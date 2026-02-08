"""
Working Agent Client - 用于调用 Working Agent LLM 评估任务完成度

论文中的核心设计：
1. Net Memory Segment Contribution: 计算记忆段对任务完成的净贡献 (s_k - s_0)
2. 需要多次调用 Working Agent 来评估不同 top-k 记忆配置下的任务表现
"""

import asyncio
import os
import yaml
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
from openai import AsyncOpenAI
from easydict import EasyDict as edict

# 导入 session 处理工具
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from utils.session_process import construct_history_text


# DeepSearch 数据集列表（使用 deepsearch_default_prompt.yaml）
DEEPSEARCH_DATASETS = [
    "2WikiMultihopQA",
    "HotpotQA",
    "MegaScience",
    "MuSiQue",
    "OneGen-TrainDataset-MultiHopQA",
    "QA-Expert-Multi-Hop-V1.0",
    "TaskCraft",
    "Voyager1.0",
    "WebDancer",
    "WebShaper",
    "WebWalkerQA-Silver",
    "WikiTables",
]


def load_chat_prompt(dataset_name: str, config_dir: str = "configs") -> dict:
    """
    根据 dataset_name 加载对应的 prompt 配置文件
    
    参考 ray_chat_infer.py 的逻辑：
    - DeepSearch 数据集使用 deepsearch_default_prompt.yaml
    - 其他数据集使用 chat_default_prompt.yaml
    - 如果存在数据集专用的 prompt 文件，优先使用
    
    Args:
        dataset_name: 数据集名称
        config_dir: 配置文件目录路径
        
    Returns:
        prompt 配置字典，包含 prompt_template.system_prompt_text 和 user_prompt_text
    """
    # 确定默认的 prompt 文件
    if dataset_name in DEEPSEARCH_DATASETS:
        default_path = os.path.join(config_dir, "deepsearch_default_prompt.yaml")
    else:
        default_path = os.path.join(config_dir, "chat_default_prompt.yaml")
    
    # 检查是否存在数据集专用的 prompt 文件
    dataset_specific_path = os.path.join(config_dir, f"chat_{dataset_name}_prompt.yaml")
    if os.path.exists(dataset_specific_path):
        prompt_config_path = dataset_specific_path
    else:
        prompt_config_path = default_path
    
    # 加载配置
    try:
        with open(prompt_config_path, "r", encoding="utf-8") as f:
            prompt_config = edict(yaml.safe_load(f))
        logger.info(f"Loaded prompt config from: {prompt_config_path}")
        return dict(prompt_config)
    except FileNotFoundError:
        logger.warning(f"Prompt config not found: {prompt_config_path}, using default hardcoded prompt")
        return None
    except Exception as e:
        logger.error(f"Error loading prompt config: {e}")
        return None


@dataclass
class WorkingAgentConfig:
    """Working Agent 配置
    
    注意：prompt 配置不再绑定到 Config 中，而是根据每条数据的 data_source 动态加载。
    参考 ray_chat_infer.py 中的 load_chat_prompt 逻辑。
    """
    api_url: str                          # LLM API 地址 (e.g., vLLM OpenAI-compatible endpoint)
    model_name: str                       # 模型名称
    api_key: str = ""                     # API Key（用于认证）
    max_tokens: int = 4096                # 最大生成 token 数
    temperature: float = 0.3              # 温度，0.0 表示确定性输出
    timeout: int = 300                    # 请求超时时间（秒）
    max_concurrent_requests: int = 32     # 最大并发请求数
    retry_times: int = 3                  # 重试次数
    config_dir: str = "configs"           # 配置文件目录路径（用于动态加载 prompt）


class WorkingAgentClient:
    """
    Working Agent 客户端
    
    用于调用 LLM 作为 Working Agent，评估给定记忆段下的任务完成度。
    支持异步批量请求以提高效率。
    
    Prompt 加载策略：
    - 根据每条数据的 data_source 动态加载对应的 prompt 配置
    - 已加载的 prompt 会缓存到 _prompt_cache 中，避免重复加载
    - 参考 ray_chat_infer.py 中的 load_chat_prompt 逻辑
    """
    
    def __init__(self, config: WorkingAgentConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        # 初始化 AsyncOpenAI 客户端
        self.client = AsyncOpenAI(
            base_url=f"{config.api_url}",
            api_key=config.api_key or "EMPTY",  # vLLM 兼容，无 key 时使用 EMPTY
        )
        
        # Prompt 缓存：{data_source: prompt_config}
        # 根据每条数据的 data_source 动态加载并缓存 prompt 配置
        self._prompt_cache: Dict[str, Optional[dict]] = {}
        logger.info("WorkingAgentClient initialized with dynamic prompt loading")
    
    def _get_prompt_config(self, data_source: str) -> Optional[dict]:
        """
        根据 data_source 获取对应的 prompt 配置（带缓存）
        
        Args:
            data_source: 数据源名称（对应 dataset_name）
            
        Returns:
            prompt 配置字典，如果加载失败则返回 None
        """
        if data_source not in self._prompt_cache:
            prompt_config = load_chat_prompt(data_source, self.config.config_dir)
            self._prompt_cache[data_source] = prompt_config
            logger.debug(f"Loaded and cached prompt config for data_source: {data_source}")
        return self._prompt_cache[data_source]
        
    def _build_working_agent_messages(
        self,
        question: str,
        sessions: List[List[Dict[str, str]]],
        session_dates: Optional[List[str]] = None,
        data_source: str = "",
    ) -> List[Dict[str, str]]:
        """
        构建 Working Agent 的输入 messages（包含 system 和 user）
        
        Args:
            question: 用户问题
            sessions: 会话列表，每个 session 是 turn 列表，每个 turn 是 {"role": str, "content": str}
            session_dates: 会话日期列表（可选）
            data_source: 数据源名称，用于动态加载对应的 prompt 配置
            
        Returns:
            messages 列表 [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        """
        # 使用 construct_history_text 构建 history_chats
        if sessions:
            history_chats = construct_history_text(sessions, session_dates)
        else:
            history_chats = ""
        
        # 根据 data_source 动态加载 prompt 配置
        prompt_config = self._get_prompt_config(data_source) if data_source else None
        
        # 如果有 prompt_config，使用模板
        if prompt_config:
            prompt_template = prompt_config.get("prompt_template", {})
            system_prompt = prompt_template.get("system_prompt_text", "You are a helpful assistant.")
            user_prompt_template = prompt_template.get("user_prompt_text", "{history_chats}\n\nQuestion: {question}")
            
            # 格式化 user prompt
            user_prompt = user_prompt_template.format(
                history_chats=history_chats,
                question=question
            )
            
            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ]
        else:
            # 回退到硬编码 prompt（保持向后兼容）
            if not sessions:
                # 无记忆情况 (s_0)
                user_prompt = f"""Please answer the following question based on your knowledge.

Question: {question}

Please provide a direct and concise answer."""
            else:
                # 有记忆情况 (s_k)
                user_prompt = f"""Please answer the following question based on the provided memory segments.

## Memory Segments
{history_chats}

## Question
{question}

Please provide a direct and concise answer based on the memory segments above. If the memory does not contain relevant information, you may use your general knowledge."""
            
            messages = [
                {"role": "user", "content": user_prompt.strip()},
            ]
        
        return messages
    
    async def _call_llm_api(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[str, bool]:
        """
        调用 LLM API（使用 AsyncOpenAI 客户端）
        
        Args:
            messages: 完整的 messages 列表，包含 system 和 user 消息
            
        Returns:
            (response_text, success)
        """
        async with self.semaphore:
            for attempt in range(self.config.retry_times):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        timeout=self.config.timeout,
                    )
                    return response.choices[0].message.content, True
                except asyncio.TimeoutError:
                    logger.warning(f"Request timeout (attempt {attempt + 1})")
                except Exception as e:
                    logger.warning(f"Request error (attempt {attempt + 1}): {e}")
                
                if attempt < self.config.retry_times - 1:
                    await asyncio.sleep(1 * (attempt + 1))  # 指数退避
            
            return "", False
    
    async def evaluate_single(
        self,
        question: str,
        sessions: List[List[Dict[str, str]]],
        session_dates: Optional[List[str]] = None,
        data_source: str = "",
    ) -> Tuple[str, bool]:
        """
        评估单个配置下 Working Agent 的回答
        
        Args:
            question: 用户问题
            sessions: 会话列表，每个 session 是 turn 列表，每个 turn 是 {"role": str, "content": str}
            session_dates: 会话日期列表
            data_source: 数据源名称，用于动态加载对应的 prompt 配置
            
        Returns:
            (agent_answer, success)
        """
        messages = self._build_working_agent_messages(question, sessions, session_dates, data_source)
        return await self._call_llm_api(messages)
    
    async def evaluate_batch(
        self,
        requests: List[Dict[str, Any]],
        data_source: str = "",
    ) -> List[Tuple[str, bool]]:
        """
        批量评估多个配置
        
        Args:
            requests: 请求列表，每个请求包含:
                - question: str
                - sessions: List[List[Dict[str, str]]]  # 会话列表
                - session_dates: Optional[List[str]]
            data_source: 数据源名称，用于动态加载对应的 prompt 配置
                
        Returns:
            结果列表 [(agent_answer, success), ...]
        """
        tasks = []
        for req in requests:
            task = self.evaluate_single(
                req["question"],
                req["sessions"],
                req.get("session_dates"),
                data_source,
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch evaluation error: {result}")
                processed_results.append(("", False))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def evaluate_batch_sync(
        self,
        requests: List[Dict[str, Any]],
        data_source: str = "",
    ) -> List[Tuple[str, bool]]:
        """
        同步版本的批量评估（便于非异步环境调用）
        
        Args:
            requests: 请求列表
            data_source: 数据源名称，用于动态加载对应的 prompt 配置
            
        Returns:
            结果列表
        """
        return asyncio.run(self.evaluate_batch(requests, data_source))


class FibonacciTierEvaluator:
    """
    基于斐波那契序列的层级评估器
    
    论文核心设计：使用斐波那契序列 (1, 2, 3, 5, 8...) 作为 top-k 层级，
    在准确度和计算开销之间取得平衡。
    """
    
    def __init__(self, working_agent_client: WorkingAgentClient):
        self.client = working_agent_client
        self._fib_cache = {}
    
    @staticmethod
    def generate_fibonacci_tiers(max_k: int) -> List[int]:
        """
        生成不超过 max_k 的斐波那契序列
        
        Args:
            max_k: 最大 k 值
            
        Returns:
            斐波那契序列列表 [1, 2, 3, 5, 8, ...]
        """
        if max_k < 1:
            return []
        
        fib = [1, 2]
        while fib[-1] < max_k:
            fib.append(fib[-1] + fib[-2])
        
        # 只保留不超过 max_k 的值
        fib = [f for f in fib if f <= max_k]
        if fib[-1] != max_k:
            fib.append(max_k)
        return fib
        
    
    async def evaluate_all_tiers(
        self,
        question: str,
        ranked_sessions: List[List[Dict[str, str]]],
        session_dates: Optional[List[str]] = None,
        data_source: str = "",
    ) -> Dict[int, Tuple[str, bool]]:
        """
        评估所有斐波那契层级的 Working Agent 回答
        
        Args:
            question: 用户问题
            ranked_sessions: 按排名排序的会话列表，每个 session 是 turn 列表
            session_dates: 会话日期列表（与 ranked_sessions 对应）
            data_source: 数据源名称，用于动态加载对应的 prompt 配置
            
        Returns:
            {tier: (answer, success)} 字典，其中 tier=0 表示无记忆情况
        """
        max_k = len(ranked_sessions)
        tiers = [0] + self.generate_fibonacci_tiers(max_k)  # 0 表示 s_0（无记忆）
        
        # 构建批量请求
        requests = []
        for tier in tiers:
            if tier == 0:
                sessions = []
                dates = None
            else:
                sessions = ranked_sessions[:tier]
                dates = session_dates[:tier] if session_dates else None
            
            requests.append({
                "question": question,
                "sessions": sessions,
                "session_dates": dates,
            })
        
        # 批量评估，传递 data_source
        results = await self.client.evaluate_batch(requests, data_source)
        
        # 组织结果
        tier_results = {}
        for tier, result in zip(tiers, results):
            tier_results[tier] = result
        
        return tier_results
    
    def evaluate_all_tiers_sync(
        self,
        question: str,
        ranked_sessions: List[List[Dict[str, str]]],
        session_dates: Optional[List[str]] = None,
        data_source: str = "",
    ) -> Dict[int, Tuple[str, bool]]:
        """
        同步版本的层级评估
        
        Args:
            question: 用户问题
            ranked_sessions: 按排名排序的会话列表
            session_dates: 会话日期列表
            data_source: 数据源名称，用于动态加载对应的 prompt 配置
        """
        return asyncio.run(self.evaluate_all_tiers(question, ranked_sessions, session_dates, data_source))


# 便捷函数
def create_working_agent_client(
    api_url: str,
    model_name: str,
    config_dir: str = "configs",
    **kwargs
) -> WorkingAgentClient:
    """
    创建 Working Agent 客户端的便捷函数
    
    注意：prompt 配置不再在创建时绑定，而是根据每条数据的 data_source 动态加载。
    
    Args:
        api_url: LLM API 地址
        model_name: 模型名称
        config_dir: 配置文件目录路径（用于动态加载 prompt）
        **kwargs: 其他配置参数（不应包含 dataset_name 或 prompt_config）
        
    Returns:
        WorkingAgentClient 实例
    """
    # 过滤掉旧版本可能传入的 dataset_name 和 prompt_config 参数
    kwargs.pop("dataset_name", None)
    kwargs.pop("prompt_config", None)
    
    api_key = os.getenv("API_KEY", os.getenv("WORKING_AGENT_API_KEY"))
    if not api_key:
        raise ValueError("API_KEY or WORKING_AGENT_API_KEY is not set")
    config = WorkingAgentConfig(
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
        config_dir=config_dir,
        **kwargs
    )
    return WorkingAgentClient(config)


if __name__ == "__main__":
    # 测试代码
    async def test():
        # 测试用的 data_source，用于动态加载 prompt
        test_data_source = "perltqa"
        
        config = WorkingAgentConfig(
            api_url="http://172.16.77.93:8000",
            api_key=os.getenv("API_KEY"),
            model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
            max_concurrent_requests=4,
            config_dir="configs",  # prompt 配置目录
        )
        client = WorkingAgentClient(config)
        evaluator = FibonacciTierEvaluator(client)
        
        # 测试斐波那契序列生成
        print("Fibonacci tiers for max_k=10:", evaluator.generate_fibonacci_tiers(10))
        print("Fibonacci tiers for max_k=20:", evaluator.generate_fibonacci_tiers(20))
        
        # 模拟测试（需要实际 API）
        question = "What is my favorite food?"
        # sessions 格式：List[List[Dict[str, str]]]
        sessions = [
            [{"role": "user", "content": "I really love pizza!"},
             {"role": "assistant", "content": "That's great! What kind of pizza do you like?"}],
            [{"role": "user", "content": "I had a busy day at work."},
             {"role": "assistant", "content": "I hope you can get some rest."}],
            [{"role": "user", "content": "I went to an Italian restaurant yesterday."},
             {"role": "assistant", "content": "Sounds delicious!"}],
            [{"role": "user", "content": "I prefer thin crust pizza."},
             {"role": "assistant", "content": "Thin crust is a classic choice!"}],
            [{"role": "user", "content": "My dog is so cute."},
             {"role": "assistant", "content": "What's your dog's name?"}],
        ]
        
        # 这里只做结构测试，实际调用需要启动 LLM 服务
        print(f"\nWould evaluate tiers: {[0] + evaluator.generate_fibonacci_tiers(len(sessions))}")
        
        # 测试 messages 构建（传入 data_source 动态加载 prompt）
        messages = client._build_working_agent_messages(question, sessions[:2], data_source=test_data_source)
        prompt_config = client._get_prompt_config(test_data_source)
        print(f"\nBuilt messages with prompt_config for '{test_data_source}': {prompt_config is not None}")
        for msg in messages:
            print(f"  [{msg['role']}]: {msg['content'][:100]}...")
        
        response = await client.client.chat.completions.create(
            model=client.config.model_name,
            messages=messages,
            max_tokens=client.config.max_tokens,
            temperature=client.config.temperature,
            timeout=client.config.timeout,
        )
        print(f"Response: {response.choices[0].message.content}")
        
    asyncio.run(test())
