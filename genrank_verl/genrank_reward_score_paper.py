"""
论文奖励设计实现 - Net Memory Segment Contribution + Discriminative Retrieved Ranking

论文核心公式：
    R_ans = Σ(i=1 to N) w_i * (s_F(i) - s_0)
    
其中：
    - F(i) 是第 i 个斐波那契数
    - s_0 是无记忆时 Working Agent 的任务完成分数
    - s_F(i) 是输入前 F(i) 个记忆时的任务完成分数
    - w_F(n) = 1/log₂(F(n)+1) - 1/log₂(F(n+1)+1) 基于 DCG 位置贡献

关键设计：
1. Net Memory Segment Contribution: 通过 s_k - s_0 衡量记忆的净贡献
2. Discriminative Retrieved Ranking: 使用斐波那契层级和 DCG 权重鼓励将重要记忆排在更前的位置
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import asyncio
from loguru import logger

import sys
sys.path.append("./")
from genrank_verl.genrank_reward_score import extract_solution, dedup_indexes
from genrank_verl.working_agent_client import (
    WorkingAgentClient, 
    WorkingAgentConfig,
    FibonacciTierEvaluator,
    create_working_agent_client,
    load_chat_prompt,
)
from utils.eval_generation_utils import (
    contains_chinese,
    tokenize_text,
    calculate_exact_match,
    calculate_f1,
    calculate_bleu,
)


def generate_fibonacci_tiers(max_k: int) -> List[int]:
    """
    生成不超过 max_k 的斐波那契序列作为 top-k 层级
    
    Args:
        max_k: 最大 k 值
        
    Returns:
        斐波那契层级列表 [1, 2, 3, 5, 8, ...] 其中所有值 <= max_k
    """
    if max_k < 1:
        return []
    
    fib = [1, 2]
    while fib[-1] < max_k:
        fib.append(fib[-1] + fib[-2])
    
    return [f for f in fib if f <= max_k]


def calculate_dcg_position_weight(position: int) -> float:
    """
    计算 DCG 中位置 k 的贡献权重 C(k)
    
    C(k) = 1 / log₂(k + 1)
    
    Args:
        position: 位置（从 1 开始）
        
    Returns:
        位置权重
    """
    if position <= 0:
        return 1.0  # 特殊处理，F(0) = 0 时
    return 1.0 / np.log2(position + 1)


def calculate_tier_weight(fib_n: int, fib_n_plus_1: int) -> float:
    """
    计算第 n 层级的权重 w_F(n)
    
    w_F(n) = C(F(n)) - C(F(n+1))
           = 1/log₂(F(n)+1) - 1/log₂(F(n+1)+1)
    
    Args:
        fib_n: F(n)，当前层级的斐波那契数
        fib_n_plus_1: F(n+1)，下一层级的斐波那契数
        
    Returns:
        层级权重
    """
    c_n = calculate_dcg_position_weight(fib_n)
    c_n_plus_1 = calculate_dcg_position_weight(fib_n_plus_1)
    return c_n - c_n_plus_1


# ============== 任务完成度评估 ==============

def compute_answer_similarity_f1(pred_answer: str, gold_answer: str) -> float:
    """
    计算预测答案与标准答案的 F1 相似度分数
    
    Args:
        pred_answer: 预测答案
        gold_answer: 标准答案
        
    Returns:
        F1 分数 (0.0 ~ 1.0)
    """
    if not pred_answer or not gold_answer:
        return 0.0
    
    # 简单的 token-level F1
    pred_tokens = set(pred_answer.lower().split())
    gold_tokens = set(gold_answer.lower().split())
    
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    intersection = pred_tokens & gold_tokens
    if not intersection:
        return 0.0
    
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_answer_similarity_rouge_l(pred_answer: str, gold_answer: str) -> float:
    """
    计算 ROUGE-L 相似度分数（基于最长公共子序列）
    
    Args:
        pred_answer: 预测答案
        gold_answer: 标准答案
        
    Returns:
        ROUGE-L 分数 (0.0 ~ 1.0)
    """
    if not pred_answer or not gold_answer:
        return 0.0
    
    pred_tokens = pred_answer.lower().split()
    gold_tokens = gold_answer.lower().split()
    
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    # 计算 LCS
    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gold_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    lcs_length = dp[m][n]
    if lcs_length == 0:
        return 0.0
    
    precision = lcs_length / m
    recall = lcs_length / n
    
    if precision + recall == 0:
        return 0.0
    
    rouge_l = 2 * precision * recall / (precision + recall)
    return rouge_l


def extract_answer_from_response(response: str) -> str:
    """
    从模型回答中提取 <answer> 标签内的内容
    
    参考 ray_chat_infer.py 中的实现
    
    Args:
        response: 模型的原始回答
        
    Returns:
        提取的答案内容，如果没有 <answer> 标签则返回原始内容
    """
    if "<answer>" in response:
        try:
            start = response.find("<answer>") + len("<answer>")
            end = response.find("</answer>")
            if end > start:
                return response[start:end].strip()
        except:
            logger.info(f"<answer>标签未关闭，原始文本: {response}")
    return response


def compute_task_score(
    agent_answer: str,
    gold_answer: str,
    score_method: str = "f1",
    is_chinese: Optional[bool] = None,
) -> float:
    """
    计算任务完成分数，支持中英文自动检测
    
    参考 utils/eval_generation_utils.py 中的 eval_generation_data 实现
    
    Args:
        agent_answer: Working Agent 的回答
        gold_answer: 标准答案
        score_method: 评分方法 ("f1", "bleu", "exact", "rouge_l")
        is_chinese: 是否为中文，如果为 None 则自动检测
        
    Returns:
        任务分数 (0.0 ~ 1.0)
    """
    if not agent_answer:
        return 0.0
    
    # 自动检测语言
    if is_chinese is None:
        is_chinese = contains_chinese(agent_answer) or contains_chinese(gold_answer)
    
    if score_method == "exact":
        return calculate_exact_match(agent_answer, gold_answer, is_chinese)
    elif score_method == "bleu":
        return calculate_bleu(agent_answer, gold_answer, is_chinese)
    elif score_method == "rouge_l":
        # 保留 rouge_l 作为兼容选项，使用原有实现
        return compute_answer_similarity_rouge_l(agent_answer, gold_answer)
    else:  # f1 (默认)
        return calculate_f1(agent_answer, gold_answer, is_chinese)


# ============== 论文奖励计算核心函数 ==============

def compute_paper_reward_from_scores(
    tier_scores: Dict[int, float],
    score_method: str = "weighted_sum",
) -> float:
    """
    根据各层级分数计算论文奖励
    
    R_ans = Σ(i=1 to N) w_i * s_F(i)
    
    Args:
        tier_scores: {tier: score} 字典，其中 tier=0 表示 s_0
        score_method: 计算方法
            - "weighted_sum": 论文公式的加权求和
            - "max_gain": 使用最大增益
            
    Returns:
        奖励分数
    """
    if not tier_scores or 0 not in tier_scores:
        logger.warning("Missing s_0 (tier=0) in tier_scores")
        return 0.0
    
    s_0 = tier_scores[0]
    
    # 获取非零层级并排序
    tiers = sorted([t for t in tier_scores.keys() if t > 0])
    
    if not tiers:
        return 0.0
    
    if score_method == "max_gain":
        # 简单方法：使用最大增益
        max_gain = max(tier_scores[t] - s_0 for t in tiers)
        return max(0.0, max_gain)
    
    # 论文公式：加权求和
    # R_ans = Σ w_i * (s_F(i) - s_0)
    
    # 计算权重
    extended_tiers = tiers
    R_ans = 0.0
    R_ans += -1 * tier_scores[0]
    for i in range(1, len(extended_tiers) - 1):
        fib_curr = extended_tiers[i]      # F(i)
        fib_next = extended_tiers[i + 1]  # F(i+1)
        
        # 权重 w_i = C(F(i)) - C(F(i+1))
        w_i = calculate_tier_weight(fib_curr, fib_next)
        
        # 使用 F(i-1) 层级的分数（如果存在）
        s_prev = tier_scores[fib_curr]
        
        # 贡献 = w_i * (s_F(i-1) - s_0)
        contribution = w_i * s_prev
        R_ans += contribution

    # 处理最后一个层级
    fib_last = extended_tiers[-1]
    s_last = tier_scores[fib_last]
    w_last = calculate_dcg_position_weight(fib_last)
    R_ans += w_last * s_last
    
    return R_ans
    

async def compute_paper_reward_async(
    solution_str: str,
    reward_model_info: Dict[str, Any],
    working_agent_client: WorkingAgentClient,
    score_method: str = "f1",
    max_ranking_len: int = 10,
) -> Tuple[float, Dict[str, Any]]:
    """
    异步计算论文奖励分数
    
    Args:
        solution_str: 模型输出的排名结果字符串
        reward_model_info: reward_model 字段信息，包含:
            - question: 原始问题
            - sessions: 候选会话列表
            - session_dates: 会话日期列表（可选）
            - task_answer: 标准答案
            - ground_truth: 原有格式的 ground_truth
            - data_source: 数据源名称，用于动态加载对应的 prompt 配置
        working_agent_client: Working Agent 客户端
        score_method: 任务完成度评分方法
        max_ranking_len: 最大排名长度
        
    Returns:
        (reward_score, debug_info)
    """
    # Step 1: 解析排名
    ground_truth = reward_model_info.get("ground_truth", "")
    if ";" in ground_truth:
        _, max_session_len = ground_truth.split(";")
        max_session_len = int(max_session_len)
    else:
        max_session_len = len(reward_model_info.get("sessions", []))
    
    ranking_dict = extract_solution(
        solution_str, 
        max_session_len, 
        min(max_ranking_len, max_session_len)
    )
    
    # 格式错误返回 0
    if ranking_dict["error"] != "":
        return 0.0, {"error": ranking_dict["error"], "stage": "parse_ranking"}
    
    ranking_list = [int(x.strip()) for x in ranking_dict["ranking"].split(",")]
    ranking_list = dedup_indexes(ranking_list)
    
    # Step 2: 获取必要信息
    question = reward_model_info.get("question", "")
    sessions = reward_model_info.get("sessions", [])
    session_dates = reward_model_info.get("session_dates", None)
    task_answer = reward_model_info.get("task_answer", "")
    data_source = reward_model_info.get("data_source", "")  # 获取 data_source
    
    if not question or not sessions:
        return 0.0, {"error": "Missing question or sessions", "stage": "get_info"}
    
    # Step 3: 根据排名重新排序会话
    try:
        ranked_sessions = [sessions[idx] for idx in ranking_list if idx < len(sessions)]
        if session_dates:
            ranked_dates = [session_dates[idx] for idx in ranking_list if idx < len(session_dates)]
        else:
            ranked_dates = None
    except IndexError as e:
        return 0.0, {"error": f"Invalid ranking index: {e}", "stage": "reorder_sessions"}
    
    # Step 4: 使用斐波那契层级评估器，传递 data_source
    evaluator = FibonacciTierEvaluator(working_agent_client)
    tier_results = await evaluator.evaluate_all_tiers(
        question=question,
        ranked_sessions=ranked_sessions,
        session_dates=ranked_dates,
        data_source=data_source,
    )
    
    # Step 5: 计算各层级的任务完成分数
    tier_scores = {}
    tier_answers = {}
    
    for tier, (answer, success) in tier_results.items():
        if success:
            # 提取 <answer> 标签中的内容
            extracted_answer = extract_answer_from_response(answer)
            score = compute_task_score(extracted_answer, task_answer, score_method)
            tier_scores[tier] = score
            tier_answers[tier] = extracted_answer  # 存储提取后的答案
        else:
            tier_scores[tier] = 0.0
            tier_answers[tier] = ""
    
    # Step 6: 计算论文奖励
    reward = compute_paper_reward_from_scores(tier_scores, score_method="weighted_sum")
    
    # 调试信息
    debug_info = {
        "ranking_list": ranking_list,
        "tier_scores": tier_scores,
        "tier_answers": tier_answers,
        "s_0": tier_scores.get(0, 0.0),
        "task_answer": task_answer,
        "reward": reward,
    }
    
    return reward, debug_info


def compute_paper_reward_sync(
    solution_str: str,
    reward_model_info: Dict[str, Any],
    working_agent_client: WorkingAgentClient,
    score_method: str = "f1",
    max_ranking_len: int = 10,
) -> Tuple[float, Dict[str, Any]]:
    """
    同步版本的论文奖励计算
    """
    return asyncio.run(
        compute_paper_reward_async(
            solution_str,
            reward_model_info,
            working_agent_client,
            score_method,
            max_ranking_len,
        )
    )


# ============== 与现有接口兼容的计算函数 ==============

# 全局客户端实例（延迟初始化）
_global_working_agent_client: Optional[WorkingAgentClient] = None


def init_working_agent_client(
    api_url: str,
    model_name: str,
    config_dir: str = "configs",
    **kwargs
):
    """
    初始化全局 Working Agent 客户端
    
    注意：prompt 配置不再在初始化时绑定，而是根据每条数据的 data_source 动态加载。
    data_source 应该包含在 reward_model_info 中传递给 compute_paper_reward_sync。
    
    Args:
        api_url: LLM API 地址
        model_name: 模型名称
        config_dir: 配置文件目录路径（用于动态加载 prompt）
        **kwargs: 其他配置参数
    """
    global _global_working_agent_client
    _global_working_agent_client = create_working_agent_client(
        api_url, 
        model_name, 
        config_dir=config_dir,
        **kwargs
    )
    logger.info(f"Initialized Working Agent client: {api_url}, model: {model_name}, config_dir: {config_dir}")


def compute_score_paper_reward(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
) -> float:
    """
    与现有接口兼容的论文奖励计算函数
    
    注意：此函数需要 reward_model_info 包含完整信息（question, sessions, task_answer 等）
    如果 Working Agent 客户端未初始化，将回退到原有的 DCG 评分
    """
    global _global_working_agent_client
    
    # 从 kwargs 中获取完整的 reward_model_info
    reward_model_info = kwargs.get("reward_model_info", {})
    
    # 兼容：如果只有 ground_truth，构造基本的 reward_model_info
    if not reward_model_info:
        reward_model_info = {"ground_truth": ground_truth}
    
    # 检查是否有足够的信息进行论文奖励计算
    has_full_info = all([
        reward_model_info.get("question"),
        reward_model_info.get("sessions"),
        reward_model_info.get("task_answer"),
    ])
    
    # 如果没有 Working Agent 客户端或信息不完整，回退到原有 DCG 评分
    if _global_working_agent_client is None or not has_full_info:
        from genrank_verl.genrank_reward_score import compute_score_instruct
        return compute_score_instruct(
            data_source,
            solution_str,
            ground_truth,
            extra_info,
            sandbox_fusion_url,
            concurrent_semaphore,
            memory_limit_mb,
            **kwargs,
        )
    
    # 使用论文奖励计算
    try:
        reward, debug_info = compute_paper_reward_sync(
            solution_str,
            reward_model_info,
            _global_working_agent_client,
            score_method=kwargs.get("score_method", "f1"),
            max_ranking_len=kwargs.get("max_ranking_len", 10),
        )
        
        if kwargs.get("debug", False):
            logger.info(f"Paper reward debug: {debug_info}")
        
        return reward
        
    except Exception as e:
        logger.error(f"Paper reward computation failed: {e}, falling back to DCG")
        from genrank_verl.genrank_reward_score import compute_score_instruct
        return compute_score_instruct(
            data_source,
            solution_str,
            ground_truth,
            extra_info,
            sandbox_fusion_url,
            concurrent_semaphore,
            memory_limit_mb,
            **kwargs,
        )


# ============== 测试代码 ==============


def test_reward_calculation():
    """测试奖励计算"""
    print("\n=== Reward Calculation Test ===")
    # 模拟的层级分数
    mock_tier_scores = {
        0: 0.2,   # s_0: 无记忆时的分数
        1: 0.4,   # s_1: 1 条记忆
        2: 0.5,   # s_2: 2 条记忆
        3: 0.6,   # s_3: 3 条记忆
        5: 0.7,   # s_5: 5 条记忆
        8: 0.75,  # s_8: 8 条记忆
    }
    
    reward = compute_paper_reward_from_scores(mock_tier_scores)
    print(f"Mock tier scores: {mock_tier_scores}")
    print(f"Computed paper reward: {reward:.4f}")


def test_text_similarity():
    """测试文本相似度计算"""
    print("\n=== Text Similarity Test ===")
    
    # 英文测试
    pred_en = "I love pizza and pasta"
    gold_en = "My favorite food is pizza"
    print(f"[English] Pred: {pred_en}")
    print(f"[English] Gold: {gold_en}")
    print(f"  F1 (new): {compute_task_score(pred_en, gold_en, 'f1'):.4f}")
    print(f"  BLEU: {compute_task_score(pred_en, gold_en, 'bleu'):.4f}")
    print(f"  Exact: {compute_task_score(pred_en, gold_en, 'exact'):.4f}")
    print(f"  ROUGE-L: {compute_task_score(pred_en, gold_en, 'rouge_l'):.4f}")
    
    # 中文测试
    pred_zh = "北京是中国的首都"
    gold_zh = "中国的首都是北京"
    print(f"\n[Chinese] Pred: {pred_zh}")
    print(f"[Chinese] Gold: {gold_zh}")
    print(f"  F1 (new): {compute_task_score(pred_zh, gold_zh, 'f1'):.4f}")
    print(f"  BLEU: {compute_task_score(pred_zh, gold_zh, 'bleu'):.4f}")
    print(f"  Exact: {compute_task_score(pred_zh, gold_zh, 'exact'):.4f}")


async def test_full_pipeline_async(api_url: str, model_name: str, data_source: str = "perltqa"):
    """
    完整的闭环测试：Memory排序 -> Working Agent多次交互 -> 输出Reward
    
    测试流程：
    1. 构造模拟的 sessions 和 question
    2. 解析 solution_str 获取 ranking
    3. 使用 FibonacciTierEvaluator 评估各层级
    4. 计算最终的论文奖励
    
    Args:
        api_url: LLM API 地址
        model_name: 模型名称
        data_source: 数据源名称，用于动态加载 prompt 配置
    """
    import os
    
    print("\n" + "=" * 60)
    print("=== Full Pipeline Test: Memory Ranking -> Agent -> Reward ===")
    print("=" * 60)
    
    # ===== Step 1: 构造测试数据 =====
    print("\n[Step 1] Constructing test data...")
    
    # 问题：用户最喜欢的食物是什么
    question = "What is my favorite food?"
    task_answer = "pizza"  # 标准答案
    
    # 模拟 8 个历史会话（每个 session 是一段对话）
    sessions = [
        # Session 0: 相关 - 明确提到喜欢 pizza
        [
            {"role": "user", "content": "I really love pizza! It's my absolute favorite food."},
            {"role": "assistant", "content": "That's great! Pizza is a delicious choice. What kind of toppings do you prefer?"},
            {"role": "user", "content": "I especially love pepperoni pizza with extra cheese."},
            {"role": "assistant", "content": "Classic combination! Pepperoni with extra cheese is always a winner."}
        ],
        # Session 1: 不相关 - 工作相关
        [
            {"role": "user", "content": "I had a very busy day at work today."},
            {"role": "assistant", "content": "Sounds exhausting! What made it so busy?"},
            {"role": "user", "content": "Too many meetings and deadlines."},
            {"role": "assistant", "content": "I hope you get some rest tonight."}
        ],
        # Session 2: 部分相关 - 提到意大利餐厅
        [
            {"role": "user", "content": "I went to an Italian restaurant yesterday."},
            {"role": "assistant", "content": "Nice! What did you order?"},
            {"role": "user", "content": "I had a margherita pizza. It was amazing!"},
            {"role": "assistant", "content": "Margherita is a classic Italian pizza. Sounds delicious!"}
        ],
        # Session 3: 相关 - 提到喜欢薄底 pizza
        [
            {"role": "user", "content": "I prefer thin crust pizza over thick crust."},
            {"role": "assistant", "content": "Thin crust is a popular choice! It's crispy and delicious."}
        ],
        # Session 4: 不相关 - 宠物相关
        [
            {"role": "user", "content": "My dog is so cute. His name is Max."},
            {"role": "assistant", "content": "Max sounds adorable! What breed is he?"},
            {"role": "user", "content": "He's a golden retriever."},
            {"role": "assistant", "content": "Golden retrievers are wonderful companions!"}
        ],
        # Session 5: 不相关 - 天气相关
        [
            {"role": "user", "content": "The weather is really nice today."},
            {"role": "assistant", "content": "That's wonderful! Are you planning to go outside?"}
        ],
        # Session 6: 部分相关 - 提到不喜欢的食物
        [
            {"role": "user", "content": "I don't like sushi very much."},
            {"role": "assistant", "content": "That's okay, everyone has different food preferences."},
            {"role": "user", "content": "Yeah, I prefer pizza and pasta over raw fish."},
            {"role": "assistant", "content": "Italian cuisine is very popular. Both are delicious!"}
        ],
        # Session 7: 不相关 - 电影相关
        [
            {"role": "user", "content": "I watched a great movie last night."},
            {"role": "assistant", "content": "What movie did you watch?"},
            {"role": "user", "content": "It was an action movie with amazing special effects."},
            {"role": "assistant", "content": "Action movies are exciting! Who was the lead actor?"}
        ],
    ]
    
    session_dates = [
        "2024-01-15", "2024-01-14", "2024-01-13", "2024-01-12",
        "2024-01-11", "2024-01-10", "2024-01-09", "2024-01-08"
    ]
    
    print(f"  Question: {question}")
    print(f"  Task answer: {task_answer}")
    print(f"  Total sessions: {len(sessions)}")
    
    # ===== Step 2: 模拟 Ranking Model 输出 =====
    print("\n[Step 2] Simulating ranking model output...")
    
    # 模拟两种不同的排名结果
    # 注意：ranking 模型输出格式为 "thinking...</think><ranking>...</ranking>"
    # <think> 标签会在调用时自动拼接（参考 compute_score_thinking）
    # ranking 长度必须等于 min(max_ranking_len, max_session_len) = min(10, 8) = 8
    
    # 好的排名：将相关会话排在前面 (0, 2, 3, 6 是相关的)
    good_ranking_str = "Analyzing memories for relevance to favorite food question. Session 0 explicitly mentions loving pizza. Session 2 mentions Italian restaurant and pizza. Session 3 discusses pizza crust preference. Session 6 mentions preferring pizza over sushi. Other sessions are not relevant.</think><ranking>0,2,3,6,1,4,5,7</ranking>"
    # 差的排名：将不相关会话排在前面
    bad_ranking_str = "Random ordering of memories without careful analysis.</think><ranking>1,4,5,7,0,2,3,6</ranking>"
    
    print(f"  Good ranking (model output): {good_ranking_str[:60]}...")
    print(f"  Bad ranking (model output): {bad_ranking_str[:60]}...")
    
    # 验证 ranking 格式解析
    print("\n[Step 2.1] Validating ranking format parsing...")
    max_session_len = len(sessions)
    max_ranking_len = min(10, max_session_len)
    
    # 拼接 <think> 前缀（模拟 compute_score_thinking 的行为）
    good_solution_str = "<think>" + good_ranking_str
    bad_solution_str = "<think>" + bad_ranking_str
    
    good_parsed = extract_solution(good_solution_str, max_session_len, max_ranking_len)
    bad_parsed = extract_solution(bad_solution_str, max_session_len, max_ranking_len)
    
    print(f"  Good ranking parsed: error='{good_parsed['error']}', ranking='{good_parsed['ranking']}'")
    print(f"  Bad ranking parsed: error='{bad_parsed['error']}', ranking='{bad_parsed['ranking']}'")
    
    if good_parsed["error"] or bad_parsed["error"]:
        print(f"  [ERROR] Parsing failed! Check ranking format.")
        return {"error": "ranking parsing failed"}
    
    # ===== Step 3: 初始化 Working Agent Client =====
    print("\n[Step 3] Initializing Working Agent client...")
    
    client = create_working_agent_client(
        api_url=api_url,
        model_name=model_name,
        config_dir="configs",
        max_concurrent_requests=8,
        temperature=0.0,  # 确定性输出
    )
    print(f"  API URL: {api_url}")
    print(f"  Model: {model_name}")
    print(f"  Data source: {data_source}")
    # 测试动态加载 prompt
    test_prompt_config = client._get_prompt_config(data_source)
    print(f"  Prompt config loaded for '{data_source}': {test_prompt_config is not None}")
    
    # ===== Step 4: 测试好的排名 =====
    print("\n[Step 4] Testing GOOD ranking...")
    
    # 构造 reward_model_info（包含 data_source 用于动态加载 prompt）
    reward_model_info_good = {
        "question": question,
        "sessions": sessions,
        "session_dates": session_dates,
        "task_answer": task_answer,
        "ground_truth": f"0,2,3,6;{len(sessions)}",  # ground_truth 格式
        "data_source": data_source,  # 用于动态加载 prompt 配置
    }
    
    # 使用拼接后的完整 solution_str（包含 <think> 前缀）
    reward_good, debug_info_good = await compute_paper_reward_async(
        solution_str=good_solution_str,
        reward_model_info=reward_model_info_good,
        working_agent_client=client,
        score_method="f1",  # 使用 F1 评分
        max_ranking_len=len(sessions),
    )
    
    print(f"\n  === Good Ranking Results ===")
    print(f"  Parsed ranking: {debug_info_good.get('ranking_list', [])}")
    print(f"  Task answer: {debug_info_good.get('task_answer', '')}")
    print(f"  s_0 (no memory): {debug_info_good.get('s_0', 0.0):.4f}")
    print(f"\n  Tier scores (F1):")
    tier_scores = debug_info_good.get('tier_scores', {})
    tier_answers = debug_info_good.get('tier_answers', {})
    for tier in sorted(tier_scores.keys()):
        answer_preview = tier_answers.get(tier, "")[:50] + "..." if len(tier_answers.get(tier, "")) > 50 else tier_answers.get(tier, "")
        gain = tier_scores[tier] - tier_scores.get(0, 0)
        print(f"    Tier {tier}: score={tier_scores[tier]:.4f}, gain={gain:+.4f}, answer='{answer_preview}'")
    print(f"\n  Final reward (good ranking): {reward_good:.4f}")
    
    # ===== Step 5: 测试差的排名 =====
    print("\n[Step 5] Testing BAD ranking...")
    
    reward_model_info_bad = {
        "question": question,
        "sessions": sessions,
        "session_dates": session_dates,
        "task_answer": task_answer,
        "ground_truth": f"0,2,3,6;{len(sessions)}",
        "data_source": data_source,  # 用于动态加载 prompt 配置
    }
    
    # 使用拼接后的完整 solution_str（包含 <think> 前缀）
    reward_bad, debug_info_bad = await compute_paper_reward_async(
        solution_str=bad_solution_str,
        reward_model_info=reward_model_info_bad,
        working_agent_client=client,
        score_method="f1",
        max_ranking_len=len(sessions),
    )
    
    print(f"\n  === Bad Ranking Results ===")
    print(f"  Parsed ranking: {debug_info_bad.get('ranking_list', [])}")
    print(f"  s_0 (no memory): {debug_info_bad.get('s_0', 0.0):.4f}")
    print(f"\n  Tier scores (F1):")
    tier_scores_bad = debug_info_bad.get('tier_scores', {})
    tier_answers_bad = debug_info_bad.get('tier_answers', {})
    for tier in sorted(tier_scores_bad.keys()):
        answer_preview = tier_answers_bad.get(tier, "")[:50] + "..." if len(tier_answers_bad.get(tier, "")) > 50 else tier_answers_bad.get(tier, "")
        gain = tier_scores_bad[tier] - tier_scores_bad.get(0, 0)
        print(f"    Tier {tier}: score={tier_scores_bad[tier]:.4f}, gain={gain:+.4f}, answer='{answer_preview}'")
    print(f"\n  Final reward (bad ranking): {reward_bad:.4f}")
    
    # ===== Step 6: 比较结果 =====
    print("\n" + "=" * 60)
    print("=== Comparison Summary ===")
    print("=" * 60)
    print(f"  Good ranking reward: {reward_good:.4f}")
    print(f"  Bad ranking reward:  {reward_bad:.4f}")
    print(f"  Difference:          {reward_good - reward_bad:+.4f}")
    
    if reward_good > reward_bad:
        print("\n  ✓ Paper reward correctly assigns higher score to good ranking!")
    else:
        print("\n  ✗ Warning: Bad ranking got higher score. Check the data or model.")
    
    return {
        "good_ranking": {
            "reward": reward_good,
            "debug_info": debug_info_good
        },
        "bad_ranking": {
            "reward": reward_bad,
            "debug_info": debug_info_bad
        }
    }


async def test_chinese_pipeline_async(api_url: str, model_name: str, data_source: str = "perltqa"):
    """
    中文闭环测试：Memory排序 -> Working Agent多次交互 -> 输出Reward
    
    Args:
        api_url: LLM API 地址
        model_name: 模型名称
        data_source: 数据源名称，用于动态加载 prompt 配置
    """
    print("\n" + "=" * 60)
    print("=== Chinese Pipeline Test ===")
    print("=" * 60)
    
    # 问题和答案
    question = "我最喜欢的运动是什么?"
    task_answer = "篮球"
    
    # 中文会话
    sessions = [
        # Session 0: 相关 - 提到喜欢篮球
        [
            {"role": "user", "content": "我特别喜欢打篮球，每周都会去球场打球。"},
            {"role": "assistant", "content": "篮球是一项很好的运动！你打什么位置？"},
            {"role": "user", "content": "我一般打后卫，喜欢控球和助攻。"},
            {"role": "assistant", "content": "后卫需要很好的视野和传球能力。"}
        ],
        # Session 1: 不相关 - 工作相关
        [
            {"role": "user", "content": "今天工作好累啊，加班到很晚。"},
            {"role": "assistant", "content": "辛苦了，要注意休息。"}
        ],
        # Session 2: 部分相关 - 提到运动
        [
            {"role": "user", "content": "周末我去看了一场NBA比赛直播。"},
            {"role": "assistant", "content": "是哪两支队伍的比赛？"},
            {"role": "user", "content": "湖人对阵勇士，特别精彩！"},
            {"role": "assistant", "content": "这两支队伍的比赛总是很有看点！"}
        ],
        # Session 3: 不相关 - 饮食相关
        [
            {"role": "user", "content": "今天中午吃了一顿火锅。"},
            {"role": "assistant", "content": "火锅很好吃！你喜欢什么口味的？"}
        ],
    ]
    
    session_dates = ["2024-01-15", "2024-01-14", "2024-01-13", "2024-01-12"]
    
    print(f"  Question: {question}")
    print(f"  Task answer: {task_answer}")
    print(f"  Total sessions: {len(sessions)}")
    
    # 好的排名：相关会话在前
    # 注意：ranking 模型输出格式为 "thinking...</think><ranking>...</ranking>"
    # <think> 标签会在调用时自动拼接
    # ranking 长度必须等于 min(max_ranking_len, max_session_len) = min(10, 4) = 4
    good_ranking_str = "分析与运动相关的记忆。Session 0 明确提到喜欢打篮球，是最相关的。Session 2 提到看NBA比赛，与篮球相关。Session 1 和 3 与运动无关。</think><ranking>0,2,1,3</ranking>"
    
    # 验证 ranking 格式解析
    print("\n  Validating ranking format parsing...")
    max_session_len = len(sessions)
    max_ranking_len = min(10, max_session_len)
    
    # 拼接 <think> 前缀
    good_solution_str = "<think>" + good_ranking_str
    good_parsed = extract_solution(good_solution_str, max_session_len, max_ranking_len)
    print(f"  Parsed: error='{good_parsed['error']}', ranking='{good_parsed['ranking']}'")
    
    if good_parsed["error"]:
        print(f"  [ERROR] Parsing failed! Check ranking format.")
        return 0.0, {"error": "ranking parsing failed"}
    
    # 初始化客户端
    client = create_working_agent_client(
        api_url=api_url,
        model_name=model_name,
        config_dir="configs",
        max_concurrent_requests=4,
    )
    
    # 计算奖励（包含 data_source 用于动态加载 prompt）
    reward_model_info = {
        "question": question,
        "sessions": sessions,
        "session_dates": session_dates,
        "task_answer": task_answer,
        "ground_truth": f"0,2;{len(sessions)}",
        "data_source": data_source,  # 用于动态加载 prompt 配置
    }
    
    # 使用拼接后的完整 solution_str
    reward, debug_info = await compute_paper_reward_async(
        solution_str=good_solution_str,
        reward_model_info=reward_model_info,
        working_agent_client=client,
        score_method="f1",  # 使用 F1 评分（支持中文分词）
        max_ranking_len=len(sessions),
    )
    
    print(f"\n  === Chinese Test Results ===")
    print(f"  Parsed ranking: {debug_info.get('ranking_list', [])}")
    print(f"  s_0 (no memory): {debug_info.get('s_0', 0.0):.4f}")
    print(f"\n  Tier scores (F1 with Chinese tokenization):")
    tier_scores = debug_info.get('tier_scores', {})
    tier_answers = debug_info.get('tier_answers', {})
    for tier in sorted(tier_scores.keys()):
        answer = tier_answers.get(tier, "")
        answer_preview = answer[:30] + "..." if len(answer) > 30 else answer
        gain = tier_scores[tier] - tier_scores.get(0, 0)
        print(f"    Tier {tier}: score={tier_scores[tier]:.4f}, gain={gain:+.4f}, answer='{answer_preview}'")
    print(f"\n  Final reward: {reward:.4f}")
    
    return reward, debug_info


def test_full_pipeline_sync(api_url: str, model_name: str, data_source: str = "perltqa"):
    """同步版本的完整闭环测试"""
    return asyncio.run(test_full_pipeline_async(api_url, model_name, data_source))


def test_chinese_pipeline_sync(api_url: str, model_name: str, data_source: str = "perltqa"):
    """同步版本的中文闭环测试"""
    return asyncio.run(test_chinese_pipeline_async(api_url, model_name, data_source))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper Reward Score Tests")
    parser.add_argument("--api_url", type=str, default="http://172.16.77.93:8000",
                        help="LLM API URL")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                        help="Model name")
    parser.add_argument("--data_source", type=str, default="perltqa",
                        help="Data source name for dynamic prompt loading")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "basic", "pipeline", "chinese"],
                        help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test in ["all", "basic"]:
        # 基础测试（无需 API）
        test_reward_calculation()
        test_text_similarity()
    
    if args.test in ["all", "pipeline"]:
        # 完整闭环测试（需要 API）
        print("\n" + "=" * 60)
        print("Starting full pipeline test (requires LLM API)...")
        print("=" * 60)
        try:
            test_full_pipeline_sync(args.api_url, args.model_name, args.data_source)
        except Exception as e:
            logger.error(f"Pipeline test failed: {e}")
            print(f"\n[Error] Pipeline test failed: {e}")
            print("Make sure the LLM API is running and accessible.")
    
    if args.test in ["all", "chinese"]:
        # 中文测试（需要 API）
        print("\n" + "=" * 60)
        print("Starting Chinese pipeline test (requires LLM API)...")
        print("=" * 60)
        try:
            test_chinese_pipeline_sync(args.api_url, args.model_name, args.data_source)
        except Exception as e:
            logger.error(f"Chinese pipeline test failed: {e}")
            print(f"\n[Error] Chinese pipeline test failed: {e}")
