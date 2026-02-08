"""
任务完成度评估模块 - LLM-as-Judge + F1/ROUGE fallback

提供三种评估策略：
1. LLM-as-Judge: 使用 LLM 评估答案质量（推荐用于开放式问答）
2. F1 Score: token-level F1 相似度
3. ROUGE-L: 基于最长公共子序列的相似度

优先使用 LLM-as-Judge，失败时自动 fallback 到文本相似度。
"""

import asyncio
import aiohttp
import json
import re
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from loguru import logger
from enum import Enum


class EvalMethod(Enum):
    """评估方法枚举"""
    LLM_JUDGE = "llm_judge"
    F1 = "f1"
    ROUGE_L = "rouge_l"
    EXACT = "exact"
    AUTO = "auto"  # 自动选择：优先 LLM-as-Judge，失败时 fallback


@dataclass
class JudgeConfig:
    """LLM-as-Judge 配置"""
    api_url: str                          # LLM API 地址
    model_name: str                       # 模型名称
    max_tokens: int = 256                 # 最大生成 token 数
    temperature: float = 0.0              # 温度
    timeout: int = 60                     # 超时时间
    retry_times: int = 2                  # 重试次数


# LLM-as-Judge Prompt 模板
JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator. Your task is to evaluate how well an answer addresses a given question, compared to the reference answer.

## Question
{question}

## Reference Answer
{gold_answer}

## Answer to Evaluate
{pred_answer}

## Evaluation Criteria
- Score 1.0: The answer is completely correct, capturing all key information from the reference.
- Score 0.8: The answer is mostly correct with minor omissions or slight inaccuracies.
- Score 0.6: The answer is partially correct, capturing the main idea but missing important details.
- Score 0.4: The answer has some relevant information but significant errors or omissions.
- Score 0.2: The answer has minimal relevance to the question.
- Score 0.0: The answer is completely wrong or irrelevant.

## Instructions
1. Compare the answer to evaluate with the reference answer.
2. Consider semantic similarity, not just exact wording.
3. Focus on factual correctness and completeness.
4. Output ONLY a JSON object with the score and a brief reason.

## Output Format
{{"score": <float between 0.0 and 1.0>, "reason": "<brief explanation>"}}

Your evaluation:"""


JUDGE_PROMPT_TEMPLATE_ZH = """你是一个专业的答案评估专家。你的任务是评估一个答案对给定问题的回答质量，并与参考答案进行比较。

## 问题
{question}

## 参考答案
{gold_answer}

## 待评估答案
{pred_answer}

## 评分标准
- 1.0 分：答案完全正确，包含参考答案中的所有关键信息。
- 0.8 分：答案基本正确，有轻微遗漏或小的不准确之处。
- 0.6 分：答案部分正确，抓住了主要思想但遗漏了重要细节。
- 0.4 分：答案包含一些相关信息，但有明显错误或重大遗漏。
- 0.2 分：答案与问题的相关性很低。
- 0.0 分：答案完全错误或不相关。

## 评估要求
1. 将待评估答案与参考答案进行比较。
2. 考虑语义相似性，而不仅仅是精确的措辞。
3. 关注事实正确性和完整性。
4. 只输出一个包含分数和简短理由的 JSON 对象。

## 输出格式
{{"score": <0.0到1.0之间的浮点数>, "reason": "<简短解释>"}}

你的评估："""


class AnswerEvaluator:
    """
    答案评估器
    
    支持多种评估方法，包括 LLM-as-Judge 和文本相似度。
    """
    
    def __init__(
        self,
        judge_config: Optional[JudgeConfig] = None,
        default_method: EvalMethod = EvalMethod.AUTO,
        use_chinese_prompt: bool = False,
    ):
        """
        初始化评估器
        
        Args:
            judge_config: LLM-as-Judge 配置（可选）
            default_method: 默认评估方法
            use_chinese_prompt: 是否使用中文 prompt
        """
        self.judge_config = judge_config
        self.default_method = default_method
        self.use_chinese_prompt = use_chinese_prompt
        
    # ============== 文本相似度方法 ==============
    
    @staticmethod
    def compute_f1(pred_answer: str, gold_answer: str) -> float:
        """
        计算 token-level F1 分数
        
        Args:
            pred_answer: 预测答案
            gold_answer: 标准答案
            
        Returns:
            F1 分数 (0.0 ~ 1.0)
        """
        if not pred_answer or not gold_answer:
            return 0.0
        
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
        
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def compute_rouge_l(pred_answer: str, gold_answer: str) -> float:
        """
        计算 ROUGE-L 分数（基于最长公共子序列）
        
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
        
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def compute_exact_match(pred_answer: str, gold_answer: str) -> float:
        """
        精确匹配
        
        Args:
            pred_answer: 预测答案
            gold_answer: 标准答案
            
        Returns:
            1.0 如果匹配，否则 0.0
        """
        if not pred_answer or not gold_answer:
            return 0.0
        return 1.0 if pred_answer.strip().lower() == gold_answer.strip().lower() else 0.0
    
    # ============== LLM-as-Judge 方法 ==============
    
    def _build_judge_prompt(
        self,
        question: str,
        pred_answer: str,
        gold_answer: str,
    ) -> str:
        """构建 LLM-as-Judge prompt"""
        template = JUDGE_PROMPT_TEMPLATE_ZH if self.use_chinese_prompt else JUDGE_PROMPT_TEMPLATE
        return template.format(
            question=question,
            pred_answer=pred_answer,
            gold_answer=gold_answer,
        )
    
    def _parse_judge_response(self, response: str) -> Tuple[float, str]:
        """
        解析 LLM-as-Judge 的响应
        
        Args:
            response: LLM 响应文本
            
        Returns:
            (score, reason)
        """
        try:
            # 尝试提取 JSON
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group())
                score = float(result.get("score", 0.0))
                reason = result.get("reason", "")
                # 确保分数在有效范围内
                score = max(0.0, min(1.0, score))
                return score, reason
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse judge response: {e}")
        
        # 尝试直接提取数字
        number_match = re.search(r'(\d+\.?\d*)', response)
        if number_match:
            score = float(number_match.group(1))
            if score > 1.0:
                score = score / 10.0 if score <= 10 else score / 100.0
            return max(0.0, min(1.0, score)), "Extracted from response"
        
        return 0.0, "Failed to parse"
    
    async def _call_judge_api(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
    ) -> Tuple[str, bool]:
        """调用 LLM-as-Judge API"""
        if not self.judge_config:
            return "", False
        
        for attempt in range(self.judge_config.retry_times):
            try:
                payload = {
                    "model": self.judge_config.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": self.judge_config.max_tokens,
                    "temperature": self.judge_config.temperature,
                }
                
                async with session.post(
                    f"{self.judge_config.api_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.judge_config.timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"], True
                    else:
                        error_text = await response.text()
                        logger.warning(f"Judge API error (attempt {attempt + 1}): {error_text}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Judge API timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Judge API error (attempt {attempt + 1}): {e}")
            
            if attempt < self.judge_config.retry_times - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
        
        return "", False
    
    async def evaluate_with_llm_judge(
        self,
        question: str,
        pred_answer: str,
        gold_answer: str,
    ) -> Tuple[float, str, bool]:
        """
        使用 LLM-as-Judge 评估答案
        
        Args:
            question: 问题
            pred_answer: 预测答案
            gold_answer: 标准答案
            
        Returns:
            (score, reason, success)
        """
        if not self.judge_config:
            return 0.0, "No judge config", False
        
        prompt = self._build_judge_prompt(question, pred_answer, gold_answer)
        
        async with aiohttp.ClientSession() as session:
            response, success = await self._call_judge_api(session, prompt)
        
        if success:
            score, reason = self._parse_judge_response(response)
            return score, reason, True
        
        return 0.0, "API call failed", False
    
    # ============== 统一评估接口 ==============
    
    async def evaluate_async(
        self,
        pred_answer: str,
        gold_answer: str,
        question: Optional[str] = None,
        method: Optional[EvalMethod] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        异步评估答案质量
        
        Args:
            pred_answer: 预测答案
            gold_answer: 标准答案
            question: 问题（LLM-as-Judge 需要）
            method: 评估方法（默认使用 self.default_method）
            
        Returns:
            (score, info)
        """
        method = method or self.default_method
        
        if method == EvalMethod.EXACT:
            score = self.compute_exact_match(pred_answer, gold_answer)
            return score, {"method": "exact"}
        
        if method == EvalMethod.F1:
            score = self.compute_f1(pred_answer, gold_answer)
            return score, {"method": "f1"}
        
        if method == EvalMethod.ROUGE_L:
            score = self.compute_rouge_l(pred_answer, gold_answer)
            return score, {"method": "rouge_l"}
        
        if method == EvalMethod.LLM_JUDGE:
            if not question:
                logger.warning("LLM-as-Judge requires question, falling back to F1")
                score = self.compute_f1(pred_answer, gold_answer)
                return score, {"method": "f1", "fallback": True}
            
            score, reason, success = await self.evaluate_with_llm_judge(
                question, pred_answer, gold_answer
            )
            if success:
                return score, {"method": "llm_judge", "reason": reason}
            else:
                # Fallback to F1
                score = self.compute_f1(pred_answer, gold_answer)
                return score, {"method": "f1", "fallback": True, "fallback_reason": reason}
        
        # AUTO: 优先 LLM-as-Judge，失败时 fallback
        if method == EvalMethod.AUTO:
            if self.judge_config and question:
                score, reason, success = await self.evaluate_with_llm_judge(
                    question, pred_answer, gold_answer
                )
                if success:
                    return score, {"method": "llm_judge", "reason": reason}
            
            # Fallback to F1
            score = self.compute_f1(pred_answer, gold_answer)
            return score, {"method": "f1", "fallback": True}
        
        # 默认 F1
        score = self.compute_f1(pred_answer, gold_answer)
        return score, {"method": "f1"}
    
    def evaluate_sync(
        self,
        pred_answer: str,
        gold_answer: str,
        question: Optional[str] = None,
        method: Optional[EvalMethod] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        同步评估答案质量
        """
        return asyncio.run(self.evaluate_async(pred_answer, gold_answer, question, method))
    
    async def evaluate_batch_async(
        self,
        items: List[Dict[str, str]],
        method: Optional[EvalMethod] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        批量异步评估
        
        Args:
            items: 评估项列表，每项包含:
                - pred_answer: str
                - gold_answer: str
                - question: Optional[str]
            method: 评估方法
            
        Returns:
            结果列表
        """
        tasks = []
        for item in items:
            task = self.evaluate_async(
                item["pred_answer"],
                item["gold_answer"],
                item.get("question"),
                method,
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    def evaluate_batch_sync(
        self,
        items: List[Dict[str, str]],
        method: Optional[EvalMethod] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        同步批量评估
        """
        return asyncio.run(self.evaluate_batch_async(items, method))


# ============== 便捷函数 ==============

def create_evaluator(
    judge_api_url: Optional[str] = None,
    judge_model_name: Optional[str] = None,
    use_chinese: bool = False,
    **kwargs
) -> AnswerEvaluator:
    """
    创建答案评估器
    
    Args:
        judge_api_url: LLM-as-Judge API 地址（可选）
        judge_model_name: LLM-as-Judge 模型名称（可选）
        use_chinese: 是否使用中文 prompt
        **kwargs: 其他 JudgeConfig 参数
        
    Returns:
        AnswerEvaluator 实例
    """
    judge_config = None
    if judge_api_url and judge_model_name:
        judge_config = JudgeConfig(
            api_url=judge_api_url,
            model_name=judge_model_name,
            **kwargs
        )
    
    return AnswerEvaluator(
        judge_config=judge_config,
        use_chinese_prompt=use_chinese,
    )


def quick_evaluate(
    pred_answer: str,
    gold_answer: str,
    method: str = "f1",
) -> float:
    """
    快速评估（无需创建评估器实例）
    
    Args:
        pred_answer: 预测答案
        gold_answer: 标准答案
        method: 评估方法 ("f1", "rouge_l", "exact")
        
    Returns:
        评分
    """
    if method == "exact":
        return AnswerEvaluator.compute_exact_match(pred_answer, gold_answer)
    elif method == "rouge_l":
        return AnswerEvaluator.compute_rouge_l(pred_answer, gold_answer)
    else:
        return AnswerEvaluator.compute_f1(pred_answer, gold_answer)


# ============== 测试代码 ==============

if __name__ == "__main__":
    # 测试文本相似度
    print("=== Text Similarity Test ===")
    pred = "I love pizza and Italian food"
    gold = "My favorite food is pizza from Italy"
    
    print(f"Pred: {pred}")
    print(f"Gold: {gold}")
    print(f"F1: {AnswerEvaluator.compute_f1(pred, gold):.4f}")
    print(f"ROUGE-L: {AnswerEvaluator.compute_rouge_l(pred, gold):.4f}")
    print(f"Exact: {AnswerEvaluator.compute_exact_match(pred, gold):.4f}")
    
    # 测试中文
    print("\n=== Chinese Text Test ===")
    pred_zh = "我喜欢吃披萨和意大利面"
    gold_zh = "我最喜欢的食物是披萨"
    
    print(f"Pred: {pred_zh}")
    print(f"Gold: {gold_zh}")
    print(f"F1: {AnswerEvaluator.compute_f1(pred_zh, gold_zh):.4f}")
    print(f"ROUGE-L: {AnswerEvaluator.compute_rouge_l(pred_zh, gold_zh):.4f}")
    
    # 测试评估器
    print("\n=== Evaluator Test ===")
    evaluator = create_evaluator()  # 无 LLM-as-Judge
    
    score, info = evaluator.evaluate_sync(pred, gold)
    print(f"Score: {score:.4f}, Info: {info}")
    
    # 批量测试
    print("\n=== Batch Evaluation Test ===")
    items = [
        {"pred_answer": "pizza", "gold_answer": "pizza"},
        {"pred_answer": "I like pizza", "gold_answer": "pizza is my favorite"},
        {"pred_answer": "completely wrong", "gold_answer": "pizza"},
    ]
    
    results = evaluator.evaluate_batch_sync(items)
    for item, (score, info) in zip(items, results):
        print(f"Pred: {item['pred_answer'][:30]}, Gold: {item['gold_answer'][:30]}, Score: {score:.4f}")
