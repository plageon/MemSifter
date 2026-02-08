"""
GenRank 奖励计算模块

支持两种奖励计算方式：
1. DCG 模式（默认）：基于排名的 DCG 分数，直接比较预测排名与真实答案
2. Paper 模式：论文中的 Net Memory Segment Contribution + Discriminative Retrieved Ranking

通过环境变量 GENRANK_REWARD_MODE 切换：
- GENRANK_REWARD_MODE=dcg (默认)
- GENRANK_REWARD_MODE=paper

Paper 模式需要额外配置：
- WORKING_AGENT_API_URL: Working Agent LLM API 地址
- WORKING_AGENT_MODEL_NAME: Working Agent 模型名称
- EVAL_SCORE_METHOD: 评估方法 (f1, rouge_l, llm_judge)
"""

import os
from typing import Dict, Optional
import numpy as np
from loguru import logger
import sys
sys.path.append("./")

# 奖励模式
REWARD_MODE_DCG = "dcg"
REWARD_MODE_PAPER = "paper"

# 全局配置
_reward_mode: str = os.environ.get("GENRANK_REWARD_MODE", REWARD_MODE_DCG)
_paper_reward_initialized: bool = False


def get_reward_mode() -> str:
    """获取当前奖励模式"""
    global _reward_mode
    return _reward_mode


def set_reward_mode(mode: str):
    """设置奖励模式"""
    global _reward_mode
    if mode not in [REWARD_MODE_DCG, REWARD_MODE_PAPER]:
        raise ValueError(f"Invalid reward mode: {mode}. Must be 'dcg' or 'paper'")
    _reward_mode = mode
    logger.info(f"Reward mode set to: {mode}")


def init_paper_reward_mode(
    working_agent_api_url: Optional[str] = None,
    working_agent_model_name: Optional[str] = None,
    **kwargs
):
    """
    初始化 Paper 奖励模式
    
    Args:
        working_agent_api_url: Working Agent LLM API 地址
        working_agent_model_name: Working Agent 模型名称
        **kwargs: 其他 WorkingAgentConfig 参数
    """
    global _paper_reward_initialized
    
    api_url = working_agent_api_url or os.environ.get("WORKING_AGENT_API_URL")
    model_name = working_agent_model_name or os.environ.get("WORKING_AGENT_MODEL_NAME")
    
    if not api_url or not model_name:
        logger.warning("Paper reward mode requires WORKING_AGENT_API_URL and WORKING_AGENT_MODEL_NAME")
        return
    
    try:
        from genrank_verl.genrank_reward_score_paper import init_working_agent_client
        init_working_agent_client(api_url, model_name, **kwargs)
        _paper_reward_initialized = True
        logger.info(f"Paper reward mode initialized: api_url={api_url}, model={model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize paper reward mode: {e}")


def calculate_dcg(gold_turns: list, pred_turns: list):
    ndcg = 0.0
    for pid, pred_turn in enumerate(pred_turns):
        if pred_turn in gold_turns:
            ndcg += (1.0 / np.log2(pid + 2))
    ndcg /= len(gold_turns)
    return ndcg


def dedup_indexes(ranking_list):
    # 去除重复的session索引，保证第一次出现的顺序不变
    ranking_set =set()
    dedup_ranking_list = []
    for ranking in ranking_list:
        if ranking not in ranking_set:
            ranking_set.add(ranking)
            dedup_ranking_list.append(ranking)
    return dedup_ranking_list


def extract_solution(solution_str: str, max_session_len: int = 1000, max_ranking_len: int = 10) -> Dict[str, str]:
    """Extract the solution from the solution string.
    Extract patterns like: <think>Your thinking process</think><ranking>27,13,34,5,12,8,21,45,6,19</ranking>
    Args:
        solution_str (str): The solution string.

    Returns:
        Dict[str, str]: The extracted solution.
    """
    solution_str = solution_str.strip()
    think_start_pattern = "<think>"
    think_end_pattern = "</think>"
    ranking_start_pattern = "<ranking>"
    ranking_end_pattern = "</ranking>"

    think_start_pos = solution_str.find(think_start_pattern)
    think_end_pos = solution_str.find(think_end_pattern, think_start_pos)
    if think_start_pos == -1 or think_end_pos == -1:
        return {
            "think": "",
            "ranking": "",
            "error": "think pattern not found"
        }
    reasoning_content = solution_str[think_start_pos + len(think_start_pattern):think_end_pos]
    ranking_start_pos = solution_str.find(ranking_start_pattern, think_end_pos)
    ranking_end_pos = solution_str.find(ranking_end_pattern, ranking_start_pos)
    if ranking_start_pos == -1 or ranking_end_pos == -1:
        return {
            "think": reasoning_content,
            "ranking": "",
            "error": "ranking pattern not found"
        }
    ranking_content = solution_str[ranking_start_pos + len(ranking_start_pattern):ranking_end_pos]
    try:
        ranking_list = [int(x.strip()) for x in ranking_content.split(",")]
    except Exception as e:
        return {
            "think": reasoning_content,
            "ranking": ranking_content,
            "error": "ranking content is not a list of integers"
        }
    # 去除重复的session索引，保证第一次出现的顺序不变
    ranking_list = dedup_indexes(ranking_list)
    if len(ranking_list) != max_ranking_len:
        return {
            "think": reasoning_content,
            "ranking": ranking_content,
            "error": f"ranking list length {len(ranking_list)} is not {max_ranking_len}"
        }
    for ranking in ranking_list:
        if ranking >= max_session_len:
            return {
                "think": reasoning_content,
                "ranking": ranking_content,
                "error": f"ranking {ranking} is out of range [0, {max_session_len - 1}]"
            }
    return {
        "think": reasoning_content,
        "ranking": ranking_content,
        "error": ""
    }


def compute_score_instruct(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """

    max_ranking_len = 10
    
    answer_session_ids, max_session_len = ground_truth.split(";")
    max_session_len = int(max_session_len)
    answer_session_ids = [int(x.strip()) for x in answer_session_ids.split(",")]
    ranking_dict = extract_solution(solution_str, max_session_len, min(max_ranking_len, max_session_len))
    format_score = 1.0 / np.log2(2 * min(max_ranking_len, max_session_len)) # 保证format score不超过正常的ndcg
    if ranking_dict["error"] != "":
        return 0.0
    ranking_list = [int(x.strip()) for x in ranking_dict["ranking"].split(",")]
    ranking_list = dedup_indexes(ranking_list)
    ndcg = calculate_dcg(answer_session_ids, ranking_list)

    # if ndcg <= format_score and ranking_dict["error"] == "":
    #     print(f"format score: {format_score}, ndcg: {ndcg}, ranking_dict: {ranking_dict}")
    #     return format_score
    # else:
    #     return ndcg
    return ndcg


def compute_score_thinking(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    solution_str = "<think>" + solution_str
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


# ============== 论文奖励模式计算函数 ==============

def compute_score_paper(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """
    论文奖励模式计算函数
    
    使用 Net Memory Segment Contribution + Discriminative Retrieved Ranking 计算奖励。
    需要 reward_model_info 包含完整信息（question, sessions, task_answer 等）。
    """
    global _paper_reward_initialized
    
    # 检查是否初始化
    if not _paper_reward_initialized:
        # 尝试自动初始化
        init_paper_reward_mode()
    
    try:
        from genrank_verl.genrank_reward_score_paper import compute_score_paper_reward
        
        # 从 kwargs 获取 reward_model_info
        reward_model_info = kwargs.get("reward_model_info", {})
        if not reward_model_info:
            reward_model_info = {"ground_truth": ground_truth}
        
        return compute_score_paper_reward(
            data_source,
            solution_str,
            ground_truth,
            extra_info,
            sandbox_fusion_url,
            concurrent_semaphore,
            memory_limit_mb,
            reward_model_info=reward_model_info,
            **kwargs,
        )
    except Exception as e:
        logger.warning(f"Paper reward computation failed: {e}, falling back to DCG")
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


def compute_score_paper_thinking(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """论文奖励模式 + Thinking 格式"""
    solution_str = "<think>" + solution_str
    return compute_score_paper(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        sandbox_fusion_url,
        concurrent_semaphore,
        memory_limit_mb,
        **kwargs,
    )


# ============== 统一入口函数 ==============

def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """
    统一的奖励计算入口函数
    
    根据环境变量 GENRANK_REWARD_MODE 自动选择计算方式：
    - dcg: 使用 DCG 分数（默认）
    - paper: 使用论文中的 Net Memory Segment Contribution + Discriminative Retrieved Ranking
    
    也可以通过 kwargs 中的 reward_mode 参数指定。
    """
    mode = kwargs.pop("reward_mode", None) or get_reward_mode()
    
    if mode == REWARD_MODE_PAPER:
        return compute_score_paper(
            data_source,
            solution_str,
            ground_truth,
            extra_info,
            sandbox_fusion_url,
            concurrent_semaphore,
            memory_limit_mb,
            **kwargs,
        )
    else:
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


def compute_score_thinking_auto(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """
    自动选择模式的 Thinking 格式奖励计算
    """
    solution_str = "<think>" + solution_str
    return compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        sandbox_fusion_url,
        concurrent_semaphore,
        memory_limit_mb,
        **kwargs,
    )


if __name__ == "__main__":
    # 测试 DCG 模式
    print("=== DCG Mode Test ===")
    data_source = "dapo"
    solution_str = "needs and interests.\n</think>\n\n<ranking>0,3,0,0,3,0,1,2,4,3</ranking>"
    ground_truth = "3;5"
    score = compute_score_thinking(
        data_source,
        solution_str,
        ground_truth,
    )
    print(f"DCG Score: {score}")
    
    # 测试统一入口
    print("\n=== Unified Entry Test ===")
    score2 = compute_score_thinking_auto(
        data_source,
        solution_str,
        ground_truth,
    )
    print(f"Auto Mode Score: {score2}")
    
    # 显示当前模式
    print(f"\nCurrent reward mode: {get_reward_mode()}")

