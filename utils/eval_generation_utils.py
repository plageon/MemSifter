import re
from typing import Dict, Any, List

# 初始化jieba分词
import jieba
import numpy as np
import pandas as pd
import ray
from loguru import logger
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pandas import DataFrame


# 判断文本是否包含中文字符
def contains_chinese(text: str) -> bool:
    """检查文本是否包含中文字符"""
    return bool(re.search(r'[\u4e00-\u9fa5]', text))

# 分词函数
def tokenize_text(text: str, is_chinese: bool = False) -> List[str]:
    """根据文本语言选择合适的分词方式"""
    if is_chinese:
        # 使用 jieba 精确模式进行中文分词
        # cut 返回的是一个生成器，用 list() 转换为列表
        return list(jieba.cut(text))
    else:
        # 英文使用 nltk 分词
        words = word_tokenize(text)
        # convert to lowercase
        return [w.lower() for w in words if w.isalpha()]

# 计算Exact Match
def calculate_exact_match(pred: str, gold: str, is_chinese: bool = False) -> float:
    """计算精确匹配分数"""
    return 1.0 if pred.strip() == gold.strip() else 0.0

# 计算F1 Score
def calculate_f1(pred: str, gold: str, is_chinese: bool = False) -> float:
    """计算F1分数"""
    pred_tokens = set(tokenize_text(pred.strip(), is_chinese))
    gold_tokens = set(tokenize_text(gold.strip(), is_chinese))
    
    if not pred_tokens and not gold_tokens:
        return 1.0
    
    # 计算精确率和召回率
    precision = len(pred_tokens.intersection(gold_tokens)) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(pred_tokens.intersection(gold_tokens)) / len(gold_tokens) if gold_tokens else 0.0
    
    # 计算F1分数
    if precision + recall == 0:
        return 0.0
    else:
        return 2 * precision * recall / (precision + recall)

# 计算BLEU分数
def calculate_bleu(pred: str, gold: str, is_chinese: bool = False) -> float:
    """计算BLEU分数"""
    pred_tokens = tokenize_text(pred.strip(), is_chinese)
    gold_tokens = [tokenize_text(gold.strip(), is_chinese)]
    
    if not pred_tokens:
        return 0.0
    
    # 使用平滑函数避免除零错误
    smooth_fn = SmoothingFunction().method1
    try:
        return sentence_bleu(gold_tokens, pred_tokens, smoothing_function=smooth_fn)
    except Exception as e:
        logger.warning(f"BLEU计算错误: {e}")
        return 0.0

metric_func = {
        "exact_match": calculate_exact_match,
        "f1_score": calculate_f1,
        "bleu_score": calculate_bleu,
    }


class GenEvalActor:
    """评测生成结果的Ray Actor"""
    
    def __init__(self, is_chinese: bool = False):
        # 初始化评测器
        self.is_chinese = is_chinese
    
    def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """评测单条生成结果"""
        try:
            pred = row.get("generated_text", "").strip()
            gold = row.get("answer", "").strip()
            
            # 计算各项指标
            row["exact_match"] = calculate_exact_match(pred, gold)
            row["f1_score"] = calculate_f1(pred, gold)
            row["bleu_score"] = calculate_bleu(pred, gold)
            
            return row
            
        except Exception as e:
            logger.error(f"评测过程中发生错误: {e}")
            row["exact_match"] = 0.0
            row["f1_score"] = 0.0
            row["bleu_score"] = 0.0
            row["eval_error"] = str(e)
            return row


def batch_evaluate(data: List[Dict[str, Any]], num_cpus: int = 4) -> List[Dict[str, Any]]:
    """批量评测生成结果"""
    # 启动本地Ray集群
    ray.init(num_cpus=num_cpus)
    
    # 创建Actor实例
    eval_actors = [GenEvalActor.remote() for _ in range(num_cpus)]
    
    # 分发任务
    results = []
    for i, row in enumerate(data):
        actor = eval_actors[i % num_cpus]
        results.append(actor.__call__.remote(row))
    
    # 收集结果
    results = ray.get(results)
    
    # 关闭Ray集群
    ray.shutdown()
    
    return results


def eval_generation_data(data: DataFrame, debug: bool = False, verbose: bool = True, is_chinese: bool = False) -> DataFrame:
    """评测生成数据"""
    if debug:
        data = data.head(10)
    metrics = ["exact_match", "f1_score", "bleu_score"]

    pred_results = {metric: [] for metric in metrics}
    for eidx, entry in data.iterrows():
        answer = entry["answer"]
        generated_text = entry["generated_text"]
        for metric in metrics:
            pred_results[metric].append(metric_func[metric](generated_text, answer, is_chinese))

    avg_pred_results = {
        metric: np.mean(pred_results[metric]) for metric in metrics
    }

    if verbose:
        # print a markdown table
        markdown_table = "| metric | score |\n"
        markdown_table += "| ------ | ------ |\n"
        for metric in metrics:
            markdown_table += f"| {metric} | {avg_pred_results[metric]:.4f} |\n"
        print(markdown_table)
    return avg_pred_results


if __name__ == "__main__":
    # 示例用法
    sample_data = [
        {
            "generated_text": "The capital of China is Beijing.",
            "answer": "The capital of China is Beijing."
        },
        {
            "generated_text": "Paris is the capital of France.",
            "answer": "London is the capital of the United Kingdom."
        }
    ]
    
    # 评测结果
    df = pd.DataFrame(sample_data)
    results = eval_generation_data(df, verbose=True, is_chinese=False)

    chinese_sample_data = [
        {
            "generated_text": "北京是中国的首都",
            "answer": "北京是中华人民共和国的首都"
        },
        {
            "generated_text": "巴黎是法国的首都",
            "answer": "伦敦是英国的首都"
        }
    ]

    # 评测结果
    df = pd.DataFrame(chinese_sample_data)
    results = eval_generation_data(df, verbose=True, is_chinese=True)
