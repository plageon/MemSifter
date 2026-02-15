import re
from typing import Dict, Any, List

# Initialize jieba tokenizer
import jieba
import numpy as np
import pandas as pd
import ray
from loguru import logger
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pandas import DataFrame


# Check if text contains Chinese characters
def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters"""
    return bool(re.search(r'[\u4e00-\u9fa5]', text))

# Tokenization function
def tokenize_text(text: str, is_chinese: bool = False) -> List[str]:
    """Choose appropriate tokenization method based on text language"""
    if is_chinese:
        # Use jieba precise mode for Chinese tokenization
        # cut returns a generator, convert to list with list()
        return list(jieba.cut(text))
    else:
        # Use nltk for English tokenization
        words = word_tokenize(text)
        # convert to lowercase
        return [w.lower() for w in words if w.isalpha()]

# Calculate Exact Match
def calculate_exact_match(pred: str, gold: str, is_chinese: bool = False) -> float:
    """Calculate exact match score"""
    return 1.0 if pred.strip() == gold.strip() else 0.0

# Calculate F1 Score
def calculate_f1(pred: str, gold: str, is_chinese: bool = False) -> float:
    """Calculate F1 score"""
    pred_tokens = set(tokenize_text(pred.strip(), is_chinese))
    gold_tokens = set(tokenize_text(gold.strip(), is_chinese))
    
    if not pred_tokens and not gold_tokens:
        return 1.0
    
    # Calculate precision and recall
    precision = len(pred_tokens.intersection(gold_tokens)) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(pred_tokens.intersection(gold_tokens)) / len(gold_tokens) if gold_tokens else 0.0
    
    # Calculate F1 score
    if precision + recall == 0:
        return 0.0
    else:
        return 2 * precision * recall / (precision + recall)

# Calculate BLEU score
def calculate_bleu(pred: str, gold: str, is_chinese: bool = False) -> float:
    """Calculate BLEU score"""
    pred_tokens = tokenize_text(pred.strip(), is_chinese)
    gold_tokens = [tokenize_text(gold.strip(), is_chinese)]
    
    if not pred_tokens:
        return 0.0
    
    # Use smoothing function to avoid division by zero error
    smooth_fn = SmoothingFunction().method1
    try:
        return sentence_bleu(gold_tokens, pred_tokens, smoothing_function=smooth_fn)
    except Exception as e:
        logger.warning(f"BLEU calculation error: {e}")
        return 0.0

metric_func = {
        "exact_match": calculate_exact_match,
        "f1_score": calculate_f1,
        "bleu_score": calculate_bleu,
    }


class GenEvalActor:
    """Ray Actor for evaluating generation results"""
    
    def __init__(self, is_chinese: bool = False):
        # Initialize evaluator
        self.is_chinese = is_chinese
    
    def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate single generation result"""
        try:
            pred = row.get("generated_text", "").strip()
            gold = row.get("answer", "").strip()
            
            # Calculate all metrics
            row["exact_match"] = calculate_exact_match(pred, gold)
            row["f1_score"] = calculate_f1(pred, gold)
            row["bleu_score"] = calculate_bleu(pred, gold)
            
            return row
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            row["exact_match"] = 0.0
            row["f1_score"] = 0.0
            row["bleu_score"] = 0.0
            row["eval_error"] = str(e)
            return row


def batch_evaluate(data: List[Dict[str, Any]], num_cpus: int = 4) -> List[Dict[str, Any]]:
    """Batch evaluate generation results"""
    # Start local Ray cluster
    ray.init(num_cpus=num_cpus)
    
    # Create Actor instances
    eval_actors = [GenEvalActor.remote() for _ in range(num_cpus)]
    
    # Distribute tasks
    results = []
    for i, row in enumerate(data):
        actor = eval_actors[i % num_cpus]
        results.append(actor.__call__.remote(row))
    
    # Collect results
    results = ray.get(results)
    
    # Shutdown Ray cluster
    ray.shutdown()
    
    return results


def eval_generation_data(data: DataFrame, debug: bool = False, verbose: bool = True, is_chinese: bool = False) -> DataFrame:
    """Evaluate generation data"""
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
    # Example usage
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
    
    # Evaluation results
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

    # Evaluation results
    df = pd.DataFrame(chinese_sample_data)
    results = eval_generation_data(df, verbose=True, is_chinese=True)
