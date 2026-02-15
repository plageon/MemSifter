import json
import os
from typing import Dict, Any
import argparse

import numpy as np
import pandas as pd
import datetime

import ray
from easydict import EasyDict as edict
import yaml
from transformers import AutoTokenizer
import sys
from loguru import logger

import os
os.environ["RAY_ERROR_PRINT_REPR_LIMIT"] = "100"
sys.path.append("./")
from utils.ray_gen_utils import parse_haystack_sessions
from utils.session_process import construct_history_text, construct_history_text_with_limited_context
from utils.eval_utils import dedup_indexes, calculate_dcg

class RLTrainDataPrepare:
    def __init__(self,
                 dataset_name: str,
                 dataset_split: str,
                 prompt_config: edict,
                 tokenizer_path: str,
                 max_prompt_tokens: int = 1024 * 128,
                 version: str = "v1",
                 ):
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.prompt_config = prompt_config
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_prompt_tokens = max_prompt_tokens
        self.version = version

    @staticmethod
    def _normalize_sessions(sessions: list) -> list:
        """
        Normalize sessions structure, keeping only content and role fields.
        Resolves session schema inconsistencies between different datasets (memory_datasets vs deepsearch_datasets).
        
        Args:
            sessions: Original sessions list, where each session is a list of turns
            
        Returns:
            Normalized sessions, where each turn only contains role and content fields
        """
        normalized = []
        for session in sessions:
            normalized_session = [
                {"role": turn.get("role", ""), "content": turn.get("content", "")}
                for turn in session
            ]
            normalized.append(normalized_session)
        return normalized

    def __call__(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        try:
            question = entry["question"]
            haystack_sessions = entry["haystack_sessions"]
            haystack_session_ids = entry["haystack_session_ids"]
            haystack_dates = entry.get("haystack_dates", None)
            answer_session_ids = entry["answer_session_ids"]  # Session IDs related to the answer
            session_sim_scores = entry.get("similarities", None)
            eidx = entry["index"]

            # locate answer session
            answer_session_idx = []
            for sid, (session, session_id) in enumerate(zip(haystack_sessions, haystack_session_ids)):
                if session_id in answer_session_ids:
                    answer_session_idx.append(sid)

            question_token_num = len(self.tokenizer.tokenize(question))
            selected_session_indices = construct_history_text_with_limited_context(
                haystack_sessions,
                answer_session_idx,
                self.max_prompt_tokens - question_token_num,
                self.tokenizer,
                session_dates=haystack_dates,
                sample_strategy="similarity",
                session_sim_scores=session_sim_scores,
            )
            selected_sessions = [haystack_sessions[sid] for sid in selected_session_indices]
            selected_session_ids = [haystack_session_ids[sid] for sid in selected_session_indices]
            assert all([asid in selected_session_ids for asid in answer_session_ids]), \
                f"answer_session_ids {answer_session_ids} not in selected_session_ids {selected_session_ids}"
            if haystack_dates is not None:
                selected_dates = [haystack_dates[sid] for sid in selected_session_indices]
            else:
                selected_dates = None
            history = construct_history_text(selected_sessions, selected_dates)
            content = self.prompt_config.prompt_template.user_prompt_text.format(history=history, current=question)

            # update answer_session_idx to selected_session_idx
            asidx_selected = []
            for sid, (session, session_id) in enumerate(zip(selected_sessions, selected_session_ids)):
                if session_id in answer_session_ids:
                    asidx_selected.append(str(sid))
            if len(asidx_selected) == 0:
                logger.info(f"answer_session_ids {answer_session_ids}")
                logger.info(f"selected_session_ids {selected_session_ids}")
                raise

            # Get task ground truth answer (if exists)
            task_answer = entry.get("answer", entry.get("gold_answer", ""))
            
            # Normalize sessions structure, keeping only content and role fields
            # Resolves session schema inconsistencies that cause concatenate_datasets to fail
            normalized_sessions = self._normalize_sessions(selected_sessions)
            
            return {
                "data_source": f"genrank_{self.dataset_name}",
                "prompt": [{
                    "role": "user",
                    "content": content
                }],
                "ability": "memory_ranking",
                "reward_model": {
                    # Original fields: answer session indices and total session count
                    "ground_truth": ",".join(asidx_selected) + f";{len(selected_session_ids)}",
                    # New fields: for Working Agent-based reward calculation in the paper
                    "question": question,                          # Original question
                    "sessions": normalized_sessions,               # Normalized session content (only role and content)
                    "session_dates": selected_dates,               # Session dates (optional, list or None)
                    "task_answer": task_answer,                    # Task ground truth answer (for evaluating s_k)
                    "answer_session_idx": [int(i) for i in asidx_selected],  # Answer-related session indices (list of integers)
                },
                "extra_info": {
                    'split': self.dataset_split,
                    'index': eidx,
                    'prompt_name': self.prompt_config.prompt_name,
                    'version': self.version,
                    'max_prompt_tokens': self.max_prompt_tokens,
                }
            }
        except Exception as e:
            # Only print key information for debugging, avoid printing entire large data rows
            eidx = entry.get("index", "unknown")
            question = entry.get("question", "")[:100]  # Truncate question
            logger.error(f"Error processing entry index={eidx}, question={question}...")
            logger.error(f"Exception: {type(e).__name__}: {e}")
            raise RuntimeError(f"Failed at index={eidx}") from e


class EvalSelectActor:
    def __init__(self,
                 dataset_name: str,
                 dataset_split: str,
                 anchor: float=0.2,
                 pred_n: int=10,
                 ):
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.anchor = anchor
        self.pred_n = pred_n

    def __call__(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        try:
            question = entry["question"]
            haystack_sessions = entry["haystack_sessions"]
            haystack_session_ids = entry["haystack_session_ids"]
            haystack_session_gen_rankings = [i for i in entry['haystack_session_gen_rankings'] if i < len(haystack_session_ids)]
            pred_rankings = dedup_indexes([haystack_session_ids[i] for i in haystack_session_gen_rankings])
            gold_sessions = entry['answer_session_ids']

            if len(pred_rankings) == 0:
                pred_rankings = haystack_session_ids
            ndcg_score = calculate_dcg(gold_turns=gold_sessions[:self.pred_n], pred_turns=pred_rankings[:self.pred_n])

            entry['ndcg'] = ndcg_score
            train_priority = 1 - abs(ndcg_score - self.anchor) # higher priority for ndcg score close to anchor
            entry['train_priority'] = train_priority

            return entry
        except Exception as e:
            # Only print key information for debugging, avoid printing entire large data rows
            eidx = entry.get("index", "unknown")
            logger.error(f"Error in EvalSelectActor at index={eidx}")
            logger.error(f"Exception: {type(e).__name__}: {e}")
            raise RuntimeError(f"Failed at index={eidx}") from e


def rl_train_data_prepare(
    data_dir: str,
    dataset_name: str,
    dataset_split: str,
    prompt_config: edict,
    tokenizer: AutoTokenizer,
    tokenizer_path: str,
    num_samples: int = None,
    max_prompt_tokens: int = 1024*128,
    version: str = "v1",
    sample_strategy: str = "anchor",  # "anchor" or "random"
):
    input_file_name = f"{data_dir}/{dataset_name}_{dataset_split}_ranking.parquet"
    if not os.path.exists(input_file_name):
        input_file_name = f"{data_dir}/{dataset_name}_{dataset_split}_embed.parquet"
        logger.info(f"result file not found, trying {input_file_name}")

    # with open(input_file_name, "r") as f:
    #     input_lines = [json.loads(line) for line in f]


    # add index
    # for eidx, entry in enumerate(input_lines):
    #     entry["index"] = eidx
    #
    # logger.info(f"Loaded {len(input_lines)} lines from {input_file_name}")

    template_token_num = len(tokenizer.tokenize(prompt_config.prompt_template.user_prompt_text))
    ds = ray.data.read_parquet(input_file_name).map(parse_haystack_sessions)
    # add index
    ds_count = ds.count()
    idx_ds = ray.data.range(ds_count).map(lambda x: {"index": x})
    ds = ds.zip(idx_ds)

    logger.info(f"Loaded {ds_count} lines from {input_file_name}")
    logger.info(f"Template token num: {template_token_num}")
    # logger.info(f"Loaded {ds.count()} lines from {input_file_name}")

    if num_samples < ds_count:
        if sample_strategy == "anchor":
            # Use anchor strategy: determine priority based on distance between NDCG score and anchor
            logger.info(f"Using anchor sampling strategy for {dataset_name} {dataset_split}")
            ds = ds.map(
                fn=EvalSelectActor,
                fn_constructor_kwargs={
                    "dataset_name": dataset_name,
                    "dataset_split": dataset_split,
                    "anchor": 0.2,
                    "pred_n": 10,
                },
                num_cpus=4,
                concurrency=(8, 32),
            )
            # sort by train_priority
            priorities_ds = ds.map(lambda row: {"p": row["train_priority"]})
            all_priorities = [row["p"] for row in priorities_ds.iter_rows()]
            all_priorities.sort(reverse=True)
            # Prevent out-of-bounds, take min
            k_index = min(num_samples, len(all_priorities)) - 1
            threshold = all_priorities[k_index]
            logger.info(f"Filtering with priority threshold: {threshold}")

            ds = ds.filter(lambda row: row["train_priority"] >= threshold)
            ds = ds.limit(num_samples)
        else:
            # Use random sampling strategy
            logger.info(f"Using random sampling strategy for {dataset_name} {dataset_split}")
            fraction = num_samples / ds_count
            ds = ds.random_sample(fraction=fraction)
        logger.info(f"Sampled {ds.count()} lines from {dataset_name} {dataset_split}")
    else:
        logger.info(f"Take all {ds_count} lines from {dataset_name} {dataset_split}")

    ds = ds.map(
        fn = RLTrainDataPrepare,
        fn_constructor_kwargs={
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "prompt_config": prompt_config,
            "tokenizer_path": tokenizer_path,
            "max_prompt_tokens": max_prompt_tokens - template_token_num,
            "version": version,
        },
        num_cpus=4,
        concurrency=(8, 32),
    )
    return ds



def load_dataset_recipe(recipe_path: str) -> Dict[str, Dict[str, int]]:
    """
    Load dataset recipe from configuration file
    
    Args:
        recipe_path: Path to yaml configuration file
        
    Returns:
        Merged dataset recipe in format:
        {"train": {"dataset_name": num_samples, ...}, "test": {...}}
    """
    with open(recipe_path, "r") as f:
        recipe_config = yaml.safe_load(f)
    
    # Merge memory_datasets and deepsearch_datasets
    included_dataset_recipe = {"train": {}, "test": {}}
    
    for split in ["train", "test"]:
        # Add memory datasets
        if "memory_datasets" in recipe_config and split in recipe_config["memory_datasets"]:
            included_dataset_recipe[split].update(recipe_config["memory_datasets"][split])
        
        # Add deepsearch datasets
        if "deepsearch_datasets" in recipe_config and split in recipe_config["deepsearch_datasets"]:
            included_dataset_recipe[split].update(recipe_config["deepsearch_datasets"][split])
        
        # Compatible with old format: directly use train/test fields
        if split in recipe_config and isinstance(recipe_config[split], dict):
            for dataset_name, num_samples in recipe_config[split].items():
                if dataset_name not in included_dataset_recipe[split]:
                    included_dataset_recipe[split][dataset_name] = num_samples
    
    logger.info(f"Loaded dataset recipe from {recipe_path}")
    logger.info(f"Train datasets: {list(included_dataset_recipe['train'].keys())}")
    logger.info(f"Test datasets: {list(included_dataset_recipe['test'].keys())}")
    
    return included_dataset_recipe


def main():
    parser = argparse.ArgumentParser(description="RL Training Data Preparation")
    parser.add_argument(
        "--recipe", 
        type=str, 
        default="configs/dataset_recipe_deepsearch_v1.yaml",
        help="Path to dataset recipe yaml file (e.g., configs/dataset_recipe_deepsearch_v1.yaml)"
    )
    parser.add_argument(
        "--primary_data_dir",
        type=str,
        default="../data/results/DAPO-GenRank/ep3-DAPO-Qwen3-4B-Thinking-merged",
        help="Primary data directory (uses anchor sampling if file exists)"
    )
    parser.add_argument(
        "--fallback_data_dir",
        type=str,
        default="../data/results/bge-m3",
        help="Fallback data directory (uses random sampling if primary file not found)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data",
        help="Output directory"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version string (default: v{MMDD}-0)"
    )
    args = parser.parse_args()
    
    output_dir = args.output_dir
    version = args.version if args.version else f"v{datetime.datetime.now().strftime('%m%d')}-0"
    output_file_name = f"{output_dir}/rl_train_data/{version}"
    os.makedirs(output_file_name, exist_ok=True)
    
    # Load dataset recipe from configuration file
    included_dataset_recipe = load_dataset_recipe(args.recipe)

    prompt_name = "gen_retrieve_instruct_prompt.yaml"
    prompt_config = edict(yaml.load(open(f"./configs/{prompt_name}"), Loader=yaml.FullLoader))
    model_path = "../models/Qwen3-4B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    for dataset_split in ["train", "test"]:
        split_data = []
        for dataset_name, num_samples in included_dataset_recipe[dataset_split].items():
            if num_samples > 0:
                # Build file path
                primary_file = f"{args.primary_data_dir}/{dataset_name}_{dataset_split}_ranking.parquet"
                
                # Check file existence to determine data directory and sampling strategy
                if os.path.exists(primary_file):
                    data_dir = args.primary_data_dir
                    sample_strategy = "anchor"
                    logger.info(f"Found {dataset_name} in primary dir, using anchor strategy")
                else:
                    data_dir = args.fallback_data_dir
                    sample_strategy = "random"
                    logger.info(f"File not found in primary dir for {dataset_name}, fallback to {data_dir}, using random strategy")
                
                split_data_ds = (
                    rl_train_data_prepare(
                        data_dir=data_dir,
                        dataset_name=dataset_name,
                        dataset_split=dataset_split,
                        prompt_config=prompt_config,
                        tokenizer=tokenizer,
                        tokenizer_path=model_path,
                        num_samples=num_samples,
                        max_prompt_tokens=1024*130,
                        version=version,
                        sample_strategy=sample_strategy,
                    )
                )

                split_data.append(split_data_ds)

        # convert to pandas
        split_data = pd.concat([ds.to_pandas() for ds in split_data], axis=0)

        if dataset_split == "test":
            # save to parquet
            split_data.to_parquet(f"{output_file_name}/{dataset_split}.parquet", index=False)
            logger.info(f"Saved {len(split_data)} lines to {output_file_name}/{dataset_split}.parquet")

        else:
            # Split train file, 1000 lines per file
            MAX_LINES_PER_FILE = 2000
            # Shuffle
            split_data = split_data.sample(frac=1, random_state=42)
            num_files = len(split_data) // MAX_LINES_PER_FILE + 1
            num_rows = len(split_data)
            num_rows_per_file = num_rows // num_files
            split_indices = np.arange(0, num_rows, num_rows_per_file)
            for i, start_index in enumerate(split_indices):
                end_index = start_index + num_rows_per_file
                df_chunk = split_data.iloc[start_index:end_index]
                filename = f"{output_file_name}/{dataset_split}_{i}.parquet"
                df_chunk.to_parquet(filename, index=False)
                logger.info(f"Saved {len(df_chunk)} lines to {filename}")


if __name__ == "__main__":
    ray.data.DataContext.enable_progress_bars = True
    ray.data.DataContext.enable_operator_progress_bars = True
    main()
