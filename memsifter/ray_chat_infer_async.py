import argparse
import asyncio
import os
import random
import sys
from typing import Dict, Any, List

import pandas as pd
import yaml
from easydict import EasyDict as edict
from loguru import logger
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append("./")

from utils.eval_utils import dedup_indexes
from utils.ray_gen_utils import parse_haystack_sessions
from utils.eval_generation_utils import eval_generation_data
from utils.session_process import construct_history_text, construct_session_text


def load_chat_prompt(dataset_name: str):
    if dataset_name in [
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
    ]:
        default_path = "configs/deepsearch_default_prompt.yaml"
    else:
        default_path = "configs/chat_default_prompt.yaml"
    if os.path.exists(os.path.join("configs", f"chat_{dataset_name}_prompt.yaml")):
        prompt_config_path = os.path.join("configs", f"{dataset_name}_prompt.yaml")
    else:
        prompt_config_path = default_path
    with open(prompt_config_path, "r") as f:
        prompt_config = edict(yaml.safe_load(f))
    return dict(prompt_config)


def construct_retrieved_history_with_limited_context(
    sessions,
    max_prompt_tokens,
    tokenizer,
    session_dates=None,
    sample_strategy="random",
    session_sim_scores=None,
):
    selected_session_indices = []
    total_num_tokens = 0
    search_order = list(range(len(sessions)))
    assert len(sessions) == len(search_order), (
        f"sessions length {len(sessions)} not equal to search_order length {len(search_order)}"
    )
    if sample_strategy == "random":
        random.shuffle(search_order)
    elif sample_strategy == "similarity":
        if session_sim_scores is None:
            raise ValueError(
                "session_sim_scores must be provided when sample_strategy is 'similarity'"
            )
        search_order.sort(key=lambda x: session_sim_scores[x], reverse=True)
    else:
        raise ValueError(f"sample_strategy {sample_strategy} not supported")
    for sid in search_order:
        session_turns = sessions[sid]
        session_text = construct_session_text(session_turns)
        if session_dates is not None:
            session_date = session_dates[sid]
            session_text = f"<session42> <date>{session_date}</date>\n{session_text}</session>\n"
        else:
            session_text = f"<session42>\n{session_text}</session>\n"
        num_tokens = len(tokenizer.tokenize(session_text))
        if total_num_tokens + num_tokens <= max_prompt_tokens:
            selected_session_indices.append(sid)
            total_num_tokens += num_tokens
        else:
            break
    selected_session_indices.sort()
    return selected_session_indices


class ChatInferAsyncActor:
    """Chat inference with AsyncOpenAI and asyncio concurrency (semaphore-limited)."""

    def __init__(
        self,
        prompt_config: Dict[str, Any],
        api_key: str,
        base_url: str,
        model_name: str,
        method: str,
        max_input_tokens: int,
        model_path: str,
        generation_params: Dict[str, Any],
        max_concurrent: int = 64,
    ):
        self.prompt_config = prompt_config
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model_name = model_name
        self.method = method
        self.max_input_tokens = max_input_tokens
        self.generation_params = generation_params
        self.max_concurrent = max_concurrent
        self.semaphore = None  # set in _process_batch_async

    def filter_history_chats(self, row: Dict[str, Any]) -> str:
        if "retrieved_memory" in row:
            return row["retrieved_memory"]
        haystack_sessions = row["haystack_sessions"]
        haystack_session_ids = row["haystack_session_ids"]
        haystack_dates = row.get("haystack_dates", None)
        answer_session_ids = row["answer_session_ids"]
        answer_session_idx = []
        for sid, (session, session_id) in enumerate(zip(haystack_sessions, haystack_session_ids)):
            if session_id in answer_session_ids:
                answer_session_idx.append(sid)
        question = row["question"]
        question_token_num = len(self.tokenizer.tokenize(question))
        haystack_session_ids = row.get("haystack_session_ids", [])
        if self.method in ["bge-m3"]:
            similarities = row["similarities"]
            selected_session_indices = construct_retrieved_history_with_limited_context(
                haystack_sessions,
                self.max_input_tokens - question_token_num,
                self.tokenizer,
                haystack_dates,
                "similarity",
                similarities,
            )
            selected_sessions = [haystack_sessions[sid] for sid in selected_session_indices]
            selected_dates = (
                [haystack_dates[sid] for sid in selected_session_indices]
                if haystack_dates is not None
                else None
            )
            return construct_history_text(selected_sessions, selected_dates)
        elif self.method in ["ep2-DAPO-Qwen3-4B-Thinking"] or "DAPO-GenRank" in self.method:
            similarities = row["similarities"]
            assert len(similarities) == len(haystack_sessions), (
                f"similarities length {len(similarities)} not equal to "
                f"haystack_sessions length {len(haystack_sessions)}"
            )
            rank_scores = similarities.copy()
            haystack_session_gen_rankings = [
                i for i in row["haystack_session_gen_rankings"]
                if i < len(haystack_session_ids)
            ]
            pred_ranking_indexes = dedup_indexes(haystack_session_gen_rankings)
            rank_score = len(pred_ranking_indexes)
            for index in pred_ranking_indexes:
                if index < len(haystack_sessions):
                    rank_scores[index] = rank_score * 10
                    rank_score -= 1
            selected_session_indices = construct_retrieved_history_with_limited_context(
                haystack_sessions,
                self.max_input_tokens - question_token_num,
                self.tokenizer,
                haystack_dates,
                "similarity",
                rank_scores,
            )
            selected_sessions = [haystack_sessions[sid] for sid in selected_session_indices]
            selected_dates = (
                [haystack_dates[sid] for sid in selected_session_indices]
                if haystack_dates is not None
                else None
            )
            return construct_history_text(selected_sessions, selected_dates)
        else:
            logger.warning(f"Method {self.method} not supported.")
            raise ValueError(f"Method {self.method} not supported.")

    async def _retry_request_async(self, messages: List[Dict[str, str]]) -> Any:
        """Retry API request with exponential backoff (async). Retries until success."""
        attempt = 0
        while True:
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    **self.generation_params,
                }
                response = await self.client.chat.completions.create(**kwargs)
                return response
            except Exception as e:
                wait_time = min(2**attempt, 300)
                logger.warning(
                    f"Chat API call failed (attempt {attempt + 1}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
                attempt += 1

    def _parse_answer(self, generated_text: str) -> str:
        if "<answer>" in generated_text:
            try:
                start = generated_text.find("<answer>") + len("<answer>")
                end = generated_text.find("</answer>")
                return generated_text[start:end]
            except Exception:
                logger.info(f"<answer> tag not closed, raw text: {generated_text}")
        return generated_text

    async def _infer_one_async(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Process one row with semaphore-limited concurrency."""
        async with self.semaphore:
            if "history_chats" in row:
                history_chats = row["history_chats"]
            else:
                history_chats = self.filter_history_chats(row)
            question = row["question"]
            user_prompt = self.prompt_config["prompt_template"]["user_prompt_text"].format(
                history_chats=history_chats, question=question
            )
            messages = [
                # {"role": "system", "content": self.prompt_config["prompt_template"]["system_prompt_text"]},
                {"role": "user", "content": user_prompt},
            ]
            try:
                response = await self._retry_request_async(messages)
                generated_text = response.choices[0].message.content or ""
                generated_text = self._parse_answer(generated_text)
                row["generated_text"] = generated_text
                row["api_status"] = "success"
            except Exception as e:
                logger.error(f"Chat API call failed: {e}")
                row["generated_text"] = ""
                row["api_status"] = f"error: {e}"
            return row

    async def _process_batch_async(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of rows concurrently (limited by semaphore)."""
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        # Preserve input order: tag rows with index, then sort results
        indexed = [dict(row, _async_idx=i) for i, row in enumerate(rows)]
        tasks = [asyncio.create_task(self._infer_one_async(r)) for r in indexed]
        results: List[Dict[str, Any]] = [None] * len(rows)
        with tqdm(total=len(tasks), desc="Chat inference", leave=False) as pbar:
            for task in asyncio.as_completed(tasks):
                result = await task
                idx = result.pop("_async_idx")
                results[idx] = result
                pbar.update(1)
        return results

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of rows with asyncio; used by batch loop."""
        rows = batch.to_dict("records")
        if not rows:
            return batch
        results = asyncio.run(self._process_batch_async(rows))
        return pd.DataFrame(results)


def _load_runtime_env_vars() -> Dict[str, str]:
    """Load env_vars from RUNTIME_ENV yaml; fallback to empty dict if not set or missing."""
    runtime_env_path = os.environ.get("RUNTIME_ENV", "")
    if not runtime_env_path or not os.path.isfile(runtime_env_path):
        return {}
    with open(runtime_env_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("env_vars", {}) if isinstance(data, dict) else {}


def chat_inference(args: Dict[str, Any]):
    """Main chat inference with pandas and async concurrency."""
    input_dir = args["input_dir"]
    output_dir = args["output_dir"]
    dataset_name = args["dataset_name"]
    runtime_env_vars = _load_runtime_env_vars()
    api_key = (
        args.get("api_key")
        or runtime_env_vars.get("API_KEY")
        or os.environ.get("API_KEY", "EMPTY")
    )
    base_url = (
        args.get("base_url")
        or runtime_env_vars.get("BASE_URL")
        or os.environ.get("BASE_URL", "")
    )
    model_name = args["model_name"]
    method = args["method"]
    dataset_split = args["dataset_split"]
    debug = args.get("debug", False)
    max_input_tokens = args.get("max_input_tokens", 8192)
    model_path = args.get("model_path", "../models/Qwen3-4B-Instruct")

    if method in ["bge-m3"]:
        input_file = f"{input_dir}/{method}/{dataset_name}_{dataset_split}_embed.parquet"
    elif method in ["hipporag", "nemori", "a_mem"]:
        input_file = f"{input_dir}/{method}/{dataset_name}_{dataset_split}_retrieved_memory.parquet"
    else:
        input_file = f"{input_dir}/{method}/{dataset_name}_{dataset_split}_ranking.parquet"

    output_file = f"{output_dir}/{model_name}/{method}/{dataset_name}_{dataset_split}_chat.parquet"
    if not str(output_dir).startswith("tos://"):
        os.makedirs(f"{output_dir}/{model_name}/{method}", exist_ok=True)

    prompt_config = load_chat_prompt(dataset_name)
    logger.info("Loaded prompt config")

    logger.info(f"Loading input: {input_file}")
    if input_file.endswith(".jsonl"):
        df = pd.read_json(input_file, lines=True)
    elif input_file.endswith(".parquet"):
        df = pd.read_parquet(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file}")

    records = [parse_haystack_sessions(row) for row in df.to_dict("records")]
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} entries from {input_file}")

    if debug:
        df = df.head(10)
        logger.info(f"Debug mode enabled, processing {len(df)} entries")

    infer_batch_size = args.get("infer_batch_size", 128)
    max_concurrent = args.get("max_concurrent", 64)
    infer_actor = ChatInferAsyncActor(
        prompt_config=prompt_config,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        method=method,
        max_input_tokens=max_input_tokens,
        model_path=model_path,
        generation_params={
            "temperature": args.get("temperature", 0.6),
            "max_tokens": args.get("max_output_tokens", 4096),
        },
        max_concurrent=max_concurrent,
    )
    batch_dfs = []
    for start in range(0, len(df), infer_batch_size):
        batch = df.iloc[start : start + infer_batch_size]
        batch_dfs.append(infer_actor(batch))
    df = pd.concat(batch_dfs, ignore_index=True)

    logger.info(f"Saving results to {output_file}")
    if output_file.endswith(".jsonl"):
        df.to_json(output_file, lines=True, orient="records", force_ascii=False)
    elif output_file.endswith(".parquet"):
        df.to_parquet(output_file, index=False, compression="zstd")
    else:
        df.to_json(output_file, lines=True, orient="records", force_ascii=False)
    logger.info(f"Saved to {output_file}")

    is_chinese = dataset_name in ["zh4o", "perltqa_zh"]
    eval_generation_data(df, debug, True, is_chinese)
    logger.info(f"Finished evaluation for {len(df)} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Chat inference with pandas and async OpenAI API (no Ray)"
    )
    parser.add_argument("--input_dir", required=True, help="Input directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--dataset_name", default="default", help="Dataset name for prompt config")
    parser.add_argument("--model_name", default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("--method", default="default", help="Retrieval method")
    parser.add_argument("--num_actors", type=int, default=4, help="Ignored (kept for CLI compatibility)")
    parser.add_argument("--debug", action="store_true", help="Debug mode (limit to 10 rows)")
    parser.add_argument("--local", action="store_true", help="Ignored (kept for CLI compatibility)")
    parser.add_argument("--concurrency", type=int, default=8, help="Ignored; use --max_concurrent")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--max_output_tokens", type=int, default=1024, help="Max output tokens")
    parser.add_argument("--max_input_tokens", type=int, default=8192, help="Max input tokens")
    parser.add_argument("--dataset_split", default="test", help="Dataset split")
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=64,
        help="Max concurrent API requests per batch (semaphore limit)",
    )
    parser.add_argument(
        "--infer_batch_size",
        type=int,
        default=128,
        help="Number of rows per batch for inference",
    )
    parser.add_argument("--model_path", type=str, default="../models/Qwen3-4B-Instruct", help="Tokenizer/model path")
    parser.add_argument("--api_key", type=str, default=None, help="Override API key (default: env API_KEY)")
    parser.add_argument("--base_url", type=str, default=None, help="Override base URL (default: env BASE_URL)")
    args = parser.parse_args()
    chat_inference(vars(args))


if __name__ == "__main__":
    main()
