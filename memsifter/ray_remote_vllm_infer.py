import argparse
import asyncio
import os
import sys
from typing import Dict, Any, List


import pandas as pd
import yaml
from easydict import EasyDict as edict
from loguru import logger
from openai import AsyncOpenAI
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append("./")

from utils.ray_gen_utils import (
    AddPromptActor,
    PreFilterActor,
    parse_haystack_sessions,
    dump_haystack_sessions,
)
from utils.eval_utils import RankingParserActor, eval_prob_data
from utils.tos_utils import Cfg, get_filesystem


class RemoteVLLMInferActor:
    """Calls remote vLLM (OpenAI-compatible API) for ranking inference with asyncio concurrency and semaphore."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        generation_params: Dict[str, Any],
        max_concurrent: int = 64,
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.generation_params = generation_params
        self.max_concurrent = max_concurrent
        # Semaphore must be created inside the same event loop that runs the tasks.
        # Each __call__ uses asyncio.run() which creates a new loop, so we create
        # the semaphore at the start of _process_batch_async instead of here.
        self.semaphore = None  # set in _process_batch_async
        # #region agent log
        try:
            _loop_init = asyncio.get_event_loop() if hasattr(asyncio, "get_event_loop") else None
            _sem_loop = getattr(self.semaphore, "_loop", None)
            _debug_log("ray_remote_vllm_infer.py:__init__", "Semaphore created", {"loop_at_init_id": id(_loop_init), "semaphore_loop_id": id(_sem_loop) if _sem_loop is not None else None}, "H1")
        except Exception as e:
            _debug_log("ray_remote_vllm_infer.py:__init__", "init log error", {"error": str(e)}, "H1")
        # #endregion

    async def _retry_request_async(self, messages: List[Dict[str, str]]) -> Any:
        """Retry API request with exponential backoff (async). Retries indefinitely until success."""
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
                wait_time = min(2**attempt, 300)  # cap at 300s to avoid unbounded sleep
                logger.warning(
                    f"Remote vLLM API call failed (attempt {attempt + 1}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
                attempt += 1

    async def _infer_one_async(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Process one row with semaphore-limited concurrency."""
        # #region agent log
        _run_loop = asyncio.get_running_loop()
        _sem_loop = getattr(self.semaphore, "_loop", None)
        _debug_log("ray_remote_vllm_infer.py:_infer_one_async", "Before acquire", {"running_loop_id": id(_run_loop), "semaphore_loop_id": id(_sem_loop) if _sem_loop is not None else None, "match": _run_loop is _sem_loop}, "H3,H5")
        # #endregion
        async with self.semaphore:
            messages = row["messages"]
            try:
                response = await self._retry_request_async(messages)
                generated_text = response.choices[0].message.content or ""
                row["generated_text"] = generated_text
                row["response"] = generated_text
            except Exception as e:
                logger.error(f"Remote vLLM inference failed: {e}")
                row["generated_text"] = ""
                row["response"] = ""
            return row

    async def _process_batch_async(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of rows concurrently (limited by semaphore)."""
        # Create semaphore in this event loop so it is bound to the same loop as the tasks.
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        # #region agent log
        _run_loop = asyncio.get_running_loop()
        _sem_loop = getattr(self.semaphore, "_loop", None)
        _debug_log("ray_remote_vllm_infer.py:_process_batch_async", "Running loop vs semaphore loop", {"running_loop_id": id(_run_loop), "semaphore_loop_id": id(_sem_loop) if _sem_loop is not None else None, "match": _run_loop is _sem_loop}, "H3")
        # #endregion
        tasks = [asyncio.create_task(self._infer_one_async(row.copy())) for row in rows]
        results: List[Dict[str, Any]] = []
        with tqdm(total=len(tasks), desc="Remote vLLM inference", leave=False) as pbar:
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                pbar.update(1)
        return results

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of rows with asyncio; used by map_batches."""
        rows = batch.to_dict("records")
        if not rows:
            return batch
        # #region agent log
        _debug_log("ray_remote_vllm_infer.py:__call__", "Before asyncio.run", {"batch_rows": len(rows)}, "H2")
        # #endregion
        results = asyncio.run(self._process_batch_async(rows))
        return pd.DataFrame(results)


def _apply_row_fn(df: pd.DataFrame, fn) -> pd.DataFrame:
    """Apply a row-wise function (row dict -> row dict) to a DataFrame."""
    records = [fn(row) for row in df.to_dict("records")]
    return pd.DataFrame(records)


def remote_vllm_inference(args: Dict[str, Any]):
    dataset_name = args["dataset_name"]
    dataset_split = args["dataset_split"]
    model_name = args["model_name"]
    debug = args["debug"]
    embed_model = args["embed_model"]
    output_dir = args["output_dir"]

    api_key = args.get("api_key") or os.environ["API_KEY"]
    base_url = args.get("base_url") or os.environ["BASE_URL"]

    data_file = f"{output_dir}/{embed_model}/{dataset_name}_{dataset_split}_embed.parquet"
    output_file = f"{output_dir}/{model_name}/{dataset_name}_{dataset_split}_ranking.parquet"
    if not str(output_dir).startswith("tos://"):
        os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)

    if str(data_file).startswith("tos://"):
        config_path = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "tos_config.yaml")
        )
        cfg = Cfg.load(config_path)
        data_fs = get_filesystem(data_file, cfg)
        with data_fs.open(data_file, "rb") as f:
            df = pd.read_parquet(f)
    else:
        df = pd.read_parquet(data_file)

    # parse_haystack_sessions: row["haystack_sessions"] = json.loads(row["haystack_sessions"])
    records = [parse_haystack_sessions(row) for row in df.to_dict("records")]
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} entries from {data_file}")
    if debug:
        df = df.head(10)
        logger.info(f"Debug mode enabled, only processing {len(df)} entries")

    prompt_name = args["prompt_name"]
    prompt_config = edict(yaml.load(open(f"./configs/{prompt_name}"), Loader=yaml.FullLoader))
    context_limit = 1024 * int(args["context_limit"].replace("k", ""))

    prefilter = PreFilterActor(
        tokenizer_name=args["model_path"],
        prompt_template=prompt_config["prompt_template"],
        filter_col=args["filter_col"],
        context_limit=context_limit,
    )
    df = _apply_row_fn(df, prefilter)

    add_prompt = AddPromptActor(
        prompt_config=prompt_config,
        filtered_idx_col="prefiltered_session_idx",
    )
    df = _apply_row_fn(df, add_prompt)

    max_concurrent = args.get("max_concurrent", 64)
    infer_batch_size = args.get("infer_batch_size", 128)
    infer_actor = RemoteVLLMInferActor(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        generation_params={
            "temperature": args.get("temperature", 0.6),
            "max_tokens": args.get("max_output_token", 4096),
        },
        max_concurrent=max_concurrent,
    )
    batch_dfs = []
    for start in range(0, len(df), infer_batch_size):
        batch = df.iloc[start : start + infer_batch_size]
        batch_dfs.append(infer_actor(batch))
    df = pd.concat(batch_dfs, ignore_index=True)

    ranking_parser = RankingParserActor(filtered_idx_col="prefiltered_session_idx")
    df = _apply_row_fn(df, ranking_parser)
    records = [dump_haystack_sessions(row) for row in df.to_dict("records")]
    df = pd.DataFrame(records)

    if str(output_file).startswith("tos://"):
        config_path = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "tos_config.yaml")
        )
        cfg = Cfg.load(config_path)
        output_fs = get_filesystem(output_file, cfg)
        with output_fs.open(output_file, "wb") as f:
            df.to_parquet(f, engine="pyarrow", index=False, compression="zstd")
    else:
        df.to_parquet(output_file, index=False, compression="zstd")
    results = df.to_dict("records")
    logger.info(f"Total {len(df)} entries processed, saved to {output_file}")

    logger.info(f"Start eval {output_file}")
    eval_prob_data(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote vLLM ranking inference (OpenAI-compatible API)")
    parser.add_argument("--local", action="store_true", help="Use local Ray cluster")
    parser.add_argument("--prompt_name", type=str, required=True, help="Prompt config file name")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split")
    parser.add_argument("--embed_model", type=str, required=True, help="Embedding model name")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--override_num", type=int, required=True, help="Concurrency for read/map stages")
    parser.add_argument("--model_path", type=str, required=True, help="Tokenizer/model path for PreFilter")
    parser.add_argument("--model_name", type=str, required=True, help="API model name for remote vLLM")
    parser.add_argument("--max_output_token", type=int, required=True, help="Max output tokens")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--num_actors", type=int, default=4, help="Number of Ray actors for remote inference")
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=64,
        help="Max concurrent API requests per actor (semaphore limit)",
    )
    parser.add_argument(
        "--infer_batch_size",
        type=int,
        default=128,
        help="Batch size per actor for map_batches (rows per batch)",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode (limit to 10 rows)")
    parser.add_argument("--filter_col", type=str, default="similarities", help="Filter column name")
    parser.add_argument("--context_limit", type=str, default="128k", help="Context limit (e.g. 128k)")
    parser.add_argument("--api_key", type=str, default=None, help="Override API key (default: env API_KEY)")
    parser.add_argument("--base_url", type=str, default=None, help="Override base URL (default: env BASE_URL)")
    args = parser.parse_args()
    p = vars(args)
    # --local and --num_actors kept for CLI compatibility; no-op (no Ray)
    remote_vllm_inference(args=p)
