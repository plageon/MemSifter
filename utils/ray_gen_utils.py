import json
import os.path
import sys
from typing import Optional, Dict, Any

import numpy as np
import ray
from easydict import EasyDict as edict
from loguru import logger
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor, Processor
from transformers import AutoTokenizer

sys.path.append("./")
# from utils.gen_infer_utils import TokenIdNode
from utils.session_process import construct_history_text, construct_session_text, construct_history_text_with_limited_context

def start_local_ray(envs: Dict = None):
    if envs and envs.get("device_ids", ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = envs["device_ids"]
    env_vars = {"OMP_NUM_THREADS": "16",
                "MKL_NUM_THREADS": "16"}
    working_dir = os.path.dirname(os.path.abspath(__file__)) + "/../"
    logger.info(f"start local ray working dir {working_dir}")
    runtime_env = {
        "env_vars": env_vars,
        "working_dir": working_dir,
        "py_modules": ["./utils"],
        "excludes": ["**/*log*", "log_*", "*.log", "*.jsonl", "*.parquet", "*.json"]
    }
    ray.init(
        runtime_env=runtime_env,
    )
    ray.data.DataContext.enable_progress_bars = True
    ray.data.DataContext.enable_operator_progress_bars = True


def parse_haystack_sessions(row):
    turn_keys = ["role", "content"]
    row["haystack_sessions"] = json.loads(row["haystack_sessions"])
    return row


def dump_haystack_sessions(row):
    row["haystack_sessions"] = json.dumps(row["haystack_sessions"], ensure_ascii=False)
    return row


class AddPromptActor():
    def __init__(self,
                 prompt_config: edict,
                 filtered_idx_col: str,
                 ):
        self.prompt_template = prompt_config.prompt_template
        self.filtered_idx_col = filtered_idx_col

    def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
        haystack_sessions = row["haystack_sessions"]
        haystack_dates = row.get("haystack_dates", None)
        if self.filtered_idx_col:
            prefiltered_session_idx = row[self.filtered_idx_col]
            haystack_sessions = [haystack_sessions[idx] for idx in prefiltered_session_idx]
            if haystack_dates is not None:
                haystack_dates = [haystack_dates[idx] for idx in prefiltered_session_idx]
        question = row["question"]
        history = construct_history_text(haystack_sessions, session_dates=haystack_dates)
        user_prompt = self.prompt_template.user_prompt_text.format(history=history, current=question)
        messages = [
            # {"role": "system", "content": self.prompt_template.system_prompt_text},
            {"role": "user", "content": user_prompt},
        ]
        row["messages"] = messages
        return row


def build_vllm_processor(
        model_path: str,
        max_num_batched_tokens: int,
        max_model_len: int,
        batch_size: int,
        max_output_token: int,
        temperature: Optional[float] = 0.6,
        concurrency: Optional[int] = 1,
        tp: Optional[int] = 1,
        pp: Optional[int] = 1
) -> Processor:
    logger.info(f"tp {tp} pp {pp}")
    assert os.path.exists(model_path)
    config = vLLMEngineProcessorConfig(
        model_source=model_path,
        engine_kwargs={
            # "kv_cache_dtype":"fp8",
            "enable_chunked_prefill": True,
            # 一个batch的最大token
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": 1,
            # 输入+输出最大长度
            "max_model_len": max_model_len,
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.85,
            "load_format": "safetensors",
        },
        # accelerator_type='L20',
        concurrency=concurrency,
        batch_size=batch_size,
    )
    processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            messages=row["messages"],
            sampling_params=dict(
                temperature=temperature,
                max_tokens=max_output_token,
            )
        ),
        postprocess=lambda row: dict(
            response=row["generated_text"],
            **row
        ),
    )
    return processor


def build_vllm_processor_with_lora(
        model_path: str,
        max_num_batched_tokens: int,
        max_model_len: int,
        batch_size: int,
        max_output_token: int,
        # 新增 LoRA 相关参数
        lora_adapter_path: Optional[str] = None,  # LoRA 权重的路径或 HF ID
        lora_rank: int = 16,  # 根据你的 LoRA config 设置 rank
        max_loras: int = 1,  # 缓存的 LoRA 数量
        temperature: Optional[float] = 0.6,
        concurrency: Optional[int] = 1,
        tp: Optional[int] = 1,
        pp: Optional[int] = 1
):
    # 注意：build_llm_processor 是旧版 API，根据第一段文档，此处建议使用 build_processor
    # 如果你的环境中只有 build_llm_processor，请自行替换回旧名

    # logger.info(f"tp {tp} pp {pp}") # 假设 logger 已在外部定义
    assert os.path.exists(model_path) or model_path.startswith("s3://") or "/" in model_path

    config = vLLMEngineProcessorConfig(
        model_source=model_path,
        engine_kwargs={
            # --- LoRA 核心配置 (参考文档第一部分) ---
            "enable_lora": True,
            "max_lora_rank": lora_rank,
            "max_loras": max_loras,
            "enable_chunked_prefill": True,
            # 一个batch的最大token
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": 1,  # 如果使用了 LoRA，通常保持较低 seqs 以避免显存压力，或者根据需求调整
            # 输入+输出最大长度
            "max_model_len": max_model_len,
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.85,
            # "load_format": "safetensors", # 根据模型实际情况选用
        },
        concurrency=concurrency,
        batch_size=batch_size,
    )

    processor = build_processor(
        config,
        preprocess=lambda row: dict(
            messages=row["messages"],
            # --- LoRA 路由配置 (参考文档第二部分) ---
            # 如果函数传入了 lora_adapter_path，则强制使用该 LoRA
            # 否则尝试从数据行的 "model" 字段读取（实现动态 LoRA）
            model=lora_adapter_path if lora_adapter_path else row.get("model"),
            # -------------------------------------
            sampling_params=dict(
                temperature=temperature,
                max_tokens=max_output_token,
            )
        ),
        postprocess=lambda row: dict(
            response=row["generated_text"],
            **row
        ),
    )
    return processor




class PreFilterActor:
    def __init__(self, tokenizer_name: str, prompt_template: edict, filter_col: str, context_limit: int):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.prompt_template = prompt_template
        self.filter_col = filter_col
        self.context_limit = context_limit

    def __call__(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        # estimate the context length of other contents
        haystack_sessions = entry["haystack_sessions"]
        haystack_dates = entry.get("haystack_dates", None)
        question = entry["question"]
        user_prompt = self.prompt_template.user_prompt_text.format(history="", current=question) # placeholder
        history_context_limit = self.context_limit - len(self.tokenizer.tokenize(user_prompt))

        # filter out the turns that are too long
        selected_session_idx = []

        filter_scores = entry[self.filter_col]
        # rank the sessions by the filter scores, higher score first
        sorted_indices = np.argsort(filter_scores)[::-1]
        # greedily select the sessions that fit into the context limit
        history_token_count = 0
        for idx in sorted_indices:
            session = haystack_sessions[idx]
            session_text = construct_session_text(session)
            if haystack_dates is not None:
                session_date = haystack_dates[idx]
                session_text = f"<session42> <date>{session_date}</date>\n{session_text}</session>\n" # not final session idx, just a placeholder for token count
            else:
                session_text = f"<session42>\n{session_text}</session>\n" # not final session idx, just a placeholder for token count
            session_token_count = len(self.tokenizer.tokenize(session_text))
            if history_token_count + session_token_count <= history_context_limit:
                selected_session_idx.append(idx)
                history_token_count += session_token_count
            else:
                break
        # reorder the selected sessions by the original index
        selected_session_idx.sort()

        entry["prefiltered_session_idx"] = selected_session_idx
        return entry


class RandomFilterActor: # not final use, just for debug
    def __init__(self, tokenizer_name: str, prompt_template: edict, filter_col: str, context_limit: int):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.prompt_template = prompt_template
        self.filter_col = filter_col
        self.context_limit = context_limit

    def __call__(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        # estimate the context length of other contents
        haystack_sessions = entry["haystack_sessions"]
        haystack_session_ids = entry["haystack_session_ids"]
        question = entry["question"]
        answer_session_ids = entry["answer_session_ids"]
        question_token_num = len(self.tokenizer.tokenize(question))

        answer_session_idx = []
        for sid, (session, session_id) in enumerate(zip(haystack_sessions, haystack_session_ids)):
            if session_id in answer_session_ids:
                answer_session_idx.append(sid)
        selected_session_idx = construct_history_text_with_limited_context(
            haystack_sessions,
            answer_session_idx,
            self.context_limit - question_token_num,
            self.tokenizer
        )
        entry["prefiltered_session_idx"] = selected_session_idx
        return entry


