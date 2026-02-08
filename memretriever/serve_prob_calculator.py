import argparse
import asyncio
import os
from collections import deque
from typing import List, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
from transformers import PreTrainedTokenizer as Tokenizer
from vllm import LLM, SamplingParams

app = FastAPI(title="vLLM Probability Calculator API")
available_calculators = deque()
calculator_semaphore = None
calculator_list = []


class VLLMProbabilityCalculator:
    def __init__(self,
                 model_path: str,
                 serve_model_name: str,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85,
                 logprobs: Optional[int] = 5):
        """初始化vllm模型，支持多worker和GPU分配"""
        logger.info(f"Initializing VLLMProbabilityCalculator with model: {model_path}")

        # 配置采样参数
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=1,  # 生成1个token
            logprobs=logprobs,
            skip_special_tokens=False
        )

        self.model_path = model_path
        if not serve_model_name:
            self.serve_model_name = model_path.split("/")[-1]
        else:
            self.serve_model_name = serve_model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size

        # 这些将在init_llm中初始化
        self.llm: Optional[LLM] = None
        self.tokenizer: Optional[Tokenizer] = None

    def init_llm(self, devices: List[int]) -> None:
        """初始化LLM模型和tokenizer"""
        logger.info(f"Initializing LLM on devices {devices}")

        # 设置环境变量指定可见设备
        if devices and devices[0] != -1:  # 不是CPU
            visible_devices = ",".join(map(str, devices))
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
            logger.info(f"Set CUDA_VISIBLE_DEVICES to {visible_devices}")

        # 创建LLM实例
        try:
            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype="bfloat16",
                max_num_batched_tokens=65536,
                max_num_seqs=2,
                enable_prefix_caching=True
            )
            logger.info("LLM instance created successfully")
        except Exception as e:
            logger.error(f"Failed to create LLM instance: {str(e)}")
            raise

        # 初始化tokenizer
        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Tokenizer initialized successfully")

    def calculate_multiple_tokens_prob(self,
                                       input_ids: List[int],
                                       force_token_ids: List[int]) -> Dict[int, float]:
        """计算单个prompt中多个指定token的概率"""
        if not self.llm or not self.tokenizer:
            logger.error("LLM or tokenizer not initialized")
            raise RuntimeError("LLM or tokenizer not initialized")

        try:
            # 将input_ids转换为单个prompt
            prompt = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            # logger.debug(f"Decoded prompt: {prompt}")

            # 生成结果
            outputs = self.llm.generate(
                prompts=[prompt],
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            # 获取输出的log概率分布
            output = outputs[0]
            logprobs_dict = output.outputs[0].logprobs[-1] if output.outputs[0].logprobs else {}
            # logger.debug(f"Received logprobs for {len(logprobs_dict)} tokens")

            # 提取目标token的log概率
            target_logprobs = []
            valid_token_ids = []

            for token_id in force_token_ids:
                if token_id == self.tokenizer.unk_token_id:
                    token = self.tokenizer.convert_ids_to_tokens(token_id)
                    logger.warning(f"Token '{token}' (ID: {token_id}) is unknown")
                    continue

                if token_id in logprobs_dict:
                    logprob_val = logprobs_dict[token_id].logprob
                    target_logprobs.append(logprob_val)
                    valid_token_ids.append(token_id)
                    # logger.debug(f"Found logprob {logprob_val} for token ID {token_id}")
                else:
                    logger.debug(f"Token ID {token_id} not found in logprobs")

            # 处理结果
            results = {tid: 0.0 for tid in force_token_ids}

            if target_logprobs:
                # 计算softmax进行归一化
                logprobs_np = np.array(target_logprobs)
                probs = np.exp(logprobs_np)
                normalized_probs = probs / np.sum(probs)

                for token_id, prob in zip(valid_token_ids, normalized_probs):
                    results[token_id] = float(prob)
                    # logger.debug(f"Normalized probability for token {token_id}: {prob}")

            return results

        except Exception as e:
            logger.error(f"Error calculating probabilities: {str(e)}")
            raise


# 请求模型
class ProbabilityRequest(BaseModel):
    input_ids: List[int]
    force_token_ids: List[int]

class ProbabilityResponse(BaseModel):
    result: Dict[int, float]

@app.on_event("startup")
def startup_event():
    """启动时初始化模型"""
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    global calculator_semaphore
    global calculator_list
    calculator = VLLMProbabilityCalculator(
        model_path=args.model_path,
        serve_model_name=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        logprobs=args.logprobs,
    )

    # 解析设备列表
    devices = list(map(int, args.devices.split(','))) if args.devices else []
    calculator.init_llm(devices)
    calculator_semaphore = asyncio.Semaphore(1)
    calculator_list.append(calculator)
    available_calculators.append(0)


@app.post("/calculate-probabilities", response_model=Dict[int, float])
async def calculate_probabilities(request: ProbabilityRequest) -> Dict[int, float]:
    """计算指定token的概率"""
    async with calculator_semaphore:
        calculator_id = available_calculators.pop()
        calculator = calculator_list[calculator_id]
        try:
            result = calculator.calculate_multiple_tokens_prob(
                input_ids=request.input_ids,
                force_token_ids=request.force_token_ids
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            available_calculators.append(calculator_id)


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "model_name": args.model_path, "available_calculators": available_calculators}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Probability Calculator API")
    parser.add_argument("--model-path", default="/mnt/local/logs/jjtan/4b-think-custom-163264/checkpoint-124", help="Path to the vLLM model")
    parser.add_argument("--host", default="0.0.0.0", help="API host address")
    parser.add_argument("--port", type=int, default=5500, help="API port number")
    parser.add_argument("--devices", default="0", help="Comma-separated list of device IDs (e.g., '0,1' or 'cpu')")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs per worker")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.93, help="GPU memory utilization")
    parser.add_argument("--logprobs", type=int, default=20, help="Number of logprobs to return")
    parser.add_argument("--log-file", default="vllm_prob_server.log", help="Log file path")

    args = parser.parse_args()
    assert args.model_path, f"model_path {args.model_path} is empty"

    # 启动FastAPI服务
    import uvicorn

    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
