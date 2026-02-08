import argparse
import asyncio
import json
import os
import sys
from asyncio import Semaphore
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.cuda
from loguru import logger
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer

sys.path.append("./")
from utils.embed_utils import compute_mdhash_id
from utils.embed_utils import STEmbeddingModel  # 替换为实际模块路径
from utils.embed_utils import EmbeddingStore  # 替换为实际模块路径


def compute_cosine_similarity(query_emb: np.ndarray, candidate_embs: List[np.ndarray]) -> np.ndarray:
    """计算查询向量与候选向量列表的余弦相似度"""
    candidate_array = np.array(candidate_embs)
    query_norm = np.linalg.norm(query_emb, axis=0)
    candidate_norms = np.linalg.norm(candidate_array, axis=1)

    # 处理零向量（避免除以零）
    if query_norm == 0:
        logger.warning("Query embedding is a zero vector, similarity will be 0")
        return np.zeros(len(candidate_embs))
    candidate_norms[candidate_norms == 0] = 1e-10  # 微小值避免除以零

    # 计算余弦相似度
    dot_product = np.dot(candidate_array, query_emb)
    similarity = dot_product / (candidate_norms * query_norm)
    return similarity


@dataclass
class GlobalConfig:
    embedding_max_seq_len: int = 8192  # Embedding模型最大序列长度
    embedding_return_as_normalized: bool = True  # 输出归一化向量
    embedding_model_dtype: str = "float32"  # Embedding数据类型
    embedding_batch_size: int = 128  # Embedding批量处理大小


class GPUDevicePool:
    """GPU设备池管理，确保每张卡同时只运行一个任务"""

    def __init__(self, max_devices: Optional[int] = None):
        # 获取可用GPU设备
        self.available_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

        if max_devices is not None and max_devices > 0:
            self.available_devices = self.available_devices[:max_devices]

        if not self.available_devices:
            logger.warning("No GPU devices available, will use CPU instead")
            self.available_devices = [-1]  # -1 表示CPU

        # 为每个设备创建一个信号量，确保同时只有一个任务运行
        self.semaphores = {
            device: Semaphore(1) for device in self.available_devices
        }

        logger.info(f"Initialized GPU device pool with devices: {self.available_devices}")

    async def acquire(self) -> int:
        """获取一个可用的设备，返回设备ID"""
        # 尝试获取任意可用设备
        while True:
            for device, semaphore in self.semaphores.items():
                if semaphore.locked():
                    continue
                # 尝试获取设备锁
                try:
                    await asyncio.wait_for(semaphore.acquire(), timeout=0.1)
                    return device
                except asyncio.TimeoutError:
                    continue

            # 如果所有设备都在忙，等待一下再重试
            await asyncio.sleep(0.5)

    def release(self, device: int) -> None:
        """释放设备锁"""
        if device in self.semaphores:
            try:
                self.semaphores[device].release()
            except ValueError:
                logger.warning(f"Trying to release an unlocked device: {device}")

    def size(self) -> int:
        """返回设备池大小"""
        return len(self.available_devices)


async def process_single_entry(
        entry: Dict,
        entry_idx: int,
        embedding_model_name: str,
        global_config: GlobalConfig,
        tokenizer: AutoTokenizer,
        dataset_name: str,
        db_filename: str,
        device: int
) -> Dict:
    """处理单个条目，在指定设备上运行"""
    try:
        # 设置当前线程的设备
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device) if device != -1 else ""
        device_str = f"cuda:{device}" if device != -1 else "cpu"

        # logger.debug(f"Processing entry {entry_idx} on device {device_str}")

        # 提取当前条目核心信息
        question = entry["question"]
        question_id = entry["question_id"]
        answer = entry["answer"]
        question_date = entry["question_date"]
        haystack_dates = entry["haystack_dates"]
        haystack_sessions = entry["haystack_sessions"]
        haystack_session_ids = entry["haystack_session_ids"]
        answer_session_ids = entry["answer_session_ids"]  # 答案相关的会话ID

        # 初始化Embedding模型（使用指定设备）
        local_config = GlobalConfig(**global_config.__dict__)
        local_config.embedding_model_device = device_str

        embedding_model = STEmbeddingModel(
            global_config=local_config,
            embedding_model_name=embedding_model_name
        )

        # 初始化Embedding存储
        embedding_store = EmbeddingStore(
            embedding_model=embedding_model,
            db_filename=db_filename,
            batch_size=global_config.embedding_batch_size,
            namespace=f"{question_id}"
        )

        # 处理历史会话Turn并计算Embedding
        history_turns_info = []

        # 遍历所有Haystack会话
        for session_idx in range(len(haystack_sessions)):
            session = haystack_sessions[session_idx]
            session_date = haystack_dates[session_idx]
            session_id = haystack_session_ids[session_idx]

            # 会话前缀
            # session_prefix = f'<session date="{session_date}">\n'

            # 处理会话内的每个Turn
            user_turn_count = 0
            assistant_turn_count = 0

            for turn in session:
                turn_role = turn["role"].strip().lower()
                turn_content = turn["content"].strip()

                if not turn_content:
                    # logger.warning(f"会话 {session_id} 中Turn {turn} 内容为空，跳过")
                    continue

                # 构建Turn文本
                if turn_role == "user":
                    user_turn_count += 1
                    # turn_text = f"{session_prefix}<user>{turn_content}</user>\n"
                    turn_name = f"session_{session_id}_user_{user_turn_count}"
                elif turn_role == "assistant":
                    assistant_turn_count += 1
                    # turn_text = f"{session_prefix}<assistant>{turn_content}</assistant>\n"
                    turn_name = f"session_{session_id}_assistant_{assistant_turn_count}"
                else:
                    logger.warning(f"未知Turn角色: {turn_role}（会话ID: {session_id}），跳过")
                    continue

                history_turns_info.append({
                    "turn_name": turn_name,
                    "session_id": session_id,
                    "session_date": session_date,
                    "full_text": turn_content,
                    "role": turn_role,
                    "content": turn_content,
                    "embedding_hash_id": compute_mdhash_id(turn_content, prefix=embedding_store.namespace + "-")
                })

        # 计算Turn的Embedding并存储
        await asyncio.to_thread(embedding_store.insert_strings, texts=[turn["full_text"] for turn in history_turns_info])


        # 处理当前查询并计算相关度排序
        # current_session_text = f'<session date="{question_date}">\n<user>{question}</user>\n</session>'
        current_session_text = question

        # 计算当前查询的Embedding
        await asyncio.to_thread(embedding_store.insert_strings, texts=[current_session_text])
        current_hash_id = embedding_store.text_to_hash_id[current_session_text]
        current_embedding = embedding_store.get_embedding(hash_id=current_hash_id)

        # 计算相似度
        if history_turns_info:
            history_hash_ids = [turn["embedding_hash_id"] for turn in history_turns_info]
            history_embeddings = embedding_store.get_embeddings(hash_ids=history_hash_ids)

            similarities = compute_cosine_similarity(
                query_emb=current_embedding,
                candidate_embs=history_embeddings
            )

            # 添加相似度分数
            for turn_idx, (turn_info, sim_score) in enumerate(zip(history_turns_info, similarities)):
                turn_info["similarity_score"] = float(sim_score)
        else:
            logger.warning(f"条目 {entry_idx}: 无历史Turn数据，跳过相似度计算")


        # update turn info in haystack_sessions
        turn_idx = 0
        for session_idx in range(len(haystack_sessions)):
            session = haystack_sessions[session_idx]
            for turn in session:
                if not turn["content"].strip():
                    turn["embedding_hash_id"] = ""
                    turn["similarity_score"] = 0.0
                else:
                    embedded_turn = history_turns_info[turn_idx]
                    turn["embedding_hash_id"] = embedded_turn["embedding_hash_id"]
                    turn["similarity_score"] = embedded_turn["similarity_score"]
                    turn_idx += 1

        # 构建输出数据
        output_entry = entry.copy()
        output_entry["current_session_text"] = current_session_text
        output_entry["current_embedding_hash_id"] = current_hash_id
        output_entry["haystack_sessions"] = haystack_sessions

        # logger.debug(f"Completed processing entry {entry_idx} on device {device_str}")
        return output_entry

    except Exception as e:
        logger.error(f"Error processing entry {entry_idx} on device {device}: {str(e)}", exc_info=True)
        raise


async def process_entries(
        entries: List[Dict],
        embedding_model_name: str,
        global_config: GlobalConfig,
        tokenizer: AutoTokenizer,
        dataset_name: str,
        device_pool: GPUDevicePool,
        db_filename: str
) -> List[Optional[Dict]]:
    """处理所有条目，使用设备池分配任务"""
    results = [None] * len(entries)  # 保持原始顺序

    async def worker(entry_idx: int, entry: Dict):
        """工作函数，获取设备并处理条目"""
        device = await device_pool.acquire()
        try:
            result = await process_single_entry(
                entry=entry,
                entry_idx=entry_idx,
                embedding_model_name=embedding_model_name,
                global_config=global_config,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                db_filename=db_filename,
                device=device
            )
            results[entry_idx] = result
        finally:
            device_pool.release(device)

    # 创建所有任务并等待完成
    tasks = [worker(i, entry) for i, entry in enumerate(entries)]
    await tqdm_asyncio.gather(*tasks, desc="Processing entries")

    return [r for r in results if r is not None]


async def process_longmemeval_data(data_dir, dataset_name, split, embed_model, embed_batch_size, debug=False):
    # 基础配置与数据加载
    data_file = f"{data_dir}/{dataset_name}/{split}.jsonl"
    output_file = f"{data_dir}/results/{embed_model}/{dataset_name}_{split}_embed.jsonl"
    embedding_model_path = f"../../models/{embed_model}"
    db_filename = f"{data_dir}/embedding_store/{dataset_name}/{embed_model}_{split}"

    # 加载原始数据
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"原始数据文件不存在: {data_file}")
    if debug:
        logger.info(f"Debug mode enabled, using only 10 samples from {data_file}")
        orig_data = [json.loads(l) for l in open(data_file)][:10]
    else:
        orig_data = [json.loads(l) for l in open(data_file)]

    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(db_filename, exist_ok=True)


    logger.info(f"成功加载数据，共 {len(orig_data)} 条记录")

    # 初始化Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)

    # 初始化全局配置
    global_config = GlobalConfig(embedding_batch_size=embed_batch_size)

    # 初始化GPU设备池
    device_pool = GPUDevicePool()
    logger.info(f"Using {device_pool.size()} devices for processing")

    # 处理所有条目
    emb_ranking_output = await process_entries(
        entries=orig_data,
        embedding_model_name=embedding_model_path,
        global_config=global_config,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        device_pool=device_pool,
        db_filename=db_filename,
    )

    # 写入jsonl文件
    with open(output_file, "w") as f:
        for entry in emb_ranking_output:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"数据处理完成！输出文件已保存至: {output_file}")
    logger.info(f"Embedding存储位置: {db_filename}")


if __name__ == '__main__':
    # 配置日志
    logger.add("embedding_processing.log", rotation="10 MB")

    arg_parser = argparse.ArgumentParser(description="vLLM Probability Calculator API")
    arg_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    arg_parser.add_argument("--dataset-name", default="LongMemEval", help="Dataset name")
    arg_parser.add_argument("--split", default="test", help="Data split")
    arg_parser.add_argument("--embed-model", default="bge-m3", help="Embedding model")
    arg_parser.add_argument("--embed-batch-size", type=int, default=64, help="Embedding batch size")
    arg_parser.add_argument("--data-dir", default="data", help="Data directory")
    args = arg_parser.parse_args()

    # 运行异步处理函数
    asyncio.run(process_longmemeval_data(args.data_dir, args.dataset_name, args.split, args.embed_model, args.embed_batch_size, args.debug))
