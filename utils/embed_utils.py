import asyncio
import os
from asyncio import Semaphore
from copy import deepcopy
from dataclasses import dataclass
from hashlib import md5
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.cuda
from loguru import logger


@dataclass
class STEmbConfig:
    embedding_max_seq_len: int = 8192  # Embedding模型最大序列长度
    embedding_return_as_normalized: bool = True  # 输出归一化向量
    embedding_model_dtype: str = "float32"  # Embedding数据类型
    embedding_batch_size: int = 256  # Embedding批量处理大小
    embedding_model_device: str="cuda"
    embedding_model_name: str = "bge-m3"


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()


class STEmbeddingModel:
    def __init__(self, global_config: STEmbConfig, embedding_model_name: Optional[str] = None) -> None:
        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")
        else:
            self.embedding_model_name = global_config.embedding_model_name

        from sentence_transformers import SentenceTransformer

        self.max_length = global_config.embedding_max_seq_len
        # "float16", "float32", "bfloat16", "auto"
        # if global_config.embedding_model_type == "float16":
        #     torch_dtype = torch.float16
        # elif global_config.embedding_model_type == "float32":
        #     torch_dtype = torch.float
        # elif global_config.embedding_model_type == "bfloat16":
        #     torch_dtype = torch.bfloat16
        # elif global_config.embedding_model_type == "auto":
        #     torch_dtype = torch.float
        self.instruction = ''
        self.normalize_embeddings = global_config.embedding_return_as_normalized
        self.device = global_config.embedding_model_device
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name, trust_remote_code=True, device=self.device, model_kwargs={
                "torch_dtype": global_config.embedding_model_dtype}
        )
        self.batch_size = global_config.embedding_batch_size
        self.max_seq_len = global_config.embedding_max_seq_len

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        if isinstance(texts, str): texts = [texts]

        params = {
            "prompt": self.instruction,
            "normalize_embeddings": self.normalize_embeddings,
            "show_progress_bar" : False,
        }
        if kwargs: params.update(kwargs)

        if "prompt" in kwargs:
            if kwargs["prompt"] != '':
                params["prompt"] = f"Instruct: {kwargs['prompt']}\nQuery: "
            # del params["prompt"]

        batch_size = params.get("batch_size", self.batch_size)

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")
        if len(texts) <= batch_size:
            params["sentences"] = texts  # self._add_eos(texts=texts)
            results = self.embedding_model.encode(**params)
        else:
            # pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                params["sentences"] = texts[i:i + batch_size]
                results.append(self.embedding_model.encode(**params))
                # pbar.update(batch_size)
            # pbar.close()
            # results = torch.cat(results, dim=0)
            results = np.concatenate(results, axis=0)

        if isinstance(results, torch.Tensor):
            results = results.cpu()
            results = results.numpy()
        if params.get("norm", False):
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results


class STEmbedActor:
    def __init__(self, emb_config: STEmbConfig, embedding_model_name: Optional[str] = None, use_gpu: bool = True):
        if not use_gpu:
            emb_config.embedding_model_device = "cpu"
        # initialize the embedding model
        self.embedding_model = STEmbeddingModel(emb_config, embedding_model_name)

    def __call__(self, rows: Dict[str, Any]) -> Dict[str, Any]:
        texts = rows["text"]
        embeddings = self.embedding_model.batch_encode(texts)
        rows["embedding"] = embeddings
        return rows


class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        """
        Initializes the class with necessary configurations and sets up the working directory.

        Parameters:
        embedding_model: The model used for embeddings.
        db_filename: The directory path where data will be stored or retrieved.
        batch_size: The batch size used for processing.
        namespace: A unique identifier for data segregation.

        Functionality:
        - Assigns the provided parameters to instance variables.
        - Checks if the directory specified by `db_filename` exists.
          - If not, creates the directory and logs the operation.
        - Constructs the filename for storing data in a parquet file format.
        - Calls the method `_load_data()` to initialize the data loading process.
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace
        # 将 namespace 中的路径分隔符替换为下划线，避免被 os.path.join 解释为子目录
        safe_namespace = namespace.replace('/', '_').replace('\\', '_')

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(
            db_filename, f"vdb_{safe_namespace}.parquet"
        )
        self._load_data()

    def get_missing_string_hash_ids(self, texts: List[str]):
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return {}

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        return {h: {"hash_id": h, "content": t} for h, t in zip(missing_ids, texts_to_encode)}

    def insert_strings(self, texts: List[str]):
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  # Nothing to insert.

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        logger.info(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist.")

        if not missing_ids:
            return {}  # All records already exist.

        # Prepare the texts to encode from the "content" field.
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)

        self._upsert(missing_ids, texts_to_encode, missing_embeddings)


    def insert_embeddings(self, hash_ids: List[str], texts: List[str], embeddings: List[np.ndarray]):
        assert len(hash_ids) == len(texts) == len(embeddings), f"hash_ids, texts, embeddings must have the same length, but got {len(hash_ids)}, {len(texts)}, {len(embeddings)}"
        self._upsert(hash_ids, texts, embeddings)


    def _load_data(self):
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids, self.texts, self.embeddings = df["hash_id"].values.tolist(), df["content"].values.tolist(), \
            df["embedding"].values.tolist()
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t}
                for h, t in zip(self.hash_ids, self.texts)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids, self.texts, self.embeddings = [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}

    def _save_data(self):
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "embedding": self.embeddings
        })
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {h: {"hash_id": h, "content": t} for h, t, e in
                               zip(self.hash_ids, self.texts, self.embeddings)}
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings):
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)

        logger.info(f"Saving new records.")
        self._save_data()

    def delete(self, hash_ids):
        indices = []

        for hash in hash_ids:
            indices.append(self.hash_id_to_idx[hash])

        sorted_indices = np.sort(indices)[::-1]

        for idx in sorted_indices:
            self.hash_ids.pop(idx)
            self.texts.pop(idx)
            self.embeddings.pop(idx)

        logger.info(f"Saving record after deletion.")
        self._save_data()

    def get_row(self, hash_id):
        return self.hash_id_to_row[hash_id]

    def get_hash_id(self, text):
        return self.text_to_hash_id[text]

    def get_rows(self, hash_ids, dtype=np.float32):
        if not hash_ids:
            return {}

        results = {id: self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_all_ids(self):
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self):
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self):
        return set(row['content'] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)

    def get_embeddings(self, hash_ids, dtype=np.float32) -> list[np.ndarray]:
        if not hash_ids:
            return []

        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings



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

