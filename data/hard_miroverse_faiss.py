import os
import json
import random
import glob
import re
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer
import shutil
from uuid import uuid4
from typing import List, Tuple, Set, Optional, Dict
import sys
sys.path.append("./")
from utils.embed_utils import (
    STEmbeddingModel, 
    STEmbConfig, 
    compute_cosine_similarity, 
    EmbeddingStore, 
    compute_mdhash_id
)
from utils.session_process import construct_session_text
import ray
from utils.embed_utils import STEmbedActor
import faiss

# ================= 配置区域 =================
# 目标上下文长度 (Token数)
TARGET_CONTEXT_LEN = 128000
# 数据路径配置
INPUT_DIR = "/mnt/jjtan/data"
SUB_DATASETS = [
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
    ]
eval_datasets = [
    "HotpotQA",
    "WebDancer",
    "WebWalkerQA-Silver",
]

OUTPUT_DIR = "/mnt/jjtan/data/MiroVerse-128k"

# Tokenizer 模型路径
TOKENIZER_PATH = "/mnt/jjtan/models/Qwen3-0.6B"

# Embedding 模型配置
EMBEDDING_MODEL_PATH = "/mnt/jjtan/models/bge-m3"  # bge-m3 模型路径
EMBEDDING_BATCH_SIZE = 512  # 批量编码大小（增大以充分利用GPU）
EMBEDDING_MAX_SEQ_LEN = 2048  # 最大序列长度
EMBEDDING_STORE_PATH = "/mnt/jjtan/data/embedding_store"  # EmbeddingStore 存储路径
DISTRACTOR_CHUNK_SIZE = 12000  # 每次处理的 distractor 数量（内存优化）
SIMILARITY_SEARCH_TOP_K = 400  # 相似度搜索时返回的 top-k 候选数

# FAISS 索引配置
FAISS_HNSW_M = 32  # HNSW 每个节点的连接数
FAISS_HNSW_EF_CONSTRUCTION = 200  # 构建时的搜索范围
FAISS_HNSW_EF_SEARCH = 400  # 搜索时的候选数量

# Ray 并行配置
RAY_EMB_CONCURRENCY = 32  # embedding 计算的并发数（建议设置为 GPU 数量的 4 倍）
RAY_EMB_MIN_CONCURRENCY = 8  # 最小并发数（等于 GPU 数量）
RAY_SIMILARITY_CONCURRENCY = 28  # 相似度计算的并发数（CPU任务，可以设置更大）

# 加载 Tokenizer
logger.info(f"Loading tokenizer from {TOKENIZER_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
logger.info("Tokenizer loaded successfully.")


def estimate_token_count(text_or_obj):
    """
    使用 Qwen3 tokenizer 精确计算文本或对象的 Token 数量。
    如果是列表/字典，先 dump 成 string 再计算。
    """
    if isinstance(text_or_obj, (list, dict)):
        text = json.dumps(text_or_obj, ensure_ascii=False)
    else:
        text = str(text_or_obj)
    return len(tokenizer.encode(text, add_special_tokens=False))


def extract_boxed_content(answer: str) -> str:
    """
    Extract all \\boxed{xxx} content from answer string.
    If multiple boxed contents are found, merge them with spaces.
    If no boxed content is found, return the original answer.
    
    Args:
        answer: The answer string that may contain \\boxed{xxx} patterns
        
    Returns:
        Extracted content from all \\boxed{xxx} patterns, or original answer if none found
    """
    if not answer or not isinstance(answer, str):
        answer = str(answer)
    
    # Match \boxed{xxx} pattern (literal backslash followed by boxed)
    # Pattern matches: \boxed{content}
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, answer)
    
    if matches:
        # Merge all extracted contents with spaces
        extracted = ' '.join(matches)
        return extracted
    else:
        # No boxed content found, return original
        return answer


def clean_question(question: str) -> str:
    """
    Clean question text by removing specific instruction strings.
    
    Removes:
    - "You should follow the format instruction in the requestion strictly and wrap the final answer in \\boxed{}."
    - "Please provide the answer and detailed supporting information of the subtask given to you."
    
    Args:
        question: The question string to clean
        
    Returns:
        Cleaned question string with specified strings removed and stripped
    """
    if not question or not isinstance(question, str):
        return question
    
    # Strings to remove (exact match)
    strings_to_remove = [
        "You should follow the format instruction in the requestion strictly and wrap the final answer in \\boxed{}.",
        "Please provide the answer and detailed supporting information of the subtask given to you.",
    ]
    
    cleaned = question
    for s in strings_to_remove:
        cleaned = cleaned.replace(s, "")
    
    return cleaned.strip()


def build_distractor_pool(sub_datasets, data_dir):
    """
    从所有子数据集的 train.parquet 中收集 haystack_sessions
    返回一个列表，包含所有的 session (list of turns)
    """
    pool = []
    logger.info("Building distractor pool from all training sets...")

    for sub_dataset in tqdm(sub_datasets, desc="Loading Pools"):
        more_train_file_split = os.path.join(data_dir, sub_dataset, "more_train_0.parquet")
        if os.path.exists(more_train_file_split):
            # collect more_train_*.parquet
            more_train_files = glob.glob(os.path.join(data_dir, sub_dataset, "more_train_*.parquet"))
            if not more_train_files:
                logger.warning(f"No more_train_*.parquet found for {sub_dataset}")
                continue
            # 合并所有 more_train_*.parquet 文件
            df = pd.concat([pd.read_parquet(f) for f in more_train_files], ignore_index=True)
            logger.info(f"Concatenated {len(more_train_files)} more_train_*.parquet files for {sub_dataset}")
        else:
            more_train_file = os.path.join(data_dir, sub_dataset, "more_train.parquet")
            if not os.path.exists(more_train_file):
                logger.warning(f"No more_train.parquet found for {sub_dataset}")
                continue
            df = pd.read_parquet(more_train_file)

        # 提取 haystack_sessions
        # 注意：存的是 JSON string，需要解析
        for json_str in df["haystack_sessions"]:
            try:
                sessions = json.loads(json_str)
                # sessions 是一个 list，里面每个元素是一个 session (list of dicts)
                pool.extend(sessions)
            except Exception as e:
                continue

    # 去重 (可选，为了防止完全一样的 session 出现多次，这里将其转为 string 做 set)
    # 考虑到性能，如果池子非常大，可以跳过去重或使用简单采样
    logger.info(f"Pool loaded. Total candidate sessions: {len(pool)}")
    return pool


class FaissIndexManager:
    """
    使用 FAISS HNSW 索引管理 embeddings，替代 EmbeddingStore
    
    功能：
    1. 从 parquet 文件（EmbeddingStore 格式）读取 embeddings
    2. 构建 FAISS HNSW 索引
    3. 提供相似度搜索功能
    4. 支持保存/加载索引
    """
    
    def __init__(self, parquet_path: str, namespace: str, 
                 embedding_dim: Optional[int] = None,
                 m: int = FAISS_HNSW_M,
                 ef_construction: int = FAISS_HNSW_EF_CONSTRUCTION,
                 ef_search: int = FAISS_HNSW_EF_SEARCH):
        """
        Args:
            parquet_path: EmbeddingStore parquet 文件路径
            namespace: namespace（用于生成 hash_id）
            embedding_dim: embedding 维度（如果为 None，从数据中推断）
            m: HNSW 每个节点的连接数
            ef_construction: 构建时的搜索范围
            ef_search: 搜索时的候选数量
        """
        self.parquet_path = parquet_path
        self.namespace = namespace
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # 索引文件路径
        safe_namespace = namespace.replace('/', '_').replace('\\', '_')
        self.index_path = os.path.join(os.path.dirname(parquet_path), f"faiss_index_{safe_namespace}.index")
        self.meta_path = os.path.join(os.path.dirname(parquet_path), f"faiss_meta_{safe_namespace}.pkl")
        
        # 加载或构建索引
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            logger.info(f"Loading existing FAISS index from {self.index_path}")
            self._load_index()
        else:
            logger.info(f"Building new FAISS index from {parquet_path}")
            self._build_index(embedding_dim)
    
    def _build_index(self, embedding_dim: Optional[int] = None):
        """从 parquet 文件构建 FAISS 索引"""
        if not os.path.exists(self.parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")
        
        # 读取 parquet 文件
        logger.info(f"Reading embeddings from {self.parquet_path}...")
        df = pd.read_parquet(self.parquet_path)
        
        if len(df) == 0:
            raise ValueError(f"Parquet file is empty: {self.parquet_path}")
        
        # 提取数据
        hash_ids = df["hash_id"].values.tolist()
        texts = df["content"].values.tolist()
        embeddings = df["embedding"].values.tolist()
        
        # 转换为 numpy 数组
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # 确定维度
        if embedding_dim is None:
            embedding_dim = embeddings_array.shape[1]
        else:
            assert embeddings_array.shape[1] == embedding_dim, \
                f"Embedding dimension mismatch: expected {embedding_dim}, got {embeddings_array.shape[1]}"
        
        logger.info(f"Loaded {len(embeddings_array)} embeddings with dimension {embedding_dim}")
        
        # 确保向量已归一化
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # 避免除以零
        embeddings_array = embeddings_array / norms
        
        # 构建 FAISS HNSW 索引（使用内积，因为向量已归一化）
        logger.info(f"Building FAISS HNSW index with M={self.m}, ef_construction={self.ef_construction}...")
        index = faiss.IndexHNSWFlat(embedding_dim, self.m)
        index.hnsw.efConstruction = self.ef_construction
        
        # 添加向量到索引
        index.add(embeddings_array)
        
        self.index = index
        self.hash_id_to_faiss_idx = {hash_id: idx for idx, hash_id in enumerate(hash_ids)}
        self.faiss_idx_to_hash_id = {idx: hash_id for idx, hash_id in enumerate(hash_ids)}
        self.faiss_idx_to_text = {idx: text for idx, text in enumerate(texts)}
        
        # 保存索引
        self._save_index()
        
        logger.info(f"FAISS index built successfully with {index.ntotal} vectors")
    
    def _save_index(self):
        """保存 FAISS 索引和元数据"""
        logger.info(f"Saving FAISS index to {self.index_path}...")
        faiss.write_index(self.index, self.index_path)
        
        meta = {
            "hash_id_to_faiss_idx": self.hash_id_to_faiss_idx,
            "faiss_idx_to_hash_id": self.faiss_idx_to_hash_id,
            "faiss_idx_to_text": self.faiss_idx_to_text,
            "namespace": self.namespace,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
        }
        
        with open(self.meta_path, "wb") as f:
            pickle.dump(meta, f)
        
        logger.info(f"FAISS index and metadata saved")
    
    def _load_index(self):
        """加载 FAISS 索引和元数据"""
        logger.info(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(self.index_path)
        
        with open(self.meta_path, "rb") as f:
            meta = pickle.load(f)
        
        self.hash_id_to_faiss_idx = meta["hash_id_to_faiss_idx"]
        self.faiss_idx_to_hash_id = meta["faiss_idx_to_hash_id"]
        self.faiss_idx_to_text = meta["faiss_idx_to_text"]
        self.namespace = meta.get("namespace", self.namespace)
        self.m = meta.get("m", self.m)
        self.ef_construction = meta.get("ef_construction", self.ef_construction)
        self.ef_search = meta.get("ef_search", self.ef_search)
        
        logger.info(f"FAISS index loaded successfully with {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, top_k: int, 
               exclude_hash_ids: Optional[Set[str]] = None) -> List[Tuple[int, float]]:
        """
        执行相似度搜索
        
        Args:
            query_embedding: 查询向量（已归一化），shape: (dim,)
            top_k: 返回 top-k 个结果
            exclude_hash_ids: 需要排除的 hash_id 集合
            
        Returns:
            List[Tuple[faiss_idx, score]]: (索引, 相似度分数) 列表，按相似度降序
        """
        # 确保 query_embedding 是 numpy array 且为 float32
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        else:
            query_embedding = query_embedding.astype(np.float32)
        
        # 归一化 query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # 设置搜索参数
        self.index.hnsw.efSearch = self.ef_search
        
        # 执行搜索（reshape 为 (1, dim)）
        query_emb_2d = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_emb_2d, min(top_k * 2, self.index.ntotal))
        
        # 处理结果
        scores = scores[0]  # 去掉 batch 维度
        indices = indices[0]
        
        # 排除指定的 hash_ids
        results = []
        exclude_set = exclude_hash_ids or set()
        
        for idx, score in zip(indices, scores):
            if idx == -1:  # FAISS 返回 -1 表示无效索引
                continue
            
            hash_id = self.faiss_idx_to_hash_id.get(idx)
            if hash_id is None:
                continue
            
            if hash_id not in exclude_set:
                results.append((idx, float(score)))
        
        # 按相似度降序排序并返回 top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_text_by_hash_id(self, hash_id: str) -> Optional[str]:
        """根据 hash_id 获取文本"""
        faiss_idx = self.hash_id_to_faiss_idx.get(hash_id)
        if faiss_idx is not None:
            return self.faiss_idx_to_text.get(faiss_idx)
        return None
    
    def get_hash_id_by_faiss_idx(self, faiss_idx: int) -> Optional[str]:
        """根据 FAISS 索引获取 hash_id"""
        return self.faiss_idx_to_hash_id.get(faiss_idx)


class DistractorEmbeddingManager:
    """
    管理 distractor embeddings 的缓存、分批加载和相似度搜索
    
    优化策略：
    1. 使用 FAISS 索引进行快速相似度搜索
    2. 从 EmbeddingStore parquet 文件读取 embeddings
    3. 支持懒加载：只在需要时加载和计算相似度
    """
    
    def __init__(self, distractor_pool: List, embedding_model: STEmbeddingModel, 
                 embed_store: EmbeddingStore, chunk_size: int = DISTRACTOR_CHUNK_SIZE):
        """
        Args:
            distractor_pool: distractor session 列表
            embedding_model: STEmbeddingModel 实例
            embed_store: EmbeddingStore 实例，用于缓存（如果 embeddings 不存在则计算）
            chunk_size: 每次处理的 distractor 数量（已废弃，保留以兼容）
        """
        self.distractor_pool = distractor_pool
        self.embedding_model = embedding_model
        self.embed_store = embed_store
        self.chunk_size = chunk_size
        self.total_distractors = len(distractor_pool)
        logger.info(f"Initialized DistractorEmbeddingManager with {self.total_distractors} distractors")
        
        # 保存配置信息供 Ray 使用
        self.emb_config = STEmbConfig()
        self.emb_config.embedding_model_name = embedding_model.embedding_model_name
        self.emb_config.embedding_batch_size = embedding_model.batch_size
        self.emb_config.embedding_max_seq_len = embedding_model.max_seq_len
        self.emb_config.embedding_model_device = embedding_model.device
        self.embedding_model_name = embedding_model.embedding_model_name
        
        # 预先将所有 distractor sessions 转换为文本并缓存 embeddings
        self._precompute_all_embeddings()
        
        # 构建 FAISS 索引
        self._build_faiss_index()
    
    def _precompute_all_embeddings(self):
        """使用 Ray 并行计算所有 distractor 的 embeddings"""
        logger.info("Precomputing distractor embeddings with Ray parallel processing...")
        
        # 将所有 distractor sessions 转换为文本
        distractor_texts = []
        for session in tqdm(self.distractor_pool, desc="Converting sessions to text"):
            session_text = construct_session_text(session)
            distractor_texts.append(session_text)
        
        # 检查哪些文本已经在 EmbeddingStore 中
        missing_texts = []
        missing_indices = []
        for idx, text in enumerate(distractor_texts):
            hash_id = compute_mdhash_id(text, prefix=self.embed_store.namespace + "-")
            if hash_id not in self.embed_store.hash_id_to_idx:
                missing_texts.append(text)
                missing_indices.append(idx)
        
        if not missing_texts:
            logger.info(f"All {len(distractor_texts)} distractor embeddings already cached")
            return
        
        logger.info(f"Found {len(missing_texts)} missing embeddings, computing with Ray...")
        
        # 创建 Ray Dataset
        text_ds = ray.data.from_items([{"text": text} for text in missing_texts])
        
        # 使用 Ray 并行编码（每个任务使用 1 个 GPU）
        # 使用更大的 batch_size 以充分利用 GPU 计算资源
        effective_batch_size = max(self.embedding_model.batch_size, EMBEDDING_BATCH_SIZE)
        embedded_ds = text_ds.map_batches(
            fn=STEmbedActor,
            fn_constructor_kwargs={
                "emb_config": self.emb_config,
                "embedding_model_name": self.embedding_model_name,
                "use_gpu": True,
            },
            batch_size=effective_batch_size,
            concurrency=(RAY_EMB_MIN_CONCURRENCY, RAY_EMB_CONCURRENCY),
            num_gpus=1,  # 每个任务使用 1 个 GPU
        )
        
        # 获取结果并写入 EmbeddingStore
        embedded_pd = embedded_ds.to_pandas()
        texts = embedded_pd["text"].tolist()
        embeddings = embedded_pd["embedding"].tolist()
        
        # 批量插入 EmbeddingStore
        hash_ids = [compute_mdhash_id(text, prefix=self.embed_store.namespace + "-") for text in texts]
        self.embed_store.insert_embeddings(hash_ids, texts, embeddings)
        
        logger.info(f"Cached embeddings for {len(texts)} distractor sessions (total: {len(distractor_texts)})")
    
    def _build_faiss_index(self):
        """构建 FAISS 索引"""
        # 获取 parquet 文件路径
        safe_namespace = self.embed_store.namespace.replace('/', '_').replace('\\', '_')
        parquet_path = os.path.join(
            os.path.dirname(self.embed_store.filename),
            f"vdb_{safe_namespace}.parquet"
        )
        
        # 获取 embedding 维度（从模型或已存储的数据中获取）
        embedding_dim = None
        if hasattr(self.embedding_model, 'embedding_model'):
            # 尝试从模型获取维度
            try:
                # bge-m3 通常是 1024 维
                embedding_dim = self.embedding_model.embedding_model.get_sentence_embedding_dimension()
            except:
                pass
        
        # 如果无法从模型获取，尝试从 EmbeddingStore 获取
        if embedding_dim is None and len(self.embed_store.embeddings) > 0:
            embedding_dim = len(self.embed_store.embeddings[0])
        
        # 如果还是无法获取，使用默认值（bge-m3 是 1024）
        if embedding_dim is None:
            embedding_dim = 1024
            logger.warning(f"Could not determine embedding dimension, using default: {embedding_dim}")
        
        # 创建 FAISS 索引管理器
        self.faiss_index = FaissIndexManager(
            parquet_path=parquet_path,
            namespace=self.embed_store.namespace,
            embedding_dim=embedding_dim,
            m=FAISS_HNSW_M,
            ef_construction=FAISS_HNSW_EF_CONSTRUCTION,
            ef_search=FAISS_HNSW_EF_SEARCH
        )
        
        # 构建 faiss_idx 到 session 的映射（通过 hash_id 匹配）
        self.faiss_idx_to_session = {}
        for sess in self.distractor_pool:
            session_text = construct_session_text(sess)
            hash_id = compute_mdhash_id(session_text, prefix=self.embed_store.namespace + "-")
            faiss_idx = self.faiss_index.hash_id_to_faiss_idx.get(hash_id)
            if faiss_idx is not None:
                self.faiss_idx_to_session[faiss_idx] = sess
    
    def find_most_similar_distractors(self, query_embedding: np.ndarray, 
                                     exclude_sessions: Set[str], 
                                     top_k: int = SIMILARITY_SEARCH_TOP_K) -> List[Tuple[List, float]]:
        """
        找到与 query 最相似的 distractor（排除 exclude_sessions）
        使用 FAISS 索引进行快速搜索
        
        Args:
            query_embedding: 查询向量（question embedding），shape: (dim,)
            exclude_sessions: 需要排除的 session 文本集合
            top_k: 返回 top-k 个最相似的
            
        Returns:
            List[Tuple[session, similarity_score]]: (session, score) 列表，按相似度降序
        """
        # 将 exclude_sessions 转换为 exclude_hash_ids
        exclude_hash_ids = set()
        for session_text in exclude_sessions:
            hash_id = compute_mdhash_id(session_text, prefix=self.embed_store.namespace + "-")
            exclude_hash_ids.add(hash_id)
        
        # 使用 FAISS 搜索
        results = self.faiss_index.search(
            query_embedding=query_embedding,
            top_k=top_k,
            exclude_hash_ids=exclude_hash_ids
        )
        
        # 转换为 (session, score) 格式
        candidates = []
        for faiss_idx, score in results:
            # 直接从映射中获取 session
            session = self.faiss_idx_to_session.get(faiss_idx)
            if session is not None:
                candidates.append((session, score))
        
        return candidates
    
    def get_distractor_embedding(self, session: List) -> Optional[np.ndarray]:
        """
        获取单个 distractor 的 embedding（从 EmbeddingStore 获取）
        
        Args:
            session: distractor session
            
        Returns:
            np.ndarray: embedding 向量，如果不存在则返回 None
        """
        session_text = construct_session_text(session)
        hash_id = compute_mdhash_id(session_text, prefix=self.embed_store.namespace + "-")
        
        if hash_id in self.embed_store.hash_id_to_idx:
            return self.embed_store.get_embedding(hash_id, dtype=np.float32)
        return None


class AugmentRowActor:
    """
    用于 ray.data.map 的数据增强 Actor
    
    对单行数据进行增强：使用相似度采样填充 distractor sessions
    """
    __name__ = "AugmentRowActor"  # 帮助 Ray 正确识别函数名
    
    def __init__(self, distractor_pool, faiss_index_path, faiss_meta_path, 
                 namespace, target_len=TARGET_CONTEXT_LEN):
        """
        初始化
        
        Args:
            distractor_pool: distractor session 列表（可序列化）
            faiss_index_path: FAISS 索引文件路径（可序列化）
            faiss_meta_path: FAISS 元数据文件路径（可序列化）
            namespace: namespace（可序列化）
            target_len: 目标 token 长度
        """
        self.distractor_pool = distractor_pool
        self.target_len = target_len
        self.total_distractors = len(distractor_pool)
        
        # 加载 FAISS 索引
        logger.info(f"Loading FAISS index from {faiss_index_path}...")
        self.index = faiss.read_index(faiss_index_path)
        
        with open(faiss_meta_path, "rb") as f:
            meta = pickle.load(f)
        
        self.hash_id_to_faiss_idx = meta["hash_id_to_faiss_idx"]
        self.faiss_idx_to_hash_id = meta["faiss_idx_to_hash_id"]
        self.faiss_idx_to_text = meta["faiss_idx_to_text"]
        self.namespace = namespace
        self.ef_search = meta.get("ef_search", FAISS_HNSW_EF_SEARCH)
        
        # 构建 faiss_idx 到 session 的映射
        self.faiss_idx_to_session = {}
        for idx, session in enumerate(distractor_pool):
            session_text = construct_session_text(session)
            hash_id = compute_mdhash_id(session_text, prefix=namespace + "-")
            faiss_idx = self.hash_id_to_faiss_idx.get(hash_id)
            if faiss_idx is not None:
                self.faiss_idx_to_session[faiss_idx] = session
        
        logger.info(f"FAISS index loaded with {self.index.ntotal} vectors")
    
    def find_most_similar_distractors(self, query_embedding: np.ndarray, 
                                     exclude_sessions: Set[str], 
                                     top_k: int = SIMILARITY_SEARCH_TOP_K) -> List[Tuple[List, float]]:
        """
        找到与 query 最相似的 distractor（排除 exclude_sessions）
        
        Args:
            query_embedding: 查询向量（question embedding），shape: (dim,)
            exclude_sessions: 需要排除的 session 文本集合
            top_k: 返回 top-k 个最相似的
            
        Returns:
            List[Tuple[session, similarity_score]]: (session, score) 列表，按相似度降序
        """
        # 确保 query_embedding 是 numpy array 且为 float32
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        else:
            query_embedding = query_embedding.astype(np.float32)
        
        # 归一化 query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # 将 exclude_sessions 转换为 exclude_hash_ids
        exclude_hash_ids = set()
        for session_text in exclude_sessions:
            hash_id = compute_mdhash_id(session_text, prefix=self.namespace + "-")
            exclude_hash_ids.add(hash_id)
        
        # 设置搜索参数
        self.index.hnsw.efSearch = self.ef_search
        
        # 执行搜索
        query_emb_2d = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_emb_2d, min(top_k * 2, self.index.ntotal))
        
        # 处理结果
        scores = scores[0]
        indices = indices[0]
        
        # 收集候选（排除 exclude_sessions）
        candidates = []
        for idx, score in zip(indices, scores):
            if idx == -1:
                continue
            
            hash_id = self.faiss_idx_to_hash_id.get(idx)
            if hash_id is None or hash_id in exclude_hash_ids:
                continue
            
            session = self.faiss_idx_to_session.get(idx)
            if session is not None:
                session_text = construct_session_text(session)
                if session_text not in exclude_sessions:
                    candidates.append((session, float(score)))
        
        # 按相似度降序排序并返回 top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def __call__(self, row: Dict) -> Dict:
        """
        处理单行数据（ray.data.map 会调用此方法）
        
        Args:
            row: 包含数据行的字典
                - 所有原始 DataFrame 列
                - question_embedding: List[float] (question 的 embedding 向量，作为列表传递)
        
        Returns:
            Dict: 增强后的数据行字典
        """
        # Process answer field: extract \boxed{xxx} content
        if "answer" in row and pd.notna(row.get("answer")):
            row["answer"] = extract_boxed_content(str(row["answer"]))
        
        # 从 row 中获取 question_embedding（作为列表传递）
        question_embedding_list = row.get("question_embedding")
        if question_embedding_list is not None:
            question_embedding = np.array(question_embedding_list, dtype=np.float32)
        else:
            question_embedding = None
        
        try:
            # haystack_sessions 应该是 JSON 字符串（从 Ray Dataset 传递）
            haystack_sessions_str = row.get("haystack_sessions", "[]")
            if isinstance(haystack_sessions_str, str):
                original_sessions = json.loads(haystack_sessions_str)
            else:
                original_sessions = haystack_sessions_str
            
            # haystack_session_ids 应该是列表
            original_ids = row.get("haystack_session_ids", [])
            if isinstance(original_ids, np.ndarray):
                original_ids = original_ids.tolist()
            
            # 展平嵌套列表（如果 haystack_session_ids 是嵌套列表格式）
            def flatten_list(lst):
                """展平嵌套列表"""
                result = []
                for item in lst:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        result.extend(flatten_list(item))
                    else:
                        result.append(item)
                return result
            
            if original_ids and isinstance(original_ids[0], (list, tuple, np.ndarray)):
                original_ids = flatten_list(original_ids)
        except Exception as e:
            logger.warning(f"Failed to parse haystack_sessions: {e}")
            # 如果解析失败，直接返回原行
            return row

        # 如果 question_embedding 为 None，回退到随机采样
        if question_embedding is None:
            logger.warning("question_embedding is None, falling back to random sampling")
            # 实际使用中应该确保 question_embedding 不为 None
            return row

        # 计算基础消耗 (Question + System prompt + Answer 等)
        current_tokens = 0
        combined_sessions = []
        combined_ids = []

        # 1. 先加入原始 Sessions (Gold)
        for sess, sess_id in zip(original_sessions, original_ids):
            combined_sessions.append(sess)
            combined_ids.append(sess_id)
            current_tokens += estimate_token_count(sess)

        # 2. 填充 Distractors（基于相似度采样）
        original_session_texts = {construct_session_text(sess) for sess in original_sessions}
        
        max_attempts = 10000
        attempts = 0
        used_distractors = set()  # 记录已使用的 distractor（通过文本）
        
        while current_tokens < self.target_len and attempts < max_attempts:
            # 找到最相似的 distractor（排除原始 sessions 和已使用的）
            exclude = original_session_texts | used_distractors
            candidates = self.find_most_similar_distractors(
                question_embedding,
                exclude,
                top_k=SIMILARITY_SEARCH_TOP_K
            )
            
            if not candidates:
                break
            
            # 选择第一个未使用的候选
            found = False
            for distractor, score in candidates:
                distractor_text = construct_session_text(distractor)
                if distractor_text not in used_distractors and distractor_text not in original_session_texts:
                    distractor_tokens = estimate_token_count(distractor)
                    combined_sessions.append(distractor)
                    distractor_id = f"distractor_{uuid4()}"
                    combined_ids.append(distractor_id)
                    used_distractors.add(distractor_text)
                    current_tokens += distractor_tokens
                    attempts += 1
                    found = True
                    break
            
            if not found:
                # 所有候选都已使用，跳出
                break

        # 3. 关键步骤：Needle In A Haystack 需要打乱顺序
        # 将 session 和 id 打包在一起 shuffle，保证对应关系不乱
        zipped = list(zip(combined_sessions, combined_ids))
        random.shuffle(zipped)

        # 解包
        final_sessions, final_ids = zip(*zipped)

        # 4. 更新 Row（确保使用 JSON 字符串避免嵌套列表）
        row["haystack_sessions"] = json.dumps(list(final_sessions), ensure_ascii=False)
        row["haystack_session_ids"] = list(final_ids)
        # 将原始的 session ids 保存到 answer_session_ids 字段
        # 确保 answer_session_ids 是扁平列表（不是嵌套列表）
        if isinstance(original_ids, (list, tuple)):
            # 确保每个元素都是字符串，不是嵌套列表
            flattened_ids = []
            for item in original_ids:
                if isinstance(item, (list, tuple, np.ndarray)):
                    flattened_ids.extend([str(x) for x in item])
                else:
                    flattened_ids.append(str(item))
            row["answer_session_ids"] = flattened_ids
        else:
            row["answer_session_ids"] = [str(original_ids)] if original_ids is not None else []
        
        # 移除 question_embedding（不再需要）
        if "question_embedding" in row:
            del row["question_embedding"]

        # 可选：更新一个字段记录实际 token 数供参考
        # row["approx_token_count"] = current_tokens

        return row


def process_dataset(sub_dataset, split, distractor_manager, question_embeddings_dict):
    """处理单个数据集的单个 split (train/test)"""
    input_file = os.path.join(INPUT_DIR, "MiroVerse-Legacy", sub_dataset, f"{split}.parquet")
    output_subdir = os.path.join(OUTPUT_DIR, sub_dataset)
    output_file = os.path.join(output_subdir, f"{split}.parquet")

    if not os.path.exists(input_file):
        logger.info(f"Input file not found: {input_file}")
        return

    os.makedirs(output_subdir, exist_ok=True)
    
    # 检测逻辑：如果输出文件已存在且数据完整，跳过处理
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_parquet(output_file)
            input_df = pd.read_parquet(input_file)
            
            # 检查行数是否一致，以及是否包含必要的列
            if len(existing_df) == len(input_df) and "haystack_sessions" in existing_df.columns:
                logger.info(f"Output file already exists and is complete: {output_file}")
                logger.info(f"Skipping processing for {sub_dataset} {split} ({len(existing_df)} rows)")
                return
            else:
                logger.warning(f"Output file exists but incomplete (existing: {len(existing_df)}, expected: {len(input_df)}), reprocessing...")
        except Exception as e:
            logger.warning(f"Error checking existing output file: {e}, reprocessing...")

    df = pd.read_parquet(input_file)
    logger.info(f"Processing {sub_dataset} - {split} ({len(df)} rows)...")

    # Clean question field (before computing embeddings)
    if "question" in df.columns:
        df["question"] = df["question"].apply(clean_question)

    # 准备数据：将 question_embedding 添加到每一行，并确保数据格式扁平化
    rows_data = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        
        # 将 question_embedding 添加到 row（作为列表，避免嵌套数组）
        question_embedding = question_embeddings_dict.get(idx, None)
        if question_embedding is not None:
            if isinstance(question_embedding, np.ndarray):
                row_dict["question_embedding"] = question_embedding.tolist()
            else:
                row_dict["question_embedding"] = question_embedding
        else:
            row_dict["question_embedding"] = None
        
        # 确保 haystack_sessions 是 JSON 字符串（避免嵌套列表）
        if "haystack_sessions" in row_dict:
            if not isinstance(row_dict["haystack_sessions"], str):
                row_dict["haystack_sessions"] = json.dumps(row_dict["haystack_sessions"], ensure_ascii=False)
        
        # 确保 haystack_session_ids 是列表（不是 numpy array）
        if "haystack_session_ids" in row_dict:
            if isinstance(row_dict["haystack_session_ids"], np.ndarray):
                row_dict["haystack_session_ids"] = row_dict["haystack_session_ids"].tolist()
        
        # 确保 messages 字段是 JSON 字符串（避免嵌套列表和 object 类型数组）
        if "messages" in row_dict:
            messages_value = row_dict["messages"]
            if not isinstance(messages_value, str):
                # 如果是 numpy array，先转换为列表
                if isinstance(messages_value, np.ndarray):
                    messages_value = messages_value.tolist()
                # 序列化为 JSON 字符串
                row_dict["messages"] = json.dumps(messages_value, ensure_ascii=False)
        
        rows_data.append(row_dict)
    
    # 处理每一行的所有字段
    logger.info("Drop legacy answer_session_ids field...")
    for row_dict in rows_data:
        row_dict["answer_session_ids"] = []

    # 获取 FAISS 索引路径
    safe_namespace = distractor_manager.embed_store.namespace.replace('/', '_').replace('\\', '_')
    faiss_index_path = distractor_manager.faiss_index.index_path
    faiss_meta_path = distractor_manager.faiss_index.meta_path
    
    # 创建 Ray Dataset
    logger.info(f"Creating Ray Dataset and applying augmentation with {len(rows_data)} rows...")
    ds = ray.data.from_items(rows_data)
    
    # 使用 Ray Dataset 并行处理
    # 传递可序列化的参数，避免序列化包含 embedding_model 的 distractor_manager
    augmented_ds = ds.map(
        fn=AugmentRowActor,
        fn_constructor_kwargs={
            "distractor_pool": distractor_manager.distractor_pool,
            "faiss_index_path": faiss_index_path,
            "faiss_meta_path": faiss_meta_path,
            "namespace": distractor_manager.embed_store.namespace,
            "target_len": TARGET_CONTEXT_LEN,
        },
        num_cpus=4,
        concurrency=(RAY_EMB_MIN_CONCURRENCY, RAY_SIMILARITY_CONCURRENCY),
    )
    
    # 转换回 pandas DataFrame
    logger.info("Converting Ray Dataset back to pandas DataFrame...")
    augmented_pd = augmented_ds.to_pandas()
    
    # 保存
    augmented_pd.to_parquet(output_file, index=False)
    logger.success(f"Saved to {output_file}")


def main():
    # 初始化 Ray（如果还没有初始化）
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        logger.info("Ray initialized")
    
    # 1. 获取子数据集列表
    logger.info(f"Found sub-datasets: {SUB_DATASETS}")
    # move legacy datafile to MiroVerse-Legacy
    os.makedirs(os.path.join(INPUT_DIR, "MiroVerse-Legacy"), exist_ok=True)
    for sub_dataset in SUB_DATASETS:
        legacy_datadir = os.path.join(INPUT_DIR, sub_dataset)
        if os.path.exists(legacy_datadir):
            shutil.move(legacy_datadir, os.path.join(INPUT_DIR, "MiroVerse-Legacy", sub_dataset))

    # 2. 构建全局候选池
    logger.info("Building distractor pool...")
    distractor_pool = build_distractor_pool(SUB_DATASETS, f"{INPUT_DIR}/MiroVerse-Legacy")
    logger.info(f"Distractor pool built. Total candidate sessions: {len(distractor_pool)}")

    if not distractor_pool:
        logger.error("No distractors found. Check data paths.")
        return

    # 2.1 初始化 embedding 模型和 EmbeddingStore
    logger.info("Initializing embedding model and EmbeddingStore...")
    emb_config = STEmbConfig()
    emb_config.embedding_model_name = EMBEDDING_MODEL_PATH
    emb_config.embedding_batch_size = EMBEDDING_BATCH_SIZE
    emb_config.embedding_max_seq_len = EMBEDDING_MAX_SEQ_LEN
    
    # 创建一个 embedding_model 实例用于 EmbeddingStore（实际编码会通过 Ray 完成）
    embedding_model = STEmbeddingModel(emb_config)
    
    # 初始化 EmbeddingStore（用于缓存 distractor embeddings）
    embed_store = EmbeddingStore(
        embedding_model=embedding_model,
        db_filename=EMBEDDING_STORE_PATH,
        batch_size=EMBEDDING_BATCH_SIZE,
        namespace="distractor_pool"
    )
    
    # 2.2 创建 DistractorEmbeddingManager
    logger.info("Creating DistractorEmbeddingManager...")
    distractor_manager = DistractorEmbeddingManager(
        distractor_pool=distractor_pool,
        embedding_model=embedding_model,
        embed_store=embed_store,
        chunk_size=DISTRACTOR_CHUNK_SIZE
    )

    # 3. 逐个处理数据集
    for sub_dataset in eval_datasets:
        for split in ["test", "train"]:
            input_file = os.path.join(INPUT_DIR, "MiroVerse-Legacy", sub_dataset, f"{split}.parquet")
            if not os.path.exists(input_file):
                continue
            
            # 预先计算所有 question 的 embedding（使用 Ray）
            logger.info(f"Precomputing question embeddings for {sub_dataset} {split}...")
            df = pd.read_parquet(input_file)
            # Clean question field before computing embeddings
            if "question" in df.columns:
                df["question"] = df["question"].apply(clean_question)
            questions = df["question"].tolist()
            
            # 初始化 EmbeddingStore（用于 question embeddings，仍然使用 EmbeddingStore）
            question_store = EmbeddingStore(
                embedding_model=embedding_model,
                db_filename=EMBEDDING_STORE_PATH,
                batch_size=EMBEDDING_BATCH_SIZE,
                namespace=f"questions_{sub_dataset}_{split}"
            )
            
            # 检查哪些 question 需要编码
            missing_questions = []
            missing_indices = []
            for idx, q in enumerate(questions):
                hash_id = compute_mdhash_id(q, prefix=question_store.namespace + "-")
                if hash_id not in question_store.hash_id_to_idx:
                    missing_questions.append(q)
                    missing_indices.append(idx)
            
            # 使用 Ray 并行编码缺失的 questions
            if missing_questions:
                logger.info(f"Found {len(missing_questions)} missing question embeddings, computing with Ray...")
                question_ds = ray.data.from_items([{"text": q} for q in missing_questions])
                
                embedded_question_ds = question_ds.map_batches(
                    fn=STEmbedActor,
                    fn_constructor_kwargs={
                        "emb_config": emb_config,
                        "embedding_model_name": EMBEDDING_MODEL_PATH,
                        "use_gpu": True,
                    },
                    batch_size=EMBEDDING_BATCH_SIZE,
                    concurrency=(RAY_EMB_MIN_CONCURRENCY, RAY_EMB_CONCURRENCY),
                    num_gpus=1,
                )
                
                # 获取结果
                embedded_question_pd = embedded_question_ds.to_pandas()
                missing_embeddings = embedded_question_pd["embedding"].tolist()
                
                # 写入 EmbeddingStore
                missing_hash_ids = [compute_mdhash_id(q, prefix=question_store.namespace + "-") for q in missing_questions]
                question_store.insert_embeddings(missing_hash_ids, missing_questions, missing_embeddings)
                logger.info(f"Cached {len(missing_questions)} new question embeddings")
            else:
                logger.info(f"All {len(questions)} question embeddings already cached")
            
            # 获取所有 question embeddings（包括已缓存的和新计算的）
            question_hash_ids = [compute_mdhash_id(q, prefix=question_store.namespace + "-") for q in questions]
            question_embeddings = question_store.get_embeddings(question_hash_ids)
            question_embeddings_dict = {idx: emb for idx, emb in enumerate(question_embeddings)}
            
            # 处理数据集
            process_dataset(sub_dataset, split, distractor_manager, question_embeddings_dict)


if __name__ == "__main__":
    # 设置随机种子以保证复现性
    random.seed(42)
    np.random.seed(42)
    main()