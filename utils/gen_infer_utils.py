import asyncio
import sys
from enum import Enum
from typing import List, Dict, Any

import aiohttp
import anytree
import numpy as np
from anytree import PreOrderIter, Node
from easydict import EasyDict as edict
from loguru import logger
from transformers import AutoTokenizer

sys.path.append("./")
from utils.session_process import construct_session_text


# enum node type including root, session, user, assistant
class NodeType(Enum):
    ROOT = 0
    SESSION = 1
    USER = 2
    ASSISTANT = 3
    UNKNOWN = 4


class HistoryNode(Node):
    def __init__(self, name, parent=None, children=None, **kwargs):
        super().__init__(name, parent, children, **kwargs)
        self.node_type: NodeType = kwargs.get('node_type', NodeType.UNKNOWN)
        self.session_id = kwargs.get('session_id', "")
        self.prefix = kwargs.get('prefix', "")
        self.text = kwargs.get('text', "")
        self.embedding_hash_id = kwargs.get('embedding_hash_id', "")
        self.similarity_score = kwargs.get('similarity_score', np.float32(0.0))
        self.prob = kwargs.get('prob', np.float32(0.0))
        self.input_ids = kwargs.get('input_ids', [])


class TokenIdNode(Node):
    def __init__(self, name, parent=None, children=None, **kwargs):
        super().__init__(name, parent, children, **kwargs)
        self.input_ids = kwargs.get('input_ids', [])
        self.prob = kwargs.get('prob', np.float32(0.0))


class APIProbabilityCalculator:
    # curl - X
    # POST
    # http: // localhost: 8000 / calculate - probabilities \
    #                     - H
    # "Content-Type: application/json" \
    # - d
    # '{"input_ids": [101, 2054, 2003], "force_token_ids": [1996, 2000, 2001]}'
    def __init__(self, base_urls: List[str], max_concurrent: int = 4):
        """
        初始化异步API客户端

        Args:
            base_urls: API服务器的基础URL列表
            max_concurrent: 最大并发数
        """
        self.base_urls = base_urls
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None

    async def __aenter__(self):
        """进入上下文管理器，创建aiohttp会话"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """退出上下文管理器，关闭aiohttp会话"""
        if self.session:
            await self.session.close()

    async def call_prob_api(self, worker_id: int, input_ids: List[int], force_token_ids: List[int]) -> Dict[str, Any]:
        """调用单个API服务器"""
        base_url = self.base_urls[worker_id]
        url = f"{base_url}/calculate-probabilities"
        payload = {
            "input_ids": input_ids,
            "force_token_ids": force_token_ids
        }

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "result": result,
                        "server": base_url,
                        "status": response.status
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": error_text,
                        "server": base_url,
                        "status": response.status
                    }
        except Exception as e:
            logger.error(f"Error calling {base_url}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "server": base_url,
                "status": None
            }

    async def check_health(self, model_name: str) -> bool:
        """检查所有服务器健康状态"""
        url = f"{self.base_urls[0]}/health"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    model_name_in_health = result.get("model_name", "")
                    return model_name_in_health == model_name
                else:
                    error_text = await response.text()
                    logger.error(f"Error calling {url}: {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Error calling {url}: {str(e)}")
            return False



def compare_input_ids(list1: List[int], list2: List[int]) -> bool:
    if len(list1) != len(list2):
        return False
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            return False
    return True


def process_haystack_sessions(tokenizer: Any, entry: Dict, max_content_length: int = 65536):
    question = entry['question']
    haystack_sessions = entry['haystack_sessions']
    haystack_session_similarity_scores = entry['haystack_session_similarity_scores']
    haystack_session_embedding_hash_ids = entry['haystack_session_embedding_hash_ids']
    haystack_session_ids = entry['haystack_session_ids']

    his_root_node = HistoryNode(name=question, parent=None, children=None, node_type=NodeType.ROOT)
    # all_turns = []  # 存储所有轮次信息
    all_sessions = []

    # 首先收集所有轮次及其元数据
    for session_idx in range(len(haystack_sessions)):
        session = haystack_sessions[session_idx]
        session_id = haystack_session_ids[session_idx]

        session_prefix = ""  # "<session>"
        session_text = construct_session_text(session)
        session_text = f"<session{session_idx}>{session_text}</session>\n"
        session_input_ids = tokenizer.encode(session_text, add_special_tokens=False)
        session_token_count = len(session_input_ids)

        embedding_hash_id = haystack_session_embedding_hash_ids[session_idx]
        similarity_score = haystack_session_similarity_scores[session_idx]

        all_sessions.append({
            'session_idx': session_idx,
            'session_id': session_id,
            'session_prefix': session_prefix,
            'session_token_count': session_token_count,
            'session': session,
            'session_type': NodeType.SESSION,
            'session_content': session_text,
            'input_ids': session_input_ids,
            'token_count': session_token_count,
            'embedding_hash_id': embedding_hash_id,
            'similarity_score': similarity_score
        })

    # 贪心算法筛选轮次：保留相似度高的轮次
    # 按相似度降序排序，但保持原有序列的相对位置（稳定排序）
    sorted_session_indices = sorted(
        range(len(all_sessions)),
        key=lambda i: (-all_sessions[i]['similarity_score'], i)
    )
    # logger.info(f"sort turn indices: {sorted_turn_indices}")

    # 计算总token数并筛选
    total_tokens = 0
    kept_session_indices = set()
    session_token_counts = {}  # 跟踪每个会话的token消耗

    for idx in sorted_session_indices:
        session = all_sessions[idx]
        session_idx = session['session_idx']

        # 会话首次被加入时，添加会话前缀的token数
        if session_idx not in session_token_counts:
            session_token_counts[session_idx] = 0  # session['session_token_count']
            # total_tokens += session['session_token_count']

        # 检查是否可以加入当前轮次
        if total_tokens + session['token_count'] <= max_content_length:
            kept_session_indices.add(idx)
            total_tokens += session['token_count']
        else:
            pass

    # 构建保留的轮次列表，保持原始顺序
    kept_sessions = [session for i, session in enumerate(all_sessions) if i in kept_session_indices]

    # 重新构建会话和节点树
    history = ""
    current_session_id = None
    current_session_node = None
    all_session_token_ids = []

    for session in kept_sessions:
        if session['session_id'] != current_session_id:
            # 处理新会话

            current_session_id = session['session_id']
            # 创建会话节点
            current_session_node = HistoryNode(
                name=f"session_{session['session_id']}",
                session_id=session['session_id'],
                node_type=NodeType.SESSION,
                text=session['session_content'],
                prefix=session['session_prefix'],
                parent=his_root_node,
                input_ids=tokenizer.encode(session['session_prefix'], add_special_tokens=False) + session['input_ids'],
                embedding_hash_id=session['embedding_hash_id'],
                similarity_score=session['similarity_score']
            )
            history += session['session_prefix'] + session['session_content']
        all_session_token_ids.append(current_session_node.input_ids)

    history = history.strip()

    return {
        "history": history,
        "all_turn_token_ids": all_session_token_ids,
        "his_root_node": his_root_node,
        "kept_turn_count": len(kept_sessions),
        "total_token_count": total_tokens
    }


class ProcessHaystackActor:
    def __init__(self, tokenizer: Any, max_content_length: int = 65536):
        self.tokenizer = tokenizer
        self.max_content_length = max_content_length

    def __call__(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        haystack_res = process_haystack_sessions(
            tokenizer=self.tokenizer,
            entry=entry,
            max_content_length=self.max_content_length
        )

        entry.update(haystack_res)
        return entry

class GatherWorkLoadActor:
    def __init__(self, tokenizer: AutoTokenizer, prompt_config: edict):
        self.tokenizer = tokenizer
        self.prompt_config = prompt_config

    def __call__(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        history = entry["history"]
        all_turn_token_ids = entry["all_turn_token_ids"]
        his_root_node = entry["his_root_node"]
        question = entry["question"]
        # construct token tree, tokenize each turn text and create token nodes under each turn node
        token_root = TokenIdNode(name="root", parent=None, children=None, input_ids=[], prob=np.float32(1.0))

        current_session = f'<session>\n<user>{question}</user>\n</session>'
        current_session = current_session.strip()
        output = "The most relevant session is:\n"

        # construct the final prompt
        prompt_template = self.prompt_config.prompt_template
        user_prompt = prompt_template.user_prompt_text.format(history=history, current=current_session)
        user_prompt = user_prompt.strip()
        messages = [
            {"role": "system", "content": prompt_template.system_prompt_text},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": output}
        ]
        full_prompt_input_ids = self.tokenizer.apply_chat_template(messages, add_special_tokens=True,
                                                              continue_final_message=True, tokenize=True,
                                                              return_tensors=None, return_dict=False)

        # logger.info(f"full prompt input ids: ... {full_prompt_input_ids[-100:]}")

        # pbar = tqdm(total=len(all_turn_token_ids), desc=f"scanning token ids to build token tree")
        infer_workloads = []
        def scan_token_ids(tgroup_token_ids, current_node):
            if len(tgroup_token_ids) == 1:
                # only one token id sequence, add all remaining token ids to current node
                current_node.input_ids += tgroup_token_ids[0]
                # pbar.update(1)
                return
            for t_idx in range(min(len(tid) for tid in tgroup_token_ids)):
                next_token_set = set(
                    tgroup_token_ids[t][t_idx] for t in range(len(tgroup_token_ids)) if t_idx < len(tgroup_token_ids[t])
                )
                if len(next_token_set) == 1:
                    token_id = next_token_set.pop()
                    # check if current token node has this child
                    current_node.input_ids.append(token_id)  # add to current input_ids
                else:
                    force_token_id = list(next_token_set)
                    complete_input_ids = full_prompt_input_ids + current_node.input_ids
                    # logger.debug(f"input_ids: {current_node.input_ids}, force_token_id: {force_token_id}")
                    # prob_res = calculator.call_prob_api(worker_id, complete_input_ids, force_token_id)
                    infer_workloads.append({
                        "complete_input_ids": complete_input_ids,
                        "force_token_id": force_token_id
                    })
                    # create child nodes for each token in next_token_set
                    for i, token_id in enumerate(next_token_set):
                        child_node = TokenIdNode(name=token_id, parent=current_node,
                                                 input_ids=current_node.input_ids + [token_id], prob=np.float32(0.0))
                        # logger.debug(f"child node input_ids: {child_node.input_ids}, prob: {child_node.prob}, token: {calculator.tokenizer.decode([token_id], skip_special_tokens=False)}")
                        # scan the remaining token ids under this child node, recursively
                        child_tgroup_token_ids = [
                            tid[t_idx + 1:] for tid in tgroup_token_ids if len(tid) > t_idx and tid[t_idx] == token_id
                        ]
                        scan_token_ids(child_tgroup_token_ids, child_node)
                    break  # stop scanning further
            return

        scan_token_ids(all_turn_token_ids, token_root)
        entry["infer_workloads"] = infer_workloads
        return entry

