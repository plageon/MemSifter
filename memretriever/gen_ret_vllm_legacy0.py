import argparse
import asyncio
import json
from enum import Enum
from typing import List, Dict, Any

import aiohttp
import anytree
import loguru
import numpy as np
import requests
import yaml
from anytree import PreOrderIter, Node
from easydict import EasyDict as edict
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer


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
    haystack_dates = entry['haystack_dates']
    haystack_session_ids = entry['haystack_session_ids']

    his_root_node = HistoryNode(name=question, parent=None, children=None, node_type=NodeType.ROOT)
    all_turns = []  # 存储所有轮次信息

    # 首先收集所有轮次及其元数据
    for session_idx in range(len(haystack_sessions)):
        session = haystack_sessions[session_idx]
        date = haystack_dates[session_idx]
        session_id = haystack_session_ids[session_idx]

        session_prefix = f'<session date="{date}">\n'
        session_input_ids = tokenizer.encode(session_prefix, add_special_tokens=False)
        session_token_count = len(session_input_ids)

        user_id, assistant_id = 0, 0

        for turn in session:
            embedding_hash_id = turn.get('embedding_hash_id', '')
            similarity_score = turn.get('similarity_score', 0.0)  # 默认最低分

            if turn["role"] == "user":
                user_id += 1
                turn_name = f"user_{user_id}"
                turn_type = NodeType.USER
                turn_content = f'<user>{turn["content"]}</user>\n'
            elif turn["role"] == "assistant":
                assistant_id += 1
                turn_name = f"assistant_{assistant_id}"
                turn_type = NodeType.ASSISTANT
                turn_content = f'<assistant>{turn["content"]}</assistant>\n'
            else:
                loguru.logger.warning(f"Unknown role {turn['role']} in session {session_idx}, skipping this turn.")
                continue

            # 计算当前轮次的token数量
            turn_input_ids = tokenizer.encode(turn_content, add_special_tokens=False)
            turn_token_count = len(turn_input_ids)

            # 存储轮次信息
            all_turns.append({
                'session_idx': session_idx,
                'session_id': session_id,
                'session_prefix': session_prefix,
                'session_token_count': session_token_count,
                'turn': turn,
                'turn_name': turn_name,
                'turn_type': turn_type,
                'turn_content': turn_content,
                'input_ids': turn_input_ids,
                'token_count': turn_token_count,
                'embedding_hash_id': embedding_hash_id,
                'similarity_score': similarity_score
            })

    # 贪心算法筛选轮次：保留相似度高的轮次
    # 按相似度降序排序，但保持原有序列的相对位置（稳定排序）
    sorted_turn_indices = sorted(
        range(len(all_turns)),
        key=lambda i: (-all_turns[i]['similarity_score'], i)
    )
    # logger.info(f"sort turn indices: {sorted_turn_indices}")

    # 计算总token数并筛选
    total_tokens = 0
    kept_turn_indices = set()
    session_token_counts = {}  # 跟踪每个会话的token消耗

    for idx in sorted_turn_indices:
        turn = all_turns[idx]
        session_idx = turn['session_idx']

        # 会话首次被加入时，添加会话前缀的token数
        if session_idx not in session_token_counts:
            session_token_counts[session_idx] = turn['session_token_count']
            total_tokens += turn['session_token_count']

        # 检查是否可以加入当前轮次
        if total_tokens + turn['token_count'] <= max_content_length:
            kept_turn_indices.add(idx)
            total_tokens += turn['token_count']
        else:
            # 如果加入当前轮次会超出限制，则尝试移除会话前缀（如果该会话没有保留任何轮次）
            if session_idx in session_token_counts and session_idx not in [all_turns[k]['session_idx'] for k in
                                                                           kept_turn_indices]:
                total_tokens -= session_token_counts[session_idx]
                del session_token_counts[session_idx]

    # 构建保留的轮次列表，保持原始顺序
    kept_turns = [turn for i, turn in enumerate(all_turns) if i in kept_turn_indices]

    # 重新构建会话和节点树
    history = ""
    current_session_id = None
    current_session_node = None
    all_turn_token_ids = []

    for turn in kept_turns:
        if turn['session_id'] != current_session_id:
            # 处理新会话
            if current_session_node:
                # 关闭上一个会话
                current_session_node.text += '</session>\n'
                history += '</session>\n'

            current_session_id = turn['session_id']
            # 创建会话节点
            current_session_node = HistoryNode(
                name=f"session_{turn['session_id']}",
                session_id=turn['session_id'],
                node_type=NodeType.SESSION,
                text=turn['session_prefix'],
                prefix=turn['session_prefix'],
                parent=his_root_node,
                input_ids=tokenizer.encode(turn['session_prefix'], add_special_tokens=False)
            )
            history += turn['session_prefix']

        # 创建轮次节点
        turn_node = HistoryNode(
            name=turn['turn_name'],
            session_id=turn['session_id'],
            node_type=turn['turn_type'],
            text=turn['turn_content'],
            prefix=turn['session_prefix'] + f'<{turn["turn"]["role"]}>',
            parent=current_session_node,
            input_ids=current_session_node.input_ids + turn['input_ids'],
            embedding_hash_id=turn['embedding_hash_id'],
            similarity_score=turn['similarity_score']
        )

        # 更新会话文本和历史
        current_session_node.text += turn['turn_content']
        history += turn['turn_content']
        all_turn_token_ids.append(current_session_node.input_ids + turn['input_ids'])

    # 关闭最后一个会话
    if current_session_node:
        current_session_node.text += '</session>\n'
        history += '</session>\n'

    history = history.strip()

    return {
        "history": history,
        "all_turn_token_ids": all_turn_token_ids,
        "his_root_node": his_root_node,
        "kept_turn_count": len(kept_turns),
        "total_token_count": total_tokens
    }


async def process_single_entry(
        prompt_config: Any,
        entry: Dict,
        calculator: APIProbabilityCalculator,
        tokenizer: Any,
        worker_id: int,
        max_content_length: int,
):
    question = entry['question']
    answer = entry['answer']
    haystack_dates = entry['haystack_dates']
    haystack_sessions = entry['haystack_sessions']
    haystack_session_ids = entry['haystack_session_ids']
    answer_session_ids = entry['answer_session_ids']

    # construct history in xml format
    # <session date="2023-10-01">
    # <user>user query</user>
    # <assistant>assistant answer</assistant>
    # ...
    # </session>
    try:
        processed_haystack = process_haystack_sessions(tokenizer, entry, max_content_length)
        history = processed_haystack["history"]
        all_turn_token_ids = processed_haystack["all_turn_token_ids"]
        his_root_node = processed_haystack["his_root_node"]
        # construct token tree, tokenize each turn text and create token nodes under each turn node
        token_root = TokenIdNode(name="root", parent=None, children=None, input_ids=[], prob=np.float32(1.0))

        current_session = f'<session date="{entry["question_date"]}">\n<user>{question}</user>\n</session>'
        current_session = current_session.strip()
        output = "The most relevant turn is:\n"

        # construct the final prompt
        prompt_template = prompt_config.prompt_template
        user_prompt = prompt_template.user_prompt_text.format(history=history, current=current_session)
        user_prompt = user_prompt.strip()
        messages = [
            {"role": "system", "content": prompt_template.system_prompt_text},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": output}
        ]
        full_prompt_input_ids = tokenizer.apply_chat_template(messages, add_special_tokens=True,
                                                                         continue_final_message=True, tokenize=True,
                                                                         return_tensors=None, return_dict=False)
        # logger.info(f"full prompt input ids: ... {full_prompt_input_ids[-100:]}")

        # pbar = tqdm(total=len(all_turn_token_ids), desc=f"scanning token ids to build token tree")


        async def scan_token_ids(tgroup_token_ids, current_node):
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
                    prob_res = await calculator.call_prob_api(worker_id, complete_input_ids, force_token_id)
                    probs = prob_res.get("result", {})
                    # create child nodes for each token in next_token_set
                    for i, token_id in enumerate(next_token_set):
                        token_prob = probs.get(str(token_id), None)
                        if not token_prob:
                            # logger.warning(f"token id {token_id} not found in probs, set to 0.0")
                            token_prob = np.float32(0.0)
                        else:
                            token_prob = np.float32(token_prob)
                        child_node = TokenIdNode(name=token_id, parent=current_node,
                                                 input_ids=current_node.input_ids + [token_id], prob=token_prob)
                        # logger.debug(f"child node input_ids: {child_node.input_ids}, prob: {child_node.prob}, token: {calculator.tokenizer.decode([token_id], skip_special_tokens=False)}")
                        # scan the remaining token ids under this child node, recursively
                        child_tgroup_token_ids = [
                            tid[t_idx + 1:] for tid in tgroup_token_ids if len(tid) > t_idx and tid[t_idx] == token_id
                        ]
                        await scan_token_ids(child_tgroup_token_ids, child_node)
                    break  # stop scanning further
            return

        # logger.info(f"token_ids example: {all_turn_token_ids[0]}")
        # logger.info(f"full_prompt_input_ids: {full_prompt_input_ids[-100:]}")
        # exit(0)
        await scan_token_ids(all_turn_token_ids, token_root)
        # pbar.close()

        # show tree infos
        # for pre, _, node in anytree.RenderTree(token_root):
        #     print(f"{pre}{node.input_ids[-10:]}|{node.prob}")

        # calculate session & turn node probabilities
        for session_node in his_root_node.children:
            # logger.info(f"session node: {session_node.input_ids}")
            curr_token_node = token_root
            session_node.prob = np.float32(1.0)
            loop_count = 0
            while len(curr_token_node.children) and len(curr_token_node.input_ids) < len(session_node.input_ids):
                # find the child node with this token id
                # logger.info(f"current token node: {curr_token_node.input_ids}")
                found_child = False
                loop_count += 1
                for child_node in curr_token_node.children:
                    # logger.debug(f"child node: {child_node.input_ids}")
                    token_length = min(len(child_node.input_ids), len(session_node.input_ids))
                    if child_node.input_ids[:token_length] == session_node.input_ids[:token_length]:
                        # logger.info(f"Match child node: {child_node.input_ids}")
                        session_node.prob *= child_node.prob
                        curr_token_node = child_node
                        found_child = True
                        break
                if not found_child:
                    # logger.warning(f"Cannot find child node for session {session_node.name}, token ids: {session_node.input_ids}")
                    for all_turn_token_id in all_turn_token_ids:
                        if session_node.input_ids == all_turn_token_id[:len(session_node.input_ids)]:
                            # logger.info(f"Found session node {session_node.name} in turn {all_turn_token_id}")
                            break
                    session_node.prob = np.float32(0.0)
                    break
            session_root = curr_token_node
            # logger.info(f"Session node {session_node.input_ids} matched with token tree node: {curr_token_node.input_ids}")
            for turn_node in session_node.children:
                # logger.info(f"turn node: {turn_node.input_ids}")
                curr_token_node = session_root
                turn_node.prob = session_node.prob
                loop_count = 0
                while len(curr_token_node.children) and len(curr_token_node.input_ids) < len(turn_node.input_ids):
                    # find the child node with this token id
                    # logger.info(f"current token node: {curr_token_node.input_ids}")
                    found_child = False
                    loop_count += 1
                    for child_node in curr_token_node.children:
                        # logger.debug(f"child node: {child_node.input_ids}")
                        token_length = min(len(child_node.input_ids), len(turn_node.input_ids))
                        if child_node.input_ids[:token_length] == turn_node.input_ids[:token_length]:
                            # logger.info(f"Match child node: {child_node.input_ids}")
                            turn_node.prob *= child_node.prob
                            curr_token_node = child_node
                            found_child = True
                            break
                    if not found_child:
                        # logger.warning(f"Cannot find child node for turn {turn_node.name}, token ids: {turn_node.input_ids}")
                        for all_turn_token_id in all_turn_token_ids:
                            if turn_node.input_ids == all_turn_token_id[:len(turn_node.input_ids)]:
                                # logger.info(f"Found turn node {turn_node.name} in turn {all_turn_token_id}")
                                break
                        turn_node.prob = np.float32(0.0)
                        break
                # logger.info(f"Turn node {turn_node.input_ids} matched with token tree node: {curr_token_node.input_ids}")


        # save turn node probabilities to the original entry
        for session_idx in range(len(haystack_sessions)):
            found_session = False
            for session_node in his_root_node.children:
                if session_node.session_id == haystack_session_ids[session_idx]:
                    found_child = False
                    for turn_idx in range(len(haystack_sessions[session_idx])):
                        for turn_node in session_node.children:
                            if turn_node.embedding_hash_id == haystack_sessions[session_idx][turn_idx]['embedding_hash_id']:
                                haystack_sessions[session_idx][turn_idx]['gen_prob'] = float(turn_node.prob)
                                found_child = True
                                break
                        if not found_child:
                            haystack_sessions[session_idx][turn_idx]['gen_prob'] = 0.0
                    found_session = True
                    break
            if not found_session:
                for turn_idx in range(len(haystack_sessions[session_idx])):
                    haystack_sessions[session_idx][turn_idx]['gen_prob'] = 0.0


        # save session node probabilities to the original entry
        gen_session_probs = []

        for session_idx in range(len(haystack_sessions)):
            found_child = False
            for session_node in his_root_node.children:
                if session_node.session_id == haystack_session_ids[session_idx]:
                    gen_session_probs.append(float(session_node.prob))
                    found_child = True
                    break
            if not found_child:
                gen_session_probs.append(0.0)

            # logger.info(f"Session {session_idx}, id {haystack_session_ids[session_idx]}, gen prob: {float(gen_session_probs[session_idx])}")
        entry['gen_session_probs'] = gen_session_probs
        return entry
    except Exception as e:
        logger.error(f"Error processing entry: {e}")
        import traceback
        traceback.print_exc()
        return None


async def process_entries(
        entries: List[Dict],
        prompt_config: Any,
        debug: bool = False,
        tokenizer: Any = None,
        semaphore: asyncio.Semaphore = None,
        max_content_length: int = 65536,

) -> List[Dict]:
    api_servers = [
        "http://localhost:5500",
        "http://localhost:5501",
        "http://localhost:5502",
        "http://localhost:5503",
        "http://localhost:5504",
        "http://localhost:5505",
        "http://localhost:5506",
        "http://localhost:5507",
    ]
    if debug:
        entries = entries[:10]
        logger.info(f"Debug mode enabled, only processing {len(entries)} entries")
    api_servers = api_servers[:args.num_servers]

    # init calculators
    calculators = []
    for i in range(len(api_servers)):
        calculator = APIProbabilityCalculator(base_urls=api_servers, max_concurrent=4)
        await calculator.__aenter__()
        calculators.append(calculator)

    results = [None for _ in range(len(entries))]
    pbar = tqdm(total=len(entries), desc="Processing entries")

    async def process_entry_wrapper(entry: Dict, idx: int):
        # find a vacant worker
        async with semaphore:
            while True:
                worker_id = None
                for i in range(len(api_servers)):
                    server_index = (idx + i) % len(api_servers)
                    health_url = f"{api_servers[server_index]}/health"
                    try:
                        health_res = requests.get(health_url, timeout=300)
                    except requests.exceptions.RequestException as e:
                        logger.error(f"Error checking health of server {server_index}: {e}")
                        continue
                    if health_res.status_code == 200:
                        worker_id = server_index
                        break
                # sleep for 0.5 seconds
                await asyncio.sleep(0.5)
                break

            # logger.info(f"Server {worker_id} is available, using it")


            result = await process_single_entry(
                prompt_config=prompt_config,
                entry=entry,
                calculator=calculator,
                tokenizer=tokenizer,
                worker_id=worker_id,
                max_content_length=max_content_length,
            )
            results[idx] = result
            pbar.update(1)

    tasks = [process_entry_wrapper(entry, idx) for idx, entry in enumerate(entries)]

    await asyncio.gather(*tasks)
    pbar.close()

    # close calculators
    for calculator in calculators:
        await calculator.__aexit__(None, None, None)

    return results


# longmemeval_s, longmemeval_m, longmemeval_oracle
async def main(api_request_semaphore: asyncio.Semaphore,
        dataset_name: str,
        output_dir: str,
        split: str,
        embed_model: str,
        model_path: str,
        max_content_length: int,
               ):
    # data_file = f"../../../memory/{dataset_name}/{file_name}"
    data_file = f"{output_dir}/{embed_model}/{dataset_name}_{split}_embed.jsonl"
    logger.info(f"Loading data from {data_file}")
    model_tag = model_path.replace("/mnt/local/logs/jjtan/", "") \
        .replace("/mnt/jjtan/ckpt/", "") \
        .replace("/home/jjtan/models/", "") \
        .replace("/", "-")
    output_file = f"{output_dir}/{model_tag}/{dataset_name}_{split}_gen_ret.jsonl"

    # orig_data = json.load(open(data_file))
    orig_data = [json.loads(line.strip()) for line in open(data_file)]
    # print(orig_data[0].keys())
    # dict_keys(['question_id', 'question_type', 'question', 'answer', 'question_date', 'haystack_dates', 'haystack_session_ids', 'haystack_sessions', 'answer_session_ids'])

    # load prompt config
    prompt_name = "gen_retrieve_prompt.yaml"
    prompt_config = edict(yaml.load(open(f"./configs/{prompt_name}"), Loader=yaml.FullLoader))
    # print(prompt_config)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # model = Qwen3ForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    # output train data format: {"messages": [{"role": "system", "content": "你是个有用无害的助手"}, {"role": "user", "content": "告诉我明天的天气"}, {"role": "assistant", "content": "明天天气晴朗"}]}
    # construct train data
    results = await process_entries(
        entries=orig_data,
        prompt_config=prompt_config,
        debug=args.debug,
        tokenizer=tokenizer,
        semaphore=api_request_semaphore,
        max_content_length=args.max_content_length,
    )

    # save haystack sessions to file
    if max_content_length == 65536:
        output_file = output_file.replace(".jsonl", f"_64k.jsonl")
    elif max_content_length == 32768:
        output_file = output_file.replace(".jsonl", f"_32k.jsonl")
    elif max_content_length == 16384:
        output_file = output_file.replace(".jsonl", f"_16k.jsonl")
    else:
        logger.error(f"Unknown max_content_length: {max_content_length}")
    output_file = output_file.replace(".jsonl", ".legacy.jsonl")

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    logger.info(f"Total {len(results)} entries processed, saved to {output_file}")

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="vLLM Probability Calculator API")
    arg_parser.add_argument("--model-path", default="/mnt/local/logs/jjtan/4b-think-custom-163264/checkpoint-124", help="Path to the vLLM model")
    arg_parser.add_argument("--num-servers", type=int, default=4, help="Number of servers to use")
    arg_parser.add_argument("--max-concurrency", type=int, default=6, help="Maximum number of concurrent requests")
    arg_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    arg_parser.add_argument("--dataset-name", default="LongMemEval", help="Dataset name")
    arg_parser.add_argument("--split", default="s", help="Split of the dataset")
    arg_parser.add_argument("--max-content-length", type=int, default=16384, help="Maximum content length")
    arg_parser.add_argument("--embed-model", default="bge-m3", help="Embedding model")
    arg_parser.add_argument("--output-dir", default="results", help="Output directory")
    args = arg_parser.parse_args()
    api_request_semaphore = asyncio.Semaphore(args.max_concurrency)
    asyncio.run(main(api_request_semaphore, args.dataset_name, args.output_dir, args.split, args.embed_model, args.model_path, args.max_content_length))
