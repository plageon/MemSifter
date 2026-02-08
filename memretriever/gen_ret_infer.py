import json
import os
import random
from enum import Enum

import anytree
import loguru
import numpy as np
import torch
import yaml
from accelerate import Accelerator, infer_auto_device_map, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory
from anytree import PreOrderIter, Node
from easydict import EasyDict as edict
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen3ForCausalLM, Qwen3Config


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
        self.prefix = kwargs.get('prefix', "")
        self.text = kwargs.get('text', "")
        self.embedding = kwargs.get('embedding', np.array([]))
        self.prob = kwargs.get('prob', np.float32(0.0))
        self.input_ids = kwargs.get('input_ids', [])


class TokenIdNode(Node):
    def __init__(self, name, parent=None, children=None, **kwargs):
        super().__init__(name, parent, children, **kwargs)
        self.input_ids = kwargs.get('input_ids', [])
        self.prob = kwargs.get('prob', np.float32(0.0))


def load_model_with_optimized_parallel(model_path, trust_remote_code=True):
    """
    优化的多卡模型加载，提升GPU利用率和负载均衡
    """
    # 初始化加速器，使用多进程分布式模式
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "1"  # 启用DeepSpeed优化
    accelerator = Accelerator()
    device_count = torch.cuda.device_count()
    accelerator.print(f"检测到 {device_count} 张GPU，开始优化分配...", main_process_only=True)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        padding_side="left"  # 优化长序列处理
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 配置模型启用FlashAttention并优化并行策略
    config = Qwen3Config.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code
    )
    config.attn_implementation = "flash_attention_2"
    config.use_cache = False  # 关闭缓存减少显存碎片（推理时可按需开启）

    # 加载模型基础结构（不加载权重）
    model = Qwen3ForCausalLM(config=config, trust_remote_code=trust_remote_code)

    # 计算更精细的内存分配方案
    max_memory = get_balanced_memory(
        model,
        dtype=torch.float16,
        max_memory={i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.9)}B"
                   for i in range(device_count)},  # 使用90%的GPU内存
        n_param_per_gpu_frac=0.9,  # 提高单卡参数占比
        low_zero=False
    )

    # 生成更均衡的设备映射
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        dtype=torch.float16,
        no_split_module_classes=model._no_split_modules,
        # 强制将大型模块分散到不同GPU
        force_hf_bit16=True
    )

    # 加载并分发模型权重
    model = load_checkpoint_and_dispatch(
        model,
        model_path,
        device_map=device_map,
        dtype=torch.float16,
        offload_folder="offload",
        offload_state_dict=False,  # 减少CPU卸载，优先使用GPU
        max_memory=max_memory
    )

    # 准备模型进行分布式推理
    model = accelerator.prepare(model)
    model.eval()  # 确保模型处于评估模式

    # 打印设备分配情况（主进程）
    if accelerator.is_main_process:
        accelerator.print("模型层设备分配情况:")
        for name, param in model.named_parameters():
            if param.device.type == "cuda":
                accelerator.print(f"  {name}: {param.device}")

    return tokenizer, model, accelerator


@torch.no_grad()
def calculate_forced_token_prob(model, accelerator, input_ids, force_token_id, **kwargs):
    """
    计算指定token的概率，支持多卡分布和FlashAttention-2

    Args:
        model: 分布式模型（使用FlashAttention-2）
        accelerator: Accelerator实例
        input_ids: 输入的token ids
        force_token_id: 需要计算概率的token id
        **kwargs: 其他模型参数

    Returns:
        probs: 计算得到的概率
    """
    # 将输入转换为张量并添加批次维度
    child_input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    # 准备模型输入
    model_inputs = model.prepare_inputs_for_generation(child_input_ids, **kwargs)

    # 将输入移动到适当的设备
    model_inputs = {k: v.to(accelerator.device) for k, v in model_inputs.items()
                    if isinstance(v, torch.Tensor)}

    # 模型前向传播（使用FlashAttention-2加速）
    outputs = model(
        **model_inputs,
        return_dict=True,
    )

    # 只在主进程上计算概率（避免重复计算）
    if accelerator.is_main_process:
        # 获取最后一个token的logits
        last_logits = outputs.logits[:, -1, :]

        # 计算softmax得到概率分布
        probs = torch.nn.functional.softmax(last_logits, dim=-1)

        # 提取目标token的概率
        force_token_id = torch.tensor(force_token_id, device=probs.device)
        target_prob = probs[0, force_token_id].detach().to(torch.float32).cpu().numpy()

        return target_prob
    else:
        return None


# @torch.no_grad()
# def calculate_forced_token_prob(model, input_ids, force_token_id, **kwargs):
#     child_input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
#     model_inputs = model.prepare_inputs_for_generation(child_input_ids, **kwargs)
#     # switch device to model device
#     for k in model_inputs:
#         if isinstance(model_inputs[k], torch.Tensor):
#             model_inputs[k] = model_inputs[k].to(model.device)
#     outputs = model(
#         **model_inputs,
#         return_dict=True,
#     )
#     #  get the probability of force_token_id
#     force_token_id = torch.tensor(force_token_id, device=model.device)
#     probs = torch.gather(outputs.logits[:, -1, :], -1, force_token_id.unsqueeze(0))
#     # softmax
#     probs = torch.nn.functional.softmax(probs, dim=-1)
#     probs = probs.squeeze(0).detach().to(torch.float32).cpu().numpy()
#     return probs


# longmemeval_s, longmemeval_m, longmemeval_oracle
if __name__ == '__main__':
    dataset_name = "LongMemEval"
    file_name = "longmemeval_s_cleaned.json"
    # data_file = f"../../../memory/{dataset_name}/{file_name}"
    data_file = f"../../data/memretriever/{dataset_name}/{file_name}"
    history_keep_rate = 0.5
    output_file = f"./data/{file_name.split('.')[0]}_his{history_keep_rate}_train.jsonl"

    orig_data = json.load(open(data_file))
    # print(orig_data[0].keys())
    # dict_keys(['question_id', 'question_type', 'question', 'answer', 'question_date', 'haystack_dates', 'haystack_session_ids', 'haystack_sessions', 'answer_session_ids'])

    # load prompt config
    prompt_name = "gen_retrieve_prompt.yaml"
    prompt_config = edict(yaml.load(open(f"./configs/{prompt_name}"), Loader=yaml.FullLoader))
    # print(prompt_config)
    train_data = []

    model_path = "/mnt/local/logs/jjtan/4b-ins-s-0.5/checkpoint-40"
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # model = Qwen3ForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer, model, accelerator = load_model_with_optimized_parallel(model_path, trust_remote_code=True)

    # output train data format: {"messages": [{"role": "system", "content": "你是个有用无害的助手"}, {"role": "user", "content": "告诉我明天的天气"}, {"role": "assistant", "content": "明天天气晴朗"}]}
    # construct train data
    for i in tqdm(range(len(orig_data)), desc="Processing data entries"):
        entry = orig_data[i]
        question = entry['question']
        answer = entry['answer']
        haystack_dates = entry['haystack_dates']
        haystack_sessions = entry['haystack_sessions']
        haystack_session_ids = entry['haystack_session_ids']
        answer_session_ids = entry['answer_session_ids']
        history = ""

        # construct history in xml format
        # <session date="2023-10-01">
        # <user>user query</user>
        # <assistant>assistant answer</assistant>
        # ...
        # </session>

        # select keep rate% of history sessions
        num_history_sessions = len(haystack_sessions) - len(answer_session_ids)
        num_kept_haystack_session = max(1, int(num_history_sessions * history_keep_rate))
        kept_haystack_session_indices = random.sample(haystack_session_ids, num_kept_haystack_session)
        kept_haystack_session_indices += answer_session_ids

        his_root_node = HistoryNode(name=question, parent=None, children=None, node_type=NodeType.ROOT)
        all_turn_token_ids = []

        for j in range(len(haystack_sessions)):
            session = haystack_sessions[j]
            date = haystack_dates[j]
            session_id = haystack_session_ids[j]
            if session_id not in kept_haystack_session_indices:
                continue
            session_prefix = f'<session date="{date}">\n' # do not include session id in case of data leakage
            session_history = session_prefix
            user_id, assistant_id = 0, 0
            session_input_ids = tokenizer.encode(session_prefix, add_special_tokens=False)
            session_node = HistoryNode(name=f"session_{session_id}", node_type=NodeType.SESSION, text="", prefix=session_prefix, parent=his_root_node, input_ids=session_input_ids)

            for turn in session:
                if turn["role"] == "user":
                    turn_prefix = session_prefix + f'<user>'
                    turn_history = f'{session_prefix}<user>{turn["content"]}</user>\n'
                    user_id += 1

                elif turn["role"] == "assistant":
                    turn_prefix = session_prefix + f'<assistant>'
                    turn_history = f'{session_prefix}<assistant>{turn["content"]}</assistant>\n'
                    assistant_id += 1
                else:
                    loguru.logger.warning(f"Unknown role {turn['role']} in session {j}, skipping this turn.")
                    continue

                turn_input_ids = tokenizer.encode(turn_history, add_special_tokens=False)
                turn_node = HistoryNode(name=f"user_{user_id}", node_type=NodeType.USER, text=turn_history, prefix=turn_prefix, parent=session_node, input_ids=turn_input_ids)
                turn_text = session_prefix + turn_history
                session_history += turn_history
                all_turn_token_ids.append(turn_node.input_ids)

            session_history += '</session>\n'
            session_node.text = session_history
            history += session_history

        history = history.strip()
        # construct token tree, tokenize each turn text and create token nodes under each turn node
        token_root = TokenIdNode(name="root", parent=None, children=None, input_ids=[], prob=np.float32(1.0))
        current_token_node = token_root

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
        full_prompt_input_ids = tokenizer.apply_chat_template(messages, add_special_tokens=True, continue_final_message=True, tokenize=True, return_tensors=None, return_dict=False)
        logger.info(f"full prompt input ids: ... {full_prompt_input_ids[-100:]}")

        pbar = tqdm(total=len(all_turn_token_ids), desc=f"scanning token ids to build token tree")

        def scan_token_ids(tgroup_token_ids, current_node):
            if len(tgroup_token_ids) == 1:
                # only one token id sequence, add all remaining token ids to current node
                current_node.input_ids += tgroup_token_ids[0]
                pbar.update(1)
                return
            for t_idx in range(min(len(tid) for tid in tgroup_token_ids)):
                next_token_set = set(
                    tgroup_token_ids[t][t_idx] for t in range(len(tgroup_token_ids)) if t_idx < len(tgroup_token_ids[t])
                )
                if len(next_token_set) == 1:
                    token_id = next_token_set.pop()
                    # check if current token node has this child
                    current_node.input_ids.append(token_id) # add to current input_ids
                else:
                    force_token_id = list(next_token_set)
                    complete_input_ids = full_prompt_input_ids + current_node.input_ids
                    probs = calculate_forced_token_prob(model, accelerator, complete_input_ids, force_token_id)

                    # create child nodes for each token in next_token_set
                    for i, token_id in enumerate(next_token_set):
                        child_node = TokenIdNode(name=token_id, parent=current_node, input_ids=current_node.input_ids + [token_id], prob=probs[i])
                        # scan the remaining token ids under this child node, recursively
                        child_tgroup_token_ids = [
                            tid[t_idx + 1:] for tid in tgroup_token_ids if len(tid) > t_idx and tid[t_idx] == token_id
                        ]
                        scan_token_ids(child_tgroup_token_ids, child_node)
                    break # stop scanning further
            return

        scan_token_ids(all_turn_token_ids, token_root)
        pbar.close()

        # show tree infos
        for pre, _, node in anytree.RenderTree(token_root):
            print(f"{pre}{node.name}|{node.prob}")

        # calculate session node probabilities
        for session_node in his_root_node.children:
            curr_token_node = token_root
            session_node.prob = np.float32(1.0)
            while len(curr_token_node.children) and len(curr_token_node.input_ids) < len(session_node.input_ids):
                # find the child node with this token id
                # logger.info(f"current token node: {curr_token_node.name}, input_ids: {curr_token_node.input_ids}, children: {[child.name for child in curr_token_node.children]}")
                found_child = False
                for child_node in token_root.children:
                    token_length = len(child_node.input_ids)
                    if child_node.input_ids == curr_token_node.input_ids[:token_length]:
                        session_node.prob *= child_node.prob
                        curr_token_node = child_node
                        found_child = True
                        break
                if not found_child:
                    logger.warning(f"Cannot find child node for session {session_node.name}, token ids: {session_node.input_ids}")
                    session_node.prob = np.float32(0.0)
                    break

        # show session node probs
        session_probs = []
        for session_node in his_root_node.children:
            session_probs.append((session_node.name, session_node.prob))
        print(session_probs)
        session_probs = sorted(session_probs, key=lambda x: x[1], reverse=True)
        print(f"sorted session probs: {session_probs}")
        break

