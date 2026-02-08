import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from loguru import logger
from typing import Dict, List
import argparse


def main():
    model_dir = "../../models/DAPO-GenRank"
    model_name = "ep3-DAPO-Qwen3-4B-Thinking"
    checkpoints = {
        "locomo": f"{model_dir}/{model_name}-step-50",
        "personamem": f"{model_dir}/{model_name}-step-10",
        "perltqa": f"{model_dir}/{model_name}-step-40",
    }

    output_ckpt = f"{model_dir}/{model_name}-merged-1"
    os.makedirs(output_ckpt, exist_ok=True)

    print(f"准备合并 {len(checkpoints)} 个模型 (使用 Float32 高精度计算)...")
    model_paths = list(checkpoints.values())
    num_models = len(model_paths)

    # 1. 加载基座模型（float32），获取可学习参数白名单
    print(f"正在加载基座模型 (1/{num_models}) [Float32]: {model_paths[0]}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_paths[0],
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(model_paths[0], trust_remote_code=True)
    trainable_keys = set(pname for pname, _ in base_model.named_parameters())
    print(f"可学习参数数量：{len(trainable_keys)}，非可学习参数（缓冲层等）将不参与合并")

    # 关键：初始化合并权重字典（从基座模型复制，确保所有 key 存在）
    merged_weights = {key: torch.zeros_like(base_model.state_dict()[key], dtype=torch.float32) for key in
                      trainable_keys}

    # 2. 逐个加载模型，累加平均权重（先除后加）
    for i, path in enumerate(model_paths, start=1):
        print(f"正在处理模型 ({i}/{num_models}) [Float32]: {path}")

        # 加载模型并提取 state_dict（加载后立即提取，减少内存占用）
        merge_model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        merge_sd = merge_model.state_dict()

        # 校验当前模型的可学习参数是否与基座一致（避免 key 缺失）
        current_trainable_keys = set(pname for pname, _ in merge_model.named_parameters())
        if current_trainable_keys != trainable_keys:
            missing_keys = trainable_keys - current_trainable_keys
            extra_keys = current_trainable_keys - trainable_keys
            raise ValueError(f"模型 {path} 的可学习参数与基座不一致！缺失：{missing_keys}，多余：{extra_keys}")

        # 累加平均权重（先除后加）
        for key in tqdm(trainable_keys, desc=f"Merging layers ({i}/{num_models})"):
            if torch.is_floating_point(merge_sd[key]):
                merged_weights[key] += merge_sd[key] / float(num_models)

        # 强制释放内存（关键：删除模型和 state_dict，避免内存溢出）
        del merge_model
        del merge_sd
        torch.cuda.empty_cache()

    # 3. 关键步骤：将合并后的权重更新到基座模型
    base_sd = base_model.state_dict()
    for key in trainable_keys:
        base_sd[key] = merged_weights[key]
    base_model.load_state_dict(base_sd)  # 应用更新后的权重

    # 4. 兼容型 dtype 转换（自动判断硬件是否支持 bfloat16）
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # Ampere+ 架构
        target_dtype = torch.bfloat16
    else:
        target_dtype = torch.float16  # 兼容旧显卡/CPU
    print(f"环境支持 {target_dtype}，正在转换格式以节省存储空间...")
    base_model = base_model.to(dtype=target_dtype)

    # 5. 校验并保存 Tokenizer（确保所有模型 Tokenizer 一致）
    print("校验所有模型的 Tokenizer 一致性...")
    for path in model_paths[1:]:
        test_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if test_tokenizer.get_vocab() != base_tokenizer.get_vocab() or test_tokenizer.special_tokens_map != base_tokenizer.special_tokens_map:
            raise ValueError(f"模型 {path} 的 Tokenizer 与基座不一致，无法合并！")
    print("Tokenizer 一致性校验通过，开始保存...")

    # 6. 保存合并后的模型和 Tokenizer
    print(f"正在保存合并后的模型到: {output_ckpt}")
    base_model.save_pretrained(output_ckpt)
    base_tokenizer.save_pretrained(output_ckpt)

    print("合并完成！")



def extract_ckpt_weight(ckpt_path: str) -> Dict[str, torch.Tensor]:
    assert os.path.exists(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    logger.info(f"ckpt keys {ckpt.keys()}")
    if 'state_dict' in ckpt:
        weights = ckpt['state_dict']
    else:
        weights = ckpt
    return weights


def extracts(ckpt_paths: List[str]):
    logger.info(f"ckpt paths is {ckpt_paths}")
    weights = {}
    if len(ckpt_paths) == 1:
        weights = extract_ckpt_weight(ckpt_paths[0])
    else:
        for ckpt in ckpt_paths:
            weight = extract_ckpt_weight(ckpt)
            for k, v in weight.items():
                if k not in weights:
                    weights[k] = v / (float(len(ckpt_paths)))
                else:
                    weights[k] += v / (float(len(ckpt_paths)))
    dir_name=os.path.dirname(ckpt_paths[0])
    base_name=[os.path.basename(e).replace('.ckpt','') for e in ckpt_paths]
    new_base_name="_".join(base_name)+"_weight.ckpt"
    new_ckpt_path = os.path.join(dir_name,new_base_name)
    logger.info(f"new ckpt path {new_ckpt_path}")
    torch.save({'state_dict': weights}, new_ckpt_path)
    logger.info(f"extract weight {ckpt_paths} ->{new_ckpt_path} size {os.path.getsize(ckpt_paths[0]) / (1024 * 1024)}MB"
                f"-> {os.path.getsize(new_ckpt_path) / (1024 * 1024)}MB")


if __name__ == "__main__":
    main()

