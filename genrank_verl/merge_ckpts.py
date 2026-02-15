import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from loguru import logger
from typing import Dict, List
import argparse


def main(args):
    model_dir = "../models/MemSifter"
    model_name = "MemSifter-Qwen3-4B-Task-Reward"
    ckpt_steps = args["ckpt_steps"].split(",")
    checkpoints = {
        f"{ckpt_step}": f"{model_dir}/{model_name}-step-{ckpt_step}"
        for ckpt_step in ckpt_steps
    }

    output_ckpt = f"{model_dir}/{model_name}-merged"
    os.makedirs(output_ckpt, exist_ok=True)

    print(f"Preparing to merge {len(checkpoints)} models (using Float32 high precision)...")
    model_paths = list(checkpoints.values())
    num_models = len(model_paths)

    # 1. Load base model (float32), get trainable parameter whitelist
    print(f"Loading base model (1/{num_models}) [Float32]: {model_paths[0]}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_paths[0],
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(model_paths[0], trust_remote_code=True)
    trainable_keys = set(pname for pname, _ in base_model.named_parameters())
    print(f"Trainable parameters: {len(trainable_keys)}, non-trainable parameters (buffers, etc.) will not participate in merging")

    # Key: Initialize merged weights dictionary (copy from base model to ensure all keys exist)
    merged_weights = {key: torch.zeros_like(base_model.state_dict()[key], dtype=torch.float32) for key in
                      trainable_keys}

    # 2. Load models one by one, accumulate average weights (divide then add)
    for i, path in enumerate(model_paths, start=1):
        print(f"Processing model ({i}/{num_models}) [Float32]: {path}")

        # Load model and extract state_dict (extract immediately after loading to reduce memory usage)
        merge_model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        merge_sd = merge_model.state_dict()

        # Validate that current model's trainable parameters match base (avoid key missing)
        current_trainable_keys = set(pname for pname, _ in merge_model.named_parameters())
        if current_trainable_keys != trainable_keys:
            missing_keys = trainable_keys - current_trainable_keys
            extra_keys = current_trainable_keys - trainable_keys
            raise ValueError(f"Model {path} trainable parameters don't match base! Missing: {missing_keys}, Extra: {extra_keys}")

        # Accumulate average weights (divide then add)
        for key in tqdm(trainable_keys, desc=f"Merging layers ({i}/{num_models})"):
            if torch.is_floating_point(merge_sd[key]):
                merged_weights[key] += merge_sd[key] / float(num_models)

        # Force release memory (key: delete model and state_dict to avoid memory overflow)
        del merge_model
        del merge_sd
        torch.cuda.empty_cache()

    # 3. Key step: Update base model with merged weights
    base_sd = base_model.state_dict()
    for key in trainable_keys:
        base_sd[key] = merged_weights[key]
    base_model.load_state_dict(base_sd)  # Apply updated weights

    # 4. Compatible dtype conversion (automatically detect if hardware supports bfloat16)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # Ampere+ architecture
        target_dtype = torch.bfloat16
    else:
        target_dtype = torch.float16  # Compatible with older GPUs/CPU
    print(f"Environment supports {target_dtype}, converting format to save storage space...")
    base_model = base_model.to(dtype=target_dtype)

    # 5. Validate and save Tokenizer (ensure all model Tokenizers are consistent)
    print("Validating Tokenizer consistency across all models...")
    for path in model_paths[1:]:
        test_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if test_tokenizer.get_vocab() != base_tokenizer.get_vocab() or test_tokenizer.special_tokens_map != base_tokenizer.special_tokens_map:
            raise ValueError(f"Model {path} Tokenizer doesn't match base, cannot merge!")
    print("Tokenizer consistency validation passed, starting save...")

    # 6. Save merged model and Tokenizer
    print(f"Saving merged model to: {output_ckpt}")
    base_model.save_pretrained(output_ckpt)
    base_tokenizer.save_pretrained(output_ckpt)

    print("Merge completed!")



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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Model directory")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--ckpt_steps", type=str, required=True, help="Checkpoint steps")
    args = parser.parse_args()
    main(vars(args))

