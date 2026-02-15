import argparse
import os
import sys

import yaml
from easydict import EasyDict as edict

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append("./")
import os

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
from typing import Dict, Any
from loguru import logger
import ray
from utils.ray_gen_utils import (build_vllm_processor, AddPromptActor, start_local_ray, PreFilterActor,
                                 parse_haystack_sessions, dump_haystack_sessions)
from utils.eval_utils import RankingParserActor
from utils.eval_utils import eval_prob_data

def vllm_inference(args: Dict[str, Any]):
    dataset_name = args["dataset_name"]
    dataset_split = args["dataset_split"]
    model_name = args["model_name"]
    debug = args["debug"]
    embed_model = args["embed_model"]
    output_dir = args["output_dir"]

    data_file = f"{output_dir}/{embed_model}/{dataset_name}_{dataset_split}_embed.parquet"
    output_file = f"{output_dir}/{model_name}/{dataset_name}_{dataset_split}_ranking.parquet"
    os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)


    ds = ray.data.read_parquet(data_file).map(parse_haystack_sessions)
    # pd_frame = pd.read_parquet(data_file, engine='fastparquet')
    # ds = ray.data.from_pandas(pd_frame)
    logger.info(f'Loaded {ds.count()} entries from {data_file}')
    if debug:
        ds = ds.limit(10)
        logger.info(f"Debug mode enabled, only processing {ds.count()} entries")

    # load prompt config
    prompt_name = args["prompt_name"]
    prompt_config = edict(yaml.load(open(f"./configs/{prompt_name}"), Loader=yaml.FullLoader))
    context_limit = 1024 * int(args["context_limit"].replace("k", ""))
    # model = Qwen3ForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    # output train data format: {"messages": [{"role": "system", "content": "你是个有用无害的助手"}, {"role": "user", "content": "告诉我明天的天气"}, {"role": "assistant", "content": "明天天气晴朗"}]}
    # construct train data
    # infer with ray
    # convert orig_data to ray dataset
    # ds = ray.data.from_items(orig_data)
    # json.loads(ds["haystack_sessions"])

    ds = ds.map(
        fn=PreFilterActor,
        fn_constructor_kwargs={
            "tokenizer_name": args["model_path"],
            "prompt_template": prompt_config["prompt_template"],
            "filter_col": args["filter_col"],
            "context_limit": context_limit,
        },
        concurrency=(1, args["override_num"] * 10)
    )
    ds = ds.map(
        fn=AddPromptActor,
        fn_constructor_kwargs={
            "prompt_config": prompt_config,
            "filtered_idx_col": "prefiltered_session_idx",
        },
        concurrency=(1, args["override_num"] * 10)
    )

    vllm_processor = build_vllm_processor(
        model_path=args["model_path"],
        max_num_batched_tokens=args["max_num_batched_tokens"],
        max_model_len=args["max_model_len"],
        batch_size=args["batch_size"],
        max_output_token=args["max_output_token"],
        temperature=args["temperature"],
        concurrency=args["concurrency"],
        tp=args["tp"],
        pp=args["pp"],
    )
    ds = vllm_processor(ds)
    ds = ds.map(
        fn = RankingParserActor,
        fn_constructor_kwargs={
            "filtered_idx_col": "prefiltered_session_idx",
        },
        concurrency=(1, args["override_num"] * 10)
    )
    # save ds
    ds = ds.map(dump_haystack_sessions)
    ds.to_pandas().to_parquet(output_file, index=False, compression="zstd")
    results = ds.take_all()
    logger.info(f"Total {ds.count()} entries processed, saved to {output_file}")

    # eval results
    logger.info(f"Start eval {output_file}")
    eval_prob_data(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action="store_true", help='是否采用采用local ray')
    parser.add_argument('--prompt_name', type=str, required=True, help='prompt配置文件名称')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称')
    parser.add_argument('--dataset_split', type=str, default="test", help='数据集划分')
    parser.add_argument('--embed_model', type=str, required=True, help='嵌入模型名称')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--override_num', type=int, required=True, help='读文件的并发')
    parser.add_argument('--model_path', type=str, required=True, help='llm模型地址')
    parser.add_argument('--model_name', type=str, required=True, help='llm模型名称')
    parser.add_argument('--max_num_batched_tokens', type=int, default=4096,
                        help='单个batch的最大token数')
    parser.add_argument('--max_model_len', type=int, default=8192,
                        help='输入+输出的最大长度')
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--pp', type=int, default=1)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--max_output_token', type=int, required=True)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--debug', action="store_true", help='是否开启debug模式')
    parser.add_argument('--filter_col', type=str, default="similarities", help='过滤列名')
    parser.add_argument('--context_limit', type=str, default="128k", help='上下文限制')
    args = parser.parse_args()
    p = vars(args)
    if p["local"]:
        start_local_ray()
    vllm_inference(args=p)