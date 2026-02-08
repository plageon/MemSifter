import sys

import datasets
import pandas as pd

sys.path.append("./")

# dataset_dir = "../../data/memretriever/LongMemEval"
# data_sizes = ["s", "m"]
# train_ids = set()
# test_ids = set()
# for size in data_sizes:
    # file_name = f"longmemeval_{size}_cleaned.json"
    #
    # with open(f"{dataset_dir}/{file_name}", "r") as f:
    #     data = json.load(f)
    #
    # # split data into train and test set
    # # set random seed
    # if not train_ids and not test_ids:
    #     random.seed(42)
    #     random.shuffle(data)
    #
    #     train_data = data[:int(len(data) * 0.7)]
    #     test_data = data[int(len(data) * 0.7):]
    #     train_ids = set([entry['question_id'] for entry in train_data])
    #     test_ids = set([entry['question_id'] for entry in test_data])
    # else:
    #     train_data, test_data = [], []
    #     for entry in data:
    #         if entry['question_id'] in train_ids:
    #             train_data.append(entry)
    #         elif entry['question_id'] in test_ids:
    #             test_data.append(entry)
    #         else:
    #             logger.warning(f"Question {entry['question_id']} not in train or test set")
    #
    # # save to split_longmemeval
    # split_data_dir = f"./data/split_longmemeval_{size}"
    # # save original data
    # os.makedirs(f"{split_data_dir}", exist_ok=True)
    # file_path = f"{split_data_dir}/{file_name.replace('.json', '.jsonl')}"
    # with open(file_path, "w") as f:
    #     for entry in data:
    #         f.write(json.dumps(entry) + "\n")
    #
    # for split in ["train", "test"]:
    #     file_path = f"{split_data_dir}/{split}.jsonl"
    #     if split == "train":
    #         data_entries = train_data
    #     else:
    #         data_entries = test_data
    #     with open(file_path, "w") as f:
    #         for entry in data_entries:
    #             f.write(json.dumps(entry) + "\n")
    #
    #     logger.info(f"Wrote {len(data_entries)} entries to {file_path}")

# split embed infer results
# load split dataset

# dataset = f"split_longmemeval_{size}"
# split_data_dir = f"./data/{dataset}"
#
# test_data = [json.loads(l) for l in open(f"{split_data_dir}/test.jsonl")]
# test_ids = [entry['question_id'] for entry in test_data]
# test_ids = set(test_ids)
#
# train_data = [json.loads(l) for l in open(f"{split_data_dir}/train.jsonl")]
# train_ids = [entry['question_id'] for entry in train_data]
# train_ids = set(train_ids)
#
# # load embed infer results
# embed_model = "bge-m3"
# embed_result_path = f"results/{embed_model}/longmemeval_{size}_cleaned_embed_records.jsonl"
#
# embed_infer_results = [json.loads(l) for l in open(embed_result_path)]
# # split embed infer results into train and test set
# test_embed_infer_results = [entry for entry in embed_infer_results if entry['question_id'] in test_ids]
# train_embed_infer_results = [entry for entry in embed_infer_results if entry['question_id'] in train_ids]
#
# # save split embed infer results
# test_embed_infer_path = f"results/{embed_model}/{dataset}_test_embed.jsonl"
# train_embed_infer_path = f"results/{embed_model}/{dataset}_train_embed.jsonl"
#
# with open(test_embed_infer_path, "w") as f:
#     for entry in test_embed_infer_results:
#         f.write(json.dumps(entry) + "\n")
#
# logger.info(f"Wrote {len(test_embed_infer_results)} entries to {test_embed_infer_path}")
#
# with open(train_embed_infer_path, "w") as f:
#     for entry in train_embed_infer_results:
#         f.write(json.dumps(entry) + "\n")
#
# logger.info(f"Wrote {len(train_embed_infer_results)} entries to {train_embed_infer_path}")

# analyze qa pair annotations
# load qa pair annotations
# qa_pair_annotations = json.load(open(f"data/custom_longmemeval/qa_pairs_v1.json"))
# # calculate `the user` occurrence
# user_occ = 0
# for qa_pair in qa_pair_annotations:
#     question = qa_pair["question_content"]["question"]
#     if "the user" in question.lower():
#         user_occ += 1
#
# logger.info(f"Total {len(qa_pair_annotations)} qa pairs, {user_occ} of them contain `the user`")

# merge sample_haystack_and_timestamp_v1_16k.json, 32k and 64k
# base_dir = "data/custom_longmemeval/"
# train_data_path = os.path.join(base_dir, "train.jsonl")
# train_data = []
# for l in ["16k", "32k", "64k"]:
#     train_data += json.load(open(os.path.join(base_dir, f"sample_haystack_and_timestamp_v1_{l}.json")))
# logger.info(f"Loaded {len(train_data)} samples from 16k, 32k and 64k")
#
# with open(train_data_path, "w") as f:
#     for entry in train_data:
#         f.write(json.dumps(entry, ensure_ascii=False) + "\n")
#
# # filter train sample according to embedding results
# embed_model = "bge-m3"
# dataset = "custom_longmemeval"
# train_embed_infer_path = f"results/{embed_model}/{dataset}_train_embed.jsonl"
# train_embed_infer_results = [json.loads(l) for l in open(train_embed_infer_path)]
#
# min_ranks, max_ranks, avg_ranks = [], [], []
# for eidx, entry in enumerate(train_embed_infer_results):
#     haystack_sessions = entry['haystack_sessions']
#     haystack_session_ids = entry['haystack_session_ids']
#
#     emb_similarities = []
#     all_turns = []
#     gold_turns = []
#     for sid, session in enumerate(haystack_sessions):
#         for tid, turn in enumerate(session):
#             emb_similarities.append(turn.get("similarity_score", 0.0))
#             all_turns.append(f"{haystack_session_ids[sid]}_{tid}")
#             if turn.get("has_answer", False):
#                 gold_turns.append(f"{haystack_session_ids[sid]}_{tid}")
#     # logger.info(f"Entry {eidx} has {len(all_turns)} turns, {len(gold_turns)} gold turns")
#
#     # calculate ranking for gold turns
#     # remove gold turn after 10%
#     num_turns = len(all_turns)
#     sorted_turns = [turn for _, turn in sorted(zip(emb_similarities, all_turns), reverse=True)]
#     gold_ranks = []
#     for gold_turn in gold_turns:
#         gold_rank = sorted_turns.index(gold_turn) + 1
#         if gold_rank > num_turns * 0.1:
#             continue
#         gold_ranks.append(gold_rank)
#     if not gold_ranks:
#         logger.info(f"Entry {eidx} has no gold turn in top 10%")
#         continue
#     # max rank, min rank, avg rank
#     max_rank = max(gold_ranks)
#     min_rank = min(gold_ranks)
#     avg_rank = sum(gold_ranks) / len(gold_ranks)
#     min_ranks.append(min_rank)
#     max_ranks.append(max_rank)
#     avg_ranks.append(avg_rank)
#
#
# logger.info(f"Min ranks: {min_ranks}")
# logger.info(f"Max ranks: {max_ranks}")
# logger.info(f"Avg ranks: {avg_ranks}")
# logger.info(f"total {len(train_embed_infer_results)} entries, {len(min_ranks)} of them have gold turns in top 10%")

# annotate thinking process before search
# import openai
# from utils.session_process import construct_session_text, construct_history_text
# BASE_URL="http://172.16.77.93:8000"
# API_KEY="sk-nrfkwelacfabbaikvxqflzztslbdrpmeduhzxqnzmisaulgy"
#
# client = openai.OpenAI(
#     api_key=API_KEY,
#     base_url=BASE_URL,
# )
# prompt_name = "gen_retrieve_ranking_prompt.yaml"
# prompt_config = edict(yaml.load(open(f"./configs/{prompt_name}"), Loader=yaml.FullLoader))
# print(prompt_config)
# train_data = []
# prompt_template = prompt_config.prompt_template
#
# training_data = [json.loads(l) for l in open("../../data/split_longmemeval_s/train.jsonl")]
# idx = 0
# haystack_sessions = training_data[idx]["haystack_sessions"]
# question = training_data[idx]["question"]
# history = construct_history_text(haystack_sessions)
# current_session = f'<session">\n<user>{question}</user>\n</session>'
# user_prompt = prompt_template.user_prompt_text.format(history=history, current=current_session)
# messages=[
#     # {"role": "system", "content": prompt_template.system_prompt_text},
#     {"role": "user", "content": user_prompt},
# ]
#
# response = client.chat.completions.create(
#     model="Pro/deepseek-ai/DeepSeek-V3.2-Exp",
#     messages=messages,
#     temperature=0.7,
#     max_tokens=16384,
#
# )
# # reasoning content
# logger.info(response.choices[0].message.reasoning_content)
# # response
# logger.info(response.choices[0].message.content)

# remove global step dir from checkpoint
# import shutil
# base_dir = "/mnt/jjtan/ckpt/4b-ins-split-s-reproduce"
# # list dir starts with checkpoint-
# checkpoint_dirs = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-")]
# # remove global step dir from checkpoint dirs
# for checkpoint_dir in checkpoint_dirs:
#     ckpt_count = checkpoint_dir.split("-")[-1]
#     global_step_dir = os.path.join(base_dir, checkpoint_dir, f"global_step{ckpt_count}")
#     # remove dir recursively
#     if os.path.exists(global_step_dir):
#         shutil.rmtree(global_step_dir)
#         logger.info(f"Removed {global_step_dir}")
#     # edit config.json
#     config_path = os.path.join(base_dir, checkpoint_dir, "config.json")
#     config = json.load(open(config_path))
#     max_position_embeddings = config.get("max_position_embeddings", 0)
#     if max_position_embeddings > 131072:
#         config["max_position_embeddings"] = 131072
#         json.dump(config, open(config_path, "w"), ensure_ascii=False, indent=4)
#         logger.info(f"Set max_position_embeddings to 131072 in {config_path}")

# decode input ids
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("../../models/Qwen3-4B-Instruct")
# # decode input ids
# input_ids = [2618, 10488, 323, 6955, 4401, 13, 1084, 594, 2244, 311, 1490, 279, 3033, 24965, 304, 279, 3853, 1075, 419, 3918, 5920, 397, 5405, 12853, 9608, 510, 27, 5920, 2400, 428, 17, 15, 17, 18, 14, 15, 20, 14, 18, 15, 320, 61048, 8, 220, 17, 15, 25, 20, 15, 881, 27, 872, 29, 4340, 1293, 572, 358, 304, 6323, 369, 26055, 872, 397, 522, 5920, 397, 5501, 23643, 279, 3403, 1378, 11127, 323, 2550, 279, 3175, 1429, 9760, 13656, 16230, 10238, 429, 1850, 9071, 279, 1482, 6236, 2266, 13, 151645, 198, 151644, 77091, 198, 785, 1429, 9760, 2484, 374, 510]
# decoded_text = tokenizer.decode(input_ids)
# logger.info(decoded_text)
# input_ids = [6955, 4401, 13, 1084, 594, 2244, 311, 1490, 279, 3033, 24965, 304, 279, 3853, 1075, 419, 3918, 872, 397, 522, 5920, 397, 5405, 12853, 9608, 510, 27, 5920, 2400, 428, 17, 15, 17, 18, 14, 15, 20, 14, 18, 15, 320, 61048, 8, 220, 17, 15, 25, 20, 15, 881, 27, 872, 29, 4340, 1293, 572, 358, 304, 6323, 369, 26055, 872, 397, 522, 5920, 397, 5501, 23643, 279, 3403, 1378, 11127, 323, 2550, 279, 3175, 1429, 9760, 13656, 16230, 10238, 429, 1850, 9071, 279, 1482, 6236, 2266, 13, 151645, 198, 151644, 77091, 198, 785, 1429, 9760, 2484, 374, 510]
# decoded_text = tokenizer.decode(input_ids)
# logger.info(decoded_text)
# logger.info(tokenizer.encode("<ranking>"))


# check file structure
# file_name = "../../data/results/bge-m3/shared_contexts_32k_embed_records.jsonl"
# lines = [json.loads(l) for l in open(file_name)]
# logger.info(lines[0].keys())
# # dict_keys(['question_id', 'question', 'answer', 'haystack_sessions', 'haystack_session_ids', 'answer_session_ids', 'haystack_session_embedding_hash_ids', 'haystack_session_similarity_scores', 'current_session_text', 'current_embedding_hash_id'])
# for k, v in lines[0].items():
#     if len(str(v)) > 1000:
#         value = str(v)[:1000]
#         logger.info(f"{k}: {value}...")
#     else:
#         logger.info(f"{k}: {v}")


# check annotated thinking process
# thinking_file = "../../data/split_longmemeval_s/train_thinking_roll_r1.jsonl"
# thinking_lines = [json.loads(l) for l in open(thinking_file)]
# logger.info(thinking_lines[0].keys())
# # dict_keys(['question_id', 'question_type', 'question', 'question_date', 'answer', 'answer_session_ids', 'haystack_dates', 'haystack_session_ids', 'haystack_sessions', 'messages', 'matched_turns'])
# # prepare train data
# train_file = "../../data/custom_longmemeval/train_thinking_r1.jsonl"
# # dict_keys(['messages', 'matched_turns', 'question_id', 'question_type', 'question', 'question_date', 'answer', 'answer_session_ids', 'haystack_dates', 'haystack_session_ids', 'haystack_sessions'])
# train_lines = []
# for li, line in enumerate(thinking_lines):
#     messages = line["messages"]
#     matched_turns = line["matched_turns"]
#     for ti, matched_turn in enumerate(matched_turns):
#         reasoning_content = matched_turn["reasoning_content"]
#         # logger.info(reasoning_content)
#         output = matched_turn["output"]
#         # logger.info(output)
#         assert messages[0]["role"] == "user", f"First message should be user, but got {messages[0]['role']}"
#         train_lines.append({
#             "id": f"r1-{li}-{ti}",
#             "messages": [
#                 {"role": "user", "reasoning_content": reasoning_content, "content": messages[0]["content"]},
#                 {"role": "assistant", "content": output},
#             ]
#         })
#
# # save train lines
# with open(train_file, "w") as f:
#     for line in train_lines:
#         f.write(json.dumps(line, ensure_ascii=False) + "\n")
#
# logger.info(f"saved {len(train_lines)} lines to {train_file}")


# from utils.gen_infer_utils import APIProbabilityCalculator, TokenIdNode
# from anytree import RenderTree
# import numpy as np
# root_node = TokenIdNode(name="root", parent=None, children=None, input_ids=[], prob=np.float32(1.0))
# c1 = TokenIdNode(name="user", parent=root_node, children=None, input_ids=[2618], prob=np.float32(0.5))
# c2 = TokenIdNode(name="user", parent=root_node, children=None, input_ids=[2618], prob=np.float32(0.5))
#
# # print token tree
# print(RenderTree(root_node))


# longmemeval_file = "../../data/personamem_1M/test.jsonl"
# idx = 52
# lines = [json.loads(l) for l in open(longmemeval_file)]
# line = lines[idx]
# logger.info(line.keys())
# # dict_keys(['question_id', 'question', 'all_options', 'correct_answer', 'answer', 'haystack_sessions', 'haystack_session_ids', 'answer_session_ids', 'gold_session_idx', 'gold_session'])
# logger.info(f"question: {line['question']}")
# logger.info(f"answer: {line['answer']}")
# logger.info(f"gold turn: {line['gold_session'][line['gold_turn_idx']]}")
# with open("./personamem_check.json", "w") as f:
#     f.write(json.dumps(line["gold_session"], ensure_ascii=False, indent=2) + "\n")


# eval rollouts during validation
# rollout_file = "../../ckpt/DAPO-GenRank/DAPO-Qwen3-4B-Thinking/val_data/80.jsonl"
# rollout_lines = [json.loads(l) for l in open(rollout_file)]
# logger.info(rollout_lines[0].keys())
# # dict_keys(['input', 'output', 'gts', 'score', 'step', 'reward', 'acc'])
# from utils.eval_utils import robust_session_ranking_parse, calculate_ndcg
# from genrank_verl.genrank_reward_score import compute_score_thinking, extract_solution
# idx = 17
# row = rollout_lines[idx]
# logger.info(f"acc: {row['acc']}")
# logger.info(f"gts: {row['gts']}")
# logger.info(f"gts: {row['gts'].split(';')[0]}")
# gen_rankings = robust_session_ranking_parse(row["output"])
# logger.info(f"gen_rankings: {gen_rankings}")
#
# golds, preds = [], []
# ndcg = []
# for i in range(len(rollout_lines)):
#     gts = rollout_lines[i]["gts"].split(";")[0]
#     gold = [int(g) for g in gts.split(",")]
#     try:
#         pred = robust_session_ranking_parse(rollout_lines[i]["output"])
#     except:
#         logger.info(f"failed to parse gen_rankings for {i}")
#         logger.info(f"output: {rollout_lines[i]['output'][-1000:]}")
#         pred = []
#         continue
#
#     golds.append(gold)
#     preds.append(pred)
#
#     # logger.info(f"gold: {gold}")
#     # logger.info(f"pred: {pred}")
#     item_ndcg = calculate_ndcg(gold, pred)
#     ndcg.append(item_ndcg)
#     if abs(item_ndcg - rollout_lines[i]["score"]) > 0.01:
#         logger.info(f"item_ndcg: {item_ndcg}, score: {rollout_lines[i]['score']}, reward: {rollout_lines[i]['reward']}, acc: {rollout_lines[i]['acc']}")
#         logger.info(f"gold: {gold}")
#         logger.info(f"pred: {pred}")
#         logger.info(f"gts: {rollout_lines[i]['gts']}")
#         logger.info(f"output: {rollout_lines[i]['output'][-1000:]}")
#         # with open("long_prompt_check.txt", "w") as f:
#         #     f.write(rollout_lines[i]["output"] + "\n")
#         # exit(0)
#
# logger.info(f"avg ndcg: {np.mean(ndcg)}")


# eval inferece results
# rollout_file = "../../data/results/DAPO-GenRank/DAPO-Qwen3-4B-Thinking-step-80/perltqa_en_test_ranking.jsonl"
# rollout_lines = [json.loads(l) for l in open(rollout_file)]
# logger.info(rollout_lines[0].keys())
# # dict_keys(['response', 'all_options', 'answer', 'answer_session_ids', 'batch_uuid', 'correct_answer', 'embeddings', 'generated_text', 'generated_tokens', 'gold_session', 'gold_session_idx', 'gold_turn_idx', 'haystack_session_ids', 'haystack_sessions', 'messages', 'metrics', 'num_generated_tokens', 'num_input_tokens', 'params', 'prefiltered_session_idx', 'prompt', 'prompt_token_ids', 'question', 'question_id', 'request_id', 'similarities', 'time_taken_llm', 'haystack_session_gen_rankings'])
# from utils.eval_utils import robust_session_ranking_parse, calculate_ndcg
# from genrank_verl.genrank_reward_score import compute_score_thinking, extract_solution
# idx = 17
# row = rollout_lines[idx]
# logger.info(f"question: {row['question']}")
# logger.info(f"answer: {row['answer']}")
# logger.info(f"generated_text: {row['generated_text'][-50:]}")
# logger.info(f"haystack_session_gen_rankings: {row['haystack_session_gen_rankings']}")
# logger.info(f"prefiltered_session_idx: {row['prefiltered_session_idx']}")
# gold_session_idx = [row["haystack_session_ids"].index(_i) for _i in row["answer_session_ids"]]
# logger.info(f"gold_session_id: {gold_session_idx}")


# eval inferece results
# rollout_lines = [json.loads(line.strip()) for line in open("../../data/results/DAPO-GenRank/DAPO-Qwen3-4B-Thinking-merged-epoch-1/perltqa_en_test_ranking.jsonl")]
# from genrank_verl.genrank_reward_score import compute_score_thinking, dedup_indexes
# logger.info(rollout_lines[0].keys())
# # dict_keys(['response', 'answer', 'answer_session_ids', 'batch_uuid', 'embeddings', 'generated_text', 'generated_tokens', 'haystack_session_ids', 'haystack_sessions', 'messages', 'metrics', 'num_generated_tokens', 'num_input_tokens', 'params', 'prefiltered_session_idx', 'prompt', 'prompt_token_ids', 'question', 'question_id', 'request_id', 'similarities', 'time_taken_llm', 'haystack_session_gen_rankings'])
# idx = 17
# logger.info(f"question: {rollout_lines[idx]['question']}")
# logger.info(f"answer: {rollout_lines[idx]['answer']}")
# logger.info(f"response: {rollout_lines[idx]['response'][-50:]}")
# logger.info(f"haystack_session_gen_rankings: {rollout_lines[idx]['haystack_session_gen_rankings']}")
# logger.info(f"prefiltered_session_idx: {rollout_lines[idx]['prefiltered_session_idx']}")
# gold_session_idx = [rollout_lines[idx]["haystack_session_ids"].index(_i) for _i in rollout_lines[idx]["answer_session_ids"]]
# logger.info(f"gold_session_id: {gold_session_idx}")
# gold_session_idx_filtered = []
# for _i in range(len(rollout_lines[idx]["prefiltered_session_idx"])):
#     if rollout_lines[idx]["prefiltered_session_idx"][_i] in gold_session_idx:
#         gold_session_idx_filtered.append(_i)
#
# logger.info(f"gold_session_idx_filtered: {gold_session_idx_filtered}")
#
# gts = ",".join([str(_i) for _i in gold_session_idx_filtered]) + f";{len(rollout_lines[idx]['prefiltered_session_idx'])}"
# logger.info(f"gts: {gts}")
#
#
# reward_scores = []
# for eidx, entry in enumerate(rollout_lines):
#     haystack_session_ids = entry['haystack_session_ids']
#     prefiltered_session_idx = entry["prefiltered_session_idx"]
#     prefiltered_session_id = [haystack_session_ids[i] for i in prefiltered_session_idx]
#     # produce gts
#     gold_session_idx = [haystack_session_ids.index(_i) for _i in entry["answer_session_ids"]]
#     gold_session_idx_filtered = []
#     for _i in range(len(prefiltered_session_idx)):
#         if prefiltered_session_idx[_i] in gold_session_idx:
#             gold_session_idx_filtered.append(_i)
#     if len(gold_session_idx_filtered) == 0:
#         reward_scores.append(0.0)
#         continue
#     gts = ",".join([str(_i) for _i in gold_session_idx_filtered]) + f";{len(prefiltered_session_id)}"
#     reward_scores.append(compute_score_thinking(
#         data_source = "perltqa",
#         solution_str = entry["response"],
#         ground_truth = gts,
#     ))
#
# avg_reward = np.mean(reward_scores)
# logger.info(f"avg reward: {avg_reward}")


# eval validation results
# from genrank_verl.genrank_reward_score import compute_score_thinking, dedup_indexes
# val_file_name = "../../ckpt/DAPO-GenRank/ep2-DAPO-Qwen3-4B-Thinking/val_data/10.jsonl"
# val_lines = [json.loads(line.strip()) for line in open(val_file_name)]
# logger.info(val_lines[0].keys())
# # dict_keys(['input', 'output', 'gts', 'score', 'step', 'reward', 'acc'])
# val_parquet_file_name = "/mnt/jjtan/data/rl_train_data/v1209-0/test_mini.parquet"
# val_dataset = datasets.load_dataset("parquet", data_files=val_parquet_file_name)['train']
# logger.info(val_dataset)
# # features: ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
# reward_scores = []
# perltqa_indexes = []
# for i in range(len(val_lines)):
#     if not "perltqa" in val_dataset[i]["data_source"]:
#         continue
#     reward_score = compute_score_thinking(
#         data_source = val_dataset[i]["data_source"],
#         solution_str = val_lines[i]["output"],
#         ground_truth = val_dataset[i]["reward_model"]["ground_truth"],
#     )
#     reward_scores.append(reward_score)
#     logger.info(f"output: {val_lines[i]['output'][-50:]}")
#     logger.info(f"gts: {val_dataset[i]['reward_model']['ground_truth']}")
#     logger.info(f"reward: {reward_score}")
#     perltqa_indexes.append(val_dataset[i]["extra_info"]["index"])
#
# avg_reward = np.mean(reward_scores)
# logger.info(f"avg reward: {avg_reward}")
# print(perltqa_indexes)

# perltqa_indexes = [209, 282, 33, 210, 93, 84, 327, 94, 262, 122, 9, 363, 56, 66, 127, 44, 273, 374, 230, 384, 75, 15, 390, 266, 0, 396, 112, 225, 256, 104, 395, 191, 278, 57, 232, 116, 129, 343, 161, 139, 385, 53, 74, 25, 82, 381, 152, 182, 22, 173, 46, 315, 336, 76, 376, 41, 223, 172, 30, 151, 120, 293, 252]

# train_dataset = datasets.load_dataset("parquet", data_files="/mnt/jjtan/data/rl_train_data/v1210-0/train_0.parquet")['train']
# for i in range(len(train_dataset)):
#     gts = train_dataset[i]["reward_model"]["ground_truth"]
#     aidx = gts.split(";")[0]
#     try:
#         aidx = [int(_i) for _i in aidx.split(",")]
#     except:
#         logger.info(f"empty answer idx in train data: {train_dataset[i]}")
#         logger.info(f"gts: {gts}")
#         raise

# (locomo split_longmemeval_s split_longmemeval_m perltqa_en perltqa_zh personamem_32k personamem_128k personamem_1M zh4o personamem_v2_32k personamem_v2_128k)
# for datasets in ["locomo", "split_longmemeval_s", "split_longmemeval_m", "perltqa_en", "perltqa_zh", "personamem_32k", "personamem_128k", "personamem_1M", "zh4o", "personamem_v2_32k", "personamem_v2_128k"]:
# for datasets in ["personamem_v2_32k", "personamem_v2_128k"]:
#     for split in ["train", "test"]:
#         # copy file
#         # results/bge-m3/perltqa_zh_test_embed.parquet
#         src_file_name = f"/mnt/jjtan/data/{datasets}/{split}.parquet"
#         # convert jsonl to parquet
#         if not os.path.exists(src_file_name):
#             logger.info(f"src_file_name not exists: {src_file_name}")
#             continue
#         df = pd.read_parquet(src_file_name)
#         df['answer'] = df['answer'].astype(str)
#         logger.info(f"dtype: {df['haystack_sessions'].dtype}")
#         if df["haystack_sessions"].dtype == str:
#             df['haystack_sessions'] = df['haystack_sessions'].apply(lambda x: json.loads(x))
#         logger.info(f"dtype: {df['haystack_sessions'].dtype}")
#         for session in df["haystack_sessions"]:
#             for turn in session:
#                 if not isinstance(turn, dict):
#                     logger.info(f"Invalid turn: {turn}")
#                     raise ValueError(f"Invalid turn: {turn}")
                # if "recommend" in turn and turn["recommend"] is None:
                #     turn["recommend"] = ""
        # if not df["haystack_sessions"].dtype == str:
        #     print(type(df["haystack_sessions"][0][0]))
        #     df['haystack_sessions'] = df['haystack_sessions'].apply(lambda x: json.dumps(x, ensure_ascii=False))
        # dst_file_name = f"/mnt/jjtan/data/results/bge-m3/{datasets}_{split}_embed.parquet"
        # df.to_parquet(dst_file_name, index=False, compression="zstd")
        # bench_file_name = f"/mnt/jjtan/data/benchmarks/{datasets}/{split}.parquet"
        # os.makedirs(os.path.dirname(bench_file_name), exist_ok=True)
        # shutil.copy(dst_file_name, bench_file_name)
        # logger.info(f"Finished {datasets}_{split}")

# for datasets in ["locomo", "split_longmemeval_s", "split_longmemeval_m", "perltqa_en", "perltqa_zh", "personamem_32k", "personamem_128k", "personamem_1M", "zh4o", "personamem_v2_32k", "personamem_v2_128k"]:
#     for split in ["train", "test"]:
#         src_file_name = f"/mnt/jjtan/data/results/bge-m3/{datasets}_{split}_embed.parquet"
#         logger.info(f"src_file_name: {src_file_name}")
#         df = pd.read_parquet(src_file_name)
#         # check answer column type
#         logger.info(f"dtype: {df['answer'].dtype}")
#         # check haystack_sessions column type
#         logger.info(f"dtype: {df['haystack_sessions'].dtype}")
#         df['haystack_sessions'] = df['haystack_sessions'].apply(lambda x: json.loads(x))
#         for sessions in df["haystack_sessions"]:
#             for session in sessions:
#                 for turn in session:
#                     assert turn["role"] != "system", f"Invalid turn: {turn}"
#
#         dst_file_name = f"/mnt/jjtan/data/{datasets}/{split}.parquet"
#         shutil.copy(src_file_name, dst_file_name)


# from a_mem.a_mem_infer import advancedMemAgent
# chat_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
# embed_model = "/mnt/jjtan/models/bge-m3"
# os.environ["API_KEY"] = "sk-deadbeef"
# agent = advancedMemAgent(chat_model, embed_model, "openai", 10, 0.5, "", "")
# assert agent.memory_system.retriever.embeddings is None, "Retriever embeddings should be None when not indexed"

output_file = "/mnt/jjtan/data/WikiTables/test.parquet"
df = pd.read_parquet(output_file)
print(df.head(1)["haystack_session_ids"])
# print(robust_parse_reasonrank(df.head(1)["response"].item()))

