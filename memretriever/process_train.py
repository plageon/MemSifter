import json
import random

import loguru
import numpy as np
import yaml
from easydict import EasyDict as edict
from transformers import AutoTokenizer

# longmemeval_s, longmemeval_m, longmemeval_oracle
if __name__ == '__main__':
    dataset_name = "custom_longmemeval"
    file_name = "emb_train.jsonl"
    # dataset_name = "split_longmemeval_s"
    # file_name = "train.jsonl"
    # data_file = f"../../../memory/{dataset_name}/{file_name}"
    data_file = f"../../data/{dataset_name}/{file_name}"
    output_tag = True
    history_keep_rate = 1.0
    emb_filter = 0.1
    if history_keep_rate < 1:
        output_file = f"../../data/{dataset_name}/{file_name.split('.')[0]}_his{history_keep_rate}_train.jsonl"
    else:
        output_file = f"../../data/{dataset_name}/{file_name.split('.')[0]}_train.jsonl"
    if emb_filter > 0:
        output_file = f"../../data/{dataset_name}/{file_name.split('.')[0]}_emb{emb_filter}_train.jsonl"

    if output_tag:
        output_file = output_file.replace(".jsonl", "_idtag.jsonl")

    if data_file.endswith(".jsonl"):
        orig_data = [json.loads(l) for l in open(data_file)]
    else:
        orig_data = json.load(open(data_file))
    print(orig_data[0].keys())
    # dict_keys(['question_id', 'question_type', 'question', 'answer', 'question_date', 'haystack_dates', 'haystack_session_ids', 'haystack_sessions', 'answer_session_ids'])
    random.seed(42)

    # load prompt config
    prompt_name = "gen_retrieve_prompt.yaml"
    prompt_config = edict(yaml.load(open(f"./configs/{prompt_name}"), Loader=yaml.FullLoader))
    print(prompt_config)
    train_data = []

    model_path = "../../models/Qwen3-4B-Instruct"

    # output train data format: {"messages": [{"role": "system", "content": "你是个有用无害的助手"}, {"role": "user", "content": "告诉我明天的天气"}, {"role": "assistant", "content": "明天天气晴朗"}]}
    # construct train data
    skipped_count = 0
    for i in range(len(orig_data)):
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

        relevant_turns = []
        emb_similarities = []
        all_turns = []
        relevant_turn_ids = []

        # select keep rate% of history sessions
        num_history_sessions = len(haystack_sessions) - len(answer_session_ids)
        num_kept_haystack_session = max(1, int(num_history_sessions * history_keep_rate))
        kept_haystack_session_indices = random.sample(haystack_session_ids, num_kept_haystack_session)
        kept_haystack_session_indices += answer_session_ids

        for sid, session in enumerate(haystack_sessions):
            date = haystack_dates[sid]
            session_id = haystack_session_ids[sid]
            if session_id not in kept_haystack_session_indices:
                continue
            # history += f'<session date="{date}">\n' # do not include session id in case of data leakage
            history += f'<session{sid}>\n'
            user_id, assistant_id = 0, 0

            for tid, turn in enumerate(session):
                if emb_filter > 0:
                    emb_similarities.append(turn.get("similarity_score", 0.0))
                    all_turns.append(f"{haystack_session_ids[sid]}_{tid}")
                    if turn.get("has_answer", False):
                        relevant_turn_ids.append(f"{haystack_session_ids[sid]}_{tid}")
                if turn["role"] == "user":
                    history += f'<user{user_id}>{turn["content"]}</user>\n'
                    user_id += 1
                elif turn["role"] == "assistant":
                    history += f'<assistant{assistant_id}>{turn["content"]}</assistant>\n'
                    assistant_id += 1

                # check if this turn has answer
                if "has_answer" in turn and turn["has_answer"]:
                    if output_tag:
                        if turn["role"] == "assistant":
                            relevant_turns.append(f"<session{sid}>\n<assistant{assistant_id}>{turn['content']}</assistant>\n</session>")
                        elif turn["role"] == "user":
                            relevant_turns.append(f"<session{sid}>\n<user{user_id}>{turn['content']}</user>\n</session>")
                    else:
                        relevant_turns.append(turn["content"])

            history += '</session>\n'

        if len(relevant_turns) == 0:
            loguru.logger.warning(f"No relevant turns found for question_id {entry['question_id']}, skipping this entry.")
            skipped_count += 1
            continue

        history = history.strip()

        # construct current session
        current_session = f'<session date="{entry["question_date"]}">\n<user>{question}</user>\n</session>'
        current_session = current_session.strip()

        num_turns = len(all_turns)
        if emb_filter > 0:
            sorted_turns = [turn for _, turn in sorted(zip(emb_similarities, all_turns), reverse=True)]
        else:
            sorted_turns = all_turns

        # construct output
        # output is the most relevant session in haystack_sessions
        added_entry_count = 0
        for ridx in range(len(relevant_turns)):
            relevant_turn = relevant_turns[ridx]
            if emb_filter > 0:
                relevant_turn_id = relevant_turn_ids[ridx]
                relevant_turn_rank = sorted_turns.index(relevant_turn_id) + 1
                if relevant_turn_rank > num_turns * 0.1:
                    continue
            relevant_turn = relevant_turn.strip()
            output = "The most relevant turn is:\n"
            output += relevant_turn

            # construct the final prompt
            prompt_template = prompt_config.prompt_template
            user_prompt = prompt_template.user_prompt_text.format(history=history, current=current_session)
            user_prompt = user_prompt.strip()

            # construct the final data entry
            data_entry = {
                "messages": [
                    {"role": "system", "content": prompt_template.system_prompt_text},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": output}
                ]
            }

            train_data.append(data_entry)
            added_entry_count += 1
        if added_entry_count == 0:
            loguru.logger.warning(f"No relevant turns found for question_id {entry['question_id']}, skipping this entry.")
            skipped_count += 1

    # save train data
    # loguru.logger.info(f"training data sample: {train_data[0]}")
    loguru.logger.info(f"total {len(orig_data)} original data, {skipped_count} lines skipped.")

    # calculate token length statistics
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_samples = len(train_data)
    token_lengths = []
    loguru.logger.info(f"total {num_samples} training samples, sample 20 for token length statistics")
    for entry in random.sample(train_data, 20):
        token_lengths.append(len(tokenizer.encode(json.dumps(entry, ensure_ascii=False))))

    token_lengths = np.array(token_lengths)
    loguru.logger.info(f"Token length statistics: mean={np.mean(token_lengths)}, std={np.std(token_lengths)}, max={np.max(token_lengths)}, min={np.min(token_lengths)}, 90th percentile={np.percentile(token_lengths, 90)}, 95th percentile={np.percentile(token_lengths, 95)}")
    loguru.logger.info(f"Total training data size: {len(train_data)}")


    # save in jsonl format
    with open(output_file, 'w') as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    loguru.logger.info(f"saved {len(train_data)} training data samples to {output_file}")


