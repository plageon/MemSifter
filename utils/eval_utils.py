import json
from typing import Dict, Any

import numpy as np
from loguru import logger
from pandas.core.interchange.dataframe_protocol import DataFrame


def calculate_mrr(gold_turns: list, pred_turns: list):
    mrr = 0.0
    for pid, pred_turn in enumerate(pred_turns):
        if pred_turn in gold_turns:
            mrr = 1.0 / (pid + 1)
            break
    return mrr


def calculate_map(gold_turns: list, pred_turns: list):
    map = 0.0
    for pid, pred_turn in enumerate(pred_turns):
        if pred_turn in gold_turns:
            map += (1.0 / (pid + 1))
    map /= len(gold_turns)
    return map


def calculate_dcg(gold_turns: list, pred_turns: list):
    dcg = 0.0
    for pid, pred_turn in enumerate(pred_turns):
        if pred_turn in gold_turns:
            dcg += (1.0 / np.log2(pid + 2))
    dcg /= len(gold_turns)
    return dcg


def calculate_ndcg(gold_turns: list, pred_turns: list):
    dcg = calculate_dcg(gold_turns, pred_turns)
    idcg = calculate_dcg(gold_turns, gold_turns)
    ndcg = dcg / idcg
    return ndcg


def calculate_recall(gold_turns: list, pred_turns: list):
    recall = 0.0
    for pred_turn in pred_turns:
        if pred_turn in gold_turns:
            recall += 1.0
    recall /= len(gold_turns)
    return recall


def calculate_precision(gold_turns: list, pred_turns: list):
    precision = 0.0
    for pred_turn in pred_turns:
        if pred_turn in gold_turns:
            precision += 1.0
    precision /= len(pred_turns)
    return precision


def calculate_f1_item(recall: float, precision: float):
    f1 = 0.0
    if recall + precision > 0.0:
        f1 = 2.0 * recall * precision / (recall + precision)
    return f1


def calculate_f1(gold_turns: list, pred_turns: list):
    recall = calculate_recall(gold_turns, pred_turns)
    precision = calculate_precision(gold_turns, pred_turns)
    f1 = calculate_f1_item(recall, precision)
    return f1


metric_functions = {
        "mrr": calculate_mrr,
        "map": calculate_map,
        "dcg": calculate_dcg,
        "ndcg": calculate_ndcg,
        "recall": calculate_recall,
        "precision": calculate_precision,
        "f1": calculate_f1,
    }

def load_prob_data(result_dir: str, model_tag: str, dataset_name: str, split: str, max_content_length: str):
    # longmemeval_s_cleaned_gen_ret.jsonl
    logger.info(f"Loading data from {result_dir}/{model_tag}/{dataset_name}_{split}_gen_ret_{max_content_length}.jsonl")
    with open(f"{result_dir}/{model_tag}/{dataset_name}_{split}_gen_ret_{max_content_length}.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    return data


def dedup_indexes(ranking_list):
    # 去除重复的session索引，保证第一次出现的顺序不变
    ranking_set =set()
    dedup_ranking_list = []
    for ranking in ranking_list:
        if ranking not in ranking_set:
            ranking_set.add(ranking)
            dedup_ranking_list.append(ranking)
    return dedup_ranking_list


def eval_prob_data(data, debug: bool = False, verbose: bool = True):
    if debug:
        data = data[:10]
    pred_ns = [1, 5, 10]
    pred_results = {pred_n: {"emb": {}, "gen": {}} for pred_n in pred_ns}
    for metric in ["mrr", "map", "ndcg", "recall", "precision", "f1"]:
        for pred_n in pred_ns:
            pred_results[pred_n]["emb"][metric] = []
            pred_results[pred_n]["gen"][metric] = []

    for eidx, entry in enumerate(data):
        haystack_session_ids = entry['haystack_session_ids']
        # remove out of scope sessions
        haystack_session_gen_rankings = [i for i in entry['haystack_session_gen_rankings'] if i < len(haystack_session_ids)]
        haystack_session_gen_rankings = dedup_indexes([haystack_session_ids[i] for i in haystack_session_gen_rankings])
        haystack_session_similarity_scores = entry['similarities']
        gen_ranking_output = entry["generated_text"]
        # logger.info(f"Pred gen rankings: {haystack_session_gen_rankings}")
        # logger.info(f"Gen ranking output: {gen_ranking_output}")

        if len(haystack_session_gen_rankings) == 0:
            logger.warning(f"Entry {eidx} has no pred gen rankings.")
            haystack_session_gen_rankings = haystack_session_ids

        haystack_session_emb_rankings = [haystack_session_ids[i] for i in np.argsort(haystack_session_similarity_scores)[::-1]]
        # logger.info(f"Pred emb rankings: {haystack_session_emb_rankings}")

        gold_sessions = entry['answer_session_ids']
        # logger.info(f"Gold sessions: {gold_sessions}")
        # for i in range(len(haystack_session_ids)):
        #     if haystack_session_ids[i] in gold_sessions:
        #         logger.info(entry["haystack_sessions"][i])


        if len(gold_sessions) == 0:
            logger.warning(f"Entry {eidx} has no gold turns.")
            continue

        for pred_n in pred_ns:
            emb_pred_sessions = haystack_session_emb_rankings[:pred_n]
            gen_pred_sessions = haystack_session_gen_rankings[:pred_n]
            for method in ["emb", "gen"]:
                for metric in ["mrr", "map", "ndcg", "recall", "precision", "f1"]:
                    pred_results[pred_n][method][metric].append(metric_functions[metric](gold_sessions[:pred_n], eval(f"{method}_pred_sessions")))

     # show pred_results
    avg_pred_results = {pred_n: {"emb": {}, "gen": {}} for pred_n in pred_ns}
    for pred_n in pred_ns:
        for method in ["emb", "gen"]:
            for metric in ["mrr", "map", "ndcg", "recall", "precision", "f1"]:
                score = np.mean(pred_results[pred_n][method][metric])
                avg_pred_results[pred_n][method][metric] = score
                # print(f"{method}@{pred_n} {metric}: {avg_pred_results[pred_n][method][metric]:.4f}")
    if verbose:
        markdown_table = "| pred_n | method | mrr | map | ndcg | recall | precision | f1 |\n"
        markdown_table += "| ------ | ------ | --- | --- | --- | ------ | -------- | --- |\n"
        for pred_n in pred_ns:
            for method in ["emb", "gen"]:
                markdown_table += (
                    f"| {pred_n} | {method} | {avg_pred_results[pred_n][method]['mrr']*100:.2f} | "
                    f"{avg_pred_results[pred_n][method]['map']*100:.2f} | {avg_pred_results[pred_n][method]['ndcg']*100:.2f} | "
                    f"{avg_pred_results[pred_n][method]['recall']*100:.2f} | {avg_pred_results[pred_n][method]['precision']*100:.2f} | "
                    f"{avg_pred_results[pred_n][method]['f1']*100:.2f} |\n"
                )

        print(markdown_table)
    return avg_pred_results


def eval_ranking_data(data: DataFrame, debug: bool = False, verbose: bool = True, mode: str = "ranking"):
    if debug:
        data = data.head(10)
    pred_ns = [1, 5, 10]
    pred_results = {pred_n: {} for pred_n in pred_ns}
    for metric in ["mrr", "map", "ndcg", "recall", "precision", "f1"]:
        for pred_n in pred_ns:
            pred_results[pred_n][metric] = []
            pred_results[pred_n][metric] = []

    for eidx, entry in data.iterrows():
        haystack_session_ids = entry['haystack_session_ids']
        if mode == "ranking":
            # remove out of scope sessions
            haystack_session_gen_rankings = [i for i in entry['haystack_session_gen_rankings'] if i < len(haystack_session_ids)]
            pred_rankings = dedup_indexes([haystack_session_ids[i] for i in haystack_session_gen_rankings])

            if len(pred_rankings) == 0:
                if verbose:
                    logger.warning(f"Entry {eidx} has no pred gen rankings.")
                pred_rankings = haystack_session_ids

        elif mode == "similarity":
            haystack_session_similarity_scores = entry['similarities']
            pred_rankings = [haystack_session_ids[i] for i in np.argsort(haystack_session_similarity_scores)[::-1]]

        else:
            raise ValueError(f"Unknown mode: {mode}")

        gold_sessions = entry['answer_session_ids']
        if len(gold_sessions) == 0:
            if verbose:
                logger.warning(f"Entry {eidx} has no gold sessions.")
            continue


        for pred_n in pred_ns:
            for metric in ["mrr", "map", "ndcg", "recall", "precision", "f1"]:
                pred_results[pred_n][metric].append(metric_functions[metric](gold_sessions[:pred_n], pred_rankings[:pred_n]))

    avg_pred_results = {pred_n: {} for pred_n in pred_ns}
    for pred_n in pred_ns:
        for metric in ["mrr", "map", "ndcg", "recall", "precision", "f1"]:
            score = np.mean(pred_results[pred_n][metric])
            avg_pred_results[pred_n][metric] = score
            # print(f"{metric}@{pred_n}: {avg_pred_results[pred_n][metric]:.4f}")

    if verbose:
        markdown_table = "| pred_n | metric | score |\n"
        markdown_table += "| ------ | ------ | --- |\n"
        for pred_n in pred_ns:
            for metric in ["mrr", "map", "ndcg", "recall", "precision", "f1"]:
                markdown_table += (
                    f"| {pred_n} | {metric} | {avg_pred_results[pred_n][metric]*100:.2f} |\n"
                )
        print(markdown_table)

    return avg_pred_results


def robust_session_ranking_parse(response_text: str):
    try:
        # get rid of think steps
        think_end_idx = response_text.find("</think>")
        if think_end_idx == -1:
            return []
        response_text = response_text[think_end_idx + len("</think>"):]
        ranking_pattern = "<ranking>"
        ranking_end_pattern = "</ranking>"
        ranking_start_idx = response_text.find(ranking_pattern)
        if ranking_start_idx == -1:
            return []
        ranking_end_idx = response_text.find(ranking_end_pattern, ranking_start_idx)
        if ranking_end_idx == -1:
            return []
        ranking_text = response_text[ranking_start_idx + len(ranking_pattern): ranking_end_idx]
        session_rankings = ranking_text.split(",")
        session_ranking_ids = []
        for i, sid in enumerate(session_rankings):
            session_ranking_ids.append(int(sid.strip()))
        return session_ranking_ids
    except Exception as e:
        logger.warning(f"Error parsing ranking: {e}")
        import traceback
        traceback.print_exc()
        return []


class RankingParserActor():
    def __init__(self, filtered_idx_col: str):
        self.filtered_idx_col = filtered_idx_col

    def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row["haystack_session_gen_rankings"] = robust_session_ranking_parse(row["response"])
        if self.filtered_idx_col:
            # convert gen rankings to prefiltered session ids
            prefiltered_session_idx = row[self.filtered_idx_col]
            gen_rankings = []
            for i in row["haystack_session_gen_rankings"]:
                if i >= len(prefiltered_session_idx):
                    logger.warning(f"Gen ranking index {i} out of range.")
                    continue
                gen_rankings.append(prefiltered_session_idx[i])
            row["haystack_session_gen_rankings"] = gen_rankings
        return row


def eval_prefilter_recall(data, debug: bool = False):
    if debug:
        data = data[:10]
    prefilter_recall = []
    ndcg_without_filter = {}
    for n in [1, 5, 10]:
        ndcg_without_filter[f"ndcg@{n}"] = []

    for eidx, entry in enumerate(data):
        haystack_session_ids = entry['haystack_session_ids']
        prefiltered_session_idx = entry["prefiltered_session_idx"]
        prefiltered_session_id = [haystack_session_ids[i] for i in prefiltered_session_idx]
        # remove out of scope sessions
        haystack_session_gen_rankings = [i for i in entry['haystack_session_gen_rankings'] if
                                         i < len(haystack_session_ids)]
        haystack_session_gen_rankings = dedup_indexes([haystack_session_ids[i] for i in haystack_session_gen_rankings])
        haystack_session_similarity_scores = entry['similarities']
        gen_ranking_output = entry["generated_text"]

        if len(haystack_session_gen_rankings) == 0:
            logger.warning(f"Entry {eidx} has no pred gen rankings.")
            haystack_session_gen_rankings = haystack_session_ids

        haystack_session_emb_rankings = [haystack_session_ids[i] for i in
                                         np.argsort(haystack_session_similarity_scores)[::-1]]
        # logger.info(f"Pred emb rankings: {haystack_session_emb_rankings}")

        gold_sessions = entry['answer_session_ids']
        # logger.info(f"Gold sessions: {gold_sessions}")
        # for i in range(len(haystack_session_ids)):
        #     if haystack_session_ids[i] in gold_sessions:
        #         logger.info(entry["haystack_sessions"][i])

        if len(gold_sessions) == 0:
            logger.warning(f"Entry {eidx} has no gold turns.")
            continue

        if all([sid in prefiltered_session_id for sid in gold_sessions]):
            prefilter_recall.append(1.0)
            for n in [1, 5, 10]:
                ndcg_without_filter[f"ndcg@{n}"].append(calculate_dcg(gold_sessions[:n], haystack_session_gen_rankings[:n]))
        else:
            prefilter_recall.append(0.0)

    # average ndcg without filter
    avg_prefilter_recall = np.mean(prefilter_recall)
    avg_ndcg_without_filter = {}
    for n in [1, 5, 10]:
        avg_ndcg_without_filter[f"ndcg@{n}"] = np.mean(ndcg_without_filter[f"ndcg@{n}"])

    return avg_prefilter_recall, avg_ndcg_without_filter

if __name__ == "__main__":
    data = [json.loads(line.strip()) for line in open("../data/results/DAPO-GenRank/DAPO-Qwen3-4B-Thinking-merged-epoch-1/perltqa_en_test_ranking.jsonl")]
    results = eval_prefilter_recall(data)
    print(results)



