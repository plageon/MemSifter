import sys
from typing import Dict, Any

from easydict import EasyDict as edict
from loguru import logger

sys.path.append("./")
from utils.session_process import construct_session_text


def construct_history_text_rankr1(sessions, query, session_dates=None):
    messages = []
    system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    user_prompt = '''Given the query: "{query}", which of the following documents is most relevant?
{docs}
After completing the reasoning process, please provide only the label of the most relevant document to the query, enclosed in square brackets, within the answer tags. For example, if the third document is the most relevant, the answer should be: <think> reasoning process here </think> <answer>[3]</answer>.'''
    history_text_list = []
    for sid, session_turns in enumerate(sessions):
        history_text = f"[{sid + 1}] " 
        if session_dates is not None: # TODO: 这个时间信息可以改
            session_date = session_dates[sid]
            history_text += f'Date: {session_date}\n'
        session_text = construct_session_text(session_turns)
        history_text += session_text
        history_text_list.append(history_text)
    history_text = "\n".join(history_text_list)
    
    user_prompt = user_prompt.format(query=query, docs=history_text)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages

def construct_history_text_rearank(sessions, query, session_dates=None):
    num = len(sessions)
    instruction =  (
            f"I will provide you with passages, each indicated by number identifier []. Rank the passages based on their relevance to the search query."
            f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query."
            f"The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be <answer> [] > [] </answer>, e.g., <answer> [1] > [2] </answer>."
        )
    
    messages = [
            {"role": "system", "content": "You are DeepRerank, an intelligent assistant that can rank passages based on their relevancy to the search query. You first thinks about the reasoning process in the mind and then provides the user with the answer."},
            {"role": "user","content": instruction},
            {"role": "assistant", "content": "Okay, please provide the passages."}
        ]
    rank = 0
    # for hit in item['hits'][rank_start: rank_end]:
    for session_turns in sessions:
        rank += 1
        content = "" 
        if session_dates is not None: # TODO: 这个时间信息可以改
            session_date = session_dates[rank - 1]
            content += f"Date: {session_date}\n"
        content += construct_session_text(session_turns)
        content = content.strip()
        content = ' '.join(content.split())
        messages.append({"role": "user", "content": f"[{rank}] {content[:400]}"})
        messages.append({"role": "assistant", "content": f"Received passage [{rank}]."})
                
    messages.append({
        "role": "user",
        "content": f"""Please rank these passages according to their relevance to the search query: "{query}"
            Follow these steps exactly:
            1. First, within <think> tags, analyze EACH passage individually:
            - Evaluate how well it addresses the query
            - Note specific relevant information or keywords

            2. Then, within <answer> tags, provide ONLY the final ranking in descending order of relevance using the format: [X] > [Y] > [Z]"""
    })

    return messages

def construct_history_text_reasonrank(sessions, query, session_dates=None):
    messages = []
    messages.append({"role": "system", "content": "You are RankLLM, an intelligent assistant that can rank passages based on their relevance to the query. Given a query and a passage list, you first thinks about the reasoning process in the mind and then provides the answer (i.e., the reranked passage list). The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."})
    num = len(sessions)
    prefix = f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"
    rank = 0
    input_context = f"{prefix}\n"
    for session_turns in sessions:
        rank += 1
        # content = convert_doc_to_prompt_content(self._tokenizer, cand.doc, max_length, truncate_by_word=False)
        content = ""
        if session_dates is not None: # TODO: 这个时间信息可以改
            session_date = session_dates[rank - 1]
            content += f"Date: {session_date}\n"
        session_text = construct_session_text(session_turns) # TODO: 我看只取了 100 个 token，这里是不是也要限制一下？
        content += session_text
        input_context += f"[{rank}] {content}\n"
    example_ordering = "[2] > [1]"
    post = f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The format of the answer should be [] > [], e.g., {example_ordering}."
    input_context += post
    messages.append({"role": "user", "content": input_context})
    return messages


def robust_parse_reasonrank(response_text: str):
    try:
        # get rid of think steps
        think_end_idx = response_text.find("</think>")
        if think_end_idx == -1:
            return []
        response_text = response_text[think_end_idx + len("</think>"):]
        ranking_pattern = "<answer>"
        ranking_end_pattern = "</answer>"
        # parse [2] > [1] > [3] > ...
        ranking_start_idx = response_text.find(ranking_pattern)
        if ranking_start_idx == -1:
            return []
        ranking_end_idx = response_text.find(ranking_end_pattern, ranking_start_idx)
        if ranking_end_idx == -1:
            return []
        ranking_text = response_text[ranking_start_idx + len(ranking_pattern): ranking_end_idx]
        session_rankings = ranking_text.split(">")
        session_ranking_ids = []
        for i, sid in enumerate(session_rankings):
            if sid.strip() == "":
                continue
            if sid.replace("[", "").replace("]", "").strip() == "":
                continue
            sid = int(sid.replace("[", "").replace("]", "").strip())
            session_ranking_ids.append(sid)
        return session_ranking_ids
    except Exception as e:
        logger.warning(f"Error parsing ranking: {e}")
        import traceback
        traceback.print_exc()
        return []


class RerankBaselineAddPromptActor():
    def __init__(self,
                 prompt_config: edict,
                 filtered_idx_col: str,
                 rerank_method: str = "none"
            ):
        self.prompt_template = prompt_config.prompt_template
        self.filtered_idx_col = filtered_idx_col
        self.rerank_method = rerank_method

    def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
        haystack_sessions = row["haystack_sessions"]
        haystack_dates = row.get("haystack_dates", None)
        if self.filtered_idx_col:
            prefiltered_session_idx = row[self.filtered_idx_col]
            haystack_sessions = [haystack_sessions[idx] for idx in prefiltered_session_idx]
            if haystack_dates is not None:
                haystack_dates = [haystack_dates[idx] for idx in prefiltered_session_idx]
        question = row["question"]

        if self.rerank_method == "rankr1":
            messages = construct_history_text_rankr1(haystack_sessions, query=question, session_dates=haystack_dates)
        elif self.rerank_method == "rearank":
            messages = construct_history_text_rearank(haystack_sessions, query=question, session_dates=haystack_dates)
        elif self.rerank_method == "reasonrank":
            messages = construct_history_text_reasonrank(haystack_sessions, query=question, session_dates=haystack_dates)
        else:
            raise ValueError(f"rerank_method {self.rerank_method} not supported")

        row["messages"] = messages
        return row


class ReasonRankParserActor():
    def __init__(self, filtered_idx_col: str):
        self.filtered_idx_col = filtered_idx_col

    def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row["haystack_session_gen_rankings"] = robust_parse_reasonrank(row["response"])
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

