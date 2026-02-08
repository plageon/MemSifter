import random


def construct_session_text(session_turns):
    session_text = ""
    user_id, assistant_id = 0, 0
    role_id_map = {}
    for tid, turn in enumerate(session_turns):
        if turn["role"] == "user":
            session_text += f'<user{user_id}>{turn["content"]}</user>\n'
            user_id += 1
        elif turn["role"] == "assistant":
            session_text += f'<assistant{assistant_id}>{turn["content"]}</assistant>\n'
            assistant_id += 1
        else:
            role = turn["role"]
            if role not in role_id_map:
                role_id_map[role] = 0
            session_text += f'<{role}{role_id_map[role]}>{turn["content"]}</{role}>\n'
            role_id_map[role] += 1
    return session_text

def construct_history_text(sessions, session_dates=None):
    history_text = ""
    for sid, session_turns in enumerate(sessions):
        if session_dates is not None:
            session_date = session_dates[sid]
            history_text += f'<session{sid}> <date>{session_date}</date>\n'
        else:
            history_text += f'<session{sid}>\n'
        session_text = construct_session_text(session_turns)
        history_text += session_text
        history_text += f'</session>\n'
    return history_text.strip()


def construct_history_text_with_limited_context(
        sessions,
        answer_session_idx,
        max_prompt_tokens,
        tokenizer,
        session_dates=None,
        sample_strategy="random",
        session_sim_scores=None,
    ):
    # 初始化选中的session索引列表和token计数
    selected_session_indices = []
    total_num_tokens = 0

    for asidx in answer_session_idx:
        # 确保answer_session_idx在有效范围内
        if asidx < 0 or asidx >= len(sessions):
            raise ValueError(f"answer_session_idx {asidx} out of range for sessions of length {len(sessions)}")

        # 首先检查answer_session_idx对应的session的token数量
        answer_session_turns = sessions[asidx]
        answer_session_text = construct_session_text(answer_session_turns)
        if session_dates is not None:
            session_date = session_dates[asidx]
            answer_session_text = f"<session42> <date>{session_date}</date>\n{answer_session_text}</session>\n" # not final session idx, just a placeholder for token count
        else:
            answer_session_text = f"<session42>\n{answer_session_text}</session>\n" # not final session idx, just a placeholder for token count
        answer_num_tokens = len(tokenizer.tokenize(answer_session_text))

        # 添加answer_session的索引
        selected_session_indices.append(asidx)
        total_num_tokens += answer_num_tokens
    
    # 创建除answer_session外的其他session的索引列表
    other_session_indices = [i for i in range(len(sessions)) if i not in answer_session_idx]
    
    # 随机打乱其他session的顺序
    if sample_strategy == "random":
        random.shuffle(other_session_indices)
    elif sample_strategy == "similarity":
        if session_sim_scores is None:
            raise ValueError("session_sim_scores must be provided when sample_strategy is 'similarity'")
        other_session_indices = sorted(other_session_indices, key=lambda x: session_sim_scores[x], reverse=True)
    else:
        raise ValueError(f"sample_strategy {sample_strategy} not supported")
    
    # 尝试添加其他session，直到达到token限制
    for sid in other_session_indices:
        session_turns = sessions[sid]
        session_text = construct_session_text(session_turns)
        if session_dates is not None:
            session_date = session_dates[sid]
            session_text = f"<session42> <date>{session_date}</date>\n{session_text}</session>\n" # placeholder for token count
        else:
            session_text = f"<session42>\n{session_text}</session>\n" # placeholder for token count
        num_tokens = len(tokenizer.tokenize(session_text))
        
        # 如果添加这个session不会超过token限制，就添加它的索引
        if total_num_tokens + num_tokens <= max_prompt_tokens:
            selected_session_indices.append(sid)
            total_num_tokens += num_tokens
        else:
            # break the loop if adding this session exceeds the token limit
            break

    # 把选中的session索引按照原始顺序排序
    selected_session_indices.sort()
    
    return selected_session_indices



