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
    # Initialize selected session indices list and token count
    selected_session_indices = []
    total_num_tokens = 0

    for asidx in answer_session_idx:
        # Ensure answer_session_idx is within valid range
        if asidx < 0 or asidx >= len(sessions):
            raise ValueError(f"answer_session_idx {asidx} out of range for sessions of length {len(sessions)}")

        # First check the token count of the session corresponding to answer_session_idx
        answer_session_turns = sessions[asidx]
        answer_session_text = construct_session_text(answer_session_turns)
        if session_dates is not None:
            session_date = session_dates[asidx]
            answer_session_text = f"<session42> <date>{session_date}</date>\n{answer_session_text}</session>\n" # not final session idx, just a placeholder for token count
        else:
            answer_session_text = f"<session42>\n{answer_session_text}</session>\n" # not final session idx, just a placeholder for token count
        answer_num_tokens = len(tokenizer.tokenize(answer_session_text))

        # Add answer_session index
        selected_session_indices.append(asidx)
        total_num_tokens += answer_num_tokens
    
    # Create index list for other sessions excluding answer_session
    other_session_indices = [i for i in range(len(sessions)) if i not in answer_session_idx]
    
    # Randomly shuffle other sessions
    if sample_strategy == "random":
        random.shuffle(other_session_indices)
    elif sample_strategy == "similarity":
        if session_sim_scores is None:
            raise ValueError("session_sim_scores must be provided when sample_strategy is 'similarity'")
        other_session_indices = sorted(other_session_indices, key=lambda x: session_sim_scores[x], reverse=True)
    else:
        raise ValueError(f"sample_strategy {sample_strategy} not supported")
    
    # Try to add other sessions until reaching token limit
    for sid in other_session_indices:
        session_turns = sessions[sid]
        session_text = construct_session_text(session_turns)
        if session_dates is not None:
            session_date = session_dates[sid]
            session_text = f"<session42> <date>{session_date}</date>\n{session_text}</session>\n" # placeholder for token count
        else:
            session_text = f"<session42>\n{session_text}</session>\n" # placeholder for token count
        num_tokens = len(tokenizer.tokenize(session_text))
        
        # If adding this session won't exceed token limit, add its index
        if total_num_tokens + num_tokens <= max_prompt_tokens:
            selected_session_indices.append(sid)
            total_num_tokens += num_tokens
        else:
            # break the loop if adding this session exceeds the token limit
            break

    # Sort selected session indices in original order
    selected_session_indices.sort()
    
    return selected_session_indices



