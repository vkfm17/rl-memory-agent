from typing import Any, Dict, List

import numpy as np


# -----------------------------
# Token utilities
# -----------------------------
def count_tokens(text: str) -> int:
    return len(text.split())


def memory_token_count(memory: List[Dict[str, Any]]) -> int:
    return sum(count_tokens(m["message"]) for m in memory)


# -----------------------------
# Age statistics
# -----------------------------
def memory_age_stats(
    memory: List[Dict[str, Any]],
    current_step: int,
):

    if not memory:
        return {
            "avg_age": 0.0,
            "max_age": 0.0,
        }

    ages = [current_step - m["step"] for m in memory]

    return {
        "avg_age": float(np.mean(ages)),
        "max_age": float(np.max(ages)),
    }


# -----------------------------
# Memory structure stats
# -----------------------------
def memory_structure_stats(
    memory: List[Dict[str, Any]],
):

    if not memory:
        return {
            "size": 0,
            "fact_ratio": 0.0,
            "distractor_ratio": 0.0,
        }

    facts = [m for m in memory if m.get("fact_value") is not None]

    distractors = [m for m in memory if m.get("fact_value") is None]

    total = len(memory)

    return {
        "size": total,
        "fact_ratio": len(facts) / total,
        "distractor_ratio": len(distractors) / total,
    }


# -----------------------------
# Retention efficiency
# -----------------------------
def retention_ratio(
    memory: List[Dict[str, Any]],
    conversation_length: int,
):

    if conversation_length == 0:
        return 0.0

    return len(memory) / conversation_length


# -----------------------------
# Compression efficiency
# -----------------------------
def compression_ratio(
    memory: List[Dict[str, Any]],
    conversation: List[Dict[str, Any]],
):

    mem_tokens = memory_token_count(memory)

    convo_tokens = sum(count_tokens(m["message"]) for m in conversation)

    if convo_tokens == 0:
        return 0.0

    return mem_tokens / convo_tokens


# -----------------------------
# Stale memory heuristic (MVP)
# -----------------------------
def stale_memory_signal(
    memory: List[Dict[str, Any]],
):
    """
    MVP heuristic:
    detects if multiple conflicting fact_values exist.
    """

    fact_values = [
        m.get("fact_value") for m in memory if m.get("fact_value") is not None
    ]

    if len(fact_values) <= 1:
        return 0

    # if multiple different facts exist → possible contradiction
    return len(set(fact_values)) > 1


# -----------------------------
# FULL FEATURE PACK
# -----------------------------
def compute_memory_stats(
    memory: List[Dict[str, Any]],
    conversation: List[Dict[str, Any]],
    current_step: int,
):

    age_stats = memory_age_stats(memory, current_step)
    struct_stats = memory_structure_stats(memory)

    return {
        "memory_tokens": memory_token_count(memory),
        **age_stats,
        **struct_stats,
        "retention_ratio": retention_ratio(
            memory,
            len(conversation),
        ),
        "compression_ratio": compression_ratio(
            memory,
            conversation,
        ),
        "stale_memory_signal": int(stale_memory_signal(memory)),
    }
