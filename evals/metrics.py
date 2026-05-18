from typing import Any


def count_message_tokens(memory: list[dict[str, Any]]) -> int:
    """Count the number of tokens."""
    return sum(len(m["message"].split()) for m in memory)


def compression_ratio(
    memory: list[dict[str, Any]],
    conversation: list[dict[str, Any]],
) -> float:
    """Token compression ratio based on number of memories saved from overall conversation."""
    memory_tokens = count_message_tokens(memory)
    conversation_tokens = count_message_tokens(conversation)
    if conversation_tokens == 0:
        return 0.0
    return memory_tokens / conversation_tokens


def retention_ratio(
    memory: list[dict[str, Any]],
    conversation: list[dict[str, Any]],
) -> float:
    """Percentage of messages saved in memory."""
    if len(conversation) == 0:
        return 0.0
    return len(memory) / len(conversation)


def distractor_retention_ratio(memory: list[dict[str, Any]]):
    """Percentage of distractors that were saved."""
    if len(memory) == 0:
        return 0.0

    distractors = 0
    for m in memory:
        if m.get("fact_value") is None:
            distractors += 1
    return distractors / len(memory)


def stale_memory_retained(
    memory: list[dict[str, Any]],
    correct_answer: str,
) -> float:
    """
    Returns True if outdated contradictory
    memories still exist.
    """

    remembered_values = []
    for m in memory:
        value = m.get("fact_value")

        if value is not None:
            remembered_values.append(value)

    unique_values = set(remembered_values)

    if len(unique_values) <= 1:
        return False

    # Multiple conflicting memories exist
    if correct_answer in unique_values:
        unique_values.remove(correct_answer)

    return len(unique_values) > 0


def action_distribution(action_history: list[int]) -> dict[int, float]:
    """Distribution of actions taken by the RL policy."""

    if len(action_history) == 0:
        return {}

    counts = {}
    for action in action_history:
        counts[action] = counts.get(action, 0) + 1

    total = len(action_history)
    return {action: count / total for action, count in counts.items()}


def efficiency_gap(reward: float, correct: float) -> float:
    return reward - correct * 10


def summarize_episode(
    memory: list[dict[str, Any]],
    conversation: list[dict[str, Any]],
    reward: float,
    correct: int,
    action_history: list[int],
    correct_answer: str,
):

    metrics = {}
    metrics["reward"] = reward
    metrics["correct"] = int(correct)
    metrics["memory_tokens"] = count_message_tokens(memory)
    metrics["compression_ratio"] = compression_ratio(
        memory,
        conversation,
    )
    metrics["retention_ratio"] = retention_ratio(
        memory,
        conversation,
    )
    metrics["distractor_retention_ratio"] = distractor_retention_ratio(memory)
    metrics["stale_memory_retained"] = int(
        stale_memory_retained(
            memory,
            correct_answer,
        )
    )
    metrics["efficiency_gap"] = efficiency_gap(reward, correct)

    action_dist = action_distribution(action_history)
    for action, pct in action_dist.items():
        metrics[f"action_{action}_pct"] = pct

    return metrics
