import numpy as np

from env.features import (
    embed_message,
)

SIMILARITY_THRESHOLD = 0.5


def cosine_similarity(a, b):

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_most_similar_memory(
    incoming_message: str,
    memory: list[str],
    threshold: int = SIMILARITY_THRESHOLD,
) -> int:
    """
    Returns index of most semantically
    similar memory item.
    """

    if len(memory) == 0:
        return None

    incoming_embedding = embed_message(incoming_message)

    similarities = []

    for i, mem in enumerate(memory):
        mem_embedding = embed_message(mem["message"])

        sim = cosine_similarity(
            incoming_embedding,
            mem_embedding,
        )

        similarities.append((i, sim))

    # Highest similarity above threshold
    best_idx, best_sim = max(similarities, key=lambda x: x[1])
    if best_sim < threshold:
        return None

    return best_idx
