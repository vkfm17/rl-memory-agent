import os
import re
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from constants import MAX_MEMORY

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "all-MiniLM-L6-v2")

# Hard-coded heuristics

KNOWN_LOCATIONS = {
    "hoboken",
    "seattle",
    "boston",
    "new york",
    "san francisco",
    "london",
}

COMMON_NAMES = {
    "john",
    "mary",
    "alice",
    "bob",
    "michael",
    "sarah",
}

TEMPORAL_WORDS = {
    "moved",
    "now",
    "currently",
    "used to",
    "formerly",
}
PERSONAL_KEYWORDS = {
    "my",
    "i am",
    "i live",
    "favorite",
    "birthday",
}


# Lazy-loaded encoder

_encoder = None


def get_encoder():
    global _encoder
    if _encoder is None:
        model_path = _MODEL_DIR if os.path.isdir(_MODEL_DIR) else "all-MiniLM-L6-v2"
        _encoder = SentenceTransformer(model_path)
    return _encoder


# Feature extractors


def contains_number(message: str) -> bool:
    """
    Detects numbers, dates, ages, etc.
    """
    return bool(re.search(r"\d", message))


def contains_location(message: str) -> bool:
    """
    Simple keyword-based location detection.
    """
    message_lower = message.lower()
    for location in KNOWN_LOCATIONS:
        if location in message_lower:
            return True
    return False


def contains_name(message: str) -> bool:
    """
    Detects likely person names.
    """
    words = re.findall(r"\b[A-Z][a-z]+\b", message)
    for word in words:
        if word.lower() in COMMON_NAMES:
            return True
    return False


def message_length(message: str) -> int:
    """
    Approximate token count using words.
    """
    return len(message.split())


def message_age(current_step: int, message_step: int) -> int:
    """
    How old a memory is relative to current timestep.
    """
    return current_step - message_step


def retrieval_frequency(message: str) -> int:
    """
    Heuristic estimate:
    how likely this message is
    to be queried later.
    """
    message_lower = message.lower()
    score = 0
    retrieval_keywords = [
        "birthday",
        "live",
        "favorite",
        "name",
        "from",
        "movie",
    ]
    for keyword in retrieval_keywords:
        if keyword in message_lower:
            score += 1
    return score


def importance_prior(message: str) -> int:
    """
    Crude estimate of memory importance.
    """
    message_lower = message.lower()
    score = 0
    for keyword in PERSONAL_KEYWORDS:
        if keyword in message_lower:
            score += 1
    return score


def contains_temporal_update(message: str) -> bool:
    """
    Detects likely memory-updating language.
    """
    message_lower = message.lower()
    for word in TEMPORAL_WORDS:
        if word in message_lower:
            return True
    return False


def embed_message(message: str) -> np.ndarray:
    encoder = get_encoder()
    embedding = encoder.encode(message)
    embedding = np.array(
        embedding,
        dtype=np.float32,
    )
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def build_feature_vector(message: str) -> np.ndarray:
    return np.array(
        [
            contains_number(message),
            contains_location(message),
            contains_name(message),
            message_length(message),
            retrieval_frequency(message),
            importance_prior(message),
            contains_temporal_update(message),
        ],
        dtype=np.float32,
    )


def build_memory_embedding(memory: list[dict[str, Any]]) -> np.ndarray:
    """Mean-pool embeddings of current memory items; zeros if memory is empty."""
    if not memory:
        return np.zeros(384, dtype=np.float32)
    embeddings = np.stack([embed_message(m["message"]) for m in memory])
    mean = embeddings.mean(axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm).astype(np.float32) if norm > 0 else mean


def build_memory_similarities(
    msg_embedding: np.ndarray,
    memory: list[dict[str, Any]],
    max_memory: int = MAX_MEMORY,
) -> np.ndarray:
    """Cosine similarity of the current message against each memory slot (padded to max_memory)."""
    sims = np.zeros(max_memory, dtype=np.float32)
    for i, m in enumerate(memory[:max_memory]):
        mem_emb = embed_message(m["message"])
        denom = np.linalg.norm(msg_embedding) * np.linalg.norm(mem_emb)
        sims[i] = float(np.dot(msg_embedding, mem_emb) / denom) if denom > 0 else 0.0
    return sims


def build_embedding_features(
    message: str,
    metadata: np.ndarray,
    memory: list[dict[str, Any]] | None = None,
    max_memory: int = MAX_MEMORY,
) -> np.ndarray:
    """
    Observation vector: current msg embedding (384) + memory mean embedding (384)
    + heuristic features (7) + metadata (4) + per-slot similarity (max_memory).
    """
    if memory is None:
        memory = []
    msg_embedding = embed_message(message)
    memory_embedding = build_memory_embedding(memory)
    heuristics = build_feature_vector(message)
    similarities = build_memory_similarities(msg_embedding, memory, max_memory)
    return np.concatenate([msg_embedding, memory_embedding, heuristics, metadata, similarities])
