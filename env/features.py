import re
from sentence_transformers import SentenceTransformer
import numpy as np


# -------------------------
# Hard-coded heuristics
# -------------------------

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


# -------------------------
# Lazy-loaded encoder
# -------------------------

_encoder = None

def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )
    return _encoder


# -------------------------
# Feature extractors
# -------------------------

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
    embedding = (
        embedding /
        np.linalg.norm(embedding)
    )
    return embedding

def build_feature_vector(message: str) -> np.ndarray:
    return np.array([
        contains_number(message),
        contains_location(message),
        contains_name(message),
        message_length(message),
        retrieval_frequency(message),
        importance_prior(message),
        contains_temporal_update(message),
    ], dtype=np.float32)

def build_embedding_features(
    message: str,
    memory_size: int,
    current_step: int,
) -> np.ndarray:
    """Use embeddings, memory size, and message age to """
    embedding = embed_message(message)
    metadata = np.array([
        memory_size,
        current_step,
    ], dtype=np.float32)
    return np.concatenate([
        embedding,
        metadata,
    ])

# [
  # 384-dim sentence embedding,
  # memory_size,
  # message_age,
# ]