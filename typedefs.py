from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

MemoryItem = dict[str, Any]
Observation = np.ndarray
InfoDict = dict[str, Any]


class Action(IntEnum):
    KEEP = 0
    DROP = 1
    REPLACE_SIMILAR = 2
    # SUMMARIZE = 3


@dataclass
class MemorySnapshot:
    step: int
    message: str
    memory: list[dict[str, Any]]
    answer: str | None


@dataclass
class MemoryEvent:
    memory_id: int
    content: str
    created_step: int
    deleted_step: int | None = None
    action: int | None = None
    replaced_by: int | None = None
