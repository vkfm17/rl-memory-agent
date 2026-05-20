import json
import os
from dataclasses import dataclass, field
from typing import Any

from env.tasks import (
    generate_contradiction_conversation,
    generate_conversation,
    generate_distractor_heavy,
    generate_hard_task,
    generate_temporal_updates,
)

_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "curriculum_state.json")


@dataclass
class CurriculumState:
    level: int = 0
    success_window: list = field(default_factory=list)
    window_size: int = 20


class Curriculum:
    """
    Controls difficulty progression for memory RL training.

    Pass level=N to pin to a fixed difficulty (benchmarking).
    Omit level to resume from the saved checkpoint (training).
    """

    def __init__(
        self,
        level: int | None = None,
        base_distractors: int = 10,
        checkpoint: str | None = None,
    ):
        self._fixed = level is not None
        self._base_distractors = base_distractors
        self._checkpoint = checkpoint or _CHECKPOINT
        self.state = CurriculumState()
        if self._fixed:
            self.state.level = level
        else:
            self._load()

    # ---------------------------------------------------------
    # PERSISTENCE
    # ---------------------------------------------------------
    def _load(self) -> None:
        if not os.path.exists(self._checkpoint):
            return
        with open(self._checkpoint) as f:
            data = json.load(f)
        self.state.level = data.get("level", 0)
        self.state.success_window = data.get("success_window", [])

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._checkpoint), exist_ok=True)
        with open(self._checkpoint, "w") as f:
            json.dump(
                {"level": self.state.level, "success_window": self.state.success_window},
                f,
            )

    # ---------------------------------------------------------
    # TASK SAMPLING
    # ---------------------------------------------------------
    def sample_task(self) -> dict[str, Any]:

        level = self.state.level
        d = self._base_distractors

        if level == 0:
            return generate_conversation(num_distractors=max(1, d // 2))

        elif level == 1:
            return generate_conversation(num_distractors=d)

        elif level == 2:
            return generate_contradiction_conversation(num_distractors=d)

        elif level == 3:
            return generate_distractor_heavy()

        elif level == 4:
            return generate_temporal_updates()

        else:
            return generate_hard_task()

    # ---------------------------------------------------------
    # UPDATE RULE
    # ---------------------------------------------------------
    def update(self, success: int) -> None:
        if self._fixed:
            return

        self.state.success_window.append(success)

        if len(self.state.success_window) < self.state.window_size:
            self._save()
            return

        avg_success = sum(self.state.success_window) / self.state.window_size

        if avg_success > 0.75:
            self.state.level += 1
            self.state.success_window.clear()
        elif avg_success < 0.3:
            self.state.level = max(0, self.state.level - 1)
            self.state.success_window.clear()

        self._save()
