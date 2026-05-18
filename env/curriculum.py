from dataclasses import dataclass
from typing import Any, Dict

from env.tasks import (
    generate_contradiction_conversation,
    generate_conversation,
    generate_distractor_heavy,
    generate_hard_task,
    generate_temporal_updates,
)


@dataclass
class CurriculumState:
    level: int = 0
    success_window: list = None
    window_size: int = 20

    def __post_init__(self):
        self.success_window = []


class Curriculum:
    """
    Controls difficulty progression for memory RL training.
    """

    def __init__(self):
        self.state = CurriculumState()

    # ---------------------------------------------------------
    # TASK SAMPLING
    # ---------------------------------------------------------
    def sample_task(self) -> Dict[str, Any]:

        level = self.state.level

        if level == 0:
            return generate_conversation(num_distractors=5)

        elif level == 1:
            return generate_conversation(num_distractors=10)

        elif level == 2:
            return generate_contradiction_conversation(num_distractors=10)

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

        self.state.success_window.append(success)

        if len(self.state.success_window) > self.state.window_size:
            self.state.success_window.pop(0)

        avg_success = sum(self.state.success_window) / len(self.state.success_window)

        # promote
        if avg_success > 0.75:
            self.state.level += 1
            self.state.success_window.clear()

        # optional demotion
        elif avg_success < 0.3:
            self.state.level = max(0, self.state.level - 1)
            self.state.success_window.clear()
