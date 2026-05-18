"""
A curriculum task scheduler that:
1. Starts easy (single fact)
2. Gradually introduces:
    - contradictions
    - distractors
    - multi-fact reasoning
    - temporal updates
3. Tracks performance per level
4. Controls sampling during training
"""

from dataclasses import dataclass
import random
from typing import Callable

# -----------------------------
# TASK IMPORTS
# -----------------------------
from env.tasks import (
    generate_contradiction_chain,
    generate_distractor_heavy,
    generate_multi_query,
    generate_profile_with_updates,
    generate_simple_fact,
    generate_temporal_updates,
)


# -----------------------------
# CURRICULUM LEVEL
# -----------------------------
@dataclass
class CurriculumLevel:
    name: str
    tasks: list[Callable]
    weights: list[float]


# -----------------------------
# CURRICULUM SCHEDULE
# -----------------------------
CURRICULUM = [
    CurriculumLevel(
        name="easy",
        tasks=[
            generate_simple_fact,
        ],
        weights=[1.0],
    ),
    CurriculumLevel(
        name="medium",
        tasks=[
            generate_simple_fact,
            generate_contradiction_chain,
        ],
        weights=[0.6, 0.4],
    ),
    CurriculumLevel(
        name="hard",
        tasks=[
            generate_contradiction_chain,
            generate_profile_with_updates,
            generate_temporal_updates,
        ],
        weights=[0.4, 0.3, 0.3],
    ),
    CurriculumLevel(
        name="very_hard",
        tasks=[
            generate_profile_with_updates,
            generate_temporal_updates,
            generate_distractor_heavy,
            generate_multi_query,
        ],
        weights=[0.25, 0.25, 0.25, 0.25],
    ),
]


class CurriculumScheduler:
    def __init__(self):

        self.level_idx = 0
        self.episode_counter = 0

    def current_level(self) -> CurriculumLevel:
        return CURRICULUM[self.level_idx]

    def sample_task(self):

        level = self.current_level()

        task = random.choices(
            level.tasks,
            weights=level.weights,
            k=1,
        )[0]

        return task()

    # -----------------------------
    # PROGRESSION LOGIC
    # -----------------------------
    def update(self, success_rate: float):

        self.episode_counter += 1

        # move up if performing well
        if success_rate > 0.7 and self.episode_counter > 50:
            if self.level_idx < len(CURRICULUM) - 1:
                self.level_idx += 1
                self.episode_counter = 0

                print(f"[CURRICULUM] Upgraded to: {self.current_level().name}")

        # optional: prevent collapse backward (can add later)
