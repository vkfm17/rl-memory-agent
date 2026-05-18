import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from constants import MAX_MEMORY
from env.features import build_embedding_features
from env.tasks import (
    generate_contradiction_conversation,
    generate_conversation,
)
from evals.memory_tracker import MemoryTracker
from memory.memory_stats import compute_memory_stats
from memory.similarity import find_most_similar_memory
from typedefs import Action


class ConversationEnv(gym.Env):
    def __init__(self):

        super().__init__()

        self.action_space = spaces.Discrete(len(Action))

        # 384 embedding + 4 metadata = 388
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(388,),
            dtype=np.float32,
        )

        self.max_memory = MAX_MEMORY
        self.action_history = []
        self.tracker = MemoryTracker()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        if random.random() < 0.5:
            sample = generate_conversation()
        else:
            sample = generate_contradiction_conversation()

        self.conversation = sample["conversation"]
        self.question = sample["question"]
        self.answer = sample["answer"]

        self.memory = []
        self.current_step = 0
        self.action_history = []

        return self._get_obs(), {}

    def _get_obs(self):

        msg = self.conversation[self.current_step]["message"]

        ages = [self.current_step - m["step"] for m in self.memory]

        avg_age = np.mean(ages) if ages else 0.0
        max_age = np.max(ages) if ages else 0.0

        metadata = np.array(
            [
                len(self.memory),
                self.current_step,
                avg_age,
                max_age,
            ],
            dtype=np.float32,
        )

        return build_embedding_features(
            message=msg,
            metadata=metadata,
        )

    def step(
        self, action: int | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:

        current_msg = self.conversation[self.current_step]
        self.action_history.append(int(action))

        # -------------------
        # KEEP
        # -------------------
        if action == Action.KEEP:
            if len(self.memory) >= self.max_memory:
                self.memory.pop(0)

            self.memory.append(current_msg)

        # -------------------
        # DROP
        # -------------------
        elif action == Action.DROP:
            pass

        # -------------------
        # REPLACE_SIMILAR
        # -------------------
        elif action == Action.REPLACE_SIMILAR:
            replace_idx = find_most_similar_memory(
                current_msg["message"],
                self.memory,
                threshold=0.7,
            )

            if replace_idx is not None:
                self.memory[replace_idx] = current_msg

            else:
                self.memory.append(current_msg)

        # -------------------
        # SUMMARIZE (DISABLED)
        # -------------------
        # elif action == Action.SUMMARIZE:
        #     summarized = current_msg.copy()
        #     summarized["message"] = summarize_message(
        #         current_msg["message"]
        #     )
        #     summarized["summarized"] = True
        #     self.memory.append(summarized)

        # -------------------
        # STEP UPDATE
        # -------------------
        self.tracker.log_step(
            step=self.current_step,
            message=current_msg["message"],
            memory=self.memory,
            answer=self.answer,
        )
        self.current_step += 1

        terminated = self.current_step >= len(self.conversation)
        truncated = False

        # -------------------
        # REWARD
        # -------------------
        reward = 0.0

        info = {}

        if terminated:
            reward = self._compute_reward()

            # -------------------
            # MEMORY STATS
            # -------------------
            memory_stats = compute_memory_stats(
                memory=self.memory,
                conversation=self.conversation,
                current_step=self.current_step,
            )
            correct = (
                len(self.memory) > 0
                and self.memory[-1].get("fact_value") == self.answer
            )
            info = {
                "correct": int(correct),
                **memory_stats,
            }

        obs = np.zeros(388, dtype=np.float32) if terminated else self._get_obs()

        return obs, reward, terminated, truncated, info

    def _compute_reward(self) -> float:

        # -----------------------------
        # 1. Find latest fact in memory
        # -----------------------------
        remembered = [m for m in self.memory if m.get("fact_value") is not None]

        if not remembered:
            return -10.0

        latest_fact = sorted(remembered, key=lambda x: x["step"])[-1]["fact_value"]

        correct = latest_fact == self.answer

        correctness_reward = 10.0 if correct else -10.0

        # -----------------------------
        # 2. Memory efficiency penalty
        # -----------------------------
        total_tokens = sum(len(m["message"].split()) for m in self.memory)

        token_penalty = total_tokens * 0.05  # reduced from 0.1

        # -----------------------------
        # 3. Contradiction penalty
        # -----------------------------
        fact_values = [
            m.get("fact_value") for m in self.memory if m.get("fact_value") is not None
        ]

        contradiction_penalty = 0.0

        if len(set(fact_values)) > 1:
            contradiction_penalty = 5.0

        # -----------------------------
        # FINAL REWARD
        # -----------------------------
        reward = correctness_reward - token_penalty - contradiction_penalty

        return reward
