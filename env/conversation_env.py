import random
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.features import (
    build_embedding_features,
)
from env.tasks import generate_contradiction_conversation, generate_conversation
from memory.similarity import find_most_similar_memory

# Hard-coded to 5 to start
MAX_MEMORY = 5


class Action(IntEnum):
    KEEP = 0
    DROP = 1
    REPLACE_SIMILAR = 2
    # SUMMARIZE = 3


class ConversationEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(388,),
            dtype=np.float32,
        )
        self.max_memory = MAX_MEMORY

    def reset(
        self, seed: int | None = None, options: Any | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
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
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        msg = self.conversation[self.current_step]["message"]

        # Get some stats about the ages of current memories
        # (How old are the memories already occupying memory?)
        ages = [self.current_step - m["step"] for m in self.memory]
        avg_age = np.mean(ages) if ages else 0
        max_age = np.max(ages) if ages else 0

        # Build metadata
        metadata = np.array(
            [len(self.memory), self.current_step, avg_age, max_age], dtype=np.float32
        )

        # Build features from message and metadata
        obs = build_embedding_features(message=msg, metadata=metadata)
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Keep or drop based on decision."""

        current_msg = self.conversation[self.current_step]

        # KEEP
        if action == Action.KEEP:
            # If we're at max memory, remove the first
            if len(self.memory) >= self.max_memory:
                self.memory.pop(0)

            self.memory.append(current_msg)

        # DROP
        elif action == Action.DROP:
            # Don't add
            pass

        # REPLACE_SIMILAR
        elif action == Action.REPLACE_SIMILAR:
            replace_idx = find_most_similar_memory(
                current_msg["message"],
                self.memory,
            )
            if replace_idx is not None:
                print(
                    "REPLACED:",
                    self.memory[replace_idx]["message"],
                    "WITH:",
                    current_msg["message"],
                )
                self.memory[replace_idx] = current_msg

            else:
                self.memory.append(current_msg)

        # SUMMARIZE
        # elif action == ACTION.SUMMARIZE:
        #     summarized = current_msg.copy()
        #     summarized["message"] = (
        #         summarize_message(
        #             current_msg["message"]
        #         )
        #     )
        #     summarized["summarized"] = True
        #     self.memory.append(summarized)

        self.current_step += 1
        done = self.current_step >= len(self.conversation)
        reward = 0
        # Only calculate reward at the end
        if done:
            reward = self._compute_reward()
            # Reset
            obs = np.zeros(3, dtype=np.float32)

        else:
            obs = self._get_obs()

        info = {}
        return obs, reward, done, False, info

    def _compute_reward(self) -> float:
        """Calculate the reward."""
        remembered_answers = []

        for msg in self.memory:
            if msg["fact_value"] is not None:
                remembered_answers.append((msg["step"], msg["fact_value"]))

        # If no memories stored, bad
        if not remembered_answers:
            return -10

        # Newest remembered fact
        newest = sorted(remembered_answers, key=lambda x: x[0])[-1][1]
        correct = newest == self.answer
        accuracy_reward = 10 if correct else -10

        # token_penalty = len(self.memory) * 0.5
        # token_penalty = sum(
        #     len(msg["message"].split())
        #     for msg in self.memory
        # ) * 0.05
        # token_penalty = len(self.memory) * 2
        # token_penalty = total_tokens * 0.25
        # if remembered: reward += 10
        # if len(memory) == 1: reward += 5
        total_tokens = sum(len(m["message"].split()) for m in self.memory)
        token_penalty = total_tokens * 0.1

        return accuracy_reward - token_penalty
