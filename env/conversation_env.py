from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.tasks import generate_conversation

# Hard-coded to 5 to start
MAX_MEMORY = 5

from env.features import (
    build_feature_vector,
    build_embedding_features,
    contains_number,
    contains_location,
    contains_name,
    message_length,
    retrieval_frequency,
)

class ConversationEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # Actions:
        # 0 = KEEP
        # 1 = DROP
        self.action_space = spaces.Discrete(2)

        # Observation:
        # [
        #   message_age,
        #   memory_size,
        #   is_fact_message,
        # ]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(386,),
            dtype=np.float32,
        )

        self.max_memory = MAX_MEMORY

    def reset(self, seed: int | None = None, options: Any | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        sample = generate_conversation()

        self.conversation = sample["conversation"]
        self.question = sample["question"]
        self.answer = sample["answer"]

        self.memory = []
        self.current_step = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        msg = self.conversation[
            self.current_step
        ]["message"]
        obs = build_embedding_features(
            message=msg,
            memory_size=len(self.memory),
            current_step=self.current_step,
        )
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Keep or drop based on decision."""

        current_msg = self.conversation[self.current_step]

        # KEEP
        if action == 0:
            # If we're at max memory, remove the first
            if len(self.memory) >= self.max_memory:
                self.memory.pop(0)

            self.memory.append(current_msg)

        # DROP
        elif action == 1:
            # Don't add
            pass

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

    def _compute_reward(self):
        """
        Compute the reward at the end based on if the answer was remembered, with appropriate penalties.
        We will later compare against an LLM response instead of just if remembered.
        """

        print("FINAL MEMORY:")
        for msg in self.memory:
            print(msg)

        remembered = False
        for msg in self.memory:
            if msg["fact_value"] == self.answer:
                remembered = True

        accuracy_reward = 10 if remembered else -10
        # Approximate token/memory penalty
        token_penalty = sum(
            len(msg["message"].split())
            for msg in self.memory
        ) * 0.05
        # token_penalty = len(self.memory) * 2
        # token_penalty = total_tokens * 0.25
        # if remembered: reward += 10
        # if len(memory) == 1: reward += 5
        reward = accuracy_reward - token_penalty
        return reward