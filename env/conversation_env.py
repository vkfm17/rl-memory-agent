from collections import defaultdict
from typing import List

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter

from constants import MAX_MEMORY
from env.curriculum import Curriculum
from env.features import build_embedding_features
from memory.memory_stats import compute_memory_stats
from memory.similarity import find_most_similar_memory
from typedefs import Action, MemoryEvent, MemoryItem


class ConversationEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # -----------------------------
        # ACTION SPACE
        # -----------------------------
        self.action_space = spaces.Discrete(len(Action))

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(388,),
            dtype=np.float32,
        )

        # -----------------------------
        # MEMORY
        # -----------------------------
        self.memory: List[MemoryItem] = []
        self.max_memory = MAX_MEMORY

        # -----------------------------
        # CONVERSATION
        # -----------------------------
        self.conversation = []
        self.question = ""
        self.answer = ""
        self.current_step = 0

        # -----------------------------
        # TRACKING
        # -----------------------------
        self.action_counts = defaultdict(int)
        self.action_history = []

        # -----------------------------
        # MEMORY EVENTS (CAUSAL GRAPH)
        # -----------------------------
        self.memory_events = {}
        self.next_memory_id = 0

        # -----------------------------
        # CURRICULUM
        # -----------------------------
        self.curriculum = Curriculum()

        # -----------------------------
        # TENSORBOARD
        # -----------------------------
        self.writer = SummaryWriter("tensorboard/env")
        self.global_step = 0

    # =========================================================
    # RESET
    # =========================================================
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        sample = self.curriculum.sample_task()

        self.conversation = sample["conversation"]
        self.question = sample["question"]
        self.answer = sample["answer"]

        self.memory = []
        self.memory_events = {}
        self.next_memory_id = 0

        self.current_step = 0
        self.action_counts = defaultdict(int)
        self.action_history = []

        return self._get_obs(), {}

    # =========================================================
    # OBSERVATION
    # =========================================================
    def _get_obs(self):

        step = min(self.current_step, len(self.conversation) - 1)
        msg = self.conversation[step]["message"]

        ages = [step - m["step"] for m in self.memory]
        avg_age = float(np.mean(ages)) if ages else 0.0
        max_age = float(np.max(ages)) if ages else 0.0

        metadata = np.array(
            [len(self.memory), step, avg_age, max_age], dtype=np.float32
        )

        return build_embedding_features(msg, metadata)

    # =========================================================
    # STEP
    # =========================================================
    def step(self, action):

        if self.current_step >= len(self.conversation):
            return self._get_obs(), 0.0, True, False, {}

        msg = self.conversation[self.current_step]
        action = int(np.array(action).item())

        self.action_history.append(action)
        self.action_counts[action] += 1

        # -----------------------------
        # CREATE MEMORY
        # -----------------------------
        def create_memory(msg, step):

            mem_id = self.next_memory_id
            self.next_memory_id += 1

            self.memory_events[mem_id] = MemoryEvent(
                memory_id=mem_id,
                content=msg["message"],
                created_step=step,
                action=action,
            )

            return {
                "id": mem_id,
                "message": msg["message"],
                "step": step,
                "fact_value": msg.get("fact_value"),
            }

        # -----------------------------
        # ACTIONS
        # -----------------------------
        if action == Action.KEEP:
            if len(self.memory) >= self.max_memory:
                old = self.memory.pop(0)
                self.memory_events[old["id"]].deleted_step = self.current_step
                self.memory_events[old["id"]].action = Action.KEEP

            self.memory.append(create_memory(msg, self.current_step))

        elif action == Action.DROP:
            pass

        elif action == Action.REPLACE_SIMILAR:
            idx = find_most_similar_memory(msg["message"], self.memory)

            if idx is not None:
                old = self.memory[idx]
                self.memory_events[old["id"]].deleted_step = self.current_step
                self.memory_events[old["id"]].action = Action.REPLACE_SIMILAR

                new_mem = create_memory(msg, self.current_step)
                self.memory_events[old["id"]].replaced_by = new_mem["id"]

                self.memory[idx] = new_mem

            else:
                self.memory.append(create_memory(msg, self.current_step))

        # -----------------------------
        # STEP ADVANCE
        # -----------------------------
        self.current_step += 1

        terminated = self.current_step >= len(self.conversation)
        truncated = False

        reward = 0.0
        info = {}

        # =====================================================
        # EPISODE END
        # =====================================================
        if terminated:
            stats = compute_memory_stats(
                memory=self.memory,
                conversation=self.conversation,
                current_step=self.current_step,
            )

            remembered = [m for m in self.memory if m.get("fact_value") is not None]

            correct = 0

            if remembered:
                latest = sorted(remembered, key=lambda x: x["step"])[-1]["fact_value"]
                correct = int(latest == self.answer)

            reward = (10 if correct else -10) - len(self.memory) * 0.05

            self.curriculum.update(correct)

            self.writer.add_scalar(
                "curriculum/level", self.curriculum.state.level, self.global_step
            )
            self.writer.add_scalar("episode/reward", reward, self.global_step)
            self.writer.add_scalar("episode/correct", correct, self.global_step)

            info = {"correct": correct, **stats}

        self.global_step += 1

        return self._get_obs(), reward, terminated, truncated, info
