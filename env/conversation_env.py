from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter

from constants import MAX_MEMORY
from env.features import build_embedding_features
from env.tasks import generate_conversation
from memory.memory_stats import compute_memory_stats
from memory.similarity import find_most_similar_memory
from typedefs import Action, InfoDict, MemoryEvent, MemoryItem, Observation


# =========================================================
# ENV
# =========================================================
class ConversationEnv(gym.Env):
    def __init__(self) -> None:

        super().__init__()

        # -----------------------------
        # SPACES
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
        self.max_memory: int = MAX_MEMORY
        self.memory: List[MemoryItem] = []

        # -----------------------------
        # CONVERSATION STATE
        # -----------------------------
        self.conversation: List[MemoryItem] = []
        self.question: str = ""
        self.answer: str = ""
        self.current_step: int = 0

        # -----------------------------
        # TRACKING
        # -----------------------------
        self.action_counts: defaultdict[int, int] = defaultdict(int)
        self.action_history: List[int] = []

        # -----------------------------
        # MEMORY EVENT GRAPH
        # -----------------------------
        self.memory_events: Dict[int, MemoryEvent] = {}
        self.next_memory_id: int = 0

        # -----------------------------
        # CURRICULUM (optional)
        # -----------------------------
        self.curriculum: Any = None

        # -----------------------------
        # TENSORBOARD
        # -----------------------------
        self.writer: SummaryWriter = SummaryWriter(log_dir="tensorboard/env")
        self.global_step: int = 0

    # =========================================================
    # RESET
    # =========================================================
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Observation, InfoDict]:

        super().reset(seed=seed)

        if self.curriculum is not None:
            sample = self.curriculum.sample_task()
        else:
            sample = generate_conversation()

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
            [
                len(self.memory),
                step,
                avg_age,
                max_age,
            ],
            dtype=np.float32,
        )

        return build_embedding_features(
            message=msg,
            metadata=metadata,
        )

    # =========================================================
    # STEP
    # =========================================================
    def step(
        self,
        action: int,
    ) -> Tuple[Observation, float, bool, bool, InfoDict]:

        current_msg = self.conversation[self.current_step]
        action_int = int(np.array(action).item())

        self.action_history.append(action_int)
        self.action_counts[action_int] += 1

        # =====================================================
        # MEMORY UPDATE HELPERS
        # =====================================================

        def create_memory(msg: Dict[str, Any], step: int) -> Dict[str, Any]:
            mem_id = self.next_memory_id
            self.next_memory_id += 1

            self.memory_events[mem_id] = MemoryEvent(
                memory_id=mem_id,
                content=msg["message"],
                created_step=step,
                action=action_int,
            )

            return {
                "id": mem_id,
                "message": msg["message"],
                "step": step,
                "fact_value": msg.get("fact_value"),
            }

        # =====================================================
        # ACTION: KEEP
        # =====================================================
        if action_int == Action.KEEP:
            if len(self.memory) >= self.max_memory:
                old = self.memory.pop(0)
                old_id = old["id"]

                self.memory_events[old_id].deleted_step = self.current_step
                self.memory_events[old_id].action = Action.KEEP

            self.memory.append(create_memory(current_msg, self.current_step))

        # =====================================================
        # ACTION: DROP
        # =====================================================
        elif action_int == Action.DROP:
            pass

        # =====================================================
        # ACTION: REPLACE_SIMILAR
        # =====================================================
        elif action_int == Action.REPLACE_SIMILAR:
            idx = find_most_similar_memory(
                current_msg["message"],
                self.memory,
            )

            if idx is not None:
                old = self.memory[idx]
                old_id = old["id"]

                self.memory_events[old_id].deleted_step = self.current_step
                self.memory_events[old_id].action = Action.REPLACE_SIMILAR

                new_mem = create_memory(current_msg, self.current_step)

                self.memory_events[old_id].replaced_by = new_mem["id"]

                self.memory[idx] = new_mem

            else:
                self.memory.append(create_memory(current_msg, self.current_step))

        # =====================================================
        # STEP ADVANCE
        # =====================================================
        self.current_step += 1

        terminated = self.current_step >= len(self.conversation)
        truncated = False

        reward = 0.0
        info: InfoDict = {}

        # =====================================================
        # EPISODE END
        # =====================================================
        if terminated or truncated:
            memory_stats = compute_memory_stats(
                memory=self.memory,
                conversation=self.conversation,
                current_step=self.current_step,
            )

            correct = 0

            remembered = [m for m in self.memory if m.get("fact_value") is not None]

            if remembered:
                latest = sorted(remembered, key=lambda x: x["step"])[-1]["fact_value"]

                correct = int(latest == self.answer)

            reward = self._compute_reward(correct)

            if self.curriculum is not None and terminated:
                self.curriculum.update(success_rate=correct)

            self._log_episode(reward, correct, memory_stats)
            self._log_memory_events()

            info = {
                "correct": correct,
                **memory_stats,
            }

        # =====================================================
        # STEP LOG
        # =====================================================
        self._log_step(action_int)
        self.global_step += 1

        return self._get_obs(), reward, terminated, truncated, info

    # =========================================================
    # REWARD
    # =========================================================
    def _compute_reward(self, correct: int) -> float:

        remembered = [m for m in self.memory if m.get("fact_value") is not None]

        if not remembered:
            return -10.0

        latest = sorted(remembered, key=lambda x: x["step"])[-1]["fact_value"]

        correctness_reward = 10.0 if latest == self.answer else -10.0

        token_penalty = sum(len(m["message"].split()) for m in self.memory) * 0.05

        contradiction_penalty = (
            5.0 if len(set(m.get("fact_value") for m in remembered)) > 1 else 0.0
        )

        return correctness_reward - token_penalty - contradiction_penalty

    # =========================================================
    # TENSORBOARD: STEP
    # =========================================================
    def _log_step(self, action: int) -> None:

        self.writer.add_scalar("step/action", action, self.global_step)
        self.writer.add_scalar("step/memory_size", len(self.memory), self.global_step)
        self.writer.add_scalar("step/index", self.current_step, self.global_step)

    # =========================================================
    # TENSORBOARD: EPISODE
    # =========================================================
    def _log_episode(
        self,
        reward: float,
        correct: int,
        memory_stats: Dict[str, Any],
    ) -> None:

        self.writer.add_scalar("episode/reward", reward, self.global_step)
        self.writer.add_scalar("episode/correct", correct, self.global_step)
        self.writer.add_scalar(
            "episode/memory_tokens",
            memory_stats.get("memory_tokens", 0),
            self.global_step,
        )
        self.writer.add_scalar(
            "episode/memory_size", len(self.memory), self.global_step
        )
        self.writer.add_scalar(
            "episode/compression_ratio",
            memory_stats.get("compression_ratio", 0.0),
            self.global_step,
        )
        self.writer.add_scalar(
            "episode/retention_ratio",
            memory_stats.get("retention_ratio", 0.0),
            self.global_step,
        )

    # =========================================================
    # MEMORY LIFECYCLE ANALYTICS
    # =========================================================
    def _log_memory_events(self) -> None:

        lifetimes = []

        for event in self.memory_events.values():
            if event.deleted_step is not None:
                lifetimes.append(event.deleted_step - event.created_step)

        if lifetimes:
            self.writer.add_scalar(
                "memory/avg_lifetime",
                float(np.mean(lifetimes)),
                self.global_step,
            )

            self.writer.add_scalar(
                "memory/max_lifetime",
                float(np.max(lifetimes)),
                self.global_step,
            )

            self.writer.add_scalar(
                "memory/min_lifetime",
                float(np.min(lifetimes)),
                self.global_step,
            )
