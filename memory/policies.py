import random

from memory.similarity import (
    find_most_similar_memory,
)
from typedefs import Action


class BasePolicy:
    def act(
        self,
        current_msg,
        memory,
    ):
        raise NotImplementedError


class KeepLastKPolicy(BasePolicy):
    def __init__(self, k=5):

        self.k = k

    def act(
        self,
        current_msg,
        memory,
    ):

        if len(memory) >= self.k:
            return Action.DROP

        return Action.KEEP


class RandomPolicy(BasePolicy):
    def act(
        self,
        current_msg,
        memory,
    ):

        return random.choice(
            [Action.KEEP, Action.DROP, Action.REPLACE_SIMILAR]  # Action.SUMMARIZE,
        )


class SummarizeOldestPolicy(BasePolicy):
    def __init__(self, k=5):

        self.k = k

    def act(
        self,
        current_msg,
        memory,
    ):

        if len(memory) < self.k:
            return Action.KEEP

        return Action.SUMMARIZE


class ReplaceSimilarPolicy(BasePolicy):
    def act(
        self,
        current_msg,
        memory,
    ):

        idx = find_most_similar_memory(
            current_msg["message"],
            memory,
            threshold=0.7,
        )

        if idx is not None:
            return Action.REPLACE_SIMILAR

        return Action.KEEP
