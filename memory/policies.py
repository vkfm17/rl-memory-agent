import random

class RandomPolicy:
    """Randomly select to drop or keep."""

    def act(self, obs):
        return random.randint(0, 1)


class KeepEverythingPolicy:
    """Keep everything"""
    def act(self, obs):
        return 0


class DropEverythingPolicy:
    """Drop everything"""
    def act(self, obs):
        return 1

"""
Also add:

keyword heuristic
TF-IDF importance
random baseline

These become evaluation baselines.
"""