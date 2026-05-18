from collections import defaultdict

import numpy as np


class EpisodeTracker:
    def __init__(self):
        self.metrics = defaultdict(list)

    def log(self, key, value):
        self.metrics[key].append(value)

    def mean(self, key):
        values = self.metrics[key]
        if len(values) == 0:
            return 0
        return np.mean(values)

    def summary(self):
        print("\n=== EVAL SUMMARY ===")
        for key in self.metrics:
            print(f"{key}: {self.mean(key):.4f}")
