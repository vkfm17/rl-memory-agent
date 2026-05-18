from typing import Any

from typedefs import MemorySnapshot


class MemoryTracker:
    def __init__(self):
        self.timeline: list[MemorySnapshot] = []

    def log_step(
        self,
        step: int,
        message: str,
        memory: list[dict[str, Any]],
        answer: str,
    ):

        # Deep copy memory state (important!)
        memory_copy = [
            {
                "message": m["message"],
                "step": m["step"],
                "fact_value": m.get("fact_value"),
            }
            for m in memory
        ]

        self.timeline.append(
            MemorySnapshot(
                step=step,
                message=message,
                memory=memory_copy,
                answer=answer,
            )
        )

    # -----------------------------
    # Did memory contain correct fact?
    # -----------------------------
    def memory_contains_truth(self, snapshot):

        for m in snapshot.memory:
            if m.get("fact_value") == snapshot.answer:
                return True

        return False

    # -----------------------------
    # Was memory contradictory?
    # -----------------------------
    def has_contradiction(self, snapshot):

        facts = [
            m.get("fact_value")
            for m in snapshot.memory
            if m.get("fact_value") is not None
        ]

        return len(set(facts)) > 1

    # -----------------------------
    # Memory size over time
    # -----------------------------
    def memory_sizes(self):

        return [len(s.memory) for s in self.timeline]

    # -----------------------------
    # Truth alignment over time
    # -----------------------------
    def correctness_over_time(self):

        return [self.memory_contains_truth(s) for s in self.timeline]

    # -----------------------------
    # Contradiction timeline
    # -----------------------------
    def contradiction_over_time(self):

        return [self.has_contradiction(s) for s in self.timeline]
