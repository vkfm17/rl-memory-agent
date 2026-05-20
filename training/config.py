import os
from dataclasses import dataclass, field

from constants import MAX_MEMORY, TOTAL_STEPS

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


@dataclass
class TrainingConfig:
    run_name: str = "default"
    max_memory: int = MAX_MEMORY
    base_distractors: int = 10
    total_steps: int = TOTAL_STEPS
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    learning_rate: float = 1e-4
    n_steps: int = 128
    batch_size: int = 32

    @property
    def run_dir(self) -> str:
        return os.path.join(_REPO_ROOT, "results", self.run_name)

    @property
    def model_path(self) -> str:
        return os.path.join(self.run_dir, "model")

    @property
    def tb_log_dir(self) -> str:
        return os.path.join(_REPO_ROOT, "tensorboard", self.run_name)

    @property
    def curriculum_checkpoint(self) -> str:
        return os.path.join(self.run_dir, "curriculum_state.json")
