"""
Scaling experiments for the memory RL agent.

Runs a grid of training configs, evaluates each against PPO + baselines,
and writes results/sweep_summary.csv.

Usage:
    python -m training.sweep
    python -m training.sweep --group memory   # run only one experiment group
"""

import argparse
import json
import os

import pandas as pd
from stable_baselines3 import PPO

from env.conversation_env import ConversationEnv
from evals.benchmark import benchmark_all, benchmark_ppo
from training.config import TrainingConfig
from training.train_ppo import train

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

EXPERIMENTS: dict[str, list[TrainingConfig]] = {
    "memory": [
        TrainingConfig(run_name="mem_3",  max_memory=3),
        TrainingConfig(run_name="mem_5",  max_memory=5),   # baseline
        TrainingConfig(run_name="mem_10", max_memory=10),
        TrainingConfig(run_name="mem_20", max_memory=20),
    ],
    "distractors": [
        TrainingConfig(run_name="dist_5",  base_distractors=5),
        TrainingConfig(run_name="dist_10", base_distractors=10),  # baseline
        TrainingConfig(run_name="dist_20", base_distractors=20),
        TrainingConfig(run_name="dist_50", base_distractors=50),
    ],
    "arch": [
        TrainingConfig(run_name="arch_small",   net_arch=[128, 128]),
        TrainingConfig(run_name="arch_default", net_arch=[256, 256]),  # baseline
        TrainingConfig(run_name="arch_large",   net_arch=[512, 512, 512]),
    ],
}


# ---------------------------------------------------------------------------
# Run one config: train + eval
# ---------------------------------------------------------------------------

def run_experiment(config: TrainingConfig, eval_episodes: int = 100) -> dict:
    print(f"\n{'='*60}")
    print(f"  Run: {config.run_name}")
    print(f"  max_memory={config.max_memory}  base_distractors={config.base_distractors}")
    print(f"  net_arch={config.net_arch}  total_steps={config.total_steps}")
    print(f"{'='*60}")

    # Train
    model = train(config)

    # Eval env uses same capacity/distractor params, fixed at level 1 for
    # consistent comparison across all configs.
    eval_env = ConversationEnv(
        max_memory=config.max_memory,
        base_distractors=config.base_distractors,
        curriculum_level=1,
        tb_log_dir=os.path.join(config.tb_log_dir, "eval"),
    )

    ppo_df = benchmark_ppo(eval_env, model, episodes=eval_episodes)
    baseline_dfs = benchmark_all(eval_env, episodes=eval_episodes)

    # Build summary row
    row = {"run": config.run_name, "agent": "ppo"}
    row.update(ppo_df.mean(numeric_only=True).to_dict())
    rows = [row]

    for name, df in baseline_dfs.items():
        r = {"run": config.run_name, "agent": name}
        r.update(df.mean(numeric_only=True).to_dict())
        rows.append(r)

    # Persist per-run summary
    summary_path = os.path.join(config.run_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"  → saved {summary_path}")
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        choices=list(EXPERIMENTS.keys()),
        default=None,
        help="Run only one experiment group (default: all)",
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    args = parser.parse_args()

    groups = {args.group: EXPERIMENTS[args.group]} if args.group else EXPERIMENTS

    all_rows = []
    for group_name, configs in groups.items():
        print(f"\n\n{'#'*60}")
        print(f"  GROUP: {group_name}")
        print(f"{'#'*60}")
        for config in configs:
            rows = run_experiment(config, eval_episodes=args.eval_episodes)
            all_rows.extend(rows)

    summary_df = pd.DataFrame(all_rows)
    out_path = os.path.join(_REPO_ROOT, "results", "sweep_summary.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary_df.to_csv(out_path, index=False)

    print(f"\n\n{'='*60}")
    print("SWEEP COMPLETE")
    print(f"{'='*60}")
    print(summary_df.to_string())
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
