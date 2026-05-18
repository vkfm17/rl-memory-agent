import pandas as pd
from stable_baselines3 import PPO

from env.conversation_env import ConversationEnv
from evals.metrics import summarize_episode
from memory.policies import (
    KeepLastKPolicy,
    RandomPolicy,
    ReplaceSimilarPolicy,
)


# -----------------------------
# Run ONE episode (policy or PPO)
# -----------------------------
def run_episode(env, policy=None, model=None):

    obs, _ = env.reset()

    done = False
    total_reward = 0

    while not done:
        current_msg = env.conversation[env.current_step]

        # -----------------------------
        # POLICY MODE (baseline)
        # -----------------------------
        if policy is not None:
            action = policy.act(
                current_msg,
                env.memory,
            )

        # -----------------------------
        # PPO MODE
        # -----------------------------
        else:
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

        obs, reward, done, _, info = env.step(action)

        total_reward += reward

    # -----------------------------
    # FINAL METRICS (single source of truth)
    # -----------------------------
    metrics = summarize_episode(
        memory=env.memory,
        conversation=env.conversation,
        reward=total_reward,
        correct=info.get("correct", 0),
        action_history=env.action_history,
        correct_answer=env.answer,
    )

    return metrics


# -----------------------------
# Benchmark single policy
# -----------------------------
def benchmark_policy(env, policy, episodes=100):

    results = []

    for _ in range(episodes):
        metrics = run_episode(
            env=env,
            policy=policy,
        )

        results.append(metrics)

    df = pd.DataFrame(results)

    print("\n=== POLICY BENCHMARK ===")
    print(df.mean(numeric_only=True))

    return df


# -----------------------------
# Benchmark PPO
# -----------------------------
def benchmark_ppo(env, model, episodes=100):

    results = []

    for _ in range(episodes):
        metrics = run_episode(
            env=env,
            model=model,
        )

        results.append(metrics)

    df = pd.DataFrame(results)

    print("\n=== PPO BENCHMARK ===")
    print(df.mean(numeric_only=True))

    return df


# -----------------------------
# Benchmark ALL baselines
# -----------------------------
def benchmark_all(env, episodes=100):

    baselines = {
        "keep_last_k": KeepLastKPolicy(),
        "random": RandomPolicy(),
        # "summarize_oldest": SummarizeOldestPolicy(),
        "replace_similar": ReplaceSimilarPolicy(),
    }

    results = {}

    for name, policy in baselines.items():
        print(f"\nRunning baseline: {name}")

        df = benchmark_policy(
            env,
            policy,
            episodes,
        )

        results[name] = df

    return results


# -----------------------------
# Main entry
# -----------------------------
if __name__ == "__main__":
    env = ConversationEnv()

    # -------------------------
    # Load PPO model
    # -------------------------
    model = PPO.load("ppo_memory_agent")

    episodes = 100

    # -------------------------
    # PPO evaluation
    # -------------------------
    print("\n======================")
    print("PPO EVALUATION")
    print("======================")

    ppo_df = benchmark_ppo(
        env,
        model,
        episodes,
    )

    # -------------------------
    # Baselines
    # -------------------------
    print("\n======================")
    print("BASELINE EVALUATION")
    print("======================")

    baseline_results = benchmark_all(
        env,
        episodes,
    )

    # -------------------------
    # FINAL COMPARISON TABLE
    # -------------------------
    print("\n======================")
    print("FINAL SUMMARY")
    print("======================")

    summary = {
        "ppo": ppo_df.mean(numeric_only=True),
    }

    for name, df in baseline_results.items():
        summary[name] = df.mean(numeric_only=True)

    summary_df = pd.DataFrame(summary)

    print(summary_df)
