from stable_baselines3 import PPO

from env.conversation_env import ConversationEnv

if __name__ == "__main__":
    env = ConversationEnv()
    model = PPO.load("ppo_memory_agent")

    episodes = 100
    total_reward = 0
    successes = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

        total_reward += ep_reward
        if ep_reward > 0:
            successes += 1

    print(f"Average Reward: {total_reward / episodes}")
    print(f"Success Rate: {successes / episodes:.2f}")
