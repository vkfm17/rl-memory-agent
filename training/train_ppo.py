from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from constants import TB_LOG_DIR, TOTAL_STEPS
from env.conversation_env import ConversationEnv

env = ConversationEnv()
check_env(env)

policy_kwargs = dict(net_arch=[256, 256])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=128,
    batch_size=32,
    policy_kwargs=policy_kwargs,
    tensorboard_log=TB_LOG_DIR,
)

model.learn(total_timesteps=TOTAL_STEPS)
model.save("ppo_memory_agent")
