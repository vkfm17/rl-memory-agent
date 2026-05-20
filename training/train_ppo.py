import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env.conversation_env import ConversationEnv
from training.config import TrainingConfig


def train(config: TrainingConfig) -> PPO:
    os.makedirs(config.run_dir, exist_ok=True)

    env = ConversationEnv(
        max_memory=config.max_memory,
        base_distractors=config.base_distractors,
        curriculum_checkpoint=config.curriculum_checkpoint,
        tb_log_dir=config.tb_log_dir,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        policy_kwargs=dict(net_arch=config.net_arch),
        tensorboard_log=config.tb_log_dir,
    )

    model.learn(total_timesteps=config.total_steps)
    model.save(config.model_path)

    return model


if __name__ == "__main__":
    train(TrainingConfig())
