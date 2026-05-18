from env.conversation_env import ConversationEnv
from evals.plots import plot_memory_dynamics

env = ConversationEnv()
obs, _ = env.reset()

done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)

plot_memory_dynamics(env.tracker)
