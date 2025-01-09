from stable_baselines3 import PPO
import gym
import time

env = gym.make('Pendulum-v1',g=9.81)

model = PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=10000, progress_bar=True)
