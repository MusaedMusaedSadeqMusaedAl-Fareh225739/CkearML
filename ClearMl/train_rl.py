import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from wandb.integration.sb3 import WandbCallback
import wandb
import time

# Initialize Weights and Biases for tracking
wandb.init(
    project="opentrons-rl",
    config={
        "policy_type": "MlpPolicy",
        "total_timesteps": 500000,  # Testing with 500k timesteps
        "env_name": "OT2Env",
    },
)

# Import the custom Gymnasium environment
from ot2_gym_wrapper import OT2Env

# Create the environment
env = DummyVecEnv([lambda: OT2Env(render=False)])

# Define the PPO model with adjusted hyperparameters
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.00005,  # Slightly reduced learning rate for stability
    gamma=0.999,  # Focus on long-term rewards
    n_steps=4096,  # Medium batch size for balanced updates
    ent_coef=0.005,  # Reduce entropy to favor exploitation over exploration
    vf_coef=0.7,  # Increase weight for value predictions
    max_grad_norm=0.7,  # Adjust gradient clipping for smoother training
    verbose=1,
)

# Add a custom reward function with time penalty in the environment
class CustomOT2Env(OT2Env):
    def step(self, action):
        # Use the base step logic
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Add a time penalty to the reward
        time_penalty = -0.01 * self.steps  # Penalize longer episodes
        reward += time_penalty
        
        return observation, reward, terminated, truncated, info

# Replace environment with the custom version
env = DummyVecEnv([lambda: CustomOT2Env(render=False)])

# Set up evaluation and callbacks
eval_callback = EvalCallback(
    env,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=10000,  # Evaluate every 10k timesteps
    deterministic=True,
    render=False,
)

# Log training start time
start_time = time.time()

print("Training started...")
# Train the model
model.learn(
    total_timesteps=500000,  # Testing with 500k timesteps
    callback=[eval_callback, WandbCallback()],
)

# Log training end time
end_time = time.time()
training_time = end_time - start_time
print(f"Training completed in {training_time:.2f} seconds (~{training_time/3600:.2f} hours).")

# Save the trained model
model.save("models/ppo_ot2_test")
print("Model saved successfully!")

# Finish Weights and Biases session
wandb.finish()
