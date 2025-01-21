import os
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
import gymnasium as gym
from clearml import Task
from dotenv import load_dotenv

# Add the ClearML directory to the Python path
import sys
sys.path.append(os.path.abspath("/path/to/CkearML/ClearMl"))  # Update the path

# Import the custom environment
from ot2_gym_wrapper_V2 import OT2Env

# Load environment variables
load_dotenv()

# Initialize ClearML Task
task = Task.init(
    project_name='Mentor Group J/Group 2/Musaed225739',  # Your project name
    task_name='Experiment2'  # Your task name
)

# Set Docker image and queue for ClearML
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default", clone=False)

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps to collect per update")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs per update")
parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor for rewards")
parser.add_argument("--clip_range", type=float, default=0.1, help="Clipping parameter for PPO")
parser.add_argument("--value_coefficient", type=float, default=0.5, help="Coefficient for value function loss")
parser.add_argument("--time_steps", type=int, default=5000000, help="Total number of timesteps for training")
args = parser.parse_args()

# Set W&B API Key
wandb_api_key = os.getenv('WANDB_API_KEY', '')
if not wandb_api_key:
    raise ValueError("WANDB_API_KEY environment variable not set.")

os.environ['WANDB_API_KEY'] = wandb_api_key

# Initialize W&B
run = wandb.init(
    project="task11",
    sync_tensorboard=True,
    settings=wandb.Settings(init_timeout=300)
)

# Custom Environment
env = OT2Env()

# Initialize PPO model
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    clip_range=args.clip_range,
    vf_coef=args.value_coefficient,
    tensorboard_log=f"runs/{run.id}"
)

# Ensure model directory exists
save_path = r"C:\Users\jimal\OneDrive - BUas\Pictures\CkearML\ClearMl\models"
os.makedirs(save_path, exist_ok=True)

# Create W&B callback
wandb_callback = WandbCallback(
    model_save_freq=100000,
    model_save_path=save_path,
    verbose=2
)

# Train the model incrementally
try:
    for i in range(args.time_steps // args.n_steps):
        print(f"Starting iteration {i + 1}")
        model.learn(
            total_timesteps=args.n_steps,
            callback=wandb_callback,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name=f"runs/{run.id}",
        )
        model_save_file = f"{save_path}/model_step_{(i + 1) * args.n_steps}.zip"
        model.save(model_save_file)
        print(f"Model saved successfully to: {model_save_file} after iteration {i + 1}")
    print("Training completed successfully!")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    run.finish()
