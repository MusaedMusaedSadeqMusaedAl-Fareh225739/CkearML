import os
import argparse
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from clearml import Task
from ot2_gym_wrapper_baseline import OT2Env  # Replace with your wrapper

# ClearML Task Initialization
task = Task.init(
    project_name='Mentor Group J/Group 2/Musaed225739',
    task_name='Experiment2'
)
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="gpu")

# W&B API Key
os.environ['WANDB_API_KEY'] = 'da30da01fd3e0628233dc693966e900058ff208e'
os.environ["WANDB_SYMLINK"] = "false"

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.00015)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_steps", type=int, default=1024)
parser.add_argument("--n_epochs", type=int, default=5)
parser.add_argument("--time_steps", type=int, default=20000)
args = parser.parse_args()

# Environment and Model
env = OT2Env()  # Pre-wrapped environment
run = wandb.init(project="sb3_experiment", sync_tensorboard=True)
save_path = f"models/{run.id}"
os.makedirs(save_path, exist_ok=True)

model = PPO(
    'MlpPolicy', env, verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    tensorboard_log=f"runs/{run.id}"
)

# Wandb Callback
wandb_callback = WandbCallback(
    model_save_freq=5000,
    model_save_path=save_path,
    verbose=2
)

# Training
model.learn(total_timesteps=args.time_steps, callback=wandb_callback, progress_bar=True)
model.save(f"{save_path}/final_model.zip")
