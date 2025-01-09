import os
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
import gym
from clearml import Task

# Initialize ClearML Task
task = Task.init(
    project_name='Mentor Group J/Group 2/Musaed225739',  # Replace with your project name
    task_name='CustomExperiment3'                       # Unique task name
)

# Set Docker image and queue for ClearML
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")  # Change queue name if GPU is needed

# Load the API key for W&B
os.environ['WANDB_API_KEY'] = 'da30da01fd3e0628233dc693966e900058ff208e'

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
parser.add_argument("--n_steps", type=int, default=1024, help="Number of steps to collect per update")
parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs per update")
parser.add_argument("--vf_coef", type=float, default=0.4, help="Value function coefficient")
parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clip range")
parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
parser.add_argument("--time_steps", type=int, default=20000, help="Total number of timesteps for training")
args = parser.parse_args()

# Disable symlinks to avoid permission issues on Windows
os.environ["WANDB_DISABLE_SYMLINK"] = "true"

try:
    # Initialize W&B
    run = wandb.init(
        project="sb3_pendulum_experiment",
        sync_tensorboard=True,
        settings=wandb.Settings(init_timeout=300)  # Increased timeout
    )

    # Create the environment
    env = gym.make('Pendulum-v1', g=9.81)

    # Initialize the PPO model with command-line hyperparameters
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
        vf_coef=args.vf_coef,
        tensorboard_log=f"runs/{run.id}"
    )

    # Ensure the directory for saving models exists
    save_path = f"models/{run.id}"
    os.makedirs(save_path, exist_ok=True)

    # Create W&B callback
    wandb_callback = WandbCallback(
        model_save_freq=5000, 
        model_save_path=save_path, 
        verbose=2
    )

    # Train the model incrementally
    for i in range(args.time_steps // args.n_steps):
        print(f"Starting iteration {i + 1}")
        model.learn(
            total_timesteps=args.n_steps,
            callback=wandb_callback,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name=f"runs/{run.id}",
        )
        # Save the model periodically
        model.save(f"{save_path}/model_step_{(i + 1) * args.n_steps}.zip")
        print(f"Model saved after iteration {i + 1}")

    print("Training completed successfully!")

except wandb.errors.CommError as e:
    print(f"W&B Communication Error: {e}")
    print("Consider running in offline mode if this persists.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # Ensure W&B run is closed properly
    if "run" in locals() and run is not None:
        run.finish()

