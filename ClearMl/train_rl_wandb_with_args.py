os.system('pip install --upgrade wandb pydantic')

import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
import gymnasium as gym
from clearml import Task

# Initialize ClearML Task
task = Task.init(
    project_name='Mentor Group J/Group 2/Musaed225739',  
    task_name='Experiment2'                              # Unique task name
)

# Set Docker image and queue for ClearML
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")  # Set to GPU queue

# Load the API key for W&B
os.environ['WANDB_API_KEY'] = 'da30da01fd3e0628233dc693966e900058ff208e'

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate for the optimizer")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps to collect per update")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs per update")
parser.add_argument("--time_steps", type=int, default=10000, help="Total number of timesteps for training")
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
    env = gym.make('Pendulum-v1', g=9.81)  # Replace with custom environment if needed

    # Initialize the PPO model with unique hyperparameters
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        gamma=0.95,  # Custom gamma value for discounting
        clip_range=0.2,  # Custom clip range
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

