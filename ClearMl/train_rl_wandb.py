import os

# Disable symlinks to avoid permission issues on Windows
os.environ["WANDB_DISABLE_SYMLINK"] = "true"

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
import gym

try:
    # Initialize W&B
    run = wandb.init(
        project="sb3_pendulum_demo",
        sync_tensorboard=True,
        settings=wandb.Settings(init_timeout=300)  # Increased timeout
    )

    # Create environment
    env = gym.make('Pendulum-v1', g=9.81)

    # Define timesteps per iteration and total iterations
    time_steps = 2000  # Save model every 2000 steps
    total_iterations = 5  # Total training steps = time_steps * total_iterations

    # Initialize PPO model with TensorBoard logging
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f"runs/{run.id}")

    # Ensure the directory for saving models exists
    os.makedirs(f"models/{run.id}", exist_ok=True)

    # Create W&B callback
    wandb_callback = WandbCallback(
        verbose=2,
    )

    # Train the model incrementally and save after each iteration
    for i in range(total_iterations):
        print(f"Starting iteration {i + 1}/{total_iterations}")
        model.learn(
            total_timesteps=time_steps,
            callback=wandb_callback,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name=f"runs/{run.id}",
        )
        # Save the model periodically
        model.save(f"models/{run.id}/model_step_{(i + 1) * time_steps}.zip")
        print(f"Model saved after iteration {i + 1}")

    print("Incremental training completed successfully!")

except wandb.errors.CommError as e:
    print(f"W&B Communication Error: {e}")
    print("Consider running in offline mode if this persists.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # Ensure W&B run is closed properly
    if "run" in locals() and run is not None:
        run.finish()
