from stable_baselines3 import PPO
import numpy as np
from ot2_gym_wrapper_V2 import OT2Env  # Updated wrapper name

# Path to your trained model
model_path = r"C:\Users\jimal\OneDrive - BUas\Pictures\CkearML\ClearMl\model.zip"  # Update with your model path

def evaluate_model_fixed_goal(model_path, env, fixed_goal, num_episodes=10):
    """
    Evaluates a single model on a fixed goal position.
    """
    model = PPO.load(model_path)  # Load the trained model
    accuracies, steps_taken = [], []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        env.goal_position = fixed_goal  # Override the goal position

        done, truncated = False, False
        step_count = 0

        while not done and not truncated:
            action, _ = model.predict(obs)  # Predict action using the model
            obs, reward, done, truncated, info = env.step(action)  # Take action in the environment
            step_count += 1

            # Calculate distance to the fixed goal
            distance_to_goal = np.linalg.norm(obs[:3] - fixed_goal)

            if done or truncated:
                accuracies.append(distance_to_goal)
                steps_taken.append(step_count)

    # Compute average and standard deviation of results
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_steps = np.mean(steps_taken)

    return avg_accuracy, std_accuracy, avg_steps

def main():
    """
    Main function to evaluate the model.
    """
    # Define a fixed goal position
    fixed_goal_position = np.array([0.1, 0.1, 0.2], dtype=np.float32)

    # Create the custom environment
    env = OT2Env(render=True, max_steps=1000)

    # Evaluate the model
    print(f"Evaluating model: {model_path} with fixed goal {fixed_goal_position}")
    avg_accuracy, std_accuracy, avg_steps = evaluate_model_fixed_goal(
        model_path, env, fixed_goal_position
    )

    # Print evaluation results
    print(
        f"Model: {model_path} | Avg Accuracy: {avg_accuracy:.4f} m | "
        f"Std Dev: {std_accuracy:.4f} m | Avg Steps: {avg_steps:.2f}"
    )

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
