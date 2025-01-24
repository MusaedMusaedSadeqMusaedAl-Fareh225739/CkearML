from stable_baselines3 import PPO
from simple_pid import PID
import numpy as np
from ot2_gym_wrapper_V2 import OT2Env
import random
import matplotlib.pyplot as plt

def evaluate_rl_model(model_path, env, fixed_goal, num_episodes=10):
    """
    Evaluates the RL model on a fixed goal position.
    """
    model = PPO.load(model_path)
    accuracies, steps_taken = [], []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        env.goal_position = fixed_goal

        done, truncated = False, False
        step_count = 0

        while not done and not truncated:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1

            # Calculate distance to the fixed goal
            distance_to_goal = np.linalg.norm(obs[:3] - fixed_goal)

            if done or truncated:
                accuracies.append(distance_to_goal)
                steps_taken.append(step_count)

    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_steps = np.mean(steps_taken)

    return avg_accuracy, std_accuracy, avg_steps


def evaluate_pid_controller(env, bounds, num_positions=10):
    """
    Evaluates the PID controller by moving to random positions.
    """
    pid_x = PID(10, 5.5, 3.0)
    pid_y = PID(10, 5.5, 3.0)
    pid_z = PID(10, 5.5, 3.0)

    for pid in [pid_x, pid_y, pid_z]:
        pid.output_limits = (-1.0, 1.0)

    accuracies, steps_taken = [], []

    for _ in range(num_positions):
        target = [
            random.uniform(bounds["x"][1], bounds["x"][0]),
            random.uniform(bounds["y"][1], bounds["y"][0]),
            random.uniform(bounds["z"][1], bounds["z"][0]),
        ]

        pid_x.setpoint = target[0]
        pid_y.setpoint = target[1]
        pid_z.setpoint = target[2]

        obs, _ = env.reset()
        pipette_position = obs[:3]

        step_count = 0
        while step_count < 1000:
            vel_x = pid_x(pipette_position[0])
            vel_y = pid_y(pipette_position[1])
            vel_z = pid_z(pipette_position[2])

            action = [vel_x, vel_y, vel_z]
            obs, _, _, _, _ = env.step(action)

            pipette_position = obs[:3]
            distance_to_target = np.linalg.norm(pipette_position - target)
            step_count += 1

            if distance_to_target < 0.001:
                break

        accuracies.append(distance_to_target)
        steps_taken.append(step_count)

    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_steps = np.mean(steps_taken)

    return avg_accuracy, std_accuracy, avg_steps


def visualize_results(rl_results, pid_results):
    """
    Visualizes the comparison between RL and PID controllers.
    """
    # Extract metrics
    methods = ["RL Model", "PID Controller"]
    avg_accuracies = [rl_results["avg_accuracy"], pid_results["avg_accuracy"]]
    std_accuracies = [rl_results["std_dev"], pid_results["std_dev"]]
    avg_steps = [rl_results["avg_steps"], pid_results["avg_steps"]]

    # Create a bar chart for average accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(methods, avg_accuracies, color=['blue', 'orange'])
    plt.ylabel("Avg Accuracy (m)")
    plt.title("Average Accuracy Comparison (RL vs PID)")
    for i, value in enumerate(avg_accuracies):
        plt.text(i, value, f"{value:.4f}", ha='center', va='bottom')
    plt.savefig("avg_accuracy_comparison.png")
    plt.show()

    # Create a bar chart for standard deviation
    plt.figure(figsize=(10, 6))
    plt.bar(methods, std_accuracies, color=['blue', 'orange'])
    plt.ylabel("Std Deviation (m)")
    plt.title("Standard Deviation of Accuracy (RL vs PID)")
    for i, value in enumerate(std_accuracies):
        plt.text(i, value, f"{value:.4f}", ha='center', va='bottom')
    plt.savefig("std_dev_comparison.png")
    plt.show()

    # Create a bar chart for average steps
    plt.figure(figsize=(10, 6))
    plt.bar(methods, avg_steps, color=['blue', 'orange'])
    plt.ylabel("Avg Steps")
    plt.title("Average Steps to Reach Target (RL vs PID)")
    for i, value in enumerate(avg_steps):
        plt.text(i, value, f"{value:.2f}", ha='center', va='bottom')
    plt.savefig("avg_steps_comparison.png")
    plt.show()


def compare_rl_vs_pid():
    """
    Compares the performance of RL and PID controllers.
    """
    # Environment setup
    env = OT2Env(render=True, max_steps=1000)
    bounds = {
        "x": [0.2531, -0.1872],
        "y": [0.2201, -0.1711],
        "z": [0.2896, 0.1691],
    }

    # RL model evaluation
    model_path = r"C:\\Users\\jimal\\OneDrive - BUas\\Pictures\\CkearML\\ClearMl\\model.zip"
    fixed_goal = np.array([0.1, 0.1, 0.2], dtype=np.float32)

    print("Evaluating RL Model...")
    rl_avg_acc, rl_std_acc, rl_avg_steps = evaluate_rl_model(model_path, env, fixed_goal)
    print(f"RL Model | Avg Accuracy: {rl_avg_acc:.4f} m | Std Dev: {rl_std_acc:.4f} m | Avg Steps: {rl_avg_steps:.2f}")

    # PID controller evaluation
    print("Evaluating PID Controller...")
    pid_avg_acc, pid_std_acc, pid_avg_steps = evaluate_pid_controller(env, bounds)
    print(f"PID Controller | Avg Accuracy: {pid_avg_acc:.4f} m | Std Dev: {pid_std_acc:.4f} m | Avg Steps: {pid_avg_steps:.2f}")

    # Store results
    rl_results = {"avg_accuracy": rl_avg_acc, "std_dev": rl_std_acc, "avg_steps": rl_avg_steps}
    pid_results = {"avg_accuracy": pid_avg_acc, "std_dev": pid_std_acc, "avg_steps": pid_avg_steps}

    # Visualize the results
    visualize_results(rl_results, pid_results)

    env.close()


if __name__ == "__main__":
    compare_rl_vs_pid()
