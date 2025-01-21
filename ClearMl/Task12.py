import time
import numpy as np
import matplotlib.pyplot as plt
import json
from ot2_gym_wrapper_V2 import OT2Env  # Your simulation environment
from simple_pid import PID
from stable_baselines3 import PPO  # Your trained RL model

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.pid_x = PID(Kp, Ki, Kd)
        self.pid_y = PID(Kp, Ki, Kd)
        self.pid_z = PID(Kp, Ki, Kd)

        # Set PID output limits (velocity bounds)
        for pid in [self.pid_x, self.pid_y, self.pid_z]:
            pid.output_limits = (-1.0, 1.0)

    def calculate_action(self, current_position, target_position):
        self.pid_x.setpoint = target_position[0]
        self.pid_y.setpoint = target_position[1]
        self.pid_z.setpoint = target_position[2]

        vel_x = self.pid_x(current_position[0])
        vel_y = self.pid_y(current_position[1])
        vel_z = self.pid_z(current_position[2])

        return [vel_x, vel_y, vel_z]

def benchmark_controller(env, controller, goal_positions, controller_type):
    """
    Benchmarks the performance of a given controller (PID or RL).

    Args:
        env: The simulation environment.
        controller: The controller to benchmark (PID or RL).
        goal_positions: List of goal positions for the benchmarking process.
        controller_type: Type of controller ("PID" or "RL").

    Returns:
        results: A list of dictionaries containing metrics for each trial.
    """
    results = []

    for goal in goal_positions:
        obs, _ = env.reset()
        env.goal_position = goal  # Set the goal position
        start_time = time.time()  # Start the timer

        while True:
            # Determine the action based on the controller type
            if controller_type == "RL":
                action, _ = controller.predict(obs, deterministic=True)
            elif controller_type == "PID":
                action = controller.calculate_action(obs[:3], goal)

            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Extract current position and calculate distance to goal
            current_pos = obs[:3]
            distance_to_goal = np.linalg.norm(np.array(goal) - np.array(current_pos))

            # Check if the goal is reached
            if distance_to_goal < 0.001:  # Accuracy threshold (1 mm)
                break

            # Check if the episode is terminated or truncated
            if terminated or truncated:
                obs, _ = env.reset()
                break

        # Record elapsed time
        elapsed_time = time.time() - start_time
        # Append results
        results.append({
            "goal": goal,
            "final_position": current_pos.tolist(),
            "error": distance_to_goal,
            "time": elapsed_time
        })

    return results

def visualize_results(pid_results, rl_results):
    """
    Visualizes the benchmarking results for speed and accuracy.

    Args:
        pid_results: Benchmarking results for the PID controller.
        rl_results: Benchmarking results for the RL controller.
    """
    # Extract average times and errors
    controllers = ["PID", "RL"]
    avg_times = [
        np.mean([r["time"] for r in pid_results]),
        np.mean([r["time"] for r in rl_results])
    ]
    avg_errors = [
        np.mean([r["error"] for r in pid_results]),
        np.mean([r["error"] for r in rl_results])
    ]

    # Plot speed comparison
    plt.figure()
    plt.bar(controllers, avg_times, color=['blue', 'orange'])
    plt.title("Speed Comparison")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.savefig("speed_comparison.png")
    plt.show()

    # Plot accuracy comparison
    plt.figure()
    plt.bar(controllers, avg_errors, color=['blue', 'orange'])
    plt.title("Accuracy Comparison")
    plt.ylabel("Error (meters)")
    plt.grid(True)
    plt.savefig("accuracy_comparison.png")
    plt.show()

if __name__ == "__main__":
    # Initialize the simulation environment
    env = OT2Env(render=False)  # Your Gym environment

    # Initialize the PID controller with tuned gains
    Kp, Ki, Kd = 10.0, 5.5, 3.0  # Tuned PID gains
    pid_controller = PIDController(Kp, Ki, Kd)
    print(f"PID Controller initialized with Kp={Kp}, Ki={Ki}, Kd={Kd}")

    # Load the trained RL model
    rl_model_path = "models/best_rl_model.zip"  # Adjust path as needed
    if not os.path.exists(rl_model_path):
        raise FileNotFoundError(f"Trained RL model not found at {rl_model_path}. Train the model before benchmarking.")

    rl_controller = PPO.load(rl_model_path)
    print(f"RL Controller loaded from {rl_model_path}")

    # Define goal positions for the benchmarking
    goal_positions = [
        [0.1, 0.1, 0.05], 
        [0.15, 0.1, 0.07], 
        [0.05, 0.15, 0.06]
    ]

    # Benchmark the PID controller
    print("Benchmarking PID controller...")
    pid_results = benchmark_controller(env, pid_controller, goal_positions, "PID")

    # Benchmark the RL controller
    print("Benchmarking RL controller...")
    rl_results = benchmark_controller(env, rl_controller, goal_positions, "RL")

    # Save results to JSON files
    with open("pid_results.json", "w") as f:
        json.dump(pid_results, f, indent=4)

    with open("rl_results.json", "w") as f:
        json.dump(rl_results, f, indent=4)

    print("Results saved successfully!")

    # Visualize results
    visualize_results(pid_results, rl_results)
