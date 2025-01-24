from simple_pid import PID
import numpy as np
from ot2_gym_wrapper_V2 import OT2Env
import random

class PIDRandomPositionSimulation:
    def __init__(self):
        # Initialize the OT2 Gym Environment
        self.env = OT2Env(render=True)
        self.dt = 0.1  # Time step

        # Initialize PID controllers for each axis
        self.pid_x = PID(10, 5.5,3.0)
        self.pid_y = PID(10, 5.5, 3.0)
        self.pid_z = PID(10, 5.5, 3.0)

        # Set PID output limits (velocity bounds)
        for pid in [self.pid_x, self.pid_y, self.pid_z]:
            pid.output_limits = (-1.0, 1.0)

        # Define bounds of the working envelope
        self.bounds = {
            "x": [0.2531, -0.1872],
            "y": [0.2201, -0.1711],
            "z": [0.2896, 0.1691]
        }

        self.positions = []
        self.distances = []
    
    def move_to_random_position(self):
        # Generate a random target position within the bounds
        target = [
            random.uniform(self.bounds["x"][1], self.bounds["x"][0]),
            random.uniform(self.bounds["y"][1], self.bounds["y"][0]),
            random.uniform(self.bounds["z"][1], self.bounds["z"][0]),
        ]

        print(f"Moving to Random Position: {target}")

        # Set PID setpoints for the target position
        self.pid_x.setpoint = target[0]
        self.pid_y.setpoint = target[1]
        self.pid_z.setpoint = target[2]

        # Reset environment and get initial position
        observation, _ = self.env.reset()
        pipette_position = observation[:3]

        # Reset tracking
        distances = []  # Track distances to target
        last_position = None

        # Use a for loop instead of while
        for step in range(1000):
            # Compute PID-controlled velocities
            vel_x = self.pid_x(pipette_position[0])
            vel_y = self.pid_y(pipette_position[1])
            vel_z = self.pid_z(pipette_position[2])

            # Apply velocities to the simulation
            action = [vel_x, vel_y, vel_z]
            observation, _, _, _, _ = self.env.step(action)

            # Update pipette position
            pipette_position = observation[:3]

            # Calculate distance to target
            distance_to_target = np.linalg.norm(pipette_position - target)
            distances.append(distance_to_target)

            if distance_to_target < 0.001:  # 1 mm accuracy
                print(f"Target reached in {step+1} steps.")
                break

            # Record position
            last_position = pipette_position

        # Log accuracy
        print(f"Final Position: {last_position}, Distance to Target: {distances[-1]:.4f} m")

        return step, target, distances[-1]

    def simulate(self, num_positions=5):
        results = []
        for i in range(num_positions):
            steps, target, final_distance = self.move_to_random_position()
            status = "Success" if final_distance < 0.001 else "Failure"
            print(f"Position {i+1} | Steps: {steps} | Target: {target} | Status: {status} | Final Distance: {final_distance:.4f} m")
            results.append((target, status, final_distance))

        return results

    def close(self):
        self.env.close()

if __name__ == "__main__":
    simulation = PIDRandomPositionSimulation()
    results = simulation.simulate(num_positions=5)
    simulation.close()

    print("\nSimulation Results:")
    for i, (target, status, final_distance) in enumerate(results):
        print(f"Position {i+1}: Target {target} | Status: {status} | Final Distance: {final_distance:.4f} m")
