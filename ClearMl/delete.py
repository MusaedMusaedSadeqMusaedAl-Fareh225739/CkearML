import numpy as np
from ot2_gym_wrapper import OT2Env
import time

class PIDController:
    def __init__(self, kp, ki, kd, setpoint, sample_time=0.01, output_limits=(-1.0, 1.0), debug=False):
        """
        Initialize the PID controllers for x, y, and z axes.

        Args:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
            setpoint (list): Target position for the axes [x, y, z].
            sample_time (float): Time interval between control updates.
            output_limits (tuple): Clamp the output to (min, max).
            debug (bool): Enable debugging output.
        """
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.setpoint = setpoint
        self.sample_time = sample_time
        self.output_limits = output_limits
        self.debug = debug

        # Internal state for each axis
        self.last_time = None
        self.integral = [0.0, 0.0, 0.0]  # Integral terms for x, y, z
        self.last_error = [0.0, 0.0, 0.0]  # Last errors for x, y, z
        self.last_output = [0.0, 0.0, 0.0]  # Last outputs for x, y, z

    def compute_control(self, current_position):
        """
        Compute the control outputs for the PID controllers.

        Args:
            current_position (list): Current position of the pipette [x, y, z].

        Returns:
            control (list): Control outputs for x, y, and z axes.
        """
        now = time.time()
        if self.last_time is None:
            self.last_time = now

        dt = now - self.last_time
        if dt < self.sample_time:
            return self.last_output  # Skip update if the sample time has not elapsed

        control = [0.0, 0.0, 0.0]
        for i in range(3):  # Loop through x, y, z axes
            error = self.setpoint[i] - current_position[i]

            # Proportional term
            P_out = self.Kp * error

            # Integral term with anti-windup
            self.integral[i] += error * dt
            if self.Ki != 0:
                self.integral[i] = max(
                    self.output_limits[0] / self.Ki,
                    min(self.integral[i], self.output_limits[1] / self.Ki),
                )  # Clamp integral
            I_out = self.Ki * self.integral[i]

            # Derivative term
            derivative = (error - self.last_error[i]) / dt if dt > 0 else 0.0
            D_out = self.Kd * derivative

            # Total PID output
            output = P_out + I_out + D_out
            output = max(self.output_limits[0], min(output, self.output_limits[1]))  # Clamp output

            control[i] = output
            self.last_error[i] = error  # Save error for the next derivative calculation

        self.last_time = now
        self.last_output = control

        # Debugging output
        if self.debug:
            print(f"PID Update: P={P_out:.6f}, I={I_out:.6f}, D={D_out:.6f}, Output={output:.6f}, Error={error:.6f}")

        return control

if __name__ == "__main__":
    # Initialize the environment
    env = OT2Env(render=False, max_steps=200)

    # Define PID gains and target setpoint
    kp, ki, kd = 0.5, 0.05, 0.1  # Better-tuned gains
    setpoint = [0.1, 0.1, 0.1]  # Target position in meters

    # Initialize PID Controller
    pid_controller = PIDController(kp, ki, kd, setpoint, debug=True)

    # Testing process
    obs, info = env.reset()
    print("Testing PID Controller...\n")

    for step in range(200):
        current_position = obs[:3]  # Extract the current pipette position

        # Compute PID control actions
        control = pid_controller.compute_control(current_position)

        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(np.array(setpoint) - np.array(current_position))

        # Normalize control to fit within action space (-1, 1)
        normalized_control = np.clip(control, -1.0, 1.0)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(normalized_control)

        # Log detailed outputs
        print(f"Step: {step + 1}")
        print(f"  Current Position: {current_position}")
        print(f"  Control: {control}")
        print(f"  Normalized Control: {normalized_control}")
        print(f"  Distance to Goal: {distance_to_goal:.6f} m")
        print(f"  Reward: {reward:.6f}\n")

        # Check termination or truncation conditions
        if terminated or truncated:
            if distance_to_goal < 0.001:  # Strict accuracy (1 mm)
                print(f"Goal reached with high precision! Distance: {distance_to_goal:.6f} m")
            elif distance_to_goal < 0.01:  # Initial accuracy (10 mm)
                print(f"Goal reached! Distance: {distance_to_goal:.6f} m")
            else:
                print(f"Goal not reached with required precision. Distance: {distance_to_goal:.6f} m")
            break

    env.close()
    print("Testing completed.\n")

    # Output summary
    print("PID Controller Testing Summary")
    print(f"  Gains: Kp={kp}, Ki={ki}, Kd={kd}")
    print(f"  Final Distance to Goal: {distance_to_goal:.6f} m")
    print("\nTune gains further if accuracy requirements are not met.")