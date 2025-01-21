# pid_controller.py

import time

class PIDController:
    """
    A Proportional-Integral-Derivative (PID) Controller with anti-windup, output limits, and debugging.
    """

    def __init__(self, Kp, Ki, Kd, setpoint=0.0, sample_time=0.01, output_limits=(-10.0, 10.0), debug=False):
        """
        Initialize the PID controller.

        :param Kp: Proportional gain
        :param Ki: Integral gain
        :param Kd: Derivative gain
        :param setpoint: Desired target value
        :param sample_time: Minimum time interval between updates
        :param output_limits: Tuple (min_output, max_output) to clamp the output
        :param debug: Boolean to enable/disable debugging output
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.sample_time = sample_time
        self.output_limits = output_limits
        self.debug = debug

        # Internal state
        self.last_time = None
        self.integral = 0.0
        self.last_error = 0.0
        self.last_output = 0.0

    def update(self, current_value):
        """
        Calculate the PID control output based on the current value.

        :param current_value: Current measurement of the process variable
        :return: Control signal to apply
        """
        now = time.time()
        if self.last_time is None:
            self.last_time = now

        dt = now - self.last_time
        if dt < self.sample_time:
            return self.last_output  # Skip update if the sample time has not elapsed

        error = self.setpoint - current_value

        # Proportional term
        P_out = self.Kp * error

        # Integral term with anti-windup
        self.integral += error * dt
        if self.Ki != 0:
            self.integral = max(self.output_limits[0] / self.Ki,
                                min(self.integral, self.output_limits[1] / self.Ki))  # Clamp integral
        I_out = self.Ki * self.integral

        # Derivative term
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        D_out = self.Kd * derivative

        # Total PID output
        output = P_out + I_out + D_out
        output = max(self.output_limits[0], min(output, self.output_limits[1]))  # Clamp output to limits

        # Debugging output
        if self.debug:
            print(f"PID Update: P={P_out:.2f}, I={I_out:.2f}, D={D_out:.2f}, Output={output:.2f}, Error={error:.2f}")

        # Save state for the next calculation
        self.last_time = now
        self.last_error = error
        self.last_output = output

        return output

    def set_target(self, setpoint):
        """
        Update the target setpoint for the PID controller.

        :param setpoint: New desired value for the process variable
        """
        self.setpoint = setpoint
        self.integral = 0.0  # Reset integral term to avoid windup
        self.last_error = 0.0  # Reset last error for derivative calculation
