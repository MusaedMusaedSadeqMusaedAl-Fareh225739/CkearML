import time

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, sample_time=0.01, output_limits=(None, None)):
        """
        Basic PID Controller for one axis.
        
        :param Kp: Proportional gain
        :param Ki: Integral gain
        :param Kd: Derivative gain
        :param setpoint: Target position
        :param sample_time: Minimum time step (sec) between PID calculations
        :param output_limits: (min_output, max_output) clamp the output if needed
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        
        self.sample_time = sample_time
        self.last_time = None
        
        self.integral = 0.0
        self.last_error = 0.0
        
        self.min_output, self.max_output = output_limits
        self.last_output = 0.0

    def update(self, current_value):
        """
        Calculate the PID output for the current axis value.
        
        :param current_value: The measured position of the axis (e.g., current X)
        :return: The control signal (e.g. velocity or step command) for the axis
        """
        now = time.time()
        if self.last_time is None:
            self.last_time = now
        
        dt = now - self.last_time
        if dt < self.sample_time:
            return self.last_output  # skip if not enough time has passed

        error = self.setpoint - current_value

        # Proportional
        P_out = self.Kp * error
        
        # Integral
        self.integral += error * dt
        I_out = self.Ki * self.integral
        
        # Derivative
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        D_out = self.Kd * derivative

        # PID summation
        output = P_out + I_out + D_out
        
        # Clamp output if needed
        if self.min_output is not None:
            output = max(self.min_output, output)
        if self.max_output is not None:
            output = min(self.max_output, output)
        
        # Save state
        self.last_time = now
        self.last_error = error
        self.last_output = output
        
        return output
