import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    """
    A Gymnasium-compatible environment wrapper for the OT-2 digital twin.
    The goal is to train an agent to move the pipette tip to a given target position.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render=False, max_steps=1000):
        """
        Initialize the OT2 environment.
        
        Args:
            render (bool): Enable rendering with PyBullet GUI.
            max_steps (int): Maximum steps per episode.
        """
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=self.render)

        # Define action space: motor velocities for x, y, z axes (-1 to 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Define observation space: pipette position (x, y, z) + goal position (x, y, z)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # Initialize steps and goal position
        self.steps = 0
        self.goal_position = None

    def reset(self, seed=None, options=None):
        """
        Reset the environment and set a random goal position.

        Args:
            seed (int, optional): Seed for reproducibility.
            options (dict, optional): Additional options.

        Returns:
            observation (np.ndarray): Initial observation (pipette and goal positions).
            info (dict): Additional info.
        """
        super().reset(seed=seed)

        # Reset simulation
        self.sim.reset(num_agents=1)

        # Retrieve pipette position
        robot_id = self.sim.robotIds[0]
        pipette_position = np.array(self.sim.get_pipette_position(robotId=robot_id), dtype=np.float32)

        # Set a random goal position
        self.goal_position = np.random.uniform(low=-0.5, high=0.5, size=(3,)).astype(np.float32)

        # Combine pipette and goal positions into observation
        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)

        # Reset step counter
        self.steps = 0

        return observation, {}

    def step(self, action):
        """
        Apply an action, update the simulation, and calculate reward.

        Args:
            action (np.ndarray): Action vector (motor velocities for x, y, z axes).

        Returns:
            observation (np.ndarray): Updated observation (pipette and goal positions).
            reward (float): Reward for the action.
            terminated (bool): Whether the episode has ended successfully.
            truncated (bool): Whether the episode was truncated due to max steps.
            info (dict): Additional info.
        """
        # Append drop action (0 for no drop)
        full_action = np.append(action, [0])

        # Apply the action to the simulation
        self.sim.run([full_action])

        # Get current pipette position
        robot_id = self.sim.robotIds[0]
        pipette_position = np.array(self.sim.get_pipette_position(robotId=robot_id), dtype=np.float32)

        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)

        # Reward: negative distance to goal
        reward = float(-distance_to_goal)

        # Termination condition: pipette reaches the goal
        goal_reached = distance_to_goal < 0.05
        terminated = bool(goal_reached)

        # Truncation condition: max steps reached
        truncated = self.steps >= self.max_steps

        # Increment step counter
        self.steps += 1

        # Combine pipette and goal positions into observation
        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)

        return observation, reward, terminated, truncated, {}

    def render(self, mode="human"):
        """
        Render the environment (if GUI is enabled).
        """
        if self.render:
            self.sim.render()

    def close(self):
        """
        Close the simulation and clean up resources.
        """
        self.sim.close()
