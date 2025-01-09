import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from ot2_gym_wrapper import OT2Env


def main():
    # Initialize the environment
    env = OT2Env(render=False, max_steps=200)  # Reduced max steps for faster testing

    # Check environment compatibility
    print("Checking environment compatibility...")
    check_env(env)
    print("Environment passed compatibility checks!")

    # Run the environment with random actions
    print("\nStarting environment testing with random actions...")
    num_episodes = 3  # Number of episodes for testing

    for episode in range(num_episodes):
        obs, info = env.reset()
        print(f"\nEpisode {episode + 1} started.")
        step = 0

        while True:
            # Take a random action from the action space
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Log detailed output every 10 steps
            if step % 10 == 0:
                print(f"  Step {step + 1}:")
                print(f"    Action: {action}")
                print(f"    Observation: {obs}")
                print(f"    Reward: {reward:.2f}")
                print(f"    Terminated: {terminated}")
                print(f"    Truncated: {truncated}")

            # Check termination or truncation
            if terminated or truncated:
                print(f"  Episode {episode + 1} ended after {step + 1} steps.")
                print(f"  Final Reward: {reward:.2f}")
                break

            step += 1

    # Close the environment
    env.close()
    print("\nEnvironment testing completed!")


if __name__ == "__main__":
    main()
