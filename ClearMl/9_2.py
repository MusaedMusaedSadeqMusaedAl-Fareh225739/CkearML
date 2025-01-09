from sim_class import Simulation

# Initialize the simulation with DIRECT mode
sim = Simulation(num_agents=1, render=True)  # Disable GUI

# Define corners of the working envelope
corners = []

# Move to each corner and log coordinates
for velocity_x, velocity_y, velocity_z in [
    (0.5, 0.5, 0.5),   # Corner 1
    (0.5, 0.5, -0.5),  # Corner 2
    (0.5, -0.5, 0.5),  # Corner 3
    (0.5, -0.5, -0.5), # Corner 4
    (-0.5, 0.5, 0.5),  # Corner 5
    (-0.5, 0.5, -0.5), # Corner 6
    (-0.5, -0.5, 0.5), # Corner 7
    (-0.5, -0.5, -0.5) # Corner 8
]:
    actions = [[velocity_x, velocity_y, velocity_z, 0]]
    state = sim.run(actions, num_steps=1600)
    position = state['robotId_1']['joint_states']['joint_0']['position']
    corners.append(position)
    print(f"Moved to corner with position: {position}")

# Print all recorded corners
print("Working Envelope Corners:")
for i, corner in enumerate(corners, start=1):
    print(f"Corner {i}: {corner}")

# Disconnect from the simulation
import pybullet as p
p.disconnect()