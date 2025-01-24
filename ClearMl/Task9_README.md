# OT-2 Robotic Simulation Environment

This project involves simulating the Opentrons OT-2 robotic system using PyBullet. The task includes:

- Setting up the simulation environment.
- Sending commands to the robot and observing its state.
- Determining the working envelope for the pipette tip by moving it to each corner of the workspace.

---

## Description

The simulation initializes a robotic system using the `sim_class` module. The robot is controlled to move the pipette tip to the 8 corners of its workspace to determine its working envelope. The coordinates of these corners are logged, providing valuable insights into the robot's operational range. The script includes proper connection management for the PyBullet server to ensure smooth operation and avoids redundant disconnection issues.

---

## Dependencies

To successfully run the simulation, the following dependencies are required:

### Python Dependencies
- **Python 3.8+**
- **PyBullet**: The physics engine used for the simulation.
- **NumPy**: For numerical operations.
- **Matplotlib** *(optional)*: For visualization.
- **ffmpeg** *(optional)*: For creating GIFs of the simulation.

### Installation
Install Python dependencies using pip:
```bash
pip install pybullet numpy matplotlib
```

Install **ffmpeg** for GIF creation (if needed):
- On Ubuntu:
  ```bash
  sudo apt-get install ffmpeg
  ```
- On Windows:
  Download the installer from [ffmpeg.org](https://ffmpeg.org/).

---

## Environment Setup

1. **Clone the Repository**:
   Clone the GitHub repository containing the OT-2 simulation environment:
   ```bash
   git clone https://github.com/BredaUniversityADSAI/Y2B-2023-OT2_Twin.git
   cd Y2B-2023-OT2_Twin
   ```

2. **Verify Files**:
   Ensure the following files and folders are present in the repository:
   - `sim_class.py`: Contains the `Simulation` class.
   - `custom.urdf`: The URDF file representing the OT-2 robot.
   - `textures/` and `meshes/`: Support files for the simulation.

3. **Run the Simulation**:
   Execute the script:
   ```bash
   python 9_2.py
   ```

---

## Working Envelope

The working envelope of the OT-2 pipette tip was determined by moving it to the 8 corners of a cuboidal workspace. The following coordinates represent the workspace limits:

| Corner | X      | Y      | Z      |
|--------|--------|--------|--------|
| 1      | 0.2531 | 0.2195 | 0.2895 |
| 2      | 0.253  | 0.2198 | 0.1693 |
| 3      | 0.2531 | -0.1705| 0.2895 |
| 4      | 0.253  | -0.1709| 0.1692 |
| 5      | -0.187 | 0.2195 | 0.2895 |
| 6      | -0.187 | 0.2197 | 0.1692 |
| 7      | -0.187 | -0.1705| 0.2895 |
| 8      | -0.187 | -0.1706| 0.1692 |

---

## Example Output

When you run the simulation, the following output will be generated:

```
Moved to corner with position: [0.2531, 0.2195, 0.2895]
Moved to corner with position: [0.253, 0.2198, 0.1693]
...
Working Envelope Corners:
Corner 1: [0.2531, 0.2195, 0.2895]
Corner 2: [0.253, 0.2198, 0.1693]
...
```

---

## Optional Features

- **Visualization**: Use `matplotlib` to visualize the simulation.
- **GIF Creation**: Use `ffmpeg` to create a GIF of the pipette's movement.

---

## Example Code Snippet

```python
from sim_class import Simulation

sim = Simulation(num_agents=1, render=True)
corners = []
for velocity_x, velocity_y, velocity_z in [
    (0.5, 0.5, 0.5),
    (0.5, 0.5, -0.5),
    (-0.5, -0.5, 0.5),
    (-0.5, -0.5, -0.5),
    # Add remaining corners
]:
    actions = [[velocity_x, velocity_y, velocity_z, 0]]
    state = sim.run(actions, num_steps=200)
    position = state['robotId_1']['pipette_position']
    corners.append(position)

print("Working Envelope Corners:", corners)
```

---

## Troubleshooting

- **Issue**: Simulation not rendering or failing to initialize.
  - **Solution**: Ensure PyBullet is installed and running in an environment with graphical rendering support.

- **Issue**: `p.disconnect()` error.
  - **Solution**: The script now includes a connection check before calling `p.disconnect()` to avoid redundant disconnection attempts.

---

## Notes

- The script is modular and can be adapted to simulate additional points or refine the working envelope.
- Modify the velocity parameters to test different corner configurations or ranges.

---

## License

This project is licensed under the MIT License.

