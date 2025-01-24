# Simulation README

This project demonstrates the simulation of a robotic system to determine the working envelope of a pipette in a 3D environment using the `sim_class` module.

---

## Description

The simulation initializes a robot with a single agent in **DIRECT mode** (render enabled) and moves it to the corners of its workspace. By logging the coordinates at each corner, the working envelope of the pipette is determined. This information is critical for understanding the robot's operational range and ensuring its movements are within safe limits. The script ensures proper connection to the PyBullet server and handles potential disconnection issues gracefully.

---

## Dependencies

To run this project, you need the following Python libraries and tools installed:

- **Python 3.8+**
- **sim_class** (custom simulation module)
- **pybullet** (simulation backend)
- **Numpy** (for any numerical operations)

Install the dependencies via pip:
```bash
pip install pybullet numpy
```

---

## Setup Instructions

1. Clone this repository or copy the script into your project directory.
2. Ensure all dependencies are installed (see the section above).
3. Run the simulation script:
   ```bash
   python 9_2.py
   ```

---

## Working Envelope of the Pipette

The simulation determines the following corners of the pipette's working envelope:

1. **Corner 1**: `[0.2531, 0.2195, 0.2895]`
2. **Corner 2**: `[0.253, 0.2198, 0.1693]`
3. **Corner 3**: `[0.2531, -0.1705, 0.2895]`
4. **Corner 4**: `[0.253, -0.1709, 0.1692]`
5. **Corner 5**: `[-0.187, 0.2195, 0.2895]`
6. **Corner 6**: `[-0.187, 0.2197, 0.1692]`
7. **Corner 7**: `[-0.187, -0.1705, 0.2895]`
8. **Corner 8**: `[-0.187, -0.1706, 0.1692]`

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
