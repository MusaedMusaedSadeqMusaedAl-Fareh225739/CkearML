# Simulation README

This project demonstrates the simulation of a robotic system to determine the working envelope of a pipette in a 3D environment using the `sim_class` module.

---

## Description

The simulation initializes a robot with a single agent in **DIRECT mode** (render enabled) and moves it to the corners of its workspace. By logging the coordinates at each corner, the working envelope of the pipette is determined. This information is critical for understanding the robot's operational range and ensuring its movements are within safe limits.

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
