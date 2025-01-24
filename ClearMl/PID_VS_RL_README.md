# README: Comparing Reinforcement Learning (RL) vs PID Controllers

## Project Overview
This project demonstrates the performance comparison between a Reinforcement Learning (RL) model and a traditional PID (Proportional-Integral-Derivative) controller for achieving precise control of a robotic arm or similar system. The RL model is trained using the Proximal Policy Optimization (PPO) algorithm, while the PID controller relies on tuned parameters to reach target positions within a simulated environment.

The project evaluates and visualizes metrics such as:
- **Average Accuracy**: The mean distance to the target position.
- **Standard Deviation**: Variability in accuracy across multiple runs.
- **Average Steps**: Number of steps taken to achieve the goal.

## Environment Details
- **Simulation Environment**: `OT2Env`
  - A custom Gym environment for simulating robotic motion.
  - Integrated with PyBullet for physics simulation.
- **RL Algorithm**: Proximal Policy Optimization (PPO) from the `stable-baselines3` library.
- **PID Controller**: Implemented using the `simple_pid` library.
- **Visualization**: Results are visualized using bar charts to compare RL and PID performance.

## Key Features
1. **RL Model Evaluation**:
   - Fixed goal position: `[0.1, 0.1, 0.2]` (meters).
   - Metrics calculated over multiple episodes (default: 10).
2. **PID Controller Evaluation**:
   - Randomly generated target positions within defined bounds:
     ```python
     bounds = {
         "x": [0.2531, -0.1872],
         "y": [0.2201, -0.1711],
         "z": [0.2896, 0.1691],
     }
     ```
   - Metrics calculated for multiple target positions (default: 10).
3. **Visualization**:
   - Bar charts for comparing:
     - Average accuracy (distance to goal).
     - Standard deviation of accuracy.
     - Average steps to reach the target.
   - Visualizations saved as PNG files:
     - `avg_accuracy_comparison.png`
     - `std_dev_comparison.png`
     - `avg_steps_comparison.png`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/compare-rl-vs-pid.git
   cd compare-rl-vs-pid
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
1. Ensure your RL model is saved at the correct path in the script:
   ```python
   model_path = r"C:\\Users\\jimal\\OneDrive - BUas\\Pictures\\CkearML\\ClearMl\\model.zip"
   ```
2. Run the comparison script:
   ```bash
   python compare_rl_vs_pid.py
   ```
3. View the generated metrics in the terminal and visualizations in the output PNG files.

## Example Results
After running the script, you’ll see outputs like:

```
Evaluating RL Model...
RL Model | Avg Accuracy: 0.0008 m | Std Dev: 0.0001 m | Avg Steps: 54.30

Evaluating PID Controller...
PID Controller | Avg Accuracy: 0.0013 m | Std Dev: 0.0010 m | Avg Steps: 541.40

Comparison Results:
RL Model | Avg Accuracy: 0.0008 m | Std Dev: 0.0001 m | Avg Steps: 54.30
PID Controller | Avg Accuracy: 0.0013 m | Std Dev: 0.0010 m | Avg Steps: 541.40
```

### Visualizations
1. **Average Accuracy Comparison**:
   ![Average Accuracy](avg_accuracy_comparison.png)

2. **Standard Deviation Comparison**:
   ![Standard Deviation](std_dev_comparison.png)

3. **Average Steps Comparison**:
   ![Average Steps](avg_steps_comparison.png)

## Technical Details
### RL Model Evaluation
- **Algorithm**: PPO from `stable-baselines3`.
- **Metrics**:
  - Average accuracy: Distance to fixed goal position across episodes.
  - Standard deviation: Variability in accuracy.
  - Average steps: Efficiency in reaching the goal.

### PID Controller Evaluation
- **Implementation**: `simple_pid` library.
- **Parameters**:
  - Proportional (Kp): 10.0
  - Integral (Ki): 5.5
  - Derivative (Kd): 3.0
- **Metrics**:
  - Average accuracy: Distance to random targets across positions.
  - Standard deviation: Consistency in accuracy.
  - Average steps: Efficiency in reaching the target.

### Visualization
- **Library**: Matplotlib.
- **Files Generated**:
  - `avg_accuracy_comparison.png`
  - `std_dev_comparison.png`
  - `avg_steps_comparison.png`

## Improvements and Suggestions
1. **RL Model**:
   - Train on a variety of goal positions for better generalization.
   - Experiment with different reward functions to optimize performance.
2. **PID Controller**:
   - Use adaptive PID tuning for dynamic environments.
   - Increase output limits to speed up convergence.
3. **Visualization**:
   - Add line plots to show step-by-step progress toward the goal.

## Conclusion
This project demonstrates how RL outperforms PID in terms of precision and efficiency for the given task. While PID is simpler to implement, RL provides better adaptability and consistency in dynamic environments.

## License
This project is licensed under the MIT License.
