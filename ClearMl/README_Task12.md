# README

## Project Overview

This project leverages reinforcement learning with the Proximal Policy Optimization (PPO) algorithm in a custom OpenAI Gym-compatible environment. The implementation integrates tools like ClearML for experiment tracking, Weights & Biases (W&B) for monitoring, and uses Docker for environment consistency. This README provides an overview of the project setup, dependencies, and execution steps.

---

## Key Features

- **Reinforcement Learning**: Trains a model using the PPO algorithm.
- **Custom Environment**: Uses a custom OpenAI Gym-compatible environment.
- **Experiment Tracking**: Integrates ClearML and Weights & Biases for seamless experiment monitoring.
- **Docker Support**: Supports remote execution with Docker containers.
- **Flexible Configuration**: Command-line arguments for key hyperparameters.

---

## Prerequisites

### Dependencies

The project requires the following Python packages:
```text
absl-py==2.1.0
annotated-types==0.7.0
attrs==23.2.0
cachetools==5.5.0
certifi==2024.12.14
charset-normalizer==3.4.1
clearml==1.17.0
click==8.1.8
cloudpickle==3.0.0
contourpy==1.2.0
cycler==0.12.1
Cython==3.0.11
distlib==0.3.9
docker-pycreds==0.4.0
Farama-Notifications==0.0.4
filelock==3.13.1
fonttools==4.47.0
fsspec==2023.12.2
furl==2.1.3
gitdb==4.0.12
GitPython==3.1.44
google-auth==2.37.0
google-auth-oauthlib==0.4.6
grpcio==1.69.0
gymnasium==1.0.0
idna==3.10
Jinja2==3.1.2
jsonschema==4.23.0
jsonschema-specifications==2024.10.1
kiwisolver==1.4.5
Markdown==3.7
markdown-it-py==3.0.0
MarkupSafe==2.1.3
matplotlib==3.8.2
mdurl==0.1.2
mpmath==1.3.0
networkx==3.2.1
numpy==1.26.2
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.18.1
nvidia-nvjitlink-cu12==12.3.101
nvidia-nvtx-cu12==12.1.105
oauthlib==3.2.2
orderedmultidict==1.0.1
packaging==23.2
pandas==2.1.4
pathlib2==2.3.7.post1
Pillow==10.1.0
platformdirs==4.3.6
protobuf==3.20.0
psutil==5.9.8
pyasn1==0.6.1
pyasn1_modules==0.4.1
pybullet==3.2.6
pydantic==2.10.4
pydantic_core==2.27.2
Pygments==2.17.2
PyJWT==2.8.0
pyparsing==3.1.1
python-dateutil==2.8.2
python-dotenv==1.0.1
pytz==2023.3.post1
PyYAML==6.0.2
referencing==0.35.1
requests==2.31.0
requests-oauthlib==2.0.0
rich==13.7.0
rpds-py==0.22.3
rsa==4.9
sentry-sdk==2.19.2
setproctitle==1.3.4
six==1.16.0
smmap==5.0.2
stable_baselines3==2.4.0
sympy==1.12
tensorboard==2.10.1
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
torch==2.1.2
tqdm==4.66.1
triton==2.1.0
typing_extensions==4.12.2
tzdata==2023.3
urllib3==1.26.20
virtualenv==20.28.1
wandb==0.19.2
Werkzeug==3.1.3
```

### Additional Requirements
- Python 3.8+
- Docker installed on the system

---

## Setup

1. Clone the repository and navigate to the project directory:
    ```bash
    git clone https://github.com/your-repo/your-project.git
    cd your-project
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    - Create a `.env` file in the root directory.
    - Add your Weights & Biases (W&B) API key:
      ```text
      WANDB_API_KEY=your_wandb_api_key
      ```

4. (Optional) Configure Docker for ClearML integration:
    ```bash
    docker pull deanis/2023y2b-rl:latest
    ```

---

## Usage

1. Run the training script with default parameters:
    ```bash
    python train.py
    ```

2. Customize the hyperparameters using command-line arguments. Example:
    ```bash
    python train.py --learning_rate 0.001 --batch_size 64 --time_steps 1000000
    ```

---

## Command-Line Arguments

| Argument            | Default Value | Description                                  |
|---------------------|---------------|----------------------------------------------|
| `--learning_rate`   | 0.0001        | Learning rate for the optimizer             |
| `--batch_size`      | 32            | Batch size for training                     |
| `--n_steps`         | 2048          | Number of steps to collect per update       |
| `--n_epochs`        | 10            | Number of training epochs per update        |
| `--gamma`           | 0.98          | Discount factor for rewards                 |
| `--clip_range`      | 0.1           | Clipping parameter for PPO                  |
| `--value_coefficient`| 0.5          | Coefficient for value function loss         |
| `--time_steps`      | 5000000       | Total number of timesteps for training      |

---

## File Structure

- `train.py`: Main script for training the PPO model.
- `requirements.txt`: Dependency list.
- `models/`: Directory where trained models are saved.
- `.env`: Environment variable configuration file.
- `ClearMl/`: Directory containing ClearML-related configurations.

---

## Experiment Tracking

### ClearML Integration
ClearML is used to track experiments and manage remote execution. The script initializes a ClearML task with the following:
- **Project Name**: `Mentor Group J/Group 2/Musaed225739`
- **Task Name**: `Experiment2`

### Weights & Biases Integration
Weights & Biases tracks the training progress and saves metrics:
1. Ensure your W&B API key is set in the `.env` file.
2. Access the W&B dashboard for real-time updates.

---

## Notes

- Ensure the custom environment `OT2Env` is located in `ot2_gym_wrapper_V2.py`.
- Update the ClearML Docker image path if using a different environment.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgements

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [ClearML](https://github.com/allegroai/clearml)
- [Weights & Biases](https://wandb.ai/site)

