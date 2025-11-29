# **Reinforcement Learning Model Training, Evaluation & Benchmarking**

This document presents our complete reinforcement learning (RL) experimentation pipeline, including hyperparameter tuning, model comparison, performance benchmarking, and code references. All models were trained using PPO (Stable-Baselines3) and logged/evaluated through ClearML.

---

## **1. Reinforcement Learning Model Experiments**

To meet the course requirements for training and comparing RL models, we trained **10 different PPO model configurations**, systematically varying:

- Learning rate  
- Batch size  
- Gamma (discount factor)

All models were trained on the same environment to allow a fair and controlled comparison.

The objective of these experiments was to evaluate how hyperparameter choices influence:

- Convergence speed  
- Stability  
- Final lowest achieved error  
- Overall policy performance  

---

## **2. Hyperparameter Comparison Table**

| Model                           | Learning Rate | Batch Size | Gamma | Lowest Error (m) | Comments                                  |
| ------------------------------- | ------------- | ---------- | ----- | ---------------- | ----------------------------------------- |
| **Model (1) (Musaed)**          | 0.0001        | 32         | 0.98  | 0.001025         | Small batch size, good performance        |
| **Model (2) (Musaed)**          | 1e-05         | 64         | 0.98  | 0.009019         | Worst performance — learning rate too low |
| **Model (3) (Musaed)**          | 0.0001        | 32         | 0.98  | 0.001019         | Best model among Musaed’s models          |
| **Model (4) (Musaed)**          | 5e-05         | 128        | 0.999 | 0.001032         | High gamma → slightly larger error        |
| **Model (5) (Musaed)**          | 0.0001        | 64         | 0.98  | 0.001047         | Similar to Model 3 but slightly worse     |
| **Model (6) (Musaed)**          | 1e-05         | 128        | 0.96  | 0.002459         | Very low LR → unstable, second-worst      |
| **Model (z4sv4e) (Edoardo)**    | 0.0003        | 64         | 0.99  | 0.006313         | Much higher error than other runs         |
| **Model (Downloads) (Edoardo)** | 0.0003        | 128        | 0.99  | 0.001005         | 2nd Best Model overall                    |
| **Model (Task11) (Edoardo)**    | 0.0001        | 64         | 0.99  | **0.000954**     | **Best Model (Lowest Error)**            |

These results clearly show how learning rate, batch size, and gamma influence the training performance of PPO in this environment.

---

## **3. Supporting Image**

![Evaluation With Edoardo](https://github.com/MusaedMusaedSadeqMusaedAl-Fareh225739/CkearML/blob/main/ClearMl/Screenshot%202025-11-29%20224131.png)

---

## **4. Analysis & Interpretation**

### **4.1 Learning Rate**
- The models with **LR = 0.0001** consistently performed best.  
- Extremely low LR values (e.g., **1e-05**) produced poor convergence and unstable learning.

### **4.2 Batch Size**
- Batch size **64** provided the best stability-to-noise balance.  
- Batch size **32** also performed well, but with more variability.  
- Large batch size (128) only performs well when paired with a good learning rate.

### **4.3 Gamma (Discount Factor)**
- Gamma **0.98–0.99** yielded stable learning across models.  
- Gamma **0.999** increased long-term weighting too much, slightly worsening results.

---

## **5. Final Evaluation — Best Model**

The best-performing model from all experiments is:

###  **Model (Task11) — Edoardo**
- **Learning Rate:** 0.0001  
- **Batch Size:** 64  
- **Gamma:** 0.99  
- **Lowest Error:** **0.000954 m**  

This model produced:
- The most stable training curve  
- Fast convergence  
- The lowest recorded final error  
- Consistent evaluation performance  

---

## **6. Code Used for Training & Evaluation**

All training and evaluation code for the best model is publicly available:

 **Task11_test_edoardo.ipynb**  
https://github.com/MusaedMusaedSadeqMusaedAl-Fareh225739/CkearML/blob/main/ClearMl/Task11_test_edoardo.ipynb

This notebook contains:
- Environment setup  
- Reward function  
- PPO configuration  
- Hyperparameter selection  
- Training loop  
- Evaluation logic  
- ClearML tracking  

This notebook directly generates the results shown in the comparison table.

---

## **7. Team Contribution — Training Responsibility**

All reinforcement learning model training, experimentation, and evaluation were performed **exclusively by Musaed Alfareh and Edoardo**.

Specifically, we were responsible for:

- Setting up the PPO training pipeline  
- Implementing and tuning reward functions  
- Running all training sessions  
- Testing hyperparameters  
- Tracking experiments with ClearML  
- Comparing performance and selecting the best model  
- Documenting all results in this README  

The models in the comparison table represent the **best-performing configurations** from all experiments conducted by Musaed and Edoardo.

---

## **8. Conclusion**

Through extensive hyperparameter testing and RL model training, we successfully demonstrated the ability to:

- Train and evaluate multiple RL models  
- Compare model performance using quantitative measures  
- Document training performance and hyperparameters  
- Benchmark results in a clear, reproducible format  
- Identify the best-performing model based on empirical evidence  

The results conclude that the optimal configuration for this task is:

➡ **Learning Rate = 0.0001**  
➡ **Batch Size = 64**  
➡ **Gamma = 0.99**

This setup consistently produced the **lowest error** and most stable learned policy.

---

