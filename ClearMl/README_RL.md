---

#  **README Section — Model Training, Hyperparameters & Performance Benchmarking**

##  Reinforcement Learning Model Experiments

To meet the requirements for training and comparing RL models, we experimented with **10 different model configurations**, varying learning rate, batch size, and gamma.
All models were trained on the same task and environment to ensure fair comparison.

The goal was to evaluate **how hyperparameters influence stability, convergence speed, and final error**.

---

##  **Hyperparameter Comparison Table**

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
| **Model (Task11) (Edoardo)**    | 0.0001        | 64         | 0.99  | **0.000954**     |  **Best Model (Lowest Error)**          |

---

##  Supporting Image



```markdown
https://github.com/MusaedMusaedSadeqMusaedAl-Fareh225739/CkearML/blob/main/ClearMl/Screenshot%202025-11-29%20224131.png

```

---

##  **Analysis & Interpretation**

### **1. Learning Rate**

* The **best-performing models** used a learning rate of **0.0001**.
* Extremely low LR (1e-05) consistently produced **poor results**, confirming slow and unstable learning.

### **2. Batch Size**

* A **moderate batch size (64)** provided the best tradeoff between stability and noise.
* Too small (32) still performed well, but (64) was more consistent across models.
* Larger batch (128) worked well only when combined with a good LR (e.g., Model Downloads).

### **3. Gamma (Discount Factor)**

* Gamma **0.98–0.99** produced stable learning.
* Very high gamma (0.999) caused *slight* performance degradation.

---

##  **Final Evaluation — Best Model**

**Model (Task11) (Edoardo)**

* **Learning Rate:** 0.0001
* **Batch Size:** 64
* **Gamma:** 0.99
* **Final Lowest Error:** **0.000954 m (Best)**

This model demonstrated:

* Fastest convergence
* Most stable policy
* Lowest final error across all experiments

---

##  **Conclusion**

Through systematic hyperparameter tuning and benchmarking, we demonstrated the ability to:

* Train multiple RL models
* Compare their performance scientifically
* Document hyperparameters and results
* Select the optimal model based on quantitative evidence

The experiments clearly show that the combination of **LR = 0.0001**, **Batch Size = 64**, **Gamma = 0.99** yields the most reliable and lowest-error policy.

---


