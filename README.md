# Character-Level Name Generation 

## 📌 Overview

This project explores **character-level sequence generation** using different recurrent neural network architectures. The goal is to generate realistic Indian names by modeling sequential dependencies between characters.

We implement and compare:

* Vanilla Recurrent Neural Network (RNN)
* Bidirectional LSTM (BiLSTM)
* Attention-based RNN

---

## 🧠 Objective

The objective of this work is to learn a probabilistic model over character sequences for the task of name generation. 
Formally, given a sequence:

x = (x₁, x₂, ..., x_T),

the model estimates the joint probability as:

P(x) = ∏ P(x_t | x₁, x₂, ..., x_{t-1})

This autoregressive formulation allows the model to generate new names sequentially, by sampling each character conditioned on the preceding context.

## 📂 Project Structure

```
.
├── rnn_model.py
├── bilstm_model.py
├── attention_rnn.py
├── train.py
├── generate.py
├── evaluate.py
├── TrainingNames.txt
├── results.txt
├── *.pth (saved models)
└── README.md
```

---

## ⚙️ Models Implemented

### 1. Vanilla RNN

* Simple recurrent architecture
* Learns sequential dependencies
* Serves as baseline model

---

### 2. BiLSTM

* Processes sequence in both directions
* Uses past + future context
* ❗ Not suitable for generation (explained below)

---

### 3. Attention RNN

* Computes context vector using attention
* Improves long-range dependency modeling
* Produces best results

---

## 🏋️ Training Details

* Framework: PyTorch
* Loss Function: CrossEntropyLoss
* Optimizer: Adam
* Learning Rate: 0.002
* Epochs: 25

Each name is processed as:

* Input: `<name`
* Target: `name>`

---

## 🔤 Generation Strategy

* Start with token `<`
* Predict next character using softmax
* Use temperature scaling for randomness
* Stop when `>` is generated

---

## 📊 Results

### 🔢 Quantitative Metrics

| Model     | Novelty | Diversity |
| --------- | ------- | --------- |
| RNN       | 0.462   | 0.678     |
| BiLSTM    | 1.000   | 0.001     |
| Attention | 0.503   | 0.567     |

---

### ✨ Sample Outputs

#### RNN

```
anika menon
aarav pandey
myra iyer
diya singh
```

#### BiLSTM

```
(empty or repetitive outputs)
```

#### Attention RNN

```
pari malhotra
aarav bajaj
raghav bajaj
myra bajpai
```

---

## ❗ Key Findings

### 🔴 BiLSTM Failure

BiLSTM performs poorly for generation because:

The poor performance of BiLSTM in sequence generation arises from a fundamental mismatch between its training objective and inference setting.

During training, BiLSTM leverages bidirectional context and models:
P(x_t | x_{<t}, x_{>t})

In contrast, autoregressive generation requires predicting tokens based only on past context:
P(x_t | x_{<t})

Since future context is unavailable at inference time, the learned representations become inconsistent, leading to degraded performance, including premature termination and mode collapse.

This mismatch causes:

* Empty outputs
* Mode collapse
* Extremely low diversity

---

### 🟢 Best Model: Attention RNN

* Produces coherent names
* Maintains diversity
* Captures long-range dependencies

---

## ⚠️ Failure Modes

* Mode collapse (BiLSTM)
* Premature sequence termination
* Repetition of common surnames
* Overfitting due to small dataset

---

## 🚀 Future Improvements

* Use embedding layers instead of one-hot encoding
* Add dropout regularization
* Train on larger datasets
* Explore Transformer-based models

---

## ▶️ How to Run

### 1. Train Model

```bash
python train.py
```

### 2. Generate Names

```bash
python generate.py
```

### 3. Evaluate Models

```bash
python evaluate.py
```

---

## 📚 Dependencies

* Python 3.x
* PyTorch
* NumPy

Install dependencies:

```bash
pip install torch numpy
```

---

## 👨‍🎓 Author

**Naman Jain**
B22BB027
IIT Jodhpur

---

## 📎 Note

This project was developed as part of the **Natural Language Understanding (NLU)** course assignment.
