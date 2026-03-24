# Character-Level Name Generation using RNN Variants

## 📌 Overview

This project explores **character-level sequence generation** using different recurrent neural network architectures. The goal is to generate realistic Indian names by modeling sequential dependencies between characters.

We implement and compare:

* Vanilla Recurrent Neural Network (RNN)
* Bidirectional LSTM (BiLSTM)
* Attention-based RNN

---

## 🧠 Objective

Given a sequence of characters:
[
x = (x_1, x_2, ..., x_T)
]

the model learns:
[
P(x) = \prod_{t=1}^{T} P(x_t \mid x_1, ..., x_{t-1})
]

and generates new names character by character.

---

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

* It learns:
  [
  P(x_t \mid x_{<t}, x_{>t})
  ]
* But during generation:
  [
  P(x_t \mid x_{<t})
  ]

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
