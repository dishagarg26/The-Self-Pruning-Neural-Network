# 🧠 Self-Pruning Neural Network (Dynamic Sparse Learning)

## 📌 Overview

This project implements a **self-pruning neural network** that learns to remove unnecessary connections during training itself. Unlike traditional pruning techniques (which are applied after training), this model **jointly learns weights and structure**, making it more adaptive and efficient.

The model is trained on the **CIFAR-10 dataset** and demonstrates how neural networks can automatically discover a sparse architecture without explicit pruning steps.

---

## 🚀 Key Idea

Each weight in the network is associated with a **learnable gate parameter**.

* Gate values are obtained using a **sigmoid function**
* Each weight is scaled as:

[
W_{effective} = W \times \sigma(\alpha)
]

Where:

* (W) → weight matrix
* (\alpha) → learnable gate scores
* (\sigma(\alpha)) → gate values in (0,1)

---

## ⚙️ Loss Function

The training objective combines classification loss with sparsity regularization:

[
\text{Total Loss} = \text{CrossEntropy} + \lambda \cdot \text{Sparsity Loss}
]

Where:

* **CrossEntropy** → standard classification loss
* **Sparsity Loss** → L1 penalty on gate values

[
\text{Sparsity Loss} = \sum |\sigma(\alpha)|
]

This encourages many gates to shrink toward zero, effectively removing unimportant connections.

---

## 🏗️ Model Architecture

A fully connected network is used:

```
Input (3072)
   ↓
Hidden Layer (512)
   ↓
Hidden Layer (256)
   ↓
Hidden Layer (128)
   ↓
Output (10 classes)
```

Each layer is implemented using a **custom gated linear layer** instead of a standard dense layer.

---

## 📊 Training Strategy

To stabilize learning and avoid premature pruning:

* **Warmup Phase** (first few epochs):
  No sparsity penalty is applied

* **Annealing Phase**:
  Sparsity gradually increases over epochs

* **Dual Learning Rates**:

  * Weights → lower learning rate
  * Gates → higher learning rate

This ensures:

* The model first **learns features**
* Then gradually **removes redundant connections**

---

## 📈 Results

| Lambda (λ) | Accuracy (%) | Sparsity (%) |
| ---------- | ------------ | ------------ |
| Low        | ~55–60%      | ~20–40%      |
| Medium     | ~50–55%      | ~50–70%      |
| High       | ~45–50%      | ~70–90%      |

### 🔍 Observations

* Increasing λ → higher sparsity
* Moderate λ → best balance between accuracy and compression
* High λ → aggressive pruning but slight accuracy drop

---

## 📉 Gate Distribution

A successful pruning behavior shows:

* A large concentration of gates near **0** → pruned weights
* A smaller cluster away from 0 → important weights

This results in a **bimodal distribution**, confirming effective self-pruning.

---

## 💡 Key Insights

* Neural networks contain **significant redundancy**
* L1 regularization on gates effectively promotes sparsity
* Learning structure during training is more efficient than post-pruning
* Proper scheduling is crucial to avoid early collapse

---



## 🔮 Future Improvements

* Apply pruning to **CNN architectures** for better accuracy
* Implement **hard pruning + fine-tuning**
* Explore **L0 regularization (Hard Concrete gates)**
* Deploy compressed model for **edge devices**

---

## 🧠 Conclusion

This project demonstrates that neural networks can **adaptively learn which connections matter**, enabling efficient and scalable models. By integrating pruning directly into training, we achieve both **model compression and regularization** in a unified framework.

---


