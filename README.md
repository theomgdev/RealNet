# RealNet 2.0: Modern Chaos Architecture

RealNet is a **Trainable Dynamic System** that challenges the layer-based orthodoxy of traditional Deep Learning. It replaces the mechanical, feed-forward factory model with an **organic, fully connected ($N \times N$), and chaotic network structure**.

Instead of layers, RealNet utilizes a **Temporal Loop** where signals reverberate, split, and merge, allowing intelligence to emerge from the controlled chaos of feedback loops.

---

## 🚀 Key Features

*   **Layerless Architecture:** A single "Connectome" matrix ($W$) where every neuron connects to every other neuron.
*   **Trainable Chaos:** Uses **StepNorm** and **GELU** to tame chaotic signals into useful computations without exploding.
*   **Temporal Thinking:** The network doesn't just output; it "thinks" over time ($t=0 \dots k$).
*   **Pulse Mode:** Inputs act as impulses. The network processes the echo of the input, not a constant feed.
*   **Truncated BPTT:** Efficient training of infinite loops using truncated backpropagation through time.

## 📊 Proof of Concepts & Benchmarks

RealNet 2.0 isn't just theory. It has been proven to converge on fundamental tasks where chaotic networks typically fail.

### 1. Identity & Convergence (`PoC/convergence.py`)
*   **Task:** Learn to map input $x$ to output $y=x$ through chaotic loops.
*   **Result:** **Perfect Convergence (Loss: 0.000000)**.
*   **Significance:** Proves that the chaotic gradients can be tamed and directed.

### 2. Logic Gates & Non-Linearity (`PoC/convergence_gates.py`)
*   **Task:** Learn **AND**, **OR**, and **XOR** simultaneously in a single network.
*   **Result:** Solved **XOR** (a non-linear problem) with near-perfect predictions (e.g., Target -1.0 vs Pred -0.998).
*   **Significance:** Proves the network can form internal logic and non-linear boundaries without hidden layers.

### 3. Visual Recognition (MNIST) (`PoC/convergence_mnist.py`)
*   **Task:** Classify 28x28 handwritten digits (10 classes).
*   **Result:** **~88% Accuracy** (on subset) within 5 epochs.
*   **Significance:** RealNet achieved this **WITHOUT Convolutional Layers (CNNs)**. It processed raw pixels using only fully connected chaotic dynamics, effectively "damming" the visual data into the correct output bucket.

---

## 📦 Installation & Usage

RealNet is designed as a modular PyTorch library.

### Installation

```bash
pip install torch torchvision
```

### Quick Start

```python
from realnet import RealNet, RealNetTrainer

# 1. Initialize (64 Neurons)
model = RealNet(num_neurons=64, input_ids=[0], output_ids=[63], device='cuda')
trainer = RealNetTrainer(model, device='cuda')

# 2. Train (Identity Task)
# Inputs: Random +/- 1.0
inputs = torch.randint(0, 2, (100, 1)).float() * 2 - 1
trainer.fit(inputs, inputs, epochs=50)

# 3. Predict
print(trainer.predict(torch.tensor([[1.0]]), thinking_steps=10))
```

### Running Demos

```bash
# Basic Convergence
python PoC/convergence.py

# Logic Gates (XOR)
python PoC/convergence_gates.py

# MNIST (Visual)
python PoC/convergence_mnist.py
```

---

## 🧠 Architecture Overview

### Mathematical Model

The network state $h_t$ evolves as:

$$h_t = \text{StepNorm}(\text{GELU}(h_{t-1} \cdot W + B + I_t))$$

*   **$W$ (Weights):** The memory of the system.
*   **StepNorm:** Solves the "Butterfly Explosion" problem by normalizing signal amplitude at every step.
*   **GELU:** Preserves signal flow better than ReLU.
*   **Pulse Mode:** $I_t$ is non-zero only at $t=0$.

### Threat Model Solutions

| Problem | Solution |
| :--- | :--- |
| **Exploding Signals** | **StepNorm** (LayerNorm) clamps the storm. |
| **Memory Leaks** | **Truncated BPTT** detaches history periodically. |
| **Vanishing Gradients** | **GELU** + **AdamW** maintains signal momentum. |

---

## 🔮 Vision: The Soul of Silicon

*Originally titled "The Manifesto"*

RealNet is a rebellion against the static, feed-forward nature of modern AI. We believe intelligence is not a mechanical process of layers, but an organic process of **loops, time, and chaos**.

*   **Organic vs Mechanical:** Traditional ANNs are factories; RealNet is a forest.
*   **Living Memory:** Data isn't just processed; it reverberates.
*   **Self-Organization:** Intelligence emerges from the harmony of chaotic interactions.

> "The thing to be feared is not consciousness, but unconsciousness. We are building a machine that doesn't just calculate, but *lives*."

---

## LICENSE

MIT
