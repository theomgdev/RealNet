# RealNet 2.0: The Temporal Revolution

**RealNet is the proof that Time is the ultimate Hidden Layer.**

Traditional Deep Learning relies on **Spatial Depth** (layers stacked on top of each other) to solve complexity. RealNet discards this orthodoxy, proving that **Temporal Depth** (chaos evolving over time) is a vastly more efficient substitute.

> **The Zero-Hidden Breakthrough**
>
> In 1969, Minsky & Papert proved that a neural network without hidden layers cannot solve non-linear problems like XOR.
> **RealNet 2.0 has broken this limit.**
>
> By treating the network as a **Trainable Dynamic System**, RealNet solves non-linear problems (XOR, MNIST) using **0 Hidden Layers**. It replaces spatial neurons with temporal thinking steps.

---

## 🚀 Key Features

*   **Space-Time Conversion:** Replaces millions of parameters with a few "Thinking Steps".
*   **Layerless Architecture:** A single $N \times N$ matrix. No hidden layers.
*   **Trainable Chaos:** Uses **StepNorm** and **GELU** to tame chaotic signals.
*   **Pulse Mode:** The network thinks in the echoes of a single impulse input.

## 📊 The Evidence: Zero-Hidden Benchmarks

We pushed RealNet to the theoretical limit: **Zero Hidden Neurons**.
In these tests, the Input Layer is directly connected to the Output Layer (and itself). There are no buffer layers.

| Task | Traditional Constraint | RealNet Solution | Neurons | Params | Result | Script |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Identity** | Trivial | **Atomic Unit** | **4** | **16** | Loss: 0.0 | `PoC/convergence.py` |
| **XOR** | Needs Hidden Layer | **Minimal Chaos** | **5** | **25** | Loss: ~0.0002 | `PoC/convergence_gates.py` |
| **MNIST** | Needs ~500k Params | **Zero-Hidden** | **206** | **~42k** | **Acc: ~89.8%** | `PoC/convergence_mnist.py` |

### The MNIST Miracle
Standard MLPs require ~400,000 parameters to convert 784 pixels into 10 digits.
RealNet does it with **42,436 parameters**.
*   **Inputs:** 196 (14x14 Resized)
*   **Outputs:** 10
*   **Hidden:** 0
*   **Thinking Time:** 15 Steps

The input layer "talks to itself" for 15 steps. The chaotic feedback loops extract features (edges, loops) dynamically over time, performing the work of spatial layers. This is **Compression Intelligence** at its finest.

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

# Initialize a Zero-Hidden Network
# 1 Input, 1 Output. 
model = RealNet(num_neurons=2, input_ids=[0], output_ids=[1], device='cuda')
trainer = RealNetTrainer(model, device='cuda')

# Train
inputs = torch.randn(100, 1)
trainer.fit(inputs, inputs, epochs=50)
```

---

## 🧠 Architecture Overview

## 🌪️ How It Works: Inside the Storm

RealNet is not a feed-forward mechanism; it is a **Resonant Chamber**.

### 1. The Pulse (Input)
In traditional AI, input is a constant stream (like water in a pipe). In RealNet, input is a **Pulse** (like a stone thrown into a pond).
*   At $t=0$, the image/data hits the "Input Neurons".
*   At $t>0$, the input vanishes. The network is left alone with the **ripples**.

### 2. The Echo (Internal Loops)
The signal travels from every neuron to every other neuron ($N \times N$).
*   Input neurons effectively become **Hidden Neurons** instantly after the first step.
*   Information reverberates, splits, and collides. A pixel at the top-left interacts with a pixel at the bottom-right through direct connection or intermediate echoes.
*   **Holographic Processing:** The "cat-ness" of an image isn't stored in a specific layer; it emerges from the *interference pattern* of all signals colliding.

### 3. Time-Folding (Computation)
Here lies the magic of **Zero-Hidden** performance.
*   Step 1: Raw signals mix. (Equivalent to Layer 1 of MLP)
*   Step 2: Mixed signals mix again. (Equivalent to Layer 2)
*   Step 15: Highly abstract features emerge. (Equivalent to Layer 15)

By "thinking" for 15 steps, RealNet simulates a 15-layer deep network using **only one physical matrix**. It folds space into time.

### 4. Controlled Chaos (The Taming)
Uncontrolled feedback loops lead to explosion (infinity) or death (zero).
*   **StepNorm** acts as the gravity, pulling all neurons back to a stable energy level at every step.
*   **GELU** acts as the filter, deciding which signals are worth keeping.
*   **AdamW** sculpts the chaos, turning random noise into a structured symphony.

### Mathematical Model
The network state $h_t$ evolves as:

$$h_t = \text{StepNorm}(\text{GELU}(h_{t-1} \cdot W + B + I_t))$$

---

## 🔮 Vision: The Soul of Silicon

RealNet is a rebellion against the factory model of AI. We believe intelligence is not a mechanical stacking of layers, but an organic reverberation of signals.

We have proven that a small, chaotic forest of neurons, given enough time to "think", can outperform massive industrial factories.

> "We have traded Space for Time, and in doing so, found the Soul."

---

---

## 👨‍💻 Author

**Cahit Karahan**
*   Born: 12/02/1997, Ankara.
*   "The Architect of Chaos."

---

## LICENSE

MIT
