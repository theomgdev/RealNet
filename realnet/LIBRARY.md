# 📘 RealNet Library Documentation

RealNet is a PyTorch-based library that implements **Zero-Hidden Layer** neural networks using **Temporal Depth**. By treating the neural network as a dynamic system that evolves over time, RealNet achieves deep learning capabilities without stacking spatial layers.

## 核心 (Core Modules)

The library is modularized into core components:
1.  **`realnet.core.network`**: Contains the `RealNet` architecture.
2.  **`realnet.training.trainer`**: Contains `RealNetTrainer`.
3.  **`realnet.utils`**: Utilities for data (`data.py`) and pruning (`pruning.py`).

---

## 🧠 RealNet Model (`realnet.core.network`)

The `RealNet` class defines the structure and dynamics of the network. It is a single layer where every neuron is connected to every other neuron (including itself).

### Initialization

```python
from realnet import RealNet

model = RealNet(
    num_neurons=10, 
    input_ids=[0, 1], 
    output_ids=[9], 
    pulse_mode=True, 
    dropout_rate=0.1, 
    device='cuda'
)
```

**Parameters:**
*   `num_neurons` (int): Total number of neurons in the single layer (No hidden layers).
*   `input_ids` (list[int]): Indices of neurons that receive external input.
*   `output_ids` (list[int]): Indices of neurons whose state is read as output.
*   `pulse_mode` (bool): 
    *   `True`: Input is applied only at $t=0$ (Impulse).
    *   `False`: Input is applied continuously at every step (Stream).
*   `dropout_rate` (float): Probability of synaptic failure during training (Biological simulation).
*   `device` (str): 'cpu' or 'cuda'.

### Key Methods

#### `model.compile()`
Optimizes the model using `torch.compile` (PyTorch 2.0+) for faster execution. Returns the compiled model.

#### `model.forward(x_input, steps=1, current_state=None)`
Runs the dynamic system.
*   `x_input`: Input tensor. Can be a single pulse or a sequence.
*   `steps`: **Thinking Time**. How many times the signal reverberates in the echo chamber.
*   **Returns**: `(all_states, final_state)`

#### `SynapticPruner.prune(model, threshold=0.001)`
Permanently kills connections that are below the absolute threshold.
*   **Location**: `realnet.utils.pruning`
*   **Returns**: `(pruned_count, total_dead, total_synapses)`

#### `SparseRealNet.from_dense(model)`
Creates an optimized Sparse version of the model for inference.
```python
from realnet import SparseRealNet
sparse_model = SparseRealNet.from_dense(model)
```
Replaces the old `make_sparse()` method.

---

## 🏋️ RealNet Trainer (`realnet.training.trainer`)

The `RealNetTrainer` handles the training loop, gradient accumulation, mixed precision (AMP), and experimental features like Ghost Gradients.

### Initialization

```python
from realnet import RealNetTrainer

trainer = RealNetTrainer(
    model, 
    optimizer=None,      # Defaults to AdamW
    loss_fn=None,        # Defaults to MSELoss
    device='cuda',
    gradient_decay=0.0   # Ghost Gradients (Persistence)
)
```

**Parameters:**
*   `gradient_decay` (float): **Ghost Gradients / Persistence**.
    *   `0.0`: Standard behavior (`zero_grad()` after every step).
    *   `> 0.0` (e.g., `0.1`): Keeps a percentage of the previous step's gradient. This creates a "momentum" over time, effectively simulating a larger batch size or longer temporal context. Useful for difficult convergence landscapes.

### Key Methods

#### `trainer.fit(...)`
Runs a full training loop with optional Darwinian Pruning.

```python
history = trainer.fit(
    input_features=X, 
    target_values=Y, 
    epochs=100, 
    batch_size=32, 
    thinking_steps=10,       # Temporal Depth
    pruning_threshold=0.0    # If > 0, prunes weak links every epoch
)
```

#### `trainer.train_batch(...)`
Runs a single custom training step. Useful for custom loops (RL, Generative, etc.).
*   `thinking_steps`: How long the model "thinks" before loss is calculated.
*   `gradient_accumulation_steps`: Simulates larger batch sizes.

#### `trainer.predict(input_features, thinking_steps)`
Runs inference in evaluation mode (no gradients, no dropout).

---

## 🌟 Advanced Capabilities

### 1. Space-Time Tradeoff (Thinking Steps)
RealNet replaces layers with time. 
*   **Standard NN**: 10 Layers = Fixed Depth.
*   **RealNet**: You can choose `thinking_steps=5`, `10`, or `100` at runtime.
    *   **Low Steps**: Fast, shallow reasoning.
    *   **High Steps**: Slow, deep reasoning (equivalent to dozens of layers).

### 2. Gradient Accumulation (Virtual Batch Size)
RealNet allows you to simulate massive batch sizes on limited hardware (e.g., consumer GPUs).
*   **How it works:** Instead of updating weights after every batch, it accumulates gradients for `N` steps and then performs a single update.
*   **Usage:**
    ```python
    # Simulates a batch size of 32 * 4 = 128
    trainer.train_batch(x, y, thinking_steps=10, gradient_accumulation_steps=4)
    ```
*   **Benefit:** Allows training large models or using large batch stability without running out of VRAM.

### 3. Ghost Gradients (Gradient Decay)
By setting `gradient_decay > 0`, you enable a form of **Temporal Momentum**. The network "remembers" the direction of the error from the previous batch.
*   **Difference from Accumulation:** Accumulation is exact math (summing). Ghost Gradients is a decaying echo (multiplying by 0.1).
*   **Use Case:** When the loss curve is extremely jagged or the model gets stuck in local minima.
*   **Effect:** Smooths out optimization and can force convergence in "Impossible" tasks (like Zero-Hidden XOR).

### 4. Darwinian Evolution (Pruning)
RealNet can evolve. By calling `prune_synapses()` during training, the network kills off useless connections.
*   **Result**: You can end up with a model that has 95% dead connections (High Sparsity) but maintains 99% accuracy. This mimics the human brain's development (synaptic pruning).

---

## 🚀 Usage Examples

### Example 1: Solving XOR (The Chaos Gate)

```python
# 2 Inputs, 1 Output. 0 Hidden Layers.
model = RealNet(3, [0, 1], [2], device='cuda')
trainer = RealNetTrainer(model, gradient_decay=0.1) # Use Ghost Gradients

# Data: Truth Table
X = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
Y = [[-1], [1], [1], [-1]]

# Train with 5 Thinking Steps
trainer.fit(X, Y, epochs=100, thinking_steps=5)
```

### Example 2: Evolutionary Training (Self-Optimizing)

```python
# Train on a task, but kill weak neurons every epoch
trainer.fit(
    X_train, Y_train, 
    epochs=50, 
    thinking_steps=10, 
    pruning_threshold=0.01 # Kill weights < 0.01
)

# Convert to sparse for efficiency
model.make_sparse()
```
