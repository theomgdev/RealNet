# 📘 RealNet Library Documentation

RealNet is a PyTorch-based library that implements **Zero-Hidden Layer** neural networks using **Temporal Depth**. By treating the neural network as a dynamic system that evolves over time, RealNet achieves deep learning capabilities without stacking spatial layers.

## Core Modules

The library is organized into three primary modules:
1.  **`realnet.core.network`**: The recurrent core architecture and update dynamics.
2.  **`realnet.training.trainer`**: Optimization engine with 8-bit support and bio-inspired regularization.
3.  **`realnet.utils`**: Data utilities, model persistence (`realstore`), and dynamic expansion (`neurogenesis`).

---

## RealNet Model (`realnet.core.network`)

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
    vocab_size=None,     # Optional: Decouples input/output size from neurons
    vocab_mode='hybrid'  # 'hybrid', 'discrete', or 'continuous'
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
*   `weight_init` (str): Initialization strategy (`'orthogonal'`, `'xavier_uniform'`, `'kaiming_normal'`, etc.). Default is `'orthogonal'`.
*   `activation` (str): Activation function used in the update step (`'tanh'`, `'relu'`, `'sigmoid'`, `'gelu'`, `'silu'`, etc.). Default is `'tanh'`.
*   `vocab_size` (int or list/tuple, optional): Size of the input/output vocabulary. 
    *   **Symmetric**: `vocab_size=50257` (GPT-2 style).
    *   **Asymmetric**: `vocab_size=[v_in, v_out]` (e.g., `[784, 10]` for MNIST to map 784 pixels to 10 classes).
    *   **Disable**: Use `-1` to disable one side (e.g., `[-1, 1000]` for direct neuron input but decoded output).
*   `vocab_mode` (str):
    *   `'hybrid'` (Default): Init both Embedding (for Int inputs) and Linear Projection (for Float inputs).
    *   `'discrete'`: Init only Embedding (Saves VRAM if only tokens are used).
    *   `'continuous'`: Init only Projection (Saves VRAM if only float vectors are used).

### Vocabulary Decoupling

When `vocab_size` is typically much larger than `num_neurons` (e.g., 50k vocab vs 1024 neurons), RealNet uses decoupled layers. This can be configured as symmetric (same size for in/out) or asymmetric.

1.  **Encoder (Input)**: Maps `v_in` -> `len(input_ids)` (Neurons).
    *   Integers (Tokens) use `nn.Embedding`.
    *   Floats (Vectors) use `nn.Linear` (Projection).
    *   *Disabled if `v_in == -1`.*
2.  **Decoder (Output)**: Maps `len(output_ids)` (Neurons) -> `v_out`.
    *   Uses `nn.Linear` (Decoding).
    *   *Disabled if `v_out == -1`.*

**Benefit:** This allows the "Thinking Core" (Neurons) to remain small and efficient while handling complex input formats or large output spaces without manual slicing.

```python
# Asymmetric Example: MNIST (784 pixels -> 10 classes)
model = RealNet(
    num_neurons=10,
    input_ids=range(10),
    output_ids=range(10),
    vocab_size=[784, 10], # Input 784, Output 10
    vocab_mode='continuous'
)
# No need for slice_output: model(x) returns (Batch, Steps, 10)
```

---

## Input Modalities and Data Handling

RealNet processes data through three distinct modalities. Choosing the right one is critical for performance and VRAM efficiency.

### 1. Pulse Mode (Impulse Computing)
**Use case**: Static data like images (MNIST) or single-shot logic (XOR).
*   **Behavior**: Set `pulse_mode=True`. Input is injected at $t=0$ only.
*   **Thinking**: The model continues computation for the specified number of `steps` without further input.
*   **VRAM Efficiency**: Optimal. Only (Batch, Neurons) is stored.

```python
# Image Classification (784 pixels -> 100 steps thinking)
model = RealNet(..., pulse_mode=True)
output = model(image_tensor, steps=100)
```

### 2. Continuous Mode (Static Control)
**Use case**: Control systems, VCO (Sine Wave), or real-time sensor monitoring.
*   **Behavior**: Set `pulse_mode=False`. The same input is injected at every time step $t$.
*   **Thinking**: The model state is constantly influenced by the static input.
*   **VRAM Efficiency**: High. Only (Batch, Neurons) is stored.

```python
# Frequency Control for Oscillator
model = RealNet(..., pulse_mode=False)
output = model(freq_input, steps=30)
```

### 3. Sequential Mode (Temporal Stretching)
**Use case**: Large Language Models (LLM), Time-Series, and reasoning agents.
*   **Behavior**: Provide a sequence `(Batch, Tokens)`. If `steps` > `tokens`, RealNet automatically scales the temporal resolution.
*   **Mechanism**: If 100 tokens are provided with 500 `steps`, the model intersperses 4 "silent" thinking steps between each token.
*   **VRAM Efficiency**: High. Eliminates the need for manually dilated/padded input tensors.

```python
# LLM: 128 tokens with 5 thinking steps per token (Total 640 steps)
tokens = torch.randint(0, 50257, (batch, 128))
output = model(tokens, steps=640)
```

#### Comparison of Sequential Input Formats
| Input Type | Format | Modality | Recommended Use Case |
| :--- | :--- | :--- | :--- |
| **Index (ID)** | `(Batch, Steps)` (Long) | Sequential | LLMs, Tokenized text. |
| **Dense** | `(Batch, Steps, Dim)` (Float) | Sequential | Audio, Video, Vector Streams. |
| **Pulse** | `(Batch, Dim)` (Float) | Instant | Static Images, Logic Gates. |
| **Continuous**| `(Batch, Dim)` (Float) | Periodic | Oscillators, Constant Signals. |

---

### Key Methods

#### `model.compile()`
Optimizes the model using `torch.compile` (PyTorch 2.0+) for faster execution. Returns the compiled model.

#### `model.forward(x_input, steps=1, current_state=None)`
Runs the dynamic system.
*   `x_input`: Input tensor. Can be a single pulse or a sequence (index-based or dense).
*   `steps`: **Thinking Time**. How many times the signal reverberates in the echo chamber.
*   `current_state`: Optional. Pass a previous state to continue from.
*   **Returns**: `(all_states, final_state)`
    *   `all_states`: Tensor of shape `(Batch, Steps, Neurons)` - **Batch-first format**.
    *   `final_state`: Tensor of shape `(Batch, Neurons)` - The last hidden state.

---

## RealNet Trainer (`realnet.training.trainer`)

The `RealNetTrainer` handles the training loop, gradient accumulation, mixed precision (AMP), and experimental features like Ghost Gradients.

### Initialization

```python
from realnet import RealNetTrainer

trainer = RealNetTrainer(
    model, 
    optimizer=None,      # Optional: Custom Optimizer
    loss_fn=None,        # Optional: Custom Loss Function (Default: MSELoss)
    lr=1e-4,             # Initial Learning Rate (Stored in trainer.initial_lr)
    device='cuda',
    gradient_persistence=0.0,   # Ghost Gradients (Persistence)
    synaptic_noise=1e-6         # Thermal Noise (Default: 1e-6)
)
```

**Auto-Optimization (bitsandbytes):**
*   If `optimizer` is `None` AND `device` is `'cuda'`, the trainer will attempt to load `bitsandbytes` and use `AdamW8bit`.
*   **VRAM Savings:** ~75% reduction in optimizer memory.
*   **Control Variables (Set before import):**
    *   `os.environ["NO_BNB"] = "1"`: Disables 8-bit optimizer (Forces standard Torch AdamW).
    *   `os.environ["VERBOSE_BNB"] = "1"`: Enables verbose loading logs for debugging.

**Parameters:**
*   `lr` (float): The initial learning rate. If no optimizer is provided, this LR is used to initialize one. It is also stored as `trainer.initial_lr` for use in Scheduler warm restarts or resets.
*   `gradient_persistence` (float): **Ghost Gradients / Persistence**.
    *   `0.0`: Standard behavior (`zero_grad()` after every step).
    *   `> 0.0` (e.g., `0.1`): Keeps a percentage of the previous step's gradient. This creates a "momentum" over time, effectively simulating a larger batch size or longer temporal context. Useful for difficult convergence landscapes.
*   `synaptic_noise` (float): **Thermal Noise**.
    *   Adds Gaussian noise (std dev = `synaptic_noise`) to all weights *before* every training step.
    *   Simulates biological thermal noise and prevents overfitting (Stochastic Resonance).
    *   **Default:** `1e-6` (Set to `0.0` for pure logic/math tasks).

### Key Methods

#### `trainer.fit(...)`
Runs a full training loop.

```python
history = trainer.fit(
    input_features=X, 
    target_values=Y, 
    epochs=100, 
    batch_size=32, 
    thinking_steps=10       # Temporal Depth
)
```

#### `trainer.train_batch(...)`
Runs a single custom training step. Useful for custom loops (RL, Generative, etc.).
*   `thinking_steps`: How long the model "thinks" before loss is calculated.
*   `gradient_accumulation_steps`: Simulates larger batch sizes.
*   `full_sequence` (bool): If `True`, calculates loss on the entire sequence output `(Batch, Steps, Out)` instead of just the last step. Essential for Seq2Seq tasks.
*   `mask` (Tensor, optional): A binary or weighted mask `(Batch, Steps, Out)` to ignore specific steps or outputs during loss calculation. Useful for tasks with "thinking delays" or variable-length sequences.
*   `output_transform` (Callable, optional): A function to transform the predicted outputs before loss calculation. Useful for reshaping logits (e.g., flatten for CrossEntropy) or applying custom activations.

#### `trainer.predict(input_features, thinking_steps, full_sequence=False)`
Runs inference in evaluation mode.
*   `full_sequence` (bool): If `True`, returns outputs for all time steps `(Batch, Steps, Out)`.

#### `trainer.regenerate_synapses(threshold=0.01)`
Triggers **Darwinian Regeneration**. Instead of pruning weak weights, this method **re-initializes** them.
*   **Logic**: If $|W| < threshold$, the synapse is considered "dead/useless". It is wiped and assigned a new random value using the model's original initialization strategy (e.g., Xavier/Orthogonal).
*   **Purpose**: Allows the network to escape local minima and constantly explore new pathways. Transforms "dead" capacity into "fresh" capacity.
*   **Returns**: `(revived_count, total_synapses)`

---

## Advanced Capabilities

### 1. Temporal Depth (Space-Time Tradeoff)
RealNet replaces spatial layers with temporal steps. 
*   **Vertical vs Horizontal**: A standard 10-layer network has fixed depth. RealNet can be run for 10 or 100 steps on-the-fly.
*   **Dynamic Complexity**: Higher `steps` allow the network more time to reverberate signals through its recurrent core, enabling deeper reasoning without increasing parameter count.

### 2. Gradient Accumulation (Virtual Batch Size)
RealNet allows you to simulate massive batch sizes on limited hardware (e.g., consumer GPUs).
*   **How it works:** Instead of updating weights after every batch, it accumulates gradients for `N` steps and then performs a single update.
*   **Usage:**
    ```python
    # Simulates a batch size of 32 * 4 = 128
    trainer.train_batch(x, y, thinking_steps=10, gradient_accumulation_steps=4)
    ```
*   **Benefit:** Allows training large models or using large batch stability without running out of VRAM.

### 3. Gradient Persistence (Ghost Gradients)
By setting `gradient_persistence > 0`, the network retains a fraction of the previous batch's gradient. 
*   **Mechanism**: Uses a decaying echo (linear scaling) of previous gradients.
*   **Use Case**: Smoothing optimization in non-convex landscapes or simulated long-context training.

### 4. Synaptic Regeneration (Darwinian Revive)
RealNet can re-initialize synapses that are no longer contributing to the loss signal (stagnant weights).
*   **Concept**: Instead of pruning, near-zero weights are re-initialized using the original weight strategy.
*   **Benefit**: Maximizes network plasticitity and parameter efficiency by converting dead capacity into fresh exploration.
*   **Usage**: 
    *   **Threshold Mode**: `trainer.regenerate_synapses(threshold=0.01)`
    *   **Percent Mode**: `trainer.regenerate_synapses(percentage=0.05)`

---

## Model Persistence (`realnet.utils.realstore`)

The `realstore` module provides checkpoint management utilities, including a unique **Weight Transplantation** feature for transferring learned knowledge between models of different sizes.

### Functions

#### `save_checkpoint(model, optimizer, epoch, loss, path, extra_data=None)`
Saves a training checkpoint to disk.

#### `load_checkpoint(model, optimizer, path, device='cpu', strict=True)`
Loads a checkpoint. Set `strict=False` to ignore size mismatches (will partially load what fits).

#### `transplant_weights(model, checkpoint_path, device='cpu', verbose=True)`
🧬 **Weight Transplantation**: Transfers learned weights from a checkpoint to a model, **even if the number of neurons is different**.

*   **Scaling Up**: Start a 512-neuron model with knowledge from a 256-neuron model. The overlapping 256×256 region is copied, the rest stays initialized.
*   **Scaling Down**: Compress a 1024-neuron model into a 256-neuron model. The most "central" 256×256 weights are preserved.
*   **Warm Starts**: Any learned weights are better than random. Gradients will find their way faster.

```python
from realnet import RealNet, transplant_weights

# Create a NEW, larger model
big_model = RealNet(num_neurons=512, ...)

# Transplant weights from a smaller, trained checkpoint
transplant_weights(big_model, 'small_model_checkpoint.pth')

# big_model now has a "warm start" - training will converge faster!
```

#### `get_checkpoint_info(path, device='cpu')`
Reads checkpoint metadata (epoch, loss, num_neurons) without loading into a model.

---

## Neurogenesis (Network Expansion)

RealNet supports dynamic growth, allowing you to add neurons to a live network during training. This mimics biological neurogenesis.

### `trainer.expand(amount=1, verbose=True)`
Dynamically adds `amount` empty neurons to the model.
*   **Continuity**: Optimizers are migrated, so momentum and history are preserved.
*   **State**: The training state is preserved.
*   **Initialization**: 
    *   **Incoming Weights**: 0 (Maintains forward pass stability, new neuron starts inactive).
    *   **Outgoing Weights**: Small random noise (Enables backpropagation / gradient flow).

```python
# Add 1 neuron if loss stagnates
if loss > prev_loss:
    trainer.expand(amount=1)
```

---

## Utilities (`realnet.utils`)

### 1. Data Utilities (`realnet.utils.data`)

#### `prepare_input(input_features, model_input_ids, num_neurons, device)`
Maps raw input features (numpy or tensor) to the full network state tensor.
*   **Pulse Mode:** Plugs data into `t=0`, leaves rest as 0.
*   **Stream Mode:** Maps sequence data `(Batch, Steps, Features)` to correct neurons.
*   **Auto-Device:** Automatically moves data to the model's device.

```python
x_in, batch_size = prepare_input(X_train, model.input_ids, model.num_neurons, 'cuda')
```

#### `to_tensor(data, device)`
Safely converts any list/array/int/float into a PyTorch tensor on the target device.

### 2. Neurogenesis (`realnet.utils.neurogenesis`)
See **Neurogenesis** section above.

### 3. RealStore (`realnet.utils.realstore`)
This module manages model serialization and the transdimensional weight transplantation feature described in the **Advanced Capabilities** section.

---

## Usage Examples

### Example 1: XOR Logic
```python
# 2 Inputs, 1 Output. 0 Hidden Layers.
model = RealNet(3, [0, 1], [2], device='cuda')
trainer = RealNetTrainer(model, gradient_persistence=0.1)

# Training logic...
trainer.fit(X, Y, epochs=100, thinking_steps=5)
```

### Example 2: MNIST Asymmetric Vocab
```python
# 784 pixels -> 10 neurons -> 10 logits
model = RealNet(num_neurons=10, input_ids=range(10), output_ids=range(10), vocab_size=[784, 10])
# Model handles projection and decoding automatically.
```