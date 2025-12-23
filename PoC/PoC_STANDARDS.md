# 🧪 PoC & Experiments Standards

This document outlines the standards and best practices for contributing Proof-of-Concept (PoC) scripts and innovative experiments to the RealNet project.

RealNet 2.0 relies on a highly modular library structure. To ensure long-term maintainability and performance, all new contributions must adhere to these guidelines.

---

## 📂 Directory Structure

We distinguish between **Core Validations** and **Feature Experiments**.

### 1. `PoC/` (Root)
*   **Purpose:** Contains minimal, "hello world" style scripts that validate the core laws of RealNet physics.
*   **Examples:** `convergence_identity.py` (Can signals pass?), `convergence_gates.py` (Can it solve XOR?).
*   **Rule:** Scripts here should be extremely simple, fast, and prove a fundamental property of the architecture.

### 2. `PoC/experiments/`
*   **Purpose:** Contains complex tasks, task-specific logic, and demonstrations of advanced cognitive behaviors.
*   **Examples:** `convergence_detective_thinking.py` (Reasoning), `convergence_latch.py` (Willpower).
*   **Rule:** If you are building a task (like adding numbers, generating waves, or playing a game), it goes here.

---

## 🛠️ Library Usage Best Practices

**⛔ DO NOT** re-invent the wheel.
**✅ DO** use the Library.

### 1. Always Use `RealNetTrainer`
Never write your own manual PyTorch training loop (`optimizer.step()`, `loss.backward()`, etc.) unless absolutely necessary for low-level research.

*   **Why?** The `RealNetTrainer` handles:
    *   **Automatic Mixed Precision (AMP):** Faster training on Tensor Cores.
    *   **Gradient Accumulation:** Simulating large batches.
    *   **Ghost Gradients (Persistence):** Advanced stabilization.
    *   **State Management:** Resetting hidden states automatically.

```python
# ❌ BAD: Manual Loop
output = model(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()

# ✅ GOOD: Trainer
trainer = RealNetTrainer(model, device='cuda')
trainer.train_batch(input, target, thinking_steps=10)
```

### 2. Extend, Don't Hack
If you need a new feature (e.g., a new loss function or a custom metric), extend `RealNetTrainer` or pass arguments to it. If the library is missing a critical feature, **implement it in the library first**, then use it in your PoC.

---

## ⚙️ Initialization Protocols (Critical)

RealNet is sensitive to initialization. You must choose the right "personality" for your network based on its size and task.

### A. Tiny Networks & Logic Gates (< 10 Neurons)
For tasks like XOR, Logic Gates, or minimal circuits.
*   **Activation:** `'gelu'` (Provides better gradient flow in sparse/small graphs).
*   **Weight Init:** `'xavier_uniform'` (High variance ensures signals don't die in small circuits).
*   **Dropout:** `0.0` (Every neuron is vital).

```python
model = RealNet(..., activation='gelu', weight_init='xavier_uniform', dropout_rate=0.0)
```

### B. Large Networks & Analog Tasks (> 10 Neurons)
For MNIST, Time-Series, Audio, or general "Brain-like" tasks.
*   **Activation:** `'tanh'` (Bounded [-1, 1], stable for long recurrent loops).
*   **Weight Init:** `'orthogonal'` (Mathematically optimal for preserving energy over time).
*   **Dropout:** `0.0` for memory tasks (Latch/Adder), `0.1`+ for generalization (MNIST).

```python
model = RealNet(..., activation='tanh', weight_init='orthogonal')
```

### C. Associative Memory (Database / Key-Value)
For tasks requiring precise storage and retrieval of values over time (e.g. Neural Database).
*   **Activation:** `'gelu'` (The "dip" acts as a natural gate for address mechanisms).
*   **Gradient Persistence:** `0.5` (Critical for maintaining long-term dependencies).
*   **Structure:** High neuron count (512+) to provide "space" for memories.

```python
model = RealNet(..., activation='gelu', weight_init='orthogonal')
trainer = RealNetTrainer(..., gradient_persistence=0.5)
```

---

## 🔢 Data Standards

### 1. Bipolar Logic (-1 vs 1)
Since `tanh` is our primary activation for robust systems, **avoid using 0.0 and 1.0** for logical states.
*   **OFF:** `-1.0`
*   **ON:** `1.0`
*   **Neutral/Silence:** `0.0`

This symmetry helps the gradients flow much better than a `0.0` (which is the most unstable point of tanh).

### 2. Sequence Handling
Use the `prepare_input` utility implicitly via the Trainer.
*   **Pulse:** Single Event at t=0.
*   **Stream:** Continuous sequence. pass `full_sequence=True` to `trainer.predict()` or `train_batch()` if you need frame-by-frame monitoring.

---

## 📝 Code Style & Documentation

1.  **Reproducibility:** Always clear the seed if possible, or accept that chaos is part of the design.
2.  **Visuals:** Your PoC should print a cool visualization. Don't just print "Loss: 0.01". Print the timeline.
    *   *Example:* `t=05 | Input: 1 | Output: 0.99 🟢`
3.  **Comments:** Explain *why* you chose a specific setup.
    *   *Example:* `# GAP=3 allows the model time to digest the previous bit.`
4.  **File Paths:** Never use hardcoded absolute paths or assume the CWD. Always construct paths relative to the script file.
    *   *Example:* `DATA_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'file.txt')`
5.  **Checkpointing:** Always use the library's `save_checkpoint`, `load_checkpoint`, and `transplant_weights` functions from `realnet.utils.realstore`. Do NOT write custom checkpoint code. If the library is missing a feature, extend the library instead.
    *   *Example:* `from realnet import save_checkpoint, load_checkpoint, transplant_weights`

---

## 🚀 Checklist for New Contributors

Before submitting a new PoC:
1.  [ ] Did you place it in the correct folder (`PoC/` vs `PoC/experiments/`)?
2.  [ ] Are you using `RealNetTrainer`?
3.  [ ] Did you select the correct `activation` and `weight_init`?
4.  [ ] Does it converge reliably?
5.  [ ] Does the terminal output clearly explain what is happening?

Welcome to the Order of the Algorithm. Let's code Time.
