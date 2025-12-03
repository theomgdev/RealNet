# RealNet: A Brain-Inspired Non-Layered Neural Architecture

![License](https://img.shields.io/badge/license-MIT-blue.svg)

[🇹🇷 Türkçe Oku](./README_TR.md) | [📜 Original Manifesto (Old Text)](./MANIFESTO.md)

## Abstract

RealNet introduces a novel paradigm in artificial neural networks, diverging fundamentally from traditional layered architectures (FNNs, CNNs, Transformers). Inspired by the biological brain's chaotic yet efficient connectivity, RealNet employs a fully interconnected, non-layered topology where every neuron has the potential to connect with every other neuron. This architecture facilitates multi-dimensional data transmission (2D to 5D+) and enables the emergence of short-term memory through circular data loops and long-term memory through a unique "Fire Forward, Wire Forward" (FFWF) learning algorithm. RealNet demonstrates theoretical capabilities for active learning, dreaming, and self-organization without the constraints of static datasets.

## 1. Introduction

The prevailing approach in deep learning relies on structured layers and backpropagation. While effective, these methods often lack the dynamic adaptability and biological plausibility of natural neural systems. RealNet addresses these limitations by mimicking the brain's "soup of neurons" approach.

In RealNet, the concept of a "layer" is abolished. The network is a chaotic web of connections where directionality is emergent rather than imposed. This structure allows for:
*   **Dynamic Topology:** The network can adapt its effective structure based on the data flow.
*   **Temporal Processing:** Information is processed in continuous time-steps, allowing for complex temporal dependencies.
*   **Self-Organization:** The network refines its own connections based on activity correlations, similar to Hebbian learning but distinct in its temporal application.

## 2. Theoretical Architecture

### 2.1. Topology and Connectivity
The network consists of a collection of neurons and connections. Unlike FNNs, where connections exist only between adjacent layers, a RealNet neuron can receive input from and send output to any other neuron in the system.
*   **Chaotic Connectivity:** This allows for the formation of complex, recurrent structures.
*   **Circular Loops (Short-Term Memory):** Data can become trapped in feedback loops, effectively serving as a short-term memory buffer. These loops emit periodic signals, influencing the network's state over multiple time-steps.
*   **Dimensionality:** The connectivity pattern supports data representation in arbitrary dimensions, transcending the flat representations of traditional layers.

### 2.2. Neuron Dynamics
Each neuron maintains an internal state that evolves over time.
*   **Accumulated Statistics:** Neurons track their average, maximum, and minimum firing values.
*   **Adaptive Sensitivity:** These statistics are used to dynamically scale the activation function, ensuring the neuron remains sensitive to novel stimuli while habituating to repetitive background noise.

### 2.3. Activation Function: The Adaptive Tanh
RealNet utilizes a custom, adaptive activation function designed to handle the dynamic range of the network's signals. It incorporates dynamic scaling and normalization to prevent saturation and ensure efficient gradient flow (conceptually).

**Mathematical Formulation:**

$$y = \frac{\tanh\left( k \cdot \frac{x - x_{avg}}{ \frac{x_{max} - x_{min}}{2} + \frac{x_{max} + x_{min} - 2x_{avg}}{2} \cdot \text{sgn}(x - x_{avg}) } \right)}{\tanh(k)}$$

Where:
*   $x$ is the input value.
*   $x_{avg}, x_{max}, x_{min}$ are the running statistics of the neuron.
*   $k$ is a constant (Golden Ratio $\phi \approx 1.618$ or $3$ is recommended).
*   $\text{sgn}(z) = \frac{z}{|z| + \epsilon}$ is a differentiable sign function.

**Mechanism:**
1.  **Dynamic Scaling:** The denominator adjusts based on whether $x$ is above or below the average, effectively normalizing the input relative to the neuron's historical range.
2.  **Normalization:** The division by $\tanh(k)$ ensures that the output range is strictly $[-1, 1]$, even when inputs hit the historical extremes.
3.  **Habituation:** As a neuron fires consistently, its $x_{avg}$ shifts, and the function desensitizes the neuron to that steady state, filtering out repetitive noise and highlighting anomalies (spikes).

## 3. Algorithmic Core

### 3.1. Inference Engine
Inference in RealNet is a time-stepped process, not a single-pass propagation.
1.  **Accumulation:** Connections deliver buffered values from the previous time-step to target neurons.
2.  **State Update:** (Optional) A training step (Standard or Dream) is executed based on the current state.
3.  **Activation:** Neurons process accumulated inputs through the adaptive activation function.
4.  **Propagation:** Neurons reset and pass outputs to connections.
5.  **Transmission:** Connections multiply outputs by weights and buffer the result for the *next* time-step.

This separation of accumulation and transmission prevents race conditions and simulates parallel processing on sequential hardware.

### 3.2. Training Protocol: Fire Forward, Wire Forward (FFWF)
RealNet abandons backpropagation for a temporally-aware local learning rule.
*   **Concept:** Instead of "Fire Together, Wire Together" (spatial correlation), FFWF focuses on "Fire Forward, Wire Forward" (temporal causality). It reinforces connections where a neuron's firing *predicts* the firing of another neuron in the subsequent time-step.
*   **Mechanism:**
    *   **Positive Correlation:** If Neuron A (t-1) fires positive and Neuron B (t) fires positive, the weight $W_{AB}$ is increased.
    *   **Negative Correlation:** If Neuron A (t-1) fires positive and Neuron B (t) fires negative, $W_{AB}$ is decreased (inhibitory).
    *   **Decay:** Connections from non-firing neurons or to non-firing neurons are decayed towards zero, pruning irrelevant pathways.
*   **Weight Explosion Control:** Instead of simple weight decay, the algorithm adjusts weights based on the *indirect* contribution of the source to the target, preventing runaway feedback loops.

### 3.3. Dream Training (Distillation)
To converge without explicit supervision for every step, RealNet employs "Dream Training."
*   **Process:** The network is periodically disconnected from external input. The output neurons are clamped to desired values (from a dataset).
*   **Distillation:** The network runs internal cycles. The FFWF algorithm propagates these "dream" states backwards (causally), reinforcing pathways that would have naturally led to these outputs.
*   **Grounding:** This process grounds the abstract internal representations to the concrete target outputs, effectively distilling the chaotic short-term memory into structured long-term weights.

## 4. Convergence and Stability
Convergence in RealNet is defined as the formation of stable, predictive pathways that map temporal input patterns to desired output states.
*   **Self-Regulation:** The adaptive activation function naturally dampens over-active neurons.
*   **Pruning:** The FFWF algorithm continuously prunes weak connections, sparsifying the network.
*   **Future Prediction:** The network inherently learns to predict its own future states, minimizing internal surprise (free energy principle).

## 5. Vision and Future Directions

RealNet represents a step towards "Organic Artificial Intelligence." It is designed not just to classify static data, but to exist, experience, and adapt in a continuous data stream.

*   **Scalability:** The non-layered nature allows for seamless addition of new neurons without retraining the entire network.
*   **Inter-Network Communication:** Multiple RealNets can be connected directly, sharing internal states and "thoughts" without the need for encoding/decoding into discrete tokens.
*   **True Multimodality:** By processing data as raw signals in time, RealNet treats text, audio, and video as fundamentally the same: temporal patterns to be learned and predicted.

## License

This project is licensed under the MIT License.
