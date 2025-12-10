import numpy as np
import random

class RealNet:
    def __init__(self, num_neurons, input_ids, output_ids):
        self.num_neurons = num_neurons
        self.input_ids = input_ids
        self.output_ids = output_ids
        
        # Initialize weights between -2 and 2
        self.weights = np.random.uniform(-2, 2, (num_neurons, num_neurons))
        
        # Neuron values (current and previous)
        self.values = np.zeros(num_neurons)
        self.prev_values = np.zeros(num_neurons)
        
        # Hyperparameters
        self.learning_rate = 0.01
        
    def activation(self, x):
        """ReLU Activation"""
        return np.maximum(0, x)
    
    def normalize_values(self, values):
        """Normalize values to [0, 1] based on population min/max"""
        v_min = np.min(values)
        v_max = np.max(values)
        if v_max - v_min == 0:
            return np.zeros_like(values)
        return (values - v_min) / (v_max - v_min)

    def normalize_weights(self):
        """Normalize all weights to [-2, 2] based on population min/max"""
        w_min = np.min(self.weights)
        w_max = np.max(self.weights)
        if w_max - w_min == 0:
            return # Avoid division by zero, though unlikely with random init
        
        # Scale to [0, 1] then to [-2, 2]
        self.weights = ((self.weights - w_min) / (w_max - w_min)) * 4 - 2

    def step(self, inputs=None, targets=None, mode='inference'):
        """
        Executes one timestep of RealNet.
        mode: 'inference' or 'dream'
        """
        # 1. Calculate Sums (Pre-activation)
        # Each neuron receives sum of (prev_value * weight) from all other neurons
        # We use a temporary sum variable as per manifesto
        sums = np.dot(self.prev_values, self.weights)
        
        # 2. Handle Inputs (Overwrite Input Neurons)
        # "Input neuron carries exactly whatever comes from the external world"
        # We do this BEFORE activation/normalization so it propagates correctly?
        # Manifesto says: "This addition process is not performed for neurons marked as Input... external data overwrites this total."
        if inputs is not None:
            for i, val in inputs.items():
                if i in self.input_ids:
                    sums[i] = val # Overwrite the sum effectively, or just the value after?
                    # "Input neuron carries exactly whatever comes from the external world"
                    # If we overwrite the sum, it passes through ReLU. If input is -1, ReLU makes it 0.
                    # If input is meant to be raw value, we should probably set it after activation?
                    # "Negative inputs (inhibitory signals) completely silence the neuron (0)."
                    # So inputs are subject to ReLU? 
                    # "Input neuron carries exactly whatever comes from the external world (0 if no data)."
                    # Let's assume inputs are positive [0, 1] usually.
                    pass

        # 3. Activation (ReLU)
        current_values = self.activation(sums)
        
        # Force inputs after activation to ensure they are exactly what is given (if we interpret "overwrites" strictly)
        if inputs is not None:
            for i, val in inputs.items():
                if i in self.input_ids:
                    current_values[i] = val

        # 4. Value Normalization [0, 1]
        # Keep track of min/max for indirect correlation calculation later
        v_min = np.min(current_values)
        v_max = np.max(current_values)
        current_values = self.normalize_values(current_values)
        
        # Re-force inputs after normalization? 
        # Manifesto: "Input neuron carries exactly whatever comes from the external world"
        # If we normalize, the input value might shift. 
        # Let's assume inputs are already in [0, 1] and should stay that way relative to others?
        # Or maybe inputs are immune to normalization?
        # Let's re-apply inputs to be safe, as they are "external truth".
        if inputs is not None:
            for i, val in inputs.items():
                if i in self.input_ids:
                    current_values[i] = val

        # 5. Dream Training (Overwrite Outputs)
        if mode == 'dream' and targets is not None:
            for i, val in targets.items():
                if i in self.output_ids:
                    current_values[i] = val
        
        # 6. Training (FFWF)
        # Compare current_values (state t) with prev_values (state t-1)
        # Update weights based on correlation.
        
        # We need to calculate 'indirect' activation for the target neuron.
        # "contribution of the source neuron to the target neuron is calculated and subtracted"
        # We assume this subtraction happens in the value space.
        
        # direct_contributions[i, j] = prev_values[i] * weights[i, j]
        direct_contributions = self.prev_values[:, np.newaxis] * self.weights # Shape (N, N)
        
        # target_vals is current_values (which includes forced dream values)
        # indirect_vals[i, j] = current_values[j] - direct_contributions[i, j]
        target_vals_matrix = current_values[np.newaxis, :] # Shape (1, N)
        indirect_vals = target_vals_matrix - direct_contributions
        
        # Calculate update
        # "Strengthening/Weakening Amount is proportional to the difference in the firing values"
        # We compare Source (prev_values[i]) with Target (indirect_vals[i, j])
        
        source_vals_matrix = self.prev_values[:, np.newaxis] # Shape (N, 1)
        
        diff = np.abs(source_vals_matrix - indirect_vals)
        
        # Logic:
        # Small diff -> Strengthen (+)
        # Large diff -> Weaken (-)
        # Formula: delta = LR * (1 - 2 * diff)
        
        updates = self.learning_rate * (1 - 2 * diff)
        
        # Apply updates
        self.weights += updates
        
        # 7. Weight Normalization [-2, 2]
        self.normalize_weights()
        
        # 8. Prepare for next step
        self.prev_values = current_values.copy()
        
        return current_values

def main():
    # Setup
    num_neurons = 32
    input_ids = [0]
    output_ids = [31]
    net = RealNet(num_neurons, input_ids, output_ids)
    
    print("Starting RealNet Convergence PoC...")
    print(f"Neurons: {num_neurons}, Input: {input_ids}, Output: {output_ids}")
    
    # Task: Simple Association / Pattern
    # Input: 1.0 -> Output: 1.0
    # Input: 0.0 -> Output: 0.0
    # We will alternate inputs and expect the network to learn the mapping.
    
    epochs = 2000
    history = []
    
    for i in range(epochs):
        # Alternate pattern
        val = 1.0 if i % 2 == 0 else 0.0
        inputs = {0: val}
        targets = {31: val}
        
        # Run Step (Dream Mode for training)
        outputs = net.step(inputs=inputs, targets=targets, mode='dream')
        
        # Monitor error (MSE on output neuron)
        if i % 100 == 0:
            # Test Inference
            print(f"Epoch {i}: Weights Min/Max: {np.min(net.weights):.2f}/{np.max(net.weights):.2f}")
            # print(f"Sample Weights (Input->First): {net.weights[0, 1:5]}")

    print("\nTraining Complete. Testing...")
    
    # Test 1: Input 1.0
    # We need to run a few steps to let signal propagate? 
    # RealNet is temporal. Input at T affects Output at T+1 (or later depending on path).
    # Since it's fully connected, T+1 should show some effect.
    
    # Reset network state (optional, but good for clean test)
    net.prev_values = np.zeros(num_neurons)
    
    print("\nTest Case: Input 1.0")
    net.step(inputs={0: 1.0}, mode='inference') # T=1 (Input injected)
    out_t1 = net.step(inputs={0: 0.0}, mode='inference') # T=2 (Signal propagates)
    print(f"Output at T+1: {out_t1[31]:.4f}")
    
    print("\nTest Case: Input 0.0")
    net.prev_values = np.zeros(num_neurons) # Reset
    net.step(inputs={0: 0.0}, mode='inference')
    out_t1 = net.step(inputs={0: 0.0}, mode='inference')
    print(f"Output at T+1: {out_t1[31]:.4f}")

if __name__ == "__main__":
    main()
