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
        self.learning_rate = 0.001
        
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

    def step(self, inputs=None, targets=None, mode='inference', num_steps=1):
        """
        Executes RealNet for num_steps timesteps.
        mode: 'inference' or 'dream'
        num_steps: Number of timesteps to run with the given inputs/targets.
        """
        total_loss = 0.0
        steps_run = 0
        
        # Ensure at least 1 step is run
        steps_to_run = max(1, num_steps)
        
        for _ in range(steps_to_run):
            # 1. Calculate Sums (Pre-activation)
            # Each neuron receives sum of (prev_value * weight) from all other neurons
            # We use a temporary sum variable as per manifesto
            sums = np.dot(self.prev_values, self.weights)
            
            # 2. Handle Inputs (Overwrite Input Neurons)
            if inputs is not None:
                for i, val in inputs.items():
                    if i in self.input_ids:
                        sums[i] = val 

            # 3. Activation (ReLU)
            current_values = self.activation(sums)
            
            # Force inputs after activation to ensure they are exactly what is given
            if inputs is not None:
                for i, val in inputs.items():
                    if i in self.input_ids:
                        current_values[i] = val

            # 4. Value Normalization [0, 1]
            # Keep track of min/max for indirect correlation calculation later
            v_min = np.min(current_values)
            v_max = np.max(current_values)
            current_values = self.normalize_values(current_values)
            
            # Re-force inputs after normalization
            if inputs is not None:
                for i, val in inputs.items():
                    if i in self.input_ids:
                        current_values[i] = val

            # 5. Dream Training (Overwrite Outputs)
            step_loss = 0.0
            if mode == 'dream' and targets is not None:
                batch_errors = []
                for i, val in targets.items():
                    if i in self.output_ids:
                        # Calculate error before overwriting
                        batch_errors.append((val - current_values[i]) ** 2)
                        current_values[i] = val
                if batch_errors:
                    step_loss = np.mean(batch_errors)
            
            total_loss += step_loss
            steps_run += 1
            
            # 6. Training (FFWF)
            # Compare current_values (state t) with prev_values (state t-1)
            # Update weights based on correlation.
            
            # direct_contributions[i, j] = prev_values[i] * weights[i, j]
            direct_contributions = self.prev_values[:, np.newaxis] * self.weights # Shape (N, N)
            
            # target_vals is current_values (which includes forced dream values)
            # indirect_vals[i, j] = current_values[j] - direct_contributions[i, j]
            target_vals_matrix = current_values[np.newaxis, :] # Shape (1, N)
            indirect_vals = target_vals_matrix - direct_contributions
            
            # Calculate update
            # We compare Source (prev_values[i]) with Target (indirect_vals[i, j])
            source_vals_matrix = self.prev_values[:, np.newaxis] # Shape (N, 1)
            
            diff = np.abs(source_vals_matrix - indirect_vals)
            
            # Formula: delta = LR * (1 - 2 * diff)
            updates = self.learning_rate * (1 - 2 * diff)
            
            # Apply updates
            self.weights += updates
            
            # 7. Weight Normalization [-2, 2]
            self.normalize_weights()
            
            # 8. Prepare for next step
            self.prev_values = current_values.copy()
        
        avg_loss = total_loss / steps_run if steps_run > 0 else 0.0
        return current_values, avg_loss

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
    
    # Hyperparameters for Training
    num_samples = 500  # Total number of pattern switches
    steps_per_sample = 10 # How long to hold each pattern (Thinking time)
    
    history = []
    
    print(f"Training for {num_samples} samples with {steps_per_sample} steps each...")

    for sample_idx in range(num_samples):
        # Alternate pattern per sample, not per step
        val = 1.0 if sample_idx % 2 == 0 else 0.0
        inputs = {0: val}
        targets = {31: val}
        
        # Run Step (Dream Mode for training)
        # We keep inputs and targets constant for 'steps_per_sample' duration
        outputs, loss = net.step(inputs=inputs, targets=targets, mode='dream', num_steps=steps_per_sample)
        
        # Monitor error
        if sample_idx % 50 == 0:
            print(f"Sample {sample_idx}: Avg Loss: {loss:.6f}, Weights Min/Max: {np.min(net.weights):.2f}/{np.max(net.weights):.2f}")

    print("\nTraining Complete. Testing with Thinking Time...")
    
    # Helper to run test
    def run_test(input_val, steps=10):
        print(f"\nTest Case: Input {input_val}")
        net.prev_values = np.zeros(num_neurons) # Reset state for clean test
        
        # Step 1: Inject Input with thinking time
        inputs = {0: input_val}
        
        # Run inference with thinking time
        out, _ = net.step(inputs=inputs, mode='inference', num_steps=steps)
        print(f"  Final Output after {steps} steps: {out[31]:.4f}")

    run_test(1.0)
    run_test(0.0)

if __name__ == "__main__":
    main()
