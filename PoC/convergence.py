import numpy as np
import random

class RealNet:
    def __init__(self, num_neurons, input_ids, output_ids, learning_rate=0.001, noise_filter=0.0000001, pulse_mode=True):
        self.num_neurons = num_neurons
        self.input_ids = input_ids
        self.output_ids = output_ids
        
        # Initialize weights between -2 and 2
        self.weights = np.random.uniform(-2, 2, (num_neurons, num_neurons))
        
        # Neuron values (current and previous)
        self.values = np.zeros(num_neurons)
        self.prev_values = np.zeros(num_neurons)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.noise_filter = noise_filter
        
        # Pulse Mode (default: True)
        # When enabled: Input is injected ONLY at the first step, Output is overwritten ONLY at the last step
        # When disabled (Continuous Mode): Input/Output are overwritten at EVERY step (legacy behavior)
        self.pulse_mode = pulse_mode
        
    def activation(self, x):
        """
        Modified Activation: ln(x) + shift_y clipped at 0.
        shift_y is derived from noise_filter: shift_y = -ln(noise_filter)
        This ensures that activation starts from 0 at x = noise_filter.
        """
        # Avoid log(0) issue if noise_filter is too small, though 1e-320 is min float
        safe_filter = max(self.noise_filter, 1e-300) 
        shift_y = -np.log(safe_filter)
        threshold = safe_filter
        
        out = np.zeros_like(x)
        mask = x > threshold
        out[mask] = np.log(x[mask]) + shift_y
        return out
    
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
        
        Pulse Mode (default):
            - Input is injected ONLY at the FIRST step (pulse in)
            - Output is overwritten ONLY at the LAST step (pulse out)
            - Between steps, the network "thinks" freely without locked signals
            
        Continuous Mode (pulse_mode=False):
            - Input/Output are overwritten at EVERY step (legacy behavior)
        """
        total_loss = 0.0
        steps_run = 0
        
        # Ensure at least 1 step is run
        steps_to_run = max(1, num_steps)
        
        for step_idx in range(steps_to_run):
            is_first_step = (step_idx == 0)
            is_last_step = (step_idx == steps_to_run - 1)
            
            # Determine if we should inject inputs/outputs this step
            if self.pulse_mode:
                # Pulse Mode: Input only at first, Output only at last
                inject_inputs = is_first_step
                inject_outputs = is_last_step
            else:
                # Continuous Mode: Always inject (legacy behavior)
                inject_inputs = True
                inject_outputs = True
            
            # 1. Calculate Sums (Pre-activation)
            # Each neuron receives sum of (prev_value * weight) from all other neurons
            # We use a temporary sum variable as per manifesto
            sums = np.dot(self.prev_values, self.weights)
            
            # 2. Handle Inputs (Overwrite Input Neurons) - based on mode
            if inject_inputs and inputs is not None:
                for i, val in inputs.items():
                    if i in self.input_ids:
                        sums[i] = val 

            # 3. Activation (ReLU)
            current_values = self.activation(sums)
            
            # Force inputs after activation to ensure they are exactly what is given
            if inject_inputs and inputs is not None:
                for i, val in inputs.items():
                    if i in self.input_ids:
                        current_values[i] = val

            # 4. Value Normalization [0, 1]
            # Keep track of min/max for indirect correlation calculation later
            v_min = np.min(current_values)
            v_max = np.max(current_values)
            current_values = self.normalize_values(current_values)
            
            # Re-force inputs after normalization (only if injecting this step)
            if inject_inputs and inputs is not None:
                for i, val in inputs.items():
                    if i in self.input_ids:
                        current_values[i] = val

            # 5. Dream Training (Overwrite Outputs) - based on mode
            step_loss = 0.0
            if mode == 'dream' and targets is not None and inject_outputs:
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
    num_neurons = 61
    input_ids = [0]
    output_ids = [60]
    net = RealNet(num_neurons, input_ids, output_ids, learning_rate=0.000376)
    
    print("Starting RealNet Convergence PoC...")
    print(f"Neurons: {num_neurons}, Input: {input_ids}, Output: {output_ids}")
    
    # Task: Simple Association / Pattern
    # Input: 1.0 -> Output: 1.0
    # Input: 0.0 -> Output: 0.0
    # We will alternate inputs and expect the network to learn the mapping.
    
    # Hyperparameters for Training
    num_samples = 2000  # Increased for better convergence
    steps_per_sample = 5 # Sweet spot for thinking time
    
    history = []
    
    print(f"Training for {num_samples} samples with {steps_per_sample} steps each...")

    for sample_idx in range(num_samples):
        # Alternate pattern per sample, not per step
        val = 1.0 if sample_idx % 2 == 0 else 0.0
        inputs = {0: val}
        targets = {output_ids[0]: val}
        
        # Run Step (Dream Mode for training)
        # We keep inputs and targets constant for 'steps_per_sample' duration
        outputs, loss = net.step(inputs=inputs, targets=targets, mode='dream', num_steps=steps_per_sample)
        
        # Monitor error
        if sample_idx % 50 == 0:
            print(f"Sample {sample_idx}: Avg Loss: {loss:.6f}, Weights Min/Max: {np.min(net.weights):.2f}/{np.max(net.weights):.2f}")

    print("\nTraining Complete. Testing with Thinking Time...")
    
    # Helper to run test
    def run_test(input_val, steps=steps_per_sample):
        print(f"\nTest Case: Input {input_val}")
        net.prev_values = np.zeros(num_neurons) # Reset state for clean test
        
        # Step 1: Inject Input with thinking time
        inputs = {input_ids[0]: input_val}
        
        # Run inference with thinking time
        out, _ = net.step(inputs=inputs, mode='inference', num_steps=steps)
        print(f"  Final Output after {steps} steps: {out[output_ids[0]]:.4f}")

    run_test(1.0)
    run_test(0.0)

if __name__ == "__main__":
    main()
