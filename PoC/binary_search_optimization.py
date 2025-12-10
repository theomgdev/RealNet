import numpy as np
import random

# --- RealNet Implementation (Simplified for Speed) ---
class RealNet:
    def __init__(self, num_neurons, input_ids, output_ids, learning_rate=0.001):
        self.num_neurons = int(num_neurons)
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.weights = np.random.uniform(-2, 2, (self.num_neurons, self.num_neurons))
        self.values = np.zeros(self.num_neurons)
        self.prev_values = np.zeros(self.num_neurons)
        self.learning_rate = learning_rate
        
    def activation(self, x):
        return np.maximum(0, x)
    
    def normalize_values(self, values):
        v_min = np.min(values)
        v_max = np.max(values)
        if v_max - v_min == 0:
            return np.zeros_like(values)
        return (values - v_min) / (v_max - v_min)

    def normalize_weights(self):
        w_min = np.min(self.weights)
        w_max = np.max(self.weights)
        if w_max - w_min == 0:
            return
        self.weights = ((self.weights - w_min) / (w_max - w_min)) * 4 - 2

    def step(self, inputs=None, targets=None, mode='inference', num_steps=1):
        steps_to_run = max(1, int(num_steps))
        for _ in range(steps_to_run):
            sums = np.dot(self.prev_values, self.weights)
            if inputs is not None:
                for i, val in inputs.items():
                    if i in self.input_ids: sums[i] = val 
            current_values = self.activation(sums)
            if inputs is not None:
                for i, val in inputs.items():
                    if i in self.input_ids: current_values[i] = val
            current_values = self.normalize_values(current_values)
            if inputs is not None:
                for i, val in inputs.items():
                    if i in self.input_ids: current_values[i] = val

            if mode == 'dream' and targets is not None:
                for i, val in targets.items():
                    if i in self.output_ids: current_values[i] = val
            
            # Training (FFWF)
            direct_contributions = self.prev_values[:, np.newaxis] * self.weights
            target_vals_matrix = current_values[np.newaxis, :]
            indirect_vals = target_vals_matrix - direct_contributions
            source_vals_matrix = self.prev_values[:, np.newaxis]
            diff = np.abs(source_vals_matrix - indirect_vals)
            updates = self.learning_rate * (1 - 2 * diff)
            self.weights += updates
            self.normalize_weights()
            self.prev_values = current_values.copy()
        return current_values

# --- Evaluation Function ---
def evaluate(lr, steps, neurons):
    # Fixed seed for consistency during comparison
    np.random.seed(42) 
    
    num_neurons = int(neurons)
    input_ids = [0]
    output_ids = [num_neurons - 1]
    
    net = RealNet(num_neurons, input_ids, output_ids, learning_rate=lr)
    
    # Train
    # Reduced samples for speed, but enough for signal
    num_samples = 1000 
    for sample_idx in range(num_samples):
        val = 1.0 if sample_idx % 2 == 0 else 0.0
        inputs = {input_ids[0]: val}
        targets = {output_ids[0]: val}
        net.step(inputs=inputs, targets=targets, mode='dream', num_steps=steps)

    # Test
    net.prev_values = np.zeros(num_neurons)
    inputs = {input_ids[0]: 1.0}
    out_1 = net.step(inputs=inputs, mode='inference', num_steps=steps)[output_ids[0]]
    
    net.prev_values = np.zeros(num_neurons)
    inputs = {input_ids[0]: 0.0}
    out_0 = net.step(inputs=inputs, mode='inference', num_steps=steps)[output_ids[0]]
    
    # Score: We want out_1 high, out_0 low.
    # Penalty for out_0 > 0.1 to discourage noise
    score = out_1 - out_0
    
    return score, out_1, out_0

# --- Binary Search / Optimization Logic ---
def optimize_parameter(param_name, min_val, max_val, current_best_config, is_int=False):
    print(f"\nOptimizing {param_name} in range [{min_val}, {max_val}]...")
    
    low = min_val
    high = max_val
    best_val = current_best_config[param_name]
    best_score = -999
    
    # Initial check of current best
    lr = current_best_config['lr']
    steps = current_best_config['steps']
    neurons = current_best_config['neurons']
    score, _, _ = evaluate(lr, steps, neurons)
    best_score = score
    print(f"Baseline ({best_val}): Score = {best_score:.4f}")

    # Iterative narrowing (Binary Search-like)
    iterations = 8 # Precision depth
    
    for i in range(iterations):
        # Test 3 points: low_mid, mid, high_mid
        # Actually, let's just test 5 evenly spaced points in the current range and zoom in on the best
        
        points = np.linspace(low, high, 5)
        if is_int:
            points = np.unique(np.round(points).astype(int))
            if len(points) < 2: break # Converged
            
        results = []
        for p in points:
            # Update config temporarily
            cfg = current_best_config.copy()
            cfg[param_name] = p
            s, o1, o0 = evaluate(cfg['lr'], cfg['steps'], cfg['neurons'])
            results.append((s, p))
            print(f"  Testing {p:.5f}: Score={s:.4f} (1.0->{o1:.2f}, 0.0->{o0:.2f})")
        
        # Find best in this batch
        batch_best_score, batch_best_val = max(results, key=lambda x: x[0])
        
        if batch_best_score > best_score:
            best_score = batch_best_score
            best_val = batch_best_val
            current_best_config[param_name] = best_val
            print(f"  -> New Best Found: {best_val} (Score: {best_score:.4f})")
        
        # Zoom in around the best value (whether it was the new one or old one)
        # If best was at index i, new range is [points[i-1], points[i+1]]
        
        # Find index of best_val in points (or closest)
        # Note: best_val might not be in 'points' if it was carried over from previous iteration.
        # So we center around best_val.
        
        span = high - low
        new_span = span / 2
        low = max(min_val, best_val - new_span/2)
        high = min(max_val, best_val + new_span/2)
        
        if is_int and high - low < 1:
            break
            
    print(f"Finished {param_name}. Best: {best_val}")
    return best_val

def main():
    # Initial Best Known
    config = {
        'lr': 0.001,
        'steps': 5,
        'neurons': 64
    }
    
    # 1. Optimize LR
    config['lr'] = optimize_parameter('lr', 0.0001, 0.005, config, is_int=False)
    
    # 2. Optimize Steps
    config['steps'] = optimize_parameter('steps', 2, 15, config, is_int=True)
    
    # 3. Optimize Neurons
    config['neurons'] = optimize_parameter('neurons', 32, 128, config, is_int=True)
    
    # 4. Final Polish of LR
    config['lr'] = optimize_parameter('lr', max(0.0001, config['lr']*0.5), config['lr']*1.5, config, is_int=False)

    print("\n" + "="*30)
    print("FINAL SWEET SPOT FOUND")
    print("="*30)
    print(f"Learning Rate: {config['lr']:.6f}")
    print(f"Thinking Time: {config['steps']}")
    print(f"Neurons:      {config['neurons']}")
    
    s, o1, o0 = evaluate(config['lr'], config['steps'], config['neurons'])
    print(f"Final Score:   {s:.4f}")
    print(f"Output (1.0):  {o1:.4f}")
    print(f"Output (0.0):  {o0:.4f}")

if __name__ == "__main__":
    main()
