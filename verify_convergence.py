import math
import random
import statistics

class Neuron:
    def __init__(self, id):
        self.id = id
        self.value = 0.0
        self.accumulator = 0.0
        self.count = 0
        self.max_val = 0.1 # Initial small values to avoid div by zero issues
        self.min_val = -0.1
        self.mean_val = 0.0
        
        # Constants
        self.k = 3.0 # Golden ratio approx (as per manifesto)
        self.epsilon = 0.000001

    def activation_function(self, x):
        # Update stats
        self.accumulator += abs(x) # Manifesto says "+-ateşleme değeri", assuming abs for magnitude accumulation or signed? 
        # "ortalama ateşleme değeri her ateşleme sonrası +-ateşleme değeri bölü... ortalamaya eklenerek"
        # Let's assume it tracks the signed mean.
        # Re-reading: "ortalama ateşleme değeri... birikimli olarak... hesaplanır"
        # Let's track simple running mean of the *input* x or the *output* y? 
        # Formula uses x_ort (mean of x). So we track x.
        
        self.count += 1
        self.mean_val = self.mean_val + (x - self.mean_val) / self.count
        
        if x > self.max_val: self.max_val = x
        if x < self.min_val: self.min_val = x
        
        x_ort = self.mean_val
        x_max = self.max_val
        x_min = self.min_val
        k = self.k
        
        # The Formula
        # y = tanh( k * (x - x_ort) / ( (x_max - x_min)/2 + (x_max + x_min - 2*x_ort)/2 * (x - x_ort) / (abs(x - x_ort) + epsilon) ) ) / tanh(k)
        
        numerator_inner = x - x_ort
        
        # Sign term: (x - x_ort) / (abs(x - x_ort) + epsilon) -> approx +1 or -1
        sign_term = numerator_inner / (abs(numerator_inner) + self.epsilon)
        
        denom_part1 = (x_max - x_min) / 2
        denom_part2 = (x_max + x_min - 2 * x_ort) / 2
        
        denominator = denom_part1 + denom_part2 * sign_term
        
        # Avoid division by zero in the main fraction
        if abs(denominator) < self.epsilon:
            denominator = self.epsilon if denominator >= 0 else -self.epsilon

        inner_term = k * (numerator_inner / denominator)
        
        y = math.tanh(inner_term) / math.tanh(k)
        return y

class RealNet:
    def __init__(self, num_neurons, input_ids, output_ids):
        self.neurons = [Neuron(i) for i in range(num_neurons)]
        self.input_ids = input_ids
        self.output_ids = output_ids
        
        # Weights: weights[source][target]
        self.weights = {}
        for i in range(num_neurons):
            self.weights[i] = {}
            for j in range(num_neurons):
                # Random weights between -2 and 2
                self.weights[i][j] = random.uniform(-2, 2)
                
        # Connection values for next step: connection_values[source][target]
        self.connection_values = {}
        for i in range(num_neurons):
            self.connection_values[i] = {}
            for j in range(num_neurons):
                self.connection_values[i][j] = 0.0
                
        self.prev_activations = [0.0] * num_neurons
        self.learning_rate = 0.01 # "Çok düşük öğrenme katsayısı tavsiye edilir"

    def inference(self, input_values, target_values=None, dream_mode=False):
        # 1. Gather inputs
        current_inputs = [0.0] * len(self.neurons)
        
        # Add external inputs
        for i, val in enumerate(input_values):
            if i < len(self.input_ids):
                neuron_id = self.input_ids[i]
                current_inputs[neuron_id] += val
                
        # Add internal connection values
        for src in range(len(self.neurons)):
            for tgt in range(len(self.neurons)):
                current_inputs[tgt] += self.connection_values[src][tgt]
                self.connection_values[src][tgt] = 0.0 # Reset after consumption

        # 2. Dream Training / Forcing Outputs
        # If dream_mode and targets exist, force output neurons to target values
        # Note: In RealNet, we might force the *activation* or the *input*?
        # "output nöronlarına datasetten alınan değerler yapay olarak ateşlenmiş gibi koyulduktan sonra"
        # Implies we override the activation result.
        
        # We need to calculate activations to know what "fired".
        # But training happens *before* propagation of the current step?
        # Let's calculate tentative activations first.
        
        current_activations = [0.0] * len(self.neurons)
        
        for i, neuron in enumerate(self.neurons):
            # Calculate activation
            # If forced (dream mode), override
            is_output = i in self.output_ids
            if dream_mode and target_values is not None and is_output:
                # Find which output index this is
                out_idx = self.output_ids.index(i)
                if out_idx < len(target_values):
                    # Force activation
                    current_activations[i] = target_values[out_idx]
                    # Also update neuron stats? Maybe.
                    neuron.activation_function(current_inputs[i]) # Just to update stats
            else:
                current_activations[i] = neuron.activation_function(current_inputs[i])

        # 3. Standard Training Step
        # Compare prev_activations (t-1) with current_activations (t)
        for src in range(len(self.neurons)):
            for tgt in range(len(self.neurons)):
                prev = self.prev_activations[src]
                curr = current_activations[tgt]
                
                # Determine correlation
                # Thresholds for "fired" could be 0, or a small epsilon.
                # Manifesto: "Pozitif ateşlenmiş", "Negatif ateşlenmiş", "Ateşlenmemiş" (0)
                
                delta = 0.0
                diff = abs(prev - curr) # "Fark arttıkça bağ zayıflar, fark azaldıkça bağ güçlenir" ??
                # Wait: "Fark arttıkça bağ zayıflar" -> This implies we want similar values?
                # But for Pos->Neg, difference is large (e.g. 1 - (-1) = 2).
                # "Pozitif ateşlenmiş -> Negatif ateşlenmiş = Weight negatife doğru güçlendirilir"
                # This means we want to capture the negative correlation.
                
                # Let's interpret "Fark" as the driving force magnitude.
                # Actually, let's stick to the explicit rules:
                
                # Rule 1: 0 -> Any OR Any -> 0 => Decay to 0
                if abs(prev) < 0.1 or abs(curr) < 0.1:
                    # Decay
                    if self.weights[src][tgt] > 0:
                        self.weights[src][tgt] -= self.learning_rate * 0.1
                        if self.weights[src][tgt] < 0: self.weights[src][tgt] = 0
                    elif self.weights[src][tgt] < 0:
                        self.weights[src][tgt] += self.learning_rate * 0.1
                        if self.weights[src][tgt] > 0: self.weights[src][tgt] = 0
                
                # Rule 2: Pos -> Pos OR Neg -> Neg => Strengthen Positive
                elif (prev > 0 and curr > 0) or (prev < 0 and curr < 0):
                    self.weights[src][tgt] += self.learning_rate * abs(curr) # Proportional to activation
                    
                # Rule 3: Pos -> Neg OR Neg -> Pos => Strengthen Negative
                elif (prev > 0 and curr < 0) or (prev < 0 and curr > 0):
                    self.weights[src][tgt] -= self.learning_rate * abs(curr)

                # Clamp weights? Manifesto doesn't explicitly say, but "Explosion" section implies unbound growth is a risk.
                # It suggests "indirect correlation" fix instead of clamping.
                # For this MVP, let's clamp loosely to prevent float overflow.
                if self.weights[src][tgt] > 10: self.weights[src][tgt] = 10
                if self.weights[src][tgt] < -10: self.weights[src][tgt] = -10

        # 4. Propagate
        for src in range(len(self.neurons)):
            val = current_activations[src]
            for tgt in range(len(self.neurons)):
                # "bağlantılar bu veriyi weight değeri ile çarpıp sıradaki timestep için bekletir"
                w = self.weights[src][tgt]
                self.connection_values[src][tgt] = val * w
        
        # Store state
        self.prev_activations = list(current_activations)
        
        return [current_activations[i] for i in self.output_ids]

def main():
    # Setup
    num_neurons = 10
    input_ids = [0, 1]
    output_ids = [8, 9]
    net = RealNet(num_neurons, input_ids, output_ids)
    
    # Task: XOR-like or simple association?
    # Let's try simple association first: Input [1, 0] -> Output [0, 1]
    # And Input [0, 1] -> Output [1, 0]
    
    data = [
        ([1.0, -1.0], [-1.0, 1.0]),
        ([-1.0, 1.0], [1.0, -1.0])
    ]
    
    print("Starting Training...")
    
    for epoch in range(100):
        total_error = 0
        
        for inputs, targets in data:
            # Run a few timesteps for each sample
            # "Dream Training"
            # Feed input
            # For a few steps, force output
            
            # Step 1: Input injection
            # We feed inputs for a few steps? Or just once?
            # "Ağın kaç timestepte bir çıktı vermesi isteniyorsa dream training step o sıklıkla çalıştırılır"
            
            # Sequence:
            # T=0: Apply Input.
            # T=1..N: Run network.
            
            # Reset network state? RealNet has memory, so maybe not.
            # But for a clean test, maybe we should let it settle or reset.
            # Let's just run continuous stream.
            
            # Dream Phase (Teaching)
            for _ in range(5):
                outputs = net.inference(inputs, targets, dream_mode=True)
            
            # Validation Phase (Testing immediately after? Or separate?)
            # Let's check error *during* dream (it's forced) or *after*?
            # To measure convergence, we should check what it *would* output without forcing.
            
            # Let's do a "Test" pass without forcing
            test_outputs = net.inference(inputs, dream_mode=False)
            
            # Calculate MSE
            mse = sum((t - o) ** 2 for t, o in zip(targets, test_outputs)) / len(targets)
            total_error += mse
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: MSE = {total_error:.4f}")
            
    print("Final Verification:")
    for inputs, targets in data:
        # Let it run for a bit to settle
        for _ in range(3):
            net.inference(inputs, dream_mode=False)
        outputs = net.inference(inputs, dream_mode=False)
        print(f"Input: {inputs}, Target: {targets}, Output: {[round(x, 2) for x in outputs]}")

if __name__ == "__main__":
    main()
