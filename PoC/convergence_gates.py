import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import 'realnet' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from realnet import RealNet, RealNetTrainer

def main():
    print("Initializing RealNet 2.0: Logic Gates Demo (AND, OR, XOR)...")
    
    # Configuration
    # We need inputs for Logic A and Logic B.
    # We need outputs for AND, OR, XOR.
    # Let's assign specific neurons.
    
    # Input Neurons
    INPUT_A = 0
    INPUT_B = 1
    
    # Output Neurons
    OUTPUT_AND = 60
    OUTPUT_OR = 61
    OUTPUT_XOR = 62
    
    NUM_NEURONS = 64
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {DEVICE}")
    
    # 1. Initialize Model
    model = RealNet(
        num_neurons=NUM_NEURONS, 
        input_ids=[INPUT_A, INPUT_B], 
        output_ids=[OUTPUT_AND, OUTPUT_OR, OUTPUT_XOR], 
        pulse_mode=True, 
        device=DEVICE
    )
    
    trainer = RealNetTrainer(model, device=DEVICE)
    print(f"Architecture: {NUM_NEURONS} Neurons. Outputs mapped to: AND({OUTPUT_AND}), OR({OUTPUT_OR}), XOR({OUTPUT_XOR})")
    
    # 2. Prepare Logic Gate Dataset
    # Truth Table: A, B -> AND, OR, XOR
    # 0, 0 -> 0, 0, 0
    # 0, 1 -> 0, 1, 1
    # 1, 0 -> 0, 1, 1
    # 1, 1 -> 1, 1, 0
    
    # NOTE: In RealNet, we typically use -1 and 1 for binary states to clearly distinguish from 0 (silence).
    # Let's map Logic 0 -> -1.0, Logic 1 -> 1.0
    
    logic_data = [
        # A,   B      AND   OR    XOR
        (-1.0, -1.0,  -1.0, -1.0, -1.0), # 0, 0
        (-1.0,  1.0,  -1.0,  1.0,  1.0), # 0, 1
        ( 1.0, -1.0,  -1.0,  1.0,  1.0), # 1, 0
        ( 1.0,  1.0,   1.0,  1.0, -1.0), # 1, 1
    ]
    
    # Create Tensors
    inputs_list = []
    targets_list = []
    
    # Augment data to have more samples (repeat the truth table)
    # This helps stochastic gradient descent to average out better in batches
    for _ in range(50): 
        for row in logic_data:
            inputs_list.append([row[0], row[1]])
            targets_list.append([row[2], row[3], row[4]])
            
    inputs_val = torch.tensor(inputs_list, device=DEVICE)
    targets_val = torch.tensor(targets_list, device=DEVICE)
    
    print(f"Dataset size: {len(inputs_val)} samples.")

    # 3. Train
    # XOR is a non-linear problem, so it might need a bit more thinking time or epochs than Identity.
    history = trainer.fit(
        inputs_val, 
        targets_val, 
        epochs=100, # 100 epochs * (200/16) batches = ~1200 updates
        batch_size=16, 
        thinking_steps=10,
        verbose=True
    )

    print("\nTraining Complete.")
    print(f"Final Loss: {history[-1]:.6f}")
    
    # 4. Verify Truth Table
    print("\nVerifying Logic Gates Truth Table:")
    print(f"{'Input A':>8} {'Input B':>8} | {'AND':>8} {'OR':>8} {'XOR':>8}")
    print("-" * 50)
    
    test_data = torch.tensor([
        [-1.0, -1.0],
        [-1.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0]
    ], device=DEVICE)
    
    preds = trainer.predict(test_data, thinking_steps=10)
    
    for i in range(len(test_data)):
        a = test_data[i][0].item()
        b = test_data[i][1].item()
        
        # Helper to format output
        def fmt(x): return f"{x:.4f}"
        
        and_out = preds[i][0].item()
        or_out = preds[i][1].item()
        xor_out = preds[i][2].item()
        
        # Simple thresholding for display logic
        def logic_val(x): return "1" if x > 0 else "0"
        
        print(f"{a:>8.1f} {b:>8.1f} | {fmt(and_out):>8} {fmt(or_out):>8} {fmt(xor_out):>8}  --> Logic: {logic_val(and_out)} {logic_val(or_out)} {logic_val(xor_out)}")

if __name__ == "__main__":
    main()
