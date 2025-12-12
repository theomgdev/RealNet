import torch
import torch.nn as nn
import sys
import os
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from realnet import RealNet, RealNetTrainer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    print("RealNet 2.0: The Impossible XOR (Zero-Hidden)...")
    set_seed(42) # Reproducibility
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2 Inputs, 1 Output, 2 Chaos Buffers = 5 Neurons
    NUM_NEURONS = 5
    INPUT_IDS = [0, 1]
    OUTPUT_ID = [2]
    
    print(f"Neurons: {NUM_NEURONS} (25 Parameters)")
    
    # CRITICAL CONFIG FOR TINY NETWORKS:
    # 1. dropout_rate=0.0 (Every neuron is vital)
    model = RealNet(
        num_neurons=NUM_NEURONS, 
        input_ids=INPUT_IDS, 
        output_ids=OUTPUT_ID, 
        pulse_mode=True, 
        device=DEVICE,
        dropout_rate=0.0 
    )
    
    # CRITICAL FIX: Rescale Weights for Tiny Network
    # Default init (0.02) is designed for massive inputs (784+). 
    # For 2 inputs, signal dies immediately. We need larger variance.
    # Rule of thumb: 1 / sqrt(Inputs) -> 1 / sqrt(2) ~= 0.7
    with torch.no_grad():
        model.W.data = torch.randn(NUM_NEURONS, NUM_NEURONS, device=DEVICE) * 0.5
        # Also reset mask just in case
        model.mask.fill_(1.0)

    trainer = RealNetTrainer(model, device=DEVICE)
    
    # CRITICAL OPTIMIZER: NO WEIGHT DECAY
    # Small networks shouldn't be penalized for magnitude.
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    
    # XOR Data
    data = [
        (-1.0, -1.0, -1.0),
        (-1.0,  1.0,  1.0),
        ( 1.0, -1.0,  1.0),
        ( 1.0,  1.0, -1.0),
    ]
    
    # Batching (Create a dataset)
    inputs_list = []
    targets_list = []
    for _ in range(50): # 200 samples total
        for row in data:
            inputs_list.append([row[0], row[1]])
            targets_list.append([row[2]])
            
    inputs_val = torch.tensor(inputs_list, device=DEVICE)
    targets_val = torch.tensor(targets_list, device=DEVICE)

    print("Training...")
    # 15 Thinking steps to allow chaotic resonance to find the XOR pattern
    history = trainer.fit(inputs_val, targets_val, epochs=100, batch_size=16, thinking_steps=15)

    print(f"Final Loss: {history[-1]:.6f}")

    print("\nVerifying Truth Table:")
    print(f"{'A':>6} {'B':>6} | {'XOR (Pred)':>12} | {'Logic'}")
    print("-" * 40)
    
    test_data = torch.tensor([[-1.0,-1.0], [-1.0,1.0], [1.0,-1.0], [1.0,1.0]], device=DEVICE)
    preds = trainer.predict(test_data, thinking_steps=15)
    
    success = True
    for i in range(4):
        a = test_data[i][0].item()
        b = test_data[i][1].item()
        out = preds[i].item()
        
        # Logic check (-1 is 0, 1 is 1)
        expected = -1.0 if (a * b) > 0 else 1.0 
        logic_pred = "0" if out < 0 else "1"
        logic_target = "0" if expected < 0 else "1"
        
        status = "OK" if logic_pred == logic_target else "FAIL"
        if status == "FAIL": success = False
        
        print(f"{a:>6.1f} {b:>6.1f} | {out:>12.4f} | {logic_pred} (Target: {logic_target}) {status}")

    if success:
        print("\nResult: CONVERGED. XOR Solved with Zero Hidden Layers.")
    else:
        print("\nResult: FAILED. (Try different seed or more epochs)")

if __name__ == "__main__":
    main()
