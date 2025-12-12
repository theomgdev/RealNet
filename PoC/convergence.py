import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
import os

# Add parent directory to path to import 'realnet' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from realnet import RealNet, RealNetTrainer

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def main():
    print("Initializing RealNet 2.0: Modern Chaos Architecture (High-Level API)...")
    
    # Configuration
    NUM_NEURONS = 64
    INPUT_ID = 0
    OUTPUT_ID = 63
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {DEVICE}")
    
    # Hyperparameters
    THINKING_STEPS = 10 
    BATCH_SIZE = 16
    NUM_EPOCHS = 500
    
    # 1. Initialize Model
    model = RealNet(
        num_neurons=NUM_NEURONS, 
        input_ids=[INPUT_ID], 
        output_ids=[OUTPUT_ID], 
        pulse_mode=True, 
        device=DEVICE
    )
    
    # 2. Initialize Trainer
    trainer = RealNetTrainer(model, device=DEVICE)
    print(f"Architecture: {NUM_NEURONS} Neurons, AdamW, GELU, StepNorm")
    
    print("\nStarting Training (Truncated BPTT)...")
    
    # 3. Prepare Data (Tensors)
    # Inputs: Random +/- 1.0 (Num_Samples, 1)
    num_samples = 200
    inputs_val = (torch.randint(0, 2, (num_samples, 1), device=DEVICE).float() * 2 - 1)
    targets_val = inputs_val # Identity mapping
    
    # 4. Train with Fit API
    history = trainer.fit(
        inputs_val, 
        targets_val, 
        epochs=NUM_EPOCHS, 
        batch_size=BATCH_SIZE, 
        thinking_steps=THINKING_STEPS,
        verbose=True
    )

    print("\nTraining Complete.")
    print(f"Final Loss: {history[-1]:.6f}")
    
    # Verification Test
    print("\nVerifying with Inference...")
    test_inputs = torch.tensor([[1.0], [-1.0]], device=DEVICE)
    
    preds = trainer.predict(test_inputs, thinking_steps=THINKING_STEPS)
    
    for i in range(len(test_inputs)):
        val = test_inputs[i].item()
        pred = preds[i].item()
        print(f"Input: {val} -> Output: {pred:.4f} (Target: {val})")

if __name__ == "__main__":
    main()
