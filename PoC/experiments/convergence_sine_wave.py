
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import math

# Adjust path to import realnet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realnet import RealNet

def generate_sine_data(batch_size, steps, device):
    """
    Generates batch of data.
    Input: Frequency multiplier (random between 0.1 and 0.5)
    Target: Sine wave sequence for that frequency.
    """
    # Random frequencies: shape (Batch, 1)
    frequencies = torch.rand(batch_size, 1, device=device) * 0.4 + 0.1
    
    # Time steps: shape (Steps, 1)
    t = torch.arange(1, steps + 1, device=device).float().unsqueeze(1)
    
    # Target Sequence: sin(w * t)
    # We need shape (Steps, Batch, 1)
    # frequencies -> (1, Batch, 1)
    # t -> (Steps, 1, 1) -> Broadcast fails?
    
    # Let's do explicit expansion
    # frequencies: (Batch, 1)
    freqs_expanded = frequencies.unsqueeze(0).expand(steps, batch_size, 1)
    
    # time: (Steps, 1)
    time_expanded = t.expand(steps, batch_size).unsqueeze(2)
    
    # Target
    targets = torch.sin(time_expanded * freqs_expanded).squeeze(2) # (Steps, Batch)
    
    return frequencies, targets

def main():
    print("RealNet Experiment: The Harmonic Oscillator (Sine Wave Generator)")
    print("Objective: One impulse at t=0 sets the frequency. The network must oscillate at that frequency for N steps.")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {DEVICE}")

    # CONFIGURATION
    NUM_NEURONS = 128
    INPUT_IDS = [0] # Input 0: Frequency Control
    OUTPUT_IDS = [1] # Output 1: Sine Wave Output
    STEPS = 30 # Length of the wave
    BATCH_SIZE = 32
    EPOCHS = 3000
    
    # Initialize Model
    model = RealNet(
        num_neurons=NUM_NEURONS,
        input_ids=INPUT_IDS,
        output_ids=OUTPUT_IDS,
        pulse_mode=False, # CHANGED: Continuous Control (VCO Mode)
        dropout_rate=0.0, # CHANGED: No random failures
        device=DEVICE
    )
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    print(f"Model: {NUM_NEURONS} Neurons. Thinking for {STEPS} steps.")
    
    # TRAINING LOOP
    history = []
    
    for epoch in range(EPOCHS):
        model.train()
        
        # 1. Generate Data
        # Input: (Batch, 1)
        # Target: (Steps, Batch) - Value of sine wave at each step
        inputs, targets = generate_sine_data(BATCH_SIZE, STEPS, DEVICE)
        
        # 2. Reset State & Run
        model.reset_state(BATCH_SIZE)
        
        # Manually map inputs to full input vector
        x_input = torch.zeros(BATCH_SIZE, NUM_NEURONS, device=DEVICE)
        x_input[:, INPUT_IDS[0]] = inputs[:, 0]
        
        # Forward pass
        # model processes for STEPS. 
        # But wait! RealNet standard forward takes x_input and applies it.
        # If pulse_mode=True, it applies it only at t=0.
        outputs_stack, final_state = model(x_input, steps=STEPS)
        
        # outputs_stack: (Steps, Batch, Neurons)
        
        # 3. Extract Output Sequence
        # We want the value of OUTPUT_ID at every step
        # shape: (Steps, Batch)
        predicted_seq = outputs_stack[:, :, OUTPUT_IDS[0]]
        
        # 4. Calculate Loss
        loss = loss_fn(predicted_seq, targets)
        
        # 5. Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        history.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.6f}")

    # TEST
    print("\nTraining Complete. Testing Generator...")
    model.eval()
    
    # Test Frequencies
    test_freqs = torch.tensor([[0.15], [0.45]], device=DEVICE) # Low and High freq
    test_batch = len(test_freqs)
    
    model.reset_state(test_batch)
    x_test = torch.zeros(test_batch, NUM_NEURONS, device=DEVICE)
    x_test[:, INPUT_IDS[0]] = test_freqs[:, 0]
    
    with torch.no_grad():
        outputs_stack, _ = model(x_test, steps=STEPS)
        predictions = outputs_stack[:, :, OUTPUT_IDS[0]] # (Steps, Batch)
    
    # Print comparison
    print("\nFrequency 0.15 (Slow Wave):")
    for t in range(0, STEPS, 5):
        target = math.sin((t+1) * 0.15)
        pred = predictions[t, 0].item()
        print(f"  t={t+1}: Target {target:.4f} | RealNet {pred:.4f}")

    print("\nFrequency 0.45 (Fast Wave):")
    for t in range(0, STEPS, 5):
        target = math.sin((t+1) * 0.45)
        pred = predictions[t, 1].item()
        print(f"  t={t+1}: Target {target:.4f} | RealNet {pred:.4f}")

if __name__ == "__main__":
    main()
