
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Adjust path to import realnet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realnet import RealNet

def generate_stopwatch_data(batch_size, seq_len, timer_durations, device):
    """
    Generates data for 'Stopwatch'.
    Input: Pulse at t=0. Value of pulse indicates 'How long to wait'.
           (e.g. 0.2 -> wait 5 steps, 0.8 -> wait 20 steps)
           Wait, let's make it simpler first. Fixed duration or Variable?
           Let's go HARD: Variable Duration.
           Input triggers at t=0.
           Target fires at t = Duration.
    """
    inputs = torch.zeros(batch_size, seq_len, 1, device=device)
    targets = torch.zeros(batch_size, seq_len, device=device)
    
    for i in range(batch_size):
        # Pick a random duration between 5 and seq_len-5
        duration = torch.randint(5, seq_len - 5, (1,)).item()
        
        # Input: Encode duration as signal strength? Or just a "Start" signal?
        # If we just give "Start", the network always waits X steps.
        # But we want "Count X steps". So we must tell it X.
        # Let's Encode X as input value: value = duration / 10.0
        
        input_val = duration / 20.0 # Normalize roughly
        inputs[i, 0, 0] = input_val
        
        # Target: Pulse at t = duration
        # Gaussian curve around the target time to make learning easier (Soft Target)
        # Instead of single 1.0, let's do 0.5, 1.0, 0.5
        targets[i, duration] = 1.0
        if duration > 0: targets[i, duration-1] = 0.5
        if duration < seq_len-1: targets[i, duration+1] = 0.5
        
    return inputs, targets

def main():
    print("RealNet Experiment: The Stopwatch (Internal Clock)")
    print("Objective: Input tells 'Wait X steps'. Network must wait in silence and fire at exactly t=X.")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {DEVICE}")

    # CONFIGURATION
    NUM_NEURONS = 128
    INPUT_ID = 0
    OUTPUT_ID = 1
    
    SEQ_LEN = 30
    BATCH_SIZE = 64
    EPOCHS = 3000
    
    model = RealNet(
        num_neurons=NUM_NEURONS,
        input_ids=[INPUT_ID],
        output_ids=[OUTPUT_ID],
        device=DEVICE
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # TRAINING
    for epoch in range(EPOCHS):
        model.train()
        
        inputs, targets = generate_stopwatch_data(BATCH_SIZE, SEQ_LEN, None, DEVICE)
        
        x_input = torch.zeros(BATCH_SIZE, SEQ_LEN, NUM_NEURONS, device=DEVICE)
        x_input[:, :, INPUT_ID] = inputs[:, :, 0]
        
        model.reset_state(BATCH_SIZE)
        
        outputs_stack, _ = model(x_input, steps=SEQ_LEN)
        preds = outputs_stack[:, :, OUTPUT_ID].transpose(0, 1) # (Batch, Steps)
        
        loss = loss_fn(preds, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.6f}")

    # TEST
    print("\nTesting Stopwatch...")
    model.eval()
    
    # Test Durations: 10 steps and 20 steps
    durations = [10, 20]
    test_batch = len(durations)
    
    x_test = torch.zeros(test_batch, SEQ_LEN, NUM_NEURONS, device=DEVICE)
    for i, d in enumerate(durations):
        x_test[i, 0, INPUT_ID] = d / 20.0
        
    model.reset_state(test_batch)
    with torch.no_grad():
        out_stack, _ = model(x_test, steps=SEQ_LEN)
        preds = out_stack[:, :, OUTPUT_ID] # (Steps, Batch)
    
    for i, d in enumerate(durations):
        print(f"\nTarget Timer: {d} steps (Input val: {d/20.0:.2f})")
        peak_t = -1
        max_val = -100
        
        for t in range(SEQ_LEN):
            val = preds[t, i].item()
            bar = "█" * int(val * 10)
            if val > max_val:
                max_val = val
                peak_t = t
                
            mark = ""
            if t == d: mark = "🎯 TARGET"
            
            # Reduces noise in print
            if val > 0.1 or t == d: 
                print(f"t={t:02d} | Out: {val:.4f} {bar} {mark}")
                
        print(f"Result: Peak at t={peak_t} (Error: {peak_t - d})")

if __name__ == "__main__":
    main()
