
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Adjust path to import realnet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realnet import RealNet

def generate_latch_data(batch_size, seq_len, device):
    """
    Generates data for 'Catch & Hold'.
    Input: Random pulse at t = trigger_time.
    Target: 0 before trigger, 1 after trigger (forever).
    """
    inputs = torch.zeros(batch_size, seq_len, 1, device=device)
    targets = torch.zeros(batch_size, seq_len, device=device)
    
    for i in range(batch_size):
        # Trigger happens somewhere between step 2 and seq_len-5
        trigger = torch.randint(2, seq_len - 5, (1,)).item()
        
        # Pulse input at trigger
        inputs[i, trigger, 0] = 1.0
        
        # Target becomes 1 AFTER trigger (inclusive or exclusive? Let's say inclusive)
        # Once turned ON, it stays ON.
        targets[i, trigger:] = 1.0
        
    return inputs, targets

def main():
    print("RealNet Experiment: Catch & Hold (The Latch)")
    print("Objective: Wait for a pulse using chaos. Once received, hold the state output at 1.0 forever.")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {DEVICE}")

    # CONFIGURATION
    NUM_NEURONS = 64
    INPUT_ID = 0
    OUTPUT_ID = 1
    
    SEQ_LEN = 20
    BATCH_SIZE = 64
    EPOCHS = 2000
    
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
        
        inputs, targets = generate_latch_data(BATCH_SIZE, SEQ_LEN, DEVICE)
        
        # Map to Input (Batch, Time, Neurons)
        x_input = torch.zeros(BATCH_SIZE, SEQ_LEN, NUM_NEURONS, device=DEVICE)
        x_input[:, :, INPUT_ID] = inputs[:, :, 0]
        
        model.reset_state(BATCH_SIZE)
        
        # Forward
        outputs_stack, _ = model(x_input, steps=SEQ_LEN)
        preds = outputs_stack[:, :, OUTPUT_ID] # (Steps, Batch)
        
        # Transpose pred to (Batch, Steps) to match target
        preds = preds.transpose(0, 1)
        
        loss = loss_fn(preds, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.6f}")

    # TEST
    print("\nTesting Latch Mechanism...")
    model.eval()
    
    # Test case: Trigger at t=5
    test_trigger = 5
    test_input = torch.zeros(1, SEQ_LEN, NUM_NEURONS, device=DEVICE)
    test_input[0, test_trigger, INPUT_ID] = 1.0
    
    model.reset_state(1)
    with torch.no_grad():
        out_stack, _ = model(test_input, steps=SEQ_LEN)
        out_seq = out_stack[:, 0, OUTPUT_ID]
        
    print(f"Trigger sent at t={test_trigger}")
    for t in range(SEQ_LEN):
        val = out_seq[t].item()
        status = "OFF" if val < 0.5 else "ON "
        visual = "🔴" if val < 0.5 else "🟢"
        if t == test_trigger: visual = "⚡ TRIGGER!"
        
        print(f"t={t:02d} | Out: {val:.4f} | {status} {visual}")

if __name__ == "__main__":
    main()
