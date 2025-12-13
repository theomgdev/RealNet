
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realnet import RealNet

def generate_dilated_data(batch_size, logic_len, gap, device):
    """
    Generates 'dilated' binary sequence.
    logic_len: How many actual bits of info.
    gap: How many empty steps between bits.
    Total steps = logic_len * (gap + 1)
    """
    # 1. Generate Raw Sequence (Logic Layer)
    raw_inputs = torch.randint(0, 2, (batch_size, logic_len), device=device).float()
    
    # Calculate total physical steps
    total_steps = logic_len * (gap + 1)
    
    # 2. Create Physical Tensors (Time Layer)
    inputs = torch.zeros(batch_size, total_steps, 1, device=device)
    targets = torch.zeros(batch_size, total_steps, device=device)
    
    # 3. Fill and Calculate Logic
    for i in range(batch_size):
        prev_bit = 0.0
        
        for step in range(logic_len):
            bit = raw_inputs[i, step].item()
            
            # Place bit at the start of the block
            t_real = step * (gap + 1)
            inputs[i, t_real, 0] = bit
            
            # Logic: If Current=1 AND Prev=1 -> Alarm MATCH
            # But where do we expect the alarm? 
            # Ideally, immediately when the second 1 arrives. 
            # And maybe sustain it for the duration of the thinking gap?
            # Let's try to demand it strictly at the moment of arrival, 
            # but allow it to ring during the gap too (Soft Target).
            
            if bit == 1.0 and prev_bit == 1.0:
                # Fire Alarm!
                # Target window: From arrival (t_real) to end of gap
                targets[i, t_real : t_real + gap + 1] = 1.0 
            
            prev_bit = bit
            
    return inputs, targets, raw_inputs

def main():
    print("RealNet Experiment: The Thinking Detective 🕵️‍♂️")
    print("Objective: Watch a stream of 0s and 1s. BUT... you have time to think between bits.")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {DEVICE}")

    # CONFIGURATION
    NUM_NEURONS = 128
    GAP = 3 # 3 steps of silence between every bit
    LOGIC_LEN = 10 # 10 bits of information
    SEQ_LEN = LOGIC_LEN * (GAP + 1)
    
    BATCH_SIZE = 64
    EPOCHS = 2000
    
    INPUT_ID = 0
    OUTPUT_ID = 1
    
    model = RealNet(
        num_neurons=NUM_NEURONS,
        input_ids=[INPUT_ID],
        output_ids=[OUTPUT_ID],
        dropout_rate=0.0, # Focused thinking
        pulse_mode=True, # Restore Pulse Mode
        device=DEVICE
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    
    print(f"Logic Steps: {LOGIC_LEN} | Thinking Gap: {GAP} | Total Physical Steps: {SEQ_LEN}")

    # TRAINING
    for epoch in range(EPOCHS):
        model.train()
        
        # Raw inputs are returned just for visualization later
        inputs, targets, _ = generate_dilated_data(BATCH_SIZE, LOGIC_LEN, GAP, DEVICE)
        
        # Map to 3D Input (Batch, Time, Neurons)
        x_input = torch.zeros(BATCH_SIZE, SEQ_LEN, NUM_NEURONS, device=DEVICE)
        x_input[:, :, INPUT_ID] = inputs[:, :, 0]
        
        model.reset_state(BATCH_SIZE)
        
        outputs_stack, _ = model(x_input, steps=SEQ_LEN)
        preds = outputs_stack[:, :, OUTPUT_ID].transpose(0, 1)
        
        loss = loss_fn(preds, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.6f}")

    # TEST
    print("\nTesting Thinking Detective...")
    model.eval()
    
    # Custom Test: 0 1 0 1 1 0
    # Expected Match at index 4 (the second 1)
    test_bits = [0, 1, 0, 1, 1, 0]
    total_len = len(test_bits) * (GAP + 1)
    
    x_test = torch.zeros(1, total_len, NUM_NEURONS, device=DEVICE)
    
    # Construct input manually
    print(f"\nTimeline Analysis (GAP={GAP}):")
    print(f"{'Time':<5} | {'Input':<5} | {'Output':<8} | {'Status'}")
    print("-" * 40)
    
    for i, bit in enumerate(test_bits):
        t_real = i * (GAP + 1)
        x_test[0, t_real, INPUT_ID] = float(bit)
        
    model.reset_state(1)
    with torch.no_grad():
        out_stack, _ = model(x_test, steps=total_len)
        preds = out_stack[:, 0, OUTPUT_ID]
        
    # Visualization
    prev_bit = 0
    for i, bit in enumerate(test_bits):
        t_start = i * (GAP + 1)
        
        # Check the logic explicitly for display
        is_match = (bit == 1 and prev_bit == 1)
        expected = "SHOULD FIRE" if is_match else ""
        
        # Print the burst for this bit
        for t in range(t_start, t_start + GAP + 1):
            val = preds[t].item()
            
            inp_display = str(bit) if t == t_start else "."
            
            bar = "█" * int(val * 10)
            alert = "🚨" if val > 0.5 else ""
            
            if t == t_start:
                 print(f"{t:<5} | {inp_display:<5} | {val:.4f} {alert} | {expected}")
            else:
                 print(f"{t:<5} | {inp_display:<5} | {val:.4f} {alert} | (Thinking...)")
                 
        prev_bit = bit

if __name__ == "__main__":
    main()
