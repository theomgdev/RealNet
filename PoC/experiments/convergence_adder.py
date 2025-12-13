
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Adjust path to import realnet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realnet import RealNet

def generate_adder_data(batch_size, seq_len, input_id, delay_1, delay_2, device):
    """
    Generates sequential data for adding two numbers.
    Input:
        t=delay_1:  Value A
        t=delay_2:  Value B
        others:     0
    Target:
        t=seq_len-1: A + B
    """
    
    # 1. Generate random numbers A and B
    # Range -0.5 to 0.5 so sum is between -1 and 1
    val_a = torch.rand(batch_size, 1, device=device) - 0.5
    val_b = torch.rand(batch_size, 1, device=device) - 0.5
    target_sum = val_a + val_b
    
    # 2. Construct Input Sequence (Batch, Steps, Neurons)
    # We don't know N yet, so return (Batch, Steps, 1) and mapped indices
    inputs = torch.zeros(batch_size, seq_len, 1, device=device)
    
    # Place pulses
    inputs[:, delay_1, 0] = val_a[:, 0]
    inputs[:, delay_2, 0] = val_b[:, 0]
    
    return inputs, target_sum

def main():
    print("RealNet Experiment: The Delayed Adder (Algorithmic Logic)")
    print("Objective: Remember Number A. Wait. Receive Number B. Output A+B.")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {DEVICE}")

    # CONFIGURATION
    NUM_NEURONS = 128
    INPUT_ID = 0
    OUTPUT_ID = 1
    
    SEQ_LEN = 15
    DELAY_1 = 2
    DELAY_2 = 8
    
    BATCH_SIZE = 1024
    EPOCHS = 10000
    
    # Initialize Model
    # Pulse Mode is irrelevant here because we will feed 3D sequence manually
    model = RealNet(
        num_neurons=NUM_NEURONS,
        input_ids=[INPUT_ID],
        output_ids=[OUTPUT_ID],
        dropout_rate=0.1,
        device=DEVICE
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    print(f"Structure: Pulse A at t={DELAY_1}. Pulse B at t={DELAY_2}. Target at t={SEQ_LEN-1}.")
    
    # TRAINING
    for epoch in range(EPOCHS):
        model.train()
        
        # Data
        inputs_seq, targets = generate_adder_data(BATCH_SIZE, SEQ_LEN, INPUT_ID, DELAY_1, DELAY_2, DEVICE)
        
        # Map to Full Neuron Space
        # inputs_seq: (Batch, Steps, 1) -> x_input: (Batch, Steps, N)
        x_input = torch.zeros(BATCH_SIZE, SEQ_LEN, NUM_NEURONS, device=DEVICE)
        x_input[:, :, INPUT_ID] = inputs_seq[:, :, 0]
        
        model.reset_state(BATCH_SIZE)
        
        # Forward
        outputs_stack, final_state = model(x_input, steps=SEQ_LEN)
        
        # We only care about the Output ID at the LAST step
        pred = final_state[:, OUTPUT_ID]
        
        loss = loss_fn(pred, targets.squeeze())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.6f}")

    # TEST
    print("\nTesting...")
    model.eval()
    
    test_a = [-0.3, 0.5, 0.1, -0.4]
    test_b = [0.1, 0.2, -0.1, -0.4]
    
    # Custom Test Batch
    batch_size = len(test_a)
    x_test = torch.zeros(batch_size, SEQ_LEN, NUM_NEURONS, device=DEVICE)
    for i in range(batch_size):
        x_test[i, DELAY_1, INPUT_ID] = test_a[i]
        x_test[i, DELAY_2, INPUT_ID] = test_b[i]
        
    model.reset_state(batch_size)
    with torch.no_grad():
        _, final = model(x_test, steps=SEQ_LEN)
        preds = final[:, OUTPUT_ID]
        
    for i in range(batch_size):
        tgt = test_a[i] + test_b[i]
        p = preds[i].item()
        print(f" {test_a[i]} + {test_b[i]} = {tgt:.2f} | RealNet: {p:.4f} (Diff: {abs(tgt-p):.4f})")
    
if __name__ == "__main__":
    main()
