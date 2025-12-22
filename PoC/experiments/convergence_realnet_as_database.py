
import torch
import torch.nn as nn
import sys
import os
import random

# Adjust path to import realnet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realnet import RealNet, RealNetTrainer
from realnet.utils.data import prepare_input

def generate_db_data(batch_size, num_keys, seq_per_op, num_ops, device):
    """
    Generates a complex CRUD sequence for RealNet Database.
    Now with 'Active Training' targets for all ops.
    """
    total_steps = num_ops * seq_per_op
    input_dim = 3 + num_keys + 1
    
    inputs = torch.zeros(batch_size, total_steps, input_dim, device=device)
    targets = torch.zeros(batch_size, total_steps, 1, device=device)
    
    memories = [{} for _ in range(batch_size)]
    
    for b in range(batch_size):
        for op in range(num_ops):
            t_start = op * seq_per_op
            t_end = t_start + seq_per_op
            
            r = random.random()
            if r < 0.5: cmd = "WRITE"
            elif r < 0.9: cmd = "READ"
            else: cmd = "DELETE"
                
            k_idx = random.randint(0, num_keys - 1)
            
            if cmd == "WRITE":
                val = random.uniform(-1.0, 1.0)
                memories[b][k_idx] = val
                
                inputs[b, t_start, 0] = 1.0 
                inputs[b, t_start, 3 + k_idx] = 1.0 
                inputs[b, t_start, 3 + num_keys] = val 
                targets[b, t_start : t_end, 0] = val
                
            elif cmd == "READ":
                inputs[b, t_start, 1] = 1.0 
                inputs[b, t_start, 3 + k_idx] = 1.0 
                
                val = memories[b].get(k_idx, 0.0)
                targets[b, t_start : t_end, 0] = val
                
            elif cmd == "DELETE":
                inputs[b, t_start, 2] = 1.0 
                inputs[b, t_start, 3 + k_idx] = 1.0 
                
                if k_idx in memories[b]:
                    del memories[b][k_idx]
                
                targets[b, t_start : t_end, 0] = 0.0
                    
    return inputs, targets

def main():
    print("🚀 RealNet Experiment: NEURAL DATABASE (Implicit CRUD)")
    print("Objective: Prove that RealNet hidden state can act as an Adressable, Updatable Memory Table.")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {DEVICE}")

    # CONFIGURATION
    NUM_KEYS = 4 
    SEQ_PER_OP = 8 
    NUM_OPS = 12 
    NUM_NEURONS = 512 
    
    input_ids = list(range(8))
    output_ids = [9]
    
    model = RealNet(
        num_neurons=NUM_NEURONS,
        input_ids=input_ids,
        output_ids=output_ids,
        device=DEVICE,
        dropout_rate=0.0,
        activation='gelu',
        weight_init='quiet'
    )
    
    trainer = RealNetTrainer(model, device=DEVICE, gradient_persistence=0.5) # Increased for deeper memory
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

    print(f"Config: {NUM_KEYS} Keys | {NUM_OPS} Ops per Batch | {NUM_NEURONS} Neurons | Persistence: 0.5")
    print("Training...")
    
    EPOCHS = 10000 
    BATCH_SIZE = 64
    total_steps = NUM_OPS * SEQ_PER_OP
    THINK_STEPS = 2 # First 2 steps of an op are 'processing', we don't demand output yet
    
    for epoch in range(EPOCHS):
        inputs, targets = generate_db_data(BATCH_SIZE, NUM_KEYS, SEQ_PER_OP, NUM_OPS, DEVICE)
        
        # Manual Train Step for masking
        model.train()
        trainer.optimizer.zero_grad()
        
        x_input, _ = prepare_input(inputs, model.input_ids, model.num_neurons, DEVICE)
        model.reset_state(BATCH_SIZE)
        preds_all, _ = model(x_input, steps=total_steps)
        preds = preds_all[:, :, output_ids].permute(1, 0, 2) # (Batch, Steps, 1)
        
        # CREATE MASK: Weight is 0 for first THINK_STEPS of every op, 1 otherwise
        mask = torch.ones_like(targets)
        for op in range(NUM_OPS):
            t_start = op * SEQ_PER_OP
            mask[:, t_start : t_start + THINK_STEPS, :] = 0.0
            
        # Calculate Weighted MSE Loss
        loss = (torch.square(preds - targets) * mask).mean()
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        trainer.optimizer.step()
        
        if epoch % 100 == 0:
            with torch.no_grad():
                # Accuracy only for masked (meaningful) steps
                diff = torch.abs(preds - targets)
                # Count accurate if diff < 0.1 on segments where mask == 1
                accurate = (diff < 0.1).float() * mask
                acc = (accurate.sum() / mask.sum()) * 100
                print(f"Epoch {epoch}: Loss {loss.item():.6f} | Acc {acc.item():.2f}%")
                
            if loss < 0.0005: 
                print("Convergence Reached!")
                break
                
    # TEST RUN (Simulation)
    print("\n🔍 Simulation Run (The Neural Database in Action):")
    test_ops = 8
    inputs, targets = generate_db_data(1, NUM_KEYS, SEQ_PER_OP, test_ops, DEVICE)
    
    with torch.no_grad():
        preds = trainer.predict(inputs, thinking_steps=(test_ops * SEQ_PER_OP), full_sequence=True)
        
    print(f"{'Step':<5} | {'Command':<8} | {'Key':<5} | {'Val_In':<8} | {'Target':<8} | {'RealNet':<8} | {'Status'}")
    print("-" * 80)
    
    for op in range(test_ops):
        t_start = op * SEQ_PER_OP
        row = inputs[0, t_start]
        cmd = ""
        if row[0] > 0.5: cmd = "WRITE"
        elif row[1] > 0.5: cmd = "READ"
        elif row[2] > 0.5: cmd = "DELETE"
        
        key = -1
        for k in range(NUM_KEYS):
            if row[3+k] > 0.5: key = k
            
        val_in = row[3 + NUM_KEYS].item()
        
        for t in range(t_start, t_start + SEQ_PER_OP):
            target = targets[0, t, 0].item()
            pred = preds[0, t, 0].item()
            diff = abs(target - pred)
            status = "✅" if diff < 0.1 else "❌"
            if t > t_start: 
                c_display = f"({t-t_start})"
                k_display = "..."
                v_display = ""
            else:
                c_display = cmd
                k_display = f"K{key}"
                v_display = f"{val_in:.4f}"
                
            print(f"{t:<5} | {c_display:<8} | {k_display:<5} | {v_display:<8} | {target:<8.4f} | {pred:<8.4f} | {status}")

if __name__ == "__main__":
    main()
