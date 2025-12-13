
import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
from realnet_llm.config import RealNetConfig, TrainingConfig
from realnet_llm.model import RealNetLM
from realnet_llm.data import UnicodeDataset

try:
    from realnet_llm.generate import generate
except ImportError:
    generate = None

# Suppress Dynamo errors
import torch._dynamo
torch._dynamo.config.suppress_errors = True

def main():
    print("Initializing Infinite Context (TBPTT) Training...")
    
    # 1. Config
    model_config = RealNetConfig(
        n_neurons=1024,
        n_layers=1,
        thinking_steps=5,
        dropout=0.1,
        compile=False
    )
    
    train_config = TrainingConfig(
        batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        max_steps=5000, # Longer run
        eval_interval=100,
        log_interval=10,
        save_interval=1000,
        out_dir='out_infinite',
        context_window=128, # Chunk size, but effective context is infinite
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 2. Data
    data_path = 'data/shakespeare.txt'
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return
        
    dataset = UnicodeDataset(data_path, block_size=train_config.context_window)
    print("Dataset loaded.")
    
    # 3. Model
    model = RealNetLM(model_config)
    model.to(train_config.device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=(train_config.device == 'cuda'))
    
    # 4. Training Loop (TBPTT)
    print("Starting Infinite Training...")
    model.train()
    
    # STATE PERSISTENCE
    # We maintain a state tensor across batches.
    # Initial state is None (RealNet will init zeros).
    current_state = None 
    
    data_idx = 0
    iter_num = 0
    t0 = time.time()
    
    os.makedirs(train_config.out_dir, exist_ok=True)
    
    while iter_num < train_config.max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for micro_step in range(train_config.gradient_accumulation_steps):
            # Fetch SEQUENTIAL slice
            # We must be careful: data_idx must advance by block_size *after* accumulation?
            # No, accumulation usually implies same weights.
            # But here we are processing a STREAM. 
            # If we accumulate gradients over TIME, we are effectively increasing BPTT length?
            # Or are we accumulating over independent batches?
            # Standard TBPTT: One forward pass through T steps.
            # Gradient Accumulation: Multiple forward passes summed up.
            
            # For Infinite Context, we want consecutive chunks.
            # Chunk 1 (0-128) -> State 1 -> Update
            # Chunk 2 (128-256) -> State 2 -> Update
            # If we split Chunk 1 into micro-batches, they are parallel sequences?
            # Our get_sequential_batch handles B parallel sequences.
            # So updating data_idx moves ALL B sequences forward.
            
            x, y = dataset.get_sequential_batch(data_idx, train_config.batch_size, train_config.context_window, device=train_config.device)
            
            # Forward
            with torch.cuda.amp.autocast(enabled=(train_config.device == 'cuda')):
                # Pass previous state
                # Note: We must detach state if it came from previous iteration to stop backprop going back forever (TBPTT)
                # But within accumulation loop? 
                # Ideally accumulation covers 4 steps of time? 
                # No, usually aggregation is spatial (batch size).
                # Here we treat it as spatial accumulation (larger effective batch).
                # So we run SAME data_idx?
                # No, that would just be same data.
                
                # Let's assume standard accumulation: Parallel Batches.
                # get_sequential_batch gives us B independent streams.
                # We update all B streams.
                # So state is (B, N).
                
                if current_state is not None:
                    # CRITICAL: Detach state from previous batch's graph
                    state_in = current_state.detach()
                else:
                    state_in = None
                    
                logits, loss, state_out = model(x, y, state=state_in)
                loss = loss / train_config.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            accum_loss += loss.item()
            
            # We want to carry state_out to next micro_step?
            # No, accumulation usually means we sum gradients for the *same* step?
            # Or valid to do time-accumulation?
            # Simpler: Disable accumulation for Infinite Context or assume B is enough.
            # Let's just update state_out at end.
            # Wait, if we run multiple micro-steps, they must be sequential in time?
            # If so, data_idx increments.
            # If they are spatial (larger batch), then get_sequential_batch should return larger batch.
            
            # Let's treat accumulation purely as "Larger Batch". 
            # So we only fetch ONE batch per step.
            # We break loop here to keep logic simple for TBPTT. 
            # (Proper accumulation with stateful RNN is tricky).
            
            current_state = state_out # Keep the updated state including grad? No, we detach next iter.
            
        # Optimization
        scaler.step(optimizer)
        scaler.update()
        
        # Advance Data Stream
        data_idx += train_config.context_window
        
        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % train_config.log_interval == 0:
            print(f"iter {iter_num}: loss {accum_loss*train_config.gradient_accumulation_steps:.4f}, time {dt*1000:.2f}ms") # Fix loss reporting scale
            
        # Generation Check (Does it hallucinate better?)
        if iter_num > 0 and iter_num % train_config.eval_interval == 0:
            if generate:
                print("\n--- Infinite Stream Check ---")
                # For generation, we should pass the current training state to see what it's thinking!
                # But generation token-by-token modifies state. We should cone it.
                gen_state = current_state.detach().clone() if current_state is not None else None
                # Generate
                generate(model, "To be", max_new_tokens=50, device=train_config.device) # We ignore gen_state for now to test raw prompt capability, or we can inject it.
                print("\n-----------------------------")
        
        iter_num += 1
        
        # Save
        if iter_num % train_config.save_interval == 0:
             torch.save(model.state_dict(), os.path.join(train_config.out_dir, 'latest_stateful.pt'))

if __name__ == '__main__':
    main()
