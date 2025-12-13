
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
import time

# Add root to path to import realnet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from realnet.model import RealNet
from realnet.vocab import RealNetVocab

# Configuration
SEQ_LEN = 1024
THINKING_STEPS = 10
BATCH_SIZE = 1024 # Small batch size due to extreme depth (10k steps)
EPOCHS = 100
LEARNING_RATE = 1e-4

# Truncated BPTT Configuration
# If False, we do Full BPTT (1024 steps). If True, we cut gradients every 64 steps.
USE_TRUNCATED_BPTT = False 

class TinyStoriesDataset(Dataset):
    def __init__(self, text_path, vocab, seq_len):
        self.vocab = vocab
        self.seq_len = seq_len
        self.text_path = text_path
        
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Data file not found: {text_path}")
            
        self.file_size = os.path.getsize(text_path)
        print(f"Dataset initialized (Lazy Loading). File: {text_path}")
        print(f"File Size: {self.file_size / 1024 / 1024:.2f} MB")
        
        # Estimate number of tokens (1 byte ~= 1 token roughly, but we have some UTF8)
        self.total_tokens_est = self.file_size 
        
    def __len__(self):
        # Approximate number of samples
        return self.total_tokens_est // self.seq_len
    
    def __getitem__(self, idx):
        # We need (seq_len + 1) characters
        # Safe read buffer size: (seq_len + 1) * 4 bytes (max utf8 width) + extra margin
        # But average is 1 byte. So 1.5x seq_len is usually enough. 
        # Let's go safe: 4x seq_len. 1024 * 4 = 4KB. Tiny.
        bytes_to_read = (self.seq_len + 20) * 4 
        
        max_start_byte = max(0, self.file_size - bytes_to_read - 100)
        
        # Random Byte Start
        # Note: 'idx' argument is ignored because we sample randomly for infinite stream feeling,
        # or we could map idx to approximate position. Ideally, pure random is fine for this phase.
        start_byte = torch.randint(0, max_start_byte, (1,)).item()
        
        with open(self.text_path, 'rb') as f:
            f.seek(start_byte)
            # Read slightly more to handle UTF-8 boundaries and discarding
            raw_bytes = f.read(bytes_to_read)
            
        # Decode ignoring errors (handles partial bytes at start/end)
        text_chunk = raw_bytes.decode('utf-8', errors='ignore')
        
        # Discard the first few chars to ensure we didn't start in middle of a multibyte sequence
        # invalidly (though errors='ignore' drops the bad byte, we might lose a char).
        # Also, to add some randomness to exact char alignment.
        valid_start = 0
        if start_byte > 0:
            # If we didn't start at 0, the first char might be result of partial byte decode
            valid_start = 1
            
        text_chunk = text_chunk[valid_start:]
        
        if len(text_chunk) < self.seq_len + 1:
            # Retry or pad? With randomly chosen large file, this happens only at EOF.
            # We controlled max_start_byte, so it shouldn'be rare. 
            # If it happens, just pad with spaces
            text_chunk = text_chunk + " " * (self.seq_len + 1 - len(text_chunk))
            
        # Take exact length
        text_chunk = text_chunk[:self.seq_len + 1]
        
        # Encode
        data = self.vocab.encode(text_chunk)
        
        x = data[:-1]
        y = data[1:]
        
        return x, y

def main():
    # Set matmul precision for performance on Ampere+ GPUs
    torch.set_float32_matmul_precision('high')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    vocab = RealNetVocab()
    print(f"Vocab Size: {vocab.get_vocab_size()}")
    
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/tinystories.txt')
    dataset = TinyStoriesDataset(data_path, vocab, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Prepare Model
    # Input/Output IDs are all neurons (since Input = Output = Hidden = All)
    all_ids = list(range(vocab.get_vocab_size()))
    
    model = RealNet(
        num_neurons=vocab.get_vocab_size(),
        input_ids=all_ids,
        output_ids=all_ids,
        pulse_mode=True, # We manually handle pulse, but setting this to True ensures model logic aligns
        device=device
    )
    
    # Compile optimizations
    # We will compile the inner loop logic if possible, but for now let's rely on standard execution
    # or model.compile() if available.
    
    # RESUME LOGIC
    # Checkpoint directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cp_dir = os.path.join(script_dir, 'ckpt')
    os.makedirs(cp_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(cp_dir, 'realnet_llm_zero_latest.pt')
    
    if os.path.exists(checkpoint_path):
        print(f"--> Resuming from checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        # Start: Fix for torch.compile prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('_orig_mod.', '')
            new_state_dict[new_key] = v
        # End: Fix
        
        model.load_state_dict(new_state_dict)
    else:
        print(f"--> No checkpoint found at {checkpoint_path}. Starting from scratch.")
        
    if hasattr(torch, 'compile'):
        # Compiling the whole step-sequence might be too much, but let's try compiling the model
        model = model.compile()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting Training...")
    print(f"Sequence Length: {SEQ_LEN}")
    print(f"Thinking Steps per Token: {THINKING_STEPS}")
    print(f"Total Graph Depth: {SEQ_LEN * THINKING_STEPS}")
    
    if USE_TRUNCATED_BPTT:
        TRUNCATE_STEPS = 64 # Backpropagate every 64 tokens to prevent gradient explosion
        print(f"Truncated BPTT: ON (Steps: {TRUNCATE_STEPS})")
    else:
        TRUNCATE_STEPS = SEQ_LEN
        print(f"Truncated BPTT: OFF (Full Sequence: {SEQ_LEN})")
    
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device) # (B, Seq)
            y_batch = y_batch.to(device) # (B, Seq)
            
            # STATEFUL FORWARD PASS
            # We must iterate through the sequence
            
            # Reset state COMPLETY at start of new sequence
            model.reset_state(BATCH_SIZE)
            current_state = model.state
            
            x_one_hot = torch.nn.functional.one_hot(x_batch, num_classes=vocab.get_vocab_size()).float()
            
            # TRUNCATED BPTT LOOP
            # We process the 1024-length sequence in chunks (e.g., 64 steps)
            # Forward context is preserved (state passed), but gradients are cut.
            
            seq_loss_sum = 0
            
            for t_start in range(0, SEQ_LEN, TRUNCATE_STEPS):
                t_end = min(t_start + TRUNCATE_STEPS, SEQ_LEN)
                chunk_steps = t_end - t_start
                
                optimizer.zero_grad()
                chunk_loss = 0
                
                for t in range(t_start, t_end):
                    # Input Pulse for this step
                    input_pulse = x_one_hot[:, t, :] # (B, N)
                    
                    # Run Thinking Steps
                    # We pass current_state recursively.
                    _, current_state = model(input_pulse, steps=THINKING_STEPS, current_state=current_state)
                    
                    # Prediction & Loss
                    logits = current_state # (B, N)
                    target = y_batch[:, t] # (B,)
                    
                    loss_t = criterion(logits, target)
                    chunk_loss += loss_t
                
                # Average loss for this chunk
                chunk_loss = chunk_loss / chunk_steps
                
                # Backward Pass for this chunk
                chunk_loss.backward()
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Update Weights
                optimizer.step()
                
                # CRITICAL: Detach state to cut gradient history
                current_state = current_state.detach()
                
                seq_loss_sum += chunk_loss.item()
            
            # Average loss for the whole sequence (for logging)
            # number of chunks = ceil(SEQ_LEN / TRUNCATE_STEPS)
            num_chunks = (SEQ_LEN + TRUNCATE_STEPS - 1) // TRUNCATE_STEPS
            avg_seq_loss = seq_loss_sum / num_chunks
            total_loss += avg_seq_loss
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {avg_seq_loss:.4f}")
                
            # Periodic Save & Gen
            if batch_idx % 100 == 0:
                print(f"--> Saving Checkpoint (Batch {batch_idx})")
                torch.save(model.state_dict(), os.path.join(cp_dir, 'realnet_llm_zero_latest.pt'))
                generate_sample(model, vocab, device)
                model.train()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Completed. Avg Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s")
        
        # Save Epoch checkpoint
        torch.save(model.state_dict(), os.path.join(cp_dir, f'realnet_llm_zero_epoch_{epoch}.pt'))

def generate_sample(model, vocab, device):
    model.eval()
    prompt = "Once upon a time"
    input_ids = vocab.encode(prompt).to(device)
    
    model.reset_state(1)
    current_state = model.state
    
    # Warmup
    for i in range(len(input_ids)):
        x_in = torch.nn.functional.one_hot(input_ids[i].unsqueeze(0), num_classes=vocab.get_vocab_size()).float()
        _, current_state = model(x_in, steps=THINKING_STEPS, current_state=current_state)
        
    # Generate
    generated = list(input_ids.cpu().numpy())
    last_state = current_state
    
    print("\n--- GENERATING ---")
    print(prompt, end='', flush=True)
    
    for _ in range(200):
        # Use last state as input? NO.
        # RealNet needs an input PULSE.
        # In generation, the input pulse is the embedding of the PREVIOUSLY GENERATED token.
        
        last_token_id = generated[-1]
        x_in = torch.nn.functional.one_hot(torch.tensor([last_token_id], device=device), num_classes=vocab.get_vocab_size()).float()
        
        _, current_state = model(x_in, steps=THINKING_STEPS, current_state=last_state)
        last_state = current_state
        
        # Sample
        logits = current_state[0]
        probs = torch.softmax(logits, dim=0)
        next_token = torch.multinomial(probs, 1).item()
        
        generated.append(next_token)
        char = vocab.id_to_char.get(next_token, '?')
        print(char, end='', flush=True)
        
    print("\n------------------\n")

if __name__ == '__main__':
    main()
