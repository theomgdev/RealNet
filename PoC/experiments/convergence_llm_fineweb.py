import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import os
import time

# Adjust path to import realnet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realnet import RealNet, RealNetTrainer, save_checkpoint, load_checkpoint, transplant_weights

# 🚀 ENABLE TF32 (Tensor Cores)
torch.set_float32_matmul_precision('high')

# --- CONFIGURATION ---
TRUNCATED_BPTT_STEPS = 512 # Set to -1 to disable
GENERATION_LENGTH = 1024
# Short sequence in full BPTT, long sequence in truncated BPTT.
SEQ_LEN = 256 if TRUNCATED_BPTT_STEPS == -1 else 4096
BATCH_SIZE = 128 # Adjusted for larger SEQ_LEN/Memory
NUM_NEURONS = -1 # Auto-size to Input+Output (Min 512)
THINK_GAP = 5 # Number of silence steps between bytes
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# NEUROGENESIS CONFIG
MAX_LOSS_INCREASE = 100000
NEUROGENESIS_AMOUNT = 1

# Byte-Level Vocabulary (0-255) to support all languages (Chinese, etc.)
VOCAB_SIZE = 256
CHAR_TO_IDX = {i: i for i in range(256)} # Identity map for bytes
IDX_TO_CHAR = {i: i for i in range(256)}

# --- DATASET ---
from datasets import load_dataset

class FineWebIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, seq_len):
        self.seq_len = seq_len
        # Streaming load - instant startup
        print("🌊 Connecting to FineWeb (CC-MAIN-2024-10)...")
        self.dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)
        
    def __iter__(self):
        iterator = iter(self.dataset)
        buffer_bytes = b""
        
        while True:
            # Replenish buffer
            while len(buffer_bytes) < self.seq_len + 1:
                try:
                    item = next(iterator)
                    text = item.get('text', '')
                    # Encode to bytes (UTF-8)
                    new_bytes = text.encode('utf-8', errors='replace') + b" " 
                    buffer_bytes += new_bytes
                except StopIteration:
                    # Reset if end of stream
                    iterator = iter(self.dataset)

            # Extract chunk
            chunk_bytes = buffer_bytes[:self.seq_len + 1]
            buffer_bytes = buffer_bytes[self.seq_len + 1:] 
            
            # Encode (Already bytes, just list inputs)
            indices = list(chunk_bytes)
            
            if len(indices) == self.seq_len + 1:
                x = torch.tensor(indices[:-1], dtype=torch.long)
                y = torch.tensor(indices[1:], dtype=torch.long)
                yield x, y

    def get_vocab_size(self):
        return VOCAB_SIZE
    
    @property
    def char_to_idx(self):
        return CHAR_TO_IDX
        
    @property
    def idx_to_char(self):
        return IDX_TO_CHAR

def one_hot_encode_dilated(x, vocab_size, num_neurons, input_ids, gap):
    # x: (Batch, Seq)
    batch_size, seq_len = x.shape
    total_steps = seq_len * (gap + 1)
    
    # Create empty canvas (Batch, Total_Steps, Neurons)
    one_hot = torch.zeros(batch_size, total_steps, num_neurons, device=DEVICE)
    
    # Map token indices to input vectors at dilated intervals
    # Input at t=0, t=gap+1, t=2*(gap+1)...
    
    for t in range(seq_len):
        step_idx = t * (gap + 1)
        token_indices = x[:, t].to(DEVICE) # (Batch)

        # Assuming input_ids are contiguous range start..end
        # We find the neuron index by adding the start_id
        neuron_indices = token_indices + input_ids[0]
        
        # one_hot[b, step, neuron] = 1
        one_hot[torch.arange(batch_size), step_idx, neuron_indices] = 1.0
        
    return one_hot

def prepare_targets_dilated(y, gap):
    # y: (Batch, Seq)
    batch_size, seq_len = y.shape
    total_steps = seq_len * (gap + 1)
    
    # Init with -100 (Ignore Index for CrossEntropy)
    y_dilated = torch.full((batch_size, total_steps), -100, dtype=torch.long, device=DEVICE)
    
    # Place Targets
    # If GAP=5: Input at 0. Silence 1-5. Target at 5 (end of thinking).
    # Logic: By the time we reach step 5 (before next input at 6), we should know the answer.
    
    for t in range(seq_len):
        target_step = t * (gap + 1) + gap
        if target_step < total_steps:
             y_dilated[:, target_step] = y[:, t].to(DEVICE)
             
    return y_dilated

def generate(model, dataset, start_str="The", length=None):
    if length is None:
        length = GENERATION_LENGTH
        
    model.eval()
    
    # Init from text -> bytes
    input_bytes = start_str.encode('utf-8', errors='replace')
    input_seq = list(input_bytes)
    
    current_state = None
    
    # 1. Warm up state with the prompt
    x_in = torch.tensor(input_seq, dtype=torch.long, device=DEVICE).unsqueeze(0) # (1, Seq)
    x_emb = one_hot_encode_dilated(x_in, dataset.get_vocab_size(), model.num_neurons, model.input_ids, THINK_GAP)
    
    generated_bytes = bytearray(input_bytes)
    
    with torch.no_grad():
        # Run full context
        _, current_state = model(x_emb, steps=x_emb.shape[1])
        
        # 2. Generate new bytes
        last_byte_idx = input_seq[-1]
        
        for _ in range(length):
            # A. Prepare Input (Single Byte + Gap)
            total_step_single = 1 + THINK_GAP
            
            # Input Vector at t=0
            x_next_emb = torch.zeros(1, total_step_single, model.num_neurons, device=DEVICE)
            neuron_idx = last_byte_idx + model.input_ids[0]
            x_next_emb[0, 0, neuron_idx] = 1.0
            
            # B. Run Model
            preds, current_state = model(x_next_emb, steps=total_step_single, current_state=current_state)
            
            # C. Extract Prediction
            logits = preds[THINK_GAP, 0, model.output_ids] # (VocabSize 256)
            
            # D. Sample
            probs = torch.softmax(logits, dim=0)
            next_idx = torch.multinomial(probs, 1).item()
            
            generated_bytes.append(next_idx)
            last_byte_idx = next_idx
            
    # Decode bytes to string
    try:
        return generated_bytes.decode('utf-8', errors='replace')
    except:
        return str(generated_bytes)

def initialize_system(vocab_size, num_neurons, device):
    """
    Creates the Model and Trainer instances.
    """
    input_ids = list(range(vocab_size))
    output_ids = list(range(vocab_size, 2 * vocab_size))
    
    model = RealNet(
        num_neurons=num_neurons,
        input_ids=input_ids,
        output_ids=output_ids,
        device=device,
        dropout_rate=0.0,
        activation='gelu', # Logic/Gating
        weight_init='quiet',
        gradient_checkpointing=True  # Save VRAM
    )
    
    trainer = RealNetTrainer(model, device=device, gradient_persistence=0, synaptic_noise=0)
    
    return model, trainer, input_ids, output_ids

def main():
    global NUM_NEURONS # Allow updating global config if needed
    
    print(f"🚀 RealNet-1B (FineWeb Streaming)...")
    print(f"--- Configuration ---")
    print(f"SEQ_LEN: {SEQ_LEN}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"NUM_NEURONS: {NUM_NEURONS}")
    print(f"TRUNCATED_BPTT_STEPS: {TRUNCATED_BPTT_STEPS}")
    print(f"GENERATION_LENGTH: {GENERATION_LENGTH}")
    print(f"THINK_GAP: {THINK_GAP}")
    print(f"VOCAB_SIZE: {VOCAB_SIZE}")
    print(f"---------------------")
    
    dataset = FineWebIterableDataset(SEQ_LEN)
    
    # DataLoader for IterableDataset
    # Streaming with 1 worker allows background downloading without duplication
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=1,        # 1 Background process for downloading
        prefetch_factor=4,    # Buffer 4 batches ahead in RAM
        persistent_workers=True, # Keep connection alive
        pin_memory=True
    )
    
    print(f"Vocab Size: {dataset.get_vocab_size()}")
    
    # --- MODEL SETUP ---
    model, trainer, input_ids, output_ids = initialize_system(dataset.get_vocab_size(), NUM_NEURONS, DEVICE)
    
    print(f"Input IDs: {input_ids[0]}-{input_ids[-1]}")
    print(f"Output IDs: {output_ids[0]}-{output_ids[-1]}")
    
    # --- CHECKPOINT SETUP (Using RealStore) ---
    CKPT_DIR = os.path.join(os.path.dirname(__file__), 'ckpt')
    os.makedirs(CKPT_DIR, exist_ok=True)
    CKPT_PATH = os.path.join(CKPT_DIR, 'llm_fineweb_latest.pth')
    CKPT_BEST_PATH = os.path.join(CKPT_DIR, 'llm_fineweb_best.pth')
    
    start_epoch = 0
    if os.path.exists(CKPT_PATH):
        # Pre-check dimensions to handle mismatches interactively
        try:
            ckpt_peek = torch.load(CKPT_PATH, map_location=DEVICE)
            if 'model_state_dict' in ckpt_peek and 'W' in ckpt_peek['model_state_dict']:
                saved_dim = ckpt_peek['model_state_dict']['W'].shape[0]
                
                if saved_dim != NUM_NEURONS:
                    print(f"\n⚠️ ARCHITECTURE MISMATCH DETECTED!")
                    print(f"   Current Model: {NUM_NEURONS} neurons")
                    print(f"   Checkpoint:    {saved_dim} neurons")
                    
                    print("Select action:")
                    print("[1] Change Model Size to Match File (Resume Training)")
                    print("[2] Transplant Weights (Adapt to New Size)")
                    action = input("Choice [1/2]: ").strip()
                    
                    if action == '1':
                        print(f"🔄 Resizing model to {saved_dim} neurons...")
                        NUM_NEURONS = saved_dim 
                        
                        # CLEAN RE-INIT using helper
                        model, trainer, _, _ = initialize_system(dataset.get_vocab_size(), NUM_NEURONS, DEVICE)
                        
                        # Now load strictly
                        checkpoint = load_checkpoint(model, trainer.optimizer, CKPT_PATH, device=DEVICE, strict=True)
                        start_epoch = checkpoint['epoch'] + 1
                        print(f"✅ Resuming from Epoch {start_epoch}")
                        
                    else:
                        print(f"⚠️ Proceeding with Weight Transplantation...")
                        transplant_weights(model, CKPT_PATH, device=DEVICE)
                        print(f"🧬 Transplant complete. Starting from Epoch 0 with warm weights.")
                        
                else:
                    # Dimensions match, standard load
                    print(f"🔄 Loading Checkpoint from {CKPT_PATH}...")
                    checkpoint = load_checkpoint(model, trainer.optimizer, CKPT_PATH, device=DEVICE, strict=True)
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"✅ Resuming from Epoch {start_epoch}")
            else:
                 # Fallback
                 print(f"🔄 Loading Checkpoint from {CKPT_PATH}...")
                 checkpoint = load_checkpoint(model, trainer.optimizer, CKPT_PATH, device=DEVICE, strict=True)
                 start_epoch = checkpoint['epoch'] + 1

        except Exception as e:
            print(f"⚠️ Failed to load/inspect checkpoint: {e}. Starting fresh.")
    
    # CrossEntropy with Masking (Thinking Gaps)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    trainer.loss_fn = criterion 
    
    # OUTPUT TRANSFORM: Flatten dilated output (Batch, Total_Steps, Out) -> (N, Out)
    def flatten_logits(out):
        return out.reshape(-1, dataset.get_vocab_size())

    # --- INITIAL GENERATION (Show current state) ---
    print("--- INITIAL GENERATION ---")
    try:
        gen_text = generate(model, dataset, start_str="The meaning of life is ")
        print(gen_text)
    except Exception as e:
        print(f"Generation Error: {e}")
    print("--------------------------")

    print("Training (Infinite)...")
    
    # Infinite Loop
    epoch = start_epoch
    prev_loss = float('inf')
    
    # Init Best Loss
    best_loss = float('inf')
    if os.path.exists(CKPT_BEST_PATH):
        try:
            # Quick peek to set best_loss baseline
            best_ckpt = torch.load(CKPT_BEST_PATH, map_location=DEVICE)
            if 'loss' in best_ckpt:
                best_loss = best_ckpt['loss']
                print(f"🏆 Historical Best Loss: {best_loss:.4f}")
        except Exception as e:
            print(f"⚠️ Could not read best checkpoint: {e}")
            
    loss_increase_counter = 0
    
    while True:
        total_loss = 0
        steps = 0
        
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(dataloader):
            # Dilate Data (Insert Silence)
            x_emb = one_hot_encode_dilated(x, dataset.get_vocab_size(), model.num_neurons, input_ids, THINK_GAP)
            y_dilated = prepare_targets_dilated(y, THINK_GAP)
            y_flat = y_dilated.reshape(-1) # Already on DEVICE
            
            # Total steps has increased due to dilation
            total_steps = x_emb.shape[1]
            
            if TRUNCATED_BPTT_STEPS != -1 and TRUNCATED_BPTT_STEPS > 0:
                # --- TRUNCATED BPTT ---
                # We need to preserve state across chunks of the diluted sequence
                # Note: SEQ_LEN is 4096 tokens, Dilation is * 6 (5+1). total_steps ~ 24k
                # BPTT Steps should probably be in "Network Steps"
                
                current_state = None
                batch_loss = 0
                steps_count = 0
                
                # Chunk size in Network Steps
                chunk_size = TRUNCATED_BPTT_STEPS 
                
                for t_start in range(0, total_steps, chunk_size):
                    t_end = min(t_start + chunk_size, total_steps)
                    actual_steps = t_end - t_start
                    
                    x_chunk = x_emb[:, t_start:t_end, :] # (Batch, Steps, Neurons)
                    y_chunk = y_flat[t_start*BATCH_SIZE : t_end*BATCH_SIZE] # y_flat is (Batch*Steps), careful!
                    # Wait, y_flat = y_dilated.reshape(-1).
                    # y_dilated is (Batch, TotalSteps).
                    # Reshape(-1) does Row-Major (Batch 0 complete, then Batch 1...)
                    # Correct slicing for Trainer expectance:
                    # Generic Traing uses: loss = (pred - target).mean()
                    # Predicted outputs (Batch, Steps, Out) -> OutputTransform -> (Batch*Steps, Out)
                    # Target Values must match (Batch*Steps)
                    
                    # Let's slice y_dilated properly first: (Batch, ChunkSteps)
                    y_chunk_2d = y_dilated[:, t_start:t_end]
                    y_chunk_flat = y_chunk_2d.reshape(-1)
                    
                    # Run Chunk
                    loss, current_state = trainer.train_batch(
                        x_chunk, 
                        y_chunk_flat, 
                        thinking_steps=actual_steps, 
                        full_sequence=True, 
                        output_transform=flatten_logits,
                        initial_state=current_state,
                        return_state=True
                    )
                    
                    # Detach State (Truncate Gradient)
                    current_state = current_state.detach()
                    
                    batch_loss += loss
                    steps_count += 1
                
                loss = batch_loss / max(steps_count, 1)

            else:
                # Standard Full Sequence Training
                loss = trainer.train_batch(
                    x_emb, 
                    y_flat, 
                    thinking_steps=total_steps, 
                    full_sequence=True, 
                    output_transform=flatten_logits 
                )
            
            total_loss += loss
            steps += 1
            
            if batch_idx % 1 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss:.4f}")
                
            if batch_idx > 10: # Limit batches per epoch for frequent view
                break
                
        avg_loss = total_loss / steps
        print(f"Epoch {epoch} Completed | Avg Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.1f}s")
        
        # --- STAGNATION / GROWTH CHECK ---
        if avg_loss > prev_loss:
            loss_increase_counter += 1
            print(f"⚠️ Loss Increased ({loss_increase_counter}/{MAX_LOSS_INCREASE})")
        else:
            # Optionally reset if we improve? 
            # User instruction: "her 5 loss artışında" (every 5 increases).
            # Strict interpretation: Count every increase. 
            # If we just hit a bump, we count it. 
            pass 
        
        if loss_increase_counter >= MAX_LOSS_INCREASE:
            trainer.expand(amount=NEUROGENESIS_AMOUNT)
            NUM_NEURONS = model.num_neurons
            loss_increase_counter = 0
            
        prev_loss = avg_loss
        
        # Generation Check (Reduced frequency if too slow, but user asked for param)
        print("--- GENERATION ---")
        try:
            # Generate short preview (100) first to not spam console
            gen_text = generate(model, dataset, start_str="The meaning of life is ")
            print(gen_text)
        except Exception as e:
            print(f"Generation Error: {e}")
        print("------------------")
        
        # SAVE CHECKPOINT (Using RealStore)
        save_checkpoint(model, trainer.optimizer, epoch, avg_loss, CKPT_PATH)
        print(f"💾 Checkpoint Saved: {CKPT_PATH}")
        
        # SAVE BEST
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, trainer.optimizer, epoch, avg_loss, CKPT_BEST_PATH)
            print(f"🏆 NEW RECORD! Best Checkpoint Saved: {CKPT_BEST_PATH} (Loss: {best_loss:.4f})")
        
        epoch += 1

if __name__ == "__main__":
    main()
