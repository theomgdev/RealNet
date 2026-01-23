import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import os
import time
import os
import random

# --- PRE-IMPORT CONFIG ---
# Remove comment below to disable bitsandbytes 8-bit Optimizer globally (Use Standard AdamW)
# os.environ["NO_BNB"] = "1" 

# Adjust path to import realnet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realnet import RealNet, RealNetTrainer, save_checkpoint, load_checkpoint, transplant_weights

# 🚀 ENABLE TF32 (Tensor Cores)
torch.set_float32_matmul_precision('high')

# --- CONFIGURATION ---
TRUNCATED_BPTT_STEPS = 16 # Set to -1 to disable
GENERATION_LENGTH = 1024
# Short sequence in full BPTT, long sequence in truncated BPTT.
SEQ_LEN = 256 if TRUNCATED_BPTT_STEPS == -1 else 64
BATCH_SIZE = 512 # Adjusted for larger SEQ_LEN/Memory
STEPS_PER_EPOCH = 10 # Number of batches per "Epoch" (for logging/saving)
LOG_INTERVAL = 1 # Print loss every N batches
MAX_START_SKIP = 1000 # Randomly skip up to N documents at start
NUM_NEURONS = -1 # Auto-size to Input+Output (Min 512)
ACTIVATION = 'swiglu' # 'gelu' (Standard) or 'swiglu' (Gated, slower but smarter)
THINK_GAP = 5 # Number of silence steps between bytes
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# NEUROGENESIS CONFIG
MAX_LOSS_INCREASE = 1
NEUROGENESIS_AMOUNT = 10

# Byte-Level Vocabulary (0-255) to support all languages (Chinese, etc.)
VOCAB_SIZE = 256
RESET_OPTIMIZER_ON_LOAD = False # Set True to discard optimizer state (Cold Restart)
LEARNING_RATE = 1e-4

# --- SCHEDULER CONFIG ---
USE_SCHEDULER = True
SCHEDULER_T0 = 1000        # Steps before first restart (~2-3 epochs)
SCHEDULER_ETA_MIN = 1e-7  # Minimum LR before restart

CHAR_TO_IDX = {i: i for i in range(256)} # Identity map for bytes
IDX_TO_CHAR = {i: i for i in range(256)}

# --- DATASET ---
from datasets import load_dataset

class FineWebIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, seq_len, debug=False):
        self.seq_len = seq_len
        self.debug = debug
        # Streaming load - instant startup
        print("🌊 Connecting to FineWeb-Edu (CC-MAIN-2024-10)...")
        self.dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="CC-MAIN-2024-10", split="train", streaming=True)
        
    def __iter__(self):
        # Random skip to vary the starting point each run
        skip_n = random.randint(0, MAX_START_SKIP)
        self.current_doc_index = 0
        
        if skip_n > 0:
            print(f"🔀 Skipping {skip_n} documents to randomize start...")
            iterator = iter(self.dataset.skip(skip_n))
            self.current_doc_index = skip_n
        else:
            iterator = iter(self.dataset)
            
        buffer_bytes = b""
        
        while True:
            # Replenish buffer
            while len(buffer_bytes) < self.seq_len + 1:
                try:
                    item = next(iterator)
                    self.current_doc_index += 1
                    if self.debug and self.current_doc_index % 1000 == 0:
                        print(f"📊 Streaming Index: Document #{self.current_doc_index}")
                    text = item.get('text', '')
                    # Encode to bytes (UTF-8)
                    new_bytes = text.encode('utf-8', errors='replace') + b" " 
                    buffer_bytes += new_bytes
                except StopIteration:
                    # Reset if end of stream
                    iterator = iter(self.dataset)
                    self.current_doc_index = 0

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

def prepare_inputs_dilated(x, gap, device=None):
    # x: (Batch, Seq) indices
    # Returns: (Batch, TotalSteps) indices with -1 as silence
    target_device = device if device is not None else x.device
    
    batch_size, seq_len = x.shape
    total_steps = seq_len * (gap + 1)
    
    # Init with -1 (Silence)
    # This tensor is ~1000x smaller than the one-hot version
    x_dilated = torch.full((batch_size, total_steps), -1, dtype=torch.long, device=target_device)
    
    # Place indices at dilated intervals
    for t in range(seq_len):
        step_idx = t * (gap + 1)
        x_dilated[:, step_idx] = x[:, t].to(target_device)
        
    return x_dilated

def prepare_targets_dilated(y, gap, device=None):
    # y: (Batch, Seq)
    target_device = device if device is not None else y.device
    
    batch_size, seq_len = y.shape
    total_steps = seq_len * (gap + 1)
    
    # Init with -100 (Ignore Index for CrossEntropy)
    y_dilated = torch.full((batch_size, total_steps), -100, dtype=torch.long, device=target_device)
    
    # Place Targets
    # If GAP=5: Input at 0. Silence 1-5. Target at 5 (end of thinking).
    # Logic: By the time we reach step 5 (before next input at 6), we should know the answer.
    
    for t in range(seq_len):
        target_step = t * (gap + 1) + gap
        if target_step < total_steps:
             y_dilated[:, target_step] = y[:, t].to(target_device)
             
    return y_dilated

def generate(model, dataset, start_str="The", length=None, temperature=0.8, top_k=40, top_p=0.9):
    """
    Generates text using the model with modern sampling techniques.
    
    Args:
        model: The RealNet model.
        dataset: The dataset (for vocab info).
        start_str: The prompt string.
        length: Length of text to generate (bytes).
        temperature: Controls randomness (1.0 = normal, <1.0 = conservative, >1.0 = creative).
        top_k: Filters to the top K most likely tokens.
        top_p: Nucleus sampling - filters to the smallest set of tokens comprising probability P.
    """
    if length is None:
        length = GENERATION_LENGTH
        
    model.eval()
    
    # Init from text -> bytes
    input_bytes = start_str.encode('utf-8', errors='replace')
    input_seq = list(input_bytes)
    
    current_state = None
    
    # 1. Warm up state with the prompt
    x_in = torch.tensor(input_seq, dtype=torch.long, device=model.device).unsqueeze(0) # (1, Seq)
    x_dilated = prepare_inputs_dilated(x_in, THINK_GAP, device=model.device)
    
    generated_bytes = bytearray(input_bytes)
    
    with torch.no_grad():
        # Run full context
        _, current_state = model(x_dilated, steps=x_dilated.shape[1])
        
        # 2. Generate new bytes
        last_byte_idx = input_seq[-1]
        
        for _ in range(length):
            # A. Prepare Input (Single Byte + Gap)
            total_step_single = 1 + THINK_GAP
            
            # Input Vector at t=0
            x_next = torch.full((1, total_step_single), -1, dtype=torch.long, device=model.device)
            x_next[0, 0] = last_byte_idx
            
            # B. Run Model
            preds, current_state = model(x_next, steps=total_step_single, current_state=current_state)
            
            # C. Extract Prediction
            logits = preds[THINK_GAP, 0, model.output_ids] # (VocabSize 256)
            
            # --- SAMPLING LOGIC ---
            
            # 1. Temperature
            if temperature > 0:
                logits = logits / temperature
            
            # 2. Top-K Filter
            if top_k is not None and top_k > 0:
                # Keep only top_k, mask others with -inf
                v, _ = torch.topk(logits, min(top_k, len(logits)))
                pivot = v[-1] # Smallest allowed value
                logits[logits < pivot] = float('-inf')
                
            # 3. Top-P (Nucleus) Filter
            if top_p is not None and top_p > 0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted indices to original indices
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # D. Sample
            probs = torch.softmax(logits, dim=0)
            
            # Handle potential NaN/Inf if filtering was too aggressive (rare fallback)
            if torch.isnan(probs).any() or torch.sum(probs) == 0:
                probs = torch.ones_like(probs) / len(probs)
                
            next_idx = torch.multinomial(probs, 1).item()
            
            generated_bytes.append(next_idx)
            last_byte_idx = next_idx
            
    # Decode bytes to string
    try:
        return generated_bytes.decode('utf-8', errors='replace')
    except:
        return str(generated_bytes)

def initialize_system(vocab_size, num_neurons, device, lr=1e-4, activation='gelu'):
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
        activation=activation, # Logic/Gating
        weight_init='quiet',
        gradient_checkpointing=True  # Save VRAM
    )
    
    trainer = RealNetTrainer(model, lr=lr, device=device, gradient_persistence=0, synaptic_noise=0)
    
    return model, trainer, input_ids, output_ids

def main():
    global NUM_NEURONS # Allow updating global config if needed
    
    print(f"🚀 RealNet-1B (FineWeb Streaming)...")
    print(f"--- Configuration ---")
    print(f"SEQ_LEN: {SEQ_LEN}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"NUM_NEURONS: {NUM_NEURONS}")
    print(f"TRUNCATED_BPTT_STEPS: {TRUNCATED_BPTT_STEPS}")
    print(f"STEPS_PER_EPOCH: {STEPS_PER_EPOCH}")
    print(f"LOG_INTERVAL: {LOG_INTERVAL}")
    print(f"GENERATION_LENGTH: {GENERATION_LENGTH}")
    print(f"THINK_GAP: {THINK_GAP}")
    print(f"ACTIVATION: {ACTIVATION}")
    print(f"VOCAB_SIZE: {VOCAB_SIZE}")
    print(f"DEVICE: {DEVICE}")
    print(f"NEUROGENESIS: MaxLossInc={MAX_LOSS_INCREASE}, Amount={NEUROGENESIS_AMOUNT}")
    print(f"RESET_OPTIM_ON_LOAD: {RESET_OPTIMIZER_ON_LOAD}")
    print(f"SCHEDULER: Enabled={USE_SCHEDULER}, T0={SCHEDULER_T0}, MinLR={SCHEDULER_ETA_MIN}")
    print(f"LEARNING_RATE: {LEARNING_RATE}")
    print(f"---------------------")
    
    dataset = FineWebIterableDataset(SEQ_LEN, debug=True)
    
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
    
    # --- MODEL SETUP ---
    model, trainer, input_ids, output_ids = initialize_system(dataset.get_vocab_size(), NUM_NEURONS, DEVICE, LEARNING_RATE, ACTIVATION)
    
    # Update global config with actual model size (if it was auto-sized)
    NUM_NEURONS = model.num_neurons
    
    print(f"Input IDs: {input_ids[0]}-{input_ids[-1]}")
    print(f"Output IDs: {output_ids[0]}-{output_ids[-1]}")
    
    # --- CHECKPOINT SETUP (Using RealStore) ---
    CKPT_DIR = os.path.join(os.path.dirname(__file__), 'ckpt')
    os.makedirs(CKPT_DIR, exist_ok=True)
    CKPT_PATH = os.path.join(CKPT_DIR, f'llm_fineweb_{ACTIVATION}_latest.pth')
    CKPT_BEST_PATH = os.path.join(CKPT_DIR, f'llm_fineweb_{ACTIVATION}_best.pth')
    
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
                        model, trainer, _, _ = initialize_system(dataset.get_vocab_size(), NUM_NEURONS, DEVICE, LEARNING_RATE, ACTIVATION)
                        
                        # Now load strictly
                        opt_arg = None if RESET_OPTIMIZER_ON_LOAD else trainer.optimizer
                        if RESET_OPTIMIZER_ON_LOAD:
                            print("⚠️ RESET_OPTIMIZER_ON_LOAD is True. Discarding saved optimizer state.")
                            
                        checkpoint = load_checkpoint(model, opt_arg, CKPT_PATH, device=DEVICE, strict=True)
                        start_epoch = checkpoint['epoch'] + 1
                        print(f"✅ Resuming from Epoch {start_epoch}")
                        
                    else:
                        print(f"⚠️ Proceeding with Weight Transplantation...")
                        transplant_weights(model, CKPT_PATH, device=DEVICE)
                        print(f"🧬 Transplant complete. Starting from Epoch 0 with warm weights.")
                        
                else:
                    # Dimensions match, standard load
                    opt_arg = None if RESET_OPTIMIZER_ON_LOAD else trainer.optimizer
                    msg = " (Optimizer Reset)" if RESET_OPTIMIZER_ON_LOAD else ""
                    print(f"🔄 Loading Checkpoint from {CKPT_PATH}{msg}...")
                    
                    checkpoint = load_checkpoint(model, opt_arg, CKPT_PATH, device=DEVICE, strict=True)
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"✅ Resuming from Epoch {start_epoch}")
            else:
                 # Fallback
                 opt_arg = None if RESET_OPTIMIZER_ON_LOAD else trainer.optimizer
                 msg = " (Optimizer Reset)" if RESET_OPTIMIZER_ON_LOAD else ""
                 print(f"🔄 Loading Checkpoint from {CKPT_PATH}{msg}...")
                 
                 checkpoint = load_checkpoint(model, opt_arg, CKPT_PATH, device=DEVICE, strict=True)
                 start_epoch = checkpoint['epoch'] + 1

        except Exception as e:
            print(f"⚠️ Failed to load/inspect checkpoint: {e}. Starting fresh.")
    
    # CrossEntropy with Masking (Thinking Gaps)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.005)
    trainer.loss_fn = criterion 
    
    # OUTPUT TRANSFORM: Flatten dilated output (Batch, Total_Steps, Out) -> (N, Out)
    def flatten_logits(out):
        return out.reshape(-1, dataset.get_vocab_size())

    # --- INITIAL GENERATION (Show current state) ---
    # --- INITIAL GENERATION (Sampling Tests) ---
    print("\n--- INITIAL GENERATION TESTS ---")
    
    test_prompts = [
        ("Classic (Old Way)",          0, 0, 1),
        ("Deterministic (Greedy)",     1e-5, 1,  1.0),
        ("Standard (Balanced)",        1.0,  40, 0.9),
        ("Creative (High Temp)",       1.2,  50, 0.95),
        ("Precise (Low Temp)",         0.7,  20, 0.8),
    ]

    for name, t, k, p in test_prompts:
        print(f"\n📝 {name}:")
        try:
            gen_text = generate(
                model, 
                dataset, 
                start_str="The meaning of life is", 
                length=256,
                temperature=t, 
                top_k=k, 
                top_p=p
            )
            print(gen_text)
        except Exception as e:
            print(f"Error: {e}")
    
    print("------------------------------\n")

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

    # SCHEDULER (Config in Global Constants)
    scheduler = None

    if USE_SCHEDULER:
        # Create Scheduler (Cosine Decay with Warm Restarts)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            trainer.optimizer, 
            T_0=SCHEDULER_T0, 
            eta_min=SCHEDULER_ETA_MIN
        )
    
    # Create persistent iterator to prevent data resetting every epoch
    data_iterator = iter(dataloader)

    while True:
        total_loss = 0
        steps = 0
        
        start_time = time.time()
        
        for batch_idx in range(STEPS_PER_EPOCH):
            try:
                x, y = next(data_iterator)
            except StopIteration:
                print("🔄 Dataset exhausted. Restarting iterator...")
                data_iterator = iter(dataloader)
                x, y = next(data_iterator)
            # Dilate Data (Insert Silence)
            x_dilated = prepare_inputs_dilated(x, THINK_GAP, device=DEVICE)
            y_dilated = prepare_targets_dilated(y, THINK_GAP, device=DEVICE)
            y_flat = y_dilated.reshape(-1) # Already on DEVICE
            
            # Total steps has increased due to dilation
            total_steps = x_dilated.shape[1]
            
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
                    
                    x_chunk = x_dilated[:, t_start:t_end] # (Batch, Steps) indices
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
                    x_dilated, 
                    y_flat, 
                    thinking_steps=total_steps, 
                    full_sequence=True, 
                    output_transform=flatten_logits 
                )
            
            # Step Scheduler
            current_lr = 0.0
            if USE_SCHEDULER and scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            elif trainer.optimizer:
                # If scheduler is off, just get fixed LR
                current_lr = trainer.optimizer.param_groups[0]['lr']
            
            total_loss += loss
            steps += 1
            
            if batch_idx % LOG_INTERVAL == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss:.4f} | LR {current_lr:.2e}")
                
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
            
            # Reset prev_loss to tolerate the "Cold Restart" spike (Momentum reset)
            # This prevents a feedback loop of (Expand -> High Loss -> Expand -> ...)
            prev_loss = float('inf')
            
            # RESET SCHEDULER (New Optimizer -> New Scheduler)
            if USE_SCHEDULER:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    trainer.optimizer, 
                    T_0=SCHEDULER_T0, 
                    eta_min=SCHEDULER_ETA_MIN
                )
            
        else:
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
        save_checkpoint(model, trainer.optimizer, epoch, avg_loss, CKPT_PATH, extra_data={'initial_lr': trainer.initial_lr})
        print(f"💾 Checkpoint Saved: {CKPT_PATH}")
        
        # SAVE BEST
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, trainer.optimizer, epoch, avg_loss, CKPT_BEST_PATH, extra_data={'initial_lr': trainer.initial_lr})
            print(f"🏆 NEW RECORD! Best Checkpoint Saved: {CKPT_BEST_PATH} (Loss: {best_loss:.4f})")
        
        epoch += 1

if __name__ == "__main__":
    main()
