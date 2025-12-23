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
DATA_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'wikisent2.txt'))
SEQ_LEN = 128
BATCH_SIZE = 512
NUM_NEURONS = 540
THINK_GAP = 5 # Number of silence steps between characters
EPOCHS = 1000 # Infinite training effectively
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- DATASET ---
class TextDataset(Dataset):
    def __init__(self, text, seq_len):
        self.text = text
        self.seq_len = seq_len
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        self.data_len = len(text) - seq_len

    def __len__(self):
        # Allow random sampling from the immense text
        return self.data_len // SEQ_LEN 

    def __getitem__(self, idx):
        # Random crop for training variety
        start_idx = np.random.randint(0, self.data_len)
        chunk = self.text[start_idx : start_idx + self.seq_len + 1]
        
        # Convert to indices
        indices = [self.char_to_idx[c] for c in chunk]
        
        # x: 0..N-1, y: 1..N
        x = torch.tensor(indices[:-1], dtype=torch.long)
        y = torch.tensor(indices[1:], dtype=torch.long)
        return x, y

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

def generate(model, dataset, start_str="The", length=100):
    model.eval()
    input_ids = dataset.char_to_idx
    idx_to_char = dataset.idx_to_char
    
    # Init String from text
    input_seq = [input_ids.get(c, 0) for c in start_str]
    current_state = None
    
    # 1. Warm up state with the prompt (dilated)
    # We need to feed the prompt exactly as we train: with gaps!
    # Otherwise the model rhythm is broken.
    
    x_in = torch.tensor(input_seq, dtype=torch.long, device=DEVICE).unsqueeze(0) # (1, Seq)
    x_emb = one_hot_encode_dilated(x_in, dataset.vocab_size, model.num_neurons, model.input_ids, THINK_GAP)
    
    with torch.no_grad():
        # Run full context
        _, current_state = model(x_emb, steps=x_emb.shape[1])
        
        # 2. Generate new characters
        last_char_idx = input_seq[-1]
        generated_text = start_str
        
        for _ in range(length):
            # A. Prepare Input (Single Char + Gap)
            # Create a sequence of length (1 input + GAP silence)
            total_step_single = 1 + THINK_GAP
            
            # Input Vector at t=0
            x_next_emb = torch.zeros(1, total_step_single, model.num_neurons, device=DEVICE)
            neuron_idx = last_char_idx + model.input_ids[0]
            x_next_emb[0, 0, neuron_idx] = 1.0
            
            # B. Run Model for (Gap+1) steps
            # We want the output at the END of this sequence (at step GAP)
            # forward returns (all_states, final_state)
            # all_states shape: (Steps, Batch, Neurons)
            
            preds, current_state = model(x_next_emb, steps=total_step_single, current_state=current_state)
            
            # C. Extract Prediction at the target step (thinking end)
            # preds shape: (Steps, Batch, Neurons)
            # Target is at step index `THINK_GAP`
            logits = preds[THINK_GAP, 0, model.output_ids] # (VocabSize)
            
            # D. Sample
            probs = torch.softmax(logits, dim=0)
            next_idx = torch.multinomial(probs, 1).item()
            
            generated_text += idx_to_char[next_idx]
            last_char_idx = next_idx
            
    return generated_text

def main():
    print(f"🚀 RealNet-1B (Prototyping on WikiSent)...")
    print(f"Loading {DATA_FILE}...")
    
    # Read text
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: Data file not found at {DATA_FILE}")
        return

    with open(DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read() 
    
    print(f"Text Loaded. Length: {len(text):,}")
    
    dataset = TextDataset(text, SEQ_LEN)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,        # Async data loading
        pin_memory=False,      # Faster GPU transfer
        prefetch_factor=2,    # Prefetch next batches
        persistent_workers=True  # Keep workers alive
    )
    
    print(f"Vocab Size: {dataset.vocab_size}")
    
    # --- MODEL SETUP ---
    # Input Neurons != Output Neurons to avoid signal collision
    input_ids = list(range(dataset.vocab_size))
    output_ids = list(range(dataset.vocab_size, 2 * dataset.vocab_size))
    
    print(f"Input IDs: {input_ids[0]}-{input_ids[-1]}")
    print(f"Output IDs: {output_ids[0]}-{output_ids[-1]}")
    
    model = RealNet(
        num_neurons=NUM_NEURONS,
        input_ids=input_ids,
        output_ids=output_ids,
        device=DEVICE,
        dropout_rate=0.0,
        activation='gelu', # Logic/Gating
        weight_init='quiet',
        gradient_checkpointing=True  # Save VRAM
    )
    
    # 🚀 COMPILE FOR SPEED (PyTorch 2.0)
    # Uncomment to enable compilation once stable
    # model = model.compile()
    
    trainer = RealNetTrainer(model, device=DEVICE, gradient_persistence=0) # Memory
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # --- CHECKPOINT SETUP (Using RealStore) ---
    CKPT_DIR = os.path.join(os.path.dirname(__file__), 'ckpt')
    os.makedirs(CKPT_DIR, exist_ok=True)
    CKPT_PATH = os.path.join(CKPT_DIR, 'llm_wikisent_latest.pth')
    
    start_epoch = 0
    if os.path.exists(CKPT_PATH):
        try:
            print(f"🔄 Loading Checkpoint from {CKPT_PATH}...")
            checkpoint = load_checkpoint(model, trainer.optimizer, CKPT_PATH, device=DEVICE, strict=True)
            start_epoch = checkpoint['epoch'] + 1
            print(f"✅ Resuming from Epoch {start_epoch}")
        except RuntimeError as e:
            # Size mismatch - try transplant
            print(f"⚠️ Size mismatch detected. Attempting Weight Transplantation...")
            transplant_weights(model, CKPT_PATH, device=DEVICE)
            print(f"🧬 Transplant complete. Starting from Epoch 0 with warm weights.")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}. Starting fresh.")
    
    # CrossEntropy with Masking (Thinking Gaps)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    trainer.loss_fn = criterion 
    
    # OUTPUT TRANSFORM: Flatten dilated output (Batch, Total_Steps, Out) -> (N, Out)
    def flatten_logits(out):
        return out.reshape(-1, dataset.vocab_size)

    # --- INITIAL GENERATION (Show current state) ---
    print("--- INITIAL GENERATION ---")
    try:
        gen_text = generate(model, dataset, start_str="The meaning of life is ", length=100)
        print(gen_text)
    except Exception as e:
        print(f"Generation Error: {e}")
    print("--------------------------")

    print("Training (Infinite)...")
    
    # Infinite Loop
    epoch = start_epoch
    while True:
        total_loss = 0
        steps = 0
        
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(dataloader):
            # Dilate Data (Insert Silence)
            x_emb = one_hot_encode_dilated(x, dataset.vocab_size, NUM_NEURONS, input_ids, THINK_GAP)
            y_dilated = prepare_targets_dilated(y, THINK_GAP)
            y_flat = y_dilated.reshape(-1) # Already on DEVICE
            
            # Total steps has increased due to dilation
            total_steps = SEQ_LEN * (THINK_GAP + 1)
            
            loss = trainer.train_batch(
                x_emb, 
                y_flat, 
                thinking_steps=total_steps, 
                full_sequence=True, 
                output_transform=flatten_logits 
            )
            
            total_loss += loss
            steps += 1
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss:.4f}")
                
            if batch_idx > 500: # Limit batches per epoch for frequent view
                break
                
        avg_loss = total_loss / steps
        print(f"Epoch {epoch} Completed | Avg Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.1f}s")
        
        # Generation Check
        print("--- GENERATION ---")
        try:
            gen_text = generate(model, dataset, start_str="The meaning of life is ", length=100)
            print(gen_text)
        except Exception as e:
            print(f"Generation Error: {e}")
        print("------------------")
        
        # SAVE CHECKPOINT (Using RealStore)
        save_checkpoint(model, trainer.optimizer, epoch, avg_loss, CKPT_PATH)
        print(f"💾 Checkpoint Saved: {CKPT_PATH}")
        
        # 🌱 PROGRESSIVE NEURAL GROWTH
        if GROW_PER_EPOCH > 0:
            old_n = model.num_neurons
            new_n = old_n + GROW_PER_EPOCH
            
            # Create new larger model
            new_input_ids = list(range(dataset.vocab_size))
            new_output_ids = list(range(dataset.vocab_size, 2 * dataset.vocab_size))
            
            new_model = RealNet(
                num_neurons=new_n,
                input_ids=new_input_ids,
                output_ids=new_output_ids,
                device=DEVICE,
                dropout_rate=0.0,
                activation='gelu',
                weight_init='zero',  # New neurons start silent
                gradient_checkpointing=True  # Save VRAM
            )
            
            # Manual transplant from old model
            old_state = model.state_dict()
            new_state = new_model.state_dict()
            
            # Copy overlapping weights
            new_state['W'][:old_n, :old_n] = old_state['W']
            new_state['B'][:old_n] = old_state['B']
            new_state['mask'][:old_n, :old_n] = old_state['mask']
            new_state['norm.weight'][:old_n] = old_state['norm.weight']
            new_state['norm.bias'][:old_n] = old_state['norm.bias']
            
            new_model.load_state_dict(new_state)
            
            print(f"🌱 Grew: {old_n} → {new_n} neurons")
            
            # Replace model and trainer
            model = new_model
            trainer = RealNetTrainer(model, device=DEVICE, gradient_persistence=0)
            trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            trainer.loss_fn = criterion
            
            # Update input_ids reference for next iteration
            input_ids = new_input_ids
        
        epoch += 1

if __name__ == "__main__":
    main()
