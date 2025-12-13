
import os
import torch
from realnet_llm.config import RealNetConfig, TrainingConfig
from realnet_llm.model import RealNetLM
from realnet_llm.data import UnicodeDataset
from realnet_llm.trainer import LMTrainer
try:
    from realnet_llm.generate import generate
except ImportError:
    generate = None
    
# Suppress Dynamo errors just in case
import torch._dynamo
torch._dynamo.config.suppress_errors = True

def main():
    print("Initializing training script...")
    # 1. Config
    # Use a small model for quick verification
    # Model Config
    model_config = RealNetConfig(
        n_neurons=1024,
        n_layers=1,
        thinking_steps=5, # Increased for bit-level reasoning
        dropout=0.1,
        compile=False # Disable compilation for comparison
    )
    
    train_config = TrainingConfig(
        batch_size=8192,
        gradient_accumulation_steps=1, # Effective 32
        learning_rate=5e-4,
        max_steps=5000000000, # Increased steps for bit learning
        eval_interval=50,
        log_interval=10,
        out_dir='out_realnet', # Shared unified brain directory
        context_window=8, # Tunable Context Length
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 2. Data
    data_path = 'data/shakespeare.txt'
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return
        
    dataset = UnicodeDataset(data_path, block_size=train_config.context_window)
    
    # Vocab Size is irrelevant now (Universal 32-bit)
    print(f"Dataset Loaded. 32-bit Unicode Mode.")
    
    # 3. Model
    model = RealNetLM(model_config)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 4. Trainer
    trainer = LMTrainer(model, dataset, train_config)
    
    # 5. Train
    print("Training...")
    
    # Auto-Resume
    ckpt_path = os.path.join(train_config.out_dir, 'latest_ckpt.pt')
    if os.path.exists(ckpt_path):
        trainer.load_checkpoint(ckpt_path)

    def gen_cb(model_ref):
        if generate:
            generate(model_ref, "To be or not to be", max_new_tokens=100, device=train_config.device)

    trainer.train(generation_callback=gen_cb)
    
    # 6. Generate
    if generate:
        print("\nGenerating sample text...")
        prompt = "To be or not to be"
        generate(model, prompt, max_new_tokens=100, device=train_config.device)

if __name__ == '__main__':
    main()
