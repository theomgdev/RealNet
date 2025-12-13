
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
    print("Initializing GSM8K (Math) Training...")
    # 1. Config (Matching Shakespeare settings)
    model_config = RealNetConfig(
        n_neurons=1024,
        n_layers=1,
        thinking_steps=4, # Fast recurrence
        dropout=0.1,
        compile=False
    )
    
    train_config = TrainingConfig(
        batch_size=8,
        gradient_accumulation_steps=1, 
        learning_rate=5e-4,
        max_steps=5000000000, 
        eval_interval=100, # Frequent checks for math
        log_interval=10,
        out_dir='out_realnet', # Shared unified brain directory
        context_window=8, # Tunable Context Length
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 2. Data
    data_path = 'data/gsm8k.txt'
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Run download_edu_data.py first.")
        return
        
    dataset = UnicodeDataset(data_path, block_size=train_config.context_window)
    print("Dataset Loaded. 32-bit Unicode Mode.")
    
    # 3. Model
    model = RealNetLM(model_config)
    
    # 4. Trainer
    trainer = LMTrainer(model, dataset, train_config)
    
    # 5. Load Checkpoint (Shared Brain)
    ckpt_path = os.path.join(train_config.out_dir, 'best_ckpt.pt')
    if os.path.exists(ckpt_path):
        trainer.load_checkpoint(ckpt_path)
    else:
        print("No existing brain found, starting fresh or run train_shakespeare.py first.")
    
    # 6. Train
    print("Training on GSM8K...")

    def gen_cb(model_ref):
        if generate:
            generate(model_ref, "Question: if I have 2 apples", max_new_tokens=100, device=train_config.device)

    trainer.train(generation_callback=gen_cb)

if __name__ == '__main__':
    main()
