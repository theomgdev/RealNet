
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import math
from .config import TrainingConfig
from .model import RealNetLM
from .model import RealNetLM
from .data import UnicodeDataset

class LMTrainer:
    def __init__(self, model: RealNetLM, dataset: UnicodeDataset, config: TrainingConfig):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = config.device
        
        self.model.to(self.device)
        self.model = self.model.compile() # PyTorch 2.0
        
        # Optimizer
        self.optimizer = self.configure_optimizers()
        
        # Scaler
        if hasattr(torch.amp, 'GradScaler'):
             self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device == 'cuda'))
        else:
             self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))
             
        # State
        self.iter_num = 0
        self.best_val_loss = float('inf')
        
    def configure_optimizers(self):
        # Separate weight decay params
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # Create optim groups. Any 2D parameter will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = optim.AdamW(optim_groups, lr=self.config.learning_rate, betas=(0.9, 0.99))
        return optimizer

    def get_lr(self, it):
        # Cosine Decay with Warmup
        if it < self.config.warmup_steps:
            return self.config.learning_rate * (it + 1) / (self.config.warmup_steps + 1)
        if it > self.config.max_steps:
            return self.config.min_lr
            
        decay_ratio = (it - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    def train(self, generation_callback=None):
        print(f"Starting training on {self.device}...")
        self.model.train()
        
        os.makedirs(self.config.out_dir, exist_ok=True)
        
        t0 = time.time()
        
        while self.iter_num < self.config.max_steps:
            # 1. Update Learning Rate
            lr = self.get_lr(self.iter_num)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            # 2. Gradient Accumulation Loop
            # We want total batch_size. We fetch micro_batches.
            # Usually input: batch_size_total = batch_size_micro * grad_accum
            # Our config.batch_size IS the micro batch size.
            
            accum_loss = 0.0
            
            for micro_step in range(self.config.gradient_accumulation_steps):
                x, y = self.dataset.get_split_batch('train', self.config.batch_size, device=self.device)
                
                # Context Management
                device_type = 'cuda' if 'cuda' in self.device else 'cpu'
                if hasattr(torch.amp, 'autocast'):
                     ctx = torch.amp.autocast(device_type=device_type, enabled=(device_type=='cuda'))
                else:
                     ctx = torch.cuda.amp.autocast(enabled=(device_type=='cuda'))
                     
                with ctx:
                     logits, loss, _ = self.model(x, y)
                     # Scale loss for accumulation
                     loss = loss / self.config.gradient_accumulation_steps
                
                # Backward
                self.scaler.scale(loss).backward()
                accum_loss += loss.item()
                
            # 3. Step
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # 4. Logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            
            if self.iter_num % self.config.log_interval == 0:
                print(f"iter {self.iter_num}: loss {accum_loss:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")
                
            # 5. Evaluation
            if self.iter_num > 0 and self.iter_num % self.config.eval_interval == 0:
                val_loss = self.evaluate()
                print(f"--> val loss {val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f"best_ckpt.pt")
            
            # 6. Regular Checkpoint
            if self.iter_num > 0 and self.iter_num % self.config.save_interval == 0:
                self.save_checkpoint(f"latest_ckpt.pt")

            # 7. Generation
            if self.iter_num > 0 and self.iter_num % self.config.eval_interval == 0:
                if generation_callback:
                    print("\n--- Generating Sample ---")
                    generation_callback(self.model)
                    print("\n-------------------------")

            self.iter_num += 1
            
    def evaluate(self):
        self.model.eval()
        losses = torch.zeros(20).to(self.device) # 20 batches
        for k in range(20):
             x, y = self.dataset.get_split_batch('val', self.config.batch_size, device=self.device)
             with torch.no_grad():
                  logits, loss, _ = self.model(x, y)
                  losses[k] = loss.item()
        self.model.train()
        return losses.mean().item()
        
    def save_checkpoint(self, filename):
        path = os.path.join(self.config.out_dir, filename)
        print(f"Saving checkpoint to {path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }, path)

    def load_checkpoint(self, path):
        print(f"Resuming from checkpoint: {path}")
        # Python 3.13 / PyTorch 2.6+ security default changed.
        # We trust our own checkpoints, so we disable weights_only to load custom configs.
        # Custom config object requires trusted load.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iter_num = checkpoint['iter_num']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed at iter {self.iter_num} with best val loss {self.best_val_loss:.4f}")
