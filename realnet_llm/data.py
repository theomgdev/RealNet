
import torch
import os
import numpy as np
from typing import Tuple, Optional

# Try importing tiktoken, fallback to simple char-level if missing
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

class TextDataset:
    """
    Handles loading text data, tokenizing it, and serving batches.
    """
    def __init__(self, file_path: str, block_size: int = 128, encoding_name: str = "cl100k_base"):
        self.file_path = file_path
        self.block_size = block_size
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            self.raw_text = f.read()
            
        if HAS_TIKTOKEN:
            print(f"Tokenizing data with {encoding_name}...")
            enc = tiktoken.get_encoding(encoding_name)
            self.tokens = enc.encode_ordinary(self.raw_text) # list of ints
            self.vocab_size = enc.n_vocab
        else:
            print("TikToken not found. Falling back to Char-Level encoding.")
            chars = sorted(list(set(self.raw_text)))
            self.vocab_size = len(chars)
            self.stoi = { ch:i for i,ch in enumerate(chars) }
            self.itos = { i:ch for i,ch in enumerate(chars) }
            self.tokens = [self.stoi[c] for c in self.raw_text]
            
        # Convert to Tensor (in-memory for NOW, use memmap for huge datasets)
        print(f"Data loaded. Total tokens: {len(self.tokens)}")
        self.data_tensor = torch.tensor(self.tokens, dtype=torch.long)
        
    def get_batch(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a random batch of (inputs, targets).
        Targets are inputs shifted by 1.
        """
        # Random offsets
        ix = torch.randint(len(self.data_tensor) - self.block_size, (batch_size,))
        
        x = torch.stack([self.data_tensor[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data_tensor[i+1:i+self.block_size+1] for i in ix])
        
        if device != 'cpu':
            x = x.to(device)
            y = y.to(device)
            
        return x, y
    
    def get_split_batch(self, split: str, batch_size: int, device: str = 'cpu'):
        # Just simple split for MVP
        n = int(0.9 * len(self.data_tensor))
        train_data = self.data_tensor[:n]
        val_data = self.data_tensor[n:]
        
        data = train_data if split == 'train' else val_data
        
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        
        if device != 'cpu':
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
        return x, y

