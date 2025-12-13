
import torch
import os
import numpy as np
from typing import Tuple, Optional

class UnicodeDataset:
    """
    Handles loading text data and converting to Unicode Code Points (32-bit Integers).
    """
    def __init__(self, file_path: str, block_size: int = 128):
        self.file_path = file_path
        self.block_size = block_size
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.raw_text = f.read()
            
        # Convert chars to Unicode Code Points (Integers)
        # 'A' -> 65, '€' -> 8364, '😀' -> 128512
        self.tokens = [ord(c) for c in self.raw_text]
        
        # Convert to Tensor (int64 is needed for full Unicode range > 65535)
        print(f"Data loaded. Total chars: {len(self.tokens)}")
        self.data_tensor = torch.tensor(self.tokens, dtype=torch.long)
        
    def __len__(self):
        return len(self.data_tensor)

    def get_sequential_batch(self, start_index: int, batch_size: int, block_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a deterministic slice of data for sequential training (TBPTT).
        Args:
            start_index: The starting token index in the dataset.
        """
        # We need (Batch, Block_Size)
        # But for strictly sequential single-stream training (Batch=1), it is easy.
        # For Batch > 1, we need B parallel streams.
        # Let's support Batch=1 for "Infinite Context" purity first, or B streams.
        
        # Implementation for B parallel streams:
        # Split dataset into B chunks. Each row of batch reads from its own chunk.
        # This is complex to manage state for.
        
        # Simpler approach:
        # Just return ONE chunk [start : start+block].
        # Batching acts as "Parallel Universes" but here we just want 1 context stream?
        # No, we usually want batch training.
        
        # Let's implement BATCHED sequential reading:
        # Dataset is split into B segments.
        # Batch[0] reads from 0..L/B
        # Batch[1] reads from L/B..2L/B
        
        total_len = len(self.data_tensor)
        segment_len = total_len // batch_size
        
        ix = torch.arange(batch_size) * segment_len + start_index
        
        # Handle wrap around or limit (simply modulo or crop)
        # For infinite training, we can wrap around?
        ix = ix % (total_len - block_size - 1) 
        
        x = torch.stack([self.data_tensor[i:i+block_size] for i in ix])
        y = torch.stack([self.data_tensor[i+1:i+block_size+1] for i in ix])
        
        if device != 'cpu':
            x = x.to(device)
            y = y.to(device)
            
        return x, y

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

