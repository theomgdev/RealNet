
import torch

class RealNetVocab:
    def __init__(self):
        # RAW BYTE LEVEL VOCAB (0-255)
        # Supports ALL languages (UTF-8) including Turkish.
        # 'ğ' becomes 2 tokens: [196, 159]
        self.vocab_size = 256
        
    def encode(self, text):
        # Convert string directly to its UTF-8 byte representation
        # 'text' -> bytes -> list of ints
        if isinstance(text, str):
            bytes_data = text.encode('utf-8')
        else:
            bytes_data = text # Assume already bytes
            
        return torch.tensor(list(bytes_data), dtype=torch.long)
    
    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        # Convert list of ints back to bytes
        bytes_data = bytes(ids)
        
        # Decode UTF-8, replacing invalid sequences (e.g. partial bytes at end)
        return bytes_data.decode('utf-8', errors='replace')

    def get_vocab_size(self):
        return self.vocab_size
