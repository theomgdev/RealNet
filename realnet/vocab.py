
import torch

class RealNetVocab:
    def __init__(self):
        # 1. Printable ASCII (32-126)
        self.chars = [chr(i) for i in range(32, 127)]
        
        # 2. Newline
        self.chars.append('\n')
        
        # 3. Turkish Symbols
        tr_chars = ['ç', 'ğ', 'ı', 'ö', 'ş', 'ü', 'Ç', 'Ğ', 'İ', 'Ö', 'Ş', 'Ü']
        self.chars.extend(tr_chars)
        
        # 4. Unknown Token
        self.unk_token = '<UNK>'
        self.chars.append(self.unk_token)
        
        # Mapping
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        self.vocab_size = len(self.chars)
        self.unk_id = self.char_to_id[self.unk_token]
        
    def encode(self, text):
        ids = []
        for ch in text:
            ids.append(self.char_to_id.get(ch, self.unk_id))
        return torch.tensor(ids, dtype=torch.long)
    
    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ''.join([self.id_to_char.get(i, self.unk_token) for i in ids])

    def get_vocab_size(self):
        return self.vocab_size
