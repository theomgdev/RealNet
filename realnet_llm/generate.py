
import torch
import torch.nn.functional as F
from .model import RealNetLM
from .config import RealNetConfig

def bits_to_int(bits):
    """
    Converts (1, 32) bit tensor to integer.
    """
    # Create mask: [2^31, ..., 2^0]
    mask = 2 ** torch.arange(31, -1, -1, device=bits.device)
    # Dot product
    return (bits * mask).sum(dim=-1).long().item()

def generate(model: RealNetLM, prompt: str, max_new_tokens: int = 50, temperature: float = 0.8, top_k: int = None, device='cpu'):
    """
    Generates text from a prompt using RealNetLM (32-bit Unicode Mode).
    """
    model.eval()
    model.to(device)
    
    # Encode Prompt: String -> Unicode Ints
    input_ids = [ord(c) for c in prompt]
    if not input_ids:
        print("Empty prompt.")
        return ""

    # Convert to tensor
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0) # (1, Seq)
    
    print(f"Prompt: {prompt}", end="", flush=True)
    
    # 1. Prefill (Process prompt to build state)
    state = None
    with torch.no_grad():
        # Feed context
        for i in range(input_tensor.size(1)):
            token = input_tensor[:, i].unsqueeze(1)
            logits, state = model.inference_step(token, state=state)
        
        # 2. Generation Loop
        next_token = input_tensor[:, -1].unsqueeze(1)
        
        for _ in range(max_new_tokens):
            # Step
            logits, state = model.inference_step(next_token, state=state)
            # logits: (1, 1, 32)
            
            bits_logits = logits[:, -1, :] # (1, 32)
            
            # Sigmoid Output: [0.0, 1.0] prob for each bit
            probs = torch.sigmoid(bits_logits)
            
            # Sampling Strategy for Bits
            # Temperature acts on logits? Or we can just use Bernoulli sampling.
            # Low temp: round(prob)
            # High temp: bernoulli(prob)
            
            # Let's simple Threshold at 0.5 for stability first, or Bernoulli?
            # Being a generative model, we want diversity.
            # But changing 1 bit in 32 drastically changes the char!
            # e.g. 'A' (65) vs 'C' (67) is bit 1 flipping.
            # With binary output, "Softmax" doesn't exist. "Top-K" doesn't exist.
            # We sample each bit independently.
            
            # Apply temperature to logits before sigmoid
            probs = torch.sigmoid(bits_logits / temperature)
            
            # Sample Bits
            next_bits = torch.bernoulli(probs) # (1, 32) floats 0.0 or 1.0
            
            # Reconstruct Integer
            idx_next = bits_to_int(next_bits)
            
            # Next Input
            next_token = torch.tensor([[idx_next]], dtype=torch.long, device=device)
            
            # Decode & Print
            try:
                char = chr(idx_next)
                print(char, end="", flush=True)
            except ValueError:
                # Invalid Unicode
                print("?", end="", flush=True)
            
    print("\n--- Done ---")

