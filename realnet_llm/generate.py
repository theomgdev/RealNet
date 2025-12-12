
import torch
import torch.nn.functional as F
from .model import RealNetLM
from .config import RealNetConfig
from .data import HAS_TIKTOKEN
import tiktoken

def generate(model: RealNetLM, prompt: str, max_new_tokens: int = 50, temperature: float = 0.8, top_k: int = 50, device='cpu'):
    """
    Generates text from a prompt using RealNetLM.
    """
    model.eval()
    model.to(device)
    
    # Encode
    if HAS_TIKTOKEN:
        enc = tiktoken.get_encoding("cl100k_base") # or config default
        input_ids = enc.encode(prompt)
    else:
        # Fallback char level (assuming dataset was trained on chars)
        # This is risky if models differ. For now assuming tiktoken.
        print("Warning: TikToken not found. Cannot encode prompt properly if model uses BPE.")
        return ""

    # Convert to tensor
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0) # (1, Seq)
    
    # 1. Prefill (Process prompt to build state)
    # We can use standard forward loop but we need the final state.
    # RealNetLM.forward doesn't return state. 
    # Let's use inference_step loop for prefill to be sure, or modify forward.
    # Loop manually:
    state = None
    print(f"Prompt: {prompt}", end="", flush=True)
    
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
            
            # Sample
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            next_token = idx_next
            
            # Decode & Print
            if HAS_TIKTOKEN:
                txt = enc.decode([idx_next.item()])
                print(txt, end="", flush=True)
            
    print("\n--- Done ---")
