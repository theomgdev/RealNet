
import torch
import os
import argparse
from realnet_llm.config import RealNetConfig
from realnet_llm.model import RealNetLM

def bits_to_int(bits):
    """Converts (1, 32) bit tensor to integer."""
    mask = 2 ** torch.arange(31, -1, -1, device=bits.device)
    return (bits * mask).sum(dim=-1).long().item()

def main():
    parser = argparse.ArgumentParser(description="RealNet Infinite Chat Interface")
    parser.add_argument('--ckpt', type=str, default='out_shakespeare_infinite/latest_stateful.pt', help='Checkpoint path')
    parser.add_argument('--len', type=int, default=100, help='Default generation length')
    parser.add_argument('--temp', type=float, default=0.8, help='Temperature')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load Config (Assume default small config for now, or try to load from ckpt if saved)
    # Ideally config should be saved with ckpt. trainer.py saves it.
    
    print(f"Loading model from {args.ckpt}...")
    
    if not os.path.exists(args.ckpt):
        # Fallback to shakespeare default if infinite not found
        fallback = 'out_shakespeare/best_ckpt.pt'
        if os.path.exists(fallback):
            print(f"Infinite ckpt not found, using {fallback}")
            args.ckpt = fallback
        else:
            print("No checkpoint found! Please train first.")
            return

    # Load Checkpoint
    checkpoint = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    
    # Extract Config
    if 'config' in checkpoint:
        # It might be TrainingConfig object which contains model config? 
        # Or did we save raw config? trainer.py saves 'config': self.config (TrainingConfig).
        # We need Model Config.
        # Wait, LMTrainer.save_checkpoint saves self.config which is TrainingConfig.
        # It does NOT save Model Config explicitly unless TrainingConfig has it?
        # Typically we just re-instantiate with known params for this hack.
        # Let's try to find it or assume default.
        pass
        
    # Hardcoded Config matches train_shakespeare.py for now
    config = RealNetConfig(
        n_neurons=1024,
        n_layers=1,
        thinking_steps=5,
        dropout=0.1,
        compile=False
    )
    
    model = RealNetLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    print("Model loaded. Infinite Context Engine Online.")
    print("------------------------------------------------")
    print("Usage:")
    print(" - Type text + Enter: Feed text to state & Generate continuation")
    print(" - Just Enter: Continue generating from current state")
    print(" - 'reset': Reset neural state")
    print(" - 'quit': Exit")
    print("------------------------------------------------")
    
    # INFINITE STATE
    current_state = None
    
    # We also need to track the last token to seed the next generation if input is empty?
    # inference_step needs an input token.
    # If user inputs "ABC", we feed A, B, C. Last input is C.
    # Next gen starts with C? Or predicted after C?
    # inference_step(C) -> predicts D.
    # So we feed C, get state, and logits for D.
    # If we want to generate, we sample D, then feed D.
    
    last_token_idx = 0 # Default NULL
    
    while True:
        try:
            user_input = input("\n>> ")
        except EOFError:
            break
            
        if user_input.strip() == 'quit':
            break
        if user_input.strip() == 'reset':
            current_state = None
            last_token_idx = 0
            print("[State Reset]")
            continue
            
        # 1. Digest User Input (if any)
        if user_input:
            input_ids = [ord(c) for c in user_input]
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=args.device)
            
            # Feed continuously
            with torch.no_grad():
                for i in range(len(input_ids)):
                    token = input_tensor[i].view(1, 1) # (1, 1)
                    # We don't care about logits here, just state update
                    logits, current_state = model.inference_step(token, state=current_state)
            
            # The last token fed is input_ids[-1]
            # The model is now predicting the NEXT token after user input.
            last_token_idx = input_ids[-1]
            
        # 2. Generate
        # If user just hit enter, we continue from `last_token_idx` and `current_state`.
        
        print(f"[{args.len} chars] ", end="", flush=True)
        
        generated_count = 0
        current_token = torch.tensor([[last_token_idx]], dtype=torch.long, device=args.device)
        
        with torch.no_grad():
            for _ in range(args.len):
                # If we just fed user input, `model.inference_step` was already called on the last char?
                # No, inside the loop we called it.
                # So `logits` from that last call corresponds to the prediction for NEXT char.
                # But `inference_step` returns state AFTER processing input.
                # So to get the prediction, we need to run inference_step?
                
                # Wait, RealNet input is the current character. Output is prediction of NEXT.
                # If user typed "Hello", we fed H, e, l, l, o.
                # After feeding 'o', `current_state` is updated `h_t`.
                # And `logits` (from the last iteration of feed loop) is the prediction for the char AFTER 'o'.
                
                # So if we have `logits` available from the feed phase, we should sample it FIRST.
                # But simplifying:
                # We can just re-feed the last token to get the prediction?
                # No, that would advance state effectively feeding 'o' twice? 
                # RealNet is generic RNN. Feeding x updates state h.
                # If we feed 'o' again, state changes again.
                
                # Handling "Just Enter":
                # We need the *prediction* from the previous step.
                # But inference_step computes (Input + State) -> (New State) -> (Output).
                # So if we fed 'o', we got New State and Output.
                # We should use that Output to sample the First generated char.
                # Then feed that char to get next.
                
                # Refined Logic:
                # We need to hold onto the `logits` from the very last operation.
                pass 
                
        # Actually simplest way:
        # Always feed `current_token`.
        # If User Input:
        #   Feed U1 -> predict U2
        #   Feed U2 -> predict U3
        #   ...
        #   Feed Un -> predict Gen1
        #   Set current_token = Gen1
        
        # If Just Enter:
        #   We don't have a "next token" to feed manually?
        #   Ah, we must use the *last generated token* as input.
        #   So `current_token` variable must persist!
        
        # Let's fix the User Input loop to update `current_token` correctly.
        
        if user_input:
             # We already ran the loop above.
             # But let's redo it cleanly.
             pass

    # REWRITE MAIN LOOP FOR CLARITY
    last_logits = None
    
    # Ensure state is valid
    # Start with a dummy input or just wait?
    # If user just hits enter at start, we need a seed.
    # Default seed: Space or 0.
    
    # Let's just run.
    
    while True:
        try:
            line = input("\n>> ")
        except EOFError:
            break
            
        if line.strip() == 'quit': break
        if line.strip() == 'reset': 
             current_state = None
             print("State cleared.")
             continue
             
        # Prepare to generate
        # If line is not empty, feed it.
        # If line is empty, continue generating from last predicted token.
        
        with torch.no_grad():
            if line:
                ids = [ord(c) for c in line]
                t_ids = torch.tensor(ids, dtype=torch.long, device=args.device)
                
                # Feed all inputs
                for i in range(len(ids)):
                    # Step
                    tok = t_ids[i].view(1, 1)
                    last_logits, current_state = model.inference_step(tok, state=current_state)
            
            # Now `last_logits` holds the prediction for the character AFTER the text (or previous gen).
            if last_logits is None:
                # Cold start with no input
                print("Start by typing something!")
                continue
                
            # Generate N chars
            print(" -> ", end="", flush=True)
            
            for _ in range(args.len):
                # Sample from last_logits
                # 32 independent bits
                bits_logits = last_logits[:, -1, :] # (1, 32)
                probs = torch.sigmoid(bits_logits / args.temp)
                next_bits = torch.bernoulli(probs)
                
                # Decode
                idx_next = bits_to_int(next_bits)
                
                # Print
                try:
                    char = chr(idx_next)
                    print(char, end="", flush=True)
                except:
                    print("?", end="", flush=True)
                    
                # Feed back as next input
                next_tok = torch.tensor([[idx_next]], dtype=torch.long, device=args.device)
                last_logits, current_state = model.inference_step(next_tok, state=current_state)

if __name__ == '__main__':
    main()
