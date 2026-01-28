import torch
import torch.nn as nn
import gc
import os
if os.environ.get('NO_BNB'):
    HAS_BNB = False
else:
    try:
        os.environ["BITSANDBYTES_NOWELCOME"] = "1"
        # Check if user wants verbose logs (VERBOSE_BNB=1)
        if os.environ.get('VERBOSE_BNB'):
            import bitsandbytes as bnb
            HAS_BNB = True
        else:
            # Suppress annoying binary_path logs on Windows during import
            import sys
            _old_out, _old_err = sys.stdout, sys.stderr
            _null_out, _null_err = open(os.devnull, 'w'), open(os.devnull, 'w')
            try:
                sys.stdout, sys.stderr = _null_out, _null_err
                import bitsandbytes as bnb
                HAS_BNB = True
            finally:
                sys.stdout, sys.stderr = _old_out, _old_err
                _null_out.close()
                _null_err.close()
    except ImportError:
        HAS_BNB = False

class Neurogenesis:
    """
    Handles dynamic network growth (Neurogenesis) for RealNet models.
    """

    @staticmethod
    def expand(model, optimizer, amount=1, verbose=True):
        """
        Dynamically adds neurons to the network while preserving state and memory.
        
        Args:
            model (RealNet): The model to expand.
            optimizer (torch.optim.Optimizer): The current optimizer.
            amount (int): Number of neurons to add.
            verbose (bool): Whether to print status.
            
        Returns:
            torch.optim.Optimizer: The new optimizer instance.
        """
        if verbose:
            print(f"\n🌱 Neurogenesis: Growing Network {model.num_neurons} -> {model.num_neurons + amount} Neurons")
        
        old_n = model.num_neurons
        new_n = old_n + amount
        device = model.device
        
        # Check for SwiGLU
        is_swiglu = getattr(model, 'is_swiglu', False)
        
        # Preserve references to old parameters for state migration
        old_W_param = model.W
        old_B_param = model.B
        old_norm_w_param = model.norm.weight
        old_norm_b_param = model.norm.bias
        
        old_W_gate_param = model.W_gate if is_swiglu else None
        old_B_gate_param = model.B_gate if is_swiglu else None
        
        old_input_scale = model.input_scale
        old_output_scale = model.output_scale
        old_tau = model.tau
        
        old_opt = optimizer
        
        # 1. Expand Weights (W)
        # Strategy: Incoming=0 (Forward safe), Outgoing=Noise (Backprop safe)
        new_W = torch.zeros(new_n, new_n, device=device)
        new_W[:old_n, :old_n] = model.W.data
        
        # Symmetry Breaking: Initialize outgoing weights to small noise
        noise_std = 1e-5
        new_W[:old_n, old_n:] = torch.randn(old_n, amount, device=device) * noise_std
        
        # 2. Expand Bias (B)
        new_B = torch.zeros(new_n, device=device)
        new_B[:old_n] = model.B.data

        # --- Expand Gates (if SwiGLU) ---
        new_W_gate = None
        new_B_gate = None
        new_mask_gate = None
        
        if is_swiglu:
            # W_gate
            new_W_gate = torch.zeros(new_n, new_n, device=device)
            new_W_gate[:old_n, :old_n] = model.W_gate.data
            new_W_gate[:old_n, old_n:] = torch.randn(old_n, amount, device=device) * noise_std # Noise for outgoing
            
            # B_gate
            new_B_gate = torch.zeros(new_n, device=device)
            new_B_gate[:old_n] = model.B_gate.data
            
            # Mask Gate
            new_mask_gate = torch.ones(new_n, new_n, device=device)
            if hasattr(model, 'mask_gate'):
                 new_mask_gate[:old_n, :old_n] = model.mask_gate

        # 3. Expand Mask
        new_mask = torch.ones(new_n, new_n, device=device)
        if hasattr(model, 'mask'):
            new_mask[:old_n, :old_n] = model.mask
        
        # 4. Expand LayerNorm
        new_norm = nn.LayerNorm(new_n).to(device)
        with torch.no_grad():
            new_norm.weight[:old_n] = model.norm.weight.data
            new_norm.bias[:old_n] = model.norm.bias.data
            
        # 5. Expand State (if exists) - Pad with 0
        if hasattr(model, 'state'):
            new_state = torch.zeros(model.state.shape[0], new_n, device=device)
            new_state[:, :old_n] = model.state
            model.state = new_state
            
        # --- APPLY CHANGES TO MODEL ---
        model.num_neurons = new_n
        model.W = nn.Parameter(new_W)
        model.B = nn.Parameter(new_B)
        model.register_buffer('mask', new_mask)
        model.norm = new_norm
        
        if is_swiglu:
            model.W_gate = nn.Parameter(new_W_gate)
            model.B_gate = nn.Parameter(new_B_gate)
            model.register_buffer('mask_gate', new_mask_gate)
            
        # Re-bind scaling params (they are safe as is)
        model.input_scale = nn.Parameter(old_input_scale.data)
        model.output_scale = nn.Parameter(old_output_scale.data)
        
        # Expand Tau (Time Constant)
        new_tau = torch.full((new_n,), 0.0, device=device) # Init new neurons to Balanced (0.5)
        new_tau[:old_n] = old_tau.data
        model.tau = nn.Parameter(new_tau)
        
        # 6. OPTIMIZER MIGRATION
        # Create new optimizer dynamically based on old optimizer type
        group = old_opt.param_groups[0]
        optimizer_cls = type(old_opt)
        
        # Helper to safely get arg
        def get_arg(name, default):
            return group.get(name, default)

        # Re-instantiate optimizer (Generic)
        try:
             new_opt = optimizer_cls(
                model.parameters(), 
                lr=get_arg('lr', 0.001), 
                weight_decay=get_arg('weight_decay', 0), 
                betas=get_arg('betas', (0.9, 0.999)), 
                eps=get_arg('eps', 1e-8)
            )
        except Exception as e:
            print(f"⚠️ Optimizer re-init failed: {e}. Falling back to standard AdamW.")
            new_opt = torch.optim.AdamW(model.parameters(), lr=group['lr'])

        
        # Helper to migrate internal optimizer state (exp_avg, exp_avg_sq, state1, state2)
        def transfer_state(old_p, new_p, is_matrix=False):
            if old_p in old_opt.state:
                old_s = old_opt.state[old_p]
                new_s = {}
                
                # Copy scalar attributes (step, etc.)
                for k, v in old_s.items():
                    if not torch.is_tensor(v):
                        new_s[k] = v
                
                # Iterate over ALL keys in state (including qmap1, absmax1, etc.)
                for key, val in old_s.items():
                    if not torch.is_tensor(val):
                         # Already copied scalars above
                         continue
                         
                    try:
                        tensor = val
                        
                        # LOGIC: Does this state tensor match the parameter shape?
                        if tensor.shape == old_p.shape:
                            # Matches Parameter Shape -> Needs Resizing (Padding)
                            new_tensor = torch.zeros(new_p.shape, dtype=tensor.dtype, device=device)
                            
                            if is_matrix:
                                # 2D Parameter (W: NxN)
                                min_rows = min(tensor.shape[0], new_p.shape[0])
                                min_cols = min(tensor.shape[1], new_p.shape[1])
                                new_tensor[:min_rows, :min_cols] = tensor[:min_rows, :min_cols]
                            else:
                                # 1D Parameter (B: N)
                                min_len = min(tensor.shape[0], new_p.shape[0])
                                new_tensor[:min_len] = tensor[:min_len]
                                
                            new_s[key] = new_tensor
                            
                        else:
                            # Does NOT match parameter shape (Metadata like qmap1)
                            # Direct Copy
                            new_s[key] = tensor.clone()
                            
                    except Exception as e:
                        print(f"      Running into issue with key '{key}': {e}. Skipping.")
                        pass
                
                # Assign gathered state to new optimizer
                if new_s:
                    new_opt.state[new_p] = new_s
                
        # Transfer state for each parameter (Only for standard optimizers)
        is_bnb = HAS_BNB and isinstance(new_opt, (bnb.optim.Adam8bit, bnb.optim.AdamW8bit))
        
        if is_bnb:
             print("   👉 BNB Optimizer Detected. Skipping State Transfer (Cold Restart) to avoid quantization explosion.")
        else:
            try:
                transfer_state(old_W_param, model.W, is_matrix=True)
                transfer_state(old_B_param, model.B, is_matrix=False)
                transfer_state(old_norm_w_param, model.norm.weight, is_matrix=False)
                transfer_state(old_norm_b_param, model.norm.bias, is_matrix=False)
                
                if is_swiglu:
                    transfer_state(old_W_gate_param, model.W_gate, is_matrix=True)
                    transfer_state(old_B_gate_param, model.B_gate, is_matrix=False)
                
                transfer_state(old_input_scale, model.input_scale, is_matrix=False)
                transfer_state(old_output_scale, model.output_scale, is_matrix=False)
                transfer_state(old_tau, model.tau, is_matrix=False)
                
                print("   ✅ Optimizer State Transferred (Momentum Preserved)")
            except Exception as e:
                print(f"   ⚠️ Optimizer State Transfer Failed ({e}). Performing Cold Restart.")
        
        # 7. CLEANUP MEMORY
        del old_W_param
        del old_B_param
        del old_norm_w_param
        del old_norm_b_param
        del old_opt
        
        if is_swiglu:
            del old_W_gate_param
            del old_B_gate_param
            
        del old_input_scale
        del old_output_scale
        del old_tau
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if verbose:
            print(f"✅ Neurogenesis Complete. Optimizer Migrated. New Size: {new_n}. VRAM Cleared.")
            
        return new_opt
