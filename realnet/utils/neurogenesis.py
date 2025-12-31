import torch
import torch.nn as nn
import gc

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
        
        # Preserve references to old parameters for state migration
        old_W_param = model.W
        old_B_param = model.B
        old_norm_w_param = model.norm.weight
        old_norm_b_param = model.norm.bias
        old_opt = optimizer
        
        # 1. Expand Weights (W) - Init new connections to 0 for continuity
        new_W = torch.zeros(new_n, new_n, device=device)
        new_W[:old_n, :old_n] = model.W.data
        
        # 2. Expand Bias (B)
        new_B = torch.zeros(new_n, device=device)
        new_B[:old_n] = model.B.data
        
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
        
        # 6. OPTIMIZER MIGRATION
        # Create new optimizer with same settings
        group = old_opt.param_groups[0]
        
        new_opt = torch.optim.AdamW(
            model.parameters(), 
            lr=group['lr'], 
            weight_decay=group['weight_decay'], 
            betas=group['betas'], 
            eps=group['eps']
        )
        
        # Helper to migrate internal optimizer state (exp_avg, exp_avg_sq)
        def transfer_state(old_p, new_p, is_matrix=False):
            if old_p in old_opt.state:
                old_s = old_opt.state[old_p]
                new_s = {}
                if 'step' in old_s: 
                     new_s['step'] = old_s['step']
                
                # Exp Avg
                if 'exp_avg' in old_s:
                    ea = old_s['exp_avg']
                    new_ea = torch.zeros_like(new_p.data)
                    if is_matrix: 
                        new_ea[:old_n, :old_n] = ea
                    else: 
                        new_ea[:old_n] = ea
                    new_s['exp_avg'] = new_ea
                    
                # Exp Avg Sq
                if 'exp_avg_sq' in old_s:
                    eas = old_s['exp_avg_sq']
                    new_eas = torch.zeros_like(new_p.data)
                    if is_matrix: 
                        new_eas[:old_n, :old_n] = eas
                    else: 
                        new_eas[:old_n] = eas
                    new_s['exp_avg_sq'] = new_eas
                
                new_opt.state[new_p] = new_s
                
        # Transfer state for each parameter
        transfer_state(old_W_param, model.W, is_matrix=True)
        transfer_state(old_B_param, model.B, is_matrix=False)
        transfer_state(old_norm_w_param, model.norm.weight, is_matrix=False)
        transfer_state(old_norm_b_param, model.norm.bias, is_matrix=False)
        
        # 7. CLEANUP MEMORY
        del old_W_param
        del old_B_param
        del old_norm_w_param
        del old_norm_b_param
        del old_opt
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if verbose:
            print(f"✅ Neurogenesis Complete. Optimizer Migrated. New Size: {new_n}. VRAM Cleared.")
            
        return new_opt
