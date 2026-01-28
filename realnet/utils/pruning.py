import torch

class SynapticPruner:
    """
    Handles Darwinian pruning logic for RealNet models.
    Operates externally on the model's weights and mask.
    """
    
    @staticmethod
    def prune(model, threshold=0.001):
        """
        Kills connections (synapses) that are too weak.
        This modifies the model in-place.
        
        Args:
            model (RealNet): The model to prune.
            threshold (float): Absolute weight threshold below which synapses die.
            
        Returns:
            (pruned_count, dead_count, total_count)
        """
        # Ensure model has a mask buffer
        if not hasattr(model, 'mask'):
            raise AttributeError("Model does not have a 'mask' buffer required for pruning.")

        with torch.no_grad():
            # Find weak connections (Absolute weight is small)
            # But DO NOT prune connections that are already dead (mask=0)
            weak_links = (torch.abs(model.W) < threshold) & (model.mask == 1.0)
            
            # Kill them
            model.mask[weak_links] = 0.0
            
            # Enforce death on the actual weights too
            model.W.data = model.W.data * model.mask
            
            pruned_count = weak_links.sum().item()
            total_count = model.mask.numel()
            dead_count = (model.mask == 0.0).sum().item()

            # --- SWIGLU GATE PRUNING ---
            # Gates also need to evolve/die
            if getattr(model, 'is_swiglu', False) and hasattr(model, 'mask_gate'):
                 weak_gate_links = (torch.abs(model.W_gate) < threshold) & (model.mask_gate == 1.0)
                 model.mask_gate[weak_gate_links] = 0.0
                 model.W_gate.data = model.W_gate.data * model.mask_gate
                 
                 pruned_gate = weak_gate_links.sum().item()
                 pruned_count += pruned_gate
                 total_count += model.mask_gate.numel()
                 dead_count += (model.mask_gate == 0.0).sum().item()

            # --- SPARSE MODEL SUPPORT ---
            # If this is a SparseRealNet, we must update the Cached Sparse Matrices
            if hasattr(model, '_sparsify_weights'):
                 model._sparsify_weights()
            
            return pruned_count, dead_count, total_count

    @staticmethod
    def get_sparsity(model):
        if not hasattr(model, 'mask'):
            return 0.0
        
        dead = (model.mask == 0.0).sum().item()
        total = model.mask.numel()
        return dead / total * 100.0
