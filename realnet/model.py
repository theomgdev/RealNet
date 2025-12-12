import torch
import torch.nn as nn
import numpy as np

class RealNet(nn.Module):
    def __init__(self, num_neurons, input_ids, output_ids, pulse_mode=True, dropout_rate=0.1, device='cpu'):
        super(RealNet, self).__init__()
        self.num_neurons = num_neurons
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.pulse_mode = pulse_mode
        self.device = device

        # Initialization
        # W: N x N weights. Anyone can talk to anyone.
        self.W = nn.Parameter(torch.randn(num_neurons, num_neurons, device=device) * 0.02)
        
        # B: Bias vector.
        self.B = nn.Parameter(torch.zeros(num_neurons, device=device))

        # Architecturally defined components
        self.norm = nn.LayerNorm(num_neurons).to(device) # StepNorm
        self.act = nn.GELU() # Flow Activation
        self.drop = nn.Dropout(p=dropout_rate) # Biological Failure Simulation

        # Internal State (hidden state h_t)
        self.state = torch.zeros(1, num_neurons, device=device)
        
        # PRUNING MASK (Synaptic Life)
        # 1 = Alive, 0 = Dead
        self.register_buffer('mask', torch.ones(num_neurons, num_neurons, device=device))

    def forward(self, x_input, steps=1, current_state=None):
        """
        Runs the dynamic system for `steps` timesteps.
        """
        if current_state is None:
            if self.state.shape[0] != (x_input.shape[0] if x_input is not None else 1):
                self.reset_state(batch_size=(x_input.shape[0] if x_input is not None else 1))
            current_state = self.state
        
        if current_state.device != self.device:
            current_state = current_state.to(self.device)

        h_t = current_state
        outputs = []

        batch_size = h_t.shape[0]

        # Apply Mask to Weights (Dead synapses transmit nothing)
        # Note: In sparse mode, we use self.W_sparse directly
        effective_W = self.W * self.mask if not self.is_sparse else self.W_sparse

        for t in range(steps):
            # 1. Chaotic Transmission
            if self.is_sparse:
                # OPTIMIZED SPARSE MULTIPLICATION
                # Matrix Math Trick: (h_t @ W) <=> (W.T @ h_t.T).T
                # PyTorch supports: Sparse(N,N) @ Dense(N,B) -> Dense(N,B)
                # So we use the transposed weight matrix (W_t_sparse) against transposed state
                
                # Check if we have cached sparse transpose
                if not hasattr(self, 'W_t_sparse'):
                    self.W_t_sparse = effective_W.t().coalesce()
                
                # h_t is (Batch, N), we need (N, Batch) for spmm
                signal_T = torch.sparse.mm(self.W_t_sparse, h_t.t())
                signal = signal_T.t() # Back to (Batch, N)
            else:
                # STANDARD DENSE MULTIPLICATION
                signal = torch.matmul(h_t, effective_W)
            
            # 2. Add Character (Bias)
            signal = signal + self.B
            
            # 3. Add Input (Impulse or Continuous)
            if x_input is not None:
                if self.pulse_mode:
                    if t == 0:
                        signal = signal + x_input
                else:
                    signal = signal + x_input

            # 4. Flow Activation (GELU) & StepNorm
            activated = self.act(signal)
            normalized = self.norm(activated)
            h_t = self.drop(normalized)
            
            outputs.append(h_t)

        return torch.stack(outputs), h_t

    def prune_synapses(self, threshold=0.001):
        """
        Kills connections (synapses) that are too weak.
        This is permanent (until manually reset).
        Returns: Number of pruned connections.
        """
        with torch.no_grad():
            # Find weak connections (Absolute weight is small)
            # But DO NOT prune connections that are already dead (mask=0)
            weak_links = (torch.abs(self.W) < threshold) & (self.mask == 1.0)
            
            # Kill them
            self.mask[weak_links] = 0.0
            
            # Enforce death on the actual weights too (optional, but cleaner)
            self.W.data = self.W.data * self.mask
            
            dead_count = (self.mask == 0.0).sum().item()
            total_count = self.mask.numel()
            return weak_links.sum().item(), dead_count, total_count

    def get_sparsity(self):
        dead = (self.mask == 0.0).sum().item()
        total = self.mask.numel()
        return dead / total * 100.0

    def reset_state(self, batch_size=1):
        self.state = torch.zeros(batch_size, self.num_neurons, device=self.device)
        if not hasattr(self, 'is_sparse'):
            self.is_sparse = False

    def make_sparse(self):
        """
        Converts the masked weights to a Sparse Tensor.
        This enables efficient 'True Sparsity' computation.
        """
        with torch.no_grad():
            masked_W = self.W * self.mask
            self.W_sparse = masked_W.to_sparse()
            self.W_t_sparse = self.W_sparse.t().coalesce()
            self.is_sparse = True
    
    def to_dense(self):
        self.is_sparse = False
        if hasattr(self, 'W_sparse'):
            del self.W_sparse
        if hasattr(self, 'W_t_sparse'):
            del self.W_t_sparse
