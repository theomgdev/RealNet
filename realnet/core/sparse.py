import torch
import torch.nn as nn
from .network import RealNet

class SparseRealNet(RealNet):
    """
    A unified RealNet implementation optimized for Sparse operations.
    Inherits continuously from RealNet but overrides the forward pass 
    to utilize PyTorch Sparse Tensors for efficiency.
    """
    def __init__(self, dense_model):
        """
        Initializes a SparseRealNet from a pre-trained RealNet.
        
        Args:
            dense_model (RealNet): The trained dense model.
        """
        # Copy attributes from dense model
        super(SparseRealNet, self).__init__(
            num_neurons=dense_model.num_neurons,
            input_ids=dense_model.input_ids,
            output_ids=dense_model.output_ids,
            pulse_mode=dense_model.pulse_mode,
            dropout_rate=dense_model.drop.p,
            device=dense_model.device
        )
        
        # Copy State Dictionary (weights, biases, mask, etc.)
        self.load_state_dict(dense_model.state_dict())
        
        # Convert W to Sparse immediately
        self._sparsify_weights()
        
    def _sparsify_weights(self):
        """
        Internal method to convert Dense W to Sparse W based on mask.
        """
        with torch.no_grad():
            masked_W = self.W * self.mask
            self.W_sparse = masked_W.to_sparse()
            # Cache the transpose for valid matrix multiplication: (h_t @ W) -> (W.t @ h_t.t).t
            self.W_t_sparse = self.W_sparse.t().coalesce()
            
            # We don't need Dense W anymore in memory for computation, 
            # but we keep it referenced in parameters if we want to save/load cleanly.
            # Ideally, for pure inference optimization, we could delete self.W, 
            # but PyTorch nn.Module logic might complain.
            
    def forward(self, x_input, steps=1, current_state=None):
        """
        Sparse forward pass.
        """
        if current_state is None:
            batch_sz = x_input.shape[0] if x_input is not None else 1
            if self.state.shape[0] != batch_sz:
                self.reset_state(batch_size=batch_sz)
            current_state = self.state
        
        if current_state.device != self.device:
            current_state = current_state.to(self.device)

        h_t = current_state
        outputs = []

        for t in range(steps):
            # 1. Chaotic Transmission (SPARSE)
            # Matrix Math Trick: (h_t @ W) <=> (W.T @ h_t.T).T
            # PyTorch supports: Sparse(N,N) @ Dense(N,B) -> Dense(N,B)
            
            # h_t is (Batch, N), we need (N, Batch) for spmm
            signal_T = torch.sparse.mm(self.W_t_sparse, h_t.t())
            signal = signal_T.t() # Back to (Batch, N)
            
            # 2. Add Character (Bias)
            signal = signal + self.B
            
            # 3. Add Input
            if x_input is not None:
                if x_input.ndim == 3:
                     if t < x_input.shape[1]:
                         signal = signal + x_input[:, t, :]
                elif self.pulse_mode:
                    if t == 0:
                        signal = signal + x_input
                else:
                    signal = signal + x_input

            # 4. Activation & Norm
            activated = self.act(signal)
            normalized = self.norm(activated)
            h_t = self.drop(normalized)
            
            outputs.append(h_t)

        return torch.stack(outputs), h_t

    @classmethod
    def from_dense(cls, model):
        """
        Factory method to create a SparseRealNet from a RealNet.
        """
        return cls(model)
