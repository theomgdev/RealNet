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
            device=dense_model.device,
            activation=getattr(dense_model, 'activation_type', 'gelu') # Default to GELU if legacy model
        )
        
        # Copy State Dictionary (weights, biases, mask, etc.)
        self.load_state_dict(dense_model.state_dict())
        
        # Copy new Learnable Parameters explicitly to be sure
        if hasattr(dense_model, 'input_scale'):
            self.input_scale = nn.Parameter(dense_model.input_scale.data.clone())
        if hasattr(dense_model, 'output_scale'):
            self.output_scale = nn.Parameter(dense_model.output_scale.data.clone())
        
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

        # Calculate Thinking Ratio (Native Temporal Stretching)
        ratio = 1
        max_outputs = steps

        if x_input is not None:
             if x_input.dtype in [torch.long, torch.int64, torch.int32] and x_input.ndim == 2:
                  # Index-based Sequential
                  if x_input.shape[1] > 0:
                       ratio = max(1, steps // x_input.shape[1])
                       max_outputs = x_input.shape[1]
             elif x_input.ndim == 3 and not self.pulse_mode:
                  # Dense Sequential
                  if x_input.shape[1] > 0:
                       ratio = max(1, steps // x_input.shape[1])
                       max_outputs = x_input.shape[1]

        for t in range(steps):
            # Prepare Input Step
            x_step = None
            if x_input is not None:
                # Handle Index-Based Input
                 if x_input.dtype in [torch.long, torch.int64, torch.int32]:
                      if x_input.ndim == 2:
                           if t % ratio == 0 and (t // ratio) < x_input.shape[1]:
                                token_indices = x_input[:, t // ratio]
                                valid_mask = token_indices != -1
                                if valid_mask.any():
                                     # Sparse injection not fully supported, convert step to dense
                                     offset = self.input_ids[0]
                                     valid_neurons = token_indices[valid_mask] + offset
                                     
                                     # Scaling
                                     scale_indices = valid_neurons - offset
                                     
                                     x_step_dense = torch.zeros(batch_sz, self.num_neurons, device=self.device)
                                     x_step_dense[valid_mask, valid_neurons] = 1.0 * self.input_scale[scale_indices]
                                     x_step = x_step_dense
                                     
                 elif x_input.ndim == 3:
                    if t % ratio == 0 and (t // ratio) < x_input.shape[1]:
                        x_step = x_input[:, t // ratio, :]
                        
                 elif self.pulse_mode:
                    if t == 0:
                        x_step = x_input
                 else:
                    x_step = x_input
            
            # Apply Input Scaling for Dense Inputs
            if x_step is not None and (x_input.dtype not in [torch.long, torch.int64, torch.int32]):
                 # Clone and Scale
                 x_step = x_step.clone()
                 x_step[:, self.input_pos] = x_step[:, self.input_pos] * self.input_scale

            # 1. Chaotic Transmission (SPARSE)
            # h_t is (Batch, N), we need (N, Batch) for spmm
            # signal = h_t @ W + B
            signal_T = torch.sparse.mm(self.W_t_sparse, h_t.t())
            signal = signal_T.t() + self.B
            
            if x_step is not None:
                signal = signal + x_step
            activated = self.act(signal)

            # 2. Dropout & 3. StepNorm
            h_t = self.norm(self.drop(activated))
            
            # Smart Output Collection
            if (t + 1) % ratio == 0 and len(outputs) < max_outputs:
                outputs.append(h_t)

        # Apply Output Scaling
        stacked_outputs = torch.stack(outputs, dim=1)
        stacked_outputs[:, :, self.output_pos] = stacked_outputs[:, :, self.output_pos] * self.output_scale

        return stacked_outputs, h_t

    def regenerate_weak_weights(self, threshold=0.01, percentage=None):
        """
        Overrides RealNet regeneration to update sparse matrices after modification.
        """
        # 1. Update Dense Weights
        revived, total = super(SparseRealNet, self).regenerate_weak_weights(threshold, percentage)
        
        # 2. Update Sparse Matrices
        if revived > 0:
             self._sparsify_weights()
             
        return revived, total

    @classmethod
    def from_dense(cls, model):
        """
        Factory method to create a SparseRealNet from a RealNet.
        """
        return cls(model)
