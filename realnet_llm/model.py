
import torch
import torch.nn as nn
from .config import RealNetConfig
from realnet import RealNet  # Core library

class RealNetLM(nn.Module):
    def __init__(self, config: RealNetConfig):
        super().__init__()
        self.config = config
        
        # 1. Input Interface: 32-bit Bitwise Projection
        # Replaces Embedding Table. We project 32 raw bits to N neurons.
        self.input_proj = nn.Linear(32, config.n_neurons)
        
        # 2. Backbone: RealNet
        # Conceptually we are driving the "whole brain" with these projected bits.
        input_ids = list(range(config.n_neurons)) 
        output_ids = list(range(config.n_neurons))
        
        self.backbone = RealNet(
            num_neurons=config.n_neurons,
            input_ids=input_ids,
            output_ids=output_ids,
            pulse_mode=config.pulse_mode,
            dropout_rate=config.dropout,
            device='cpu' # Will be moved later
        )
        
        # 3. Output Interface: N Neurons -> 32 Bits
        self.ln_f = nn.LayerNorm(config.n_neurons)
        self.output_proj = nn.Linear(config.n_neurons, 32, bias=True) # Bias allows default 0/1 lean
        
        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def to_bits(self, input_ids):
        """
        Converts Integer Code Points to 32-bit Binary Float Vectors.
        Args:
            input_ids: (Batch, Seq) or (Batch, 1) int64
        Returns:
            bits: (Batch, Seq, 32) float32
        """
        # Create a bit mask: [2^31, 2^30, ..., 2^0]
        # We cache this mask in the model buffers if possible, but here dynamic is fine.
        mask = 2 ** torch.arange(31, -1, -1, device=input_ids.device)
        
        # Bitwise AND to extract bits, then Boolean to Float
        # unsqueeze(-1) extends inputs to (B, S, 1) to broadcast against mask (32)
        # Result (B, S, 32)
        return (input_ids.unsqueeze(-1) & mask).ne(0).float()

    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids: (Batch, Seq_Len) - Sequence of Unicode Code Points (int64)
            targets: (Batch, Seq_Len) - Next Code Points (int64)
        Returns:
            logits: (Batch, Seq_Len, 32) - Logits for each bit
            loss: scalar (BCE)
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Convert to Bits
        x_bits = self.to_bits(input_ids) # (Batch, Seq, 32)
        
        # 2. Reset Backbone State
        self.backbone.reset_state(batch_size=batch_size)
        
        logits_list = []
        
        # 3. Iterate Sequence
        for t in range(seq_len):
            # Input Bits for this timestep
            xt = x_bits[:, t, :] # (Batch, 32)
            
            # Project to Neurons
            xt_neuron = self.input_proj(xt) # (Batch, N)
            
            # RealNet Step (Thinking)
            _, final_state = self.backbone(x_input=xt_neuron, steps=self.config.thinking_steps)
            
            # Readout
            h_norm = self.ln_f(final_state)
            logit = self.output_proj(h_norm) # (Batch, 32)
            
            logits_list.append(logit)
            
        # Stack: (Batch, Seq, 32)
        logits = torch.stack(logits_list, dim=1)
        
        loss = None
        if targets is not None:
            # We must predict the bits of the target
            target_bits = self.to_bits(targets) # (Batch, Seq, 32)
            
            # Binary Cross Entropy with Logits
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target_bits)
            
        return logits, loss

    def inference_step(self, input_ids, state=None):
        """
        O(1) Step for generation.
        Args:
            input_ids: (Batch, 1) Int64
        Returns:
            logits: (Batch, 1, 32)
            new_state: (Batch, N)
        """
        # 1. Convert to Bits
        x_bits = self.to_bits(input_ids)
        xt = x_bits[:, 0, :] # (Batch, 32)
        
        # 2. Project
        xt_neuron = self.input_proj(xt)
        
        # 3. RealNet Step
        _, new_state = self.backbone(x_input=xt_neuron, steps=self.config.thinking_steps, current_state=state)
        
        # 4. Readout
        h_norm = self.ln_f(new_state)
        logit = self.output_proj(h_norm) # (Batch, 32)
        
        return logit.unsqueeze(1), new_state

    def compile(self):
        """PyTorch 2.0 Compile"""
        if not self.config.compile:
            print("RealNetLM: Compilation disabled in config. Skipping.")
            return self

        if hasattr(torch, 'compile'):
            try:
                print("RealNetLM: Compiling...")
                compiled_model = torch.compile(self)
                
                # DRY RUN
                print("RealNetLM: Performing dry run...")
                dummy_ids = torch.zeros(1, 1, dtype=torch.long, device='cpu')
                p = next(self.parameters())
                dummy_ids = dummy_ids.to(p.device)
                
                with torch.no_grad():
                     compiled_model.inference_step(dummy_ids)
                
                print("RealNetLM: Compilation successful!")
                return compiled_model
            except Exception as e:
                print(f"RealNetLM: Compilation failed ({e}). Fallback to eager.")
                return self
        return self
