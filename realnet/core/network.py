import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np

class RealNet(nn.Module):
    def __init__(self, num_neurons, input_ids, output_ids, pulse_mode=True, dropout_rate=0.1, device='cpu', weight_init='orthogonal', activation='tanh', gradient_checkpointing=False):
        super(RealNet, self).__init__()
        self.num_neurons = num_neurons
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.pulse_mode = pulse_mode
        self.activation_type = activation
        self.gradient_checkpointing = gradient_checkpointing
        self._device = device # Private variable for property
        
        # Initialization
        # W: N x N weights. Anyone can talk to anyone.
        self.W = nn.Parameter(torch.empty(num_neurons, num_neurons, device=device))
        self._init_weights(weight_init)
        
        # B: Bias vector.
        self.B = nn.Parameter(torch.zeros(num_neurons, device=device))

        # Architecturally defined components
        self.norm = nn.LayerNorm(num_neurons).to(device) # StepNorm
        
        # Activation Function
        if activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'silu':
             self.act = nn.SiLU()
        else:
             raise ValueError(f"Unknown activation function: {activation}")

        self.drop = nn.Dropout(p=dropout_rate) # Biological Failure Simulation

        # Internal State (hidden state h_t)
        self.state = torch.zeros(1, num_neurons, device=device)
        
        # PRUNING MASK (Synaptic Life)
        # 1 = Alive, 0 = Dead
        self.register_buffer('mask', torch.ones(num_neurons, num_neurons, device=device))

    def _init_weights(self, strategy):
        """
        Applies requested weight initialization strategy.
        """
        with torch.no_grad():
            if strategy == 'quiet':
                nn.init.normal_(self.W, mean=0.0, std=0.02)
            elif strategy == 'classic':
                nn.init.normal_(self.W)
            elif strategy == 'xavier_uniform':
                nn.init.xavier_uniform_(self.W)
            elif strategy == 'xavier_normal':
                nn.init.xavier_normal_(self.W)
            elif strategy == 'kaiming_uniform':
                nn.init.kaiming_uniform_(self.W, mode='fan_in', nonlinearity='relu') # GELU is close to ReLU
            elif strategy == 'kaiming_normal':
                nn.init.kaiming_normal_(self.W, mode='fan_in', nonlinearity='relu')
            elif strategy == 'orthogonal':
                nn.init.orthogonal_(self.W)
            elif strategy == 'sparse':
                # Experimental sparse init (connect only 10%)
                nn.init.sparse_(self.W, sparsity=0.9, std=0.02)
            elif strategy == 'zero':
                # Zero init - useful for transplantation (new neurons start silent)
                nn.init.zeros_(self.W)
            elif strategy == 'one':
                # One init - useful for transplantation (new neurons act as just a data bridge)
                nn.init.ones_(self.W)
            else:
                raise ValueError(f"Unknown weight_init strategy: {strategy}")

    def compile(self):
        """
        Compiles the model using PyTorch 2.0 torch.compile for faster execution.
        Returns the compiled model (in-place modification where possible).
        """
        if hasattr(torch, 'compile'):
            try:
                print("RealNet: Compiling model with torch.compile...")
                # Use 'inductor' backend explicitly or let it pick default.
                compiled_model = torch.compile(self)
                
                # FORCE DRY RUN to catch lazy errors now
                print("RealNet: Performing dry run to verify compilation...")
                dummy_input = torch.zeros(1, self.num_neurons, device=self.device)
                with torch.no_grad():
                    compiled_model(dummy_input, steps=1)
                
                print("RealNet: Compilation successful!")
                return compiled_model
            except Exception as e:
                print(f"RealNet: Compilation failed ({e}). Fallback to eager execution.")
                return self
        else:
            print("RealNet: torch.compile not found. Skipping compilation.")
            return self

    def forward(self, x_input, steps=1, current_state=None):
        """
        Runs the dynamic system for `steps` timesteps.
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

        # Apply Mask to Weights (Dead synapses transmit nothing)
        effective_W = self.W * self.mask

        def _single_step(h_t_in, t_idx, x_input_step):
            """Single timestep computation - can be checkpointed."""
            # 1. Chaotic Transmission (DENSE)
            signal = torch.matmul(h_t_in, effective_W)
            
            # 2. Add Character (Bias)
            signal = signal + self.B
            
            # 3. Add Input
            if x_input_step is not None:
                signal = signal + x_input_step
            
            # 4. Flow Activation (GELU) & StepNorm
            activated = self.act(signal)
            normalized = self.norm(activated)
            h_t_out = self.drop(normalized)
            
            return h_t_out

        for t in range(steps):
            # Prepare input for this step
            x_step = None
            if x_input is not None:
                if x_input.ndim == 3:
                    # Sequential Input: (Batch, Steps, Neurons)
                    if t < x_input.shape[1]:
                        x_step = x_input[:, t, :]
                elif self.pulse_mode:
                    if t == 0:
                        x_step = x_input
                else:
                    x_step = x_input
            
            # Use gradient checkpointing if enabled (saves VRAM, costs recomputation)
            if self.gradient_checkpointing and self.training:
                # checkpoint requires tensors, not None - use dummy if needed
                if x_step is None:
                    x_step = torch.zeros_like(h_t)
                h_t = checkpoint.checkpoint(_single_step, h_t, torch.tensor(t), x_step, use_reentrant=False)
            else:
                h_t = _single_step(h_t, t, x_step)
            
            outputs.append(h_t)

        return torch.stack(outputs), h_t

    def reset_state(self, batch_size=1):
        self.state = torch.zeros(batch_size, self.num_neurons, device=self.device)

    @property
    def device(self):
        return self.W.device
