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
        self._device = device # Private variable for property
        
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

        for t in range(steps):
            # 1. Chaotic Transmission (DENSE)
            signal = torch.matmul(h_t, effective_W)
            
            # 2. Add Character (Bias)
            signal = signal + self.B
            
            # 3. Add Input (Impulse, Continuous, or Sequence)
            if x_input is not None:
                if x_input.ndim == 3:
                     # Sequential Input: (Batch, Steps, Neurons)
                     if t < x_input.shape[1]:
                         signal = signal + x_input[:, t, :]
                elif self.pulse_mode:
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

    def reset_state(self, batch_size=1):
        self.state = torch.zeros(batch_size, self.num_neurons, device=self.device)

    @property
    def device(self):
        return self.W.device
