import torch
import torch.nn as nn
import numpy as np

class RealNet(nn.Module):
    def __init__(self, num_neurons, input_ids, output_ids, pulse_mode=True, device='cpu'):
        super(RealNet, self).__init__()
        self.num_neurons = num_neurons
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.pulse_mode = pulse_mode
        self.device = device

        # Initialization
        # W: N x N weights. Anyone can talk to anyone.
        # Initialized with small random values N(0, 0.02) to start with "Silence".
        self.W = nn.Parameter(torch.randn(num_neurons, num_neurons, device=device) * 0.02)
        
        # B: Bias vector. Each neuron has a tendency.
        # Zeros or small random values.
        self.B = nn.Parameter(torch.zeros(num_neurons, device=device))

        # Architecturally defined components
        self.norm = nn.LayerNorm(num_neurons).to(device) # StepNorm
        self.act = nn.GELU() # Flow Activation

        # Internal State (hidden state h_t)
        self.state = torch.zeros(1, num_neurons, device=device)

    def forward(self, x_input, steps=1, current_state=None):
        """
        Runs the dynamic system for `steps` timesteps.
        
        x_input: Tensor of shape (batch, num_neurons) or None.
                 If pulse_mode is True, this is added only at t=0.
                 If pulse_mode is False, this is added at every step.
        
        current_state: Initial state h_{t-1}. If None, uses internal self.state.
        
        Returns:
            outputs: list of state tensors for each step [h_t, h_{t+1}, ...]
            final_state: tensor h_{t+steps}
        """
        
        if current_state is None:
            # If internal state doesn't match batch size, reset it
            if self.state.shape[0] != (x_input.shape[0] if x_input is not None else 1):
                self.reset_state(batch_size=(x_input.shape[0] if x_input is not None else 1))
            current_state = self.state
        
        # Ensure state is on correct device
        if current_state.device != self.device:
            current_state = current_state.to(self.device)

        h_t = current_state
        outputs = []

        batch_size = h_t.shape[0]

        for t in range(steps):
            # 1. Chaotic Transmission: h_{t-1} @ W
            # Shape: (Batch, N) @ (N, N) -> (Batch, N)
            signal = torch.matmul(h_t, self.W)
            
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
            # The Manifesto says: StepNorm(GELU(...))
            activated = self.act(signal)
            h_t = self.norm(activated)
            
            outputs.append(h_t)

        return torch.stack(outputs), h_t

    def reset_state(self, batch_size=1):
        self.state = torch.zeros(batch_size, self.num_neurons, device=self.device)
