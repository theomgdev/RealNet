import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ..utils.data import prepare_input, to_tensor
from ..utils.pruning import SynapticPruner
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

class RealNetTrainer:
    def __init__(self, model, optimizer=None, loss_fn=None, lr=1e-4, device='cpu', gradient_persistence=0.0, synaptic_noise=1e-6):
        """
        Initializes the trainer.
        
        Args:
            model (nn.Module): The RealNet model to train.
            optimizer (torch.optim.Optimizer): Custom optimizer (Optional).
            loss_fn (callable): Custom loss function (Optional).
            lr (float): Initial learning rate (Default: 1e-4). Used if optimizer is None.
            device (str): Device to run training on.
            gradient_persistence (float): How much gradient to keep from previous step (0.0 - 0.9).
            synaptic_noise (float): Scale of noise added to weights during training (Regularization). Default 1e-6.
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.gradient_persistence = gradient_persistence
        self.synaptic_noise = synaptic_noise
        self.initial_lr = lr
        
        # --- OPTIMIZER (8-bit AdamW if available) ---
        if optimizer:
            self.optimizer = optimizer
        elif HAS_BNB and device == 'cuda': 
             # HAS_BNB is strictly False if NO_BNB was set at import time.
             print("RealNetTrainer: Using bitsandbytes 8-bit AdamW for VRAM efficiency.")
             self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.01)
        else:
             print("RealNetTrainer: bitsandbytes not found or CPU mode. Using standard AdamW.")
             self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        self.loss_fn = loss_fn if loss_fn else nn.MSELoss()

    def train_batch(self, input_features, target_values, thinking_steps, gradient_accumulation_steps=1, full_sequence=False, mask=None, output_transform=None, initial_state=None, return_state=False):
        """
        Runs a single training step on a batch.
        """
        self.model.train()
        
        # Initialize Scaler (AMP)
        if not hasattr(self, 'scaler'):
            if hasattr(torch.amp, 'GradScaler'):
                self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device == 'cuda'))
            else:
                self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))

        # 1. Synaptic Noise (Thermal Noise / Langevin Dynamics)
        if self.synaptic_noise > 0.0:
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.requires_grad:
                        noise = torch.randn_like(param) * self.synaptic_noise
                        param.add_(noise)

        # 2. Prepare Data
        # Optimization: If passing indices (Long/Int), bypass prepare_input expansion
        if isinstance(input_features, torch.Tensor) and input_features.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
             x_input = input_features.to(self.device)
             batch_size = x_input.shape[0]
        else:
             x_input, batch_size = prepare_input(input_features, self.model.input_ids, self.model.num_neurons, self.device)
        
        target_values = to_tensor(target_values, self.device)
        if mask is not None:
            mask = to_tensor(mask, self.device)

        # 2. Reset State & Run (with AMP)
        device_type = 'cuda' if self.device == 'cuda' else 'cpu'
        if hasattr(torch.amp, 'autocast'):
             autocast_ctx = torch.amp.autocast(device_type=device_type, enabled=(self.device == 'cuda'))
        else:
             autocast_ctx = torch.cuda.amp.autocast(enabled=(self.device == 'cuda'))
             
        with autocast_ctx:
            # Use initial_state if provided, otherwise reset
            if initial_state is not None:
                current_state_in = initial_state
            else:
                self.model.reset_state(batch_size)
                current_state_in = None # Let model use its internal state or resetting logic if needed, but model.forward checks this.
                # Actually model.forward(..., current_state=None) triggers reset if needed inside.
            
            all_states, final_state = self.model(x_input, steps=thinking_steps, current_state=current_state_in)
            
            # 3. Extract Outputs & Calculate Loss
            output_indices = self.model.output_ids
            
            if full_sequence:
                # all_states: (Steps, Batch, Neurons)
                # read_outputs handles permutation and scaling
                predicted_outputs = self.model.read_outputs(all_states)
                # Note: target_values must also be (Batch, Steps, Output_Size) or compatible
            else:
                # read_outputs handles scaling
                predicted_outputs = self.model.read_outputs(final_state)
            
            # OPTIONAL TRANSFORM (e.g. for CrossEntropy reshaping)
            if output_transform:
                predicted_outputs = output_transform(predicted_outputs)

            if mask is not None:
                # Masked MSE Loss
                # Apply mask to squared errors
                loss = (torch.square(predicted_outputs - target_values) * mask).mean()
            else:
                loss = self.loss_fn(predicted_outputs, target_values)
            
            # Normalize loss for gradient accumulation
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

        # 4. Backward (with Scaler)
        self.scaler.scale(loss).backward()
        
        # Step optimizer only if accumulation cycle is complete
        step_now = True
        if gradient_accumulation_steps > 1:
             if not hasattr(self, '_acc_counter'):
                 self._acc_counter = 0
             self._acc_counter += 1
             if self._acc_counter % gradient_accumulation_steps != 0:
                 step_now = False

        if step_now:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.gradient_persistence > 0.0:
                 # Gradient Persistence (Ghost Gradients)
                 with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            # CRITICAL: If NaN/Inf, reset completely
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                param.grad.zero_()
                            else:
                                # Keep a fraction of the gradient
                                param.grad.mul_(self.gradient_persistence)
            else:
                 self.optimizer.zero_grad()
                 
            if hasattr(self, '_acc_counter'):
                self._acc_counter = 0

        # Return the SCALED loss for logging
        loss_val = loss.item() * gradient_accumulation_steps
        
        if return_state:
            return loss_val, final_state
        return loss_val

    def predict(self, input_features, thinking_steps, full_sequence=False):
        """
        Runs inference.
        """
        self.model.eval()
        with torch.no_grad():
            x_input, batch_size = prepare_input(input_features, self.model.input_ids, self.model.num_neurons, self.device)
            
            self.model.reset_state(batch_size)
            all_states, final_state = self.model(x_input, steps=thinking_steps)
            
            output_indices = self.model.output_ids
            
            if full_sequence:
                 return self.model.read_outputs(all_states)
            else:
                 return self.model.read_outputs(final_state)

    def evaluate(self, input_features, target_values, thinking_steps):
        """
        Evaluates the model on a dataset.
        """
        self.model.eval()
        with torch.no_grad():
            preds = self.predict(input_features, thinking_steps)
            target_values = to_tensor(target_values, self.device)
                
            loss = self.loss_fn(preds, target_values)
            return loss.item()

    def prune(self, threshold=0.01):
        """
        Triggers synaptic pruning on the model.
        """
        return SynapticPruner.prune(self.model, threshold)

    def regenerate_synapses(self, threshold=0.01, percentage=None):
        """
        Triggers synaptic regeneration (Darwinian Revive) on weak connections.
        Re-initializes weights below threshold (or bottom percentage) instead of pruning them.
        """
        revived, total = self.model.regenerate_weak_weights(threshold, percentage)
        return revived, total

    def fit(self, input_features, target_values, epochs, batch_size=32, thinking_steps=10, verbose=True, pruning_threshold=0.0):
        """
        Trains the model for a fixed number of epochs.
        """
        input_features = to_tensor(input_features, self.device)
        target_values = to_tensor(target_values, self.device)
            
        history = []
        
        # Prepare Data
        batch_size = min(batch_size, len(input_features))
        dataset_len = len(input_features)
        
        # Simple Batching Loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # Helper for random permutation
            indices = torch.randperm(dataset_len)
            
            for i in range(0, dataset_len, batch_size):
                batch_indices = indices[i:i+batch_size]
                x_batch = input_features[batch_indices]
                y_batch = target_values[batch_indices]
                
                loss = self.train_batch(x_batch, y_batch, thinking_steps)
                epoch_loss += loss
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            history.append(avg_loss)
            
            # --- DARWINIAN PRUNING (If Enabled) ---
            pruning_info = ""
            if pruning_threshold > 0.0:
                 # Prune weak connections
                 pruned, dead, total = self.prune(threshold=pruning_threshold)
                 sparsity = (dead / total) * 100.0
                 pruning_info = f" | Dead: {sparsity:.1f}%"
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}/{epochs}: Loss {avg_loss:.6f}{pruning_info}")
                
        return history

    def expand(self, amount=1, verbose=True):
        """
        Dynamically adds neurons to the model (Neurogenesis).
        Ensures optimizer continuity and memory cleanup.
        """
        from ..utils.neurogenesis import Neurogenesis
        self.optimizer = Neurogenesis.expand(self.model, self.optimizer, amount, verbose)
