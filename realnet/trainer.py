import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RealNetTrainer:
    def __init__(self, model, optimizer=None, loss_fn=None, device='cpu', gradient_decay=0.0):
        """
        Trainer for RealNet models.
        
        Args:
            model (RealNet): The RealNet model instance.
            optimizer (torch.optim.Optimizer): Optimizer instance. If None, defaults to AdamW.
            loss_fn (callable): Loss function. If None, defaults to MSELoss.
            loss_fn (callable): Loss function. If None, defaults to MSELoss.
            device (str): Device to run on ('cpu' or 'cuda').
            gradient_decay (float): Fraction of gradients to keep from previous step (0.0 to 1.0). Default 0.0 (Standard zero_grad).
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.gradient_decay = gradient_decay
        
        self.optimizer = optimizer if optimizer else optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.loss_fn = loss_fn if loss_fn else nn.MSELoss()

    def _prepare_input(self, input_features, batch_size):
        """
        Maps input features (Batch, Input_Size) to the full neuron tensor (Batch, N).
        """
        x_input = torch.zeros(batch_size, self.model.num_neurons, device=self.device)
        
        # Ensure input_features is on correct device
        if not isinstance(input_features, torch.Tensor):
            input_features = torch.tensor(input_features, dtype=torch.float32, device=self.device)
        else:
            input_features = input_features.to(self.device)
            
        # Map features to input_ids
        # Assuming input_features has columns corresponding to input_ids order
        # If input_features is (Batch, 1) and len(input_ids) == 1, it works.
        if len(self.model.input_ids) > 0:
            # Handle case where input_features might be 1D (Batch,) -> (Batch, 1)
            if input_features.dim() == 1:
                input_features = input_features.unsqueeze(1)
                
            num_features = input_features.shape[1]
            num_assigned = min(num_features, len(self.model.input_ids))
            
            # Efficient indexing
            # We assign input_features[:, k] to neuron input_ids[k]
            for k in range(num_assigned):
                x_input[:, self.model.input_ids[k]] = input_features[:, k]
                
        return x_input

    def train_batch(self, input_features, target_values, thinking_steps, gradient_accumulation_steps=1):
        """
        Runs a single training step on a batch.
        
        Args:
            input_features (Tensor or np.array): Shape (Batch, Len(Input_IDs))
            target_values (Tensor or np.array): Shape (Batch, Len(Output_IDs))
            thinking_steps (int): Number of steps to run the network.
            gradient_accumulation_steps (int): Number of steps to accumulate gradients before stepping optimizer.
            
        Returns:
            float: Loss value
        """
        self.model.train()
        
        # Initialize Scaler if not exists (for AMP)
        if not hasattr(self, 'scaler'):
            # Use new torch.amp API if available, else fallback
            if hasattr(torch.amp, 'GradScaler'):
                self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device == 'cuda'))
            else:
                self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))

        batch_size = len(input_features)
        
        # 1. Prepare Data
        x_input = self._prepare_input(input_features, batch_size)
        
        if not isinstance(target_values, torch.Tensor):
            target_values = torch.tensor(target_values, dtype=torch.float32, device=self.device)
        else:
            target_values = target_values.to(self.device)

        # 2. Reset State & Run (with AMP)
        # Use autocast for mixed precision
        device_type = 'cuda' if self.device == 'cuda' else 'cpu'
        if hasattr(torch.amp, 'autocast'):
             autocast_ctx = torch.amp.autocast(device_type=device_type, enabled=(self.device == 'cuda'))
        else:
             autocast_ctx = torch.cuda.amp.autocast(enabled=(self.device == 'cuda'))
             
        with autocast_ctx:
            self.model.reset_state(batch_size)
            _, final_state = self.model(x_input, steps=thinking_steps)
            
            # 3. Extract Outputs & Calculate Loss
            output_indices = self.model.output_ids
            predicted_outputs = final_state[:, output_indices]
            
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

            if self.gradient_decay > 0.0:
                 # Gradient Persistence (Ghost Gradients)
                 with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            # CRITICAL: If NaN/Inf, reset completely
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                param.grad.zero_()
                            else:
                                # Keep a fraction of the gradient
                                param.grad.mul_(self.gradient_decay)
            else:
                 self.optimizer.zero_grad()
            if hasattr(self, '_acc_counter'):
                self._acc_counter = 0

        # Return the SCALED loss for logging (un-normalize if accumulated)
        return loss.item() * gradient_accumulation_steps

    def predict(self, input_features, thinking_steps):
        """
        Runs inference.
        
        Returns:
            Tensor: Predicted values for output neurons.
        """
        self.model.eval()
        with torch.no_grad():
            batch_size = len(input_features)
            x_input = self._prepare_input(input_features, batch_size)
            
            self.model.reset_state(batch_size)
            _, final_state = self.model(x_input, steps=thinking_steps)
            
            output_indices = self.model.output_ids
            return final_state[:, output_indices]

    def evaluate(self, input_features, target_values, thinking_steps):
        """
        Evaluates the model on a dataset.
        
        Returns:
            avg_loss (float)
        """
        self.model.eval()
        with torch.no_grad():
            preds = self.predict(input_features, thinking_steps)
            
            if not isinstance(target_values, torch.Tensor):
                target_values = torch.tensor(target_values, dtype=torch.float32, device=self.device)
            else:
                target_values = target_values.to(self.device)
                
            loss = self.loss_fn(preds, target_values)
            return loss.item()

    def prune(self, threshold=0.01):
        """
        Triggers synaptic pruning on the model.
        """
        return self.model.prune_synapses(threshold)

    def fit(self, input_features, target_values, epochs, batch_size=32, thinking_steps=10, verbose=True, pruning_threshold=0.0):
        """
        Trains the model for a fixed number of epochs.
        
        Args:
            input_features (Tensor or np.array): Training features.
            target_values (Tensor or np.array): Training targets.
            epochs (int): Number of epochs.
            batch_size (int): Size of batches.
            thinking_steps (int): Timesteps for RealNet.
            verbose (bool): Print progress.
            pruning_threshold (float): If > 0, performs Darwinian pruning after each epoch.
                                       Recommended: 0.03 for aggressive optimization.
            
        Returns:
            history (list): List of average loss per epoch.
        """
        if not isinstance(input_features, torch.Tensor):
            input_features = torch.tensor(input_features, dtype=torch.float32, device=self.device)
        else:
            input_features = input_features.to(self.device)
            
        if not isinstance(target_values, torch.Tensor):
            target_values = torch.tensor(target_values, dtype=torch.float32, device=self.device)
        else:
            target_values = target_values.to(self.device)
            
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
