
import torch
import torch.nn as nn
from .config import RealNetConfig
from realnet import RealNet  # Core library

class RealNetLM(nn.Module):
    def __init__(self, config: RealNetConfig):
        super().__init__()
        self.config = config
        
        # 1. Embedding Layer: Token -> Vector
        self.tokenizer_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        # 2. Input Projection (if n_embd != n_neurons input area)
        # To map embeddings to RealNet neurons, we can either:
        # A) Use first n_embd neurons as input.
        # B) Linearly project n_embd -> n_neurons (or partial).
        # We choose (A) for now, but we need to supply input_ids to RealNet.
        # However, RealNet core uses `input_ids` list. 
        # If n_embd matches n_neurons, we might need no projection.
        # But usually n_embd < n_neurons.
        # Let's use a Linear project to mix inputs into the system if sizes differ widely.
        # For simplicity and "Deep Learning" standard:
        self.input_proj = nn.Linear(config.n_embd, config.n_neurons) if config.n_embd != config.n_neurons else nn.Identity()
        
        # 3. Backbone: RealNet
        # Construct input_ids/output_ids logic? 
        # Actually RealNet core is flexible. If we feed it a vector of size (Batch, N), it works.
        # But RealNet.forward expects `x_input` of size (Batch, N).
        # If we use `input_proj`, we get (Batch, N). Perfect.
        # We just need to ensure RealNet knows which neurons are "input" if we want special initialization?
        # No, RealNet 2.0 treats all neurons as potentially active. 
        # We just need to pass an initialization mask? 
        # Actually, in `realnet/model.py`, `input_ids` is used mainly for `_prepare_input` in Trainer.
        # Here we are bypassing Trainer's input handling and feeding directly.
        # So we can define input_ids as ALL neurons or None.
        
        input_ids = list(range(config.n_neurons)) # Theoretically we can drive all neurons
        output_ids = list(range(config.n_neurons)) # We can read all neurons
        
        self.backbone = RealNet(
            num_neurons=config.n_neurons,
            input_ids=input_ids,
            output_ids=output_ids, # We will read all and project back
            pulse_mode=config.pulse_mode,
            dropout_rate=config.dropout,
            device='cpu' # Will be moved later
        )
        
        # 4. Unembedding Head: Vector -> Token Logits
        # We project the internal state (n_neurons) back to n_embd, then to vocab?
        # Or directly n_neurons -> vocab?
        # Usually: State -> LayerNorm -> Linear(n_embd) -> Linear(vocab)
        self.ln_f = nn.LayerNorm(config.n_neurons)
        
        # Project back to embedding dimension first (optional, but good for weight tying)
        self.output_proj = nn.Linear(config.n_neurons, config.n_embd, bias=False) 
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight Tying (Optional but standard in GPT)
        # self.tokenizer_embedding.weight = self.head.weight 
        
        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids: (Batch, Seq_Len) - Sequence of tokens
            targets: (Batch, Seq_Len) - Next token targets
        Returns:
            logits: (Batch, Seq_Len, Vocab_Size)
            loss: scalar (if targets provided)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. Embed
        # (Batch, Seq, Emb)
        x_emb = self.tokenizer_embedding(input_ids)
        
        # 2. Process Sequence (Time-Folding)
        # RealNet expects (Batch, N). We have a sequence.
        # We need to run RealNet step-by-step for each token?
        # OR is the sequence the "Thinking Steps"?
        # NO. In LLMs, Sequence is T=1, T=2... each step feeds a new token.
        # So we iterate over Seq_Len.
        
        # Initialize State
        self.backbone.reset_state(batch_size=batch_size)
        
        logits_list = []
        
        # Iterate over sequence
        for t in range(seq_len):
            # Input for this timestep
            xt = x_emb[:, t, :] # (Batch, Emb)
            
            # Project to Neurons
            xt_neuron = self.input_proj(xt) # (Batch, N)
            
            # RealNet Step (Micro-Thinking per token)
            # "Thinking Steps" determines how many internal recurrent loops run PER TOKEN.
            # This allows "System 2" thinking between words.
            _, final_state = self.backbone(x_input=xt_neuron, steps=self.config.thinking_steps)
            
            # Readout
            # Normalize
            h_norm = self.ln_f(final_state)
            
            # Project to Logits
            h_emb = self.output_proj(h_norm)
            logit = self.head(h_emb) # (Batch, Vocab)
            
            logits_list.append(logit)
            
        # Stack: (Batch, Seq, Vocab)
        logits = torch.stack(logits_list, dim=1)
        
        loss = None
        if targets is not None:
            # Shift Logic is usually handled by caller (Input[:-1], Target[1:])
            # But standard HF style: Loss is calculated on shifted internally or mismatched?
            # Let's assume input_ids and targets are aligned such that logits[i] predicts targets[i].
            
            # Flatten
            B, T, V = logits.shape
            logits_flat = logits.view(B*T, V)
            targets_flat = targets.view(B*T)
            
            loss = nn.functional.cross_entropy(logits_flat, targets_flat)
            
        return logits, loss

    def inference_step(self, input_ids, state=None):
        """
        Runs a single step for generation (O(1) complexity).
        Args:
            input_ids: (Batch, 1) - Current token
            state: (Batch, N) - Previous state
        Returns:
            logits: (Batch, 1, Vocab)
            new_state: (Batch, N)
        """
        # 1. Embed
        x_emb = self.tokenizer_embedding(input_ids) # (Batch, 1, Emb)
        xt = x_emb[:, 0, :] # (Batch, Emb)
        
        # 2. Project
        xt_neuron = self.input_proj(xt)
        
        # 3. RealNet Step (Preserve State)
        # Note: We pass 'state' as 'current_state' to RealNet
        _, new_state = self.backbone(x_input=xt_neuron, steps=self.config.thinking_steps, current_state=state)
        
        # 4. Readout
        h_norm = self.ln_f(new_state)
        h_emb = self.output_proj(h_norm)
        logit = self.head(h_emb) # (Batch, Vocab)
        
        return logit.unsqueeze(1), new_state

    def compile(self):
        """PyTorch 2.0 Compile"""
        if not self.config.compile:
            print("RealNetLM: Compilation disabled in config. Skipping.")
            return self

        if hasattr(torch, 'compile'):
            try:
                print("RealNetLM: Compiling...")
                # Attempt compilation
                compiled_model = torch.compile(self)
                
                # DRY RUN to force error now if backend fails (e.g. Windows/Triton)
                print("RealNetLM: Performing dry run...")
                # Create dummy input (Batch=1, Seq=1)
                dummy_ids = torch.zeros(1, 1, dtype=torch.long, device='cpu') # We are on CPU during init usually, or check device
                # Model parameters device?
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
