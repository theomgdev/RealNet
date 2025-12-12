from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RealNetConfig:
    """Configuration for the RealNet Language Model."""
    vocab_size: int = 50257  # GPT-2/3 standard (TikToken cl100k usually ~100k, but let's default to typical)
    n_embd: int = 768        # Embedding dimension (Input/Output interface width)
    n_neurons: int = 2048    # Size of the internal RealNet matrix (Global Workspace)
    n_layers: int = 1        # Number of stacked RealNet blocks (Recurrent loop usually handles depth, but we allow stacking)
    thinking_steps: int = 5  # Recurrence steps per token (Time-Folding)
    dropout: float = 0.1
    bias: bool = True
    pulse_mode: bool = True
    compile: bool = True     # Enable torch.compile
    
    # Pruning / Sparsity
    pruning_threshold: float = 0.0

@dataclass
class TrainingConfig:
    """Configuration for Training Loop."""
    batch_size: int = 32 # Micro-batch size
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    max_steps: int = 10000
    warmup_steps: int = 100
    context_window: int = 128 # Sequence length (BPTT or just input chunk)
    device: str = 'cuda'
    precision: str = 'mixed' # 'mixed', 'float32'
    compile: bool = True     # Use torch.compile (Triton)
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    out_dir: str = 'out'
