from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class CompressionConfig:
    # Model Architecture
    img_size: int = 256
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    
    # Training
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    
    # Compression
    lambda_factor: float = 0.01  # Rate-distortion trade-off
    num_entropy_bins: int = 256  # For entropy modeling
    
    # Dataset
    train_data_dir: str = "data/train"
    val_data_dir: str = "data/val"
    num_workers: int = 4
    
    # Logging and Checkpoints
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_freq: int = 5  # Save checkpoint every N epochs
    
    # Device
    device: str = "cuda"  # or "cpu"
    
    # Entropy Model
    entropy_model_type: str = "gaussian"  # Options: "gaussian", "laplace", "hyperprior"
    hyperprior_channels: int = 192  # For hyperprior entropy model
    
    def __post_init__(self):
        assert self.img_size % self.patch_size == 0, "Image size must be divisible by patch size"
        assert self.embed_dim % self.num_heads == 0, "Embedding dimension must be divisible by number of heads" 