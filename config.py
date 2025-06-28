from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import os

@dataclass
class CompressionConfig:
    # Model Architecture
    img_size: int = 256  # Reduced from default to save memory
    patch_size: int = 16
    embed_dim: int = 384  # Reduced from 768 to save memory
    depth: int = 6  # Reduced from 12 to save memory
    num_heads: int = 6  # Reduced from 12 to match embed_dim
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.1  # Added dropout for regularization
    attn_drop_rate: float = 0.1
    num_layers: int =12
    
    # Training
    batch_size: int = 8  # Reduced for laptop GPU
    num_epochs: int = 50  # Reduced for faster training
    learning_rate: float = 2e-4  # Slightly increased for faster convergence
    weight_decay: float = 1e-4
    warmup_epochs: int = 2  # Reduced warmup
    lambda_factor: float = 0.1  # Rate-distortion trade-off
    max_grad_norm: float = 1.0
    
    # Compression
    num_entropy_bins: int = 256
    
    # Dataset
    train_data_dir: str = "data/train"
    val_data_dir: str = "data/val"
    num_workers: int = 2  # Reduced for laptop
    
    # Logging and Checkpoints
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_freq: int = 5
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Entropy Model
    entropy_model_type: str = "gaussian"
    hyperprior_channels: int = 128  # Reduced from 192
    
    # Mixed Precision Training
    use_amp: bool = True  # Enable automatic mixed precision
    
    # Experiment Tracking
    use_wandb: bool = True
    project_name: str = "vit-compression"
    experiment_name: str = "laptop-training-2"
    
    w_y: float = 1.0  # Weight for Y channel in loss
    w_c: float = 1.0  # Weight for chroma channels in loss

    def __post_init__(self):
        assert self.img_size % self.patch_size == 0, "Image size must be divisible by patch size"
        assert self.embed_dim % self.num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Set environment variables for better performance
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
