import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
import os

# ------------------------------------------------------------
# NumPy 2.0 backward‑compat: many libs still reference `np.dtypes`.
# Provide an alias so they keep working.
# ------------------------------------------------------------
import numpy as np
if not hasattr(np, "dtypes"):
    np.dtypes = np.dtype

class HybridViTCompressor(nn.Module):
    # ---------- colour‑space utilities ----------
    @staticmethod
    def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of RGB images ∈ [0,1] to YCbCr.
        Args:
            x: (B, 3, H, W) tensor in range [0,1]
        Returns:
            (B, 3, H, W) tensor (Y, Cb, Cr) also in [0,1]
        """
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y  = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr =  0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        return torch.cat([y, cb, cr], dim=1)

    @staticmethod
    def ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of `rgb_to_ycbcr`.
        """
        y, cb, cr = x[:, 0:1], x[:, 1:2] - 0.5, x[:, 2:3] - 0.5
        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb
        return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)

    def __init__(self, img_size: int, patch_size: int, embed_dim: int, num_layers: int = 12):
        super(HybridViTCompressor, self).__init__()
        
        # Initial convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # First convolution
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsampling
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),  # Final feature extraction
            nn.ReLU()
        )
        
        # Vision Transformer for global context modeling
        self.vit = VisionTransformer(
            img_size=img_size // 4,  # Adjusted for downsampling
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=num_layers,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            in_chans=embed_dim           # <-- match feature‑map channels
        )
        
        # Reconstruction layers (decoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # Reconstruct original image
        )
    
    def forward(self, x):
        # Convert to YCbCr so that the network can compress chroma more aggressively
        x_ycbcr = self.rgb_to_ycbcr(x)

        # Extract features using convolutional layers (operating on YCbCr)
        features = self.conv_layers(x_ycbcr)
        
        b, c, h, w = features.shape
        # Pass through Vision Transformer (timm will patchify internally)
        vit_features = self.vit(features)
        
        # Reshape back to image dimensions
        vit_features = vit_features.permute(0, 2, 1).view(b, c, h, w)
        
        # Reconstruct image (still YCbCr), then convert back to RGB
        reconstructed_ycbcr = self.decoder(vit_features)
        reconstructed_rgb   = self.ycbcr_to_rgb(reconstructed_ycbcr)

        return reconstructed_rgb, None  # likelihoods placeholder
import logging
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, Tuple
# import wandb
from config import CompressionConfig
from dataset import get_dataloaders
from tic_vit_encoder import ViTCompressor
from tic_vit_encoder import HybridViTCompressor
from loss import CombinedLoss

def setup_logging(config: CompressionConfig) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(config.log_dir, 'training.log'))
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

def get_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_steps_per_epoch: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create warmup scheduler"""
    from torch.optim.lr_scheduler import LinearLR
    return LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs * total_steps_per_epoch
    )

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    metrics: Dict[str, float],
    config: CompressionConfig
) -> None:
    """Save model checkpoint"""
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
        'config': config
    }
    
    torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')

def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    warmup_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    main_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    writer: SummaryWriter,
    config: CompressionConfig
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    epoch_metrics = {}
    total_steps = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, x in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        current_step = epoch * total_steps + batch_idx
        
        # Forward pass with mixed precision
        with autocast(enabled=config.use_amp):
            x_hat, likelihoods = model(x)
            loss, metrics = criterion(x, x_hat, likelihoods)
        
        # Backward pass with gradient scaling
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        # Optimizer step with gradient scaling
        scaler.step(optimizer)
        scaler.update()
        
        # Update learning rate
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        elif main_scheduler is not None:
            main_scheduler.step()
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, current_step)
        writer.add_scalar('train/grad_norm', grad_norm, current_step)
        
        # Update metrics
        for k, v in metrics.items():
            epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            writer.add_scalar(f'train/{k}', v, current_step)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'psnr': f'{metrics["psnr"]:.2f}',
            'bpp': f'{metrics["bpp"]:.4f}',
            'grad_norm': f'{grad_norm:.2f}',
            'lr': f'{current_lr:.2e}'
        })
        
        # Log to wandb
        # if config.use_wandb:
        #     wandb.log({
        #         'train/loss': loss.item(),
        #         'train/psnr': metrics['psnr'],
        #         'train/bpp': metrics['bpp'],
        #         'train/grad_norm': grad_norm,
        #         'train/learning_rate': current_lr,
        #         'epoch': epoch,
        #         'step': current_step
        #     })
    
    # Average metrics
    for k in epoch_metrics:
        epoch_metrics[k] /= len(train_loader)
    
    return epoch_metrics

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logger: logging.Logger,
    epoch: int,
    writer: SummaryWriter,
    config: CompressionConfig
) -> Dict[str, float]:
    """Validate the model"""
    model.eval()
    val_metrics = {}
    
    for batch_idx, x in enumerate(tqdm(val_loader, desc='Validation')):
        x = x.to(device, non_blocking=True)
        
        with autocast(enabled=config.use_amp):
            x_hat, likelihoods = model(x)
            _, metrics = criterion(x, x_hat, likelihoods)
        
        for k, v in metrics.items():
            val_metrics[k] = val_metrics.get(k, 0) + v
            writer.add_scalar(f'val/{k}', v, epoch * len(val_loader) + batch_idx)
    
    # Average metrics
    for k in val_metrics:
        val_metrics[k] /= len(val_loader)
        writer.add_scalar(f'val/{k}_epoch', val_metrics[k], epoch)
        
        # Log to wandb
        # if config.use_wandb:
        #     wandb.log({
        #         f'val/{k}': val_metrics[k],
        #         'epoch': epoch
        #     })
    
    return val_metrics

def main():
    # Load configuration
    config = CompressionConfig()
    device = torch.device(config.device)
    
    # Initialize wandb
    # if config.use_wandb:
    #     wandb.init(
    #         project=config.project_name,
    #         name=config.experiment_name,
    #         config=vars(config)
    #     )
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting training with configuration:")
    for k, v in vars(config).items():
        logger.info(f"{k}: {v}")
    
    # Create tensorboard writer
    writer = SummaryWriter(config.log_dir)
    
    # Create model
    model = HybridViTCompressor(
        img_size=config.img_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers
    ).to(device)
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(config)
    
    # Create loss function
    criterion = CombinedLoss(
        lambda_bpp=config.lambda_factor,
        w_y=config.w_y,
        w_c=config.w_c
    )
    
    # Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler(enabled=config.use_amp)
    
    # Create schedulers
    total_steps_per_epoch = len(train_loader)
    warmup_scheduler = get_warmup_scheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_steps_per_epoch=total_steps_per_epoch
    )
    
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs * total_steps_per_epoch - config.warmup_epochs * total_steps_per_epoch,
        eta_min=config.learning_rate * 0.01
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.num_epochs):
        # Train
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            warmup_scheduler=warmup_scheduler if epoch < config.warmup_epochs else None,
            main_scheduler=main_scheduler if epoch >= config.warmup_epochs else None,
            device=device,
            epoch=epoch,
            logger=logger,
            writer=writer,
            config=config
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            logger=logger,
            epoch=epoch,
            writer=writer,
            config=config
        )
        
        # Save checkpoint
        if (epoch + 1) % config.save_freq == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=main_scheduler,
                scaler=scaler,
                epoch=epoch,
                metrics=val_metrics,
                config=config
            )
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=main_scheduler,
                scaler=scaler,
                epoch=epoch,
                metrics=val_metrics,
                config=config
            )
            logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        # Log epoch summary
        logger.info(
            f"Epoch {epoch} - "
            f"Train Loss: {train_metrics['total_loss']:.4f}, "
            f"Val Loss: {val_metrics['total_loss']:.4f}, "
            f"Train PSNR: {train_metrics['psnr']:.2f}, "
            f"Val PSNR: {val_metrics['psnr']:.2f}, "
            f"Train BPP: {train_metrics['bpp']:.4f}, "
            f"Val BPP: {val_metrics['bpp']:.4f}"
        )
    
    writer.close()
    # if config.use_wandb:
    #     wandb.finish()
    # logger.info("Training completed!")

if __name__ == '__main__':
    main()