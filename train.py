import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import CompressionConfig
from dataset import get_dataloaders
from loss import CombinedLoss
from tic_vit_encoder import ViTCompressor

def setup_logging(config: CompressionConfig) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
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
        'metrics': metrics,
        'config': config
    }
    
    torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')

def get_warmup_scheduler(optimizer, warmup_epochs, total_steps_per_epoch):
    """Create a learning rate scheduler with warmup"""
    warmup_steps = warmup_epochs * total_steps_per_epoch
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    return warmup_scheduler

def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    warmup_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    main_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    writer: SummaryWriter,
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    epoch_metrics = {}
    total_steps = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, x in enumerate(pbar):
        x = x.to(device)
        current_step = epoch * total_steps + batch_idx
        
        # Forward pass
        x_hat, likelihoods = model(x)
        loss, metrics = criterion(x, x_hat, likelihoods)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Log gradient norm
        writer.add_scalar('train/grad_norm', grad_norm, current_step)
        
        optimizer.step()
        
        # Update learning rate
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        elif main_scheduler is not None:
            main_scheduler.step()
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, current_step)
        
        # Update metrics
        for k, v in metrics.items():
            epoch_metrics[k] = epoch_metrics.get(k, 0) + v
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'psnr': f'{metrics["psnr"]:.2f}',
            'bpp': f'{metrics["bpp"]:.4f}',
            'grad_norm': f'{grad_norm:.2f}',
            'lr': f'{current_lr:.2e}'
        })
    
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
    logger: logging.Logger
) -> Dict[str, float]:
    """Validate the model"""
    model.eval()
    val_metrics = {}
    
    for x in tqdm(val_loader, desc='Validation'):
        x = x.to(device)
        x_hat, likelihoods = model(x)
        _, metrics = criterion(x, x_hat, likelihoods)
        
        for k, v in metrics.items():
            val_metrics[k] = val_metrics.get(k, 0) + v
    
    # Average metrics
    for k in val_metrics:
        val_metrics[k] /= len(val_loader)
    
    return val_metrics

def main():
    # Load configuration
    config = CompressionConfig()
    device = torch.device(config.device)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting training with configuration:")
    for k, v in vars(config).items():
        logger.info(f"{k}: {v}")
    
    # Create tensorboard writer
    writer = SummaryWriter(config.log_dir)
    
    # Create model
    model = ViTCompressor(
        img_size=config.img_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim
    ).to(device)
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(config)
    
    # Create loss function
    criterion = CombinedLoss(lambda_factor=config.lambda_factor)
    
    # Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)  # Standard Adam betas
    )
    
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
            warmup_scheduler=warmup_scheduler if epoch < config.warmup_epochs else None,
            main_scheduler=main_scheduler if epoch >= config.warmup_epochs else None,
            device=device,
            epoch=epoch,
            logger=logger,
            writer=writer,
            max_grad_norm=1.0
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, logger)
        
        # Log metrics
        for k, v in train_metrics.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'val/{k}', v, epoch)
        
        # Save checkpoint
        if (epoch + 1) % config.save_freq == 0:
            save_checkpoint(model, optimizer, main_scheduler, epoch, val_metrics, config)
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            save_checkpoint(model, optimizer, main_scheduler, epoch, val_metrics, config)
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
    logger.info("Training completed!")

if __name__ == '__main__':
    main() 