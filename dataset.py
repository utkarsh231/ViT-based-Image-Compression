import os
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

class ImageCompressionDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        img_size: int = 256,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.is_train = is_train
        
        # Get all image files
        self.image_files = [
            f for f in self.data_dir.glob("**/*")
            if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}
        ]
        
        if not self.image_files:
            raise RuntimeError(f"No images found in {data_dir}")
            
        # Default transforms
        if transform is None:
            if is_train:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(10),
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image

def get_dataloaders(
    config: 'CompressionConfig',
    pin_memory: bool = True,
    num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders with GPU optimizations."""
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
    ])
    
    train_dataset = ImageCompressionDataset(
        config.train_data_dir,
        img_size=config.img_size,
        transform=transform,
        is_train=True
    )
    val_dataset = ImageCompressionDataset(
        config.val_data_dir,
        img_size=config.img_size,
        transform=transform,
        is_train=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None  # Prefetch 2 batches per worker
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, val_loader 