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
    train_transform: Optional[transforms.Compose] = None,
    val_transform: Optional[transforms.Compose] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    """
    train_dataset = ImageCompressionDataset(
        config.train_data_dir,
        img_size=config.img_size,
        transform=train_transform,
        is_train=True
    )
    
    val_dataset = ImageCompressionDataset(
        config.val_data_dir,
        img_size=config.img_size,
        transform=val_transform,
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 