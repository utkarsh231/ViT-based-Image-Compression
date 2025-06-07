import os
import requests
import zipfile
from pathlib import Path
import shutil
from tqdm import tqdm
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def download_file(url: str, destination: Path, chunk_size: int = 8192):
    """Download a file with progress bar."""
    logger = logging.getLogger(__name__)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            pbar.update(size)

def prepare_kodak_dataset():
    """Download and prepare the Kodak dataset."""
    logger = setup_logging()
    
    # Create directories
    data_dir = Path('data')
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    
    # Create directories if they don't exist
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Download Kodak dataset
    kodak_url = "http://r0k.us/graphics/kodak/kodak.zip"
    zip_path = data_dir / "kodak.zip"
    
    if not zip_path.exists():
        logger.info("Downloading Kodak dataset...")
        download_file(kodak_url, zip_path)
    else:
        logger.info("Kodak dataset already downloaded.")
    
    # Extract the dataset
    if not (data_dir / "kodak").exists():
        logger.info("Extracting Kodak dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    
    # Move images to train/val directories (80/20 split)
    kodak_dir = data_dir / "kodak"
    images = sorted(list(kodak_dir.glob("*.png")))
    num_val = len(images) // 5  # 20% for validation
    
    logger.info("Splitting dataset into train/val...")
    for i, img_path in enumerate(images):
        if i < num_val:
            shutil.copy2(img_path, val_dir / img_path.name)
        else:
            shutil.copy2(img_path, train_dir / img_path.name)
    
    # Clean up
    logger.info("Cleaning up...")
    if zip_path.exists():
        zip_path.unlink()
    if kodak_dir.exists():
        shutil.rmtree(kodak_dir)
    
    logger.info(f"Dataset preparation complete!")
    logger.info(f"Training images: {len(list(train_dir.glob('*.png')))}")
    logger.info(f"Validation images: {len(list(val_dir.glob('*.png')))}")

if __name__ == "__main__":
    prepare_kodak_dataset() 