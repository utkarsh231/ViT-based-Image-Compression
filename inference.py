import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tic_vit_encoder import ViTCompressor
from config import CompressionConfig

def load_image(image_path: str, img_size: int = 256) -> torch.Tensor:
    """Load and preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def save_image(tensor: torch.Tensor, path: str):
    """Convert tensor to image and save."""
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = tensor * std + mean
    
    # Clamp to valid range
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to image
    image = (tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype('uint8')
    Image.fromarray(image).save(path)

def run_inference(
    model: nn.Module,
    image_path: str,
    output_dir: str,
    img_size: int = 256
) -> None:
    """Run model inference on a single image."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process image
    x = load_image(image_path, img_size)
    
    # Run model
    model.eval()
    with torch.no_grad():
        x_hat, likelihoods = model(x)
    
    # Calculate metrics
    mse = torch.mean((x - x_hat) ** 2)
    psnr = 10 * torch.log10(1.0 / mse)
    bpp = -torch.log2(likelihoods).mean() / (img_size * img_size)
    
    # Save original and reconstructed images
    save_image(x, os.path.join(output_dir, 'original.png'))
    save_image(x_hat, os.path.join(output_dir, 'reconstructed.png'))
    
    # Plot and save metrics
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(Image.open(os.path.join(output_dir, 'original.png')))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(Image.open(os.path.join(output_dir, 'reconstructed.png')))
    plt.title(f'Reconstructed\nPSNR: {psnr:.2f} dB, BPP: {bpp:.4f}')
    plt.axis('off')
    
    plt.savefig(os.path.join(output_dir, 'comparison.png'))
    plt.close()

if __name__ == '__main__':
    # Load model
    config = CompressionConfig()
    model = ViTCompressor(
        img_size=config.img_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim
    )
    
    # Run inference
    run_inference(
        model,
        'cat.png',
        'outputs',
        img_size=config.img_size
    )
    print('Inference completed. Check outputs/ for results.') 