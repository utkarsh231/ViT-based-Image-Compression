import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class RateDistortionLoss(nn.Module):
    def __init__(self, lambda_factor: float = 0.01):
        super().__init__()
        self.lambda_factor = lambda_factor
        
    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        likelihoods: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute rate-distortion loss.
        
        Args:
            x: Original image tensor [B, C, H, W]
            x_hat: Reconstructed image tensor [B, C, H, W]
            likelihoods: Entropy model likelihoods
            
        Returns:
            total_loss: Combined rate-distortion loss
            metrics: Dictionary containing individual loss components
        """
        # Distortion loss (MSE)
        distortion_loss = F.mse_loss(x_hat, x)
        
        # Rate loss (negative log-likelihood)
        rate_loss = -torch.log2(likelihoods).mean()
        
        # Total loss
        total_loss = distortion_loss + self.lambda_factor * rate_loss
        
        # Calculate PSNR
        mse = F.mse_loss(x_hat, x)
        psnr = 10 * torch.log10(1.0 / mse)
        
        # Calculate bits per pixel (bpp)
        bpp = rate_loss / (x.shape[2] * x.shape[3])
        
        metrics = {
            'distortion_loss': distortion_loss.item(),
            'rate_loss': rate_loss.item(),
            'total_loss': total_loss.item(),
            'psnr': psnr.item(),
            'bpp': bpp.item()
        }
        
        return total_loss, metrics

class MS_SSIM_Loss(nn.Module):
    """Multi-Scale Structural Similarity Index Loss"""
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self._create_window(window_size, self.channel)

    def _gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2/float(sigma**2)))
            for x in range(window_size)
        ])
        return gauss/gauss.sum()

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.device != img1.device:
            self.window = self.window.to(img1.device)
            
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

class CombinedLoss(nn.Module):
    """Combines MSE, MS-SSIM, and Rate losses"""
    def __init__(
        self,
        lambda_factor: float = 0.01,
        alpha: float = 0.84,  # Weight for MS-SSIM
        beta: float = 0.16,   # Weight for MSE
    ):
        super().__init__()
        self.lambda_factor = lambda_factor
        self.alpha = alpha
        self.beta = beta
        self.ms_ssim_loss = MS_SSIM_Loss()
        
    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        likelihoods: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Distortion losses
        mse_loss = F.mse_loss(x_hat, x)
        ms_ssim_loss = self.ms_ssim_loss(x_hat, x)
        
        # Combined distortion loss
        distortion_loss = self.alpha * ms_ssim_loss + self.beta * mse_loss
        
        # Rate loss
        rate_loss = -torch.log2(likelihoods).mean()
        
        # Total loss
        total_loss = distortion_loss + self.lambda_factor * rate_loss
        
        # Calculate metrics
        psnr = 10 * torch.log10(1.0 / mse_loss)
        bpp = rate_loss / (x.shape[2] * x.shape[3])
        
        metrics = {
            'mse_loss': mse_loss.item(),
            'ms_ssim_loss': ms_ssim_loss.item(),
            'distortion_loss': distortion_loss.item(),
            'rate_loss': rate_loss.item(),
            'total_loss': total_loss.item(),
            'psnr': psnr.item(),
            'bpp': bpp.item()
        }
        
        return total_loss, metrics 