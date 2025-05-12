import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

class ContextAdaptiveEntropyModel(nn.Module):
    def __init__(
        self,
        channels: int,
        num_entropy_bins: int = 256,
        context_channels: int = 192,
        use_hyperprior: bool = True
    ):
        super().__init__()
        self.channels = channels
        self.num_entropy_bins = num_entropy_bins
        self.context_channels = context_channels
        self.use_hyperprior = use_hyperprior

        # Spatial context model for local entropy estimation
        self.spatial_conv = nn.Conv2d(
            channels, context_channels, kernel_size=5, padding=2, stride=1
        )
        self.spatial_entropy_param = nn.Conv2d(
            context_channels, 2 * num_entropy_bins, kernel_size=1
        )

        if use_hyperprior:
            # Hyperprior model for global context
            self.hyper_encoder = nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels, channels, 5, stride=2, padding=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels, channels, 3, stride=1, padding=1)
            )
            
            self.hyper_decoder = nn.Sequential(
                nn.ConvTranspose2d(channels, channels, 3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(channels, channels, 5, stride=2, padding=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels, 2 * num_entropy_bins, 1)
            )

    def get_entropy_params(
        self, 
        x: torch.Tensor,
        hyper: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get entropy parameters (mean and scale) for entropy coding.
        
        Args:
            x: Input tensor [B, C, H, W]
            hyper: Optional hyperprior tensor [B, C, H, W]
            
        Returns:
            Tuple of (mean, scale) tensors
        """
        # Get spatial context
        spatial = self.spatial_conv(x)
        spatial_params = self.spatial_entropy_param(spatial)
        mean, scale = torch.chunk(spatial_params, 2, dim=1)
        
        if self.use_hyperprior and hyper is not None:
            # Get hyperprior context
            h = self.hyper_encoder(hyper)
            h = self.hyper_decoder(h)
            h_mean, h_scale = torch.chunk(h, 2, dim=1)
            
            # Combine spatial and hyperprior
            mean = mean + h_mean
            scale = F.softplus(scale + h_scale)
            
        return mean, F.softplus(scale)

    def forward(
        self,
        x: torch.Tensor,
        hyper: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to get entropy model parameters.
        
        Args:
            x: Input tensor [B, C, H, W]
            hyper: Optional hyperprior tensor [B, C, H, W]
            
        Returns:
            Dictionary with entropy parameters
        """
        mean, scale = self.get_entropy_params(x, hyper)
        
        # Get entropy model likelihoods
        likelihoods = self.entropy_model(mean, scale)
        
        return {
            'likelihoods': likelihoods,
            'mean': mean,
            'scale': scale
        }

    def entropy_model(
        self,
        mean: torch.Tensor,
        scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Get entropy model likelihoods using Gaussian entropy model.
        
        Args:
            mean: Mean tensor [B, num_entropy_bins, H, W]
            scale: Scale tensor [B, num_entropy_bins, H, W]
            
        Returns:
            Likelihoods tensor
        """
        # Get Gaussian entropy model likelihoods
        x = torch.arange(self.num_entropy_bins, device=mean.device)
        x = x.view(1, -1, 1, 1)
        x = (x - mean) / scale
        
        # Get Gaussian CDF
        cdf = 0.5 * (1 + torch.erf(x / 2**0.5))
        likelihoods = cdf[:, 1:] - cdf[:, :-1]
        
        return likelihoods 