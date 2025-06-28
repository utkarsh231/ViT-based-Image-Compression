import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Union

from pytorch_msssim import ms_ssim          # fast CUDA version
# Optional: pip install lpips and uncomment next two lines
# import lpips
# _lpips = lpips.LPIPS(net="vgg").eval()

LOG2 = float(torch.log2(torch.tensor(2.0)))  # convenience

def _bpp_from_likelihoods(
    likelihoods: Union[torch.Tensor, Dict[str, torch.Tensor]],
    num_pixels: int
) -> torch.Tensor:
    """
    Compute bits‑per‑pixel given one tensor or a dict of tensors
    containing PMFs or PMF‑like likelihood values.
    """
    if isinstance(likelihoods, dict):
        bits = sum((-torch.log2(t.clamp_(min=1e-9))).sum() for t in likelihoods.values())
    else:
        bits = (-torch.log2(likelihoods.clamp_(min=1e-9))).sum()
    return bits / num_pixels


class RateDistortionLoss(nn.Module):
    """MSE + λ·BPP  (baseline)"""
    def __init__(self, lambda_bpp: float = 0.1):
        super().__init__()
        self.lambda_bpp = lambda_bpp
        self.mse = nn.MSELoss()

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        likelihoods: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        mse_loss = self.mse(x_hat, x)
        bpp = _bpp_from_likelihoods(likelihoods, x.numel() / x.size(0))
        total = mse_loss + self.lambda_bpp * bpp

        with torch.no_grad():
            psnr = 10 * torch.log10(1.0 / (mse_loss + 1e-9))

        return total, {
            "mse": mse_loss.item(),
            "bpp": bpp.item(),
            "psnr": psnr.item(),
            "total_loss": total.item(),
        }


class CombinedLoss(nn.Module):
    """
    (α · MS‑SSIM  +  β · MSE)  +  λ · BPP  [+ γ · LPIPS]
    Default α+β=1.  Set β=0 for pure MS‑SSIM, etc.
    """
    def __init__(
        self,
        lambda_bpp: float = 0.1,
        w_y: float = 1.0,
        w_c: float = 1.0,
        alpha: float = 0.84,
        beta: float = 0.16,
        use_lpips: bool = False,
        gamma_lpips: float = 0.0,
    ):
        super().__init__()
        self.lambda_bpp = lambda_bpp
        self.alpha = alpha
        self.beta = beta
        self.use_lpips = use_lpips
        self.gamma_lpips = gamma_lpips
        self.mse = nn.MSELoss()
        self.w_y = w_y
        self.w_c = w_c

        if use_lpips:
            import lpips
            self.lpips_fn = lpips.LPIPS(net="vgg").eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        likelihoods: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        # Distortion terms
        ms_ssim_val = ms_ssim(x_hat, x, data_range=1.0)
        ms_ssim_loss = 1.0 - ms_ssim_val

        mse_loss = self.mse(x_hat, x)

        lpips_loss = torch.tensor(0.0, device=x.device)
        if self.use_lpips:
            lpips_loss = self.lpips_fn(x * 2 - 1, x_hat * 2 - 1).mean()

        distortion = (
            self.alpha * ms_ssim_loss +
            self.beta * mse_loss +
            self.gamma_lpips * lpips_loss
        )

        # Rate term
        bpp = _bpp_from_likelihoods(likelihoods, x.numel() / x.size(0)) if likelihoods is not None else torch.tensor(0.0, device=x.device)
        total = distortion + self.lambda_bpp * bpp

        # Metrics (detach to avoid extra graph traversals)
        with torch.no_grad():
            psnr = 10 * torch.log10(1.0 / (mse_loss + 1e-9))

        metrics = {
            "ms_ssim_loss": ms_ssim_loss.item(),
            "mse_loss": mse_loss.item(),
            "lpips_loss": lpips_loss.item() if self.use_lpips else 0.0,
            "distortion": distortion.item(),
            "bpp": bpp.item(),
            "total_loss": total.item(),
            "psnr": psnr.item(),
        }
        return total, metrics