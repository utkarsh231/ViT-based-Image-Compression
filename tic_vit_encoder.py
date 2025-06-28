import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import VisionTransformer
#from compressai.entropy_models import EntropyBottleneck
#from compressai.layers import GDN, IGDN  # Removed unused import
from entropy_model import ContextAdaptiveEntropyModel

class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x

class ViTEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size)**2, embed_dim))
        
        # Add layer norm after patch embedding
        self.pre_norm = nn.LayerNorm(embed_dim)
        
        # Create transformer encoder layers with proper normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,  # Standard ViT feedforward ratio
            dropout=0.1,  # Add dropout for regularization
            activation='gelu',  # Use GELU activation
            batch_first=True,  # Use batch first for better compatibility
            norm_first=True  # Use pre-norm architecture for better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # Initialize patch embedding
        nn.init.trunc_normal_(self.patch_embed.proj.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.proj.bias)
    
    def forward(self, x):
        x = self.patch_embed(x)  # [B, N, D]
        x = self.pre_norm(x)  # Add pre-norm
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        return x  # [B, N, D]

class SimpleViTDecoder(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, img_size=256, out_chans=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.linear = nn.Linear(embed_dim, patch_size * patch_size * out_chans)

    def forward(self, x):
        B, N, D = x.shape
        x = self.linear(x)
        x = x.view(B, self.grid_size, self.grid_size, 3, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, 3, self.img_size, self.img_size)
        return x

class ViTCompressor(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=768, use_hyperprior=True):
        super().__init__()
        self.encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.entropy_model = ContextAdaptiveEntropyModel(
            channels=embed_dim,
            num_entropy_bins=256,
            context_channels=192,
            use_hyperprior=use_hyperprior
        )
        self.decoder = SimpleViTDecoder(embed_dim=embed_dim, patch_size=patch_size, img_size=img_size)

    def forward(self, x):
        y = self.encoder(x)  # [B, N, D]
        B, N, D = y.shape
        grid_size = int(N ** 0.5)
        y_4d = y.transpose(1, 2).contiguous().view(B, D, grid_size, grid_size)  # [B, D, H', W']
        entropy_out = self.entropy_model(y_4d, hyper=None)
        # For now, just use the input as y_hat (no quantization/entropy coding implemented)
        y_hat_4d = y_4d
        y_likelihoods = entropy_out['likelihoods']
        # Reshape back to [B, N, D]
        y_hat = y_hat_4d.view(B, D, -1).transpose(1, 2).contiguous()  # [B, N, D]
        x_hat = self.decoder(y_hat)
        return x_hat, y_likelihoods

class HybridViTCompressor(nn.Module):
    def __init__(self, img_size: int, patch_size: int, embed_dim: int, num_layers: int = 12):
        super(HybridViTCompressor, self).__init__()
        
        # Initial convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Use your own ViTEncoder that supports arbitrary input channels
        self.vit = ViTEncoder(
            img_size=img_size // 4,      # Adjusted for downsampling (stride 2 x 2)
            patch_size=patch_size,
            in_chans=embed_dim,          # Match conv output channels
            embed_dim=embed_dim,
            depth=num_layers,
            num_heads=8
        )
        
        # Reconstruction layers (decoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),        # 32→64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),         # 64→128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),          # 128→256
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)                                       # Final RGB
        )
    
    def rgb_to_ycbcr(self, x):
        matrix = torch.tensor([[0.299, 0.587, 0.114],
                               [-0.168736, -0.331264, 0.5],
                               [0.5, -0.418688, -0.081312]], device=x.device)
        ycbcr = torch.tensordot(x, matrix, dims=([1], [1])).permute(0, 3, 1, 2)
        ycbcr[:, 1:] += 128.0
        return ycbcr

    def ycbcr_to_rgb(self, x):
        matrix = torch.tensor([[1.0, 0.0, 1.402],
                               [1.0, -0.344136, -0.714136],
                               [1.0, 1.772, 0.0]], device=x.device)
        x[:, 1:] -= 128.0
        rgb = torch.tensordot(x, matrix, dims=([1], [1])).permute(0, 3, 1, 2)
        return rgb

    def forward(self, x):
        x_ycbcr = self.rgb_to_ycbcr(x)
        features = self.conv_layers(x_ycbcr)  # [B, embed_dim, H', W']

        # ViTEncoder expects [B, embed_dim, H', W']
        vit_tokens = self.vit.patch_embed(features)  # [B, N, embed_dim]
        vit_tokens = self.vit.pre_norm(vit_tokens)
        vit_tokens = vit_tokens + self.vit.pos_embed
        vit_tokens = self.vit.transformer(vit_tokens)
        vit_tokens = self.vit.norm(vit_tokens)

        # Reshape tokens back to feature map
        B, N, D = vit_tokens.shape
        grid_size = int(N ** 0.5)
        vit_features = vit_tokens.transpose(1, 2).contiguous().view(B, D, grid_size, grid_size)

        reconstructed_ycbcr = self.decoder(vit_features)
        reconstructed_rgb = self.ycbcr_to_rgb(reconstructed_ycbcr)
        return reconstructed_rgb, None  # likelihoods placeholder

if __name__ == '__main__':
    dummy_input = torch.randn(1, 3, 256, 256)
    model = ViTCompressor()
    out, likelihoods = model(dummy_input)
    print(f"Output image shape: {out.shape}")
    print(f"Likelihoods shape: {likelihoods.shape}")
