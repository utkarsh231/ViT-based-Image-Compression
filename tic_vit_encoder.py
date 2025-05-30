import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import VisionTransformer
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN, IGDN

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
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=depth
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)  # [B, N, D]
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
    def __init__(self, img_size=256, patch_size=16, embed_dim=768):
        super().__init__()
        self.encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.entropy_bottleneck = EntropyBottleneck(embed_dim)
        self.decoder = SimpleViTDecoder(embed_dim=embed_dim, patch_size=patch_size, img_size=img_size)

    def forward(self, x):
        y = self.encoder(x)
        y_flat = y.view(-1, y.shape[-1])  # [B*N, C]
        y_hat, y_likelihoods = self.entropy_bottleneck(y_flat)
        y_hat = y_hat.view(x.size(0), -1, y.shape[-1])  # [B, N, C]
        x_hat = self.decoder(y_hat)
        return x_hat, y_likelihoods

if __name__ == '__main__':
    dummy_input = torch.randn(1, 3, 256, 256)
    model = ViTCompressor()
    out, likelihoods = model(dummy_input)
    print(f"Output image shape: {out.shape}")
    print(f"Likelihoods shape: {likelihoods.shape}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   