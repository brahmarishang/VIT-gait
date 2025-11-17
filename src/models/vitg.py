
import math
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=32, in_chans=1, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B,N,D
        return x

class VITG(nn.Module):
    def __init__(self, img_size=128, patch_size=32, in_chans=1,
                 embed_dim=64, depth=16, num_heads=8, mlp_ratio=4.0,
                 num_classes=100, dropout=0.1, learned_pos=True):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.num_patches
        if learned_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_buffer("pos_embed", self._build_sincos_pos_embed(n_patches, embed_dim), persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim*mlp_ratio),
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * n_patches, 2048), nn.GELU(),
            nn.Linear(2048, 1024), nn.GELU(),
            nn.Linear(1024, 512), nn.GELU(),
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, num_classes),
        )

    def _build_sincos_pos_embed(self, n_patches: int, dim: int):
        pe = torch.zeros(1, n_patches, dim)
        position = torch.arange(0, n_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        x = x.flatten(1)
        return self.head(x)
