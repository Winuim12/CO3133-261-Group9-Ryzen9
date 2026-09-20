"""
Small Vision Transformer (ViT) classifier for Fashion-MNIST.
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Splits the image into non-overlapping patches and linearly projects each to embed_dim."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()

        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size})")

        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.proj(x)       # [B, embed_dim, H/p, W/p] dùng conv để chia ảnh thành các patch ko trùng lắp + nối thành embed vector
        x = x.flatten(2)       # [B, embed_dim, num_patches] gộp 2 chiều không gian (H/p, W/p) thành 1 chiều num_patches
        x = x.transpose(1, 2)     # [B, num_patches, embed_dim] đổi thứ tự để khớp format chuẩn (T, feature_dim) mà Transformer cần
        return x


class VisionTransformerClassifier(nn.Module):
    def __init__(
        self,
        img_size: int = 28,
        in_channels: int = 1,
        num_classes: int = 10,
        patch_size: int = 4,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

#thêm token đặc biệt [CLS] để dùng cho classification, sẽ được thêm vào đầu chuỗi patch tokens representation của toàn bộ ảnh.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  

        #embedding cho vị trí của các patch tokens + token [CLS]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  

        #chuẩn hóa giảm overfitting
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN: more stable to train from scratch
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()


    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)

        tokens = self.patch_embed(x)                          # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(b, -1, -1)          # [B, 1, embed_dim]
        tokens = torch.cat((cls_tokens, tokens), dim=1)        # [B, num_patches+1, embed_dim]

        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        encoded = self.encoder(tokens)                         # [B, num_patches+1, embed_dim]
        cls_out = encoded[:, 0]                                # [B, embed_dim]
        cls_out = self.norm(cls_out)
        logits = self.head(cls_out)                            # [B, num_classes]
        return logits
