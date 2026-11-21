#image_encoder.py
"""ResNet-18 image features with optional encoder freezing."""

import torch
from torch import nn
from torchvision.models import resnet18

class ImageEncoder(nn.Module):
    def __init__(
        self,
        weights: str | None = "IMAGENET1K_V1",
        freeze: bool = True,
    ):
        super().__init__()

        self.freeze = freeze
        self.backbone = resnet18(weights=weights)

        #replace ImageNet's classifier with and identity operation
        self.output_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.backbone.requires_grad_(not freeze)

        if freeze:
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)

        #keep frozen BatchNorm statistics fixed during head training
        if self.freeze:
            self.backbone.eval()

        return self

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                return self.backbone(images)

        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)
