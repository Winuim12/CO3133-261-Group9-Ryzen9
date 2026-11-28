#image_encoder.py
"""ResNet image features with optional encoder freezing."""

import torch
from torch import nn
from torchvision.models import resnet18, resnet50

class ImageEncoder(nn.Module):
    def __init__(
        self,
        weights: str | None = "IMAGENET1K_V1",
        freeze: bool = True,
        *,
        architecture: str = "resnet18",
    ):
        super().__init__()

        self.freeze = freeze
        builders = {"resnet18": resnet18, "resnet50": resnet50}
        if architecture not in builders:
            raise ValueError(f"Unknown image encoder {architecture!r}.")
        self.backbone = builders[architecture](weights=weights)

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
