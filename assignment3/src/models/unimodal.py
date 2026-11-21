#unimodal.py
"""Classifiers that use one input modality."""

from torch import Tensor, nn
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder

class ImageOnlyClassifier(nn.Module):
    def __init__(
        self,
        encoder: ImageEncoder,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.output_dim, num_classes),
        )

    def forward(self, batch) -> Tensor:
        features = self.encoder(batch["image"])
        return self.head(features)

class TextOnlyClassifier(nn.Module):
    def __init__(
        self,
        encoder: TextEncoder,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.output_dim, num_classes),
        )

    def forward(self, batch) -> Tensor:
        features = self.encoder(batch["text_ids"], batch["text_lengths"])
        return self.head(features)