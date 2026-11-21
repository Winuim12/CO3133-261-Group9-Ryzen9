"""Concatenation, learned-gate, and fixed-average fusion over shared encoders.

All three designs share the same encoders and the same per-modality projection
(Linear -> ReLU -> Dropout into `projection_dim`). They differ only in how the
two projected vectors are combined, so the comparison is controlled:

    concat:        h = [p_img ; p_txt]                          head: 2d -> 5
    gated:         g = sigmoid(W [p_img ; p_txt] + b)            (per dimension)
                   h = g * p_img + (1 - g) * p_txt              head:  d -> 5
    fixed_average: h = a * p_img + (1 - a) * p_txt, a constant  head:  d -> 5

`fixed_average` is the ablation of `gated`: identical architecture with the
learned gate replaced by the constant `gate_value` (0.5 by default).
The remaining difference to `concat` is the head size; parameter counts are
recorded for every run.

g close to 1 means the image dominates that dimension; close to 0 means text.
The last gate is kept in `last_gate` for modality-interaction analysis.
"""

import torch
from torch import Tensor, nn

from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder

FUSION_MODES = ("concat", "gated", "fixed_average")


def _projection(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))


class FusionClassifier(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoder,
        text_encoder: TextEncoder,
        mode: str,
        projection_dim: int = 256,
        num_classes: int = 5,
        dropout: float = 0.3,
        gate_value: float = 0.5,
    ):
        super().__init__()
        if mode not in FUSION_MODES:
            raise ValueError(f"Unknown fusion mode {mode!r}; expected one of {FUSION_MODES}.")
        if not 0.0 <= gate_value <= 1.0:
            raise ValueError("gate_value must be in [0, 1].")

        self.mode = mode
        self.projection_dim = projection_dim
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_projection = _projection(image_encoder.output_dim, projection_dim, dropout)
        self.text_projection = _projection(text_encoder.output_dim, projection_dim, dropout)

        if mode == "gated":
            self.gate = nn.Linear(2 * projection_dim, projection_dim)
        else:
            self.gate = None
        # A buffer, not a parameter: the ablation gate is never trained.
        self.register_buffer("gate_value", torch.tensor(float(gate_value)), persistent=mode == "fixed_average")

        fused_dim = 2 * projection_dim if mode == "concat" else projection_dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fused_dim, num_classes))
        self.last_gate: Tensor | None = None

    def project(self, batch) -> tuple[Tensor, Tensor]:
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(batch["text_ids"], batch["text_lengths"])
        return self.image_projection(image_features), self.text_projection(text_features)

    def fuse(self, image: Tensor, text: Tensor) -> tuple[Tensor, Tensor | None]:
        if self.mode == "concat":
            return torch.cat([image, text], dim=1), None
        if self.mode == "gated":
            gate = torch.sigmoid(self.gate(torch.cat([image, text], dim=1)))
        else:
            gate = self.gate_value.to(image.dtype).expand_as(image)
        return gate * image + (1.0 - gate) * text, gate

    def forward(self, batch) -> Tensor:
        image, text = self.project(batch)
        fused, gate = self.fuse(image, text)
        self.last_gate = None if gate is None else gate.detach()
        return self.head(fused)
