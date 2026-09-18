"""
Implementing RNN (LSTM/GRU) classifier for Fashion-MNIST.
- "row":   each of the 28 rows is one timestep -> seq_len=28, input_size=28
- "col":   each of the 28 columns is one timestep -> seq_len=28, input_size=28
- "patch": the image is split into non-overlapping patch_size x patch_size
           patches, each patch is one timestep (a small "ViT-style" tokenization)
"""

import torch
import torch.nn as nn


class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        img_size: int = 28,
        num_classes: int = 10,
        cell_type: str = "gru",
        mode: str = "row",
        patch_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        if cell_type not in ("lstm", "gru"):
            raise ValueError(f"Unknown cell_type: {cell_type}")

        if mode not in ("row", "col", "patch"):
            raise ValueError(f"Unknown mode: {mode}")

        self.img_size = img_size
        self.mode = mode
        self.patch_size = patch_size

        if mode in ("row", "col"):
            self.seq_len = img_size
            self.input_size = img_size
        else:
            if img_size % patch_size != 0:
                raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size})")
            numpatches_per_side = img_size // patch_size
            self.seq_len = numpatches_per_side * numpatches_per_side
            self.input_size = patch_size * patch_size

        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU

        self.rnn = rnn_cls(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        rnn_out_dim = hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Linear(rnn_out_dim, num_classes)

    def _image_to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W] (output of the data pipeline: ToTensor + Normalize)
        returns: [B, seq_len, input_size]
        """
        b = x.size(0)
        x = x.squeeze(1)  # [B, H, W]

        if self.mode == "row":
            seq = x  # [B, H, W] -> each row is a timestep

        elif self.mode == "col":
            seq = x.transpose(1, 2)  # [B, W, H] -> each column is a timestep

        else:  # patch
            p = self.patch_size
            n = self.img_size // p

            seq = (x.view(b, n, p, n, p).permute(0, 1, 3, 2, 4).contiguous())
            seq = seq.view(b, n * n, p * p)

        return seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._image_to_sequence(x)
        out, _ = self.rnn(seq)
        last_step = out[:, -1, :]
        logits = self.classifier(last_step)
        return logits

