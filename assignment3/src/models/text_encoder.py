#text_encoder.py
"""Encode corrected meme text with an embedding layer and a GRU."""

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.config import DISTILBERT_CACHE, DISTILBERT_ID, DISTILBERT_REVISION

class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        feature_dim: int = 256,
        pad_id: int = 0,
    ):
        super().__init__()
        self.output_dim = feature_dim
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_id
        )
        self.gru = nn.GRU(embedding_dim, feature_dim, batch_first=True)

    def encode(self, text_ids: Tensor, text_lengths: Tensor) -> Tensor:
        embeddings = self.embedding(text_ids)
        packed = pack_padded_sequence(
            embeddings,
            text_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.gru(packed)
        return hidden[-1]

    def forward(self, text_ids: Tensor, text_lengths: Tensor) -> Tensor:
        return self.encode(text_ids, text_lengths)


class DistilBertEncoder(nn.Module):
    """Frozen DistilBERT-base features pooled over unpadded tokens."""

    def __init__(self, load_pretrained: bool = True):
        super().__init__()
        from transformers import DistilBertConfig, DistilBertModel

        if load_pretrained:
            self.backbone = DistilBertModel.from_pretrained(
                DISTILBERT_ID, revision=DISTILBERT_REVISION,
                cache_dir=DISTILBERT_CACHE, use_safetensors=True,
            )
        else:
            # The pinned distilbert-base-uncased uses the base architecture.
            self.backbone = DistilBertModel(DistilBertConfig())
        self.output_dim = self.backbone.config.dim
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def encode(self, text_ids: Tensor, text_lengths: Tensor) -> Tensor:
        # DistilBERT uses token 0 for padding. Derive the mask on the model device.
        mask = text_ids.ne(self.backbone.config.pad_token_id)
        hidden = self.backbone(input_ids=text_ids, attention_mask=mask).last_hidden_state
        mask = mask.unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def forward(self, text_ids: Tensor, text_lengths: Tensor) -> Tensor:
        return self.encode(text_ids, text_lengths)
