#text_encoder.py
"""Encode corrected meme text with an embedding layer and a GRU."""

from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

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