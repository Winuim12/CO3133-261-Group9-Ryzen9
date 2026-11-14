"""Pad tokens and return the shared paired-batch contract."""
import torch


def make_collate(pad_id: int = 0):
    def collate(batch):
        lengths = torch.tensor([len(b["text_ids"]) for b in batch], dtype=torch.long)
        ids = torch.full((len(batch), int(lengths.max())), pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            ids[i, : lengths[i]] = torch.tensor(b["text_ids"], dtype=torch.long)
        return {
            "image": torch.stack([b["image"] for b in batch]),   # [B,3,224,224] float32
            "text_ids": ids,                                      # [B,T] long, padded with pad_id
            "text_lengths": lengths,                              # [B] long, true lengths (>=1)
            "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),  # 0..4
            "image_name": [b["image_name"] for b in batch],
        }
    return collate
