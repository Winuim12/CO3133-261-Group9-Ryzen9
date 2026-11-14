"""Load paired image-text records from the frozen manifests."""
import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .tokenizer import Vocabulary
from .transforms import ImageTransform

ROOT = Path(__file__).resolve().parents[2]


class MemeDataset(Dataset):
    def __init__(self, manifest, vocab: Vocabulary, train=False, image_size=224,
                 max_text_length=64, root=ROOT):
        self.root, self.vocab, self.max_len = Path(root), vocab, max_text_length
        self.transform = ImageTransform(image_size, train=train)
        with open(self.root / manifest, newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(self.root / r["image_path"]) as im:
            image = self.transform(im)
        return {
            "image": image,
            "text_ids": self.vocab.encode(r["text_corrected"], self.max_len),
            "label": int(r["label_id"]),
            "image_name": r["image_name"],
        }
