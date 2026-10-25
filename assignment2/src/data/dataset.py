#dataset.py
'''
Dataset that reads image paths and labels from a manifest CSV.

Each item is (image, label) like Assignment 1:
    image: tensor [3, 224, 224] after the transform
    label: int in 0..29

Extra attributes:
    classes: the 30 class names ordered by class index
    records: manifest rows (image_id, path, class_name, class_idx, group_id),
             so error analysis can link dataset[i] to records[i]["image_id"]
'''

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.data.audit import load_csv


class ManifestDataset(Dataset):
    def __init__(self, manifest_path, raw_dir, class_to_idx_path=None, transform=None):
        manifest_path = Path(manifest_path)
        if class_to_idx_path is None:
            class_to_idx_path = manifest_path.parent / "class_to_idx.json"

        class_to_idx = json.loads(Path(class_to_idx_path).read_text(encoding="utf-8"))

        self.raw_dir = Path(raw_dir)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.classes = sorted(class_to_idx, key=class_to_idx.get)
        self.records = load_csv(manifest_path)

        for record in self.records:
            record["class_idx"] = int(record["class_idx"])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.raw_dir / record["path"]) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, record["class_idx"]


def get_food_datasets(
    manifests_dir,
    raw_dir,
    train_transforms=None,
    val_transforms=None,
    test_transforms=None,
):
    manifests_dir = Path(manifests_dir)

    train_dataset = ManifestDataset(manifests_dir / "train.csv", raw_dir, transform=train_transforms)
    val_dataset = ManifestDataset(manifests_dir / "val.csv", raw_dir, transform=val_transforms)
    test_dataset = ManifestDataset(manifests_dir / "test.csv", raw_dir, transform=test_transforms)

    return train_dataset, val_dataset, test_dataset
