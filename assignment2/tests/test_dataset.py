#test_dataset.py
'''
RUN: python -m pytest tests/test_dataset.py
Checks the data contract agreed with the model/trainer owners:
images [B, 3, 224, 224], integer labels [B] in 0..29, .classes and .records.
'''

import numpy as np
import torch
from PIL import Image

from src.data.audit import create_audit_table, find_duplicate_groups
from src.data.dataloader import create_dataloaders
from src.data.dataset import get_food_datasets
from src.data.split import create_class_to_idx, save_manifests
from src.data.transforms import get_train_transforms, get_val_transforms, get_test_transforms


def make_manifests(tmp_path):
    raw_dir = tmp_path / "raw"
    rng = np.random.default_rng(0)
    for class_name in ("pho", "bun_bo", "com_tam"):
        for i in range(4):
            path = raw_dir / class_name / f"{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "L" if i == 0 else "RGB"          # one grayscale image must become RGB
            size = (60 + 10 * i, 50) if mode == "RGB" else (80, 80)
            array = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
            Image.fromarray(array).convert(mode).save(path)

    records = create_audit_table(raw_dir)
    find_duplicate_groups(records, raw_dir)
    class_to_idx = create_class_to_idx(record["class_name"] for record in records)
    manifests_dir = tmp_path / "manifests"
    hashes = save_manifests(records, records[:3], records[3:6], class_to_idx, manifests_dir)
    return raw_dir, manifests_dir, hashes


def test_dataset_returns_rgb_tensor_and_int_label(tmp_path):
    raw_dir, manifests_dir, _ = make_manifests(tmp_path)

    train, val, test = get_food_datasets(
        manifests_dir, raw_dir,
        get_train_transforms(), get_val_transforms(), get_test_transforms(),
    )

    for dataset in (train, val, test):
        image, label = dataset[0]
        assert image.shape == (3, 224, 224)
        assert image.dtype == torch.float32
        assert isinstance(label, int)

    assert train.classes == ["bun_bo", "com_tam", "pho"]
    assert train.records[0]["image_id"] == "bun_bo/0.png"
    assert len(train) == 12


def test_dataloader_batch_shape(tmp_path):
    raw_dir, manifests_dir, _ = make_manifests(tmp_path)
    train, val, test = get_food_datasets(
        manifests_dir, raw_dir,
        get_train_transforms(), get_val_transforms(), get_test_transforms(),
    )

    train_loader, val_loader, test_loader = create_dataloaders(train, val, test, batch_size=3, num_workers=0)

    for loader in (train_loader, val_loader, test_loader):
        images, labels = next(iter(loader))
        assert images.shape == (3, 3, 224, 224)
        assert labels.shape == (3,)
        assert labels.dtype == torch.int64


def test_manifest_hashes_are_reproducible(tmp_path):
    _, _, first = make_manifests(tmp_path / "a")
    _, _, second = make_manifests(tmp_path / "b")

    assert first == second
