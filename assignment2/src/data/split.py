#split.py
'''
Group-aware, class-stratified train/validation/test split.

All images with the same group_id (near duplicates) stay in one split, so a
model cannot see an almost identical picture in train and in val/test.

For every class (a group belongs to the most common class among its images):
1. shuffle the class's groups with the seed,
2. give groups to test until about test_size of the class's images are there,
3. then to validation until about val_size,
4. the remaining groups go to train.
Group integrity comes first, so a class can overshoot by less than one group.

sklearn's StratifiedGroupKFold was tried first, but with shuffling its folds
were uneven per class (a 100-image class got 8-24 test images instead of 15).
'''

import json
from pathlib import Path

import numpy as np

from src.data.audit import compute_sha256, save_csv

MANIFEST_FIELDS = ["image_id", "path", "class_name", "class_idx", "group_id"]


def create_class_to_idx(class_names):
    return {name: index for index, name in enumerate(sorted(set(class_names)))}


def create_train_val_test_split(labels, groups, val_size, test_size, seed):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    members_by_group = {}
    for index, group in enumerate(groups):
        members_by_group.setdefault(group, []).append(index)

    groups_by_class = {}
    for group, members in members_by_group.items():
        values, counts = np.unique(labels[members], return_counts=True)
        groups_by_class.setdefault(values[np.argmax(counts)], []).append(group)

    train_indices, val_indices, test_indices = [], [], []
    for label in sorted(groups_by_class):
        class_groups = sorted(groups_by_class[label])
        rng.shuffle(class_groups)

        class_size = sum(len(members_by_group[group]) for group in class_groups)
        test_target = round(test_size * class_size)
        val_target = round(val_size * class_size)
        test_count = val_count = 0

        for group in class_groups:
            members = members_by_group[group]
            if test_count < test_target:
                test_indices.extend(members)
                test_count += len(members)
            elif val_count < val_target:
                val_indices.extend(members)
                val_count += len(members)
            else:
                train_indices.extend(members)

    return np.sort(train_indices), np.sort(val_indices), np.sort(test_indices)


def split_records(records, val_size, test_size, seed):
    '''Split the audit records whose decision is "keep"; return three record lists.'''
    kept = [record for record in records if record["decision"] == "keep"]
    labels = [record["class_name"] for record in kept]
    groups = [record["group_id"] for record in kept]

    train_indices, val_indices, test_indices = create_train_val_test_split(
        labels, groups, val_size, test_size, seed,
    )

    return (
        [kept[i] for i in train_indices],
        [kept[i] for i in val_indices],
        [kept[i] for i in test_indices],
    )


def save_manifests(train_records, val_records, test_records, class_to_idx, manifests_dir):
    '''Write train/val/test.csv and class_to_idx.json; return their SHA-256 hashes.'''
    manifests_dir = Path(manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    hashes = {}
    for split_name, records in (
        ("train", train_records), ("val", val_records), ("test", test_records),
    ):
        rows = [{field: record[field] for field in MANIFEST_FIELDS} for record in records]
        path = manifests_dir / f"{split_name}.csv"
        save_csv(rows, path, MANIFEST_FIELDS)
        hashes[split_name] = compute_sha256(path)

    class_path = manifests_dir / "class_to_idx.json"
    class_path.write_text(
        json.dumps(class_to_idx, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    hashes["class_to_idx"] = compute_sha256(class_path)

    return hashes


def count_per_class(records, class_to_idx):
    counts = {name: 0 for name in class_to_idx}
    for record in records:
        counts[record["class_name"]] += 1
    return counts
