#test_split.py
'''
RUN: python -m pytest tests/test_split.py
Small synthetic records/images only; no dataset download needed.
'''

import numpy as np
import pytest
from PIL import Image

from src.data.audit import create_audit_table, find_duplicate_groups, apply_review_decisions
from src.data.split import create_class_to_idx, create_train_val_test_split, split_records

NUM_CLASSES = 30


def make_labels_and_groups(images_per_class=100, group_size=2):
    labels, groups = [], []
    for class_idx in range(NUM_CLASSES):
        for image_idx in range(images_per_class):
            labels.append(class_idx)
            groups.append(f"c{class_idx}-g{image_idx // group_size}")
    return np.array(labels), np.array(groups)


def split():
    labels, groups = make_labels_and_groups()
    return labels, groups, create_train_val_test_split(labels, groups, 0.15, 0.15, seed=42)


def test_split_has_no_overlapping_images_or_groups():
    labels, groups, (train, val, test) = split()

    assert len(train) + len(val) + len(test) == len(labels)
    assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)

    train_groups, val_groups, test_groups = set(groups[train]), set(groups[val]), set(groups[test])
    assert not train_groups & val_groups
    assert not train_groups & test_groups
    assert not val_groups & test_groups


def test_split_is_about_70_15_15_and_covers_all_classes():
    labels, _, (train, val, test) = split()

    assert abs(len(train) / len(labels) - 0.70) <= 0.02
    assert abs(len(val) / len(labels) - 0.15) <= 0.02
    assert abs(len(test) / len(labels) - 0.15) <= 0.02

    for indices in (train, val, test):
        assert set(labels[indices]) == set(range(NUM_CLASSES))

    # Each class: 15 of its 100 images in val/test, +1 at most because groups have 2 images.
    for indices in (val, test):
        per_class = np.bincount(labels[indices], minlength=NUM_CLASSES)
        assert per_class.min() >= 15 and per_class.max() <= 16


def test_split_is_reproducible_with_same_seed():
    _, _, first = split()
    _, _, second = split()

    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_class_to_idx_is_sorted_and_contiguous():
    class_to_idx = create_class_to_idx(["pho", "banh_mi", "pho", "com_tam"])

    assert class_to_idx == {"banh_mi": 0, "com_tam": 1, "pho": 2}


def save_image(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def test_audit_marks_unreadable_and_duplicates_with_same_label(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 245, size=(64, 64, 3))
    other = rng.integers(0, 245, size=(64, 64, 3))

    save_image(tmp_path / "pho" / "a.png", base)
    save_image(tmp_path / "pho" / "a_copy.png", base)          # exact duplicate
    save_image(tmp_path / "pho" / "a_brighter.png", base + 10)  # near duplicate
    save_image(tmp_path / "bun_bo" / "b.png", other)
    (tmp_path / "bun_bo" / "broken.jpg").write_bytes(b"not an image")

    records = create_audit_table(tmp_path)
    review_rows = find_duplicate_groups(records, tmp_path)
    by_id = {record["image_id"]: record for record in records}

    assert by_id["bun_bo/broken.jpg"]["decision"] == "exclude"
    assert by_id["pho/a_copy.png"]["decision"] == "exclude"
    assert by_id["pho/a.png"]["decision"] == "keep"
    assert by_id["pho/a_brighter.png"]["decision"] == "keep"
    assert by_id["pho/a_brighter.png"]["group_id"] == by_id["pho/a.png"]["group_id"]
    assert by_id["bun_bo/b.png"]["group_id"] != by_id["pho/a.png"]["group_id"]
    assert [row["decision"] for row in review_rows] == ["same_group"]


def test_audit_handles_label_conflicts(tmp_path):
    rng = np.random.default_rng(1)
    exact = rng.integers(0, 245, size=(64, 64, 3))
    near = rng.integers(0, 245, size=(64, 64, 3))

    save_image(tmp_path / "pho" / "exact.png", exact)
    save_image(tmp_path / "bun_bo" / "exact.png", exact)       # same file, other label
    save_image(tmp_path / "pho" / "near.png", near)
    save_image(tmp_path / "bun_bo" / "near.png", near + 10)    # near duplicate, other label

    records = create_audit_table(tmp_path)
    review_rows = find_duplicate_groups(records, tmp_path)
    by_id = {record["image_id"]: record for record in records}

    assert by_id["pho/exact.png"]["decision"] == "exclude"
    assert by_id["bun_bo/exact.png"]["decision"] == "exclude"
    assert [row["decision"] for row in review_rows] == ["needs_review"]

    with pytest.raises(ValueError):
        apply_review_decisions(records, review_rows)

    review_rows[0]["decision"] = "exclude_a" if review_rows[0]["image_a"] == "bun_bo/near.png" else "exclude_b"
    apply_review_decisions(records, review_rows)

    assert by_id["bun_bo/near.png"]["decision"] == "exclude"
    assert by_id["pho/near.png"]["decision"] == "keep"


def test_pairs_reviewed_as_different_are_not_grouped(tmp_path):
    rng = np.random.default_rng(2)
    near = rng.integers(0, 245, size=(64, 64, 3))
    save_image(tmp_path / "pho" / "near.png", near)
    save_image(tmp_path / "bun_bo" / "near.png", near + 10)

    records = create_audit_table(tmp_path)
    review_rows = find_duplicate_groups(records, tmp_path)
    review_rows[0]["decision"] = "different"
    apply_review_decisions(records, review_rows)

    assert records[0]["decision"] == records[1]["decision"] == "keep"
    assert records[0]["group_id"] != records[1]["group_id"]


def test_split_records_uses_only_kept_records():
    labels, groups = make_labels_and_groups()
    records = [
        {"image_id": str(i), "class_name": str(label), "group_id": group, "decision": "keep"}
        for i, (label, group) in enumerate(zip(labels, groups))
    ]
    records[0]["decision"] = "exclude"

    train, val, test = split_records(records, 0.15, 0.15, seed=42)
    ids = [record["image_id"] for record in train + val + test]

    assert "0" not in ids
    assert len(ids) == len(set(ids)) == len(records) - 1
