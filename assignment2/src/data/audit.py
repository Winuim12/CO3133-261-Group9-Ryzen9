#audit.py
'''
Audit the raw 30VNFoods images before splitting.

1. create_audit_table: one record per image file (readable?, size, format,
   SHA-256 and a 64-bit difference hash "dHash").
2. find_duplicate_groups: exact duplicates (same SHA-256) and near duplicates.
   Near duplicates are found in two steps:
   - candidates: dHash Hamming distance <= max_distance (8 of 64 bits),
   - confirmation: mean absolute difference of 32x32 grayscale thumbnails <= max_mae.
   dHash alone joined different dishes with a similar layout (round plate in
   the centre) into chains of 250+ images; the thumbnail check removes most of
   those false matches. Images in one duplicate group get the same group_id so
   the split keeps them together.
3. apply_review_decisions: apply the manual decisions written in
   duplicate_review.csv for near-duplicate pairs with different labels.

The class label of an image is the name of its parent folder, so the code works
whether Kaggle's package has Train/Validate/Test folders or not.
'''

import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

AUDIT_FIELDS = [
    "image_id", "path", "class_name", "class_idx", "width", "height",
    "format", "mode", "readable", "sha256", "dhash", "group_id",
    "decision", "reason",
]

REVIEW_FIELDS = [
    "image_a", "image_b", "class_a", "class_b", "distance", "mae", "decision",
]


def compute_sha256(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_dhash(image, hash_size=8):
    '''
    Difference hash: shrink to (hash_size + 1) x hash_size grayscale pixels and
    compare each pixel with its right neighbour -> hash_size * hash_size bits.
    Resizing or re-compressing an image barely changes these bits.
    '''
    small = image.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    pixels = np.asarray(small, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def list_image_paths(raw_dir):
    raw_dir = Path(raw_dir)
    return sorted(
        path for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def audit_image(path, raw_dir):
    '''Audit one image file; class_idx is filled in later by create_audit_table.'''
    path = Path(path)
    relative_path = path.relative_to(raw_dir).as_posix()
    record = {
        "image_id": relative_path,
        "path": relative_path,
        "class_name": path.parent.name,
        "class_idx": "",
        "width": "", "height": "", "format": "", "mode": "",
        "readable": False,
        "sha256": compute_sha256(path),
        "dhash": "",
        "group_id": "",
        "decision": "keep",
        "reason": "",
    }

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            record["width"], record["height"] = image.size
            record["format"] = image.format
            record["mode"] = image.mode
            record["dhash"] = compute_dhash(image)
        record["readable"] = True
    except Exception as error:
        record["decision"] = "exclude"
        record["reason"] = f"unreadable: {type(error).__name__}"

    return record


def add_class_idx(records):
    class_names = sorted({record["class_name"] for record in records})
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    for record in records:
        record["class_idx"] = class_to_idx[record["class_name"]]
    return records


def create_audit_table(raw_dir):
    raw_dir = Path(raw_dir)
    records = [audit_image(path, raw_dir) for path in list_image_paths(raw_dir)]
    return add_class_idx(records)


def _hamming_distances(hashes, start, stop):
    '''Distances between hashes[start:stop] and all hashes, shape [stop-start, N].'''
    block = hashes[start:stop, None] ^ hashes[None, :]
    return np.bitwise_count(block)


def thumbnail_mae(path_a, path_b, size=32):
    '''Mean absolute difference (0-255) between small grayscale versions of two images.'''
    thumbnails = []
    for path in (path_a, path_b):
        with Image.open(path) as image:
            small = image.convert("L").resize((size, size), Image.BILINEAR)
            thumbnails.append(np.asarray(small, dtype=np.float32))
    return float(np.abs(thumbnails[0] - thumbnails[1]).mean())


def find_duplicate_groups(records, raw_dir, max_distance=8, max_mae=30, block_size=256):
    '''
    Mark exact duplicates and assign group_id to every readable record.
    Returns one review row per dHash candidate pair (written to duplicate_review.csv):
        same_group   - confirmed near duplicate with the same label
        needs_review - confirmed near duplicate with different labels
        different    - dHash candidate rejected by the thumbnail check
    Records are changed in place.
    '''
    raw_dir = Path(raw_dir)
    # 1. Exact duplicates: same file content.
    by_sha = {}
    for record in records:
        if record["readable"] in (True, "True"):
            by_sha.setdefault(record["sha256"], []).append(record)

    for copies in by_sha.values():
        if len(copies) < 2:
            continue
        copies.sort(key=lambda record: record["path"])
        labels = {record["class_name"] for record in copies}
        if len(labels) > 1:
            for record in copies:
                record["decision"] = "exclude"
                record["reason"] = "exact duplicate with conflicting labels"
        else:
            for record in copies[1:]:
                record["decision"] = "exclude"
                record["reason"] = f"exact duplicate of {copies[0]['image_id']}"

    # 2. Near duplicates among the images that are still kept.
    kept = [record for record in records if record["decision"] == "keep"]
    hashes = np.array([int(record["dhash"], 16) for record in kept], dtype=np.uint64)

    review_rows = []
    for start in range(0, len(kept), block_size):
        stop = min(start + block_size, len(kept))
        distances = _hamming_distances(hashes, start, stop)
        rows, cols = np.nonzero(distances <= max_distance)
        for row, col in zip(rows, cols):
            i, j = start + int(row), int(col)
            if j <= i:
                continue

            a, b = kept[i], kept[j]
            mae = thumbnail_mae(raw_dir / a["path"], raw_dir / b["path"])
            if mae <= max_mae:
                decision = "same_group" if a["class_name"] == b["class_name"] else "needs_review"
            else:
                decision = "different"

            review_rows.append({
                "image_a": a["image_id"],
                "image_b": b["image_id"],
                "class_a": a["class_name"],
                "class_b": b["class_name"],
                "distance": int(distances[row, col]),
                "mae": round(mae, 1),
                "decision": decision,
            })

    # 3. One group_id per connected set of near duplicates.
    assign_group_ids(records, review_rows)

    return review_rows


def assign_group_ids(records, review_rows):
    '''
    Give every kept record a group_id; pairs whose decision links them
    (same_group, needs_review, keep_both) share one group. Excluded records get "".
    '''
    kept = [record for record in records if record["decision"] == "keep"]
    position = {record["image_id"]: index for index, record in enumerate(kept)}
    parent = list(range(len(kept)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for row in review_rows:
        if row["decision"] not in ("same_group", "needs_review", "keep_both"):
            continue
        if row["image_a"] in position and row["image_b"] in position:
            parent[find(position[row["image_a"]])] = find(position[row["image_b"]])

    roots = {}
    for index, record in enumerate(kept):
        root = find(index)
        if root not in roots:
            roots[root] = f"g{len(roots):05d}"
        record["group_id"] = roots[root]

    for record in records:
        if record["decision"] != "keep":
            record["group_id"] = ""

    return records


def apply_review_decisions(records, review_rows):
    '''
    Apply manual decisions for label-conflicting near duplicates.
    Allowed decisions: same_group, different, keep_both, exclude_a, exclude_b, exclude_both.
    Groups are rebuilt afterwards, so pairs marked "different" are no longer linked.
    Raises if any pair is still "needs_review".
    '''
    pending = [row for row in review_rows if row["decision"] == "needs_review"]
    if pending:
        raise ValueError(f"{len(pending)} near-duplicate pairs still need review")

    by_id = {record["image_id"]: record for record in records}
    for row in review_rows:
        targets = {
            "exclude_a": [row["image_a"]],
            "exclude_b": [row["image_b"]],
            "exclude_both": [row["image_a"], row["image_b"]],
        }.get(row["decision"], [])
        for image_id in targets:
            by_id[image_id]["decision"] = "exclude"
            by_id[image_id]["reason"] = "near duplicate removed after review"

    return assign_group_ids(records, review_rows)


def save_csv(rows, path, fields):
    '''Write rows with "\n" line endings so file hashes match on Windows and Linux.'''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))
