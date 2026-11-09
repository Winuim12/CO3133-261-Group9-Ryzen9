#review_exact_duplicates.py

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "data/audit/samples.csv"
LABELS_PATH = ROOT / "data/raw/memotion/labels.csv"
OUTPUT_PATH = ROOT / "data/review/exact_duplicates.csv"


def main():
    with AUDIT_PATH.open(encoding="utf-8-sig", newline="") as file:
        audit_rows = list(csv.DictReader(file))

    with LABELS_PATH.open(encoding="utf-8-sig", newline="") as file:
        labels = {
            row["image_name"]: row
            for row in csv.DictReader(file)
        }

    groups = defaultdict(list)

    for row in audit_rows:
        if row["decode_ok"] == "True" and row["pixel_sha256"]:
            groups[row["pixel_sha256"]].append(row)

    duplicate_groups = sorted(
        (digest, members)
        for digest, members in groups.items()
        if len(members) > 1
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "group_id",
        "image_name",
        "label",
        "text_corrected",
        "label_conflict",
        "decision",
        "reviewer",
        "notes",
    ]

    # Exclusive creation protects any existing manual review.
    with OUTPUT_PATH.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for digest, members in duplicate_groups:
            conflict = len({row["label"] for row in members}) > 1

            for row in sorted(members, key=lambda item: item["image_name"]):
                writer.writerow({
                    "group_id": digest,
                    "image_name": row["image_name"],
                    "label": row["label"],
                    "text_corrected": labels[row["image_name"]]["text_corrected"],
                    "label_conflict": conflict,
                    "decision": "",
                    "reviewer": "",
                    "notes": "",
                })

    print("Duplicate groups:", len(duplicate_groups))
    print("Rows exported:", sum(len(members) for _, members in duplicate_groups))
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()