import csv
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data/raw/memotion/labels.csv"

LABELS = [
    "very_negative",
    "negative",
    "neutral",
    "positive",
    "very_positive",
]

REQUIRED_COLUMNS = {
    "image_name",
    "text_ocr",
    "text_corrected",
    "overall_sentiment",
}

def main():
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []

        missing_columns = REQUIRED_COLUMNS - set(columns)
        if missing_columns:
            raise ValueError(f"Missing columns: {sorted(missing_columns)}")

        rows = list(reader)

    if not rows:
        raise ValueError("The CSV contains no data rows.")

    counts = Counter(row["overall_sentiment"] for row in rows)

    unknown_labels = set(counts) - set(LABELS)
    if unknown_labels:
        raise ValueError(f"Unknown labels: {sorted(unknown_labels)}")

    missing_labels = set(LABELS) - set(counts)
    if missing_labels:
        raise ValueError(f"Absent classes: {sorted(missing_labels)}")

    if any(not row["image_name"].strip() for row in rows):
        raise ValueError("Some rows have blank image names.")

    print("CSV:", CSV_PATH)
    print("Columns:", columns)
    print("Total rows:", len(rows))

    print("\nSentiment distribution:")
    for label in LABELS:
        percentage = counts[label] / len(rows) * 100
        print(f" {label:15} {counts[label]:5} ({percentage:.2f}%)")

    print("\nMissing text:")
    for column in ["text_ocr", "text_corrected"]:
        missing = sum(not row[column].strip() for row in rows)
        print(f" {column}: {missing}")

    image_names = [row["image_name"] for row in rows]
    duplicate_names = len(image_names) - len(set(image_names))
    print("\nDuplicate image names:", duplicate_names)

if __name__ == "__main__":
    main()