#audit_dataset.py

import csv
from pathlib import Path
import hashlib

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data/raw/memotion"
CSV_PATH = DATA_ROOT / "labels.csv"
IMAGE_ROOT = DATA_ROOT / "images/images"
OUTPUT_PATH = PROJECT_ROOT / "data/audit/samples.csv"

FIELDS = [
    "image_name",
    "label",
    "image_exists",
    "decode_ok",
    "width",
    "height",
    "mode",
    "error",
    "corrected_text_blank",
    "ocr_text_blank",
    "corrected_word_count",
    "eligible",
    "exclusion_reason",
    "file_sha256",
    "pixel_sha256",
]

def main():
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    missing_count = 0
    failed_count = 0

    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        for index, row in enumerate(rows, start=1):
            image_path = IMAGE_ROOT / row["image_name"]

            result = {
                "image_name": row["image_name"],
                "label": row["overall_sentiment"],
                "image_exists": image_path.is_file(),
                "decode_ok": False,
                "width": "",
                "height": "",
                "mode": "",
                "error": "",
                "file_sha256": "",
                "pixel_sha256": "",
            }

            if not result["image_exists"]:
                missing_count += 1
                result["error"] = "File not found"
            else:
                try:
                    with Image.open(image_path) as image:
                        image.load()

                        with image_path.open("rb") as image_file:
                            result["file_sha256"] = hashlib.file_digest(
                                image_file, "sha256"
                            ).hexdigest()

                        rgb = image.convert("RGB")

                        pixel_hasher = hashlib.sha256()
                        pixel_hasher.update(f"{rgb.width},{rgb.height}:RGB:".encode("ascii"))
                        pixel_hasher.update(rgb.tobytes())

                        result["pixel_sha256"] = pixel_hasher.hexdigest()
                        
                        result["width"], result["height"] = image.size

                        result["mode"] = image.mode
                        result["decode_ok"] = True

                except Exception as error:
                    failed_count += 1
                    result["error"] = (
                        f"{type(error).__name__}: {error}"
                    )

            corrected_text = row["text_corrected"].strip()
            ocr_text = row["text_ocr"].strip()

            result["corrected_text_blank"] = not corrected_text
            result["ocr_text_blank"] = not ocr_text
            result["corrected_word_count"] = len(corrected_text.split())

            reasons = []

            if not result["image_exists"]:
                reasons.append("missing_image")
            elif not result["decode_ok"]:
                reasons.append("image_decode_failed")

            if result["corrected_text_blank"]:
                reasons.append("blank_corrected_text")

            result["eligible"] = not reasons
            result["exclusion_reason"] = ";".join(reasons)

            writer.writerow(result)

            if index % 500 == 0 or index == len(rows):
                print(f"Checked {index}/{len(rows)} images")

    print("\nAudit complete")
    print("CSV rows:", len(rows))
    print("Missing images:", missing_count)
    print("Failed image decodes:", failed_count)
    print("Successfully decoded:", len(rows) - missing_count - failed_count)
    print("Saved:", OUTPUT_PATH)

if __name__ == "__main__":
    main()