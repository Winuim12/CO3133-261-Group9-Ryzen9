"""Package local source and images as two Kaggle datasets, using only the standard library."""

import argparse
import csv
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


def write_metadata(folder, user, slug, title):
    (folder / "dataset-metadata.json").write_text(json.dumps({
        "title": title, "id": f"{user}/{slug}", "licenses": [{"name": "other"}],
    }, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--kaggle-user", help="Write dataset-metadata.json for the Kaggle CLI")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--raw-dir", type=Path, help="Image root containing archive/Images")
    source.add_argument("--data-zip", type=Path, help="Existing downloaded dataset ZIP; no extraction needed")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (args.output or root / ".cache/kaggle_upload").resolve()
    source_dir, data_dir = output / "source", output / "data"
    source_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.kaggle_user:
        write_metadata(source_dir, args.kaggle_user, "co3133-a2-source", "CO3133 A2 source")
        write_metadata(data_dir, args.kaggle_user, "co3133-a2-food-raw", "CO3133 A2 food images")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", "."], cwd=root, text=True
    ).strip())
    with zipfile.ZipFile(source_dir / "assignment2_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for directory in ("src", "experiments", "config", "data/manifests"):
            for path in sorted((root / directory).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("SOURCE_COMMIT.txt", commit + "\n")
        archive.writestr("SOURCE_DIRTY.txt", str(dirty) + "\n")
    print(f"Source ready: {source_dir / 'assignment2_source.zip'}", flush=True)
    print(f"Source commit: {commit}; uncommitted changes: {dirty}")
    if args.skip_data:
        return
    data_zip = data_dir / "food_raw.zip"
    if args.data_zip:
        if args.data_zip.resolve() == data_zip.resolve():
            parser.error("Input dataset ZIP must differ from output food_raw.zip")
        if not args.data_zip.is_file():
            parser.error(f"Dataset ZIP does not exist: {args.data_zip}")
        expected = set()
        for split in ("train", "val", "test"):
            with (root / "data/manifests" / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
                expected.update(row["path"] for row in csv.DictReader(stream))
        with zipfile.ZipFile(args.data_zip) as dataset:
            members = {}
            for item in dataset.infolist():
                name = item.filename.replace("\\", "/")
                if item.is_dir() or "Images/" not in name:
                    continue
                target = "archive/Images/" + name.split("Images/", 1)[1]
                if target in expected:
                    if target in members:
                        parser.error(f"Ambiguous image in dataset ZIP: {target}")
                    members[target] = item
            missing = expected - members.keys()
            if missing:
                parser.error(f"Dataset ZIP lacks {len(missing)} manifest images; example: {sorted(missing)[0]}")
            with zipfile.ZipFile(data_zip, "w", zipfile.ZIP_STORED) as archive:
                for name, item in sorted(members.items()):
                    with dataset.open(item) as image, archive.open(name, "w", force_zip64=True) as destination:
                        shutil.copyfileobj(image, destination)
        print(f"Data ready: {data_zip} ({len(expected)} images)")
        return
    raw = (args.raw_dir or root / "data/raw").resolve()
    images = [p for p in raw.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    if not images:
        parser.error(
            f"No images found in {raw}. Source ZIP was created. "
            "Use --raw-dir PATH for images elsewhere, --data-zip PATH for the downloaded "
            "dataset ZIP, or --skip-data if the image dataset is already on Kaggle. "
            "Cloning this repository does not download dataset images."
        )
    # JPEGs are already compressed; storing avoids expensive recompression.
    with zipfile.ZipFile(data_zip, "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(images):
            archive.write(path, path.relative_to(raw).as_posix())
    print(f"Data ready: {data_zip} ({len(images)} files)")


if __name__ == "__main__":
    main()
