from pathlib import Path
import shutil
import zipfile

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "data" / "raw" / "memotion"
ARCHIVE = DESTINATION / "dataset.zip"

URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "hammadjavaid/6992-labeled-meme-images-dataset"
)

def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset...")
    with requests.get(URL, stream=True, timeout=(30,300)) as response:
        response.raise_for_status()

        with ARCHIVE.open("wb") as output:
            shutil.copyfileobj(response.raw, output)

    if not zipfile.is_zipfile(ARCHIVE):
        raise RuntimeError(
            "Kaggle did not return a ZIP file."
            "Download it manually from the dataset page instead."
        )

    print("Extracting dataset...")
    with zipfile.ZipFile(ARCHIVE) as archive:
        #Check that archive paths remain inside the destination
        root = DESTINATION.resolve()
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"Unsafe archive path: {member.filename}")

        archive.extractall(DESTINATION)

    ARCHIVE.unlink()

    print(f"\nDataset saved to: {DESTINATION}")
    csv_files = sorted(DESTINATION.rglob("*.csv"))

    if csv_files:
        print("\nCSV annotation files:")
        for path in csv_files:
            print(f"{path.relative_to(DESTINATION)}")
    else:
        print(
            "\nNo CSV files found. This upload may contain only images; "
            "the matching Memotion text and labels would still be needed."
        )

if __name__ == "__main__":
    main()