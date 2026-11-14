"""Clean the Memotion dataset and resolve exact duplicates.

Prints every step with the number of samples removed and remaining.
Raw files are never modified. Outputs (under data/clean/):
  cleaned.csv        retained samples (+ group_id for later group-aware splitting)
  removed.csv        every excluded sample with step and reason (audit trail)

Run from anywhere:  python assignment3/scripts/clean_dataset.py
"""
import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["very_negative", "negative", "neutral", "positive", "very_positive"]


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def pixel_hash(path: Path):
    """SHA-256 of decoded RGB pixels (+ size). Returns None if the image is unreadable."""
    try:
        with Image.open(path) as im:
            im.load()
            rgb = im.convert("RGB")
            h = hashlib.sha256(f"{rgb.size}".encode())
            h.update(rgb.tobytes())
            return h.hexdigest()
    except Exception:
        return None


NEAR_DUP_THRESHOLD = 5.0   # mean abs gray diff (0-255) on 64x64 thumbnails; re-encoded/resized copies are < ~3


def thumb(path: Path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((64, 64)), dtype=float)


def near_dup_clusters(names, images):
    """Cluster file names whose thumbnails are near-identical (union-find, single link)."""
    th = {n: thumb(images / n) for n in names}
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    names = list(names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if np.abs(th[a] - th[b]).mean() <= NEAR_DUP_THRESHOLD:
                parent[find(a)] = find(b)
    return {n: find(n) for n in names}


class Pipeline:
    def __init__(self, df):
        self.df = df
        self.removed = []
        self.n0 = len(df)
        self.step_no = 0
        print(f"{'Step':<58}{'removed':>9}{'remaining':>11}")
        print("-" * 78)
        self._row("0. Raw rows in labels.csv", 0)

    def _row(self, name, removed):
        print(f"{name:<58}{removed:>9}{len(self.df):>11}")

    def drop(self, mask, name, reason):
        """Remove rows where mask is True, log them, print the step."""
        self.step_no += 1
        gone = self.df[mask]
        if len(gone):
            self.removed.append(gone.assign(step=name, reason=reason))
        self.df = self.df[~mask]
        self._row(f"{self.step_no}. {name}", len(gone))
        print(f"     reason: {reason}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data/raw/memotion")
    ap.add_argument("--out", type=Path, default=ROOT / "data/clean")
    args = ap.parse_args()
    images = args.raw / "images/images"
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.raw / "labels.csv", encoding="utf-8-sig", keep_default_na=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.rename(columns={"overall_sentiment": "label"})
    df["text_corrected"] = df["text_corrected"].astype(str)

    p = Pipeline(df)

    # 1. Duplicate file names (alignment key must be unique)
    p.drop(p.df.duplicated("image_name", keep="first"),
           "Repeated image_name rows", "image_name must be a unique key for aligning image and text")

    # 2. Invalid / missing labels
    p.drop(~p.df["label"].isin(LABELS), "Invalid or missing sentiment label", "label is not one of the 5 sentiment classes")

    # 3. Missing image files
    exists = p.df["image_name"].map(lambda f: (images / f).is_file())
    p.drop(~exists, "Image file missing", "image file does not exist, so the image modality is missing")

    # 4. Undecodable images (full decode) + compute pixel hash on the way
    print("   (decoding all images, this takes a minute...)")
    hashes = p.df["image_name"].map(lambda f: pixel_hash(images / f))
    p.df = p.df.assign(pixel_sha256=hashes)
    p.drop(p.df["pixel_sha256"].isna(), "Image cannot be fully decoded", "image file is truncated/corrupt and cannot be decoded")

    # 5. Blank corrected text
    blank = p.df["text_corrected"].str.strip().eq("")
    p.drop(blank, "Blank text_corrected", "no text, so the image-text pair is incomplete")

    # 5b. Junk / non-caption text (scrape or spreadsheet artifacts: "#NAME?", "<html...")
    junk = p.df["text_corrected"].str.strip().str.lower().str.match(
        r"^(#name\?|<html|<!doctype|<\?xml)|^[^a-z0-9]*$")
    p.drop(junk, "Junk text (#NAME?, <html>, no alphanumerics)", "text is not a caption (#NAME? / HTML dump / no alphanumerics)")

    # 6. Exact duplicates: same pixels AND same normalized text = equivalent pair
    p.df = p.df.assign(text_norm=p.df["text_corrected"].map(norm_text))
    key = ["pixel_sha256", "text_norm"]
    nlabels = p.df.groupby(key)["label"].transform("nunique")
    p.drop(nlabels > 1, "Equivalent pairs with conflicting labels (drop all)",
           "identical image+text with different labels: label cannot be trusted (A3)")

    # 7. Equivalent pairs with same label: keep lexicographically smallest filename
    ordered = p.df.sort_values("image_name")
    redundant = ordered.duplicated(key, keep="first")
    p.drop(p.df.index.isin(ordered.index[redundant]),
           "Equivalent pairs, same label (keep 1 copy)", "identical image+text+label: redundant copy, keep 1 (A1)")

    # 7b. Same image (pixels) with conflicting labels, even if the text differs slightly
    px_conflict = p.df.groupby("pixel_sha256")["label"].transform("nunique") > 1
    p.drop(px_conflict, "Same image, conflicting labels (drop all)",
           "same image with different labels (text differs slightly): label noise (A4)")

    # 7c. Same image + same label but slightly different text (OCR/typo variants): keep 1  (A2)
    ordered = p.df.sort_values("image_name")
    redundant = ordered.duplicated("pixel_sha256", keep="first")
    p.drop(p.df.index.isin(ordered.index[redundant]),
           "Same image, same label, text differs slightly (keep 1 copy)",
           "same image+label, only OCR/typo differences in text, redundant copy (A2)")

    # 7d. Same text on near-identical images (re-encoded / resized copies)  (B1 / B2)
    #     Different images that merely share a caption (meme templates) are KEPT.
    tg = p.df.groupby("text_norm")["image_name"].transform("size") > 1
    cand = p.df[tg]
    cluster = {}
    for _, g in cand.groupby("text_norm"):
        cluster.update(near_dup_clusters(g["image_name"], images))
    p.df = p.df.assign(near_cluster=p.df["image_name"].map(cluster))
    in_cl = p.df["near_cluster"].notna()
    csize = p.df.groupby("near_cluster")["image_name"].transform("size")
    clab = p.df.groupby("near_cluster")["label"].transform("nunique")
    p.drop(in_cl & (csize > 1) & (clab > 1),
           "Same text, near-identical image, conflicting labels (drop all)",
           "re-encoded/resized copy of the same meme with different labels: label noise (B2)")
    ordered = p.df.sort_values("image_name")
    sub = ordered[ordered["near_cluster"].notna()]
    redundant = sub.duplicated("near_cluster", keep="first")
    p.drop(p.df.index.isin(sub.index[redundant]),
           "Same text, near-identical image, same label (keep 1 copy)",
           "re-encoded/resized copy of the same meme, same label: redundant copy (B1)")
    p.df = p.df.drop(columns=["near_cluster"])

    # 7e. Remaining repeated captions on different images (decision after manual review):
    #     same label -> keep 1 of the group; different labels -> drop the whole group
    tsize = p.df.groupby("text_norm")["image_name"].transform("size")
    tlab = p.df.groupby("text_norm")["label"].transform("nunique")
    p.drop((tsize > 1) & (tlab > 1),
           "Same text, different images, conflicting labels (drop all)",
           "same caption with different labels after manual review: label noise (B2)")
    ordered = p.df.sort_values("image_name")
    redundant = ordered.duplicated("text_norm", keep="first")
    p.drop(p.df.index.isin(ordered.index[redundant]),
           "Same text, different images, same label (keep 1 copy)",
           "same caption and same label after manual review: keep 1 copy (B1)")

    # 8. Group id: shared pixel hash => same group (kept together when splitting)
    df = p.df.copy()
    df["group_id"] = df["pixel_sha256"].map({h: i for i, h in enumerate(sorted(df["pixel_sha256"].unique()))})

    print("-" * 78)
    print(f"{'Final cleaned samples':<58}{p.n0 - len(df):>9}{len(df):>11}   (total removed / kept)")

    # Flags that are reported but NOT removed (need manual review per proposal)
    same_px = df.groupby("pixel_sha256")["image_name"].transform("size") > 1
    rep_txt = df.groupby("text_norm")["image_name"].transform("size") > 1
    print("\nFlagged, kept for manual review (not removed):")
    print(f"  same image, different text : {same_px.sum()} samples in "
          f"{df.loc[same_px, 'pixel_sha256'].nunique()} groups")
    print(f"  repeated text, other image : {rep_txt.sum()} samples in "
          f"{df.loc[rep_txt, 'text_norm'].nunique()} text groups")

    print("\nClass distribution after cleaning:")
    cnt = df["label"].value_counts().reindex(LABELS)
    for lab, c in cnt.items():
        print(f"  {lab:<14}{c:>6}  ({c / len(df):.1%})")

    df = df.drop(columns=["text_norm"])
    df.to_csv(args.out / "cleaned.csv", index=False, encoding="utf-8-sig")
    removed = pd.concat(p.removed) if p.removed else pd.DataFrame()
    removed.drop(columns=["text_norm"], errors="ignore").to_csv(
        args.out / "removed.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved {args.out / 'cleaned.csv'} and {args.out / 'removed.csv'}")


if __name__ == "__main__":
    main()
