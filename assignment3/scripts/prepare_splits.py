"""Create group-aware, label-stratified train/val/test manifests from data/clean/cleaned.csv.

Outputs (data/manifests/): train.csv, val.csv, test.csv, split_report.json (counts + SHA-256 hashes).
Columns: image_name,image_path,text_corrected,label_id,group_id
Run:  python assignment3/scripts/prepare_splits.py
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
COLS = ["image_name", "image_path", "text_corrected", "label_id", "group_id"]


def sha256(path: Path) -> str:
    """Hash with line endings normalised to LF so Windows/Linux checkouts agree."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def split(df, seed, tries=200):
    """~70/15/15 via two StratifiedGroupKFold passes; pick the seed offset with all classes in every split."""
    for k in range(tries):
        s = seed + k
        # 1) hold out ~30% (3 of 10 folds is not exact, so use 10 folds: 7 train / 3 held out)
        outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=s)
        folds = list(outer.split(df, df["label_id"], df["group_id"]))
        held = set()
        for _, te in folds[:3]:
            held.update(te)
        held = sorted(held)
        train = df.iloc[[i for i in range(len(df)) if i not in set(held)]]
        rest = df.iloc[held]
        # 2) split held-out half/half into val/test
        inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=s)
        v, t = next(inner.split(rest, rest["label_id"], rest["group_id"]))
        val, test = rest.iloc[v], rest.iloc[t]
        if all(p["label_id"].nunique() == 5 for p in (train, val, test)):
            return train, val, test, s
    raise RuntimeError("no split with all 5 classes in every partition")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=Path, default=ROOT / "data/clean/cleaned.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data/manifests")
    ap.add_argument("--config", type=Path, default=ROOT / "config/base.yaml")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    labels = cfg["labels"]
    raw_rel = cfg["data"]["raw_dir"]

    df = pd.read_csv(a.clean, encoding="utf-8-sig", keep_default_na=False)
    df["label_id"] = df["label"].map(labels)
    assert df["label_id"].notna().all(), "unknown label"
    df["label_id"] = df["label_id"].astype(int)
    df["image_path"] = df["image_name"].map(lambda n: f"{raw_rel}/images/images/{n}")
    df = df.reset_index(drop=True)

    train, val, test, used = split(df, a.seed)

    # integrity checks
    parts = {"train": train, "val": val, "test": test}
    for x, y in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(parts[x]["group_id"]) & set(parts[y]["group_id"]), f"group overlap {x}/{y}"
        assert not set(parts[x]["image_name"]) & set(parts[y]["image_name"]), f"image overlap {x}/{y}"
    assert sum(map(len, parts.values())) == len(df)

    a.out.mkdir(parents=True, exist_ok=True)
    report = {"seed": a.seed, "seed_used": used, "total": len(df), "splits": {}}
    inv = {v: k for k, v in labels.items()}
    for name, p in parts.items():
        f = a.out / f"{name}.csv"
        p[COLS].sort_values("image_name").to_csv(f, index=False, encoding="utf-8", lineterminator="\n")
        dist = p["label_id"].value_counts().sort_index()
        report["splits"][name] = {
            "n": len(p), "frac": round(len(p) / len(df), 4),
            "groups": int(p["group_id"].nunique()),
            "classes": {inv[i]: int(c) for i, c in dist.items()},
            "sha256": sha256(f),
        }
        print(f"{name:<6}{len(p):>6} ({len(p)/len(df):.1%})  " +
              "  ".join(f"{inv[i]}={c}" for i, c in dist.items()))
    (a.out / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("No group/image overlap. Saved manifests + split_report.json in", a.out)


if __name__ == "__main__":
    main()
