"""Build per-model error-analysis figures for the A2 M3 slides.

Reads predictions_test.csv and per_class_test.csv of the three final Kaggle runs
and writes, per model: normalised confusion matrix, per-class F1 bar chart, and a
grid of 8 misclassified test images (4 most confident + 4 random, seed 0).
Prints the numbers quoted on the slides.

Run from the repository root:
    assignment2/.venv/Scripts/python.exe assignment2/experiments/error_figures.py
"""
import csv
import io
import random
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "results" / "kaggle"
OUT = ROOT / "results" / "figures"
ZIP = ROOT / "data" / "raw" / "archive.zip"
RUNS = {
    "cnn": ("baseline_cnn/20261103T145419Z_7a003524", "#C97A20"),
    "frozen": ("resnet50_frozen/20261103T160427Z_d37a574f", "#4277A5"),
    "full": ("resnet50_finetune/20261103T170311Z_ce0f6099", "#258352"),
}


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def zip_lookup(zf):
    names = {n.replace("\\", "/"): n for n in zf.namelist()}
    return lambda p: names.get(p) or names.get(p.split("/", 1)[1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(ZIP)
    find = zip_lookup(zf)
    f1s = {}
    for key, (run, color) in RUNS.items():
        rows = read(KAGGLE / run / "predictions_test.csv")
        pc = read(KAGGLE / run / "per_class_test.csv")
        classes = [r["class_name"] for r in pc]
        n = len(classes)
        cm = np.zeros((n, n))
        for r in rows:
            cm[int(r["true_idx"]), int(r["pred_idx"])] += 1
        cmn = cm / cm.sum(1, keepdims=True)

        fig, ax = plt.subplots(figsize=(4.4, 4.4), dpi=250)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n), classes, rotation=90, fontsize=7.5)
        ax.set_yticks(range(n), classes, fontsize=7.5)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / f"ea_{key}_confusion.png")
        plt.close(fig)

        f1 = np.array([float(r["f1"]) for r in pc])
        f1s[key] = dict(zip(classes, f1))
        order = np.argsort(f1)
        fig, ax = plt.subplots(figsize=(2.7, 4.4), dpi=250)
        cols = ["#A83F49" if i in order[:3] else color for i in order]
        ax.barh(range(n), f1[order], color=cols)
        ax.set_yticks(range(n), [classes[i] for i in order], fontsize=7.5)
        ax.set_xlim(0, 1)
        ax.axvline(f1.mean(), color="#536472", ls="--", lw=0.8)
        ax.set_xlabel("F1 (dashed = macro-F1)", fontsize=8)
        ax.tick_params(axis="x", labelsize=6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / f"ea_{key}_f1.png")
        plt.close(fig)

        wrong = [r for r in rows if r["correct"] == "False"]
        conf = sorted(wrong, key=lambda r: -float(r["confidence"]))[:4]
        rest = [r for r in wrong if r not in conf]
        pick = conf + random.Random(0).sample(rest, 4)
        fig, axes = plt.subplots(2, 4, figsize=(10, 6.0), dpi=170)
        for ax, r in zip(axes.flat, pick):
            name = find(r["path"])
            img = Image.open(io.BytesIO(zf.read(name))).convert("RGB")
            img = ImageOps.fit(img, (300, 300))
            ax.imshow(img)
            ax.set_title(f"T: {r['true_class']}\nP: {r['pred_class']}\nconf = {float(r['confidence']):.2f}",
                         fontsize=13, fontweight="bold")
            ax.axis("off")
        fig.subplots_adjust(left=0.05, right=0.99, bottom=0.01, top=0.86, wspace=0.08, hspace=0.42)
        for row, label in enumerate(("Most confident errors", "Random errors")):
            box = axes[row, 0].get_position()
            fig.text(0.025, (box.y0 + box.y1) / 2, label, rotation=90, ha="center", va="center",
                     fontsize=13, fontweight="bold", color=color)
        fig.savefig(OUT / f"ea_{key}_errors.png")
        plt.close(fig)

        pairs = Counter((r["true_class"], r["pred_class"]) for r in wrong).most_common(3)
        sinks = Counter(r["pred_class"] for r in wrong).most_common(3)
        print(f"== {key}: errors {len(wrong)}/{len(rows)} macroF1 {f1.mean():.4f}")
        print("  weakest:", [(classes[i], round(f1[i], 3)) for i in order[:3]])
        print("  best:", [(classes[i], round(f1[i], 3)) for i in order[-3:]])
        print("  top pairs:", pairs)
        print("  sink classes:", sinks)
        print("  examples:", [(r["true_class"], r["pred_class"], round(float(r["confidence"]), 2)) for r in pick])
    print("== F1 of classes that are weakest for any model")
    weak = sorted({c for d in f1s.values() for c in sorted(d, key=d.get)[:3]})
    for c in weak:
        print(f"  {c}: " + " / ".join(f"{f1s[k][c]:.3f}" for k in RUNS))
    for a, b in (("cnn", "frozen"), ("frozen", "full")):
        gain = sorted(f1s[b], key=lambda c: f1s[a][c] - f1s[b][c])[:3]
        print(f"  largest gain {a}->{b}:", [(c, round(f1s[b][c] - f1s[a][c], 3)) for c in gain])


if __name__ == "__main__":
    main()
