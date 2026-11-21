"""Collect finished runs into comparison tables and figures.

    python -m experiments.compare                      # validation results
    python -m experiments.compare --split test         # after locked test evaluation

Writes to results/comparison/ (outside the ignored results/runs/):
    runs_<split>.csv           one row per run
    summary_<split>.csv / .md  mean ± std over seeds per model
    macro_f1_<split>.png       model comparison (one dot per seed)
    per_class_f1_<split>.png   per-class F1 by model
    gate_<split>.png           mean image-gate weight per true class (gated fusion)

Debugging runs are excluded unless --include-debug is given. Only numbers read
from saved run files are reported; nothing is estimated or filled in.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

from src.config import PROJECT_ROOT
from src.reproducibility import load_json

MODEL_ORDER = ["image_only", "text_only", "concat", "gated", "fixed_average"]


def collect_runs(runs_dir: Path, split: str, include_debug: bool = False) -> list[dict]:
    rows = []
    for training_path in sorted(runs_dir.glob("*/*/training.json")):
        run_dir = training_path.parent
        evaluation_path = run_dir / f"evaluation_{split}.json"
        if not evaluation_path.exists():
            continue
        training, evaluation = load_json(training_path), load_json(evaluation_path)
        metadata_path = run_dir / "run_metadata.json"
        metadata = load_json(metadata_path) if metadata_path.exists() else {}
        if training.get("debug_run") and not include_debug:
            continue
        row = {
            "model": training["model"],
            "run_id": training["run_id"],
            "seed": training["seed"],
            "debug_run": training.get("debug_run", False),
            "accuracy": evaluation["accuracy"],
            "macro_f1": evaluation["macro_f1"],
            "weighted_f1": evaluation.get("weighted_f1"),
            "best_epoch": training["best_epoch"],
            "epochs_completed": training["epochs_completed"],
            "trainable_params": training["parameters"]["trainable"],
            "total_params": training["parameters"]["total"],
            "training_seconds": training["compute"]["training_seconds"],
            "seconds_per_epoch": training["compute"]["seconds_per_epoch"],
            "peak_vram_mib": training["compute"].get("peak_vram_allocated_mib"),
            "device": training["compute"]["device"],
            "git_commit": (metadata.get("source") or {}).get("commit"),
            "git_dirty": (metadata.get("source") or {}).get("dirty"),
            "gate_mean": ((evaluation.get("gate") or {}).get("overall") or {}).get("mean"),
        }
        for name, scores in evaluation["per_class"].items():
            row[f"f1_{name}"] = scores["f1"]
        row["_evaluation"] = evaluation
        rows.append(row)
    rows.sort(key=lambda r: (_order(r["model"]), r["seed"], r["run_id"]))
    return rows


def _order(model):
    return MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)


def _mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def summarise(rows: list[dict]) -> list[dict]:
    summary = []
    for model in sorted({r["model"] for r in rows}, key=_order):
        group = [r for r in rows if r["model"] == model]
        entry = {"model": model, "runs": len(group), "seeds": " ".join(str(r["seed"]) for r in group)}
        for key in ("accuracy", "macro_f1", "weighted_f1", "training_seconds", "seconds_per_epoch",
                    "peak_vram_mib", "gate_mean"):
            entry[f"{key}_mean"], entry[f"{key}_std"] = _mean_std([r[key] for r in group])
        entry["trainable_params"] = group[0]["trainable_params"]
        for key in [k for k in group[0] if k.startswith("f1_")]:
            entry[f"{key}_mean"], _ = _mean_std([r[key] for r in group])
        summary.append(entry)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(mean, std, digits=4):
    if mean is None:
        return "—"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def write_markdown(path: Path, summary: list[dict], split: str) -> None:
    lines = [
        f"# Model comparison — {split} split",
        "",
        "Mean ± sample standard deviation over seeds (± 0 when only one run exists).",
        "",
        "| Model | Runs | Accuracy | Macro-F1 | Trainable params | s/epoch | Peak VRAM (MiB) | Image gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for e in summary:
        vram = "—" if e["peak_vram_mib_mean"] is None else f"{e['peak_vram_mib_mean']:.0f}"
        gate = "—" if e["gate_mean_mean"] is None else f"{e['gate_mean_mean']:.3f}"
        lines.append(
            f"| {e['model']} | {e['runs']} | {_fmt(e['accuracy_mean'], e['accuracy_std'])} | "
            f"{_fmt(e['macro_f1_mean'], e['macro_f1_std'])} | {e['trainable_params']:,} | "
            f"{e['seconds_per_epoch_mean']:.1f} | {vram} | {gate} |"
        )
    names = {e["model"]: e for e in summary}
    if {"gated", "fixed_average"} <= names.keys():
        delta = names["gated"]["macro_f1_mean"] - names["fixed_average"]["macro_f1_mean"]
        lines += ["", f"Ablation: gated − fixed_average mean Macro-F1 = {delta:+.4f}."]
    if {"gated", "concat"} <= names.keys():
        delta = names["gated"]["macro_f1_mean"] - names["concat"]["macro_f1_mean"]
        lines += [f"Gated − concat mean Macro-F1 = {delta:+.4f}."]
    lines += ["", "Differences of this size should be judged against the seed standard deviation."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_figures(rows: list[dict], summary: list[dict], output_dir: Path, split: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    written = []
    models = [e["model"] for e in summary]

    figure, axis = plt.subplots(figsize=(7.5, 4))
    for i, entry in enumerate(summary):
        values = [r["macro_f1"] for r in rows if r["model"] == entry["model"]]
        axis.bar(i, entry["macro_f1_mean"], color="tab:blue", alpha=0.6,
                 yerr=entry["macro_f1_std"] if len(values) > 1 else None, capsize=4)
        axis.scatter([i] * len(values), values, color="black", s=14, zorder=3)
    axis.set(xticks=range(len(models)), xticklabels=models, ylabel="Macro-F1",
             title=f"Macro-F1 by model ({split}); dots are individual seeds")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path = output_dir / f"macro_f1_{split}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)

    class_keys = [k for k in summary[0] if k.startswith("f1_") and k.endswith("_mean")]
    if class_keys:
        figure, axis = plt.subplots(figsize=(9, 4))
        width = 0.8 / len(summary)
        x = np.arange(len(class_keys))
        for i, entry in enumerate(summary):
            axis.bar(x + i * width - 0.4 + width / 2, [entry[k] for k in class_keys], width, label=entry["model"])
        axis.set(xticks=x, xticklabels=[k[3:-5] for k in class_keys], ylabel="F1",
                 title=f"Per-class F1 ({split}, mean over seeds)")
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        path = output_dir / f"per_class_f1_{split}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)

    gated = [r for r in rows if r["model"] == "gated" and (r["_evaluation"].get("gate") or {}).get("per_true_class")]
    if gated:
        classes = list(gated[0]["_evaluation"]["gate"]["per_true_class"])
        figure, axis = plt.subplots(figsize=(7.5, 4))
        for r in gated:
            per_class = r["_evaluation"]["gate"]["per_true_class"]
            axis.plot(classes, [(per_class[c] or {}).get("mean") for c in classes], marker="o",
                      label=f"seed {r['seed']}")
        axis.axhline(0.5, linestyle="--", color="grey", label="fixed average (0.5)")
        axis.set(ylabel="Mean image gate", ylim=(0, 1),
                 title=f"Learned gate by true class ({split}); 1 = image, 0 = text")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
        figure.tight_layout()
        path = output_dir / f"gate_{split}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output-dir", default="results/comparison")
    parser.add_argument("--include-debug", action="store_true")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    runs_dir = runs_dir if runs_dir.is_absolute() else PROJECT_ROOT / runs_dir
    output_dir = Path(args.output_dir)
    output_dir = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir

    rows = collect_runs(runs_dir, args.split, args.include_debug)
    if not rows:
        parser.exit(1, f"No runs with evaluation_{args.split}.json under {runs_dir}"
                       f"{'' if args.include_debug else ' (debug runs excluded)'}.\n")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarise(rows)
    write_csv(output_dir / f"runs_{args.split}.csv", rows)
    write_csv(output_dir / f"summary_{args.split}.csv", summary)
    write_markdown(output_dir / f"summary_{args.split}.md", summary, args.split)
    figures = plot_figures(rows, summary, output_dir, args.split)

    print((output_dir / f"summary_{args.split}.md").read_text(encoding="utf-8"))
    if any(r["debug_run"] for r in rows):
        print("WARNING: debugging runs are included; do not report these numbers.")
    if any(r["git_dirty"] for r in rows):
        print("NOTE: some runs were trained from a working tree with uncommitted changes.")
    print(f"Wrote {len(rows)} runs to {output_dir} ({len(figures)} figures).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
