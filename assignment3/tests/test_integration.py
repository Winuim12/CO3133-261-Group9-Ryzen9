"""End-to-end pipeline on a tiny synthetic dataset (no raw data or downloads)."""

import csv

import pytest
import torch
from PIL import Image

from experiments.compare import collect_runs, summarise, write_markdown
from experiments.context import build_data
from experiments.evaluate import evaluate_model, load_checkpoint_into
from src.config import load_config
from src.data.tokenizer import Vocabulary
from src.engine import fit
from src.metrics import macro_f1
from src.models.factory import build_model
from src.reproducibility import (
    count_parameters, create_run_paths, data_fingerprint, file_sha256, make_generator,
    resolve_device, save_json, set_seed,
)

WORDS = ["sad", "bad", "ok", "good", "great"]


@pytest.fixture
def project(tmp_path):
    """Fake project root with images, manifests, and a training-only vocabulary."""
    image_dir = tmp_path / "data/raw/memotion/images/images"
    image_dir.mkdir(parents=True)
    manifests = tmp_path / "data/manifests"
    manifests.mkdir(parents=True)
    texts = {}
    for split, count in (("train", 20), ("val", 10), ("test", 10)):
        rows = []
        for i in range(count):
            label = i % 5
            name = f"{split}_{i}.png"
            Image.new("RGB", (40 + i, 30), color=(label * 50, 100, 200 - label * 40)).save(image_dir / name)
            rows.append({"image_name": name, "image_path": f"data/raw/memotion/images/images/{name}",
                         "text_corrected": f"this meme is {WORDS[label]} {WORDS[label]}",
                         "label_id": label, "group_id": f"{split}{i}"})
        texts[split] = [row["text_corrected"] for row in rows]
        with (manifests / f"{split}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    Vocabulary.build(texts["train"], min_freq=1).save(tmp_path / "data/processed/vocabulary.json")
    return tmp_path


def small_config(name):
    config = load_config(f"config/{name}.yaml")
    config["model"].update(image_weights=None, embedding_dim=8, text_feature_dim=12, projection_dim=16)
    config["training"].update(batch_size=8, epochs=2, early_stopping_patience=2, device="cpu")
    config["data"]["image_size"] = 64
    return config


def test_seed_and_device_helpers():
    set_seed(7)
    first = torch.rand(3)
    set_seed(7)
    assert torch.equal(first, torch.rand(3))
    assert torch.randperm(10, generator=make_generator(1)).tolist() == \
        torch.randperm(10, generator=make_generator(1)).tolist()
    assert resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device("cuda")


def test_manifest_hash_is_line_ending_independent(tmp_path):
    (tmp_path / "lf.csv").write_bytes(b"a,b\n1,2\n")
    (tmp_path / "crlf.csv").write_bytes(b"a,b\r\n1,2\r\n")
    assert file_sha256(tmp_path / "lf.csv", True) == file_sha256(tmp_path / "crlf.csv", True)


def test_real_manifests_match_split_report():
    fingerprint = data_fingerprint(load_config("config/concat.yaml"))
    if fingerprint.get("matches_split_report") is None:
        pytest.skip("split report not available")
    assert fingerprint["matches_split_report"]


@pytest.mark.parametrize("name", ["text_only", "concat", "gated", "fixed_average"])
def test_train_evaluate_compare_pipeline(project, name):
    config = small_config(name)
    set_seed(42)
    data = build_data(config, root=project)
    batch = next(iter(data.loaders["train"]))
    assert batch["image"].shape[1:] == (3, 64, 64) and batch["label"].dtype == torch.long

    model = build_model(config, vocab_size=len(data.vocab))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    paths = create_run_paths(config, name, seed=42, run_name="test", root=project)
    history = fit(model, data.loaders["train"], data.loaders["val"], optimizer, "cpu",
                  epochs=2, patience=2, score_fn=macro_f1,
                  checkpoint_path=paths["checkpoint_path"], label_mapping=config["labels"])
    assert len(history) == 2

    fresh = build_model(config, vocab_size=len(data.vocab), load_pretrained=False)
    checkpoint = load_checkpoint_into(fresh, paths["checkpoint_path"], config)
    result = evaluate_model(fresh, data.loaders["val"], "cpu", config, "val", paths["run_dir"])
    assert result["macro_f1"] == pytest.approx(checkpoint["val_macro_f1"], abs=1e-6)
    assert result["num_samples"] == 10
    assert (paths["run_dir"] / "predictions_val.csv").exists()
    assert (result["gate"] is not None) == (name in {"gated", "fixed_average"})

    save_json(paths["training_path"], {
        "run_id": paths["run_id"], "model": name, "seed": 42, "debug_run": False,
        "best_epoch": checkpoint["epoch"], "epochs_completed": len(history),
        "parameters": count_parameters(fresh),
        "compute": {"training_seconds": 1.0, "seconds_per_epoch": 0.5, "device": "cpu"},
    })
    rows = collect_runs(project / "results/runs", "val")
    assert [row["model"] for row in rows] == [name]
    write_markdown(project / "summary.md", summarise(rows), "val")
    assert name in (project / "summary.md").read_text(encoding="utf-8")
