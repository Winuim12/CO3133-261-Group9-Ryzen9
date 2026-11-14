import csv
import json
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from src.data.collate import make_collate
from src.data.dataset import MemeDataset
from src.data.tokenizer import Vocabulary
from src.data.transforms import resize_pad

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config/base.yaml").read_text(encoding="utf-8"))["data"]
needs_data = pytest.mark.skipif(not (ROOT / CFG["train_manifest"]).exists(), reason="manifests not generated")
# The raw images and the vocabulary are git-ignored, so the batch test only runs where they exist locally.
needs_raw = pytest.mark.skipif(
    not ((ROOT / CFG["vocabulary"]).exists() and (ROOT / CFG["raw_dir"] / "images").exists()),
    reason="raw images or vocabulary not available (git-ignored)")


def rows(name):
    with open(ROOT / CFG[f"{name}_manifest"], newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_resize_pad_keeps_aspect_and_size():
    out = resize_pad(Image.new("RGB", (400, 100), (255, 0, 0)), 224)
    assert out.size == (224, 224)
    assert out.getpixel((112, 112)) == (255, 0, 0)       # image centred
    assert out.getpixel((112, 2)) != (255, 0, 0)          # padded, not stretched


def test_vocab_special_ids_and_unk():
    v = Vocabulary.build(["a a b b c"], min_freq=2)
    assert v.token_to_id["<pad>"] == 0 and v.token_to_id["<unk>"] == 1
    assert v.encode("a zzz") == [v.token_to_id["a"], 1]
    assert v.encode("!!!") == [1]


@needs_data
def test_splits_disjoint_and_all_classes():
    sp = {n: rows(n) for n in ("train", "val", "test")}
    for n, r in sp.items():
        assert {int(x["label_id"]) for x in r} == {0, 1, 2, 3, 4}, n
        assert list(r[0].keys()) == ["image_name", "image_path", "text_corrected", "label_id", "group_id"]
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not {x["group_id"] for x in sp[a]} & {x["group_id"] for x in sp[b]}
        assert not {x["image_name"] for x in sp[a]} & {x["image_name"] for x in sp[b]}


@needs_data
def test_manifest_hashes_match_report():
    import hashlib
    rep = json.loads((ROOT / "data/manifests/split_report.json").read_text())
    for n in ("train", "val", "test"):
        raw = (ROOT / CFG[f"{n}_manifest"]).read_bytes().replace(b"\r\n", b"\n")  # LF-normalised
        h = hashlib.sha256(raw).hexdigest()
        assert h == rep["splits"][n]["sha256"]


@needs_data
@needs_raw
def test_one_batch_contract():
    v = Vocabulary.load(ROOT / CFG["vocabulary"])
    ds = MemeDataset(CFG["train_manifest"], v, train=True)
    b = next(iter(DataLoader(ds, batch_size=8, shuffle=False, collate_fn=make_collate(v.pad_id))))
    assert b["image"].shape == (8, 3, 224, 224) and b["image"].dtype == torch.float32
    assert b["text_ids"].dtype == torch.long and b["text_ids"].shape[0] == 8
    assert b["text_ids"].shape[1] <= CFG["max_text_length"]
    assert (b["text_lengths"] >= 1).all() and b["text_lengths"].max() == b["text_ids"].shape[1]
    assert b["label"].min() >= 0 and b["label"].max() <= 4
    assert len(b["image_name"]) == 8
    for i in range(8):  # padding only after true length
        assert (b["text_ids"][i, b["text_lengths"][i]:] == v.pad_id).all()
