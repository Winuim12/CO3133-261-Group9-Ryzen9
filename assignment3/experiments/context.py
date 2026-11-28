"""Shared setup for train/evaluate: configuration overrides, vocabulary, and loaders."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.config import PROJECT_ROOT
from src.data.collate import make_collate
from src.data.dataset import MemeDataset
from src.data.tokenizer import DistilBertVocabulary, Vocabulary
from src.reproducibility import make_generator, seed_worker

SPLITS = ("train", "val", "test")


@dataclass
class DataContext:
    vocab: Vocabulary | DistilBertVocabulary
    datasets: dict
    loaders: dict


def apply_overrides(config: dict, args) -> dict:
    """Command-line values override YAML; the resolved config is saved per run."""
    training = config["training"]
    for option, key in (("seed", "seed"), ("epochs", "epochs"), ("batch_size", "batch_size"),
                        ("num_workers", "num_workers"), ("device", "device"),
                        ("learning_rate", "learning_rate"), ("patience", "early_stopping_patience")):
        value = getattr(args, option, None)
        if value is not None:
            training[key] = value
    return config


def load_vocabulary(config: dict, root: Path = PROJECT_ROOT) -> Vocabulary:
    path = root / config["data"]["vocabulary"]
    if not path.exists():
        raise FileNotFoundError(
            f"Vocabulary not found: {path}\n"
            "Build it from the training manifest first:  python -m src.data.tokenizer"
        )
    vocab = Vocabulary.load(path)
    if (vocab.pad_id, vocab.unk_id) != (config["data"]["pad_id"], config["data"]["unk_id"]):
        raise ValueError("Vocabulary pad/unk ids differ from config/base.yaml.")
    return vocab


def check_raw_images(config: dict, root: Path = PROJECT_ROOT) -> None:
    raw = root / config["data"]["raw_dir"]
    if not raw.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {raw}\n"
            "Download the Kaggle dataset and extract it so that "
            "data/raw/memotion/images/images/<image files> exists (see README)."
        )


def _limited(dataset, limit, seed):
    """Deterministic random subset for debugging runs only."""
    if not limit or limit >= len(dataset):
        return dataset
    order = torch.randperm(len(dataset), generator=make_generator(seed)).tolist()
    return Subset(dataset, sorted(order[:limit]))


def build_data(config: dict, splits=SPLITS, limits: dict | None = None,
               root: Path = PROJECT_ROOT) -> DataContext:
    check_raw_images(config, root)
    data_cfg, train_cfg = config["data"], config["training"]
    if config["model"].get("text_encoder") == "distilbert":
        vocab = DistilBertVocabulary()
    else:
        vocab = load_vocabulary(config, root)
    seed = int(train_cfg["seed"])
    num_workers = int(train_cfg["num_workers"])
    limits = limits or {}

    datasets, loaders = {}, {}
    for split in splits:
        dataset = MemeDataset(
            data_cfg[f"{split}_manifest"], vocab, train=split == "train",
            image_size=int(data_cfg["image_size"]), max_text_length=int(data_cfg["max_text_length"]),
            root=root,
        )
        dataset = _limited(dataset, limits.get(split), seed)
        datasets[split] = dataset
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=split == "train",
            num_workers=num_workers,
            collate_fn=make_collate(vocab.pad_id),
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=make_generator(seed),
            persistent_workers=num_workers > 0,
        )
    return DataContext(vocab=vocab, datasets=datasets, loaders=loaders)


def move_batch(batch: dict, device) -> dict:
    """Same rule as src.engine: text lengths stay on CPU for packed GRU sequences."""
    return {
        name: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor) and name != "text_lengths" else value
        for name, value in batch.items()
    }
