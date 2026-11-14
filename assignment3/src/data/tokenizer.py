"""Training-only vocabulary; preserves the agreed pad/unknown token IDs (config/base.yaml).

Build:  python -m src.data.tokenizer   (run from assignment3/)
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

PAD, UNK = "<pad>", "<unk>"
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str):
    return _TOKEN.findall(str(text).lower())


class Vocabulary:
    def __init__(self, token_to_id, pad_id=0, unk_id=1):
        self.token_to_id = dict(token_to_id)
        self.pad_id, self.unk_id = pad_id, unk_id

    def __len__(self):
        return len(self.token_to_id)

    @classmethod
    def build(cls, texts, min_freq=2, pad_id=0, unk_id=1):
        counts = Counter(t for s in texts for t in tokenize(s))
        t2i = {PAD: pad_id, UNK: unk_id}
        nxt = max(pad_id, unk_id) + 1
        # sort by (-freq, token) so the vocabulary is deterministic
        for tok, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if c >= min_freq:
                t2i[tok] = nxt
                nxt += 1
        return cls(t2i, pad_id, unk_id)

    def encode(self, text, max_len=64):
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokenize(text)][:max_len]
        return ids or [self.unk_id]  # never return an empty sequence

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"pad_id": self.pad_id, "unk_id": self.unk_id, "token_to_id": self.token_to_id},
            ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["token_to_id"], d["pad_id"], d["unk_id"])


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=root / "config/base.yaml")
    ap.add_argument("--min-freq", type=int, default=2)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))["data"]
    train = pd.read_csv(root / cfg["train_manifest"], keep_default_na=False)  # train only: no leakage
    v = Vocabulary.build(train[cfg["text_column"]], a.min_freq, cfg["pad_id"], cfg["unk_id"])
    v.save(root / cfg["vocabulary"])
    print(f"vocab size {len(v)} (min_freq={a.min_freq}) -> {root / cfg['vocabulary']}")
    for name in ("val", "test"):
        df = pd.read_csv(root / cfg[f"{name}_manifest"], keep_default_na=False)
        toks = [t for s in df[cfg["text_column"]] for t in tokenize(s)]
        unk = sum(t not in v.token_to_id for t in toks)
        print(f"{name}: unknown-token rate {unk / len(toks):.1%}")


if __name__ == "__main__":
    main()
