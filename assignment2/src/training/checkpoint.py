import math
from pathlib import Path

import torch


class BestCheckpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.best_score = float("-inf")
        self.best_epoch = None

    def update(
        self,
        *,
        model,
        optimizer,
        epoch,
        val_macro_f1,
        run_metadata,
        scheduler=None,
    ):
        score = float(val_macro_f1)

        if not math.isfinite(score):
            raise ValueError("Validation Macro-F1 must be finite.")

        if score <= self.best_score:
            return False

        checkpoint = {
            "epoch": epoch,
            "val_macro_f1": score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "run_metadata": run_metadata,
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, self.path)

        self.best_score = score
        self.best_epoch = epoch

        return True