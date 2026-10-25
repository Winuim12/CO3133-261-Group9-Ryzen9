import pytest
import torch
from torch import nn

from src.training.checkpoint import BestCheckpoint


def test_checkpoint_keeps_highest_score_and_earlier_tie(tmp_path):
    path = tmp_path / "run" / "best.pt"
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = BestCheckpoint(path)

    metadata = {"run_id": "synthetic-test"}

    def update(epoch, score):
        return checkpoint.update(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_macro_f1=score,
            run_metadata=metadata,
        )

    assert update(epoch=1, score=0.4) is True
    assert path.exists()

    assert update(epoch=2, score=0.6) is True

    assert update(epoch=3, score=0.6) is False
    assert update(epoch=4, score=0.5) is False

    saved = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint.best_epoch == 2
    assert checkpoint.best_score == pytest.approx(0.6)
    assert saved["epoch"] == 2
    assert saved["val_macro_f1"] == pytest.approx(0.6)


def test_checkpoint_restores_saved_model(tmp_path):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "best.pt"

    with torch.no_grad():
        model.weight.fill_(1.0)

    checkpoint = BestCheckpoint(path)
    checkpoint.update(
        model=model,
        optimizer=optimizer,
        epoch=1,
        val_macro_f1=0.5,
        run_metadata={},
    )

    # Đổi trọng số hiện tại thành 0.
    with torch.no_grad():
        model.weight.fill_(0.0)

    saved = torch.load(path, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])

    assert torch.all(model.weight == 1.0).item()