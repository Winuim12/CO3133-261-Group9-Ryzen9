from copy import deepcopy
import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.context import ExperimentContext
from experiments.run import train_epochs
from src.data.dataloader import create_dataloaders
from src.training.trainer import run_epoch


def test_epoch_loss_is_sample_weighted_and_metrics_unchanged():
    model = nn.Linear(2, 3)
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    y = torch.tensor([0, 1, 2])
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        expected_loss = criterion(model(x), y).item()
        expected_accuracy = (model(x).argmax(1) == y).float().mean().item()
    metrics = run_epoch(model=model, loader=DataLoader(TensorDataset(x, y), batch_size=2),
                        criterion=criterion, device="cpu", num_classes=3)
    assert metrics["loss"] == pytest.approx(expected_loss)
    assert metrics["accuracy"] == pytest.approx(expected_accuracy)
    assert metrics["timing"]["samples_per_second"] > 0
    with pytest.raises(ValueError, match="empty"):
        run_epoch(model=model, loader=[], criterion=criterion, device="cpu")


def test_resume_matches_uninterrupted_cpu_training(tmp_path):
    torch.manual_seed(17)
    initial = nn.Linear(2, 3)
    dataset = TensorDataset(torch.rand(7, 2), torch.tensor([0, 1, 2, 1, 2, 0, 1]))

    def train(path, epochs, model, resume=False):
        loader = DataLoader(dataset, batch_size=3, shuffle=True,
                            generator=torch.Generator().manual_seed(42))
        context = ExperimentContext(tmp_path, {
            "training": {"epochs": epochs, "save_last": True, "amp": True},
            "data": {"num_classes": 3},
        }, torch.device("cpu"))
        history = train_epochs(context=context, model=model, criterion=nn.CrossEntropyLoss(),
                               optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
                               train_loader=loader, val_loader=DataLoader(dataset, batch_size=3),
                               checkpoint_path=path / "best_model.pt", run_metadata={}, resume=resume)
        return history

    continuous = deepcopy(initial)
    expected = train(tmp_path / "continuous", 2, continuous)
    interrupted = deepcopy(initial)
    train(tmp_path / "resumed", 1, interrupted)
    # Simulate disconnect after a new best is written but before last.pt.
    best_path = tmp_path / "resumed/best_model.pt"
    interrupted_best = torch.load(best_path, weights_only=True)
    best_path.replace(tmp_path / "resumed/best_model.previous.pt")
    interrupted_best["epoch"] = 2
    interrupted_best["val_macro_f1"] = 1.0
    torch.save(interrupted_best, best_path)
    restored = nn.Linear(2, 3)  # Deliberately different initial weights.
    actual = train(tmp_path / "resumed", 2, restored, resume=True)
    assert [r["epoch"] for r in actual] == [1, 2]
    for key, value in continuous.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    assert actual[-1]["val"]["loss"] == expected[-1]["val"]["loss"]
    saved = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    assert saved["epoch"] == 2
    assert "scaler_state_dict" in saved
    best = torch.load(best_path, weights_only=True)
    assert best["val_macro_f1"] == max(row["val"]["macro_f1"] for row in actual)
    assert json.loads((tmp_path / "resumed/progress.json").read_text())["epochs_completed"] == 2


def test_zero_workers_accepts_persistent_prefetch_options():
    dataset = TensorDataset(torch.rand(3, 2), torch.tensor([0, 1, 2]))
    loaders = create_dataloaders(dataset, dataset, dataset, batch_size=2, num_workers=0,
                                 persistent_workers=True, prefetch_factor=2)
    assert all(sum(len(labels) for _, labels in loader) == 3 for loader in loaders)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_cuda_amp_channels_last_update_and_fp32_validation():
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1),
                          nn.Flatten(), nn.Linear(4, 3)).cuda().to(memory_format=torch.channels_last)
    loader = [(torch.rand(4, 3, 8, 8), torch.tensor([0, 1, 2, 0]))]
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.amp.GradScaler("cuda")
    before = model[0].weight.detach().clone()
    result = run_epoch(model=model, loader=loader, criterion=nn.CrossEntropyLoss(),
                       device="cuda", optimizer=optimizer, scaler=scaler,
                       channels_last=True, num_classes=3)
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert not torch.equal(before, model[0].weight)
    dtypes = []
    handle = model[-1].register_forward_hook(lambda module, inputs, output: dtypes.append(output.dtype))
    run_epoch(model=model, loader=loader, criterion=nn.CrossEntropyLoss(), device="cuda",
              scaler=scaler, channels_last=True, num_classes=3)
    handle.remove()
    assert dtypes == [torch.float32]
