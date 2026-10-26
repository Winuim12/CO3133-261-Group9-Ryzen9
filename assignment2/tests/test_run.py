#test_run.py

import torch
from torch import nn

from experiments.context import ExperimentContext
from experiments.run import train_epochs


def test_train_epochs_saves_best_validation_checkpoint(tmp_path):
    context = ExperimentContext(
        project_root=tmp_path,
        config={
            "training": {"epochs": 2},
            "data": {"num_classes": 30},
        },
        device=torch.device("cpu"),
    )

    # Small real model keeps this integration test fast.
    model = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(3, 30),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = [
        (torch.rand(2, 3, 224, 224), torch.tensor([0, 1]))
    ]
    val_loader = [
        (torch.rand(2, 3, 224, 224), torch.tensor([1, 2]))
    ]
    checkpoint_path = tmp_path / "best_model.pt"

    history = train_epochs(
        context=context,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=checkpoint_path,
        run_metadata={"config": context.config},
    )

    assert [record["epoch"] for record in history] == [1, 2]

    saved = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert saved["val_macro_f1"] == max(
        record["val"]["macro_f1"] for record in history
    )