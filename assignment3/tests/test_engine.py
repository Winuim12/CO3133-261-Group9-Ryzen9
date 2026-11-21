import math

import torch

from src.engine import train_one_epoch, validate_one_epoch, fit
from src.models.text_encoder import TextEncoder
from src.models.unimodal import TextOnlyClassifier


def test_train_one_epoch_updates_classifier():
    model = TextOnlyClassifier(
        TextEncoder(vocab_size=16, embedding_dim=8, feature_dim=12),
        dropout=0.0,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.head[-1].bias.detach().clone()

    batch = {
        "text_ids": torch.tensor([[2, 3, 0], [4, 5, 6]]),
        "text_lengths": torch.tensor([2, 3]),
        "label": torch.tensor([0, 4]),
    }

    loss = train_one_epoch(model, [batch], optimizer, device="cpu")

    assert math.isfinite(loss)
    assert loss > 0
    assert not torch.equal(model.head[-1].bias, before)

def test_validation_collects_predictions_across_batches():
    model = TextOnlyClassifier(
        TextEncoder(vocab_size=16, embedding_dim=8, feature_dim=12),
    )
    with torch.no_grad():
        # Give the model a predictable output for every sample.
        model.head[-1].weight.zero_()
        model.head[-1].bias.copy_(torch.tensor([0., 0., 0., 0., 2.]))

    batches = [
        {
            "text_ids": torch.tensor([[2, 3], [4, 5]]),
            "text_lengths": torch.tensor([2, 2]),
            "label": torch.tensor([0, 1]),
        },
        {
            "text_ids": torch.tensor([[6, 7]]),
            "text_lengths": torch.tensor([2]),
            "label": torch.tensor([4]),
        },
    ]

    result = validate_one_epoch(model, batches, device="cpu")

    assert result["targets"].tolist() == [0, 1, 4]
    assert result["predictions"].tolist() == [4, 4, 4]
    expected_loss = torch.nn.functional.cross_entropy(
        model.head[-1].bias.detach().expand(3, -1),
        torch.tensor([0, 1, 4]),
    )
    assert math.isclose(result["loss"], expected_loss.item(), rel_tol=1e-6)
    assert not model.training

def test_fit_keeps_earlier_checkpoint_on_tie(tmp_path):
    model = TextOnlyClassifier(
        TextEncoder(vocab_size=16, embedding_dim=8, feature_dim=12),
        dropout=0.0,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [{
        "text_ids": torch.tensor([[2, 3], [4, 5]]),
        "text_lengths": torch.tensor([2, 2]),
        "label": torch.tensor([0, 4]),
    }]
    scores = iter([0.20, 0.40, 0.40, 0.30])
    path = tmp_path / "best_model.pt"

    history = fit(
        model, batches, batches, optimizer, "cpu",
        epochs=10,
        patience=2,
        score_fn=lambda targets, predictions: next(scores),
        checkpoint_path=path,
        label_mapping={
            "very_negative": 0, "negative": 1, "neutral": 2,
            "positive": 3, "very_positive": 4,
        },
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert len(history) == 4
    assert checkpoint["epoch"] == 2
    assert checkpoint["val_macro_f1"] == 0.40

    restored_model = TextOnlyClassifier(
        TextEncoder(vocab_size=16, embedding_dim=8, feature_dim=12),
        dropout=0.0,
    )
    restored_model.load_state_dict(checkpoint["model_state_dict"])

    restored = validate_one_epoch(restored_model, batches, device="cpu")
    selected_epoch = history[checkpoint["epoch"] - 1]

    assert math.isclose(
        restored["loss"], selected_epoch["val_loss"], rel_tol=1e-6
    )