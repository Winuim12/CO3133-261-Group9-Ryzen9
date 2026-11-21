import torch

from src.models.text_encoder import TextEncoder


def test_text_encoder_ignores_padding_and_preserves_batch_order():
    encoder = TextEncoder(vocab_size=16)

    with torch.no_grad():
        batched = encoder(
            torch.tensor([[2, 3, 0, 0], [4, 5, 6, 0]]),
            torch.tensor([2, 3]),
        )
        short = encoder(torch.tensor([[2, 3]]), torch.tensor([2]))
        long = encoder(torch.tensor([[4, 5, 6]]), torch.tensor([3]))

    assert batched.shape == (2, 256)
    assert torch.isfinite(batched).all()
    torch.testing.assert_close(batched, torch.cat([short, long], dim=0))

def test_text_only_classifier_backpropagates_through_encoder():
    from src.models.unimodal import TextOnlyClassifier

    encoder = TextEncoder(vocab_size=16)
    model = TextOnlyClassifier(encoder)
    batch = {
        "text_ids": torch.tensor([[2, 3, 0], [4, 5, 6]]),
        "text_lengths": torch.tensor([2, 3]),
        "label": torch.tensor([0, 4]),
    }

    logits = model(batch)
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()

    loss = torch.nn.functional.cross_entropy(logits, batch["label"])
    loss.backward()

    assert encoder.embedding.weight.grad is not None
    assert all(p.grad is not None for p in encoder.gru.parameters())
    assert all(p.grad is not None for p in model.head.parameters())