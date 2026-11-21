import torch

from src.models.image_encoder import ImageEncoder


def test_image_encoder_returns_frozen_features():
    # No pretrained download is needed for these behavior tests.
    encoder = ImageEncoder(weights=None, freeze=True)
    images = torch.randn(2, 3, 224, 224)

    features = encoder(images)

    assert features.shape == (2, 512)
    assert torch.isfinite(features).all()
    assert not features.requires_grad
    assert all(not p.requires_grad for p in encoder.parameters())


def test_frozen_batchnorm_state_does_not_change():
    encoder = ImageEncoder(weights=None, freeze=True)
    before = {
        name: buffer.clone()
        for name, buffer in encoder.named_buffers()
    }

    # The training loop will call train() on the containing model.
    encoder.train()
    encoder(torch.randn(2, 3, 224, 224))

    for name, buffer in encoder.named_buffers():
        torch.testing.assert_close(
            buffer, before[name], rtol=0, atol=0
        )

def test_image_only_head_learns_with_frozen_encoder():
    from src.models.unimodal import ImageOnlyClassifier

    encoder = ImageEncoder(weights=None, freeze=True)
    model = ImageOnlyClassifier(encoder)
    model.train()

    batch = {
        "image": torch.randn(2, 3, 224, 224),
        "label": torch.tensor([0, 4]),
    }

    logits = model(batch)
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()

    loss = torch.nn.functional.cross_entropy(logits, batch["label"])
    loss.backward()

    assert all(p.grad is not None for p in model.head.parameters())
    assert all(p.grad is None for p in encoder.parameters())