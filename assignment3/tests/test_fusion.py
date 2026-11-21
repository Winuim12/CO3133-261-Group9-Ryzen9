import pytest
import torch

from src.config import load_config
from src.models.factory import MODEL_NAMES, build_model
from src.models.fusion import FusionClassifier
from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder


def make_batch(batch_size=3):
    torch.manual_seed(0)
    return {
        "image": torch.randn(batch_size, 3, 64, 64),
        "text_ids": torch.tensor([[2, 3, 4], [5, 6, 0], [7, 0, 0]])[:batch_size],
        "text_lengths": torch.tensor([3, 2, 1])[:batch_size],
        "label": torch.tensor([0, 2, 4])[:batch_size],
    }


def make_fusion(mode, **kwargs):
    return FusionClassifier(
        ImageEncoder(weights=None, freeze=True),
        TextEncoder(vocab_size=10, embedding_dim=8, feature_dim=12),
        mode=mode, projection_dim=16, **kwargs,
    )


@pytest.mark.parametrize("mode", ["concat", "gated", "fixed_average"])
def test_fusion_returns_finite_logits_and_trains_heads(mode):
    model = make_fusion(mode, dropout=0.0)
    model.train()
    logits = model(make_batch())
    assert logits.shape == (3, 5)
    assert torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, make_batch()["label"]).backward()
    for module in (model.image_projection, model.text_projection, model.head, model.text_encoder):
        assert all(p.grad is not None for p in module.parameters())
    assert all(p.grad is None for p in model.image_encoder.parameters())


def test_frozen_image_encoder_is_unchanged_by_an_update():
    model = make_fusion("gated")
    before = {k: v.clone() for k, v in model.image_encoder.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model.train()
    torch.nn.functional.cross_entropy(model(make_batch()), make_batch()["label"]).backward()
    optimizer.step()
    after = model.image_encoder.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_gate_is_in_unit_interval_and_recorded():
    model = make_fusion("gated").eval()
    model(make_batch())
    assert model.last_gate.shape == (3, 16)
    assert ((model.last_gate > 0) & (model.last_gate < 1)).all()


def test_fixed_average_has_no_learned_gate_and_mixes_equally():
    model = make_fusion("fixed_average", gate_value=0.5).eval()
    assert model.gate is None
    image, text = torch.ones(2, 16), torch.zeros(2, 16)
    fused, gate = model.fuse(image, text)
    assert torch.allclose(fused, torch.full((2, 16), 0.5))
    assert torch.equal(gate, torch.full((2, 16), 0.5))
    assert "gate_value" in model.state_dict()


def test_gated_and_fixed_average_differ_only_by_the_gate_layer():
    gated, fixed = make_fusion("gated"), make_fusion("fixed_average")
    count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert count(gated) - count(fixed) == 2 * 16 * 16 + 16


def test_unknown_mode_and_gate_value_are_rejected():
    with pytest.raises(ValueError):
        make_fusion("sum")
    with pytest.raises(ValueError):
        make_fusion("fixed_average", gate_value=1.5)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_factory_builds_every_config(name):
    config = load_config(f"config/{name}.yaml")
    model = build_model(config, vocab_size=20, load_pretrained=False).eval()
    batch = make_batch()
    batch["text_ids"] = batch["text_ids"].clamp(max=19)
    with torch.no_grad():
        assert model(batch).shape == (3, 5)


def test_factory_rejects_unknown_model():
    config = load_config("config/concat.yaml")
    config["model"]["name"] = "late_fusion"
    with pytest.raises(ValueError):
        build_model(config, vocab_size=20, load_pretrained=False)
