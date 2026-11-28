"""Build any of the five compared classifiers from one merged configuration."""

from torch import nn

from .fusion import FUSION_MODES, FusionClassifier
from .image_encoder import ImageEncoder
from .text_encoder import DistilBertEncoder, TextEncoder
from .unimodal import ImageOnlyClassifier, TextOnlyClassifier

MODEL_NAMES = ("image_only", "text_only", *FUSION_MODES)
_NO_WEIGHTS = {None, "", "none", "null", "random"}


def build_model(config: dict, vocab_size: int, load_pretrained: bool = True) -> nn.Module:
    """Create the model named by `config["model"]["name"]`.

    `load_pretrained=False` skips downloading encoder weights; use it only when a
    checkpoint will overwrite every parameter and buffer (evaluation) or in tests.
    """
    model_cfg, data_cfg = config["model"], config["data"]
    name = model_cfg.get("name")
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}.")
    image_architecture = model_cfg.get("image_encoder", "resnet18")
    text_architecture = model_cfg.get("text_encoder", "gru")
    if text_architecture not in ("gru", "distilbert"):
        raise ValueError(f"Unknown text encoder {text_architecture!r}.")

    num_classes = int(model_cfg["num_classes"])
    if num_classes != len(config["labels"]):
        raise ValueError("model.num_classes does not match the label mapping.")
    dropout = float(model_cfg["dropout"])

    def image_encoder():
        weights = model_cfg.get("image_weights")
        if not load_pretrained or str(weights).lower() in _NO_WEIGHTS:
            weights = None
        return ImageEncoder(
            weights=weights, freeze=bool(model_cfg["freeze_image_encoder"]),
            architecture=image_architecture,
        )

    def text_encoder():
        if text_architecture == "distilbert":
            return DistilBertEncoder(load_pretrained=load_pretrained)
        return TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=int(model_cfg["embedding_dim"]),
            feature_dim=int(model_cfg["text_feature_dim"]),
            pad_id=int(data_cfg["pad_id"]),
        )

    if name == "image_only":
        return ImageOnlyClassifier(image_encoder(), num_classes=num_classes, dropout=dropout)
    if name == "text_only":
        return TextOnlyClassifier(text_encoder(), num_classes=num_classes, dropout=dropout)
    return FusionClassifier(
        image_encoder(),
        text_encoder(),
        mode=name,
        projection_dim=int(model_cfg["projection_dim"]),
        num_classes=num_classes,
        dropout=dropout,
        gate_value=float(model_cfg.get("gate_value", 0.5)),
    )
