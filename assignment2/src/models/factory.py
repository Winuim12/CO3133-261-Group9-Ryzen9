import torch.nn as nn

from .baseline_cnn import BaselineCNN
from .resnet50 import ResNet50


def build_model(config: dict) -> nn.Module:
    name = config["name"]
    num_classes = config.get("num_classes", 30)

    if name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)

    if name == "resnet50":
        return ResNet50(
            num_classes=num_classes,
            freeze_backbone=config.get("freeze_backbone", True),
            pretrained=config.get("pretrained", True),
        )

    raise ValueError(f"Unknown model: {name!r}")