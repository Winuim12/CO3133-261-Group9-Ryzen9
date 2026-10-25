import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50(nn.Module):
    def __init__(
        self,
        num_classes: int = 30,
        freeze_backbone: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = (
            ResNet50_Weights.IMAGENET1K_V2
            if pretrained else None
        )
        self.model = resnet50(weights=weights)
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        self.train()

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_backbone:
            self.model.eval()
            self.model.fc.train(mode)

        return self

    def forward(self, X):
        return self.model(X)