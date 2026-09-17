import torch
import torch.nn as nn
from typing import List

class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dims: List[int] = [256, 128],
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], num_classes),
        )

    def forward(self, X):
        X = self.flatten(X)
        return self.classifier(X)
