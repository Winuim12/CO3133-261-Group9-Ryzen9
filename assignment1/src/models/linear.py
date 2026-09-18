import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, X): # X: (B, 1, 28, 28)
        X = self.flatten(X) # (B, 1, 28, 28) -> (B, 784)
        return self.linear(X) # (B, 784) -> Logits: (B, 10) 