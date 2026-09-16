import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            num_classes: int = 10,
            out_channels: int = 32,
            hidden_dim: int = 128,
            dropout: float = 0.3,
    ):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 1, 28, 28) -> (B, 32, 28, 28)

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 32, 28, 28) -> (B, 32, 14, 14)
            nn.Dropout2d(p=0.1),

            # Block 2
            nn.Conv2d(
                out_channels,
                out_channels * 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 32, 14, 14) -> (B, 64, 14, 14)

            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 64, 14, 14) -> (B, 64, 7, 7)
            nn.Dropout2d(p=0.2)
        )

        # [B, 64, 7, 7] -> [B, 64, 2, 2]
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Flatten(), # (B, 64, 2, 2) -> (B, 256)
            nn.Linear(out_channels * 2 * 2 * 2, hidden_dim), # (B, 256) -> (B, 128)
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes) # (B, 128) -> (B, 10)
        )

    def forward(self, X):
        X = self.features(X)
        X = self.pool(X)
        return self.classifier(X)

        
