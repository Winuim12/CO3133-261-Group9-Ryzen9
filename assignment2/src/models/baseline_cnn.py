import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(
            self,
            num_classes: int = 30,
    ):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ), # (B, 3, 224, 224) -> (B, 64, 112, 112)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # (B, 64, 112, 112) -> (B, 64, 56, 56)
            
            # Block 2
            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 64, 56, 56) -> (B, 64, 56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 64, 56, 56) -> (B, 64, 56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 64, 56, 56) -> (B, 64, 56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 64, 56, 56) -> (B, 64, 56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 64, 56, 56) -> (B, 64, 56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 64, 56, 56) -> (B, 64, 56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ), # (B, 64, 56, 56) -> (B, 128, 28, 28)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,  
                padding=1,
                bias=False,
            ), # (B, 128, 28, 28) -> (B, 128, 28, 28)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,  
                padding=1,
                bias=False,
            ), # (B, 128, 28, 28) -> (B, 128, 28, 28)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,  
                padding=1,
                bias=False,
            ), # (B, 128, 28, 28) -> (B, 128, 28, 28)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,  
                padding=1,
                bias=False,
            ), # (B, 128, 28, 28) -> (B, 128, 28, 28)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,  
                padding=1,
                bias=False,
            ), # (B, 128, 28, 28) -> (B, 128, 28, 28)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,  
                padding=1,
                bias=False,
            ), # (B, 128, 28, 28) -> (B, 128, 28, 28)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,  
                padding=1,
                bias=False,
            ), # (B, 128, 28, 28) -> (B, 128, 28, 28)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ), # (B, 128, 28, 28) -> (B, 256, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 256, 14, 14) -> (B, 256, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 256, 14, 14) -> (B, 256, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 256, 14, 14) -> (B, 256, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 256, 14, 14) -> (B, 256, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 256, 14, 14) -> (B, 256, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 256, 14, 14) -> (B, 256, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 256, 14, 14) -> (B, 256, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 256, 14, 14) -> (B, 256, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 256, 14, 14) -> (B, 256, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 256, 14, 14) -> (B, 256, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 256, 14, 14) -> (B, 256, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 5
            nn.Conv2d(
                256,
                512,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ), # (B, 256, 14, 14) -> (B, 512, 7, 7)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 512, 7, 7) -> (B, 512, 7, 7)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 512, 7, 7) -> (B, 512, 7, 7)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 512, 7, 7) -> (B, 512, 7, 7)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 512, 7, 7) -> (B, 512, 7, 7)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # (B, 512, 7, 7) -> (B, 512, 7, 7)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # (B, 512, 7, 7) -> (B, 512, 1, 1) 

        self.classifier = nn.Sequential(
            nn.Flatten(), # (B, 512, 1, 1) -> (B, 512)
            nn.Linear(512, 256), # (B, 512) -> (B, 256)
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes), # (B, 256) -> (B, num_classes) 
        )

    def forward(self, X):
        X = self.features(X)
        X = self.avgpool(X)
        return self.classifier(X)
