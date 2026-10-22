"""A small CNN for Fashion-MNIST classification.

Architecture (input: 1x28x28 grayscale):
    Conv block 1: Conv2d(1 -> 32, 3x3) + BatchNorm + ReLU + MaxPool  -> 32x14x14
    Conv block 2: Conv2d(32 -> 64, 3x3) + BatchNorm + ReLU + MaxPool -> 64x7x7
    Classifier:   Flatten -> Linear(64*7*7 -> 128) + ReLU + Dropout -> Linear(128 -> 10)

Total parameters: ~424k.
"""

import torch.nn as nn

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


class FashionCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1x28x28 -> 32x14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32x14x14 -> 64x7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
