"""CNN classifier used for EVERY experiment (same architecture everywhere, only the training data changes).

Shape flow for one 28x28 grayscale image:
1x28x28 -> block1 -> 32x14x14 -> block2 -> 64x7x7 -> block3 -> 128x3x3 -> global avg pool -> 128 -> dropout -> linear -> n_classes logits
About 0.29 M trainable parameters (small on purpose: the dataset is tiny and training runs on CPU).
"""
from torch import nn


def _block(c_in, c_out):  # one VGG-style block: two 3x3 convs then halve the resolution
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),  # 3x3 conv, padding=1 keeps H,W; no bias because BatchNorm follows
        nn.BatchNorm2d(c_out),  # normalises activations -> faster, more stable training
        nn.ReLU(inplace=True),  # non-linearity
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),  # second 3x3 conv: bigger receptive field (5x5) with fewer params than one 5x5
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),  # downsample by 2 (28->14->7->3), keeps the strongest response
    )


class CNNClassifier(nn.Module):
    """Three conv blocks (28 -> 14 -> 7 -> 3), global average pooling, linear head."""

    def __init__(self, n_classes=2, in_ch=1, width=32, dropout=0.3):  # in_ch=1 grayscale; width=channels of block1; dropout fights overfitting
        super().__init__()
        self.features = nn.Sequential(  # feature extractor
            _block(in_ch, width),  # 1 -> 32 channels, 28 -> 14 px
            _block(width, width * 2),  # 32 -> 64 channels, 14 -> 7 px
            _block(width * 2, width * 4),  # 64 -> 128 channels, 7 -> 3 px
        )
        self.head = nn.Sequential(  # classifier head
            nn.AdaptiveAvgPool2d(1),  # global average pooling: 128x3x3 -> 128x1x1 (fewer params than flatten, less overfitting)
            nn.Flatten(),  # -> vector of 128
            nn.Dropout(dropout),  # randomly zero 30% of features during training only
            nn.Linear(width * 4, n_classes),  # 128 -> 2 raw scores (logits); softmax is applied in the loss / at prediction
        )

    def forward(self, x):  # x: (batch, 1, 28, 28) in [-1, 1] -> logits (batch, n_classes)
        return self.head(self.features(x))
