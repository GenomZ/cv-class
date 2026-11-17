"""Reference GoogLeNet (Inception v1) for the practical notebook P3, written exactly like its solution cells.

The notebook falls back on these when a step is not finished yet (``ptests_p3.yours_or_reference``) and the
checks in ``ptests_p3`` compare the student's layers with them. Follows the original paper (Szegedy et al.,
"Going deeper with convolutions", 2014, table 1): every convolution has a bias and is followed by a ReLU, no
batch normalisation, no local response normalisation, and without the two auxiliary classifiers.

    conv_relu(in_channels, out_channels, kernel_size, stride=1, padding=0) -> nn.Sequential(Conv2d, ReLU)
    make_branch1(in_channels, c1)                  1x1
    make_branch2(in_channels, c3_reduce, c3)       1x1 then 3x3
    make_branch3(in_channels, c5_reduce, c5)       1x1 then 5x5
    make_branch4(in_channels, pool_proj)           3x3 max-pool then 1x1
    Inception(in_channels, c1, c3_reduce, c3, c5_reduce, c5, pool_proj)
    make_stem()                                    3 x 224 x 224 -> 192 x 28 x 28
    make_inception_blocks()                        192 x 28 x 28 -> 1024 x 7 x 7 (blocks 3a ... 5b)
    make_head(in_channels, num_classes)            global average pool, dropout 0.4, linear
    GoogLeNet(num_classes=1000)
    SmallGoogLeNet(num_classes=10, inception=Inception)   the CIFAR-10 (3 x 32 x 32) variant of the notebook

BLOCKS holds the published table: name -> (in_channels, c1, c3_reduce, c3, c5_reduce, c5, pool_proj).
"""
from __future__ import annotations

import torch
import torch.nn as nn

# the published table (paper, table 1): in_channels, #1x1, #3x3 reduce, #3x3, #5x5 reduce, #5x5, pool proj
BLOCKS = {
    "inception3a": (192, 64, 96, 128, 16, 32, 32),
    "inception3b": (256, 128, 128, 192, 32, 96, 64),
    "inception4a": (480, 192, 96, 208, 16, 48, 64),
    "inception4b": (512, 160, 112, 224, 24, 64, 64),
    "inception4c": (512, 128, 128, 256, 24, 64, 64),
    "inception4d": (512, 112, 144, 288, 32, 64, 64),
    "inception4e": (528, 256, 160, 320, 32, 128, 128),
    "inception5a": (832, 256, 160, 320, 32, 128, 128),
    "inception5b": (832, 384, 192, 384, 48, 128, 128),
}


def out_channels(numbers) -> int:
    """Channels leaving a block with these seven numbers: c1 + c3 + c5 + pool_proj."""
    return numbers[1] + numbers[3] + numbers[5] + numbers[6]


def conv_relu(in_channels, out_channels, kernel_size, stride=1, padding=0):
    layers = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
    )
    return layers


def make_branch1(in_channels, c1):
    branch = nn.Sequential(
        conv_relu(in_channels, c1, kernel_size=1),
    )
    return branch


def make_branch2(in_channels, c3_reduce, c3):
    branch = nn.Sequential(
        conv_relu(in_channels, c3_reduce, kernel_size=1),
        conv_relu(c3_reduce, c3, kernel_size=3, padding=1),
    )
    return branch


def make_branch3(in_channels, c5_reduce, c5):
    branch = nn.Sequential(
        conv_relu(in_channels, c5_reduce, kernel_size=1),
        conv_relu(c5_reduce, c5, kernel_size=5, padding=2),
    )
    return branch


def make_branch4(in_channels, pool_proj):
    branch = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        conv_relu(in_channels, pool_proj, kernel_size=1),
    )
    return branch


class Inception(nn.Module):
    """GoogLeNet's Inception block: four branches side by side, their outputs stacked along the channels."""

    def __init__(self, in_channels, c1, c3_reduce, c3, c5_reduce, c5, pool_proj):
        super().__init__()
        self.branch1 = make_branch1(in_channels, c1)
        self.branch2 = make_branch2(in_channels, c3_reduce, c3)
        self.branch3 = make_branch3(in_channels, c5_reduce, c5)
        self.branch4 = make_branch4(in_channels, pool_proj)

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        out = torch.cat([y1, y2, y3, y4], dim=1)
        return out


def make_stem():
    stem = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Conv2d(64, 64, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    return stem


def make_inception_blocks(inception=None):
    """The nine blocks and the two max-pools between the stages; `inception` = the block class to use."""
    block = inception or Inception
    blocks = nn.Sequential()
    blocks.add_module("inception3a", block(192, 64, 96, 128, 16, 32, 32))
    blocks.add_module("inception3b", block(256, 128, 128, 192, 32, 96, 64))
    blocks.add_module("pool3", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    blocks.add_module("inception4a", block(480, 192, 96, 208, 16, 48, 64))
    blocks.add_module("inception4b", block(512, 160, 112, 224, 24, 64, 64))
    blocks.add_module("inception4c", block(512, 128, 128, 256, 24, 64, 64))
    blocks.add_module("inception4d", block(512, 112, 144, 288, 32, 64, 64))
    blocks.add_module("inception4e", block(528, 256, 160, 320, 32, 128, 128))
    blocks.add_module("pool4", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    blocks.add_module("inception5a", block(832, 256, 160, 320, 32, 128, 128))
    blocks.add_module("inception5b", block(832, 384, 192, 384, 48, 128, 128))
    return blocks


def make_head(in_channels, num_classes):
    head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(p=0.4),
        nn.Linear(in_channels, num_classes),
    )
    return head


class GoogLeNet(nn.Module):
    """GoogLeNet (Inception v1), the main trunk: stem, nine Inception blocks, head."""

    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = make_stem()
        self.blocks = make_inception_blocks()
        self.head = make_head(1024, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class SmallGoogLeNet(nn.Module):
    """The notebook's CIFAR-10 variant: a light stem for 32 x 32 photos, blocks 3a, 3b and 4a at a quarter of their
    width, the same head, He initialisation of the convolutions."""

    def __init__(self, num_classes=10, inception=None):
        super().__init__()
        block = inception or Inception
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential()
        self.blocks.add_module("inception3a", block(48, 16, 24, 32, 4, 8, 8))
        self.blocks.add_module("inception3b", block(64, 32, 32, 48, 8, 24, 16))
        self.blocks.add_module("pool3", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.blocks.add_module("inception4a", block(120, 48, 24, 52, 4, 12, 16))
        self.head = make_head(128, num_classes)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
