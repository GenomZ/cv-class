"""Reference solutions for the practical notebook P6 (GoogLeNet with auxiliary classifiers), written exactly like
its solution cells.

P6 adds only the two auxiliary classifiers of the original GoogLeNet (Szegedy et al., "Going deeper with
convolutions", 2014, section 5); the stem, the nine Inception blocks and the head are P3's, imported from
``cvcourse.pmodels_p3``. The notebook falls back on these when a step is not finished yet
(``ptests_p6.yours_or_reference``) and the checks in ``ptests_p6`` compare the student's code with them.

    make_aux_head(in_channels, num_classes)    5x5 average pool stride 3, 1x1 conv to 128, linear 2048 -> 1024,
                                               dropout 0.7, linear 1024 -> num_classes
    GoogLeNetAux(num_classes=1000)             P3's GoogLeNet + aux heads after blocks 4a and 4d;
                                               training mode: (main, aux1, aux2), eval mode: main
    aux_loss(outputs, labels, aux_weight=0.3)  main + aux_weight * aux1 + aux_weight * aux2 (cross-entropies)
    make_small_aux_head(in_channels, num_classes)   the smaller head of the CIFAR-10 network
    SmallGoogLeNetAux(num_classes=10)          P3's SmallGoogLeNet with two small aux heads (after 3a and 3b)
"""
from __future__ import annotations

import torch.nn as nn

from . import pmodels_p3 as p3

AUX1_IN = 512           # maps out of block 4a
AUX2_IN = 528           # maps out of block 4d


def make_aux_head(in_channels, num_classes):
    head = nn.Sequential(
        nn.AvgPool2d(kernel_size=5, stride=3),
        nn.Conv2d(in_channels, 128, kernel_size=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.7),
        nn.Linear(1024, num_classes),
    )
    return head


class GoogLeNetAux(nn.Module):
    """GoogLeNet (P3) with its two auxiliary classifiers. ``aux_head``: the function that builds them."""

    def __init__(self, num_classes=1000, aux_head=None):
        super().__init__()
        make_aux = aux_head or make_aux_head
        self.stem = p3.make_stem()
        blocks = p3.make_inception_blocks()
        self.blocks_3a_to_4a = blocks[0:4]
        self.blocks_4b_to_4d = blocks[4:7]
        self.blocks_4e_to_5b = blocks[7:11]
        self.head = p3.make_head(1024, num_classes)
        self.aux1 = make_aux(512, num_classes)
        self.aux2 = make_aux(528, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks_3a_to_4a(x)
        if self.training:
            aux1 = self.aux1(x)
        x = self.blocks_4b_to_4d(x)
        if self.training:
            aux2 = self.aux2(x)
        x = self.blocks_4e_to_5b(x)
        main = self.head(x)
        if self.training:
            return main, aux1, aux2
        return main


cross_entropy = nn.CrossEntropyLoss()


def aux_loss(outputs, labels, aux_weight=0.3):
    main, aux1, aux2 = outputs
    loss_main = cross_entropy(main, labels)
    loss_aux1 = cross_entropy(aux1, labels)
    loss_aux2 = cross_entropy(aux2, labels)
    loss = loss_main + aux_weight * loss_aux1 + aux_weight * loss_aux2
    return loss


def make_small_aux_head(in_channels, num_classes):
    head = nn.Sequential(
        nn.AvgPool2d(kernel_size=5, stride=3),
        nn.Conv2d(in_channels, 16, kernel_size=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(p=0.7),
        nn.Linear(64, num_classes),
    )
    return head


class SmallGoogLeNetAux(nn.Module):
    """P3's SmallGoogLeNet for 32 x 32 photos, with a small auxiliary head after block 3a and one after 3b."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block_3a = p3.Inception(48, 16, 24, 32, 4, 8, 8)
        self.block_3b = p3.Inception(64, 32, 32, 48, 8, 24, 16)
        self.pool_and_4a = nn.Sequential()
        self.pool_and_4a.add_module("pool3", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.pool_and_4a.add_module("inception4a", p3.Inception(120, 48, 24, 52, 4, 12, 16))
        self.head = p3.make_head(128, num_classes)
        self.aux1 = make_small_aux_head(64, num_classes)
        self.aux2 = make_small_aux_head(120, num_classes)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.stem(x)
        x = self.block_3a(x)
        if self.training:
            aux1 = self.aux1(x)
        x = self.block_3b(x)
        if self.training:
            aux2 = self.aux2(x)
        x = self.pool_and_4a(x)
        main = self.head(x)
        if self.training:
            return main, aux1, aux2
        return main
