"""Reference code for the practical notebook P7 (object detection with boxes), written exactly like its cells.

The notebook falls back on these when a step is not finished yet (``ptests_p7.yours_or_reference``) and the
checks in ``ptests_p7`` compare the student's answers with them.

    box_iou_one(box_a, box_b)              step 1: IoU of two boxes (x1, y1, x2, y2), plain arithmetic
    make_body()                            given: 3 x 64 x 64 picture -> 1024 numbers
    make_class_head(in_features, num_classes)   step 2: Linear -> 64, ReLU, Linear -> num_classes
    make_box_head(in_features)             step 2: Linear -> 64, ReLU, Linear -> 4, Sigmoid
    TinyDetector(num_classes=3)            given: body + the two heads; forward returns (class_scores, box)
    make_grid_net(num_classes=3)           given: the grid detector of section 4, 3 x 64 x 64 -> 8 x 8 x 8
"""
from __future__ import annotations

import torch.nn as nn


def box_iou_one(box_a, box_b):
    """IoU of two boxes, each (x1, y1, x2, y2) in pixels: from 0 (no overlap) to 1 (the same box)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    left = max(ax1, bx1)
    top = max(ay1, by1)
    right = min(ax2, bx2)
    bottom = min(ay2, by2)
    width = max(0, right - left)
    height = max(0, bottom - top)
    intersection = width * height

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    iou = intersection / union
    return iou


def make_body():
    """The body: four convolution blocks (8, 16, 32, 64 maps), then Flatten. 3 x 64 x 64 -> 64 x 4 x 4 -> 1024."""
    body = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
    )
    return body


def make_class_head(in_features, num_classes):
    """The class head: in_features numbers -> one score per class."""
    head = nn.Sequential(
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    )
    return head


def make_box_head(in_features):
    """The box head: in_features numbers -> 4 numbers between 0 and 1 (x1, y1, x2, y2 divided by 64)."""
    head = nn.Sequential(
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Linear(64, 4),
        nn.Sigmoid(),
    )
    return head


class TinyDetector(nn.Module):
    """One object per picture: a body that reads the picture, a class head and a box head."""

    def __init__(self, num_classes=3, class_head=None, box_head=None):
        super().__init__()
        self.body = make_body()
        self.class_head = (class_head or make_class_head)(1024, num_classes)
        self.box_head = (box_head or make_box_head)(1024)

    def forward(self, x):
        features = self.body(x)
        class_scores = self.class_head(features)
        box = self.box_head(features)
        return class_scores, box


def make_grid_net(num_classes=3):
    """The grid detector: 3 x 64 x 64 picture -> (1 + num_classes + 4) numbers for each of the 8 x 8 cells."""
    net = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 1 + num_classes + 4, kernel_size=1),
    )
    return net
