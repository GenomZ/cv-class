"""Reference vision transformer for the practical notebook P8, written exactly like its solution cells.

The notebook falls back on these when a step is not finished yet (``ptests_p8.yours_or_reference``) and the
checks in ``ptests_p8`` compare the student's code with them. The design follows the Vision Transformer (ViT,
Dosovitskiy et al., "An image is worth 16x16 words", 2020) at a tiny size for 28 x 28 grey pictures:

    PatchEmbedding(image_size=28, patch_size=7, in_channels=1, dim=32)
        picture -> 1 class token + one token per patch, each `dim` numbers, plus a learned position row
    transformer_block(dim, heads)
        one nn.TransformerEncoderLayer: layer norm, attention, skip connection; layer norm, MLP, skip connection
    TinyViT(image_size=28, patch_size=7, dim=32, depth=2, heads=4, num_classes=10)
        PatchEmbedding -> `depth` transformer blocks -> layer norm of the class token -> linear layer to the scores
    small_cnn()
        network B of P1 (two convolutions, two max-pools, two linear layers), for the comparison
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """A picture -> a table of tokens: one class token, then one token per patch, each `dim` numbers long."""

    def __init__(self, image_size=28, patch_size=7, in_channels=1, dim=32):
        super().__init__()
        patches_per_side = image_size // patch_size
        number_of_patches = patches_per_side * patches_per_side
        self.to_tokens = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.randn(1, number_of_patches + 1, dim))

    def forward(self, images):
        x = self.to_tokens(images)                  # batch x dim x 4 x 4
        x = x.flatten(2)                            # batch x dim x 16
        x = x.transpose(1, 2)                       # batch x 16 x dim
        batch_size = x.shape[0]
        class_tokens = self.class_token.expand(batch_size, -1, -1)     # batch x 1 x dim
        x = torch.cat([class_tokens, x], dim=1)     # batch x 17 x dim
        x = x + self.position                       # batch x 17 x dim
        return x


def transformer_block(dim, heads):
    """One transformer block: attention and a small MLP, each with a layer norm before and a skip connection."""
    block = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dim_feedforward=2 * dim,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return block


class TinyViT(nn.Module):
    """A tiny vision transformer: patch embedding, `depth` transformer blocks, the class token -> scores."""

    def __init__(self, image_size=28, patch_size=7, dim=32, depth=2, heads=4, num_classes=10):
        super().__init__()
        self.embedding = PatchEmbedding(image_size, patch_size, 1, dim)
        self.blocks = nn.Sequential()
        for k in range(depth):
            self.blocks.add_module("block" + str(k + 1), transformer_block(dim, heads))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, images):
        tokens = self.embedding(images)             # batch x 17 x dim
        tokens = self.blocks(tokens)                # batch x 17 x dim
        class_token = tokens[:, 0]                  # batch x dim: row 0 of every picture
        class_token = self.norm(class_token)        # batch x dim
        scores = self.head(class_token)             # batch x 10
        return scores


def small_cnn():
    """Network B of P1: the small convolutional network, for the comparison."""
    cnn = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(1568, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    return cnn
