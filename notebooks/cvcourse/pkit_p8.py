"""Figures and small helpers for the practical notebook P8 (vision transformers).

Import it in the notebook as ``from cvcourse import pkit_p8 as V``. Every figure function draws its figure under
the cell (also when the notebook runs headless) and returns it, like the figures of ``cvcourse.practical``.

    V.show_idea(image)                          section 1: a convolution's small window vs attention between patches
    V.show_patches(image, patch_size=7)         the picture cut into patches, and the patches one by one
    V.show_table(table, title, ...)             a table of numbers (rows x columns) as a coloured grid
    V.show_tokens(tokens)                       the 17 tokens of one picture: the class token, then one per patch
    V.copy_linear_into_conv(linear, conv)       give a patch convolution the weights of a linear layer (to compare)
    V.patch_table(image, patch_size=7)          the loop of section 2.1: one row of numbers per patch
    V.shuffle_patches(image, patch_size, seed)  the same patches in another order
    V.show_pictures(images, titles)             pictures side by side
    V.same_rows(table_a, table_b)               do two tables hold the same rows, in any order?
    V.show_toy_attention(tokens, weights, new_tokens, names)   the hand-worked attention example
    V.show_weights(weights, names=None)         an attention table as a heat map (row = who looks, column = at whom)
    V.position_test(embedding, attention, image, shuffled)     why the position rows are needed
    V.attention_maps(model, images)             the attention tables of every block of a trained TinyViT
    V.show_class_attention(model, loader)       where the class token looks, for a few validation pictures
    V.show_position_similarity(model)           how alike the learned position rows of the patches are
    V.load_pretrained_vit()                     torchvision's ViT-B/16 with its ImageNet weights, if they are on disk
    V.pet_photo(name)                           the path of one Oxford-IIIT Pet photo, e.g. "pug_12"
    V.classify_photos(vit, weights, names)      the pretrained ViT's top-3 answers for a few photos
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .practical import _new_figure, _show

__all__ = [
    "show_idea", "show_patches", "show_table", "show_tokens", "copy_linear_into_conv", "patch_table",
    "shuffle_patches", "show_pictures", "same_rows", "show_toy_attention", "show_weights", "position_test",
    "attention_maps", "show_class_attention", "show_position_similarity", "load_pretrained_vit", "pet_photo",
    "classify_photos", "VIT_FILE",
]

_INK = "#0b0b0b"
_INK2 = "#52514e"
_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_GREEN = "#1baf7a"
_RED = "#d03b3b"
VIT_FILE = "vit_b_16-c867db91.pth"      # the file name torchvision gives the ImageNet weights of ViT-B/16


# ============================================================================================== small tools
def _picture(image) -> np.ndarray:
    """A picture (1 x H x W, H x W or 3 x H x W tensor) as numbers 0 to 1 for drawing (min -> 0, max -> 1)."""
    t = torch.as_tensor(image).detach().float().cpu()
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 3 and t.shape[0] == 1:
        t = t[0]
    low = t.min()
    high = t.max()
    t = (t - low) / (high - low).clamp_min(1e-8)
    if t.dim() == 3:
        return t.permute(1, 2, 0).numpy()
    return t.numpy()


def _clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def _numpy(t) -> np.ndarray:
    return torch.as_tensor(t).detach().float().cpu().numpy()


def patch_table(image, patch_size: int = 7) -> torch.Tensor:
    """The loop of section 2.1: one row per patch (left to right, top to bottom), patch_size x patch_size numbers."""
    picture = torch.as_tensor(image).detach()
    if picture.dim() == 3:
        picture = picture[0]
    size = picture.shape[-1]
    rows = []
    for top in range(0, size, patch_size):
        for left in range(0, size, patch_size):
            patch = picture[top:top + patch_size, left:left + patch_size]
            rows.append(patch.flatten())
    return torch.stack(rows)


# ============================================================================================== section 1
def show_idea(image, patch_size: int = 7, patch: int = 5):
    """Two panels: a 3 x 3 convolution sees 9 neighbouring pixels; in a ViT one patch can look at every patch."""
    pic = _picture(image)
    size = pic.shape[0]
    per_side = size // patch_size
    fig = _new_figure(10.5, 5.0)
    left, right = fig.subplots(1, 2)

    left.imshow(pic, cmap="gray", extent=(0, size, size, 0))
    row, col = 13, 9                                       # a pixel on the stroke of most digits
    left.add_patch(_rect(col - 1, row - 1, 3, 3, _ORANGE, 2.5))
    left.set_title("convolution (P1 to P5): a 3 x 3 filter\nsees 9 neighbouring pixels at a time", fontsize=11,
                   color=_INK)
    _clean(left)

    right.imshow(pic, cmap="gray", extent=(0, size, size, 0))
    for k in range(1, per_side):
        right.axhline(k * patch_size, color="white", lw=1.2, alpha=0.8)
        right.axvline(k * patch_size, color="white", lw=1.2, alpha=0.8)
    pr = patch // per_side
    pc = patch % per_side
    cy = pr * patch_size + patch_size / 2
    cx = pc * patch_size + patch_size / 2
    for r in range(per_side):
        for c in range(per_side):
            if r == pr and c == pc:
                continue
            ty = r * patch_size + patch_size / 2
            tx = c * patch_size + patch_size / 2
            right.plot([cx, tx], [cy, ty], color=_ORANGE, lw=1.6, alpha=0.9)
            right.plot([tx], [ty], marker="o", ms=5, color=_ORANGE)
    right.add_patch(_rect(pc * patch_size, pr * patch_size, patch_size, patch_size, _ORANGE, 3))
    right.set_title(f"vision transformer: {per_side * per_side} patches of {patch_size} x {patch_size};\n"
                    f"with attention, one patch can look at every other patch", fontsize=11, color=_INK)
    _clean(right)
    return _show(fig)


def _rect(x, y, w, h, color, lw):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=lw)


# ============================================================================================== section 2
def show_patches(image, patch_size: int = 7):
    """Left: the picture with the patch grid and the patch numbers. Right: the patches as separate little pictures."""
    pic = _picture(image)
    size = pic.shape[0]
    per_side = size // patch_size
    fig = _new_figure(10.5, 5.4)
    left_fig, right_fig = fig.subfigures(1, 2, wspace=0.04)
    ax = left_fig.subplots()
    ax.imshow(pic, cmap="gray", extent=(0, size, size, 0), vmin=0, vmax=1)
    for k in range(1, per_side):
        ax.axhline(k * patch_size, color=_ORANGE, lw=1.5)
        ax.axvline(k * patch_size, color=_ORANGE, lw=1.5)
    number = 0
    for r in range(per_side):
        for c in range(per_side):
            if per_side <= 8:
                ax.text(c * patch_size + 0.5, r * patch_size + 0.5, str(number), color=_ORANGE, fontsize=10,
                        va="top", ha="left", fontweight="bold")
            number += 1
    ax.set_title(f"the picture, {size} x {size}, cut into {per_side} x {per_side} = {per_side * per_side} patches",
                 fontsize=10.5, color=_INK)
    _clean(ax)
    axes = right_fig.subplots(per_side, per_side, squeeze=False)
    number = 0
    for r in range(per_side):
        for c in range(per_side):
            sub = axes[r, c]
            piece = pic[r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size]
            sub.imshow(piece, cmap="gray", vmin=0, vmax=1)
            if per_side <= 8:
                sub.set_title(f"patch {number}", fontsize=8.5, color=_INK2, pad=2)
            sub.set_xticks([])
            sub.set_yticks([])
            for side in sub.spines.values():
                side.set_edgecolor("#bbbbbb")
            number += 1
    right_fig.suptitle(f"the {per_side * per_side} patches one by one, each {patch_size} x {patch_size} pixels",
                       fontsize=10.5, color=_INK)
    return _show(fig)


def show_table(table, title: str | None = None, row_names=None, column_word: str = "numbers",
               kind: str = "tokens", highlight_first: bool = False):
    """A table of numbers as a grid of coloured cells, one row per patch (or token).

    kind="pixels": grey values as in the picture (dark = background); kind="tokens": blue for negative numbers,
    red for positive ones, white for 0.
    """
    data = _numpy(table)
    if data.ndim == 3:
        data = data[0]
    n_rows, n_cols = data.shape
    fig = _new_figure(min(11.5, 2.4 + n_cols * 0.17), 1.1 + n_rows * 0.24)
    ax = fig.subplots()
    if kind == "pixels":
        image = ax.imshow(data, cmap="gray", aspect="auto", interpolation="nearest")
        label = "pixel value"
    else:
        top = float(np.abs(data).max()) or 1.0
        image = ax.imshow(data, cmap="RdBu_r", vmin=-top, vmax=top, aspect="auto", interpolation="nearest")
        label = "value (blue < 0 < red)"
    if row_names is None:
        row_names = []
        for k in range(n_rows):
            row_names.append(f"patch {k}")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_names, fontsize=8.5)
    step = 4 if n_cols > 20 else 1
    ax.set_xticks(range(0, n_cols, step))
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel(f"{n_cols} {column_word} per row (column 0 to {n_cols - 1})", fontsize=9.5)
    if highlight_first:
        ax.add_patch(_rect(-0.5, -0.5, n_cols, 1, _GREEN, 3))
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.01)
    bar.ax.tick_params(labelsize=8)
    bar.set_label(label, fontsize=8.5)
    ax.set_title(title or f"a table of {n_rows} rows x {n_cols} {column_word}", fontsize=10.5, color=_INK, loc="left")
    return _show(fig)


def show_tokens(tokens, title: str | None = None):
    """The tokens of one picture: row 0 is the class token (green frame), rows 1 ... are the patches 0 ..."""
    data = _numpy(tokens)
    if data.ndim == 3:
        data = data[0]
    names = ["class token"]
    for k in range(data.shape[0] - 1):
        names.append(f"patch {k}")
    head = title or (f"the {data.shape[0]} tokens of the digit: the class token, then one per patch "
                     f"({data.shape[0]} x {data.shape[1]})")
    return show_table(data, head, row_names=names, kind="tokens", highlight_first=True)


def copy_linear_into_conv(linear: nn.Linear, conv: nn.Conv2d) -> None:
    """Give the patch convolution exactly the weights of the linear layer, so that the two can be compared."""
    with torch.no_grad():
        conv.weight.copy_(linear.weight.reshape(conv.weight.shape))
        conv.bias.copy_(linear.bias)
    print(f"copied the {linear.out_features} x {linear.in_features} weights and {linear.out_features} biases of the "
          f"linear layer into the {conv.out_channels} filters of {conv.in_channels} x {conv.kernel_size[0]} x "
          f"{conv.kernel_size[1]} of the convolution")


def shuffle_patches(image, patch_size: int = 7, seed: int = 0) -> torch.Tensor:
    """The same picture with its patches moved to other places (a fixed random order chosen by `seed`)."""
    picture = torch.as_tensor(image).detach().clone()
    size = picture.shape[-1]
    per_side = size // patch_size
    g = torch.Generator()
    g.manual_seed(seed)
    order = torch.randperm(per_side * per_side, generator=g)
    out = picture.clone()
    for new_place in range(per_side * per_side):
        old_place = int(order[new_place])
        r_new, c_new = new_place // per_side, new_place % per_side
        r_old, c_old = old_place // per_side, old_place % per_side
        out[..., r_new * patch_size:(r_new + 1) * patch_size, c_new * patch_size:(c_new + 1) * patch_size] = \
            picture[..., r_old * patch_size:(r_old + 1) * patch_size, c_old * patch_size:(c_old + 1) * patch_size]
    return out


def show_pictures(images, titles, patch_size: int | None = 7):
    """Pictures side by side (with the patch grid when patch_size is given)."""
    fig = _new_figure(3.2 * len(images), 3.6)
    axes = fig.subplots(1, len(images), squeeze=False)[0]
    for ax, image, title in zip(axes, images, titles):
        pic = _picture(image)
        size = pic.shape[0]
        ax.imshow(pic, cmap="gray", extent=(0, size, size, 0))
        if patch_size:
            for k in range(1, size // patch_size):
                ax.axhline(k * patch_size, color=_ORANGE, lw=1)
                ax.axvline(k * patch_size, color=_ORANGE, lw=1)
        ax.set_title(title, fontsize=10.5, color=_INK)
        _clean(ax)
    return _show(fig)


class _Answer:
    """A yes / no answer that has already been printed: true or false in Python, shows nothing in Jupyter."""

    def __init__(self, value: bool):
        self.value = bool(value)

    def __bool__(self) -> bool:
        return self.value

    def __repr__(self) -> str:
        return str(self.value)

    def _ipython_display_(self) -> None:
        return None


def same_rows(table_a, table_b):
    """Prints and returns whether every row of table_a is also a row of table_b (and the other way round)."""
    a = torch.as_tensor(table_a).detach()
    b = torch.as_tensor(table_b).detach()
    found_all = a.shape == b.shape
    same_order = found_all and bool(torch.equal(a, b))
    if found_all:
        for k in range(a.shape[0]):
            found = False
            for j in range(b.shape[0]):
                if torch.equal(a[k], b[j]):
                    found = True
                    break
            if not found:
                found_all = False
                break
    print(f"both tables have the shape {' x '.join(str(int(v)) for v in a.shape)}"
          if a.shape == b.shape else f"different shapes: {tuple(a.shape)} and {tuple(b.shape)}")
    print(f"every row of the first table is also a row of the second: {found_all}")
    print(f"in the same order: {same_order}")
    return _Answer(found_all)


# ============================================================================================== section 3
def show_toy_attention(tokens, weights, new_tokens, names):
    """Left: the tokens before (filled) and after (open) attention. Right: the weights table with its numbers."""
    old = _numpy(tokens)
    new = _numpy(new_tokens)
    w = _numpy(weights)
    fig = _new_figure(10.5, 4.3)
    ax, axw = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.15, 1]})
    colors = [_BLUE, _ORANGE, _GREEN, "#6250d6", "#e87ba4"]
    for k in range(len(old)):
        color = colors[k % len(colors)]
        ax.annotate("", xy=(new[k, 0], new[k, 1]), xytext=(old[k, 0], old[k, 1]),
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1.6})
        ax.plot([old[k, 0]], [old[k, 1]], "o", ms=11, color=color, markeredgecolor="white", zorder=3)
        ax.plot([new[k, 0]], [new[k, 1]], "o", ms=11, markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=2, zorder=3)
        ax.text(old[k, 0] + 0.07, old[k, 1] + 0.07, names[k], fontsize=12, color=_INK, fontweight="bold")
        ax.text(new[k, 0] + 0.07, new[k, 1] - 0.16, f"new {names[k]}", fontsize=9.5, color=_INK2)
    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=0)
    ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=0)
    ax.set_xlim(-1.6, 1.7)
    ax.set_ylim(-0.6, 1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("number 1 of each token", fontsize=9.5)
    ax.set_ylabel("number 2", fontsize=9.5)
    ax.set_title("each token moves towards the tokens it is similar to\n(filled: before, open: after attention)",
                 fontsize=10.5, color=_INK, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    _weights_axes(axw, w, names, numbers=True)
    axw.set_title("the weights: row = the token that looks,\ncolumn = the token it looks at (rows add up to 1)",
                  fontsize=10.5, color=_INK, loc="left")
    return _show(fig)


def _weights_axes(ax, w, names, numbers: bool):
    image = ax.imshow(w, cmap="Blues", vmin=0, vmax=max(float(w.max()), 1e-6), interpolation="nearest")
    n = w.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=8.5 if n > 6 else 11)
    ax.set_yticklabels(names, fontsize=8.5 if n > 6 else 11)
    if n > 6:
        for label in ax.get_xticklabels():
            label.set_rotation(90)
    if numbers:
        top = float(w.max())
        for i in range(n):
            for j in range(n):
                color = "white" if w[i, j] > 0.55 * top else _INK
                ax.text(j, i, f"{w[i, j]:.3f}", ha="center", va="center", fontsize=11, color=color)
    ax.tick_params(length=0)
    return image


def _token_names(n_tokens: int) -> list:
    names = ["class"]
    for k in range(n_tokens - 1):
        names.append(str(k))
    return names


def show_weights(weights, names=None, title: str | None = None):
    """An attention table as a heat map: row i = how token i spreads its attention over the tokens (columns)."""
    w = _numpy(weights)
    if w.ndim == 3:
        w = w[0]
    n = w.shape[0]
    names = names or _token_names(n)
    fig = _new_figure(6.4, 5.6)
    ax = fig.subplots()
    image = _weights_axes(ax, w, names, numbers=n <= 6)
    ax.set_xlabel("the token that is looked at", fontsize=9.5)
    ax.set_ylabel("the token that looks", fontsize=9.5)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.ax.tick_params(labelsize=8)
    bar.set_label("weight", fontsize=9)
    ax.set_title(title or f"attention weights, {n} x {n} (each row adds up to 1)", fontsize=10.5, color=_INK,
                 loc="left")
    return _show(fig)


def _patch_tokens(embedding: nn.Module, image) -> torch.Tensor:
    """The tokens of one picture without the position rows: class token, then the patches (1 x 17 x dim)."""
    batch = torch.as_tensor(image)
    if batch.dim() == 3:
        batch = batch.unsqueeze(0)
    x = embedding.to_tokens(batch)
    x = x.flatten(2)
    x = x.transpose(1, 2)
    class_tokens = embedding.class_token.expand(x.shape[0], -1, -1)
    x = torch.cat([class_tokens, x], dim=1)
    return x


@torch.no_grad()
def position_test(embedding: nn.Module, attention: nn.MultiheadAttention, image, shuffled) -> None:
    """Runs one attention layer on the digit and on its shuffled version, without and with the position rows,
    and compares what comes out for the class token."""
    plain_a = _patch_tokens(embedding, image)
    plain_b = _patch_tokens(embedding, shuffled)
    rows = [("without the position rows", plain_a, plain_b),
            ("with the position rows", plain_a + embedding.position, plain_b + embedding.position)]
    for words, a, b in rows:
        out_a, _ = attention(a, a, a)
        out_b, _ = attention(b, b, b)
        first_a = out_a[0, 0, :4].tolist()
        first_b = out_b[0, 0, :4].tolist()
        diff = float((out_a[0, 0] - out_b[0, 0]).abs().max())
        same = diff < 1e-5
        print(f"{words}:")
        print(f"    class token after attention, the digit:    {_fmt4(first_a)} ...")
        print(f"    class token after attention, shuffled:     {_fmt4(first_b)} ...")
        if same:
            print("    -> the same numbers: attention cannot tell the two pictures apart")
        else:
            print(f"    -> different (by up to {diff:.4f}): now the place of each patch counts")


def _fmt4(values) -> str:
    parts = []
    for v in values:
        parts.append(f"{v:+.4f}")
    return "[" + ", ".join(parts) + "]"


# ============================================================================================== section 5
@torch.no_grad()
def attention_maps(model: nn.Module, images: torch.Tensor) -> list:
    """The attention weights of every transformer block of a TinyViT, averaged over its heads.

    Returns one tensor per block, of shape batch x tokens x tokens (row = the token that looks).
    """
    was = model.training
    model.eval()
    try:
        x = model.embedding(images)
        maps = []
        for block in model.blocks:
            h = block.norm1(x) if block.norm_first else x
            _, w = block.self_attn(h, h, h, need_weights=True, average_attn_weights=True)
            maps.append(w)
            x = block(x)
    finally:
        model.train(was)
    return maps


def show_class_attention(model: nn.Module, loader, n: int = 8, block: int | None = None, classes=None):
    """For the first n validation pictures: the picture, and where the class token looks (block = which block;
    None = the average over all blocks), as a heat map over the patches."""
    ds = loader.dataset if hasattr(loader, "dataset") else loader
    images = []
    labels = []
    for k in range(n):
        image, label = ds[k]
        images.append(image)
        labels.append(int(label))
    batch = torch.stack(images)
    maps = attention_maps(model, batch)
    if block is None:
        total = maps[0]
        for m in maps[1:]:
            total = total + m
        chosen = total / len(maps)
        which = f"averaged over the {len(maps)} blocks"
    else:
        chosen = maps[block]
        which = f"block {block + 1 if block >= 0 else len(maps) + block + 1} of {len(maps)}"
    from .practical import predict
    preds = predict(model, batch)
    class_row = chosen[:, 0, 1:]                                     # the class token looking at the patches
    per_side = int(round(math.sqrt(class_row.shape[1])))
    fig = _new_figure(1.55 * n, 3.6)
    axes = fig.subplots(2, n, squeeze=False)
    for k in range(n):
        pic = _picture(images[k])
        size = pic.shape[0]
        axes[0, k].imshow(pic, cmap="gray")
        name = classes[preds[k]] if classes is not None else str(int(preds[k]))
        ok = int(preds[k]) == labels[k]
        axes[0, k].set_title(f"answer: {name}", fontsize=9.5, color="#008300" if ok else _RED)
        _clean(axes[0, k])
        heat = class_row[k].reshape(per_side, per_side).numpy()
        axes[1, k].imshow(heat, cmap="Blues", extent=(0, size, size, 0), interpolation="nearest", vmin=0,
                          vmax=float(heat.max()))
        axes[1, k].contour(np.arange(size) + 0.5, np.arange(size) + 0.5, pic, levels=[0.4], colors=_ORANGE,
                           linewidths=1.6)
        _clean(axes[1, k])
    axes[0, 0].set_ylabel("the picture", fontsize=9.5)
    axes[1, 0].set_ylabel("where the class\ntoken looks", fontsize=9.5)
    fig.suptitle(f"top: the picture and the network's answer\nbottom: the class token's attention to each patch, "
                 f"{which} (dark blue = large weight, orange = outline of the digit)", fontsize=10.5, color=_INK)
    return _show(fig)


@torch.no_grad()
def show_position_similarity(model: nn.Module):
    """For every patch place: how similar its learned position row is to the rows of all other places (cosine
    similarity, 1 = same direction), drawn as a small map in that place."""
    pos = model.embedding.position.detach()[0, 1:]                     # the rows of the patches (not the class token)
    length = pos.norm(dim=1, keepdim=True).clamp_min(1e-8)
    unit = pos / length
    sim = (unit @ unit.T).numpy()
    count = sim.shape[0]
    per_side = int(round(math.sqrt(count)))
    fig = _new_figure(5.6, 5.6)
    axes = fig.subplots(per_side, per_side, squeeze=False)
    image = None
    for k in range(count):
        ax = axes[k // per_side, k % per_side]
        image = ax.imshow(sim[k].reshape(per_side, per_side), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.plot([k % per_side], [k // per_side], marker="s", ms=4 if per_side > 4 else 6, markerfacecolor="none",
                markeredgecolor=_INK, markeredgewidth=1)
        _clean(ax)
    bar = fig.colorbar(image, ax=axes, fraction=0.04, pad=0.03)
    bar.ax.tick_params(labelsize=8)
    bar.set_label("similarity of the position rows", fontsize=9)
    fig.suptitle("one small map per patch place: how similar its position row is to\nthe position rows of all "
                 "places (black square = the place itself)", fontsize=10.5, color=_INK)
    return _show(fig)


# ============================================================================================== section 6
def load_pretrained_vit():
    """(vit, weights): torchvision's ViT-B/16 with its ImageNet weights if the file is in the torch hub cache;
    otherwise the same network with random weights and weights = None. Never downloads anything."""
    import torchvision
    from torchvision.models import ViT_B_16_Weights

    folder = Path(torch.hub.get_dir()) / "checkpoints"
    path = folder / VIT_FILE
    if path.exists():
        vit = torchvision.models.vit_b_16(weights=None)
        vit.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        vit.eval()
        size_mb = path.stat().st_size / 1_000_000
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        print(f"loaded ViT-B/16 with its ImageNet weights from {path} ({size_mb:.0f} MB)")
        top1 = weights.meta.get("_metrics", {}).get("ImageNet-1K", {}).get("acc@1")
        if top1 is not None:
            print(f"with these weights it names {top1:.1f}% of ImageNet's validation photos right "
                  f"(torchvision's measurement)")
        return vit, weights
    vit = torchvision.models.vit_b_16(weights=None)
    vit.eval()
    print(f"note: the trained weights of ViT-B/16 are not on this computer (looked for {path}).")
    print("      The network is built with random weights: its sizes and shapes below are right, but it cannot")
    print("      recognise anything, so the photos are not classified. To get the weights (about 330 MB), run")
    print("      torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1) once")
    print("      with an internet connection, or copy the file from a computer that has it into that folder.")
    return vit, None


def pet_photo(name: str) -> Path:
    """The path of one Oxford-IIIT Pet photo, e.g. pet_photo("pug_12") -> data/oxford-iiit-pet/images/pug_12.jpg."""
    from .torchkit import data_root
    return data_root() / "oxford-iiit-pet" / "images" / (name + ".jpg")


@torch.no_grad()
def classify_photos(vit: nn.Module, weights, names, top: int = 3):
    """Show a few pet photos with the pretrained ViT's `top` answers (ImageNet class names and probabilities)."""
    from PIL import Image

    if weights is None:
        print("no trained weights: skipping the classification (see the note above)")
        return None
    prepare = weights.transforms()
    categories = weights.meta["categories"]
    photos = []
    batch = []
    used = []
    for name in names:
        path = pet_photo(name)
        if not path.exists():
            print(f"photo not found: {path}")
            continue
        photo = Image.open(path).convert("RGB")
        photos.append(photo)
        batch.append(prepare(photo))
        used.append(name)
    if not batch:
        print("none of the photos was found: is data/oxford-iiit-pet/images there? (P5 uses it)")
        return None
    was = vit.training
    vit.eval()
    try:
        scores = vit(torch.stack(batch))
    finally:
        vit.train(was)
    probs = scores.softmax(dim=1)
    best_p, best_i = probs.topk(top, dim=1)
    fig = _new_figure(3.3 * len(photos), 4.5)
    axes = fig.subplots(1, len(photos), squeeze=False)[0]
    for k, ax in enumerate(axes):
        shown = prepare(photos[k])
        ax.imshow(_picture(shown))
        lines = []
        for j in range(top):
            lines.append(f"{100 * float(best_p[k, j]):.0f}%  {categories[int(best_i[k, j])]}")
        ax.set_title(f"file: {used[k]}.jpg", fontsize=9.5, color=_INK2)
        ax.set_xlabel("\n".join(lines), fontsize=10, color=_INK)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"the pretrained ViT-B/16: its {top} most likely ImageNet classes for each photo", fontsize=11,
                 color=_INK)
    for k in range(len(used)):
        answers = []
        for j in range(top):
            answers.append(f"{categories[int(best_i[k, j])]} {100 * float(best_p[k, j]):.1f}%")
        print(f"{used[k]}: " + ", ".join(answers))
    return _show(fig)
