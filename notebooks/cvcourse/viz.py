"""Plotting helpers used across the course notebooks (Matplotlib only)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# A small, colour-blind-friendly palette used for classes / boxes everywhere.
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2",
           "#FF9DA6", "#9D755D", "#BAB0AC", "#72B7B2", "#EECA3B"]


_LABEL_INDEX: dict = {}


def color(i) -> str:
    """Palette colour for an integer class id or any hashable label (strings get a stable slot)."""
    try:
        return PALETTE[int(i) % len(PALETTE)]
    except (TypeError, ValueError):
        idx = _LABEL_INDEX.setdefault(i, len(_LABEL_INDEX))
        return PALETTE[idx % len(PALETTE)]


def sample_photo(gray: bool = False) -> np.ndarray:
    """A real photograph that ships with Matplotlib (no download needed).

    Returns float32 array in [0, 1], shape (H, W, 3) or (H, W) if gray.
    """
    from matplotlib import cbook
    from PIL import Image

    with cbook.get_sample_data("grace_hopper.jpg") as f:
        img = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
    if gray:
        img = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return img


def show_images(images, titles=None, cols=None, size=3.0, cmap="gray",
                vmin=None, vmax=None, suptitle=None):
    """Show a list of images in a grid. Returns the figure."""
    n = len(images)
    cols = cols or min(n, 6)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k >= n:
            continue
        im = images[k]
        if im.ndim == 2:
            ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(np.clip(im, 0, 1))
        if titles is not None:
            ax.set_title(titles[k], fontsize=10)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def draw_boxes(ax, boxes, labels=None, scores=None, color_by_label=True,
               edgecolor=None, lw=1.5, fmt="xyxy", alpha=1.0):
    """Draw boxes (N, 4) on an axis. fmt: 'xyxy' | 'xywh' | 'cxcywh'."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    for i, b in enumerate(boxes):
        if fmt == "xyxy":
            x1, y1, x2, y2 = b
        elif fmt == "xywh":
            x1, y1, w, h = b
            x2, y2 = x1 + w, y1 + h
        elif fmt == "cxcywh":
            cx, cy, w, h = b
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:
            raise ValueError(fmt)
        c = edgecolor or (color(labels[i]) if (labels is not None and color_by_label) else PALETTE[0])
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=c, lw=lw, alpha=alpha))
        txt = []
        if labels is not None:
            txt.append(str(labels[i]))
        if scores is not None:
            txt.append(f"{scores[i]:.2f}")
        if txt:
            ax.text(x1, y1 - 2, " ".join(txt), color="white", fontsize=8,
                    bbox=dict(facecolor=c, edgecolor="none", pad=1, alpha=0.8))


def overlay_mask(ax, mask, alpha=0.45, cmap="tab10"):
    """Overlay an integer mask (H, W) on an axis; 0 is transparent."""
    m = np.ma.masked_where(mask == 0, mask)
    ax.imshow(m, cmap=cmap, alpha=alpha, interpolation="nearest", vmin=0, vmax=9)


def heatmap_overlay(ax, image, heat, alpha=0.5, cmap="jet"):
    """Overlay a (h, w) heat map (any size) on an image (H, W[, 3])."""
    ax.imshow(image if image.ndim == 3 else image, cmap=None if image.ndim == 3 else "gray")
    H, W = image.shape[:2]
    ax.imshow(heat, cmap=cmap, alpha=alpha, extent=(0, W, H, 0), interpolation="bilinear")
    ax.axis("off")


def annotate_grid(ax, arr, fmt="{:.0f}", fontsize=8, color="black"):
    """Write the value of every cell of a small 2-D array on top of an imshow."""
    for (i, j), v in np.ndenumerate(arr):
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize, color=color)


def plot_curves(history: dict, keys=("train_loss", "val_loss"), ax=None, title=None):
    """history: dict of lists. Plots the given keys against epoch."""
    ax = ax or plt.gca()
    for k in keys:
        if k in history:
            ax.plot(range(1, len(history[k]) + 1), history[k], marker="o", ms=3, label=k)
    ax.set_xlabel("epoch")
    ax.legend()
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax
