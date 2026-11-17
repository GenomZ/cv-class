"""P4 (U-Net, practical track): the reference U-Net, the segmentation data and the pictures of the notebook.

Import it in the notebook as::

    from cvcourse import pmodels_p4 as R

Reference blocks (the notebook's solution, one source of truth for the tests in ``ptests_p4``)
-----------------------------------------------------------------------------------------------
DoubleConv(in_channels, out_channels)                    conv 3x3 -> BatchNorm -> ReLU, twice; size kept
Down(in_channels, out_channels)                          MaxPool 2x2, then DoubleConv; size halved
Up(in_channels, skip_channels, out_channels, use_skip=True)
                                                         ConvTranspose 2x2 stride 2 (size doubled, channels
                                                         halved), torch.cat([skip, x], dim=1), DoubleConv
UNet(in_channels=3, n_classes=4, base=16, depth=4, use_skips=True)
                                                         inc, down1..down<depth>, up1..up<depth>, outc; with
                                                         depth=4 the same names and layers as the notebook's UNet

Data and pictures
-----------------
get_data(name="shapes", n_train=1200, n_val=200, batch_size=16, size=64) -> (train_loader, val_loader, classes)
show_examples(loader, n=6)                               images above their masks
show_predictions(model, loader, n=6, title=None)         image, true mask, predicted mask, wrong pixels
mean_iou(model, loader) -> dict(mean_iou, per_class, pixel_accuracy)
draw_unet(base=16, size=64, in_channels=3, n_classes=4)  the U-shaped map of the network
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from . import practical as P

__all__ = ["DoubleConv", "Down", "Up", "UNet", "get_data", "show_examples", "show_predictions", "mean_iou",
           "draw_unet", "MASK_COLORS"]

# mask colours, per data set: a neutral background, then hues from the toolkit's categorical palette chosen so
# that a shape's mask has the colour family of the shape itself (circles are drawn red, squares blue, triangles green)
MASK_COLORS = {
    "shapes": ["#e6e5e0", "#eb6834", "#2a78d6", "#1baf7a"],
    "pets": ["#e6e5e0", "#2a78d6", "#eda100"],
}
_WRONG = "#3d3c3a"
_INK = "#0b0b0b"
_INK2 = "#52514e"


# =============================================================================================== reference blocks
class DoubleConv(nn.Module):
    """conv 3x3 (padding 1) -> BatchNorm -> ReLU, twice. (N, in, H, W) -> (N, out, H, W)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Down(nn.Module):
    """MaxPool 2x2, then DoubleConv. (N, in, H, W) -> (N, out, H/2, W/2)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """ConvTranspose 2x2 stride 2 (in -> in // 2 maps, size doubled), glue the skip in front, DoubleConv.

    forward(x (N, in, h, w), skip (N, skip_channels, 2h, 2w)) -> (N, out, 2h, 2w).
    use_skip=False ignores the skip (the experiment "U-Net without skip connections").
    """

    def __init__(self, in_channels, skip_channels, out_channels, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        if use_skip:
            conv_in = in_channels // 2 + skip_channels
        else:
            conv_in = in_channels // 2
        self.conv = DoubleConv(conv_in, out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)
        if self.use_skip:
            x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """The U-Net of the notebook, with two extra settings for the experiments.

    depth: how many Down blocks (and as many Up blocks); 4 gives exactly the notebook's network
    (inc, down1..down4, up1..up4, outc). The input height and width must be divisible by 2 ** depth.
    use_skips=False: the Up blocks ignore the skip connections.
    """

    def __init__(self, in_channels=3, n_classes=4, base=16, depth=4, use_skips=True):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.depth = depth
        self.use_skips = use_skips
        self.inc = DoubleConv(in_channels, base)
        channels = base
        for level in range(1, depth + 1):
            setattr(self, "down" + str(level), Down(channels, channels * 2))
            channels = channels * 2
        for level in range(1, depth + 1):
            setattr(self, "up" + str(level), Up(channels, channels // 2, channels // 2, use_skip=use_skips))
            channels = channels // 2
        self.outc = nn.Conv2d(base, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.inc(x)
        skips = [x]                                         # x1, x2, ...: kept for the decoder
        for level in range(1, self.depth + 1):
            x = getattr(self, "down" + str(level))(x)
            skips.append(x)
        skips.pop()                                         # the bottleneck output is x itself, not a skip
        for level in range(1, self.depth + 1):
            skip = skips.pop()
            if not self.use_skips:
                skip = None                                 # nothing crosses the U (P.show_model then draws no arcs)
            x = getattr(self, "up" + str(level))(x, skip)
        return self.outc(x)


# =============================================================================================== data
SHAPES_CLASSES = ["background", "circle", "square", "triangle"]
PETS_CLASSES = ["background", "pet", "border"]


def _shapes(n_train: int, n_val: int, size: int):
    from .torchkit import ShapesDataset
    kw = dict(min_r=max(3, size * 5 // 64), max_r=max(6, size * 14 // 64))
    tr = ShapesDataset(n_train, size, seed=0, task="segmentation", **kw)
    va = ShapesDataset(n_val, size, seed=1, task="segmentation", **kw)
    zero, one = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    train_set = P.ImageSet(tr.images.clone(), tr.masks.long().clone(), SHAPES_CLASSES, zero, one, name="shapes")
    val_set = P.ImageSet(va.images.clone(), va.masks.long().clone(), SHAPES_CLASSES, zero, one, name="shapes")
    return train_set, val_set


def _pets(n_train: int, n_val: int, size: int, verbose: bool):
    from .torchkit import CHECKPOINTS, IMAGENET_MEAN, IMAGENET_STD, get_pet_datasets
    cache = CHECKPOINTS / f"p4_pets_{size}px_{n_train}_{n_val}.pt"
    if cache.exists():
        d = torch.load(cache)
    else:
        if verbose:
            print(f"reading {n_train + n_val} Oxford-IIIT Pet photos and their masks at {size} x {size} pixels "
                  f"(once; then cached in {cache.name})")
        tr, va, _ = get_pet_datasets(size=size, target="segmentation", subset=(n_train, n_val))
        d = {}
        for key, ds in (("train", tr), ("val", va)):
            xs, ys = [], []
            for i in range(len(ds)):
                x, y = ds[i]
                xs.append(x)
                ys.append(y)
            d[key] = (torch.stack(xs), torch.stack(ys))
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        torch.save(d, cache)
    train_set = P.ImageSet(d["train"][0], d["train"][1], PETS_CLASSES, IMAGENET_MEAN, IMAGENET_STD, name="pets")
    val_set = P.ImageSet(d["val"][0], d["val"][1], PETS_CLASSES, IMAGENET_MEAN, IMAGENET_STD, name="pets")
    return train_set, val_set


def get_data(name: str = "shapes", n_train: int = 1200, n_val: int = 200, batch_size: int = 16, size: int = 64,
             verbose: bool = True):
    """A segmentation data set -> (train_loader, val_loader, classes). Every item is (image, mask).

    name="shapes": generated pictures of circles, squares and triangles (no download), 4 classes:
                   background, circle, square, triangle. Images 3 x size x size, values 0 to 1.
    name="pets":   Oxford-IIIT Pet photos (data/oxford-iiit-pet, see tools/download_data.py), 3 classes:
                   background, pet, border (the outline band). Images normalised with the ImageNet mean / std.
    A mask is a size x size tensor of class numbers, one per pixel.
    """
    t0 = time.time()
    if name == "shapes":
        train_set, val_set = _shapes(n_train, n_val, size)
    elif name == "pets":
        train_set, val_set = _pets(n_train, n_val, size, verbose)
    else:
        raise ValueError(f'name must be "shapes" or "pets", not {name!r}')
    train_loader, val_loader = P.loaders(train_set, val_set, batch_size=batch_size)
    if verbose:
        x, y = train_set[0]
        print(f"{name}: {len(train_set)} training and {len(val_set)} validation images of "
              f"{' x '.join(str(v) for v in x.shape)}; masks of {' x '.join(str(v) for v in y.shape)} with the "
              f"classes {', '.join(str(k) + ' ' + c for k, c in enumerate(train_set.classes))} "
              f"({time.time() - t0:.1f} s)")
    return train_loader, val_loader, list(train_set.classes)


# =============================================================================================== evaluation
@torch.no_grad()
def _confusion(model: nn.Module, loader, n_classes: int) -> torch.Tensor:
    dev = P._model_device(model)
    was = model.training
    model.eval()
    cm = torch.zeros(n_classes * n_classes, dtype=torch.long)
    try:
        for x, y in loader:
            pred = model(x.to(dev)).argmax(1).cpu()
            y = y.long()
            cm += torch.bincount(n_classes * y.reshape(-1) + pred.reshape(-1), minlength=n_classes * n_classes)
    finally:
        model.train(was)
    return cm.view(n_classes, n_classes)


def mean_iou(model: nn.Module, loader, verbose: bool = True, name: str | None = None) -> dict:
    """IoU of every class over all validation pixels, their mean (the mean IoU), and the pixel accuracy.

    IoU of a class = pixels that are this class in both the prediction and the true mask, divided by the
    pixels that are this class in at least one of them.
    """
    classes = list(P._dataset_of(loader).classes)
    cm = _confusion(model, loader, len(classes)).double()
    inter = torch.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    per_class = {}
    values = []
    for k, c in enumerate(classes):
        if union[k] > 0:
            v = float(inter[k] / union[k])
            per_class[c] = v
            values.append(v)
    miou = float(np.mean(values)) if values else float("nan")
    acc = float(inter.sum() / cm.sum()) * 100
    who = f"{name}: " if name else ""
    parts = []
    for c, v in per_class.items():
        parts.append(f"{c} {v:.3f}")
    rep = P._Report(mean_iou=miou, per_class=per_class, pixel_accuracy=acc)
    rep.text = (f"{who}mean IoU {miou:.3f} on {len(P._dataset_of(loader))} validation images "
                f"(per class: {', '.join(parts)}); pixel accuracy {acc:.1f}%")
    if verbose:
        print(rep.text)
    return rep._mark()


# =============================================================================================== pictures
def _colors(ds) -> list:
    return MASK_COLORS.get(getattr(ds, "name", ""), MASK_COLORS["shapes"])


def _mask_rgb(mask: torch.Tensor, ds) -> np.ndarray:
    rgb = []
    for h in _colors(ds):
        rgb.append([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    rgb = np.array(rgb)
    m = np.clip(np.asarray(mask, dtype=np.int64), 0, len(rgb) - 1)
    return rgb[m]


def _wrong_rgb(pred: torch.Tensor, true: torch.Tensor) -> np.ndarray:
    bg = np.array([1.0, 1.0, 1.0])
    red = np.array([int(_WRONG[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    wrong = (np.asarray(pred) != np.asarray(true))
    out = np.empty(wrong.shape + (3,))
    out[...] = bg
    out[wrong] = red
    return out


def _image_rgb(img: torch.Tensor, ds) -> np.ndarray:
    return P._display_image(img, ds)


def _legend(fig, ds, extra=None):
    handles = []
    colors = _colors(ds)
    for k, c in enumerate(ds.classes):
        handles.append(Patch(facecolor=colors[k], edgecolor="#9a9893", label=f"{k} {c}"))
    if extra:
        handles.extend(extra)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.0))


def _first_items(loader, n: int):
    ds = P._dataset_of(loader)
    xs, ys = [], []
    for i in range(min(n, len(ds))):
        x, y = ds[i]
        xs.append(x)
        ys.append(y)
    return ds, torch.stack(xs), torch.stack(ys)


def _frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bbbbbb")
        s.set_linewidth(0.8)


def show_examples(loader, n: int = 6, title: str | None = None):
    """The first n images of a loader's data set above their masks (one colour per class)."""
    ds, xs, ys = _first_items(loader, n)
    n = len(xs)
    fig = P._new_figure(1.75 * n + 0.9, 4.3)
    axes = fig.subplots(2, n, squeeze=False)
    for j in range(n):
        axes[0, j].imshow(_image_rgb(xs[j], ds))
        axes[1, j].imshow(_mask_rgb(ys[j], ds), interpolation="nearest")
        _frame(axes[0, j])
        _frame(axes[1, j])
    axes[0, 0].set_ylabel("image", fontsize=10)
    axes[1, 0].set_ylabel("mask", fontsize=10)
    _legend(fig, ds)
    fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
    fig.suptitle(title or "the task: one class for every pixel", fontsize=11.5, color=_INK)
    return P._show(fig)


@torch.no_grad()
def show_predictions(model: nn.Module, loader, n: int = 6, title: str | None = None):
    """For the first n validation images: the image, the true mask, the network's mask, and its wrong pixels."""
    ds, xs, ys = _first_items(loader, n)
    n = len(xs)
    dev = P._model_device(model)
    was = model.training
    model.eval()
    try:
        preds = model(xs.to(dev)).argmax(1).cpu()
    finally:
        model.train(was)
    fig = P._new_figure(1.75 * n + 1.2, 7.9)
    axes = fig.subplots(4, n, squeeze=False)
    for j in range(n):
        n_wrong = int((preds[j] != ys[j]).sum())
        axes[0, j].imshow(_image_rgb(xs[j], ds))
        axes[1, j].imshow(_mask_rgb(ys[j], ds), interpolation="nearest")
        axes[2, j].imshow(_mask_rgb(preds[j], ds), interpolation="nearest")
        axes[3, j].imshow(_wrong_rgb(preds[j], ys[j]), interpolation="nearest")
        axes[3, j].set_title(f"{n_wrong} wrong", fontsize=9, color=_INK2)
        for r in range(4):
            _frame(axes[r, j])
    for r, lab in enumerate(["image", "true mask", "predicted", "wrong pixels"]):
        axes[r, 0].set_ylabel(lab, fontsize=10)
    extra = [Patch(facecolor=_WRONG, edgecolor=_WRONG, label="wrong pixel")]
    _legend(fig, ds, extra)
    fig.get_layout_engine().set(rect=(0, 0.045, 1, 0.95))
    fig.suptitle(title or "what the network predicts on validation images", fontsize=11.5, color=_INK)
    return P._show(fig)


def _box(ax, x, y, w, h, face, edge, lines, bold_first=True):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=face, edgecolor=edge, linewidth=1.4, zorder=3))
    n = len(lines)
    for k, line in enumerate(lines):
        yy = y + (n - 1) * 0.105 - k * 0.21
        ax.text(x, yy, line, ha="center", va="center", fontsize=9.3 if k == 0 else 8.6, color=_INK,
                fontweight="bold" if (k == 0 and bold_first) else "normal", zorder=4)


def _arrow(ax, p0, p1, color, lw=1.6, style="-|>", ls="-", z=2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, linestyle=ls, zorder=z))


def _tint(hex_color: str, a: float) -> tuple:
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    out = []
    for v in rgb:
        out.append(1 - a * (1 - v))
    return tuple(out)


def draw_unet(base: int = 16, size: int = 64, in_channels: int = 3, n_classes: int = 4, depth: int = 4):
    """The U-Net as a U: encoder down the left, decoder up the right, skip connections across.

    Every box is one block of the notebook's UNet with the shape of its output (maps x height x width).
    """
    enc_c, dec_c, skip_c, pool_c, up_c = "#2a78d6", "#1baf7a", "#e34948", "#eb6834", "#1baf7a"
    dx, dy, w, h = 1.32, 1.05, 1.18, 0.66
    fig = P._new_figure(12.4, 1.15 * depth + 2.6, constrained=False)
    ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))
    ax.axis("off")
    names_enc = ["inc"]
    for k in range(1, depth + 1):
        names_enc.append("down" + str(k))
    pos = {}
    for lvl in range(depth + 1):
        ch, s = base * 2 ** lvl, size // 2 ** lvl
        x, y = lvl * dx, -lvl * dy
        if lvl == depth:
            x = depth * dx
        kind = "DoubleConv" if lvl == 0 else "Down"
        face = _tint(pool_c if lvl == depth else enc_c, 0.16)
        edge = pool_c if lvl == depth else enc_c
        label3 = f"x{lvl + 1}" + ("  (bottleneck)" if lvl == depth else "")
        _box(ax, x, y, w, h, face, edge, [names_enc[lvl], f"{ch} × {s} × {s}", label3])
        pos[names_enc[lvl]] = (x, y)
    for k in range(1, depth + 1):
        lvl = depth - k
        ch, s = base * 2 ** lvl, size // 2 ** lvl
        x, y = (2 * depth - lvl) * dx, -lvl * dy
        _box(ax, x, y, w, h, _tint(dec_c, 0.16), dec_c, ["up" + str(k), f"{ch} × {s} × {s}", "Up"])
        pos["up" + str(k)] = (x, y)
    # encoder arrows (max-pool) and decoder arrows (transposed convolution)
    for lvl in range(depth):
        x0, y0 = pos[names_enc[lvl]]
        x1, y1 = pos[names_enc[lvl + 1]]
        _arrow(ax, (x0 + 0.18, y0 - h / 2), (x1 - 0.28, y1 + h / 2), pool_c)
        ax.text((x0 + x1) / 2 - 0.42, (y0 + y1) / 2, "MaxPool\nsize ÷ 2", ha="right", va="center", fontsize=8,
                color=_INK2)
    prev = names_enc[depth]
    for k in range(1, depth + 1):
        x0, y0 = pos[prev]
        x1, y1 = pos["up" + str(k)]
        _arrow(ax, (x0 + 0.28, y0 + h / 2), (x1 - 0.18, y1 - h / 2), up_c)
        ax.text((x0 + x1) / 2 + 0.42, (y0 + y1) / 2, "ConvTranspose\nsize × 2", ha="left", va="center",
                fontsize=8, color=_INK2)
        prev = "up" + str(k)
    # skip connections
    for lvl in range(depth):
        x0, y0 = pos[names_enc[lvl]]
        x1, y1 = pos["up" + str(depth - lvl)]
        ch = base * 2 ** lvl
        _arrow(ax, (x0 + w / 2, y0), (x1 - w / 2, y1), skip_c, lw=1.8, ls=(0, (5, 3)))
        if lvl == 0:
            text = f"skip connection: x{lvl + 1} ({ch} maps) is copied across and glued onto the decoder's maps"
        else:
            text = f"skip: x{lvl + 1} ({ch} maps)"
        ax.text((x0 + x1) / 2, y0 + 0.09, text, ha="center", va="bottom", fontsize=8.4, color=_INK)
    # input and output
    xi, yi = pos["inc"]
    ax.text(xi, yi + h / 2 + 0.62, f"input image\n{in_channels} × {size} × {size}", ha="center", va="center",
            fontsize=9, color=_INK)
    _arrow(ax, (xi, yi + h / 2 + 0.38), (xi, yi + h / 2), _INK2, lw=1.2)
    xo, yo = pos["up" + str(depth)]
    xc = xo + dx * 0.95
    _box(ax, xc, yo, w * 0.82, h, _tint(enc_c, 0.16), enc_c, ["outc", "Conv 1×1", f"{n_classes} × {size} × {size}"])
    _arrow(ax, (xo + w / 2, yo), (xc - w * 0.41, yo), _INK2, lw=1.2)
    ax.text(xc, yo + h / 2 + 0.62, f"output: {n_classes} scores\nfor every pixel", ha="center", va="center",
            fontsize=9, color=_INK)
    _arrow(ax, (xc, yo + h / 2), (xc, yo + h / 2 + 0.38), _INK2, lw=1.2)
    # the three ideas
    yb = -depth * dy
    ax.text(0, yb + 0.1, "ENCODER\nthe image shrinks, the maps grow:\nit learns WHAT is in the image",
            ha="left", va="center", fontsize=9.5, color=_INK, fontweight="bold")
    ax.text(2 * depth * dx + dx * 0.95 + w * 0.41, yb + 0.1,
            "DECODER\nthe image grows back:\nit restores WHERE things are", ha="right", va="center",
            fontsize=9.5, color=_INK, fontweight="bold")
    ax.set_xlim(-w / 2 - 0.1, 2 * depth * dx + dx * 0.95 + w * 0.41 + 0.1)
    ax.set_ylim(yb - h / 2 - 0.15, h / 2 + 1.05)
    ax.set_aspect("equal")
    fig.set_size_inches(12.4, 12.4 * (h + 1.2 + depth * dy) / (2 * depth * dx + dx * 0.95 + w * 0.91 + 0.2))
    return P._show(fig)
