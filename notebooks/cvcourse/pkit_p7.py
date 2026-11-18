"""Pictures, training loops and drawings for P7 (object detection with boxes).

In the notebook: ``from cvcourse import pkit_p7 as D``. Everything that is plumbing lives here, so that the
notebook reads as text, short calls and the few cells the students write.

Boxes are (x1, y1, x2, y2) in pixels everywhere: x to the right, y DOWNWARDS, (x1, y1) the top-left corner and
(x2, y2) the bottom-right corner; width = x2 - x1, height = y2 - y1 (no "+1"). A box (2, 3, 6, 5) covers the
pixel columns 2, 3, 4, 5 and the pixel rows 3, 4: 4 x 2 = 8 pixels.

    D.CLASSES                                        ["ellipse", "rectangle", "triangle"]
    picture, boxes, labels = D.example_picture()     one 64 x 64 picture with three shapes
    D.show_boxes(picture, boxes, labels, scores=None, title=None)
    D.show_iou(box_a, box_b)                         the two boxes, their intersection and their union
    D.show_iou_pairs(box_iou_one)                    four pairs of boxes with the IoU a function gives

    one object per picture (section 3)
    train_set = D.one_object_set(n, seed)            .images (n, 3, 64, 64), .labels (n,), .boxes (n, 4)
    D.show_set(train_set, n=8)
    history = D.train_detector(model, train_set, val_set, epochs, lr, box_weight)
    D.plot_detector_history(history)
    D.evaluate_detector(model, val_set)              {"accuracy": %, "mean_iou": 0..1, "found": %}
    D.show_detector_predictions(model, val_set, n=12)

    several objects per picture (section 4)
    many_set = D.many_objects_set(n, seed)           .images, .boxes (list of (k, 4)), .labels (list of (k,))
    D.show_grid_idea(picture, boxes, labels)
    history = D.train_grid_detector(net, train_set, val_set, epochs, lr)     also draws D.plot_grid_history
    boxes, scores, labels = D.grid_candidates(net, picture)     one box per cell: 64 boxes
    D.show_before_after(picture, boxes, labels, scores, keep, before_title, after_title)
    D.show_grid_results(net, val_set, score_threshold, iou_threshold, n=8)

    a pretrained detector on photos (section 5)
    detector, categories = D.load_coco_detector()   Faster R-CNN with COCO weights (None if the file is missing)
    photo, true_boxes = D.load_photo("street", index)   Penn-Fudan street photos, or "pets": Oxford pets
    result = D.detect(detector, photo)               {"boxes", "labels", "scores"}
    D.print_result(result, categories, n=8)
    D.show_photo_detections(photo, result, categories, score_threshold, true_boxes)
"""
from __future__ import annotations

import colorsys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .practical import _new_figure, _Report, _show

__all__ = [
    "CLASSES", "SIZE", "GRID", "CELL", "example_picture", "show_boxes", "show_iou", "show_iou_pairs",
    "one_object_set", "many_objects_set", "show_set", "train_detector", "evaluate_detector",
    "plot_detector_history", "show_detector_predictions", "show_grid_idea", "grid_targets",
    "train_grid_detector", "plot_grid_history", "grid_candidates", "final_detections", "evaluate_grid", "show_before_after",
    "show_grid_results", "load_coco_detector", "load_photo", "detect", "print_result", "show_photo_detections",
    "box_iou", "OneObjectSet", "ManyObjectsSet",
]

CLASSES = ["ellipse", "rectangle", "triangle"]
SIZE = 64                  # pictures are SIZE x SIZE pixels
GRID = 8                   # the grid detector of section 4: GRID x GRID cells
CELL = SIZE // GRID        # 8 pixels per cell
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]         # ellipse blue, rectangle orange, triangle green
TRUE_COLOR = "#0b0b0b"
PRED_COLOR = "#e34948"
GOOD = "#008300"
BAD = "#e34948"
INK = "#0b0b0b"
INK2 = "#52514e"


# ============================================================================================== pictures
def _shape_mask(kind: int, cx: float, cy: float, rx: float, ry: float, size: int) -> np.ndarray:
    """True for every pixel whose centre lies inside the shape."""
    yy, xx = np.mgrid[0:size, 0:size] + 0.5
    if kind == 0:                                            # ellipse
        return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    if kind == 1:                                            # rectangle
        return (np.abs(xx - cx) <= rx) & (np.abs(yy - cy) <= ry)
    t = (yy - (cy - ry)) / (2.0 * ry)                        # triangle, apex up: 0 at the apex, 1 at the base
    return (t >= 0) & (t <= 1) & (np.abs(xx - cx) <= rx * t)


def _mask_box(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _background(rng: np.random.Generator, size: int) -> np.ndarray:
    grey = rng.uniform(0.72, 0.95)
    canvas = np.full((size, size, 3), grey, dtype=np.float32)
    canvas += rng.normal(0.0, 0.035, canvas.shape).astype(np.float32)
    return canvas


def _random_color(rng: np.random.Generator) -> np.ndarray:
    """Any hue (the colour says nothing about the class), always darker than the background."""
    r, g, b = colorsys.hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.5, 1.0), rng.uniform(0.25, 0.7))
    return np.array([r, g, b], dtype=np.float32)


def _overlaps(a, b, margin: int = 2) -> bool:
    return not (a[2] + margin <= b[0] or b[2] + margin <= a[0] or a[3] + margin <= b[1] or b[3] + margin <= a[1])


def _make_picture(rng: np.random.Generator, n_objects: int, r_min: int, r_max: int, size: int = SIZE):
    """One picture with n_objects shapes that do not overlap. Returns (picture (H, W, 3), boxes, labels)."""
    canvas = _background(rng, size)
    boxes = []
    labels = []
    tries = 0
    while len(boxes) < n_objects and tries < 200:
        tries += 1
        kind = int(rng.integers(0, len(CLASSES)))
        rx = float(rng.uniform(r_min, r_max))
        ry = float(rng.uniform(r_min, r_max))
        cx = float(rng.uniform(rx + 1, size - rx - 1))
        cy = float(rng.uniform(ry + 1, size - ry - 1))
        mask = _shape_mask(kind, cx, cy, rx, ry, size)
        box = _mask_box(mask)
        if box is None:
            continue
        clash = False
        for other in boxes:
            if _overlaps(box, other):
                clash = True
        if clash:
            continue
        noise = rng.normal(0.0, 0.03, 3).astype(np.float32)
        canvas[mask] = np.clip(_random_color(rng) + noise, 0, 1)
        boxes.append(box)
        labels.append(kind)
    return np.clip(canvas, 0, 1), boxes, labels


def _to_tensor(picture: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(picture.transpose(2, 0, 1))).float()


def example_picture(seed: int = 7):
    """The picture of section 1: 64 x 64 pixels, three shapes. Returns (picture (3, 64, 64), boxes, labels).

    boxes is a list of [x1, y1, x2, y2] (whole numbers, pixels), labels a list of class numbers (0, 1, 2).
    """
    rng = np.random.default_rng(seed)
    while True:
        picture, boxes, labels = _make_picture(rng, 3, 6, 15)
        if len(boxes) == 3 and len(set(labels)) == 3:
            return _to_tensor(picture), boxes, labels


class OneObjectSet:
    """Pictures with exactly one shape each: images (n, 3, 64, 64) in 0..1, labels (n,), boxes (n, 4) in pixels."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, boxes: torch.Tensor):
        self.images = images
        self.labels = labels
        self.boxes = boxes
        self.classes = list(CLASSES)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.labels[i], self.boxes[i]

    def __repr__(self) -> str:
        return f"{len(self)} pictures of 3 x {SIZE} x {SIZE}, one shape each"


class ManyObjectsSet:
    """Pictures with 1 to 3 shapes: images (n, 3, 64, 64); boxes[i] (k, 4) pixels and labels[i] (k,) of picture i."""

    def __init__(self, images: torch.Tensor, boxes: list, labels: list):
        self.images = images
        self.boxes = boxes
        self.labels = labels
        self.classes = list(CLASSES)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.boxes[i], self.labels[i]

    def __repr__(self) -> str:
        total = 0
        for b in self.boxes:
            total += len(b)
        return f"{len(self)} pictures of 3 x {SIZE} x {SIZE}, {total} shapes in total"


def one_object_set(n: int = 4000, seed: int = 0) -> OneObjectSet:
    """n pictures, each with ONE shape of random class, colour, size (10 to 40 pixels) and position."""
    rng = np.random.default_rng(1000 + seed)
    images = torch.zeros(n, 3, SIZE, SIZE)
    labels = torch.zeros(n, dtype=torch.int64)
    boxes = torch.zeros(n, 4)
    for i in range(n):
        picture, bxs, lbs = _make_picture(rng, 1, 5, 20)
        images[i] = _to_tensor(picture)
        labels[i] = lbs[0]
        boxes[i] = torch.tensor(bxs[0], dtype=torch.float32)
    return OneObjectSet(images, labels, boxes)


def many_objects_set(n: int = 4000, seed: int = 0) -> ManyObjectsSet:
    """n pictures with 1, 2 or 3 shapes each (10 to 28 pixels), never overlapping."""
    rng = np.random.default_rng(2000 + seed)
    images = torch.zeros(n, 3, SIZE, SIZE)
    boxes = []
    labels = []
    for i in range(n):
        k = int(rng.integers(1, 4))
        picture, bxs, lbs = _make_picture(rng, k, 5, 14)
        images[i] = _to_tensor(picture)
        boxes.append(torch.tensor(bxs, dtype=torch.float32).reshape(-1, 4))
        labels.append(torch.tensor(lbs, dtype=torch.int64))
    return ManyObjectsSet(images, boxes, labels)


# ============================================================================================== IoU (library side)
def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """IoU of aligned pairs: boxes_a[i] with boxes_b[i], both (N, 4) xyxy. Returns (N,)."""
    left = torch.maximum(boxes_a[:, 0], boxes_b[:, 0])
    top = torch.maximum(boxes_a[:, 1], boxes_b[:, 1])
    right = torch.minimum(boxes_a[:, 2], boxes_b[:, 2])
    bottom = torch.minimum(boxes_a[:, 3], boxes_b[:, 3])
    inter = (right - left).clamp(min=0) * (bottom - top).clamp(min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-9)


# ============================================================================================== drawing basics
def _as_list(boxes) -> list:
    if boxes is None:
        return []
    if torch.is_tensor(boxes):
        return boxes.detach().cpu().reshape(-1, 4).tolist()
    if isinstance(boxes, np.ndarray):
        return boxes.reshape(-1, 4).tolist()
    out = []
    for b in boxes:
        if torch.is_tensor(b):
            b = b.tolist()
        out.append(list(b))
    return out


def _as_numbers(values) -> list:
    if values is None:
        return []
    if torch.is_tensor(values):
        return values.detach().cpu().reshape(-1).tolist()
    if isinstance(values, np.ndarray):
        return values.reshape(-1).tolist()
    return list(values)


def _picture_hwc(picture) -> np.ndarray:
    if torch.is_tensor(picture):
        picture = picture.detach().cpu().float()
        if picture.dim() == 3 and picture.shape[0] in (1, 3):
            picture = picture.permute(1, 2, 0)
        picture = picture.numpy()
    picture = np.asarray(picture, dtype=np.float32)
    if picture.ndim == 3 and picture.shape[2] == 1:
        picture = picture[:, :, 0]
    return np.clip(picture, 0, 1)


def _imshow(ax, picture, axes: bool = False):
    img = _picture_hwc(picture)
    h, w = img.shape[:2]
    ax.imshow(img, extent=(0, w, h, 0), interpolation="nearest", cmap="gray" if img.ndim == 2 else None)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    if axes:
        step = 8 if w <= 64 else (16 if w <= 128 else 100)
        ax.set_xticks(list(range(0, w + 1, step)))
        ax.set_yticks(list(range(0, h + 1, step)))
        ax.tick_params(labelsize=7, length=2, colors=INK2)
        ax.xaxis.tick_top()
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color("#bbbbbb")


def _box(ax, box, color, lw=2.0, ls="-", text=None, alpha=1.0, fontsize=7.5, labels=None):
    """One rectangle; its text tag goes through `labels` (a _Labels) when given, so that tags do not overlap."""
    from matplotlib.patches import Rectangle
    x1, y1, x2, y2 = [float(v) for v in box]
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=lw, ls=ls, alpha=alpha))
    if text:
        if labels is not None:
            labels.add((x1, y1, x2, y2), text, color, alpha)
        else:
            ax.text(x1, y1, text, color="white", fontsize=fontsize, va="bottom", ha="left", alpha=max(alpha, 0.6),
                    bbox=dict(fc=color, ec="none", pad=1.0, alpha=0.85 * max(alpha, 0.6)), clip_on=True)


class _Labels:
    """Places the text tags of boxes so that they do not cover each other (best box first).

    A tag goes above the box's top-left corner if there is room, otherwise just inside the box, one tag height
    lower each time, until it finds a free place. Sizes are estimated from the font size and the axes' width.
    """

    def __init__(self, ax, axes_width_in: float, data_w: float, data_h: float, fontsize: float = 7.5):
        self.ax = ax
        self.unit = data_w / (axes_width_in * 72.0)            # data units per point
        self.data_w = data_w
        self.data_h = data_h
        self.fontsize = fontsize
        self.placed = []

    def _free(self, x, y, w, h) -> bool:
        if y < 0 or y + h > self.data_h:
            return False
        for px, py, pw, ph in self.placed:
            if x < px + pw and px < x + w and y < py + ph and py < y + h:
                return False
        return True

    def add(self, box, text, color, alpha=1.0):
        x1, y1, x2, y2 = box
        w = (len(text) * 0.6 * self.fontsize + 3) * self.unit
        h = self.fontsize * 1.5 * self.unit
        x = min(max(x1, 0.0), max(self.data_w - w, 0.0))
        tops = [y1 - h]
        k = 0
        while k < 12:
            tops.append(y1 + k * h)
            k += 1
        top = None
        for t in tops:
            if self._free(x, t, w, h):
                top = t
                break
        if top is None:
            top = min(max(y1 - h, 0.0), self.data_h - h)
        self.placed.append((x, top, w, h))
        self.ax.text(x + 1.5 * self.unit, top + 0.12 * h, text, color="white", fontsize=self.fontsize, va="top",
                     ha="left", alpha=max(alpha, 0.6), clip_on=True,
                     bbox=dict(fc=color, ec="none", pad=0.8, alpha=0.88 * max(alpha, 0.6)))


def _name(label: int, names=None) -> str:
    names = names or CLASSES
    if 0 <= int(label) < len(names):
        return str(names[int(label)])
    return str(int(label))


def show_boxes(picture, boxes, labels=None, scores=None, title=None, axes: bool = True, names=None):
    """Draw a picture and its boxes, each with its class name (and score, if given)."""
    box_list = _as_list(boxes)
    label_list = _as_numbers(labels)
    score_list = _as_numbers(scores)
    img = _picture_hwc(picture)
    h, w = img.shape[:2]
    scale = 4.6 if max(h, w) <= 64 else 7.0
    fig_w = scale * w / max(h, w) + 0.3
    fig = _new_figure(fig_w, scale * h / max(h, w) + 0.6)
    ax = fig.subplots()
    _imshow(ax, picture, axes=axes)
    tags = _Labels(ax, fig_w - 0.5, w, h, 8.0)
    for k in range(len(box_list)):
        text = None
        color = PRED_COLOR
        if k < len(label_list):
            color = CLASS_COLORS[int(label_list[k]) % len(CLASS_COLORS)] if names is None else PRED_COLOR
            text = _name(label_list[k], names)
        if k < len(score_list):
            text = (text + " " if text else "") + f"{score_list[k]:.2f}"
        _box(ax, box_list[k], color, lw=2.0, text=text, labels=tags)
    if title:
        ax.set_title(title, fontsize=10, loc="left", color=INK)
    return _show(fig)


# ============================================================================================== IoU drawings
def show_iou(box_a, box_b, title: str | None = None):
    """Two boxes on a grid of pixels: their intersection (hatched) and their union (shaded), with the areas."""
    from matplotlib.patches import Rectangle

    a = [float(v) for v in box_a]
    b = [float(v) for v in box_b]
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    w = max(0.0, right - left)
    h = max(0.0, bottom - top)
    inter = w * h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    xmax = int(max(a[2], b[2])) + 2
    ymax = int(max(a[3], b[3])) + 2

    fig = _new_figure(9.6, 3.9)
    axs = fig.subplots(1, 2)
    panels = [("intersection: the overlap", "inter"), ("union: everything covered by A or B", "union")]
    for ax, (name, which) in zip(axs, panels):
        if which == "union":
            ax.add_patch(Rectangle((a[0], a[1]), a[2] - a[0], a[3] - a[1], fc="#eda100", ec="none", alpha=0.35))
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fc="#eda100", ec="none", alpha=0.35))
        if which == "inter" and w > 0 and h > 0:
            ax.add_patch(Rectangle((left, top), w, h, fc="#6250d6", ec="#6250d6", alpha=0.45, hatch="///"))
        ax.add_patch(Rectangle((a[0], a[1]), a[2] - a[0], a[3] - a[1], fill=False, ec=CLASS_COLORS[0], lw=2.5))
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=CLASS_COLORS[1], lw=2.5))
        ax.text(a[0] + 0.2, a[1] + 0.2, "A", color=CLASS_COLORS[0], fontsize=11, fontweight="bold", va="top")
        ax.text(b[2] - 0.2, b[3] - 0.2, "B", color=CLASS_COLORS[1], fontsize=11, fontweight="bold", va="bottom",
                ha="right")
        ax.set_xlim(0, xmax)
        ax.set_ylim(ymax, 0)
        ax.set_aspect("equal")
        ax.set_xticks(list(range(0, xmax + 1)))
        ax.set_yticks(list(range(0, ymax + 1)))
        ax.tick_params(labelsize=7, length=2, colors=INK2)
        ax.xaxis.tick_top()
        ax.grid(True, color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
        if which == "inter":
            text = f"{name}\n{w:g} x {h:g} = {inter:g} pixels"
        else:
            text = f"{name}\n{area_a:g} + {area_b:g} - {inter:g} = {union:g} pixels"
        ax.set_title(text, fontsize=9.5, loc="left", color=INK)
    fig.suptitle(title or f"IoU = intersection / union = {inter:g} / {union:g} = {iou:.3f}", fontsize=11, x=0.02,
                 ha="left", color=INK)
    return _show(fig)


IOU_PAIRS = [
    ("the same box", (16, 16, 48, 48), (16, 16, 48, 48)),
    ("moved by half its width", (8, 16, 40, 48), (24, 16, 56, 48)),
    ("a small box inside a big one", (8, 8, 56, 56), (24, 24, 40, 40)),
    ("apart", (4, 8, 24, 28), (36, 32, 60, 56)),
]


def show_iou_pairs(iou_function, title: str | None = None):
    """Four pairs of boxes, each with the IoU that iou_function(box_a, box_b) gives (and the right value)."""
    from matplotlib.patches import Rectangle

    fig = _new_figure(11.0, 3.3)
    axs = fig.subplots(1, len(IOU_PAIRS))
    for ax, (name, a, b) in zip(axs, IOU_PAIRS):
        right_value = float(box_iou(torch.tensor([a], dtype=torch.float32), torch.tensor([b], dtype=torch.float32))[0])
        try:
            yours = float(iou_function(a, b))
            text = f"IoU = {yours:.2f}"
            ok = abs(yours - right_value) < 1e-6
        except Exception as e:  # noqa: BLE001  (the student's function may not be finished)
            text = f"error: {type(e).__name__}"
            ok = False
        ax.add_patch(Rectangle((0, 0), SIZE, SIZE, fc="#f4f3ef", ec="none"))
        left = max(a[0], b[0])
        top = max(a[1], b[1])
        right = min(a[2], b[2])
        bottom = min(a[3], b[3])
        if right > left and bottom > top:
            ax.add_patch(Rectangle((left, top), right - left, bottom - top, fc="#6250d6", alpha=0.35, ec="none"))
        ax.add_patch(Rectangle((a[0], a[1]), a[2] - a[0], a[3] - a[1], fill=False, ec=CLASS_COLORS[0], lw=2.5))
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=CLASS_COLORS[1], lw=2.5,
                               ls="--"))
        ax.set_xlim(0, SIZE)
        ax.set_ylim(SIZE, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if not ok and not text.startswith("error"):
            text += f"  (expected {right_value:.2f})"
        ax.set_title(f"{name}\n{text}", fontsize=9, loc="left", color=GOOD if ok else BAD)
    fig.suptitle(title or "box A (blue), box B (orange, dashed), their intersection (purple)", fontsize=10,
                 x=0.02, ha="left", color=INK)
    return _show(fig)


# ============================================================================================== section 3: one object
def show_set(dataset, n: int = 8, cols: int = 8, title: str | None = None):
    """The first n pictures of a dataset with their true boxes and class names."""
    rows = (n + cols - 1) // cols
    fig = _new_figure(1.45 * cols, 1.6 * rows + 0.3)
    axs = np.atleast_1d(fig.subplots(rows, cols)).reshape(-1)
    for k, ax in enumerate(axs):
        if k >= n or k >= len(dataset):
            ax.axis("off")
            continue
        _imshow(ax, dataset.images[k])
        if isinstance(dataset, OneObjectSet):
            box_list = [dataset.boxes[k].tolist()]
            label_list = [int(dataset.labels[k])]
        else:
            box_list = dataset.boxes[k].tolist()
            label_list = dataset.labels[k].tolist()
        names = []
        for box, label in zip(box_list, label_list):
            _box(ax, box, CLASS_COLORS[label], lw=1.6)
            names.append(CLASSES[label])
        ax.set_title(", ".join(names), fontsize=7.5, color=INK)
    if title:
        fig.suptitle(title, fontsize=10, x=0.01, ha="left", color=INK)
    return _show(fig)


def _batches(n: int, batch_size: int, generator: torch.Generator):
    order = torch.randperm(n, generator=generator)
    out = []
    for start in range(0, n, batch_size):
        out.append(order[start:start + batch_size])
    return out


@torch.no_grad()
def _predict_one_object(model: nn.Module, images: torch.Tensor, batch_size: int = 256):
    model.eval()
    all_scores = []
    all_boxes = []
    for start in range(0, len(images), batch_size):
        scores, boxes = model(images[start:start + batch_size])
        all_scores.append(scores)
        all_boxes.append(boxes)
    return torch.cat(all_scores), torch.cat(all_boxes) * SIZE


def evaluate_detector(model: nn.Module, dataset: OneObjectSet, verbose: bool = True) -> dict:
    """accuracy (% right class), mean IoU of predicted and true boxes, found (% right class AND IoU >= 0.5)."""
    scores, boxes = _predict_one_object(model, dataset.images)
    predicted = scores.argmax(dim=1)
    right = predicted == dataset.labels
    ious = box_iou(boxes, dataset.boxes)
    found = right & (ious >= 0.5)
    result = _Report({
        "accuracy": round(100.0 * float(right.float().mean()), 1),
        "mean_iou": round(float(ious.mean()), 3),
        "found": round(100.0 * float(found.float().mean()), 1),
    })
    result.text = (f"{len(dataset)} validation pictures: class right {result['accuracy']:.1f}%, mean IoU "
                   f"{result['mean_iou']:.3f}, found (right class and IoU >= 0.5) {result['found']:.1f}%")
    if verbose:
        print(result.text)
    return result._mark()


def train_detector(model: nn.Module, train_set: OneObjectSet, val_set: OneObjectSet, epochs: int = 8,
                   lr: float = 0.002, box_weight: float = 5.0, batch_size: int = 64, seed: int = 0):
    """Train a model that returns (class_scores, box) with box in 0..1 (x1/64, y1/64, x2/64, y2/64).

    loss = cross-entropy(class scores, true class) + box_weight x L1(predicted box, true box / 64).
    Adam optimiser; prints one line per epoch; draws the curves at the end. Returns the history (a dict).
    """
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"epoch": [], "class_loss": [], "box_loss": [], "loss": [], "accuracy": [], "mean_iou": [],
               "found": [], "seconds": []}
    start_time = time.time()
    print(f"training on {len(train_set)} pictures, {epochs} epochs, learning rate {lr}, box weight {box_weight}")
    for epoch in range(1, epochs + 1):
        model.train()
        sum_class = 0.0
        sum_box = 0.0
        batches = _batches(len(train_set), batch_size, generator)
        for idx in batches:
            images = train_set.images[idx]
            labels = train_set.labels[idx]
            true_boxes = train_set.boxes[idx] / SIZE                     # pixels -> 0..1
            class_scores, boxes = model(images)
            class_loss = F.cross_entropy(class_scores, labels)
            box_loss = F.l1_loss(boxes, true_boxes)
            loss = class_loss + box_weight * box_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_class += class_loss.item()
            sum_box += box_loss.item()
        val = evaluate_detector(model, val_set, verbose=False)
        history["epoch"].append(epoch)
        history["class_loss"].append(sum_class / len(batches))
        history["box_loss"].append(sum_box / len(batches))
        history["loss"].append(history["class_loss"][-1] + box_weight * history["box_loss"][-1])
        history["accuracy"].append(val["accuracy"])
        history["mean_iou"].append(val["mean_iou"])
        history["found"].append(val["found"])
        history["seconds"].append(time.time() - start_time)
        print(f"epoch {epoch:2d}/{epochs}  training loss {history['loss'][-1]:.3f} (class {history['class_loss'][-1]:.3f}, "
              f"box {history['box_loss'][-1]:.4f})   validation: class right {val['accuracy']:5.1f}%, "
              f"mean IoU {val['mean_iou']:.3f}   ({history['seconds'][-1]:.0f} s)")
    model.eval()
    plot_detector_history(history)
    return history


def plot_detector_history(history: dict, title: str | None = None):
    """Left: the two training losses. Right: validation accuracy and mean IoU, per epoch."""
    fig = _new_figure(10.0, 3.3)
    ax1, ax2 = fig.subplots(1, 2)
    ep = history["epoch"]
    ax1.plot(ep, history["class_loss"], "o-", color="#2a78d6", lw=1.8, ms=4, label="class loss (cross-entropy)")
    ax1.plot(ep, history["box_loss"], "o-", color="#eb6834", lw=1.8, ms=4, label="box loss (L1, 0..1 units)")
    ax1.set_xlabel("epoch", fontsize=9)
    ax1.set_title("training losses (lower is better)", fontsize=10, loc="left")
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_ylim(bottom=0)
    ax2.plot(ep, history["accuracy"], "o-", color="#1baf7a", lw=1.8, ms=4, label="class right (%)")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("%", fontsize=9)
    ax2.set_xlabel("epoch", fontsize=9)
    ax3 = ax2.twinx()
    ax3.plot(ep, history["mean_iou"], "s-", color="#6250d6", lw=1.8, ms=4, label="mean IoU")
    ax3.set_ylim(0, 1.0)
    ax3.set_ylabel("mean IoU", fontsize=9)
    lines = ax2.get_lines() + ax3.get_lines()
    ax2.legend(lines, [ln.get_label() for ln in lines], fontsize=8, frameon=False, loc="lower right")
    ax2.set_title("validation pictures (higher is better)", fontsize=10, loc="left")
    for ax in (ax1, ax2, ax3):
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=10.5, x=0.01, ha="left")
    return _show(fig)


def show_detector_predictions(model: nn.Module, dataset: OneObjectSet, n: int = 12, cols: int = 6,
                              title: str | None = None):
    """The first n pictures: true box (black, dashed) and predicted box (red), predicted class, IoU."""
    scores, boxes = _predict_one_object(model, dataset.images[:n])
    probabilities = scores.softmax(dim=1)
    rows = (n + cols - 1) // cols
    fig = _new_figure(1.85 * cols, 2.05 * rows + 0.35)
    axs = np.atleast_1d(fig.subplots(rows, cols)).reshape(-1)
    for k, ax in enumerate(axs):
        if k >= n:
            ax.axis("off")
            continue
        _imshow(ax, dataset.images[k])
        _box(ax, dataset.boxes[k].tolist(), TRUE_COLOR, lw=1.3, ls="--")
        _box(ax, boxes[k].tolist(), PRED_COLOR, lw=2.0)
        label = int(probabilities[k].argmax())
        sure = float(probabilities[k, label])
        iou = float(box_iou(boxes[k:k + 1], dataset.boxes[k:k + 1])[0])
        right = label == int(dataset.labels[k])
        ok = right and iou >= 0.5
        text = f"{CLASSES[label]} {100 * sure:.0f}%\nIoU {iou:.2f}"
        if not right:
            text += f" (is {CLASSES[int(dataset.labels[k])]})"
        ax.set_title(text, fontsize=7.5, color=GOOD if ok else BAD)
    fig.suptitle(title or "dashed black: the true box; red: the predicted box; title: predicted class, IoU",
                 fontsize=10, x=0.01, ha="left", color=INK)
    return _show(fig)


# ============================================================================================== section 4: a grid
def _cell_centres():
    centres = (torch.arange(GRID, dtype=torch.float32) + 0.5) * CELL
    cy = centres.view(GRID, 1).expand(GRID, GRID)
    cx = centres.view(1, GRID).expand(GRID, GRID)
    return cx, cy


def grid_targets(boxes: torch.Tensor, labels: torch.Tensor):
    """What each cell of the 8 x 8 grid should say for one picture.

    A cell is responsible for an object when the cell's centre lies inside the object's box (if no centre
    does, the cell that holds the box's centre). Returns object (8, 8) 0/1, label (8, 8) (-1 = none) and
    distances (4, 8, 8): from the cell's centre to the left, top, right and bottom edge of its box, / 64.
    """
    cx, cy = _cell_centres()
    obj = torch.zeros(GRID, GRID)
    lab = torch.full((GRID, GRID), -1, dtype=torch.int64)
    dist = torch.zeros(4, GRID, GRID)
    for box, label in zip(boxes.tolist(), labels.tolist()):
        x1, y1, x2, y2 = box
        inside = (cx > x1) & (cx < x2) & (cy > y1) & (cy < y2)
        if not bool(inside.any()):
            col = min(GRID - 1, int(((x1 + x2) / 2) // CELL))
            row = min(GRID - 1, int(((y1 + y2) / 2) // CELL))
            inside = torch.zeros(GRID, GRID, dtype=torch.bool)
            inside[row, col] = True
        obj[inside] = 1.0
        lab[inside] = int(label)
        dist[0][inside] = (cx[inside] - x1) / SIZE
        dist[1][inside] = (cy[inside] - y1) / SIZE
        dist[2][inside] = (x2 - cx[inside]) / SIZE
        dist[3][inside] = (y2 - cy[inside]) / SIZE
    return obj, lab, dist


def _all_grid_targets(dataset: ManyObjectsSet):
    objs = []
    labs = []
    dists = []
    for boxes, labels in zip(dataset.boxes, dataset.labels):
        obj, lab, dist = grid_targets(boxes, labels)
        objs.append(obj)
        labs.append(lab)
        dists.append(dist)
    return torch.stack(objs), torch.stack(labs), torch.stack(dists)


def _grid_loss(out: torch.Tensor, obj: torch.Tensor, lab: torch.Tensor, dist: torch.Tensor, box_weight: float):
    """out (N, 8, 8, 8): channel 0 object score, 1-3 class scores, 4-7 distances (before the sigmoid)."""
    object_loss = F.binary_cross_entropy_with_logits(out[:, 0], obj)
    positive = obj > 0.5
    class_scores = out[:, 1:4].permute(0, 2, 3, 1)[positive]          # (P, 3)
    predicted = torch.sigmoid(out[:, 4:8]).permute(0, 2, 3, 1)[positive]
    wanted = dist.permute(0, 2, 3, 1)[positive]
    class_loss = F.cross_entropy(class_scores, lab[positive])
    box_loss = F.l1_loss(predicted, wanted)
    total = object_loss + class_loss + box_weight * box_loss
    return total, object_loss.item(), class_loss.item(), box_loss.item()


def _decode(out: torch.Tensor):
    """(N, 8, 8, 8) raw outputs -> boxes (N, 64, 4) pixels, scores (N, 64), labels (N, 64), cell by cell."""
    n = out.shape[0]
    cx, cy = _cell_centres()
    scores = torch.sigmoid(out[:, 0]).reshape(n, -1)
    labels = out[:, 1:4].argmax(dim=1).reshape(n, -1)
    d = torch.sigmoid(out[:, 4:8]) * SIZE
    x1 = cx - d[:, 0]
    y1 = cy - d[:, 1]
    x2 = cx + d[:, 2]
    y2 = cy + d[:, 3]
    boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(n, -1, 4)
    return boxes, scores, labels


@torch.no_grad()
def grid_candidates(net: nn.Module, picture: torch.Tensor):
    """Run the grid detector on ONE picture: 64 candidate boxes (64, 4) in pixels, their scores (64,) and
    labels (64,), one per cell, row by row."""
    net.eval()
    out = net(picture.unsqueeze(0))
    boxes, scores, labels = _decode(out)
    return boxes[0], scores[0], labels[0]


def final_detections(boxes, scores, labels, score_threshold: float = 0.5, iou_threshold: float = 0.5):
    """Score threshold, then non-maximum suppression: the detections that remain (boxes, scores, labels)."""
    keep = scores > score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    kept = torchvision.ops.nms(boxes, scores, iou_threshold)
    return boxes[kept], scores[kept], labels[kept]


def _match(boxes, labels, true_boxes, true_labels, iou_needed: float = 0.5):
    """How many true objects are found (a box of the right class with IoU >= 0.5), and how many boxes are extra."""
    used = [False] * len(boxes)
    found = 0
    for t in range(len(true_boxes)):
        best = -1
        best_iou = iou_needed
        for k in range(len(boxes)):
            if used[k] or int(labels[k]) != int(true_labels[t]):
                continue
            iou = float(box_iou(boxes[k:k + 1], true_boxes[t:t + 1])[0])
            if iou >= best_iou:
                best = k
                best_iou = iou
        if best >= 0:
            used[best] = True
            found += 1
    extra = used.count(False)
    return found, extra


@torch.no_grad()
def evaluate_grid(net: nn.Module, dataset: ManyObjectsSet, score_threshold: float = 0.5,
                  iou_threshold: float = 0.5, verbose: bool = True) -> dict:
    """Found objects and extra boxes over a whole set, after the score threshold and NMS."""
    net.eval()
    total = 0
    found = 0
    extra = 0
    for start in range(0, len(dataset), 256):
        out = net(dataset.images[start:start + 256])
        boxes, scores, labels = _decode(out)
        for k in range(out.shape[0]):
            i = start + k
            b, s, lb = final_detections(boxes[k], scores[k], labels[k], score_threshold, iou_threshold)
            f, e = _match(b, lb, dataset.boxes[i], dataset.labels[i])
            total += len(dataset.boxes[i])
            found += f
            extra += e
    result = _Report({"objects": total, "found": found, "extra": extra,
                      "found_percent": round(100.0 * found / max(total, 1), 1)})
    result.text = (f"{len(dataset)} validation pictures with {total} shapes: {found} found "
                   f"({result['found_percent']:.1f}%: a box of the right class with IoU >= 0.5), {extra} extra boxes "
                   f"(score threshold {score_threshold}, NMS IoU threshold {iou_threshold})")
    if verbose:
        print(result.text)
    return result._mark()


def train_grid_detector(net: nn.Module, train_set: ManyObjectsSet, val_set: ManyObjectsSet, epochs: int = 10,
                        lr: float = 0.002, box_weight: float = 5.0, batch_size: int = 64, seed: int = 0):
    """Train the grid detector. loss = object loss (binary cross-entropy, every cell) + class loss (cross-entropy,
    responsible cells) + box_weight x box loss (L1 on the 4 distances, responsible cells). Adam optimiser."""
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    obj, lab, dist = _all_grid_targets(train_set)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    history = {"epoch": [], "object_loss": [], "class_loss": [], "box_loss": [], "found": [], "extra": [],
               "seconds": []}
    start_time = time.time()
    print(f"training on {len(train_set)} pictures, {epochs} epochs, learning rate {lr}")
    for epoch in range(1, epochs + 1):
        net.train()
        sums = [0.0, 0.0, 0.0]
        batches = _batches(len(train_set), batch_size, generator)
        for idx in batches:
            out = net(train_set.images[idx])
            loss, lo, lc, lb = _grid_loss(out, obj[idx], lab[idx], dist[idx], box_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sums[0] += lo
            sums[1] += lc
            sums[2] += lb
        val = evaluate_grid(net, val_set, verbose=False)
        history["epoch"].append(epoch)
        history["object_loss"].append(sums[0] / len(batches))
        history["class_loss"].append(sums[1] / len(batches))
        history["box_loss"].append(sums[2] / len(batches))
        history["found"].append(val["found_percent"])
        history["extra"].append(val["extra"])
        history["seconds"].append(time.time() - start_time)
        print(f"epoch {epoch:2d}/{epochs}  losses: object {history['object_loss'][-1]:.3f}, class "
              f"{history['class_loss'][-1]:.3f}, box {history['box_loss'][-1]:.4f}   validation: "
              f"{val['found']} of {val['objects']} shapes found ({val['found_percent']:.1f}%), "
              f"{val['extra']} extra boxes   ({history['seconds'][-1]:.0f} s)")
    net.eval()
    plot_grid_history(history)
    return history


def plot_grid_history(history: dict, title: str | None = None):
    """Left: the three training losses. Right: shapes found (%) and extra boxes on the validation pictures."""
    fig = _new_figure(10.0, 3.3)
    ax1, ax2 = fig.subplots(1, 2)
    ep = history["epoch"]
    ax1.plot(ep, history["object_loss"], "o-", color="#6250d6", lw=1.8, ms=4, label="object loss (every cell)")
    ax1.plot(ep, history["class_loss"], "o-", color="#2a78d6", lw=1.8, ms=4, label="class loss")
    ax1.plot(ep, history["box_loss"], "o-", color="#eb6834", lw=1.8, ms=4, label="box loss (L1, 0..1 units)")
    ax1.set_xlabel("epoch", fontsize=9)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title("training losses (lower is better)", fontsize=10, loc="left")
    ax2.plot(ep, history["found"], "o-", color="#1baf7a", lw=1.8, ms=4, label="shapes found (%)")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("found (%)", fontsize=9)
    ax2.set_xlabel("epoch", fontsize=9)
    ax3 = ax2.twinx()
    ax3.plot(ep, history["extra"], "s-", color="#e34948", lw=1.8, ms=4, label="extra boxes")
    ax3.set_ylim(bottom=0)
    ax3.set_ylabel("extra boxes", fontsize=9)
    lines = ax2.get_lines() + ax3.get_lines()
    ax2.legend(lines, [ln.get_label() for ln in lines], fontsize=8, frameon=False, loc="center right")
    ax2.set_title("validation pictures", fontsize=10, loc="left")
    for ax in (ax1, ax2, ax3):
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=10.5, x=0.01, ha="left")
    return _show(fig)


def show_grid_idea(picture, boxes, labels, title: str | None = None):
    """The 8 x 8 grid on a picture: the cells responsible for each shape shaded, and one cell's four distances."""
    from matplotlib.patches import FancyArrowPatch, Rectangle

    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    labels = torch.as_tensor(labels, dtype=torch.int64).reshape(-1)
    obj, lab, dist = grid_targets(boxes, labels)
    fig = _new_figure(10.0, 5.0)
    ax1, ax2 = fig.subplots(1, 2)
    for ax in (ax1, ax2):
        _imshow(ax, picture, axes=True)
        for k in range(GRID + 1):
            ax.axhline(k * CELL, color="white", lw=0.8, alpha=0.9)
            ax.axvline(k * CELL, color="white", lw=0.8, alpha=0.9)
        tags = _Labels(ax, 4.6, SIZE, SIZE, 8.0)
        for box, label in zip(boxes.tolist(), labels.tolist()):
            _box(ax, box, CLASS_COLORS[label], lw=2.0, text=CLASSES[label], labels=tags)
    for row in range(GRID):
        for col in range(GRID):
            cx = (col + 0.5) * CELL
            cy = (row + 0.5) * CELL
            if obj[row, col] > 0:
                color = CLASS_COLORS[int(lab[row, col])]
                ax1.add_patch(Rectangle((col * CELL, row * CELL), CELL, CELL, fc=color, alpha=0.35, ec="none"))
                ax1.plot(cx, cy, "o", color=color, ms=3.5)
            else:
                ax1.plot(cx, cy, ".", color="white", ms=2.5)
    ax1.set_title("the cells whose centre lies inside a box are responsible\nfor that object (shaded)", fontsize=9.5,
                  loc="left")
    # one responsible cell, near the middle of the first object: its four distances
    best = None
    x1, y1, x2, y2 = boxes[0].tolist()
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    for row in range(GRID):
        for col in range(GRID):
            if obj[row, col] > 0 and int(lab[row, col]) == int(labels[0]):
                cx = (col + 0.5) * CELL
                cy = (row + 0.5) * CELL
                d = (cx - mx) ** 2 + (cy - my) ** 2 + 0.01 * col
                if best is None or d < best[0]:
                    best = (d, row, col)
    if best is not None:
        _, row, col = best
        cx = (col + 0.5) * CELL
        cy = (row + 0.5) * CELL
        color = CLASS_COLORS[int(labels[0])]
        ax2.add_patch(Rectangle((col * CELL, row * CELL), CELL, CELL, fc=color, alpha=0.45, ec=INK, lw=1.2))
        ends = [(x1, cy, "left"), (cx, y1, "top"), (x2, cy, "right"), (cx, y2, "bottom")]
        for ex, ey, name in ends:
            ax2.add_patch(FancyArrowPatch((cx, cy), (ex, ey), arrowstyle="-|>", mutation_scale=9, color=INK, lw=1.3))
        ax2.plot(cx, cy, "o", color=INK, ms=4)
        ax2.text(x1 - 0.5, cy - 0.8, f"left {cx - x1:.0f}", fontsize=8, ha="right", color=INK,
                 bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
        ax2.text(cx + 0.8, y1 - 0.6, f"top {cy - y1:.0f}", fontsize=8, ha="left", va="bottom", color=INK,
                 bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
        ax2.text(x2 + 0.5, cy - 0.8, f"right {x2 - cx:.0f}", fontsize=8, ha="left", color=INK,
                 bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
        ax2.text(cx + 0.8, y2 + 0.6, f"bottom {y2 - cy:.0f}", fontsize=8, ha="left", va="top", color=INK,
                 bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
        ax2.set_title(f"one cell (row {row}, column {col}, centre at x = {cx:.0f}, y = {cy:.0f}) and the four\n"
                      f"distances from its centre to the edges of the box", fontsize=9.5, loc="left")
    if title:
        fig.suptitle(title, fontsize=10.5, x=0.01, ha="left")
    return _show(fig)


def show_before_after(picture, boxes, labels, scores, keep, before_title: str = "before", after_title: str = "after"):
    """Two pictures side by side: all the given boxes, and only the kept ones (keep: indices or True/False)."""
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    labels = torch.as_tensor(labels).reshape(-1)
    scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    keep = torch.as_tensor(keep)
    fig = _new_figure(10.0, 5.2)
    ax1, ax2 = fig.subplots(1, 2)
    panels = [(ax1, boxes, labels, scores, before_title),
              (ax2, boxes[keep], labels[keep], scores[keep], after_title)]
    for ax, b, lb, s, title in panels:
        _imshow(ax, picture, axes=True)
        order = torch.argsort(s)                      # draw the best boxes last, on top
        tags = _Labels(ax, 4.6, SIZE, SIZE, 7.5)
        with_text = len(b) <= 6                       # with many boxes, tags would hide the picture
        for k in order.tolist():
            alpha = 0.25 + 0.75 * float(s[k])
            text = None
            if with_text:
                text = f"{CLASSES[int(lb[k])]} {float(s[k]):.2f}"
            _box(ax, b[k].tolist(), CLASS_COLORS[int(lb[k])], lw=0.7 + 0.9 * float(s[k]), alpha=alpha, text=text,
                 labels=tags)
        word = "box" if len(b) == 1 else "boxes"
        ax.set_title(f"{title}: {len(b)} {word}", fontsize=10, loc="left")
    return _show(fig)


def show_grid_results(net: nn.Module, dataset: ManyObjectsSet, score_threshold: float = 0.5,
                      iou_threshold: float = 0.5, n: int = 8, cols: int = 4, title: str | None = None):
    """The final detections (after the score threshold and NMS) on the first n pictures of a set."""
    rows = (n + cols - 1) // cols
    fig = _new_figure(2.6 * cols, 2.75 * rows + 0.35)
    axs = np.atleast_1d(fig.subplots(rows, cols)).reshape(-1)
    for k, ax in enumerate(axs):
        if k >= n:
            ax.axis("off")
            continue
        boxes, scores, labels = grid_candidates(net, dataset.images[k])
        b, s, lb = final_detections(boxes, scores, labels, score_threshold, iou_threshold)
        _imshow(ax, dataset.images[k])
        for true_box in dataset.boxes[k].tolist():
            _box(ax, true_box, TRUE_COLOR, lw=1.0, ls="--")
        tags = _Labels(ax, 2.45, SIZE, SIZE, 6.5)
        for j in range(len(b)):
            _box(ax, b[j].tolist(), CLASS_COLORS[int(lb[j])], lw=2.0,
                 text=f"{CLASSES[int(lb[j])]} {float(s[j]):.2f}", labels=tags)
        found, extra = _match(b, lb, dataset.boxes[k], dataset.labels[k])
        total = len(dataset.boxes[k])
        ok = found == total and extra == 0
        ax.set_title(f"{found} of {total} found, {extra} extra", fontsize=8.5, color=GOOD if ok else BAD)
    fig.suptitle(title or f"final detections (score > {score_threshold}, NMS at IoU {iou_threshold}); "
                          f"dashed black: the true boxes", fontsize=10, x=0.01, ha="left", color=INK)
    return _show(fig)


# ============================================================================================== section 5: photos
WEIGHTS_FILE = "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"


def _weights_path() -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints" / WEIGHTS_FILE


def load_coco_detector(download: bool = False):
    """torchvision's Faster R-CNN (ResNet-50 FPN v2) with its COCO weights, in evaluation mode.

    Returns (detector, categories): categories[label] is the name of a label number (91 entries, 0 is
    "__background__"). If the weights file is not on this computer and download is False, prints what to do and
    returns (None, categories): the cells that use the detector then say "skipped".
    """
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    categories = list(weights.meta["categories"])
    path = _weights_path()
    if not path.is_file() and not download:
        print("The pretrained weights are not on this computer, so section 5 is skipped.")
        print(f"  expected file: {path}")
        print("  to get it: run  detector, categories = D.load_coco_detector(download=True)  once with an internet")
        print(f"  connection (it downloads {WEIGHTS_FILE}, 167 MB), or copy that file into the folder above.")
        return None, categories
    t0 = time.time()
    detector = fasterrcnn_resnet50_fpn_v2(weights=weights)
    detector.eval()
    n_params = 0
    for p in detector.parameters():
        n_params += p.numel()
    print(f"Faster R-CNN (ResNet-50 FPN v2) with its COCO weights: {n_params / 1e6:.1f} million parameters, "
          f"{len(categories) - 1} label numbers (1 to {len(categories) - 1}), loaded in {time.time() - t0:.1f} s")
    return detector, categories


def _photo_files(source: str) -> list:
    from .torchkit import data_root
    if source == "street":
        folder = data_root() / "penn-fudan" / "PennFudanPed" / "PNGImages"
        return sorted(folder.glob("*.png"))
    if source == "pets":
        folder = data_root() / "oxford-iiit-pet" / "images"
        return sorted(folder.glob("*.jpg"))
    raise ValueError('source must be "street" (Penn-Fudan) or "pets" (Oxford-IIIT Pet)')


def load_photo(source: str = "street", index: int = 0):
    """One photo as a (3, H, W) tensor with values 0..1, and the boxes the dataset gives for it.

    source "street": Penn-Fudan street photos (170), with the boxes of the pedestrians marked by hand.
    source "pets": Oxford-IIIT Pet photos (7 390), no boxes. Returns (None, None) with a message if missing.
    """
    from PIL import Image

    files = _photo_files(source)
    if len(files) == 0:
        print(f"No {source} photos found (expected them under the data folder): run  python tools/download_data.py")
        return None, None
    index = int(index) % len(files)
    path = files[index]
    photo = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0)
    photo = photo.permute(2, 0, 1).contiguous()
    true_boxes = None
    if source == "street":
        mask_path = path.parent.parent / "PedMasks" / (path.stem + "_mask.png")
        if mask_path.is_file():
            ids = np.asarray(Image.open(mask_path))
            true_boxes = []
            for value in np.unique(ids):
                if value == 0:
                    continue
                true_boxes.append(_mask_box(ids == value))
            true_boxes = torch.tensor(true_boxes, dtype=torch.float32).reshape(-1, 4)
    marked = ""
    if true_boxes is not None:
        word = "person" if len(true_boxes) == 1 else "people"
        marked = f", {len(true_boxes)} {word} marked by hand in the dataset"
    print(f"{source} photo {index} (numbered 0 to {len(files) - 1}), {path.name}: {photo.shape[2]} x {photo.shape[1]} "
          f"pixels (width x height){marked}")
    return photo, true_boxes


def _empty_result() -> dict:
    return {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64), "scores": torch.zeros(0)}


def detect(detector, photo) -> dict:
    """Run the detector on one photo: {"boxes": (k, 4) pixels, "labels": (k,), "scores": (k,)}, best first."""
    if detector is None or photo is None:
        print("skipped: no detector or no photo (see the messages above)")
        return _empty_result()
    t0 = time.time()
    detector.eval()
    with torch.no_grad():
        outputs = detector([photo])
    result = outputs[0]
    print(f"one forward pass on the CPU: {time.time() - t0:.1f} s, {len(result['boxes'])} boxes")
    return result


def print_result(result: dict, categories, n: int = 8):
    """The first n rows of a detector's result: box, label number, its name, score."""
    print(f"keys: {list(result.keys())}")
    print(f"boxes: {tuple(result['boxes'].shape)}   labels: {tuple(result['labels'].shape)}   scores: "
          f"{tuple(result['scores'].shape)}")
    if len(result["boxes"]) == 0:
        return
    print()
    print(f"{'#':>3}  {'box (x1, y1, x2, y2) in pixels':<32} {'label':>5}  {'name':<14} {'score':>6}")
    for k in range(min(n, len(result["boxes"]))):
        x1, y1, x2, y2 = result["boxes"][k].tolist()
        label = int(result["labels"][k])
        name = categories[label] if categories is not None and label < len(categories) else str(label)
        box_text = f"({x1:6.1f}, {y1:6.1f}, {x2:6.1f}, {y2:6.1f})"
        print(f"{k:>3}  {box_text:<32} {label:>5}  {name:<14} {float(result['scores'][k]):6.3f}")
    if len(result["boxes"]) > n:
        print(f"... and {len(result['boxes']) - n} more rows")


_PHOTO_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#e34948", "#008300"]


def show_photo_detections(photo, result: dict, categories, score_threshold: float = 0.5, true_boxes=None,
                          title: str | None = None):
    """The photo with the detections whose score is at least score_threshold (and the dataset's own boxes)."""
    if photo is None:
        print("skipped: no photo")
        return None
    keep = result["scores"] >= score_threshold
    boxes = result["boxes"][keep]
    labels = result["labels"][keep]
    scores = result["scores"][keep]
    img = _picture_hwc(photo)
    h, w = img.shape[:2]
    panels = 1 if true_boxes is None else 2
    panel_w = 5.4
    panel_h = panel_w * h / w
    if panel_h > 6.0:                                   # a tall photo: limit the height instead
        panel_h = 6.0
        panel_w = panel_h * w / h
    fig = _new_figure(panel_w * panels + 0.2, panel_h + 0.6)
    axs = np.atleast_1d(fig.subplots(1, panels)).reshape(-1)
    if true_boxes is not None:
        _imshow(axs[0], photo)
        n_true = len(_as_list(true_boxes))
        for box in _as_list(true_boxes):
            _box(axs[0], box, TRUE_COLOR, lw=2.0, ls="--")
        word = "person" if n_true == 1 else "people"
        axs[0].set_title(f"the dataset: {n_true} {word} marked by hand", fontsize=9.5, loc="left")
    ax = axs[-1]
    _imshow(ax, photo)
    tags = _Labels(ax, panel_w - 0.1, w, h, 7.5)
    colors = {}
    for k in range(len(boxes)):
        label = int(labels[k])
        if label not in colors:
            colors[label] = _PHOTO_COLORS[len(colors) % len(_PHOTO_COLORS)]
        name = categories[label] if categories is not None and label < len(categories) else str(label)
        _box(ax, boxes[k].tolist(), colors[label], lw=2.2, text=f"{name} {float(scores[k]):.2f}", labels=tags)
    word = "box" if len(boxes) == 1 else "boxes"
    ax.set_title(title or f"the detector: {len(boxes)} {word} with score >= {score_threshold}", fontsize=9.5,
                 loc="left")
    return _show(fig)
