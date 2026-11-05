"""Synthetic 'shapes' dataset: circles, squares and triangles on a canvas.

No download, deterministic with a seed, and every mechanism of detection and
segmentation (boxes, anchors, matching, masks) is visible at a glance.

    img, boxes, labels, mask = shapes.make_sample(rng, size=128)
    X, B, L, M = shapes.make_dataset(n=200, size=128, seed=0)

Classes: 0 = circle, 1 = square, 2 = triangle  (background in masks is 0, classes +1).
"""
from __future__ import annotations

import numpy as np

CLASS_NAMES = ["circle", "square", "triangle"]
CLASS_COLORS = np.array([[0.93, 0.35, 0.30], [0.30, 0.55, 0.93], [0.35, 0.80, 0.45]])  # RGB in [0,1]


def _draw(canvas, mask, cls, cx, cy, r, color, inst_id):
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W]
    if cls == 0:                                   # circle
        region = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    elif cls == 1:                                 # square (axis aligned)
        region = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
    else:                                          # triangle, apex up
        # barycentric-free test: inside if below the two slanted edges and above the base
        base_y = cy + r * 0.6
        apex_y = cy - r
        t = np.clip((yy - apex_y) / max(base_y - apex_y, 1e-6), 0, None)
        region = (yy >= apex_y) & (yy <= base_y) & (np.abs(xx - cx) <= r * t)
    canvas[region] = color
    mask[region] = inst_id
    ys, xs = np.where(region)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)


def make_sample(rng: np.random.Generator, size: int = 128, max_objects: int = 4,
                min_r: int = 8, max_r: int = 24, noise: float = 0.03):
    """One image. Returns (img (H,W,3) float32, boxes (N,4) xyxy, labels (N,), mask (H,W) int).

    mask holds semantic class ids: 0 background, 1 circle, 2 square, 3 triangle.
    Instance masks can be rebuilt from boxes if needed (objects do not overlap).
    """
    img = np.full((size, size, 3), 0.95, dtype=np.float32)
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    mask = np.zeros((size, size), dtype=np.int64)
    boxes, labels = [], []
    n = rng.integers(1, max_objects + 1)
    tries = 0
    while len(boxes) < n and tries < 50:
        tries += 1
        cls = int(rng.integers(0, 3))
        r = int(rng.integers(min_r, max_r + 1))
        cx = int(rng.integers(r + 1, size - r - 1))
        cy = int(rng.integers(r + 1, size - r - 1))
        cand = np.array([cx - r, cy - r, cx + r, cy + r], float)
        # reject overlaps so the targets stay unambiguous
        if any(_overlap(cand, b) for b in boxes):
            continue
        color = CLASS_COLORS[cls] * rng.uniform(0.7, 1.0) + rng.normal(0, 0.02, 3)
        box = _draw(img, mask, cls, cx, cy, r, np.clip(color, 0, 1).astype(np.float32), cls + 1)
        if box is None:
            continue
        boxes.append(box); labels.append(cls)
    img = np.clip(img, 0, 1)
    return img, np.array(boxes).reshape(-1, 4), np.array(labels, int), mask


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def make_dataset(n: int = 200, size: int = 128, seed: int = 0, **kw):
    """Returns images (n,H,W,3), list of boxes, list of labels, masks (n,H,W)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, size, size, 3), np.float32)
    M = np.zeros((n, size, size), np.int64)
    B, L = [], []
    for i in range(n):
        img, boxes, labels, mask = make_sample(rng, size, **kw)
        X[i] = img; M[i] = mask; B.append(boxes); L.append(labels)
    return X, B, L, M


def grid_targets(boxes, labels, grid: int, size: int, num_classes: int = 3):
    """YOLOv1-style dense target: (grid, grid, 1 + 4 + C).

    channel 0  : objectness (1 if an object's centre falls in the cell)
    channels 1-4: box as (cx, cy, w, h) normalised to [0,1] of the *image*
    channels 5+: one-hot class
    """
    T = np.zeros((grid, grid, 5 + num_classes), np.float32)
    cell = size / grid
    for b, l in zip(boxes, labels):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        gi, gj = int(cy // cell), int(cx // cell)
        T[gi, gj, 0] = 1
        T[gi, gj, 1:5] = [cx / size, cy / size, (b[2] - b[0]) / size, (b[3] - b[1]) / size]
        T[gi, gj, 5 + int(l)] = 1
    return T


def decode_grid(T, size: int, score_threshold: float = 0.5):
    """Inverse of grid_targets for a prediction tensor (grid, grid, 5 + C).

    Returns boxes (N,4) xyxy, scores (N,), labels (N,).
    """
    grid = T.shape[0]
    boxes, scores, labels = [], [], []
    for i in range(grid):
        for j in range(grid):
            p = T[i, j]
            if p[0] < score_threshold:
                continue
            cx, cy, w, h = p[1:5] * size
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            scores.append(float(p[0]))
            labels.append(int(np.argmax(p[5:])))
    return np.array(boxes).reshape(-1, 4), np.array(scores), np.array(labels, int)
