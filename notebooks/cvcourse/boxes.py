"""Reference implementations for object detection geometry (NumPy).

Box convention everywhere in the course: **xyxy, continuous coordinates**,
(x1, y1, x2, y2) with x to the right and y downwards, area = (x2-x1)*(y2-y1).
No "+1" pixel convention (the old course used it; modern libraries do not).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------
def xyxy_to_xywh(b):
    b = np.asarray(b, dtype=float)
    return np.stack([b[..., 0], b[..., 1], b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]], -1)


def xywh_to_xyxy(b):
    b = np.asarray(b, dtype=float)
    return np.stack([b[..., 0], b[..., 1], b[..., 0] + b[..., 2], b[..., 1] + b[..., 3]], -1)


def xyxy_to_cxcywh(b):
    b = np.asarray(b, dtype=float)
    w, h = b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]
    return np.stack([b[..., 0] + w / 2, b[..., 1] + h / 2, w, h], -1)


def cxcywh_to_xyxy(b):
    b = np.asarray(b, dtype=float)
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def box_area(b):
    b = np.asarray(b, dtype=float)
    return np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)


# ---------------------------------------------------------------------------
# IoU family
# ---------------------------------------------------------------------------
def box_iou(a, b):
    """Pairwise IoU. a: (N, 4), b: (M, 4) -> (N, M)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def iou_single(a, b) -> float:
    return float(box_iou(np.asarray(a)[None], np.asarray(b)[None])[0, 0])


def generalized_iou(a, b):
    """Elementwise GIoU for aligned pairs. a, b: (N, 4) -> (N,). In [-1, 1]."""
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, :2], b[:, :2]); rb = np.minimum(a[:, 2:], b[:, 2:])
    wh = np.clip(rb - lt, 0, None); inter = wh[:, 0] * wh[:, 1]
    union = box_area(a) + box_area(b) - inter
    iou = inter / np.maximum(union, 1e-12)
    elt = np.minimum(a[:, :2], b[:, :2]); erb = np.maximum(a[:, 2:], b[:, 2:])
    ewh = np.clip(erb - elt, 0, None); enclose = ewh[:, 0] * ewh[:, 1]
    return iou - (enclose - union) / np.maximum(enclose, 1e-12)


def distance_iou(a, b):
    """Elementwise DIoU: IoU minus normalised centre distance."""
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    iou = np.diag(box_iou(a, b))
    ca = xyxy_to_cxcywh(a)[:, :2]; cb = xyxy_to_cxcywh(b)[:, :2]
    d2 = np.sum((ca - cb) ** 2, -1)
    elt = np.minimum(a[:, :2], b[:, :2]); erb = np.maximum(a[:, 2:], b[:, 2:])
    c2 = np.sum((erb - elt) ** 2, -1)
    return iou - d2 / np.maximum(c2, 1e-12)


def complete_iou(a, b):
    """Elementwise CIoU: DIoU minus an aspect-ratio consistency term."""
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    diou = distance_iou(a, b)
    iou = np.diag(box_iou(a, b))
    wa, ha = a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]
    wb, hb = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    v = (4 / np.pi ** 2) * (np.arctan(wb / np.maximum(hb, 1e-12)) - np.arctan(wa / np.maximum(ha, 1e-12))) ** 2
    alpha = v / np.maximum(1 - iou + v, 1e-12)
    return diou - alpha * v


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------
def nms(boxes, scores, iou_threshold: float = 0.5):
    """Greedy NMS. Returns indices of kept boxes, in decreasing score order."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    scores = np.asarray(scores, float).reshape(-1)
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = box_iou(boxes[i][None], boxes[order[1:]])[0]
        order = order[1:][ious < iou_threshold]
    return np.array(keep, dtype=int)


def soft_nms(boxes, scores, sigma: float = 0.5, score_threshold: float = 0.001):
    """Gaussian Soft-NMS. Returns (kept indices, new scores)."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    scores = np.asarray(scores, float).copy()
    idx = np.arange(len(scores))
    keep, kept_scores = [], []
    while idx.size > 0:
        m = idx[np.argmax(scores[idx])]
        keep.append(int(m)); kept_scores.append(float(scores[m]))
        idx = idx[idx != m]
        if idx.size == 0:
            break
        ious = box_iou(boxes[m][None], boxes[idx])[0]
        scores[idx] *= np.exp(-(ious ** 2) / sigma)
        idx = idx[scores[idx] > score_threshold]
    return np.array(keep, int), np.array(kept_scores)


def batched_nms(boxes, scores, labels, iou_threshold=0.5):
    """Class-wise NMS by offsetting boxes per class (the torchvision trick)."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    labels = np.asarray(labels)
    if boxes.size == 0:
        return np.array([], int)
    offset = (boxes.max() + 1.0) * labels[:, None]
    return nms(boxes + offset, scores, iou_threshold)


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------
def make_anchors(feature_size, stride, scales, ratios):
    """Anchors for one feature-map level, in input-image pixel coordinates.

    feature_size: (fh, fw) cells
    stride      : input pixels per cell
    scales      : iterable of box side lengths (pixels) for ratio 1
    ratios      : iterable of h/w aspect ratios
    Returns (fh * fw * A, 4) xyxy, ordered cell-major (row, col, anchor).
    """
    fh, fw = feature_size
    scales = np.asarray(scales, float); ratios = np.asarray(ratios, float)
    # all (scale, ratio) combinations -> widths / heights preserving area = scale**2
    s, r = np.meshgrid(scales, ratios, indexing="ij")
    w = (s / np.sqrt(r)).reshape(-1)
    h = (s * np.sqrt(r)).reshape(-1)
    base = np.stack([-w / 2, -h / 2, w / 2, h / 2], -1)            # (A, 4)
    cy = (np.arange(fh) + 0.5) * stride
    cx = (np.arange(fw) + 0.5) * stride
    cx, cy = np.meshgrid(cx, cy)                                     # (fh, fw)
    centers = np.stack([cx, cy, cx, cy], -1).reshape(-1, 1, 4)       # (fh*fw, 1, 4)
    return (centers + base[None]).reshape(-1, 4)


def match_anchors(anchors, gt_boxes, pos_iou=0.7, neg_iou=0.3, allow_low_quality=True):
    """Assign each anchor a ground-truth index.

    Returns (matched_idx, labels) where labels is 1 positive, 0 negative,
    -1 ignore, and matched_idx is the GT index (valid where labels == 1).
    """
    anchors = np.asarray(anchors, float).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
    N = len(anchors)
    if len(gt_boxes) == 0:
        return np.zeros(N, int), np.zeros(N, int)
    ious = box_iou(anchors, gt_boxes)                    # (N, G)
    best_gt = ious.argmax(1); best_iou = ious.max(1)
    labels = np.full(N, -1, int)
    labels[best_iou < neg_iou] = 0
    labels[best_iou >= pos_iou] = 1
    if allow_low_quality:   # every GT gets at least its best anchor
        best_anchor_per_gt = ious.argmax(0)
        labels[best_anchor_per_gt] = 1
        best_gt[best_anchor_per_gt] = np.arange(len(gt_boxes))
    return best_gt, labels


# ---------------------------------------------------------------------------
# Box encoding (Faster R-CNN parametrisation)
# ---------------------------------------------------------------------------
def encode_boxes(gt, anchors, weights=(1.0, 1.0, 1.0, 1.0)):
    """gt, anchors: (N, 4) xyxy aligned -> deltas (N, 4) = (tx, ty, tw, th)."""
    g = xyxy_to_cxcywh(gt); a = xyxy_to_cxcywh(anchors)
    wx, wy, ww, wh = weights
    tx = wx * (g[:, 0] - a[:, 0]) / a[:, 2]
    ty = wy * (g[:, 1] - a[:, 1]) / a[:, 3]
    tw = ww * np.log(g[:, 2] / a[:, 2])
    th = wh * np.log(g[:, 3] / a[:, 3])
    return np.stack([tx, ty, tw, th], -1)


def decode_boxes(deltas, anchors, weights=(1.0, 1.0, 1.0, 1.0), clip=np.log(1000.0 / 16)):
    """Inverse of encode_boxes. Returns xyxy (N, 4)."""
    a = xyxy_to_cxcywh(anchors)
    wx, wy, ww, wh = weights
    dx, dy = deltas[:, 0] / wx, deltas[:, 1] / wy
    dw, dh = np.clip(deltas[:, 2] / ww, None, clip), np.clip(deltas[:, 3] / wh, None, clip)
    cx = a[:, 0] + dx * a[:, 2]
    cy = a[:, 1] + dy * a[:, 3]
    w = a[:, 2] * np.exp(dw)
    h = a[:, 3] * np.exp(dh)
    return cxcywh_to_xyxy(np.stack([cx, cy, w, h], -1))


def smooth_l1(x, beta=1.0 / 9):
    x = np.abs(np.asarray(x, float))
    return np.where(x < beta, 0.5 * x ** 2 / beta, x - 0.5 * beta)


# ---------------------------------------------------------------------------
# Evaluation: AP / mAP
# ---------------------------------------------------------------------------
def precision_recall(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """Single image, single class. Returns (precision, recall, tp_flags) ordered by score.

    Greedy matching: each prediction (by decreasing score) is a TP if its best
    unmatched GT has IoU >= threshold, otherwise a FP. Each GT matches once.
    """
    pred_boxes = np.asarray(pred_boxes, float).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
    order = np.argsort(-np.asarray(pred_scores, float))
    matched = np.zeros(len(gt_boxes), bool)
    tp = np.zeros(len(order), bool)
    for k, i in enumerate(order):
        if len(gt_boxes) == 0:
            break
        ious = box_iou(pred_boxes[i][None], gt_boxes)[0]
        ious[matched] = -1
        j = int(ious.argmax())
        if ious[j] >= iou_threshold:
            tp[k] = True
            matched[j] = True
    tp_c = np.cumsum(tp); fp_c = np.cumsum(~tp)
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-12)
    recall = tp_c / max(len(gt_boxes), 1)
    return precision, recall, tp


def average_precision(precision, recall, method="interp101"):
    """Area under the (monotone) PR curve.

    method 'all'       : exact area with precision made monotonically decreasing (VOC2010+)
    method 'interp101' : COCO-style 101-point interpolation
    method 'interp11'  : VOC2007 11-point
    """
    p = np.concatenate([[0.0], precision, [0.0]])
    r = np.concatenate([[0.0], recall, [1.0]])
    # make precision monotonically decreasing from the right
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    if method == "all":
        idx = np.where(r[1:] != r[:-1])[0]
        return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))
    n = 101 if method == "interp101" else 11
    rs = np.linspace(0, 1, n)
    ap = 0.0
    for t in rs:
        mask = r >= t
        ap += (p[mask].max() if mask.any() else 0.0)
    return float(ap / n)


def mean_average_precision(predictions, ground_truths, num_classes, iou_thresholds=(0.5,), method="interp101"):
    """Dataset-level mAP.

    predictions  : list over images of dict(boxes (N,4), scores (N,), labels (N,))
    ground_truths: list over images of dict(boxes (M,4), labels (M,))
    Returns (mAP, per-class AP array averaged over thresholds).
    """
    aps = np.zeros((len(iou_thresholds), num_classes))
    for ti, thr in enumerate(iou_thresholds):
        for c in range(num_classes):
            records = []   # (score, tp)
            n_gt = 0
            for pred, gt in zip(predictions, ground_truths):
                pb = np.asarray(pred["boxes"]).reshape(-1, 4)[np.asarray(pred["labels"]) == c]
                ps = np.asarray(pred["scores"]).reshape(-1)[np.asarray(pred["labels"]) == c]
                gb = np.asarray(gt["boxes"]).reshape(-1, 4)[np.asarray(gt["labels"]) == c]
                n_gt += len(gb)
                if len(pb) == 0:
                    continue
                _, _, tp = precision_recall(pb, ps, gb, thr)
                order = np.argsort(-ps)
                records += list(zip(ps[order], tp))
            if n_gt == 0:
                aps[ti, c] = np.nan
                continue
            if not records:
                aps[ti, c] = 0.0
                continue
            records.sort(key=lambda t: -t[0])
            tp = np.array([t[1] for t in records], bool)
            tp_c = np.cumsum(tp); fp_c = np.cumsum(~tp)
            precision = tp_c / np.maximum(tp_c + fp_c, 1e-12)
            recall = tp_c / n_gt
            aps[ti, c] = average_precision(precision, recall, method)
    per_class = np.nanmean(aps, 0)
    return float(np.nanmean(per_class)), per_class
