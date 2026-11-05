"""Segmentation metrics and losses (NumPy reference implementations)."""
from __future__ import annotations

import numpy as np


def confusion_matrix(pred, target, num_classes):
    """pred, target: integer arrays of the same shape -> (C, C) matrix, rows = target."""
    pred = np.asarray(pred).reshape(-1); target = np.asarray(target).reshape(-1)
    mask = (target >= 0) & (target < num_classes)
    return np.bincount(num_classes * target[mask] + pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)


def pixel_accuracy(pred, target):
    pred = np.asarray(pred); target = np.asarray(target)
    return float((pred == target).mean())


def iou_per_class(pred, target, num_classes):
    cm = confusion_matrix(pred, target, num_classes)
    inter = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - np.diag(cm)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / union, np.nan)
    return iou


def mean_iou(pred, target, num_classes):
    return float(np.nanmean(iou_per_class(pred, target, num_classes)))


def dice_coefficient(pred_mask, target_mask, eps=1e-7):
    """Binary Dice on {0,1} masks (any shape)."""
    p = np.asarray(pred_mask, bool); t = np.asarray(target_mask, bool)
    inter = np.logical_and(p, t).sum()
    return float((2 * inter + eps) / (p.sum() + t.sum() + eps))


def soft_dice_loss(prob, target, eps=1e-7):
    """Differentiable Dice loss on probabilities in [0,1] and binary targets."""
    prob = np.asarray(prob, float); target = np.asarray(target, float)
    inter = (prob * target).sum()
    return float(1 - (2 * inter + eps) / (prob.sum() + target.sum() + eps))


def bce_loss(prob, target, eps=1e-7):
    prob = np.clip(np.asarray(prob, float), eps, 1 - eps); target = np.asarray(target, float)
    return float(-(target * np.log(prob) + (1 - target) * np.log(1 - prob)).mean())


def focal_loss(prob, target, gamma=2.0, alpha=0.25, eps=1e-7):
    """Binary focal loss (Lin et al. 2017), mean over elements."""
    prob = np.clip(np.asarray(prob, float), eps, 1 - eps); target = np.asarray(target, float)
    p_t = np.where(target == 1, prob, 1 - prob)
    a_t = np.where(target == 1, alpha, 1 - alpha)
    return float((-a_t * (1 - p_t) ** gamma * np.log(p_t)).mean())


def dilate_kernel(k: np.ndarray, dilation: int) -> np.ndarray:
    """Insert (dilation-1) zeros between kernel taps: the 'atrous' kernel."""
    if dilation == 1:
        return k
    kh, kw = k.shape
    out = np.zeros((dilation * (kh - 1) + 1, dilation * (kw - 1) + 1), dtype=k.dtype)
    out[::dilation, ::dilation] = k
    return out


def pixel_shuffle(x: np.ndarray, r: int) -> np.ndarray:
    """(C*r*r, H, W) -> (C, H*r, W*r), the sub-pixel convolution rearrangement."""
    C_r2, H, W = x.shape
    C = C_r2 // (r * r)
    x = x.reshape(C, r, r, H, W)
    x = x.transpose(0, 3, 1, 4, 2)         # C, H, r, W, r
    return x.reshape(C, H * r, W * r)
