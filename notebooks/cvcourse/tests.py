"""Exercise checkers. Each `check_*` takes the student's function and prints a verdict.

Pattern inherited from the original course (`tests.py` next to every notebook):
run the cell, get an immediate PASS / FAIL with a hint, no instructor needed.
"""
from __future__ import annotations

import numpy as np

from . import convops, boxes, seg, shapes

_PASS = "[PASS]"
_FAIL = "[FAIL]"


def _report(name, ok, hint=""):
    print(f"{_PASS if ok else _FAIL}  {name}" + (f"\n      hint: {hint}" if (hint and not ok) else ""))
    return ok


def _close(a, b, tol=1e-5):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return a.shape == b.shape and np.allclose(a, b, atol=tol, rtol=tol)


# ----------------------------------------------------------------- Day 1 ---
def check_output_size(fn):
    cases = [((28, 7, 1, 0, 1), 22), ((28, 3, 1, 1, 1), 28), ((32, 3, 2, 1, 1), 16),
             ((64, 5, 1, 0, 2), 56), ((7, 7, 1, 0, 1), 1), ((224, 7, 2, 3, 1), 112)]
    ok = True
    for args, want in cases:
        try:
            got = fn(*args)
        except TypeError:
            got = fn(*args[:4])
        if got != want:
            ok = False
            print(f"   n_in={args[0]}, k={args[1]}, stride={args[2]}, pad={args[3]}, dilation={args[4]}: got {got}, expected {want}")
    return _report("conv_output_size", ok, "formula: (n + 2p - d(k-1) - 1) // s + 1")


def check_cross_correlate(fn):
    rng = np.random.default_rng(0)
    ok = True
    for stride, pad in [(1, 0), (1, 1), (2, 0), (2, 1)]:
        x = rng.normal(size=(9, 11)); k = rng.normal(size=(3, 3))
        want = convops.cross_correlate2d(x, k, stride, pad)
        try:
            got = fn(x, k, stride, pad)
        except TypeError:
            got = fn(x, k)
            if (stride, pad) != (1, 0):
                continue
        if not _close(got, want):
            ok = False
            print(f"   stride={stride}, pad={pad}: shape {np.shape(got)} vs {want.shape}, or values differ")
    return _report("cross_correlate2d", ok, "slide the kernel, multiply elementwise, sum; do not flip the kernel")


def check_conv2d(fn):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 8, 8)); w = rng.normal(size=(4, 3, 3, 3)); b = rng.normal(size=4)
    ok = True
    for stride, pad in [(1, 0), (1, 1), (2, 1)]:
        want = convops.conv2d(x, w, b, stride, pad)
        got = fn(x, w, b, stride, pad)
        if not _close(got, want):
            ok = False
            print(f"   stride={stride}, pad={pad}: got shape {np.shape(got)}, expected {want.shape}")
    return _report("conv2d (multi-channel)", ok, "sum over input channels AND kernel positions, then add the bias of that output channel")


def check_receptive_field(fn):
    layers = [(3, 1, 1), (3, 1, 1), (2, 2, 0), (3, 1, 1), (3, 1, 1), (2, 2, 0), (3, 1, 1)]
    want = convops.receptive_field(layers)[-1]["rf"]
    got = fn(layers)
    got = got[-1]["rf"] if isinstance(got, list) else got
    return _report("receptive_field", int(got) == int(want), f"expected {want}: rf += (k-1)*jump; jump *= stride")


def check_pool2d(fn):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 8, 8))
    ok = _close(fn(x, 2, 2, "max"), convops.pool2d(x, 2, 2, "max")) and _close(fn(x, 2, 2, "avg"), convops.pool2d(x, 2, 2, "avg"))
    return _report("pool2d", ok, "reduce over each k x k window with the given stride")


def check_count_params(fn):
    """fn(c_in, c_out, k, bias=True) -> int"""
    cases = [((3, 64, 7, True), 9472), ((64, 64, 3, True), 36928), ((256, 64, 1, False), 16384)]
    ok = all(fn(*a) == w for a, w in cases)
    return _report("conv parameter count", ok, "c_out * c_in * k * k + c_out (bias)")


def check_inception_shapes(fn):
    """fn(c_in, f1, f3r, f3, f5r, f5, fpool) -> output channels"""
    ok = fn(192, 64, 96, 128, 16, 32, 32) == 256 and fn(256, 128, 128, 192, 32, 96, 64) == 480
    return _report("inception output channels", ok, "concatenate along channels: f1 + f3 + f5 + fpool")


# ----------------------------------------------------------------- Day 2 ---
def check_box_conversions(to_xywh, to_cxcywh, from_cxcywh):
    b = np.array([[10., 20., 50., 80.], [0., 0., 5., 5.]])
    ok = _close(to_xywh(b), boxes.xyxy_to_xywh(b)) and _close(to_cxcywh(b), boxes.xyxy_to_cxcywh(b)) \
        and _close(from_cxcywh(boxes.xyxy_to_cxcywh(b)), b)
    return _report("box format conversions", ok, "xywh: (x1, y1, x2-x1, y2-y1); cxcywh: centre and size")


def check_iou(fn):
    cases = [
        ([0, 0, 10, 10], [0, 0, 10, 10], 1.0),
        ([0, 0, 10, 10], [5, 5, 15, 15], 25 / 175),
        ([0, 0, 10, 10], [10, 10, 20, 20], 0.0),
        ([0, 0, 10, 10], [20, 20, 30, 30], 0.0),
        ([0, 0, 10, 10], [2, 2, 8, 8], 36 / 100),
    ]
    ok = True
    for a, b, want in cases:
        got = float(np.asarray(fn(np.array(a, float), np.array(b, float))).reshape(-1)[0])
        if not np.isclose(got, want, atol=1e-6):
            ok = False
            print(f"   IoU({a}, {b}) = {got:.4f}, expected {want:.4f}")
    return _report("IoU", ok, "intersection = clip(min(x2) - max(x1), 0) * clip(min(y2) - max(y1), 0); union = A + B - inter")


def check_box_iou_matrix(fn):
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 50, (6, 2)); a = np.concatenate([a, a + rng.uniform(5, 40, (6, 2))], 1)
    b = rng.uniform(0, 50, (4, 2)); b = np.concatenate([b, b + rng.uniform(5, 40, (4, 2))], 1)
    ok = _close(fn(a, b), boxes.box_iou(a, b))
    return _report("pairwise box_iou (N, M)", ok, "broadcast a[:, None, :] against b[None, :, :]")


def check_nms(fn):
    rng = np.random.default_rng(4)
    ok = True
    for _ in range(5):
        base = rng.uniform(0, 60, (3, 2))
        bx = []
        for c in base:
            for _ in range(4):
                j = rng.normal(0, 3, 2)
                bx.append(np.concatenate([c + j, c + j + 30]))
        bx = np.array(bx); sc = rng.uniform(0.1, 1.0, len(bx))
        want = boxes.nms(bx, sc, 0.5)
        got = np.asarray(fn(bx, sc, 0.5)).reshape(-1)
        if not (len(got) == len(want) and np.array_equal(np.sort(got), np.sort(want))):
            ok = False
    return _report("non-maximum suppression", ok, "pick highest score, drop everything with IoU >= threshold to it, repeat")


def check_make_anchors(fn):
    want = boxes.make_anchors((4, 6), 16, [32, 64], [0.5, 1, 2])
    got = np.asarray(fn((4, 6), 16, [32, 64], [0.5, 1, 2]), float).reshape(-1, 4)
    ok = got.shape == want.shape and np.allclose(np.sort(got, 0), np.sort(want, 0), atol=1e-4)
    return _report("make_anchors", ok, f"expected {want.shape[0]} anchors = 4*6 cells x 6 (scale, ratio) pairs; centres at (i+0.5)*stride")


def check_match_anchors(fn):
    rng = np.random.default_rng(5)
    anchors = boxes.make_anchors((8, 8), 16, [24, 48], [0.5, 1, 2])
    gt = np.array([[20, 20, 60, 70], [70, 80, 120, 120]], float)
    want_idx, want_lab = boxes.match_anchors(anchors, gt, 0.7, 0.3)
    got = fn(anchors, gt, 0.7, 0.3)
    got_idx, got_lab = got
    ok = np.array_equal(np.asarray(got_lab), want_lab) and np.array_equal(np.asarray(got_idx)[want_lab == 1], want_idx[want_lab == 1])
    return _report("match_anchors", ok, "labels: 1 if max IoU >= pos, 0 if < neg, -1 otherwise; plus each GT's best anchor is positive")


def check_encode_decode(encode, decode):
    rng = np.random.default_rng(6)
    a = boxes.make_anchors((3, 3), 32, [40], [1])
    g = a + rng.normal(0, 5, a.shape); g[:, 2:] = np.maximum(g[:, 2:], g[:, :2] + 4)
    d = encode(g, a)
    ok = _close(d, boxes.encode_boxes(g, a), 1e-4) and _close(decode(d, a), g, 1e-4)
    return _report("encode / decode boxes", ok, "tx = (gx - ax)/aw, tw = log(gw/aw); decode is the exact inverse")


def check_average_precision(fn):
    p = np.array([1.0, 1.0, 0.67, 0.75, 0.6, 0.67])
    r = np.array([0.25, 0.5, 0.5, 0.75, 0.75, 1.0])
    want = boxes.average_precision(p, r, "all")
    got = float(fn(p, r))
    return _report("average precision", np.isclose(got, want, atol=1e-3), f"expected {want:.3f}: make precision monotone from the right, then integrate over recall")


def check_focal_loss(fn):
    rng = np.random.default_rng(7)
    p = rng.uniform(0.01, 0.99, 200); t = (rng.uniform(size=200) < 0.2).astype(float)
    ok = np.isclose(fn(p, t, 2.0, 0.25), seg.focal_loss(p, t, 2.0, 0.25), atol=1e-5) and \
        np.isclose(fn(p, t, 0.0, 0.5), 0.5 * seg.bce_loss(p, t), atol=1e-5)
    return _report("focal loss", ok, "-alpha_t * (1 - p_t)^gamma * log(p_t); with gamma=0, alpha=0.5 it is half the BCE")


def check_grid_targets(fn):
    rng = np.random.default_rng(8)
    img, bx, lb, _ = shapes.make_sample(rng, 128)
    ok = _close(fn(bx, lb, 8, 128, 3), shapes.grid_targets(bx, lb, 8, 128, 3))
    return _report("dense grid targets", ok, "cell = (cy // cell_size, cx // cell_size); store objectness, normalised cxcywh, one-hot")


# ----------------------------------------------------------------- Day 3 ---
def check_transposed_conv(fn):
    rng = np.random.default_rng(9)
    x = rng.normal(size=(4, 5)); k = rng.normal(size=(3, 3))
    ok = all(_close(fn(x, k, s), convops.transposed_conv2d(x, k, s)) for s in (1, 2, 3))
    return _report("transposed_conv2d", ok, "each input pixel scatters x[i,j]*kernel at (i*stride, j*stride); overlaps add")


def check_dice(fn):
    rng = np.random.default_rng(10)
    p = rng.uniform(size=(16, 16)) > 0.5; t = rng.uniform(size=(16, 16)) > 0.5
    ok = np.isclose(fn(p, t), seg.dice_coefficient(p, t), atol=1e-6) and np.isclose(fn(t, t), 1.0, atol=1e-6)
    return _report("dice coefficient", ok, "2 * |A and B| / (|A| + |B|)")


def check_mean_iou(fn):
    rng = np.random.default_rng(11)
    p = rng.integers(0, 3, (20, 20)); t = rng.integers(0, 3, (20, 20))
    ok = np.isclose(fn(p, t, 3), seg.mean_iou(p, t, 3), atol=1e-6)
    return _report("mean IoU", ok, "per class: TP / (TP + FP + FN) from the confusion matrix, then average")


def check_unet_shapes(fn):
    """fn(input_size, depth, base_channels) -> list of (channels, size) for encoder levels + bottleneck."""
    got = fn(128, 4, 16)
    want = [(16, 128), (32, 64), (64, 32), (128, 16), (256, 8)]
    return _report("U-Net shape schedule", list(map(tuple, got)) == want, f"expected {want}")


def check_pixel_shuffle(fn):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(8, 3, 3))
    ok = _close(fn(x, 2), seg.pixel_shuffle(x, 2))
    return _report("pixel shuffle", ok, "(C*r*r, H, W) -> reshape (C, r, r, H, W) -> transpose (C, H, r, W, r) -> reshape")


# ---------------------------------------------------------------------------
# Day-specific checkers live in their own modules so they can be developed in
# parallel; they are re-exported here so notebooks only ever `from cvcourse import tests`.
# ---------------------------------------------------------------------------
for _mod in ("tests_day0", "tests_day1", "tests_day2", "tests_day3"):
    try:
        exec(f"from .{_mod} import *")  # noqa: S102
    except ImportError:
        pass
