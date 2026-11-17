"""Tests for P7 (object detection with boxes): checks, reference code and helpers. Plain asserts, no pytest needed.

    python tools/test_p7.py            # about half a minute on a laptop CPU; no download

Covers: the reference answers pass their checks; typical wrong answers fail with a hint that names the mistake;
yours_or_reference falls back on the reference (also for untouched stubs); the generated boxes lie inside the
pictures and fit the shapes; a few training steps reduce the losses of both detectors; NMS and the drawings run;
the photo helpers degrade gracefully.
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from cvcourse import pkit_p7 as D  # noqa: E402
from cvcourse import pmodels_p7 as R  # noqa: E402
from cvcourse import ptests_p7 as T  # noqa: E402

RESULTS: list = []
T0 = time.time()


def check(name):
    def wrap(fn):
        t0 = time.time()
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:300]}")
            return fn
        RESULTS.append((name, True, ""))
        print(f"[PASS] {name} ({time.time() - t0:.1f} s)")
        return fn
    return wrap


def run_quiet(fn, *args):
    """(return value, printed text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        value = fn(*args)
    return value, buf.getvalue()


def assert_ascii(text: str):
    text.encode("ascii")                       # the Windows console is cp1250: checker output must be ASCII


# ------------------------------------------------------------------ step 1: IoU
def iou_stub(box_a, box_b):
    raise NotImplementedError   # delete this line once every ____ is filled in


def iou_no_clamp(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    w = min(ax2, bx2) - max(ax1, bx1)
    h = min(ay2, by2) - max(ay1, by1)
    inter = w * h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union


def iou_double_count(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    w = max(0, min(ax2, bx2) - max(ax1, bx1))
    h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = w * h
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1))


def iou_intersection_only(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    w = max(0, min(ax2, bx2) - max(ax1, bx1))
    h = max(0, min(ay2, by2) - max(ay1, by1))
    return w * h


def iou_swapped_minmax(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    w = max(0, max(ax2, bx2) - min(ax1, bx1))
    h = max(0, max(ay2, by2) - min(ay1, by1))
    inter = w * h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union


def iou_blank(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    top = ____  # noqa: F821  (a blank left in, as in the student notebook)
    return top


@check("check_box_iou_one: the reference passes; output is ASCII")
def _():
    result, text = run_quiet(T.check_box_iou_one, R.box_iou_one)
    assert result, text
    assert "[FAIL]" not in text and "all 10 checks passed" in text, text
    assert_ascii(text)


@check("check_box_iou_one: the untouched stub says 'not written yet'")
def _():
    result, text = run_quiet(T.check_box_iou_one, iou_stub)
    assert not result
    assert "not written yet" in text, text


@check("check_box_iou_one: a ____ blank left in is named")
def _():
    result, text = run_quiet(T.check_box_iou_one, iou_blank)
    assert not result
    assert "____ blank is still empty" in text, text


@check("check_box_iou_one: typical mistakes fail with a hint that names them")
def _():
    cases = [
        (iou_no_clamp, "max(0, right - left)"),
        (iou_double_count, "counts the overlap twice"),
        (iou_intersection_only, "you return the intersection"),
        (iou_swapped_minmax, "LARGER of the two (max)"),
    ]
    for fn, words in cases:
        result, text = run_quiet(T.check_box_iou_one, fn)
        assert not result, fn.__name__
        assert words in text, (fn.__name__, text)
        assert_ascii(text)


@check("box_iou_one: the reference agrees with torchvision.ops.box_iou on random boxes")
def _():
    import torchvision
    g = torch.Generator().manual_seed(0)
    xy = torch.rand(50, 2, generator=g) * 40
    wh = torch.rand(50, 2, generator=g) * 30 + 1
    boxes = torch.cat([xy, xy + wh], dim=1)
    lib = torchvision.ops.box_iou(boxes, boxes)
    for i in range(0, 50, 7):
        for j in range(0, 50, 5):
            mine = R.box_iou_one(boxes[i].tolist(), boxes[j].tolist())
            assert abs(mine - float(lib[i, j])) < 1e-5, (i, j)
    mine = D.box_iou(boxes[:10], boxes[10:20])
    assert torch.allclose(mine, lib[:10, 10:20].diagonal(), atol=1e-6)


# ------------------------------------------------------------------ step 2: heads
def class_head_stub(in_features, num_classes):
    raise NotImplementedError   # delete this line once every ____ is filled in


def box_head_stub(in_features):
    raise NotImplementedError   # delete this line once every ____ is filled in


def box_head_no_sigmoid(in_features):
    return nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, 4))


def box_head_three_numbers(in_features):
    return nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, 3), nn.Sigmoid())


def box_head_hard_coded(in_features):
    return nn.Sequential(nn.Linear(1024, 64), nn.ReLU(), nn.Linear(64, 4), nn.Sigmoid())


def class_head_softmax(in_features, num_classes):
    return nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, num_classes), nn.Softmax(dim=1))


def class_head_no_relu(in_features, num_classes):
    return nn.Sequential(nn.Linear(in_features, 64), nn.Linear(64, num_classes))


def class_head_hard_coded(in_features, num_classes):
    return nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, 3))


@check("check_heads: the reference passes; output is ASCII")
def _():
    result, text = run_quiet(T.check_heads, R.make_class_head, R.make_box_head)
    assert result, text
    assert "[FAIL]" not in text and "step 2 is done" in text, text
    assert_ascii(text)


@check("check_heads: the untouched stubs say 'not written yet'")
def _():
    result, text = run_quiet(T.check_heads, class_head_stub, box_head_stub)
    assert not result
    assert text.count("not written yet") == 2, text


@check("check_heads: typical mistakes fail with a hint that names them")
def _():
    cases = [
        (R.make_class_head, box_head_no_sigmoid, "ends with nn.Sigmoid()"),
        (R.make_class_head, box_head_three_numbers, "The box head gives 4 numbers"),
        (R.make_class_head, box_head_hard_coded, "not the number 1024"),
        (class_head_softmax, R.make_box_head, "no softmax in the class head"),
        (class_head_no_relu, R.make_box_head, "nn.ReLU() between the two nn.Linear layers"),
        (class_head_hard_coded, R.make_box_head, "Use the names in_features and num_classes"),
    ]
    for class_fn, box_fn, words in cases:
        result, text = run_quiet(T.check_heads, class_fn, box_fn)
        assert not result, (class_fn.__name__, box_fn.__name__)
        assert words in text, (class_fn.__name__, box_fn.__name__, text)
        assert_ascii(text)


# ------------------------------------------------------------------ never stuck
@check("yours_or_reference: keeps a passing version, falls back on stubs and wrong answers, rejects other names")
def _():
    def box_iou_one(a, b):
        return R.box_iou_one(a, b)
    got, text = run_quiet(T.yours_or_reference, box_iou_one)
    assert got is box_iou_one and "using your box_iou_one" in text, text
    iou_stub.__name__ = "box_iou_one"
    got, text = run_quiet(T.yours_or_reference, iou_stub)
    assert got is R.box_iou_one and "note: using the reference box_iou_one" in text, text
    box_head_no_sigmoid.__name__ = "make_box_head"
    got, _ = run_quiet(T.yours_or_reference, box_head_no_sigmoid)
    assert got is R.make_box_head
    class_head_stub.__name__ = "make_class_head"
    got, _ = run_quiet(T.yours_or_reference, class_head_stub)
    assert got is R.make_class_head
    got, text = run_quiet(T.yours_or_reference, R.make_class_head)
    assert got is R.make_class_head and "still using the reference" in text
    try:
        T.yours_or_reference(lambda x: x, "something_else")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown name must raise ValueError")


@check("checks do not change PyTorch's random numbers")
def _():
    torch.manual_seed(5)
    a = torch.rand(3)
    torch.manual_seed(5)
    run_quiet(T.check_heads, R.make_class_head, R.make_box_head)
    run_quiet(T.check_box_iou_one, R.box_iou_one)
    b = torch.rand(3)
    assert torch.equal(a, b)


# ------------------------------------------------------------------ data
@check("one_object_set: shapes, one box per picture, boxes inside and tight around the shape, repeatable")
def _():
    s = D.one_object_set(200, seed=3)
    assert s.images.shape == (200, 3, 64, 64) and s.labels.shape == (200,) and s.boxes.shape == (200, 4)
    assert float(s.images.min()) >= 0 and float(s.images.max()) <= 1
    b = s.boxes
    assert bool((b[:, 0] >= 0).all() and (b[:, 1] >= 0).all() and (b[:, 2] <= 64).all() and (b[:, 3] <= 64).all())
    assert bool((b[:, 2] - b[:, 0] >= 8).all() and (b[:, 3] - b[:, 1] >= 8).all())
    assert set(s.labels.tolist()) == {0, 1, 2}
    for i in range(20):                                   # the box is tight: the shape touches all four edges
        x1, y1, x2, y2 = [int(v) for v in b[i].tolist()]
        background = s.images[i, :, 0, 0]
        inside = (s.images[i] - background[:, None, None]).abs().sum(0) > 0.25
        assert bool(inside[y1:y2, x1].any()) and bool(inside[y1:y2, x2 - 1].any())
        assert bool(inside[y1, x1:x2].any()) and bool(inside[y2 - 1, x1:x2].any())
    again = D.one_object_set(200, seed=3)
    assert torch.equal(s.images, again.images) and torch.equal(s.boxes, again.boxes)
    other = D.one_object_set(200, seed=4)
    assert not torch.equal(s.images, other.images)


@check("many_objects_set: 1 to 3 shapes, boxes inside the picture and not overlapping")
def _():
    s = D.many_objects_set(300, seed=1)
    counts = set()
    for boxes, labels in zip(s.boxes, s.labels):
        counts.add(len(boxes))
        assert len(boxes) == len(labels)
        assert bool((boxes[:, :2] >= 0).all() and (boxes[:, 2:] <= 64).all())
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert float(D.box_iou(boxes[i:i + 1], boxes[j:j + 1])[0]) == 0.0
    assert counts == {1, 2, 3}


@check("example_picture: three shapes of three classes, whole-number boxes")
def _():
    picture, boxes, labels = D.example_picture()
    assert picture.shape == (3, 64, 64)
    assert len(boxes) == 3 and sorted(labels) == [0, 1, 2]
    for box in boxes:
        assert all(isinstance(v, int) for v in box)


@check("grid_targets: every shape gets at least one responsible cell; distances rebuild the box")
def _():
    s = D.many_objects_set(100, seed=2)
    cx, cy = D._cell_centres()
    for boxes, labels in zip(s.boxes, s.labels):
        obj, lab, dist = D.grid_targets(boxes, labels)
        for box, label in zip(boxes.tolist(), labels.tolist()):
            cells = (lab == label) & (obj > 0)
            assert bool(cells.any())
        rows, cols = torch.nonzero(obj > 0, as_tuple=True)
        for r, c in zip(rows.tolist(), cols.tolist()):
            rebuilt = [float(cx[r, c] - dist[0, r, c] * 64), float(cy[r, c] - dist[1, r, c] * 64),
                       float(cx[r, c] + dist[2, r, c] * 64), float(cy[r, c] + dist[3, r, c] * 64)]
            ious = D.box_iou(torch.tensor([rebuilt]).expand(len(boxes), 4), boxes)
            assert float(ious.max()) > 0.999


# ------------------------------------------------------------------ models and training
@check("TinyDetector: two outputs of the right shapes; a few training steps reduce both losses")
def _():
    import torch.nn.functional as F
    torch.manual_seed(0)
    model = R.TinyDetector()
    s = D.one_object_set(256, seed=0)
    scores, box = model(s.images[:5])
    assert scores.shape == (5, 3) and box.shape == (5, 4)
    assert bool(((box >= 0) & (box <= 1)).all())
    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    x, y, b = s.images[:64], s.labels[:64], s.boxes[:64] / 64

    def losses():
        sc, bx = model(x)
        return F.cross_entropy(sc, y), F.l1_loss(bx, b)
    c0, b0 = losses()
    for _ in range(40):
        c, bl = losses()
        loss = c + 5 * bl
        opt.zero_grad()
        loss.backward()
        opt.step()
    c1, b1 = losses()
    c0, b0, c1, b1 = c0.item(), b0.item(), c1.item(), b1.item()
    # the box loss falls first (the class loss plateaus for a while, as in the notebook's curves)
    assert c1 < c0 and b1 < 0.8 * b0 and c1 + 5 * b1 < 0.8 * (c0 + 5 * b0), (c0, c1, b0, b1)


@check("train_detector and evaluate_detector run for one short epoch and report sensible numbers")
def _():
    torch.manual_seed(0)
    model = R.TinyDetector()
    train = D.one_object_set(300, seed=0)
    val = D.one_object_set(60, seed=1)
    history, text = run_quiet(D.train_detector, model, train, val, 2, 0.002, 5.0)
    assert len(history["epoch"]) == 2 and history["loss"][1] < history["loss"][0], history
    assert "epoch  2/2" in text, text
    result, text = run_quiet(D.evaluate_detector, model, val)
    assert 0 <= result["accuracy"] <= 100 and 0 <= result["mean_iou"] <= 1, result
    assert_ascii(text)
    run_quiet(D.show_detector_predictions, model, val, 6)


@check("grid detector: output 8 x 8 x 8; a few training steps reduce the loss; candidates, NMS, drawings")
def _():
    torch.manual_seed(0)
    net = R.make_grid_net()
    s = D.many_objects_set(128, seed=0)
    out = net(s.images[:2])
    assert out.shape == (2, 8, 8, 8)
    obj, lab, dist = D._all_grid_targets(s)
    opt = torch.optim.Adam(net.parameters(), lr=0.002)
    first = None
    for step in range(30):
        loss, _, _, _ = D._grid_loss(net(s.images[:64]), obj[:64], lab[:64], dist[:64], 5.0)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    last, _, _, _ = D._grid_loss(net(s.images[:64]), obj[:64], lab[:64], dist[:64], 5.0)
    assert last.item() < 0.7 * first, (first, last.item())
    boxes, scores, labels = D.grid_candidates(net, s.images[0])
    assert boxes.shape == (64, 4) and scores.shape == (64,) and labels.shape == (64,)
    b, sc, lb = D.final_detections(boxes, scores, labels, 0.0, 0.5)
    assert len(b) <= 64 and bool((sc[:-1] >= sc[1:]).all())
    run_quiet(D.show_before_after, s.images[0], boxes, labels, scores, scores > 0.5, "a", "b")
    result, text = run_quiet(D.evaluate_grid, net, s, 0.5, 0.5)
    assert result["objects"] > 0
    assert_ascii(text)
    run_quiet(D.show_grid_results, net, s, 0.5, 0.5, 4)
    run_quiet(D.train_grid_detector, net, D.many_objects_set(64, seed=5), D.many_objects_set(32, seed=6), 1)


@check("drawings: boxes, IoU pictures, sets, grid idea (also with an unfinished IoU function)")
def _():
    picture, boxes, labels = D.example_picture()
    run_quiet(D.show_boxes, picture, boxes, labels)
    run_quiet(D.show_iou, (2, 2, 10, 8), (4, 4, 12, 10))
    run_quiet(D.show_iou_pairs, R.box_iou_one)
    run_quiet(D.show_iou_pairs, iou_stub)
    run_quiet(D.show_set, D.one_object_set(8, seed=0), 8)
    run_quiet(D.show_set, D.many_objects_set(8, seed=0), 8)
    run_quiet(D.show_grid_idea, picture, boxes, labels)


@check("photos: missing detector or photo gives a message, not an error")
def _():
    result, text = run_quiet(D.detect, None, None)
    assert len(result["boxes"]) == 0 and "skipped" in text
    _, text = run_quiet(D.print_result, result, None)
    assert "boxes: (0, 4)" in text
    _, text = run_quiet(D.show_photo_detections, None, result, None)
    assert "skipped" in text
    photo, people = run_quiet(D.load_photo, "street", 0)[0]
    if photo is not None:
        assert photo.shape[0] == 3 and float(photo.max()) <= 1
        assert people is not None and people.shape[1] == 4 and len(people) >= 1
        run_quiet(D.show_photo_detections, photo, result, None, 0.5, people)
    photo, people = run_quiet(D.load_photo, "pets", 0)[0]
    if photo is not None:
        assert people is None


print()
n_fail = 0
for _name, ok, _msg in RESULTS:
    if not ok:
        n_fail += 1
print(f"{len(RESULTS) - n_fail} of {len(RESULTS)} tests passed in {time.time() - T0:.0f} s")
sys.exit(1 if n_fail else 0)
