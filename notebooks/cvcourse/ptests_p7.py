"""Checks for P7 (object detection with boxes). In the notebook: ``from cvcourse import ptests_p7 as T``.

Every check prints one line per property, ``[PASS]`` or ``[FAIL]`` with a plain-language hint, then one summary
line, and never stops the notebook with a traceback: an unfinished step (the ``raise NotImplementedError`` line
still there, or a ``____`` blank left) is reported as "not written yet". The reference versions, written exactly
like the solution cells, live in ``cvcourse.pmodels_p7``.

    T.check_box_iou_one(box_iou_one)                  step 1: the IoU of two boxes
    T.check_heads(make_class_head, make_box_head)     step 2: the two heads of the detector
    thing = T.yours_or_reference(thing)               yours if it passes its checks, else the reference (+ a note)

A check returns a ``Result`` (true when everything passed; Jupyter shows nothing for it). Checks do not change
PyTorch's random numbers, so the rest of the notebook gives the same results whether a check passed or not.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import pmodels_p7 as R
from .ptests_p3 import Result, _Lines, _try, explain, leaves

__all__ = ["check_box_iou_one", "check_heads", "yours_or_reference", "Result"]


def _with_rng(fn):
    """Run fn() without changing PyTorch's random numbers for the rest of the notebook; never raise."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  (a check must never end in a traceback)
            print(f"[FAIL]  the check could not finish: {explain(e)}")
            return Result(False, 0, 1)


def _number(value):
    """value as a float, or None if it is not one number."""
    if isinstance(value, bool):
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================================================== step 1: IoU
def _parts(a, b):
    """The pieces of the IoU computation, right and wrong, for the hints."""
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    w, h = max(0, right - left), max(0, bottom - top)
    inter = w * h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    raw_inter = (right - left) * (bottom - top)                     # without max(0, ...)
    swapped_l, swapped_t = min(a[0], b[0]), min(a[1], b[1])          # min / max the wrong way round
    swapped_r, swapped_b = max(a[2], b[2]), max(a[3], b[3])
    swapped_inter = max(0, swapped_r - swapped_l) * max(0, swapped_b - swapped_t)
    return {"inter": inter, "area_a": area_a, "area_b": area_b, "union": union, "raw_inter": raw_inter,
            "swapped_inter": swapped_inter}


def _iou_hint(got: float, a, b) -> str:
    """Which typical mistake gives the number the student's function returned."""
    p = _parts(a, b)
    right = p["inter"] / p["union"] if p["union"] else 0.0

    def same(x):
        return x is not None and abs(got - x) < 1e-6

    if same(100 * right) and right != 0:
        return "this is the IoU in percent: return the plain ratio intersection / union (between 0 and 1)"
    if same(p["inter"]) and p["inter"] != right:
        return (f"you return the intersection ({p['inter']:g} pixels): divide it by the union "
                f"(iou = intersection / union)")
    if p["inter"] and same(p["union"] / p["inter"]):
        return "the ratio is upside down: iou = intersection / union"
    if same(p["inter"] / (p["area_a"] + p["area_b"])) and p["inter"] != 0:
        return ("the union counts the overlap twice: union = area_a + area_b - intersection (the overlap belongs "
                "to both boxes)")
    if p["area_a"] and same(p["inter"] / p["area_a"]) and p["area_a"] != p["union"]:
        return "you divide by the area of box a only: divide by the union, area_a + area_b - intersection"
    if p["area_b"] and same(p["inter"] / p["area_b"]) and p["area_b"] != p["union"]:
        return "you divide by the area of box b only: divide by the union, area_a + area_b - intersection"
    if p["inter"] == 0 and p["raw_inter"] != 0:
        return ("these boxes do not overlap, so the intersection must be 0: width = max(0, right - left) and "
                "height = max(0, bottom - top) keep a negative width or height at 0")
    union_raw = p["area_a"] + p["area_b"] - p["raw_inter"]
    if union_raw and same(p["raw_inter"] / union_raw):
        return ("these boxes do not overlap, so the intersection must be 0: width = max(0, right - left) and "
                "height = max(0, bottom - top)")
    swapped_union = p["area_a"] + p["area_b"] - p["swapped_inter"]
    if swapped_union and same(p["swapped_inter"] / swapped_union):
        return ("the intersection's left and top edges are the LARGER of the two (max), its right and bottom edges "
                "the SMALLER of the two (min)")
    return (f"step by step for these boxes: left {max(a[0], b[0]):g}, top {max(a[1], b[1]):g}, right "
            f"{min(a[2], b[2]):g}, bottom {min(a[3], b[3]):g}; intersection {p['inter']:g}; areas {p['area_a']:g} "
            f"and {p['area_b']:g}; union {p['union']:g}. Compare with your lines")


IOU_CASES = [
    ("the example of the text", (2, 2, 10, 8), (4, 4, 12, 10), "24 / 72 = 0.333"),
    ("the same box twice", (3, 4, 9, 12), (3, 4, 9, 12), "1.0"),
    ("B moved right by half its width", (0, 0, 10, 10), (5, 0, 15, 10), "50 / 150 = 0.333"),
    ("a small box inside a big one", (0, 0, 20, 20), (5, 5, 15, 15), "100 / 400 = 0.25"),
    ("B to the right of A, no overlap", (0, 0, 10, 10), (20, 0, 30, 10), "0.0"),
    ("B below and to the right of A, no overlap", (0, 0, 10, 10), (20, 20, 30, 30), "0.0"),
    ("edges touching, no overlap", (0, 0, 10, 10), (10, 0, 20, 10), "0.0"),
    ("boxes of different shapes, decimal numbers", (1.5, 2.0, 7.5, 6.0), (4.0, 0.5, 9.0, 10.5), "14 / 60 = 0.233"),
]


def _iou(fn, verbose: bool) -> Result:
    lines = _Lines("Checking step 1: box_iou_one(box_a, box_b)", verbose)
    first, err = _try(fn, IOU_CASES[0][1], IOU_CASES[0][2])
    if err:
        lines.fail("box_iou_one((2, 2, 10, 8), (4, 4, 12, 10)) runs", err)
        return lines.done("1", "")
    value = _number(first)
    if not lines.check(value is not None, "it returns one number",
                       f"it returns {type(first).__name__} {first!r}: end with  return iou  where iou is "
                       f"intersection / union"):
        return lines.done("1", "")
    for name, a, b, want_text in IOU_CASES:
        p = _parts(a, b)
        want = p["inter"] / p["union"]
        got, err = _try(fn, a, b)
        title = f"{name}: box_iou_one({a}, {b}) = {want_text}"
        if err:
            lines.fail(title, err)
            continue
        got = _number(got)
        if got is None or math.isnan(got):
            lines.fail(title, "the result is not a number")
            continue
        if abs(got - want) < 1e-6:
            lines.ok(title)
        else:
            lines.fail(title, f"yours gives {got:.4g}. {_iou_hint(got, a, b)}")
    a, b = (1, 1, 7, 5), (3, 2, 11, 9)
    ab, err1 = _try(fn, a, b)
    ba, err2 = _try(fn, b, a)
    same = err1 is None and err2 is None and _number(ab) is not None and _number(ba) is not None \
        and abs(_number(ab) - _number(ba)) < 1e-9
    lines.check(same, "the order of the boxes does not matter: box_iou_one(a, b) = box_iou_one(b, a)",
                err1 or err2 or "the two orders give different numbers: use max for left and top, min for right and "
                                "bottom, for both boxes alike")
    return lines.done("1", "Your function measures how well two boxes match.")


def check_box_iou_one(box_iou_one) -> Result:
    """Step 1: box_iou_one(box_a, box_b) = intersection / union, 0 for boxes that do not overlap."""
    return _with_rng(lambda: _iou(box_iou_one, True))


# ============================================================================================== step 2: the heads
def _linear(cin, cout):
    return {"kind": "linear", "in": cin, "out": cout}


_WORDS = {"linear": "nn.Linear", "relu": "nn.ReLU()", "sigmoid": "nn.Sigmoid()"}


def _kind(m: nn.Module) -> str:
    if isinstance(m, nn.Linear):
        return "linear"
    if isinstance(m, nn.ReLU):
        return "relu"
    if isinstance(m, nn.Sigmoid):
        return "sigmoid"
    return type(m).__name__


def _describe(spec: dict) -> str:
    if spec["kind"] == "linear":
        return f"nn.Linear, {spec['in']} -> {spec['out']}"
    return _WORDS.get(spec["kind"], spec["kind"])


def _found(m: nn.Module) -> str:
    if isinstance(m, nn.Linear):
        return f"nn.Linear {m.in_features} -> {m.out_features}"
    return type(m).__name__


def _differences(spec: dict, m: nn.Module) -> list:
    out = []
    if spec["kind"] == "linear":
        if m.in_features != spec["in"]:
            out.append(f"in_features is {m.in_features}, expected {spec['in']}")
        if m.out_features != spec["out"]:
            out.append(f"out_features is {m.out_features}, expected {spec['out']}")
        if m.bias is None:
            out.append("it has no bias (bias=False): leave bias at its default")
    return out


def _cost(spec: dict, m: nn.Module) -> float:
    if spec["kind"] != _kind(m):
        return 1.5
    n = len(_differences(spec, m))
    if n == 0:
        return 0.0
    return 0.4 + 0.1 * min(n, 5)


def _align(expected: list, found: list) -> list:
    """Like ptests_p3.align, with this notebook's layer kinds (Linear, ReLU, Sigmoid)."""
    n, m = len(expected), len(found)
    cost = []
    for i in range(n + 1):
        cost.append([0.0] * (m + 1))
    for i in range(1, n + 1):
        cost[i][0] = float(i)
    for j in range(1, m + 1):
        cost[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = cost[i - 1][j - 1] + _cost(expected[i - 1], found[j - 1][1])
            best = min(best, cost[i - 1][j] + 1.0, cost[i][j - 1] + 1.0)
            cost[i][j] = best
    steps = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(cost[i][j] - cost[i - 1][j - 1] - _cost(expected[i - 1], found[j - 1][1])) < 1e-9:
            steps.append(("match", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and abs(cost[i][j] - cost[i - 1][j] - 1.0) < 1e-9:
            steps.append(("missing", i - 1, None))
            i -= 1
        else:
            steps.append(("extra", None, j - 1))
            j -= 1
    steps.reverse()
    return steps


def _kind_hint(head: str, spec: dict, m: nn.Module) -> str:
    k = _kind(m)
    if head == "class" and k in ("Softmax", "LogSoftmax"):
        return ("no softmax in the class head: it gives raw scores, as every classifier of P1 to P5 (the loss "
                "turns them into probabilities itself)")
    if head == "class" and k == "sigmoid":
        return "the class head ends with its last nn.Linear: no nn.Sigmoid() there (that belongs to the box head)"
    if spec["kind"] == "sigmoid" and k == "relu":
        return ("the last layer of the box head is nn.Sigmoid(), not nn.ReLU(): a ReLU lets numbers above 1 through, "
                "the sigmoid keeps all four between 0 and 1")
    if spec["kind"] == "sigmoid" and k in ("Softmax", "Tanh"):
        return f"found {k}: the last layer of the box head is nn.Sigmoid() (it keeps every number between 0 and 1)"
    return f"found {_found(m)} instead of {_WORDS.get(spec['kind'], spec['kind'])}"


def _compare(lines: _Lines, head: str, expected: list, module: nn.Module, name_hint: str) -> bool:
    found = leaves(module)
    all_ok = True
    for step, a, b in _align(expected, found):
        if step == "extra":
            m = found[b][1]
            hint = "the table has no such layer here: remove it"
            if _kind(m) in ("Softmax", "LogSoftmax") and head == "class":
                hint = _kind_hint(head, {"kind": "linear"}, m)
            if _kind(m) == "sigmoid" and head == "class":
                hint = _kind_hint(head, {"kind": "linear"}, m)
            lines.fail(f"{head} head: an extra layer, {_found(m)}", hint)
            all_ok = False
            continue
        title = f"{head} head, layer {a + 1}: {_describe(expected[a])}"
        if step == "missing":
            spec = expected[a]
            if spec["kind"] == "sigmoid":
                hint = ("missing: the box head ends with nn.Sigmoid(), which squeezes the 4 box numbers into 0..1 "
                        "(fractions of the picture's width and height)")
            elif spec["kind"] == "relu":
                hint = "missing: nn.ReLU() between the two nn.Linear layers"
            else:
                hint = f"missing: the table has {_describe(spec)} here, your code has none"
            lines.fail(title, hint)
            all_ok = False
            continue
        m = found[b][1]
        if _kind(m) != expected[a]["kind"]:
            lines.fail(title, _kind_hint(head, expected[a], m))
            all_ok = False
            continue
        problems = _differences(expected[a], m)
        hint = "; ".join(problems)
        if problems and expected[a]["kind"] == "linear" and expected[a]["out"] == 4 and m.out_features != 4:
            hint += ". The box head gives 4 numbers: x1, y1, x2, y2"
        if problems and name_hint:
            hint += ". " + name_hint
        all_ok &= lines.check(not problems, title, hint)
    return all_ok


def _params(m: nn.Module) -> int:
    total = 0
    for p in m.parameters():
        total += p.numel()
    return total


def _class_spec(cin, n):
    return [_linear(cin, 64), {"kind": "relu"}, _linear(64, n)]


def _box_spec(cin):
    return [_linear(cin, 64), {"kind": "relu"}, _linear(64, 4), {"kind": "sigmoid"}]


def _same_layers(expected: list, module) -> str:
    if not isinstance(module, nn.Module):
        return f"it returns {type(module).__name__}, not the nn.Sequential"
    found = leaves(module)
    for step, a, b in _align(expected, found):
        if step == "extra":
            return f"there is an extra layer, {_found(found[b][1])}"
        if step == "missing":
            return f"layer {a + 1} ({_describe(expected[a])}) is missing"
        m = found[b][1]
        if _kind(m) != expected[a]["kind"]:
            return f"layer {a + 1} should be {_describe(expected[a])}, found {_found(m)}"
        problems = _differences(expected[a], m)
        if problems:
            return f"layer {a + 1} ({_describe(expected[a])}): {problems[0]}"
    return ""


def _run(module, x):
    module.eval()
    with torch.no_grad():
        return module(x)


def _class_head(fn, verbose: bool) -> Result:
    lines = _Lines("Checking make_class_head (in_features=1024, num_classes=3)", verbose)
    head, err = _try(fn, 1024, 3)
    if not lines.check(err is None and isinstance(head, nn.Module), "make_class_head(1024, 3) can be built",
                       err or f"it returns {type(head).__name__}: end with  return head"):
        return lines.done("2", "")
    ok = _compare(lines, "class", _class_spec(1024, 3), head, "")
    out, err = _try(_run, head, torch.randn(2, 1024))
    shape_ok = err is None and torch.is_tensor(out) and tuple(out.shape) == (2, 3)
    hint = err or (f"the output is {tuple(out.shape) if torch.is_tensor(out) else type(out).__name__}, "
                   f"expected (2, 3)")
    lines.check(shape_ok, "2 pictures' 1024 numbers in -> 2 x 3 class scores out", hint)
    got = _params(head)
    lines.check(got == 65_795, "65,795 parameters (1024 x 64 + 64, then 64 x 3 + 3)",
                f"yours has {got:,}, expected 65,795")
    other, err = _try(fn, 256, 5)
    if err:
        lines.fail("also right with other numbers: make_class_head(256, 5)", err)
    else:
        diff = _same_layers(_class_spec(256, 5), other)
        more = ("Use the names in_features and num_classes, not the numbers 1024 and 3." if ok
                else "Fix the [FAIL] lines above first.")
        lines.check(not diff, "also right with other numbers: make_class_head(256, 5)", f"with these numbers {diff}. {more}")
    return lines.done("2", "")


def _box_head(fn, verbose: bool) -> Result:
    lines = _Lines("Checking make_box_head (in_features=1024)", verbose)
    head, err = _try(fn, 1024)
    if not lines.check(err is None and isinstance(head, nn.Module), "make_box_head(1024) can be built",
                       err or f"it returns {type(head).__name__}: end with  return head"):
        return lines.done("2", "")
    ok = _compare(lines, "box", _box_spec(1024), head, "")
    x = torch.randn(2, 1024) * 50.0                     # large inputs: only a sigmoid keeps the output in 0..1
    out, err = _try(_run, head, x)
    shape_ok = err is None and torch.is_tensor(out) and tuple(out.shape) == (2, 4)
    hint = err or (f"the output is {tuple(out.shape) if torch.is_tensor(out) else type(out).__name__}, "
                   f"expected (2, 4)")
    lines.check(shape_ok, "2 pictures' 1024 numbers in -> 2 x 4 box numbers out", hint)
    if shape_ok:
        inside = bool(((out >= 0) & (out <= 1)).all())
        lines.check(inside, "every box number is between 0 and 1, even for large inputs",
                    f"some numbers lie outside 0..1 (from {float(out.min()):.2f} to {float(out.max()):.2f}): the box "
                    f"head must end with nn.Sigmoid()")
    got = _params(head)
    lines.check(got == 65_860, "65,860 parameters (1024 x 64 + 64, then 64 x 4 + 4)",
                f"yours has {got:,}, expected 65,860")
    other, err = _try(fn, 256)
    if err:
        lines.fail("also right with other numbers: make_box_head(256)", err)
    else:
        diff = _same_layers(_box_spec(256), other)
        more = "Use the name in_features, not the number 1024." if ok else "Fix the [FAIL] lines above first."
        lines.check(not diff, "also right with other numbers: make_box_head(256)", f"with these numbers {diff}. {more}")
    return lines.done("2", "")


def check_heads(make_class_head, make_box_head) -> Result:
    """Step 2: the class head (Linear, ReLU, Linear) and the box head (Linear, ReLU, Linear, Sigmoid)."""
    def run():
        print("Checking step 2: the two heads")
        print()
        a = _class_head(make_class_head, True)
        print()
        b = _box_head(make_box_head, True)
        passed = bool(a) and bool(b)
        print()
        if passed:
            print("=> step 2 is done: both heads pass. Next: the whole detector.")
        else:
            print("=> step 2 is not finished: fix the [FAIL] lines above, run your cell, then this check again.")
        return Result(passed, a.n_pass + b.n_pass, a.n_fail + b.n_fail)
    return _with_rng(run)


# ============================================================================================== never stuck
_STEPS = {
    "box_iou_one": (_iou, R.box_iou_one, "1"),
    "make_class_head": (_class_head, R.make_class_head, "2"),
    "make_box_head": (_box_head, R.make_box_head, "2"),
}


def yours_or_reference(yours, name: str | None = None):
    """Your function if it passes all its checks; otherwise the course's reference version, with a note.

    So that the rest of the notebook works while a step is unfinished: use it as
    ``box_iou_one = T.yours_or_reference(box_iou_one)``. Fix your cell later, run it, its check and this line again.
    """
    name = name or getattr(yours, "__name__", "")
    if name not in _STEPS:
        raise ValueError(f"yours_or_reference knows {', '.join(_STEPS)}; got {name!r}")
    check, reference, step = _STEPS[name]
    if yours is reference:
        print(f"note: still using the reference {name}: once your step {step} passes, run your cell, its check and "
              f"this cell again")
        return reference
    result = _with_rng(lambda: check(yours, False))
    if result:
        print(f"using your {name} (it passes all its checks)")
        return yours
    print(f"note: using the reference {name} so you can continue; come back to step {step} later "
          f"(your version does not pass its checks yet)")
    return reference
