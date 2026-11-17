"""Checks and helpers for P3 (GoogLeNet step by step). In the notebook: ``from cvcourse import ptests_p3 as T``.

Every check prints one line per property, ``[PASS]`` or ``[FAIL]`` with a plain-language hint, then one summary
line, and never stops the notebook with a traceback: an unfinished step (the ``raise NotImplementedError`` line
still there, or a ``____`` blank left) is reported as "not written yet". The reference versions, written exactly
like the solution cells, live in ``cvcourse.pmodels_p3``.

    T.check_conv_relu(conv_relu)                        step (a), given complete
    T.check_branches_1_2(make_branch1, make_branch2)    step (b)
    T.check_branches_3_4(make_branch3, make_branch4)    step (c)
    T.check_inception(Inception)                        step (d)
    T.check_stem(make_stem)                             step (e)
    T.check_inception_blocks(make_inception_blocks)     step (f)
    T.check_head(make_head)                             step (g)
    T.check_googlenet(GoogLeNet)                        step (h), also against torchvision's GoogLeNet
    thing = T.yours_or_reference(thing)                 yours if it passes its checks, else the reference (+ a note)
    T.show_params_per_block(net)                        bar chart: parameters of the stem, of each block, of the head

A check returns a ``Result`` (true when everything passed; Jupyter shows nothing for it). Checks do not change
PyTorch's random numbers, so the rest of the notebook gives the same results whether a check passed or not.
"""
from __future__ import annotations

import itertools
import traceback

import torch
import torch.nn as nn

from . import pmodels_p3 as R

__all__ = [
    "check_conv_relu", "check_branches_1_2", "check_branches_3_4", "check_inception", "check_stem",
    "check_inception_blocks", "check_head", "check_googlenet", "yours_or_reference", "show_params_per_block",
    "Result",
]

BLOCK_3A = R.BLOCKS["inception3a"]          # (192, 64, 96, 128, 16, 32, 32)
BLOCK_4B = R.BLOCKS["inception4b"]          # (512, 160, 112, 224, 24, 64, 64): the second set of numbers
COLUMNS = ("in_channels", "c1", "c3_reduce", "c3", "c5_reduce", "c5", "pool_proj")
NOT_WRITTEN = "not written yet: fill in the ____ blanks and delete the raise line"


# ============================================================================================== reporting
class Result:
    """What a check returns: true when every line passed. Prints nothing itself in Jupyter."""

    def __init__(self, passed: bool, n_pass: int, n_fail: int):
        self.passed, self.n_pass, self.n_fail = passed, n_pass, n_fail

    def __bool__(self):
        return self.passed

    def __repr__(self):
        return f"Result(passed={self.passed}, {self.n_pass} passed, {self.n_fail} failed)"

    def _ipython_display_(self):
        pass                                 # the lines are already printed


class _Lines:
    """Collects [PASS] / [FAIL] lines; prints them only when verbose."""

    def __init__(self, title: str, verbose: bool):
        self.verbose = verbose
        self.n_pass = 0
        self.n_fail = 0
        if verbose:
            print(title)

    def ok(self, name: str):
        self.n_pass += 1
        if self.verbose:
            print(f"[PASS]  {name}")

    def fail(self, name: str, hint: str):
        self.n_fail += 1
        if self.verbose:
            print(f"[FAIL]  {name}")
            if hint:
                print(f"        hint: {hint}")

    def check(self, condition: bool, name: str, hint: str = "") -> bool:
        if condition:
            self.ok(name)
        else:
            self.fail(name, hint)
        return bool(condition)

    def section(self, text: str):
        if self.verbose:
            print(text)

    def done(self, step: str, what: str) -> Result:
        passed = self.n_fail == 0 and self.n_pass > 0
        if self.verbose:
            if passed:
                print(f"=> all {self.n_pass} checks passed: step {step} is done. {what}")
            else:
                total = self.n_pass + self.n_fail
                print(f"=> {self.n_fail} of {total} checks failed: fix the [FAIL] lines in your step {step} cell, "
                      f"run it, then run this check again.")
        return Result(passed, self.n_pass, self.n_fail)


def _first_line(e: BaseException) -> str:
    text = str(e).strip()
    if not text:
        return type(e).__name__
    return text.splitlines()[0]


def _where(e: BaseException) -> str:
    """' (in make_branch2, line 7 of its cell)': the last place in the student's own code."""
    mine = None
    for frame in traceback.extract_tb(e.__traceback__):
        f = frame.filename.replace("\\", "/")
        if "/torch/" in f or "/cvcourse/" in f or "site-packages" in f or "/lib/python" in f.lower():
            continue
        mine = frame
    if mine is None:
        return ""
    if mine.name == "<module>":
        return f" (line {mine.lineno} of the cell)"
    return f" (in {mine.name}, line {mine.lineno} of its cell)"


def explain(e: BaseException) -> str:
    """A plain-language reason for an error raised by the student's code."""
    msg = _first_line(e)
    if isinstance(e, NotImplementedError):
        return NOT_WRITTEN
    if isinstance(e, NameError):
        if "'____'" in msg:
            return ("a ____ blank is still empty" + _where(e) + ": replace every ____ with what its # YOUR CODE "
                    "comment asks for")
        return (f"Python does not know a name ({msg}){_where(e)}: check the spelling, and run all the cells above "
                f"this one first")
    if isinstance(e, TypeError) and "cat()" in msg:
        return ("torch.cat takes ONE list of tensors and the axis: torch.cat([y1, y2, y3, y4], dim=1)" + _where(e))
    if isinstance(e, TypeError):
        return (f"Python stopped with a TypeError{_where(e)}: {msg}. Compare the brackets and the arguments of "
                f"that line with the table")
    if isinstance(e, RuntimeError):
        if "Sizes of tensors must match except in dimension 1" in msg:
            return ("the four branch outputs do not have the same height and width, so they cannot be stacked "
                    "(3x3 convolutions need padding 1, 5x5 need padding 2, the max-pool stride 1 and padding 1)"
                    + _where(e))
        if "Sizes of tensors must match except in dimension" in msg:
            return ("torch.cat stacks along the wrong axis: use dim=1, the channel axis (axis 0 counts the images "
                    "of the batch)" + _where(e))
        if "must match the size of tensor" in msg:
            return (f"tensors of different sizes cannot be added with +{_where(e)} ({msg}). The block stacks the "
                    f"branch outputs with torch.cat([y1, y2, y3, y4], dim=1) instead")
        if "channels, but got" in msg or "to have" in msg and "channels" in msg:
            return (f"a convolution receives a different number of channels than it was built for{_where(e)}: "
                    f"{msg}. The in channels of each layer must equal the out channels of the layer before it")
        if "mat1 and mat2" in msg:
            return (f"the linear layer receives a different number of numbers than it was built for{_where(e)}: "
                    f"{msg}")
        if "Output size is too small" in msg or "too small" in msg:
            return f"the picture became too small inside the network{_where(e)}: {msg}. Check strides and padding"
    return f"Python stopped with {type(e).__name__}{_where(e)}: {msg}"


def _try(fn, *args, **kwargs):
    """(value, None) or (None, plain-language reason)."""
    try:
        return fn(*args, **kwargs), None
    except Exception as e:  # noqa: BLE001  (anything the student's code raises becomes a hint)
        return None, explain(e)


def _shape(t) -> str:
    return " x ".join(str(int(v)) for v in t)


def _params(m: nn.Module) -> int:
    total = 0
    for p in m.parameters():
        total += p.numel()
    return total


def _run(model: nn.Module, x: torch.Tensor):
    model.eval()
    with torch.no_grad():
        return model(x)


# ============================================================================================== layers
_KIND_WORDS = {"conv": "a convolution (Conv2d)", "relu": "a ReLU", "maxpool": "a max-pool (MaxPool2d)",
               "gap": "a global average pooling (AdaptiveAvgPool2d)", "flatten": "a Flatten",
               "dropout": "a Dropout", "linear": "a linear layer (Linear)"}


def _kind(m: nn.Module) -> str:
    if isinstance(m, nn.Conv2d):
        return "conv"
    if isinstance(m, nn.ReLU):
        return "relu"
    if isinstance(m, nn.MaxPool2d):
        return "maxpool"
    if isinstance(m, nn.AdaptiveAvgPool2d):
        return "gap"
    if isinstance(m, nn.Flatten):
        return "flatten"
    if isinstance(m, nn.Dropout):
        return "dropout"
    if isinstance(m, nn.Linear):
        return "linear"
    return type(m).__name__


def conv(cin, cout, k, s=1, p=0):
    return {"kind": "conv", "in": cin, "out": cout, "k": k, "s": s, "p": p}


def relu():
    return {"kind": "relu"}


def maxpool(k, s, p):
    return {"kind": "maxpool", "k": k, "s": s, "p": p}


def describe(spec: dict) -> str:
    """The expected layer in words, as in the notebook's tables."""
    k = spec["kind"]
    if k == "conv":
        return (f"{spec['k']}x{spec['k']} convolution, {spec['in']} -> {spec['out']} channels, stride {spec['s']}, "
                f"padding {spec['p']}")
    if k == "relu":
        return "ReLU"
    if k == "maxpool":
        return f"{spec['k']}x{spec['k']} max-pool, stride {spec['s']}, padding {spec['p']}"
    if k == "gap":
        return "global average pooling to 1 x 1, nn.AdaptiveAvgPool2d(1)"
    if k == "flatten":
        return "nn.Flatten()"
    if k == "dropout":
        return f"dropout with p = {spec['p']}"
    if k == "linear":
        return f"linear layer, {spec['in']} -> {spec['out']}"
    return k


def _pair(v):
    if isinstance(v, (tuple, list)):
        return tuple(v)
    return (v, v)


def _num(v) -> str:
    t = _pair(v)
    if t[0] == t[1]:
        return str(t[0])
    return " x ".join(str(a) for a in t)


def found_text(m: nn.Module) -> str:
    """The student's layer in words."""
    k = _kind(m)
    if k == "conv":
        return (f"Conv2d {m.in_channels} -> {m.out_channels}, kernel {_num(m.kernel_size)}, stride {_num(m.stride)}, "
                f"padding {m.padding if isinstance(m.padding, str) else _num(m.padding)}")
    if k == "maxpool":
        return f"MaxPool2d kernel {_num(m.kernel_size)}, stride {_num(m.stride)}, padding {_num(m.padding)}"
    if k == "gap":
        return f"AdaptiveAvgPool2d to {_num(m.output_size)}"
    if k == "dropout":
        return f"Dropout p = {m.p}"
    if k == "linear":
        return f"Linear {m.in_features} -> {m.out_features}"
    if k in ("relu", "flatten"):
        return type(m).__name__
    if isinstance(m, nn.AvgPool2d):
        return f"AvgPool2d kernel {_num(m.kernel_size)}"
    return type(m).__name__


def _differences(spec: dict, m: nn.Module) -> list:
    """What differs between an expected layer and the student's layer of the same kind."""
    out = []
    k = spec["kind"]
    if k == "conv":
        if m.in_channels != spec["in"]:
            out.append(f"in channels is {m.in_channels}, expected {spec['in']}")
        if m.out_channels != spec["out"]:
            out.append(f"out channels is {m.out_channels}, expected {spec['out']}")
        if _pair(m.kernel_size) != (spec["k"], spec["k"]):
            out.append(f"kernel_size is {_num(m.kernel_size)}, expected {spec['k']} (a {spec['k']}x{spec['k']} "
                       f"convolution)")
        if _pair(m.stride) != (spec["s"], spec["s"]):
            out.append(f"stride is {_num(m.stride)}, expected {spec['s']}")
        if isinstance(m.padding, str) or _pair(m.padding) != (spec["p"], spec["p"]):
            pad = m.padding if isinstance(m.padding, str) else _num(m.padding)
            out.append(f"padding is {pad}, expected {spec['p']}")
        if m.bias is None:
            out.append("it has no bias (bias=False): leave bias at its default")
        if m.groups != 1 or _pair(m.dilation) != (1, 1):
            out.append("groups and dilation must stay at their defaults")
    elif k == "maxpool":
        stride = m.stride if m.stride is not None else m.kernel_size
        if _pair(m.kernel_size) != (spec["k"], spec["k"]):
            out.append(f"kernel_size is {_num(m.kernel_size)}, expected {spec['k']}")
        if _pair(stride) != (spec["s"], spec["s"]):
            word = " (without stride=..., a max-pool steps by its kernel size)" if stride == m.kernel_size else ""
            out.append(f"stride is {_num(stride)}, expected {spec['s']}{word}")
        if _pair(m.padding) != (spec["p"], spec["p"]):
            out.append(f"padding is {_num(m.padding)}, expected {spec['p']}")
        if m.ceil_mode:
            out.append("ceil_mode must stay False")
    elif k == "gap":
        if _pair(m.output_size) != (1, 1):
            out.append(f"output size is {_num(m.output_size)}, expected 1 (one average per map)")
    elif k == "dropout":
        if abs(m.p - spec["p"]) > 1e-9:
            out.append(f"p is {m.p}, expected {spec['p']}")
    elif k == "linear":
        if m.in_features != spec["in"]:
            out.append(f"in_features is {m.in_features}, expected {spec['in']}")
        if m.out_features != spec["out"]:
            out.append(f"out_features is {m.out_features}, expected {spec['out']}")
        if m.bias is None:
            out.append("it has no bias (bias=False): leave bias at its default")
    return out


def leaves(module: nn.Module) -> list:
    """(name, layer) of every layer without sub-layers, in the order they were created."""
    out = []
    for name, m in module.named_modules():
        if not any(True for _ in m.children()):
            out.append((name, m))
    return out


def _missing_hint(spec: dict, after: str, relu_hint: str) -> str:
    if spec["kind"] == "relu":
        return f"missing: there is no ReLU{after}. {relu_hint}"
    return f"missing: the table has {_KIND_WORDS.get(spec['kind'], spec['kind'])} here, your code has none"


def _pair_cost(spec: dict, m: nn.Module) -> float:
    if spec["kind"] != _kind(m):
        return 1.5                                   # a different kind of layer in this place
    n = len(_differences(spec, m))
    if n == 0:
        return 0.0
    return 0.4 + 0.1 * min(n, 5)                     # the right kind with wrong numbers: still the same layer


def align(expected: list, found: list) -> list:
    """Line the student's layers up with the table (fewest missing / extra / different layers).

    Returns steps in order: ("match", i, j), ("missing", i, None) or ("extra", None, j).
    """
    n, m = len(expected), len(found)
    cost = []
    for i in range(n + 1):
        row = []
        for j in range(m + 1):
            row.append(0.0)
        cost.append(row)
    for i in range(1, n + 1):
        cost[i][0] = float(i)
    for j in range(1, m + 1):
        cost[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = cost[i - 1][j - 1] + _pair_cost(expected[i - 1], found[j - 1][1])
            best = min(best, cost[i - 1][j] + 1.0, cost[i][j - 1] + 1.0)
            cost[i][j] = best
    steps = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(cost[i][j] - cost[i - 1][j - 1] - _pair_cost(expected[i - 1], found[j - 1][1])) < 1e-9:
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


def compare_layers(lines: _Lines, where: str, expected: list, module: nn.Module, relu_hint: str,
                   name_hint: str = "") -> bool:
    """One [PASS] / [FAIL] line per expected layer (in order), plus one per extra layer. True if all match."""
    found = leaves(module)
    all_ok = True
    for step, a, b in align(expected, found):
        if step == "extra":
            lines.fail(f"{where}: an extra layer, {found_text(found[b][1])}",
                       "the table has no such layer here: remove it")
            all_ok = False
            continue
        name = f"{where}, layer {a + 1}: {describe(expected[a])}"
        if step == "missing":
            after = ""
            if a > 0:
                after = f" after the {describe(expected[a - 1]).split(',')[0]}"
            lines.fail(name, _missing_hint(expected[a], after, relu_hint))
            all_ok = False
            continue
        m = found[b][1]
        if _kind(m) != expected[a]["kind"]:
            hint = f"found {found_text(m)} instead of {_KIND_WORDS.get(expected[a]['kind'], '?')}"
            if expected[a]["kind"] == "gap" and isinstance(m, nn.AvgPool2d):
                hint += (": nn.AvgPool2d(7) works only for maps of 7 x 7, nn.AdaptiveAvgPool2d(1) averages maps of "
                         "any size (the small network of section 6 needs that)")
            lines.fail(name, hint)
            all_ok = False
            continue
        problems = _differences(expected[a], m)
        hint = "; ".join(problems)
        if problems and name_hint:
            hint += ". " + name_hint
        all_ok &= lines.check(not problems, name, hint)
    return all_ok


def _same_layers(expected: list, module: nn.Module) -> str:
    """'' if the layers match, else the first difference (for the second set of numbers)."""
    found = leaves(module)
    for step, a, b in align(expected, found):
        if step == "extra":
            return f"there is an extra layer, {found_text(found[b][1])}"
        if step == "missing":
            return f"layer {a + 1} ({describe(expected[a])}) is missing"
        m = found[b][1]
        if _kind(m) != expected[a]["kind"]:
            return f"layer {a + 1} should be {describe(expected[a])}, found {found_text(m)}"
        problems = _differences(expected[a], m)
        if problems:
            return f"layer {a + 1} ({describe(expected[a]).split(',')[0]}): {problems[0]}"
    return ""


def _check_output(lines: _Lines, name: str, model: nn.Module, x: torch.Tensor, want: tuple, size_hint: str = "",
                  channel_hint: str = ""):
    out, err = _try(_run, model, x)
    if err:
        lines.fail(name, err)
        return None
    if not isinstance(out, torch.Tensor):
        lines.fail(name, f"the result is a {type(out).__name__}, not a tensor: does forward end with return?")
        return None
    hint = f"the output is {_shape(out.shape)}, expected {_shape(want)}"
    if len(want) == 4 and out.dim() == 4 and tuple(out.shape[2:]) != tuple(want[2:]) and size_hint:
        hint += ". " + size_hint
    elif out.dim() >= 2 and out.shape[1] != want[1] and channel_hint:
        hint += ". " + channel_hint
    lines.check(tuple(out.shape) == tuple(want), name, hint)
    return out


def _check_params(lines: _Lines, name: str, model: nn.Module, want: int, how: str = ""):
    got = _params(model)
    hint = f"yours has {got:,}, expected {want:,}"
    if how:
        hint += f" ({how})"
    lines.check(got == want, f"{name}: {want:,} parameters", hint)


# ============================================================================================== the specifications
def spec_branch1(cin, c1):
    return [conv(cin, c1, 1), relu()]


def spec_branch2(cin, c3r, c3):
    return [conv(cin, c3r, 1), relu(), conv(c3r, c3, 3, 1, 1), relu()]


def spec_branch3(cin, c5r, c5):
    return [conv(cin, c5r, 1), relu(), conv(c5r, c5, 5, 1, 2), relu()]


def spec_branch4(cin, pool_proj):
    return [maxpool(3, 1, 1), conv(cin, pool_proj, 1), relu()]


def spec_stem():
    return [conv(3, 64, 7, 2, 3), relu(), maxpool(3, 2, 1), conv(64, 64, 1), relu(), conv(64, 192, 3, 1, 1), relu(),
            maxpool(3, 2, 1)]


def spec_head(cin, n):
    return [{"kind": "gap"}, {"kind": "flatten"}, {"kind": "dropout", "p": 0.4}, {"kind": "linear", "in": cin, "out": n}]


_RELU_BRANCH = "Every convolution is followed by a ReLU: write conv_relu(...), which makes both, not nn.Conv2d alone."
_RELU_STEM = "Every convolution of the stem is followed by nn.ReLU()."
_NAMES_HINT = ("Use the names of the arguments (in_channels, c1, c3_reduce, ...), not numbers: the same function "
               "must build every block.")


def _with_rng(fn):
    """Run fn() without changing PyTorch's random numbers for the rest of the notebook; never raise."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  (a check must never end in a traceback)
            print(f"[FAIL]  the check could not finish: {explain(e)}")
            return Result(False, 0, 1)


# ============================================================================================== step (a)
def _conv_relu(fn, verbose: bool) -> Result:
    lines = _Lines("Checking step (a): conv_relu", verbose)
    layers, err = _try(fn, 3, 8, kernel_size=3, stride=2, padding=1)
    if not lines.check(err is None, "conv_relu(3, 8, kernel_size=3, stride=2, padding=1) can be built", err):
        return lines.done("(a)", "")
    compare_layers(lines, "conv_relu", [conv(3, 8, 3, 2, 1), relu()], layers, _RELU_BRANCH)
    out = _check_output(lines, "a 2 x 3 x 16 x 16 batch gives 2 x 8 x 8 x 8 (stride 2 halves height and width)",
                        layers, torch.randn(2, 3, 16, 16), (2, 8, 8, 8))
    if out is not None:
        lines.check(bool((out >= 0).all()), "no negative numbers come out (the ReLU sets them to 0)",
                    "some outputs are negative: the ReLU is missing")
    return lines.done("(a)", "This is the pattern of every step: build, look at the layers, run a batch.")


def check_conv_relu(conv_relu) -> Result:
    """Step (a): conv_relu(in, out, kernel_size, stride, padding) is one Conv2d followed by one ReLU."""
    return _with_rng(lambda: _conv_relu(conv_relu, True))


# ============================================================================================== steps (b) and (c)
def _branch(lines: _Lines, label: str, fn, spec_fn, args_3a: tuple, args_4b: tuple, want_params: int, how: str):
    call = f"{fn.__name__ if hasattr(fn, '__name__') else label}{args_3a}"
    branch, err = _try(fn, *args_3a)
    if not lines.check(err is None, f"{label}: {call} can be built", err):
        return False
    ok = compare_layers(lines, label, spec_fn(*args_3a), branch, _RELU_BRANCH)
    out_c = spec_fn(*args_3a)[-2]["out"] if spec_fn(*args_3a)[-1]["kind"] == "relu" else 0
    x = torch.randn(2, args_3a[0], 28, 28)
    out = _check_output(lines, f"{label}: a 2 x {args_3a[0]} x 28 x 28 batch gives 2 x {out_c} x 28 x 28 "
                               f"(height and width unchanged)", branch, x, (2, out_c, 28, 28),
                        "Height and width must stay 28 x 28: check the padding and the stride",
                        "The channels that come out are the out channels of the branch's last convolution")
    _check_params(lines, label, branch, want_params, how)
    other, err = _try(fn, *args_4b)
    if err:
        lines.fail(f"{label}: also right with the numbers of block 4b, {args_4b}", err)
        return False
    diff = _same_layers(spec_fn(*args_4b), other)
    hint = f"with these numbers {diff}. " + (_NAMES_HINT if ok else "Fix the [FAIL] lines above first.")
    lines.check(not diff, f"{label}: also right with the numbers of block 4b, {args_4b}", hint)
    return ok and out is not None and not diff


def _branch1(fn, verbose):
    lines = _Lines("Checking make_branch1 (block 3a: in_channels=192, c1=64)", verbose)
    _branch(lines, "branch 1", fn, spec_branch1, (192, 64), (512, 160), 12_352, "192 x 64 weights + 64 biases")
    return lines.done("(b)", "")


def _branch2(fn, verbose):
    lines = _Lines("Checking make_branch2 (block 3a: in_channels=192, c3_reduce=96, c3=128)", verbose)
    _branch(lines, "branch 2", fn, spec_branch2, (192, 96, 128), (512, 112, 224), 129_248,
            "192 x 96 + 96 for the 1x1, 96 x 128 x 3 x 3 + 128 for the 3x3")
    return lines.done("(b)", "")


def _branch3(fn, verbose):
    lines = _Lines("Checking make_branch3 (block 3a: in_channels=192, c5_reduce=16, c5=32)", verbose)
    _branch(lines, "branch 3", fn, spec_branch3, (192, 16, 32), (512, 24, 64), 15_920,
            "192 x 16 + 16 for the 1x1, 16 x 32 x 5 x 5 + 32 for the 5x5")
    return lines.done("(c)", "")


def _branch4(fn, verbose):
    lines = _Lines("Checking make_branch4 (block 3a: in_channels=192, pool_proj=32)", verbose)
    _branch(lines, "branch 4", fn, spec_branch4, (192, 32), (512, 64), 6_176,
            "the max-pool has none; 192 x 32 + 32 for the 1x1")
    return lines.done("(c)", "")


def _two(step: str, first, second, verbose: bool, what: str) -> Result:
    if verbose:
        print(f"Checking step {step}")
    a = first(verbose)
    if verbose:
        print()
    b = second(verbose)
    passed = bool(a) and bool(b)
    if verbose:
        print()
        if passed:
            print(f"=> step {step} is done: both functions pass. {what}")
        else:
            print(f"=> step {step} is not finished: fix the [FAIL] lines above, run your cell, then this check again.")
    return Result(passed, a.n_pass + b.n_pass, a.n_fail + b.n_fail)


def check_branches_1_2(make_branch1, make_branch2) -> Result:
    """Step (b): branch 1 (1x1) and branch 2 (1x1 then 3x3), with the numbers of blocks 3a and 4b."""
    return _with_rng(lambda: _two("(b)", lambda v: _branch1(make_branch1, v), lambda v: _branch2(make_branch2, v),
                                  True, "Next: branches 3 and 4."))


def check_branches_3_4(make_branch3, make_branch4) -> Result:
    """Step (c): branch 3 (1x1 then 5x5) and branch 4 (3x3 max-pool then 1x1)."""
    return _with_rng(lambda: _two("(c)", lambda v: _branch3(make_branch3, v), lambda v: _branch4(make_branch4, v),
                                  True, "All four branches exist: next, the block that runs them side by side."))


# ============================================================================================== step (d)
def _copy_weights(src: nn.Module, dst: nn.Module) -> bool:
    """Copy src's parameters into dst, in order; False if the shapes do not line up."""
    a = list(src.parameters())
    b = list(dst.parameters())
    if len(a) != len(b):
        return False
    for p, q in zip(a, b):
        if p.shape != q.shape:
            return False
    with torch.no_grad():
        for p, q in zip(a, b):
            q.copy_(p)
    return True


def _branch_order(block, x, out) -> str:
    """'' if out == cat([b1(x), b2(x), b3(x), b4(x)]); else what the order is (or why it is not that)."""
    ys = []
    for k in range(1, 5):
        branch = getattr(block, f"branch{k}", None)
        if branch is None:
            return f"the block has no attribute branch{k}: keep the four self.branch... lines of __init__"
        ys.append(_run(branch, x))
    if torch.allclose(out, torch.cat(ys, dim=1), atol=1e-5):
        return ""
    for order in itertools.permutations(range(4)):
        pieces = []
        for k in order:
            pieces.append(ys[k])
        cand = torch.cat(pieces, dim=1)
        if cand.shape == out.shape and torch.allclose(out, cand, atol=1e-5):
            names = []
            for k in order:
                names.append(f"y{k + 1}")
            return (f"your output stacks the branches in the order {', '.join(names)}; the specification (and the "
                    f"real GoogLeNet) uses y1, y2, y3, y4: torch.cat([y1, y2, y3, y4], dim=1)")
    # not a reordering: say, slot by slot, what each group of channels is
    problems = []
    start = 0
    for k in range(4):
        width = ys[k].shape[1]
        piece = out[:, start:start + width]
        source = ""
        for j in range(4):
            if ys[j].shape == piece.shape and torch.allclose(piece, ys[j], atol=1e-5):
                source = f"branch {j + 1}'s"
                break
        if source != f"branch {k + 1}'s":
            what = source or "none of the branch outputs"
            problems.append(f"channels {start}-{start + width - 1} should be branch {k + 1}'s output, but they are "
                            f"{what}: is y{k + 1} = self.branch{k + 1}(x)?")
        start += width
    if problems:
        return "; ".join(problems)
    return ("the output is not the four branch outputs stacked: y1 ... y4 must be self.branch1(x) ... "
            "self.branch4(x), each applied to the same x")


def _inception(cls, verbose: bool) -> Result:
    lines = _Lines("Checking step (d): Inception, with the numbers of block 3a "
                   "(192, 64, 96, 128, 16, 32, 32)", verbose)
    block, err = _try(cls, *BLOCK_3A)
    if not lines.check(err is None, "Inception(192, 64, 96, 128, 16, 32, 32) can be built", err):
        return lines.done("(d)", "")
    has = True
    for k in range(1, 5):
        has &= isinstance(getattr(block, f"branch{k}", None), nn.Module)
    lines.check(has, "it has the four branches self.branch1 ... self.branch4",
                "keep the four self.branch... = make_branch...(...) lines of __init__ as they are")
    x = torch.randn(2, 192, 28, 28)
    out, err = _try(_run, block, x)
    if not lines.check(err is None and isinstance(out, torch.Tensor), "forward runs on a 2 x 192 x 28 x 28 batch",
                       err or "forward must end with: return out"):
        return lines.done("(d)", "")
    lines.check(tuple(out.shape) == (2, 256, 28, 28), "the output is 2 x 256 x 28 x 28",
                f"the output is {_shape(out.shape)}, expected 2 x 256 x 28 x 28")
    lines.check(out.shape[1] == 64 + 128 + 32 + 32, "output channels = c1 + c3 + c5 + pool_proj = 64 + 128 + 32 + 32 "
                "= 256", f"the output has {out.shape[1]} channels: stack all four branch outputs, each once")
    lines.check(tuple(out.shape[2:]) == (28, 28), "height and width are unchanged: 28 x 28 in, 28 x 28 out",
                f"the output is {_shape(out.shape[2:])} high and wide, expected 28 x 28")
    _check_params(lines, "the block", block, 163_696, "the four branches: 12,352 + 129,248 + 15,920 + 6,176")
    if has and tuple(out.shape) == (2, 256, 28, 28):
        order = _branch_order(block, x, out)
        lines.check(not order, "branch order: channels 0-63 come from branch 1, 64-191 from branch 2, 192-223 from "
                    "branch 3, 224-255 from branch 4", order)
        ref = R.Inception(*BLOCK_3A)
        same = _copy_weights(block, ref) and torch.allclose(_run(ref, x), out, atol=1e-5)
        lines.check(same, "with the same weights, your block computes exactly what the reference block computes",
                    "the numbers differ from the reference block: compare forward with the table, line by line")
    other, err = _try(cls, *BLOCK_4B)
    if err:
        lines.fail("also right with the numbers of block 4b, (512, 160, 112, 224, 24, 64, 64)", err)
    else:
        y, err = _try(_run, other, torch.randn(2, 512, 14, 14))
        ok = err is None and isinstance(y, torch.Tensor) and tuple(y.shape) == (2, 512, 14, 14) and _params(other) == 449_160
        hint = err or (f"output {_shape(y.shape) if isinstance(y, torch.Tensor) else type(y).__name__} and "
                       f"{_params(other):,} parameters, expected 2 x 512 x 14 x 14 and 449,160")
        lines.check(ok, "also right with the numbers of block 4b: 2 x 512 x 14 x 14 in, 2 x 512 x 14 x 14 out, "
                    "449,160 parameters", hint)
    return lines.done("(d)", "You have built an Inception block.")


def check_inception(Inception) -> Result:
    """Step (d): the Inception block runs its four branches on the same input and concatenates them, in order."""
    return _with_rng(lambda: _inception(Inception, True))


# ============================================================================================== step (e)
def _stem(fn, verbose: bool) -> Result:
    lines = _Lines("Checking step (e): make_stem", verbose)
    stem, err = _try(fn)
    if not lines.check(err is None, "make_stem() can be built", err):
        return lines.done("(e)", "")
    compare_layers(lines, "stem", spec_stem(), stem, _RELU_STEM)
    _check_output(lines, "a 1 x 3 x 224 x 224 picture gives 1 x 192 x 28 x 28", stem, torch.randn(1, 3, 224, 224),
                  (1, 192, 28, 28), "Each stride-2 layer halves height and width: 224 -> 112 -> 56 -> 28",
                  "The last convolution of the stem makes 192 maps")
    _check_params(lines, "the stem", stem, 124_416, "9,472 + 4,160 + 110,784 for the three convolutions")
    return lines.done("(e)", "")


def check_stem(make_stem) -> Result:
    """Step (e): the stem, 3 x 224 x 224 -> 192 x 28 x 28, layer by layer from the table."""
    return _with_rng(lambda: _stem(make_stem, True))


# ============================================================================================== step (f)
_BLOCK_ORDER = ["inception3a", "inception3b", "pool3", "inception4a", "inception4b", "inception4c", "inception4d",
                "inception4e", "pool4", "inception5a", "inception5b"]


def block_numbers(block: nn.Module):
    """The seven numbers an Inception block was built with, read from its six convolutions (None if not 6)."""
    convs = []
    for m in block.modules():
        if isinstance(m, nn.Conv2d):
            convs.append(m)
    if len(convs) != 6:
        return None
    return (convs[0].in_channels, convs[0].out_channels, convs[1].out_channels, convs[2].out_channels,
            convs[3].out_channels, convs[4].out_channels, convs[5].out_channels)


def _row_of(numbers) -> str:
    for name, row in R.BLOCKS.items():
        if tuple(row) == tuple(numbers):
            return name.replace("inception", "")
    return ""


def _blocks(fn, verbose: bool) -> Result:
    lines = _Lines("Checking step (f): make_inception_blocks", verbose)
    blocks, err = _try(fn)
    if not lines.check(err is None, "make_inception_blocks() can be built", err):
        return lines.done("(f)", "")
    names = list(blocks._modules.keys()) if isinstance(blocks, nn.Module) else []
    lines.check(names == _BLOCK_ORDER, "the parts are, in order: 3a, 3b, pool3, 4a, 4b, 4c, 4d, 4e, pool4, 5a, 5b",
                f"found {', '.join(names) or 'nothing'}: keep the add_module lines and their names as given")
    previous_out = 192
    previous_name = "the stem"
    for name, row in R.BLOCKS.items():
        short = name.replace("inception", "")
        want_out = R.out_channels(row)
        title = f"block {short}: Inception{row} -> {want_out} channels"
        block = blocks._modules.get(name) if isinstance(blocks, nn.Module) else None
        if block is None:
            lines.fail(title, f'missing: there is no blocks.add_module("{name}", ...) line')
            continue
        got = block_numbers(block)
        if got is None:
            lines.fail(title, f"{name} is not an Inception block (it should have 6 convolutions)")
            continue
        if tuple(got) == tuple(row):
            lines.ok(title)
        else:
            wrong = []
            for col, g, w in zip(COLUMNS, got, row):
                if g != w:
                    wrong.append(f"{col} is {g}, expected {w}")
            hint = f"found Inception{got}: " + "; ".join(wrong)
            other = _row_of(got)
            if other:
                hint += f" (these are the numbers of row {other})"
            if got[0] != row[0]:
                hint += (f". in_channels must be what comes before: {previous_name} gives {previous_out} channels")
            lines.fail(title, hint)
        previous_out = want_out
        previous_name = f"block {short}"
    pools_ok = True
    for name in ("pool3", "pool4"):
        pool = blocks._modules.get(name) if isinstance(blocks, nn.Module) else None
        pools_ok &= isinstance(pool, nn.MaxPool2d) and not _differences(maxpool(3, 2, 1), pool)
    lines.check(pools_ok, "pool3 and pool4: 3x3 max-pools, stride 2, padding 1 (they halve height and width)",
                "keep the two given max-pool lines as they are")
    _check_output(lines, "a 1 x 192 x 28 x 28 input gives 1 x 1024 x 7 x 7", blocks, torch.randn(1, 192, 28, 28),
                  (1, 1024, 7, 7), "the two max-pools halve height and width: 28 -> 14 -> 7",
                  "Read the [FAIL] lines above: a block with wrong numbers changes the channels")
    _check_params(lines, "the nine blocks", blocks, 5_849_136, "sum of the nine blocks of the table")
    return lines.done("(f)", "")


def check_inception_blocks(make_inception_blocks) -> Result:
    """Step (f): the nine Inception blocks 3a ... 5b with the published numbers, and the two max-pools."""
    return _with_rng(lambda: _blocks(make_inception_blocks, True))


# ============================================================================================== step (g)
def _head(fn, verbose: bool) -> Result:
    lines = _Lines("Checking step (g): make_head (in_channels=1024, num_classes=1000)", verbose)
    head, err = _try(fn, 1024, 1000)
    if not lines.check(err is None, "make_head(1024, 1000) can be built", err):
        return lines.done("(g)", "")
    layers_ok = compare_layers(lines, "head", spec_head(1024, 1000), head, "")
    for size in (7, 3):
        _check_output(lines, f"a 2 x 1024 x {size} x {size} input gives 2 x 1000 scores", head,
                      torch.randn(2, 1024, size, size), (2, 1000), "",
                      "nn.AdaptiveAvgPool2d(1) averages each map down to 1 x 1, whatever its size, and the linear "
                      "layer gives num_classes scores")
    _check_params(lines, "the head", head, 1_025_000, "1024 x 1000 weights + 1000 biases of the linear layer")
    other, err = _try(fn, 256, 10)
    if err:
        lines.fail("also right with other numbers, make_head(256, 10)", err)
    else:
        diff = _same_layers(spec_head(256, 10), other)
        more = "Use in_channels and num_classes, not numbers." if layers_ok else "Fix the [FAIL] lines above first."
        lines.check(not diff, "also right with other numbers, make_head(256, 10)", f"with these numbers {diff}. {more}")
    return lines.done("(g)", "")


def check_head(make_head) -> Result:
    """Step (g): global average pooling, flatten, dropout 0.4, linear in_channels -> num_classes."""
    return _with_rng(lambda: _head(make_head, True))


# ============================================================================================== step (h)
def _record_calls(net: nn.Module, x: torch.Tensor) -> list:
    calls = []
    handles = []
    for name in ("stem", "blocks", "head"):
        part = getattr(net, name, None)
        if isinstance(part, nn.Module):
            handles.append(part.register_forward_hook(_recorder(calls, name)))
    try:
        _run(net, x)
    finally:
        for h in handles:
            h.remove()
    return calls


def _recorder(calls: list, name: str):
    def hook(module, inputs, output):
        calls.append(name)
    return hook


def kernel_3x3_saving() -> int:
    """Weights torchvision saves by using 3x3 instead of 5x5 kernels in branch 3 of every block."""
    total = 0
    for row in R.BLOCKS.values():
        total += row[4] * row[5] * (25 - 9)
    return total


def conv_channels_total(net: nn.Module) -> int:
    """Sum of the out channels of every convolution = the biases (one per map)."""
    total = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            total += m.out_channels
    return total


def _googlenet(cls, verbose: bool) -> Result:
    lines = _Lines("Checking step (h): GoogLeNet", verbose)
    net, err = _try(cls)
    if not lines.check(err is None, "GoogLeNet() can be built", err):
        return lines.done("(h)", "")
    x = torch.randn(1, 3, 224, 224)
    out, err = _try(_run, net, x)
    if not lines.check(err is None and isinstance(out, torch.Tensor), "forward runs on a 1 x 3 x 224 x 224 picture",
                       err or "forward must end with: return x"):
        return lines.done("(h)", "")
    calls, err = _try(_record_calls, net, x)
    lines.check(calls == ["stem", "blocks", "head"], "forward applies the stem, then the blocks, then the head, "
                "each once", f"your forward applies: {', '.join(calls or ['nothing'])}")
    lines.check(tuple(out.shape) == (1, 1000), "the output is 1 x 1000: one score per class",
                f"the output is {_shape(out.shape)}, expected 1 x 1000")
    small, err = _try(cls, num_classes=10)
    if err:
        lines.fail("GoogLeNet(num_classes=10) gives 1 x 10 scores", err)
    else:
        y, err = _try(_run, small, x)
        ok = err is None and isinstance(y, torch.Tensor) and tuple(y.shape) == (1, 10)
        hint = err
        if not err and isinstance(y, torch.Tensor):
            hint = f"the output is {_shape(y.shape)}, expected 1 x 10"
            if tuple(y.shape) == (1, 1000):
                hint += ": keep make_head(1024, num_classes) in __init__ as given"
        lines.check(ok, "GoogLeNet(num_classes=10) gives 1 x 10 scores", hint or "forward must end with: return x")
    _check_params(lines, "the whole network", net, 6_998_552,
                  "stem 124,416 + blocks 5,849,136 + head 1,025,000")
    n_conv = 0
    n_relu = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            n_conv += 1
        if isinstance(m, nn.ReLU):
            n_relu += 1
    lines.check(n_conv == 57 and n_relu == 57, "57 convolutions (3 in the stem, 6 in each of the 9 blocks), each "
                "followed by a ReLU", f"found {n_conv} convolutions and {n_relu} ReLUs")
    ref = R.GoogLeNet()
    same = _copy_weights(net, ref)
    if same:
        y = _run(ref, x)
        same = y.shape == out.shape and torch.allclose(y, out, atol=1e-4)
    lines.check(same, "with the same weights, your network computes exactly what the reference network computes",
                "the scores differ from the reference network: check forward")
    _torchvision(lines, net, out)
    return lines.done("(h)", "You have built GoogLeNet.")


def _torchvision(lines: _Lines, net: nn.Module, out: torch.Tensor):
    """Compare with torchvision's GoogLeNet: an implementation written by other people."""
    from torchvision.models import googlenet
    tv = googlenet(weights=None, aux_logits=False, init_weights=False)
    y = _run(tv, torch.zeros(1, 3, 224, 224))
    lines.check(tuple(y.shape) == tuple(out.shape), f"same output shape as torchvision's GoogLeNet: {_shape(y.shape)}",
                f"yours gives {_shape(out.shape)}")
    mine = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            mine.append(m)
    theirs = []
    for m in tv.modules():
        if isinstance(m, nn.Conv2d):
            theirs.append(m)
    first_diff = ""
    kernel_diff = 0
    if len(mine) != len(theirs):
        first_diff = f"you have {len(mine)} convolutions, torchvision {len(theirs)}"
    else:
        for k, (a, b) in enumerate(zip(mine, theirs)):
            if (a.in_channels, a.out_channels) != (b.in_channels, b.out_channels) and not first_diff:
                first_diff = (f"convolution {k + 1}: yours {a.in_channels} -> {a.out_channels}, torchvision "
                              f"{b.in_channels} -> {b.out_channels}")
            if a.kernel_size != b.kernel_size:
                kernel_diff += 1
    lines.check(not first_diff, f"all {len(theirs)} convolutions have the same in and out channels as torchvision's, "
                "in the same order", first_diff)
    lines.check(kernel_diff == 9, "the same kernel sizes as torchvision's, except the 9 '5x5' convolutions, which "
                "torchvision makes 3x3", f"{kernel_diff} kernel sizes differ from torchvision's")
    n_mine = _params(net)
    n_tv = _params(tv)
    saving = kernel_3x3_saving()
    extra = conv_channels_total(net)
    lines.check(n_mine - saving + extra == n_tv,
                f"parameters: torchvision {n_tv:,}, yours {n_mine:,}; the difference is exactly torchvision's two "
                f"changes: {n_mine:,} - {saving:,} (3x3 instead of 5x5) + {extra:,} (batch normalisation instead of "
                f"biases) = {n_tv:,}",
                f"{n_mine:,} - {saving:,} + {extra:,} = {n_mine - saving + extra:,}, not {n_tv:,}")


def check_googlenet(GoogLeNet) -> Result:
    """Step (h): the whole network, against the reference and against torchvision's GoogLeNet."""
    return _with_rng(lambda: _googlenet(GoogLeNet, True))


# ============================================================================================== never stuck
_STEPS = {
    "conv_relu": (_conv_relu, R.conv_relu, "(a)"),
    "make_branch1": (_branch1, R.make_branch1, "(b)"),
    "make_branch2": (_branch2, R.make_branch2, "(b)"),
    "make_branch3": (_branch3, R.make_branch3, "(c)"),
    "make_branch4": (_branch4, R.make_branch4, "(c)"),
    "Inception": (_inception, R.Inception, "(d)"),
    "make_stem": (_stem, R.make_stem, "(e)"),
    "make_inception_blocks": (_blocks, R.make_inception_blocks, "(f)"),
    "make_head": (_head, R.make_head, "(g)"),
    "GoogLeNet": (_googlenet, R.GoogLeNet, "(h)"),
}


def yours_or_reference(yours, name: str | None = None):
    """Your function or class if it passes all its checks; otherwise the course's reference version, with a note.

    So that the rest of the notebook works while a step is unfinished: use it as
    ``make_branch2 = T.yours_or_reference(make_branch2)``. Fix your cell later, run it, its check and this line again.
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


# ============================================================================================== chart
def show_params_per_block(net: nn.Module, title: str | None = None):
    """Bar chart: the parameters of the stem, of each of the nine Inception blocks and of the head."""
    from .practical import _new_figure, _show

    labels = []
    counts = []
    for name, part in net.named_children():
        if name == "blocks":
            for sub_name, sub in part.named_children():
                if _params(sub) == 0:
                    continue                         # the max-pools have no parameters
                labels.append(sub_name.replace("inception", ""))
                counts.append(_params(sub))
        else:
            labels.append(name)
            counts.append(_params(part))
    fig = _new_figure(9.5, 3.8)
    ax = fig.subplots()
    xs = list(range(len(labels)))
    ax.bar(xs, counts, color="#2a78d6", width=0.68, edgecolor="white", linewidth=2)
    top = max(counts)
    for xpos, value in zip(xs, counts):
        ax.text(xpos, value + top * 0.015, f"{round(value / 1000):,} K", ha="center", va="bottom", fontsize=8.5,
                color="#0b0b0b")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, top * 1.15)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.tick_params(axis="x", length=0)
    total = _params(net)
    ax.set_title(title or f"parameters of each part ({total:,} in total; K = thousand)", fontsize=10.5, loc="left",
                 color="#0b0b0b")
    return _show(fig)
