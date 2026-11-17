"""P4 (U-Net, practical track): the tests after every building step.

Import it in the notebook as::

    from cvcourse import ptests_p4 as T

    ok = T.check_double_conv(DoubleConv)                       # one [PASS] / [FAIL] line per property
    DoubleConv = T.yours_or_reference(ok, DoubleConv, "DoubleConv")

check_double_conv(DoubleConv), check_down(Down), check_up(Up), check_unet(UNet): build the student's
block, compare every layer with the specification table of the notebook, run it on a small random batch
and check the output, the order of the layers, the wiring of the skip connections and the parameter
count against the reference in ``cvcourse.pmodels_p4``. They print, never raise: an unfinished block
(the ``raise NotImplementedError`` line still there, or a ``____`` blank left) is reported as "not written
yet". Each returns True when everything passed.

yours_or_reference(ok, yours, name): the student's class if ok, else the reference class with a note, so
that the rest of the notebook always runs.
"""
from __future__ import annotations

import traceback

import torch
import torch.nn as nn

from . import pmodels_p4 as R

__all__ = ["check_double_conv", "check_down", "check_up", "check_unet", "yours_or_reference"]

_PASS = "[PASS]"
_FAIL = "[FAIL]"
_NOT_WRITTEN = "not written yet: fill in the ____ blanks and delete the raise line"


# =============================================================================================== printing
def _line(ok: bool, text: str, hint: str = "") -> bool:
    out = f"{_PASS if ok else _FAIL}  {text}"
    if hint and not ok:
        out += "\n        hint: " + hint
    print(out)
    return bool(ok)


def _head(text: str) -> None:
    print(text)


def _n(v: int) -> str:
    return f"{v:,}"


def _shape(t) -> str:
    return " x ".join(str(int(v)) for v in t)


# =============================================================================================== errors in student code
def _student_line(err: BaseException) -> str:
    """The source line of the student's notebook cell where the error happened (or '')."""
    frames = traceback.extract_tb(err.__traceback__)
    found = ""
    for fr in frames:
        fn = (fr.filename or "").replace("\\", "/")
        if "site-packages" in fn or "/cvcourse/" in fn or fn.startswith("<frozen"):
            continue
        if fr.line:
            found = fr.line.strip()
    return found


def _why(err: BaseException) -> str:
    """A plain explanation of an error raised while building or running the student's block."""
    line = _student_line(err)
    where = f" (in the line `{line}`)" if line else ""
    msg = str(err).strip().splitlines()[0].rstrip(".") if str(err).strip() else ""
    if isinstance(err, NotImplementedError):
        return _NOT_WRITTEN
    if isinstance(err, NameError) and "____" in msg:
        return "a ____ blank is still empty" + (f": `{line}`" if line else "") + ". Fill it in as its comment says"
    if isinstance(err, NameError):
        return f"Python does not know a name: {msg}{where}. A typing mistake, or a cell above not run yet?"
    if isinstance(err, AttributeError) and "object has no attribute" in msg:
        return f"{msg}{where}: a layer is used in forward but was never created in __init__ (or has another name)"
    if isinstance(err, TypeError) and ("argument" in msg or "takes" in msg):
        return f"{msg}{where}: a layer or block was given the wrong number of arguments"
    return f"{type(err).__name__}: {msg}{where}"


def _build(cls, args: tuple, label: str):
    """(module, '') or (None, explanation)."""
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            return cls(*args), ""
    except Exception as err:  # noqa: BLE001
        return None, _why(err)


# =============================================================================================== tracing a forward pass
class _Trace:
    """Run a block once and record, for every direct child, the call order, its inputs and its output."""

    def __init__(self, module: nn.Module, inputs: tuple):
        self.calls = []                 # [name, args, output or None] in call order
        self.out = None
        self.err = None
        names = {}
        for name, child in module.named_children():
            names[id(child)] = name
        handles = []

        def pre(mod, args):
            self.calls.append([names.get(id(mod), "?"), args, None])

        def post(mod, args, out):
            for c in reversed(self.calls):
                if c[0] == names.get(id(mod), "?") and c[2] is None:
                    c[2] = out
                    break

        for child in module.children():
            handles.append(child.register_forward_pre_hook(pre))
            handles.append(child.register_forward_hook(post))
        was = module.training
        module.eval()
        try:
            with torch.no_grad():
                self.out = module(*inputs)
        except Exception as err:  # noqa: BLE001
            self.err = err
        finally:
            module.train(was)
            for h in handles:
                h.remove()

    def order(self) -> list:
        out = []
        for c in self.calls:
            out.append(c[0])
        return out

    def first(self, name: str):
        for c in self.calls:
            if c[0] == name:
                return c
        return None


def _order_message(got: list, want: list) -> str:
    parts = []
    for name in want:
        k = got.count(name)
        if k == 0:
            parts.append(f"{name} is never used")
        elif k > 1:
            parts.append(f"{name} is used {k} times")
    for name in got:
        if name not in want and f"{name} is not in the table" not in parts:
            parts.append(f"{name} is not in the table")
    if not parts:
        for k, name in enumerate(got):
            if name != want[k]:
                before = f"after {got[k - 1]} comes" if k > 0 else "the first layer is"
                parts.append(f"{before} {name}, the table says {want[k]}")
                break
    return "; ".join(parts)


def _check_order(trace: _Trace, want: list, block: str) -> bool:
    got = trace.order()
    if got == want:
        return _line(True, f"forward uses the layers in the table's order: {', '.join(want)}")
    return _line(False, f"forward uses the layers in this order: {', '.join(got) or '(none)'}; expected "
                        f"{', '.join(want)}", _order_message(got, want) + f". Compare the forward lines of {block} "
                                                                       f"with the table.")


def _check_chain(trace: _Trace, x, want: list) -> bool:
    """Each layer works on the result of the one before (x = self.layer(x) line after line)."""
    prev, prev_name = x, "the input x"
    for name in want:
        c = trace.first(name)
        if c is None:
            return _line(False, "each layer works on the result of the layer before",
                         f"{name} is never used")
        if not c[1] or c[1][0] is not prev:
            return _line(False, "each layer works on the result of the layer before",
                         f"{name} is not applied to {prev_name}: every forward line is "
                         f"x = self.<layer>(x), so that x always holds the latest result")
        prev, prev_name = c[2], f"the output of {name}"
    if trace.out is not prev:
        return _line(False, "each layer works on the result of the layer before",
                     f"forward does not return the output of {want[-1]}: its last line is `return x`")
    return _line(True, "each layer works on the result of the layer before, and forward returns the last one")


# =============================================================================================== layer checks
def _pair(v):
    if isinstance(v, (tuple, list)):
        return tuple(v)
    return (v, v)


def _layer(m: nn.Module, name: str, kind, settings: dict, label: str = "") -> tuple:
    """(ok, description, problem) for one layer of a block: its type and its settings."""
    label = label or name
    layer = getattr(m, name, None)
    if layer is None:
        return False, f"{label}: not found", f"__init__ has no line self.{name} = ..."
    if not isinstance(layer, kind):
        return False, f"{label}: is a {type(layer).__name__}", f"expected nn.{kind.__name__}"
    problems = []
    for attr, want in settings.items():
        got = getattr(layer, attr, None)
        if attr in ("kernel_size", "stride", "padding"):
            got_v = _pair(got)
            want_v = _pair(want)
            if attr == "stride" and isinstance(layer, nn.MaxPool2d) and got is None:
                got_v = _pair(layer.kernel_size)
            if got_v != want_v:
                problems.append(f"{attr} is {got_v[0] if len(set(got_v)) == 1 else got_v}, expected {want}")
        elif got != want:
            problems.append(f"{attr} is {got}, expected {want}")
    text = _describe(layer)
    if problems:
        return False, f"{label}: {text}", "; ".join(problems)
    return True, f"{label}: {text}", ""


def _describe(layer: nn.Module) -> str:
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        return (f"{type(layer).__name__}({layer.in_channels} -> {layer.out_channels} maps, kernel "
                f"{_pair(layer.kernel_size)[0]}, stride {_pair(layer.stride)[0]}, padding {_pair(layer.padding)[0]})")
    if isinstance(layer, nn.BatchNorm2d):
        return f"BatchNorm2d({layer.num_features} maps)"
    if isinstance(layer, nn.MaxPool2d):
        stride = layer.stride if layer.stride is not None else layer.kernel_size
        return f"MaxPool2d(window {_pair(layer.kernel_size)[0]}, stride {_pair(stride)[0]})"
    return type(layer).__name__ + "()"


def _report_layers(rows: list) -> bool:
    ok_all = True
    for ok, text, problem in rows:
        ok_all = _line(ok, text, problem) and ok_all
    return ok_all


def _dc_rows(m: nn.Module, cin: int, cout: int, prefix: str = "") -> list:
    conv = {"in_channels": cin, "out_channels": cout, "kernel_size": 3, "stride": 1, "padding": 1}
    conv2 = {"in_channels": cout, "out_channels": cout, "kernel_size": 3, "stride": 1, "padding": 1}
    return [
        _layer(m, "conv1", nn.Conv2d, conv, prefix + "conv1"),
        _layer(m, "bn1", nn.BatchNorm2d, {"num_features": cout}, prefix + "bn1"),
        _layer(m, "relu1", nn.ReLU, {}, prefix + "relu1"),
        _layer(m, "conv2", nn.Conv2d, conv2, prefix + "conv2"),
        _layer(m, "bn2", nn.BatchNorm2d, {"num_features": cout}, prefix + "bn2"),
        _layer(m, "relu2", nn.ReLU, {}, prefix + "relu2"),
    ]


def _params(m: nn.Module) -> int:
    total = 0
    for p in m.parameters():
        total += p.numel()
    return total


def _check_params(mine: nn.Module, ref: nn.Module, label: str) -> bool:
    a, b = _params(mine), _params(ref)
    return _line(a == b, f"parameters of {label}: {_n(a)} (reference {_n(b)})",
                 "a different count means a layer has other channels or kernel sizes than the table")


def _rand(*shape, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _forward_error(trace: _Trace, block: str) -> str:
    """Explain an error raised inside forward, naming the layer it happened in."""
    err = trace.err
    if isinstance(err, (NotImplementedError, NameError, AttributeError)):
        return _why(err)
    failing = None
    for c in trace.calls:
        if c[2] is None:
            failing = c
    msg = str(err).strip().splitlines()[0].rstrip(".") if str(err).strip() else type(err).__name__
    if failing is not None and failing[1] and torch.is_tensor(failing[1][0]):
        return (f"the layer {failing[0]} could not work on its input of {_shape(failing[1][0].shape[1:])} "
                f"(maps x height x width): {msg}")
    return _why(err)


# =============================================================================================== step 1: DoubleConv
def check_double_conv(DoubleConv) -> bool:
    """DoubleConv(in_channels, out_channels): conv 3x3 -> BatchNorm -> ReLU, twice; height and width kept."""
    try:
        return _check_double_conv(DoubleConv)
    except Exception as err:  # noqa: BLE001
        return _line(False, "DoubleConv could not be checked", _why(err))


def _check_double_conv(DoubleConv) -> bool:
    _head("Testing DoubleConv(3, 16): 3 maps in, 16 maps out")
    m, why = _build(DoubleConv, (3, 16), "DoubleConv(3, 16)")
    if m is None:
        return _line(False, f"DoubleConv is {why}" if why == _NOT_WRITTEN else "DoubleConv(3, 16) can be built",
                     "" if why == _NOT_WRITTEN else why)
    ok = _report_layers(_dc_rows(m, 3, 16))
    x = _rand(2, 3, 64, 64)
    tr = _Trace(m, (x,))
    if tr.err is not None:
        return _line(False, "forward runs on a batch of 2 x 3 x 64 x 64", _forward_error(tr, "DoubleConv"))
    want = (2, 16, 64, 64)
    got = tuple(tr.out.shape) if torch.is_tensor(tr.out) else None
    ok = _line(got == want, f"output shape: {_shape(x.shape)} in -> {_shape(got) if got else type(tr.out).__name__} "
                            f"out (expected {_shape(want)})", "the lines above and below show what differs") and ok
    if got is not None:
        ok = _line(got[2:] == x.shape[2:], f"height and width are kept: {_shape(x.shape[2:])} in, "
                                           f"{_shape(got[2:])} out", "every 3 x 3 convolution needs padding=1") and ok
    order = ["conv1", "bn1", "relu1", "conv2", "bn2", "relu2"]
    ok = _check_order(tr, order, "DoubleConv") and ok
    ok = _check_chain(tr, x, order) and ok
    # other channel numbers: catches numbers typed in instead of in_channels / out_channels
    m2, why2 = _build(DoubleConv, (16, 32), "DoubleConv(16, 32)")
    if m2 is None:
        ok = _line(False, "DoubleConv(16, 32) can be built too", why2) and ok
    else:
        bad = []
        for row_ok, text, problem in _dc_rows(m2, 16, 32):
            if not row_ok:
                bad.append(f"{text} ({problem})")
        ok = _line(not bad, "DoubleConv(16, 32) is right too (the numbers come from in_channels and out_channels)",
                   "; ".join(bad) + ". Use the names in_channels and out_channels, not fixed numbers") and ok
    ok = _check_params(m, R.DoubleConv(3, 16), "DoubleConv(3, 16)") and ok
    return ok


# =============================================================================================== step 2: Down
def _block_channels(m: nn.Module):
    """(in, out) of a DoubleConv-like block, from its first and last convolution (None if unknown)."""
    convs = []
    for sub in m.modules():
        if isinstance(sub, nn.Conv2d):
            convs.append(sub)
    if not convs:
        return None
    return convs[0].in_channels, convs[-1].out_channels


def _double_conv_row(m: nn.Module, name: str, cin: int, cout: int, label: str) -> tuple:
    block = getattr(m, name, None)
    if block is None:
        return False, f"{label}: not found", f"__init__ has no line self.{name} = ..."
    if type(block).__name__ != "DoubleConv":
        return False, f"{label}: is a {type(block).__name__}", "expected a DoubleConv block (step 1)"
    ch = _block_channels(block)
    if ch is None:
        return False, f"{label}: DoubleConv without convolutions", "check your DoubleConv"
    text = f"{label}: DoubleConv({ch[0]} -> {ch[1]} maps)"
    problems = []
    if ch[0] != cin:
        problems.append(f"it takes {ch[0]} maps in, expected {cin}")
    if ch[1] != cout:
        problems.append(f"it gives {ch[1]} maps out, expected {cout}")
    return not problems, text, "; ".join(problems)


def check_down(Down) -> bool:
    """Down(in_channels, out_channels): max-pool 2 x 2, then DoubleConv; height and width halved."""
    try:
        return _check_down(Down)
    except Exception as err:  # noqa: BLE001
        return _line(False, "Down could not be checked", _why(err))


def _check_down(Down) -> bool:
    _head("Testing Down(16, 32): 16 maps in, 32 maps out, half the height and width")
    m, why = _build(Down, (16, 32), "Down(16, 32)")
    if m is None:
        return _line(False, f"Down is {why}" if why == _NOT_WRITTEN else "Down(16, 32) can be built",
                     "" if why == _NOT_WRITTEN else why)
    rows = [_layer(m, "pool", nn.MaxPool2d, {"kernel_size": 2, "stride": 2}),
            _double_conv_row(m, "conv", 16, 32, "conv")]
    ok = _report_layers(rows)
    x = _rand(2, 16, 64, 64)
    tr = _Trace(m, (x,))
    if tr.err is not None:
        return _line(False, "forward runs on a batch of 2 x 16 x 64 x 64", _forward_error(tr, "Down"))
    want = (2, 32, 32, 32)
    got = tuple(tr.out.shape) if torch.is_tensor(tr.out) else None
    ok = _line(got == want, f"output shape: {_shape(x.shape)} in -> {_shape(got) if got else type(tr.out).__name__} "
                            f"out (expected {_shape(want)})", "the lines above and below show what differs") and ok
    if got is not None:
        ok = _line(got[2:] == (32, 32), f"height and width are halved: {_shape(x.shape[2:])} in, {_shape(got[2:])} out",
                   "the max-pooling layer halves them: nn.MaxPool2d(2)") and ok
    order = ["pool", "conv"]
    ok = _check_order(tr, order, "Down") and ok
    ok = _check_chain(tr, x, order) and ok
    m2, why2 = _build(Down, (32, 64), "Down(32, 64)")
    if m2 is None:
        ok = _line(False, "Down(32, 64) can be built too", why2) and ok
    else:
        row = _double_conv_row(m2, "conv", 32, 64, "conv")
        ok = _line(row[0], "Down(32, 64) is right too (the numbers come from in_channels and out_channels)",
                   f"{row[1]}: {row[2]}. Use the names in_channels and out_channels, not fixed numbers") and ok
    ok = _check_params(m, R.Down(16, 32), "Down(16, 32)") and ok
    return ok


# =============================================================================================== step 3: Up
def _up_rows(m: nn.Module, cin: int, cskip: int, cout: int) -> list:
    rows = [_layer(m, "up", nn.ConvTranspose2d,
                   {"in_channels": cin, "out_channels": cin // 2, "kernel_size": 2, "stride": 2, "padding": 0})]
    row = _double_conv_row(m, "conv", cin // 2 + cskip, cout, "conv")
    if not row[0] and "maps in" in row[2]:
        row = (row[0], row[1], row[2] + f" = in_channels // 2 + skip_channels = {cin // 2} + {cskip}")
    rows.append(row)
    return rows


def check_up(Up) -> bool:
    """Up(in_channels, skip_channels, out_channels): ConvTranspose 2 x 2 stride 2, torch.cat([skip, x], dim=1), DoubleConv."""
    try:
        return _check_up(Up)
    except Exception as err:  # noqa: BLE001
        return _line(False, "Up could not be checked", _why(err))


def _check_up(Up) -> bool:
    _head("Testing Up(64, 32, 32): x with 64 maps of 16 x 16, skip with 32 maps of 32 x 32 -> 32 maps of 32 x 32")
    m, why = _build(Up, (64, 32, 32), "Up(64, 32, 32)")
    if m is None:
        return _line(False, f"Up is {why}" if why == _NOT_WRITTEN else "Up(64, 32, 32) can be built",
                     "" if why == _NOT_WRITTEN else why)
    ok = _report_layers(_up_rows(m, 64, 32, 32))
    x = _rand(2, 64, 16, 16, seed=1)
    skip = _rand(2, 32, 32, 32, seed=2)
    tr = _Trace(m, (x, skip))
    if tr.err is not None:
        return _line(False, "forward runs on x of 2 x 64 x 16 x 16 and a skip of 2 x 32 x 32 x 32",
                     _up_error(tr, skip))
    want = (2, 32, 32, 32)
    got = tuple(tr.out.shape) if torch.is_tensor(tr.out) else None
    ok = _line(got == want, f"output shape: x {_shape(x.shape)} and skip {_shape(skip.shape)} in -> "
                            f"{_shape(got) if got else type(tr.out).__name__} out (expected {_shape(want)})",
               "the lines above and below show what differs") and ok
    if got is not None:
        ok = _line(got[2:] == skip.shape[2:], f"the output has the skip's height and width (twice x's): "
                                              f"{_shape(got[2:])}", "self.up doubles height and width") and ok
    ok = _check_order(tr, ["up", "conv"], "Up") and ok
    ok = _check_concat(tr, skip) and ok
    ok = _check_inputs_used(m, x, skip) and ok
    # other numbers: in_channels // 2 is not always skip_channels
    m2, why2 = _build(Up, (64, 16, 24), "Up(64, 16, 24)")
    if m2 is None:
        ok = _line(False, "Up(64, 16, 24) can be built too", why2) and ok
    else:
        bad = []
        for row_ok, text, problem in _up_rows(m2, 64, 16, 24):
            if not row_ok:
                bad.append(f"{text}: {problem}")
        ok = _line(not bad, "Up(64, 16, 24) is right too (a skip with fewer maps than x: the channel rule holds)",
                   "; ".join(bad) + ". In the U-Net in_channels // 2 and skip_channels happen to be equal; the table's "
                                    "rule in_channels // 2 + skip_channels works for every block") and ok
    ok = _check_params(m, R.Up(64, 32, 32), "Up(64, 32, 32)") and ok
    return ok


def _up_error(tr: _Trace, skip) -> str:
    err = tr.err
    if isinstance(err, (NotImplementedError, NameError, AttributeError)):
        return _why(err)
    msg = str(err).strip().splitlines()[0].rstrip(".") if str(err).strip() else type(err).__name__
    conv = tr.first("conv")
    up = tr.first("up")
    if conv is not None and conv[1] and torch.is_tensor(conv[1][0]):
        got = conv[1][0].shape
        if got[2:] != skip.shape[2:]:
            return (f"self.conv received maps of {_shape(got[2:])} pixels instead of {_shape(skip.shape[2:])}: "
                    f"torch.cat glued the tensors along the height or width. Use dim=1, the channel axis: "
                    f"torch.cat([skip, x], dim=1)")
        if up is not None and conv[1][0] is up[2]:
            return (f"self.conv received only the {got[1]} maps of self.up, without the skip: the line "
                    f"x = torch.cat([skip, x], dim=1) is missing ({msg})")
        return f"self.conv could not work on its input of {_shape(got[1:])} (maps x height x width): {msg}"
    if "Sizes of tensors must match" in msg or "Expected size" in msg:
        return (f"torch.cat could not glue the skip and x: {msg}. Both must have the same height and width (apply "
                f"self.up to x first), and are glued along dim=1")
    return _why(err)


def _check_concat(tr: _Trace, skip) -> bool:
    up = tr.first("up")
    conv = tr.first("conv")
    title = "the skip is glued in front of the upsampled x along the channels: torch.cat([skip, x], dim=1)"
    if up is None or conv is None or up[2] is None or not conv[1]:
        return _line(False, title, "forward must use self.up, then torch.cat, then self.conv")
    got = conv[1][0]
    right = torch.cat([skip, up[2]], dim=1)
    if got.shape == right.shape and torch.equal(got, right):
        return _line(True, title)
    if got.shape == right.shape and torch.equal(got, torch.cat([up[2], skip], dim=1)):
        return _line(False, title, "you glued x first and the skip second; the table puts the skip first: "
                                   "torch.cat([skip, x], dim=1). (Both orders could learn, but every Up block and "
                                   "the reference use this one.)")
    return _line(False, title, f"self.conv receives {_shape(got.shape[1:])}, which is not the skip and the upsampled x "
                               f"glued together ({_shape(right.shape[1:])})")


def _check_inputs_used(m: nn.Module, x, skip) -> bool:
    other_skip = _rand(*skip.shape, seed=3)
    other_x = _rand(*x.shape, seed=4)
    zeros = torch.zeros_like(x)
    was = m.training
    m.eval()
    try:
        with torch.no_grad():
            a = m(zeros, skip)
            b = m(zeros, other_skip)
            c = m(x, skip)
            d = m(other_x, skip)
    finally:
        m.train(was)
    ok = _line(not torch.allclose(a, b),
               "the skip is used: with x set to zeros, a different skip still gives a different output",
               "the output does not depend on the skip: it must be glued on with torch.cat([skip, x], dim=1)")
    ok = _line(not torch.allclose(c, d), "the upsampled x is used: a different x gives a different output",
               "the output does not depend on x: apply self.up to x and glue the result after the skip") and ok
    return ok


# =============================================================================================== step 4: UNet
_ENC_VARS = {"inc": "x1", "down1": "x2", "down2": "x3", "down3": "x4", "down4": "x5"}


def _unet_spec(cin: int, ncls: int, base: int) -> list:
    """(name, kind, numbers) of every block of the U-Net, in the table's order."""
    b = base
    return [
        ("inc", "DoubleConv", (cin, b)),
        ("down1", "Down", (b, 2 * b)),
        ("down2", "Down", (2 * b, 4 * b)),
        ("down3", "Down", (4 * b, 8 * b)),
        ("down4", "Down", (8 * b, 16 * b)),
        ("up1", "Up", (16 * b, 8 * b, 8 * b)),
        ("up2", "Up", (8 * b, 4 * b, 4 * b)),
        ("up3", "Up", (4 * b, 2 * b, 2 * b)),
        ("up4", "Up", (2 * b, b, b)),
        ("outc", "Conv2d", (b, ncls)),
    ]


def _numbers_of(block: nn.Module, kind: str):
    """The channel numbers of a block as the table writes them, read from its layers (None if unknown)."""
    if kind == "Conv2d":
        return block.in_channels, block.out_channels
    if kind == "Up":
        up = getattr(block, "up", None)
        conv = getattr(block, "conv", None)
        if up is None or conv is None or not hasattr(up, "in_channels"):
            return None
        ch = _block_channels(conv)
        if ch is None:
            return None
        return up.in_channels, ch[0] - up.out_channels, ch[1]
    ch = _block_channels(block)
    return ch


def _unet_rows(m: nn.Module, cin: int, ncls: int, base: int, ref: nn.Module | None) -> list:
    rows = []
    for name, kind, want in _unet_spec(cin, ncls, base):
        block = getattr(m, name, None)
        call = f"{kind}({', '.join(str(v) for v in want)})" if kind != "Conv2d" else \
            f"nn.Conv2d({want[0]}, {want[1]}, kernel_size=1)"
        if block is None:
            rows.append((False, f"{name}: not found", f"__init__ needs the line self.{name} = {call}"))
            continue
        if type(block).__name__ != kind:
            rows.append((False, f"{name}: is a {type(block).__name__}", f"expected {call}"))
            continue
        got = _numbers_of(block, kind)
        params = _params(block)
        ref_params = _params(getattr(ref, name)) if ref is not None else None
        p_text = f", {_n(params)} parameters" + (f" (reference {_n(ref_params)})" if ref_params is not None else "")
        if got is None:
            rows.append((False, f"{name}: {kind} of unknown shape", f"expected {call}"))
            continue
        shown = f"{kind}({', '.join(str(v) for v in got)})" if kind != "Conv2d" else \
            f"Conv2d({got[0]} -> {got[1]} maps, kernel {_pair(block.kernel_size)[0]})"
        problems = []
        if tuple(got) != tuple(want):
            problems.append(f"expected {call} (the table's row {name})")
        if kind == "Conv2d" and _pair(block.kernel_size) != (1, 1):
            problems.append("the last layer is a 1 x 1 convolution: kernel_size=1")
        if ref_params is not None and params != ref_params and not problems:
            problems.append("same channels as the table but another parameter count: check the block's own layers")
        rows.append((not problems, f"{name}: {shown}{p_text}", "; ".join(problems)))
    return rows


def check_unet(UNet) -> bool:
    """UNet(in_channels, n_classes, base): the blocks of the table (__init__) and their wiring (forward)."""
    try:
        return _check_unet(UNet)
    except Exception as err:  # noqa: BLE001
        return _line(False, "UNet could not be checked", _why(err))


def _check_unet(UNet) -> bool:
    _head("Testing UNet(in_channels=3, n_classes=4, base=16) on images of 3 x 64 x 64")
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            m = UNet(in_channels=3, n_classes=4, base=16)
    except Exception as err:  # noqa: BLE001
        why = _why(err)
        return _line(False, f"UNet is {why}" if why == _NOT_WRITTEN else "UNet(3, 4, base=16) can be built",
                     "" if why == _NOT_WRITTEN else why)
    ref = R.UNet(3, 4, base=16)
    _head("the blocks (__init__), one per row of the table:")
    ok = _report_layers(_unet_rows(m, 3, 4, 16, ref))
    try:
        with torch.random.fork_rng(devices=[]):
            m2 = UNet(in_channels=1, n_classes=3, base=8)
        bad = []
        for row_ok, text, problem in _unet_rows(m2, 1, 3, 8, None):
            if not row_ok:
                bad.append(f"{text}: {problem}")
        ok = _line(not bad, "UNet(in_channels=1, n_classes=3, base=8) is right too (the numbers come from the "
                            "settings)", "; ".join(bad) + ". Write every number with in_channels, n_classes and base, as "
                                                          "in the table") and ok
    except Exception as err:  # noqa: BLE001
        ok = _line(False, "UNet(in_channels=1, n_classes=3, base=8) can be built too", _why(err)) and ok
    ok = _line(_params(m) == _params(ref), f"total: {_n(_params(m))} parameters (reference {_n(_params(ref))})",
               "a block above has other channels than the table") and ok
    _head("the wiring (forward):")
    x = _rand(2, 3, 64, 64)
    tr = _Trace(m, (x,))
    if tr.err is not None:
        return _line(False, "forward runs on a batch of 2 x 3 x 64 x 64", _unet_error(tr, x)) and False
    got = tuple(tr.out.shape) if torch.is_tensor(tr.out) else None
    if got is None:
        return _line(False, "forward returns a tensor", f"it returns a {type(tr.out).__name__}: the last line is "
                                                        f"`return x`") and False
    ok = _line(got == (2, 4, 64, 64), f"output shape: {_shape(x.shape)} in -> {_shape(got)} out (expected 2 x 4 x 64 x 64)",
               "see the lines below") and ok
    ok = _line(got[1] == 4, f"the output has n_classes = 4 maps: one score per class for every pixel (found {got[1]})",
               "the last step of forward is self.outc, the 1 x 1 convolution to n_classes maps") and ok
    ok = _line(got[2:] == (64, 64), f"the output has the input's height and width: {_shape(got[2:])}",
               "every Down halves the size and every Up doubles it: use all four of each") and ok
    order = ["inc", "down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4", "outc"]
    ok = _check_order(tr, order, "UNet") and ok
    ok = _check_wiring(tr, x) and ok
    try:
        x2 = _rand(1, 3, 128, 128, seed=5)
        was = m.training
        m.eval()
        with torch.no_grad():
            out2 = m(x2)
        m.train(was)
        ok = _line(tuple(out2.shape) == (1, 4, 128, 128),
                   f"a 3 x 128 x 128 image gives {_shape(out2.shape[1:])} (any size divisible by 16 works)",
                   "the output must always have the input's height and width") and ok
    except Exception as err:  # noqa: BLE001
        ok = _line(False, "a 3 x 128 x 128 image works too", _why(err)) and ok
    return ok


def _name_of(tr: _Trace, t, x=None) -> str:
    """'x3 (the output of down2)' for a tensor recorded in the trace."""
    if t is None:
        return "nothing"
    if x is not None and t is x:
        return "the input image x"
    for c in tr.calls:
        if c[2] is t:
            var = _ENC_VARS.get(c[0])
            return f"{var} (the output of {c[0]})" if var else f"the output of {c[0]}"
    return "another tensor"


def _wiring_problems(tr: _Trace, x, finished: bool = True) -> list:
    """What is wired wrongly: every block must get x, then the result of the block before, and the right skip."""
    outs = {}
    for c in tr.calls:
        if c[0] not in outs:
            outs[c[0]] = c[2]
    wants = [
        ("inc", [x], ["the input image x"]),
        ("down1", [outs.get("inc")], ["x1"]),
        ("down2", [outs.get("down1")], ["x2"]),
        ("down3", [outs.get("down2")], ["x3"]),
        ("down4", [outs.get("down3")], ["x4"]),
        ("up1", [outs.get("down4"), outs.get("down3")], ["x5", "the skip x4"]),
        ("up2", [outs.get("up1"), outs.get("down2")], ["x (the output of up1)", "the skip x3"]),
        ("up3", [outs.get("up2"), outs.get("down1")], ["x (the output of up2)", "the skip x2"]),
        ("up4", [outs.get("up3"), outs.get("inc")], ["x (the output of up3)", "the skip x1"]),
        ("outc", [outs.get("up4")], ["x (the output of up4)"]),
    ]
    problems = []
    for name, want_ts, want_names in wants:
        c = tr.first(name)
        if c is None:
            continue
        args = c[1]
        for k in range(len(want_ts)):
            wt, wn = want_ts[k], want_names[k]
            got_t = args[k] if k < len(args) else None
            if wt is None or got_t is wt:
                continue
            what = "input" if k == 0 else "skip"
            problems.append(f"{name} gets as its {what} {_name_of(tr, got_t, x)}, expected {wn}")
    if finished and tr.out is not outs.get("outc"):
        problems.append(f"forward returns {_name_of(tr, tr.out, x)}, expected the output of outc")
    return problems


def _check_wiring(tr: _Trace, x) -> bool:
    problems = _wiring_problems(tr, x)
    title = "every block gets the right tensors: x1 .. x5 down the encoder, the matching skip at every Up"
    if problems:
        return _line(False, title, "; ".join(problems))
    return _line(True, title)


def _unet_error(tr: _Trace, x) -> str:
    err = tr.err
    if isinstance(err, (NotImplementedError, NameError, AttributeError, TypeError)):
        return _why(err)
    failing = None
    for c in tr.calls:
        if c[2] is None:
            failing = c
    msg = str(err).strip().splitlines()[0].rstrip(".") if str(err).strip() else type(err).__name__
    if failing is None:
        return _why(err)
    name, args = failing[0], failing[1]
    problems = _wiring_problems(tr, x, finished=False)
    if problems:
        return "; ".join(problems) + ". Compare the forward lines with the table"
    if name.startswith("up") and len(args) >= 2 and torch.is_tensor(args[0]) and torch.is_tensor(args[1]):
        level = int(name[2:])
        want_skip = {1: "x4", 2: "x3", 3: "x2", 4: "x1"}.get(level, "the skip of the same size")
        x_hw, s_hw = args[0].shape[2:], args[1].shape[2:]
        if tuple(s_hw) != (x_hw[0] * 2, x_hw[1] * 2):
            return (f"{name} got x of {_shape(x_hw)} pixels and a skip of {_shape(s_hw)} pixels: after the upsampling "
                    f"x has {_shape((x_hw[0] * 2, x_hw[1] * 2))}, so the skip must have that size too. {name} takes "
                    f"the skip {want_skip}")
        return f"{name} could not work on x of {_shape(args[0].shape[1:])} and a skip of {_shape(args[1].shape[1:])}: {msg}"
    if args and torch.is_tensor(args[0]):
        return f"{name} could not work on its input of {_shape(args[0].shape[1:])} (maps x height x width): {msg}"
    return _why(err)


# =============================================================================================== never stuck
def yours_or_reference(ok: bool, yours, name: str):
    """Your class if its test passed; otherwise the reference class from cvcourse.pmodels_p4, with a note."""
    if ok:
        print(f"-> the cells below use your {name}.")
        return yours
    print(f"-> note: your {name} has not passed every test yet, so the cells below use the course's reference "
          f"{name} instead (from cvcourse.pmodels_p4). Fix yours, run its cell and this test cell again, and the "
          f"cells below will use yours.")
    return getattr(R, name)
