"""Checks and helpers for P6 (GoogLeNet with auxiliary classifiers). In the notebook:
``from cvcourse import ptests_p6 as T``.

Every check prints one line per property, ``[PASS]`` or ``[FAIL]`` with a plain-language hint, then one summary
line, and never stops the notebook with a traceback: an unfinished step (the ``raise NotImplementedError`` line
still there, or a ``____`` blank left) is reported as "not written yet". The reference versions, written exactly
like the solution cells, live in ``cvcourse.pmodels_p6``; the reporting and layer-comparison machinery is P3's
(``cvcourse.ptests_p3``).

    T.check_aux_head(make_aux_head)            step (a)
    T.check_googlenet_aux(GoogLeNetAux)        step (b), also against torchvision's GoogLeNet with aux_logits=True
    T.check_aux_loss(aux_loss)                 step (c)
    thing = T.yours_or_reference(thing)        yours if it passes its checks, else the reference (+ a note)
    T.show_in_training_mode(model, input_shape, depth=None, title=None)
                                               P.show_model, but with the network in training mode (every output)
    T.parameters_per_part(net)                 a small table: the parameters of each part of a network
    T.gradient_per_part(net, photos, labels, loss_fn, aux_weight=0.3)
                                               the size of each part's gradient, main loss alone vs with aux losses
    T.train_with_aux(model, train_loader, val_loader, loss_fn, aux_weight=0.3, epochs=5, lr=0.001, name=None)
                                               P.train for a network with auxiliary heads; returns a History
    T.compare_runs(history_without, history_with)   validation accuracy of two runs, epoch by epoch

A check returns a ``Result`` (true when everything passed; Jupyter shows nothing for it). Checks do not change
PyTorch's random numbers, so the rest of the notebook gives the same results whether a check passed or not.
"""
from __future__ import annotations

import math
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pmodels_p3 as R3
from . import pmodels_p6 as R
from . import ptests_p3 as T3
from .ptests_p3 import Result, _Lines, _params, _run, _shape, conv, relu

__all__ = [
    "check_aux_head", "check_googlenet_aux", "check_aux_loss", "yours_or_reference", "show_in_training_mode",
    "parameters_per_part", "gradient_per_part", "train_with_aux", "compare_runs", "Result",
]

NOT_WRITTEN = T3.NOT_WRITTEN
MAIN_PARAMS = 6_998_552          # P3's GoogLeNet: stem 124,416 + blocks 5,849,136 + head 1,025,000
AUX1_PARAMS = 3_188_840          # 512 x 128 + 128, 2048 x 1024 + 1024, 1024 x 1000 + 1000
AUX2_PARAMS = 3_190_888          # 528 x 128 + 128, the rest the same
PARTS = ["stem", "blocks_3a_to_4a", "blocks_4b_to_4d", "blocks_4e_to_5b", "head", "aux1", "aux2"]
TRAIN_ORDER = ["stem", "blocks_3a_to_4a", "aux1", "blocks_4b_to_4d", "aux2", "blocks_4e_to_5b", "head"]
EVAL_ORDER = ["stem", "blocks_3a_to_4a", "blocks_4b_to_4d", "blocks_4e_to_5b", "head"]


# ============================================================================================== errors
def explain(e: BaseException) -> str:
    """P3's plain-language reasons, plus the mistakes typical of this notebook."""
    msg = T3._first_line(e)
    where = T3._where(e)
    if isinstance(e, RuntimeError) and "expected input" in msg and "channels" in msg:
        found = re.search(r"but got (\d+) channels", msg)
        got = f"{found.group(1)} maps" if found else "another number of maps"
        if "[128, 528, 1, 1]" in msg:
            return (f"aux2 (built for the 528 maps of block 4d) received {got}{where}: aux2 must read x right after "
                    f"self.blocks_4b_to_4d (aux1 reads the 512 maps of block 4a)")
        if "[128, 512, 1, 1]" in msg:
            return (f"aux1 (built for the 512 maps of block 4a) received {got}{where}: aux1 must read x right after "
                    f"self.blocks_3a_to_4a (aux2 reads the 528 maps of block 4d)")
    if isinstance(e, RuntimeError) and "mat1 and mat2" in msg:
        return (f"a linear layer receives a different number of numbers than it was built for{where}: {msg}. In the "
                f"auxiliary head the average pool must give 4 x 4 (kernel_size=5, stride=3) and the 1x1 convolution "
                f"128 maps, so that Flatten gives 128 x 4 x 4 = 2048 numbers")
    if isinstance(e, TypeError) and "not callable" in msg:
        return (f"Python stopped with a TypeError{where}: {msg}. A layer is applied with round brackets, "
                f"self.aux1(x); a number or a tensor cannot be called")
    if isinstance(e, ValueError) and "values to unpack" in msg:
        return (f"Python stopped with a ValueError{where}: {msg}. The line main, aux1, aux2 = outputs needs the "
                f"three sets of scores the network returns in training mode")
    return T3.explain(e)


def _try(fn, *args, **kwargs):
    """(value, None) or (None, plain-language reason)."""
    try:
        return fn(*args, **kwargs), None
    except Exception as e:  # noqa: BLE001  (anything the student's code raises becomes a hint)
        return None, explain(e)


def _with_rng(fn):
    """Run fn() without changing PyTorch's random numbers for the rest of the notebook; never raise."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  (a check must never end in a traceback)
            print(f"[FAIL]  the check could not finish: {explain(e)}")
            return Result(False, 0, 1)


# ============================================================================================== layers
def avgpool(k, s):
    return {"kind": "AvgPool2d", "k": k, "s": s, "p": 0}


def linear(cin, cout):
    return {"kind": "linear", "in": cin, "out": cout}


def dropout(p):
    return {"kind": "dropout", "p": p}


def spec_aux_head(cin, n):
    return [avgpool(5, 3), conv(cin, 128, 1), relu(), {"kind": "flatten"}, linear(2048, 1024), relu(), dropout(0.7),
            linear(1024, n)]


_KIND_WORDS = dict(T3._KIND_WORDS)
_KIND_WORDS["AvgPool2d"] = "an average pool (AvgPool2d)"


def describe(spec: dict) -> str:
    if spec["kind"] == "AvgPool2d":
        return f"{spec['k']}x{spec['k']} average pool, stride {spec['s']}"
    return T3.describe(spec)


def found_text(m: nn.Module) -> str:
    if isinstance(m, nn.AvgPool2d):
        stride = m.stride if m.stride is not None else m.kernel_size
        return f"AvgPool2d kernel {T3._num(m.kernel_size)}, stride {T3._num(stride)}, padding {T3._num(m.padding)}"
    return T3.found_text(m)


def differences(spec: dict, m: nn.Module) -> list:
    """What differs between an expected layer and the student's layer of the same kind."""
    if spec["kind"] != "AvgPool2d":
        return T3._differences(spec, m)
    out = []
    stride = m.stride if m.stride is not None else m.kernel_size
    if T3._pair(m.kernel_size) != (spec["k"], spec["k"]):
        out.append(f"kernel_size is {T3._num(m.kernel_size)}, expected {spec['k']}")
    if T3._pair(stride) != (spec["s"], spec["s"]):
        word = ""
        if stride == m.kernel_size:
            word = " (without stride=..., an average pool steps by its kernel size)"
        out.append(f"stride is {T3._num(stride)}, expected {spec['s']}{word}")
    if T3._pair(m.padding) != (spec["p"], spec["p"]):
        out.append(f"padding is {T3._num(m.padding)}, expected {spec['p']}")
    if m.ceil_mode:
        out.append("ceil_mode must stay False")
    return out


def _kind_hint(spec: dict, m: nn.Module) -> str:
    hint = f"found {found_text(m)} instead of {_KIND_WORDS.get(spec['kind'], spec['kind'])}"
    if spec["kind"] == "AvgPool2d" and isinstance(m, nn.AdaptiveAvgPool2d):
        hint += (": nn.AdaptiveAvgPool2d((4, 4)) also gives 4 x 4 (torchvision does that), but the table has the "
                 "paper's nn.AvgPool2d(kernel_size=5, stride=3)")
    elif spec["kind"] == "AvgPool2d" and isinstance(m, nn.MaxPool2d):
        hint += ": the auxiliary head averages, nn.AvgPool2d, it does not take the largest value"
    return hint


def compare_layers(lines: _Lines, where: str, expected: list, module: nn.Module, relu_hint: str) -> bool:
    """One [PASS] / [FAIL] line per expected layer (in order), plus one per extra layer. True if all match."""
    found = T3.leaves(module)
    all_ok = True
    for step, a, b in T3.align(expected, found):
        if step == "extra":
            lines.fail(f"{where}: an extra layer, {found_text(found[b][1])}",
                       "the table has no such layer here: remove it")
            all_ok = False
            continue
        name = f"{where}, layer {a + 1}: {describe(expected[a])}"
        if step == "missing":
            if expected[a]["kind"] == "relu":
                hint = f"missing: there is no ReLU after the {describe(expected[a - 1]).split(',')[0]}. {relu_hint}"
            else:
                hint = (f"missing: the table has {_KIND_WORDS.get(expected[a]['kind'], expected[a]['kind'])} here, "
                        f"your code has none")
            lines.fail(name, hint)
            all_ok = False
            continue
        m = found[b][1]
        if T3._kind(m) != expected[a]["kind"]:
            lines.fail(name, _kind_hint(expected[a], m))
            all_ok = False
            continue
        problems = differences(expected[a], m)
        all_ok &= lines.check(not problems, name, "; ".join(problems))
    return all_ok


def same_layers(expected: list, module: nn.Module) -> str:
    """'' if the layers match, else the first difference (for the second set of numbers)."""
    found = T3.leaves(module)
    for step, a, b in T3.align(expected, found):
        if step == "extra":
            return f"there is an extra layer, {found_text(found[b][1])}"
        if step == "missing":
            return f"layer {a + 1} ({describe(expected[a])}) is missing"
        m = found[b][1]
        if T3._kind(m) != expected[a]["kind"]:
            return f"layer {a + 1} should be {describe(expected[a])}, found {found_text(m)}"
        problems = differences(expected[a], m)
        if problems:
            return f"layer {a + 1} ({describe(expected[a]).split(',')[0]}): {problems[0]}"
    return ""


def _check_params(lines: _Lines, name: str, model: nn.Module, want: int, how: str = ""):
    got = _params(model)
    hint = f"yours has {got:,}, expected {want:,}"
    if how:
        hint += f" ({how})"
    lines.check(got == want, f"{name}: {want:,} parameters", hint)


# ============================================================================================== step (a)
_RELU_AUX = "The 1x1 convolution and the first linear layer are each followed by nn.ReLU()."


def _flatten_input(head: nn.Module, x: torch.Tensor):
    """The shape that goes into the head's first nn.Flatten (batch, maps, height, width), or None."""
    flat = None
    for m in head.modules():
        if isinstance(m, nn.Flatten):
            flat = m
            break
    if flat is None:
        return None
    seen = []

    def hook(module, inputs, output):
        seen.append(tuple(inputs[0].shape))

    handle = flat.register_forward_hook(hook)
    try:
        _try(_run, head, x)
    finally:
        handle.remove()
    if not seen:
        return None
    return seen[0]


def _aux_head(fn, verbose: bool) -> Result:
    lines = _Lines("Checking step (a): make_aux_head (in_channels=512, num_classes=1000: the head after block 4a)",
                   verbose)
    head, err = _try(fn, 512, 1000)
    if not lines.check(err is None, "make_aux_head(512, 1000) can be built", err):
        return lines.done("(a)", "")
    if not isinstance(head, nn.Module):
        lines.fail("make_aux_head returns the layers", f"it returns {type(head).__name__}: end with return head")
        return lines.done("(a)", "")
    layers_ok = compare_layers(lines, "aux head", spec_aux_head(512, 1000), head, _RELU_AUX)
    x = torch.randn(2, 512, 14, 14)
    into = _flatten_input(head, x)
    hint = "nothing reached nn.Flatten(): read the [FAIL] lines above"
    good = into is not None and tuple(into[1:]) == (128, 4, 4)
    if into is not None and not good:
        hint = f"Flatten receives {_shape(into[1:])} per photo, expected 128 x 4 x 4"
        if len(into) == 4 and tuple(into[2:]) != (4, 4):
            hint += ": the average pool must turn 14 x 14 into 4 x 4, (14 - 5) / 3 + 1 = 4 (kernel_size=5, stride=3)"
        elif len(into) == 4 and into[1] != 128:
            hint += ": the 1x1 convolution must make 128 maps"
    lines.check(good, "the pool and the 1x1 convolution leave 128 maps of 4 x 4, and Flatten lays them out as "
                "128 x 4 x 4 = 2048 numbers", hint)
    out, err = _try(_run, head, x)
    if err:
        lines.fail("a 2 x 512 x 14 x 14 batch gives 2 x 1000 scores", err)
    elif not isinstance(out, torch.Tensor):
        lines.fail("a 2 x 512 x 14 x 14 batch gives 2 x 1000 scores", f"the result is a {type(out).__name__}")
    else:
        lines.check(tuple(out.shape) == (2, 1000), "a 2 x 512 x 14 x 14 batch gives 2 x 1000 scores",
                    f"the output is {_shape(out.shape)}, expected 2 x 1000")
    _check_params(lines, "aux head", head, AUX1_PARAMS,
                  "512 x 128 + 128 for the 1x1 convolution, 2048 x 1024 + 1024 and 1024 x 1000 + 1000 for the two "
                  "linear layers")
    other, err = _try(fn, 528, 10)
    title = "also right with other numbers, make_aux_head(528, 10): 528 maps in (block 4d), 10 classes"
    if err:
        lines.fail(title, err)
    else:
        diff = same_layers(spec_aux_head(528, 10), other)
        more = "Fix the [FAIL] lines above first."
        if layers_ok:
            more = "Use the names in_channels and num_classes where the table has them, not 512 and 1000."
        lines.check(not diff, title, f"with these numbers {diff}. {more}")
    return lines.done("(a)", "You have built an auxiliary classifier.")


def check_aux_head(make_aux_head) -> Result:
    """Step (a): the auxiliary head, layer by layer from the table (the paper's section 5)."""
    return _with_rng(lambda: _aux_head(make_aux_head, True))


# ============================================================================================== step (b)
def _recorder(calls: list, inputs_seen: dict, outputs_seen: dict, name: str):
    def hook(module, inputs, output):
        calls.append(name)
        if inputs and isinstance(inputs[0], torch.Tensor):
            inputs_seen[name] = inputs[0].detach()
        if isinstance(output, torch.Tensor):
            outputs_seen[name] = output.detach()
    return hook


def _run_recorded(net: nn.Module, x: torch.Tensor, training: bool, seed: int = 5):
    """(output, calls, inputs, outputs): what forward returns and, for each part, the order and its tensors."""
    calls = []
    inputs_seen = {}
    outputs_seen = {}
    handles = []
    for name in PARTS:
        part = getattr(net, name, None)
        if isinstance(part, nn.Module):
            handles.append(part.register_forward_hook(_recorder(calls, inputs_seen, outputs_seen, name)))
    was = net.training
    net.train(training)
    try:
        torch.manual_seed(seed)
        with torch.no_grad():
            out = net(x)
    finally:
        for h in handles:
            h.remove()
        net.train(was)
    return out, calls, inputs_seen, outputs_seen


def _same(a, b) -> bool:
    return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape and \
        torch.allclose(a, b, atol=1e-4)


def _source_of(t: torch.Tensor, outputs_seen: dict) -> str:
    """Which part produced this tensor ('' if none)."""
    for name in PARTS:
        if _same(t, outputs_seen.get(name)):
            return name
    return ""


def _training_mode(lines: _Lines, net: nn.Module, x: torch.Tensor):
    """The training-mode lines. Returns (the three outputs or None, what each part received)."""
    lines.section("-- training mode (net.train()): a batch of 2 photos of 3 x 224 x 224")
    result, err = _try(_run_recorded, net, x, True)
    if not lines.check(err is None, "forward runs", err):
        return None, {}
    out, calls, inputs_seen, outputs_seen = result
    is_three = isinstance(out, (tuple, list)) and len(out) == 3
    what = type(out).__name__
    if isinstance(out, (tuple, list)):
        what = f"{len(out)} things"
    if not lines.check(is_three, "it returns three things: main, aux1, aux2",
                       f"it returns {what}: keep the given lines if self.training: return main, aux1, aux2"):
        return None, inputs_seen
    shapes_ok = True
    bad = ""
    for k, name in enumerate(["main", "aux1", "aux2"]):
        item = out[k]
        if not isinstance(item, torch.Tensor):
            shapes_ok = False
            bad = (f"{name} is a {type(item).__name__}, not a tensor of scores: apply the layers to x, as in "
                   f"self.aux1(x), with (x)")
            break
        if tuple(item.shape) != (2, 1000):
            shapes_ok = False
            bad = f"{name} is {_shape(item.shape)}, expected 2 x 1000"
            break
    lines.check(shapes_ok, "all three are 2 x 1000: 1000 scores for each of the 2 photos", bad)
    want = ", ".join(TRAIN_ORDER)
    got = ", ".join(calls) or "nothing"
    order_hint = f"your forward runs: {got}"
    if calls.count("blocks_4b_to_4d") == 0:
        order_hint += ". The blocks 4b, 4c, 4d are missing: x = self.blocks_4b_to_4d(x)"
    elif calls.count("blocks_4e_to_5b") == 0:
        order_hint += ". The blocks 4e to 5b are missing: x = self.blocks_4e_to_5b(x)"
    elif "head" not in calls:
        order_hint += ". The main head is missing: main = self.head(x)"
    lines.check(calls == TRAIN_ORDER, f"the parts run in this order, each once: {want}", order_hint)
    for name, after, maps, part in (("aux1", "4a", 512, "blocks_3a_to_4a"), ("aux2", "4d", 528, "blocks_4b_to_4d")):
        seen = inputs_seen.get(name)
        title = f"{name} reads the output of block {after}: 2 x {maps} x 14 x 14"
        if seen is None:
            lines.fail(title, f"{name} did not run in training mode: {name} = self.{name}(x)")
            continue
        source = _source_of(seen, outputs_seen)
        hint = f"{name} received {_shape(seen.shape)}"
        if source and source != part:
            hint += f", the output of self.{source}: write the {name} line right after x = self.{part}(x)"
        elif not source:
            hint += f": {name} must read x right after self.{part}"
        lines.check(source == part, title, hint)
    if shapes_ok:
        problems = []
        for k, name in enumerate(["main", "aux1", "aux2"]):
            source = _source_of(out[k], outputs_seen)
            if name == "main" and source != "head":
                problems.append(f"output 1 should be the main head's scores, it is {source or 'something else'}'s")
            if name != "main" and source != name:
                problems.append(f"output {k + 1} should be {name}'s scores, it is {source or 'something else'}'s")
        lines.check(not problems, "the three outputs are, in order, the scores of the main head, aux1 and aux2",
                    "; ".join(problems) + ". The return line is: return main, aux1, aux2")
    return out, inputs_seen


def _eval_mode(lines: _Lines, net: nn.Module, x: torch.Tensor):
    lines.section("-- prediction mode (net.eval()): the same 2 photos")
    result, err = _try(_run_recorded, net, x, False)
    if not lines.check(err is None, "forward runs", err):
        return None
    out, calls, inputs_seen, outputs_seen = result
    ok = isinstance(out, torch.Tensor) and tuple(out.shape) == (2, 1000) and _same(out, outputs_seen.get("head"))
    hint = f"it returns {type(out).__name__}"
    if isinstance(out, torch.Tensor):
        hint = f"it returns {_shape(out.shape)}"
    if isinstance(out, nn.Module):
        hint += ": the layers themselves, not scores. Apply them to x: main = self.head(x)"
    if isinstance(out, (tuple, list)):
        hint += ": in prediction mode only the main scores come back (keep the given last line: return main)"
    lines.check(ok, "it returns one tensor: the main head's 2 x 1000 scores", hint)
    ran = []
    for name in ("aux1", "aux2"):
        if name in calls:
            ran.append(name)
    lines.check(not ran, "the auxiliary heads do not run (they are only used in training)",
                f"{' and '.join(ran)} ran in prediction mode: compute them only under if self.training:")
    return out


def _vs_reference(lines: _Lines, net: nn.Module, x: torch.Tensor, out_train, out_eval):
    ref = R.GoogLeNetAux()
    if not T3._copy_weights(net, ref):
        lines.fail("with the same weights, your network computes exactly what the reference computes",
                   "the parameters do not line up with the reference: keep __init__ as given")
        return
    ref_train, _, _, _ = _run_recorded(ref, x, True)
    ref_eval, _, _, _ = _run_recorded(ref, x, False)
    same = isinstance(out_train, (tuple, list)) and len(out_train) == 3
    if same:
        for k in range(3):
            same = same and _same(out_train[k], ref_train[k])
    same = same and _same(out_eval, ref_eval)
    lines.check(same, "with the same weights (and the same dropout), your network computes exactly what the "
                "reference computes, in both modes", "the scores differ from the reference: compare forward with "
                "the table, line by line")


def _torchvision(lines: _Lines, net: nn.Module, mine_in: dict):
    """Compare with torchvision's GoogLeNet with aux_logits=True: an implementation written by other people.
    mine_in: what the student's parts received in training mode (from _training_mode)."""
    from torchvision.models import googlenet
    lines.section("-- torchvision's GoogLeNet with aux_logits=True")
    tv = googlenet(weights=None, aux_logits=True, init_weights=False)
    shapes_in = {}

    def keep(name):
        def hook(module, inputs, output):
            shapes_in[name] = tuple(inputs[0].shape)
        return hook

    handles = [tv.aux1.register_forward_hook(keep("aux1")), tv.aux2.register_forward_hook(keep("aux2"))]
    tv.train()
    try:
        with torch.no_grad():
            out = tv(torch.zeros(2, 3, 224, 224))
    finally:
        for h in handles:
            h.remove()
    shapes = []
    for item in out:
        shapes.append(_shape(item.shape))
    lines.check(len(out) == 3 and shapes == ["2 x 1000", "2 x 1000", "2 x 1000"],
                "in training mode it also returns three sets of 2 x 1000 scores (torchvision lists them as main, "
                "aux2, aux1)", f"torchvision gives {', '.join(shapes)}")
    ok = True
    hint = ""
    for name in ("aux1", "aux2"):
        mine = mine_in.get(name)
        if mine is None:
            ok = False
            hint = "your auxiliary heads did not run in training mode: fix the training-mode lines above first"
        elif tuple(mine.shape) != shapes_in.get(name):
            ok = False
            hint = (f"torchvision's {name} reads {_shape(shapes_in.get(name))}, yours {_shape(mine.shape)}: see the "
                    f"training-mode lines above")
    lines.check(ok, "its auxiliary heads read the same maps as yours: 2 x 512 x 14 x 14 (after 4a) and "
                "2 x 528 x 14 x 14 (after 4d)", hint)
    tv1 = _params(tv.aux1)
    tv2 = _params(tv.aux2)
    mine1 = _params(net.aux1)
    mine2 = _params(net.aux2)
    lines.check(tv1 - mine1 == 128 and tv2 - mine2 == 128,
                f"parameters of the auxiliary heads: torchvision {tv1:,} and {tv2:,}, yours {mine1:,} and {mine2:,}. "
                f"The difference, 128 each, is batch normalisation (2 parameters per map) instead of a bias (1 per "
                f"map) on the 128 maps of the 1x1 convolution",
                f"the differences are {tv1 - mine1:,} and {tv2 - mine2:,}, expected 128 and 128")
    tv.eval()
    with torch.no_grad():
        y = tv(torch.zeros(2, 3, 224, 224))
    lines.check(isinstance(y, torch.Tensor) and tuple(y.shape) == (2, 1000),
                "in prediction mode it also returns one tensor of 2 x 1000 scores",
                f"torchvision gives {type(y).__name__}")


def _googlenet_aux(cls, verbose: bool) -> Result:
    lines = _Lines("Checking step (b): GoogLeNetAux", verbose)
    net, err = _try(cls)
    if not lines.check(err is None, "GoogLeNetAux() can be built", err):
        return lines.done("(b)", "")
    missing = []
    for name in PARTS:
        if not isinstance(getattr(net, name, None), nn.Module):
            missing.append(name)
    lines.check(not missing, "it has the parts of __init__: " + ", ".join(PARTS),
                f"missing: {', '.join(missing)}. Keep __init__ as it is given")
    if missing:
        return lines.done("(b)", "")
    _check_params(lines, "the whole network", net, MAIN_PARAMS + AUX1_PARAMS + AUX2_PARAMS,
                  f"P3's GoogLeNet {MAIN_PARAMS:,} + aux1 {AUX1_PARAMS:,} + aux2 {AUX2_PARAMS:,}")
    x = torch.randn(2, 3, 224, 224)
    out_train, inputs_seen = _training_mode(lines, net, x)
    out_eval = _eval_mode(lines, net, x)
    if out_train is not None and out_eval is not None:
        _vs_reference(lines, net, x, out_train, out_eval)
    _torchvision(lines, net, inputs_seen)
    return lines.done("(b)", "You have built GoogLeNet with its auxiliary classifiers.")


def check_googlenet_aux(GoogLeNetAux) -> Result:
    """Step (b): three outputs in training mode, one in prediction mode; against the reference and torchvision."""
    return _with_rng(lambda: _googlenet_aux(GoogLeNetAux, True))


# ============================================================================================== step (c)
def _ce(scores, labels) -> float:
    return float(F.cross_entropy(scores, labels))


def _wrong_formula(v: float, m: float, a1: float, a2: float, w: float) -> str:
    """A guess at what the student's formula computes, from its value."""
    guesses = [
        (m, "only the main loss: add aux_weight * loss_aux1 + aux_weight * loss_aux2"),
        (m + a1 + a2, "the auxiliary losses are added without their weight: multiply each by aux_weight"),
        (w * (m + a1 + a2), "aux_weight multiplies all three losses: the main loss counts in full (weight 1)"),
        (w * m + a1 + a2, "aux_weight multiplies the main loss: it belongs to the two auxiliary losses"),
        (w * m + w * a1 + a2, "aux_weight multiplies the main loss: it belongs to the two auxiliary losses"),
        (m + w * a1, "the loss of aux2 is missing: + aux_weight * loss_aux2"),
        (m + w * a2, "the loss of aux1 is missing: + aux_weight * loss_aux1"),
        (m + 2 * w * a1, "loss_aux1 is counted twice and loss_aux2 not at all: loss_aux2 uses aux2"),
        (m + 2 * w * a2, "loss_aux2 is counted twice and loss_aux1 not at all: loss_aux1 uses aux1"),
        (m + 2 * w * m, "loss_aux1 and loss_aux2 are computed from the main scores: use aux1 and aux2"),
        (m + w * a1 + w * m, "loss_aux2 is computed from the main scores: cross_entropy(aux2, labels)"),
        (m + w * m + w * a2, "loss_aux1 is computed from the main scores: cross_entropy(aux1, labels)"),
        ((m + a1 + a2) / 3, "this is the average of the three losses; the formula is "
                            "loss_main + aux_weight * loss_aux1 + aux_weight * loss_aux2"),
        (m + 0.3 * a1 + 0.3 * a2, "the number 0.3 is written instead of the argument aux_weight"),
    ]
    for value, text in guesses:
        if abs(v - value) < 1e-4:
            return text
    return ""


def _aux_loss(fn, verbose: bool) -> Result:
    lines = _Lines("Checking step (c): aux_loss, on the scores of 4 photos for 10 classes", verbose)
    g = torch.Generator()
    g.manual_seed(6)
    main = torch.randn(4, 10, generator=g)
    aux1 = torch.randn(4, 10, generator=g) * 1.5
    aux2 = torch.randn(4, 10, generator=g) * 0.7
    labels = torch.tensor([3, 0, 7, 3])
    m = _ce(main, labels)
    a1 = _ce(aux1, labels)
    a2 = _ce(aux2, labels)
    leaves = [main.clone().requires_grad_(True), aux1.clone().requires_grad_(True), aux2.clone().requires_grad_(True)]
    value, err = _try(fn, (leaves[0], leaves[1], leaves[2]), labels)
    if not lines.check(err is None, "aux_loss(outputs, labels) runs", err):
        return lines.done("(c)", "")
    is_tensor = isinstance(value, torch.Tensor)
    hint = f"it returns a {type(value).__name__}: the loss must stay a tensor (no .item(), no float(...))"
    if value is None:
        hint = "it returns None: end with return loss"
    if not lines.check(is_tensor, "it returns a tensor (so that loss.backward() can work)", hint):
        return lines.done("(c)", "")
    if not lines.check(value.dim() == 0, "it is one number (a tensor with no axes), not one per photo",
                       f"it has shape {tuple(value.shape)}: use the cross_entropy of the cell, which averages over "
                       f"the photos"):
        return lines.done("(c)", "")
    want = m + 0.3 * a1 + 0.3 * a2
    got = float(value.detach())
    hint = f"yours gives {got:.4f}"
    guess = _wrong_formula(got, m, a1, a2, 0.3)
    if guess:
        hint += f": {guess}"
    lines.check(abs(got - want) < 1e-4,
                f"with aux_weight=0.3: main {m:.3f} + 0.3 x aux1 {a1:.3f} + 0.3 x aux2 {a2:.3f} = {want:.3f}", hint)
    for w in (0.0, 1.0):
        v, err = _try(fn, (main, aux1, aux2), labels, aux_weight=w)
        want_w = m + w * a1 + w * a2
        title = f"with aux_weight={w:g}: {want_w:.3f}"
        if w == 0.0:
            title += " (the main loss alone)"
        else:
            title += " (the plain sum of the three)"
        if err:
            lines.fail(title, err)
            continue
        if not isinstance(v, torch.Tensor):
            lines.fail(title, f"it returns a {type(v).__name__}")
            continue
        hint = f"yours gives {float(v):.4f}"
        guess = _wrong_formula(float(v), m, a1, a2, w)
        if guess:
            hint += f": {guess}"
        lines.check(abs(float(v) - want_w) < 1e-4, title, hint)
    grads_ok = value.requires_grad
    hint = "the loss is cut off from the scores (no gradient): do not use .item(), float(...) or .detach()"
    if grads_ok:
        for p in leaves:
            p.grad = None
        _, err = _try(value.backward)
        if err:
            grads_ok = False
            hint = err
        else:
            share = []
            for k, weight in enumerate((1.0, 0.3, 0.3)):
                expected = torch.softmax([main, aux1, aux2][k], dim=1)
                expected[torch.arange(4), labels] -= 1.0
                expected = expected * weight / 4
                grad = leaves[k].grad
                share.append(grad is not None and torch.allclose(grad, expected, atol=1e-5))
            grads_ok = share[0] and share[1] and share[2]
            names = []
            for k, name in enumerate(["main", "aux1", "aux2"]):
                if not share[k]:
                    names.append(name)
            hint = f"the gradient of {', '.join(names)} is not right: each output must be used once, with its weight"
    lines.check(grads_ok, "loss.backward() reaches all three outputs, with weights 1, 0.3 and 0.3", hint)
    return lines.done("(c)", "This is the loss GoogLeNet was trained with.")


def check_aux_loss(aux_loss) -> Result:
    """Step (c): main + aux_weight x aux1 + aux_weight x aux2, three cross-entropies, one number."""
    return _with_rng(lambda: _aux_loss(aux_loss, True))


# ============================================================================================== never stuck
_STEPS = {
    "make_aux_head": (_aux_head, R.make_aux_head, "(a)"),
    "GoogLeNetAux": (_googlenet_aux, R.GoogLeNetAux, "(b)"),
    "aux_loss": (_aux_loss, R.aux_loss, "(c)"),
}


def yours_or_reference(yours, name: str | None = None):
    """Your function or class if it passes all its checks; otherwise the course's reference version, with a note.

    So that the rest of the notebook works while a step is unfinished: use it as
    ``make_aux_head = T.yours_or_reference(make_aux_head)``. Fix your cell later, run it, its check and this line
    again.
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


# ============================================================================================== drawings and tables
def show_in_training_mode(model: nn.Module, input_shape, depth=None, title: str | None = None):
    """P.show_model, but with the network in training mode, so that every output is drawn.

    P.show_model always draws a network in prediction mode (model.eval()), where the auxiliary heads do not run.
    This keeps the network in training mode while it is drawn; dropout then sets random numbers to 0, which
    changes nothing in the drawing. The network's mode is restored afterwards.
    """
    from . import practical as P

    was = model.training
    model.train(True)

    def stay_in_training_mode(mode=True):
        return model

    object.__setattr__(model, "train", stay_in_training_mode)      # model.eval() inside show_model does nothing
    try:
        fig = P.show_model(model, input_shape, depth=depth, title=title)
    finally:
        del model.__dict__["train"]
        model.train(was)
    return fig


def parameters_per_part(net: nn.Module):
    """Print the parameters of each top-level part of a network, the auxiliary heads apart."""
    main = 0
    aux = 0
    rows = []
    for name, part in net.named_children():
        n = _params(part)
        rows.append((name, n))
        if name.startswith("aux"):
            aux += n
        else:
            main += n
    width = 18
    for name, n in rows:
        word = "used in training only" if name.startswith("aux") else ""
        print(f"  {name:<{width}} {n:>12,}   {word}")
    print(f"  {'main network':<{width}} {main:>12,}   used in training and for predictions")
    print(f"  {'auxiliary heads':<{width}} {aux:>12,}   thrown away after training")
    print(f"  {'total in training':<{width}} {main + aux:>12,}")
    return None


def _fmt_size(v: float) -> str:
    """A gradient size with 3 significant digits, without powers of ten (0.0000585, 0.00518, 0.571)."""
    if v == 0:
        return "0"
    digits = max(0, 2 - int(math.floor(math.log10(abs(v)))))
    return f"{v:.{digits}f}"


def gradient_per_part(net: nn.Module, photos: torch.Tensor, labels: torch.Tensor, loss_fn, aux_weight: float = 0.3,
                      parts=("stem", "blocks_3a_to_4a", "blocks_4b_to_4d", "blocks_4e_to_5b", "head")):
    """Print, for each part of the network, how big its gradient is with the main loss alone and with the
    auxiliary losses added. The size of a part's gradient: every number of the gradient of every weight of the
    part, squared, summed, square root. The network's weights are not changed."""
    was = net.training
    net.train()
    params = []
    owners = []
    for name in parts:
        for p in getattr(net, name).parameters():
            params.append(p)
            owners.append(name)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        outputs = net(photos)
    loss_main = loss_fn(outputs, labels, 0.0)
    loss_all = loss_fn(outputs, labels, aux_weight)
    grads_main = torch.autograd.grad(loss_main, params, retain_graph=True)
    grads_all = torch.autograd.grad(loss_all, params)
    net.train(was)
    print(f"loss of the main head alone: {float(loss_main.detach()):.3f};  with the auxiliary losses "
          f"(aux_weight={aux_weight:g}): {float(loss_all.detach()):.3f}")
    print()
    print(f"{'part':<18} {'main loss alone':>16} {'with the auxiliary losses':>26} {'times bigger':>13}")
    for name in parts:
        main_sq = 0.0
        all_sq = 0.0
        for k in range(len(params)):
            if owners[k] == name:
                main_sq += float(grads_main[k].pow(2).sum())
                all_sq += float(grads_all[k].pow(2).sum())
        main_size = main_sq ** 0.5
        all_size = all_sq ** 0.5
        ratio = all_size / main_size if main_size > 0 else float("nan")
        print(f"{name:<18} {_fmt_size(main_size):>16} {_fmt_size(all_size):>26} {ratio:>13.1f}")
    return None


# ============================================================================================== training
def train_with_aux(model: nn.Module, train_loader, val_loader, loss_fn, aux_weight: float = 0.3, epochs: int = 5,
                   lr: float = 1e-3, name: str | None = None, seed: int = 0, live: bool = True, verbose: bool = True):
    """P.train for a network that returns (main, aux1, aux2) in training mode. Returns a History (a dict).

    Each batch: outputs = model(images) in training mode, loss = loss_fn(outputs, labels, aux_weight), then the
    usual zero_grad / backward / step with Adam. The training loss and accuracy that are printed and drawn are the
    main head's, so that runs with different aux_weight are measured the same way (the total loss is kept in
    history["train_total_loss"]). Validation runs in prediction mode, where the network returns the main scores
    only. seed fixes the shuffling order and dropout, as in P.train.
    """
    from . import practical as P

    name = name or type(model).__name__
    dev = P._device()
    model.to(dev)
    ds = P._dataset_of(train_loader)
    x0, _ = ds[0]
    in_shape = tuple(x0.shape)
    out0 = P._dry_run(model, in_shape)
    P._check_labels(model, out0, train_loader)
    P._seed(seed)
    P._reseed_loader(train_loader, seed)
    opt = P._make_optimizer("adam", model, lr, 0.0)
    n_train = len(ds)
    hist = P.History(name=name, epoch=[], train_loss=[], train_acc=[], val_loss=[], val_acc=[], seconds=[],
                     total_seconds=0.0, best_val_acc=None, best_epoch=None, lr=lr, optimizer="adam",
                     batch_size=getattr(train_loader, "batch_size", None), n_train=n_train,
                     params=P._count_params(model), input_shape=in_shape, stopped_early=False,
                     train_total_loss=[], aux_weight=aux_weight)
    live_out = P._Live(live, quiet=not verbose)
    live_out.log(f"training '{name}': {epochs} epoch(s) over {n_train:,} images in batches of {hist['batch_size']}, "
                 f"adam with learning rate {lr}; loss = main + {aux_weight:g} x aux1 + {aux_weight:g} x aux2")
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        total = 0.0
        main_total = 0.0
        correct = 0
        n_items = 0
        stop_reason = None
        for x, y in train_loader:
            x = x.to(dev)
            y = y.to(dev)
            outputs = model(x)
            if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
                raise TypeError("in training mode the network must return three sets of scores: main, aux1, aux2")
            loss = loss_fn(outputs, y, aux_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            value = float(loss.detach())
            if not math.isfinite(value):
                stop_reason = (f"the loss became {value} (not a number) in epoch {epoch}: training stopped. The "
                               f"learning rate {lr} is too high for this model; try one 10 times smaller.")
                break
            main = outputs[0].detach()
            total += value * len(x)
            main_total += float(F.cross_entropy(main, y)) * len(x)
            correct += int((main.argmax(1) == y).sum())
            n_items += len(x)
        tl = main_total / max(n_items, 1)
        tt = total / max(n_items, 1)
        ta = 100.0 * correct / max(n_items, 1)
        vl, va = P.evaluate(model, val_loader)
        sec = time.time() - t_ep
        hist["epoch"].append(epoch)
        hist["train_loss"].append(tl)
        hist["train_total_loss"].append(tt)
        hist["train_acc"].append(ta)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        hist["seconds"].append(sec)
        if hist["best_val_acc"] is None or va > hist["best_val_acc"]:
            hist["best_val_acc"] = va
            hist["best_epoch"] = epoch
        line = f"epoch {epoch:>2}/{epochs}   training: loss {tl:.3f}"
        if aux_weight != 0:
            line += f" (with the auxiliary losses {tt:.3f})"
        line += f", accuracy {ta:5.1f}%   |   validation: loss {vl:.3f}, accuracy {va:5.1f}%   |   {sec:.1f} s"
        live_out.log(line)
        live_out.draw(hist)
        if stop_reason:
            live_out.log(stop_reason)
            hist["stopped_early"] = True
            break
    hist["total_seconds"] = time.time() - t_start
    live_out.log(f"done in {hist['total_seconds']:.1f} s; best validation accuracy {hist['best_val_acc']:.1f}% "
                 f"(epoch {hist['best_epoch']})")
    live_out.draw(hist, final=True)
    return hist._mark()


def compare_runs(history_without, history_with):
    """Validation accuracy of two runs side by side, epoch by epoch, and the difference."""
    a = history_without["val_acc"]
    b = history_with["val_acc"]
    print(f"{'epoch':>5}   {'without auxiliary heads':>23}   {'with auxiliary heads':>20}   {'difference':>10}")
    for k in range(min(len(a), len(b))):
        print(f"{k + 1:>5}   {a[k]:>22.1f}%   {b[k]:>19.1f}%   {b[k] - a[k]:>+10.1f}")
    best_a = history_without["best_val_acc"]
    best_b = history_with["best_val_acc"]
    print(f"{'best':>5}   {best_a:>22.1f}%   {best_b:>19.1f}%   {best_b - best_a:>+10.1f}")
    seconds_a = history_without["total_seconds"]
    seconds_b = history_with["total_seconds"]
    print(f"{'time':>5}   {seconds_a:>21.0f} s   {seconds_b:>18.0f} s")
    return None
