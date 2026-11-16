"""Day-1 exercise checkers (D1-02 training recipe, D1-03 architectures, D1-04 transfer learning).

Re-exported by `cvcourse.tests`, so notebooks call `tests.check_mixup(mixup)` etc.
Every checker prints ASCII [PASS] / [FAIL] lines with a hint and returns True / False.
An unfinished exercise (NotImplementedError) or a crash is reported as [FAIL] instead of
stopping "Run all"; later cells then fall back to the reference in `cvcourse.day1`
(see `use_or_reference`).
"""
from __future__ import annotations

import functools
import math
import traceback

import numpy as np

__all__ = [
    "use_or_reference",
    "check_augment_pipeline", "check_mixup", "check_warmup_cosine", "check_label_smoothing",
    "check_train_step", "check_overfit_one_batch",
    "check_inception_shapes", "check_inception_block", "check_residual_block", "check_bottleneck", "check_dw_params",
    "check_mini_resnet",
    "check_frozen", "check_param_groups", "check_gradcam", "check_extract_features",
]

_PASS = "[PASS]"
_FAIL = "[FAIL]"


def _report(name, ok, hint=""):
    print(f"{_PASS if ok else _FAIL}  {name}" + (f"\n      hint: {hint}" if (hint and not ok) else ""))
    return bool(ok)


def _safe(name):
    """Decorator: turn NotImplementedError / any exception inside a checker into a [FAIL] line."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            try:
                return fn(*a, **k)
            except NotImplementedError:
                return _report(name, False, "not implemented yet: fill in the blanks, then delete the `raise NotImplementedError` line")
            except Exception as e:  # noqa: BLE001
                tb = traceback.extract_tb(e.__traceback__)
                where = f" (line {tb[-1].lineno} in {tb[-1].name})" if tb else ""
                return _report(name, False, f"your code raised {type(e).__name__}: {e}{where}")
        return wrapper
    return deco


def use_or_reference(ok: bool, mine, reference, name: str = ""):
    """Return the student's implementation if its checker passed, else the reference (with a note)."""
    if ok:
        return mine
    print(f"note: using the reference implementation of {name or getattr(reference, '__name__', 'this exercise')} "
          f"(yours has not passed its checker yet)")
    return reference


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


# =========================================================================== D1-02
@_safe("augmentation pipeline")
def check_augment_pipeline(tf, size: int = 128):
    """tf: a transform (uint8 (3, S, S) tensor -> normalised float (3, size, size), random),
    or a factory build_tf(size) returning one."""
    torch, _, _ = _torch()
    import inspect
    if inspect.isfunction(tf):
        tf = tf(size)
    g = torch.Generator().manual_seed(0)
    img = (torch.rand(3, 144, 144, generator=g) * 255).to(torch.uint8)
    img[:, :72] = 30                                     # some structure so crops / flips differ
    torch.manual_seed(0)
    outs = [tf(img) for _ in range(4)]
    o = outs[0]
    ok = True
    if not isinstance(o, torch.Tensor) or tuple(o.shape) != (3, size, size):
        print(f"   output should be a tensor of shape (3, {size}, {size}), got {getattr(o, 'shape', type(o))}")
        ok = False
    elif o.dtype != torch.float32:
        print(f"   output dtype should be float32 (v2.ToDtype(torch.float32, scale=True)), got {o.dtype}")
        ok = False
    else:
        if float(o.min()) >= 0 and float(o.max()) <= 1:
            print("   values are in [0, 1]: did you forget v2.Normalize(mean, std)?")
            ok = False
        if all(torch.equal(outs[0], x) for x in outs[1:]):
            print("   four calls gave identical outputs: the pipeline has no randomness")
            ok = False
    return _report("augmentation pipeline", ok,
                   "RandomResizedCrop -> RandomHorizontalFlip -> (TrivialAugmentWide) -> ToDtype(float32, scale=True) -> Normalize")


@_safe("mixup")
def check_mixup(fn):
    """fn(x, y, alpha, num_classes, lam=None, perm=None) -> (x_mix, y_soft)."""
    torch, _, F = _torch()
    from . import day1
    g = torch.Generator().manual_seed(1)
    x = torch.randn(6, 3, 8, 8, generator=g)
    y = torch.tensor([0, 3, 1, 4, 2, 3])
    perm = torch.tensor([3, 0, 5, 1, 2, 4])
    ok = True
    for lam in (0.7, 0.25):
        xm, ym = fn(x, y, 0.2, 5, lam=lam, perm=perm)
        xr, yr = day1.mixup(x, y, 0.2, 5, lam=lam, perm=perm)
        if tuple(ym.shape) != (6, 5):
            print(f"   soft labels should have shape (B, num_classes) = (6, 5), got {tuple(ym.shape)}")
            ok = False
            break
        if not torch.allclose(xm, xr, atol=1e-6):
            print(f"   lam={lam}: mixed images differ from lam * x + (1 - lam) * x[perm]")
            ok = False
        if not torch.allclose(ym.float(), yr, atol=1e-6):
            print(f"   lam={lam}: soft labels differ from lam * onehot(y) + (1 - lam) * onehot(y[perm])")
            ok = False
    # random path: must run, keep shapes, rows of y must sum to 1
    torch.manual_seed(0); np.random.seed(0)
    xm, ym = fn(x, y, 0.4, 5)
    if tuple(xm.shape) != tuple(x.shape) or not torch.allclose(ym.float().sum(1), torch.ones(6), atol=1e-5):
        print("   with lam=None / perm=None: shapes must be kept and each soft-label row must sum to 1")
        ok = False
    return _report("mixup", ok, "y1 = F.one_hot(y, num_classes).float(); x_mix = lam * x + (1 - lam) * x[perm]; same for y1")


@_safe("warmup_cosine")
def check_warmup_cosine(fn):
    """fn(step, total, warmup, lr_max) -> lr."""
    total, warm, lr_max = 100, 10, 0.1
    got = np.array([float(fn(s, total, warm, lr_max)) for s in range(total + 1)])
    ok = True
    w = got[:warm]
    a = lr_max * (np.arange(warm) + 1) / warm
    b = lr_max * np.arange(warm) / warm
    if not (np.allclose(w, a, atol=1e-6) or np.allclose(w, b, atol=1e-6)):
        print(f"   warm-up should rise linearly to lr_max: got {np.round(w, 4)}")
        ok = False
    s = np.arange(warm, total + 1)
    want = 0.5 * lr_max * (1 + np.cos(np.pi * (s - warm) / (total - warm)))
    if not np.allclose(got[warm:], want, atol=1e-6):
        i = int(np.argmax(np.abs(got[warm:] - want)))
        print(f"   cosine part: at step {warm + i} got {got[warm + i]:.5f}, expected {want[i]:.5f}")
        ok = False
    if ok and not (abs(got[warm] - lr_max) < 1e-6 and abs(got[total]) < 1e-6):
        ok = False
    return _report("warmup_cosine", ok,
                   "step < warmup: lr_max * (step + 1) / warmup; then p = (step - warmup) / (total - warmup), "
                   "lr = 0.5 * lr_max * (1 + cos(pi * p))")


@_safe("label_smoothing_ce")
def check_label_smoothing(fn):
    torch, _, F = _torch()
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(16, 37, generator=g) * 3
    t = torch.randint(0, 37, (16,), generator=g)
    ok = True
    for eps in (0.0, 0.1, 0.3):
        got = float(fn(logits, t, eps))
        want = float(F.cross_entropy(logits, t, label_smoothing=eps))
        if not math.isclose(got, want, rel_tol=1e-4, abs_tol=1e-5):
            print(f"   eps={eps}: got {got:.5f}, expected {want:.5f}")
            ok = False
    return _report("label_smoothing_ce", ok,
                   "q = eps / C everywhere, plus (1 - eps) on the true class; loss = -(q * log_softmax(logits)).sum(-1).mean()")


@_safe("train_step")
def check_train_step(fn):
    """fn(model, x, y, loss_fn, optimizer) -> float loss; checks order of zero_grad / backward / step."""
    torch, nn, F = _torch()
    import copy

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Dropout(0.0), nn.Linear(8, 3))
    ref = copy.deepcopy(model)
    x = torch.randn(10, 5)
    y = torch.randint(0, 3, (10,))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    ropt = torch.optim.SGD(ref.parameters(), lr=0.1)
    model.eval()                       # the step must switch to train mode itself
    ok = True
    losses = []
    for _ in range(3):
        lv = fn(model, x, y, F.cross_entropy, opt)
        rl = F.cross_entropy(ref(x), y)
        ropt.zero_grad()
        rl.backward()
        ropt.step()
        losses.append((lv, rl.item()))
    if not isinstance(losses[0][0], float):
        print(f"   return loss.item() (a Python float), got {type(losses[0][0]).__name__}")
        ok = False
    elif not all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6) for a, b in losses):
        print(f"   losses {[round(a, 4) for a, _ in losses]} vs expected {[round(b, 4) for _, b in losses]}")
        ok = False
    same = all(torch.allclose(p, q, atol=1e-6) for p, q in zip(model.parameters(), ref.parameters()))
    if not same:
        print("   parameters after 3 steps differ from the reference: zero the gradients before backward "
              "(otherwise they accumulate) and call optimizer.step() after backward")
        ok = False
    if not model.training:
        print("   call model.train() inside the step (dropout / BatchNorm behave differently in eval mode)")
        ok = False
    return _report("train_step", ok, "model.train(); out = model(x); loss = loss_fn(out, y); "
                                     "optimizer.zero_grad(); loss.backward(); optimizer.step(); return loss.item()")


@_safe("overfit one batch")
def check_overfit_one_batch(fn):
    """fn(model, x, y, steps, lr) -> list of losses; must drive the loss of one batch to ~0."""
    torch, nn, _ = _torch()
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 37))
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 37, (16,))
    losses = list(map(float, fn(model, x, y, 150, 1e-2)))
    ok = len(losses) >= 10
    if not ok:
        print("   return the list of per-step losses")
    else:
        first, last = losses[0], min(losses[-5:])
        print(f"   loss: step 0 = {first:.3f} (ln 37 = {math.log(37):.3f}), last = {last:.4f}")
        if abs(first - math.log(37)) > 1.0:
            print("   the initial loss should be close to ln(37): is the model freshly initialised, the loss cross-entropy?")
            ok = False
        if last > 0.05:
            print("   the loss did not go to ~0 on a single batch: check that you reuse the SAME batch every step "
                  "and that zero_grad / backward / step are all there")
            ok = False
    return _report("overfit one batch", ok,
                   "loop `steps` times over the same (x, y): loss = F.cross_entropy(model(x), y); zero_grad; backward; step")


# =========================================================================== D1-03
def _nparams(m):
    return sum(p.numel() for p in m.parameters())


@_safe("inception output channels")
def check_inception_shapes(fn):
    """fn(c_in, f1, f3r, f3, f5r, f5, fpool) -> output channels.

    Same test as the original `tests.check_inception_shapes` (which this re-export replaces),
    wrapped so an unfinished stub prints [FAIL] instead of raising."""
    ok = fn(192, 64, 96, 128, 16, 32, 32) == 256 and fn(256, 128, 128, 192, 32, 96, 64) == 480
    return _report("inception output channels", ok, "concatenate along channels: f1 + f3 + f5 + fpool")


@_safe("InceptionBlock")
def check_inception_block(cls):
    torch, _, _ = _torch()
    from . import day1
    ok = True
    for cfg, hw, want_c in [((192, 64, 96, 128, 16, 32, 32), 28, 256), ((256, 128, 128, 192, 32, 96, 64), 14, 480)]:
        m = cls(*cfg)
        out = m(torch.randn(2, cfg[0], hw, hw))
        if tuple(out.shape) != (2, want_c, hw, hw):
            print(f"   InceptionBlock{cfg}: output {tuple(out.shape)}, expected {(2, want_c, hw, hw)}")
            ok = False
        pr, pw = _nparams(m), _nparams(day1.InceptionBlock(*cfg))
        if pr != pw:
            print(f"   InceptionBlock{cfg}: {pr:,} parameters, expected {pw:,} (convs with bias, no BatchNorm; "
                  f"the pool branch has a 1x1 conv after the max-pool)")
            ok = False
        if ok and float(out.detach().min()) < 0:
            print("   outputs should be >= 0: every branch ends with a ReLU")
            ok = False
    return _report("InceptionBlock", ok, "4 branches with padding 0 / 1 / 2 / (pool: kernel 3, stride 1, padding 1); "
                                         "torch.cat([...], dim=1)")


def _zero_last_bn(block, name):
    bn = getattr(block, name)
    bn.weight.data.zero_()
    bn.bias.data.zero_()


@_safe("BasicBlock (residual)")
def check_residual_block(cls):
    torch, nn, F = _torch()
    from . import day1
    ok = True
    x = torch.randn(2, 64, 16, 16)
    for c_in, c_out, stride in [(64, 64, 1), (64, 128, 2)]:
        torch.manual_seed(0)
        m = cls(c_in, c_out, stride).eval()
        out = m(x)
        want_shape = (2, c_out, 16 // stride, 16 // stride)
        if tuple(out.shape) != want_shape:
            print(f"   BasicBlock({c_in}, {c_out}, stride={stride}): output {tuple(out.shape)}, expected {want_shape}")
            ok = False
            continue
        pr, pw = _nparams(m), _nparams(day1.BasicBlock(c_in, c_out, stride))
        if pr != pw:
            kind = "identity shortcut (no parameters)" if (c_in == c_out and stride == 1) else "1x1 conv (stride) + BN projection shortcut"
            print(f"   BasicBlock({c_in}, {c_out}, stride={stride}): {pr:,} params, expected {pw:,} -> {kind}")
            ok = False
        # switch the residual branch off: the block must reduce to ReLU(shortcut(x))
        _zero_last_bn(m, "bn2")
        sc = m.shortcut(x) if hasattr(m, "shortcut") else x
        if not torch.allclose(m(x), F.relu(sc), atol=1e-5):
            print(f"   BasicBlock({c_in}, {c_out}, stride={stride}): with bn2 zeroed the output should be "
                  f"relu(shortcut(x)) - is the shortcut added BEFORE the final ReLU?")
            ok = False
        if c_in == c_out and stride == 1 and not torch.allclose(m(x), F.relu(x), atol=1e-5):
            print("   identity case: with the residual branch off the block must pass x through (relu(x))")
            ok = False
    return _report("BasicBlock (residual)", ok,
                   "out = bn2(conv2(relu(bn1(conv1(x))))); return relu(out + self.shortcut(x))")


@_safe("Bottleneck")
def check_bottleneck(cls):
    torch, nn, F = _torch()
    from torchvision.models.resnet import Bottleneck as TVB
    ok = True
    for c_in, mid, stride, hw in [(256, 64, 1, 14), (256, 128, 2, 14), (64, 64, 1, 8)]:
        torch.manual_seed(0)
        m = cls(c_in, mid, stride).eval()
        x = torch.randn(2, c_in, hw, hw)
        out = m(x)
        want_shape = (2, 4 * mid, hw // stride, hw // stride)
        if tuple(out.shape) != want_shape:
            print(f"   Bottleneck({c_in}, {mid}, stride={stride}): output {tuple(out.shape)}, expected {want_shape}")
            ok = False
            continue
        down = None
        if stride != 1 or c_in != 4 * mid:
            down = nn.Sequential(nn.Conv2d(c_in, 4 * mid, 1, stride, bias=False), nn.BatchNorm2d(4 * mid))
        pw = _nparams(TVB(c_in, mid, stride, down))
        if _nparams(m) != pw:
            print(f"   Bottleneck({c_in}, {mid}, stride={stride}): {_nparams(m):,} params, torchvision's has {pw:,}")
            ok = False
        _zero_last_bn(m, "bn3")
        sc = m.shortcut(x) if hasattr(m, "shortcut") else x
        if not torch.allclose(m(x), F.relu(sc), atol=1e-5):
            print(f"   Bottleneck({c_in}, {mid}, stride={stride}): with bn3 zeroed the output should be relu(shortcut(x))")
            ok = False
    return _report("Bottleneck", ok, "1x1 (c_in->mid) BN ReLU, 3x3 (mid->mid, stride) BN ReLU, 1x1 (mid->4*mid) BN; "
                                     "+ shortcut; ReLU. Projection when stride != 1 or c_in != 4*mid")


@_safe("DepthwiseSeparable parameter count")
def check_dw_params(cls):
    torch, nn, _ = _torch()
    ok = True
    for c_in, c_out, stride in [(32, 64, 1), (128, 256, 2)]:
        m = cls(c_in, c_out, stride)
        out = m(torch.randn(2, c_in, 16, 16))
        want_shape = (2, c_out, 16 // stride, 16 // stride)
        if tuple(out.shape) != want_shape:
            print(f"   DepthwiseSeparable({c_in}, {c_out}, {stride}): output {tuple(out.shape)}, expected {want_shape}")
            ok = False
        want = c_in * 9 + 2 * c_in + c_in * c_out + 2 * c_out
        got = _nparams(m)
        if got != want:
            print(f"   DepthwiseSeparable({c_in}, {c_out}): {got:,} params, expected {want:,} = "
                  f"{c_in}*9 (depthwise) + {c_in}*{c_out} (pointwise) + BN (2 per channel), no conv bias")
            ok = False
        dw = [mm for mm in m.modules() if isinstance(mm, nn.Conv2d) and mm.groups == c_in and mm.groups > 1]
        if not dw:
            print("   no depthwise conv found: the 3x3 conv needs groups=c_in")
            ok = False
    std = 64 * 32 * 9
    print(f"   for comparison: a standard 3x3 conv 32->64 has {std:,} weights vs {32 * 9 + 32 * 64:,} "
          f"for depthwise + pointwise ({std / (32 * 9 + 32 * 64):.1f}x fewer)")
    return _report("DepthwiseSeparable parameter count", ok,
                   "nn.Conv2d(c_in, c_in, 3, stride, 1, groups=c_in, bias=False) then nn.Conv2d(c_in, c_out, 1, bias=False)")


@_safe("MiniResNet")
def check_mini_resnet(cls):
    torch, nn, _ = _torch()
    m = cls(num_classes=37).eval()
    ok = True
    for hw in (128, 96):
        out = m(torch.randn(2, 3, hw, hw))
        if tuple(out.shape) != (2, 37):
            print(f"   input {hw} px: output {tuple(out.shape)}, expected (2, 37) - global average pooling makes any size work")
            ok = False
    blocks = [mm for mm in m.modules() if hasattr(mm, "shortcut")]
    if len(blocks) < 3:
        print(f"   found {len(blocks)} residual blocks (modules with a .shortcut); expected at least 3 (one per stage)")
        ok = False
    n = _nparams(m)
    if not (1e5 <= n <= 3e6):
        print(f"   {n:,} parameters; expected roughly 0.1 M - 3 M for widths (32, 64, 128)")
        ok = False
    if ok:
        strided = [mm for mm in blocks if any(isinstance(c, nn.Conv2d) and c.stride == (2, 2) for c in mm.modules())]
        if len(strided) < 2:
            print("   stages 2 and 3 should start with a stride-2 block (downsample + double the channels)")
            ok = False
    print(f"   MiniResNet: {n:,} parameters, {len(blocks)} residual blocks")
    return _report("MiniResNet", ok, "stem -> stage1 (stride 1) -> stage2 (stride 2) -> stage3 (stride 2) -> "
                                     "AdaptiveAvgPool2d(1) -> flatten -> Linear")


# =========================================================================== D1-04
def _resnet18():
    from torchvision.models import resnet18
    return resnet18(weights=None)


def _stage_params(model, stages):
    return sum(p.numel() for n, p in model.named_parameters()
               if ("stem" if n.split(".")[0] in ("conv1", "bn1") else n.split(".")[0]) in stages)


@_safe("freeze")
def check_frozen(fn):
    ok = True
    cases = {"layer3": ("layer4", "fc"), "layer4": ("fc",), "stem": ("layer1", "layer2", "layer3", "layer4", "fc"),
             "layer1": ("layer2", "layer3", "layer4", "fc")}
    for until, trainable in cases.items():
        m = _resnet18()
        r = fn(m, until)
        m = r if r is not None else m
        got = sum(p.numel() for p in m.parameters() if p.requires_grad)
        want = _stage_params(m, trainable)
        if got != want:
            print(f"   freeze(model, until={until!r}): {got:,} trainable params, expected {want:,} "
                  f"(only {', '.join(trainable)} trainable)")
            ok = False
    m = _resnet18()
    for p in m.parameters():
        p.requires_grad = False
    fn(m, "layer2")
    if sum(p.numel() for p in m.parameters() if p.requires_grad) != _stage_params(m, ("layer3", "layer4", "fc")):
        print("   freeze must also UNfreeze the later stages (set requires_grad=True there), so it can be called repeatedly")
        ok = False
    return _report("freeze", ok, "for name, p in model.named_parameters(): p.requires_grad = stage(name) comes after `until`")


@_safe("param_groups_with_lrs")
def check_param_groups(fn):
    from . import day1
    ok = True
    m = _resnet18()
    groups = fn(m, 1e-3, 0.3)
    ids = [id(p) for g in groups for p in g["params"]]
    if len(ids) != len(set(ids)):
        print("   a parameter appears in more than one group")
        ok = False
    if set(ids) != {id(p) for p in m.parameters()}:
        print("   the groups must cover every trainable parameter exactly once")
        ok = False
    want = {"stem": 1e-3 * 0.3 ** 5, "layer1": 1e-3 * 0.3 ** 4, "layer2": 1e-3 * 0.3 ** 3,
            "layer3": 1e-3 * 0.3 ** 2, "layer4": 1e-3 * 0.3, "fc": 1e-3}
    name_of = {}
    for n, p in m.named_parameters():
        top = n.split(".")[0]
        name_of[id(p)] = "stem" if top in ("conv1", "bn1") else top
    for g in groups:
        stages = {name_of[id(p)] for p in g["params"]}
        for s in stages:
            if not math.isclose(g["lr"], want[s], rel_tol=1e-6):
                print(f"   parameters of {s} have lr {g['lr']:.2e}, expected {want[s]:.2e}")
                ok = False
                break
    # frozen params must be excluded
    m2 = day1.freeze(_resnet18(), "layer2")
    g2 = fn(m2, 1e-3, 0.3)
    if any(not p.requires_grad for g in g2 for p in g["params"]):
        print("   frozen parameters (requires_grad=False) should not be passed to the optimiser")
        ok = False
    if any(len(g["params"]) == 0 for g in g2):
        print("   drop empty groups")
        ok = False
    return _report("param_groups_with_lrs", ok,
                   "stage index i in [stem, layer1, ..., layer4, fc]; lr = base_lr * decay ** (5 - i)")


@_safe("gradcam")
def check_gradcam(fn):
    torch, nn, _ = _torch()
    from . import torchkit
    torch.manual_seed(0)
    feat = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
    model = nn.Sequential(feat, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 5)).eval()
    x = torch.randn(3, 32, 32)
    ok = True
    for c in (None, 2, 4):
        got = np.asarray(fn(model, feat[3], x, c), float)
        want = torchkit.gradcam(model, feat[3], x, c)
        if got.shape != want.shape:
            print(f"   class_idx={c}: map shape {got.shape}, expected {want.shape} (upsample to the input size)")
            ok = False
            break
        if not np.allclose(got, want, atol=1e-4):
            print(f"   class_idx={c}: max |diff| = {np.abs(got - want).max():.3f} vs the reference map")
            ok = False
    hooks = sum(len(m._forward_hooks) + len(m._backward_hooks) for m in model.modules())
    if hooks:
        print(f"   {hooks} hook(s) left registered on the model: call handle.remove() in a finally block")
        ok = False
    return _report("gradcam", ok, "weights = grads.mean((2, 3)); cam = relu((weights[..., None, None] * acts).sum(1)); "
                                  "upsample bilinearly, min-max scale to [0, 1]")


@_safe("extract_features")
def check_extract_features(fn):
    torch, nn, _ = _torch()
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(0)
    backbone = nn.Sequential(nn.Conv2d(3, 6, 3), nn.BatchNorm2d(6), nn.ReLU(), nn.Dropout(0.5),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten())
    backbone.train()
    x = torch.randn(10, 3, 12, 12)
    y = torch.arange(10)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    feats, labels = fn(backbone, loader)
    ok = True
    if tuple(feats.shape) != (10, 6) or tuple(labels.shape) != (10,):
        print(f"   got features {tuple(feats.shape)} and labels {tuple(labels.shape)}, expected (10, 6) and (10,)")
        ok = False
    else:
        backbone.eval()
        with torch.no_grad():
            want = backbone(x)
        if not torch.allclose(feats, want, atol=1e-5):
            print("   features differ from backbone(x) in eval mode: call backbone.eval() (BatchNorm / Dropout!)")
            ok = False
        if feats.requires_grad:
            print("   features still require grad: wrap the loop in torch.no_grad() (saves memory and time)")
            ok = False
        if not torch.equal(labels, y):
            print("   labels are not in loader order")
            ok = False
    return _report("extract_features", ok, "backbone.eval(); with torch.no_grad(): collect backbone(x) and y per batch; torch.cat")
