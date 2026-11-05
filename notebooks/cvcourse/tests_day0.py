"""Checkers for D0-00 (setup and tensors) and the PyTorch part of D1-01.

Re-exported by `cvcourse.tests`, so notebooks call `tests.check_to_chw(to_chw)` etc.
"""
from __future__ import annotations

import numpy as np

__all__ = ["check_to_chw", "check_normalize", "check_d0_train_step", "check_linear_fit", "check_tiny_convnet"]

_PASS = "[PASS]"
_FAIL = "[FAIL]"


def _report(name, ok, hint=""):
    print(f"{_PASS if ok else _FAIL}  {name}" + (f"\n      hint: {hint}" if (hint and not ok) else ""))
    return ok


# ----------------------------------------------------------------- Day 0 ---
def check_to_chw(fn):
    """fn(img_hwc: np.ndarray) -> torch float32 tensor (C, H, W) in [0, 1]."""
    import torch

    rng = np.random.default_rng(0)
    ok, msgs = True, []
    u8 = rng.integers(0, 256, (5, 7, 3), dtype=np.uint8)
    f32 = rng.uniform(size=(6, 4, 3)).astype(np.float32)
    for name, x, want in [("uint8 (5, 7, 3)", u8, u8.astype(np.float32) / 255), ("float32 (6, 4, 3)", f32, f32)]:
        try:
            got = fn(x)
        except Exception as e:  # noqa: BLE001
            ok = False
            msgs.append(f"   {name}: raised {type(e).__name__}: {e}")
            continue
        if not isinstance(got, torch.Tensor):
            ok = False
            msgs.append(f"   {name}: returned {type(got).__name__}, expected a torch.Tensor")
            continue
        if tuple(got.shape) != want.shape[::-1][:1] + want.shape[:2]:
            ok = False
            msgs.append(f"   {name}: shape {tuple(got.shape)}, expected {(want.shape[2], *want.shape[:2])}")
            continue
        if got.dtype != torch.float32:
            ok = False
            msgs.append(f"   {name}: dtype {got.dtype}, expected torch.float32")
            continue
        if not np.allclose(got.numpy(), want.transpose(2, 0, 1), atol=1e-6):
            ok = False
            msgs.append(f"   {name}: values differ (uint8 images must be divided by 255; float images kept as they are)")
    for m in msgs:
        print(m)
    return _report("to_chw", ok, "torch.from_numpy(img).permute(2, 0, 1); divide by 255 only if the dtype is uint8")


def check_normalize(fn):
    """fn(x, mean, std) for x of shape (C, H, W) or (B, C, H, W); mean / std are length-C sequences."""
    import torch

    g = torch.Generator().manual_seed(1)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ok = True
    for shape in [(3, 5, 4), (2, 3, 5, 4)]:
        x = torch.rand(shape, generator=g)
        want = (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
        try:
            got = fn(x, mean, std)
        except Exception as e:  # noqa: BLE001
            print(f"   input {shape}: raised {type(e).__name__}: {e}")
            ok = False
            continue
        if not (isinstance(got, torch.Tensor) and tuple(got.shape) == shape and torch.allclose(got, want, atol=1e-5)):
            print(f"   input {shape}: wrong result" + (f" (shape {tuple(got.shape)})" if isinstance(got, torch.Tensor) else ""))
            ok = False
    return _report("normalize (broadcasting)", ok,
                   "reshape mean and std to (C, 1, 1) so they broadcast over H and W (and over B in front)")


def _ref_model_and_data():
    import torch
    import torch.nn as nn

    torch.manual_seed(3)
    model = nn.Linear(3, 1)
    x = torch.randn(8, 3)
    y = torch.randn(8, 1)
    return model, x, y


def check_d0_train_step(fn):
    """fn(model, x, y, loss_fn, optimizer) -> float loss (before the update)."""
    import copy

    import torch
    import torch.nn as nn

    model, x, y = _ref_model_and_data()
    ref = copy.deepcopy(model)
    lr = 0.1
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    ok = True
    # reference: two clean SGD steps by hand
    expected_losses = []
    for _ in range(2):
        loss = nn.functional.mse_loss(ref(x), y)
        expected_losses.append(loss.item())
        grads = torch.autograd.grad(loss, list(ref.parameters()))
        with torch.no_grad():
            for p, g in zip(ref.parameters(), grads):
                p -= lr * g
    try:
        got = [fn(model, x, y, nn.MSELoss(), opt) for _ in range(2)]
    except Exception as e:  # noqa: BLE001
        print(f"   raised {type(e).__name__}: {e}")
        return _report("train_step", False, "forward -> loss -> optimizer.zero_grad() -> loss.backward() -> optimizer.step()")
    try:
        got = [float(v) for v in got]
    except (TypeError, ValueError):
        print("   return the loss as a Python float: loss.item()")
        ok = False
    if ok and not np.allclose(got, expected_losses, atol=1e-5):
        print(f"   returned losses {np.round(got, 4)}, expected {np.round(expected_losses, 4)}")
        ok = False
    same = all(torch.allclose(a, b, atol=1e-5) for a, b in zip(model.parameters(), ref.parameters()))
    if not same:
        print("   weights after two calls differ from two clean SGD steps "
              "(missing optimizer.step()? or missing optimizer.zero_grad(), so gradients accumulate?)")
        ok = False
    return _report("train_step", ok, "forward -> loss -> optimizer.zero_grad() -> loss.backward() -> optimizer.step(); return loss.item()")


def check_linear_fit(model, true_w=3.0, true_b=-1.0, tol=0.15):
    """The trained nn.Linear(1, 1) should recover y = 3 x - 1."""
    w = float(model.weight.detach().reshape(-1)[0])
    b = float(model.bias.detach().reshape(-1)[0])
    print(f"   learned w = {w:.3f} (true {true_w}),  b = {b:.3f} (true {true_b})")
    ok = abs(w - true_w) < tol and abs(b - true_b) < tol
    return _report("linear model fitted", ok, "loop over epochs and over the loader, call train_step on every batch; "
                                               "if the loss stalls, train for more epochs")


# ----------------------------------------------------------------- Day 1 (PyTorch part of D1-01) ---
def check_tiny_convnet(model):
    """model: TinyConvNet instance with conv1, conv2, conv3, pool, fc (conv32-pool-conv64-pool-conv64-GAP-fc10)."""
    import torch
    import torch.nn.functional as F

    ok = True
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != 56_394:
        print(f"   {n_params:,d} parameters, expected 56,394 (keep __init__ as given)")
        ok = False
    x = torch.randn(4, 1, 28, 28, generator=torch.Generator().manual_seed(0))
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(x)
    except NotImplementedError:
        model.train(was)
        return _report("TinyConvNet.forward", False, "write the forward pass")
    except Exception as e:  # noqa: BLE001
        model.train(was)
        print(f"   forward raised {type(e).__name__}: {e}")
        return _report("TinyConvNet.forward", False, "check the shapes: (B,1,28,28) -> (B,32,14,14) -> (B,64,7,7) -> (B,64) -> (B,10)")
    if tuple(out.shape) != (4, 10):
        print(f"   output shape {tuple(out.shape)}, expected (4, 10)")
        ok = False
    elif all(hasattr(model, a) for a in ("conv1", "conv2", "conv3", "fc")):
        with torch.no_grad():
            h = F.max_pool2d(F.relu(model.conv1(x)), 2)
            h = F.max_pool2d(F.relu(model.conv2(h)), 2)
            h = F.relu(model.conv3(h)).mean(dim=(2, 3))
            want = model.fc(h)
        if not torch.allclose(out, want, atol=1e-5):
            print("   output shape is right but the values differ from conv-ReLU-pool, conv-ReLU-pool, conv-ReLU, GAP, fc")
            ok = False
    model.train(was)
    return _report("TinyConvNet.forward", ok,
                   "ReLU after every conv; pool after conv1 and conv2 only; GAP = x.mean(dim=(2, 3)); no softmax (the loss does it)")
