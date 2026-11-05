"""PyTorch helpers shared by every course notebook.

Import with ``from cvcourse import torchkit`` (the notebooks put the ``notebooks/``
folder on ``sys.path``). Nothing in here is "the concept" of a notebook: these are
the plumbing pieces (device, data, training loop, checkpoints, plots) that would
otherwise be copy-pasted into every notebook.

Public API
----------
get_device(verbose=True) -> torch.device
seed_everything(seed=0)
data_root() -> Path ; CHECKPOINTS: Path
count_parameters(model, trainable_only=False) -> int
summarize(model, input_size) -> str            (forward hooks; params and MACs per layer)
profile(model, input_size) -> dict(params, macs)
tune_cpu_threads() -> int                         (fastest intra-op thread count on CPU)
fit(model, train_loader, val_loader, loss_fn, optimizer, epochs, ...) -> history dict
evaluate(model, loader, loss_fn=None, metric_fn=None, device=None) -> (loss, metric)
accuracy(logits, targets) -> float
save_checkpoint(model, path, **extra) ; load_checkpoint(model, path, map_location="cpu") -> dict extras
lr_range_test(model, loader, loss_fn, optimizer_cls, lr_min=1e-6, lr_max=1, steps=100) -> (lrs, losses)
IMAGENET_MEAN, IMAGENET_STD ; denormalize(tensor) -> numpy HWC in [0, 1]
show_batch(images, labels=None, classes=None, n=16) -> Figure
get_pet_datasets(size=224, train_tf=None, eval_tf=None, subset=None, target="category") -> (train, val, classes)
get_fashion_mnist(root=None) -> (train, test, classes)
get_penn_fudan(root=None, download=True) -> Dataset
ShapesDataset(n, size, seed, task="detection" | "segmentation" | "grid", grid=8)
detection_collate(batch) -> (list of images, list of targets)
gradcam(model, target_layer, image_tensor, class_idx=None) -> (H, W) numpy heat map in [0, 1]
"""
from __future__ import annotations

import copy
import math
import os
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset

if __package__ in (None, ""):  # allow `python notebooks/cvcourse/torchkit.py` for the smoke test
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "cvcourse"

from . import shapes as _shapes

__all__ = [
    "get_device", "seed_everything", "data_root", "CHECKPOINTS", "count_parameters", "summarize",
    "profile", "tune_cpu_threads", "fit", "evaluate", "accuracy", "save_checkpoint", "load_checkpoint", "lr_range_test",
    "IMAGENET_MEAN", "IMAGENET_STD", "denormalize", "show_batch", "get_pet_datasets",
    "get_fashion_mnist", "get_penn_fudan", "ShapesDataset", "detection_collate", "gradcam",
    "FASHION_MNIST_MEAN", "FASHION_MNIST_STD", "PET_SEG_CLASSES",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD = (0.3530,)
PET_SEG_CLASSES = ["background", "pet", "border"]


# =========================================================================== basics
def get_device(verbose: bool = True) -> torch.device:
    """Pick the best available device: CUDA > Apple MPS > CPU, and say which one."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        msg = f"cuda ({torch.cuda.get_device_name(0)})"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = torch.device("mps")
        msg = "mps (Apple GPU)"
    else:
        dev = torch.device("cpu")
        msg = f"cpu ({torch.get_num_threads()} threads) - training cells are sized for this"
    if verbose:
        print(f"device: {msg}")
    return dev


def seed_everything(seed: int = 0) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _repo_root() -> Path:
    # notebooks/cvcourse/torchkit.py -> repo root is two levels above the package
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """`<repo>/data` (or $CVCOURSE_DATA if set), created on demand."""
    root = Path(os.environ.get("CVCOURSE_DATA", _repo_root() / "data"))
    root.mkdir(parents=True, exist_ok=True)
    return root


CHECKPOINTS: Path = data_root() / "checkpoints"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters (all of them, or only those with requires_grad)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# =========================================================================== summary
class _Text(str):
    """A str that displays without quotes when it is the last expression of a cell."""

    def __repr__(self) -> str:  # noqa: D401
        return str(self)


def _shape_of(out: Any) -> str:
    if isinstance(out, torch.Tensor):
        return str(tuple(out.shape))
    if isinstance(out, (list, tuple)):
        return "[" + ", ".join(_shape_of(o) for o in out[:3]) + (", ..." if len(out) > 3 else "") + "]"
    if isinstance(out, dict):
        return "{" + ", ".join(f"{k}: {_shape_of(v)}" for k, v in list(out.items())[:3]) + "}"
    return type(out).__name__


def _macs(module: nn.Module, out: Any) -> int:
    """Multiply-accumulates per sample for conv / linear layers (0 for everything else)."""
    if not isinstance(out, torch.Tensor) or out.dim() == 0:
        return 0
    per_sample = out.numel() // max(out.shape[0], 1)
    if isinstance(module, nn.modules.conv._ConvNd):
        k = int(np.prod(module.kernel_size))
        return per_sample * (module.in_channels // module.groups) * k
    if isinstance(module, nn.Linear):
        return per_sample * module.in_features
    return 0


def _trace(model: nn.Module, input_size: Sequence[int], device=None, depth: int | None = None):
    """Hook every module, run a dummy forward, return (display_rows, total_macs).

    MACs are computed on leaf conv / linear modules and rolled up into whichever rows
    are displayed (so container rows such as `layer1` show the MACs of everything inside).
    """
    if len(input_size) == 3:
        input_size = (1, *input_size)
    device = device or next((p.device for p in model.parameters()), torch.device("cpu"))
    records, hooks = [], []
    for name, m in model.named_modules():
        def hook(mod, inp, out, name=name):
            records.append(dict(name=name, type=type(mod).__name__, shape=_shape_of(out),
                                params=sum(p.numel() for p in mod.parameters()),
                                leaf=len(list(mod.children())) == 0, macs=_macs(mod, out)))
        hooks.append(m.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(*input_size, device=device))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    leaf_macs = [(r["name"], r["macs"]) for r in records if r["leaf"]]
    total_macs = int(sum(m for _, m in leaf_macs))

    def rolled(name: str) -> int:
        if name == "":
            return total_macs
        return int(sum(m for n, m in leaf_macs if n == name or n.startswith(name + ".")))

    rows = []
    for r in records:
        name = r["name"]
        if name == "":
            if len(records) > 1:
                continue            # root is only shown when the model is a single layer
            r = dict(r, name="(model)")
        level = name.count(".")
        if depth is None and not r["leaf"]:
            continue
        if depth is not None and (level > depth or (level < depth and not r["leaf"])):
            continue
        rows.append(dict(r, macs=rolled(name)))
    return rows, total_macs


def summarize(model: nn.Module, input_size: Sequence[int], device=None, depth: int | None = None) -> str:
    """Layer-by-layer table: name, type, output shape, parameters, MACs.

    Built with forward hooks on a dummy zero input of `input_size` ((C, H, W) or (B, C, H, W)).
    `depth=None` lists the leaf modules; `depth=k` lists modules k dots deep (e.g. depth=0 for
    the top-level stages of a torchvision ResNet); container rows show the MACs of everything
    inside them. Returns a string (prints nicely in a cell).
    """
    rows, total_macs = _trace(model, input_size, device, depth)
    w_name = max([len("layer")] + [len(r["name"]) for r in rows]) + 2
    w_type = max([len("type")] + [len(r["type"]) for r in rows]) + 2
    w_shape = max([len("output shape")] + [len(r["shape"]) for r in rows]) + 2
    lines = [f"{'layer':<{w_name}}{'type':<{w_type}}{'output shape':<{w_shape}}{'params':>12}{'MACs':>16}"]
    lines.append("-" * len(lines[0]))
    for r in rows:
        lines.append(f"{r['name']:<{w_name}}{r['type']:<{w_type}}{r['shape']:<{w_shape}}{r['params']:>12,d}{r['macs']:>16,d}")
    lines.append("-" * len(lines[0]))
    total, trainable = count_parameters(model), count_parameters(model, trainable_only=True)
    lines.append(f"total params: {total:,d}   trainable: {trainable:,d}   "
                 f"MACs per image (conv + linear): {total_macs:,d} (~{total_macs / 1e9:.2f} G)")
    return _Text("\n".join(lines))


def profile(model: nn.Module, input_size: Sequence[int], device=None) -> dict:
    """{'params': int, 'macs': int} for one image of `input_size` (conv + linear MACs only)."""
    _, total_macs = _trace(model, input_size, device, None)
    return {"params": count_parameters(model), "macs": total_macs}


def tune_cpu_threads(candidates: Sequence[int] = (1, 2, 4, 8), verbose: bool = True) -> int:
    """Pick the fastest intra-op thread count for small models on this CPU and set it.

    On a shared or heavily loaded machine, the PyTorch default (all cores) can be several
    times slower than 1-2 threads for the small batches used in class. Takes about 1 s.
    No-op (returns the current setting) when a CUDA / MPS device is available.
    """
    if torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return torch.get_num_threads()
    max_t = os.cpu_count() or 1
    cands = sorted({min(c, max_t) for c in candidates})
    model = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10))
    x = torch.randn(16, 3, 64, 64)
    timings = {}
    for t in cands:
        torch.set_num_threads(t)
        model(x).sum().backward()                      # warm-up
        t0 = time.perf_counter()
        for _ in range(3):
            model(x).sum().backward()
        timings[t] = (time.perf_counter() - t0) / 3
    best = min(timings, key=timings.get)
    torch.set_num_threads(best)
    if verbose:
        print("cpu threads: " + ", ".join(f"{t}: {v * 1e3:.0f} ms" for t, v in timings.items()) + f"  -> using {best}")
    return best


# =========================================================================== training
def _to(obj: Any, device) -> Any:
    """Move tensors (possibly nested in lists / tuples / dicts) to `device`."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _to(v, device) for k, v in obj.items()}
    return obj


def _batch_len(x: Any) -> int:
    return x.shape[0] if isinstance(x, torch.Tensor) else len(x)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of correct argmax predictions.

    Works for classification (logits (B, C), targets (B,)), dense prediction
    (logits (B, C, H, W), targets (B, H, W)) and soft / one-hot targets (B, C) such as
    Mixup labels (the target class is then the argmax of the target).
    """
    if targets.dim() == logits.dim() and targets.shape == logits.shape and targets.is_floating_point():
        targets = targets.argmax(1)
    return (logits.argmax(1) == targets).float().mean().item()


def _default_metric(out: Any, y: Any) -> float:
    """Accuracy for class logits with integer targets ((B, C) / (B,) or (B, C, H, W) / (B, H, W)), else NaN.

    Float targets (regression, boxes, Mixup soft labels) get NaN: pass metric_fn=accuracy explicitly
    for soft labels.
    """
    if (isinstance(out, torch.Tensor) and isinstance(y, torch.Tensor) and out.dim() >= 2 and out.shape[1] > 1
            and not y.is_floating_point() and y.dim() == out.dim() - 1 and out.shape[0] == y.shape[0]):
        return accuracy(out, y)
    return float("nan")


def _forward_loss(model, x, y, loss_fn):
    """Return (outputs, loss). With loss_fn=None the model computes its own losses (torchvision detection)."""
    if loss_fn is None:
        out = model(x, y)
        loss = sum(out.values()) if isinstance(out, dict) else out
        return out, loss
    out = model(x)
    return out, loss_fn(out, y)


_PER_BATCH_SCHEDULERS = ("OneCycleLR", "CyclicLR", "LambdaLR", "MultiplicativeLR")


def _scheduler_mode(scheduler) -> str | None:
    if scheduler is None:
        return None
    name = type(scheduler).__name__
    if name in _PER_BATCH_SCHEDULERS:
        return "batch"
    if name == "ReduceLROnPlateau":
        return "plateau"
    return "epoch"


@torch.no_grad()
def evaluate(model: nn.Module, loader: Iterable, loss_fn: Callable | None = None,
             metric_fn: Callable | None = None, device=None, max_batches: int | None = None) -> tuple[float, float]:
    """Average loss and metric over a loader (weighted by batch size).

    metric_fn(outputs, targets) -> float. If None, accuracy is used when the outputs look like
    class logits ((B, C) or (B, C, H, W) with integer targets); otherwise the metric is NaN.
    With loss_fn=None (torchvision detection models) the loss is computed in train mode
    under no_grad (those models only return losses in train mode; their BatchNorm is frozen)
    and metric_fn, if given, receives the eval-mode predictions.
    """
    device = device or next((p.device for p in model.parameters()), torch.device("cpu"))
    was_training = model.training
    tot_loss = tot_metric = 0.0
    n = n_metric = 0
    try:
        for b, batch in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            x, y = _to(batch[0], device), _to(batch[1], device)
            bs = _batch_len(x)
            if loss_fn is None:
                model.train()
                _, loss = _forward_loss(model, x, y, None)
                model.eval()
                m = metric_fn(model(x), y) if metric_fn is not None else float("nan")
            else:
                model.eval()
                out, loss = _forward_loss(model, x, y, loss_fn)
                m = metric_fn(out, y) if metric_fn is not None else _default_metric(out, y)
            tot_loss += float(loss) * bs
            n += bs
            if not (isinstance(m, float) and math.isnan(m)):
                tot_metric += float(m) * bs
                n_metric += bs
    finally:
        model.train(was_training)
    return tot_loss / max(n, 1), (tot_metric / n_metric if n_metric else float("nan"))


def fit(model: nn.Module, train_loader: Iterable, val_loader: Iterable | None, loss_fn: Callable | None,
        optimizer: torch.optim.Optimizer, epochs: int, scheduler=None, device=None,
        metric_fn: Callable | None = None, max_minutes: float | None = None, amp: bool = False,
        log_every: int | None = 50, checkpoint: str | Path | None = None, *,
        scheduler_step: str | None = None, grad_clip: float | None = None, mode: str = "max",
        restore_best: bool = False, max_val_batches: int | None = None, progress: bool = False,
        verbose: bool = True) -> dict:
    """Train `model` and return a history dict.

    history keys (one entry per epoch): ``train_loss``, ``val_loss``, ``val_metric``, ``lr``
    (learning rate of the first param group at the end of the epoch), ``time`` (seconds,
    cumulative). Also ``stopped_early`` (bool) and ``best_epoch``.

    Batches are ``(x, y)``; x and y may be tensors or (nested) lists / dicts of tensors and are
    moved to `device`. ``loss_fn(outputs, y)`` gives the loss; with ``loss_fn=None`` the model
    is called as ``model(x, y)`` and must return a loss tensor or a dict of losses that are
    summed (torchvision detection models).

    metric_fn(outputs, y) -> float, higher is better (``mode="min"`` for lower is better).
    Default: accuracy for class logits, else NaN.

    Scheduler rule: OneCycleLR, CyclicLR, LambdaLR and MultiplicativeLR are stepped after every
    batch (their schedule is defined in optimiser steps); ReduceLROnPlateau is stepped once per
    epoch with the validation loss; every other scheduler (StepLR, CosineAnnealingLR, ...) once
    per epoch. Override with ``scheduler_step="batch" | "epoch"``.

    max_minutes: when the wall-clock budget is exceeded, the current epoch stops after the
    running batch, the model is validated on what it has learned so far, and training ends
    (``history["stopped_early"] = True``).

    checkpoint: path; the weights are saved every time the validation metric improves (or
    the validation loss when there is no metric), and the final history is written into
    the same file. ``load_checkpoint`` returns it as ``extras["history"]``.

    amp: mixed precision (float16 + GradScaler on CUDA, bfloat16 autocast on CPU).
    log_every: print the running training loss every N batches (0 / None = only epoch lines).
    """
    device = device or next((p.device for p in model.parameters()), torch.device("cpu"))
    model.to(device)
    sched_mode = scheduler_step or _scheduler_mode(scheduler)
    if sched_mode == "epoch" and type(scheduler).__name__ == "ReduceLROnPlateau":
        sched_mode = "plateau"
    use_cuda_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_cuda_amp else None
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    hist: dict = {"train_loss": [], "val_loss": [], "val_metric": [], "lr": [], "time": [],
                  "stopped_early": False, "best_epoch": None}
    best = None
    t_start = time.time()
    budget = max_minutes * 60 if max_minutes is not None else None
    stop = False

    try:
        from tqdm.auto import tqdm as _tqdm  # optional
    except ImportError:  # pragma: no cover
        _tqdm = None

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, run_n, seen = 0.0, 0, 0
        it = train_loader
        if progress and _tqdm is not None:
            it = _tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for b, batch in enumerate(it, 1):
            x, y = _to(batch[0], device), _to(batch[1], device)
            bs = _batch_len(x)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                _, loss = _forward_loss(model, x, y, loss_fn)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            if sched_mode == "batch":
                scheduler.step()
            lv = float(loss.detach())
            if not math.isfinite(lv):
                print(f"[fit] loss is {lv} at epoch {epoch}, batch {b}: stopping (try a lower learning rate)")
                stop = True
                hist["stopped_early"] = True
                break
            run_loss += lv * bs
            run_n += bs
            seen += 1
            if verbose and log_every and b % log_every == 0:
                print(f"  epoch {epoch} batch {b:>5d}  train_loss {run_loss / run_n:.4f}")
            if budget is not None and time.time() - t_start > budget:
                print(f"[fit] time budget of {max_minutes} min reached after {seen} batches of epoch {epoch}; "
                      f"validating and stopping")
                stop = True
                hist["stopped_early"] = True
                break

        train_loss = run_loss / max(run_n, 1)
        if val_loader is not None:
            val_loss, val_metric = evaluate(model, val_loader, loss_fn, metric_fn, device, max_val_batches)
        else:
            val_loss, val_metric = float("nan"), float("nan")
        if sched_mode == "epoch":
            scheduler.step()
        elif sched_mode == "plateau":
            scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_start
        for k, v in zip(("train_loss", "val_loss", "val_metric", "lr", "time"),
                        (train_loss, val_loss, val_metric, lr, elapsed)):
            hist[k].append(float(v))
        if verbose:
            print(f"epoch {epoch}/{epochs}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
                  f"val_metric {val_metric:.4f}  lr {lr:.2e}  ({elapsed:.0f} s)")

        # best-model bookkeeping
        if math.isfinite(val_metric):
            score = val_metric if mode == "max" else -val_metric
        elif math.isfinite(val_loss):
            score = -val_loss
        else:
            score = -train_loss
        if best is None or score > best[0]:
            best = (score, copy.deepcopy(model.state_dict()) if restore_best else None)
            hist["best_epoch"] = epoch
            if checkpoint is not None:
                save_checkpoint(model, checkpoint, epoch=epoch, val_loss=val_loss, val_metric=val_metric,
                                history=_plain(hist))
        if stop:
            break

    if checkpoint is not None and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state["history"] = _plain(hist)
        torch.save(state, checkpoint)
    if restore_best and best is not None and best[1] is not None:
        model.load_state_dict(best[1])
    return hist


def _plain(obj: Any) -> Any:
    """Convert numbers / arrays to plain Python types so checkpoints load with weights_only=True."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor) and obj.dim() == 0:
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_checkpoint(model: nn.Module, path: str | Path, **extra) -> Path:
    """Save {'model': state_dict, **extra} to `path` (directories are created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), **_plain(extra)}, path)
    return path


def load_checkpoint(model: nn.Module, path: str | Path, map_location: str = "cpu", strict: bool = True) -> dict:
    """Load weights saved by `save_checkpoint` (or a bare state_dict) into `model`.

    Returns the extra entries (e.g. ``epoch``, ``val_metric``, ``history``) as a dict.
    """
    state = torch.load(Path(path), map_location=map_location, weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        model.load_state_dict(state["model"], strict=strict)
        return {k: v for k, v in state.items() if k != "model"}
    model.load_state_dict(state, strict=strict)
    return {}


def lr_range_test(model: nn.Module, loader: Iterable, loss_fn: Callable, optimizer_cls,
                  lr_min: float = 1e-6, lr_max: float = 1.0, steps: int = 100, device=None,
                  smooth: float = 0.05, diverge: float = 4.0, **opt_kwargs) -> tuple[list, list]:
    """Leslie Smith's LR range test.

    Trains for up to `steps` batches while the learning rate grows exponentially from
    `lr_min` to `lr_max`, recording the exponentially smoothed loss. Stops early when the
    loss exceeds `diverge` x the best loss seen. The model's weights are restored afterwards.
    Returns (lrs, smoothed_losses). Pick an LR about 10x below the minimum of the curve.
    """
    device = device or next((p.device for p in model.parameters()), torch.device("cpu"))
    saved = copy.deepcopy(model.state_dict())
    opt = optimizer_cls(model.parameters(), lr=lr_min, **opt_kwargs)
    gamma = (lr_max / lr_min) ** (1 / max(steps - 1, 1))
    lrs, losses = [], []
    avg, best = None, float("inf")
    model.train()
    step = 0
    try:
        while step < steps:
            for batch in loader:
                if step >= steps:
                    break
                lr = lr_min * gamma ** step
                for g in opt.param_groups:
                    g["lr"] = lr
                x, y = _to(batch[0], device), _to(batch[1], device)
                _, loss = _forward_loss(model, x, y, loss_fn)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                lv = float(loss.detach())
                avg = lv if avg is None else smooth * lv + (1 - smooth) * avg
                lrs.append(lr)
                losses.append(avg)
                best = min(best, avg)
                step += 1
                if not math.isfinite(avg) or avg > diverge * best:
                    step = steps
                    break
    finally:
        model.load_state_dict(saved)
    return lrs, losses


# =========================================================================== visualisation
def denormalize(tensor: torch.Tensor, mean: Sequence[float] | None = IMAGENET_MEAN,
                std: Sequence[float] | None = IMAGENET_STD) -> np.ndarray:
    """Undo `Normalize(mean, std)`: (C, H, W) -> (H, W, C) numpy in [0, 1] ((B, C, H, W) -> (B, H, W, C)).

    Pass mean=None to skip the un-normalisation (image already in [0, 1]). If the channel
    count does not match `mean` (e.g. a grayscale image with ImageNet stats), each image is
    min-max scaled instead.
    """
    t = tensor.detach().float().cpu()
    batched = t.dim() == 4
    if not batched:
        t = t.unsqueeze(0)
    C = t.shape[1]
    if mean is not None and std is not None and len(mean) == C:
        t = t * torch.tensor(std).view(1, C, 1, 1) + torch.tensor(mean).view(1, C, 1, 1)
    elif mean is not None:
        lo = t.amin(dim=(1, 2, 3), keepdim=True)
        hi = t.amax(dim=(1, 2, 3), keepdim=True)
        t = (t - lo) / (hi - lo).clamp_min(1e-8)
    arr = t.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    if C == 1:
        arr = arr[..., 0]
    return arr if batched else arr[0]


def _auto_denorm(img: torch.Tensor, mean, std) -> np.ndarray:
    if mean == "auto":
        if float(img.min()) >= 0 and float(img.max()) <= 1:
            return denormalize(img, None, None)
        if img.shape[0] == 1:
            return denormalize(img, FASHION_MNIST_MEAN, FASHION_MNIST_STD)
        return denormalize(img, IMAGENET_MEAN, IMAGENET_STD)
    return denormalize(img, mean, std)


def show_batch(images, labels=None, classes: Sequence[str] | None = None, n: int = 16, cols: int | None = None,
               mean="auto", std=IMAGENET_STD, size: float = 2.2, preds=None, title: str | None = None):
    """Show up to `n` images of a batch in a grid. Returns the figure.

    images: tensor (B, C, H, W) or list of (C, H, W) tensors. Normalisation is detected
    automatically (values in [0, 1] are shown as-is; 3-channel images are assumed ImageNet-
    normalised; 1-channel FashionMNIST-normalised); pass mean / std explicitly to override.
    labels: per image an int class (title), a (H, W) mask (overlaid), or a target dict with
    'boxes' (drawn) and optionally 'labels' and 'masks' (overlaid). preds: optional predicted
    class ids; titles turn red when they differ from the label.
    """
    import matplotlib.pyplot as plt
    from . import viz

    k = min(n, len(images))
    cols = cols or min(k, 8)
    rows = int(math.ceil(k / cols))
    has_titles = labels is not None and not isinstance(labels[0], (dict, torch.Tensor)) or (
        isinstance(labels, torch.Tensor) and labels.dim() == 1)
    fig_h = size * rows + (0.25 * rows if has_titles else 0) + (0.4 if title else 0)
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, fig_h), squeeze=False, layout="constrained")
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= k:
            continue
        img = _auto_denorm(images[i], mean, std)
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        if labels is None:
            continue
        lab = labels[i]
        if isinstance(lab, dict):
            if "masks" in lab and len(lab["masks"]):
                inst = torch.zeros(lab["masks"].shape[-2:], dtype=torch.long)
                for j, m in enumerate(lab["masks"]):
                    inst[m.cpu().bool()] = j + 1
                viz.overlay_mask(ax, inst.numpy(), alpha=0.45)
            if "boxes" in lab:
                lb = lab.get("labels")
                names = None
                if lb is not None:
                    names = [classes[int(c)] if classes is not None else int(c) for c in lb]
                viz.draw_boxes(ax, lab["boxes"].cpu().numpy(), labels=names, color_by_label=False,
                               edgecolor="#F58518", lw=2)
        elif isinstance(lab, torch.Tensor) and lab.dim() == 2:
            m = lab.cpu().numpy().astype(int)
            viz.overlay_mask(ax, m, alpha=0.45)
        else:
            c = int(lab)
            txt = classes[c] if classes is not None else str(c)
            color = "black"
            if preds is not None:
                p = int(preds[i])
                if p != c:
                    txt = f"{txt}\n(pred: {classes[p] if classes is not None else p})"
                    color = "#E45756"
            ax.set_title(txt, fontsize=8, color=color)
    if title:
        fig.suptitle(title)
    return fig


# =========================================================================== datasets
def get_fashion_mnist(root: str | Path | None = None, normalize: bool = True, as_tensors: bool = False,
                      train_tf=None, test_tf=None, download: bool = True):
    """FashionMNIST (60k train / 10k test, 1x28x28, 10 classes) -> (train_ds, test_ds, classes).

    Default items are float tensors (1, 28, 28), normalised with mean 0.286 / std 0.353
    (normalize=False keeps [0, 1]). as_tensors=True returns in-memory TensorDatasets
    (much faster on CPU, no per-item transform).
    """
    from torchvision import datasets, transforms as T

    root = Path(root) if root else data_root()
    classes = list(datasets.FashionMNIST.classes)
    if as_tensors:
        out = []
        for train in (True, False):
            ds = datasets.FashionMNIST(root, train=train, download=download)
            x = ds.data.float().div(255).unsqueeze(1)
            if normalize:
                x = (x - FASHION_MNIST_MEAN[0]) / FASHION_MNIST_STD[0]
            out.append(TensorDataset(x, ds.targets.clone()))
        return out[0], out[1], classes
    base = [T.ToTensor()] + ([T.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD)] if normalize else [])
    train = datasets.FashionMNIST(root, train=True, download=download, transform=train_tf or T.Compose(base))
    test = datasets.FashionMNIST(root, train=False, download=download, transform=test_tf or T.Compose(base))
    return train, test, classes


def _pet_root() -> Path:
    return data_root() / "oxford-iiit-pet"


def _balanced_subset(labels: Sequence[int], n: int, seed: int = 0) -> list[int]:
    """Deterministic class-balanced subset: shuffle each class with `seed`, then round-robin."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    per_class = [list(rng.permutation(np.flatnonzero(labels == c))) for c in np.unique(labels)]
    out: list[int] = []
    while len(out) < min(n, len(labels)):
        for idx in per_class:
            if idx and len(out) < n:
                out.append(int(idx.pop(0)))
    return sorted(out)


def _default_pet_tf(size: int):
    from torchvision.transforms import v2

    return v2.Compose([v2.ToImage(), v2.Resize((size, size), antialias=True),
                       v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def _parse_voc(path: Path) -> tuple[list[list[float]], list[str]]:
    root = ET.parse(path).getroot()
    boxes, names = [], []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        boxes.append([float(bb.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")])
        names.append(obj.findtext("name", "pet").strip())
    return boxes, names


class _PetDataset(Dataset):
    """Index view over torchvision's OxfordIIITPet with a joint (image, target) transform."""

    def __init__(self, base, indices: Sequence[int], target: str, tf, det_labels: str = "head"):
        self.base, self.indices, self.target, self.tf, self.det_labels = base, list(indices), target, tf, det_labels
        self.classes = base.classes

    def __len__(self) -> int:
        return len(self.indices)

    def label(self, i: int) -> int:
        """Breed index of item i without loading the image."""
        return int(self.base._labels[self.indices[i]])

    def image_path(self, i: int) -> Path:
        return Path(self.base._images[self.indices[i]])

    def __getitem__(self, i: int):
        from PIL import Image
        from torchvision import tv_tensors

        j = self.indices[i]
        img = Image.open(self.base._images[j]).convert("RGB")
        if self.target == "category":
            return self.tf(img), int(self.base._labels[j])
        if self.target == "segmentation":
            tri = np.asarray(Image.open(self.base._segs[j]))
            lut = np.array([0, 1, 0, 2], dtype=np.int64)          # 1 pet -> 1, 2 background -> 0, 3 border -> 2
            mask = tv_tensors.Mask(torch.from_numpy(lut[np.clip(tri, 0, 3)]))
            img, mask = self.tf(img, mask)
            return img, mask.as_subclass(torch.Tensor).long()
        # detection
        stem = Path(self.base._images[j]).stem
        boxes, names = _parse_voc(_pet_root() / "annotations" / "xmls" / f"{stem}.xml")
        if self.det_labels == "species":
            labels = [1 if n == "cat" else 2 for n in names]
        elif self.det_labels == "breed":
            labels = [int(self.base._labels[j]) + 1] * len(boxes)
        else:
            labels = [1] * len(boxes)
        bb = tv_tensors.BoundingBoxes(torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), format="XYXY",
                                      canvas_size=(img.height, img.width))
        img, bb = self.tf(img, bb)
        return img, {"boxes": bb.as_subclass(torch.Tensor).float(), "labels": torch.tensor(labels, dtype=torch.int64)}


def get_pet_datasets(size: int = 224, train_tf=None, eval_tf=None, subset: int | tuple[int, int] | None = None,
                     target: str = "category", root: str | Path | None = None, det_labels: str = "head",
                     seed: int = 0):
    """Oxford-IIIT Pet (37 breeds) -> (train_ds, val_ds, classes).

    target:
      "category"     item = (image, breed index 0..36); classes = 37 breed names.
                     train = official 'trainval' split (3 680), val = official 'test' split (3 669).
      "segmentation" item = (image, long mask (H, W)) with 0 = background, 1 = pet, 2 = border
                     (remapped from the trimap 1 = pet, 2 = background, 3 = border); classes = PET_SEG_CLASSES.
                     Same splits as "category".
      "detection"    item = (image, {"boxes": float (N, 4) xyxy pixels, "labels": int64 (N,)}) with the
                     head boxes from annotations/xmls (PASCAL VOC). Only 'trainval' images have an xml,
                     so they are split 80 / 20 (class-stratified, deterministic) into train / val.
                     det_labels="head" -> every label is 1, classes = ["background", "head"];
                     "species" -> 1 cat / 2 dog; "breed" -> breed index + 1.
    size: default transforms resize to (size, size) and normalise with ImageNet stats (no
          augmentation). Custom train_tf / eval_tf for "segmentation" / "detection" must be
          torchvision.transforms.v2 transforms: they are called as tf(image, target).
    subset: int N -> deterministic class-balanced N training images and N // 4 validation images
            (2000 -> 2000 / 500); or a tuple (n_train, n_val).
    """
    from torchvision.datasets import OxfordIIITPet

    if target not in ("category", "segmentation", "detection"):
        raise ValueError("target must be 'category', 'segmentation' or 'detection'")
    root = Path(root) if root else data_root()
    if not (root / "oxford-iiit-pet" / "images").is_dir():
        raise FileNotFoundError(f"Oxford-IIIT Pet not found under {root / 'oxford-iiit-pet'}; "
                                f"run `python tools/download_data.py` first")
    tt = ["category", "segmentation"]
    trainval = OxfordIIITPet(str(root), split="trainval", target_types=tt, download=False)
    test = OxfordIIITPet(str(root), split="test", target_types=tt, download=False)
    train_tf = train_tf or _default_pet_tf(size)
    eval_tf = eval_tf or _default_pet_tf(size)

    if target == "detection":
        xml_dir = root / "oxford-iiit-pet" / "annotations" / "xmls"
        have = [i for i, p in enumerate(trainval._images) if (xml_dir / f"{Path(p).stem}.xml").exists()]
        labs = [int(trainval._labels[i]) for i in have]
        rng = np.random.default_rng(seed)
        tr_idx, va_idx = [], []
        for c in sorted(set(labs)):
            idx = [have[k] for k, l in enumerate(labs) if l == c]
            idx = [idx[k] for k in rng.permutation(len(idx))]
            cut = int(round(0.8 * len(idx)))
            tr_idx += idx[:cut]
            va_idx += idx[cut:]
        tr_base = va_base = trainval
        tr_idx, va_idx = sorted(tr_idx), sorted(va_idx)
        classes = {"head": ["background", "head"], "species": ["background", "cat", "dog"],
                   "breed": ["background"] + list(trainval.classes)}[det_labels]
    else:
        tr_base, va_base = trainval, test
        tr_idx, va_idx = list(range(len(trainval))), list(range(len(test)))
        classes = list(trainval.classes) if target == "category" else list(PET_SEG_CLASSES)

    if subset is not None:
        n_tr, n_va = (subset, max(subset // 4, 1)) if isinstance(subset, int) else subset
        tr_idx = [tr_idx[k] for k in _balanced_subset([tr_base._labels[i] for i in tr_idx], n_tr, seed)]
        va_idx = [va_idx[k] for k in _balanced_subset([va_base._labels[i] for i in va_idx], n_va, seed)]

    return (_PetDataset(tr_base, tr_idx, target, train_tf, det_labels),
            _PetDataset(va_base, va_idx, target, eval_tf, det_labels), classes)


class PennFudanDataset(Dataset):
    """Penn-Fudan pedestrians (170 images): (image float [0, 1] (3, H, W), target dict).

    target: boxes float (N, 4) xyxy, labels int64 (N,) all 1 (= person; 0 is background),
    masks uint8 (N, H, W), image_id, area, iscrowd (torchvision detection format).
    transforms: optional torchvision.transforms.v2 transform called as tf(image, target).
    """

    classes = ["background", "person"]

    def __init__(self, root: str | Path, transforms=None):
        self.root = Path(root)
        self.transforms = transforms
        self.imgs = sorted((self.root / "PNGImages").glob("*.png"))
        self.masks = [self.root / "PedMasks" / (p.stem + "_mask.png") for p in self.imgs]

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int):
        from PIL import Image
        from torchvision import tv_tensors
        from torchvision.ops import masks_to_boxes
        from torchvision.transforms.v2 import functional as TF

        img = TF.to_dtype(TF.to_image(Image.open(self.imgs[i]).convert("RGB")), torch.float32, scale=True)
        inst = torch.from_numpy(np.asarray(Image.open(self.masks[i]), dtype=np.int64))
        ids = torch.unique(inst)
        ids = ids[ids != 0]
        masks = (inst[None] == ids[:, None, None]).to(torch.uint8)
        boxes = masks_to_boxes(masks).float()
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=img.shape[-2:]),
            "labels": torch.ones(len(ids), dtype=torch.int64),
            "masks": tv_tensors.Mask(masks),
            "image_id": i,
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(ids), dtype=torch.int64),
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        target["boxes"] = target["boxes"].as_subclass(torch.Tensor).float()
        target["masks"] = target["masks"].as_subclass(torch.Tensor).to(torch.uint8)
        return img.as_subclass(torch.Tensor), target


PENN_FUDAN_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"


def get_penn_fudan(root: str | Path | None = None, download: bool = True, transforms=None) -> PennFudanDataset:
    """Penn-Fudan pedestrians from `<data>/penn-fudan/PennFudanPed`, downloaded if missing."""
    base = Path(root) if root else data_root() / "penn-fudan"
    folder = base / "PennFudanPed"
    if not (folder / "PNGImages").is_dir():
        if not download:
            raise FileNotFoundError(f"{folder} not found; run `python tools/download_data.py`")
        from torchvision.datasets.utils import download_and_extract_archive

        download_and_extract_archive(PENN_FUDAN_URL, str(base))
        (base / ".complete").touch()
    return PennFudanDataset(folder, transforms)


class ShapesDataset(Dataset):
    """Synthetic shapes (cvcourse.shapes) as a PyTorch Dataset; generated in memory, no download.

    Items are (image float (3, size, size) in [0, 1], target) with target depending on `task`:
      "detection"    {"boxes": float (N, 4) xyxy, "labels": int64 (N,)}; labels are 0 circle,
                     1 square, 2 triangle (+ label_offset: use 1 for torchvision detection models)
      "segmentation" long (size, size): 0 background, 1 circle, 2 square, 3 triangle
      "grid"         float (5 + 3, grid, grid): shapes.grid_targets, channels-first
      "classify"     int64 class of the largest object (handy for tiny classifiers)
    Extra keyword arguments go to shapes.make_dataset (max_objects, min_r, max_r, noise).
    """

    classes = list(_shapes.CLASS_NAMES)

    def __init__(self, n: int = 512, size: int = 128, seed: int = 0, task: str = "detection", grid: int = 8,
                 label_offset: int = 0, **kw):
        if task not in ("detection", "segmentation", "grid", "classify"):
            raise ValueError("task must be 'detection', 'segmentation', 'grid' or 'classify'")
        self.size, self.task, self.grid, self.label_offset = size, task, grid, label_offset
        X, B, L, M = _shapes.make_dataset(n=n, size=size, seed=seed, **kw)
        self.images = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
        self.boxes, self.labels, self.masks = B, L, torch.from_numpy(M)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img = self.images[i]
        if self.task == "segmentation":
            return img, self.masks[i]
        if self.task == "grid":
            T = _shapes.grid_targets(self.boxes[i], self.labels[i], self.grid, self.size, len(self.classes))
            return img, torch.from_numpy(T).permute(2, 0, 1).contiguous()
        if self.task == "classify":
            b = self.boxes[i]
            k = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
            return img, torch.tensor(int(self.labels[i][k]), dtype=torch.int64)
        return img, {"boxes": torch.as_tensor(self.boxes[i], dtype=torch.float32).reshape(-1, 4),
                     "labels": torch.as_tensor(self.labels[i], dtype=torch.int64) + self.label_offset}


def detection_collate(batch):
    """Keep variable-size targets as lists: [(img, tgt), ...] -> (list of imgs, list of tgts)."""
    images, targets = zip(*batch)
    return list(images), list(targets)


# =========================================================================== Grad-CAM
def gradcam(model: nn.Module, target_layer: nn.Module, image_tensor: torch.Tensor,
            class_idx: int | None = None) -> np.ndarray:
    """Grad-CAM heat map (H, W) in [0, 1] at the input resolution.

    A forward hook stores the activations A of `target_layer` and a hook on that output
    tensor stores dScore/dA during backward; channel weights are the spatially averaged
    gradients, the map is ReLU(sum_c w_c A_c), bilinearly upsampled and min-max scaled.
    class_idx=None explains the predicted class.
    """
    x = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor[:1]
    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    x = x.to(device)
    store: dict = {}

    def fwd_hook(mod, inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = target_layer.register_forward_hook(fwd_hook)
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            logits = model(x)
            c = int(logits.argmax(1)) if class_idx is None else int(class_idx)
            model.zero_grad(set_to_none=True)
            logits[0, c].backward()
    finally:
        h.remove()
        model.train(was_training)
    act, grad = store["act"].detach(), store["grad"].detach()
    w = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max().clamp_min(1e-8)
    return cam.cpu().numpy()


# =========================================================================== smoke test
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    t0 = time.time()
    seed_everything(0)
    dev = get_device()
    train_ds = ShapesDataset(n=256, size=64, seed=0, task="segmentation", max_r=14, min_r=6)
    val_ds = ShapesDataset(n=64, size=64, seed=1, task="segmentation", max_r=14, min_r=6)
    print("ShapesDataset:", len(train_ds), "items; image", tuple(train_ds[0][0].shape), "mask", tuple(train_ds[0][1].shape))
    net = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(16, 4, 1))
    print(summarize(net, (3, 64, 64)))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-2, total_steps=len(train_ds) // 32)
    ck = CHECKPOINTS / "_smoke_test.pt"
    hist = fit(net, DataLoader(train_ds, 32, shuffle=True), DataLoader(val_ds, 64), nn.CrossEntropyLoss(),
               opt, epochs=1, scheduler=sched, device=dev, log_every=4, checkpoint=ck)
    extras = load_checkpoint(net, ck)
    print("history:", {k: v for k, v in hist.items()})
    print("checkpoint extras:", sorted(extras))
    ck.unlink()
    cam = gradcam(nn.Sequential(net, nn.AdaptiveAvgPool2d(1), nn.Flatten()), net[2], val_ds[0][0])
    print("gradcam map:", cam.shape, f"range [{cam.min():.2f}, {cam.max():.2f}]")
    print(f"smoke test done in {time.time() - t0:.1f} s")
