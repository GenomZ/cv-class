"""The practical track's toolkit: short names, one import, works on any ``nn.Module``.

Import it in a notebook as::

    from cvcourse import practical as P

    P.setup()                                                   # seed, CPU threads, prints the device
    train_loader, val_loader, classes = P.get_mnist()           # data (downloaded on first use)
    P.show_samples(train_loader, classes)                       # a grid of images with their labels
    P.show_model(model, (1, 28, 28))                            # a diagram of the network
    P.summary(model, (1, 28, 28))                               # a table: one row per layer
    P.cost(model, (1, 28, 28))                                  # parameters, multiply-adds, timing
    history = P.train(model, train_loader, val_loader)          # training with live curves
    P.show_predictions(model, val_loader, classes)              # what the network answers
    P.remember("cnn v1", model, history); P.leaderboard()       # keep track of every version

Public functions (full reference with examples: ``docs/10-practical-track.md``)
-------------------------------------------------------------------------------
setup(seed=0, threads=None)
get_mnist(n_train=10000, n_val=2000, batch_size=64, seed=0, shuffle_pixels=False) -> (train_loader, val_loader, classes)
get_cifar10(n_train=10000, n_val=2000, batch_size=64, seed=0, augment=False, size=None, normalize="cifar")
ImageSet(x, y, classes, mean, std) ; loaders(train_set, val_set, batch_size=64, seed=0) -> (train_loader, val_loader)
show_samples(loader, classes=None, n=16) -> Figure
show_model(model, input_shape, depth=None, title=None) -> Figure
summary(model, input_shape, depth=None) -> list of row dicts
output_shape(model, input_shape) -> tuple
cost(model, input_shape, name=None, batch_size=32, epoch_images=10000) -> dict
compare(models, input_shape, histories=None) -> list of row dicts (also draws a bar chart)
train(model, train_loader, val_loader=None, epochs=3, lr=1e-3, optimizer="adam", max_minutes=None, live=True, seed=0, name=None) -> History
evaluate(model, loader) -> (loss, accuracy in %)
predict(model, images) -> predicted class indices
plot_history(history or {name: history}) -> Figure
show_predictions(model, loader, classes=None, n=16, only_wrong=False) -> Figure
confusion(model, loader, classes=None) -> numpy array (rows = true class, columns = predicted class)
show_filters(model, layer=None) -> Figure
show_feature_maps(model, image, layer=None, n=8) -> Figure
remember(name, model, history=None, input_shape=None) ; leaderboard(sort="accuracy") -> rows ; forget(name=None)

Every figure function draws its figure under the cell (also when the notebook runs headless with
nbconvert) and returns it; Jupyter does not show it a second time. Nothing needs a GPU. Errors
about shapes are raised as ``ModelShapeError`` with a message that names the layer and the fix.
"""
from __future__ import annotations

import copy
import io
import math
import operator
import random
import statistics
import time
from collections import OrderedDict
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath

from .torchkit import IMAGENET_MEAN, IMAGENET_STD, data_root

__all__ = [
    "setup", "get_mnist", "get_cifar10", "ImageSet", "loaders", "show_samples", "show_model", "summary",
    "output_shape", "cost", "compare", "train", "evaluate", "predict", "plot_history", "show_predictions",
    "confusion", "show_filters", "show_feature_maps", "remember", "leaderboard", "forget", "History",
    "ModelShapeError", "KIND_COLORS", "MNIST_MEAN", "MNIST_STD", "CIFAR10_MEAN", "CIFAR10_STD",
]

MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Colours of the layer kinds in show_model (validated categorical palette, fixed order) and the text inks.
KIND_COLORS = {
    "convolution": "#2a78d6", "pooling": "#eb6834", "linear": "#1baf7a", "activation": "#eda100",
    "normalisation": "#e87ba4", "dropout": "#008300", "flatten": "#6250d6", "merge": "#e34948",
    "other": "#9a9893", "input": "#e6e5e0", "output": "#e6e5e0",
}
_KIND_NAMES = {
    "convolution": "convolution", "pooling": "pooling", "linear": "linear (fully connected)",
    "activation": "activation", "normalisation": "normalisation", "dropout": "dropout",
    "flatten": "flatten", "merge": "add / concatenate", "other": "other",
}
_INK = "#0b0b0b"
_INK2 = "#52514e"
_RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
_DPI = 100
_STATE: dict = {"seed": 0}


# =============================================================================================== notebook plumbing
def _shell():
    """The IPython kernel shell when running inside Jupyter / nbconvert, else None."""
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    sh = get_ipython()
    if sh is None or not hasattr(sh, "kernel"):
        return None
    return sh


def _cell():
    sh = _shell()
    return None if sh is None else getattr(sh, "execution_count", None)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, facecolor="white")
    return buf.getvalue()


def _publish(fig) -> None:
    from IPython.display import Image, display
    display(Image(data=_png(fig)))


class _Figure(Figure):
    """A Matplotlib figure that the toolkit has already drawn under the cell.

    Jupyter shows the return value of the last line of a cell; without this, every figure
    would appear twice. In a later cell, ``fig`` alone shows it again as usual.
    """

    _p_cell = None

    def _ipython_display_(self):
        if self._p_cell is not None and self._p_cell == _cell():
            return
        _publish(self)


def _new_figure(w: float, h: float, constrained: bool = True) -> _Figure:
    return _Figure(figsize=(w, h), layout="constrained" if constrained else None)


def _show(fig: _Figure) -> _Figure:
    _STATE["last_figure"] = fig
    if _shell() is not None:
        _publish(fig)
        fig._p_cell = _cell()
    return fig


class _Quiet:
    """Mixin for results that printed themselves: not repeated by Jupyter in the same cell."""

    _p_cell = None

    def _ipython_display_(self):
        if self._p_cell is not None and self._p_cell == _cell():
            return
        from IPython.display import Pretty, display
        display(Pretty(repr(self)))

    def _mark(self):
        self._p_cell = _cell()
        return self


class _Rows(_Quiet, list):
    """A list of row dicts that remembers the table it was printed as."""

    text = ""

    def __repr__(self):
        return self.text or list.__repr__(self)


class _Report(_Quiet, dict):
    text = ""

    def __repr__(self):
        return self.text or dict.__repr__(self)


class History(_Quiet, dict):
    """What ``P.train`` returns: a dict of lists, one entry per epoch.

    Keys: ``name``, ``epoch``, ``train_loss``, ``train_acc``, ``val_loss``, ``val_acc`` (accuracies in %),
    ``seconds`` (per epoch), ``total_seconds``, ``best_val_acc``, ``best_epoch``, ``lr``, ``optimizer``,
    ``batch_size``, ``n_train``, ``params``, ``input_shape``, ``stopped_early``.
    """

    def __repr__(self):
        n = len(self.get("epoch", []))
        best = self.get("best_val_acc")
        if best is None:
            return f"History({self.get('name')!r}: {n} epochs)"
        return (f"History({self.get('name')!r}: {n} epochs, best validation accuracy {best:.1f}% "
                f"at epoch {self.get('best_epoch')}, {self.get('total_seconds', 0):.1f} s)")


# =============================================================================================== setup
def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _model_device(model: nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup(seed: int = 0, threads: int | None = None, verbose: bool = True) -> None:
    """Prepare a notebook: fix the random seed, set the CPU thread count once, print the device.

    seed: the starting point of every random number (weights, shuffling), so results repeat.
    threads: how many CPU cores PyTorch may use; None keeps PyTorch's default (your number of cores).
    """
    if threads is not None:
        torch.set_num_threads(int(threads))
    _STATE["seed"] = seed
    _seed(seed)
    dev = _device()
    if dev.type == "cuda":
        where = f"GPU ({torch.cuda.get_device_name(0)})"
    else:
        where = "CPU"
    if verbose:
        print(f"PyTorch {torch.__version__} | running on the {where} | {torch.get_num_threads()} CPU threads | "
              f"random seed {seed}")


# =============================================================================================== data
class ImageSet(Dataset):
    """Images and labels held in memory as tensors (fast on a CPU).

    x: float tensor (N, C, H, W), already normalised; y: int64 tensor (N,) of class indices (or masks).
    classes: class names; mean, std: the normalisation used (show_samples undoes it for display).
    augment: random 4-pixel shifts and horizontal flips on every access (for colour photos).
    size: resize every image to size x size when it is read (for networks that need bigger inputs).
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, classes: Sequence[str], mean: Sequence[float],
                 std: Sequence[float], name: str = "", augment: bool = False, size: int | None = None):
        self.x, self.y = x, y
        self.classes = list(classes)
        self.mean, self.std = tuple(mean), tuple(std)
        self.name, self.augment, self.size = name, augment, size

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i):
        img = self.x[i]
        if self.size is not None and img.shape[-1] != self.size:
            img = F.interpolate(img[None], size=(self.size, self.size), mode="bilinear", align_corners=False,
                                antialias=True)[0]
        if self.augment:
            if torch.rand(1).item() < 0.5:
                img = img.flip(-1)
            h, w = img.shape[-2:]
            pad = max(1, h // 8)
            padded = F.pad(img, (pad, pad, pad, pad))
            top = int(torch.randint(0, 2 * pad + 1, (1,)))
            left = int(torch.randint(0, 2 * pad + 1, (1,)))
            img = padded[:, top:top + h, left:left + w]
        return img, self.y[i]


def loaders(train_set: Dataset, val_set: Dataset | None = None, batch_size: int = 64, seed: int = 0):
    """(train_loader, val_loader): the training loader shuffles with its own seeded generator; no workers."""
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g, num_workers=0)
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=max(256, batch_size), shuffle=False, num_workers=0)
    return train_loader, val_loader


_RAW: dict = {}


def _raw(name: str):
    """Full train / test tensors of a torchvision dataset (uint8, N x C x H x W), loaded once per session."""
    if name in _RAW:
        return _RAW[name]
    from torchvision import datasets

    root = data_root()
    if name == "mnist":
        if not (root / "MNIST" / "raw" / "t10k-labels-idx1-ubyte").exists():
            print("downloading MNIST (about 12 MB) into data/ - only the first time")
        tr = datasets.MNIST(str(root), train=True, download=True)
        te = datasets.MNIST(str(root), train=False, download=True)
        out = (tr.data.unsqueeze(1), tr.targets.clone(), te.data.unsqueeze(1), te.targets.clone(),
               [str(i) for i in range(10)])
    elif name == "cifar10":
        if not (root / "cifar-10-batches-py" / "test_batch").exists():
            print("downloading CIFAR-10 (about 170 MB) into data/ - only the first time")
        tr = datasets.CIFAR10(str(root), train=True, download=True)
        te = datasets.CIFAR10(str(root), train=False, download=True)
        out = (torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(tr.targets),
               torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(te.targets),
               list(tr.classes))
    else:
        raise ValueError(name)
    _RAW[name] = out
    return out


def _pick(n_total: int, n: int | None, seed: int) -> torch.Tensor:
    if n is None or n >= n_total:
        return torch.arange(n_total)
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randperm(n_total, generator=g)[:n]


def _normalise(x_uint8: torch.Tensor, mean, std) -> torch.Tensor:
    x = x_uint8.float().div_(255)
    m = torch.tensor(mean).view(1, -1, 1, 1)
    s = torch.tensor(std).view(1, -1, 1, 1)
    return (x - m) / s


def get_mnist(n_train: int | None = 10000, n_val: int | None = 2000, batch_size: int = 64, seed: int = 0,
              shuffle_pixels: bool = False, verbose: bool = True):
    """MNIST hand-written digits -> (train_loader, val_loader, classes).

    Images are 1 x 28 x 28, normalised with MNIST's mean 0.1307 and std 0.3081; labels 0-9.
    n_train / n_val: how many images to use (a fixed random subset chosen by `seed`; None = all
    60 000 / 10 000). The validation images come from MNIST's separate test set, so the network
    never trains on them. shuffle_pixels=True moves every pixel to another (fixed) place in every
    image: the digits become unreadable to us, but every image is scrambled in the same way.
    """
    x_tr, y_tr, x_te, y_te, classes = _raw("mnist")
    i_tr, i_va = _pick(len(x_tr), n_train, seed), _pick(len(x_te), n_val, seed + 1)
    xt, xv = _normalise(x_tr[i_tr], MNIST_MEAN, MNIST_STD), _normalise(x_te[i_va], MNIST_MEAN, MNIST_STD)
    if shuffle_pixels:
        g = torch.Generator()
        g.manual_seed(seed + 12345)
        perm = torch.randperm(28 * 28, generator=g)
        xt = xt.reshape(len(xt), 1, -1)[:, :, perm].reshape(-1, 1, 28, 28)
        xv = xv.reshape(len(xv), 1, -1)[:, :, perm].reshape(-1, 1, 28, 28)
    name = "MNIST, pixels shuffled" if shuffle_pixels else "MNIST"
    train_set = ImageSet(xt, y_tr[i_tr], classes, MNIST_MEAN, MNIST_STD, name=name)
    val_set = ImageSet(xv, y_te[i_va], classes, MNIST_MEAN, MNIST_STD, name=name)
    tl, vl = loaders(train_set, val_set, batch_size, seed)
    if verbose:
        print(f"{name}: {len(train_set):,} training images, {len(val_set):,} validation images, "
              f"each 1 x 28 x 28, 10 classes; batches of {batch_size}")
    return tl, vl, classes


def get_cifar10(n_train: int | None = 10000, n_val: int | None = 2000, batch_size: int = 64, seed: int = 0,
                augment: bool = False, size: int | None = None, normalize: str = "cifar", verbose: bool = True):
    """CIFAR-10 colour photos (3 x 32 x 32, 10 classes) -> (train_loader, val_loader, classes).

    n_train / n_val: fixed random subsets (None = all 50 000 / 10 000; validation comes from the test set).
    augment: random shifts and flips of the training images (the validation images stay as they are).
    size: resize images to size x size when read (e.g. 64, or 224 for ImageNet-pretrained networks).
    normalize: "cifar" (CIFAR-10's own mean / std) or "imagenet" (for ImageNet-pretrained networks).
    """
    x_tr, y_tr, x_te, y_te, classes = _raw("cifar10")
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if normalize == "imagenet" else (CIFAR10_MEAN, CIFAR10_STD)
    i_tr, i_va = _pick(len(x_tr), n_train, seed), _pick(len(x_te), n_val, seed + 1)
    train_set = ImageSet(_normalise(x_tr[i_tr], mean, std), y_tr[i_tr], classes, mean, std, "CIFAR-10",
                         augment=augment, size=size)
    val_set = ImageSet(_normalise(x_te[i_va], mean, std), y_te[i_va], classes, mean, std, "CIFAR-10", size=size)
    tl, vl = loaders(train_set, val_set, batch_size, seed)
    if verbose:
        s = size or 32
        print(f"CIFAR-10: {len(train_set):,} training images, {len(val_set):,} validation images, each 3 x {s} x {s}, "
              f"10 classes; batches of {batch_size}" + ("; training images randomly shifted and flipped" if augment else ""))
    return tl, vl, classes


def _dataset_of(obj):
    return obj.dataset if isinstance(obj, DataLoader) else obj


def _display_image(img: torch.Tensor, ds) -> np.ndarray:
    """(C, H, W) normalised tensor -> numpy image in [0, 1] for imshow ((H, W) for grey)."""
    t = img.detach().float().cpu()
    mean, std = getattr(ds, "mean", None), getattr(ds, "std", None)
    if mean is not None and len(mean) == t.shape[0]:
        t = t * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)
    else:
        t = (t - t.min()) / (t.max() - t.min()).clamp_min(1e-8)
    t = t.clamp(0, 1)
    if t.shape[0] == 1:
        return t[0].numpy()
    return t.permute(1, 2, 0).numpy()


def _grid(n: int, cols: int):
    cols = max(1, min(cols, n))
    return cols, int(math.ceil(n / cols))


def show_samples(loader, classes: Sequence[str] | None = None, n: int = 16, cols: int = 8, title: str | None = None):
    """A grid of the first `n` images of a loader (or dataset) with their class names. Returns the figure."""
    ds = _dataset_of(loader)
    classes = classes if classes is not None else getattr(ds, "classes", None)
    n = min(n, len(ds))
    cols, rows = _grid(n, cols)
    fig = _new_figure(1.45 * cols, 1.55 * rows + 0.35)
    axes = fig.subplots(rows, cols, squeeze=False)
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k >= n:
            continue
        img, label = ds[k]
        arr = _display_image(img, ds)
        ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
        lab = int(label) if torch.is_tensor(label) and label.dim() == 0 or isinstance(label, int) else None
        if lab is not None:
            ax.set_title(classes[lab] if classes is not None else str(lab), fontsize=10, color=_INK)
    fig.suptitle(title or f"{n} images from the {getattr(ds, 'name', '') or 'data'} set and their labels",
                 fontsize=11, color=_INK)
    return _show(fig)


# =============================================================================================== shapes and errors
class ModelShapeError(ValueError):
    """The model cannot process the input: the message names the layer, what it got, what it expected, the fix."""


def _fmt_shape(shape, sep: str = " x ") -> str:
    shape = tuple(int(v) for v in shape)
    if len(shape) == 0:
        return "a single number"
    return sep.join(str(v) for v in shape)


def _norm_shape(input_shape) -> tuple:
    if isinstance(input_shape, torch.Size):
        input_shape = tuple(input_shape)
    if isinstance(input_shape, int):
        input_shape = (input_shape,)
    if not isinstance(input_shape, (tuple, list)) or not all(isinstance(v, (int, np.integer)) for v in input_shape):
        raise ModelShapeError(f"input_shape must be a tuple of whole numbers such as (1, 28, 28), not {input_shape!r}")
    shape = tuple(int(v) for v in input_shape)
    if len(shape) == 2:
        raise ModelShapeError(
            f"input_shape {shape} has 2 numbers, but one image needs 3: (channels, height, width). "
            f"For MNIST that is (1, 28, 28), for CIFAR-10 (3, 32, 32).")
    if len(shape) == 4:
        raise ModelShapeError(
            f"input_shape {shape} has 4 numbers: it looks like it includes the batch size. Give the shape of "
            f"ONE image instead, e.g. {shape[1:]}.")
    return shape


def _first_line(err: BaseException) -> str:
    text = str(err).strip().splitlines()
    return f"{type(err).__name__}: {text[0] if text else ''}"


_POOL_TYPES = (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.LPPool2d)


def _explain(name: str, m: nn.Module, args, err: BaseException, x: torch.Tensor) -> str:
    """Plain-language message for an exception raised inside module `m` (named `name`)."""
    inp = args[0] if args and isinstance(args[0], torch.Tensor) else None
    where = f"Layer '{name}' ({type(m).__name__})" if name else f"The model ({type(m).__name__})"
    head = f"The model cannot process one image of shape {_fmt_shape(x.shape[1:])}.\n"
    tail = f"\n(PyTorch's own message: {_first_line(err)})"
    if inp is None:
        return head + f"{where} raised an error." + tail
    got = _fmt_shape(inp.shape[1:])
    if isinstance(m, nn.Linear):
        if inp.dim() > 2:
            n = int(np.prod(inp.shape[1:]))
            msg = (f"{where} works on a flat list of numbers, but it received a {got} block per image. "
                   f"Put nn.Flatten() in front of it. It then receives {n} numbers, so it needs in_features={n}: "
                   f"nn.Linear({n}, {m.out_features}).")
        elif inp.shape[-1] != m.in_features:
            n = int(inp.shape[-1])
            msg = (f"{where} expects {m.in_features} numbers per image (in_features={m.in_features}), but the layers "
                   f"before it produce {n}.\nFix: make this layer nn.Linear({n}, {m.out_features}), or change the "
                   f"layers before it so that they produce {m.in_features} numbers. P.summary(...) of the layers "
                   f"before it shows what they produce.")
        else:
            msg = f"{where} could not process its input of {got} numbers per image."
        return head + msg + tail
    if isinstance(m, nn.modules.conv._ConvNd):
        nd = len(m.kernel_size)
        if inp.dim() != nd + 2:
            msg = (f"{where} expects a batch of images (batch, channels, height, width), but it received a tensor "
                   f"of shape {tuple(inp.shape)}.")
            if inp.dim() == 2:
                msg += " Is there an nn.Flatten() before this convolution? Convolutions come before Flatten."
        elif inp.shape[1] != m.in_channels:
            c = int(inp.shape[1])
            msg = (f"{where} expects images with {m.in_channels} channel(s) (in_channels={m.in_channels}), but it "
                   f"received {c} (its input is {got} per image).\nFix: in_channels={c}. The in_channels of a "
                   f"convolution must equal the out_channels of the convolution before it (or the number of "
                   f"colour channels of the images for the first one: 1 for grey, 3 for colour).")
        else:
            k = m.kernel_size[0]
            pad = m.padding if isinstance(m.padding, str) else m.padding[0]
            msg = (f"{where} has a {k} x {k} kernel, but its input is only {_fmt_shape(inp.shape[2:])} pixels"
                   + (f" (plus padding {pad})" if pad else "") + ": the image has become smaller than the filter. "
                   f"Every convolution without padding, every stride of 2 and every pooling layer shrinks the "
                   f"image.\nFix: padding={k // 2}, fewer pooling layers, stride=1, or a smaller kernel_size.")
        return head + msg + tail
    if isinstance(m, _POOL_TYPES):
        k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
        msg = (f"{where} has a {k} x {k} window, but its input is only {_fmt_shape(inp.shape[2:])} pixels "
               f"({got} per image): too small to pool.\nFix: remove a pooling layer, use stride=1 or padding in the "
               f"convolutions before it, or start from bigger images.")
        return head + msg + tail
    if isinstance(m, nn.modules.batchnorm._BatchNorm) and inp.dim() >= 2 and inp.shape[1] != m.num_features:
        c = int(inp.shape[1])
        msg = (f"{where} expects {m.num_features} channels (num_features={m.num_features}), but it received {c}.\n"
               f"Fix: {type(m).__name__}({c}): BatchNorm's number must equal the out_channels of the layer before it.")
        return head + msg + tail
    if any(True for _ in m.children()):
        msg = (f"The error happened in the forward() code of {where[0].lower() + where[1:]} itself (between its "
               f"layers), while its input was {got} per image.")
        return head + msg + tail
    return head + f"{where} could not process its input of shape {got} per image." + tail


def _forward_checked(model: nn.Module, x: torch.Tensor):
    """model(x), but an error raises ModelShapeError naming the layer that failed (innermost running module)."""
    names = {}
    for n, mod in model.named_modules():
        names.setdefault(id(mod), n)
    stack: list = []

    def pre(mod, args):
        stack.append((mod, args))

    def post(mod, args, out):
        if stack:
            stack.pop()

    handles = []
    for mod in model.modules():
        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))
    try:
        return model(x)
    except ModelShapeError:
        raise
    except Exception as err:  # noqa: BLE001
        if not stack:
            raise
        mod, args = stack[-1]
        raise ModelShapeError(_explain(names.get(id(mod), ""), mod, args, err, x)) from None
    finally:
        for h in handles:
            h.remove()


def _dry_run(model: nn.Module, input_shape, batch: int = 1):
    """Run one dummy batch in eval mode without gradients; returns the output. RNG state untouched."""
    shape = _norm_shape(input_shape)
    x = torch.zeros((batch,) + shape, device=_model_device(model))
    was = model.training
    model.eval()
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            return _forward_checked(model, x)
    finally:
        model.train(was)


def output_shape(model: nn.Module, input_shape) -> tuple:
    """The shape of what the model gives back for ONE image of `input_shape` (no batch axis).

    Runs a black dummy image through the model. If a layer does not fit, raises ModelShapeError
    with the layer's name and the fix.
    """
    out = _dry_run(model, input_shape)
    if isinstance(out, torch.Tensor):
        return tuple(out.shape[1:])
    if isinstance(out, (tuple, list)):
        return tuple(tuple(o.shape[1:]) for o in out if isinstance(o, torch.Tensor))
    raise ModelShapeError(f"the model returned a {type(out).__name__}, not a tensor")


# =============================================================================================== layer description
_CONV_T = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
_POOL_ALL = _POOL_TYPES + (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d, nn.AdaptiveMaxPool1d,
                           nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d)
_ACT_T = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU, nn.ELU, nn.SELU, nn.CELU, nn.GELU, nn.SiLU, nn.Mish, nn.Sigmoid,
          nn.Tanh, nn.Softmax, nn.LogSoftmax, nn.Softmax2d, nn.Hardswish, nn.Hardsigmoid, nn.Hardtanh, nn.Softplus,
          nn.Softsign, nn.Tanhshrink, nn.Threshold, nn.GLU)
_NORM_T = (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm, nn.modules.instancenorm._InstanceNorm,
           nn.LocalResponseNorm)
_DROP_T = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)
_THIN = ("activation", "normalisation", "dropout")


def _module_kind(m: nn.Module) -> str:
    if isinstance(m, nn.modules.conv._ConvNd):
        return "convolution"
    if isinstance(m, _POOL_ALL):
        return "pooling"
    if isinstance(m, (nn.Linear, nn.Bilinear)):
        return "linear"
    if isinstance(m, _ACT_T):
        return "activation"
    if isinstance(m, _NORM_T):
        return "normalisation"
    if isinstance(m, _DROP_T):
        return "dropout"
    if isinstance(m, nn.Flatten):
        return "flatten"
    return "other"


def _block_kind(m: nn.Module) -> str:
    """Colour of a box that stands for several layers: the most telling kind inside it."""
    kinds = [_module_kind(sub) for sub in m.modules() if not any(True for _ in sub.children())]
    for k in ("convolution", "linear", "pooling", "normalisation", "activation"):
        if k in kinds:
            return k
    return "other"


def _pair(v) -> tuple:
    if isinstance(v, (tuple, list)):
        return tuple(v)
    return (v, v)


def _k(v, sep: str) -> str:
    t = _pair(v)
    if all(a == t[0] for a in t):
        return f"{t[0]}{sep}{t[0]}" if len(t) >= 2 else str(t[0])
    return sep.join(str(a) for a in t)


def _single(v) -> str:
    t = _pair(v) if not isinstance(v, str) else (v,)
    if all(a == t[0] for a in t):
        return str(t[0])
    return "x".join(str(a) for a in t)


def _describe_module(m: nn.Module, out_shape: tuple | None):
    """(title, figure detail lines, one-line ASCII description) of a single layer."""
    X = "×"
    if isinstance(m, nn.modules.conv._ConvNd):
        name = "ConvTranspose" if isinstance(m, _CONV_T) else "Conv"
        k_f, k_a = _k(m.kernel_size, X), _k(m.kernel_size, "x")
        pad = m.padding if isinstance(m.padding, str) else _single(m.padding)
        lines = [f"{m.out_channels} maps", f"stride {_single(m.stride)}, pad {pad}"]
        text = f"{name} {k_a}, {m.out_channels} maps, stride {_single(m.stride)}, pad {pad}"
        if _single(m.dilation) != "1":
            lines.append(f"dilation {_single(m.dilation)}")
            text += f", dilation {_single(m.dilation)}"
        if m.groups != 1:
            lines.append(f"groups {m.groups}")
            text += f", groups {m.groups}"
        return f"{name} {k_f}", lines, text
    if isinstance(m, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d, nn.AdaptiveMaxPool1d,
                      nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d)):
        kind = "Max" if "Max" in type(m).__name__ else "Avg"
        size = _k(m.output_size, X)
        return f"{kind}Pool", [f"to {size}"], f"Adaptive{kind}Pool to {_k(m.output_size, 'x')}"
    if isinstance(m, _POOL_TYPES):
        kind = "MaxPool" if "Max" in type(m).__name__ else ("AvgPool" if "Avg" in type(m).__name__ else "LPPool")
        stride = m.stride if m.stride is not None else m.kernel_size
        pad = getattr(m, "padding", 0)
        lines = [f"stride {_single(stride)}" + (f", pad {_single(pad)}" if _single(pad) != "0" else "")]
        text = f"{kind} {_k(m.kernel_size, 'x')}, " + lines[0]
        return f"{kind} {_k(m.kernel_size, X)}", lines, text
    if isinstance(m, nn.Linear):
        return "Linear", [f"{m.in_features} → {m.out_features}"], f"Linear {m.in_features} -> {m.out_features}"
    if isinstance(m, nn.Flatten):
        n = _fmt_shape(out_shape) if out_shape else "?"
        return "Flatten", [], f"Flatten to {n} numbers"
    if isinstance(m, _DROP_T):
        return "Dropout", [f"p = {m.p}"], f"Dropout p={m.p}"
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        return "BatchNorm", [], f"BatchNorm, {m.num_features} channels"
    if isinstance(m, nn.LeakyReLU):
        return "LeakyReLU", [], f"LeakyReLU slope {m.negative_slope}"
    if isinstance(m, nn.Upsample):
        f = m.scale_factor if m.scale_factor is not None else m.size
        return "Upsample", [f"{X}{_single(f)}" if m.scale_factor is not None else f"to {_single(f)}"], \
            f"Upsample x{_single(f)} ({m.mode})"
    name = type(m).__name__
    return name, [], name


_FN_ACT = {"relu": "ReLU", "relu6": "ReLU6", "leaky_relu": "LeakyReLU", "elu": "ELU", "selu": "SELU", "celu": "CELU",
           "gelu": "GELU", "silu": "SiLU", "mish": "Mish", "sigmoid": "Sigmoid", "tanh": "Tanh", "softmax": "Softmax",
           "log_softmax": "LogSoftmax", "hardswish": "Hardswish", "hardsigmoid": "Hardsigmoid",
           "hardtanh": "Hardtanh", "softplus": "Softplus"}
_FN_POOL = {"max_pool1d", "max_pool2d", "max_pool3d", "avg_pool1d", "avg_pool2d", "avg_pool3d", "adaptive_avg_pool1d",
            "adaptive_avg_pool2d", "adaptive_avg_pool3d", "adaptive_max_pool1d", "adaptive_max_pool2d", "lp_pool2d",
            "max_pool2d_with_indices"}
_FN_DROP = {"dropout", "dropout1d", "dropout2d", "dropout3d", "alpha_dropout", "feature_alpha_dropout"}
_FN_NORM = {"batch_norm", "layer_norm", "group_norm", "instance_norm", "local_response_norm"}
_FN_TRIVIAL = {"contiguous", "to", "float", "double", "half", "bfloat16", "type", "type_as", "clone", "detach", "cpu",
               "cuda", "requires_grad_", "identity"}
_FN_MERGE_CAT = {"cat", "concat", "concatenate", "stack", "hstack", "vstack"}
_FN_MERGE_ARITH = {"add": "Add", "iadd": "Add", "sub": "Subtract", "isub": "Subtract", "mul": "Multiply",
                   "imul": "Multiply", "multiply": "Multiply", "subtract": "Subtract"}


def _op_name(n) -> str:
    t = n.target
    if n.op == "call_method":
        name = str(t)
    else:
        name = getattr(t, "__name__", str(t))
    return name.strip("_")


# =============================================================================================== the graph of a model
class _Node:
    def __init__(self, nid: str, kind: str, title: str, details=(), text: str = "", shapes=(), params: int = 0,
                 macs: int = 0, inputs=(), name: str = "", module=None, op: str = "", block: bool = False,
                 shared: bool = False):
        self.id, self.kind, self.title, self.details, self.text = nid, kind, title, list(details), text
        self.shapes = [tuple(s) for s in shapes]
        self.params, self.macs, self.inputs = params, macs, list(inputs)
        self.name, self.module, self.op, self.block, self.shared = name, module, op, block, shared

    @property
    def shape(self):
        return self.shapes[0] if self.shapes else ()


class _Graph:
    def __init__(self, nodes, traced: bool, depth, note: str = ""):
        self.nodes = nodes
        self.by_id = {n.id: n for n in nodes}
        self.traced, self.depth, self.note = traced, depth, note


def _levels(model: nn.Module) -> dict:
    """qualified module name -> nesting level, not counting ModuleList / ModuleDict containers."""
    mods = dict(model.named_modules())
    levels = {"": 0}
    for name in mods:
        if name == "":
            continue
        parent = name.rsplit(".", 1)[0] if "." in name else ""
        pm = mods.get(parent)
        step = 0 if isinstance(pm, (nn.ModuleList, nn.ModuleDict)) else 1
        levels[name] = levels.get(parent, 0) + step
    return levels


def _is_cut(m: nn.Module, qualname: str, depth, levels: dict) -> bool:
    """Is this module drawn as one box (not opened up) at this depth?"""
    if isinstance(m, (nn.ModuleList, nn.ModuleDict)):
        return False
    if not any(True for _ in m.children()):
        return True
    if depth is not None and qualname and levels.get(qualname, 0) >= depth:
        return True
    mod = type(m).__module__ or ""
    return mod.startswith("torch.nn") and not isinstance(m, nn.Sequential)


def _layer_macs(m: nn.Module, inp, out) -> int:
    """Multiply-adds of one call of a convolution / linear layer for the whole batch."""
    if not isinstance(out, torch.Tensor):
        return 0
    if isinstance(m, _CONV_T):
        x = inp[0]
        return int(x.numel() * (m.out_channels // m.groups) * int(np.prod(m.kernel_size)))
    if isinstance(m, nn.modules.conv._ConvNd):
        return int(out.numel() * (m.in_channels // m.groups) * int(np.prod(m.kernel_size)))
    if isinstance(m, nn.Linear):
        return int(out.numel() * m.in_features)
    if isinstance(m, nn.Bilinear):
        return int(out.numel() * m.in1_features * m.in2_features)
    return 0


def _macs_hooks(model: nn.Module, log: list):
    handles = []

    def hook(mod, inp, out):
        log.append(_layer_macs(mod, inp, out))

    for mod in model.modules():
        if isinstance(mod, (nn.modules.conv._ConvNd, nn.Linear, nn.Bilinear)):
            handles.append(mod.register_forward_hook(hook))
    return handles


def _info(v):
    if isinstance(v, torch.Tensor):
        return ("t", tuple(v.shape))
    if isinstance(v, (tuple, list)):
        return ("seq", [_info(a) for a in v])
    return ("x", type(v).__name__)


def _shapes_of(info) -> list:
    if info is None:
        return []
    if info[0] == "t":
        return [info[1][1:]]
    if info[0] == "seq":
        out = []
        for a in info[1]:
            out.extend(_shapes_of(a))
        return out
    return []


def _params_of(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _n_layers(m: nn.Module) -> int:
    return sum(1 for sub in m.modules() if not any(True for _ in sub.children()) and not isinstance(sub, nn.Identity))


def _module_node(nid, name, mod, shapes, macs, seen: set, cut_block: bool) -> _Node:
    # a layer object used twice (shared weights) is marked, so its parameters are not counted twice;
    # parameter-free layers such as ReLU are often reused, which is not worth a remark
    shared = id(mod) in seen and _params_of(mod) > 0
    seen.add(id(mod))
    params = 0 if shared else _params_of(mod)
    if cut_block:
        kind = _block_kind(mod)
        cls = type(mod).__name__
        n = _n_layers(mod)
        details = [cls]
        text = f"{cls}: {n} layers inside"
        return _Node(nid, kind, name or cls, details, text, shapes, params, macs, name=name, module=mod, block=True,
                     shared=shared)
    title, details, text = _describe_module(mod, shapes[0] if shapes else None)
    if shared:
        text += " (same layer again)"
    return _Node(nid, _module_kind(mod), title, details, text, shapes, params, macs, name=name, module=mod,
                 shared=shared)


class _Wrap(nn.Module):
    """Gives a model that is a single layer a parent, so that it is drawn as one box."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)


def _fx_graph(model: nn.Module, x: torch.Tensor, depth) -> _Graph:
    import torch.fx as fx

    levels = _levels(model)

    class Tracer(fx.Tracer):
        def is_leaf_module(self, m, qualname):
            return _is_cut(m, qualname, depth, levels)

    tracer = Tracer()
    graph = tracer.trace(model)
    gm = fx.GraphModule(tracer.root, graph)
    log: list = []

    class Recorder(fx.Interpreter):
        def run_node(self, n):
            start = len(log)
            out = super().run_node(n)
            n.meta["p_info"] = _info(out)
            n.meta["p_macs"] = sum(log[start:])
            return out

    handles = _macs_hooks(model, log)
    try:
        with torch.no_grad():
            Recorder(gm).run(x)
    finally:
        for h in handles:
            h.remove()

    nodes: list = []
    src: dict = {}
    seen: set = set()
    count = [0]
    first_placeholder = [True]

    def new_id():
        count[0] += 1
        return f"n{count[0]}"

    def arg_nodes(args) -> list:
        out = []

        def visit(a):
            if isinstance(a, fx.Node):
                out.append(a)
            elif isinstance(a, (list, tuple)):
                for b in a:
                    visit(b)
            elif isinstance(a, dict):
                for b in a.values():
                    visit(b)
        visit(args)
        return out

    def srcs(args) -> list:
        ids = []
        for a in arg_nodes(args):
            for i in src.get(a, []):
                if i not in ids:
                    ids.append(i)
        return ids

    for n in graph.nodes:
        info = n.meta.get("p_info")
        if n.op == "placeholder":
            if first_placeholder[0] and info and info[0] == "t":
                first_placeholder[0] = False
                nid = new_id()
                nodes.append(_Node(nid, "input", "input", [], "input image", _shapes_of(info), name="input"))
                src[n] = [nid]
            else:
                src[n] = []
            continue
        if n.op == "get_attr":
            src[n] = []
            continue
        if n.op == "output":
            ins = srcs(n.args)
            nodes.append(_Node(new_id(), "output", "output", [], "output", _shapes_of(info), inputs=ins,
                               name="output"))
            continue
        ins = srcs((n.args, n.kwargs))
        if n.op == "call_module":
            mod = gm.get_submodule(n.target)
            if isinstance(mod, nn.Identity):
                src[n] = ins
                continue
            is_block = any(True for _ in mod.children())
            node = _module_node(new_id(), str(n.target), mod, _shapes_of(info), n.meta.get("p_macs", 0), seen,
                                is_block)
            node.inputs = ins
            nodes.append(node)
            src[n] = [node.id]
            continue
        # call_function / call_method
        name = _op_name(n)
        if info is None or info[0] != "t":
            src[n] = ins if info is not None and info[0] == "seq" else []
            continue
        tensor_args = [a for a in arg_nodes((n.args, n.kwargs)) if src.get(a)]
        first = tensor_args[0].meta.get("p_info") if tensor_args else None
        in_shape = first[1] if first and first[0] == "t" else None
        out_shape = info[1]
        shapes = _shapes_of(info)
        if name == "getitem":
            if first is not None and first[0] == "seq":
                src[n] = ins
                continue
            node = _Node(new_id(), "other", "Slice", [], "take a slice", shapes, inputs=ins, name=n.name, op=name)
        elif name in _FN_MERGE_CAT:
            node = _Node(new_id(), "merge", "Concat", [f"{len(ins)} inputs"],
                         f"concatenate {len(ins)} inputs along the channels", shapes, inputs=ins, name=n.name,
                         op="cat")
        elif name in _FN_MERGE_ARITH:
            if len(ins) < 2:
                src[n] = ins
                continue
            word = _FN_MERGE_ARITH[name]
            node = _Node(new_id(), "merge", word, [], f"{word.lower()} {len(ins)} inputs element by element", shapes,
                         inputs=ins, name=n.name, op=word.lower())
        elif name in ("flatten", "view", "reshape"):
            if in_shape is not None and tuple(in_shape) == tuple(out_shape):
                src[n] = ins
                continue
            if len(out_shape) == 2 and in_shape is not None and len(in_shape) > 2:
                node = _Node(new_id(), "flatten", "Flatten", [], f"Flatten to {_fmt_shape(out_shape[1:])} numbers",
                             shapes, inputs=ins, name=n.name, op="flatten")
            else:
                node = _Node(new_id(), "other", "Reshape", [], f"reshape to {_fmt_shape(out_shape[1:])}", shapes,
                             inputs=ins, name=n.name, op=name)
        elif name in _FN_ACT:
            node = _Node(new_id(), "activation", _FN_ACT[name], [], _FN_ACT[name], shapes, inputs=ins, name=n.name,
                         op=name)
        elif name in _FN_POOL:
            if name.startswith("adaptive"):
                title = "AvgPool" if "avg" in name else "MaxPool"
                details = [f"to {_fmt_shape(out_shape[2:], chr(215))}"]
                text = f"{title} to {_fmt_shape(out_shape[2:])}"
            else:
                title = "MaxPool" if "max" in name else "AvgPool"
                k = n.args[1] if len(n.args) > 1 else n.kwargs.get("kernel_size")
                if isinstance(k, (int, tuple, list)):
                    title += " " + _k(k, chr(215))
                details, text = [], title.replace(chr(215), "x")
            node = _Node(new_id(), "pooling", title, details, text, shapes, inputs=ins, name=n.name, op=name)
        elif name in ("mean", "amax", "sum") and in_shape is not None and len(in_shape) == 4 and len(out_shape) <= 2:
            node = _Node(new_id(), "pooling", "AvgPool" if name == "mean" else "Pool", ["over H, W"],
                         f"{name} over height and width", shapes, inputs=ins, name=n.name, op=name)
        elif name in _FN_DROP:
            p = n.kwargs.get("p", n.args[1] if len(n.args) > 1 else 0.5)
            node = _Node(new_id(), "dropout", "Dropout", [f"p = {p}"], f"Dropout p={p}", shapes, inputs=ins,
                         name=n.name, op=name)
        elif name in _FN_NORM:
            node = _Node(new_id(), "normalisation", "Norm", [], name, shapes, inputs=ins, name=n.name, op=name)
        elif name in ("interpolate", "upsample", "upsample_bilinear", "upsample_nearest"):
            node = _Node(new_id(), "other", "Upsample", [], f"upsample to {_fmt_shape(out_shape[1:])}", shapes,
                         inputs=ins, name=n.name, op=name)
        elif name in ("conv1d", "conv2d", "conv3d", "conv_transpose2d"):
            node = _Node(new_id(), "convolution", "Conv", [], name, shapes, inputs=ins, name=n.name, op=name)
        elif name == "linear":
            node = _Node(new_id(), "linear", "Linear", [], "linear", shapes, inputs=ins, name=n.name, op=name)
        elif name in _FN_TRIVIAL:
            src[n] = ins
            continue
        else:
            node = _Node(new_id(), "other", name, [], name, shapes, inputs=ins, name=n.name, op=name)
        nodes.append(node)
        src[n] = [node.id]
    return _prune(_Graph(nodes, True, depth))


def _cut_modules(model: nn.Module, depth) -> list:
    levels = _levels(model)
    cut, inside = [], []
    for name, m in model.named_modules():
        if any(name.startswith(p + ".") for p in inside):
            continue
        if name == "" and any(True for _ in m.children()) and not _is_cut(m, "", None, levels):
            continue
        if _is_cut(m, name, depth, levels) or (name == "" and not any(True for _ in m.children())):
            cut.append((name, m))
            inside.append(name)
    return cut


def _hook_graph(model: nn.Module, x: torch.Tensor, depth) -> _Graph:
    """Fallback when torch.fx cannot trace the model: layers in the order they run, as a chain."""
    cut = _cut_modules(model, depth)
    log: list = []
    records: list = []
    handles = _macs_hooks(model, log)
    starts: dict = {}

    def make(name):
        def pre(mod, args):
            starts.setdefault(id(mod), []).append(len(log))

        def post(mod, args, out):
            start = starts[id(mod)].pop()
            records.append((name, mod, _info(out), start))
        return pre, post

    for name, m in cut:
        if isinstance(m, nn.Identity):
            continue
        pre, post = make(name)
        handles.append(m.register_forward_pre_hook(pre))
        handles.append(m.register_forward_hook(post))
    try:
        with torch.no_grad():
            out = model(x)
    finally:
        for h in handles:
            h.remove()
    # post hooks fire in completion order; MACs window = log entries recorded during the call
    nodes = [_Node("n0", "input", "input", [], "input image", [tuple(x.shape[1:])], name="input")]
    seen: set = set()
    prev = "n0"
    for k, (name, mod, info, start) in enumerate(records):
        end = records[k + 1][3] if k + 1 < len(records) else len(log)
        macs = sum(log[start:]) if k + 1 == len(records) else sum(log[start:end])
        is_block = any(True for _ in mod.children())
        node = _module_node(f"n{k + 1}", name, mod, _shapes_of(info), macs, seen, is_block)
        node.inputs = [prev]
        nodes.append(node)
        prev = node.id
    nodes.append(_Node(f"n{len(records) + 1}", "output", "output", [], "output", _shapes_of(_info(out)),
                       inputs=[prev], name="output"))
    return _Graph(nodes, False, depth, note="torch.fx could not follow this model's forward(), so the layers are "
                                            "shown in the order they run; branches, skips and functions such as "
                                            "torch.relu are missing")


def _prune(g: _Graph) -> _Graph:
    """Keep nodes on a path from the input to the output."""
    nodes = g.nodes
    if not nodes or nodes[0].kind != "input" or nodes[-1].kind != "output":
        return g
    fwd = {nodes[0].id}
    for n in nodes[1:]:
        if any(i in fwd for i in n.inputs):
            fwd.add(n.id)
    back = {nodes[-1].id}
    for n in reversed(nodes):
        if n.id in back:
            back.update(n.inputs)
    keep = []
    for n in nodes:
        if n.id in fwd and n.id in back:
            n.inputs = [i for i in n.inputs if i in fwd and i in back]
            keep.append(n)
    return _Graph(keep, g.traced, g.depth, g.note)


def _graph(model: nn.Module, input_shape, depth=None) -> _Graph:
    shape = _norm_shape(input_shape)
    root = model if any(True for _ in model.children()) else _Wrap(model)
    x = torch.zeros((1,) + shape, device=_model_device(model))
    was = model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=[]):
            with torch.no_grad():
                _forward_checked(model, x)
            try:
                g = _fx_graph(root, x, depth)
            except ModelShapeError:
                raise
            except Exception:  # noqa: BLE001  (tracing failed: data-dependent Python code in forward)
                g = _hook_graph(root, x, depth)
    finally:
        model.train(was)
    g.warnings = []
    for n in g.nodes:
        if isinstance(n.module, nn.Linear) and len(n.shape) >= 2 and len(shape) == 3:
            g.warnings.append(
                f"warning: layer '{n.name}' (Linear) works on a {_fmt_shape(n.shape)} block per image, not on a flat "
                f"list of numbers: it only mixes the last axis. In an image classifier, put nn.Flatten() before the "
                f"first nn.Linear.")
    return g


def _max_level(model: nn.Module) -> int:
    lv = _levels(model)
    return max(lv.values()) if lv else 0


# =============================================================================================== summary
def _shape_text(shapes, sep=" x ") -> str:
    if not shapes:
        return ""
    return " | ".join(_fmt_shape(s, sep) for s in shapes)


def _human(n: float) -> str:
    n = float(n)
    if n >= 1e9:
        return f"{n / 1e9:.2f} billion"
    if n >= 1e6:
        return f"{n / 1e6:.2f} million"
    if n >= 1e3:
        return f"{n / 1e3:.1f} thousand"
    return f"{int(n)}"


def _table(headers, rows, aligns) -> list:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    lines = []

    def fmt(cells):
        parts = []
        for c, w, a in zip(cells, widths, aligns):
            parts.append(c.rjust(w) if a == ">" else c.ljust(w))
        return "  ".join(parts).rstrip()

    lines.append(fmt(headers))
    lines.append("  ".join("-" * w for w in widths))
    for r in rows:
        lines.append(fmt(r))
    return lines


def _count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def summary(model: nn.Module, input_shape, depth: int | None = None, verbose: bool = True):
    """Print one row per layer: name, kind, what it does, output shape, parameters, multiply-adds.

    The rows follow the data from the input to the output (same order as P.show_model). Multiply-adds
    count the work of the convolution and linear layers for one image (everything else is small).
    depth: like P.show_model (1 = one row per top-level block). Returns the rows as a list of dicts with
    keys layer, kind, settings, output_shape, params, mult_adds.
    """
    g = _graph(model, input_shape, depth)
    rows = _Rows()
    cells = []
    for n in g.nodes:
        if n.kind in ("input", "output"):
            continue
        kind = n.kind if not n.block else f"block ({n.kind})"
        rows.append({"layer": n.name, "kind": kind, "settings": n.text,
                     "output_shape": n.shapes[0] if len(n.shapes) == 1 else tuple(n.shapes),
                     "params": n.params, "mult_adds": n.macs})
        cells.append([n.name, kind, n.text, _shape_text(n.shapes), f"{n.params:,}", f"{n.macs:,}"])
    shape = _norm_shape(input_shape)
    total, trainable = _count_params(model), _count_params(model, True)
    macs = sum(n.macs for n in g.nodes)
    head = ["layer", "kind", "what it does", "output shape", "parameters", "mult-adds"]
    lines = _table(head, [["(input)", "", "one image", _fmt_shape(shape), "", ""]] + cells +
                   [["total", f"{len(cells)} layers", "", "", f"{total:,}", f"{macs:,}"]],
                   ["<", "<", "<", "<", ">", ">"])
    lines.insert(len(lines) - 1, "-" * len(lines[1]))
    lines.append("")
    lines.append(f"parameters: {total:,} ({trainable:,} trainable) = {total * 4 / 2 ** 20:.2f} MB as float32 numbers")
    lines.append(f"multiply-adds for one image: {macs:,} ({_human(macs)})")
    if not g.traced:
        lines.append(f"note: {g.note}")
    lines.extend(g.warnings)
    rows.text = "\n".join(lines)
    if verbose:
        print(rows.text)
    return rows._mark()


# =============================================================================================== show_model: layout
_FS_TITLE, _FS = 10, 9
_LINE_T, _LINE = 0.175, 0.158          # line heights (inches) of a box title and a detail line
_GAP = 0.3                             # horizontal gap between two boxes (arrow)
_PAD = 0.08                            # box bottom -> first label line
_FAN = 0.32                            # space for the fan-out / fan-in lines of parallel branches
_LANE_GAP = 0.14
_ROW_GAP = 0.42
_ARC_STEP = 0.15
_TAG_H = 0.34
_MARGIN = 0.4
_MAX_W = 11.2

_TW_CACHE: dict = {}


def _text_w(s: str, size: float, bold: bool = False) -> float:
    key = (s, size, bold)
    if key not in _TW_CACHE:
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextPath
        fp = FontProperties(size=size, weight="bold" if bold else "normal")
        _TW_CACHE[key] = TextPath((0, 0), s, prop=fp).get_extents().width / 72 if s else 0.0
    return _TW_CACHE[key]


class _Scale:
    """Box size from the output shape: height ~ image size H, width ~ number of maps (log scale)."""

    def __init__(self, nodes):
        hs, cs, vs = [1], [1], [1]
        for n in nodes:
            s = n.shape
            if len(s) == 3:
                hs.append(s[1])
                cs.append(s[0])
            elif len(s) == 1:
                vs.append(s[0])
        self.hmax, self.cmax, self.vmax = max(hs), max(cs), max(vs)

    def box(self, n: _Node):
        s = n.shape
        if n.kind == "merge" and n.op in ("add", "subtract", "multiply"):
            return 0.3, 0.3
        if len(s) == 3:
            h = 0.32 + 0.93 * math.sqrt(max(s[1], 1) / self.hmax)
            w = 0.2 + 0.42 * (math.log2(max(s[0], 1)) / math.log2(self.cmax) if self.cmax > 1 else 0)
        elif len(s) == 1:
            h = 0.32 + 0.93 * (math.log10(max(s[0], 1)) / math.log10(self.vmax) if self.vmax > 1 else 0)
            w = 0.16
        elif len(s) == 2:
            h, w = 0.32 + 0.93 * math.sqrt(max(s[1], 1) / self.hmax), 0.3
        else:
            h, w = 0.5, 0.3
        if n.kind in _THIN:
            w = 0.12
        return w, h


def _label_lines(n: _Node) -> list:
    """[(text, style)] under a box; style: title | detail | shape | params."""
    X = "×"
    if n.kind == "input":
        return [("input", "title"), (_shape_text(n.shapes, X), "shape")]
    if n.kind == "output":
        s = n.shapes[0] if n.shapes else ()
        txt = f"{s[0]} numbers" if len(s) == 1 else _shape_text(n.shapes, X)
        return [("output", "title"), (txt, "shape")]
    lines = [(n.title, "title")]
    for d in n.details:
        lines.append((d, "detail"))
    if n.block:
        for k, s in enumerate(n.shapes[:3]):
            lines.append((("→ " if k == 0 else "+ ") + _fmt_shape(s, X), "shape"))
    elif n.kind not in _THIN and not (n.kind == "linear" and n.details):
        lines.append(("→ " + _shape_text(n.shapes[:2], X), "shape"))
    if n.shared:
        lines.append(("same layer again", "params"))
    elif n.params:
        lines.append((f"{n.params:,} params", "params"))
    return lines


class _U:
    """A unit of the layout: one box, or a group of parallel branches (lanes)."""

    def __init__(self, node=None, lanes=None):
        self.node, self.lanes = node, lanes
        self.is_box = node is not None


class _Track:
    def __init__(self, units, tid):
        self.units, self.id = units, tid
        self.arcs: list = []      # (i, j, src_id, dst_id, level)
        self.tags = False


def _series(start, stop, succ, ipdom, used, arcs):
    items = []
    node = start
    while node != stop:
        if node in used:
            raise ValueError("not series-parallel")
        used.add(node)
        items.append(("box", node))
        nexts = succ[node]
        if len(nexts) == 1:
            node = nexts[0]
            continue
        if not nexts:
            raise ValueError("dead end")
        merge = ipdom[node]
        lanes = []
        direct = False
        for s in nexts:
            if s == merge:
                direct = True
            else:
                lanes.append(_series(s, merge, succ, ipdom, used, arcs))
        if direct:
            arcs.append((node, merge))
        if len(lanes) == 1:
            items.extend(lanes[0])
        elif lanes:
            items.append(("par", lanes))
        node = merge
    return items


def _structure(g: _Graph):
    """Series-parallel decomposition: (items, arcs, ok). Items: ("box", id) or ("par", [lane items])."""
    ids = [n.id for n in g.nodes]
    order = {nid: k for k, nid in enumerate(ids)}
    succ = {nid: [] for nid in ids}
    edges = set()
    for n in g.nodes:
        for i in n.inputs:
            if n.id not in succ[i]:
                succ[i].append(n.id)
            edges.add((i, n.id))
    for nid in ids:
        succ[nid].sort(key=order.get)
    sink = ids[-1]
    pdom: dict = {}
    for nid in reversed(ids):
        if nid == sink:
            pdom[nid] = {sink}
            continue
        sets = [pdom[s] for s in succ[nid] if s in pdom]
        inter = set.intersection(*sets) if sets else set()
        pdom[nid] = inter | {nid}
    ipdom = {}
    for nid in ids:
        rest = pdom[nid] - {nid}
        ipdom[nid] = min(rest, key=order.get) if rest else sink
    arcs: list = []
    if g.traced:
        try:
            items = _series(ids[0], sink, succ, ipdom, set(), arcs)
            items.append(("box", sink))
            return items, arcs, edges, True
        except (ValueError, KeyError, RecursionError):
            pass
    items = [("box", nid) for nid in ids]
    arcs = []
    for a, b in edges:
        if order[b] != order[a] + 1:
            arcs.append((a, b))
    return items, arcs, edges, False


class _Diagram:
    def __init__(self, g: _Graph):
        self.g = g
        self.scale = _Scale(g.nodes)
        self.box = {}          # node id -> (w, h)
        self.lines = {}        # node id -> label lines
        self.slot = {}         # node id -> slot width
        self.lab_h = {}
        for n in g.nodes:
            self.box[n.id] = self.scale.box(n)
            lines = _label_lines(n)
            self.lines[n.id] = lines
            lw = 0.0
            lh = 0.0
            for text, style in lines:
                bold = style == "title"
                lw = max(lw, _text_w(text, _FS_TITLE if bold else _FS, bold))
                lh += _LINE_T if bold else _LINE
            self.slot[n.id] = max(self.box[n.id][0], lw) + 0.04
            self.lab_h[n.id] = lh
        self.pos: dict = {}    # node id -> (cx, cy, w, h, label_top)
        self.where: dict = {}  # node id -> (track, index)
        self.tracks: list = []

    # ---------------------------------------------------------------- build units from the structure
    def units(self, items) -> list:
        out = []
        for kind, val in items:
            if kind == "box":
                out.append(_U(node=val))
            else:
                out.append(_U(lanes=[self.units(lane) for lane in val]))
        return out

    def width(self, u: _U) -> float:
        if u.is_box:
            return self.slot[u.node]
        return 2 * _FAN + max(self.track_width(lane) for lane in u.lanes)

    def track_width(self, units) -> float:
        if not units:
            return 0.0
        return sum(self.width(u) for u in units) + _GAP * (len(units) - 1)

    # ---------------------------------------------------------------- tracks, arcs, heights
    def register(self, units, tid) -> _Track:
        t = _Track(units, tid)
        self.tracks.append(t)
        for k, u in enumerate(units):
            if u.is_box:
                self.where[u.node] = (t, k)
            else:
                u.lane_tracks = []
                for lane in u.lanes:
                    u.lane_tracks.append(self.register(lane, len(self.tracks)))
        return t

    def assign_arcs(self, arcs):
        self.tags: list = []
        for a, b in arcs:
            ta, ia = self.where[a]
            tb, ib = self.where[b]
            if ta is tb:
                i, j = (ia, ib) if ia < ib else (ib, ia)
                ta.arcs.append([i, j, a, b, 1])
            else:
                self.tags.append((a, b))
                ta.tags = True
                tb.tags = True
        for t in self.tracks:
            t.arcs.sort(key=lambda r: r[1] - r[0])
            placed = []
            for arc in t.arcs:
                level = 1
                for other in placed:
                    if arc[0] < other[1] and other[0] < arc[1]:
                        level = max(level, other[4] + 1)
                arc[4] = level
                placed.append(arc)

    def measure(self, t: _Track):
        band, lab = 0.3, 0.0
        above, below = 0.0, 0.0
        for u in t.units:
            if u.is_box:
                band = max(band, self.box[u.node][1])
                lab = max(lab, self.lab_h[u.node])
        for u in t.units:
            if not u.is_box:
                h = 0.0
                for lt in u.lane_tracks:
                    self.measure(lt)
                    h += lt.height
                h += _LANE_GAP * (len(u.lane_tracks) - 1)
                u.height = h
                above = max(above, h / 2)
                below = max(below, h / 2)
        t.band = band
        t.core_above = max(above, band / 2)
        t.below = max(below, band / 2 + _PAD + lab)
        levels = max([a[4] for a in t.arcs] + [0])
        t.tag_h = _TAG_H if t.tags else 0.0
        t.arc_h = (t.tag_h + 0.16 + (levels - 1) * _ARC_STEP + 0.06) if levels else t.tag_h
        t.height = t.arc_h + t.core_above + t.below

    # ---------------------------------------------------------------- placement
    def place(self, t: _Track, x0: float, yc: float):
        x = x0
        t.yc = yc
        t.x0 = x0
        for u in t.units:
            w = self.width(u)
            u.x0, u.x1, u.yc = x, x + w, yc
            if u.is_box:
                bw, bh = self.box[u.node]
                self.pos[u.node] = (x + w / 2, yc, bw, bh, yc - t.band / 2 - _PAD)
                u.entry = (x + w / 2 - bw / 2, yc)
                u.exit = (x + w / 2 + bw / 2, yc)
            else:
                top = yc + u.height / 2
                for lt in u.lane_tracks:
                    lyc = top - lt.arc_h - lt.core_above
                    self.place(lt, x + _FAN, lyc)
                    top -= lt.height + _LANE_GAP
                u.entry = (x, yc)
                u.exit = (x + w, yc)
            x += w + _GAP
        t.x1 = x - _GAP

    # ---------------------------------------------------------------- rows
    def break_rows(self, units, arcs, max_w):
        n = len(units)
        widths = [self.width(u) for u in units]
        index = {}
        for k, u in enumerate(units):
            for nid in self.ids_in(u):
                index[nid] = k
        cross = [0] * (n + 1)
        for a, b in arcs:
            i, j = sorted((index[a], index[b]))
            for k in range(i + 1, j + 1):
                cross[k] += 1
        no_break = [False] * (n + 1)
        for k, u in enumerate(units):
            if not u.is_box:
                no_break[k] = True
                no_break[k + 1] = True
        best = [math.inf] * (n + 1)
        prev = [0] * (n + 1)
        best[0] = 0.0
        for j in range(1, n + 1):
            w = -_GAP
            for i in range(j - 1, -1, -1):
                w += widths[i] + _GAP
                if i > 0 and no_break[i]:
                    if w > max_w * 3:
                        break
                    continue
                cost = best[i] + 3.0 + 10 * ((max_w - w) / max_w) ** 2
                if w > max_w:
                    cost += 1000 + 100 * (w - max_w) if j - i > 1 else 0
                if j < n:
                    cost += 6 * cross[j]
                if j - i == 1 and n > 1:
                    cost += 4
                if cost < best[j]:
                    best[j], prev[j] = cost, i
                if w > max_w * 3:
                    break
        rows = []
        j = n
        while j > 0:
            i = prev[j]
            rows.append(units[i:j])
            j = i
        return list(reversed(rows))

    def ids_in(self, u: _U) -> list:
        if u.is_box:
            return [u.node]
        out = []
        for lane in u.lanes:
            for v in lane:
                out.extend(self.ids_in(v))
        return out


def _name_of(title, model) -> str:
    return title or type(model).__name__


def _darker(hex_color: str, f: float = 0.72) -> tuple:
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hex_color)
    return (r * f, g * f, b * f)


def _arrow(ax, pts, head=True, color=_INK2, lw=1.0, z=1, ls="-"):
    verts = [tuple(p) for p in pts]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 1)
    patch = FancyArrowPatch(path=MplPath(verts, codes), arrowstyle="-|>" if head else "-", mutation_scale=9,
                            color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0, linestyle=ls)
    ax.add_patch(patch)


def show_model(model: nn.Module, input_shape, depth: int | None = None, title: str | None = None):
    """Draw the network as boxes in the order the data flows, and return the figure.

    One box per layer, coloured by its kind; under it what it does ("Conv 3x3, 32 maps, stride 1, pad 1")
    and the shape of its output (maps x height x width). Box height follows the image size, box width
    the number of maps, so you see the image shrink and the maps grow. Parallel branches are drawn
    one above the other and joined where they are concatenated or added; skip connections are red
    arcs (or numbered tags when they jump to another row).

    input_shape: the shape of ONE image, e.g. (1, 28, 28).
    depth: how far blocks are opened up. None = every layer (very big networks are drawn at a
    smaller depth automatically, with a note); 1 = one box per top-level part of the model; 2 = one
    level deeper, and so on.
    """
    note = ""
    g = _graph(model, input_shape, depth)
    full = g if depth is None else _graph(model, input_shape, None)
    g.n_total = len(full.nodes) - 2
    if depth is None and len(g.nodes) > 90:
        chosen = None
        for d in range(1, _max_level(model) + 1):
            gd = _graph(model, input_shape, d)
            if len(gd.nodes) <= 60 or chosen is None:
                chosen = (d, gd)
            if len(gd.nodes) > 60:
                break
        note = (f"{len(g.nodes) - 2} layers in total: drawn at depth={chosen[0]}, each stacked box is a block of "
                f"layers (show_model(..., depth=...) opens them)")
        n_total = g.n_total
        g = chosen[1]
        g.n_total = n_total
    for w in getattr(g, "warnings", []):
        print(w)
    fig = _draw(g, model, input_shape, title, note)
    return _show(fig)


def _draw(g: _Graph, model, input_shape, title, note) -> _Figure:
    D = _Diagram(g)
    items, arcs, edges, sp = _structure(g)
    units = D.units(items)
    rows_units = D.break_rows(units, arcs, _MAX_W)
    rows = []
    for k, ru in enumerate(rows_units):
        rows.append(D.register(ru, len(D.tracks)))
    D.assign_arcs(arcs)
    for t in rows:
        D.measure(t)
    content_w = max(D.track_width(t.units) for t in rows)
    shape = _norm_shape(input_shape)
    out_shapes = g.nodes[-1].shapes
    params = _count_params(model)
    macs = sum(n.macs for n in g.nodes)
    n_boxes = sum(1 for n in g.nodes if n.kind not in ("input", "output"))
    X = "×"
    if any(n.block for n in g.nodes):
        count_txt = f"{n_boxes} boxes ({getattr(g, 'n_total', _n_layers(model))} layers inside)"
    else:
        count_txt = f"{n_boxes} layers"
    sub = (f"input {_fmt_shape(shape, X)}  →  output {_shape_text(out_shapes, X)}   ·   {count_txt}   ·   "
           f"{params:,} parameters   ·   {_human(macs)} multiply-adds per image")
    legend_note = ("box height = image size (height × width) · box width = number of maps · "
                   "under each box: the layer and the shape of its output")
    note_txt = ("note: " + (note or g.note)) if (note or not g.traced) else ""
    text_w = max(_text_w(sub, 10), _text_w(note_txt, 9), _text_w(_name_of(title, model), 13, True))
    W = max(content_w + 2 * _MARGIN + 0.3, text_w + 2 * _MARGIN, 6.5)
    note_lines = [legend_note]
    if _text_w(legend_note, 9) > W - 2 * _MARGIN:
        note_lines = ["box height = image size (height × width) · box width = number of maps",
                      "under each box: the layer and the shape of its output"]
    kinds = []
    for kname in KIND_COLORS:
        if kname in ("input", "output"):
            continue
        for n in g.nodes:
            if n.kind == kname:
                kinds.append(kname)
                break
    has_block = any(n.block for n in g.nodes)
    has_skip = bool(arcs) and sp or bool(D.tags)
    # legend: rows of entries
    entries = [(KIND_COLORS[k], _KIND_NAMES[k], "patch") for k in kinds]
    if has_skip:
        label = "skip connection" + (" (a numbered circle continues at the same number)" if D.tags else "")
        entries.append((KIND_COLORS["merge"], label, "line"))
    if has_block:
        entries.append(("#bbbbbb", "stacked box = a block of layers", "stack"))
    leg_rows, cur, cur_w = [], [], 0.0
    for e in entries:
        ew = 0.32 + _text_w(e[1], 9.5) + 0.35
        if cur and cur_w + ew > W - 2 * _MARGIN:
            leg_rows.append(cur)
            cur, cur_w = [], 0.0
        cur.append((e, ew))
        cur_w += ew
    if cur:
        leg_rows.append(cur)
    title_h = 0.78 + (0.22 if note_txt else 0)
    legend_h = 0.28 * len(leg_rows) + 0.2 + 0.2 * len(note_lines)
    H = title_h + sum(t.height for t in rows) + _ROW_GAP * (len(rows) - 1) + legend_h + 0.15
    fig = _Figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    # title
    ax.text(_MARGIN, H - 0.22, _name_of(title, model), fontsize=13, fontweight="bold", color=_INK, va="top")
    ax.text(_MARGIN, H - 0.52, sub, fontsize=10, color=_INK2, va="top")
    if note_txt:
        ax.text(_MARGIN, H - 0.76, note_txt, fontsize=9, color=_INK2, va="top", style="italic")
    # place rows
    y = H - title_h
    for t in rows:
        yc = y - t.arc_h - t.core_above
        D.place(t, _MARGIN + 0.15, yc)
        y = yc - t.below - _ROW_GAP
    # ------------------------------------------------ edges
    def entry_of(u):
        return u.entry

    def draw_track_edges(t: _Track):
        for k, u in enumerate(t.units):
            if not u.is_box:
                # fan-out and fan-in of a parallel group
                xt_in = u.x0 + _FAN * 0.5
                xt_out = u.x1 - _FAN * 0.5
                _arrow(ax, [u.entry, (xt_in, u.yc)], head=False)
                _arrow(ax, [(xt_out, u.yc), u.exit], head=False)
                for lt in u.lane_tracks:
                    first, last = lt.units[0], lt.units[-1]
                    fy, ly = first.entry[1], last.exit[1]
                    _arrow(ax, [(xt_in, u.yc), (xt_in, fy), first.entry], head=first.is_box)
                    _arrow(ax, [last.exit, (xt_out, ly), (xt_out, u.yc)], head=False)
                    draw_track_edges(lt)
            if k + 1 < len(t.units):
                v = t.units[k + 1]
                connected = True
                if u.is_box and v.is_box:
                    connected = (u.node, v.node) in edges
                if connected:
                    _arrow(ax, [u.exit, v.entry], head=v.is_box)

    for t in rows:
        draw_track_edges(t)
    # wrap connectors between rows
    for r in range(len(rows) - 1):
        a, b = rows[r], rows[r + 1]
        last, first = a.units[-1], b.units[0]
        if last.is_box and first.is_box and (last.node, first.node) not in edges:
            continue
        xr = a.x1 + 0.14
        ymid = a.yc - a.below - _ROW_GAP / 2
        xl = _MARGIN - 0.12
        _arrow(ax, [last.exit, (xr, last.exit[1]), (xr, ymid), (xl, ymid), (xl, first.entry[1]), first.entry],
               head=first.is_box)
    # skip arcs within a track
    skip_c = KIND_COLORS["merge"]
    for t in D.tracks:
        for i, j, a, b, level in t.arcs:
            ax_, ay_, aw, ah, _ = D.pos[a]
            bx_, by_, bw, bh, _ = D.pos[b]
            ya = t.yc + t.core_above + t.tag_h + 0.16 + (level - 1) * _ARC_STEP
            sx = ax_ + aw * 0.25
            dx = bx_ - bw * 0.25
            _arrow(ax, [(sx, ay_ + ah / 2), (sx, ya), (dx, ya), (dx, by_ + bh / 2)], color=skip_c, lw=1.3, z=2)
    # numbered tags for skips that change rows
    for k, (a, b) in enumerate(D.tags):
        for nid, is_dst in ((a, False), (b, True)):
            cx, cy, bw, bh, _ = D.pos[nid]
            top = cy + bh / 2
            tx = cx + (-0.08 if is_dst else 0.08)
            ty = top + 0.22
            if is_dst:
                _arrow(ax, [(tx, ty - 0.1), (tx, top)], color=skip_c, lw=1.3, z=2)
            else:
                _arrow(ax, [(tx, top), (tx, ty - 0.1)], head=False, color=skip_c, lw=1.3, z=2)
            ax.add_patch(Circle((tx, ty), 0.1, facecolor="white", edgecolor=skip_c, lw=1.3, zorder=5))
            ax.text(tx, ty, str(k + 1), fontsize=8.5, color=skip_c, ha="center", va="center", fontweight="bold",
                    zorder=6)
    # ------------------------------------------------ boxes and labels
    for n in g.nodes:
        if n.id not in D.pos:
            continue
        cx, cy, bw, bh, lab_top = D.pos[n.id]
        color = KIND_COLORS.get(n.kind, KIND_COLORS["other"])
        if n.kind == "merge" and n.op in ("add", "subtract", "multiply"):
            ax.add_patch(Circle((cx, cy), 0.15, facecolor=color, edgecolor=_darker(color), lw=1, zorder=3))
            sym = {"add": "+", "subtract": "−", "multiply": "×"}[n.op]
            ax.text(cx, cy - 0.01, sym, fontsize=12, color="white", ha="center", va="center", fontweight="bold",
                    zorder=4)
        else:
            if n.block:
                for off in (0.08, 0.04):
                    ax.add_patch(FancyBboxPatch((cx - bw / 2 + off, cy - bh / 2 + off), bw, bh,
                                                boxstyle="round,pad=0,rounding_size=0.03", facecolor=color,
                                                edgecolor=_darker(color), lw=0.8, alpha=0.45, zorder=3))
            edge = _INK2 if n.kind in ("input", "output") else _darker(color)
            ax.add_patch(FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.03",
                                        facecolor=color, edgecolor=edge, lw=0.9, zorder=3))
        yy = lab_top
        for text, style in D.lines[n.id]:
            bold = style == "title"
            size = _FS_TITLE if bold else _FS
            ax.text(cx, yy, text, fontsize=size, fontweight="bold" if bold else "normal",
                    color=_INK2 if style == "params" else _INK, ha="center", va="top", zorder=4)
            yy -= _LINE_T if bold else _LINE
    # ------------------------------------------------ legend
    ly = 0.15 + 0.28 * len(leg_rows) + 0.2 * len(note_lines) - 0.08
    for line in note_lines:
        ax.text(_MARGIN, ly + 0.02, line, fontsize=9, color=_INK2, va="bottom")
        ly -= 0.2
    ly -= 0.1
    for row in leg_rows:
        x = _MARGIN
        for (color, label, style), ew in row:
            if style == "patch":
                ax.add_patch(FancyBboxPatch((x, ly - 0.07), 0.22, 0.16, boxstyle="round,pad=0,rounding_size=0.02",
                                            facecolor=color, edgecolor=_darker(color), lw=0.8))
            elif style == "line":
                _arrow(ax, [(x, ly + 0.01), (x + 0.24, ly + 0.01)], color=color, lw=1.3)
            else:
                for off in (0.06, 0.03, 0.0):
                    ax.add_patch(Rectangle((x + off, ly - 0.07 + off), 0.18, 0.13, facecolor="#cfcfcf",
                                           edgecolor="#777777", lw=0.7))
            ax.text(x + 0.32, ly + 0.01, label, fontsize=9.5, color=_INK, va="center")
            x += ew
        ly -= 0.28
    return fig


# =============================================================================================== timing and cost
def _time(fn, budget: float = 0.4, min_runs: int = 5, max_runs: int = 40) -> float:
    """Median seconds of fn() after warm-up; fewer runs for slow functions."""
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    if first > 0.5:
        min_runs, max_runs = 2, 3
    else:
        fn()
    times = []
    start = time.perf_counter()
    while len(times) < max_runs and (len(times) < min_runs or time.perf_counter() - start < budget):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _generic_loss(out, n_classes_hint: int = 10):
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.float().pow(2).mean()


def _measure(model: nn.Module, shape: tuple, batch_size: int = 32, train_step: bool = True) -> dict:
    dev = _model_device(model)
    was = model.training
    res = {}
    with torch.random.fork_rng(devices=[]):
        g = torch.Generator()
        g.manual_seed(0)
        x1 = torch.zeros((1,) + shape, device=dev)
        model.eval()
        try:
            with torch.no_grad():
                t1 = _time(lambda: model(x1))
                b = batch_size
                if t1 * b > 3.0:
                    b = max(2, int(3.0 / max(t1, 1e-6)))
                xb = torch.randn((b,) + shape, generator=g).to(dev)
                tb = _time(lambda: model(xb))
            res.update(ms_single=t1 * 1e3, ms_batch=tb * 1e3 / b, batch=b)
            if train_step:
                state = copy.deepcopy(model.state_dict())
                model.train()
                opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad] or [torch.zeros(1)], lr=1e-3)

                def step():
                    out = model(xb)
                    if isinstance(out, torch.Tensor) and out.dim() == 2:
                        y = torch.randint(0, out.shape[1], (out.shape[0],), generator=g).to(dev)
                        loss = F.cross_entropy(out, y)
                    else:
                        loss = _generic_loss(out)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                try:
                    res["ms_step"] = _time(step, budget=0.6) * 1e3
                finally:
                    model.load_state_dict(state)
                    model.zero_grad(set_to_none=True)
        finally:
            model.train(was)
    return res


def _fmt_ms(v: float) -> str:
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.3g}"


def _fmt_s(v: float) -> str:
    if v >= 120:
        return f"{v / 60:.1f} min"
    if v >= 10:
        return f"{v:.0f} s"
    return f"{v:.1f} s"


def cost(model: nn.Module, input_shape, name: str | None = None, batch_size: int = 32, epoch_images: int = 10000,
         verbose: bool = True):
    """What the model costs: parameters, memory, multiply-adds, and measured time on this machine.

    Prints a short report and returns it as a dict with keys name, params, trainable_params, size_mb,
    mult_adds, ms_per_image_single (one image alone), ms_per_image_batch (in a batch of `batch_size`),
    train_step_ms (forward + backward + Adam update of one batch), epoch_images, epoch_seconds (the
    training steps of one epoch over `epoch_images` images). Times are medians of several runs after a
    warm-up; the model's weights are restored afterwards, so the model is not changed.
    """
    shape = _norm_shape(input_shape)
    _dry_run(model, shape)
    name = name or type(model).__name__
    log: list = []
    handles = _macs_hooks(model, log)
    try:
        _dry_run(model, shape)
    finally:
        for h in handles:
            h.remove()
    macs = sum(log)
    total, trainable = _count_params(model), _count_params(model, True)
    m = _measure(model, shape, batch_size)
    b = m["batch"]
    epoch_s = m["ms_step"] / 1e3 / b * epoch_images
    rep = _Report(name=name, input_shape=shape, params=total, trainable_params=trainable,
                  size_mb=total * 4 / 2 ** 20, mult_adds=macs, ms_per_image_single=m["ms_single"],
                  ms_per_image_batch=m["ms_batch"], batch_size=b, train_step_ms=m["ms_step"],
                  epoch_images=epoch_images, epoch_seconds=epoch_s)
    lines = [f"Cost of '{name}' for one {_fmt_shape(shape)} image (measured on this machine)",
             f"  parameters      {total:,}" + (" (all trainable)" if trainable == total else f" ({trainable:,} trainable)")
             + f" = {rep['size_mb']:.2f} MB as float32 numbers",
             f"  multiply-adds   {macs:,} per image ({_human(macs)})",
             f"  prediction      {_fmt_ms(m['ms_single'])} ms for one image alone, {_fmt_ms(m['ms_batch'])} ms per image "
             f"in a batch of {b}",
             f"  training step   {_fmt_ms(m['ms_step'])} ms for a batch of {b} (forward + backward + Adam update)",
             f"  one epoch       about {_fmt_s(epoch_s)} of training steps for {epoch_images:,} images"]
    rep.text = "\n".join(lines)
    if verbose:
        print(rep.text)
    return rep._mark()


def compare(models: dict, input_shape, histories: dict | None = None, batch_size: int = 32, epoch_images: int = 10000):
    """Cost of several models side by side: one table and one bar chart. Returns the rows.

    models: {"name": model, ...}; histories: optional {"name": history from P.train} to add the best
    validation accuracy.
    """
    histories = histories or {}
    rows = _Rows()
    cells = []
    for name, model in models.items():
        r = cost(model, input_shape, name=name, batch_size=batch_size, epoch_images=epoch_images, verbose=False)
        h = histories.get(name)
        acc = h.get("best_val_acc") if h else None
        row = dict(r)
        row["best_val_acc"] = acc
        rows.append(row)
        cells.append([name, f"{r['params']:,}", f"{r['size_mb']:.2f}", _human(r["mult_adds"]),
                      _fmt_ms(r["ms_per_image_single"]), _fmt_ms(r["ms_per_image_batch"]),
                      _fmt_ms(r["train_step_ms"]), _fmt_s(r["epoch_seconds"]),
                      f"{acc:.1f}%" if acc is not None else "-"])
    head = ["model", "parameters", "MB", "mult-adds / image", "ms / image (alone)", f"ms / image (batch {batch_size})",
            f"train step ms ({batch_size})", f"epoch ({epoch_images:,} images)", "best val acc"]
    lines = _table(head, cells, ["<", ">", ">", ">", ">", ">", ">", ">", ">"])
    rows.text = "\n".join(lines)
    print(rows.text)
    # chart
    panels = [("parameters", "params", "{:,.0f}"), ("multiply-adds per image", "mult_adds", None),
              (f"ms per image (batch of {batch_size})", "ms_per_image_batch", "{:.3g} ms")]
    if any(r["best_val_acc"] is not None for r in rows):
        panels.append(("best validation accuracy", "best_val_acc", "{:.1f}%"))
    names = [r["name"] for r in rows]
    fig = _new_figure(3.1 * len(panels), 0.42 * len(rows) + 1.3)
    axes = fig.subplots(1, len(panels), squeeze=False)[0]
    for ax, (label, key, fmt) in zip(axes, panels):
        vals = [r[key] if r[key] is not None else 0 for r in rows]
        ypos = np.arange(len(rows))[::-1]
        colors = [_RUN_COLORS[k % len(_RUN_COLORS)] for k in range(len(rows))]
        ax.barh(ypos, vals, color=colors, height=0.62, edgecolor="white", linewidth=2)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names if ax is axes[0] else [""] * len(names), fontsize=10)
        ax.set_title(label, fontsize=10.5, color=_INK, loc="left")
        top = max(vals) if max(vals) > 0 else 1
        ax.set_xlim(0, top * 1.38)
        for yv, v in zip(ypos, vals):
            txt = _human(v) if fmt is None else fmt.format(v)
            ax.text(v + top * 0.02, yv, txt, va="center", fontsize=9.5, color=_INK)
        ax.set_xticks([])
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#bbbbbb")
        ax.tick_params(axis="y", length=0)
    rows.figure = _show(fig)
    return rows._mark()


# =============================================================================================== training
def _reseed_loader(loader, seed: int) -> None:
    sampler = getattr(loader, "sampler", None)
    gen = getattr(sampler, "generator", None)
    if gen is not None:
        gen.manual_seed(seed)


def _make_optimizer(optimizer, model: nn.Module, lr: float, weight_decay: float):
    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("the model has no trainable parameters (all frozen?)")
    name = str(optimizer).lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"optimizer must be 'adam', 'adamw' or 'sgd' (or a torch.optim optimizer), not {optimizer!r}")


@torch.no_grad()
def evaluate(model: nn.Module, loader, loss_fn=None) -> tuple:
    """(average loss, accuracy in %) of the model on every batch of a loader; the model is not changed."""
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    dev = _model_device(model)
    was = model.training
    model.eval()
    tot, correct, count, n_items = 0.0, 0, 0, 0
    try:
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            tot += float(loss_fn(out, y)) * len(x)
            n_items += len(x)
            correct += int((out.argmax(1) == y).sum())
            count += y.numel()
    finally:
        model.train(was)
    return tot / max(n_items, 1), 100.0 * correct / max(count, 1)


def _check_labels(model, out, loader) -> None:
    """Does the model's output fit the labels? One score per class for each image (or each pixel)."""
    ds = _dataset_of(loader)
    classes = getattr(ds, "classes", None)
    if not isinstance(out, torch.Tensor):
        raise ModelShapeError(f"the model returned a {type(out).__name__}; P.train needs one tensor of class scores")
    _, label = ds[0]
    label_dims = label.dim() if torch.is_tensor(label) else 0
    n = len(classes) if classes is not None else None
    if label_dims == 0 and out.dim() != 2:
        raise ModelShapeError(
            f"The model gives a {_fmt_shape(out.shape[1:])} block of numbers per image, but a classifier must give "
            f"one score per class" + (f" ({n} numbers)" if n else "") + ".\nFix: put nn.Flatten() before the first "
            f"nn.Linear layer, so that the Linear layers work on a flat list of numbers.")
    if n is not None and out.dim() >= 2 and out.shape[1] != n:
        raise ModelShapeError(
            f"The model gives {out.shape[1]} numbers per image, but there are {n} classes: it needs one score per "
            f"class.\nFix: make the last layer produce {n} numbers, e.g. nn.Linear(..., {n}).")


def _history_figure(hists: dict, title: str | None = None) -> _Figure:
    fig = _new_figure(11, 3.9)
    ax_l, ax_a = fig.subplots(1, 2)
    single = len(hists) == 1
    max_e = 1
    best_lines = []
    for k, (name, h) in enumerate(hists.items()):
        ep = h.get("epoch", [])
        if not ep:
            continue
        max_e = max(max_e, ep[-1])
        if single:
            c_tr, c_va = _RUN_COLORS[0], _RUN_COLORS[1]
            lab_tr, lab_va = "training", "validation"
        else:
            c_tr = c_va = _RUN_COLORS[k % len(_RUN_COLORS)]
            lab_tr, lab_va = f"{name} (training)", f"{name} (validation)"
        ls_tr = "-" if single else "--"
        ax_l.plot(ep, h["train_loss"], ls_tr, color=c_tr, lw=2, marker="o", ms=5, label=lab_tr)
        ax_a.plot(ep, h["train_acc"], ls_tr, color=c_tr, lw=2, marker="o", ms=5, label=lab_tr)
        if h.get("val_loss") and h["val_loss"][0] is not None:
            ax_l.plot(ep, h["val_loss"], "-", color=c_va, lw=2, marker="s", ms=5, label=lab_va)
            ax_a.plot(ep, h["val_acc"], "-", color=c_va, lw=2, marker="s", ms=5, label=lab_va)
            be, bv = h.get("best_epoch"), h.get("best_val_acc")
            if be is not None:
                ax_a.plot([be], [bv], marker="*", ms=17, color=c_va, markeredgecolor="white", zorder=5)
                who = "" if single else f"{name}: "
                best_lines.append(f"★ {who}best validation accuracy {bv:.1f}% (epoch {be})")
    for ax, lab in ((ax_l, "loss (lower is better)"), (ax_a, "accuracy in % (higher is better)")):
        ax.set_xlabel("epoch", fontsize=10)
        ax.set_title(lab, fontsize=10.5, loc="left", color=_INK)
        ax.set_xlim(0.6, max_e + 0.4)
        ax.set_xticks(range(1, max_e + 1) if max_e <= 20 else np.linspace(1, max_e, 10).round().astype(int))
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax_l.legend(fontsize=9, frameon=False, loc="upper right")
    # the accuracy curves rise, so the lower right corner is free for the best values
    lo, hi = ax_a.get_ylim()
    span = hi - lo
    ax_a.set_ylim(lo - span * (0.08 + 0.09 * len(best_lines)), hi + span * 0.05)
    for k, line in enumerate(best_lines):
        ax_a.text(0.98, 0.04 + 0.075 * (len(best_lines) - 1 - k), line, transform=ax_a.transAxes, ha="right",
                  va="bottom", fontsize=9.5, color=_INK)
    if title:
        fig.suptitle(title, fontsize=12, color=_INK)
    return fig


class _Live:
    """Epoch lines and curves that update in place under a notebook cell; plain prints elsewhere."""

    def __init__(self, enabled: bool):
        self.notebook = _shell() is not None
        self.enabled = enabled and self.notebook
        self.lines: list = []
        if self.enabled:
            from IPython.display import Pretty, display
            self.text = display(Pretty(""), display_id=True)
            self.plot = display(Pretty("(the curves appear here after the first epoch)"), display_id=True)

    def log(self, line: str) -> None:
        self.lines.append(line)
        if self.enabled:
            from IPython.display import Pretty
            self.text.update(Pretty("\n".join(self.lines)))
        else:
            print(line)

    def draw(self, hist, final: bool = False) -> None:
        if self.enabled:
            from IPython.display import Image
            fig = _history_figure({hist["name"]: hist}, hist["name"])
            self.plot.update(Image(data=_png(fig)))
        elif final and self.notebook:
            _show(_history_figure({hist["name"]: hist}, hist["name"]))


def train(model: nn.Module, train_loader, val_loader=None, epochs: int = 3, lr: float = 1e-3, optimizer="adam",
          max_minutes: float | None = None, live: bool = True, seed: int = 0, name: str | None = None,
          loss_fn=None, weight_decay: float = 0.0):
    """Train a classifier and draw its training and validation curves. Returns a History (a dict).

    Every epoch shows the model all training images once (in batches), adjusting the weights after
    each batch, then measures loss and accuracy on the validation images. One line per epoch, and
    the loss and accuracy curves update after every epoch (live=False draws them once at the end).

    epochs: passes over the training data. lr: learning rate, the size of each weight update.
    optimizer: "adam" (default), "adamw", "sgd" (with momentum 0.9) or a torch.optim optimizer.
    max_minutes: stop early after this many minutes (the last epoch is then validated as it is).
    seed: fixes the shuffling order and dropout, so a run can be repeated exactly (the initial weights
    come from torch.manual_seed before the model was built). loss_fn: default cross-entropy.
    Works for image classification and also per-pixel classification (accuracy over pixels).
    """
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    name = name or type(model).__name__
    dev = _device()
    model.to(dev)
    ds = _dataset_of(train_loader)
    x0, _ = ds[0]
    in_shape = tuple(x0.shape)
    out0 = _dry_run(model, in_shape)
    _check_labels(model, out0, train_loader)
    _seed(seed)
    _reseed_loader(train_loader, seed)
    opt = _make_optimizer(optimizer, model, lr, weight_decay)
    opt_name = optimizer if isinstance(optimizer, str) else type(optimizer).__name__
    n_train = len(ds)
    hist = History(name=name, epoch=[], train_loss=[], train_acc=[], val_loss=[], val_acc=[], seconds=[],
                   total_seconds=0.0, best_val_acc=None, best_epoch=None, lr=lr, optimizer=opt_name,
                   batch_size=getattr(train_loader, "batch_size", None), n_train=n_train, params=_count_params(model),
                   input_shape=in_shape, stopped_early=False)
    live_out = _Live(live)
    live_out.log(f"training '{name}': {epochs} epoch(s) over {n_train:,} images in batches of {hist['batch_size']}, "
                 f"{opt_name} with learning rate {lr}")
    t_start = time.time()
    budget = max_minutes * 60 if max_minutes is not None else None
    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        tot, correct, count, n_items = 0.0, 0, 0, 0
        stop_reason = None
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lv = float(loss.detach())
            if not math.isfinite(lv):
                stop_reason = (f"the loss became {lv} (not a number) in epoch {epoch}: training stopped. The learning "
                               f"rate {lr} is too high for this model; try one 10 times smaller.")
                break
            tot += lv * len(x)
            n_items += len(x)
            correct += int((out.detach().argmax(1) == y).sum())
            count += y.numel()
            if budget is not None and time.time() - t_start > budget:
                stop_reason = f"time limit of {max_minutes} min reached during epoch {epoch}: stopping after it"
                break
        tl = tot / max(n_items, 1) if n_items else float("nan")
        ta = 100.0 * correct / max(count, 1)
        if val_loader is not None:
            vl, va = evaluate(model, val_loader, loss_fn)
        else:
            vl, va = None, None
        sec = time.time() - t_ep
        hist["epoch"].append(epoch)
        hist["train_loss"].append(tl)
        hist["train_acc"].append(ta)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        hist["seconds"].append(sec)
        if va is not None and (hist["best_val_acc"] is None or va > hist["best_val_acc"]):
            hist["best_val_acc"], hist["best_epoch"] = va, epoch
        line = f"epoch {epoch:>2}/{epochs}   training: loss {tl:.3f}, accuracy {ta:5.1f}%"
        if va is not None:
            line += f"   |   validation: loss {vl:.3f}, accuracy {va:5.1f}%"
        line += f"   |   {sec:.1f} s"
        live_out.log(line)
        live_out.draw(hist)
        if stop_reason:
            live_out.log(stop_reason)
            hist["stopped_early"] = True
            break
    hist["total_seconds"] = time.time() - t_start
    if hist["best_val_acc"] is not None:
        live_out.log(f"done in {hist['total_seconds']:.1f} s; best validation accuracy {hist['best_val_acc']:.1f}% "
                     f"(epoch {hist['best_epoch']})")
    else:
        live_out.log(f"done in {hist['total_seconds']:.1f} s")
    live_out.draw(hist, final=True)
    return hist._mark()


def plot_history(history, title: str | None = None):
    """Draw the loss and accuracy curves of one History, or overlay several: plot_history({"A": h1, "B": h2})."""
    if isinstance(history, History) or (isinstance(history, dict) and "epoch" in history):
        hists = {history.get("name", "run"): history}
    else:
        hists = dict(history)
        if len(hists) > len(_RUN_COLORS):
            print(f"showing the first {len(_RUN_COLORS)} runs only")
            hists = dict(list(hists.items())[:len(_RUN_COLORS)])
    fig = _history_figure(hists, title or ("comparison" if len(hists) > 1 else None))
    return _show(fig)


@torch.no_grad()
def predict(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Predicted class index of one image (C, H, W) -> 0-dim tensor, or of a batch (B, C, H, W) -> (B,)."""
    dev = _model_device(model)
    single = images.dim() == 3
    x = images[None] if single else images
    was = model.training
    model.eval()
    try:
        out = model(x.to(dev)).argmax(1).cpu()
    finally:
        model.train(was)
    return out[0] if single else out


@torch.no_grad()
def _predict_loader(model, loader, max_items=None):
    dev = _model_device(model)
    was = model.training
    model.eval()
    xs, ys, ps, confs = [], [], [], []
    n = 0
    try:
        for x, y in loader:
            out = model(x.to(dev)).float()
            prob = out.softmax(1)
            conf, pred = prob.max(1)
            xs.append(x.cpu())
            ys.append(y.cpu())
            ps.append(pred.cpu())
            confs.append(conf.cpu())
            n += len(x)
            if max_items is not None and n >= max_items:
                break
    finally:
        model.train(was)
    return torch.cat(xs), torch.cat(ys), torch.cat(ps), torch.cat(confs)


def show_predictions(model: nn.Module, loader, classes: Sequence[str] | None = None, n: int = 16,
                     only_wrong: bool = False, cols: int = 8, title: str | None = None):
    """Images with the model's answer: correct ones in green, wrong ones marked WRONG in red with the true label.

    only_wrong=True shows only mistakes (from the whole loader). Returns the figure.
    """
    ds = _dataset_of(loader)
    classes = classes if classes is not None else getattr(ds, "classes", None)
    x, y, p, c = _predict_loader(model, loader, None if only_wrong else n)
    idx = torch.arange(len(y))
    if only_wrong:
        idx = idx[p != y]
    idx = idx[:n]
    k = len(idx)
    if k == 0:
        print("no mistakes to show")
        return None
    cols, rows = _grid(k, cols)
    fig = _new_figure(1.45 * cols, 1.75 * rows + 0.4)
    axes = fig.subplots(rows, cols, squeeze=False)
    right = 0
    for a, ax in enumerate(axes.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        if a >= k:
            ax.axis("off")
            continue
        i = int(idx[a])
        arr = _display_image(x[i], ds)
        ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
        pred, true = int(p[i]), int(y[i])
        pn = classes[pred] if classes is not None else str(pred)
        tn = classes[true] if classes is not None else str(true)
        ok = pred == true
        right += ok
        color = "#008300" if ok else "#d03b3b"
        if ok:
            ax.set_title(f"{pn}  ({100 * float(c[i]):.0f}%)", fontsize=9.5, color=color)
        else:
            ax.set_title(f"WRONG: {pn}\ntrue: {tn}", fontsize=9.5, color=color, fontweight="bold")
        for side in ax.spines.values():
            side.set_edgecolor(color)
            side.set_linewidth(1.2 if ok else 3)
    head = title or (f"{k} mistakes of the model" if only_wrong else f"the model's answers: {right} of {k} correct "
                                                                      f"(percent = how sure it is)")
    fig.suptitle(head, fontsize=11, color=_INK)
    return _show(fig)


def confusion(model: nn.Module, loader, classes: Sequence[str] | None = None, title: str | None = None):
    """Confusion matrix over a loader: row = true class, column = predicted class. Prints the most common mistakes.

    Returns the matrix as a numpy array of counts.
    """
    ds = _dataset_of(loader)
    classes = list(classes if classes is not None else getattr(ds, "classes", None) or [])
    _, y, p, _ = _predict_loader(model, loader)
    k = max(len(classes), int(y.max()) + 1, int(p.max()) + 1)
    if not classes:
        classes = [str(i) for i in range(k)]
    m = np.zeros((k, k), dtype=int)
    for t, q in zip(y.tolist(), p.tolist()):
        m[t, q] += 1
    errors = []
    for i in range(k):
        for j in range(k):
            if i != j and m[i, j] > 0:
                errors.append((m[i, j], i, j))
    errors.sort(reverse=True)
    total = int(m.sum())
    wrong = total - int(np.trace(m))
    lines = [f"{wrong} of {total} validation images wrong ({100 * wrong / max(total, 1):.1f}%). Most common mistakes:"]
    for cnt, i, j in errors[:5]:
        lines.append(f"  true {classes[i]:>10} -> predicted {classes[j]:<10} {cnt} times")
    print("\n".join(lines))
    side = 0.48 * k + 1.6
    fig = _new_figure(side + 0.6, side)
    ax = fig.subplots()
    off = m.astype(float).copy()
    np.fill_diagonal(off, 0)
    vmax = off.max() if off.max() > 0 else 1
    off[off == 0] = np.nan                       # no mistakes: white; the diagonal is drawn grey below
    import matplotlib as mpl
    cmap = mpl.colormaps["Oranges"].copy()
    cmap.set_bad("white")
    ax.imshow(off, cmap=cmap, vmin=0, vmax=vmax * 1.15)
    for i in range(k):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, facecolor="#e6e5e0", edgecolor="none"))
    for i in range(k + 1):
        ax.axhline(i - 0.5, color="#f0efec", lw=0.8)
        ax.axvline(i - 0.5, color="#f0efec", lw=0.8)
    for i in range(k):
        for j in range(k):
            v = m[i, j]
            if v == 0:
                continue
            strong = i != j and v > 0.55 * vmax
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9 if k <= 12 else 7,
                    color="white" if strong else _INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(classes, rotation=45 if max(len(c) for c in classes) > 3 else 0, ha="right"
                       if max(len(c) for c in classes) > 3 else "center", fontsize=9.5)
    ax.set_yticklabels(classes, fontsize=9.5)
    ax.set_xlabel("predicted class", fontsize=10)
    ax.set_ylabel("true class", fontsize=10)
    ax.set_title(title or "confusion matrix: grey diagonal = correct, orange = mistakes", fontsize=10.5, color=_INK)
    for s in ax.spines.values():
        s.set_visible(False)
    _show(fig)
    return m


def _find_layer(model: nn.Module, layer, want=nn.modules.conv._ConvNd):
    """(name, module) for layer given as a name, a module, or None (the first layer of type `want`)."""
    named = list(model.named_modules())
    if layer is None:
        for name, m in named:
            if isinstance(m, want):
                return name, m
        raise ValueError(f"the model has no {want.__name__} layer")
    if isinstance(layer, nn.Module):
        for name, m in named:
            if m is layer:
                return name, m
        raise ValueError("that layer is not part of this model")
    key = str(layer)
    for name, m in named:
        if name == key:
            return name, m
    choices = ", ".join(repr(n) for n, m in named if isinstance(m, want))
    raise ValueError(f"no layer named {key!r}; layers of that kind: {choices} (see the 'layer' column of P.summary)")


def show_filters(model: nn.Module, layer=None, n: int = 64, title: str | None = None):
    """Show the kernels (filters) a convolution layer has learned; default: the first convolution.

    layer: its name as in P.summary (e.g. "0" or "features.0") or the layer itself. Grey images: one
    picture per filter, blue = negative weight, red = positive. Colour inputs: each filter as a colour patch.
    """
    name, conv = _find_layer(model, layer)
    w = conv.weight.detach().float().cpu()
    k = min(n, w.shape[0])
    cols, rows = _grid(k, 8 if k > 8 else k)
    size = 1.0 if w.shape[-1] <= 5 else 1.2
    fig = _new_figure(size * cols + 0.9, size * rows + 0.6)
    axes = fig.subplots(rows, cols, squeeze=False)
    colour = w.shape[1] == 3
    lim = float(w[:k].abs().max())
    im = None
    for a, ax in enumerate(axes.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        if a >= k:
            ax.axis("off")
            continue
        if colour:
            f = w[a]
            f = (f - f.min()) / (f.max() - f.min()).clamp_min(1e-8)
            ax.imshow(f.permute(1, 2, 0).numpy(), interpolation="nearest")
        else:
            im = ax.imshow(w[a, 0].numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_title(f"filter {a}", fontsize=9, color=_INK2)
    kk = "x".join(str(s) for s in conv.kernel_size)
    extra = "" if w.shape[1] in (1, 3) else f" (input channel 0 of {w.shape[1]})"
    fig.suptitle(title or f"layer '{name}': {w.shape[0]} filters of {kk}{extra}"
                 + ("" if colour else ": red = positive weight, blue = negative"), fontsize=10.5, color=_INK)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.8, aspect=25)
    return _show(fig)


def show_feature_maps(model: nn.Module, image: torch.Tensor, layer=None, n: int = 8, title: str | None = None):
    """What the convolution layers make of one image: the feature maps they output.

    image: one image (C, H, W) as it comes from the data, e.g. ``image, label = val_loader.dataset[0]``.
    layer: None = one row per convolution layer (first `n` maps each); or one layer's name / the layer
    itself = up to 2 x n of its maps. Bright = the filter found its pattern there.
    """
    if isinstance(image, (tuple, list)):
        image = image[0]
    if image.dim() == 4:
        image = image[0]
    if layer is None:
        targets = [(nm, m) for nm, m in model.named_modules() if isinstance(m, nn.modules.conv._ConvNd)]
        if not targets:
            raise ValueError("the model has no convolution layer")
        per = n
    else:
        targets = [_find_layer(model, layer, nn.Module)]
        per = 2 * n
    outs = {}
    handles = []
    for nm, m in targets:
        def hook(mod, inp, out, nm=nm):
            outs[nm] = out.detach().float().cpu()
        handles.append(m.register_forward_hook(hook))
    was = model.training
    model.eval()
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            _forward_checked(model, image[None].to(_model_device(model)))
    finally:
        model.train(was)
        for h in handles:
            h.remove()
    rows = []
    for nm, m in targets:
        o = outs.get(nm)
        if o is None or o.dim() != 4:
            continue
        rows.append((nm, o[0]))
    if layer is not None:
        nm, o = rows[0]
        chunks = [(nm, o[:n]), (nm + " (continued)", o[n:2 * n])] if o.shape[0] > n else [(nm, o[:n])]
        rows = [r for r in chunks if r[1].shape[0] > 0]
    ncols = 1 + min(n, max(r[1].shape[0] for r in rows))
    fig = _new_figure(1.35 * ncols + 0.3, 1.5 * len(rows) + 0.5)
    axes = fig.subplots(len(rows), ncols, squeeze=False)
    ds_like = type("D", (), {"mean": None, "std": None})()
    img = _display_image(image, ds_like)
    for r, (nm, o) in enumerate(rows):
        ax0 = axes[r, 0]
        ax0.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_ylabel(f"layer '{nm}'\n{o.shape[0]} maps of {o.shape[1]}x{o.shape[2]}", fontsize=9, color=_INK)
        if r == 0:
            ax0.set_title("input", fontsize=9.5, color=_INK)
        for c in range(1, ncols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            k = c - 1
            if k >= o.shape[0]:
                ax.axis("off")
                continue
            fmap = o[k].clamp(min=0)               # what passes on after ReLU: only the positive responses
            top = float(fmap.max())
            ax.imshow(fmap.numpy(), cmap="gray", vmin=0, vmax=top if top > 0 else 1, interpolation="nearest")
            label = f"map {k if 'continued' not in nm else k + n}"
            ax.set_title(label if top > 0 else label + " (empty)", fontsize=9, color=_INK2)
    fig.suptitle(title or "feature maps: where each filter found its pattern in this image (bright = strong match, "
                          "black = nothing)", fontsize=10.5, color=_INK)
    return _show(fig)


# =============================================================================================== remembering versions
_BOARD: "OrderedDict[str, dict]" = OrderedDict()


def remember(name: str, model: nn.Module, history: dict | None = None, input_shape=None, verbose: bool = True) -> dict:
    """Keep a record of one version of a model (accuracy, size, time) for P.leaderboard().

    Re-using a name replaces that record, so re-running a cell does not add duplicates.
    input_shape: only needed without a history (P.train stores it in the history).
    """
    shape = input_shape or (history or {}).get("input_shape")
    if shape is None:
        raise ValueError("give input_shape=(C, H, W) or the history returned by P.train")
    shape = _norm_shape(shape)
    log: list = []
    handles = _macs_hooks(model, log)
    try:
        _dry_run(model, shape)
    finally:
        for h in handles:
            h.remove()
    m = _measure(model, shape, 32, train_step=False)
    h = history or {}
    secs = h.get("seconds") or []
    rec = {"name": name, "best_val_acc": h.get("best_val_acc"), "params": _count_params(model), "mult_adds": sum(log),
           "seconds_per_epoch": (sum(secs) / len(secs)) if secs else None, "epochs": len(h.get("epoch", [])),
           "ms_per_image": m["ms_batch"], "lr": h.get("lr"), "batch_size": h.get("batch_size"),
           "n_train": h.get("n_train"), "model": model, "history": history}
    # same name but a different network or run settings: keep both (re-running a cell replaces its own row)
    sig = (rec["params"], rec["mult_adds"], rec["epochs"], rec["lr"], rec["batch_size"], rec["n_train"])
    base, k = name, 2
    while name in _BOARD and _BOARD[name].get("sig") != sig:
        name = f"{base} ({k})"
        k += 1
    rec["name"], rec["sig"] = name, sig
    _BOARD[name] = rec
    acc = f"{rec['best_val_acc']:.1f}% best validation accuracy, " if rec["best_val_acc"] is not None else ""
    out = _Report({k: v for k, v in rec.items() if k != "sig"})
    out.text = (f"remembered '{name}': {acc}{rec['params']:,} parameters"
                + (f", {rec['seconds_per_epoch']:.1f} s per epoch" if rec["seconds_per_epoch"] else "")
                + f" ({len(_BOARD)} version(s) so far)")
    if verbose:
        print(out.text)
    return out._mark()


def forget(name: str | None = None) -> None:
    """Remove one remembered version, or all of them with no name."""
    if name is None:
        _BOARD.clear()
    else:
        _BOARD.pop(name, None)


def leaderboard(sort: str = "accuracy", chart: bool = False):
    """Every remembered version in one table, best validation accuracy first (sort="params" or "time" also work).

    chart=True also draws accuracy against the number of parameters. Returns the rows.
    """
    recs = list(_BOARD.values())
    if not recs:
        print("nothing remembered yet: call P.remember(name, model, history) after training")
        return _Rows()._mark()
    if sort == "params":
        recs.sort(key=lambda r: r["params"])
    elif sort == "time":
        recs.sort(key=lambda r: r["seconds_per_epoch"] or 0)
    else:
        recs.sort(key=lambda r: -(r["best_val_acc"] or 0))
    rows = _Rows()
    cells = []
    for k, r in enumerate(recs, 1):
        row = {key: v for key, v in r.items() if key not in ("model", "history", "sig")}
        rows.append(row)
        cells.append([str(k), r["name"], f"{r['best_val_acc']:.1f}%" if r["best_val_acc"] is not None else "-",
                      f"{r['params']:,}", _human(r["mult_adds"]),
                      f"{r['seconds_per_epoch']:.1f}" if r["seconds_per_epoch"] else "-", str(r["epochs"] or "-"),
                      f"{r['lr']:g}" if r["lr"] is not None else "-", str(r["batch_size"] or "-"),
                      f"{r['n_train']:,}" if r["n_train"] else "-"])
    head = ["#", "version", "best val acc", "parameters", "mult-adds / image", "s / epoch", "epochs",
            "learning rate", "batch", "training images"]
    rows.text = "\n".join(_table(head, cells, [">", "<", ">", ">", ">", ">", ">", ">", ">", ">"]))
    print(rows.text)
    if chart and len(recs) > 1:
        fig = _new_figure(7.5, 4.2)
        ax = fig.subplots()
        for r in recs:
            if r["best_val_acc"] is None:
                continue
            ax.scatter(r["params"], r["best_val_acc"], s=70, color=_RUN_COLORS[0], edgecolor="white", zorder=3)
            ax.annotate(r["name"], (r["params"], r["best_val_acc"]), textcoords="offset points", xytext=(6, 4),
                        fontsize=9, color=_INK)
        ax.set_xscale("log")
        ax.set_xlabel("parameters (log scale)", fontsize=10)
        ax.set_ylabel("best validation accuracy (%)", fontsize=10)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title("every version so far: accuracy against size", fontsize=10.5, loc="left", color=_INK)
        _show(fig)
    return rows._mark()
