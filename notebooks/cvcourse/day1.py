"""Day-1 (convnets, training, architectures, transfer learning) shared PyTorch code.

Everything the D1-02 / D1-03 / D1-04 notebooks share lives here so the notebooks stay
about the *ideas*. Reference implementations of every exercise are included (the
notebooks fall back to them if a student skipped an exercise, and the checkers in
`tests_day1` compare against them).

* Pet subset as in-memory uint8 tensors (`pet_tensors`) + `TensorImageDataset`: decoding
  2 500 JPEGs once and caching them makes every epoch on CPU ~10x faster.
* `PetNet`            : the small from-scratch convnet of D1-02
* `recipe_transforms`, `train_recipe`: the D1-02 training recipe, reused in D1-03
* `make_deep_net`, `first_layer_grad_norm`: the plain-vs-residual gradient-flow experiment
* reference blocks    : `InceptionBlock`, `BasicBlock`, `Bottleneck`, `DepthwiseSeparable`,
  `SqueezeExcite`, `DenseLayer`, `MiniResNet`
* reference functions : `mixup`, `warmup_cosine`, `label_smoothing_ce`, `train_step`,
  `freeze`, `param_groups_with_lrs`, `gradcam`, `extract_features`
* `ARCHITECTURES`     : published ImageNet numbers for the D1-03 timeline / scatter
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import torchkit

IMAGENET_MEAN = torchkit.IMAGENET_MEAN
IMAGENET_STD = torchkit.IMAGENET_STD


# =========================================================================== CPU threads
def pick_cpu_threads(max_threads: int = 4, verbose: bool = True) -> int:
    """Choose torch's intra-op thread count for this CPU *right now*.

    On a busy or shared CPU, many threads fight for the cores and small ops (augmentation,
    small batches) get 3-30x slower than with a single thread. Time a small conv workload
    with 1 thread and with `max_threads` and keep the faster setting. (Set the environment
    variable OMP_WAIT_POLICY=PASSIVE before importing torch to make idle threads sleep.)
    """
    import os

    n = max(1, min(max_threads, os.cpu_count() or 1))
    if n == 1:
        torch.set_num_threads(1)
        return 1
    net = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
    x = torch.randn(16, 3, 64, 64)
    best = None
    for t in (1, n):
        torch.set_num_threads(t)
        with torch.no_grad():
            net(x)
            t0 = time.perf_counter()
            for _ in range(5):
                net(x)
            dt = time.perf_counter() - t0
        if best is None or dt < best[1] * 0.8:
            best = (t, dt)
    torch.set_num_threads(best[0])
    if verbose:
        print(f"using {best[0]} CPU thread(s) for torch (the faster of 1 and {n} on this machine right now)")
    return best[0]


# =========================================================================== data
def _load_resized(path, size: int) -> torch.Tensor:
    """Decode a JPEG at reduced resolution (PIL draft mode), resize the short side to `size`,
    centre-crop to (size, size). Returns uint8 (3, size, size)."""
    from PIL import Image

    img = Image.open(path)
    img.draft("RGB", (size * 2, size * 2))         # JPEG DCT scaling: decode at ~1/2-1/8 resolution, fast
    img = img.convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BILINEAR)
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    img = img.crop((l, t, l + size, t + size))
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).contiguous()


def pet_tensors(size: int = 144, subset=(2000, 500), cache: bool = True, verbose: bool = True):
    """The class-balanced Pet subset of `torchkit.get_pet_datasets(subset=...)` as uint8 tensors.

    Returns dict(x_train (N,3,S,S) uint8, y_train (N,) int64, x_val, y_val, classes, paths_train, paths_val).
    Cached in data/checkpoints/d1_pet_{size}_{n_train}_{n_val}.pt so the decode happens once.
    """
    n_tr, n_va = subset if isinstance(subset, (tuple, list)) else (subset, subset // 4)
    path = torchkit.CHECKPOINTS / f"d1_pet_{size}_{n_tr}_{n_va}.pt"
    if cache and path.exists():
        d = torch.load(path, weights_only=False)
        if verbose:
            print(f"Pet subset loaded from cache {path.name}: {len(d['y_train'])} train / {len(d['y_val'])} val, {size} px")
        return d
    t0 = time.time()
    tr, va, classes = torchkit.get_pet_datasets(size=size, subset=(n_tr, n_va))
    d = {"classes": list(classes)}
    for split, ds in (("train", tr), ("val", va)):
        paths = [ds.image_path(i) for i in range(len(ds))]
        with ThreadPoolExecutor(max_workers=8) as ex:        # PIL releases the GIL while decoding
            d[f"x_{split}"] = torch.stack(list(ex.map(lambda p: _load_resized(p, size), paths)))
        d[f"y_{split}"] = torch.tensor([ds.label(i) for i in range(len(ds))], dtype=torch.long)
        d[f"paths_{split}"] = [str(p) for p in paths]
    if cache:
        torch.save(d, path)
    if verbose:
        print(f"Pet subset decoded in {time.time() - t0:.0f} s: {n_tr} train / {n_va} val at {size} px (cached to {path.name})")
    return d


class TensorImageDataset(Dataset):
    """(uint8 image tensor, label) pairs with an optional transforms.v2 pipeline applied on the fly."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, transform=None):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[i]


def eval_transform(size: int = 128):
    """Resize (short side) + centre crop + float + ImageNet normalisation (no randomness)."""
    from torchvision.transforms import v2

    return v2.Compose([v2.Resize(size, antialias=True), v2.CenterCrop(size), v2.ToDtype(torch.float32, scale=True),
                       v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def plain_transform(size: int = 128):
    """The 'no augmentation' training pipeline: identical to evaluation."""
    return eval_transform(size)


def recipe_transforms(size: int = 128, strength: float = 1.0):
    """The D1-02 reference augmentation pipeline (input uint8 tensor (3, S, S) with S >= size)."""
    from torchvision.transforms import v2

    return v2.Compose([
        v2.RandomResizedCrop(size, scale=(max(0.08, 1 - 0.65 * strength), 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([v2.TrivialAugmentWide()], p=min(1.0, strength)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        v2.RandomErasing(p=0.25 * strength),
    ])


def loaders(data: dict, train_tf, eval_tf=None, batch_size: int = 64, n_train: int | None = None,
            seed: int = 0):
    """DataLoaders over `pet_tensors()` output. n_train limits the training set (class-balanced prefix)."""
    eval_tf = eval_tf or eval_transform()
    xt, yt = data["x_train"], data["y_train"]
    if n_train is not None and n_train < len(yt):
        idx = torchkit._balanced_subset(yt.tolist(), n_train, seed)
        xt, yt = xt[idx], yt[idx]
    g = torch.Generator().manual_seed(seed)
    tr = DataLoader(TensorImageDataset(xt, yt, train_tf), batch_size=batch_size, shuffle=True, generator=g,
                    drop_last=True)
    va = DataLoader(TensorImageDataset(data["x_val"], data["y_val"], eval_tf), batch_size=128)
    return tr, va


# =========================================================================== D1-02 models
def conv_bn_relu(c_in: int, c_out: int, k: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, k, stride, k // 2, bias=False), nn.BatchNorm2d(c_out),
                         nn.ReLU(inplace=True))


class PetNet(nn.Module):
    """Small from-scratch convnet for 128 px Pet images (~0.4 M parameters, ~65 M MACs).

    conv3x3/2 (32) -> conv3x3/2 (64) -> conv3x3 (64) -> conv3x3/2 (128) -> conv3x3/2 (256) -> GAP -> dropout -> fc
    """

    def __init__(self, num_classes: int = 37, width: int = 32, dropout: float = 0.0):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            conv_bn_relu(3, w, 3, 2),           # 64
            conv_bn_relu(w, 2 * w, 3, 2),       # 32
            conv_bn_relu(2 * w, 2 * w, 3, 1),   # 32
            conv_bn_relu(2 * w, 4 * w, 3, 2),   # 16
            conv_bn_relu(4 * w, 8 * w, 3, 2),   # 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(8 * w, num_classes)

    def forward(self, x):
        return self.fc(self.drop(self.pool(self.features(x)).flatten(1)))


class FashionNet(nn.Module):
    """The D1-01 tiny convnet for 28x28 grayscale: conv32-pool-conv64-pool-conv64-GAP-dropout-fc10.
    Used in D1-02 as a fast proxy for the LR range test and the random search."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(conv_bn_relu(1, 32), nn.MaxPool2d(2), conv_bn_relu(32, 64), nn.MaxPool2d(2),
                                      conv_bn_relu(64, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, 10))

    def forward(self, x):
        return self.head(self.features(x))


def fashion_subsets(n_train: int = 4000, n_val: int = 2000, seed: int = 0):
    """In-memory FashionMNIST subsets (normalised tensors): (TensorDataset train, TensorDataset val)."""
    from torch.utils.data import TensorDataset

    tr, te, _ = torchkit.get_fashion_mnist(as_tensors=True)
    g = torch.Generator().manual_seed(seed)
    i_tr = torch.randperm(len(tr), generator=g)[:n_train]
    i_va = torch.randperm(len(te), generator=g)[:n_val]
    return (TensorDataset(tr.tensors[0][i_tr], tr.tensors[1][i_tr]),
            TensorDataset(te.tensors[0][i_va], te.tensors[1][i_va]))


# =========================================================================== D1-02 reference exercises
def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2, num_classes: int = 37, lam: float | None = None,
          perm: torch.Tensor | None = None):
    """Mixup: x' = lam x + (1-lam) x[perm], y' = lam onehot(y) + (1-lam) onehot(y[perm]).

    lam ~ Beta(alpha, alpha) unless given; perm = random permutation of the batch unless given.
    Returns (x_mixed, y_soft (B, num_classes) float).
    """
    if lam is None:
        lam = float(np.random.beta(alpha, alpha))
    if perm is None:
        perm = torch.randperm(x.shape[0])
    y1 = F.one_hot(y, num_classes).float()
    x_mix = lam * x + (1 - lam) * x[perm]
    y_mix = lam * y1 + (1 - lam) * y1[perm]
    return x_mix, y_mix


def warmup_cosine(step: int, total: int, warmup: int, lr_max: float, lr_min: float = 0.0) -> float:
    """Linear warm-up from ~0 to lr_max over `warmup` steps, then cosine decay to lr_min at `total`."""
    if step < warmup:
        return lr_max * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(progress, 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def label_smoothing_ce(logits: torch.Tensor, target: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    """Cross-entropy against (1 - eps) * onehot + eps / C (matches F.cross_entropy(label_smoothing=eps))."""
    C = logits.shape[-1]
    logp = F.log_softmax(logits, dim=-1)
    q = torch.full_like(logp, eps / C)
    q.scatter_(-1, target.unsqueeze(-1), 1 - eps + eps / C)
    return -(q * logp).sum(-1).mean()


def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn, optimizer) -> float:
    """One optimisation step. Returns the loss as a float."""
    model.train()
    logits = model(x)
    loss = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def overfit_one_batch(model: nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int = 100, lr: float = 1e-2) -> list:
    """Train on the same batch `steps` times with AdamW; return the losses (should go to ~0)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    return [train_step(model, x, y, F.cross_entropy, opt) for _ in range(steps)]


def soft_target_ce(logits: torch.Tensor, target: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """Cross-entropy that accepts integer targets (B,) or soft targets (B, C) (Mixup / CutMix)."""
    return F.cross_entropy(logits, target, label_smoothing=label_smoothing)


def make_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_steps: int, schedule_fn=None):
    """LambdaLR that multiplies each group's initial LR by schedule_fn(step, total, warmup, 1.0)."""
    fn = schedule_fn or warmup_cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: fn(s, total_steps, warmup_steps, 1.0))


def train_recipe(model: nn.Module, data: dict, epochs: int = 6, lr: float = 3e-3, weight_decay: float = 0.05,
                 label_smoothing: float = 0.1, strength: float = 1.0, batch_size: int = 64,
                 warmup_frac: float = 0.15, max_minutes: float | None = None, checkpoint=None,
                 size: int = 128, n_train: int | None = None, verbose: bool = True, device=None,
                 train_tf=None, schedule_fn=None, smoothing_loss=None):
    """The full D1-02 recipe: augmentation + AdamW (decoupled wd) + warm-up/cosine + label smoothing.

    train_tf / schedule_fn(step, total, warmup, lr_max) / smoothing_loss(logits, y, eps) let the
    notebook plug in the students' own pipeline, schedule and loss (defaults: the references here).
    """
    if train_tf is None:
        train_tf = recipe_transforms(size, strength) if strength > 0 else plain_transform(size)
    tr, va = loaders(data, train_tf, eval_transform(size), batch_size, n_train)
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)        # no weight decay on BN / biases
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    total = epochs * len(tr)
    sched = make_warmup_cosine_scheduler(opt, total, max(1, int(warmup_frac * total)), schedule_fn)
    if smoothing_loss is None:
        loss_fn = lambda out, y: soft_target_ce(out, y, label_smoothing)  # noqa: E731
    else:
        loss_fn = lambda out, y: smoothing_loss(out, y, label_smoothing)  # noqa: E731
    return torchkit.fit(model, tr, va, loss_fn, opt, epochs, scheduler=sched, max_minutes=max_minutes,
                        log_every=0, checkpoint=checkpoint, verbose=verbose, device=device)


def train_plain(model: nn.Module, data: dict, epochs: int = 6, lr: float = 3e-3, batch_size: int = 64,
                max_minutes: float | None = None, checkpoint=None, size: int = 128, verbose: bool = True,
                device=None):
    """Baseline: no augmentation, Adam with a constant LR, no weight decay, no smoothing."""
    tr, va = loaders(data, plain_transform(size), eval_transform(size), batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return torchkit.fit(model, tr, va, nn.CrossEntropyLoss(), opt, epochs, max_minutes=max_minutes, log_every=0,
                        checkpoint=checkpoint, verbose=verbose, device=device)


def load_or_train(model: nn.Module, name: str, train_fn, retrain: bool = False):
    """If data/checkpoints/<name>.pt exists, load it (weights + history); otherwise call train_fn(checkpoint_path).

    Returns the history dict. Delete the file (or pass retrain=True) to train again.
    """
    path = torchkit.CHECKPOINTS / f"{name}.pt"
    if path.exists() and not retrain:
        extras = torchkit.load_checkpoint(model, path)
        hist = extras.get("history", {})
        print(f"loaded {path.name} (trained earlier: {len(hist.get('train_loss', []))} epochs, "
              f"best val metric {max(hist.get('val_metric', [float('nan')])):.3f}); delete it to retrain")
        return hist
    hist = train_fn(path)
    torchkit.load_checkpoint(model, path)               # best epoch's weights
    return hist


def grad_norms_per_layer(model: nn.Module) -> dict:
    """{parameter name: gradient L2 norm} for weight tensors that have a gradient."""
    return {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None and p.ndim > 1}


# =========================================================================== D1-03 reference blocks
class InceptionBlock(nn.Module):
    """GoogLeNet Inception (v1) block: four parallel branches concatenated along channels.

    b1: 1x1 (f1) | b2: 1x1 (f3r) -> 3x3 (f3) | b3: 1x1 (f5r) -> 5x5 (f5) | b4: maxpool 3x3/1 -> 1x1 (fpool)
    Every conv has a bias and is followed by ReLU; spatial size is preserved.
    """

    def __init__(self, c_in, f1, f3r, f3, f5r, f5, fpool):
        super().__init__()
        relu = lambda: nn.ReLU(inplace=True)  # noqa: E731
        self.b1 = nn.Sequential(nn.Conv2d(c_in, f1, 1), relu())
        self.b2 = nn.Sequential(nn.Conv2d(c_in, f3r, 1), relu(), nn.Conv2d(f3r, f3, 3, padding=1), relu())
        self.b3 = nn.Sequential(nn.Conv2d(c_in, f5r, 1), relu(), nn.Conv2d(f5r, f5, 5, padding=2), relu())
        self.b4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(c_in, fpool, 1), relu())

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class BasicBlock(nn.Module):
    """ResNet-18/34 block: conv3x3(stride)-BN-ReLU-conv3x3-BN, + shortcut, then ReLU.

    shortcut = identity when shape is unchanged, else 1x1 conv (stride) + BN ("projection").
    """

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class Bottleneck(nn.Module):
    """ResNet-50 block: 1x1 (reduce) - 3x3 (stride) - 1x1 (expand x4), BN after each, + shortcut, ReLU."""

    expansion = 4

    def __init__(self, c_in, c_mid, stride=1):
        super().__init__()
        c_out = c_mid * self.expansion
        self.conv1 = nn.Conv2d(c_in, c_mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.conv2 = nn.Conv2d(c_mid, c_mid, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_mid)
        self.conv3 = nn.Conv2d(c_mid, c_out, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c_out)
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x))


class DepthwiseSeparable(nn.Module):
    """MobileNet block: depthwise 3x3 (groups=c_in) - BN - ReLU - pointwise 1x1 - BN - ReLU (convs without bias)."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(c_in, c_in, 3, stride, 1, groups=c_in, bias=False)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.pointwise = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        return F.relu(self.bn2(self.pointwise(x)))


class SqueezeExcite(nn.Module):
    """Squeeze-and-excitation: GAP -> fc (C/r) -> ReLU -> fc (C) -> sigmoid -> rescale channels."""

    def __init__(self, c, r=4):
        super().__init__()
        self.fc1 = nn.Conv2d(c, max(1, c // r), 1)
        self.fc2 = nn.Conv2d(max(1, c // r), c, 1)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s


class DenseLayer(nn.Module):
    """DenseNet layer: BN-ReLU-conv3x3 producing `growth` new channels, CONCATENATED to the input."""

    def __init__(self, c_in, growth=32):
        super().__init__()
        self.bn = nn.BatchNorm2d(c_in)
        self.conv = nn.Conv2d(c_in, growth, 3, 1, 1, bias=False)

    def forward(self, x):
        return torch.cat([x, self.conv(F.relu(self.bn(x)))], dim=1)


class MiniResNet(nn.Module):
    """stem (conv3x3/2 + BN + ReLU + maxpool 2) -> 3 stages of BasicBlocks -> GAP -> fc.

    widths (32, 64, 128); stage 1 stride 1, stages 2-3 stride 2. On a 128 px image the
    stages run at 32, 16 and 8 px.
    """

    def __init__(self, num_classes=37, widths=(32, 64, 128), blocks=(1, 1, 1), block=None):
        super().__init__()
        block = block or BasicBlock
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 2, 1, bias=False), nn.BatchNorm2d(widths[0]),
                                  nn.ReLU(inplace=True), nn.MaxPool2d(2))
        stages, c = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            layers = []
            for j in range(n):
                layers.append(block(c, w, 2 if (j == 0 and i > 0) else 1))
                c = w
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c, num_classes)

    def forward(self, x):
        return self.fc(self.pool(self.stages(self.stem(x))).flatten(1))


# =========================================================================== D1-03 model summaries
def summarize(model: nn.Module, input_size, depth: int | None = None) -> str:
    """Thin wrapper kept for the D1-03 notebook: `torchkit.summarize` now rolls leaf MACs up
    into container rows and handles single-layer models, so this just delegates."""
    return torchkit.summarize(model, input_size, depth=depth)


def profile_macs(model: nn.Module, input_size) -> int:
    """Conv + linear MACs for one image (delegates to `torchkit.profile`)."""
    return torchkit.profile(model, input_size)["macs"]


# =========================================================================== D1-03 gradient flow
class _PlainUnit(nn.Module):
    def __init__(self, c, bn):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1, bias=not bn)
        self.bn = nn.BatchNorm2d(c) if bn else nn.Identity()

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class _ResUnit(nn.Module):
    """Pre-activation residual unit with a single conv per branch so depth counts conv layers."""

    def __init__(self, c, bn):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1, bias=not bn)
        self.bn = nn.BatchNorm2d(c) if bn else nn.Identity()

    def forward(self, x):
        return x + self.conv(F.relu(self.bn(x)))


def make_deep_net(depth: int, residual: bool, width: int = 16, bn: bool = False, num_classes: int = 10) -> nn.Sequential:
    """Stem conv + (depth - 2) plain or residual 3x3 conv units + GAP + fc: `depth` weight layers."""
    unit = _ResUnit if residual else _PlainUnit
    return nn.Sequential(nn.Conv2d(3, width, 3, 1, 1), nn.ReLU(),
                         *[unit(width, bn) for _ in range(depth - 2)],
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, num_classes))


def first_layer_grad_norm(net: nn.Sequential, x: torch.Tensor, y: torch.Tensor) -> float:
    """||dL/dW|| of the first conv for one cross-entropy backward pass at initialisation."""
    net.zero_grad()
    F.cross_entropy(net(x), y).backward()
    return net[0].weight.grad.norm().item()


# =========================================================================== D1-03 published numbers
# ImageNet-1k top-1 (single crop, 224 px unless noted) and parameters, from the papers /
# torchvision model cards. Used for the timeline and the accuracy-vs-size scatter.
ARCHITECTURES = [
    # name,            year, top1,  params (M), GMACs, idea
    ("LeNet-5",         1998, None,  0.06,  None, "conv + pool + fc, trained with backprop"),
    ("AlexNet",         2012, 56.5,  61.1,  0.71, "ReLU, dropout, GPUs, data augmentation"),
    ("VGG-16",          2014, 71.6, 138.4, 15.5,  "only 3x3 convs, depth"),
    ("GoogLeNet",       2014, 69.8,   6.6,  1.5,  "Inception: parallel branches, 1x1 bottlenecks, GAP"),
    ("ResNet-50",       2015, 76.1,  25.6,  4.1,  "residual connections + BN: 100+ layers train"),
    ("ResNet-152",      2015, 78.3,  60.2, 11.5,  "depth scales once optimisation is fixed"),
    ("DenseNet-121",    2017, 74.4,   8.0,  2.8,  "concatenate all previous feature maps"),
    ("MobileNetV2",     2018, 71.9,   3.5,  0.30, "depthwise-separable + inverted residuals"),
    ("EfficientNet-B0", 2019, 77.7,   5.3,  0.39, "compound scaling of depth / width / resolution"),
    ("EfficientNet-B4", 2019, 83.4,  19.3,  4.5,  "same family, 380 px"),
    ("ViT-B/16",        2020, 81.1,  86.6, 17.6,  "patches + transformer; needs lots of data"),
    ("ConvNeXt-T",      2022, 82.1,  28.6,  4.5,  "a ResNet modernised with ViT-era design + training"),
    ("ResNet-50 (2021 recipe)", 2021, 80.9, 25.6, 4.1, "same net as 2015, modern training recipe"),
]


# =========================================================================== D1-04 reference exercises
def freeze(model: nn.Module, until: str | None = "layer3") -> nn.Module:
    """Freeze every parameter of a torchvision ResNet up to and including stage `until`
    (one of None, 'stem', 'layer1', ..., 'layer4'); later stages and fc stay trainable.

    until=None freezes nothing; until='layer4' freezes the whole backbone (head-only).
    BatchNorm layers of frozen stages are also put in eval mode by `set_bn_eval`.
    """
    order = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]
    stage_of = lambda name: "stem" if name.split(".")[0] in ("conv1", "bn1") else name.split(".")[0]  # noqa: E731
    cut = order.index(until) if until is not None else -1
    for name, p in model.named_parameters():
        p.requires_grad = order.index(stage_of(name)) > cut
    return model


def set_bn_eval(model: nn.Module) -> nn.Module:
    """Put BatchNorm layers whose parameters are all frozen in eval mode (keep running stats fixed)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if all(not p.requires_grad for p in m.parameters()):
                m.eval()
    return model


def keep_frozen_bn_eval(model: nn.Module) -> nn.Module:
    """Make `model.train()` keep fully-frozen BatchNorm layers in eval mode (running stats untouched).

    torchkit.fit calls model.train() every epoch; this wraps the instance's `train` method.
    """
    orig_train = model.train

    def train(mode: bool = True):
        orig_train(mode)
        if mode:
            set_bn_eval(model)
        return model

    model.train = train
    return model


def keep_bn_eval(model: nn.Module) -> nn.Module:
    """Make `model.train()` keep EVERY BatchNorm layer in eval mode (running stats frozen, affine
    parameters still trainable) - the usual choice when fine-tuning on small data / small batches."""
    orig_train = model.train

    def train(mode: bool = True):
        orig_train(mode)
        if mode:
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return model

    model.train = train
    return model


def param_groups_with_lrs(model: nn.Module, base_lr: float, decay: float = 0.3) -> list[dict]:
    """Discriminative learning rates for a torchvision ResNet.

    fc gets base_lr, layer4 base_lr * decay, layer3 base_lr * decay**2, ... stem base_lr * decay**5.
    Frozen parameters (requires_grad=False) are left out; empty groups are dropped.
    Returns [{"params": [...], "lr": lr, "name": stage}, ...] ordered from stem to fc.
    """
    order = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]
    groups = {s: [] for s in order}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        top = name.split(".")[0]
        groups["stem" if top in ("conv1", "bn1") else top].append(p)
    out = []
    for i, s in enumerate(order):
        if groups[s]:
            out.append({"params": groups[s], "lr": base_lr * decay ** (len(order) - 1 - i), "name": s})
    return out


def gradcam(model: nn.Module, target_layer: nn.Module, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
    """Reference Grad-CAM (delegates to torchkit.gradcam). x: (3, H, W). Returns (H, W) in [0, 1]."""
    return torchkit.gradcam(model, target_layer, x, class_idx)


@torch.no_grad()
def extract_features(backbone: nn.Module, loader, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a frozen backbone (in eval mode) over a loader -> (features (N, D) float, labels (N,) long)."""
    device = device or next(backbone.parameters()).device
    backbone.eval()
    feats, labels = [], []
    for x, y in loader:
        feats.append(backbone(x.to(device)).flatten(1).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def train_linear_probe(f_tr, y_tr, f_va, y_va, num_classes: int = 37, epochs: int = 100, lr: float = 1e-2,
                       weight_decay: float = 1e-3, batch_size: int = 256, seed: int = 0, verbose: bool = False):
    """Logistic regression on cached features (standardised with train statistics). Returns (head, val_acc, history)."""
    g = torch.Generator().manual_seed(seed)
    mu, sd = f_tr.mean(0, keepdim=True), f_tr.std(0, keepdim=True) + 1e-6
    xtr, xva = (f_tr - mu) / sd, (f_va - mu) / sd
    head = nn.Linear(f_tr.shape[1], num_classes)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = {"val_acc": []}
    n = len(y_tr)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            loss = F.cross_entropy(head(xtr[b]), y_tr[b], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if ep % 10 == 9 or ep == epochs - 1:
            with torch.no_grad():
                hist["val_acc"].append((head(xva).argmax(1) == y_va).float().mean().item())
    # fold the standardisation into the linear layer so it applies to raw features
    with torch.no_grad():
        w = head.weight / sd
        b = head.bias - (head.weight * (mu / sd)).sum(1)
        folded = nn.Linear(f_tr.shape[1], num_classes)
        folded.weight.copy_(w)
        folded.bias.copy_(b)
        acc = (folded(f_va).argmax(1) == y_va).float().mean().item()
    if verbose:
        print(f"linear probe: val acc {acc:.3f}")
    return folded, acc, hist


def cam_on_mask_fraction(cam: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of the Grad-CAM mass that falls on the pet (mask > 0)."""
    cam = np.asarray(cam, float)
    return float((cam * (mask > 0)).sum() / max(cam.sum(), 1e-8))


def pet_masks_for(paths, size: int) -> np.ndarray:
    """Binary pet masks (trimap 'pet' + 'border' -> 1) for image paths, resized / centre-cropped like `_load_resized`."""
    from PIL import Image

    out = []
    for p in paths:
        p = Path(p)
        tri = p.parent.parent / "annotations" / "trimaps" / (p.stem + ".png")
        m = Image.open(tri)
        w, h = m.size
        s = size / min(w, h)
        m = m.resize((max(size, round(w * s)), max(size, round(h * s))), Image.NEAREST)
        w, h = m.size
        l, t = (w - size) // 2, (h - size) // 2
        arr = np.asarray(m.crop((l, t, l + size, t + size)))
        out.append(((arr == 1) | (arr == 3)).astype(np.uint8))
    return np.stack(out)
