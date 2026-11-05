"""Day 3 shared code: segmentation models, mask utilities, Penn-Fudan targets, ViT helpers.

Everything the Day-3 notebooks need beyond `torchkit` lives here, so the
notebooks stay short and the reference solutions of the exercises have a single
source of truth (the checkers in `tests_day3` compare against these functions).

Sections
--------
RLE                 : rle_encode / rle_decode (COCO uncompressed convention)
losses              : soft_dice_loss, multiclass_dice_loss, bce_dice_loss
U-Net               : double_conv, EncoderBlock, DecoderBlock, UNet, unet_shapes
autoencoder         : ConvAutoencoder (FashionMNIST, D3-01)
pretrained encoder  : ResNetUNet (torchvision resnet18 encoder, D3-02)
evaluation          : confusion matrix over a loader, mIoU, sliding window, error maps
instance masks      : paste_mask, mask_iou, penn_fudan_target, mask_average_precision
Mask R-CNN          : build / train / predict helpers (D3-03)
ViT                 : patchify, attention, torchvision-ViT attention + tokens, PCA colours (D3-04)
zero-shot           : zero_shot_classify, toy CLIP embedding space
drawings            : schematic figures (U-Net, mask head, variants, CLIP, SAM, map of the field)
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ASSETS = Path(__file__).resolve().parent.parent / "assets"
SEG_COLORS = np.array([[0.10, 0.10, 0.10], [0.93, 0.35, 0.30], [0.30, 0.55, 0.93],
                       [0.35, 0.80, 0.45], [0.95, 0.75, 0.20], [0.70, 0.45, 0.85]])
PET_CLASSES = ["background", "pet", "border"]


# =============================================================================
# Small utilities
# =============================================================================
def colorize(mask, colors=None) -> np.ndarray:
    """Integer mask (H, W) -> RGB float image using a fixed palette (0 = dark background)."""
    colors = SEG_COLORS if colors is None else np.asarray(colors, float)
    m = np.asarray(mask, int)
    return colors[np.clip(m, 0, len(colors) - 1)]


def tune_cpu_threads(verbose: bool = True) -> int:
    """On CPU, pick the fastest torch thread count for a small conv workload.

    A laptop shared with a browser, an IDE and other jobs is often FASTER with fewer threads
    (thread oversubscription). Takes about a second; does nothing when a GPU is present.
    """
    import os
    if torch.cuda.is_available():
        return torch.get_num_threads()
    n_cpu = os.cpu_count() or 1
    x = torch.randn(8, 32, 48, 48)
    conv = nn.Conv2d(32, 32, 3, padding=1)
    best = None
    for n in sorted({1, 2, 4, max(1, n_cpu // 2)}):
        torch.set_num_threads(n)
        with torch.no_grad():
            conv(x)
            t = time.perf_counter()
            for _ in range(3):
                conv(x)
            dt = time.perf_counter() - t
        if best is None or dt < best[1]:
            best = (n, dt)
    torch.set_num_threads(best[0])
    if verbose:
        print(f"CPU threads for torch: {best[0]} (fastest of a quick benchmark on this machine)")
    return best[0]


def pet_root() -> Path:
    from .torchkit import data_root
    return data_root() / "oxford-iiit-pet"


def load_pet_example(stem: str = "Abyssinian_10") -> dict:
    """Raw Pet image (H, W, 3) float, raw trimap (1 pet, 2 background, 3 border), head boxes, species."""
    import xml.etree.ElementTree as ET
    from PIL import Image
    root = pet_root()
    img = np.asarray(Image.open(root / "images" / f"{stem}.jpg").convert("RGB"), np.float32) / 255.0
    tri = np.asarray(Image.open(root / "annotations" / "trimaps" / f"{stem}.png")).astype(np.int64)
    boxes = []
    xml = root / "annotations" / "xmls" / f"{stem}.xml"
    if xml.exists():
        for obj in ET.parse(xml).getroot().iter("object"):
            bb = obj.find("bndbox")
            boxes.append([float(bb.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")])
    return {"image": img, "trimap": tri, "head_boxes": np.array(boxes).reshape(-1, 4),
            "species": "cat" if stem[0].isupper() else "dog", "breed": stem.rsplit("_", 1)[0].replace("_", " ")}


def trimap_to_target(trimap) -> np.ndarray:
    """Raw Pet trimap (1 pet, 2 background, 3 border) -> 0 background, 1 pet, 2 border."""
    lut = np.array([0, 1, 0, 2], dtype=np.int64)
    return lut[np.clip(np.asarray(trimap, np.int64), 0, 3)]


def show_asset(name: str, title: str | None = None, figsize=(10, 5)):
    """Display a pre-rendered figure from notebooks/assets (used as an offline fallback)."""
    import matplotlib.pyplot as plt
    from PIL import Image
    path = ASSETS / name
    if not path.exists():
        print(f"(static figure {name} not found under {ASSETS})")
        return
    img = np.asarray(Image.open(path).convert("RGB"))
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img); ax.axis("off")
    ax.set_title(title or f"pre-rendered figure: {name}", fontsize=10)
    plt.tight_layout(); plt.show()


def to_numpy_image(t: torch.Tensor, mean=None, std=None) -> np.ndarray:
    """(3, H, W) tensor -> (H, W, 3) numpy in [0, 1], optionally undoing a normalisation."""
    x = t.detach().cpu().float()
    if mean is not None:
        x = x * torch.as_tensor(std).view(3, 1, 1) + torch.as_tensor(mean).view(3, 1, 1)
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


# =============================================================================
# RLE (COCO "uncompressed RLE": column-major, counts start with a run of zeros)
# =============================================================================
def rle_encode(mask) -> dict:
    """Binary mask (H, W) -> {"size": [H, W], "counts": [run lengths]}.

    Pixels are read column by column (Fortran order, as in COCO). The first count
    is always the number of leading zeros (it may be 0 if the mask starts with a 1).
    """
    m = np.asarray(mask).astype(bool)
    flat = m.reshape(-1, order="F").astype(np.int8)
    change = np.flatnonzero(np.diff(flat)) + 1                 # indices where the value flips
    bounds = np.concatenate([[0], change, [flat.size]])
    counts = np.diff(bounds).tolist()
    if flat.size and flat[0] == 1:
        counts = [0] + counts
    return {"size": [int(m.shape[0]), int(m.shape[1])], "counts": [int(c) for c in counts]}


def rle_decode(rle: dict) -> np.ndarray:
    """Inverse of `rle_encode`: returns a uint8 mask (H, W)."""
    H, W = rle["size"]
    flat = np.zeros(H * W, np.uint8)
    pos, val = 0, 0
    for c in rle["counts"]:
        flat[pos:pos + c] = val
        pos += c
        val = 1 - val
    return flat.reshape((H, W), order="F")


# =============================================================================
# Losses (torch)
# =============================================================================
def soft_dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - soft Dice over the whole batch. prob in [0, 1], target in {0, 1}, same shape."""
    target = target.float()
    inter = (prob * target).sum()
    return 1 - (2 * inter + eps) / (prob.sum() + target.sum() + eps)


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    """bce_weight * BCE-with-logits + (1 - bce_weight) * soft Dice (binary segmentation)."""
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = soft_dice_loss(torch.sigmoid(logits), target)
    return bce_weight * bce + (1 - bce_weight) * dice


def multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0,
                         ignore_background: bool = False) -> torch.Tensor:
    """Soft Dice averaged over classes. logits (N, C, H, W), target (N, H, W) integer."""
    C = logits.shape[1]
    prob = logits.softmax(1)
    onehot = F.one_hot(target.long(), C).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (prob * onehot).sum(dims)
    denom = prob.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    if ignore_background:
        dice = dice[1:]
    return 1 - dice.mean()


class CEDiceLoss(nn.Module):
    """Cross-entropy + multi-class soft Dice, the default loss for the Day-3 U-Nets."""

    def __init__(self, dice_weight: float = 0.5, class_weights=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.register_buffer("w", None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target.long(), weight=self.w)
        return (1 - self.dice_weight) * ce + self.dice_weight * multiclass_dice_loss(logits, target)


# =============================================================================
# U-Net
# =============================================================================
def double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """(conv3x3 -> BN -> ReLU) x 2, 'same' padding: the basic U-Net unit."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class EncoderBlock(nn.Module):
    """double_conv, keep the result as the skip, then 2x2 max-pool.

    forward(x (N, in_ch, H, W)) -> (pooled (N, out_ch, H/2, W/2), skip (N, out_ch, H, W))
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """2x2 transposed conv (x2 upsample), concatenate the skip, double_conv.

    forward(x (N, in_ch, h, w), skip (N, skip_ch, 2h, 2w)) -> (N, out_ch, 2h, 2w)
    With use_skip=False the skip is ignored (ablation).
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_skip: bool = True):
        super().__init__()
        self.use_skip = use_skip
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = double_conv(out_ch + (skip_ch if use_skip else 0), out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if self.use_skip:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Classic U-Net. Encoder channels base * 2**i (i < depth), bottleneck base * 2**depth."""

    def __init__(self, in_ch: int = 3, num_classes: int = 4, base: int = 16, depth: int = 4,
                 use_skips: bool = True):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth)]
        self.encoders = nn.ModuleList()
        c_prev = in_ch
        for c in chs:
            self.encoders.append(EncoderBlock(c_prev, c))
            c_prev = c
        self.bottleneck = double_conv(chs[-1], chs[-1] * 2)
        self.decoders = nn.ModuleList([DecoderBlock(c * 2, c, c, use_skips) for c in reversed(chs)])
        self.head = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x, s = enc(x)
            skips.append(s)
        x = self.bottleneck(x)
        for dec, s in zip(self.decoders, reversed(skips)):
            x = dec(x, s)
        return self.head(x)


def unet_shapes(input_size: int, depth: int, base: int):
    """[(channels, size)] for every encoder level and the bottleneck."""
    return [(base * 2 ** i, input_size // 2 ** i) for i in range(depth + 1)]


# =============================================================================
# Autoencoder (D3-01)
# =============================================================================
class ConvAutoencoder(nn.Module):
    """28x28 -> 14 -> 7 -> 4 (bottleneck) -> 7 -> 14 -> 28. No skips: the bottleneck must carry 'where'."""

    def __init__(self, bottleneck_ch: int = 8, skip: bool = False):
        super().__init__()
        self.skip = skip
        self.enc1 = nn.Sequential(nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True))     # 16 x 14 x 14
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True))    # 32 x 7 x 7
        self.enc3 = nn.Conv2d(32, bottleneck_ch, 3, stride=2, padding=1)                                # C x 4 x 4
        self.dec3 = nn.Sequential(nn.ReLU(inplace=True),
                                  nn.ConvTranspose2d(bottleneck_ch, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True))   # 7
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True))  # 14
        self.dec1 = nn.ConvTranspose2d(16 + (16 if skip else 0), 1, 3, stride=2, padding=1, output_padding=1)  # 28

    def encode(self, x):
        return self.enc3(self.enc2(self.enc1(x)))

    def forward(self, x):
        e1 = self.enc1(x)
        z = self.enc3(self.enc2(e1))
        d = self.dec2(self.dec3(z))
        if self.skip:                                   # U-Net style: give the decoder the 14x14 'where'
            d = torch.cat([d, e1], 1)
        return self.dec1(d)


# =============================================================================
# U-Net with a pretrained torchvision ResNet-18 encoder (D3-02)
# =============================================================================
class ResNetUNet(nn.Module):
    """U-Net whose encoder is torchvision's ImageNet ResNet-18.

    encoder (input 128 px)          channels  size
      stem  conv1-bn-relu            64       64   (/2)   -> skip
      layer1 (after maxpool)         64       32   (/4)   -> skip
      layer2                        128       16   (/8)   -> skip
      layer3                        256        8   (/16)  -> skip
      layer4                        512        4   (/32)  bottleneck
    decoder: four DecoderBlocks back to /2, one more x2 upsample, 1x1 head.
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.head = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32),
                                  nn.ReLU(inplace=True), nn.Conv2d(32, num_classes, 1))

    def encoder_parameters(self):
        for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            yield from m.parameters()

    def decoder_parameters(self):
        enc = {id(p) for p in self.encoder_parameters()}
        return [p for p in self.parameters() if id(p) not in enc]

    def forward(self, x):
        s0 = self.stem(x)                      # /2
        s1 = self.layer1(self.pool(s0))        # /4
        s2 = self.layer2(s1)                   # /8
        s3 = self.layer3(s2)                   # /16
        b = self.layer4(s3)                    # /32
        x = self.dec4(b, s3)
        x = self.dec3(x, s2)
        x = self.dec2(x, s1)
        x = self.dec1(x, s0)
        return self.head(self.up(x))


# =============================================================================
# Segmentation evaluation and inference helpers
# =============================================================================
@torch.no_grad()
def confusion_over_loader(model, loader, num_classes: int, device=None, max_batches=None) -> np.ndarray:
    """Accumulate a (C, C) confusion matrix (rows = target) over a DataLoader of (image, mask)."""
    device = device or next(model.parameters()).device
    model.eval()
    cm = torch.zeros(num_classes * num_classes, dtype=torch.long)
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        pred = model(x.to(device)).argmax(1).cpu()
        y = y.long()
        valid = (y >= 0) & (y < num_classes)
        cm += torch.bincount(num_classes * y[valid] + pred[valid], minlength=num_classes ** 2)
    return cm.view(num_classes, num_classes).numpy()


def iou_from_confusion(cm: np.ndarray):
    """Per-class IoU and mIoU from a confusion matrix (rows = target, cols = prediction)."""
    cm = np.asarray(cm, float)
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / union, np.nan)
    return iou, float(np.nanmean(iou))


def miou_metric(num_classes: int):
    """A metric_fn(logits, targets) -> float for torchkit.fit (batch mIoU)."""
    def fn(logits, targets):
        pred = logits.argmax(1).detach().cpu().numpy(); t = targets.detach().cpu().numpy()
        from . import seg
        return seg.mean_iou(pred, t, num_classes)
    return fn


def train_segmentation(model, train_loader, loss_fn, optimizer, epochs=1, max_minutes=None, device=None,
                       scheduler=None, log_every=20, val_loader=None, num_classes=None):
    """Plain training loop with a wall-clock cap. Returns history dict(step_loss, epoch_miou, seconds)."""
    device = device or next(model.parameters()).device
    hist = {"step_loss": [], "epoch_miou": [], "seconds": 0.0}
    t0 = time.time()
    step = 0
    stop = False
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            hist["step_loss"].append(float(loss.detach()))
            step += 1
            if log_every and step % log_every == 0:
                print(f"  epoch {ep + 1} step {step:4d}  loss {np.mean(hist['step_loss'][-log_every:]):.4f}  ({time.time() - t0:5.0f}s)")
            if max_minutes is not None and time.time() - t0 > 60 * max_minutes:
                stop = True
                break
        if val_loader is not None and num_classes:
            _, m = iou_from_confusion(confusion_over_loader(model, val_loader, num_classes, device))
            hist["epoch_miou"].append(m)
            print(f"epoch {ep + 1}: val mIoU {m:.3f}  ({time.time() - t0:.0f}s)")
        if stop:
            print(f"stopped after {max_minutes} min (wall-clock cap)")
            break
    hist["seconds"] = time.time() - t0
    return hist


def sliding_window_predict(model_fn, image: torch.Tensor, window: int, stride: int) -> torch.Tensor:
    """Run `model_fn` on overlapping windows and average the outputs.

    image (C, H, W); model_fn maps (1, C, window, window) -> (1, K, window, window).
    The last window in each direction is shifted to end exactly at the border.
    Returns (K, H, W).
    """
    C, H, W = image.shape
    ys = list(range(0, max(H - window, 0) + 1, stride))
    xs = list(range(0, max(W - window, 0) + 1, stride))
    if ys[-1] + window < H:
        ys.append(H - window)
    if xs[-1] + window < W:
        xs.append(W - window)
    out, count = None, torch.zeros(1, H, W)
    for y in ys:
        for x in xs:
            pred = model_fn(image[None, :, y:y + window, x:x + window])[0]
            if out is None:
                out = torch.zeros(pred.shape[0], H, W, dtype=pred.dtype)
            out[:, y:y + window, x:x + window] += pred
            count[:, y:y + window, x:x + window] += 1
    return out / count


def error_map(pred, target, cls: int = 1) -> np.ndarray:
    """RGB error image for one class: TP grey, FP red, FN blue, TN black."""
    p = np.asarray(pred) == cls
    t = np.asarray(target) == cls
    img = np.zeros(p.shape + (3,))
    img[p & t] = [0.75, 0.75, 0.75]
    img[p & ~t] = [0.90, 0.15, 0.15]
    img[~p & t] = [0.15, 0.35, 0.95]
    return img


def best_threshold(prob, target, thresholds):
    """Binary IoU for every threshold; returns (best_threshold, ious array)."""
    prob = np.asarray(prob); t = np.asarray(target).astype(bool)
    ious = []
    for th in thresholds:
        p = prob >= th
        inter = np.logical_and(p, t).sum(); union = np.logical_or(p, t).sum()
        ious.append(inter / union if union else 1.0)
    ious = np.array(ious, float)
    return float(np.asarray(thresholds)[int(np.argmax(ious))]), ious


# =============================================================================
# Instance masks (D3-03)
# =============================================================================
def paste_mask(mask28, box, image_hw, threshold: float = 0.5) -> np.ndarray:
    """Resize a small (M, M) mask probability map to its box and paste it into an empty image.

    box = (x1, y1, x2, y2) in pixels (floats allowed); image_hw = (H, W).
    Returns a bool array (H, W). Parts of the box outside the image are clipped.
    """
    m = torch.as_tensor(np.asarray(mask28, np.float32))
    H, W = image_hw
    x1, y1, x2, y2 = [float(v) for v in box]
    x0, y0 = int(math.floor(x1)), int(math.floor(y1))
    w = max(int(math.ceil(x2)) - x0, 1)
    h = max(int(math.ceil(y2)) - y0, 1)
    resized = F.interpolate(m[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
    out = np.zeros((H, W), bool)
    ya, yb = max(y0, 0), min(y0 + h, H)
    xa, xb = max(x0, 0), min(x0 + w, W)
    if ya < yb and xa < xb:
        out[ya:yb, xa:xb] = resized[ya - y0:yb - y0, xa - x0:xb - x0] >= threshold
    return out


def mask_iou(a, b) -> np.ndarray:
    """Pairwise IoU between two stacks of binary masks: a (N, H, W), b (M, H, W) -> (N, M)."""
    a = np.asarray(a).astype(bool).reshape(len(a), -1).astype(np.float64)
    b = np.asarray(b).astype(bool).reshape(len(b), -1).astype(np.float64)
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def penn_fudan_target(instance_mask) -> dict:
    """Penn-Fudan instance-id mask (H, W) with 0 = background, k = person k -> Mask R-CNN target.

    Returns dict(boxes float32 (N, 4) xyxy with x2/y2 = last pixel index (torchvision
    `masks_to_boxes` convention), labels int64 (N,) all 1, masks uint8 (N, H, W)).
    """
    m = torch.as_tensor(np.array(instance_mask))
    ids = torch.unique(m)
    ids = ids[ids != 0]
    masks = (m[None] == ids[:, None, None]).to(torch.uint8)
    boxes = torch.zeros((len(ids), 4), dtype=torch.float32)
    for i, mk in enumerate(masks):
        ys, xs = torch.where(mk > 0)
        boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], dtype=torch.float32)
    return {"boxes": boxes, "labels": torch.ones(len(ids), dtype=torch.int64), "masks": masks}


def _ap_from_records(records, n_gt):
    from . import boxes as _boxes
    if n_gt == 0:
        return float("nan")
    if not records:
        return 0.0
    records = sorted(records, key=lambda t: -t[0])
    tp = np.array([t[1] for t in records], bool)
    tp_c = np.cumsum(tp); fp_c = np.cumsum(~tp)
    return _boxes.average_precision(tp_c / np.maximum(tp_c + fp_c, 1e-12), tp_c / n_gt, "interp101")


def mask_average_precision(predictions, ground_truths, iou_threshold: float = 0.5) -> float:
    """Single-class AP with mask IoU, COCO 101-point interpolation.

    predictions  : list over images of dict(masks (N, H, W) bool, scores (N,))
    ground_truths: list over images of dict(masks (M, H, W) bool)
    Greedy matching by decreasing score, each GT matched at most once.
    """
    records, n_gt = [], 0
    for pred, gt in zip(predictions, ground_truths):
        gm = np.asarray(gt["masks"]).astype(bool)
        n_gt += len(gm)
        pm = np.asarray(pred["masks"]).astype(bool)
        ps = np.asarray(pred["scores"], float).reshape(-1)
        if len(pm) == 0:
            continue
        order = np.argsort(-ps)
        ious = mask_iou(pm[order], gm) if len(gm) else np.zeros((len(pm), 0))
        matched = np.zeros(len(gm), bool)
        for k in range(len(order)):
            tp = False
            if len(gm):
                row = ious[k].copy(); row[matched] = -1
                j = int(row.argmax())
                if row[j] >= iou_threshold:
                    matched[j] = True; tp = True
            records.append((ps[order[k]], tp))
    return _ap_from_records(records, n_gt)


def coco_style_ap(predictions, ground_truths, kind: str = "mask"):
    """AP@0.5, AP@0.75 and AP@[.5:.95] for one class. kind = 'mask' or 'box'."""
    from . import boxes as _boxes
    thrs = np.arange(0.5, 0.96, 0.05)
    out = []
    for thr in thrs:
        if kind == "mask":
            out.append(mask_average_precision(predictions, ground_truths, thr))
        else:
            p = [dict(boxes=d["boxes"], scores=d["scores"], labels=np.zeros(len(d["scores"]), int)) for d in predictions]
            g = [dict(boxes=d["boxes"], labels=np.zeros(len(d["boxes"]), int)) for d in ground_truths]
            out.append(_boxes.mean_average_precision(p, g, 1, (thr,))[0])
    out = np.array(out)
    return {"AP50": float(out[0]), "AP75": float(out[5]), "AP": float(out.mean())}


# =============================================================================
# Mask R-CNN (D3-03)
# =============================================================================
CPU_FRIENDLY_RCNN = dict(rpn_pre_nms_top_n_train=1000, rpn_post_nms_top_n_train=500, rpn_pre_nms_top_n_test=500,
                         rpn_post_nms_top_n_test=200, box_batch_size_per_image=128, box_detections_per_img=50)


def coco_maskrcnn(min_size: int = 320, max_size: int = 512, **kw):
    """torchvision maskrcnn_resnet50_fpn_v2 with COCO weights (91 classes, person = 1), CPU-friendly RoI counts."""
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    opts = {**CPU_FRIENDLY_RCNN, **kw}
    return maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1, min_size=min_size,
                                    max_size=max_size, **opts)


def maskrcnn_for_finetuning(num_classes: int = 2, min_size: int = 320, max_size: int = 512,
                            pretrained: bool = True, **kw):
    """torchvision maskrcnn_resnet50_fpn_v2 with COCO weights and BOTH predictors replaced."""
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    if pretrained:
        model = coco_maskrcnn(min_size, max_size, **kw)
    else:
        model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=91, min_size=min_size,
                                         max_size=max_size, **{**CPU_FRIENDLY_RCNN, **kw})
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_ch = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_ch, 256, num_classes)
    return model


@torch.no_grad()
def warm_start_from_coco(model, coco_model, coco_class: int = 1):
    """Copy the COCO background + `coco_class` rows of both predictors into a 2-class model.

    COCO already has 'person' (class 1): instead of starting the new heads from random
    numbers we can start them from the rows of the classes we keep.
    """
    keep = [0, coco_class]
    src, dst = coco_model.roi_heads.box_predictor, model.roi_heads.box_predictor
    dst.cls_score.weight.copy_(src.cls_score.weight[keep]); dst.cls_score.bias.copy_(src.cls_score.bias[keep])
    rows = [4 * c + k for c in keep for k in range(4)]
    dst.bbox_pred.weight.copy_(src.bbox_pred.weight[rows]); dst.bbox_pred.bias.copy_(src.bbox_pred.bias[rows])
    sm, dm = coco_model.roi_heads.mask_predictor, model.roi_heads.mask_predictor
    dm.conv5_mask.load_state_dict(sm.conv5_mask.state_dict())
    dm.mask_fcn_logits.weight.copy_(sm.mask_fcn_logits.weight[keep]); dm.mask_fcn_logits.bias.copy_(sm.mask_fcn_logits.bias[keep])
    return model


def train_detector(model, loader, optimizer, max_iters: int = 200, max_minutes: float | None = None,
                   device=None, log_every: int = 10, scheduler=None):
    """Training loop for torchvision detection models (the model returns a dict of losses)."""
    device = device or next(model.parameters()).device
    model.train()
    hist = {k: [] for k in ("loss", "loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg")}
    t0 = time.time(); it = 0
    while it < max_iters:
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            hist["loss"].append(float(loss.detach()))
            for k, v in losses.items():
                hist.setdefault(k, []).append(float(v.detach()))
            it += 1
            if log_every and it % log_every == 0:
                parts = "  ".join(f"{k.replace('loss_', '')} {np.mean(hist[k][-log_every:]):.3f}" for k in losses)
                print(f"  iter {it:4d}  loss {np.mean(hist['loss'][-log_every:]):.3f}  [{parts}]  ({time.time() - t0:.0f}s)")
            if it >= max_iters or (max_minutes is not None and time.time() - t0 > 60 * max_minutes):
                hist["seconds"] = time.time() - t0
                hist["iters"] = it
                return hist
    hist["seconds"] = time.time() - t0
    hist["iters"] = it
    return hist


@torch.no_grad()
def predict_instances(model, images, score_threshold: float = 0.5, mask_threshold: float = 0.5, label=None):
    """Run a torchvision Mask R-CNN; returns list of numpy dicts(boxes, scores, labels, masks bool)."""
    model.eval()
    device = next(model.parameters()).device
    outs = model([im.to(device) for im in images])
    res = []
    for o in outs:
        keep = o["scores"] >= score_threshold
        if label is not None:
            keep &= o["labels"] == label
        res.append({"boxes": o["boxes"][keep].cpu().numpy(), "scores": o["scores"][keep].cpu().numpy(),
                    "labels": o["labels"][keep].cpu().numpy(),
                    "masks": (o["masks"][keep, 0] >= mask_threshold).cpu().numpy(),
                    "soft_masks": o["masks"][keep, 0].cpu().numpy()})
    return res


def draw_instances(ax, image, masks, boxes=None, scores=None, alpha=0.5, title=None):
    """Image with one random colour per instance mask (+ boxes and scores)."""
    from matplotlib.patches import Rectangle
    img = np.asarray(image, float).copy()
    rng = np.random.default_rng(3)
    for i, m in enumerate(masks):
        c = rng.uniform(0.2, 1.0, 3)
        m = np.asarray(m, bool)
        img[m] = (1 - alpha) * img[m] + alpha * c
        if boxes is not None:
            x1, y1, x2, y2 = boxes[i]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=c, lw=1.5))
            if scores is not None:
                ax.text(x1, y1 - 2, f"{scores[i]:.2f}", fontsize=7, color="white",
                        bbox=dict(fc=c, ec="none", pad=1, alpha=0.9))
    ax.imshow(np.clip(img, 0, 1)); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


# =============================================================================
# ViT helpers (D3-04)
# =============================================================================
def patchify(img: torch.Tensor, p: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, C*p*p), patches in row-major order, each flattened as (C, p, p)."""
    B, C, H, W = img.shape
    x = img.unfold(2, p, p).unfold(3, p, p)            # B, C, H/p, W/p, p, p
    x = x.permute(0, 2, 3, 1, 4, 5)                    # B, H/p, W/p, C, p, p
    return x.reshape(B, (H // p) * (W // p), C * p * p)


def attention(q, k, v):
    """Scaled dot-product attention. q (.., Nq, d), k (.., Nk, d), v (.., Nk, dv) -> (out, weights)."""
    w = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(-1)
    return w @ v, w


@torch.no_grad()
def torchvision_vit_attention(model, x: torch.Tensor, layer: int = -1) -> torch.Tensor:
    """Attention weights (B, heads, N+1, N+1) of one encoder layer of torchvision's vit_b_16.

    torchvision calls nn.MultiheadAttention with need_weights=False, so we capture the
    layer's normalised input with a hook and recompute the attention with weights.
    """
    block = model.encoder.layers[layer]
    store = {}
    h = block.ln_1.register_forward_hook(lambda m, i, o: store.__setitem__("x", o))
    model(x)
    h.remove()
    y = store["x"]
    _, w = block.self_attention(y, y, y, need_weights=True, average_attn_weights=False)
    return w


@torch.no_grad()
def torchvision_vit_tokens(model, x: torch.Tensor) -> torch.Tensor:
    """Final normalised tokens (B, 1 + N, D) of torchvision's vit_b_16 (cls token first)."""
    t = model._process_input(x)
    cls = model.class_token.expand(t.shape[0], -1, -1)
    return model.encoder(torch.cat([cls, t], dim=1))


@torch.no_grad()
def timm_last_attention(model, x: torch.Tensor) -> torch.Tensor:
    """Attention weights (B, heads, N, N) of the last block of a timm ViT."""
    attn = model.blocks[-1].attn
    old = getattr(attn, "fused_attn", False)
    attn.fused_attn = False
    store = {}
    h = attn.attn_drop.register_forward_hook(lambda m, i, o: store.__setitem__("w", o))
    model(x)
    h.remove()
    attn.fused_attn = old
    return store["w"]


def pca_rgb(feats: torch.Tensor, n: int = 3, fit_on: torch.Tensor | None = None) -> torch.Tensor:
    """Project features (P, D) on their top-n principal components, min-max scaled to [0, 1] per component."""
    X = feats.float()
    ref = X if fit_on is None else fit_on.float()
    mu = ref.mean(0, keepdim=True)
    _, _, V = torch.linalg.svd(ref - mu, full_matrices=False)
    proj = (X - mu) @ V[:n].T
    lo, hi = proj.min(0, keepdim=True).values, proj.max(0, keepdim=True).values
    return (proj - lo) / (hi - lo + 1e-8)


def knn_predict(train_feats, train_labels, test_feats, k: int = 10, temperature: float = 0.07):
    """Weighted k-NN on cosine similarity (as in the DINO paper). Returns predicted labels."""
    a = F.normalize(torch.as_tensor(train_feats).float(), dim=1)
    b = F.normalize(torch.as_tensor(test_feats).float(), dim=1)
    sim = b @ a.T
    top, idx = sim.topk(k, dim=1)
    labels = torch.as_tensor(train_labels)[idx]
    C = int(torch.as_tensor(train_labels).max()) + 1
    votes = torch.zeros(len(b), C).scatter_add_(1, labels, (top / temperature).exp())
    return votes.argmax(1)


def linear_probe(train_feats, train_labels, test_feats, epochs: int = 200, lr: float = 1e-2, wd: float = 1e-4):
    """Logistic regression on frozen features (full-batch AdamW). Returns predicted labels."""
    X = torch.as_tensor(train_feats).float(); y = torch.as_tensor(train_labels).long()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    Xt = (torch.as_tensor(test_feats).float() - mu) / sd
    clf = nn.Linear(X.shape[1], int(y.max()) + 1)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        loss = F.cross_entropy(clf(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return clf(Xt).argmax(1)


# =============================================================================
# Zero-shot classification (CLIP-style) and a toy embedding space
# =============================================================================
def zero_shot_classify(image_feats, text_feats, logit_scale: float = 100.0):
    """Cosine similarity between L2-normalised image (N, D) and text (C, D) features.

    Returns (pred (N,), probs (N, C)) with probs = softmax(logit_scale * cos).
    """
    img = F.normalize(torch.as_tensor(image_feats).float(), dim=-1)
    txt = F.normalize(torch.as_tensor(text_feats).float(), dim=-1)
    probs = (logit_scale * img @ txt.T).softmax(-1)
    return probs.argmax(-1), probs


def toy_clip_space(n_classes: int = 37, dim: int = 64, n_per_class: int = 20, n_templates: int = 8,
                   image_noise: float = 1.0, template_noise: float = 0.9, shared_noise: float = 0.0,
                   seed: int = 0):
    """A synthetic joint embedding space to experiment with zero-shot classification offline.

    Each class has a hidden 'concept' vector. Images = concept + image noise.
    A text prompt built from template t = concept + shared class noise (what the text encoder
    gets wrong about the class name, whatever the wording) + template-specific noise (a badly
    worded prompt drifts further). Averaging templates removes only the template part.
    Returns dict(img (N, D), labels (N,), text (T, C, D)).
    """
    g = torch.Generator().manual_seed(seed)
    concept = F.normalize(torch.randn(n_classes, dim, generator=g), dim=-1)
    labels = torch.arange(n_classes).repeat_interleave(n_per_class)
    img = concept[labels] + image_noise * torch.randn(len(labels), dim, generator=g) / math.sqrt(dim)
    tn = torch.as_tensor(template_noise, dtype=torch.float32).reshape(-1, 1, 1)   # scalar or one value per template
    if tn.numel() > 1:
        n_templates = tn.numel()
    shared = shared_noise * torch.randn(1, n_classes, dim, generator=g) / math.sqrt(dim)
    text = concept[None] + shared + tn * torch.randn(n_templates, n_classes, dim, generator=g) / math.sqrt(dim)
    return {"img": F.normalize(img, dim=-1), "labels": labels, "text": F.normalize(text, dim=-1)}


# =============================================================================
# Schematic drawings (Matplotlib only)
# =============================================================================
def _box(ax, x, y, w, h, fc, text="", fs=8, ec="k", lw=0.8, tc="black", **kw):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, **kw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def _arrow(ax, x0, y0, x1, y1, color="k", lw=1.2, style="->", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, lw=lw, color=color, ls=ls, shrinkA=0, shrinkB=0))


def draw_unet(ax, input_size: int = 128, depth: int = 4, base: int = 16, in_ch: int = 3, num_classes: int = 4):
    """The U drawn with tensor shapes; skip connections in orange."""
    shapes_ = unet_shapes(input_size, depth, base)
    ax.axis("off")
    step, dy, bw, bh = 1.7, 1.35, 1.9, 0.95
    enc_col, dec_col, bott_col = "#cfe0f5", "#cde8cf", "#f8d0b8"
    W = 2 * depth * step + bw
    for i in range(depth):
        c, s = shapes_[i]
        ex, y = i * step, -i * dy
        dx_ = 2 * depth * step - i * step
        _box(ax, ex, y, bw, bh, enc_col, f"enc {i + 1}\n{c} x {s} x {s}", fs=9)
        _box(ax, dx_, y, bw, bh, dec_col, f"dec {i + 1}\n{c} x {s} x {s}", fs=9)
        _arrow(ax, ex + bw, y + bh / 2, dx_, y + bh / 2, color="#F58518", lw=2.0)
        ax.text((ex + bw + dx_) / 2, y + bh / 2 + 0.1, f"skip {c} ch", ha="center", fontsize=8, color="#c46a10")
        nx, ny = (i + 1) * step, -(i + 1) * dy
        _arrow(ax, ex + 0.5, y, nx + 0.4, ny + bh, color="#4C78A8", lw=1.5)          # down
        ux = 2 * depth * step - (i + 1) * step
        _arrow(ax, ux + bw - 0.4, ny + bh, dx_ + bw - 0.5, y, color="#54A24B", lw=1.5)  # up
    c, s = shapes_[-1]
    _box(ax, depth * step, -depth * dy, bw, bh, bott_col, f"bottleneck\n{c} x {s} x {s}", fs=9)
    ax.text(0.0, bh + 0.55, f"input {in_ch} x {input_size} x {input_size}", fontsize=10)
    _arrow(ax, 0.6, bh + 0.5, 0.7, bh)
    ax.text(W - 2.6, bh + 0.55, f"logits {num_classes} x {input_size} x {input_size} (1x1 conv)", fontsize=10)
    _arrow(ax, W - 0.8, bh, W - 0.7, bh + 0.5)
    ax.text(0.0, -depth * dy + 0.3, "blue: double conv (3x3-BN-ReLU x2) + 2x2 max-pool\n"
            "green: 2x2 transposed conv (x2 up) + concat + double conv\norange: skip connection", fontsize=9, va="top")
    ax.set_xlim(-0.2, W + 0.2); ax.set_ylim(-depth * dy - 0.4, bh + 1.0)


def draw_mask_head(ax):
    """Mask R-CNN mask branch: RoIAlign 14x14x256 -> 4 conv -> deconv -> 28x28xC."""
    ax.axis("off")
    _box(ax, 0.0, 1.0, 1.7, 1.4, "#e8e8e8", "FPN features\n(P2-P5)\n256 ch", fs=8)
    _box(ax, 2.3, 1.2, 1.4, 1.0, "#cfe0f5", "RoIAlign\n14 x 14 x 256", fs=8)
    for k in range(4):
        _box(ax, 4.2 + k * 1.05, 1.2, 0.85, 1.0, "#cde8cf", f"conv3x3\n256\n+ReLU", fs=7)
    _box(ax, 8.5, 1.05, 1.3, 1.3, "#f8d0b8", "deconv 2x2/2\n28 x 28\nx 256", fs=8)
    _box(ax, 10.3, 1.05, 1.5, 1.3, "#fde2a2", "1x1 conv\n28 x 28 x C\n(sigmoid)", fs=8)
    for (x0, x1) in [(1.7, 2.3), (3.7, 4.2), (5.05, 5.25), (6.1, 6.3), (7.15, 7.35), (8.2, 8.5), (9.8, 10.3)]:
        _arrow(ax, x0, 1.7, x1, 1.7)
    ax.text(10.3, 0.75, "loss: per-pixel BCE on the channel\nof the GT class only (no competition\nbetween classes)", fontsize=7.5, va="top")
    ax.text(2.3, 0.95, "bilinear sampling, no\nrounding of the RoI", fontsize=7.5, va="top")
    _box(ax, 2.3, 2.75, 5.0, 0.7, "#f3f3f3", "box branch (Faster R-CNN): RoIAlign 7x7 -> box head -> class + box deltas", fs=7.5)
    _arrow(ax, 1.7, 2.1, 2.3, 3.05)
    ax.set_xlim(-0.2, 12.0); ax.set_ylim(0.0, 3.7)


def draw_clip(ax):
    """CLIP: two encoders, a shared embedding space, contrastive matrix and zero-shot use."""
    ax.axis("off")
    _box(ax, 0.0, 2.3, 1.6, 0.8, "#cfe0f5", "image\nencoder (ViT)", fs=8)
    _box(ax, 0.0, 0.5, 1.6, 0.8, "#cde8cf", "text encoder\n(transformer)", fs=8)
    ax.text(0.8, 3.35, "N photos", ha="center", fontsize=8)
    ax.text(0.8, 0.25, "N captions", ha="center", fontsize=8, va="top")
    n = 5
    x0, y0, s = 3.0, 0.6, 0.45
    for i in range(n):
        for j in range(n):
            fc = "#54A24B" if i == j else "#f0f0f0"
            ax.add_patch(__import__("matplotlib").patches.Rectangle((x0 + j * s, y0 + (n - 1 - i) * s), s, s, fc=fc, ec="white"))
    _arrow(ax, 1.6, 2.7, x0 - 0.1, y0 + n * s - 0.2, color="#4C78A8")
    _arrow(ax, 1.6, 0.9, x0 + 0.2, y0 - 0.05, color="#54A24B")
    ax.text(x0 + n * s / 2, y0 + n * s + 0.15, "cosine similarities I_i . T_j\ncontrastive loss: diagonal = match",
            ha="center", fontsize=8)
    # zero-shot side
    xs = 6.0
    ax.text(xs, 3.3, "zero-shot classification", fontsize=9, weight="bold")
    prompts = ["a photo of a {Abyssinian}, a type of pet", "a photo of a {beagle}, a type of pet", "...  (one prompt per class)"]
    for k, p in enumerate(prompts):
        _box(ax, xs, 2.5 - 0.55 * k, 3.5, 0.42, "#cde8cf", p, fs=7)
    _box(ax, xs, 0.35, 1.5, 0.6, "#cfe0f5", "test image", fs=8)
    _arrow(ax, xs + 1.5, 0.65, xs + 2.2, 0.65)
    ax.text(xs + 2.25, 0.65, "argmax_c cos(image, text_c)\n= predicted class, no training", va="center", fontsize=8)
    ax.set_xlim(-0.2, 10.2); ax.set_ylim(0.0, 3.6)


def draw_sam(ax):
    """SAM: heavy image encoder once, light prompt encoder + mask decoder per prompt."""
    ax.axis("off")
    _box(ax, 0.0, 1.0, 1.4, 1.2, "#e8e8e8", "image\n1024 x 1024", fs=8)
    _box(ax, 1.9, 0.9, 1.8, 1.4, "#cfe0f5", "image encoder\nViT (MAE pre-trained)\nrun ONCE per image", fs=7.5)
    _box(ax, 4.2, 1.0, 1.6, 1.2, "#dbe7f7", "image embedding\n256 x 64 x 64", fs=7.5)
    _box(ax, 4.2, 2.8, 1.6, 0.9, "#cde8cf", "prompt encoder\npoints / box / mask\n(SAM 3: text concept)", fs=7)
    _box(ax, 6.4, 1.6, 1.8, 1.4, "#f8d0b8", "mask decoder\n2-layer two-way\ntransformer (~ms)", fs=7.5)
    _box(ax, 8.8, 1.6, 2.1, 1.4, "#fde2a2", "3 candidate masks\n(whole / part / subpart)\n+ predicted IoU each", fs=7.5)
    _arrow(ax, 1.4, 1.6, 1.9, 1.6); _arrow(ax, 3.7, 1.6, 4.2, 1.6); _arrow(ax, 5.8, 1.6, 6.4, 2.0)
    _arrow(ax, 5.8, 3.2, 6.4, 2.7); _arrow(ax, 8.2, 2.3, 8.8, 2.3)
    ax.text(0.0, 0.55, "Data engine: SAM labelled SA-1B itself (11M images, 1.1B masks); SAM 2 adds a memory for video;\n"
                       "SAM 3 accepts short noun phrases ('striped cat') and returns ALL matching instances.", fontsize=7.5, va="top")
    ax.set_xlim(-0.2, 11.2); ax.set_ylim(-0.2, 3.9)


def draw_field_map(ax):
    """Map of the field in 2026: tasks x what to start with."""
    ax.axis("off")
    rows = [
        ("Classification", "ResNet-50 / ConvNeXt-T fine-tune", "DINOv2/v3 linear probe or k-NN; CLIP / SigLIP zero-shot", "fine-tune a pretrained backbone"),
        ("Detection", "Faster R-CNN, RetinaNet, FCOS", "YOLO (v8-v12), RT-DETR, D-FINE; Grounding DINO / OWLv2 open-vocab", "YOLO or RT-DETR fine-tune"),
        ("Semantic seg.", "U-Net, DeepLabv3+", "SegFormer, Mask2Former; DINOv2 + linear head", "U-Net w/ pretrained encoder"),
        ("Instance seg.", "Mask R-CNN", "YOLO-seg, Mask2Former; SAM / SAM 2 masks from boxes", "Mask R-CNN or YOLO-seg"),
        ("Promptable / open-vocab", "-", "SAM 2 / SAM 3, Grounding DINO + SAM, Florence-2", "SAM 2 + a detector"),
        ("Video", "2-stream, 3D conv", "SAM 2 tracking, VideoMAE, V-JEPA", "per-frame model + tracker"),
        ("Depth / 3D", "stereo, SfM", "Depth Anything v2, DUSt3R / VGGT, Gaussian splatting", "Depth Anything for monocular"),
        ("Image + language", "captioning CNN+RNN", "VLMs (Qwen-VL, PaliGemma, LLaVA, GPT/Claude/Gemini)", "a VLM API, then distil"),
    ]
    cols = ["task", "classic (2015-2020)", "foundation-model era (2023-2026)", "recommended first try"]
    widths = [1.9, 2.6, 4.6, 2.6]
    xs = np.cumsum([0] + widths)
    hdr_c = ["#4C78A8", "#9D755D", "#54A24B", "#F58518"]
    for j, c in enumerate(cols):
        _box(ax, xs[j] + 0.03, len(rows) + 0.05, widths[j] - 0.06, 0.8, hdr_c[j], c, fs=9, tc="white", ec="none")
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        for j, txt in enumerate(r):
            fc = ["#e9eef6", "#f3ece6", "#e7f3e6", "#fdeedd"][j]
            _box(ax, xs[j] + 0.03, y + 0.05, widths[j] - 0.06, 0.88, fc, txt, fs=7.5 if j != 0 else 8.5, ec="none")
    ax.set_xlim(-0.1, xs[-1] + 0.1); ax.set_ylim(-0.1, len(rows) + 1.0)


def draw_variants(fig=None):
    """Six schematic panels: U-Net++, Attention U-Net, DeepLabv3+, SegFormer, Mask2Former, diffusion U-Net."""
    import matplotlib.pyplot as plt
    fig = fig or plt.figure(figsize=(16, 9.5))
    axes = fig.subplots(2, 3)
    for ax in axes.flat:
        ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    E, D, O, Y, G = "#cfe0f5", "#cde8cf", "#f8d0b8", "#fde2a2", "#e8e8e8"

    # 1. U-Net++ : nested dense skips
    ax = axes[0, 0]; ax.set_title("U-Net++ (2018): nested, dense skip paths", fontsize=10)
    pos = {}
    for i in range(4):
        for j in range(4 - i):
            x, y = 1.0 + j * 2.1 + i * 1.05, 5.8 - i * 1.6
            pos[i, j] = (x, y)
            _box(ax, x - 0.45, y - 0.35, 0.9, 0.7, E if j == 0 else D, f"X{i},{j}", fs=7)
    for i in range(4):
        for j in range(1, 4 - i):
            for jj in range(j):
                (x0, y0), (x1, y1) = pos[i, jj], pos[i, j]
                _arrow(ax, x0 + 0.45, y0 + 0.1 * (jj - j), x1 - 0.45, y1, color="#F58518", lw=0.8)
            xa, ya = pos[i + 1, j - 1]
            _arrow(ax, xa, ya + 0.35, pos[i, j][0], pos[i, j][1] - 0.35, color="#54A24B", lw=0.8)
    for i in range(3):
        _arrow(ax, pos[i, 0][0], pos[i, 0][1] - 0.35, pos[i + 1, 0][0], pos[i + 1, 0][1] + 0.35, color="#4C78A8", lw=0.8)
    ax.text(0.3, 0.3, "every decoder node sees ALL shallower nodes of its level;\ndeep supervision on X0,1..X0,3 allows pruning at inference", fontsize=7.5)

    # 2. Attention U-Net : gated skip
    ax = axes[0, 1]; ax.set_title("Attention U-Net (2018): gated skips", fontsize=10)
    _box(ax, 0.3, 4.6, 2.0, 1.0, E, "encoder skip x\n(fine, 'where')", fs=8)
    _box(ax, 0.3, 1.6, 2.0, 1.0, O, "gating g from the\ncoarser decoder ('what')", fs=8)
    _box(ax, 3.4, 3.1, 2.2, 1.2, Y, "attention gate\nsigmoid(psi(relu(\nWx x + Wg g)))", fs=7.5)
    _box(ax, 6.6, 3.1, 1.4, 1.2, D, "alpha * x", fs=8)
    _box(ax, 8.4, 3.1, 1.4, 1.2, D, "concat\n+ convs", fs=8)
    _arrow(ax, 2.3, 5.1, 3.4, 4.0); _arrow(ax, 2.3, 2.1, 3.4, 3.4); _arrow(ax, 5.6, 3.7, 6.6, 3.7); _arrow(ax, 8.0, 3.7, 8.4, 3.7)
    _arrow(ax, 2.3, 5.3, 7.3, 4.3, color="#F58518", ls="--")
    ax.text(0.3, 0.5, "alpha in [0, 1] per pixel suppresses irrelevant\nbackground in the skip (popular in medical imaging)", fontsize=7.5)

    # 3. DeepLabv3+
    ax = axes[0, 2]; ax.set_title("DeepLabv3+ (2018): ASPP + light decoder", fontsize=10)
    _box(ax, 0.2, 2.8, 1.6, 1.4, G, "image", fs=8)
    _box(ax, 2.2, 2.6, 1.8, 1.8, E, "dilated\nResNet / Xception\nOS 16", fs=7.5)
    _box(ax, 4.5, 4.4, 1.9, 1.8, Y, "ASPP\n1x1, rates 6/12/18,\nimage pooling", fs=7)
    _box(ax, 4.5, 0.9, 1.9, 1.2, E, "low-level feats\n1/4, 1x1 conv", fs=7.5)
    _box(ax, 7.0, 2.6, 1.3, 1.8, D, "up x4,\nconcat,\n3x3 convs", fs=7.5)
    _box(ax, 8.7, 2.9, 1.1, 1.2, O, "up x4\nmask", fs=7.5)
    _arrow(ax, 1.8, 3.5, 2.2, 3.5); _arrow(ax, 4.0, 3.9, 4.5, 5.2); _arrow(ax, 4.0, 3.0, 4.5, 1.6)
    _arrow(ax, 6.4, 5.2, 7.0, 4.1); _arrow(ax, 6.4, 1.5, 7.0, 2.9); _arrow(ax, 8.3, 3.5, 8.7, 3.5)
    ax.text(0.2, 0.2, "context from atrous convs, boundaries from ONE skip", fontsize=7.5)

    # 4. SegFormer
    ax = axes[1, 0]; ax.set_title("SegFormer (2021): transformer encoder + all-MLP decoder", fontsize=10)
    for k, (s, c) in enumerate([("1/4", 64), ("1/8", 128), ("1/16", 320), ("1/32", 512)]):
        _box(ax, 0.3 + k * 1.15, 1.2 + k * 0.9, 1.0, 1.0, E, f"stage {k + 1}\n{s}, {c}", fs=6.5)
        _arrow(ax, 1.3 + k * 1.15, 1.7 + k * 0.9, 6.0, 3.6, color="#F58518", lw=0.8)
    _box(ax, 6.0, 2.8, 1.7, 1.6, D, "MLP each\n-> up to 1/4\n-> concat", fs=7.5)
    _box(ax, 8.1, 2.9, 1.6, 1.4, O, "MLP\n-> mask 1/4", fs=7.5)
    _arrow(ax, 7.7, 3.6, 8.1, 3.6)
    ax.text(0.3, 0.3, "efficient self-attention (reduced keys), overlapping\npatch embeddings, no positional encoding", fontsize=7.5)

    # 5. Mask2Former
    ax = axes[1, 1]; ax.set_title("Mask2Former (2022): queries predict masks", fontsize=10)
    _box(ax, 0.2, 4.6, 2.0, 1.2, E, "backbone +\npixel decoder", fs=8)
    _box(ax, 0.2, 2.2, 2.0, 1.2, D, "per-pixel\nembeddings E", fs=8)
    _box(ax, 3.0, 4.6, 2.4, 1.2, Y, "N learned queries\n(e.g. 100)", fs=8)
    _box(ax, 3.0, 2.2, 2.4, 1.6, O, "transformer decoder\nmasked attention\n(to its own mask)", fs=7.5)
    _box(ax, 6.2, 3.8, 3.5, 1.0, G, "class_i (K+1, incl. 'no object')", fs=7.5)
    _box(ax, 6.2, 2.2, 3.5, 1.0, G, "mask_i = sigmoid(q_i . E)", fs=7.5)
    _arrow(ax, 1.2, 4.6, 1.2, 3.4); _arrow(ax, 4.2, 4.6, 4.2, 3.8); _arrow(ax, 2.2, 2.8, 3.0, 2.8)
    _arrow(ax, 5.4, 3.3, 6.2, 4.3); _arrow(ax, 5.4, 2.9, 6.2, 2.7)
    ax.text(0.2, 0.4, "Hungarian matching as in DETR: ONE model for semantic,\ninstance and panoptic segmentation", fontsize=7.5)

    # 6. Diffusion U-Net
    ax = axes[1, 2]; ax.set_title("U-Net in diffusion models (DDPM 2020, Stable Diffusion)", fontsize=10)
    xs = [0.5, 2.0, 3.5]
    for k, x in enumerate(xs):
        _box(ax, x, 5.0 - k * 1.3, 1.2, 0.8, E, "ResBlock", fs=7)
        _box(ax, 8.6 - x, 5.0 - k * 1.3, 1.2, 0.8, D, "ResBlock", fs=7)
        _arrow(ax, x + 1.2, 5.4 - k * 1.3, 8.6 - x, 5.4 - k * 1.3, color="#F58518", lw=0.8)
    _box(ax, 4.1, 0.9, 1.6, 0.9, Y, "self-attn\n(+ cross-attn text)", fs=6.5)
    _box(ax, 0.3, 0.6, 2.6, 1.1, O, "timestep t -> sinusoid\n-> MLP, ADDED in\nevery ResBlock", fs=7)
    ax.text(5.0, 6.4, "input: noisy image x_t ; output: predicted noise (same size)", ha="center", fontsize=7.5)
    ax.text(6.2, 1.1, "same encoder / decoder / skips\nas the segmentation U-Net", fontsize=7.5)
    fig.tight_layout()
    return fig


def pet_seg_transforms(size: int = 128):
    """(train_tf, eval_tf) for Pet segmentation: joint image + mask transforms (torchvision v2)."""
    from torchvision.transforms import v2
    from .torchkit import IMAGENET_MEAN, IMAGENET_STD
    train_tf = v2.Compose([v2.ToImage(), v2.RandomResizedCrop((size, size), scale=(0.5, 1.0), antialias=True),
                           v2.RandomHorizontalFlip(), v2.ToDtype(torch.float32, scale=True),
                           v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    eval_tf = v2.Compose([v2.ToImage(), v2.Resize((size, size), antialias=True), v2.ToDtype(torch.float32, scale=True),
                          v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return train_tf, eval_tf


def pet_unet_optimizer(model, lr_decoder: float = 1e-3, lr_encoder: float = 2e-4, weight_decay: float = 1e-4):
    """AdamW with a 5x smaller learning rate for the pretrained encoder (discriminative LRs, D1-04)."""
    return torch.optim.AdamW([{"params": list(model.encoder_parameters()), "lr": lr_encoder},
                              {"params": model.decoder_parameters(), "lr": lr_decoder}], weight_decay=weight_decay)


def draw_instance_alternatives(fig=None):
    """Three schematics: YOLO-seg / YOLACT prototypes, Mask2Former queries, SAM as a mask generator."""
    import matplotlib.pyplot as plt
    fig = fig or plt.figure(figsize=(16, 4.6))
    axes = fig.subplots(1, 3)
    for ax in axes:
        ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    E, D, O, Y, G = "#cfe0f5", "#cde8cf", "#f8d0b8", "#fde2a2", "#e8e8e8"
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:24, 0:24] / 23.0
    protos = [np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / s) for cx, cy, s in [(0.3, 0.4, 0.08), (0.7, 0.5, 0.1), (0.5, 0.2, 0.3), (0.5, 0.8, 0.05)]]
    ax = axes[0]; ax.set_title("YOLACT / YOLO-seg: prototypes x coefficients", fontsize=10)
    for k, p in enumerate(protos):
        ax.imshow(p, extent=(0.3 + k * 1.25, 1.35 + k * 1.25, 3.6, 4.65), cmap="viridis", aspect="auto")
    ax.text(2.8, 4.85, "k prototype masks for the WHOLE image (one per image)", ha="center", fontsize=7.5)
    _box(ax, 0.3, 1.9, 5.2, 0.9, D, "per detection: box, class, k coefficients c", fs=7.5)
    comb = 1 / (1 + np.exp(-(6 * (0.9 * protos[0] - 0.4 * protos[2] + 0.2 * protos[3]) - 2)))
    ax.imshow(comb, extent=(6.2, 8.4, 2.4, 4.6), cmap="gray", aspect="auto"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.add_patch(__import__("matplotlib").patches.Rectangle((6.35, 2.9), 1.3, 1.5, fill=False, ec="#E45756", lw=2))
    _arrow(ax, 5.2, 3.9, 6.1, 3.6); _arrow(ax, 5.1, 2.4, 6.1, 3.0)
    ax.text(7.3, 2.0, "mask = sigmoid(c . P),\ncropped to the box", ha="center", fontsize=8)
    ax.text(0.3, 0.6, "one-stage, real time; mask quality limited by the\nprototype resolution (1/4)", fontsize=7.5)
    ax = axes[1]; ax.set_title("Mask2Former: one model, three tasks", fontsize=10)
    for k in range(5):
        _box(ax, 0.3 + k * 0.9, 4.4, 0.7, 0.7, Y, f"q{k}", fs=7)
    _box(ax, 0.3, 2.6, 4.6, 1.1, O, "transformer decoder\n(masked attention)", fs=8)
    _arrow(ax, 2.5, 4.4, 2.5, 3.7)
    for k, (c, t) in enumerate([("person", "#4C78A8"), ("person", "#54A24B"), ("road (stuff)", "#9D755D"), ("no object", "#BAB0AC")]):
        _box(ax, 5.4, 4.6 - k * 1.05, 4.2, 0.8, G, f"query {k}: class '{c}' + its mask", fs=7.5)
    _arrow(ax, 4.7, 3.15, 5.4, 3.3)
    ax.text(0.3, 1.5, "semantic: merge masks by class\ninstance: keep 'thing' queries\npanoptic: both; no NMS, no anchors", fontsize=7.5, va="top")
    ax = axes[2]; ax.set_title("SAM: class-agnostic masks from prompts", fontsize=10)
    _box(ax, 0.3, 3.3, 2.2, 1.6, E, "image\nencoder\n(once)", fs=8)
    _box(ax, 0.3, 1.0, 2.2, 1.4, D, "prompts: points,\nboxes (e.g. from\na detector)", fs=7.5)
    _box(ax, 3.3, 2.2, 2.2, 1.6, O, "mask decoder\n(milliseconds)", fs=8)
    _box(ax, 6.3, 2.2, 3.4, 1.6, Y, "masks + IoU scores,\nNO class names", fs=8)
    _arrow(ax, 2.5, 4.1, 3.3, 3.4); _arrow(ax, 2.5, 1.7, 3.3, 2.6); _arrow(ax, 5.5, 3.0, 6.3, 3.0)
    ax.text(0.3, 0.3, "detector boxes + SAM = instance segmentation without\nmask training; see D3-04", fontsize=7.5)
    fig.tight_layout()
    return fig


# =============================================================================
# More ViT helpers (D3-04)
# =============================================================================
@torch.no_grad()
def torchvision_vit_all_attentions(model, x: torch.Tensor):
    """Attention weights of EVERY encoder layer of torchvision's vit_b_16 in one forward pass.

    Returns a list of (B, heads, N+1, N+1) tensors (index 0 = class token).
    """
    store = []
    hooks = [blk.ln_1.register_forward_hook(lambda m, i, o: store.append(o)) for blk in model.encoder.layers]
    model(x)
    for h in hooks:
        h.remove()
    out = []
    for blk, y in zip(model.encoder.layers, store):
        _, w = blk.self_attention(y, y, y, need_weights=True, average_attn_weights=False)
        out.append(w)
    return out


def attention_rollout(attentions, residual: float = 0.5) -> torch.Tensor:
    """Abnar & Zuidema (2020): multiply head-averaged attention (+ identity for the residual) through the layers.

    Returns (B, N+1, N+1); row 0 is how much the class token 'sees' of each input token.
    """
    rollout = None
    for a in attentions:
        a = a.mean(1)
        a = residual * a + (1 - residual) * torch.eye(a.shape[-1])
        a = a / a.sum(-1, keepdim=True)
        rollout = a if rollout is None else a @ rollout
    return rollout


def mean_attention_distance(attentions, grid: int, patch: int = 16) -> np.ndarray:
    """(layers, heads) mean distance in pixels between a query patch and the patches it attends to."""
    ys, xs = np.divmod(np.arange(grid * grid), grid)
    pos = torch.tensor(np.stack([ys, xs], 1), dtype=torch.float32) * patch
    dist = torch.cdist(pos, pos)                                         # (N, N)
    out = []
    for a in attentions:
        p = a[:, :, 1:, 1:]                                              # patch-to-patch part
        p = p / p.sum(-1, keepdim=True)
        out.append((p * dist).sum(-1).mean((0, 2)).numpy())              # mean over batch and queries
    return np.stack(out)


@torch.no_grad()
def extract_features(fn, loader, max_batches=None):
    """Run fn(images) -> (B, D) over a loader of (image, label). Returns (features, labels)."""
    feats, labels = [], []
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        feats.append(fn(x).float().cpu())
        labels.append(torch.as_tensor(y))
    return torch.cat(feats), torch.cat(labels)


def draw_swin(ax_pair):
    """Swin Transformer: window attention (layer l) and shifted windows (layer l+1) on an 8x8 patch grid."""
    from matplotlib.patches import Rectangle
    for k, ax in enumerate(ax_pair):
        ax.set_xlim(0, 8); ax.set_ylim(8, 0); ax.set_aspect("equal")
        for i in range(9):
            ax.plot([0, 8], [i, i], color="#dddddd", lw=0.8); ax.plot([i, i], [0, 8], color="#dddddd", lw=0.8)
        shift = 0 if k == 0 else 2
        cols = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2", "#EECA3B", "#9D755D", "#FF9DA6"]
        c = 0
        edges = [0, 4, 8] if shift == 0 else [0, 2, 6, 8]
        for yi in range(len(edges) - 1):
            for xi in range(len(edges) - 1):
                y0, y1, x0, x1 = edges[yi], edges[yi + 1], edges[xi], edges[xi + 1]
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=cols[c % len(cols)], alpha=0.35, ec="k", lw=2))
                c += 1
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("layer l: 4 windows of 4x4 patches\n(attention only inside a window)" if k == 0 else
                     "layer l+1: windows shifted by 2\n(information crosses window borders)", fontsize=9)
