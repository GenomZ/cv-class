"""Day-2 (object detection) shared code: models, targets, training and evaluation glue.

Everything the four Day-2 notebooks share lives here so the notebooks stay about
the *ideas*. Reference implementations of every exercise are included (the
notebooks fall back to them if a student skipped an exercise), plus:

* synthetic shapes as tensors (`shapes_tensors`, `grid_target_tensor`)
* `MiniDetector`  : the YOLOv1-style dense detector of D2-03 and its loss / decoding
* `TinyRPN`       : a small region-proposal network trained on shapes in D2-02
* `RPNHead`, `FPN`, `RetinaHead`, `roi_align`, `dense_loss`, `detr_cost`, `hungarian_match`
* torchvision glue: `train_detector`, `predict_detector`, `evaluate_detector`,
  `penn_fudan_splits`, `load_or_train`
* `mini_selective_search` : a 30-line, NumPy-only version of Selective Search

Box convention: xyxy in pixels unless a name says otherwise (`cxcywh`, `norm`).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import boxes as B
from . import shapes as S

NUM_SHAPE_CLASSES = len(S.CLASS_NAMES)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def fastest_num_threads(step_fn, candidates=None, repeats: int = 2, log=print):
    """Time `step_fn()` (one training step) with several torch thread counts and return the fastest.

    Tiny models on a CPU are dominated by thread-synchronisation overhead, and on a busy machine
    all-cores can be several times *slower* than 1-2 threads. Benchmark, do not guess.
    """
    default = torch.get_num_threads()
    candidates = candidates or sorted({default, max(1, default // 2), 2, 1}, reverse=True)
    times = {}
    for n in candidates:
        torch.set_num_threads(n)
        step_fn()                                         # warm-up
        t0 = time.perf_counter()
        for _ in range(repeats):
            step_fn()
        times[n] = (time.perf_counter() - t0) / repeats
    torch.set_num_threads(default)
    best = min(times, key=times.get)
    log("threads -> s/step: " + ", ".join(f"{n}: {t:.2f}" for n, t in times.items()) + f"  => using {best}")
    return best


def draw_detections(ax, boxes_xyxy, labels, scores=None, names=None, lw=1.6):
    """Draw boxes coloured by integer class label, with 'name score' tags."""
    import matplotlib.patches as mpatches
    from . import viz
    boxes_xyxy = np.asarray(boxes_xyxy, float).reshape(-1, 4)
    for k, b in enumerate(boxes_xyxy):
        c = viz.color(int(labels[k]))
        ax.add_patch(mpatches.Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec=c, lw=lw))
        txt = (names[int(labels[k])] if names is not None else str(int(labels[k]))) + (f" {scores[k]:.2f}" if scores is not None else "")
        ax.text(b[0], b[1] - 2, txt, color="white", fontsize=7, bbox=dict(fc=c, ec="none", pad=0.8, alpha=0.85))


# ---------------------------------------------------------------------------
# Synthetic shapes as tensors
# ---------------------------------------------------------------------------
def shapes_tensors(n: int = 1000, size: int = 128, seed: int = 0, **kw):
    """Generate `n` shapes images. Returns (images (n,3,H,W) float tensor, boxes list, labels list)."""
    X, Bx, L, _ = S.make_dataset(n, size, seed, **kw)
    return torch.from_numpy(X).permute(0, 3, 1, 2).contiguous(), Bx, L


def grid_target_tensor(boxes_list, labels_list, grid: int = 8, size: int = 128, num_classes: int = 3):
    """Stack `shapes.grid_targets` for a list of images -> (n, 5 + C, grid, grid) tensor (channels first)."""
    T = np.stack([S.grid_targets(b, l, grid, size, num_classes) for b, l in zip(boxes_list, labels_list)])
    return torch.from_numpy(T).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# D2-03: the mini dense detector (YOLOv1 style, one box per cell)
# ---------------------------------------------------------------------------
def conv_bn(c_in, c_out, stride=1):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class MiniDetector(nn.Module):
    """128 px image -> (5 + C, 8, 8) grid.

    Output channels per cell (same layout as `shapes.grid_targets`):
      0     objectness *logit*
      1..4  box (cx, cy, w, h) normalised to the image, already decoded:
            cx = (col + sigmoid(tx)) / G, cy = (row + sigmoid(ty)) / G, w = sigmoid(tw), h = sigmoid(th)
      5..   class logits
    """

    def __init__(self, num_classes: int = 3, width: int = 16):
        super().__init__()
        w = width
        self.backbone = nn.Sequential(                    # 4 stride-2 blocks: 128 -> 64 -> 32 -> 16 -> 8
            conv_bn(3, w, 2),
            conv_bn(w, 2 * w, 2), conv_bn(2 * w, 2 * w),
            conv_bn(2 * w, 4 * w, 2), conv_bn(4 * w, 4 * w),
            conv_bn(4 * w, 8 * w, 2), conv_bn(8 * w, 8 * w),
        )
        self.head = nn.Conv2d(8 * w, 5 + num_classes, 1)
        self.num_classes = num_classes

    def forward(self, x):
        raw = self.head(self.backbone(x))                 # (N, 5 + C, G, G)
        return decode_raw_grid(raw)


def decode_raw_grid(raw):
    """Turn raw head outputs into the target layout (boxes decoded, logits untouched)."""
    N, _, G, _ = raw.shape
    rows = torch.arange(G, device=raw.device).view(1, G, 1).float()
    cols = torch.arange(G, device=raw.device).view(1, 1, G).float()
    cx = (cols + torch.sigmoid(raw[:, 1])) / G
    cy = (rows + torch.sigmoid(raw[:, 2])) / G
    wh = torch.sigmoid(raw[:, 3:5])
    return torch.cat([raw[:, :1], cx[:, None], cy[:, None], wh, raw[:, 5:]], 1)


def dense_loss(pred, target, w_obj: float = 1.0, w_box: float = 5.0, w_cls: float = 1.0):
    """Reference three-term loss of the mini detector.

    pred, target: (N, 5 + C, G, G). pred[:, 0] and pred[:, 5:] are logits, pred[:, 1:5] decoded boxes.
    obj : BCE-with-logits over *all* cells (mean)
    box : L1 over the 4 box numbers, summed over coordinates, mean over positive cells
    cls : cross-entropy over classes, mean over positive cells
    Returns (total, {"obj": ..., "box": ..., "cls": ...}).
    """
    obj_t = target[:, 0]
    pos = obj_t > 0.5
    l_obj = F.binary_cross_entropy_with_logits(pred[:, 0], obj_t)
    n_pos = pos.sum()
    if n_pos > 0:
        pb = pred[:, 1:5].permute(0, 2, 3, 1)[pos]          # (P, 4)
        tb = target[:, 1:5].permute(0, 2, 3, 1)[pos]
        l_box = (pb - tb).abs().sum(-1).mean()
        pc = pred[:, 5:].permute(0, 2, 3, 1)[pos]            # (P, C)
        tc = target[:, 5:].permute(0, 2, 3, 1)[pos].argmax(-1)
        l_cls = F.cross_entropy(pc, tc)
    else:
        l_box = pred.sum() * 0.0
        l_cls = pred.sum() * 0.0
    total = w_obj * l_obj + w_box * l_box + w_cls * l_cls
    return total, {"obj": l_obj, "box": l_box, "cls": l_cls}


def grid_to_probs(pred):
    """(N, 5+C, G, G) model output -> (N, G, G, 5+C) numpy with sigmoid objectness and softmax classes."""
    p = pred.detach().float().cpu().clone()
    p[:, 0] = torch.sigmoid(p[:, 0])
    p[:, 5:] = torch.softmax(p[:, 5:], 1)
    return p.permute(0, 2, 3, 1).numpy()


def decode_predictions(pred, size: int = 128, score_threshold: float = 0.3, iou_threshold: float = 0.5):
    """Decode a batch of grid predictions into per-image dicts (boxes, scores, labels), class-wise NMS."""
    out = []
    for g in grid_to_probs(pred):
        bx, sc, lb = S.decode_grid(g, size, score_threshold)
        if len(bx):
            sc = sc * g[..., 5:].max(-1)[g[..., 0] >= score_threshold]   # score = objectness x class prob
            keep = B.batched_nms(bx, sc, lb, iou_threshold)
            bx, sc, lb = bx[keep], sc[keep], lb[keep]
        out.append({"boxes": bx.reshape(-1, 4), "scores": np.asarray(sc).reshape(-1), "labels": np.asarray(lb, int).reshape(-1)})
    return out


def train_mini_detector(model, images, targets, loss_fn=dense_loss, epochs: int = 10, batch_size: int = 32,
                        lr: float = 3e-3, max_seconds: float | None = None, device="cpu", seed: int = 0,
                        log=print, augment_flip: bool = True, threads="auto"):
    """Custom training loop (AdamW + one-cycle). Returns history dict of per-epoch mean losses.

    threads="auto" benchmarks a few torch thread counts on one batch first (CPU only) and trains with
    the fastest; the previous setting is restored afterwards.
    """
    torch.manual_seed(seed)
    model.to(device).train()
    old_threads = torch.get_num_threads()
    if threads == "auto" and str(device) == "cpu":
        xb, tb = images[:batch_size].to(device), targets[:batch_size].to(device)
        def _step():
            model.zero_grad(); loss_fn(model(xb), tb)[0].backward()
        threads = fastest_num_threads(_step, log=log)
        model.zero_grad()
    if isinstance(threads, int):
        torch.set_num_threads(threads)
    n = len(images)
    steps = epochs * math.ceil(n / batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    hist = {"total": [], "obj": [], "box": [], "cls": [], "seconds": []}
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n)
        sums = {"total": 0.0, "obj": 0.0, "box": 0.0, "cls": 0.0}
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x, t = images[idx].to(device), targets[idx].to(device)
            if augment_flip and torch.rand(1).item() < 0.5:   # horizontal flip: image, grid columns and cx
                x = x.flip(-1); t = t.flip(-1).clone(); t[:, 1] = torch.where(t[:, 0] > 0, 1 - t[:, 1], t[:, 1])
            pred = model(x)
            total, parts = loss_fn(pred, t)
            opt.zero_grad(); total.backward(); opt.step(); sched.step()
            sums["total"] += float(total.detach()); nb += 1
            for k in ("obj", "box", "cls"):
                sums[k] += float(parts[k].detach())
        for k in sums:
            hist[k].append(sums[k] / nb)
        hist["seconds"].append(time.time() - t0)
        log(f"epoch {ep + 1:2d}/{epochs}  loss {hist['total'][-1]:.3f}  (obj {hist['obj'][-1]:.3f}  box {hist['box'][-1]:.3f}  "
            f"cls {hist['cls'][-1]:.3f})  {time.time() - t0:5.1f}s")
        if max_seconds and time.time() - t0 > max_seconds and ep < epochs - 1:
            log(f"stopping early: time budget of {max_seconds:.0f}s reached")
            break
    torch.set_num_threads(old_threads)
    model.eval()
    return hist


@torch.no_grad()
def predict_batches(model, images, batch_size: int = 64, device="cpu"):
    model.eval()
    return torch.cat([model(images[i:i + batch_size].to(device)).cpu() for i in range(0, len(images), batch_size)])


def gt_dicts(boxes_list, labels_list):
    return [{"boxes": np.asarray(b).reshape(-1, 4), "labels": np.asarray(l, int)} for b, l in zip(boxes_list, labels_list)]


# ---------------------------------------------------------------------------
# D2-02: RPN head, tiny RPN on shapes, RoIAlign reference
# ---------------------------------------------------------------------------
class RPNHead(nn.Module):
    """3x3 conv + ReLU, then two sibling 1x1 convs: objectness (A) and box deltas (4A)."""

    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, 4 * num_anchors, 1)

    def forward(self, x):
        h = F.relu(self.conv(x))
        return self.cls_logits(h), self.bbox_pred(h)


def flatten_rpn_outputs(obj, deltas):
    """(N, A, H, W), (N, 4A, H, W) -> (N, H*W*A), (N, H*W*A, 4) ordered (row, col, anchor) like make_anchors."""
    N, A, H, W = obj.shape
    obj = obj.permute(0, 2, 3, 1).reshape(N, -1)
    deltas = deltas.view(N, A, 4, H, W).permute(0, 3, 4, 1, 2).reshape(N, -1, 4)
    return obj, deltas


def encode_boxes_torch(gt, anchors):
    """Torch version of boxes.encode_boxes (weights 1)."""
    aw, ah = anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1]
    ax, ay = anchors[:, 0] + aw / 2, anchors[:, 1] + ah / 2
    gw, gh = gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]
    gx, gy = gt[:, 0] + gw / 2, gt[:, 1] + gh / 2
    return torch.stack([(gx - ax) / aw, (gy - ay) / ah, torch.log(gw / aw), torch.log(gh / ah)], -1)


def decode_boxes_torch(d, anchors, clip=math.log(1000.0 / 16)):
    aw, ah = anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1]
    ax, ay = anchors[:, 0] + aw / 2, anchors[:, 1] + ah / 2
    cx, cy = ax + d[:, 0] * aw, ay + d[:, 1] * ah
    w, h = aw * torch.exp(d[:, 2].clamp(max=clip)), ah * torch.exp(d[:, 3].clamp(max=clip))
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


class TinyRPN(nn.Module):
    """A 3-block backbone (stride 8) + an RPN head. 128 px input -> 16 x 16 feature map."""

    def __init__(self, head_cls=RPNHead, channels: int = 64, scales=(16, 32, 48), ratios=(0.5, 1.0, 2.0), size: int = 128):
        super().__init__()
        c = channels
        self.backbone = nn.Sequential(conv_bn(3, c // 4, 2), conv_bn(c // 4, c // 2, 2), conv_bn(c // 2, c // 2),
                                      conv_bn(c // 2, c, 2), conv_bn(c, c))
        self.stride = 8
        self.fsize = size // self.stride
        self.anchors = torch.from_numpy(B.make_anchors((self.fsize, self.fsize), self.stride, scales, ratios)).float()
        self.num_anchors = len(scales) * len(ratios)
        self.head = head_cls(c, self.num_anchors)

    def forward(self, x):
        obj, deltas = self.head(self.backbone(x))
        return flatten_rpn_outputs(obj, deltas)


def rpn_targets(anchors_np, boxes_list, pos_iou=0.5, neg_iou=0.3):
    """Matching labels (1 / 0 / -1) and regression targets for every image (precomputed once)."""
    labels, deltas = [], []
    for gt in boxes_list:
        idx, lab = B.match_anchors(anchors_np, gt, pos_iou, neg_iou)
        d = np.zeros((len(anchors_np), 4), np.float32)
        if len(gt):
            d[lab == 1] = B.encode_boxes(np.asarray(gt)[idx[lab == 1]], anchors_np[lab == 1])
        labels.append(lab); deltas.append(d)
    return torch.from_numpy(np.stack(labels)), torch.from_numpy(np.stack(deltas))


def sample_anchors(labels, batch_per_image: int = 128, pos_fraction: float = 0.25, generator=None):
    """RPN minibatch sampling: up to 25% positives, rest negatives. Returns a bool mask (N, A)."""
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for i in range(labels.shape[0]):
        pos = torch.where(labels[i] == 1)[0]; neg = torch.where(labels[i] == 0)[0]
        n_pos = min(len(pos), int(batch_per_image * pos_fraction))
        n_neg = min(len(neg), batch_per_image - n_pos)
        pos = pos[torch.randperm(len(pos), generator=generator)[:n_pos]]
        neg = neg[torch.randperm(len(neg), generator=generator)[:n_neg]]
        mask[i, pos] = True; mask[i, neg] = True
    return mask


def rpn_loss(obj, deltas, labels, target_deltas, sample_mask, beta: float = 1.0 / 9):
    """BCE objectness on sampled anchors + smooth-L1 on positive anchors' deltas."""
    l_obj = F.binary_cross_entropy_with_logits(obj[sample_mask], (labels[sample_mask] == 1).float())
    pos = sample_mask & (labels == 1)
    if pos.any():
        l_box = F.smooth_l1_loss(deltas[pos], target_deltas[pos], beta=beta, reduction="sum") / sample_mask.sum()
    else:
        l_box = deltas.sum() * 0.0
    return l_obj, l_box


def train_tiny_rpn(model, images, labels, target_deltas, epochs: int = 6, batch_size: int = 32, lr: float = 3e-3,
                   max_seconds: float | None = None, log=print, seed: int = 0, threads="auto"):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    old_threads = torch.get_num_threads()
    if threads == "auto":
        xb, lb, db = images[:batch_size], labels[:batch_size], target_deltas[:batch_size]
        mb = sample_anchors(lb, generator=torch.Generator().manual_seed(seed))
        def _step():
            model.zero_grad(); o, d = model(xb); sum(rpn_loss(o, d, lb, db, mb)).backward()
        threads = fastest_num_threads(_step, log=log)
        model.zero_grad()
    if isinstance(threads, int):
        torch.set_num_threads(threads)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(images) / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    hist = {"obj": [], "box": []}
    t0 = time.time()
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(images))
        so = sb = 0.0; nb = 0
        for i in range(0, len(images), batch_size):
            idx = perm[i:i + batch_size]
            obj, d = model(images[idx])
            m = sample_anchors(labels[idx], generator=gen)
            l_obj, l_box = rpn_loss(obj, d, labels[idx], target_deltas[idx], m)
            loss = l_obj + l_box
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            so += float(l_obj.detach()); sb += float(l_box.detach()); nb += 1
        hist["obj"].append(so / nb); hist["box"].append(sb / nb)
        log(f"epoch {ep + 1}/{epochs}  objectness {so / nb:.3f}  box {sb / nb:.3f}  {time.time() - t0:5.1f}s")
        if max_seconds and time.time() - t0 > max_seconds:
            log(f"stopping early: time budget of {max_seconds:.0f}s reached")
            break
    torch.set_num_threads(old_threads)
    model.eval()
    return hist


@torch.no_grad()
def rpn_proposals(model, image, pre_nms_top_n: int = 300, post_nms_top_n: int = 20, nms_iou: float = 0.5,
                  min_size: float = 4.0, return_steps: bool = False):
    """Proposal generation for one image (3, H, W): decode -> clip -> remove tiny -> top-k -> NMS -> top-N."""
    from torchvision.ops import nms, clip_boxes_to_image, remove_small_boxes
    model.eval()
    obj, d = model(image[None])
    scores = torch.sigmoid(obj[0])
    bx = decode_boxes_torch(d[0], model.anchors)
    steps = {"all": (bx, scores)}
    bx = clip_boxes_to_image(bx, image.shape[-2:])
    keep = remove_small_boxes(bx, min_size); bx, scores = bx[keep], scores[keep]
    steps["clipped"] = (bx, scores)
    top = scores.argsort(descending=True)[:pre_nms_top_n]; bx, scores = bx[top], scores[top]
    steps["top-k"] = (bx, scores)
    keep = nms(bx, scores, nms_iou)[:post_nms_top_n]; bx, scores = bx[keep], scores[keep]
    steps["nms"] = (bx, scores)
    return (bx, scores, steps) if return_steps else (bx, scores)


def roi_align(feature, rois, output_size: int = 7, spatial_scale: float = 1.0, sampling_ratio: int = 2):
    """Reference RoIAlign (aligned=True) for one feature map (C, H, W) and rois (K, 4) xyxy in image coords.

    Matches `torchvision.ops.roi_align(feature[None], [rois], output_size, spatial_scale, sampling_ratio, aligned=True)`
    for boxes that lie inside the image.
    """
    C, H, W = feature.shape
    out = feature.new_zeros((len(rois), C, output_size, output_size))
    for k, (x1, y1, x2, y2) in enumerate(rois.tolist()):
        x1, y1, x2, y2 = x1 * spatial_scale - 0.5, y1 * spatial_scale - 0.5, x2 * spatial_scale - 0.5, y2 * spatial_scale - 0.5
        bw, bh = (x2 - x1) / output_size, (y2 - y1) / output_size
        s = sampling_ratio
        frac = (torch.arange(s, dtype=feature.dtype) + 0.5) / s                         # sample offsets inside a bin
        ys = y1 + (torch.arange(output_size, dtype=feature.dtype)[:, None] + frac[None]) * bh   # (P, s)
        xs = x1 + (torch.arange(output_size, dtype=feature.dtype)[:, None] + frac[None]) * bw
        yy = ys.reshape(-1)[:, None].expand(-1, output_size * s)
        xx = xs.reshape(-1)[None, :].expand(output_size * s, -1)
        vals = bilinear_sample(feature, yy, xx)                                         # (C, P*s, P*s)
        out[k] = vals.reshape(C, output_size, s, output_size, s).mean((2, 4))
    return out


def bilinear_sample(feature, y, x):
    """Bilinear interpolation of feature (C, H, W) at float coordinates y, x (same shape) -> (C, *y.shape)."""
    C, H, W = feature.shape
    y = y.clamp(0, H - 1); x = x.clamp(0, W - 1)
    y0 = y.floor().long().clamp(max=H - 1); x0 = x.floor().long().clamp(max=W - 1)
    y1 = (y0 + 1).clamp(max=H - 1); x1 = (x0 + 1).clamp(max=W - 1)
    ly, lx = y - y0.float(), x - x0.float()
    hy, hx = 1 - ly, 1 - lx
    return (feature[:, y0, x0] * hy * hx + feature[:, y0, x1] * hy * lx +
            feature[:, y1, x0] * ly * hx + feature[:, y1, x1] * ly * lx)


def roi_pool_quantized(feature, roi, output_size: int = 3):
    """Classic RoI pooling (Fast R-CNN): round the box to the grid, split into integer bins, max-pool each.

    Returns (pooled (C, P, P), list of integer bin rectangles (x0, y0, x1, y1)) for drawing.
    """
    C, H, W = feature.shape
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
    out = feature.new_zeros((C, output_size, output_size)); bins = []
    for i in range(output_size):
        for j in range(output_size):
            ya = y1 + int(math.floor(i * (y2 - y1) / output_size)); yb = y1 + int(math.ceil((i + 1) * (y2 - y1) / output_size))
            xa = x1 + int(math.floor(j * (x2 - x1) / output_size)); xb = x1 + int(math.ceil((j + 1) * (x2 - x1) / output_size))
            out[:, i, j] = feature[:, ya:yb, xa:xb].amax((1, 2))
            bins.append((xa, ya, xb, yb))
    return out, bins


# ---------------------------------------------------------------------------
# D2-03: FPN and RetinaNet head references
# ---------------------------------------------------------------------------
class FPN(nn.Module):
    """Feature Pyramid Network: lateral 1x1, top-down nearest upsample + add, 3x3 smoothing.

    forward(list of C_i, finest first) -> list of P_i (same order, all `out_channels`).
    """

    def __init__(self, in_channels_list, out_channels: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.smooth = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])

    def forward(self, feats):
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        out = [None] * len(lat)
        top = lat[-1]
        out[-1] = self.smooth[-1](top)
        for i in range(len(lat) - 2, -1, -1):
            top = lat[i] + F.interpolate(top, size=lat[i].shape[-2:], mode="nearest")
            out[i] = self.smooth[i](top)
        return out


class RetinaHead(nn.Module):
    """Shared RetinaNet heads: two towers of `num_convs` 3x3 conv + ReLU; outputs A*C and A*4 channels."""

    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, num_convs: int = 4, prior: float = 0.01):
        super().__init__()

        def tower():
            layers = []
            for _ in range(num_convs):
                layers += [nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        self.cls_tower, self.box_tower = tower(), tower()
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01); nn.init.zeros_(m.bias)
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior) / prior))

    def forward(self, feats):
        return ([self.cls_logits(self.cls_tower(f)) for f in feats],
                [self.bbox_pred(self.box_tower(f)) for f in feats])


def sigmoid_focal_loss(logits, targets, gamma: float = 2.0, alpha: float = 0.25):
    """Reference focal loss on logits (mean). Same as torchvision.ops.sigmoid_focal_loss(..., reduction='mean')."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t) ** gamma
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss.mean()


def fcos_ltrb(points, box):
    """FCOS regression targets: distances (l, t, r, b) from points (P, 2) xy to the box sides."""
    points = np.asarray(points, float); x1, y1, x2, y2 = box
    return np.stack([points[:, 0] - x1, points[:, 1] - y1, x2 - points[:, 0], y2 - points[:, 1]], -1)


def centerness(ltrb):
    """FCOS centre-ness = sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b)); 0 outside the box."""
    ltrb = np.asarray(ltrb, float)
    l, t, r, b = ltrb[..., 0], ltrb[..., 1], ltrb[..., 2], ltrb[..., 3]
    inside = ltrb.min(-1) > 0
    c = np.sqrt(np.clip(np.minimum(l, r) / np.maximum(np.maximum(l, r), 1e-12), 0, None) *
                np.clip(np.minimum(t, b) / np.maximum(np.maximum(t, b), 1e-12), 0, None))
    return np.where(inside, c, 0.0)


# ---------------------------------------------------------------------------
# D2-04: DETR matching
# ---------------------------------------------------------------------------
def box_cxcywh_to_xyxy_t(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def detr_cost(pred_logits, pred_boxes, gt_labels, gt_boxes, w_class: float = 1.0, w_l1: float = 5.0, w_giou: float = 2.0):
    """DETR matching cost (Q, G): -w_class * p(gt class) + w_l1 * L1(cxcywh) - w_giou * GIoU(xyxy).

    pred_logits (Q, C+1) incl. a 'no object' class, pred_boxes (Q, 4) normalised cxcywh,
    gt_labels (G,), gt_boxes (G, 4) normalised cxcywh.
    """
    from torchvision.ops import generalized_box_iou
    prob = pred_logits.softmax(-1)
    c_class = -prob[:, gt_labels]
    c_l1 = torch.cdist(pred_boxes, gt_boxes, p=1)
    c_giou = -generalized_box_iou(box_cxcywh_to_xyxy_t(pred_boxes), box_cxcywh_to_xyxy_t(gt_boxes))
    return w_class * c_class + w_l1 * c_l1 + w_giou * c_giou


def hungarian_match(cost):
    """Optimal one-to-one assignment. cost (Q, G) -> (query_idx, gt_idx) sorted by gt? (sorted by query)."""
    from scipy.optimize import linear_sum_assignment
    c = cost.detach().cpu().numpy() if torch.is_tensor(cost) else np.asarray(cost)
    r, c_ = linear_sum_assignment(c)
    return r, c_


# ---------------------------------------------------------------------------
# torchvision detectors: data, training, prediction, evaluation
# ---------------------------------------------------------------------------
def detection_collate(batch):
    return tuple(zip(*batch))


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _as_image_tensor(img):
    """Accept a PIL image, an HWC uint8/float array or a CHW tensor; return CHW float tensor in [0,1]."""
    if torch.is_tensor(img):
        t = img.float()
        if t.ndim == 3 and t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3):
            t = t.permute(2, 0, 1)
        return t / 255.0 if t.max() > 1.5 else t
    a = np.asarray(img)
    t = torch.from_numpy(a.copy()).float()
    if t.ndim == 2:
        t = t[None].repeat(3, 1, 1)
    else:
        t = t.permute(2, 0, 1)
    return t / 255.0 if a.dtype == np.uint8 or t.max() > 1.5 else t


def materialise(ds, indices, max_side: int | None = 512):
    """Load `indices` of a detection dataset into memory as (image CHW float, target{boxes, labels}) pairs.

    Images are downscaled so the longer side is at most `max_side` (boxes scaled too) - this keeps
    CPU fine-tuning fast; torchvision detectors resize internally anyway (`min_size`).
    """
    items = []
    for i in indices:
        img, tgt = ds[int(i)]
        img = _as_image_tensor(img)
        bx = torch.as_tensor(tgt["boxes"], dtype=torch.float32).reshape(-1, 4)
        lb = torch.as_tensor(tgt["labels"], dtype=torch.int64).reshape(-1)
        if max_side is not None and max(img.shape[-2:]) > max_side:
            s = max_side / max(img.shape[-2:])
            img = F.interpolate(img[None], scale_factor=s, mode="bilinear", align_corners=False, antialias=True)[0]
            bx = bx * s
        items.append((img.clamp(0, 1).contiguous(), {"boxes": bx, "labels": lb}))
    return ListDataset(items)


def penn_fudan_splits(n_train: int = 40, n_val: int = 20, seed: int = 0, max_side: int = 512):
    """Penn-Fudan via torchkit, split into a small in-memory train / val subset (labels: 1 = person)."""
    from . import torchkit
    ds = torchkit.get_penn_fudan()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))
    return materialise(ds, idx[:n_train], max_side), materialise(ds, idx[n_train:n_train + n_val], max_side)


def train_detector(model, dataset, epochs: int = 1, batch_size: int = 2, lr: float = 0.005, momentum: float = 0.9,
                   weight_decay: float = 1e-4, max_minutes: float | None = None, log_every: int = 5,
                   params=None, log=print, seed: int = 0, optimizer: str = "sgd"):
    """Minimal torchvision-detection training loop (the model returns a dict of losses in train mode)."""
    torch.manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=detection_collate)
    params = [p for p in (params if params is not None else model.parameters()) if p.requires_grad]
    if optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(loader)
    warm = min(10, total_steps // 4 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    history = []
    t0 = time.time(); step = 0
    model.train()
    for ep in range(epochs):
        for images, targets in loader:
            losses = model(list(images), [dict(t) for t in targets])
            loss = sum(losses.values())
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
            history.append({k: float(v.detach()) for k, v in losses.items()} | {"total": float(loss.detach())})
            if step % log_every == 0 or step == 1:
                parts = "  ".join(f"{k.replace('loss_', '')} {float(v):.3f}" for k, v in losses.items())
                log(f"ep {ep + 1} it {step:3d}/{total_steps}  loss {float(loss):.3f}  ({parts})  {time.time() - t0:5.1f}s")
            if max_minutes and time.time() - t0 > 60 * max_minutes:
                log(f"stopping: {max_minutes} min budget reached after {step} iterations")
                model.eval()
                return history
    model.eval()
    return history


@torch.no_grad()
def predict_detector(model, images, score_threshold: float = 0.0):
    """Run a torchvision detector on a list of CHW tensors -> list of numpy dicts."""
    model.eval()
    outs = []
    for img in images:
        o = model([img])[0]
        keep = o["scores"] >= score_threshold
        outs.append({k: o[k][keep].cpu().numpy() for k in ("boxes", "scores", "labels")})
    return outs


def evaluate_detector(model, dataset, iou_thresholds=(0.5,), label_offset: int = 1, num_classes: int = 1):
    """mAP of a torchvision detector on an in-memory dataset with the course's `boxes.mean_average_precision`.

    torchvision uses label 0 for background, so labels are shifted by `label_offset` to start at 0.
    Returns (mAP, predictions, ground truths).
    """
    imgs = [dataset[i][0] for i in range(len(dataset))]
    preds = predict_detector(model, imgs)
    gts = []
    for i in range(len(dataset)):
        t = dataset[i][1]
        gts.append({"boxes": t["boxes"].numpy(), "labels": t["labels"].numpy() - label_offset})
    preds_shift = []
    for p in preds:
        keep = (p["labels"] - label_offset >= 0) & (p["labels"] - label_offset < num_classes)
        preds_shift.append({"boxes": p["boxes"][keep], "scores": p["scores"][keep], "labels": p["labels"][keep] - label_offset})
    m, _ = B.mean_average_precision(preds_shift, gts, num_classes, iou_thresholds)
    return m, preds, gts


COCO_THRESHOLDS = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))


def evaluate_map(model, dataset, label_offset: int = 1, num_classes: int = 1):
    """One inference pass over `dataset`, then mAP@0.5 and COCO mAP@[.5:.95] from the same predictions.

    Returns dict(map50, map, preds, gts); labels are shifted by `label_offset` (torchvision's background = 0).
    """
    imgs = [dataset[i][0] for i in range(len(dataset))]
    preds = predict_detector(model, imgs)
    gts = [{"boxes": dataset[i][1]["boxes"].numpy(), "labels": dataset[i][1]["labels"].numpy() - label_offset}
           for i in range(len(dataset))]
    shifted = []
    for p in preds:
        lab = p["labels"] - label_offset
        keep = (lab >= 0) & (lab < num_classes)
        shifted.append({"boxes": p["boxes"][keep], "scores": p["scores"][keep], "labels": lab[keep]})
    m50, _ = B.mean_average_precision(shifted, gts, num_classes, (0.5,))
    mc, _ = B.mean_average_precision(shifted, gts, num_classes, COCO_THRESHOLDS)
    return {"map50": m50, "map": mc, "preds": preds, "gts": gts}


def load_or_train(model, name: str, train_fn, log=print):
    """If `torchkit.CHECKPOINTS / name` exists, load it into `model`; otherwise call `train_fn()` and save."""
    from . import torchkit
    path = Path(torchkit.CHECKPOINTS) / name
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        log(f"loaded checkpoint {path.name}  (delete it to retrain)")
        return state.get("history") if isinstance(state, dict) else None
    history = train_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "history": history}, path)
    log(f"saved checkpoint {path.name}")
    return history


@torch.no_grad()
def latency_ms(model, size: int, repeats: int = 3, warmup: int = 1):
    """Median CPU latency (ms) of one forward pass on a single size x size image (batch of 1)."""
    model.eval()
    x = [torch.rand(3, size, size)]
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); model(x); ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts))


# ---------------------------------------------------------------------------
# D2-01: a NumPy-only "Selective Search" (hierarchical grouping of colour regions)
# ---------------------------------------------------------------------------
def oversegment(img, n_segments: int = 80, compactness: float = 0.6, iters: int = 6, seed: int = 0):
    """SLIC-like superpixels: k-means on (r, g, b, y, x) started on a regular grid. Returns a label map (H, W)."""
    from scipy import ndimage
    H, W = img.shape[:2]
    step = max(int(np.sqrt(H * W / n_segments)), 2)
    gy, gx = np.mgrid[step // 2:H:step, step // 2:W:step]
    yy, xx = np.mgrid[0:H, 0:W]
    feats = np.concatenate([img.reshape(-1, 3), compactness * np.stack([yy, xx], -1).reshape(-1, 2) / step], 1)
    centres = feats[(gy * W + gx).reshape(-1)]
    for _ in range(iters):
        d = ((feats[:, None, :] - centres[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for k in range(len(centres)):
            m = lab == k
            if m.any():
                centres[k] = feats[m].mean(0)
    lab = lab.reshape(H, W)
    out = np.zeros_like(lab); nxt = 1
    for k in np.unique(lab):                                   # split clusters into connected pieces
        cc, n = ndimage.label(lab == k)
        out[cc > 0] = cc[cc > 0] + nxt - 1; nxt += n
    return out


def mini_selective_search(img, n_segments: int = 80, min_area: int = 4):
    """Tiny Selective Search (Uijlings et al. 2013): start from superpixels, repeatedly merge the most
    similar *adjacent* pair (colour + small size + good fill), and emit the bounding box of every
    region ever formed.

    img: (H, W, 3) float in [0, 1]. Returns (proposals (N, 4) xyxy, initial label map, merge history).
    """
    seg = oversegment(img, n_segments)
    H, W = seg.shape
    regions = {}
    for r in np.unique(seg):
        m = seg == r
        ys, xs = np.where(m)
        regions[int(r)] = dict(box=np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float),
                               area=int(m.sum()), col=img[m].mean(0))
    nbrs = {r: set() for r in regions}
    for a, b in ((seg[:, :-1], seg[:, 1:]), (seg[:-1, :], seg[1:, :])):
        d = a != b
        for u, v in set(zip(a[d].tolist(), b[d].tolist())):
            nbrs[u].add(v); nbrs[v].add(u)

    def sim(u, v):
        ru, rv = regions[u], regions[v]
        colour = 1 - np.abs(ru["col"] - rv["col"]).sum() / 3
        size = 1 - (ru["area"] + rv["area"]) / (H * W)
        bb = np.concatenate([np.minimum(ru["box"][:2], rv["box"][:2]), np.maximum(ru["box"][2:], rv["box"][2:])])
        fill = 1 - (B.box_area(bb) - ru["area"] - rv["area"]) / (H * W)
        return 2 * colour + size + fill

    sims = {(min(u, v), max(u, v)): sim(u, v) for u in nbrs for v in nbrs[u]}
    out = [r["box"] for r in regions.values() if r["area"] >= min_area]
    history = []
    nid = max(regions) + 1
    while sims:
        (u, v), _ = max(sims.items(), key=lambda kv: kv[1])
        ru, rv = regions.pop(u), regions.pop(v)
        box = np.concatenate([np.minimum(ru["box"][:2], rv["box"][:2]), np.maximum(ru["box"][2:], rv["box"][2:])])
        a = ru["area"] + rv["area"]
        regions[nid] = dict(box=box, area=a, col=(ru["col"] * ru["area"] + rv["col"] * rv["area"]) / a)
        nb = (nbrs.pop(u) | nbrs.pop(v)) - {u, v}
        sims = {k: s_ for k, s_ in sims.items() if u not in k and v not in k}
        for w in nb:
            nbrs[w] -= {u, v}; nbrs[w].add(nid)
            sims[(min(w, nid), max(w, nid))] = sim(w, nid)
        nbrs[nid] = nb
        out.append(box); history.append((u, v, nid)); nid += 1
    return np.unique(np.array(out).reshape(-1, 4), axis=0), seg, history
