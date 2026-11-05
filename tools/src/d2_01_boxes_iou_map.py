from nb import md, code, exercise, header

NAME = "D2-01-boxes-iou-map"

SETUP = r"""
import sys, pathlib, time
for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:   # find the cvcourse package
    if (p / "cvcourse").is_dir():
        sys.path.insert(0, str(p)); break

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from cvcourse import viz, boxes, shapes, tests, torchkit, day2

%matplotlib inline
plt.rcParams["figure.dpi"] = 90
np.set_printoptions(precision=3, suppress=True)
torch.manual_seed(0)
device = torchkit.get_device()
GREEN, RED, GREY, BLUE, ORANGE = "#54A24B", "#E45756", "#9a9a9a", "#4C78A8", "#F58518"
"""

CELLS = [
    header(
        "D2-01 · Boxes, IoU, mAP and the region-based lineage",
        steps=[
            "The detection problem on Pet heads; three box formats and their conversions",
            "Intersection over Union: vectorised, corner cases, how fast it falls",
            "IoU-family losses (GIoU, DIoU, CIoU) and why plain IoU cannot be a loss",
            "Evaluation: TP / FP by greedy matching, the precision-recall curve, AP, mAP, COCO mAP@[.5:.95]",
            "History as a chain of fixes: sliding window -> R-CNN -> Fast -> Faster -> one-stage -> anchor-free -> DETR",
            "The fully-convolutional trick: a classifier *is* a sliding window; Selective Search in 40 lines",
            "Non-maximum suppression: greedy, Soft-NMS, class-wise; hard-negative mining",
        ],
        topics=[
            "Represent boxes in xyxy / xywh / cxcywh, pixels or normalised, and convert without bugs",
            "Compute IoU for N x M boxes with broadcasting, and reason about its corner cases",
            "Explain why GIoU / DIoU / CIoU give gradients where IoU does not",
            "Compute AP and mAP exactly the way COCO / VOC do, and read a PR curve",
            "Place every detector of the next three notebooks on the family tree",
            "Remove duplicate detections with NMS / Soft-NMS and know when NMS hurts",
        ],
        minutes=90,
        prereq="Day 1 (convnets, training, transfer learning). NumPy broadcasting.",
    ),
    md(r"""
    ## 0 · Setup

    Mechanisms are written in NumPy (they are a few lines each, and you will reuse them in
    every detector of the day); models are PyTorch. Every exercise is followed by a checker
    that prints `[PASS]` or `[FAIL]` with a hint.
    """),
    code(SETUP),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · The detection problem

    A classifier answers *what* is in the image. A detector answers **what, where and how
    many**: its output is a *variable-length set* of `(box, class, score)` triples. That one
    sentence explains everything that makes detection harder than classification:

    * the output size is not fixed -> we predict a fixed, large number of candidates and filter them (NMS);
    * "where" needs a geometry -> boxes, and a way to compare boxes (IoU);
    * "is this prediction right?" depends on a threshold -> precision / recall / AP instead of accuracy.

    Our running example is the Oxford-IIIT Pet dataset, which ships one *head* box per animal.
    """),
    code(r"""
    from torchvision.transforms import v2
    show_tf = v2.Compose([v2.ToImage(), v2.Resize((224, 224), antialias=True), v2.ToDtype(torch.float32, scale=True)])
    pet_train, pet_val, det_classes = torchkit.get_pet_datasets(target="detection", train_tf=show_tf, eval_tf=show_tf)
    print(f"Pet heads: {len(pet_train)} train / {len(pet_val)} val images, classes = {det_classes}")

    fig, axes = plt.subplots(2, 6, figsize=(15, 5.4))
    for ax, i in zip(axes.flat, range(0, 720, 60)):
        img, t = pet_val[i]
        ax.imshow(img.permute(1, 2, 0).numpy()); ax.axis("off")
        viz.draw_boxes(ax, t["boxes"].numpy(), labels=["head"] * len(t["boxes"]), lw=2, edgecolor=ORANGE)
        ax.set_title(pet_val.image_path(i).stem.rsplit("_", 1)[0], fontsize=8)
    fig.suptitle("Oxford-IIIT Pet: one head box per animal (boxes follow the resize to 224 x 224)", y=1.0)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Three box formats

    | format | numbers | used by |
    |---|---|---|
    | **xyxy** | $(x_1, y_1, x_2, y_2)$ corners | torchvision, PASCAL VOC xml, this course |
    | **xywh** | $(x_1, y_1, w, h)$ corner + size | COCO json |
    | **cxcywh** | $(c_x, c_y, w, h)$ centre + size | YOLO, DETR, anchor encodings |

    Coordinates are **continuous**: x to the right, y *downwards*, area $=(x_2-x_1)(y_2-y_1)$
    (the old "+1 pixel" convention is gone). Any of them can also be **normalised** by the image
    width / height to $[0, 1]$, which makes them independent of the resolution (YOLO, DETR).
    Mixing formats silently is the #1 detection bug, so let us draw one box in all three.
    """),
    code(r"""
    img, t = pet_val[300]
    b = t["boxes"][0].numpy()
    x1, y1, x2, y2 = b; w, h = x2 - x1, y2 - y1; cx, cy = x1 + w / 2, y1 + h / 2
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, title in zip(axes, ["xyxy  (x1, y1, x2, y2)", "xywh  (x1, y1, w, h)", "cxcywh  (cx, cy, w, h)"]):
        ax.imshow(img.permute(1, 2, 0).numpy()); ax.set_title(title)
        ax.add_patch(Rectangle((x1, y1), w, h, fill=False, ec=ORANGE, lw=2.5))
    axes[0].plot([x1, x2], [y1, y2], "o", color=RED, ms=8)
    lbl = dict(color="white", fontsize=10, weight="bold", bbox=dict(fc="black", alpha=0.6, ec="none", pad=1.5))
    axes[0].annotate(f"({x1:.0f}, {y1:.0f})", (x1, y1), (x1 + 4, y1 - 6), **lbl)
    axes[0].annotate(f"({x2:.0f}, {y2:.0f})", (x2, y2), (x2 - 30, y2 + 16), **lbl)
    axes[1].plot([x1], [y1], "o", color=RED, ms=8)
    axes[1].annotate("", (x2, y2 + 8), (x1, y2 + 8), arrowprops=dict(arrowstyle="<->", color="white", lw=2))
    axes[1].text((x1 + x2) / 2, y2 + 22, f"w = {w:.0f}", ha="center", **lbl)
    axes[1].annotate("", (x2 + 8, y2), (x2 + 8, y1), arrowprops=dict(arrowstyle="<->", color="white", lw=2))
    axes[1].text(x2 + 12, (y1 + y2) / 2, f"h = {h:.0f}", va="center", **lbl)
    axes[2].plot([cx], [cy], "+", color=RED, ms=16, mew=3)
    axes[2].annotate(f"({cx:.0f}, {cy:.0f})", (cx, cy), (cx + 6, cy - 6), **lbl)
    for ax in axes: ax.set_xlim(0, 224); ax.set_ylim(224, 0)
    plt.tight_layout(); plt.show()
    print("xyxy  :", b.round(1))
    print("xywh  :", boxes.xyxy_to_xywh(b).round(1))
    print("cxcywh:", boxes.xyxy_to_cxcywh(b).round(1))
    print("cxcywh normalised by the 224 px image:", (boxes.xyxy_to_cxcywh(b) / 224).round(3))
    """),
    md(r"""
    ### Exercise 1 · box conversions

    Write the three conversions for arrays of shape `(..., 4)` (use `b[..., 0]` etc. so they
    work for one box and for N boxes; `np.stack(..., -1)` puts the columns back together).
    """),
    exercise(
        stub=r"""
        def xyxy_to_xywh(b):
            b = np.asarray(b, float)
            # YOUR CODE HERE
            raise NotImplementedError

        def xyxy_to_cxcywh(b):
            b = np.asarray(b, float)
            # YOUR CODE HERE
            raise NotImplementedError

        def cxcywh_to_xyxy(b):
            b = np.asarray(b, float)
            # YOUR CODE HERE
            raise NotImplementedError

        tests.check_box_conversions(xyxy_to_xywh, xyxy_to_cxcywh, cxcywh_to_xyxy)
        """,
        solution=r"""
        def xyxy_to_xywh(b):
            b = np.asarray(b, float)
            return np.stack([b[..., 0], b[..., 1], b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]], -1)

        def xyxy_to_cxcywh(b):
            b = np.asarray(b, float)
            w, h = b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]
            return np.stack([b[..., 0] + w / 2, b[..., 1] + h / 2, w, h], -1)

        def cxcywh_to_xyxy(b):
            b = np.asarray(b, float)
            cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
            return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)

        tests.check_box_conversions(xyxy_to_xywh, xyxy_to_cxcywh, cxcywh_to_xyxy)
        """),
    md(r"""
    > **Normalised vs pixel coordinates.** Resizing an image by a factor $s$ multiplies pixel
    > boxes by $s$ but leaves normalised boxes unchanged. Libraries differ: torchvision wants
    > *pixels* (xyxy), YOLO label files store *normalised* cxcywh, COCO json stores *pixel* xywh.
    > Always check which one a loader returns before you train for a day on wrong boxes.
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · Intersection over Union

    How good is a predicted box $P$ for a ground-truth box $G$? The standard answer is

    $$\mathrm{IoU}(P, G) = \frac{|P \cap G|}{|P \cup G|} = \frac{|P \cap G|}{|P| + |G| - |P \cap G|} \in [0, 1].$$

    The intersection of two axis-aligned boxes is again a box: its top-left corner is the
    **max** of the two top-left corners, its bottom-right the **min** of the two bottom-right
    corners, and its width / height are clipped at 0 when the boxes do not overlap. IoU is
    *scale invariant*: it does not care whether the boxes are 10 or 1000 pixels wide.
    """),
    code(r"""
    cases = [("identical", [10, 10, 50, 50], [10, 10, 50, 50]),
             ("partial overlap", [10, 10, 50, 50], [30, 25, 70, 65]),
             ("touching edges", [10, 10, 40, 50], [40, 10, 70, 50]),
             ("disjoint", [5, 5, 30, 30], [45, 40, 70, 70]),
             ("one inside the other", [5, 5, 75, 75], [25, 25, 55, 55]),
             ("same centre, other shape", [25, 5, 55, 75], [5, 25, 75, 55])]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6))
    for ax, (name, a, b) in zip(axes.flat, cases):
        a, b = np.array(a, float), np.array(b, float)
        ax.add_patch(Rectangle(a[:2], *(a[2:] - a[:2]), fc=BLUE, alpha=0.35, ec=BLUE, lw=2))
        ax.add_patch(Rectangle(b[:2], *(b[2:] - b[:2]), fc=ORANGE, alpha=0.35, ec=ORANGE, lw=2))
        lt, rb = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
        if (rb > lt).all():
            ax.add_patch(Rectangle(lt, *(rb - lt), fill=False, hatch="///", ec="k", lw=0))
        ax.set_xlim(0, 80); ax.set_ylim(80, 0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}\nIoU = {boxes.iou_single(a, b):.3f}")
    fig.suptitle("IoU corner cases (hatched = intersection)", y=1.0)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Exercise 2 · pairwise IoU, vectorised

    `box_iou(a, b)` takes `a (N, 4)` and `b (M, 4)` in xyxy and returns the `(N, M)` matrix of
    IoUs **without Python loops**: broadcast `a[:, None, :2]` against `b[None, :, :2]`. Detectors
    call this on thousands of anchors x tens of boxes per image, so it must be vectorised.
    """),
    exercise(
        stub=r"""
        def box_area(b):
            return (b[..., 2] - b[..., 0]).clip(0) * (b[..., 3] - b[..., 1]).clip(0)

        def box_iou(a, b):
            a = np.asarray(a, float).reshape(-1, 4)
            b = np.asarray(b, float).reshape(-1, 4)
            # YOUR CODE HERE: top-left = max of corners, bottom-right = min, clip w/h at 0,
            # inter (N, M), union = area_a[:, None] + area_b[None, :] - inter
            raise NotImplementedError

        tests.check_iou(lambda a, b: box_iou(a, b)[0, 0])
        tests.check_box_iou_matrix(box_iou)
        """,
        solution=r"""
        def box_area(b):
            return (b[..., 2] - b[..., 0]).clip(0) * (b[..., 3] - b[..., 1]).clip(0)

        def box_iou(a, b):
            a = np.asarray(a, float).reshape(-1, 4)
            b = np.asarray(b, float).reshape(-1, 4)
            lt = np.maximum(a[:, None, :2], b[None, :, :2])      # (N, M, 2)
            rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
            wh = (rb - lt).clip(0)
            inter = wh[..., 0] * wh[..., 1]
            union = box_area(a)[:, None] + box_area(b)[None, :] - inter
            return inter / np.maximum(union, 1e-12)

        tests.check_iou(lambda a, b: box_iou(a, b)[0, 0])
        tests.check_box_iou_matrix(box_iou)
        """),
    code(r"""
    # The library equivalent: torchvision.ops.box_iou (same maths, on tensors, GPU-ready)
    rng = np.random.default_rng(0)
    A = rng.uniform(0, 100, (1000, 2)); A = np.concatenate([A, A + rng.uniform(5, 60, (1000, 2))], 1)
    G = rng.uniform(0, 100, (20, 2));   G = np.concatenate([G, G + rng.uniform(5, 60, (20, 2))], 1)
    ours = boxes.box_iou(A, G)
    lib = torchvision.ops.box_iou(torch.tensor(A), torch.tensor(G)).numpy()
    print("IoU matrix", ours.shape, "| max |ours - torchvision| =", np.abs(ours - lib).max())
    """),
    md(r"""
    ### How fast does IoU fall?

    Slide a predicted box across a ground-truth box of the same size. IoU drops **much faster
    than intuition suggests**: shifting a box by one third of its width already brings IoU down
    to 0.5, the usual "correct detection" threshold. And because the shift is measured in pixels,
    the same 4-pixel error that is harmless on a 128-pixel dog head is fatal on a 16-pixel one -
    which is why COCO reports AP separately for small, medium and large objects.
    """),
    code(r"""
    gt = np.array([40, 30, 100, 90], float)           # 60 x 60
    shifts = np.linspace(-75, 75, 301)
    ious = [boxes.iou_single(gt, gt + [s, 0, s, 0]) for s in shifts]
    snap = [-70, -45, -20, 0, 20, 45]

    fig = plt.figure(figsize=(14, 5.4))
    for k, s in enumerate(snap):
        ax = fig.add_axes([0.03 + k * 0.16, 0.64, 0.15, 0.3])
        ax.add_patch(Rectangle(gt[:2], 60, 60, fc=GREEN, alpha=0.4, ec=GREEN, lw=2))
        ax.add_patch(Rectangle(gt[:2] + [s, 0], 60, 60, fill=False, ec=RED, lw=2.5))
        ax.set_xlim(-40, 180); ax.set_ylim(110, 10); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"shift {s:+d} px\nIoU = {boxes.iou_single(gt, gt + [s, 0, s, 0]):.2f}", fontsize=9)
    ax = fig.add_axes([0.06, 0.09, 0.42, 0.45])
    ax.plot(shifts, ious, color=BLUE, lw=2); ax.axhline(0.5, ls="--", color=GREY)
    ax.plot(snap, [boxes.iou_single(gt, gt + [s, 0, s, 0]) for s in snap], "o", color=RED)
    ax.set_xlabel("horizontal shift of the prediction (px), box width 60 px"); ax.set_ylabel("IoU"); ax.grid(alpha=.3)
    ax.text(-78, 0.53, "IoU = 0.5 at shift = w/3", color="k", fontsize=9)
    ax = fig.add_axes([0.56, 0.09, 0.42, 0.45])
    px = np.linspace(0, 20, 81)
    for size, c in zip([16, 32, 64, 128], [RED, ORANGE, GREEN, BLUE]):
        g = np.array([0, 0, size, size], float)
        ax.plot(px, [boxes.iou_single(g, g + [d, d, d, d]) for d in px], color=c, lw=2, label=f"{size} px box")
    ax.axhline(0.5, ls="--", color=GREY); ax.set_xlabel("diagonal shift (px)"); ax.set_ylabel("IoU")
    ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_title("same pixel error, different object sizes", fontsize=10)
    plt.show()
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · IoU-family losses: GIoU, DIoU, CIoU

    The metric is IoU, so why not train with $\mathcal{L} = 1 - \mathrm{IoU}$? Two problems:

    1. When the boxes **do not overlap**, IoU is 0 whatever the distance: the loss is flat and the
       gradient is exactly zero. The network gets no hint which way to move.
    2. An L1 loss on the four corners has gradients everywhere but is **not scale invariant**
       (a 5 px error costs the same on a huge and on a tiny box) and does not match the metric.

    The fixes add a penalty that keeps working without overlap. With $C$ the smallest box that
    encloses both, $\rho$ the distance between centres and $c$ the diagonal of $C$:

    $$\mathrm{GIoU} = \mathrm{IoU} - \frac{|C| - |P \cup G|}{|C|}, \qquad
      \mathrm{DIoU} = \mathrm{IoU} - \frac{\rho^2}{c^2}, \qquad
      \mathrm{CIoU} = \mathrm{DIoU} - \alpha v$$

    where $v$ measures the aspect-ratio mismatch. All three lie in $[-1, 1]$; the losses are $1 - (\cdot)$.
    GIoU (2019) is in DETR; DIoU / CIoU (2020) in the YOLO family.
    """),
    md(r"""
    ### Exercise 3 · GIoU

    `giou(a, b)` for *aligned* pairs: `a (N, 4)` and `b (N, 4)` -> `(N,)`. Compute IoU of each pair,
    then the enclosing box (min of top-left corners, max of bottom-right corners) and its area.
    """),
    exercise(
        stub=r"""
        def giou(a, b):
            a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
            lt = np.maximum(a[:, :2], b[:, :2]); rb = np.minimum(a[:, 2:], b[:, 2:])
            inter = (rb - lt).clip(0).prod(-1)
            union = box_area(a) + box_area(b) - inter
            iou = inter / np.maximum(union, 1e-12)
            # YOUR CODE HERE: enclosing box C, its area, and the GIoU
            raise NotImplementedError

        tests.check_giou(giou)
        """,
        solution=r"""
        def giou(a, b):
            a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
            lt = np.maximum(a[:, :2], b[:, :2]); rb = np.minimum(a[:, 2:], b[:, 2:])
            inter = (rb - lt).clip(0).prod(-1)
            union = box_area(a) + box_area(b) - inter
            iou = inter / np.maximum(union, 1e-12)
            c_lt = np.minimum(a[:, :2], b[:, :2]); c_rb = np.maximum(a[:, 2:], b[:, 2:])
            c_area = (c_rb - c_lt).clip(0).prod(-1)
            return iou - (c_area - union) / np.maximum(c_area, 1e-12)

        tests.check_giou(giou)
        """),
    code(r"""
    # 1-D loss landscape: the GT is fixed, the prediction (40 x 40, then 40 x 80) slides horizontally
    gt = np.array([80, 40, 120, 80], float)
    xs = np.linspace(-60, 220, 400)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
    for ax, (pw, ph) in zip(axes, [(40, 40), (40, 80)]):
        P = np.stack([xs, np.full_like(xs, 60 - ph / 2), xs + pw, np.full_like(xs, 60 + ph / 2)], 1)
        G = np.repeat(gt[None], len(xs), 0)
        l1 = np.abs(P - G).sum(1); l1 = l1 / l1.max() * 2
        ax.plot(xs + pw / 2, l1, color=GREY, lw=2, label="L1 on corners (rescaled)")
        ax.plot(xs + pw / 2, 1 - np.diag(boxes.box_iou(P, G)), color=BLUE, lw=2.5, label="1 - IoU")
        ax.plot(xs + pw / 2, 1 - boxes.generalized_iou(P, G), color=ORANGE, lw=2, label="1 - GIoU")
        ax.plot(xs + pw / 2, 1 - boxes.distance_iou(P, G), color=GREEN, lw=2, ls="--", label="1 - DIoU")
        ax.plot(xs + pw / 2, 1 - boxes.complete_iou(P, G), color=RED, lw=1.5, ls=":", label="1 - CIoU")
        ax.axvspan(xs[0] + pw / 2, 80 - pw / 2, color=BLUE, alpha=0.06); ax.axvspan(120 + pw / 2, xs[-1] + pw / 2, color=BLUE, alpha=0.06)
        ax.text(5, 1.9, "no overlap:\n1-IoU is flat", color=BLUE, fontsize=9)
        ax.axvline(100, color="k", lw=0.8); ax.set_xlabel("x-centre of the prediction (GT centre = 100)")
        ax.set_title(f"prediction {pw} x {ph}, GT 40 x 40"); ax.grid(alpha=.3)
    axes[0].set_ylabel("loss"); axes[0].legend(fontsize=8, loc="lower left")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Read the right panel: the prediction has the wrong aspect ratio, so even perfectly centred
    its IoU is 0.5. Only CIoU adds an extra penalty for that; GIoU and DIoU agree with IoU once
    the boxes overlap. Now let the losses *drive* a box. We start a prediction far away from
    the GT and run gradient descent on its four coordinates with each loss
    (`torchvision.ops` has all of them).
    """),
    code(r"""
    from torchvision.ops import generalized_box_iou_loss, distance_box_iou_loss, complete_box_iou_loss
    def iou_loss(p, g):
        return 1 - torchvision.ops.box_iou(p, g).diag()
    losses = {"1 - IoU": iou_loss, "GIoU loss": generalized_box_iou_loss,
              "DIoU loss": distance_box_iou_loss, "CIoU loss": complete_box_iou_loss}
    gt_t = torch.tensor([[60., 50., 100., 110.]])
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, (name, fn) in zip(axes, losses.items()):
        p = torch.tensor([[130., 120., 160., 140.]], requires_grad=True)
        opt = torch.optim.Adam([p], lr=1.0)               # Adam: same step length for every loss
        traj = []
        for step in range(150):
            loss = fn(p, gt_t).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            traj.append(p.detach().numpy()[0].copy())
        traj = np.array(traj)
        ax.add_patch(Rectangle((60, 50), 40, 60, fc=GREEN, alpha=0.35, ec=GREEN, lw=2))
        for k in [0, 20, 40, 60, 149]:
            b = traj[k]; ax.add_patch(Rectangle(b[:2], *(b[2:] - b[:2]), fill=False, ec=RED, alpha=0.25 + 0.75 * k / 149, lw=1.5))
        c = (traj[:, :2] + traj[:, 2:]) / 2
        ax.plot(c[:, 0], c[:, 1], ".-", color=RED, ms=3, lw=1)
        ious = boxes.box_iou(traj, gt_t.numpy())[:, 0]
        reach = f"IoU 0.5 after {np.argmax(ious >= 0.5) + 1} steps" if (ious >= 0.5).any() else "never overlaps"
        ax.set_title(f"{name}\n{reach}, final IoU {ious[-1]:.2f}", fontsize=10)
        ax.set_xlim(30, 190); ax.set_ylim(170, 20); ax.set_aspect("equal")
    fig.suptitle("Gradient descent on the box coordinates, starting with no overlap (green = GT)", y=1.02)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    `1 - IoU` never moves: zero gradient, the box is stuck forever. GIoU moves because it
    penalises the empty part of the enclosing box: its gradient first stretches the box towards
    the GT, then the IoU term takes over. DIoU pulls the **centre** straight at the target, CIoU
    additionally matches the aspect ratio. (We use Adam so that every loss gets the same step
    length and only the *direction* of the gradient matters.) In practice detectors combine an IoU-family loss with L1 (DETR: L1 + GIoU).
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · Evaluation: precision, recall, AP, mAP

    A detector outputs scored boxes. To score the detector on one class:

    1. **Sort** all predictions by score, highest first.
    2. Walk down the list. A prediction is a **true positive** if its best-matching *still
       unmatched* GT box has IoU $\ge$ threshold (0.5 for PASCAL VOC); that GT is then consumed.
       Otherwise it is a **false positive** (a duplicate of an already-found object is a FP!).
       GT boxes never matched are **false negatives**.
    3. After each prediction $k$: $\;\text{precision}_k = \frac{TP_k}{k}$, $\;\text{recall}_k = \frac{TP_k}{\#GT}$.

    Every score threshold you could choose corresponds to one point of this list, so the
    **precision-recall curve** summarises all thresholds at once. **Average precision (AP)** is
    the area under it (after making precision monotonically decreasing). **mAP** is the mean of
    AP over classes. COCO additionally averages over IoU thresholds $0.5, 0.55, \dots, 0.95$, which
    rewards *precise* boxes: that is "mAP@[.5:.95]", the number every paper reports.
    """),
    code(r"""
    # A toy image: 4 GT objects and 8 predictions with hand-picked quality
    rng = np.random.default_rng(7)
    toy_img, toy_gt, _, _ = shapes.make_sample(np.random.default_rng(4), 160, max_objects=4, min_r=14, max_r=26)
    g = toy_gt
    toy_pred = np.array([g[0] + [1, -2, 2, 1],                 # good box on object 0
                         g[1] + [-2, 1, 1, 2],                 # good box on object 1
                         g[0] + [6, 5, 8, 6],                  # duplicate of object 0
                         g[2] + [2, 2, -1, -2],                # good box on object 2
                         [5, 120, 35, 150],                    # background
                         g[3] + [14, 12, 16, 14],              # badly localised object 3
                         g[3] + [-1, 1, 2, 0],                 # good box on object 3
                         [120, 5, 150, 30]], float)            # background
    toy_scores = np.array([0.95, 0.91, 0.84, 0.77, 0.62, 0.55, 0.43, 0.31])
    P_toy, R_toy, tp_toy = boxes.precision_recall(toy_pred, toy_scores, toy_gt, 0.5)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax[0].imshow(toy_img); ax[0].axis("off"); ax[0].set_title("GT (green) and 8 predictions (TP blue, FP red)")
    viz.draw_boxes(ax[0], toy_gt, edgecolor=GREEN, lw=3)
    for k, (b, ok) in enumerate(zip(toy_pred, tp_toy)):
        c = BLUE if ok else RED
        ax[0].add_patch(Rectangle(b[:2], *(b[2:] - b[:2]), fill=False, ec=c, lw=1.8, ls="-" if ok else "--"))
        ax[0].text(b[2] - 2, b[1] + 1, f"#{k + 1}", color="white", fontsize=8, ha="right", va="top", bbox=dict(fc=c, ec="none", pad=1))
    ax[1].axis("off")
    rows = [[f"#{k + 1}", f"{s:.2f}", f"{boxes.box_iou(b[None], toy_gt).max():.2f}", "TP" if t else "FP",
             f"{int(tp_toy[:k + 1].sum())}", f"{p:.2f}", f"{r:.2f}"] for k, (b, s, t, p, r) in enumerate(zip(toy_pred, toy_scores, tp_toy, P_toy, R_toy))]
    tab = ax[1].table(cellText=rows, colLabels=["rank", "score", "best IoU", "", "cum TP", "precision", "recall"], loc="center", cellLoc="center")
    tab.scale(1, 1.6); tab.set_fontsize(10)
    for k, t in enumerate(tp_toy):
        tab[k + 1, 3].set_facecolor("#cfe6cf" if t else "#f6cccc")
    ax[1].set_title("walk down the list sorted by score", pad=12)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Prediction #3 overlaps object 0 well, but object 0 was already claimed by #1: **duplicates
    are false positives**. Prediction #6 overlaps object 3 with IoU < 0.5: a localisation error
    *and* a miss at that point of the list. Now you write the matching.

    ### Exercise 4 · precision and recall by greedy matching

    `precision_recall(pred_boxes, pred_scores, gt_boxes, iou_threshold)` returns
    `(precision, recall, tp)` arrays in score order. Use `boxes.box_iou`; mark matched GTs so they
    cannot be used twice (setting their IoU to -1 is a neat trick).
    """),
    exercise(
        stub=r"""
        def precision_recall(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
            order = np.argsort(-np.asarray(pred_scores))
            matched = np.zeros(len(gt_boxes), bool)
            tp = np.zeros(len(order), bool)
            for k, i in enumerate(order):
                # YOUR CODE HERE: IoU of prediction i with all GTs, ignore matched ones, decide TP/FP
                raise NotImplementedError
            tp_c, fp_c = np.cumsum(tp), np.cumsum(~tp)
            precision = tp_c / (tp_c + fp_c)
            recall = tp_c / max(len(gt_boxes), 1)
            return precision, recall, tp

        tests.check_precision_recall(precision_recall)
        """,
        solution=r"""
        def precision_recall(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
            order = np.argsort(-np.asarray(pred_scores))
            matched = np.zeros(len(gt_boxes), bool)
            tp = np.zeros(len(order), bool)
            for k, i in enumerate(order):
                if len(gt_boxes) == 0:
                    break
                ious = boxes.box_iou(pred_boxes[i][None], gt_boxes)[0]
                ious[matched] = -1                    # each GT can be found only once
                j = ious.argmax()
                if ious[j] >= iou_threshold:
                    tp[k] = True; matched[j] = True
            tp_c, fp_c = np.cumsum(tp), np.cumsum(~tp)
            precision = tp_c / (tp_c + fp_c)
            recall = tp_c / max(len(gt_boxes), 1)
            return precision, recall, tp

        tests.check_precision_recall(precision_recall)
        """),
    code(r"""
    # The PR curve, built one prediction at a time
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    for k, ax in enumerate(axes.flat):
        ax.plot(R_toy[:k + 1], P_toy[:k + 1], "-", color=GREY, lw=1.5)
        ax.scatter(R_toy[:k], P_toy[:k], c=[BLUE if t else RED for t in tp_toy[:k]], s=30, alpha=0.6)
        ax.scatter([R_toy[k]], [P_toy[k]], c=[BLUE if tp_toy[k] else RED], s=140, edgecolor="k", zorder=3)
        ax.set_title(f"#{k + 1} score {toy_scores[k]:.2f}: {'TP' if tp_toy[k] else 'FP'}\nP = {P_toy[k]:.2f}, R = {R_toy[k]:.2f}", fontsize=10)
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.08); ax.grid(alpha=.3)
    for ax in axes[1]: ax.set_xlabel("recall")
    for ax in axes[:, 0]: ax.set_ylabel("precision")
    fig.suptitle("A TP moves the curve right (recall up); a FP moves it down (precision down)", y=1.0)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### From the zig-zag to one number

    The raw curve zig-zags. All modern protocols first replace precision at recall $r$ by the
    **best precision at any recall $\ge r$** (the "envelope": you would never pick a threshold
    that is worse in both precision and recall), then integrate:

    $$\mathrm{AP} = \int_0^1 p_{\text{interp}}(r)\,dr, \qquad p_{\text{interp}}(r) = \max_{\tilde r \ge r} p(\tilde r).$$

    VOC2007 sampled $p_{\text{interp}}$ at 11 recall points, VOC2010+ integrates exactly ("all
    points"), COCO samples 101 points.

    ### Exercise 5 · average precision (all points)

    Pad the curve with `(recall 0, precision 0)` at the start and `(recall 1, precision 0)` at the end
    (as in the code), make precision monotone from the right, then sum `(r[i+1] - r[i]) * p[i+1]`
    wherever recall changes.
    """),
    exercise(
        stub=r"""
        def average_precision(precision, recall):
            p = np.concatenate([[0.0], precision, [0.0]])
            r = np.concatenate([[0.0], recall, [1.0]])
            # YOUR CODE HERE: 1) envelope: p[i] = max(p[i], p[i+1]) from right to left
            #                 2) area: sum over the steps where recall changes
            raise NotImplementedError

        tests.check_average_precision(average_precision)
        """,
        solution=r"""
        def average_precision(precision, recall):
            p = np.concatenate([[0.0], precision, [0.0]])
            r = np.concatenate([[0.0], recall, [1.0]])
            p = np.maximum.accumulate(p[::-1])[::-1]          # envelope, right to left
            idx = np.where(r[1:] != r[:-1])[0]
            return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))

        tests.check_average_precision(average_precision)
        """),
    code(r"""
    p = np.concatenate([[0.0], P_toy, [0.0]]); r = np.concatenate([[0.0], R_toy, [1.0]])
    env = np.maximum.accumulate(p[::-1])[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.fill_between(r, env, step="pre", color=BLUE, alpha=0.18, label="AP = shaded area")
    ax.step(r, env, where="pre", color=BLUE, lw=2.5, label="interpolated (envelope)")
    ax.plot(R_toy, P_toy, "o-", color=GREY, lw=1, label="raw precision / recall")
    r11 = np.linspace(0, 1, 11)
    ax.plot(r11, [env[r >= t].max() if (r >= t).any() else 0 for t in r11], "x", color=RED, ms=9, mew=2, label="VOC2007 11-point samples")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05); ax.grid(alpha=.3); ax.legend(loc="lower left", fontsize=9)
    ax.set_title("AP of the toy example")
    plt.show()
    for m in ["all", "interp11", "interp101"]:
        print(f"AP ({m:9s}) = {boxes.average_precision(P_toy, R_toy, m):.3f}")
    """),
    md(r"""
    ### mAP over classes, IoU thresholds and object sizes

    Let us evaluate a *simulated* detector on 300 shapes images: every object is found with a
    few pixels of localisation noise, sometimes with the wrong class, plus some background false
    positives. `boxes.mean_average_precision` pools the predictions of all images per class
    (this is how VOC / COCO do it - the PR curve is per *dataset*, not per image).
    """),
    code(r"""
    sim_imgs, sim_gt_boxes, sim_gt_labels, _ = shapes.make_dataset(300, 128, seed=5)
    rng = np.random.default_rng(1)
    sim_preds, sim_gts = [], []
    for gb, gl in zip(sim_gt_boxes, sim_gt_labels):
        pb = gb + rng.normal(0, 2.5, gb.shape)                                 # ~2.5 px noise per coordinate
        pl = np.where(rng.uniform(size=len(gl)) < 0.08, rng.integers(0, 3, len(gl)), gl)   # 8% wrong class
        ps = np.clip(0.5 + 0.5 * np.diag(boxes.box_iou(pb, gb)) + rng.normal(0, 0.1, len(gl)), 0, 1)
        nf = rng.integers(0, 3)                                                # 0-2 background false positives
        fb = rng.uniform(0, 100, (nf, 2)); fb = np.concatenate([fb, fb + rng.uniform(10, 30, (nf, 2))], 1)
        sim_preds.append({"boxes": np.concatenate([pb, fb]), "scores": np.concatenate([ps, rng.uniform(0.2, 0.8, nf)]),
                          "labels": np.concatenate([pl, rng.integers(0, 3, nf)])})
        sim_gts.append({"boxes": gb, "labels": gl})

    thr = np.round(np.arange(0.5, 0.96, 0.05), 2)
    ap_per_thr = [boxes.mean_average_precision(sim_preds, sim_gts, 3, (t,))[1] for t in thr]
    map50, per_class50 = boxes.mean_average_precision(sim_preds, sim_gts, 3, (0.5,))
    map_coco, _ = boxes.mean_average_precision(sim_preds, sim_gts, 3, tuple(thr))

    # size buckets (COCO uses 32^2 / 96^2 px on ~640 px images; we scale them to our 128 px images)
    def size_filter(d, lo, hi):
        a = boxes.box_area(d["boxes"]); k = (a >= lo) & (a < hi)
        return {key: v[k] for key, v in d.items()}
    buckets = {"small (< 24^2)": (0, 24 ** 2), "medium": (24 ** 2, 36 ** 2), "large (>= 36^2)": (36 ** 2, 1e9)}
    size_map = {k: boxes.mean_average_precision([size_filter(p, *v) for p in sim_preds], [size_filter(g, *v) for g in sim_gts], 3, tuple(thr))[0]
                for k, v in buckets.items()}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for c, name in enumerate(shapes.CLASS_NAMES):
        axes[0].plot(thr, [a[c] for a in ap_per_thr], "o-", color=shapes.CLASS_COLORS[c], label=name)
    axes[0].plot(thr, [np.mean(a) for a in ap_per_thr], "k--", lw=2, label="mAP")
    axes[0].set_xlabel("IoU threshold"); axes[0].set_ylabel("AP"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
    axes[0].set_title(f"mAP@0.5 = {map50:.3f}   mAP@[.5:.95] = {map_coco:.3f}")
    axes[1].bar(shapes.CLASS_NAMES, per_class50, color=shapes.CLASS_COLORS); axes[1].set_ylim(0, 1)
    axes[1].set_title("AP@0.5 per class"); [axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center") for i, v in enumerate(per_class50)]
    axes[2].bar(list(size_map), list(size_map.values()), color=[RED, ORANGE, GREEN]); axes[2].set_ylim(0, 1)
    axes[2].set_title("mAP@[.5:.95] by object size (same 2.5 px noise)")
    for i, v in enumerate(size_map.values()): axes[2].text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Three lessons in one figure: (1) with a few pixels of noise, mAP@0.5 is nearly perfect but
    AP collapses at IoU 0.9 - **mAP@[.5:.95] measures localisation quality**; (2) mAP is the plain
    average over classes, so a rare class counts as much as a common one; (3) the *same* pixel
    noise hurts small objects most. When someone quotes "mAP", always ask: which IoU thresholds,
    which dataset, which object sizes. (Library equivalent: `torchmetrics.detection.MeanAveragePrecision`
    or `pycocotools`; they implement exactly this protocol.)
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · Ten years of detectors as a chain of fixes

    Every detector you will meet today is the previous one with **one bottleneck removed**.
    Keep this map in mind; D2-02 opens the two-stage branch, D2-03 the one-stage branch,
    D2-04 the transformer branch.
    """),
    code(r"""
    steps = [
        ("Sliding window\n+ CNN classifier", "2013 (OverFeat)", "classify every crop at\nevery position and scale", "~10^5 crops / image"),
        ("R-CNN", "2014", "~2000 Selective-Search\nproposals, CNN on each", "~47 s / image"),
        ("Fast R-CNN", "2015", "ONE conv pass, crop\nfeatures with RoI pooling", "~2 s (proposals dominate)"),
        ("Faster R-CNN", "2015", "proposals by a learned\nRPN on the same features", "~0.2 s (5 fps)"),
        ("One-stage\nYOLO / SSD /\nRetinaNet", "2016-17", "no proposals: dense grid\npredicts boxes + classes", "20-60 fps"),
        ("Anchor-free\nFCOS / CenterNet", "2019", "no anchor boxes: points\npredict distances / peaks", "fewer hyper-parameters"),
        ("Set prediction\nDETR / RT-DETR", "2020-23", "transformer queries +\nHungarian matching,\nno NMS", "end to end"),
    ]
    fixes = ["too many crops", "one CNN per proposal", "proposals on CPU", "two stages", "anchor tuning", "NMS + hand rules"]
    fig, ax = plt.subplots(figsize=(16, 4.6)); ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0.3, 5.2)
    cols = [GREY, "#9ecae1", "#6baed6", BLUE, ORANGE, GREEN, "#B279A2"]
    for k, ((name, year, idea, speed), c) in enumerate(zip(steps, cols)):
        x = 0.15 + k * 2.27
        ax.add_patch(FancyBboxPatch((x, 1.35), 1.95, 3.2, boxstyle="round,pad=0.05", fc=c, ec="k", alpha=0.85))
        ax.text(x + 0.97, 4.25, name, ha="center", va="top", fontsize=10, weight="bold")
        ax.text(x + 0.97, 3.3, year, ha="center", fontsize=9, style="italic")
        ax.text(x + 0.97, 2.5, idea, ha="center", va="center", fontsize=8.2)
        ax.text(x + 0.97, 1.55, speed, ha="center", fontsize=8, color="#222")
        if k < len(fixes):
            ax.annotate("", (x + 2.27, 2.9), (x + 1.98, 2.9), arrowprops=dict(arrowstyle="-|>", lw=2))
            ax.text(x + 2.12, 0.95, "fixes:\n" + fixes[k], ha="center", va="top", fontsize=8, color=RED)
    ax.text(8, 5.0, "Each step removes the bottleneck of the previous one (speeds are rough, different GPUs - read the trend)",
            ha="center", fontsize=11)
    plt.show()
    """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · The fully-convolutional trick: a classifier *is* a sliding window

    The naive detector (and the 2013 OverFeat paper) slides a classifier over the image. That
    is thousands of forward passes - but look at what a ResNet does: convolutions, then **global
    average pooling**, then a linear layer. A linear layer applied after averaging equals the
    average of the linear layer applied *at every position*, and a linear layer applied at every
    position is a **1x1 convolution**. So: drop the pooling, turn `fc` into a 1x1 conv, feed a
    bigger image, and you get a **map of class scores** - one forward pass, a sliding window with
    stride 32 for free, all computation shared between overlapping windows.

    We use an ImageNet ResNet-50 (it knows ~120 dog breeds and 5 cat classes) on a canvas with
    several Pet images pasted on it. (Swap in your D1-04 fine-tuned Pet model if you have it.)
    """),
    code(r"""
    from PIL import Image
    net = torchvision.models.resnet50(weights="IMAGENET1K_V2").eval()
    body = nn.Sequential(*list(net.children())[:-2])          # everything up to layer4 (no pooling, no fc)
    fc_conv = nn.Conv2d(2048, 1000, 1)                        # the fc layer as a 1x1 convolution
    fc_conv.weight.data = net.fc.weight.data[:, :, None, None].clone(); fc_conv.bias.data = net.fc.bias.data.clone()
    norm = lambda x: (x - torch.tensor(torchkit.IMAGENET_MEAN)[:, None, None]) / torch.tensor(torchkit.IMAGENET_STD)[:, None, None]

    # sanity check on one 224 px image: mean of the score map == the classifier's logits
    x224 = norm(pet_val[0][0])[None]
    with torch.no_grad():
        a = net(x224); b = fc_conv(body(x224)).mean((2, 3))
    print("max |classifier - mean(score map)| =", float((a - b).abs().max()))

    # a 448 x 704 canvas with three pets (from the validation set) on a textured background
    rng = np.random.default_rng(3)
    canvas = np.clip(0.55 + 0.08 * rng.normal(size=(448, 704, 1)) + np.linspace(-.1, .1, 704)[None, :, None], 0, 1).repeat(3, 2)
    cat_idx = next(i for i in range(len(pet_val)) if pet_val.image_path(i).stem[0].isupper())
    dog_idx = [i for i in range(len(pet_val)) if pet_val.image_path(i).stem[0].islower()][5:7]
    for i, (y, x, s) in zip([cat_idx] + dog_idx, [(40, 30, 230), (190, 300, 230), (60, 500, 190)]):
        im = Image.open(pet_val.image_path(i)).convert("RGB"); r = s / max(im.size)
        im = np.asarray(im.resize((int(im.width * r), int(im.height * r)))) / 255.0
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
    xc = norm(torch.tensor(canvas, dtype=torch.float32).permute(2, 0, 1))[None]
    with torch.no_grad():
        body(xc)                                              # warm-up (memory allocation)
    t0 = time.perf_counter()
    with torch.no_grad():
        prob = fc_conv(body(xc)).softmax(1)[0]                # (1000, 14, 22): a class distribution per position
    t_fc = time.perf_counter() - t0
    dog_map = prob[151:269].sum(0).numpy(); cat_map = prob[281:286].sum(0).numpy()
    print(f"score map {tuple(prob.shape)} in ONE forward pass: {t_fc:.2f} s  "
          f"(compute = {448 * 704 / 224 ** 2:.1f} crops' worth of 224 x 224)")
    """),
    code(r"""
    # The same thing done the slow way: crop 224 x 224 windows with stride 64 and classify each
    wins = [(y, x) for y in range(0, 448 - 223, 64) for x in range(0, 704 - 223, 64)]
    crops = torch.stack([xc[0, :, y:y + 224, x:x + 224] for y, x in wins])
    t0 = time.perf_counter()
    with torch.no_grad():
        pw = net(crops).softmax(1)
    t_sw = time.perf_counter() - t0
    ny, nx = len(range(0, 448 - 223, 64)), len(range(0, 704 - 223, 64))
    sw_dog = pw[:, 151:269].sum(1).reshape(ny, nx).numpy(); sw_cat = pw[:, 281:286].sum(1).reshape(ny, nx).numpy()

    fig, axes = plt.subplots(2, 3, figsize=(16, 7.8))
    axes[0, 0].imshow(canvas); axes[0, 0].set_title("input canvas 448 x 704"); axes[0, 0].axis("off")
    viz.heatmap_overlay(axes[0, 1], canvas, dog_map, alpha=0.55); axes[0, 1].set_title(f"fully conv.: P(dog)  [1 pass, {t_fc:.1f} s]")
    viz.heatmap_overlay(axes[0, 2], canvas, cat_map, alpha=0.55); axes[0, 2].set_title("fully conv.: P(cat)  (stride 32)")
    axes[1, 0].imshow(canvas); axes[1, 0].axis("off"); axes[1, 0].set_title(f"{len(wins)} sliding windows (stride 64)")
    for (y, x) in wins[::5]:
        axes[1, 0].add_patch(Rectangle((x, y), 224, 224, fill=False, ec=ORANGE, lw=1, alpha=0.7))
    for ax, m, nm in [(axes[1, 1], sw_dog, "dog"), (axes[1, 2], sw_cat, "cat")]:
        ax.imshow(canvas); ax.imshow(m, cmap="jet", alpha=0.55, vmin=0, vmax=1, extent=(112, 112 + 64 * nx, 112 + 64 * ny, 112))
        ax.set_xlim(0, 704); ax.set_ylim(448, 0); ax.axis("off")
        ax.set_title(f"sliding window: P({nm})  [{len(wins)} crops, {t_sw:.1f} s]")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Both maps light up on the animals - but the fully-convolutional one is computed in a single
    pass and at a finer stride (32 px instead of 64). This is the seed of every one-stage
    detector: *a convnet's feature map is already a grid of window descriptors*. What is missing
    is (a) a box that fits the object instead of a fixed window, and (b) a way to handle very
    different object sizes. Region proposals were the first answer to both.

    ### Region proposals: Selective Search in 40 lines

    R-CNN did not slide windows; it asked a bottom-up grouping algorithm for ~2000 "object-like"
    regions. **Selective Search** over-segments the image into superpixels, then repeatedly merges
    the two most similar *neighbouring* regions (similar colour, small, filling their joint
    bounding box well) and emits the bounding box of every region it ever forms. Objects appear
    at some level of the hierarchy. OpenCV-contrib ships the real thing
    (`cv2.ximgproc.segmentation`); we fall back to `day2.mini_selective_search`, a NumPy version of
    the same algorithm, when it is not installed.
    """),
    code(r"""
    try:
        import cv2
        ss_impl = "OpenCV ximgproc"
        def selective_search(img):
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            ss.setBaseImage((img * 255).astype(np.uint8)[..., ::-1]); ss.switchToSelectiveSearchFast()
            r = ss.process()[:500].astype(float)                   # x, y, w, h
            return boxes.xywh_to_xyxy(r), None
    except (ImportError, AttributeError):
        ss_impl = "cvcourse.day2.mini_selective_search (OpenCV-contrib not installed)"
        def selective_search(img):
            props, seg, _ = day2.mini_selective_search(img, n_segments=120)
            return props, seg
    print("Selective Search implementation:", ss_impl)

    pi = 40
    im = Image.open(pet_val.image_path(pi)).convert("RGB"); r = 192 / max(im.size)
    ss_img = np.asarray(im.resize((int(im.width * r), int(im.height * r)))) / 255.0
    ss_gt = pet_val[pi][1]["boxes"].numpy() / 224 * np.array([ss_img.shape[1], ss_img.shape[0]] * 2)
    t0 = time.perf_counter(); props, seg = selective_search(ss_img); t_ss = time.perf_counter() - t0
    best = boxes.box_iou(ss_gt, props)[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    if seg is not None:
        axes[0].imshow(seg % 20, cmap="tab20"); axes[0].set_title(f"initial superpixels ({seg.max()})")
    else:
        axes[0].imshow(ss_img); axes[0].set_title("input")
    axes[1].imshow(ss_img); axes[1].set_title(f"{len(props)} proposals (every 3rd drawn), {t_ss:.1f} s")
    for b in props[::3]:
        axes[1].add_patch(Rectangle(b[:2], *(b[2:] - b[:2]), fill=False, ec=viz.color(int(b.sum()) % 10), lw=0.6))
    top = np.argsort(-best)[:3]
    axes[2].imshow(ss_img); axes[2].set_title(f"best proposals for the head: IoU {best[top[0]]:.2f}, {best[top[1]]:.2f}, {best[top[2]]:.2f}")
    viz.draw_boxes(axes[2], ss_gt, edgecolor=GREEN, lw=3)
    viz.draw_boxes(axes[2], props[top], edgecolor=RED, lw=1.5)
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()
    print(f"proposal recall at IoU 0.5 for the head box: {(best >= 0.5).any()}  (R-CNN then ran the CNN on each proposal...)")
    """),

    # ------------------------------------------------------------------ 7
    md(r"""
    ## 7 · Non-maximum suppression

    A good detector fires *several* times on each object (neighbouring windows, anchors or grid
    cells all see it). **Greedy NMS** keeps the best and removes its look-alikes:

    1. take the highest-scoring remaining box, **keep** it;
    2. **remove** every remaining box with IoU $\ge$ threshold (typically 0.5) to it;
    3. repeat until no boxes remain.

    It is a hand-written post-processing step, not learned - remember that, it comes back in D2-04.
    """),
    code(r"""
    # Candidate detections around the objects of a shapes image
    nms_img, nms_gt, nms_lab, _ = shapes.make_sample(np.random.default_rng(16), 160, max_objects=3, min_r=16, max_r=28)
    rng = np.random.default_rng(2)
    cand, cand_s = [], []
    for g in nms_gt:
        for k in range(5):
            cand.append(g + rng.normal(0, 4, 4)); cand_s.append(rng.uniform(0.5, 0.97))
    cand = np.array(cand); cand_s = np.array(cand_s)
    print(f"{len(cand)} candidates for {len(nms_gt)} objects")
    """),
    md(r"""
    ### Exercise 6 · greedy NMS

    Return the indices of the kept boxes (highest score first). Use `boxes.box_iou`.
    """),
    exercise(
        stub=r"""
        def nms(bxs, scores, iou_threshold=0.5):
            order = np.argsort(-scores)
            keep = []
            while len(order) > 0:
                # YOUR CODE HERE: keep order[0], drop the rest whose IoU with it is >= threshold
                raise NotImplementedError
            return np.array(keep)

        tests.check_nms(nms)
        """,
        solution=r"""
        def nms(bxs, scores, iou_threshold=0.5):
            order = np.argsort(-scores)
            keep = []
            while len(order) > 0:
                i = order[0]
                keep.append(int(i))
                ious = boxes.box_iou(bxs[i][None], bxs[order[1:]])[0]
                order = order[1:][ious < iou_threshold]
            return np.array(keep)

        tests.check_nms(nms)
        lib = torchvision.ops.nms(torch.tensor(cand), torch.tensor(cand_s), 0.5).numpy()
        print("same result as torchvision.ops.nms:", np.array_equal(np.sort(nms(cand, cand_s, 0.5)), np.sort(lib)))
        """),
    code(r"""
    # Greedy NMS step by step (reference implementation, recording every iteration)
    order = list(np.argsort(-cand_s)); kept, removed_at = [], {}
    snapshots = []
    while order:
        i = order.pop(0); kept.append(i)
        ious = boxes.box_iou(cand[i][None], cand[order])[0] if order else np.array([])
        gone = [o for o, v in zip(order, ious) if v >= 0.5]
        order = [o for o, v in zip(order, ious) if v < 0.5]
        snapshots.append((list(kept), list(gone), list(order)))
    fig, axes = plt.subplots(1, len(snapshots) + 1, figsize=(4 * (len(snapshots) + 1), 4.3))
    axes[0].imshow(nms_img); axes[0].set_title(f"{len(cand)} candidates")
    viz.draw_boxes(axes[0], cand, scores=cand_s, edgecolor=GREY, lw=1)
    for ax, (k, gone, rest) in zip(axes[1:], snapshots):
        ax.imshow(nms_img)
        viz.draw_boxes(ax, cand[rest], edgecolor=GREY, lw=1)
        viz.draw_boxes(ax, cand[gone], edgecolor=RED, lw=1.2)
        viz.draw_boxes(ax, cand[k[:-1]], edgecolor=GREEN, lw=2)
        viz.draw_boxes(ax, cand[k[-1:]], scores=cand_s[k[-1:]], edgecolor=GREEN, lw=3.5)
        ax.set_title(f"step {len(k)}: keep 1 (thick), suppress {len(gone)} (red)", fontsize=10)
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### When greedy NMS hurts: crowds, and Soft-NMS

    NMS assumes *one object per neighbourhood*. Two people standing close together, or a
    cat in front of a dog, can overlap with IoU > 0.5 - and greedy NMS deletes the second
    (correct!) detection. **Soft-NMS** (Bodla et al. 2017) does not delete: it *decays* the
    score of each overlapping box, $s_i \leftarrow s_i\, e^{-\mathrm{IoU}(M, b_i)^2/\sigma}$, so heavy
    duplicates sink to near zero while a moderately overlapping neighbour survives with a lower
    score. Recall goes up at no cost in precision.

    ### Exercise 7 · Soft-NMS (Gaussian)

    Repeatedly pick the remaining box with the highest *current* score, record it, then decay
    the scores of the others and drop those below `score_threshold`. Return
    `(kept indices, their scores at the time they were picked)`.
    """),
    exercise(
        stub=r"""
        def soft_nms(bxs, scores, sigma=0.5, score_threshold=0.001):
            scores = np.asarray(scores, float).copy()
            idx = np.arange(len(scores))
            keep, kept_scores = [], []
            while idx.size > 0:
                # YOUR CODE HERE: m = remaining index with max score; record it; remove it from idx;
                # decay scores[idx] *= exp(-IoU(m, idx)^2 / sigma); keep only scores > score_threshold
                raise NotImplementedError
            return np.array(keep), np.array(kept_scores)

        tests.check_soft_nms(soft_nms)
        """,
        solution=r"""
        def soft_nms(bxs, scores, sigma=0.5, score_threshold=0.001):
            scores = np.asarray(scores, float).copy()
            idx = np.arange(len(scores))
            keep, kept_scores = [], []
            while idx.size > 0:
                m = idx[np.argmax(scores[idx])]
                keep.append(int(m)); kept_scores.append(float(scores[m]))
                idx = idx[idx != m]
                if idx.size == 0:
                    break
                ious = boxes.box_iou(bxs[m][None], bxs[idx])[0]
                scores[idx] *= np.exp(-ious ** 2 / sigma)
                idx = idx[scores[idx] > score_threshold]
            return np.array(keep), np.array(kept_scores)

        tests.check_soft_nms(soft_nms)
        """),
    code(r"""
    # Two overlapping pedestrians-like objects, each with its own duplicates
    crowd_gt = np.array([[30, 20, 90, 150], [44, 22, 104, 152]], float)
    print("IoU between the two people:", round(boxes.iou_single(*crowd_gt), 2))
    crowd = np.array([crowd_gt[0] + [1, 2, -1, 0], crowd_gt[0] + [3, -3, 4, 1], crowd_gt[1] + [1, 1, 1, 1], crowd_gt[1] + [-4, 2, -3, 0]])
    crowd_s = np.array([0.95, 0.80, 0.90, 0.70])
    hard = boxes.nms(crowd, crowd_s, 0.5)
    soft_k, soft_s = boxes.soft_nms(crowd, crowd_s, sigma=0.5)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.3]))
    for ax, title, k, s in [(axes[0], "candidates", np.arange(4), crowd_s), (axes[1], f"greedy NMS @0.5: {len(hard)} kept", hard, crowd_s[hard]),
                            (axes[2], "Soft-NMS: all kept, re-scored", soft_k, soft_s)]:
        ax.set_xlim(0, 150); ax.set_ylim(170, 0); ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
        viz.draw_boxes(ax, crowd_gt, edgecolor=GREEN, lw=4, alpha=0.3)
        for i, sc in zip(k, s):
            b = crowd[i]; c = BLUE if i < 2 else ORANGE
            ax.add_patch(Rectangle(b[:2], *(b[2:] - b[:2]), fill=False, ec=c, lw=1.8))
            ax.text(b[0] + 2 if i < 2 else b[2] - 2, b[3] - 6 - 14 * (i % 2), f"#{i} {sc:.2f}", color="white", fontsize=8,
                    ha="left" if i < 2 else "right", bbox=dict(fc=c, ec="none", pad=1))
    x = np.arange(4)
    axes[3].bar(x - 0.2, crowd_s, 0.4, color=GREY, label="before")
    after = np.zeros(4); after[soft_k] = soft_s
    axes[3].bar(x + 0.2, after, 0.4, color=BLUE, label="after Soft-NMS")
    axes[3].set_xticks(x); axes[3].set_xticklabels([f"#{i}\n{'person A' if i < 2 else 'person B'}" for i in x])
    axes[3].legend(); axes[3].set_title("scores (duplicates sink, person B survives)", fontsize=10)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Class-wise NMS

    A circle and its bounding square can overlap perfectly and both be correct if they are
    *different classes* (person on a horse!). Detectors therefore run NMS **per class**. The
    vectorised trick used by `torchvision.ops.batched_nms`: shift every box by
    `label * (max coordinate + 1)` so that boxes of different classes can never overlap, then run
    one ordinary NMS.
    """),
    code(r"""
    cw_boxes = np.array([[20, 20, 80, 80], [24, 22, 82, 84], [22, 18, 78, 82], [90, 30, 140, 90]], float)
    cw_scores = np.array([0.9, 0.85, 0.6, 0.7]); cw_labels = np.array([0, 1, 0, 1])
    agnostic = boxes.nms(cw_boxes, cw_scores, 0.5)
    classwise = torchvision.ops.batched_nms(torch.tensor(cw_boxes), torch.tensor(cw_scores), torch.tensor(cw_labels), 0.5).numpy()
    offset = (cw_boxes.max() + 1) * cw_labels[:, None]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, title, k, bx in [(axes[0], f"class-agnostic NMS: keeps {sorted(agnostic.tolist())}", agnostic, cw_boxes),
                             (axes[1], f"class-wise NMS: keeps {sorted(classwise.tolist())}", classwise, cw_boxes),
                             (axes[2], "the trick: offset boxes by label", np.arange(4), cw_boxes + offset)]:
        for i in k:
            b = bx[i]; c = viz.color(cw_labels[i])
            ax.add_patch(Rectangle(b[:2], *(b[2:] - b[:2]), fill=False, ec=c, lw=2))
            ax.text(b[0], b[1] - 3 - 9 * (i == 2) + 70 * (i == 1), f"#{i} {shapes.CLASS_NAMES[cw_labels[i]]} {cw_scores[i]:.2f}", color=c, fontsize=8)
        ax.set_xlim(0, 300 if ax is axes[2] else 150); ax.set_ylim(260 if ax is axes[2] else 110, 0); ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Hard-negative mining

    A detector looks at thousands of candidate windows per image and only a handful contain
    objects. If you train on all of them, the loss is dominated by **easy negatives** (empty
    background the classifier already rejects); if you sample negatives uniformly, you mostly
    pick easy ones too. **Hard-negative mining** (R-CNN, SSD) trains on the negatives with the
    *highest loss*: typically all positives + the top-scoring negatives at a 1:3 ratio. Below, a
    toy "detector" that fires on colourful windows scores every 32 x 32 window of a shapes image.
    """),
    code(r"""
    hn_img, hn_gt, _, _ = shapes.make_sample(np.random.default_rng(4), 160, max_objects=4, min_r=12, max_r=20)
    wins = np.array([[x, y, x + 32, y + 32] for y in range(0, 129, 4) for x in range(0, 129, 4)], float)
    sat = hn_img.max(-1) - hn_img.min(-1)
    score = np.array([sat[int(b[1]):int(b[3]), int(b[0]):int(b[2])].mean() for b in wins])
    score = 1 / (1 + np.exp(-30 * (score - 0.25)))                 # toy classifier: P(object | colourful window)
    is_pos = boxes.box_iou(wins, hn_gt).max(1) >= 0.5
    neg_loss = -np.log(1 - score[~is_pos] + 1e-6)                 # BCE of each negative
    n_hard = 3 * is_pos.sum()
    hard_idx = np.where(~is_pos)[0][np.argsort(-neg_loss)[:n_hard]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
    axes[0].imshow(hn_img); axes[0].set_title(f"{len(wins)} windows: {is_pos.sum()} positive (IoU >= 0.5)")
    viz.draw_boxes(axes[0], wins[is_pos], edgecolor=GREEN, lw=1.5)
    axes[1].imshow(hn_img); axes[1].set_title(f"the {n_hard} hardest negatives (1:3 ratio)")
    viz.draw_boxes(axes[1], wins[hard_idx], edgecolor=RED, lw=1)
    for ax in axes[:2]: ax.axis("off")
    axes[2].hist(neg_loss, bins=40, color=GREY, log=True); axes[2].axvline(np.sort(neg_loss)[-n_hard], color=RED, ls="--", label="hard-negative cut")
    axes[2].set_xlabel("loss of a negative window"); axes[2].set_ylabel("count (log)"); axes[2].legend()
    axes[2].set_title(f"{(~is_pos).sum()} negatives: {100 * (neg_loss < 0.05).mean():.0f}% are easy (loss < 0.05)")
    plt.tight_layout(); plt.show()
    print(f"share of total negative loss carried by the {n_hard} hard negatives: {np.sort(neg_loss)[-n_hard:].sum() / neg_loss.sum():.0%}")
    """),
    md(r"""
    The hard negatives are exactly the windows that *half* cover a shape - the confusing cases
    the detector must learn to reject. Focal loss (D2-03) replaces this hand-made sampling with a
    smooth re-weighting of every example, which is why one-stage detectors could finally train
    on all anchors.

    **NMS-free detectors.** DETR (2020) and YOLOv10 (2024) are trained with a *one-to-one*
    assignment between predictions and objects (Hungarian matching, D2-04), so the network itself
    learns not to produce duplicates and the NMS step - with its threshold and its latency - disappears.
    """),

    # ------------------------------------------------------------------ 8
    md(r"""
    ## 8 · Summary

    * A detector outputs a variable-length set of `(box, class, score)`. Boxes come in xyxy / xywh /
      cxcywh, in pixels or normalised - know which one every tool expects.
    * IoU = intersection / union; vectorise it with broadcasting. IoU falls to 0.5 after a shift of
      one third of the box width, and the same pixel error is much worse for small objects.
    * $1-\mathrm{IoU}$ has no gradient without overlap; GIoU (enclosing box), DIoU (centre distance)
      and CIoU (+ aspect ratio) fix that and are the box losses of DETR and YOLO.
    * Evaluation: sort by score, greedy matching (duplicates are FPs), PR curve, AP = area under the
      monotone envelope, mAP = mean over classes; COCO averages over IoU 0.5:0.95 and reports S / M / L.
    * The lineage is a chain of removed bottlenecks; the fully-convolutional trick turns a classifier
      into a dense detector in one pass.
    * NMS removes duplicates greedily; Soft-NMS decays instead of deleting (crowds); NMS is run per class;
      hard-negative mining keeps the loss from drowning in easy background.

    ### Self-check
    1. Box `(10, 20, 50, 80)` in xyxy: what are its xywh and cxcywh forms? Its area?
    2. Two 40 x 40 boxes, one shifted 10 px right and 10 px down. IoU? (Draw it.)
    3. Why is $1-\mathrm{IoU}$ useless as a loss for a prediction that does not overlap the GT? What does GIoU add?
    4. A detector finds every object but also produces one duplicate per object, with a lower score. What happens to its AP@0.5?
    5. Why can a mAP@0.5 of 0.95 coexist with a mAP@[.5:.95] of 0.55?
    6. In a crowded scene, what goes wrong with an NMS threshold of 0.3? With 0.9?

    ### Go deeper
    * Girshick et al., *Rich feature hierarchies for accurate object detection and semantic segmentation* (R-CNN, 2014) and Girshick, *Fast R-CNN* (2015).
    * Uijlings et al., *Selective Search for Object Recognition* (IJCV 2013).
    * Rezatofighi et al., *Generalized Intersection over Union* (CVPR 2019); Zheng et al., *Distance-IoU Loss* (AAAI 2020, DIoU and CIoU).
    * COCO detection evaluation documentation (cocodataset.org/#detection-eval) - the exact protocol behind mAP@[.5:.95].
    * Padilla et al., *A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit* (Electronics 2021).
    * Bodla et al., *Soft-NMS - Improving Object Detection With One Line of Code* (ICCV 2017).
    """),
]
