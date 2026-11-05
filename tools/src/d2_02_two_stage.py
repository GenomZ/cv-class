from nb import md, code, exercise, header

NAME = "D2-02-two-stage"

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
        "D2-02 · Two-stage detectors: anchors, matching, RPN, RoIAlign",
        steps=[
            "Anchors: a grid of reference boxes at several scales and aspect ratios",
            "Matching anchors to ground truth (positive / negative / ignore) and the imbalance it creates",
            "Box encoding (tx, ty, tw, th), decoding, smooth-L1",
            "The Region Proposal Network: head, training on shapes, proposal filtering",
            "RoI pooling vs RoIAlign: the quantisation problem and bilinear sampling",
            "The second stage and the four-term Faster R-CNN loss",
            "Fine-tune torchvision's Faster R-CNN on Penn-Fudan pedestrians and measure mAP",
        ],
        topics=[
            "Generate anchors for any feature map and count them",
            "Implement the IoU-based matching rule and see why only ~1% of anchors are positive",
            "Encode / decode boxes relative to anchors and explain the log for width and height",
            "Build and train an RPN head; turn its dense output into a few hundred proposals",
            "Implement RoIAlign and verify it against torchvision",
            "Adapt a pretrained Faster R-CNN to a new dataset in ~15 lines",
        ],
        minutes=90,
        prereq="D2-01 (boxes, IoU, NMS, mAP).",
    ),
    md(r"""
    ## 0 · Setup
    """),
    code(SETUP),
    md(r"""
    ## The big picture

    Faster R-CNN (Ren et al. 2015) is the model this notebook takes apart. One backbone computes
    a feature map *once*; a small **Region Proposal Network** slides over it and scores a fixed
    set of reference boxes (**anchors**) for "objectness"; the best few hundred become
    **proposals**; **RoIAlign** cuts a fixed-size feature tile out of the map for each proposal;
    a second head classifies each tile and refines its box. Two stages: *where might objects be?*
    then *what exactly is there?*
    """),
    code(r"""
    fig, ax = plt.subplots(figsize=(15, 3.6)); ax.axis("off"); ax.set_xlim(0, 15); ax.set_ylim(0, 3.6)
    blocks = [(0.1, "image\n3 x H x W", "#dddddd"), (2.1, "backbone\n(ResNet + FPN)", "#9ecae1"), (4.3, "feature map\nC x H/16 x W/16", "#c6dbef"),
              (6.5, "RPN head\nobjectness A\ndeltas 4A per cell", ORANGE), (8.7, "proposals\ndecode, NMS,\ntop-N", "#fdd0a2"),
              (10.9, "RoIAlign\nN x C x 7 x 7", GREEN), (13.0, "box head\nclass + box\nper RoI", "#B279A2")]
    for x, t, c in blocks:
        ax.add_patch(FancyBboxPatch((x, 1.0), 1.8, 1.8, boxstyle="round,pad=0.05", fc=c, ec="k", alpha=0.85))
        ax.text(x + 0.9, 1.9, t, ha="center", va="center", fontsize=9)
    for x0, x1 in [(1.95, 2.1), (3.95, 4.3), (6.15, 6.5), (8.35, 8.7), (10.55, 10.9), (12.75, 13.0)]:
        ax.annotate("", (x1, 1.9), (x0, 1.9), arrowprops=dict(arrowstyle="-|>", lw=1.8))
    ax.annotate("", (11.4, 2.85), (5.2, 2.85), arrowprops=dict(arrowstyle="-|>", lw=1.5, ls="--", color=GREY, connectionstyle="arc3,rad=-0.25"))
    ax.text(8.3, 3.45, "the same feature map is cropped for every proposal", ha="center", fontsize=9, color="#555")
    ax.text(7.6, 0.55, "stage 1: where? (class-agnostic)", ha="center", fontsize=10, color=ORANGE, weight="bold")
    ax.text(12.9, 0.55, "stage 2: what? (per class)", ha="center", fontsize=10, color="#8a4f86", weight="bold")
    plt.show()
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · Anchors

    A feature map with stride $s$ has one cell per $s \times s$ block of input pixels; cell
    $(i, j)$ is centred at $\big((j + 0.5)s,\ (i + 0.5)s\big)$ in the image. At every cell we
    place the same family of $A$ **anchors**: every combination of a few *scales* (box side for a
    square anchor) and *aspect ratios* $r = h/w$, keeping the area fixed:

    $$w = \frac{\text{scale}}{\sqrt r}, \qquad h = \text{scale} \cdot \sqrt r .$$

    The network never predicts boxes from scratch - it predicts **corrections** to these anchors.
    A feature map of $H_f \times W_f$ cells therefore produces $H_f W_f A$ candidate boxes.
    """),
    md(r"""
    ### Exercise 1 · `make_anchors`

    Return all anchors of one level as `(fh * fw * A, 4)` xyxy, ordered cell by cell (row, column,
    anchor). Build the `A` base boxes centred at 0, the grid of centres, and add them with broadcasting.
    """),
    exercise(
        stub=r"""
        def make_anchors(feature_size, stride, scales, ratios):
            fh, fw = feature_size
            s, r = np.meshgrid(np.asarray(scales, float), np.asarray(ratios, float), indexing="ij")
            # YOUR CODE HERE:
            # 1) widths w = s / sqrt(r), heights h = s * sqrt(r)   (flatten -> A values)
            # 2) base boxes (A, 4) = [-w/2, -h/2, w/2, h/2]
            # 3) centres cx = (j + 0.5) * stride, cy = (i + 0.5) * stride on a (fh, fw) grid
            # 4) centres (fh*fw, 1, 4) + base (1, A, 4) -> reshape (-1, 4)
            raise NotImplementedError

        tests.check_make_anchors(make_anchors)
        """,
        solution=r"""
        def make_anchors(feature_size, stride, scales, ratios):
            fh, fw = feature_size
            s, r = np.meshgrid(np.asarray(scales, float), np.asarray(ratios, float), indexing="ij")
            w = (s / np.sqrt(r)).reshape(-1)
            h = (s * np.sqrt(r)).reshape(-1)
            base = np.stack([-w / 2, -h / 2, w / 2, h / 2], -1)                  # (A, 4)
            cx, cy = np.meshgrid((np.arange(fw) + 0.5) * stride, (np.arange(fh) + 0.5) * stride)
            centres = np.stack([cx, cy, cx, cy], -1).reshape(-1, 1, 4)          # (fh*fw, 1, 4)
            return (centres + base[None]).reshape(-1, 4)

        tests.check_make_anchors(make_anchors)
        """),
    code(r"""
    from PIL import Image
    pf = torchkit.get_penn_fudan()
    img0 = pf[3][0]
    s = 512 / max(img0.shape[-2:])
    img0 = F.interpolate(img0[None], scale_factor=s, mode="bilinear", antialias=True)[0].permute(1, 2, 0).numpy()
    H, W = img0.shape[:2]
    scales, ratios = [64, 128, 256], [0.5, 1.0, 2.0]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    ax = axes[0]; ax.imshow(img0); ax.set_title("one cell (stride 32): 3 scales x 3 ratios = 9 anchors")
    cell = (6, 8)                                   # (row, col)
    fam = boxes.make_anchors((1, 1), 32, scales, ratios) + [cell[1] * 32, cell[0] * 32] * 2
    for k, b in enumerate(fam):
        ax.add_patch(Rectangle(b[:2], *(b[2:] - b[:2]), fill=False, ec=[RED, ORANGE, BLUE][k // 3], lw=2, ls=["-", "--", ":"][k % 3]))
    ax.plot(*((np.array(cell[::-1]) + 0.5) * 32), "w*", ms=14, mec="k")
    ax.set_xlim(-150, W + 150); ax.set_ylim(H + 150, -150)
    for c, sc in zip([RED, ORANGE, BLUE], scales):
        ax.plot([], [], color=c, lw=2, label=f"scale {sc}")
    for ls, r in zip(["-", "--", ":"], ratios):
        ax.plot([], [], color="k", ls=ls, label=f"ratio h/w = {r}")
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]; ax.imshow(img0)
    fh, fw = H // 32, W // 32
    grid = boxes.make_anchors((fh, fw), 32, [64], [1.0])
    ctr = (grid[:, :2] + grid[:, 2:]) / 2
    ax.plot(ctr[:, 0], ctr[:, 1], ".", color="yellow", ms=4)
    for x in range(0, W + 1, 32): ax.axvline(x, color="white", lw=0.4, alpha=0.6)
    for y in range(0, H + 1, 32): ax.axhline(y, color="white", lw=0.4, alpha=0.6)
    for k in range(0, len(grid), 37):
        b = grid[k]; ax.add_patch(Rectangle(b[:2], *(b[2:] - b[:2]), fill=False, ec=ORANGE, lw=1.2))
    ax.set_title(f"stride 32 -> {fh} x {fw} cells; every cell gets the same 9 anchors (a few 64 px ones drawn)")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    plt.tight_layout(); plt.show()

    print(f"{'stride':>7} {'feature map':>12} {'anchors (A=9)':>14}")
    tot = 0
    for st in [8, 16, 32]:
        n = (512 // st) ** 2 * 9; tot += n
        print(f"{st:>7} {f'{512 // st} x {512 // st}':>12} {n:>14,d}")
    print(f"{'total':>7} {'':>12} {tot:>14,d}   for a 512 x 512 image - and maybe 5 of them contain an object")
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · Matching anchors to ground truth

    Training needs a target for every anchor. Faster R-CNN's RPN rule, from the IoU matrix
    `anchors x GT`:

    * **positive** (label 1) if its best IoU with any GT is $\ge 0.7$; it regresses to that GT;
    * **negative** (label 0, background) if its best IoU is $< 0.3$;
    * **ignore** (label -1) in between - ambiguous anchors are simply left out of the loss;
    * **low-quality matches**: every GT also makes its single best anchor positive, even if that IoU
      is below 0.7 - otherwise small or oddly shaped objects would get no positive anchor at all.

    ### Exercise 2 · `match_anchors`

    Return `(matched_gt_index, labels)` for `anchors (N, 4)` and `gt (G, 4)`. Use `boxes.box_iou`.
    """),
    exercise(
        stub=r"""
        def match_anchors(anchors, gt, pos_iou=0.7, neg_iou=0.3, allow_low_quality=True):
            ious = boxes.box_iou(anchors, gt)                 # (N, G)
            best_gt, best_iou = ious.argmax(1), ious.max(1)
            labels = np.full(len(anchors), -1)
            # YOUR CODE HERE: negatives, positives, then the low-quality rule
            # (hint: ious.argmax(0) is the best anchor of every GT)
            raise NotImplementedError
            return best_gt, labels

        tests.check_match_anchors(match_anchors)
        """,
        solution=r"""
        def match_anchors(anchors, gt, pos_iou=0.7, neg_iou=0.3, allow_low_quality=True):
            ious = boxes.box_iou(anchors, gt)                 # (N, G)
            best_gt, best_iou = ious.argmax(1), ious.max(1)
            labels = np.full(len(anchors), -1)
            labels[best_iou < neg_iou] = 0
            labels[best_iou >= pos_iou] = 1
            if allow_low_quality:
                best_anchor = ious.argmax(0)                  # (G,) best anchor of every GT
                labels[best_anchor] = 1
                best_gt[best_anchor] = np.arange(len(gt))
            return best_gt, labels

        tests.check_match_anchors(match_anchors)
        """),
    code(r"""
    # Matching on a shapes image: 128 px, stride 8 -> 16 x 16 cells, 3 scales x 3 ratios = 2304 anchors
    m_img, m_gt, m_lab, _ = shapes.make_sample(np.random.default_rng(7), 128)
    anchors = boxes.make_anchors((16, 16), 8, [16, 32, 48], [0.5, 1.0, 2.0])
    for thr in (0.7, 0.5):
        print(f"positive threshold {thr}: {(boxes.match_anchors(anchors, m_gt, thr, 0.3)[1] == 1).sum()} positive anchors for {len(m_gt)} objects")
    idx, lab = boxes.match_anchors(anchors, m_gt, 0.5, 0.3)        # 0.5 for our tiny images (see below)
    max_iou = boxes.box_iou(anchors, m_gt).max(1)
    ctr = (anchors[:, :2] + anchors[:, 2:]) / 2

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    axes[0].imshow(m_img); axes[0].set_title(f"positive anchors ({(lab == 1).sum()}), coloured by their GT")
    for b, g in zip(anchors[lab == 1], idx[lab == 1]):
        axes[0].add_patch(Rectangle(b[:2], *(b[2:] - b[:2]), fill=False, ec=viz.color(g + 2), lw=1.2))
    viz.draw_boxes(axes[0], m_gt, edgecolor="k", lw=2.5)
    axes[1].imshow(m_img); axes[1].set_title(f"anchor centres: ignore ({(lab == -1).sum()}) / negative ({(lab == 0).sum()})")
    ig = lab == -1
    axes[1].scatter(ctr[lab == 0, 0], ctr[lab == 0, 1], s=3, c=RED, alpha=0.3)
    for b in anchors[ig]:
        axes[1].add_patch(Rectangle(b[:2], *(b[2:] - b[:2]), fill=False, ec=GREY, lw=0.6, alpha=0.6))
    for ax in axes[:2]: ax.set_xlim(0, 128); ax.set_ylim(128, 0); ax.axis("off")
    axes[2].hist([max_iou[lab == 0], max_iou[lab == -1], max_iou[lab == 1]], bins=30, stacked=True,
                 color=[RED, GREY, GREEN], label=["negative", "ignore", "positive"], log=True)
    axes[2].axvline(0.3, color="k", ls="--"); axes[2].axvline(0.5, color="k", ls="--"); axes[2].axvline(0.7, color=GREY, ls=":")
    axes[2].set_xlabel("max IoU of each anchor with any GT"); axes[2].set_ylabel("anchors (log scale)"); axes[2].legend()
    axes[2].set_title(f"{100 * (lab == 1).mean():.1f}% positive, {100 * (lab == 0).mean():.0f}% negative")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    With the 0.7 rule only the forced "best anchor per GT" matches survive here: on a 128 px image
    a stride-8 anchor can be 4 px off-centre, and 4 px on a 24 px object already pushes IoU below 0.7.
    Faster R-CNN's thresholds assume ~800 px images where objects are large relative to the stride;
    for our tiny images we use **0.5 / 0.3** (RetinaNet uses 0.5 / 0.4). Anchor design and matching
    thresholds are dataset-dependent hyper-parameters - one of the motivations for anchor-free detectors.

    The histogram is the core difficulty of dense detection: **almost every anchor is
    background**. If the objectness loss averaged over all anchors, "always say background"
    would score 99% accuracy. The RPN therefore **samples** a minibatch of 256 anchors per image
    with at most 50% positives (Faster R-CNN; torchvision's default) - a sampling cousin of
    hard-negative mining. One-stage detectors cannot afford to throw anchors away; their answer
    is focal loss (D2-03).
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · Encoding boxes relative to anchors

    A positive anchor $a$ learns to regress its GT $g$. Instead of raw coordinates the target is
    a scale-free **delta**:

    $$t_x = \frac{g_x - a_x}{a_w},\quad t_y = \frac{g_y - a_y}{a_h},\quad t_w = \log\frac{g_w}{a_w},\quad t_h = \log\frac{g_h}{a_h}$$

    (centres and sizes, i.e. cxcywh). Dividing by the anchor size makes a "half an anchor to the
    right" mean the same for small and large anchors. The **log** makes size changes symmetric
    (twice as big = $+0.69$, half as big = $-0.69$) and keeps the predicted size positive after
    decoding with $\exp$. Decoding is the exact inverse.
    """),
    md(r"""
    ### Exercise 3 · `encode` / `decode`

    Both take aligned `(N, 4)` xyxy arrays. Convert with `boxes.xyxy_to_cxcywh` /
    `boxes.cxcywh_to_xyxy`.
    """),
    exercise(
        stub=r"""
        def encode(gt, anchors):
            g, a = boxes.xyxy_to_cxcywh(gt), boxes.xyxy_to_cxcywh(anchors)
            # YOUR CODE HERE -> (N, 4) deltas
            raise NotImplementedError

        def decode(deltas, anchors):
            a = boxes.xyxy_to_cxcywh(anchors)
            # YOUR CODE HERE -> (N, 4) xyxy
            raise NotImplementedError

        tests.check_encode_decode(encode, decode)
        """,
        solution=r"""
        def encode(gt, anchors):
            g, a = boxes.xyxy_to_cxcywh(gt), boxes.xyxy_to_cxcywh(anchors)
            return np.stack([(g[:, 0] - a[:, 0]) / a[:, 2], (g[:, 1] - a[:, 1]) / a[:, 3],
                             np.log(g[:, 2] / a[:, 2]), np.log(g[:, 3] / a[:, 3])], -1)

        def decode(deltas, anchors):
            a = boxes.xyxy_to_cxcywh(anchors)
            cx = a[:, 0] + deltas[:, 0] * a[:, 2]
            cy = a[:, 1] + deltas[:, 1] * a[:, 3]
            w = a[:, 2] * np.exp(deltas[:, 2])
            h = a[:, 3] * np.exp(deltas[:, 3])
            return boxes.cxcywh_to_xyxy(np.stack([cx, cy, w, h], -1))

        tests.check_encode_decode(encode, decode)
        """),
    code(r"""
    k = np.where(lab == 1)[0][len(np.where(lab == 1)[0]) // 3]
    a, g = anchors[k], m_gt[idx[k]]
    d = boxes.encode_boxes(g[None], a[None])[0]
    ac, gc = boxes.xyxy_to_cxcywh(a), boxes.xyxy_to_cxcywh(g)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    ax = axes[0]; ax.imshow(m_img)
    ax.add_patch(Rectangle(a[:2], *(a[2:] - a[:2]), fill=False, ec=ORANGE, lw=2.5, ls="--", label="anchor"))
    ax.add_patch(Rectangle(g[:2], *(g[2:] - g[:2]), fill=False, ec=GREEN, lw=2.5, label="GT"))
    ax.annotate("", gc[:2], ac[:2], arrowprops=dict(arrowstyle="->", lw=2, color="k"))
    ax.plot(*ac[:2], "o", color=ORANGE); ax.plot(*gc[:2], "o", color=GREEN)
    pad = 20
    ax.set_xlim(min(a[0], g[0]) - pad, max(a[2], g[2]) + pad); ax.set_ylim(max(a[3], g[3]) + pad, min(a[1], g[1]) - pad)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"tx={d[0]:+.2f}  ty={d[1]:+.2f}  tw={d[2]:+.2f}  th={d[3]:+.2f}", fontsize=10)
    r = np.linspace(0.25, 4, 200)
    axes[1].plot(r, np.log(r), color=BLUE, lw=2, label="t_w = log(g_w / a_w)")
    axes[1].plot(r, r - 1, color=GREY, ls="--", label="naive (g_w - a_w) / a_w")
    for x in [0.5, 2]: axes[1].plot(x, np.log(x), "o", color=RED)
    axes[1].text(2.05, np.log(2) - 0.35, "x2: +0.69", color=RED); axes[1].text(0.55, np.log(0.5) - 0.3, "/2: -0.69", color=RED)
    axes[1].set_xscale("log"); axes[1].set_xticks([0.25, 0.5, 1, 2, 4]); axes[1].set_xticklabels(["1/4", "1/2", "1", "2", "4"])
    axes[1].xaxis.set_minor_formatter(plt.NullFormatter()); axes[1].set_xlabel("size ratio GT / anchor (log axis)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    axes[1].set_title("log encoding: symmetric for growing and shrinking")
    x = np.linspace(-1.5, 1.5, 400)
    axes[2].plot(x, np.abs(x), color=GREY, label="L1"); axes[2].plot(x, 0.5 * x ** 2, color=ORANGE, label="L2 / 2")
    for beta, c in [(1 / 9, BLUE), (1.0, GREEN)]:
        axes[2].plot(x, boxes.smooth_l1(x, beta), color=c, lw=2, label=f"smooth-L1, beta={beta:.2g}")
    axes[2].set_ylim(0, 1.3); axes[2].legend(fontsize=8); axes[2].grid(alpha=.3); axes[2].set_xlabel("prediction - target")
    axes[2].set_title("smooth-L1: quadratic near 0, linear for outliers")
    plt.tight_layout(); plt.show()
    print("round trip decode(encode(g, a), a) == g:", np.allclose(boxes.decode_boxes(d[None], a[None])[0], g))
    """),
    md(r"""
    **Smooth-L1** (Huber) is the box loss of the R-CNN family: quadratic for errors below $\beta$
    (smooth gradient near the optimum), linear above (a badly matched anchor cannot dominate the
    batch the way it would with L2). Faster R-CNN's RPN uses $\beta = 1/9$.
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · The Region Proposal Network

    The RPN head is tiny: a $3\times 3$ conv (256 or 512 channels) + ReLU shared by two sibling
    $1\times 1$ convs. At every cell it outputs $A$ **objectness logits** ("is anchor $a$ of this cell
    on an object?") and $4A$ **deltas**. It is class-agnostic: it does not know about people or
    cats, only "thing vs stuff".
    """),
    code(r"""
    fig, ax = plt.subplots(figsize=(14, 4.4)); ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(-0.6, 4.2)
    def slab(x, y, w, h, d, c):
        for k in range(d - 1, -1, -1):
            ax.add_patch(Rectangle((x + 0.08 * k, y + 0.08 * k), w, h, fc=c, ec="k", lw=0.7))
    slab(0.3, 1.0, 1.8, 1.8, 5, "#c6dbef"); ax.text(1.35, 0.55, "feature map\n(N, C, 32, 32)", ha="center", va="top", fontsize=9)
    slab(4.0, 1.0, 1.8, 1.8, 5, "#fdd0a2"); ax.text(5.05, 0.55, "3x3 conv + ReLU\n(N, C, 32, 32)", ha="center", va="top", fontsize=9)
    slab(8.4, 2.5, 1.2, 1.0, 3, ORANGE)
    slab(8.4, 0.0, 1.2, 1.0, 8, "#9ecae1")
    ax.text(10.3, 3.0, "objectness logits (N, A, 32, 32)\n-> (N, 32*32*A)", fontsize=9, va="center")
    ax.text(10.3, 0.7, "box deltas (N, 4A, 32, 32)\n-> (N, 32*32*A, 4)", fontsize=9, va="center")
    ax.annotate("", (4.0, 1.9), (2.5, 1.9), arrowprops=dict(arrowstyle="-|>", lw=1.8))
    ax.annotate("", (8.35, 3.0), (6.3, 2.3), arrowprops=dict(arrowstyle="-|>", lw=1.5)); ax.text(6.9, 2.95, "1x1 conv, A", fontsize=9)
    ax.annotate("", (8.35, 0.5), (6.3, 1.5), arrowprops=dict(arrowstyle="-|>", lw=1.5)); ax.text(6.9, 0.65, "1x1 conv, 4A", fontsize=9)
    plt.show()
    """),
    md(r"""
    ### Exercise 4 · `RPNHead`

    `RPNHead(in_channels, num_anchors)`; `forward(x)` returns `(objectness, deltas)` with shapes
    `(N, A, H, W)` and `(N, 4A, H, W)`. Name the layers `conv`, `cls_logits`, `bbox_pred` (like torchvision).
    """),
    exercise(
        stub=r"""
        class RPNHead(nn.Module):
            def __init__(self, in_channels, num_anchors):
                super().__init__()
                # YOUR CODE HERE: self.conv (3x3, padding 1), self.cls_logits (1x1), self.bbox_pred (1x1)
                raise NotImplementedError

            def forward(self, x):
                # YOUR CODE HERE
                raise NotImplementedError

        tests.check_rpn_head(RPNHead)
        """,
        solution=r"""
        class RPNHead(nn.Module):
            def __init__(self, in_channels, num_anchors):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
                self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
                self.bbox_pred = nn.Conv2d(in_channels, 4 * num_anchors, 1)

            def forward(self, x):
                h = F.relu(self.conv(x))
                return self.cls_logits(h), self.bbox_pred(h)

        tests.check_rpn_head(RPNHead)
        """),
    md(r"""
    ### Train a tiny RPN on shapes (about a minute on CPU)

    `day2.TinyRPN` = a 5-layer backbone with stride 8 (128 px -> 16 x 16 cells) + **your** head
    (or the reference one if your checker failed), 9 anchors per cell. Targets come from
    `match_anchors` (0.5 / 0.3) + `encode_boxes`, computed once for the whole dataset. Each step samples 128
    anchors per image (<= 25% positives), and the loss is BCE on the sampled anchors' objectness +
    smooth-L1 on the positive anchors' deltas: exactly Faster R-CNN's RPN loss.
    """),
    code(r"""
    head_cls = RPNHead if tests.check_rpn_head(RPNHead, verbose=False) else day2.RPNHead
    print("using", "your RPNHead" if head_cls is RPNHead else "the reference day2.RPNHead")
    torch.manual_seed(0)
    rpn = day2.TinyRPN(head_cls, channels=64, scales=(16, 32, 48), ratios=(0.5, 1.0, 2.0))
    X_tr, B_tr, _ = day2.shapes_tensors(1000, 128, seed=0)
    X_va, B_va, _ = day2.shapes_tensors(100, 128, seed=1)
    anc_np = rpn.anchors.numpy()
    lab_tr, dt_tr = day2.rpn_targets(anc_np, B_tr)
    print(f"{len(anc_np)} anchors per image; positives per image: {(lab_tr == 1).sum(1).float().mean():.1f} on average")

    untrained = day2.TinyRPN(head_cls); untrained.load_state_dict(rpn.state_dict())   # keep a copy for the "before" picture
    t0 = time.time()
    rpn_hist = day2.train_tiny_rpn(rpn, X_tr, lab_tr, dt_tr, epochs=6, batch_size=32, lr=3e-3, max_seconds=150)
    print(f"trained in {time.time() - t0:.0f} s")
    """),
    code(r"""
    # Objectness map and proposals before / after training, on validation images
    fig, axes = plt.subplots(3, 4, figsize=(15, 11.5))
    for c, i in enumerate([0, 3, 7, 11]):
        img = X_va[i]
        with torch.no_grad():
            obj, _ = rpn(img[None])
        objmap = torch.sigmoid(obj[0]).view(16, 16, 9).max(-1).values.numpy()
        p_before, s_before = day2.rpn_proposals(untrained, img, post_nms_top_n=10)
        p_after, s_after = day2.rpn_proposals(rpn, img, post_nms_top_n=10)
        im = img.permute(1, 2, 0).numpy()
        axes[0, c].imshow(im); viz.draw_boxes(axes[0, c], p_before.numpy(), edgecolor=RED, lw=1); axes[0, c].set_title("before training: top-10 proposals", fontsize=9)
        axes[1, c].imshow(im); axes[1, c].imshow(objmap, cmap="viridis", alpha=0.6, extent=(0, 128, 128, 0), vmin=0, vmax=1)
        axes[1, c].set_title("objectness (max over 9 anchors)", fontsize=9)
        axes[2, c].imshow(im); viz.draw_boxes(axes[2, c], B_va[i], edgecolor=GREEN, lw=3, alpha=0.6)
        viz.draw_boxes(axes[2, c], p_after.numpy()[:5], scores=s_after.numpy()[:5], edgecolor=BLUE, lw=1.5)
        axes[2, c].set_title("after: top-5 proposals (GT green)", fontsize=9)
    for ax in axes.flat: ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### From 2304 scored anchors to a handful of proposals

    The RPN's raw output is one decoded box per anchor. torchvision's `RegionProposalNetwork`
    turns it into proposals with a fixed recipe (default numbers in brackets for a full-size model):

    1. decode all deltas against their anchors; 2. clip to the image and drop tiny boxes;
    3. keep the top-k by objectness *per level* (2000 train / 1000 test); 4. NMS at IoU 0.7;
    5. keep the top-N (2000 train / 1000 test). The second stage only ever sees these.
    """),
    code(r"""
    i = 3
    _, _, steps = day2.rpn_proposals(rpn, X_va[i], pre_nms_top_n=100, post_nms_top_n=12, nms_iou=0.5, return_steps=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    titles = {"all": "all decoded anchors (top 300 shown)", "clipped": "clipped, tiny ones removed",
              "top-k": "top-100 by objectness", "nms": "after NMS: top-12 proposals"}
    for ax, (k, (bx, sc)) in zip(axes, steps.items()):
        o = sc.argsort(descending=True)[:300]
        ax.imshow(X_va[i].permute(1, 2, 0).numpy())
        cols = plt.cm.viridis(sc[o].numpy())
        for b, c in zip(bx[o].numpy()[::-1], cols[::-1]):
            ax.add_patch(Rectangle(b[:2], *(b[2:] - b[:2]), fill=False, ec=c, lw=0.8))
        ax.set_title(f"{titles[k]}\n{len(bx)} boxes", fontsize=9); ax.set_xlim(-10, 138); ax.set_ylim(138, -10); ax.axis("off")
    plt.tight_layout(); plt.show()
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · RoI pooling vs RoIAlign

    The second stage needs a **fixed-size** feature tile (7 x 7) for every proposal, whatever its
    size. Fast R-CNN's **RoI pooling** snaps the proposal to the feature grid (round the box
    coordinates), splits it into 7 x 7 *integer* bins (rounding again) and max-pools each bin.
    With stride 16, one rounding step can move the box by up to 8 image pixels - harmless for
    classification, harmful for precise boxes and fatal for masks.

    **RoIAlign** (Mask R-CNN, 2017) never rounds: the box stays in continuous coordinates, each
    bin gets a few regularly spaced **sampling points** (2 x 2), the feature value at each point is
    obtained by **bilinear interpolation** of its 4 neighbouring cells, and the samples are averaged.
    """),
    code(r"""
    fmap = torch.tensor(np.add.outer(np.arange(8), np.arange(8)) % 5 + np.random.default_rng(0).uniform(0, 1, (8, 8)), dtype=torch.float32)[None]
    roi = np.array([1.3, 1.7, 6.1, 5.6])              # in feature-map coordinates (stride 1 for the drawing)
    pooled, bins = day2.roi_pool_quantized(fmap, roi, 3)
    aligned = torchvision.ops.roi_align(fmap[None], [torch.tensor(roi[None], dtype=torch.float32)], 3, 1.0, 2, aligned=True)[0, 0]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), gridspec_kw=dict(width_ratios=[1, 0.45, 1, 0.45]))
    for ax in (axes[0], axes[2]):
        ax.imshow(fmap[0], cmap="Blues", extent=(0, 8, 8, 0), vmin=0, vmax=6)
        ax.set_xticks(range(9)); ax.set_yticks(range(9)); ax.grid(color="k", lw=0.5)
        ax.add_patch(Rectangle(roi[:2], *(roi[2:] - roi[:2]), fill=False, ec=RED, lw=2.5, ls="--"))
    for (xa, ya, xb, yb) in bins:
        axes[0].add_patch(Rectangle((xa, ya), xb - xa, yb - ya, fill=False, ec=ORANGE, lw=2.2))
    axes[0].set_title("RoI pooling: box rounded to the grid,\ninteger bins (orange) - some cells used twice", fontsize=10)
    bw, bh = (roi[2] - roi[0]) / 3, (roi[3] - roi[1]) / 3
    for i in range(3):
        for j in range(3):
            axes[2].add_patch(Rectangle((roi[0] + j * bw, roi[1] + i * bh), bw, bh, fill=False, ec=GREEN, lw=1.8))
            for sy in range(2):
                for sx in range(2):
                    axes[2].plot(roi[0] + (j + (sx + 0.5) / 2) * bw, roi[1] + (i + (sy + 0.5) / 2) * bh, "o", color=RED, ms=5)
    axes[2].set_title("RoIAlign: exact bins (green), 2 x 2 bilinear\nsamples per bin (red dots), averaged", fontsize=10)
    for ax, m, t in [(axes[1], pooled[0], "RoI pool 3x3"), (axes[3], aligned, "RoIAlign 3x3")]:
        ax.imshow(m, cmap="Blues", vmin=0, vmax=6); viz.annotate_grid(ax, m.numpy(), fmt="{:.2f}"); ax.set_title(t); ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    code(r"""
    # Quantisation in action: move the RoI by fractions of a feature cell and watch one output bin
    shifts = np.linspace(0, 2, 81)
    pool_v, align_v = [], []
    smooth = torch.tensor(np.add.outer(np.sin(np.arange(16) / 2.0), np.cos(np.arange(16) / 3.0)), dtype=torch.float32)[None]
    for sft in shifts:
        r = np.array([3.0 + sft, 4.0, 9.0 + sft, 10.0])
        pool_v.append(float(day2.roi_pool_quantized(smooth, r, 3)[0][0, 1, 1]))
        align_v.append(float(torchvision.ops.roi_align(smooth[None], [torch.tensor(r[None], dtype=torch.float32)], 3, 1.0, 2, aligned=True)[0, 0, 1, 1]))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(shifts, pool_v, color=ORANGE, lw=2, drawstyle="steps-mid", label="RoI pooling (rounds the box)")
    ax.plot(shifts, align_v, color=GREEN, lw=2, label="RoIAlign (bilinear)")
    ax.set_xlabel("sub-cell shift of the RoI (feature cells; x16 = image pixels)"); ax.set_ylabel("centre bin value")
    ax.legend(); ax.grid(alpha=.3); ax.set_title("RoI pooling is piecewise constant in the box position: no gradient, jumps")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Exercise 5 · RoIAlign in PyTorch

    Two functions. `bilinear(feature, y, x)` interpolates a `(C, H, W)` map at float coordinates
    (the clamping is given). `roi_align(feature, rois, output_size, spatial_scale, sampling_ratio)`
    loops over RoIs: scale the box to feature coordinates, shift by -0.5 (`aligned=True`: pixel *centres*),
    build the `P*s` sample coordinates along y and along x, sample the full grid, and average the
    `s x s` samples of every bin. The checker compares with `torchvision.ops.roi_align`.
    """),
    exercise(
        stub=r"""
        def bilinear(feature, y, x):
            C, H, W = feature.shape
            y = y.clamp(0, H - 1); x = x.clamp(0, W - 1)
            y0 = y.floor().long(); x0 = x.floor().long()
            y1 = (y0 + 1).clamp(max=H - 1); x1 = (x0 + 1).clamp(max=W - 1)
            ly, lx = y - y0, x - x0                     # fractional parts in [0, 1)
            # YOUR CODE HERE: weighted sum of feature[:, y0, x0], [:, y0, x1], [:, y1, x0], [:, y1, x1]
            raise NotImplementedError

        def roi_align(feature, rois, output_size=7, spatial_scale=1.0, sampling_ratio=2):
            C = feature.shape[0]; P, s = output_size, sampling_ratio
            out = feature.new_zeros((len(rois), C, P, P))
            for k, box in enumerate(rois):
                x1, y1, x2, y2 = (box * spatial_scale - 0.5).tolist()
                bw, bh = (x2 - x1) / P, (y2 - y1) / P
                # YOUR CODE HERE:
                # offsets of the s samples inside a bin: (arange(s) + 0.5) / s
                # ys = y1 + (arange(P)[:, None] + offsets[None]) * bh  -> flatten to (P*s,); same for xs
                # grid: yy (P*s, P*s), xx (P*s, P*s); vals = bilinear(feature, yy, xx) -> (C, P*s, P*s)
                # out[k] = vals.reshape(C, P, s, P, s).mean((2, 4))
                raise NotImplementedError
            return out

        tests.check_roi_align(roi_align)
        """,
        solution=r"""
        def bilinear(feature, y, x):
            C, H, W = feature.shape
            y = y.clamp(0, H - 1); x = x.clamp(0, W - 1)
            y0 = y.floor().long(); x0 = x.floor().long()
            y1 = (y0 + 1).clamp(max=H - 1); x1 = (x0 + 1).clamp(max=W - 1)
            ly, lx = y - y0, x - x0                     # fractional parts in [0, 1)
            return (feature[:, y0, x0] * (1 - ly) * (1 - lx) + feature[:, y0, x1] * (1 - ly) * lx +
                    feature[:, y1, x0] * ly * (1 - lx) + feature[:, y1, x1] * ly * lx)

        def roi_align(feature, rois, output_size=7, spatial_scale=1.0, sampling_ratio=2):
            C = feature.shape[0]; P, s = output_size, sampling_ratio
            out = feature.new_zeros((len(rois), C, P, P))
            for k, box in enumerate(rois):
                x1, y1, x2, y2 = (box * spatial_scale - 0.5).tolist()
                bw, bh = (x2 - x1) / P, (y2 - y1) / P
                offs = (torch.arange(s, dtype=feature.dtype) + 0.5) / s
                ys = (y1 + (torch.arange(P, dtype=feature.dtype)[:, None] + offs[None]) * bh).reshape(-1)
                xs = (x1 + (torch.arange(P, dtype=feature.dtype)[:, None] + offs[None]) * bw).reshape(-1)
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                vals = bilinear(feature, yy, xx)
                out[k] = vals.reshape(C, P, s, P, s).mean((2, 4))
            return out

        tests.check_roi_align(roi_align)
        """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · The second stage and the four-term loss

    Each RoI tile (256 x 7 x 7) goes through a small head (two FC layers of 1024 in the original,
    four convs + one FC in torchvision's `_v2`) and two sibling outputs: **class logits** over
    $K + 1$ classes (background included) and **class-specific box deltas** $4(K+1)$ - a refinement
    *of the proposal*, encoded exactly like the RPN's deltas relative to anchors. RoIs are matched to
    GT with IoU $\ge 0.5$ (foreground) / $< 0.5$ (background) and sampled 512 per image with 25% foreground.

    | term | stage | on which samples | loss |
    |---|---|---|---|
    | `loss_objectness` | RPN | 256 sampled anchors / image | binary cross-entropy |
    | `loss_rpn_box_reg` | RPN | positive anchors | smooth-L1 on $(t_x, t_y, t_w, t_h)$ |
    | `loss_classifier` | RoI head | 512 sampled RoIs / image | cross-entropy over $K+1$ classes |
    | `loss_box_reg` | RoI head | foreground RoIs, GT class only | smooth-L1 on deltas vs the proposal |

    These four names are exactly the keys of the dict torchvision returns in training mode - you
    will see them in the training log below.
    """),

    # ------------------------------------------------------------------ 7
    md(r"""
    ## 7 · Fine-tune torchvision's Faster R-CNN on Penn-Fudan

    Penn-Fudan has 170 street images with 423 pedestrians (instance masks too - Day 3). The
    recipe that works for almost any small detection dataset:

    1. load a COCO-pretrained detector (here `fasterrcnn_resnet50_fpn_v2`, 43 M parameters);
    2. **replace the box predictor** so it outputs your number of classes (+1 for background);
    3. fine-tune everything with SGD, a small LR and a warm-up.

    **CPU budget.** We train on **40 images** at **320 px** (`min_size=320`), with fewer proposals
    (`rpn_post_nms_top_n_train=300`, `box_batch_size_per_image=64`), for 2 epochs (40 iterations)
    capped at 3 minutes, and save `data/checkpoints/d2_02_frcnn_pennfudan.pt` so re-runs just load it.
    The provided checkpoint was made by exactly this cell without the time cap (about 10 CPU minutes);
    delete it to train yourself. With a GPU and the standard recipe (120 / 50 split, 800 px, 2-10
    epochs) you get mAP@0.5 of ~0.97-0.99 and mAP@[.5:.95] of ~0.75-0.85.
    """),
    code(r"""
    pf_train, pf_val = day2.penn_fudan_splits(n_train=40, n_val=20, seed=0, max_side=512)
    print(f"Penn-Fudan subset: {len(pf_train)} train / {len(pf_val)} val images, "
          f"{sum(len(t['boxes']) for _, t in pf_train)} / {sum(len(t['boxes']) for _, t in pf_val)} people")
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, i in zip(axes, range(5)):
        im, t = pf_train[i]
        ax.imshow(im.permute(1, 2, 0)); viz.draw_boxes(ax, t["boxes"].numpy(), edgecolor=GREEN, lw=2); ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Before training anything: the COCO model already detects people (COCO class 1 = person, and
    torchvision also uses label 1 for "person" in our dataset). How good is it out of the box?
    """),
    code(r"""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    FAST = dict(min_size=320, max_size=512, rpn_pre_nms_top_n_test=500, rpn_post_nms_top_n_test=100,
                rpn_pre_nms_top_n_train=1000, rpn_post_nms_top_n_train=300, box_batch_size_per_image=64)
    coco_model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", **FAST).eval()
    t0 = time.time()
    coco_res = day2.evaluate_map(coco_model, pf_val)             # one inference pass; label 1 = person in both
    print(f"COCO-pretrained, no fine-tuning: mAP@0.5 = {coco_res['map50']:.3f}, mAP@[.5:.95] = {coco_res['map']:.3f}  ({time.time() - t0:.0f} s)")
    del coco_model
    """),
    md(r"""
    ### Exercise 6 · replace the box predictor

    `model.roi_heads.box_predictor` is a `FastRCNNPredictor(in_features, num_classes)` with
    `cls_score` and `bbox_pred` linear layers. Replace it with a fresh one for **2 classes**
    (background + person). Everything else keeps its COCO weights.
    """),
    exercise(
        stub=r"""
        torch.manual_seed(0)
        model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", **FAST)
        # YOUR CODE HERE (2 lines): read in_features from the old predictor, install a new one
        raise NotImplementedError

        tests.check_box_predictor(model, num_classes=2)
        """,
        solution=r"""
        torch.manual_seed(0)
        model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", **FAST)
        in_features = model.roi_heads.box_predictor.cls_score.in_features      # 1024
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

        tests.check_box_predictor(model, num_classes=2)
        """),
    code(r"""
    print(f"parameters: {torchkit.count_parameters(model) / 1e6:.1f} M")
    history = day2.load_or_train(
        model, "d2_02_frcnn_pennfudan.pt",
        lambda: day2.train_detector(model, pf_train, epochs=2, batch_size=2, lr=0.01, max_minutes=3, log_every=5))
    model.eval()
    if history:
        keys = ["loss_objectness", "loss_rpn_box_reg", "loss_classifier", "loss_box_reg"]
        fig, ax = plt.subplots(figsize=(10, 3.6))
        for k, c in zip(keys, [ORANGE, "#fdae6b", "#B279A2", GREEN]):
            v = np.array([h[k] for h in history]); ax.plot(v, color=c, alpha=0.35)
            ax.plot(np.convolve(v, np.ones(5) / 5, "valid"), color=c, lw=2, label=k)
        ax.set_xlabel("iteration (batch of 2 images)"); ax.set_ylabel("loss"); ax.legend(); ax.grid(alpha=.3)
        ax.set_title("the four Faster R-CNN loss terms during fine-tuning (thin: raw, thick: moving average)")
        plt.tight_layout(); plt.show()
    """),
    code(r"""
    t0 = time.time()
    ft_res = day2.evaluate_map(model, pf_val)
    ft_preds, pf_gts = ft_res["preds"], ft_res["gts"]
    print(f"{'':28s}{'mAP@0.5':>9}{'mAP@[.5:.95]':>14}")
    print(f"{'COCO weights, no training':28s}{coco_res['map50']:9.3f}{coco_res['map']:14.3f}")
    print(f"{'fine-tuned (40 imgs, CPU)':28s}{ft_res['map50']:9.3f}{ft_res['map']:14.3f}   (eval {time.time() - t0:.0f} s)")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, i in zip(axes.flat, range(8)):
        im = pf_val[i][0].permute(1, 2, 0).numpy(); ax.imshow(im); ax.axis("off")
        viz.draw_boxes(ax, pf_gts[i]["boxes"], edgecolor=GREEN, lw=3, alpha=0.5)
        p = ft_preds[i]; k = p["scores"] > 0.5
        viz.draw_boxes(ax, p["boxes"][k], scores=p["scores"][k], edgecolor=RED, lw=1.5)
    fig.suptitle("fine-tuned Faster R-CNN on validation images (green: GT, red: predictions with score > 0.5)", y=1.0)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    **Read the table carefully.** With the provided checkpoint you should see roughly mAP@0.5 = 0.99
    and mAP@[.5:.95] = 0.78 after fine-tuning - but the *untouched* COCO model already scores about
    0.99 / 0.85. Persons are COCO's most frequent class, learned from 64 000 images at 800 px; our
    40 images at 320 px with a freshly initialised predictor match its recall and lose some box
    precision. Lessons: (1) **always measure the zero-shot / pretrained baseline first**; (2) fine-tuning
    pays off when your classes or your box definition differ from the pretraining data (see the Pet
    heads in D2-04); (3) on 20 validation images a difference of a few points is noise.

    Then look at the failures: partly occluded people at the image border, groups where two people
    merge into one box (NMS!), and small background pedestrians that were not annotated - a
    "false positive" that is actually a labelling gap. Inspecting errors like this is worth more
    than another decimal of mAP.
    """),

    # ------------------------------------------------------------------ 8
    md(r"""
    ## 8 · Summary

    * **Anchors** = a fixed family of scales x ratios at every feature-map cell; a 512 px image
      easily gets 50 000 of them. Networks predict *corrections* to anchors.
    * **Matching**: IoU >= 0.7 positive, < 0.3 negative, in between ignored, plus the best anchor of
      every GT. ~1% of anchors are positive, hence minibatch sampling (two-stage) or focal loss (one-stage).
    * **Encoding**: $t_x, t_y$ = centre offset / anchor size, $t_w, t_h$ = log size ratio; smooth-L1.
    * **RPN**: 3x3 conv + two 1x1 convs (A objectness, 4A deltas); proposals = decode, clip, top-k, NMS, top-N.
    * **RoIAlign** samples bilinearly at exact positions; RoI pooling rounds twice and loses up to half a
      stride of precision.
    * **Faster R-CNN loss** = RPN objectness + RPN box + RoI classification + RoI box.
    * Fine-tuning a library detector = replace the predictor for $K+1$ classes and train with a small LR.

    ### Self-check
    1. How many anchors does a 640 x 480 image get from a single stride-16 level with 3 scales and 3 ratios?
    2. An anchor has IoU 0.55 with its best GT. What is its label under the 0.7 / 0.3 rule? When could it still be positive?
    3. Why encode the width as $\log(g_w/a_w)$ rather than $g_w - a_w$?
    4. The RPN predicts objectness for 2 classes or for 81? Why?
    5. A proposal spans x = 33.7 ... 97.2 px on a stride-16 map. Which x-range does RoI pooling actually use? RoIAlign?
    6. Which of the four loss terms would you watch if proposals look good but final boxes are badly placed?

    ### Go deeper
    * Ren, He, Girshick, Sun, *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks* (NeurIPS 2015).
    * He, Gkioxari, Dollar, Girshick, *Mask R-CNN* (ICCV 2017) - section 3 introduces RoIAlign.
    * torchvision "TorchVision Object Detection Finetuning Tutorial" (Penn-Fudan, the recipe used here) and the source of `torchvision/models/detection/rpn.py`.
    """),
]
