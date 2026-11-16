from nb import md, code, exercise, header, api

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
from cvcourse import viz, boxes, shapes, tests, torchkit, day2, labs

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
    api("np.set_printoptions", "torch.manual_seed", "torchkit.get_device", brief=True),
    code(SETUP),
    md(r"""
    ## The big picture

    Faster R-CNN (Ren et al. 2015) is the model this notebook takes apart. One backbone computes
    a feature map *once*; a small **Region Proposal Network** slides over it and scores a fixed
    set of reference boxes (**anchors**) for "objectness" (the probability that the box holds *some*
    object, whatever its class); the best few hundred become
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
    api("np.sqrt", title="Functions you will need"),
    api("np.array", "np.arange", "np.stack", "Tensor.reshape", brief=True),
    exercise(
        stub=r"""
        def make_anchors(feature_size, stride, scales, ratios):
            '''feature_size = (fh, fw) cells, stride in pixels, scales and ratios: lists ->
            all anchors of the level, (fh * fw * A, 4) xyxy, A = len(scales) * len(ratios),
            ordered cell by cell (row, then column), then anchor by anchor.'''
            fh, fw = feature_size
            # 1. the A base boxes, centred at (0, 0): one per (scale, ratio) pair
            base = []
            for scale in scales:
                for ratio in ratios:
                    # YOUR CODE: w and h of this anchor: w = scale / sqrt(ratio), h = scale * sqrt(ratio)
                    # YOUR CODE: append the box [-w / 2, -h / 2, w / 2, h / 2] to base
                    raise NotImplementedError
            base = np.array(base)                            # (A, 4)
            # 2. the centre of every cell, row by row, in pixels
            centres = []
            for i in range(fh):                              # row
                for j in range(fw):                          # column
                    # YOUR CODE: cx = (j + 0.5) * stride, cy = (i + 0.5) * stride; append [cx, cy, cx, cy]
                    raise NotImplementedError
            centres = np.array(centres)                      # (fh * fw, 4)
            # 3. YOUR CODE: every centre plus every base box, with broadcasting:
            #               centres[:, None, :] + base[None, :, :] -> (fh * fw, A, 4)
            # 4. YOUR CODE: reshape to (fh * fw * A, 4) and return it
            raise NotImplementedError

        tests.check_make_anchors(make_anchors)
        """,
        solution=r"""
        def make_anchors(feature_size, stride, scales, ratios):
            '''feature_size = (fh, fw) cells, stride in pixels, scales and ratios: lists ->
            all anchors of the level, (fh * fw * A, 4) xyxy, A = len(scales) * len(ratios),
            ordered cell by cell (row, then column), then anchor by anchor.'''
            fh, fw = feature_size
            # 1. the A base boxes, centred at (0, 0): one per (scale, ratio) pair
            base = []
            for scale in scales:
                for ratio in ratios:
                    # same area for every ratio: w * h = scale ** 2, and h / w = ratio
                    w = scale / np.sqrt(ratio)
                    h = scale * np.sqrt(ratio)
                    base.append([-w / 2, -h / 2, w / 2, h / 2])
            base = np.array(base)                            # (A, 4)
            # 2. the centre of every cell, row by row, in pixels
            centres = []
            for i in range(fh):                              # row
                for j in range(fw):                          # column
                    cx = (j + 0.5) * stride
                    cy = (i + 0.5) * stride
                    centres.append([cx, cy, cx, cy])
            centres = np.array(centres)                      # (fh * fw, 4)
            # 3. every centre plus every base box, with broadcasting
            anchors = centres[:, None, :] + base[None, :, :] # (fh * fw, 1, 4) + (1, A, 4) -> (fh * fw, A, 4)
            # 4. one row per anchor
            return anchors.reshape(-1, 4)                    # (fh * fw * A, 4)

        tests.check_make_anchors(make_anchors)
        """),
    api("torchkit.get_penn_fudan", "F.interpolate", "Tensor.permute", "Tensor.numpy", "np.array", brief=True),
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
    md(r"""
    Left: the family of one cell - colour is the scale, line style the ratio; all nine share the
    centre (star). Right: that family is repeated at every cell of the grid. The table is the
    point to remember: anchors grow with $1/s^2$, so the fine stride-8 level alone holds three
    quarters of the ~48 000 candidates.
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

    0.7 / 0.3 are the paper's thresholds, chosen for ~800 px images. They are arguments of the
    function, and further down you will see why our 128 px images need 0.5 / 0.3 instead.
    """),
    md(r"""
    ### Lab · Anchor matching

    **Predict first:** two objects in a 256 px image, stride 16, 3 scales x 3 ratios = 2304 anchors,
    the paper's 0.7 / 0.3 rule. How many anchors are positive: about 6, 60 or 600?
    Check in the lab, then work through its "Try this" list (8-10 minutes).
    """),
    code(r"""
    labs.show("anchor-matching")
    """),
    md(r"""
    ### Exercise 2 · `match_anchors`

    Return `(matched_gt_index, labels)` for `anchors (N, 4)` and `gt (G, 4)`: two integer arrays of
    length `N`, labels in {1, 0, -1}; `matched_gt_index` only matters where `labels == 1`. Use
    `boxes.box_iou`. The green anchors in the lab are the `labels == 1` entries; the dashed "forced"
    ones come from the `allow_low_quality` branch.
    """),
    api("np.full", "Tensor.argmax", "Tensor.max", "np.random.default_rng", "Tensor.sum", "Tensor.mean", brief=True),
    exercise(
        stub=r"""
        def match_anchors(anchors, gt, pos_iou=0.7, neg_iou=0.3, allow_low_quality=True):
            '''anchors (N, 4), gt (G, 4), xyxy -> (best_gt, labels): two integer arrays of length N.
            labels: 1 positive, 0 negative (background), -1 ignore; best_gt[n] = the GT that anchor n regresses to.'''
            # 1. the IoU of every anchor with every GT
            ious = boxes.box_iou(anchors, gt)                 # (N, G)
            # 2. for every anchor: its best GT, and that IoU
            best_gt = ious.argmax(1)                          # (N,) index of the best GT (along axis 1 = over the GTs)
            best_iou = ious.max(1)                            # (N,) its IoU
            # 3. start with every anchor on "ignore"
            labels = np.full(len(anchors), -1)                # (N,)
            # 4. YOUR CODE: negatives: label 0 where best_iou < neg_iou (a boolean mask as the index)
            # 5. YOUR CODE: positives: label 1 where best_iou >= pos_iou
            # 6. low-quality rule: every GT also makes its best anchor positive
            if allow_low_quality:
                # YOUR CODE: best_anchor, the best anchor of every GT: argmax over the anchors (axis 0) -> (G,)
                # YOUR CODE: label those anchors 1
                # YOUR CODE: and point them at their GT: best_gt[best_anchor] = np.arange(len(gt))
                raise NotImplementedError
            return best_gt, labels

        tests.check_match_anchors(match_anchors)
        """,
        solution=r"""
        def match_anchors(anchors, gt, pos_iou=0.7, neg_iou=0.3, allow_low_quality=True):
            '''anchors (N, 4), gt (G, 4), xyxy -> (best_gt, labels): two integer arrays of length N.
            labels: 1 positive, 0 negative (background), -1 ignore; best_gt[n] = the GT that anchor n regresses to.'''
            # 1. the IoU of every anchor with every GT
            ious = boxes.box_iou(anchors, gt)                 # (N, G)
            # 2. for every anchor: its best GT, and that IoU
            best_gt = ious.argmax(1)                          # (N,) index of the best GT (along axis 1 = over the GTs)
            best_iou = ious.max(1)                            # (N,) its IoU
            # 3. start with every anchor on "ignore"
            labels = np.full(len(anchors), -1)                # (N,)
            # 4. negatives
            labels[best_iou < neg_iou] = 0
            # 5. positives
            labels[best_iou >= pos_iou] = 1
            # 6. low-quality rule: every GT also makes its best anchor positive
            if allow_low_quality:
                best_anchor = ious.argmax(0)                  # (G,) the best anchor of every GT
                labels[best_anchor] = 1
                best_gt[best_anchor] = np.arange(len(gt))     # anchor best_anchor[g] regresses to GT g
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
    With the 0.7 rule the four objects get only 5 positive anchors: two objects reach 0.7 with one
    or two anchors, the other two only through the forced "best anchor per GT" rule (the dashed
    anchors of the lab). On a 128 px image a stride-8 anchor can be 4 px off-centre, and 4 px on a
    24 px object already pushes IoU below 0.7.
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
    ### Lab · Box encoding

    **Predict first:** take the same anchor and ground truth and double both (the same scene at twice
    the resolution). Which of $t_x, t_y, t_w, t_h$ change? Check with the "x2 scale both boxes"
    button, then work through the "Try this" list (6-8 minutes).
    """),
    code(r"""
    labs.show("box-encoding")
    """),
    md(r"""
    ### Exercise 3 · `encode` / `decode`

    Both take aligned `(N, 4)` xyxy arrays and return `(N, 4)` arrays (`encode`: deltas
    $(t_x, t_y, t_w, t_h)$, `decode`: xyxy boxes). Convert with `boxes.xyxy_to_cxcywh` /
    `boxes.cxcywh_to_xyxy`. The checker tests what the lab's "snap deltas to the target" button
    showed: `decode(encode(g, a), a)` gives back `g`.
    """),
    api("np.log", "np.exp", brief=True),
    exercise(
        stub=r"""
        def encode(gt, anchors):
            '''gt, anchors: (N, 4) xyxy, aligned (gt[n] belongs to anchors[n]) -> deltas (N, 4) = (tx, ty, tw, th).'''
            # 1. both in cxcywh: columns 0, 1, 2, 3 are cx, cy, w, h
            g = boxes.xyxy_to_cxcywh(gt)                     # (N, 4)
            a = boxes.xyxy_to_cxcywh(anchors)                # (N, 4)
            # 2. YOUR CODE: tx, ty, the centre shift in units of the anchor's size:
            #               (g_cx - a_cx) / a_w and (g_cy - a_cy) / a_h -> (N,) each
            # 3. YOUR CODE: tw, th, the log of the size ratio: log(g_w / a_w) and log(g_h / a_h) -> (N,) each
            # 4. YOUR CODE: put the columns tx, ty, tw, th together with np.stack([...], -1) -> (N, 4); return it
            raise NotImplementedError

        def decode(deltas, anchors):
            '''deltas (N, 4) = (tx, ty, tw, th), anchors (N, 4) xyxy -> boxes (N, 4) xyxy; the inverse of encode.'''
            # 1. the anchors in cxcywh
            a = boxes.xyxy_to_cxcywh(anchors)                # (N, 4)
            # 2. YOUR CODE: cx, cy, the centre: the anchor's centre plus tx (ty) times the anchor's width (height)
            # 3. YOUR CODE: w, h, the size: the anchor's width (height) times exp(tw) (exp(th))
            # 4. YOUR CODE: stack cx, cy, w, h -> (N, 4), convert with boxes.cxcywh_to_xyxy and return it
            raise NotImplementedError

        tests.check_encode_decode(encode, decode)
        """,
        solution=r"""
        def encode(gt, anchors):
            '''gt, anchors: (N, 4) xyxy, aligned (gt[n] belongs to anchors[n]) -> deltas (N, 4) = (tx, ty, tw, th).'''
            # 1. both in cxcywh: columns 0, 1, 2, 3 are cx, cy, w, h
            g = boxes.xyxy_to_cxcywh(gt)                     # (N, 4)
            a = boxes.xyxy_to_cxcywh(anchors)                # (N, 4)
            # 2. tx, ty, the centre shift in units of the anchor's size
            tx = (g[:, 0] - a[:, 0]) / a[:, 2]               # (N,)
            ty = (g[:, 1] - a[:, 1]) / a[:, 3]               # (N,)
            # 3. tw, th, the log of the size ratio
            tw = np.log(g[:, 2] / a[:, 2])                   # (N,)
            th = np.log(g[:, 3] / a[:, 3])                   # (N,)
            # 4. put the columns together
            return np.stack([tx, ty, tw, th], -1)            # (N, 4)

        def decode(deltas, anchors):
            '''deltas (N, 4) = (tx, ty, tw, th), anchors (N, 4) xyxy -> boxes (N, 4) xyxy; the inverse of encode.'''
            # 1. the anchors in cxcywh
            a = boxes.xyxy_to_cxcywh(anchors)                # (N, 4)
            # 2. cx, cy, the centre: the anchor's centre plus tx (ty) times the anchor's width (height)
            cx = a[:, 0] + deltas[:, 0] * a[:, 2]            # (N,)
            cy = a[:, 1] + deltas[:, 1] * a[:, 3]            # (N,)
            # 3. w, h, the size: the anchor's width (height) times exp(tw) (exp(th))
            w = a[:, 2] * np.exp(deltas[:, 2])               # (N,)
            h = a[:, 3] * np.exp(deltas[:, 3])               # (N,)
            # 4. stack and convert back to xyxy
            boxes_cxcywh = np.stack([cx, cy, w, h], -1)      # (N, 4)
            return boxes.cxcywh_to_xyxy(boxes_cxcywh)        # (N, 4)

        tests.check_encode_decode(encode, decode)
        """),
    api("np.allclose"),
    api("np.where", "np.linspace", "np.abs", brief=True),
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
    Left: the arrow is the centre shift that $t_x, t_y$ encode, in units of the anchor's size (deltas in
    the title). Middle: on a log axis, growing x2 and shrinking /2 give targets of equal size
    with opposite signs; the naive ratio (grey) does not.

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
    api("nn.Conv2d", "F.relu", brief=True),
    exercise(
        stub=r"""
        class RPNHead(nn.Module):
            '''RPNHead(C, A): input feature map x (N, C, H, W) ->
            (objectness logits (N, A, H, W), box deltas (N, 4A, H, W)): A scores and 4A deltas at every cell.'''
            def __init__(self, in_channels, num_anchors):
                super().__init__()
                # 1. YOUR CODE: self.conv, a 3x3 convolution from in_channels to in_channels, padding=1 (same size)
                # 2. YOUR CODE: self.cls_logits, a 1x1 convolution from in_channels to num_anchors (one score per anchor)
                # 3. YOUR CODE: self.bbox_pred, a 1x1 convolution from in_channels to 4 * num_anchors (4 deltas per anchor)
                raise NotImplementedError

            def forward(self, x):
                # 4. YOUR CODE: h, the shared features: self.conv, then F.relu -> (N, C, H, W)
                # 5. YOUR CODE: objectness = self.cls_logits(h) -> (N, A, H, W)
                # 6. YOUR CODE: deltas = self.bbox_pred(h) -> (N, 4A, H, W)
                # 7. YOUR CODE: return objectness, deltas
                raise NotImplementedError

        tests.check_rpn_head(RPNHead)
        """,
        solution=r"""
        class RPNHead(nn.Module):
            '''RPNHead(C, A): input feature map x (N, C, H, W) ->
            (objectness logits (N, A, H, W), box deltas (N, 4A, H, W)): A scores and 4A deltas at every cell.'''
            def __init__(self, in_channels, num_anchors):
                super().__init__()
                # 1. self.conv, a 3x3 convolution from in_channels to in_channels, padding=1 (same size)
                self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
                # 2. self.cls_logits, a 1x1 convolution from in_channels to num_anchors (one score per anchor)
                self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
                # 3. self.bbox_pred, a 1x1 convolution from in_channels to 4 * num_anchors (4 deltas per anchor)
                self.bbox_pred = nn.Conv2d(in_channels, 4 * num_anchors, 1)

            def forward(self, x):
                # 4. h, the shared features
                h = self.conv(x)
                h = F.relu(h)                                # (N, C, H, W)
                # 5. one objectness logit per anchor and cell
                objectness = self.cls_logits(h)              # (N, A, H, W)
                # 6. four deltas per anchor and cell
                deltas = self.bbox_pred(h)                   # (N, 4A, H, W)
                # 7. both outputs
                return objectness, deltas

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
    api("torch.sigmoid"),
    api("Module.state_dict", "Module.load_state_dict", "Tensor.float", "torch.no_grad", "Tensor.view", brief=True),
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
    Top row: before training the highest-scoring anchors are arbitrary boxes. Middle: after training
    the objectness map is bright only on the shapes, background cells stay dark. Bottom: the
    best-scoring decoded proposals now hug the green ground truth, although no anchor matched it
    exactly - the deltas moved them there. A few lower-scoring duplicates are still off; NMS and the
    second stage deal with those.
    """),
    md(r"""
    ### From 2304 scored anchors to a handful of proposals

    The RPN's raw output is one decoded box per anchor. torchvision's `RegionProposalNetwork`
    turns it into proposals with a fixed recipe (default numbers in brackets for a full-size model):

    1. decode all deltas against their anchors; 2. clip to the image and drop tiny boxes;
    3. keep the top-k by objectness *per level* (2000 train / 1000 test); 4. NMS at IoU 0.7;
    5. keep the top-N (2000 train / 1000 test). The second stage only ever sees these.
    """),
    api("Tensor.argsort", brief=True),
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
    api("torchvision.ops.roi_align", "np.add.outer", "np.sin", "np.cos"),
    api("torch.tensor", brief=True),
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
    ### Lab · RoI pooling vs RoIAlign

    **Predict first:** a RoIAlign sample point lies a quarter of a cell to the right of and a quarter
    of a cell below the centre of a feature cell. What are its four bilinear weights? Hover a sample
    in the lab to check, then work through the "Try this" list (8 minutes).
    """),
    code(r"""
    labs.show("roi-align")
    """),
    md(r"""
    ### Exercise 5 · RoIAlign in PyTorch

    Two functions. `bilinear(feature, y, x)` interpolates a `(C, H, W)` map at float coordinates
    `y`, `x` (tensors of the same shape) and returns `(C, *y.shape)` (the clamping is given).
    `roi_align(feature, rois, output_size, spatial_scale, sampling_ratio)` returns `(K, C, P, P)`
    and loops over RoIs: scale the box to feature coordinates, shift by -0.5 (`aligned=True`: pixel *centres*),
    build the `P*s` sample coordinates along y and along x, sample the full grid, and average the
    `s x s` samples of every bin. These are exactly the numbers in the lab's calculation box: the
    "-0.5 -> index" line, then four neighbours times their weights. The checker compares with
    `torchvision.ops.roi_align`.
    """),
    api("Tensor.floor", "Tensor.long", "Tensor.new_zeros", "torch.meshgrid", title="Functions you will need"),
    api("Tensor.clamp", "Tensor.tolist", "torch.arange", brief=True),
    exercise(
        stub=r"""
        def bilinear(feature, y, x):
            '''feature (C, H, W); y, x: float tensors of one same shape S (row and column of each point) ->
            (C, *S): the feature map interpolated at every point (y, x).'''
            C, H, W = feature.shape
            # 1. keep the points inside the map
            y = y.clamp(0, H - 1)
            x = x.clamp(0, W - 1)
            # 2. the four neighbouring cells of every point: rows y0 (above) and y1 (below),
            #    columns x0 (left) and x1 (right), as integer indices
            y0 = y.floor().long()
            x0 = x.floor().long()
            y1 = (y0 + 1).clamp(max=H - 1)
            x1 = (x0 + 1).clamp(max=W - 1)
            ly = y - y0                                  # in [0, 1): how far the point lies below row y0
            lx = x - x0                                  # in [0, 1): how far it lies right of column x0
            # 3. YOUR CODE: top_left, feature[:, y0, x0] times its weight (1 - ly) * (1 - lx) -> (C, *S)
            # 4. YOUR CODE: top_right, feature[:, y0, x1] times (1 - ly) * lx
            # 5. YOUR CODE: bottom_left, feature[:, y1, x0] times ly * (1 - lx)
            # 6. YOUR CODE: bottom_right, feature[:, y1, x1] times ly * lx
            # 7. YOUR CODE: return the sum of the four (the four weights add up to 1)
            raise NotImplementedError

        def roi_align(feature, rois, output_size=7, spatial_scale=1.0, sampling_ratio=2):
            '''feature (C, H, W), rois (K, 4) xyxy in image pixels -> (K, C, P, P), P = output_size:
            a P x P tile for every RoI, each bin the mean of sampling_ratio x sampling_ratio bilinear samples.'''
            C = feature.shape[0]
            P = output_size                                  # bins per side
            s = sampling_ratio                               # samples per bin and side
            out = feature.new_zeros((len(rois), C, P, P))    # (K, C, P, P), zeros of the same dtype as feature
            for k, box in enumerate(rois):
                # 1. the box in feature-map coordinates, shifted by -0.5 (aligned: pixel centres)
                x1, y1, x2, y2 = (box * spatial_scale - 0.5).tolist()
                bw = (x2 - x1) / P                           # width of one bin, in feature cells
                bh = (y2 - y1) / P                           # height of one bin
                # 2. where the s samples sit inside one bin, as a fraction of the bin: e.g. [0.25, 0.75] for s = 2
                offsets = (torch.arange(s, dtype=feature.dtype) + 0.5) / s     # (s,)
                bins = torch.arange(P, dtype=feature.dtype)                    # (P,) bin numbers 0 .. P-1
                # 3. YOUR CODE: ys, the P*s sample rows: y1 + (bin number + offset) * bh, for every bin and
                #               every offset: broadcast bins[:, None] against offsets[None, :] -> (P, s),
                #               then reshape(-1) -> (P*s,)
                # 4. YOUR CODE: xs, the P*s sample columns, the same with x1 and bw -> (P*s,)
                # 5. YOUR CODE: yy, xx, the full grid of sample points: torch.meshgrid(ys, xs, indexing="ij")
                #               -> (P*s, P*s) each
                # 6. YOUR CODE: vals, the feature at every sample point: bilinear(feature, yy, xx) -> (C, P*s, P*s)
                # 7. YOUR CODE: for every bin (i, j), i, j = 0 .. P-1: its s x s samples are
                #               vals[:, i*s:(i+1)*s, j*s:(j+1)*s] -> (C, s, s); out[k, :, i, j] = their mean over the
                #               last two axes -> (C,)
                raise NotImplementedError
            return out

        tests.check_roi_align(roi_align)
        """,
        solution=r"""
        def bilinear(feature, y, x):
            '''feature (C, H, W); y, x: float tensors of one same shape S (row and column of each point) ->
            (C, *S): the feature map interpolated at every point (y, x).'''
            C, H, W = feature.shape
            # 1. keep the points inside the map
            y = y.clamp(0, H - 1)
            x = x.clamp(0, W - 1)
            # 2. the four neighbouring cells of every point: rows y0 (above) and y1 (below),
            #    columns x0 (left) and x1 (right), as integer indices
            y0 = y.floor().long()
            x0 = x.floor().long()
            y1 = (y0 + 1).clamp(max=H - 1)
            x1 = (x0 + 1).clamp(max=W - 1)
            ly = y - y0                                  # in [0, 1): how far the point lies below row y0
            lx = x - x0                                  # in [0, 1): how far it lies right of column x0
            # 3. - 6. every neighbour times its weight: the closer the point, the larger the weight
            top_left = feature[:, y0, x0] * (1 - ly) * (1 - lx)        # (C, *S)
            top_right = feature[:, y0, x1] * (1 - ly) * lx             # (C, *S)
            bottom_left = feature[:, y1, x0] * ly * (1 - lx)           # (C, *S)
            bottom_right = feature[:, y1, x1] * ly * lx                # (C, *S)
            # 7. their sum (the four weights add up to 1)
            return top_left + top_right + bottom_left + bottom_right

        def roi_align(feature, rois, output_size=7, spatial_scale=1.0, sampling_ratio=2):
            '''feature (C, H, W), rois (K, 4) xyxy in image pixels -> (K, C, P, P), P = output_size:
            a P x P tile for every RoI, each bin the mean of sampling_ratio x sampling_ratio bilinear samples.'''
            C = feature.shape[0]
            P = output_size                                  # bins per side
            s = sampling_ratio                               # samples per bin and side
            out = feature.new_zeros((len(rois), C, P, P))    # (K, C, P, P), zeros of the same dtype as feature
            for k, box in enumerate(rois):
                # 1. the box in feature-map coordinates, shifted by -0.5 (aligned: pixel centres)
                x1, y1, x2, y2 = (box * spatial_scale - 0.5).tolist()
                bw = (x2 - x1) / P                           # width of one bin, in feature cells
                bh = (y2 - y1) / P                           # height of one bin
                # 2. where the s samples sit inside one bin, as a fraction of the bin: e.g. [0.25, 0.75] for s = 2
                offsets = (torch.arange(s, dtype=feature.dtype) + 0.5) / s     # (s,)
                bins = torch.arange(P, dtype=feature.dtype)                    # (P,) bin numbers 0 .. P-1
                # 3. ys, the P*s sample rows: row i of the (P, s) array holds the s samples of bin row i
                ys = y1 + (bins[:, None] + offsets[None, :]) * bh              # (P, 1) + (1, s) -> (P, s)
                ys = ys.reshape(-1)                                            # (P*s,)
                # 4. xs, the P*s sample columns
                xs = x1 + (bins[:, None] + offsets[None, :]) * bw              # (P, s)
                xs = xs.reshape(-1)                                            # (P*s,)
                # 5. the full grid of sample points
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")                 # (P*s, P*s) each
                # 6. the feature at every sample point
                vals = bilinear(feature, yy, xx)                               # (C, P*s, P*s)
                # 7. every bin is the mean of its s x s samples
                for i in range(P):
                    for j in range(P):
                        bin_samples = vals[:, i * s:(i + 1) * s, j * s:(j + 1) * s]   # (C, s, s)
                        out[k, :, i, j] = bin_samples.mean((1, 2))                    # (C,)
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
    api("torchvision.models.detection.fasterrcnn_resnet50_fpn_v2"),
    api("Module.eval", brief=True),
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
    api("torchvision.models.detection.faster_rcnn.FastRCNNPredictor", title="Functions you will need"),
    exercise(
        stub=r"""
        torch.manual_seed(0)
        model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", **FAST)
        # 1. the old predictor (COCO: 91 classes) and the number of features its cls_score layer reads
        old_predictor = model.roi_heads.box_predictor
        # 2. YOUR CODE: in_features, the input size of old_predictor.cls_score (a Linear layer: .in_features)
        # 3. YOUR CODE: install a new FastRCNNPredictor(in_features, 2) as model.roi_heads.box_predictor
        #               (2 classes: background + person)
        raise NotImplementedError

        tests.check_box_predictor(model, num_classes=2)
        """,
        solution=r"""
        torch.manual_seed(0)
        model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", **FAST)
        # 1. the old predictor (COCO: 91 classes) and the number of features its cls_score layer reads
        old_predictor = model.roi_heads.box_predictor
        # 2. in_features, the input size of old_predictor.cls_score
        in_features = old_predictor.cls_score.in_features                       # 1024
        # 3. install a new predictor for 2 classes: background + person
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

        tests.check_box_predictor(model, num_classes=2)
        """),
    api("np.convolve"),
    api("torchkit.count_parameters", "np.ones", brief=True),
    code(r"""
    print(f"parameters: {torchkit.count_parameters(model) / 1e6:.1f} M")

    def train():
        # only called when there is no saved checkpoint yet
        return day2.train_detector(model, pf_train, epochs=2, batch_size=2, lr=0.01, max_minutes=3, log_every=5)

    history = day2.load_or_train(model, "d2_02_frcnn_pennfudan.pt", train)
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
      every GT (0.5 / 0.3 for our 128 px images). ~1% of anchors are positive, hence minibatch sampling
      (two-stage) or focal loss (one-stage).
    * **Encoding**: $t_x, t_y$ = centre offset / anchor size, $t_w, t_h$ = log size ratio; smooth-L1.
    * **RPN**: 3x3 conv + two 1x1 convs (A objectness, 4A deltas); proposals = decode, clip, top-k, NMS, top-N.
    * **RoIAlign** samples bilinearly at exact positions; RoI pooling rounds twice - the box edges (up to
      half a stride) and then the bin borders (up to about one more cell) - and its output jumps instead
      of following the box.
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
