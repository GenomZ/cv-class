from nb import md, code, exercise, header

NAME = "D2-04-modern-detectors"

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
GREEN, RED, GREY, BLUE, ORANGE, PURPLE = "#54A24B", "#E45756", "#9a9a9a", "#4C78A8", "#F58518", "#B279A2"
"""

CELLS = [
    header(
        "D2-04 · Modern detectors in practice: YOLO, DETR, fine-tuning, deployment",
        steps=[
            "The YOLO family in one page, and where today's models sit on the speed / accuracy curve",
            "DETR: set prediction with Hungarian matching - no anchors, no NMS",
            "Open-vocabulary detection: classes as text prompts",
            "Fine-tune a modern anchor-free detector (FCOS) and compare it with D2-02's Faster R-CNN",
            "Deployment: input size vs latency, ONNX export, quantisation, monitoring",
            "Which detector would you pick? Three scenarios",
        ],
        topics=[
            "Name what each YOLO generation changed and why NMS-free training matters",
            "Implement DETR's matching cost and the Hungarian assignment",
            "Adapt a pretrained detector to a new dataset and evaluate it honestly",
            "Measure latency and reason about the resolution / accuracy / speed trade-off",
            "Choose a detector for a real constraint (latency, data, licence, open vocabulary)",
        ],
        minutes=75,
        prereq="D2-01 to D2-03.",
    ),
    md(r"""
    ## 0 · Setup
    """),
    code(SETUP),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · The YOLO family in one page

    "YOLO" is no longer one model but a brand shared by several groups. What each step changed:

    | version | year | what changed |
    |---|---|---|
    | YOLOv1 | 2016 | one grid, 2 boxes per cell, one forward pass (our D2-03 mini detector) |
    | YOLOv2 / 9000 | 2017 | anchors from k-means on the dataset, BatchNorm, multi-scale training |
    | YOLOv3 | 2018 | 3 scales (FPN-like), Darknet-53, independent sigmoid classes |
    | YOLOv4 / v5 | 2020 | CSP backbones, Mosaic augmentation (four training images tiled into one), CIoU loss; v5 = PyTorch + Ultralytics tooling |
    | YOLOX | 2021 | **anchor-free**, **decoupled** cls / box heads, SimOTA dynamic assignment |
    | YOLOv6 / v7 | 2022 | re-parameterised blocks (train multi-branch, deploy as one conv), better assignment |
    | YOLOv8 | 2023 | anchor-free decoupled head with **Distribution Focal Loss**, task-aligned assignment |
    | YOLOv9 | 2024 | programmable gradient information (auxiliary reversible branch) |
    | YOLOv10 | 2024 | **NMS-free**: one-to-many + one-to-one heads trained together, only the one-to-one head at inference |
    | YOLO11 / 12 | 2024-25 | efficiency tweaks; v12 brings attention blocks into the backbone |
    | YOLO26 | 2025-26 | end-to-end NMS-free by default, DFL removed for simpler export to edge devices |

    The arc: anchors appeared (v2), multi-scale arrived (v3), anchors disappeared again (YOLOX, v8),
    and the last hand-written step - NMS - is being trained away (v10, 26), following DETR.
    Meanwhile transformer detectors became real-time (RT-DETR, 2023).
    """),
    code(r"""
    # Speed / accuracy of current real-time detectors (published numbers: COCO val mAP@[.5:.95],
    # latency per image on an NVIDIA T4 GPU with TensorRT FP16, YOLO numbers without NMS; approximate - read the trend)
    families = {
        "YOLOv8 (n-x)":      ([1.47, 2.66, 5.86, 9.06, 14.37], [37.3, 44.9, 50.2, 52.9, 53.9], BLUE),
        "YOLOv10 (n-x)":     ([1.84, 2.49, 4.74, 5.74, 7.28, 10.70], [38.5, 46.3, 51.1, 52.5, 53.2, 54.4], ORANGE),
        "YOLO11 (n-x)":      ([1.5, 2.5, 4.7, 6.2, 11.3], [39.5, 47.0, 51.5, 53.4, 54.7], GREEN),
        "RT-DETR (R18-R101)": ([4.6, 9.2, 13.5], [46.5, 53.1, 54.3], PURPLE),
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, (lat, ap, c) in families.items():
        ax.plot(lat, ap, "o-", color=c, lw=2, label=name)
    for name, ap in [("torchvision Faster R-CNN R50-FPN v2", 46.7), ("torchvision FCOS R50-FPN", 39.2), ("torchvision RetinaNet R50-FPN", 36.4)]:
        ax.axhline(ap, color=GREY, ls=":", lw=1); ax.text(24.8, ap + 0.15, name, fontsize=8, ha="right", va="bottom", color="#555")
    ax.set_xlabel("latency (ms / image, T4 TensorRT FP16)"); ax.set_ylabel("COCO mAP@[.5:.95]"); ax.set_xlim(0, 25)
    ax.grid(alpha=.3); ax.legend(loc="lower right"); ax.set_title("Real-time detectors 2023-2025: each curve = one family, small to large")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    The torchvision models we use in class (dotted lines, accuracy only - their latency was never
    optimised for deployment) are **teaching models**: readable, BSD-licensed, easy to modify. For
    production on a GPU you would pick a curve on the left part of this plot. Note the licences:
    Ultralytics YOLO (v5, v8, 11, 26) is **AGPL-3.0** (or a paid licence), RT-DETR (Apache-2.0) and
    torchvision (BSD) are permissive - a real constraint at work.
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · DETR: detection as set prediction

    DETR (Carion et al. 2020) removes every hand-designed component of the previous notebooks:

    1. a CNN backbone produces a feature map; it is flattened into a sequence of tokens (+ positional encodings:
       a vector added to every token that says where in the image it came from, because attention itself ignores order);
    2. a **transformer encoder** lets every location attend to every other one (global context);
    3. a **decoder** takes $N = 100$ learned **object queries**; each query attends to the image and outputs one
       prediction: class probabilities over $C + 1$ classes (incl. "no object") and a box (normalised cxcywh);
    4. training: **Hungarian matching** finds the one-to-one assignment between the $N$ predictions and the
       $G$ ground-truth objects with the lowest total cost; matched queries learn their object, all others
       learn "no object".

    Because each object is assigned to exactly one query, the network learns **not to produce duplicates**:
    no anchors, no NMS, no IoU thresholds.
    """),
    code(r"""
    fig, ax = plt.subplots(figsize=(15, 3.6)); ax.axis("off"); ax.set_xlim(0, 15); ax.set_ylim(0, 3.6)
    items = [(0.1, "image", "#dddddd"), (1.9, "CNN\nbackbone", "#9ecae1"), (3.7, "flatten +\npositional enc.\n(HW tokens)", "#c6dbef"),
             (5.7, "transformer\nencoder", ORANGE), (7.8, "transformer\ndecoder", "#fdd0a2"), (10.1, "N predictions\n(class, box)", GREEN),
             (12.4, "Hungarian\nmatching\n(train only)", PURPLE)]
    for x, t, c in items:
        ax.add_patch(FancyBboxPatch((x, 1.1), 1.6, 1.5, boxstyle="round,pad=0.05", fc=c, ec="k", alpha=0.85))
        ax.text(x + 0.8, 1.85, t, ha="center", va="center", fontsize=9)
    for x0, x1 in [(1.75, 1.9), (3.55, 3.7), (5.35, 5.7), (7.35, 7.8), (9.45, 10.1), (11.75, 12.4)]:
        ax.annotate("", (x1, 1.85), (x0, 1.85), arrowprops=dict(arrowstyle="-|>", lw=1.6))
    ax.text(8.6, 3.25, "N learned object queries", ha="center", fontsize=9)
    ax.annotate("", (8.6, 2.65), (8.6, 3.1), arrowprops=dict(arrowstyle="-|>", lw=1.6))
    ax.text(13.2, 0.6, "vs ground-truth set", ha="center", fontsize=9)
    ax.text(10.9, 0.6, "no NMS at inference", ha="center", fontsize=9, color=GREEN, weight="bold")
    plt.show()
    """),
    md(r"""
    ### The matching cost

    For prediction $i$ and ground truth $j$ DETR uses

    $$\mathcal{C}_{ij} = -\lambda_{\text{cls}}\,\hat p_i(c_j) + \lambda_{L1}\,\lVert \hat b_i - b_j \rVert_1 - \lambda_{\text{giou}}\,\mathrm{GIoU}(\hat b_i, b_j),
    \qquad \lambda_{\text{cls}} = 1,\ \lambda_{L1} = 5,\ \lambda_{\text{giou}} = 2,$$

    where $\hat p_i(c_j)$ is the probability that prediction $i$ gives to the class $c_j$ of ground truth $j$,
    and boxes are normalised $(c_x, c_y, w, h)$. The **Hungarian algorithm** (Kuhn 1955) finds the
    assignment $\sigma$ (ground truth $j$ gets query $\sigma(j)$, no query used twice) that minimises
    $\sum_j \mathcal{C}_{\sigma(j) j}$ exactly in $O(N^3)$. `scipy.optimize.linear_sum_assignment` implements it
    and accepts rectangular matrices (more queries than objects).

    ### Exercise 1 · `hungarian_match`

    Return `(query_idx, gt_idx)` for a cost matrix `(Q, G)`, numpy or torch: two integer arrays of
    length `G` (one pair per ground-truth object), so that `cost[query_idx, gt_idx].sum()` is minimal.
    """),
    exercise(
        stub=r"""
        from scipy.optimize import linear_sum_assignment

        def hungarian_match(cost):
            cost = cost.detach().cpu().numpy() if torch.is_tensor(cost) else np.asarray(cost)
            # YOUR CODE HERE
            raise NotImplementedError

        tests.check_hungarian(hungarian_match)
        """,
        solution=r"""
        from scipy.optimize import linear_sum_assignment

        def hungarian_match(cost):
            cost = cost.detach().cpu().numpy() if torch.is_tensor(cost) else np.asarray(cost)
            return linear_sum_assignment(cost)

        tests.check_hungarian(hungarian_match)
        """),
    code(r"""
    # The cartoon: 3 queries, 2 objects. Greedy "each GT takes its cheapest query" vs the optimal matching
    cost = np.array([[0.20, 0.30],
                     [0.25, 0.90],
                     [0.80, 0.85]])
    r, c = day2.hungarian_match(cost)
    greedy = [int(cost[:, 0].argmin()), int(np.delete(np.arange(3), cost[:, 0].argmin())[cost[np.delete(np.arange(3), cost[:, 0].argmin()), 1].argmin()])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    for ax, assign, title in [(axes[0], list(zip(greedy, [0, 1])), f"greedy (GT A first): total {cost[greedy[0], 0] + cost[greedy[1], 1]:.2f}"),
                              (axes[1], list(zip(r, c)), f"Hungarian (optimal): total {cost[r, c].sum():.2f}")]:
        ax.imshow(cost, cmap="Reds", vmin=0, vmax=1)
        viz.annotate_grid(ax, cost, fmt="{:.2f}", fontsize=11)
        for qi, gj in assign:
            ax.add_patch(Rectangle((gj - 0.5, qi - 0.5), 1, 1, fill=False, ec=BLUE, lw=4))
        ax.set_xticks([0, 1]); ax.set_xticklabels(["GT A", "GT B"]); ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["query 1", "query 2", "query 3"])
        ax.set_title(title, fontsize=10)
    ax = axes[2]; ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    q = [(1.0, 4.5), (1.0, 3.0), (1.0, 1.5)]; g = [(8.5, 4.0), (8.5, 2.0)]
    for k, (x, y) in enumerate(q):
        ax.add_patch(FancyBboxPatch((x - 0.6, y - 0.4), 1.2, 0.8, boxstyle="round,pad=0.05", fc="#fdd0a2", ec="k")); ax.text(x, y, f"query {k + 1}", ha="center", va="center", fontsize=9)
    for k, (x, y) in enumerate(g):
        ax.add_patch(FancyBboxPatch((x - 0.6, y - 0.4), 1.2, 0.8, boxstyle="round,pad=0.05", fc="#c7e9c0", ec="k")); ax.text(x, y, f"GT {'AB'[k]}", ha="center", va="center", fontsize=9)
    for qi, gj in zip(r, c):
        ax.annotate("", (g[gj][0] - 0.65, g[gj][1]), (q[qi][0] + 0.65, q[qi][1]), arrowprops=dict(arrowstyle="-|>", lw=2.5, color=BLUE))
        ax.text(q[qi][0] + 0.25 * (g[gj][0] - q[qi][0]) + 0.6, q[qi][1] + 0.25 * (g[gj][1] - q[qi][1]) + 0.15, f"cost {cost[qi, gj]:.2f}", color=BLUE, ha="center")
    unmatched = [k for k in range(3) if k not in r]
    for k in unmatched:
        ax.text(q[k][0] + 1.0, q[k][1], "-> learns 'no object'", fontsize=9, va="center", color=GREY)
    ax.set_title("one-to-one: every GT gets exactly one query", fontsize=10)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: greedy lets GT A take query 1, its cheapest (0.20), which leaves GT B only the
    expensive query 3 (0.85), total 1.05. The optimum gives query 1 to B and query 2 to A: each
    choice is slightly worse for one object, but the sum is 0.55.
    """),
    md(r"""
    ### Lab · Hungarian matching
    **Predict first:** two queries sit almost exactly on the same object. After matching, how many of
    them are trained to predict that object, and what is the other one trained to predict?
    Drag the boxes to check, then work through the "Try this" list under the lab (6 minutes).
    """),
    code("""
    labs.show("hungarian-matching")
    """),
    md(r"""
    ### Exercise 2 · the DETR matching cost

    `detr_cost(pred_logits (Q, C+1), pred_boxes (Q, 4) cxcywh in [0, 1], gt_labels (G,), gt_boxes (G, 4) cxcywh)`
    -> `(Q, G)`. Use `softmax` for the class probabilities, `torch.cdist(..., p=1)` for the L1 term and
    `torchvision.ops.generalized_box_iou` (needs xyxy: use `day2.box_cxcywh_to_xyxy_t`). This is the
    matrix of the lab: rows are queries, columns are ground-truth objects.
    """),
    exercise(
        stub=r"""
        from torchvision.ops import generalized_box_iou

        def detr_cost(pred_logits, pred_boxes, gt_labels, gt_boxes, w_class=1.0, w_l1=5.0, w_giou=2.0):
            # YOUR CODE HERE: three (Q, G) matrices, then their weighted sum
            raise NotImplementedError

        tests.check_detr_cost(detr_cost)
        """,
        solution=r"""
        from torchvision.ops import generalized_box_iou

        def detr_cost(pred_logits, pred_boxes, gt_labels, gt_boxes, w_class=1.0, w_l1=5.0, w_giou=2.0):
            c_class = -pred_logits.softmax(-1)[:, gt_labels]                     # (Q, G)
            c_l1 = torch.cdist(pred_boxes, gt_boxes, p=1)                         # (Q, G)
            c_giou = -generalized_box_iou(day2.box_cxcywh_to_xyxy_t(pred_boxes), day2.box_cxcywh_to_xyxy_t(gt_boxes))
            return w_class * c_class + w_l1 * c_l1 + w_giou * c_giou

        tests.check_detr_cost(detr_cost)
        """),
    md(r"""
    ### Watch set prediction happen

    No network yet - just **6 free "queries"**, each a learnable box and a learnable class
    distribution over `circle / square / triangle / no-object`. Every step we match them to the 3
    objects of a shapes image with your cost + Hungarian matching and apply DETR's loss: cross-entropy
    for all queries (unmatched ones -> "no object", down-weighted 0.1), L1 + GIoU for the matched ones.
    The queries *compete*; the losers drift into "no object", like the dashed queries in the lab.
    That is why DETR needs no NMS.
    """),
    code(r"""
    cost_fn = detr_cost if tests.check_detr_cost(detr_cost, verbose=False) else day2.detr_cost
    match_fn = hungarian_match if tests.check_hungarian(hungarian_match, verbose=False) else day2.hungarian_match
    s_img, s_boxes, s_labels, _ = shapes.make_sample(np.random.default_rng(5), 128, max_objects=3, min_r=14, max_r=26)
    gt_b = torch.tensor(boxes.xyxy_to_cxcywh(s_boxes) / 128, dtype=torch.float32); gt_l = torch.tensor(s_labels)
    torch.manual_seed(3)
    Q, NOOBJ = 6, 3
    q_box = torch.nn.Parameter(torch.randn(Q, 4) * 0.8)                       # sigmoid -> cxcywh in (0, 1)
    q_cls = torch.nn.Parameter(torch.zeros(Q, 4))
    opt = torch.optim.Adam([q_box, q_cls], lr=0.05)
    weight = torch.tensor([1.0, 1.0, 1.0, 0.1])                              # "no object" down-weighted
    snaps, hist_match = {}, []
    for step in range(301):
        pb = torch.sigmoid(q_box)
        if step in (0, 15, 40, 300):                                         # snapshot before the update
            snaps[step] = (pb.detach().numpy().copy(), q_cls.softmax(-1).detach().numpy().copy())
        with torch.no_grad():
            qi, gj = match_fn(cost_fn(q_cls, pb, gt_l, gt_b))
        tgt = torch.full((Q,), NOOBJ); tgt[qi] = gt_l[gj]
        l_cls = F.cross_entropy(q_cls, tgt, weight=weight)
        l_l1 = (pb[qi] - gt_b[gj]).abs().sum(-1).mean()
        l_giou = (1 - torchvision.ops.generalized_box_iou(day2.box_cxcywh_to_xyxy_t(pb[qi]), day2.box_cxcywh_to_xyxy_t(gt_b[gj])).diag()).mean()
        loss = l_cls + 5 * l_l1 + 2 * l_giou
        opt.zero_grad(); loss.backward(); opt.step()
        hist_match.append(dict(zip(gj.tolist(), qi.tolist())))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, (step, (pb, prob)) in zip(axes, snaps.items()):
        ax.imshow(s_img); ax.axis("off"); ax.set_xlim(-10, 138); ax.set_ylim(138, -10)
        xyxy = boxes.cxcywh_to_xyxy(pb * 128)
        for k in range(Q):
            lab_k = prob[k].argmax(); obj = lab_k != NOOBJ
            c = viz.color(k)
            ax.add_patch(Rectangle(xyxy[k, :2], *(xyxy[k, 2:] - xyxy[k, :2]), fill=False, ec=c, lw=2.2 if obj else 1, ls="-" if obj else ":"))
            if obj:
                ax.text(xyxy[k, 0] + 1, xyxy[k, 1] + 8, f"q{k}: {shapes.CLASS_NAMES[lab_k]} {prob[k].max():.2f}",
                        fontsize=7, color="white", bbox=dict(fc=c, ec="none", pad=0.5, alpha=0.85))
        n_obj = int((prob.argmax(1) != NOOBJ).sum())
        ax.set_title(f"step {step}: {n_obj} queries predict an object, {Q - n_obj} 'no object'", fontsize=10)
    fig.suptitle("6 learnable queries, 3 objects: matched queries converge, the others learn 'no object' (dotted)", y=1.02)
    plt.tight_layout(); plt.show()
    switches = sum(hist_match[t] != hist_match[t - 1] for t in range(1, len(hist_match)))
    print(f"the query-object assignment changed {switches} times in 300 steps. In a real DETR the queries are produced by a "
          "network from changing images, and early on the assignment flips constantly - one reason DETR trains slowly.")
    """),
    md(r"""
    **DETR's weak spots and their fixes.** The original needs ~500 epochs on COCO (vs ~12-36 for Faster
    R-CNN) because the matching is unstable early on and global attention over all pixels is slow to
    learn locality; it is also weak on small objects (single-scale, stride 32). **Deformable DETR** (2021)
    attends to a few learned sampling points on multi-scale features (10x faster convergence);
    **DN-DETR / DINO** (2022) add denoising queries; **RT-DETR** (2023) uses an efficient hybrid encoder and
    IoU-aware query selection to run in real time and is now a standard choice alongside YOLO.
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · Open-vocabulary detection

    Every detector so far has a fixed list of classes baked into its last layer. **Open-vocabulary**
    detectors (OWL-ViT 2022, Grounding DINO 2023, OWLv2, YOLO-World 2024) replace the class layer by a
    **text encoder**: each candidate box gets an embedding, each prompt ("a cat's head", "a red
    collar") gets an embedding, and the class score is their similarity - the same trick as CLIP
    (Day 3). New classes cost nothing but a sentence. Great for prototyping and for auto-labelling
    data that you then use to train a small fixed-class detector.

    The demo below runs OWL-ViT through Hugging Face `transformers` if it is installed
    (`pip install transformers`, ~600 MB of weights on first use); otherwise it draws the mechanism.
    """),
    code(r"""
    from PIL import Image
    _, pet_val, _ = torchkit.get_pet_datasets(target="detection")
    ov_img = Image.open(pet_val.image_path(40)).convert("RGB")
    prompts = ["a dog's head", "a dog's ear", "a dog's nose", "grass"]
    try:
        from transformers import pipeline
        ov = pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32")
        t0 = time.time(); found = ov(ov_img, candidate_labels=prompts, threshold=0.1); dt = time.time() - t0
        fig, ax = plt.subplots(figsize=(7, 6)); ax.imshow(ov_img); ax.axis("off")
        for d in found:
            b = d["box"]; c = viz.color(prompts.index(d["label"]))
            ax.add_patch(Rectangle((b["xmin"], b["ymin"]), b["xmax"] - b["xmin"], b["ymax"] - b["ymin"], fill=False, ec=c, lw=2))
            ax.text(b["xmin"], b["ymin"] - 3, f"{d['label']} {d['score']:.2f}", color="white", fontsize=8, bbox=dict(fc=c, ec="none", pad=1))
        ax.set_title(f"OWL-ViT, prompts = {prompts}  ({dt:.1f} s)"); plt.show()
    except Exception as e:  # ImportError, no internet, ...
        print(f"open-vocabulary demo skipped ({type(e).__name__}: {str(e)[:80]}). Showing the mechanism instead.")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw=dict(width_ratios=[1, 1.25]))
        axes[0].imshow(ov_img); axes[0].axis("off"); axes[0].set_title("image -> ViT -> one box + one embedding per patch token")
        W_, H_ = ov_img.size
        cand = [(0.25, 0.08, 0.75, 0.55), (0.2, 0.05, 0.38, 0.32), (0.4, 0.3, 0.6, 0.48), (0.0, 0.7, 1.0, 1.0)]
        for k, (a, b_, c_, d_) in enumerate(cand):
            axes[0].add_patch(Rectangle((a * W_, b_ * H_), (c_ - a) * W_, (d_ - b_) * H_, fill=False, ec=viz.color(k), lw=2, ls="--"))
            axes[0].text(a * W_ + 3, b_ * H_ + 14, f"box {k}", color="white", fontsize=9, bbox=dict(fc=viz.color(k), ec="none", pad=1))
        ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
        sim = np.array([[0.9, 0.2, 0.3, 0.0], [0.3, 0.8, 0.1, 0.0], [0.4, 0.1, 0.85, 0.0], [0.0, 0.0, 0.0, 0.7]])
        ax.imshow(sim, cmap="Greens", extent=(3, 9, 0.5, 4.5), vmin=0, vmax=1, aspect="auto")
        for i in range(4):
            ax.text(2.9, 4.0 - i, f"box {i}", ha="right", va="center", color=viz.color(i), weight="bold")
            ax.text(3.75 + 1.5 * i, 4.7, prompts[i], ha="center", fontsize=9, rotation=15)
        ax.text(6, 0.1, "illustrative similarity = cos(box embedding, text embedding) -> class score", ha="center", fontsize=9)
        ax.text(6, 5.6, "text encoder: one embedding per prompt", ha="center", fontsize=10, weight="bold")
        plt.tight_layout(); plt.show()
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · Fine-tune a modern anchor-free detector and compare

    We fine-tune torchvision's **FCOS** (anchor-free, focal loss, centre-ness - everything from D2-03)
    on **exactly the same** Penn-Fudan subset, resolution and budget as the Faster R-CNN of D2-02, so the
    comparison is fair: 40 training images at 320 px, 2 epochs (40 iterations of 2 images) capped at
    3 minutes, evaluated with the same mAP code on the same 20 validation images. The provided checkpoints
    in `data/checkpoints/` were produced by these exact cells without the time cap; delete them to train
    yourself. The D2-02 model is re-trained here if its checkpoint is missing.

    One difference: FCOS gets **LR 0.003** instead of 0.01. With 0.01 (fine for Faster R-CNN) the dense
    focal-loss head kept oscillating and reached only mAP@0.5 = 0.72 in the same 40 iterations; with 0.003
    it converges. Learning rate is the first knob to check when a fine-tune underperforms.

    ### Exercise 3 · adapt FCOS to 2 classes

    FCOS has no separate predictor module: its `head.classification_head` ends in a 3 x 3 conv
    `cls_logits` with `num_anchors * num_classes` outputs (num_anchors = 1, anchor-free). Replace
    **only that conv** (keep the pretrained tower), set `classification_head.num_classes = 2`, and
    initialise the new conv like RetinaNet (weights $\mathcal N(0, 0.01)$, bias $-\log\frac{1-0.01}{0.01}$).
    """),
    exercise(
        stub=r"""
        import math
        from torchvision.models.detection import fcos_resnet50_fpn
        torch.manual_seed(0)
        fcos = fcos_resnet50_fpn(weights="DEFAULT", min_size=320, max_size=512)
        cls_head = fcos.head.classification_head
        # YOUR CODE HERE (4-5 lines)
        raise NotImplementedError

        tests.check_box_predictor(fcos, num_classes=2)
        """,
        solution=r"""
        import math
        from torchvision.models.detection import fcos_resnet50_fpn
        torch.manual_seed(0)
        fcos = fcos_resnet50_fpn(weights="DEFAULT", min_size=320, max_size=512)
        cls_head = fcos.head.classification_head
        cls_head.num_classes = 2
        cls_head.cls_logits = nn.Conv2d(cls_head.cls_logits.in_channels, cls_head.num_anchors * 2, 3, padding=1)
        nn.init.normal_(cls_head.cls_logits.weight, std=0.01)
        nn.init.constant_(cls_head.cls_logits.bias, -math.log((1 - 0.01) / 0.01))

        tests.check_box_predictor(fcos, num_classes=2)
        """),
    code(r"""
    pf_train, pf_val = day2.penn_fudan_splits(n_train=40, n_val=20, seed=0, max_side=512)
    fcos_hist = day2.load_or_train(
        fcos, "d2_04_fcos_pennfudan.pt",
        lambda: day2.train_detector(fcos, pf_train, epochs=2, batch_size=2, lr=0.003, max_minutes=3, log_every=5))
    fcos.eval()

    # the D2-02 Faster R-CNN (same data, same budget) - loaded from its checkpoint, or trained now
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    FAST = dict(min_size=320, max_size=512, rpn_pre_nms_top_n_test=500, rpn_post_nms_top_n_test=100,
                rpn_pre_nms_top_n_train=1000, rpn_post_nms_top_n_train=300, box_batch_size_per_image=64)
    torch.manual_seed(0)
    frcnn = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", **FAST)
    frcnn.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
    day2.load_or_train(frcnn, "d2_02_frcnn_pennfudan.pt",
                       lambda: day2.train_detector(frcnn, pf_train, epochs=2, batch_size=2, lr=0.01, max_minutes=3, log_every=5))
    frcnn.eval();
    """),
    md(r"""
    ### Exercise 4 · a latency benchmark

    Write `benchmark(model, sizes, repeats=3)` returning the **median** milliseconds of one forward
    pass (batch of 1, `torch.no_grad()`, one warm-up call first) for a random `3 x s x s` image at
    each size, as a dict `{size: milliseconds}` (e.g. `{320: 241.0, 480: ...}`). torchvision detectors resize internally, so set `model.transform.min_size = (s,)` and
    `model.transform.max_size = s` before timing each size. Use `time.perf_counter()`.
    """),
    exercise(
        stub=r"""
        @torch.no_grad()
        def benchmark(model, sizes=(320, 480, 640), repeats=3):
            model.eval()
            out = {}
            for s in sizes:
                model.transform.min_size, model.transform.max_size = (s,), s
                x = [torch.rand(3, s, s)]
                # YOUR CODE HERE: warm-up, then `repeats` timed calls, store the median in ms
                raise NotImplementedError
            return out
        """,
        solution=r"""
        @torch.no_grad()
        def benchmark(model, sizes=(320, 480, 640), repeats=3):
            model.eval()
            out = {}
            for s in sizes:
                model.transform.min_size, model.transform.max_size = (s,), s
                x = [torch.rand(3, s, s)]
                model(x)                                              # warm-up
                ts = []
                for _ in range(repeats):
                    t0 = time.perf_counter(); model(x); ts.append(time.perf_counter() - t0)
                out[s] = 1000 * float(np.median(ts))
            return out
        """),
    code(r"""
    sizes = (320, 480, 640)
    lat = {"Faster R-CNN v2": benchmark(frcnn, sizes), "FCOS": benchmark(fcos, sizes)}
    for m in (frcnn, fcos):
        m.transform.min_size, m.transform.max_size = (320,), 512           # back to the training resolution
    fig, ax = plt.subplots(figsize=(8, 4))
    for (name, d), c in zip(lat.items(), [BLUE, ORANGE]):
        ax.plot(list(d), list(d.values()), "o-", color=c, lw=2, label=name)
        for s, v in d.items(): ax.text(s, v * 1.04, f"{v:.0f}", ha="center", fontsize=8, color=c)
    ax.plot(sizes, [lat["FCOS"][320] * (s / 320) ** 2 for s in sizes], ":", color=GREY, label="ideal ~ pixels (s^2) from FCOS@320")
    ax.set_xlabel("input size (px)"); ax.set_ylabel("ms / image (CPU, batch 1)"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("latency grows with the number of pixels")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: from 320 to 640 px the side doubles and the pixel count grows 4x, but the measured
    latency usually grows by less (compare each curve with the dotted line): part of the work, such as
    the RoI head, NMS and framework overhead, does not depend on the resolution.
    """),
    md(r"""
    ### Exercise 5 · the comparison table

    Fill `rows` with one dict per model: name, parameters (millions, `torchkit.count_parameters`),
    mAP@0.5 and mAP@[.5:.95] on `pf_val` (`day2.evaluate_map(model, pf_val)` returns both from one
    inference pass, plus the predictions), and the latency at 320 px from `lat`. Then print it.
    """),
    exercise(
        stub=r"""
        rows = []
        for name, key, model in [("Faster R-CNN v2 (two-stage)", "Faster R-CNN v2", frcnn), ("FCOS (one-stage, anchor-free)", "FCOS", fcos)]:
            # YOUR CODE HERE: append dict(name=..., params=..., map50=..., map=..., ms=..., preds=...)
            raise NotImplementedError
        print(f"{'model':32s}{'params':>9}{'mAP@0.5':>9}{'mAP@[.5:.95]':>14}{'ms @320':>9}")
        for r_ in rows:
            print(f"{r_['name']:32s}{r_['params']:8.1f}M{r_['map50']:9.3f}{r_['map']:14.3f}{r_['ms']:9.0f}")
        """,
        solution=r"""
        rows = []
        for name, key, model in [("Faster R-CNN v2 (two-stage)", "Faster R-CNN v2", frcnn), ("FCOS (one-stage, anchor-free)", "FCOS", fcos)]:
            res = day2.evaluate_map(model, pf_val)
            rows.append(dict(name=name, params=torchkit.count_parameters(model) / 1e6, map50=res["map50"], map=res["map"],
                             ms=lat[key][320], preds=res["preds"]))
        print(f"{'model':32s}{'params':>9}{'mAP@0.5':>9}{'mAP@[.5:.95]':>14}{'ms @320':>9}")
        for r_ in rows:
            print(f"{r_['name']:32s}{r_['params']:8.1f}M{r_['map50']:9.3f}{r_['map']:14.3f}{r_['ms']:9.0f}")
        """),
    md(r"""
    **What to expect.** With the provided checkpoints both models land at mAP@0.5 = 0.99 and
    mAP@[.5:.95] = 0.78-0.79 on our 20 validation images - a tie (20 images cannot separate them).
    With a GPU, the full 120-image training split and 800 px, both reach ~0.97-0.99 / ~0.75-0.85.
    FCOS has ~25% fewer parameters and no RoI stage, so it is usually faster at the same resolution
    (check the latency column for your machine). The differences between modern detectors on an easy,
    small dataset are smaller than the differences caused by **resolution, data, learning rate and
    training length** - measure on *your* data before choosing.
    """),
    md(r"""
    ### Fine-tuning on your own data: Pet heads (single class)

    The same 10 lines adapt FCOS to the course's running dataset, the Pet *head* boxes, at 320 px. We
    use 48 training images and 2 epochs (about 3 CPU minutes, capped at 2.5 here; a provided checkpoint
    makes re-runs instant). COCO has no "head" class: the detector has to learn a new definition of
    "object" from 48 examples. Expect mAP@0.5 around 0.6-0.7 on 16 validation images: the top-1 box
    usually sits on the head, but often too large (whole face + ears) or too small. With a GPU, the
    2 900 training images and ~10 epochs, the same code reaches mAP@0.5 above 0.95.
    """),
    code(r"""
    from torchvision.transforms import v2
    to_float = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    pet_ds_train, pet_ds_val, _ = torchkit.get_pet_datasets(target="detection", subset=(48, 16), seed=0, train_tf=to_float, eval_tf=to_float)
    heads_train = day2.materialise(pet_ds_train, range(len(pet_ds_train)), max_side=400)
    heads_val = day2.materialise(pet_ds_val, range(len(pet_ds_val)), max_side=400)
    torch.manual_seed(0)
    fcos_heads = fcos_resnet50_fpn(weights="DEFAULT", min_size=320, max_size=400)
    h = fcos_heads.head.classification_head
    h.num_classes = 2; h.cls_logits = nn.Conv2d(256, 2, 3, padding=1)
    nn.init.normal_(h.cls_logits.weight, std=0.01); nn.init.constant_(h.cls_logits.bias, -math.log(99))
    day2.load_or_train(fcos_heads, "d2_04_fcos_petheads.pt",
                       lambda: day2.train_detector(fcos_heads, heads_train, epochs=2, batch_size=2, lr=0.003, max_minutes=2.5, log_every=6))
    heads_res = day2.evaluate_map(fcos_heads, heads_val)
    heads_preds, heads_gts = heads_res["preds"], heads_res["gts"]
    print(f"FCOS on Pet heads (16 val images): mAP@0.5 = {heads_res['map50']:.3f}, mAP@[.5:.95] = {heads_res['map']:.3f}")
    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    for ax, i in zip(axes.flat, range(12)):
        ax.imshow(heads_val[i % len(heads_val)][0].permute(1, 2, 0)); ax.axis("off")
        viz.draw_boxes(ax, heads_gts[i % len(heads_val)]["boxes"], edgecolor=GREEN, lw=3, alpha=0.5)
        p = heads_preds[i % len(heads_val)]; k = np.argsort(-p["scores"])[:1]
        viz.draw_boxes(ax, p["boxes"][k], scores=p["scores"][k], edgecolor=RED, lw=1.8)
    fig.suptitle("FCOS fine-tuned on Pet heads: top-1 prediction (red) vs GT (green)", y=1.0)
    plt.tight_layout(); plt.show()
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · Deployment

    * **Resolution** is the biggest latency knob (cost grows with the number of pixels, see the plot);
      it is also an accuracy knob for small objects. Choose it on the validation set, per use case.
    * **Batching** amortises overhead on GPUs; on CPUs and edge devices latency at batch 1 is what matters.
    * **Export**: `torch.onnx.export` produces a framework-neutral graph that ONNX Runtime, TensorRT
      (NVIDIA), OpenVINO (Intel) or CoreML can optimise (layer fusion, FP16 / INT8 kernels). Detection
      post-processing (NMS) is often the hardest part to export - one more argument for NMS-free models.
    * **Quantisation**: INT8 weights and activations cut size by 4x and latency by 2-4x on supported
      hardware; post-training quantisation needs a few hundred calibration images, quantisation-aware
      training recovers the last accuracy points.
    * **Monitoring**: log the score distribution, the number of detections per image and box sizes in
      production. A shift (new camera, night images, a new product) shows up there long before anyone
      labels new data - compare with the validation distribution below.
    """),
    code(r"""
    # ONNX export of the D2-03-style mini detector (small, so the demo is instant)
    import io
    mini = day2.MiniDetector().eval()
    try:
        import onnx  # noqa: F401
        buf = io.BytesIO()
        torch.onnx.export(mini, torch.randn(1, 3, 128, 128), buf, input_names=["image"], output_names=["grid"],
                          dynamic_axes={"image": {0: "batch"}, "grid": {0: "batch"}})
        print(f"exported MiniDetector to ONNX: {len(buf.getvalue()) / 1e6:.2f} MB")
    except Exception as e:
        print(f"ONNX export skipped ({type(e).__name__}: {str(e)[:100]}). `pip install onnx onnxscript` to try it.")

    # Production-style monitoring: the distribution of detection scores on validation images
    fcos_preds = [r_ for r_ in rows if r_["name"].startswith("FCOS")][0]["preds"]
    all_scores = np.concatenate([p["scores"] for p in fcos_preds])
    n_det = [int((p["scores"] > 0.5).sum()) for p in fcos_preds]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    axes[0].hist(all_scores, bins=40, color=ORANGE); axes[0].set_xlabel("detection score"); axes[0].set_ylabel("count")
    axes[0].set_title("score histogram, FCOS on Penn-Fudan val (torchvision drops scores < 0.2)")
    cnt = np.bincount(n_det); axes[1].bar(range(len(cnt)), cnt, color=BLUE, width=0.7); axes[1].set_xlabel("detections > 0.5 per image")
    axes[1].set_ylabel("images"); axes[1].set_xticks(range(len(cnt)))
    axes[1].set_title("detections per image")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: the scores form two groups, many weak detections just above the 0.2 cut-off and a
    cluster of confident ones near 0.7-0.9, with few in between. A production monitor would watch the
    share of scores in the middle and the detections-per-image histogram: both shift when the input
    data changes.
    """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · Which detector would you pick?

    ### Exercise 6 · three scenarios (written)

    For each scenario, name a detector family (and a size / resolution), the main risk, and how you
    would validate the choice. There is no single right answer - argue with the trade-offs of today.

    1. **Factory line**: count screws on a conveyor belt, fixed camera, 60 fps, an industrial PC with a
       small NVIDIA GPU, 500 labelled images, product must ship commercially next quarter.
    2. **Wildlife camera traps**: 2 million unlabelled images, 30 rare species, latency irrelevant
       (batch processing overnight), biologists can label ~2 000 boxes.
    3. **Retail shelf audit app**: phone camera, detect ~1 000 different product SKUs that change monthly,
       a few examples per SKU.
    """),
    exercise(
        stub=r"""
        answers = {
            "factory line": "YOUR ANSWER HERE",
            "camera traps": "YOUR ANSWER HERE",
            "shelf audit": "YOUR ANSWER HERE",
        }
        for k, v in answers.items():
            print(f"{k}: {v}\n")
        """,
        solution=r"""
        answers = {
            "factory line": "Small real-time one-stage model (RT-DETR-R18 for a permissive licence, or a "
                            "YOLO-n/s with a commercial licence), fine-tuned at the lowest resolution that "
                            "still resolves a screw, exported to TensorRT FP16/INT8. Risk: tiny, touching "
                            "objects -> check recall on crowded frames and NMS (or an NMS-free model). "
                            "Validate on a held-out week of production video, measure fps on the target PC.",
            "camera traps": "Accuracy over speed: a large two-stage or DETR-family model (or an open-vocabulary "
                            "detector to pre-label 'animal' boxes, then human verification = cheap labels). "
                            "Detect 'animal' class-agnostically, classify species with a separate classifier "
                            "on crops (rare classes!). Risk: long tail and empty frames -> report per-species "
                            "recall and AP, not one mAP. Validate on cameras not seen in training.",
            "shelf audit": "Classes change monthly -> do not bake 1 000 SKUs into a detector head. Use a "
                           "class-agnostic 'product' detector (fine-tuned, mobile-sized, quantised) + an "
                           "embedding / retrieval model to recognise the SKU from a few reference photos "
                           "(or an open-vocabulary detector for prototyping). Risk: look-alike packages, "
                           "drift with new packaging -> monitor scores, re-index embeddings without retraining.",
        }
        for k, v in answers.items():
            print(f"{k}: {v}\n")
        """),

    # ------------------------------------------------------------------ 7
    md(r"""
    ## 7 · Summary

    * The YOLO line went grid -> anchors -> multi-scale -> anchor-free decoupled heads -> NMS-free;
      today's real-time detectors (YOLO 8/10/11, RT-DETR) span 1-15 ms on a T4 at 37-55 COCO mAP.
    * DETR casts detection as set prediction: N queries, Hungarian matching with a class + L1 + GIoU
      cost, a "no object" class, no anchors and no NMS. Deformable DETR, DINO and RT-DETR fixed its slow
      training and made it real-time.
    * Open-vocabulary detectors score boxes against text embeddings: classes become prompts.
    * Fine-tuning recipe: COCO-pretrained model, replace the class layer (prior-initialised bias),
      small LR, short schedule; compare models on the *same* data, resolution and budget.
    * Deployment: resolution drives latency; export (ONNX / TensorRT), quantise, and monitor score
      distributions for drift.

    ### Self-check
    1. Why does DETR need a "no object" class, and why is its loss weight reduced (0.1)?
    2. With 100 queries and 3 objects, how many predictions receive a box loss in one image?
    3. What does YOLOv10 change at training time to remove NMS at inference?
    4. You double the input resolution of a detector. Roughly how does latency change? What might improve?
    5. Name two reasons to prefer RT-DETR or torchvision models over Ultralytics YOLO in a commercial product.
    6. When is an open-vocabulary detector the right production choice, and when is it only a labelling tool?

    ### Go deeper
    * Carion et al., *End-to-End Object Detection with Transformers* (DETR, ECCV 2020).
    * Zhu et al., *Deformable DETR* (ICLR 2021).
    * Lv et al. / Zhao et al., *DETRs Beat YOLOs on Real-time Object Detection* (RT-DETR, CVPR 2024).
    * Wang et al., *YOLOv10: Real-Time End-to-End Object Detection* (NeurIPS 2024).
    * Liu et al., *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection* (2023).
    * Ultralytics documentation (docs.ultralytics.com) - training, export and licence notes for the YOLO family.
    """),
]
