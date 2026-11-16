from nb import md, code, exercise, header, api

NAME = "D3-03-instance-segmentation"

CELLS = [
    header(
        "D3-03 · Instance segmentation: Mask R-CNN and friends",
        steps=[
            "From boxes to masks: the Mask R-CNN mask branch",
            "Paste a 28x28 mask back into the image; why RoIAlign matters for masks",
            "Mask IoU, mask AP and panoptic quality",
            "Turn Penn-Fudan instance PNGs into Mask R-CNN targets",
            "Replace both heads of a COCO Mask R-CNN and fine-tune it on pedestrians",
            "Evaluate (box and mask AP), visualise, find the failure cases",
            "Alternatives: YOLO-seg prototypes, Mask2Former queries, SAM",
        ],
        topics=[
            "Mask R-CNN = Faster R-CNN + a small FCN per RoI (14x14 RoIAlign -> 28x28 mask per class)",
            "Decoupled mask and class: per-pixel BCE on the ground-truth class channel only",
            "Instance metrics: mask IoU, AP@[.5:.95] with mask matching, PQ = SQ x RQ",
            "Fine-tuning a torchvision detector: dataset targets, head replacement, warm start",
            "Reading failure cases: overlaps, occlusion, small and truncated people",
        ],
        minutes=75,
        prereq="D2-02 (anchors, RPN, RoIAlign, Faster R-CNN), D3-01 (IoU, Dice).",
    ),
    md(r"""
    ## 0 · Setup

    Mask R-CNN is a ResNet-50-FPN detector: on a CPU every image costs seconds, so this
    notebook uses small subsets, `min_size=320` and a provided checkpoint. The markdown
    gives the numbers you would see on a GPU with the full data.
    """),
    api("np.set_printoptions", "torch.manual_seed", "torchkit.get_device", "torchkit.get_penn_fudan", brief=True),
    code("""
    import sys, pathlib
    for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:   # find the cvcourse package
        if (p / "cvcourse").is_dir():
            sys.path.insert(0, str(p)); break

    import copy, time
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from torchvision.ops import roi_align
    from PIL import Image
    from cvcourse import viz, boxes, tests, torchkit, day3, labs

    %matplotlib inline
    plt.rcParams["figure.dpi"] = 90
    np.set_printoptions(precision=3, suppress=True)
    torch.manual_seed(0)
    device = torchkit.get_device()
    if device.type == "cpu":
        day3.tune_cpu_threads()

    pf = torchkit.get_penn_fudan()                       # 170 images, 423 pedestrians
    print(len(pf), "images; classes:", pf.classes)
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · From boxes to masks

    Mask R-CNN (He et al., 2017) adds **one small branch** to Faster R-CNN. For every RoI
    the box branch already classifies and refines; the mask branch predicts a
    **28x28 mask for every class** from a 14x14 RoIAlign crop of the FPN features:
    """),
    code("""
    fig, ax = plt.subplots(figsize=(14, 3.8))
    day3.draw_mask_head(ax)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Three design decisions that made it work:

    | decision | why |
    |---|---|
    | **RoIAlign** instead of RoIPool | masks need pixel-accurate alignment; rounding the RoI to the feature grid shifts the mask by up to half a stride (8 px at stride 16) |
    | **one mask per class, sigmoid + BCE on the GT class only** | the box branch decides *which* class; the mask branch only decides *which pixels*. A softmax across classes per pixel was 5.5 AP worse in the paper |
    | **28x28, small FCN** | cheap (only positive RoIs at training, top-100 detections at inference); resized to the box at the end |

    At inference: take the 28x28 channel of the predicted class, **resize it to the box,
    paste it into an empty image, threshold at 0.5**. That last step is the first exercise.

    The training target is built the other way round: the full-resolution GT mask is cropped
    to the proposal box and resampled to 28x28 with `roi_align` (torchvision's
    `project_masks_on_boxes`). Let us build one:
    """),
    api("torchvision.ops.roi_align", "Tensor.permute", "Tensor.numpy", "Tensor.float", "np.argmax", "torch.tensor", brief=True),
    code("""
    IDX = 7
    img_t, tgt = pf[IDX]
    img_np = img_t.permute(1, 2, 0).numpy()
    H, W = img_np.shape[:2]
    k = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in tgt["boxes"]]))   # the biggest person
    gt_mask = tgt["masks"][k].float()
    box = tgt["boxes"][k] + torch.tensor([-6.3, -4.7, 5.2, 3.9])             # a slightly-off proposal, fractional coords
    m28 = roi_align(gt_mask[None, None], [box[None]], output_size=28, spatial_scale=1.0, sampling_ratio=2, aligned=True)[0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(img_np); viz.draw_boxes(axes[0], box[None].numpy(), edgecolor="#E45756", lw=2)
    axes[0].set_title(f"image {H}x{W}, proposal box (red)")
    axes[1].imshow(gt_mask, cmap="gray"); viz.draw_boxes(axes[1], box[None].numpy(), edgecolor="#E45756", lw=2)
    axes[1].set_title("ground-truth instance mask")
    axes[2].imshow(m28, cmap="viridis", vmin=0, vmax=1); axes[2].set_title("28x28 training target\\n(roi_align of the GT mask)")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: the target (right) is soft along the person's outline, because `roi_align`
    averages the binary mask inside each of the 28x28 cells, and it does not quite reach the grid's
    borders, because the proposal box is a few pixels larger than the person. At inference the
    predicted 28x28 map makes the opposite trip: stretched back over the detected box.

    ### Lab · Mask paste-back
    **Predict first:** the same person fills a 32 px box in one image and a 340 px box in another,
    and both get a perfect 28x28 mask. Which pasted mask has the higher mask IoU with the true shape?
    Use the lab's box-size buttons to check, then work through its "Try this" list (5 minutes).
    """),
    code("""
    labs.show("mask-paste")
    """),
    md(r"""
    ### Exercise 1 · `paste_mask`

    Given a `(28, 28)` probability map, a box `(x1, y1, x2, y2)` (floats) and the image size
    `(H, W)`: resize the map to the box (`x0 = floor(x1)`, `w = ceil(x2) - x0`, same for y;
    bilinear via `F.interpolate`), threshold at 0.5, write it at `(y0, x0)` into an empty
    `(H, W)` boolean image, and **clip** the parts of the box that fall outside the image.
    `F.interpolate` expects a `(N, C, H, W)` tensor, so give the map two leading dimensions
    and use `mode="bilinear", align_corners=False`: exactly the 4-cell weights you hovered in
    the lab. Return the `(H, W)` bool array.
    """),
    api("torch.as_tensor", title="Functions you will need"),
    api("F.interpolate", "np.asarray", "np.zeros", "Tensor.sum", "np.ma.masked_where", brief=True),
    exercise(
        stub="""
        import math

        def paste_mask(mask28, box, image_hw, threshold=0.5):
            '''mask28 (28, 28) probabilities, box (x1, y1, x2, y2) in pixels (floats), image_hw = (H, W) ->
            (H, W) bool image: the map stretched over the box, thresholded, clipped to the image.'''
            m = torch.as_tensor(np.asarray(mask28, np.float32))     # (28, 28) tensor
            H, W = image_hw
            # 1. the box corners as Python floats
            x1 = float(box[0])
            y1 = float(box[1])
            x2 = float(box[2])
            y2 = float(box[3])
            # 2. the box on the pixel grid: top-left pixel (x0, y0) and size w x h (at least 1 pixel)
            x0 = int(math.floor(x1))
            y0 = int(math.floor(y1))
            w = max(int(math.ceil(x2)) - x0, 1)
            h = max(int(math.ceil(y2)) - y0, 1)
            out = np.zeros((H, W), bool)                            # the empty image
            # 3. YOUR CODE: resized, m stretched to (h, w): F.interpolate(m[None, None], size=(h, w), mode="bilinear",
            #               align_corners=False) -> (1, 1, h, w); then [0, 0] and .numpy() -> (h, w)
            # 4. YOUR CODE: the part of the box inside the image: rows ya = max(y0, 0) to yb = min(y0 + h, H),
            #               columns xa = max(x0, 0) to xb = min(x0 + w, W)
            # 5. YOUR CODE: if that part is not empty (ya < yb and xa < xb): piece, the matching part of resized,
            #               resized[ya - y0:yb - y0, xa - x0:xb - x0]; out[ya:yb, xa:xb] = piece >= threshold
            raise NotImplementedError
            return out

        tests.check_paste_mask(paste_mask)
        """,
        solution="""
        import math

        def paste_mask(mask28, box, image_hw, threshold=0.5):
            '''mask28 (28, 28) probabilities, box (x1, y1, x2, y2) in pixels (floats), image_hw = (H, W) ->
            (H, W) bool image: the map stretched over the box, thresholded, clipped to the image.'''
            m = torch.as_tensor(np.asarray(mask28, np.float32))     # (28, 28) tensor
            H, W = image_hw
            # 1. the box corners as Python floats
            x1 = float(box[0])
            y1 = float(box[1])
            x2 = float(box[2])
            y2 = float(box[3])
            # 2. the box on the pixel grid: top-left pixel (x0, y0) and size w x h (at least 1 pixel)
            x0 = int(math.floor(x1))
            y0 = int(math.floor(y1))
            w = max(int(math.ceil(x2)) - x0, 1)
            h = max(int(math.ceil(y2)) - y0, 1)
            out = np.zeros((H, W), bool)                            # the empty image
            # 3. stretch the 28x28 map over the box (F.interpolate wants (N, C, H, W))
            resized = F.interpolate(m[None, None], size=(h, w), mode="bilinear", align_corners=False)   # (1, 1, h, w)
            resized = resized[0, 0].numpy()                         # (h, w)
            # 4. clip the box to the image
            ya = max(y0, 0)
            yb = min(y0 + h, H)
            xa = max(x0, 0)
            xb = min(x0 + w, W)
            # 5. threshold the visible part and write it into the image
            if ya < yb and xa < xb:
                piece = resized[ya - y0:yb - y0, xa - x0:xb - x0]   # the part of the box inside the image
                out[ya:yb, xa:xb] = piece >= threshold
            return out

        tests.check_paste_mask(paste_mask)
        """),
    code("""
    back = paste_mask(m28, box, (H, W))
    g = gt_mask.numpy() > 0.5
    iou_rt = (back & g).sum() / (back | g).sum()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(m28, cmap="viridis"); axes[0].set_title("28x28 mask")
    axes[1].imshow(img_np); axes[1].imshow(np.ma.masked_where(~back, back), cmap="autumn", alpha=0.6)
    viz.draw_boxes(axes[1], box[None].numpy(), edgecolor="#E45756", lw=2); axes[1].set_title("pasted into the box")
    err = np.zeros((H, W, 3)); err[back & g] = 0.75; err[back & ~g] = [0.9, 0.15, 0.15]; err[~back & g] = [0.15, 0.35, 0.95]
    axes[2].imshow(err); axes[2].set_title(f"vs GT: IoU {iou_rt:.3f}  (red FP, blue FN)")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Even a *perfect* 28x28 prediction loses some detail (fingers, the gap between legs):
    the mask head's resolution is the ceiling of Mask R-CNN's mask quality. In the lab this
    showed up as a boundary IoU that falls as the box grows while the mask IoU barely moves.
    PointRend and query-based models (Mask2Former) predict at higher resolution for that reason.

    ### Why RoIAlign matters for masks: a round-trip experiment

    For every pedestrian in the dataset, crop its GT mask to its box, shrink to 28x28 and
    paste back - once with exact (RoIAlign-style) coordinates, once with the box **snapped to
    the stride-16 feature grid** as RoIPool does. The pasted mask always goes to the exact
    box (the box branch regresses it); only the crop is misaligned.
    """),
    api("torch.round"),
    api("np.array", "np.linspace", "Tensor.mean", brief=True),
    code("""
    def raw_instance_mask(i):
        return np.asarray(Image.open(pf.masks[i]))

    rt = {"RoIAlign (exact)": [], "RoIPool (snapped to stride 16)": []}
    for i in range(len(pf)):
        t = day3.penn_fudan_target(raw_instance_mask(i))
        for mk, b in zip(t["masks"], t["boxes"]):
            m = mk.float()                                         # (H, W) mask of one person
            hw = m.shape
            snapped = torch.round(b / 16) * 16                     # the box rounded to the stride-16 grid
            for name, bb in [("RoIAlign (exact)", b), ("RoIPool (snapped to stride 16)", snapped)]:
                crop = roi_align(m[None, None], [bb[None]], output_size=28, sampling_ratio=2, aligned=True)[0, 0]
                p = paste_mask(crop, b, hw)
                g_ = m.numpy() > 0.5
                rt[name].append((p & g_).sum() / max((p | g_).sum(), 1))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for name, col in zip(rt, ["#4C78A8", "#E45756"]):
        v = np.array(rt[name]); ax.hist(v, bins=np.linspace(0.4, 1, 49), alpha=0.65, color=col, label=f"{name}: mean IoU {v.mean():.3f}")
    ax.set_xlabel("mask IoU after crop -> 28x28 -> paste back"); ax.set_ylabel("instances"); ax.legend(fontsize=9)
    ax.set_title(f"{len(rt['RoIAlign (exact)'])} pedestrians: quantised crops lose ~0.1 IoU before any learning")
    plt.tight_layout(); plt.show()
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · Instance metrics

    Detection AP (D2-01) matches predictions to ground truth by **box** IoU. Instance
    segmentation uses exactly the same machinery with **mask** IoU: sort predictions by
    score, greedily match each to the best unmatched GT with IoU >= threshold, build the
    precision-recall curve, take the area. COCO reports AP@[.5:.95] (mean over 10 IoU
    thresholds), AP50, AP75 and AP for small / medium / large objects, for boxes (`bbox`)
    and masks (`segm`) separately. Mask AP is always lower than box AP: a mask must be right
    at every pixel, a box only at four edges.

    ### Exercise 2 · pairwise mask IoU

    `a (N, H, W)`, `b (M, H, W)` boolean -> `(N, M)`. Flatten each mask into a row: the
    intersections of all pairs are then **one matrix product**.
    """),
    api("Tensor.reshape", "np.maximum", brief=True, title="Functions you will need"),
    exercise(
        stub="""
        def mask_iou(a, b):
            '''a (N, H, W), b (M, H, W): binary masks -> (N, M), the mask IoU of every pair (0 where both are empty).'''
            # 1. every mask as one row of 0.0 / 1.0
            a = np.asarray(a).astype(bool)
            a = a.reshape(len(a), -1).astype(np.float64)        # (N, H*W)
            b = np.asarray(b).astype(bool)
            b = b.reshape(len(b), -1).astype(np.float64)        # (M, H*W)
            # 2. YOUR CODE: inter, the intersections of all pairs in one matrix product, a @ b.T -> (N, M)
            #               (entry [i, j] adds up a[i] * b[j] over the pixels: 1 exactly where both masks are 1)
            # 3. YOUR CODE: area_a, area_b, the number of pixels of every mask: the sum over axis 1 -> (N,), (M,)
            # 4. YOUR CODE: union = area_a[:, None] + area_b[None, :] - inter -> (N, M)
            # 5. YOUR CODE: return inter / np.maximum(union, 1) (two empty masks: 0 / 1 = 0 instead of 0 / 0)
            raise NotImplementedError

        tests.check_mask_iou(mask_iou)
        """,
        solution="""
        def mask_iou(a, b):
            '''a (N, H, W), b (M, H, W): binary masks -> (N, M), the mask IoU of every pair (0 where both are empty).'''
            # 1. every mask as one row of 0.0 / 1.0
            a = np.asarray(a).astype(bool)
            a = a.reshape(len(a), -1).astype(np.float64)        # (N, H*W)
            b = np.asarray(b).astype(bool)
            b = b.reshape(len(b), -1).astype(np.float64)        # (M, H*W)
            # 2. the intersections of all pairs: one matrix product
            inter = a @ b.T                                     # (N, M)
            # 3. the area of every mask
            area_a = a.sum(1)                                   # (N,)
            area_b = b.sum(1)                                   # (M,)
            # 4. the unions of all pairs
            union = area_a[:, None] + area_b[None, :] - inter   # (N, 1) + (1, M) - (N, M) -> (N, M)
            # 5. the IoU; a union of 0 means two empty masks, whose intersection is 0 too: 0 / 1 = 0
            return inter / np.maximum(union, 1)                 # (N, M)

        tests.check_mask_iou(mask_iou)
        """),
    md(r"""
    **Box IoU vs mask IoU.** People walking side by side have heavily overlapping boxes but
    almost disjoint masks. That is why box-based NMS hurts crowded scenes (it deletes the
    second person) and why mask metrics are stricter. The pair with the largest box overlap
    in Penn-Fudan:
    """),
    api("np.fill_diagonal", "np.unravel_index"),
    api("Tensor.max", "Tensor.argmax", brief=True),
    code("""
    best = (0, None)
    for i in range(len(pf)):
        t = day3.penn_fudan_target(raw_instance_mask(i))
        if len(t["boxes"]) < 2: continue
        bi = boxes.box_iou(t["boxes"].numpy(), t["boxes"].numpy()); np.fill_diagonal(bi, 0)
        if bi.max() > best[0]:
            a_, b_ = np.unravel_index(bi.argmax(), bi.shape); best = (bi.max(), (i, a_, b_))
    i, a_, b_ = best[1]
    t = day3.penn_fudan_target(raw_instance_mask(i)); im = pf[i][0].permute(1, 2, 0).numpy()
    miou_ab = mask_iou(t["masks"][[a_]].numpy(), t["masks"][[b_]].numpy())[0, 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].imshow(im); viz.draw_boxes(axes[0], t["boxes"][[a_, b_]].numpy(), labels=[0, 1], lw=2)
    axes[0].set_title(f"image {i}: box IoU {best[0]:.2f}")
    day3.draw_instances(axes[1], im, t["masks"][[a_, b_]].numpy(), title=f"same two people: mask IoU {miou_ab:.2f}")
    axes[0].axis("off"); plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Exercise 3 · mask AP

    Implement single-class AP with mask matching. `predictions[i] = {"masks": (N, H, W),
    "scores": (N,)}`, `ground_truths[i] = {"masks": (M, H, W)}`. Collect `(score, is_tp)` over
    all images, then build precision / recall and call `boxes.average_precision(p, r,
    "interp101")` (the D2-01 function).
    """),
    api("np.argsort", "np.cumsum", "np.maximum", brief=True, title="Functions you will need"),
    exercise(
        stub="""
        def mask_ap(predictions, ground_truths, iou_threshold=0.5):
            '''predictions[i] = {"masks": (N, H, W), "scores": (N,)}, ground_truths[i] = {"masks": (M, H, W)}, one class ->
            AP (COCO 101-point) over all images, with greedy matching by mask IoU.'''
            all_scores = []                                     # the score of every prediction of every image
            all_tp = []                                         # whether that prediction is a true positive
            n_gt = 0                                            # the number of GT masks in all images
            for pred, gt in zip(predictions, ground_truths):
                gm = np.asarray(gt["masks"]).astype(bool)       # (M, H, W)
                pm = np.asarray(pred["masks"]).astype(bool)     # (N, H, W)
                ps = np.asarray(pred["scores"], float)          # (N,)
                n_gt += len(gm)
                if len(pm) == 0:
                    continue
                # 1. this image's predictions in score order, and their IoU with every GT mask
                order = np.argsort(-ps)                         # (N,)
                if len(gm) > 0:
                    ious = mask_iou(pm[order], gm)              # (N, M); row k = the k-th best prediction
                else:
                    ious = np.zeros((len(pm), 0))               # no GT: every prediction will be a FP
                matched = np.zeros(len(gm), bool)               # (M,) becomes True once a GT has been found
                # 2. greedy matching, best score first (as precision_recall in D2-01)
                for k in range(len(order)):
                    # YOUR CODE: is_tp = False; if there is at least one GT: row = a copy of ious[k] (.copy()),
                    #            row[matched] = -1, j = the GT with the highest IoU in row; if row[j] >= iou_threshold:
                    #            GT j is matched and is_tp = True
                    # YOUR CODE: append the score ps[order[k]] to all_scores and is_tp to all_tp
                    raise NotImplementedError
            if len(all_scores) == 0:
                return 0.0
            # 3. all images together, highest score first (kind="stable": equal scores keep their order)
            order = np.argsort(-np.array(all_scores), kind="stable")
            tp = np.array(all_tp, bool)
            tp = tp[order]                                      # (number of predictions,)
            # 4. YOUR CODE: tp_count and fp_count, the running counts (np.cumsum of tp and of ~tp)
            # 5. YOUR CODE: precision = tp_count / np.maximum(tp_count + fp_count, 1e-12); recall = tp_count / max(n_gt, 1)
            # 6. YOUR CODE: return boxes.average_precision(precision, recall, "interp101")
            raise NotImplementedError

        tests.check_mask_ap(mask_ap)
        """,
        solution="""
        def mask_ap(predictions, ground_truths, iou_threshold=0.5):
            '''predictions[i] = {"masks": (N, H, W), "scores": (N,)}, ground_truths[i] = {"masks": (M, H, W)}, one class ->
            AP (COCO 101-point) over all images, with greedy matching by mask IoU.'''
            all_scores = []                                     # the score of every prediction of every image
            all_tp = []                                         # whether that prediction is a true positive
            n_gt = 0                                            # the number of GT masks in all images
            for pred, gt in zip(predictions, ground_truths):
                gm = np.asarray(gt["masks"]).astype(bool)       # (M, H, W)
                pm = np.asarray(pred["masks"]).astype(bool)     # (N, H, W)
                ps = np.asarray(pred["scores"], float)          # (N,)
                n_gt += len(gm)
                if len(pm) == 0:
                    continue
                # 1. this image's predictions in score order, and their IoU with every GT mask
                order = np.argsort(-ps)                         # (N,)
                if len(gm) > 0:
                    ious = mask_iou(pm[order], gm)              # (N, M); row k = the k-th best prediction
                else:
                    ious = np.zeros((len(pm), 0))               # no GT: every prediction will be a FP
                matched = np.zeros(len(gm), bool)               # (M,) becomes True once a GT has been found
                # 2. greedy matching, best score first (as precision_recall in D2-01)
                for k in range(len(order)):
                    is_tp = False
                    if len(gm) > 0:
                        row = ious[k].copy()                    # (M,) IoU of this prediction with every GT
                        row[matched] = -1                       # a GT can be found only once
                        j = int(row.argmax())
                        if row[j] >= iou_threshold:
                            matched[j] = True
                            is_tp = True
                    all_scores.append(ps[order[k]])
                    all_tp.append(is_tp)
            if len(all_scores) == 0:
                return 0.0
            # 3. all images together, highest score first (kind="stable": equal scores keep their order)
            order = np.argsort(-np.array(all_scores), kind="stable")
            tp = np.array(all_tp, bool)
            tp = tp[order]                                      # (number of predictions,)
            # 4. running counts down the list
            tp_count = np.cumsum(tp)
            fp_count = np.cumsum(~tp)
            # 5. precision and recall after every prediction
            precision = tp_count / np.maximum(tp_count + fp_count, 1e-12)
            recall = tp_count / max(n_gt, 1)
            # 6. the area under the curve, COCO style
            return boxes.average_precision(precision, recall, "interp101")

        tests.check_mask_ap(mask_ap)
        """),
    md(r"""
    ### Panoptic quality (PQ)

    Panoptic segmentation (Kirillov et al., 2019) scores *things and stuff* with one number.
    Predicted and GT segments match if their IoU > 0.5 (unique by construction, since
    panoptic segments do not overlap):

    $$\text{PQ} = \underbrace{\frac{\sum_{(p,g) \in TP} \text{IoU}(p, g)}{|TP|}}_{\text{SQ: segmentation quality}}
    \times \underbrace{\frac{|TP|}{|TP| + \frac12 |FP| + \frac12 |FN|}}_{\text{RQ: recognition quality (an F1)}}$$

    | metric | task | matches by | typical COCO value (2024 models) |
    |---|---|---|---|
    | mIoU | semantic | pixels per class | ADE20K ~55-60 |
    | mask AP | instance | mask IoU, score-ranked | COCO ~45-50 (Mask R-CNN R50: 37-41) |
    | PQ | panoptic | segment IoU > 0.5 | COCO ~55-58 (Mask2Former) |

    We compute PQ for the *person* class after fine-tuning (section 4).

    ## 3 · From instance PNGs to Mask R-CNN targets

    Penn-Fudan stores one PNG per image where pixel value `k` means "person number `k`"
    (0 = background). torchvision detection models want, per image, a dict with
    `boxes (N, 4)` float xyxy, `labels (N,)` int64 and `masks (N, H, W)` uint8.
    """),
    api("np.unique", brief=True),
    code("""
    raw = raw_instance_mask(IDX)
    print("raw mask PNG:", raw.shape, raw.dtype, "values", np.unique(raw))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].imshow(img_np); axes[0].set_title("image")
    axes[1].imshow(raw, cmap="gray", vmin=0, vmax=255); axes[1].set_title("raw PNG shown as 0-255: ids 1, 2 look black")
    axes[2].imshow(raw, cmap="tab10", vmin=0, vmax=9, interpolation="nearest"); axes[2].set_title("same PNG with a colour map")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Exercise 4 · `penn_fudan_target`

    One binary mask per instance id (excluding 0, in sorted order), its box as
    `(x_min, y_min, x_max, y_max)` of its pixels (the `torchvision.ops.masks_to_boxes`
    convention: max = last pixel index), label 1 (person) for everyone. The broadcasting
    trick `m[None] == ids[:, None, None]` builds all masks in one line.
    """),
    api("torch.unique", "torch.where", "torch.zeros", "torch.ones", "Tensor.to", "Tensor.min", brief=True,
        title="Functions you will need"),
    exercise(
        stub="""
        def penn_fudan_target(instance_mask):
            '''instance_mask (H, W): pixel value k = person number k, 0 = background ->
            {"boxes": (N, 4) float32 xyxy, "labels": (N,) int64 (all 1), "masks": (N, H, W) uint8}.'''
            m = torch.as_tensor(np.array(instance_mask))        # (H, W)
            # 1. the instance ids, sorted, without the background 0
            ids = torch.unique(m)
            ids = ids[ids != 0]                                 # (N,)
            # 2. YOUR CODE: masks, one binary mask per id, all at once with broadcasting:
            #               m[None] (1, H, W) == ids[:, None, None] (N, 1, 1) -> (N, H, W) bool; then .to(torch.uint8)
            # 3. YOUR CODE: boxes_ = torch.zeros((len(ids), 4), dtype=torch.float32); for every mask i (enumerate):
            #               ys, xs = torch.where(mask > 0), the rows and columns of its pixels;
            #               boxes_[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], dtype=torch.float32)
            # 4. YOUR CODE: labels, N ones of dtype torch.int64 (every instance is a person)
            raise NotImplementedError
            return {"boxes": boxes_, "labels": labels, "masks": masks}

        tests.check_penn_fudan_target(penn_fudan_target)
        """,
        solution="""
        def penn_fudan_target(instance_mask):
            '''instance_mask (H, W): pixel value k = person number k, 0 = background ->
            {"boxes": (N, 4) float32 xyxy, "labels": (N,) int64 (all 1), "masks": (N, H, W) uint8}.'''
            m = torch.as_tensor(np.array(instance_mask))        # (H, W)
            # 1. the instance ids, sorted, without the background 0
            ids = torch.unique(m)
            ids = ids[ids != 0]                                 # (N,)
            # 2. one binary mask per id, all at once: (1, H, W) == (N, 1, 1) -> (N, H, W)
            masks = m[None] == ids[:, None, None]               # (N, H, W) bool
            masks = masks.to(torch.uint8)
            # 3. the box of every mask: the extreme columns and rows of its pixels
            boxes_ = torch.zeros((len(ids), 4), dtype=torch.float32)
            for i, mask in enumerate(masks):
                ys, xs = torch.where(mask > 0)                  # rows and columns of the pixels of instance i
                boxes_[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], dtype=torch.float32)
            # 4. every instance is a person: label 1
            labels = torch.ones(len(ids), dtype=torch.int64)
            return {"boxes": boxes_, "labels": labels, "masks": masks}

        tests.check_penn_fudan_target(penn_fudan_target)
        """),
    api("torch.allclose", "torch.equal"),
    code("""
    mine = penn_fudan_target(raw)
    lib = pf[IDX][1]
    print("same boxes as torchkit's dataset:", torch.allclose(mine["boxes"], lib["boxes"]),
          "| same masks:", torch.equal(mine["masks"], lib["masks"]), "| keys torchkit adds:", sorted(set(lib) - set(mine)))
    fig, ax = plt.subplots(figsize=(5, 5))
    day3.draw_instances(ax, img_np, mine["masks"].numpy(), mine["boxes"].numpy(), title=f"{len(mine['boxes'])} instances from the PNG")
    plt.show()
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · Fine-tune Mask R-CNN on pedestrians

    `torchvision.models.detection.maskrcnn_resnet50_fpn_v2` with COCO weights has 91 class
    outputs in **two** predictors: the box predictor (`cls_score`, `bbox_pred`) and the mask
    predictor (`mask_fcn_logits`, one 28x28 map per class). For "background + person" both
    must be replaced. `day3.coco_maskrcnn(min_size=320)` builds the COCO model with
    CPU-friendly RoI counts (500 proposals for training, 200 at test time).

    ### Exercise 5 · replace both heads

    Assign a new `FastRCNNPredictor` to `model.roi_heads.box_predictor` and a new
    `MaskRCNNPredictor` to `model.roi_heads.mask_predictor`, both with 2 classes. Take their
    input sizes from the old predictors printed below (`cls_score.in_features`,
    `conv5_mask.in_channels`).
    """),
    api("torchvision.models.detection.mask_rcnn.MaskRCNNPredictor", title="Functions you will need"),
    api("torchvision.models.detection.faster_rcnn.FastRCNNPredictor", brief=True, title="Functions you will need (met before)"),
    exercise(
        stub="""
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        model = day3.coco_maskrcnn(min_size=320, max_size=512)
        coco_model = copy.deepcopy(model)                       # keep the original for comparison
        print("COCO box predictor:", model.roi_heads.box_predictor)
        print("COCO mask predictor:", model.roi_heads.mask_predictor)
        # 1. YOUR CODE: in_features, the input size of the old box predictor's cls_score layer (.in_features)
        # 2. YOUR CODE: model.roi_heads.box_predictor = a new FastRCNNPredictor(in_features, 2)
        # 3. YOUR CODE: in_ch, the input channels of the old mask predictor's conv5_mask layer (.in_channels)
        # 4. YOUR CODE: model.roi_heads.mask_predictor = a new MaskRCNNPredictor(in_ch, 256, 2) (256 hidden channels)
        raise NotImplementedError

        tests.check_maskrcnn_heads(model, num_classes=2)
        """,
        solution="""
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        model = day3.coco_maskrcnn(min_size=320, max_size=512)
        coco_model = copy.deepcopy(model)                       # keep the original for comparison
        print("COCO box predictor:", model.roi_heads.box_predictor)
        print("COCO mask predictor:", model.roi_heads.mask_predictor)
        # 1. the input size of the old box predictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features              # 1024
        # 2. a new box predictor for background + person
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        # 3. the input channels of the old mask predictor
        in_ch = model.roi_heads.mask_predictor.conv5_mask.in_channels                  # 256
        # 4. a new mask predictor: 256 hidden channels, one 28x28 mask per class
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_ch, 256, 2)

        tests.check_maskrcnn_heads(model, num_classes=2)
        """),
    md(r"""
    **A free lunch: warm start.** COCO already has a *person* class (index 1). Instead of
    starting the new 2-class heads from random numbers, copy the COCO rows for
    *background* and *person* into them (`day3.warm_start_from_coco`). The model then
    starts from "COCO person detector" and fine-tuning only adapts it to Penn-Fudan's
    annotation style. This works whenever your classes are a subset of the pretraining
    classes; for new classes, the random init is all you have.

    We also **freeze the ResNet body** (the FPN, RPN and heads still train): it halves the
    cost of a step, and 20 images are not enough to improve ImageNet+COCO features anyway.
    """),
    api("torch.utils.data.Subset"),
    api("Module.parameters", "Tensor.requires_grad_", "torchkit.count_parameters", "torchvision.transforms.v2.RandomHorizontalFlip",
        "torch.utils.data.DataLoader", brief=True),
    code("""
    day3.warm_start_from_coco(model, coco_model)
    for p in model.backbone.body.parameters():
        p.requires_grad_(False)
    print(f"trainable parameters: {torchkit.count_parameters(model, trainable_only=True):,} of {torchkit.count_parameters(model):,}")

    from torchvision.transforms import v2
    train_idx = list(range(0, 120, 6))                            # 20 training images (the checkpoint used all 120)
    val_idx = list(range(120, 170, 3))                            # 17 held-out images
    pf_train = torchkit.get_penn_fudan(transforms=v2.RandomHorizontalFlip(0.5))   # flips image, boxes and masks together
    train_loader = DataLoader(Subset(pf_train, train_idx), batch_size=2, shuffle=True, collate_fn=torchkit.detection_collate)
    print(len(train_idx), "training images,", len(val_idx), "validation images")
    """),
    md(r"""
    The cell below fine-tunes for at most 2.5 minutes (SGD, lr 0.005, momentum 0.9) when no
    checkpoint exists. The provided checkpoint `d3_maskrcnn_pennfudan.pt` was trained the
    same way on 120 images for 120 iterations (batch 2).

    > **Expected numbers** with the provided checkpoint on the 17 held-out images at
    > `min_size=320`: box AP50 ~0.99, mask AP50 ~0.97, mask AP@[.5:.95] ~0.73, person PQ ~0.8.
    > A 2.5-minute CPU run (about 10 iterations) lands within a few points of that thanks to
    > the warm start. The torchvision tutorial (full training split, 800 px, 10 epochs on a
    > GPU, ~5 min) reports box AP ~0.8 and mask AP ~0.75 at [.5:.95]. Note how good the
    > *untouched* COCO model already is: pedestrians are COCO's most frequent class.
    """),
    api("torchkit.load_checkpoint", "torchkit.save_checkpoint", "torch.optim.SGD", "np.convolve", "np.ones", brief=True),
    code("""
    FORCE_TRAIN = False
    CKPT = torchkit.CHECKPOINTS / "d3_maskrcnn_pennfudan.pt"
    model.to(device)
    if CKPT.exists() and not FORCE_TRAIN:
        extra = torchkit.load_checkpoint(model, CKPT)
        hist = extra.get("history", {})
        print(f"loaded {CKPT.name}: {extra.get('iters')} iterations on {extra.get('images')} images")
    else:
        params = []                                               # only the parameters that are not frozen
        for p in model.parameters():
            if p.requires_grad:
                params.append(p)
        opt = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=1e-4)
        hist = day3.train_detector(model, train_loader, opt, max_iters=30, max_minutes=2.5, log_every=5)
        torchkit.save_checkpoint(model, CKPT, iters=hist["iters"], images=len(train_idx), history=hist)
        print(f"trained {hist['iters']} iterations in {hist['seconds']:.0f} s -> saved {CKPT.name}")
    if hist.get("loss"):
        fig, ax = plt.subplots(figsize=(9, 3.4))
        for k_ in ["loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"]:
            if hist.get(k_): ax.plot(np.convolve(hist[k_], np.ones(3) / 3, mode="valid"), label=k_)
        ax.set_xlabel("iteration"); ax.set_ylabel("loss (smoothed)"); ax.set_yscale("log"); ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
        ax.set_title("the five Mask R-CNN loss terms"); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.show()
    """),
    md(r"""
    Five losses: RPN objectness and box regression (stage 1), classifier and box regression
    (stage 2), and the mask BCE. With the warm start the classifier loss is small from the
    first step; most of the learning goes into the mask and box regression.

    ### Evaluate: fine-tuned vs the original COCO model
    """),
    api("np.arange", "np.mean", "np.argmin", "np.round", "Tensor.any", brief=True),
    code("""
    val_imgs = []                                                         # (3, H, W) image tensors
    for i in val_idx:
        val_imgs.append(pf[i][0])
    val_gts = []                                                          # boxes and boolean masks, as NumPy
    for i in val_idx:
        t_ = pf[i][1]
        val_gts.append({"boxes": t_["boxes"].numpy(), "masks": t_["masks"].numpy().astype(bool)})

    t0 = time.time()
    preds_ft, preds_coco = [], []
    for im in val_imgs:                                                   # one image at a time keeps memory low
        preds_ft += day3.predict_instances(model, [im], score_threshold=0.05)
        preds_coco += day3.predict_instances(coco_model.to(device), [im], score_threshold=0.05, label=1)   # COCO 'person'
    print(f"inference on {2 * len(val_imgs)} images: {time.time() - t0:.0f} s")

    rows = {}
    for name, P in [("COCO model (no fine-tuning)", preds_coco), ("fine-tuned on Penn-Fudan", preds_ft)]:
        rows[name] = {"box": day3.coco_style_ap(P, val_gts, "box"), "mask": day3.coco_style_ap(P, val_gts, "mask")}
        print(f"{name:30s} box AP50 {rows[name]['box']['AP50']:.3f}  AP {rows[name]['box']['AP']:.3f}   |   "
              f"mask AP50 {rows[name]['mask']['AP50']:.3f}  AP75 {rows[name]['mask']['AP75']:.3f}  AP {rows[name]['mask']['AP']:.3f}")
    print("your mask_ap (AP50, fine-tuned):", round(mask_ap(preds_ft, val_gts, 0.5), 3))
    """),
    code("""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    labels_ = ["box AP50", "box AP", "mask AP50", "mask AP75", "mask AP"]
    xk = np.arange(len(labels_))
    for j, (name, r) in enumerate(rows.items()):
        vals = [r["box"]["AP50"], r["box"]["AP"], r["mask"]["AP50"], r["mask"]["AP75"], r["mask"]["AP"]]
        ax.bar(xk + (j - 0.5) * 0.38, vals, 0.38, label=name, color=["#BAB0AC", "#4C78A8"][j])
        for x_, v in zip(xk, vals): ax.text(x_ + (j - 0.5) * 0.38, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xk); ax.set_xticklabels(labels_); ax.set_ylim(0, 1.12); ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"Penn-Fudan, {len(val_idx)} held-out images, min_size 320")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Read the bars in pairs (grey = COCO model, blue = fine-tuned). At AP50 both models are near
    the ceiling, so look at the stricter mask AP75 and AP, which reward pixel-accurate masks.

    ### Panoptic-style quality for the *person* class
    """),
    code("""
    def pq_things(preds, gts, score_threshold=0.5):
        tp_iou = []                                              # the IoU of every matched (prediction, GT) pair
        fp = 0                                                   # predictions without a GT
        fn = 0                                                   # GTs without a prediction
        for p, g in zip(preds, gts):
            pm = p["masks"][p["scores"] >= score_threshold]      # the confident predicted masks
            gm = g["masks"]
            if len(pm) > 0 and len(gm) > 0:
                ious = mask_iou(pm, gm)                          # (predictions, GTs)
            else:
                ious = np.zeros((len(pm), len(gm)))
            used = np.zeros(len(pm), bool)                       # predictions already matched
            for j in range(len(gm)):                             # IoU > 0.5 -> a unique match
                if len(pm) > 0:
                    i = int(ious[:, j].argmax())                 # the best prediction for GT j
                else:
                    i = -1
                if i >= 0 and ious[i, j] > 0.5 and not used[i]:
                    used[i] = True
                    tp_iou.append(ious[i, j])
                else:
                    fn += 1
            fp += int((~used).sum())
        tp = len(tp_iou)
        if tp > 0:
            sq = float(np.mean(tp_iou))                          # segmentation quality: mean IoU of the matches
        else:
            sq = 0.0
        rq = tp / max(tp + 0.5 * fp + 0.5 * fn, 1e-9)            # recognition quality: an F1 score
        return sq * rq, sq, rq, tp, fp, fn

    for name, P in [("COCO model", preds_coco), ("fine-tuned", preds_ft)]:
        pq, sq, rq, tp, fp, fn = pq_things(P, val_gts)
        print(f"{name:11s} PQ {pq:.3f} = SQ {sq:.3f} x RQ {rq:.3f}   (TP {tp}, FP {fp}, FN {fn})")
    """),
    md(r"""
    Reading the numbers: on 17 images, AP differences of a couple of points are noise, and
    the COCO model's AP is already excellent. What fine-tuning changes is visible in PQ at
    a fixed score threshold: the COCO model also fires (correctly!) on the small
    background pedestrians that Penn-Fudan does not annotate, which count as false
    positives; the fine-tuned model has learned the dataset's annotation style and keeps
    far fewer of them. "Better on this benchmark" often means "matches its labelling
    conventions".

    ### Look at the predictions
    """),
    code("""
    show = [0, 2, 5, 9]
    fig, axes = plt.subplots(2, len(show), figsize=(16, 8.4))
    for j, s in enumerate(show):
        im = val_imgs[s].permute(1, 2, 0).numpy()
        day3.draw_instances(axes[0, j], im, val_gts[s]["masks"], val_gts[s]["boxes"], title=f"GT: {len(val_gts[s]['masks'])} people")
        p = preds_ft[s]; keep = p["scores"] >= 0.5
        day3.draw_instances(axes[1, j], im, p["masks"][keep], p["boxes"][keep], p["scores"][keep],
                            title=f"fine-tuned: {keep.sum()} detections (score >= 0.5)")
    plt.tight_layout(); plt.show()
    """),
    code("""
    # failure analysis: the held-out image with the lowest per-image mask AP50
    per_img = []                                                  # mask AP50 of every image on its own
    for p, g in zip(preds_ft, val_gts):
        per_img.append(mask_ap([p], [g], 0.5))
    w = int(np.argmin(per_img))                                   # the worst image
    im = val_imgs[w].permute(1, 2, 0).numpy()
    p = preds_ft[w]
    keep = p["scores"] >= 0.5
    pm = p["masks"][keep]
    gm = val_gts[w]["masks"]
    if len(pm) > 0 and len(gm) > 0:
        ious = mask_iou(pm, gm)                                   # (predictions, GT people)
    else:
        ious = np.zeros((len(pm), len(gm)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    day3.draw_instances(axes[0], im, gm, val_gts[w]["boxes"], title=f"GT ({len(gm)} people)")
    day3.draw_instances(axes[1], im, pm, p["boxes"][keep], p["scores"][keep], title=f"prediction (mask AP50 {per_img[w]:.2f})")
    err = np.zeros(im.shape)
    if len(pm) > 0:
        union_p = pm.any(0)                                       # pixels covered by any predicted mask
    else:
        union_p = np.zeros(im.shape[:2], bool)
    union_g = gm.any(0)                                           # pixels covered by any GT mask
    err[union_p & union_g] = 0.75; err[union_p & ~union_g] = [0.9, 0.15, 0.15]; err[~union_p & union_g] = [0.15, 0.35, 0.95]
    axes[2].imshow(err); axes[2].axis("off"); axes[2].set_title("red: extra pixels, blue: missed pixels")
    plt.tight_layout(); plt.show()
    print("best IoU of each GT person:", np.round(ious.max(0), 2) if ious.size else "no predictions")
    """),
    md(r"""
    Typical failure modes on Penn-Fudan (and in crowds in general):

    * **occlusion / overlap**: two people merged into one mask, or the occluded person's mask
      leaks into the front person (each RoI's 28x28 mask has no idea about the other RoIs);
    * **truncated / small people** at the image border or in the background: Penn-Fudan
      annotates only some of them, so a correct detection can count as a false positive;
    * **thin parts** (arms, legs, bags): limited by the 28x28 mask resolution.

    ## 5 · Alternatives to Mask R-CNN
    """),
    code("""
    day3.draw_instance_alternatives(plt.figure(figsize=(16, 4.8)))
    plt.show()
    """),
    md(r"""
    * **YOLO-seg / YOLACT** (one-stage, real time): the network predicts $k$ *prototype masks*
      for the whole image once, and for every detection a box, a class and $k$
      coefficients. The instance mask is `sigmoid(coefficients . prototypes)` cropped to
      the box - no per-RoI head. Ultralytics YOLOv8/11-seg is the practical default when
      you need masks at video rate.
    * **Mask2Former / query-based models**: a fixed set of learned queries, each predicting
      a class and a mask (dot product of the query embedding with per-pixel embeddings),
      trained with Hungarian matching like DETR. One architecture for semantic, instance
      and panoptic segmentation; no NMS. OneFormer, Mask DINO and SAM 3's detector follow
      this design.
    * **SAM as a mask generator**: Segment Anything turns a *box* (or a point) into a
      high-quality mask without knowing the class. Any detector + SAM = instance
      segmentation with zero mask labels; Grounding DINO + SAM = text-prompted instance
      segmentation. That is the bridge to D3-04.

    ## 6 · Summary

    * Mask R-CNN = Faster R-CNN + a 4-conv FCN per RoI that outputs one 28x28 mask per class; BCE on the GT-class channel only.
    * RoIAlign keeps the crop aligned to the pixels; quantised crops cost ~0.1 mask IoU before any learning.
    * Inference: resize the 28x28 mask to the box, paste, threshold; mask quality is capped by that resolution.
    * Instance metrics reuse detection AP with mask IoU; panoptic quality = SQ x RQ.
    * Fine-tuning a torchvision detector = targets in the right format + replace (or warm-start) both predictors + a few hundred iterations.

    ### Self-check (answers in the Day 3 quiz)
    1. Why does Mask R-CNN predict one mask per class with a sigmoid instead of a softmax over classes?
    2. A 28x28 mask is pasted into a 140x280 box. What limits the quality of thin structures?
    3. Two people overlap heavily. Why can box NMS remove a correct detection, and what would a mask-based or query-based model do differently?
    4. Which two modules must be replaced to fine-tune a COCO Mask R-CNN on 3 new classes? How many outputs does each get?
    5. PQ = 0.6 with SQ = 0.8. What is RQ, and what does it measure?

    ### Go deeper
    * He, Gkioxari, Dollar & Girshick, *Mask R-CNN* (ICCV 2017).
    * Kirillov et al., *Panoptic Segmentation* (CVPR 2019) and *PointRend* (CVPR 2020).
    * Bolya et al., *YOLACT: Real-time Instance Segmentation* (ICCV 2019): prototype masks.
    * Cheng et al., *Masked-attention Mask Transformer for Universal Image Segmentation* (Mask2Former, CVPR 2022).
    * torchvision *TorchVision Object Detection Finetuning Tutorial* (Penn-Fudan, the source of this notebook's recipe).
    * COCO detection evaluation (`bbox` vs `segm` AP, small / medium / large).
    """),
]
