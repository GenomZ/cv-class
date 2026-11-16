from nb import md, code, exercise, header, api

NAME = "D3-01-segmentation-basics"

CELLS = [
    header(
        "D3-01 · Segmentation basics: tasks, metrics, losses, upsampling",
        steps=[
            "Semantic vs instance vs panoptic segmentation on one Pet image; label formats and RLE",
            "Metrics: why pixel accuracy lies; confusion matrix -> IoU -> mIoU; Dice",
            "Losses: cross-entropy / BCE, soft Dice, focal; who gets the gradient on a small object",
            "Getting resolution back: transposed convolution, the checkerboard, bilinear + conv, pixel shuffle",
            "Context without downsampling: dilated (atrous) convolutions and ASPP",
            "The autoencoder ancestor: what a bottleneck forgets",
            "Build the Pet trimap target and look at the class frequencies",
        ],
        topics=[
            "The three segmentation tasks and how their labels are stored (masks, trimaps, polygons, RLE)",
            "mIoU and Dice from a confusion matrix; Dice = 2 IoU / (1 + IoU)",
            "Why Dice / focal losses help small objects, measured with autograd",
            "Transposed convolution = scatter-add; why stride 2 / kernel 3 makes a checkerboard",
            "Pixel shuffle, dilation, ASPP: the toolbox every segmentation decoder is built from",
            "Encoder-decoder without skips loses 'where' - the motivation for U-Net (next notebook)",
        ],
        minutes=90,
        prereq="Day 1 (convolutions, transfer learning) and Day 2 (detection, RoIAlign, FPN).",
    ),
    md(r"""
    ## 0 · Setup

    NumPy for the mechanisms, PyTorch for the losses, the upsampling layers and one small
    autoencoder. Everything runs on a CPU in a few minutes.
    """),
    api("np.set_printoptions", "torch.manual_seed", "torchkit.get_device", brief=True),
    code("""
    import sys, pathlib
    for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:   # find the cvcourse package
        if (p / "cvcourse").is_dir():
            sys.path.insert(0, str(p)); break

    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    import torch, torch.nn as nn, torch.nn.functional as F
    from cvcourse import viz, convops, seg, shapes, tests, torchkit, day3, labs

    %matplotlib inline
    plt.rcParams["figure.dpi"] = 90
    np.set_printoptions(precision=2, suppress=True)
    torch.manual_seed(0)
    device = torchkit.get_device()
    if device.type == "cpu":
        day3.tune_cpu_threads()
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · Three segmentation tasks

    Detection (Day 2) answers *"what is where, roughly?"* with boxes. Segmentation answers
    the same question **per pixel**:

    | task | output per pixel | two cats of the same breed | counts objects? |
    |---|---|---|---|
    | **semantic** | a class | one blob labelled "cat" | no |
    | **instance** | a class + an instance id, *things only* | "cat #1", "cat #2" | yes |
    | **panoptic** | class + id for *things*, class only for *stuff* (sky, road, grass) | "cat #1", "cat #2", "background" | yes |

    (*Things* are countable objects such as cats, cars, people; *stuff* is amorphous background such as sky, road or grass.)

    The Oxford-IIIT Pet image below happens to contain **two** Bengal cats. The dataset gives
    a *trimap* (pet / background / uncertain border) and one head box per animal.
    """),
    api("scipy.ndimage.label", "scipy.ndimage.binary_opening", "np.unique", "np.dstack"),
    api("np.zeros", "np.where", "Tensor.min", "Tensor.max", brief=True),
    code("""
    from scipy import ndimage

    ex = day3.load_pet_example("Bengal_105")
    img, tri = ex["image"], ex["trimap"]
    target = day3.trimap_to_target(tri)                    # 0 background, 1 pet, 2 border
    pet = target > 0                                       # pet + its border
    inst, n_inst = ndimage.label(ndimage.binary_opening(pet, iterations=2))   # connected components = instances here
    print("image", img.shape, "| raw trimap values", np.unique(tri), "| instances found:", n_inst,
          "| head boxes:", len(ex["head_boxes"]))

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3))
    axes[0].imshow(img); axes[0].set_title(f"image ({ex['breed']})")
    # semantic: one colour per CLASS
    sem_rgb = np.zeros(img.shape); sem_rgb[target == 1] = day3.SEG_COLORS[1]; sem_rgb[target == 2] = [0.8, 0.8, 0.8]
    axes[1].imshow(img); axes[1].imshow(np.dstack([sem_rgb, 0.65 * (target > 0)]))
    axes[1].set_title("semantic: every pet pixel = 'cat'\\n(grey = uncertain border)")
    # instance: one colour per OBJECT, plus boxes
    axes[2].imshow(img)
    inst_rgb = np.zeros(img.shape); inst_alpha = np.zeros(img.shape[:2])
    for k in range(1, n_inst + 1):
        inst_rgb[inst == k] = day3.SEG_COLORS[k + 1]; inst_alpha[inst == k] = 0.6
        ys, xs = np.where(inst == k)
        axes[2].add_patch(Rectangle((xs.min(), ys.min()), xs.max() - xs.min(), ys.max() - ys.min(),
                                    fill=False, ec=day3.SEG_COLORS[k + 1], lw=2))
        axes[2].text(xs.min(), ys.min() - 5, f"cat #{k}", color="white", fontsize=9,
                     bbox=dict(fc=day3.SEG_COLORS[k + 1], ec="none", pad=1))
    axes[2].imshow(np.dstack([inst_rgb, inst_alpha]))
    viz.draw_boxes(axes[2], ex["head_boxes"], edgecolor="white", lw=1)
    axes[2].set_title("instance: one mask + box per cat\\n(white = head boxes from the XML)")
    # panoptic: things with ids, stuff without
    pan = np.zeros(img.shape); pan[...] = [0.55, 0.55, 0.45]
    for k in range(1, n_inst + 1):
        pan[inst == k] = day3.SEG_COLORS[k + 1]
    axes[3].imshow(pan); axes[3].set_title("panoptic: every pixel labelled\\nstuff 'background' + things 'cat #1/#2'")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    > Instance segmentation is not "semantic + connected components" in general: two
    > touching cats form one blob. Here they do not touch, so `ndimage.label` recovers the
    > instances; D3-03 trains Mask R-CNN to separate touching pedestrians.

    ### How masks are stored

    | format | what | used by |
    |---|---|---|
    | dense mask PNG | one integer per pixel | Pet trimaps, Cityscapes, ADE20K, Penn-Fudan |
    | polygon | list of (x, y) vertices per object | COCO (`segmentation: [[x1, y1, ...]]`), LabelMe, CVAT |
    | RLE | run lengths of 0s and 1s | COCO crowd regions and *all* model outputs (`pycocotools`), SAM |

    Polygons are tiny and editable but cannot express holes easily; RLE is lossless and
    compresses large blobs very well.
    """),
    code("""
    import contourpy
    small = pet[::2, ::2]                                         # half resolution to keep the plot light
    lines = contourpy.contour_generator(z=small.astype(float)).lines(0.5)
    poly = max(lines, key=len)                                    # outline of the largest blob
    poly_s = poly[::max(len(poly) // 60, 1)]                     # an annotator would click ~60 points

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(small, cmap="gray"); axes[0].set_title(f"dense mask: {small.size:,} values")
    axes[1].imshow(img[::2, ::2]); axes[1].plot(poly_s[:, 0], poly_s[:, 1], "-o", ms=3, color="#F58518")
    axes[1].set_title(f"polygon: {len(poly_s)} vertices ({2 * len(poly_s)} numbers)")
    axes[2].imshow(tri, cmap="gray", vmin=0, vmax=255); axes[2].set_title("raw trimap PNG shown as 0-255:\\nvalues 1, 2, 3 look black!")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Exercise 1 · run-length encoding

    COCO's *uncompressed RLE* reads the mask **column by column** (Fortran order) and stores
    the lengths of alternating runs, **always starting with a run of 0s** (length 0 if the
    first pixel is 1). Example: the 2x3 mask `[[0, 1, 1], [0, 1, 0]]` read column-wise is
    `0 0 | 1 1 | 1 0` -> runs `0 0`, `1 1 1`, `0` -> `counts = [2, 3, 1]`.

    Implement `rle_encode(mask) -> {"size": [H, W], "counts": [...]}` and `rle_decode(rle)`.
    """),
    api("np.shape"),
    api("np.asarray", "Tensor.reshape", "np.array_equal", brief=True),
    exercise(
        stub="""
        def rle_encode(mask):
            '''mask (H, W) of 0 / 1 (or bool) -> {"size": [H, W], "counts": [run lengths]}: COCO's uncompressed RLE.'''
            mask = np.asarray(mask).astype(np.uint8)            # (H, W), values 0 / 1
            H, W = mask.shape
            flat = mask.reshape(-1, order="F")                  # (H * W,) column by column, like COCO
            counts = []
            current = 0                                         # the value of the run being counted: the first run is 0s
            run = 0                                             # its length so far (stays 0 if the mask starts with a 1)
            for value in flat:
                # YOUR CODE: if value == current, the run is one pixel longer (run += 1); otherwise the run is
                #            over: append run to counts, then a new run of `value` starts (current = value, run = 1)
                raise NotImplementedError
            # YOUR CODE: the last run is still open: append it to counts
            return {"size": [int(H), int(W)], "counts": counts}

        def rle_decode(rle):
            '''{"size": [H, W], "counts": [...]} -> mask (H, W) uint8 of 0 / 1: the inverse of rle_encode.'''
            H, W = rle["size"]
            flat = np.zeros(H * W, np.uint8)                    # (H * W,) all 0 to start
            pos = 0                                             # where the next run starts in flat
            value = 0                                           # the value of the next run: 0s first
            for length in rle["counts"]:
                # YOUR CODE: write value into the next `length` pixels, flat[pos:pos + length]
                # YOUR CODE: move pos to the end of this run
                # YOUR CODE: the next run has the other value: value = 1 - value
                raise NotImplementedError
            return flat.reshape((H, W), order="F")              # back to (H, W), column by column

        tests.check_rle(rle_encode, rle_decode)
        """,
        solution="""
        def rle_encode(mask):
            '''mask (H, W) of 0 / 1 (or bool) -> {"size": [H, W], "counts": [run lengths]}: COCO's uncompressed RLE.'''
            mask = np.asarray(mask).astype(np.uint8)            # (H, W), values 0 / 1
            H, W = mask.shape
            flat = mask.reshape(-1, order="F")                  # (H * W,) column by column, like COCO
            counts = []
            current = 0                                         # the value of the run being counted: the first run is 0s
            run = 0                                             # its length so far (stays 0 if the mask starts with a 1)
            for value in flat:
                if value == current:
                    run += 1                                    # the run is one pixel longer
                else:
                    counts.append(run)                          # the run is over
                    current = value                             # a new run of the other value starts
                    run = 1
            counts.append(run)                                  # the last run is still open
            return {"size": [int(H), int(W)], "counts": counts}

        def rle_decode(rle):
            '''{"size": [H, W], "counts": [...]} -> mask (H, W) uint8 of 0 / 1: the inverse of rle_encode.'''
            H, W = rle["size"]
            flat = np.zeros(H * W, np.uint8)                    # (H * W,) all 0 to start
            pos = 0                                             # where the next run starts in flat
            value = 0                                           # the value of the next run: 0s first
            for length in rle["counts"]:
                flat[pos:pos + length] = value                  # the run
                pos += length                                   # the next run starts here
                value = 1 - value                               # 0 -> 1 -> 0 ...
            return flat.reshape((H, W), order="F")              # back to (H, W), column by column

        tests.check_rle(rle_encode, rle_decode)
        """),
    code("""
    rle = rle_encode(pet)
    back = rle_decode(rle)
    print(f"pet mask {pet.shape}: {pet.size:,} pixels -> {len(rle['counts'])} run lengths "
          f"({pet.size / len(rle['counts']):.0f}x smaller), lossless: {np.array_equal(back, pet)}")
    print("first counts:", rle["counts"][:12], "...")
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · Metrics: pixel accuracy lies

    The obvious metric, the fraction of correctly labelled pixels, is dominated by the
    biggest class. A model that predicts **"background everywhere"** on an image where the
    object covers 2% of the pixels scores **98% pixel accuracy** and is useless.

    The fix is to score **each class separately** with the intersection over union you know
    from boxes, now computed on pixel sets, and average over classes:

    $$\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}, \qquad \text{mIoU} = \frac{1}{C}\sum_c \text{IoU}_c,
    \qquad \text{Dice}_c = \frac{2\,TP_c}{2\,TP_c + FP_c + FN_c}$$
    """),
    api("np.zeros_like", "np.nanmean"),
    api("np.arange", "Tensor.mean", brief=True),
    code("""
    H = W = 128
    gt = np.zeros((H, W), int); yy, xx = np.mgrid[:H, :W]
    gt[(yy - 70) ** 2 + (xx - 50) ** 2 <= 10 ** 2] = 1            # a small object: ~2% of the image
    lazy = np.zeros_like(gt)                                       # predicts background everywhere
    shifted = np.zeros_like(gt); shifted[(yy - 72) ** 2 + (xx - 54) ** 2 <= 11 ** 2] = 1   # decent but offset

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.5]))
    rows = []
    for ax, (name, p) in zip(axes, [("ground truth", gt), ("'all background'", lazy), ("offset circle", shifted)]):
        ax.imshow(day3.colorize(p)); ax.axis("off")
        acc = seg.pixel_accuracy(p, gt); iou = seg.iou_per_class(p, gt, 2); miou = np.nanmean(iou)
        ax.set_title(f"{name}\\nacc {acc:.3f} | IoU(obj) {iou[1]:.2f} | mIoU {miou:.2f}", fontsize=9)
        rows.append((name, acc, iou[1], miou))
    names = [r[0] for r in rows[1:]]
    x = np.arange(2)
    for k, (lab, col) in enumerate([("pixel accuracy", "#BAB0AC"), ("IoU of the object", "#E45756"), ("mIoU", "#4C78A8")]):
        axes[3].bar(x + (k - 1) * 0.27, [r[k + 1] for r in rows[1:]], 0.27, label=lab, color=col)
    axes[3].set_xticks(x); axes[3].set_xticklabels(names); axes[3].set_ylim(0, 1.1); axes[3].legend(fontsize=8, loc="upper right")
    axes[3].set_title("accuracy cannot tell them apart; IoU can")
    plt.tight_layout(); plt.show()
    print(f"object covers {gt.mean():.1%} of the image")
    """),
    md(r"""
    Compare the two predictions in the bar chart: the model that found nothing and the decent offset
    circle both score about 0.98 pixel accuracy; only the IoU of the object (0 vs 0.57) tells them apart.

    ### Lab · segmentation metrics
    **Predict first:** a perfect prediction is shifted one pixel to the right. Which loses more IoU, a large
    blob or a vessel 2 pixels wide, and roughly how much? Pick the presets in the lab to check, then work through
    the "Try this" list under it (4-5 minutes).
    """),
    code("""
    labs.show("seg-metrics")
    """),
    md(r"""
    ### Exercise 2 · Dice coefficient

    For two binary masks $A$ (prediction) and $B$ (target): $\text{Dice} = \frac{2|A \cap B|}{|A| + |B|}$
    (in the lab's terms, $2TP / (2TP + FP + FN)$). Two lines with NumPy boolean arrays; return a Python
    `float`, with `eps` added to numerator and denominator so that two empty masks score 1.
    """),
    api("np.logical_and", title="Functions you will need"),
    api("Tensor.sum", brief=True, title="Functions you will need (met before)"),
    exercise(
        stub="""
        def dice_coefficient(pred_mask, target_mask, eps=1e-7):
            '''two masks of the same shape (anything that converts to bool) -> Dice = 2|A and B| / (|A| + |B|), a float.'''
            p = np.asarray(pred_mask, bool)                     # A, the prediction
            t = np.asarray(target_mask, bool)                   # B, the target
            # 1. YOUR CODE: inter, the number of pixels that are True in both (np.logical_and, then .sum())
            # 2. YOUR CODE: return (2 * inter + eps) / (p.sum() + t.sum() + eps) as a Python float
            raise NotImplementedError

        tests.check_dice(dice_coefficient)
        """,
        solution="""
        def dice_coefficient(pred_mask, target_mask, eps=1e-7):
            '''two masks of the same shape (anything that converts to bool) -> Dice = 2|A and B| / (|A| + |B|), a float.'''
            p = np.asarray(pred_mask, bool)                     # A, the prediction
            t = np.asarray(target_mask, bool)                   # B, the target
            # 1. inter, the number of pixels that are True in both
            inter = np.logical_and(p, t).sum()
            # 2. the Dice formula; eps makes two empty masks score 1 instead of 0 / 0
            return float((2 * inter + eps) / (p.sum() + t.sum() + eps))

        tests.check_dice(dice_coefficient)
        """),
    md(r"""
    ### The confusion matrix holds every metric

    For $C$ classes, count pixels in a $C \times C$ matrix: row = true class, column =
    predicted class. The diagonal is $TP_c$; the rest of column $c$ is $FP_c$; the rest of
    row $c$ is $FN_c$. (The 2x2 matrix in the lab is the case $C = 2$.) Below: a shapes image
    and a deliberately flawed "prediction" (the square shifted, the triangle called a square,
    the circles shrunk).
    """),
    api("scipy.ndimage.binary_erosion", "np.roll", "np.log1p", "np.isnan"),
    api("np.random.default_rng", "np.ndenumerate", "np.nan_to_num", brief=True),
    code("""
    rng = np.random.default_rng(7)
    s_img, s_boxes, s_labels, s_mask = shapes.make_sample(rng, 128, max_objects=4)
    pred = s_mask.copy()
    objs = [(k, b, l) for k, (b, l) in enumerate(zip(s_boxes, s_labels))]
    for k, b, l in objs:
        region = np.zeros_like(s_mask, bool); x1, y1, x2, y2 = b.astype(int); region[y1:y2, x1:x2] = s_mask[y1:y2, x1:x2] == l + 1
        if l == 2:   pred[region] = 2                                            # triangle -> square
        elif l == 0: pred[region & ~ndimage.binary_erosion(region, iterations=3)] = 0   # circle shrunk
        else:        pred[region] = 0; pred[np.roll(region, (4, 6), (0, 1))] = l + 1   # square shifted
    cm = seg.confusion_matrix(pred, s_mask, 4)
    names4 = ["bg"] + shapes.CLASS_NAMES

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), gridspec_kw=dict(width_ratios=[1, 1, 1.2, 1.2]))
    axes[0].imshow(s_img); axes[0].set_title("image"); axes[1].imshow(day3.colorize(pred)); axes[1].set_title("flawed prediction")
    for ax in axes[:2]: ax.axis("off")
    im = axes[2].imshow(np.log1p(cm), cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        axes[2].text(j, i, f"{v}", ha="center", va="center", fontsize=8, color="white" if np.log1p(v) > np.log1p(cm).max() * 0.6 else "black")
    axes[2].set_xticks(range(4)); axes[2].set_xticklabels(names4, rotation=30); axes[2].set_yticks(range(4)); axes[2].set_yticklabels(names4)
    axes[2].set_xlabel("predicted"); axes[2].set_ylabel("true"); axes[2].set_title("confusion matrix (pixels)")
    iou = seg.iou_per_class(pred, s_mask, 4)
    axes[3].bar(names4, np.nan_to_num(iou), color=[day3.SEG_COLORS[i] for i in range(4)])
    for i, v in enumerate(iou):
        axes[3].text(i, np.nan_to_num(v) + 0.02, "absent" if np.isnan(v) else f"{v:.2f}", ha="center", fontsize=9)
    axes[3].set_ylim(0, 1.12); axes[3].set_title(f"per-class IoU -> mIoU = {np.nanmean(iou):.3f}")
    plt.tight_layout(); plt.show()
    print(f"pixel accuracy {seg.pixel_accuracy(pred, s_mask):.3f} vs mIoU {np.nanmean(iou):.3f}")
    """),
    md(r"""
    Read the off-diagonal cells: every triangle pixel sits in row "triangle", column "square", so the
    triangle's IoU is 0, while pixel accuracy barely notices because the background dominates the count.

    ### Exercise 3 · mIoU from the confusion matrix

    Use the matrix (rows = target, columns = prediction). A class absent from both the
    prediction and the target has an undefined IoU (`nan`) and is skipped by `np.nanmean`.
    """),
    api("np.diag", "np.nanmean", brief=True, title="Functions you will need (met before)"),
    exercise(
        stub="""
        def mean_iou(pred, target, num_classes):
            '''pred, target: integer masks of the same shape, values 0 .. num_classes-1 -> mIoU (a float),
            the mean IoU over the classes present in the prediction or the target.'''
            cm = seg.confusion_matrix(pred, target, num_classes).astype(float)   # (C, C) rows = target, cols = prediction
            # 1. YOUR CODE: tp, the true positives of every class: the diagonal of cm (np.diag) -> (C,)
            # 2. YOUR CODE: fp, pixels predicted as class c that are something else: column sums (cm.sum(0)) - tp -> (C,)
            # 3. YOUR CODE: fn, pixels of class c predicted as something else: row sums (cm.sum(1)) - tp -> (C,)
            # 4. YOUR CODE: iou, an empty list; for every class c: union = tp[c] + fp[c] + fn[c]; append
            #               tp[c] / union if union > 0, otherwise np.nan (class absent from both: undefined)
            # 5. YOUR CODE: return the mean that skips the nan values, float(np.nanmean(iou))
            raise NotImplementedError

        tests.check_mean_iou(mean_iou)
        """,
        solution="""
        def mean_iou(pred, target, num_classes):
            '''pred, target: integer masks of the same shape, values 0 .. num_classes-1 -> mIoU (a float),
            the mean IoU over the classes present in the prediction or the target.'''
            cm = seg.confusion_matrix(pred, target, num_classes).astype(float)   # (C, C) rows = target, cols = prediction
            # 1. tp, the true positives of every class: the diagonal
            tp = np.diag(cm)                                    # (C,)
            # 2. fp: column c holds everything predicted as c; minus the correct ones
            fp = cm.sum(0) - tp                                 # (C,)
            # 3. fn: row c holds every pixel that really is c; minus the correct ones
            fn = cm.sum(1) - tp                                 # (C,)
            # 4. the IoU of every class
            iou = []
            for c in range(num_classes):
                union = tp[c] + fp[c] + fn[c]                   # pixels of class c in the prediction or the target
                if union > 0:
                    iou.append(tp[c] / union)
                else:
                    iou.append(np.nan)                          # class absent from both: undefined
            # 5. the mean over the classes, skipping the nan values
            return float(np.nanmean(iou))

        tests.check_mean_iou(mean_iou)
        """),
    md(r"""
    ### Dice vs IoU

    Both count the same pixels, so they are monotonically related: with $I$ = IoU,
    $\text{Dice} = \frac{2I}{1 + I}$. Dice is always $\geq$ IoU (it looks more generous), and
    they agree at 0 and 1 (the curve in the lab: every mask you painted sat on it). Rankings of models are identical *per image*; averaged over a
    dataset they can differ slightly. Medical imaging reports Dice, natural-image benchmarks
    report mIoU - convert before you compare papers.
    """),
    api("np.array", "np.linspace", "np.clip", brief=True),
    code("""
    rng = np.random.default_rng(0)
    pts = []
    for _ in range(300):
        a = np.zeros((64, 64), bool); b = np.zeros((64, 64), bool)
        x, y, w, h = rng.integers(0, 40, 4); a[y:y + 10 + h // 2, x:x + 10 + w // 2] = True
        x, y, w, h = np.clip([x, y, w, h] + rng.integers(-12, 12, 4), 0, 40); b[y:y + 10 + h // 2, x:x + 10 + w // 2] = True
        pts.append((seg.iou_per_class(a.astype(int), b.astype(int), 2)[1], seg.dice_coefficient(a, b)))
    pts = np.array(pts)
    i = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(i, 2 * i / (1 + i), lw=2.5, label="Dice = 2 IoU / (1 + IoU)")
    ax.plot(i, i, "--", color="grey", label="Dice = IoU")
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color="#F58518", alpha=0.6, label="300 random mask pairs")
    ax.annotate("IoU 0.5 is Dice 0.67", (0.5, 2 / 3), xytext=(0.55, 0.35), arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.set_xlabel("IoU"); ax.set_ylabel("Dice"); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · Losses

    A segmentation network outputs a logit **per pixel and class**: shape `(N, C, H, W)`.
    The default loss is the classification loss applied at every pixel:

    * **cross-entropy** (`F.cross_entropy(logits, target)` with `target (N, H, W)` integers) for
      multi-class; **BCE with logits** for a single foreground channel;
    * optional **class weights** (`weight=`) to up-weight rare classes;
    * **soft Dice loss**: replace the hard sets in Dice by probabilities so it is differentiable,
      $\;\mathcal{L}_\text{Dice} = 1 - \frac{2\sum_i p_i t_i + \epsilon}{\sum_i p_i + \sum_i t_i + \epsilon}$;
    * **focal loss** (Day 2): down-weights easy pixels by $(1 - p_t)^\gamma$;
    * in practice **BCE + Dice** (or CE + Dice): CE gives smooth per-pixel gradients, Dice
      directly optimises the overlap and does not care how many background pixels there are.
    """),
    md(r"""
    ### Exercise 4 · soft Dice loss in PyTorch

    `prob` and `target` have the same shape (any shape); compute **one** Dice over the whole
    batch and return the loss $1 - \text{Dice}$ as a 0-dim tensor. Use torch operations only, so
    autograd can differentiate it.
    """),
    api("Tensor.float", "Tensor.sum", brief=True, title="Functions you will need (met before)"),
    exercise(
        stub="""
        def soft_dice_loss(prob, target, eps=1e-6):
            '''prob: probabilities in [0, 1], target: 0 / 1, same shape (any) -> 1 - Dice over the whole batch, a 0-d tensor.'''
            target = target.float()
            # 1. YOUR CODE: inter, the soft intersection: the sum of prob * target over all elements
            # 2. YOUR CODE: total, the soft sizes: prob.sum() + target.sum()
            # 3. YOUR CODE: dice = (2 * inter + eps) / (total + eps)
            # 4. YOUR CODE: return 1 - dice
            raise NotImplementedError

        tests.check_soft_dice_torch(soft_dice_loss)
        """,
        solution="""
        def soft_dice_loss(prob, target, eps=1e-6):
            '''prob: probabilities in [0, 1], target: 0 / 1, same shape (any) -> 1 - Dice over the whole batch, a 0-d tensor.'''
            target = target.float()
            # 1. the soft intersection: a pixel with p = 0.8 on the object counts 0.8
            inter = (prob * target).sum()
            # 2. the soft sizes of prediction and target
            total = prob.sum() + target.sum()
            # 3. the soft Dice
            dice = (2 * inter + eps) / (total + eps)
            # 4. the loss: 0 for a perfect prediction
            return 1 - dice

        tests.check_soft_dice_torch(soft_dice_loss)
        """),
    md(r"""
    ### Who gets the gradient on a small object?

    A 1-D "image" of 4 096 pixels holds an object of `n` pixels. The model currently predicts
    $p = 0.3$ on the object (too low) and $0.1$ on the background (slightly too high). We let
    autograd compute $\partial \mathcal{L} / \partial p_i$ for every pixel and ask: **what share of
    the total gradient comes from the object pixels?** With mean BCE, each pixel has the
    same voice, so 4 000 background pixels outvote a 40-pixel object. Dice normalises by the
    object size.
    """),
    api("torch.where", "F.binary_cross_entropy", "torch.log", "np.logspace", "np.log10"),
    api("torch.zeros", "torch.tensor", "Tensor.requires_grad_", "Tensor.backward", "Tensor.grad", "Tensor.abs", brief=True),
    code("""
    def grad_share(loss_name, n_obj, N=4096, p_obj=0.3, p_bg=0.1):
        t = torch.zeros(N); t[:n_obj] = 1
        p = torch.where(t > 0, torch.tensor(float(p_obj)), torch.tensor(float(p_bg))).requires_grad_(True)
        if loss_name == "BCE":   loss = F.binary_cross_entropy(p, t)
        elif loss_name == "Dice": loss = soft_dice_loss(p, t)
        else:
            p_t = torch.where(t > 0, p, 1 - p); a_t = torch.where(t > 0, torch.tensor(0.25), torch.tensor(0.75))
            loss = (-a_t * (1 - p_t) ** 2 * torch.log(p_t)).mean()
        loss.backward()
        g = p.grad.abs()
        return float(g[:n_obj].sum() / g.sum()), float(g[0])

    sizes = np.unique(np.logspace(np.log10(4), np.log10(2048), 30).astype(int))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name, col in [("BCE", "#4C78A8"), ("focal (gamma 2)", "#B279A2"), ("Dice", "#E45756")]:
        key = "focal" if name.startswith("focal") else name
        share = [grad_share(key, n)[0] for n in sizes]
        axes[0].plot(100 * sizes / 4096, share, lw=2.2, label=name, color=col)
        pp = np.linspace(0.02, 0.98, 60)
        g1 = [grad_share(key, 41, p_obj=v)[1] for v in pp]
        axes[1].plot(pp, g1, lw=2.2, label=name, color=col)
    axes[0].axhline(0.5, ls=":", color="grey")
    axes[0].set_xscale("log"); axes[0].set_xlabel("object size (% of pixels, log)"); axes[0].set_ylabel("share of |gradient| on object pixels")
    axes[0].set_title("small objects are drowned out by BCE"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_yscale("log"); axes[1].set_xlabel("predicted probability on the object pixels"); axes[1].set_ylabel("|dL/dp| for ONE object pixel")
    axes[1].set_title("gradient on one pixel of a 1% object"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Reading the plot: with BCE a 1% object receives only a few percent of the gradient.
    Dice hands it about a quarter **whatever its size** (its gradient is normalised by the
    object area), and focal loss helps too by silencing the easy background pixels - but
    its share still collapses for the tiniest objects. The right panel shows the flip side of
    focal: once a pixel is right (p -> 1) it stops learning from it. That is why Dice / BCE + Dice is the default for medical masks and
    thin structures. Dice has its own failure mode: on an *empty* target its gradient is
    unstable, which the BCE term smooths out.
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · Getting the resolution back

    We can now score a per-pixel prediction and train it with a per-pixel loss; the missing piece is a
    network whose output has the resolution of the input.
    A classification backbone shrinks a 224 px image to a 7x7 grid (stride 32): perfect for
    "what", fatal for "where". The first fully convolutional segmenter (FCN, 2015) simply
    upsampled that 7x7 map 32x. See what that does to a mask:
    """),
    api("F.adaptive_avg_pool2d"),
    api("F.interpolate", "Tensor.numpy", brief=True),
    code("""
    t = torch.tensor(pet[None, None].astype(np.float32))
    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    axes[0].imshow(pet, cmap="gray"); axes[0].set_title(f"mask {pet.shape[0]}x{pet.shape[1]}")
    for ax, s in zip(axes[1:], [8, 16, 32]):
        small_t = F.adaptive_avg_pool2d(t, (pet.shape[0] // s, pet.shape[1] // s))
        up = F.interpolate(small_t, size=pet.shape, mode="bilinear", align_corners=False)[0, 0] > 0.5
        iou = (up.numpy() & pet).sum() / (up.numpy() | pet).sum()
        ax.imshow(up, cmap="gray"); ax.set_title(f"stride {s}: {small_t.shape[-2]}x{small_t.shape[-1]} -> up\\nIoU {iou:.3f}")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Watch the IoU in the titles and the thin parts (ears, paws, tail): at stride 32 the coarse grid still
    knows *what* is there, but not *where* its outline runs.

    Three ways to learn an upsampling, plus one way to avoid downsampling:

    | tool | idea | parameters | watch out |
    |---|---|---|---|
    | **transposed conv** | each input pixel *paints* a scaled copy of the kernel | yes | checkerboard when kernel % stride != 0 |
    | **bilinear (or nearest) + conv** | fixed interpolation, then a normal 3x3 conv | yes (the conv) | slightly blurrier, no artefacts |
    | **pixel shuffle** | predict $r^2$ channels at low res, rearrange into an $r \times$ bigger map | yes (the conv before) | needs careful init (ICNR) |
    | **dilated conv** | keep the resolution, spread the kernel taps | yes | gridding if all rates equal |

    ### Transposed convolution = scatter-add

    A normal strided convolution *gathers* a window into one output pixel. A transposed
    convolution does the reverse: every **input** pixel *scatters* `x[i, j] * kernel` into
    the output at position `(i * stride, j * stride)`, and overlapping contributions add up.
    Output size: $(n - 1)\,s + k - 2p$.
    """),
    code("""
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML

    xs = np.array([[1., 2., 0.], [0., 1., 3.], [2., 0., 1.]])
    ks = np.array([[1., 1., 1.], [1., 2., 1.], [1., 1., 1.]])
    stride = 2
    full = convops.transposed_conv2d(xs, ks, stride)
    fig, (a_in, a_k, a_out) = plt.subplots(1, 3, figsize=(10, 3.4), gridspec_kw=dict(width_ratios=[3, 3, 7]))
    a_in.imshow(xs, cmap="Blues", vmin=0, vmax=3); viz.annotate_grid(a_in, xs); a_in.set_title("input 3x3")
    a_k.imshow(ks, cmap="Oranges", vmin=0, vmax=2); viz.annotate_grid(a_k, ks); a_k.set_title("kernel 3x3")
    canvas = np.zeros_like(full)
    im = a_out.imshow(canvas, cmap="Greens", vmin=0, vmax=full.max())
    txt = [[a_out.text(j, i, "", ha="center", va="center", fontsize=8) for j in range(full.shape[1])] for i in range(full.shape[0])]
    r_in = Rectangle((-0.5, -0.5), 1, 1, fill=False, ec="#E45756", lw=2.5); a_in.add_patch(r_in)
    r_out = Rectangle((-0.5, -0.5), 3, 3, fill=False, ec="#E45756", lw=2.5); a_out.add_patch(r_out)
    for ax in (a_in, a_k, a_out): ax.set_xticks([]); ax.set_yticks([])

    def update(f):
        i, j = divmod(f, 3)
        canvas[i * stride:i * stride + 3, j * stride:j * stride + 3] += xs[i, j] * ks
        im.set_data(canvas)
        for a in range(full.shape[0]):
            for b in range(full.shape[1]):
                txt[a][b].set_text(f"{canvas[a, b]:.0f}" if canvas[a, b] else "")
        r_in.set_xy((j - 0.5, i - 0.5)); r_out.set_xy((j * stride - 0.5, i * stride - 0.5))
        a_out.set_title(f"output 7x7: add x[{i},{j}]={xs[i, j]:.0f} * kernel at ({i * stride},{j * stride})")
        return [im, r_in, r_out]

    anim = FuncAnimation(fig, update, frames=9, interval=900, blit=False)
    plt.close(fig)
    HTML(anim.to_jshtml())
    """),
    md(r"""
    ### Lab · transposed convolution
    **Predict first:** stride 2, a 3x3 kernel of ones and an input of ones. Is the output constant away
    from the border? If not, which values appear? Step through the stamps in the lab, hover the output
    pixels, then work through the "Try this" list (4-5 minutes).
    """),
    code("""
    labs.show("transposed-conv")
    """),
    md(r"""
    ### Exercise 5 · `transposed_conv2d`

    Single channel, no padding: allocate the `(H-1)*s + kh` by `(W-1)*s + kw` output and
    add `x[i, j] * k` into the window starting at `(i*s, j*s)` - the stamp you stepped through in
    the lab. Return the output as a float NumPy array.
    """),
    exercise(
        stub="""
        def transposed_conv2d(x, k, stride=1):
            '''x (H, W), kernel k (kh, kw) -> out ((H-1)*stride + kh, (W-1)*stride + kw), float:
            every input pixel adds x[i, j] * k into the output window that starts at (i*stride, j*stride).'''
            H, W = x.shape
            kh, kw = k.shape
            # 1. YOUR CODE: out, an array of zeros of the output size
            # 2. for every input pixel (row i, column j):
            for i in range(H):
                for j in range(W):
                    # YOUR CODE: top = i * stride and left = j * stride, where its stamp starts in out
                    # YOUR CODE: add x[i, j] * k into out[top:top + kh, left:left + kw]
                    raise NotImplementedError
            return out

        tests.check_transposed_conv(transposed_conv2d)
        """,
        solution="""
        def transposed_conv2d(x, k, stride=1):
            '''x (H, W), kernel k (kh, kw) -> out ((H-1)*stride + kh, (W-1)*stride + kw), float:
            every input pixel adds x[i, j] * k into the output window that starts at (i*stride, j*stride).'''
            H, W = x.shape
            kh, kw = k.shape
            # 1. out, zeros of the output size
            out = np.zeros(((H - 1) * stride + kh, (W - 1) * stride + kw))
            # 2. for every input pixel (row i, column j):
            for i in range(H):
                for j in range(W):
                    top = i * stride                            # where its stamp starts in out
                    left = j * stride
                    out[top:top + kh, left:left + kw] += x[i, j] * k   # overlapping stamps add up
            return out

        tests.check_transposed_conv(transposed_conv2d)
        """),
    md(r"""
    Your loop is exactly what `F.conv_transpose2d` computes. It is also a *normal*
    convolution of a **zero-stuffed** input (insert `s - 1` zeros between pixels, pad by
    `k - 1`) with the **flipped** kernel - the reason it was nicknamed "fractionally strided
    convolution" (and, wrongly, "deconvolution").
    """),
    api("F.conv_transpose2d", "np.pad"),
    api("np.allclose", brief=True),
    code("""
    out_torch = F.conv_transpose2d(torch.tensor(xs)[None, None], torch.tensor(ks)[None, None], stride=2)[0, 0].numpy()
    stuffed = np.zeros((5, 5)); stuffed[::2, ::2] = xs                       # zero-stuffing
    out_conv = convops.cross_correlate2d(stuffed, ks[::-1, ::-1], pad=2)      # normal conv, flipped kernel
    print("loop == F.conv_transpose2d:", np.allclose(transposed_conv2d(xs, ks, 2), out_torch),
          "| == conv on zero-stuffed input:", np.allclose(out_conv, out_torch))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    for ax, a, t in zip(axes, [xs, np.pad(stuffed, 2), out_conv], ["input 3x3", "zero-stuffed + padded 9x9", "3x3 conv of that = 7x7 output"]):
        ax.imshow(a, cmap="Greens"); viz.annotate_grid(ax, a, fontsize=7); ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### The checkerboard artefact

    With stride 2 and kernel 3, output pixels receive contributions from **1, 2 or 4**
    input pixels depending on their parity; with kernel 4 every interior pixel receives
    exactly 4. Uneven overlap = a periodic pattern baked into every upsampled map, and stacking
    layers makes it worse (Odena et al., Distill 2016). Count the overlaps with an all-ones
    input and kernel (the "stamps per output pixel" map of the lab):
    """),
    api("np.outer", "Tensor.expand"),
    api("np.ones", "Tensor.transpose", "Tensor.contiguous", "Tensor.item", "Tensor.permute", brief=True),
    code("""
    ones = np.ones((6, 6))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, k in zip(axes[:2], [3, 4]):
        o = convops.transposed_conv2d(ones, np.ones((k, k)), 2)
        ax.imshow(o, cmap="YlOrRd", vmin=0, vmax=6); viz.annotate_grid(ax, o, fontsize=6)
        ax.set_title(f"stride 2, kernel {k}: overlap count"); ax.axis("off")
    # a real image upsampled x4 with two stride-2 transposed convs and FIXED smoothing kernels
    small_img = torch.tensor(img[::8, ::8].transpose(2, 0, 1).copy())[None].float()
    kernels = {"k=3 box [1 1 1]": [1.0, 1.0, 1.0], "k=4 bilinear [.25 .75 .75 .25]": [0.25, 0.75, 0.75, 0.25]}
    for ax, (name, k1) in zip(axes[2:], kernels.items()):
        kk = torch.tensor(np.outer(k1, k1), dtype=torch.float32)
        up = small_img
        for _ in range(2):
            up = F.conv_transpose2d(up, kk.expand(3, 1, *kk.shape).contiguous(), stride=2, groups=3)
        up = up[0].permute(1, 2, 0).numpy()
        crop = up[100:160, 120:180]
        ax.imshow(np.clip(crop / up.mean() * small_img.mean().item(), 0, 1), interpolation="nearest")
        ax.set_title(f"Pet image x4 (crop), {name}", fontsize=9); ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    The same effect with **learned (random-init)** layers: three x2 stages applied to a
    *constant* input. A perfect upsampler would return a constant image.
    """),
    api("nn.ConvTranspose2d", "nn.Upsample", "nn.PixelShuffle", "Module.modules", "nn.init.uniform_", "nn.init.zeros_"),
    api("nn.Sequential", "nn.Conv2d", "torch.ones", "torch.no_grad", "Tensor.std", brief=True),
    code("""
    torch.manual_seed(0)
    def stack(kind):
        layers = []
        for _ in range(3):
            if kind == "convT k3 s2":   layers.append(nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1, output_padding=1))
            elif kind == "convT k4 s2": layers.append(nn.ConvTranspose2d(8, 8, 4, stride=2, padding=1))
            elif kind == "bilinear + conv3x3": layers += [nn.Upsample(scale_factor=2, mode="bilinear"), nn.Conv2d(8, 8, 3, padding=1, padding_mode="replicate")]
            else: layers += [nn.Conv2d(8, 32, 3, padding=1, padding_mode="replicate"), nn.PixelShuffle(2)]
        net = nn.Sequential(*layers)
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.uniform_(m.weight, 0, 1); nn.init.zeros_(m.bias)   # positive weights make overlap visible
        return net

    x_const = torch.ones(1, 8, 8, 8)
    kinds = ["convT k3 s2", "convT k4 s2", "bilinear + conv3x3", "conv + pixel shuffle"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, kind in zip(axes, kinds):
        with torch.no_grad():
            y = stack(kind)(x_const)[0, 0].numpy()
        c = y[8:-8, 8:-8]                                       # ignore the border
        ax.imshow(c / c.mean(), cmap="RdBu_r", vmin=0.8, vmax=1.2); ax.axis("off")
        ax.set_title(f"{kind}\\nstd/mean in the interior: {c.std() / c.mean():.3f}")
    plt.suptitle("constant input, three random-init x2 upsampling stages (64 x 64 output, interior shown;\\n"
                 "colour = value / mean on a fixed 0.8-1.2 scale: a perfect upsampler is uniformly white)", y=1.06)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Transposed conv with k=3 shows a strong grid; k=4 overlaps evenly but its four
    sub-kernels are different random numbers, so a faint ripple remains (and can grow
    during training); resize + conv is exactly flat. Pixel shuffle with independently
    initialised sub-kernels also leaves a periodic ripple at init - which is why it is used
    with *ICNR* initialisation (all $r^2$ sub-kernels start identical). Modern decoders: **bilinear + conv** (DeepLab,
    SegFormer, FPN) or **k=2, s=2 transposed conv** (U-Net: no overlap at all).

    ### Bilinear upsampling vs the library

    `convops.bilinear_upsample` is a 15-line reference; check it against `F.interpolate` and
    compare the two learned upsamplers on one feature map.
    """),
    code("""
    fmap = torch.tensor(img[::16, ::16].mean(-1), dtype=torch.float32)
    ref = F.interpolate(fmap[None, None], scale_factor=2, mode="bilinear", align_corners=False)[0, 0].numpy()
    print("convops.bilinear_upsample == F.interpolate(bilinear, align_corners=False):",
          np.allclose(convops.bilinear_upsample(fmap.numpy(), 2), ref, atol=1e-5))

    torch.manual_seed(1)
    up_t = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)
    up_b = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(1, 1, 3, padding=1))
    with torch.no_grad():
        outs = {"input (1/16 of the Pet image)": fmap.numpy(), "nn.ConvTranspose2d(k3, s2)": up_t(fmap[None, None])[0, 0].numpy(),
                "nn.Upsample(bilinear) + Conv2d": up_b(fmap[None, None])[0, 0].numpy()}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (k, v) in zip(axes, outs.items()):
        ax.imshow(v, cmap="gray"); ax.set_title(f"{k}\\n{v.shape[0]}x{v.shape[1]}", fontsize=9); ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Pixel shuffle (sub-pixel convolution)

    Predict $C \cdot r^2$ channels at low resolution, then **rearrange**: channel group
    $(c, a, b)$ becomes the pixel at offset $(a, b)$ inside each $r \times r$ output cell. No
    zeros are inserted and every output pixel is computed at low resolution, which makes it
    cheap (ESPCN super-resolution, many real-time decoders).
    """),
    api("np.floor"),
    api("np.stack", "np.full", "torch.randn", brief=True),
    code("""
    r = 2
    xps = np.stack([np.full((3, 3), c + 1.0) for c in range(4)])     # 4 channels = 1 output channel x r^2
    xps += np.arange(9).reshape(3, 3) / 10                           # tag each position: channel.position
    fig, axes = plt.subplots(1, 5, figsize=(14, 3), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 2]))
    cols = ["Blues", "Oranges", "Greens", "Reds"]
    for c in range(4):
        axes[c].imshow(np.ones((3, 3)), cmap=cols[c], vmin=0, vmax=2); viz.annotate_grid(axes[c], xps[c], fmt="{:.1f}", fontsize=8)
        axes[c].set_title(f"channel {c} = offset {divmod(c, 2)}", fontsize=9); axes[c].set_xticks([]); axes[c].set_yticks([])
    ps = seg.pixel_shuffle(xps, r)[0]
    rgb = np.zeros(ps.shape + (3,))
    for c, cm_ in enumerate(cols):
        rgb[np.floor(ps).astype(int) - 1 == c] = plt.get_cmap(cm_)(0.5)[:3]
    axes[4].imshow(rgb); viz.annotate_grid(axes[4], ps, fmt="{:.1f}", fontsize=8)
    axes[4].set_title("pixel shuffle -> 1 x 6 x 6"); axes[4].set_xticks([]); axes[4].set_yticks([])
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Exercise 6 · `pixel_shuffle`

    `(C*r*r, H, W) -> (C, H*r, W*r)`: one `reshape`, one `transpose`, one `reshape`. The
    order of the axes in the transpose is the whole exercise.
    """),
    api("Tensor.transpose", "Tensor.reshape", brief=True, title="Functions you will need (met before)"),
    exercise(
        stub="""
        def pixel_shuffle(x, r):
            '''x (C*r*r, H, W) NumPy array -> (C, H*r, W*r): channel (c, a, b) becomes the pixel at offset (a, b)
            inside every r x r output cell.'''
            C_r2, H, W = x.shape
            C = C_r2 // (r * r)
            # 1. YOUR CODE: split the channel axis into (c, a, b): x.reshape(C, r, r, H, W)
            # 2. YOUR CODE: transpose the axes to the order (C, H, a, W, b), so that a sits next to H and b next to W
            # 3. YOUR CODE: merge (H, a) into H*r and (W, b) into W*r with one reshape; return the result
            raise NotImplementedError

        tests.check_pixel_shuffle(pixel_shuffle)
        """,
        solution="""
        def pixel_shuffle(x, r):
            '''x (C*r*r, H, W) NumPy array -> (C, H*r, W*r): channel (c, a, b) becomes the pixel at offset (a, b)
            inside every r x r output cell.'''
            C_r2, H, W = x.shape
            C = C_r2 // (r * r)
            # 1. split the channel axis into (c, a, b)
            x = x.reshape(C, r, r, H, W)          # axes: c, a, b, H, W
            # 2. a next to H, b next to W
            x = x.transpose(0, 3, 1, 4, 2)        # axes: c, H, a, W, b
            # 3. merge (H, a) and (W, b)
            return x.reshape(C, H * r, W * r)

        tests.check_pixel_shuffle(pixel_shuffle)
        # the library layer gives the same numbers
        xt = torch.randn(1, 12, 5, 5)
        ours = pixel_shuffle(xt[0].numpy(), 2)                  # (3, 10, 10)
        lib = nn.PixelShuffle(2)(xt)[0].numpy()                 # (3, 10, 10)
        print("matches nn.PixelShuffle:", np.allclose(ours, lib))
        """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · Context without losing resolution: dilated convolution and ASPP

    A **dilated** (atrous, "with holes") 3x3 convolution with rate $d$ samples the input at
    offsets $\{-d, 0, d\}^2$: same 9 weights, receptive field $2d + 1$. Stacking rates
    $1, 2, 4$ grows the receptive field *exponentially* while the feature map keeps its
    resolution - DeepLab removes the last downsampling stages of a ResNet and dilates the
    rest instead.
    """),
    md(r"""
    ### Exercise 7 · `dilate_kernel`

    Build the equivalent dense kernel: insert `d - 1` zeros between the taps. Size
    $d(k - 1) + 1$.
    """),
    exercise(
        stub="""
        def dilate_kernel(k, dilation):
            '''k (kh, kw) -> the equivalent dense kernel (dilation*(kh-1) + 1, dilation*(kw-1) + 1),
            with dilation - 1 zeros between the taps.'''
            kh, kw = k.shape
            # 1. YOUR CODE: out_h and out_w, the size of the dense kernel: dilation * (kh - 1) + 1 and the same for kw
            # 2. YOUR CODE: out, zeros of that size, with the same dtype as k
            # 3. YOUR CODE: put the taps at every dilation-th row and column: out[::dilation, ::dilation] = k
            # 4. YOUR CODE: return out
            raise NotImplementedError

        tests.check_dilate_kernel(dilate_kernel)
        """,
        solution="""
        def dilate_kernel(k, dilation):
            '''k (kh, kw) -> the equivalent dense kernel (dilation*(kh-1) + 1, dilation*(kw-1) + 1),
            with dilation - 1 zeros between the taps.'''
            kh, kw = k.shape
            # 1. the size of the dense kernel
            out_h = dilation * (kh - 1) + 1
            out_w = dilation * (kw - 1) + 1
            # 2. zeros of that size
            out = np.zeros((out_h, out_w), dtype=k.dtype)
            # 3. the taps at every dilation-th row and column
            out[::dilation, ::dilation] = k
            # 4. the result
            return out

        tests.check_dilate_kernel(dilate_kernel)
        """),
    md(r"""
    Which input pixels can influence the centre output after three 3x3 layers? Ask autograd:
    the gradient of the centre output w.r.t. the input is non-zero exactly on the receptive
    field. Equal rates (2, 2, 2) leave **holes** (the "gridding" problem), which is why
    DeepLabv3 and HDC (hybrid dilated convolution, Wang et al. 2018) use rates with no common factor.
    """),
    api("np.nonzero"),
    api("nn.init.constant_", brief=True),
    code("""
    def rf_map(dilations, n=33):
        layers = []
        for d in dilations:
            c = nn.Conv2d(1, 1, 3, padding=d, dilation=d, bias=False); nn.init.constant_(c.weight, 1.0); layers.append(c)
        x = torch.zeros(1, 1, n, n, requires_grad=True)
        nn.Sequential(*layers)(x)[0, 0, n // 2, n // 2].backward()
        return x.grad[0, 0].numpy()

    configs = [(1, 1, 1), (1, 2, 4), (2, 2, 2), (1, 2, 4, 8)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, d in zip(axes, configs):
        g = rf_map(d)
        ys_, xs_ = np.nonzero(g)
        ax.imshow(np.log1p(g), cmap="magma"); ax.set_xticks([]); ax.set_yticks([])
        holes = (g[ys_.min():ys_.max() + 1, xs_.min():xs_.max() + 1] == 0).sum()
        ax.set_title(f"dilations {d}\\nRF {ys_.max() - ys_.min() + 1}x{xs_.max() - xs_.min() + 1}, holes: {holes}")
    plt.suptitle("receptive field of the centre pixel (colour = number of paths, log)", y=1.02)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    **ASPP** (Atrous Spatial Pyramid Pooling, DeepLabv3) runs several dilated 3x3 convs in
    *parallel* on the same feature map (rates 6, 12, 18 at output stride 16, i.e. a feature map
    16x smaller than the input), plus a 1x1
    conv and a global-average-pooled branch, and concatenates them: multi-scale context in
    one layer. Each branch samples the map like this:
    """),
    api("nn.ModuleList", "torch.cat", brief=True),
    code("""
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    n = 45; c = n // 2
    ax.imshow(np.zeros((n, n)), cmap="Greys", vmin=0, vmax=1)
    ax.add_patch(Rectangle((-0.5, -0.5), n, n, fc="#f3f3f3", ec="none"))
    for rate, col in [(1, "#4C78A8"), (6, "#F58518"), (12, "#54A24B"), (18, "#E45756")]:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ax.add_patch(Rectangle((c + dx * rate - 0.5, c + dy * rate - 0.5), 1, 1, fc=col, ec="k", lw=0.4))
        ax.plot([], [], "s", color=col, label=f"3x3, rate {rate}")
    ax.plot([], [], "s", color="#BAB0AC", label="+ 1x1 conv + image pooling")
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.01, 1)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("ASPP: four branches sample the same feature map at four scales")
    plt.tight_layout(); plt.show()
    aspp = nn.ModuleList()                                  # four parallel branches, one per rate
    for rate in (1, 6, 12, 18):
        aspp.append(nn.Conv2d(256, 64, 3, padding=rate, dilation=rate))   # padding = rate keeps the size
    f = torch.randn(1, 256, 32, 32)
    branch_outputs = []
    branch_shapes = []
    for branch in aspp:
        branch_out = branch(f)                              # (1, 64, 32, 32) for every rate
        branch_outputs.append(branch_out)
        branch_shapes.append(tuple(branch_out.shape))
    concat = torch.cat(branch_outputs, 1)                   # (1, 4 * 64, 32, 32): all scales side by side
    print("ASPP branch outputs:", branch_shapes, "-> concat:", tuple(concat.shape))
    """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · The ancestor: a convolutional autoencoder

    An **encoder-decoder** compresses the image into a small bottleneck and reconstructs it.
    Replace "reconstruct the image" by "predict the mask" and you have a segmentation network.
    We train a 3-level conv autoencoder on FashionMNIST for one epoch
    (28 -> 14 -> 7 -> **4x4x8 bottleneck** -> 7 -> 14 -> 28, transposed convs on the way up),
    and an identical one with **one skip connection** that hands the decoder the 14x14
    encoder features.
    """),
    api("F.mse_loss", "scipy.ndimage.sobel", "np.hypot", "np.corrcoef"),
    api("torchkit.get_fashion_mnist", "torch.utils.data.DataLoader", "torch.optim.AdamW", "Module.parameters", "Tensor.to",
        "F.binary_cross_entropy_with_logits", "Optimizer.zero_grad", "Optimizer.step", "Module.eval", "torch.sigmoid",
        "Tensor.cpu", "Tensor.ravel", brief=True),
    code("""
    train_fm, test_fm, fm_classes = torchkit.get_fashion_mnist(normalize=False, as_tensors=True)
    sub = train_fm                                                  # one epoch = 60k images (capped at 60 s below)
    loader = torch.utils.data.DataLoader(sub, batch_size=128, shuffle=True)
    x_test = test_fm.tensors[0][:512]

    aes = {}                                                        # name -> trained autoencoder
    curves = {}                                                     # name -> loss of every step
    for name, skip in [("no skip", False), ("one skip (14x14)", True)]:
        torch.manual_seed(0)
        ae = day3.ConvAutoencoder(bottleneck_ch=8, skip=skip).to(device)
        opt = torch.optim.AdamW(ae.parameters(), lr=3e-3)
        t0 = time.time()
        losses = []
        for xb, _ in loader:
            xb = xb.to(device)
            # the target is the input itself: reconstruct every pixel
            loss = F.binary_cross_entropy_with_logits(ae(xb), xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if time.time() - t0 > 60:                                # wall-clock cap for slow machines
                break
        aes[name] = ae.eval()
        curves[name] = losses
        with torch.no_grad():
            mse = F.mse_loss(torch.sigmoid(ae(x_test.to(device))), x_test.to(device)).item()
        print(f"{name:18s} {len(losses)} steps in {time.time() - t0:4.0f}s   test MSE {mse:.4f}")
    """),
    code("""
    from scipy.ndimage import sobel
    idx = [0, 1, 2, 5, 13, 16]
    rec = {}                                                        # name -> reconstructions of the 6 images
    with torch.no_grad():
        for name, ae in aes.items():
            logits = ae(x_test[idx].to(device))                     # (6, 1, 28, 28)
            rec[name] = torch.sigmoid(logits).cpu()
    fig, axes = plt.subplots(4, len(idx), figsize=(1.9 * len(idx), 8.2))
    for j, i in enumerate(idx):
        axes[0, j].imshow(x_test[i, 0], cmap="gray"); axes[0, j].set_title(fm_classes[int(test_fm.tensors[1][i])], fontsize=8)
        axes[1, j].imshow(rec["no skip"][j, 0], cmap="gray", vmin=0, vmax=1)
        err = (rec["no skip"][j, 0] - x_test[i, 0]).abs().numpy()
        axes[2, j].imshow(err, cmap="inferno", vmin=0, vmax=0.6)
        axes[3, j].imshow(rec["one skip (14x14)"][j, 0], cmap="gray", vmin=0, vmax=1)
    for r_, lab in enumerate(["input", "no skip", "|error| (no skip)", "with skip"]):
        axes[r_, 0].set_ylabel(lab, fontsize=10)
    for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.show()

    with torch.no_grad():
        err_all = (torch.sigmoid(aes["no skip"](x_test.to(device))).cpu() - x_test).abs()[:, 0].numpy()
    edge_maps = []                                                  # edge strength of every test image
    for im in x_test[:, 0].numpy():                                 # im: (28, 28)
        edge_maps.append(np.hypot(sobel(im, 0), sobel(im, 1)))      # length of the image gradient
    edges = np.stack(edge_maps)                                     # (512, 28, 28)
    print(f"correlation between reconstruction error and edge strength: {np.corrcoef(err_all.ravel(), edges.ravel())[0, 1]:.2f}")
    """),
    md(r"""
    The errors of the plain autoencoder sit on **edges, stripes and logos**: exactly the
    "where" information (precise positions of boundaries) that does not fit through a 4x4
    bottleneck. The bottleneck keeps *what* the object is; it forgets *exactly where* its
    outline runs. For segmentation the boundary is the whole job.

    The skip connection hands the decoder the high-resolution encoder features directly, and
    the reconstruction becomes sharp. (For an autoencoder that is cheating - it can copy
    the input - but for segmentation it is precisely what we want: **semantics from the
    bottleneck, location from the skips**.) That is the U-Net, next notebook.
    """),

    # ------------------------------------------------------------------ 7
    md(r"""
    ## 7 · The Pet trimap target

    Oxford-IIIT Pet trimaps use `1 = pet`, `2 = background`, `3 = border/uncertain`. We remap
    to the convention "0 is background": `0 background, 1 pet, 2 border`
    (`torchkit.get_pet_datasets(target="segmentation")` does this for you). Two common
    choices for the *border* class: keep it as a third class (what we do, it is the
    official benchmark), or **ignore** it in the loss (`ignore_index=2` in
    `F.cross_entropy`) because it is genuinely ambiguous.
    """),
    api("torchkit.get_pet_datasets", "torchkit.denormalize", "Tensor.unique", "Tensor.tolist", "np.bincount", "np.median",
        "np.round", brief=True),
    code("""
    pet_tr, pet_va, pet_classes = torchkit.get_pet_datasets(size=128, target="segmentation", subset=(240, 60))
    print(len(pet_tr), "train /", len(pet_va), "val images; classes:", pet_classes)
    xb, yb = next(iter(torch.utils.data.DataLoader(pet_tr, batch_size=8, shuffle=True)))
    print("batch:", tuple(xb.shape), xb.dtype, "| masks:", tuple(yb.shape), yb.dtype, "values", yb.unique().tolist())

    fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
    for j in range(8):
        axes[0, j].imshow(torchkit.denormalize(xb[j])); axes[0, j].axis("off")
        axes[1, j].imshow(torchkit.denormalize(xb[j])); axes[1, j].imshow(day3.colorize(yb[j]), alpha=0.55); axes[1, j].axis("off")
    fig.legend(handles=[Patch(color=day3.SEG_COLORS[i], label=c) for i, c in enumerate(pet_classes)],
               loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.06))
    plt.tight_layout(); plt.show()
    """),
    code("""
    counts = np.zeros(3)
    for _, m in torch.utils.data.DataLoader(pet_tr, batch_size=60):
        counts += np.bincount(m.numpy().ravel(), minlength=3)
    freq = counts / counts.sum()
    w_inv = 1 / freq; w_inv /= w_inv.mean()
    w_med = np.median(freq) / freq                                   # "median frequency balancing" (SegNet)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].bar(pet_classes, freq, color=day3.SEG_COLORS[:3]); axes[0].set_title("pixel frequency per class (240 train images)")
    axes[0].set_ylim(0, freq.max() * 1.18)
    for i, v in enumerate(freq): axes[0].text(i, v + 0.01, f"{v:.1%}", ha="center")
    xw = np.arange(3)
    axes[1].bar(xw - 0.2, w_inv, 0.4, label="inverse frequency (mean 1)"); axes[1].bar(xw + 0.2, w_med, 0.4, label="median frequency")
    axes[1].set_xticks(xw); axes[1].set_xticklabels(pet_classes); axes[1].legend(fontsize=8); axes[1].set_title("class weights for the loss")
    plt.tight_layout(); plt.show()
    print("loss with class weights:  F.cross_entropy(logits, target, weight=torch.tensor(", np.round(w_med, 2).tolist(), "))")
    print("pixel accuracy of 'always predict the most frequent class':", f"{freq.max():.1%}")
    """),
    md(r"""
    Background and pet are roughly balanced here, but the **border** class is a thin band:
    a model that never predicts "border" still scores high pixel accuracy. Median-frequency
    weights push the border class up several times. Keep this picture in mind when you read
    the per-class IoU of the U-Net in D3-02: the border IoU is always the lowest.

    ## 8 · Summary

    * Semantic = class per pixel; instance = class + id for countable *things*; panoptic = both, every pixel.
    * Masks are stored as dense PNGs, polygons or RLE (column-major runs starting with zeros).
    * Never report pixel accuracy alone: use per-class IoU and mIoU from the confusion matrix; Dice = 2 IoU / (1 + IoU).
    * Mean BCE lets the background drown small objects; Dice (or BCE + Dice, or focal) rebalances the gradient.
    * Resolution comes back with transposed convs (scatter-add; keep kernel divisible by stride), bilinear + conv, or pixel shuffle; dilation and ASPP add context without downsampling.
    * An encoder-decoder without skips keeps *what* but loses *where*; skip connections put it back.

    ### Self-check (answers in the Day 3 quiz)
    1. A model predicts "background" everywhere on an image where the object covers 5% of the pixels. Pixel accuracy? IoU of the object? mIoU over the two classes?
    2. IoU = 0.6. What is the Dice score?
    3. Output size of `ConvTranspose2d(k=4, s=2, p=1)` on a 16x16 input? Does it produce a checkerboard from overlap?
    4. Receptive field of three stacked 3x3 convs with dilations 1, 2, 4?
    5. Why does a plain autoencoder blur edges, and what does a skip connection change?

    ### Go deeper
    * Long, Shelhamer & Darrell, *Fully Convolutional Networks for Semantic Segmentation* (CVPR 2015).
    * Odena, Dumoulin & Olah, *Deconvolution and Checkerboard Artifacts* (Distill, 2016).
    * Chen et al., *DeepLab* (2017) and *Rethinking Atrous Convolution* (DeepLabv3, ASPP).
    * Shi et al., *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel CNN* (pixel shuffle, 2016).
    * Milletari, Navab & Ahmadi, *V-Net* (2016): the Dice loss.
    * COCO data format and `pycocotools` mask API (RLE).
    """),
]
