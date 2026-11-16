from nb import md, code, exercise, header, api

NAME = "D2-03-one-stage"

SETUP = r"""
import sys, pathlib, time
for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:   # find the cvcourse package
    if (p / "cvcourse").is_dir():
        sys.path.insert(0, str(p)); break

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
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
        "D2-03 · One-stage detectors: dense grids, SSD, FPN, RetinaNet, focal loss, anchor-free",
        steps=[
            "Dense prediction: one output vector per grid cell (the YOLOv1 encoding)",
            "Build and train a mini detector end to end on shapes (custom loop, 3-term loss, decode + NMS, mAP)",
            "Multi-scale: SSD default boxes and the Feature Pyramid Network",
            "RetinaNet: shared heads over the pyramid",
            "Class imbalance and focal loss, derived and plotted",
            "Anchor-free heads: FCOS, CenterNet, the modern YOLO head",
            "Library RetinaNet and FCOS with COCO weights on Pet images",
        ],
        topics=[
            "Write the target tensor of a dense detector and the loss that trains it",
            "Train a working detector from scratch in about two minutes on a CPU",
            "Explain why detectors use feature pyramids and draw an FPN with its tensor shapes",
            "Derive focal loss from cross-entropy and say what gamma and alpha do",
            "Compare anchor-based and anchor-free heads",
        ],
        minutes=90,
        prereq="D2-01 (IoU, NMS, mAP) and D2-02 (anchors, matching, encoding).",
    ),
    md(r"""
    ## 0 · Setup
    """),
    api("np.set_printoptions", "torch.manual_seed", "torchkit.get_device", brief=True),
    code(SETUP),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · Dense prediction on a grid

    The fully-convolutional trick of D2-01 gave us a *grid of class scores*. A one-stage detector
    makes every grid cell predict the **whole detection**: is there an object here, where exactly
    is its box, and which class is it. With $A$ anchors (or "slots") per cell and $C$ classes the
    head is one convolution with

    $$A \times (\underbrace{1}_{\text{objectness}} + \underbrace{4}_{\text{box}} + \underbrace{C}_{\text{classes}}) \text{ output channels.}$$

    No proposals, no RoIAlign, no second stage - one forward pass, then decoding and NMS.

    **YOLOv1's assignment rule** (2016): the cell that contains the **centre** of an object is
    *responsible* for it; all other cells must predict "no object". We use $A=1$, an 8 x 8 grid
    on 128 px shapes images (cell = 16 px), and store each box as $(c_x, c_y, w, h)$ normalised by
    the image size.
    """),
    md(r"""
    ### Lab · Grid assignment
    **Predict first:** two small objects whose box centres fall into the same cell. What does the
    target tensor hold for that cell: both boxes, their average, or one of them? And which class?
    Drag the shapes to check, then work through the "Try this" list under the lab (5 minutes).
    """),
    code("""
    labs.show("grid-assignment")
    """),
    md(r"""
    ### Exercise 1 · the dense target tensor

    `grid_targets(boxes, labels, grid, size, num_classes)` -> a float32 array `(grid, grid, 5 + C)`
    indexed `[row, col, channel]` (channels last; the training code permutes it to channels-first
    later). `boxes` is `(N, 4)` xyxy in pixels, `labels` holds class indices `0 .. C-1`.
    Channel 0 objectness, 1-4 normalised `(cx, cy, w, h)`, then a one-hot class.
    The responsible cell is `(row, col) = (int(cy // cell), int(cx // cell))`.
    These are exactly the 8 numbers per object in the table under the lab.
    """),
    api("np.zeros", brief=True),
    exercise(
        stub=r"""
        def grid_targets(bxs, labels, grid=8, size=128, num_classes=3):
            '''bxs (N, 4) xyxy in pixels, labels (N,) class indices -> T (grid, grid, 5 + num_classes) float32,
            indexed [row, col, channel]: 0 objectness, 1-4 (cx, cy, w, h) / size, then a one-hot class.'''
            T = np.zeros((grid, grid, 5 + num_classes), np.float32)
            cell = size / grid                               # side of one cell in pixels (16 for 128 / 8)
            for box, label in zip(bxs, labels):
                # 1. the centre of the box, in pixels
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                # 2. YOUR CODE: the responsible cell: row i = int(cy // cell), column j = int(cx // cell)
                # 3. YOUR CODE: objectness: channel 0 of that cell is 1
                # 4. YOUR CODE: w and h of the box; channels 1-4 of the cell get [cx, cy, w, h], each divided by size
                # 5. YOUR CODE: the one-hot class: channel 5 + int(label) of the cell is 1
                raise NotImplementedError
            return T

        tests.check_grid_targets(grid_targets)
        """,
        solution=r"""
        def grid_targets(bxs, labels, grid=8, size=128, num_classes=3):
            '''bxs (N, 4) xyxy in pixels, labels (N,) class indices -> T (grid, grid, 5 + num_classes) float32,
            indexed [row, col, channel]: 0 objectness, 1-4 (cx, cy, w, h) / size, then a one-hot class.'''
            T = np.zeros((grid, grid, 5 + num_classes), np.float32)
            cell = size / grid                               # side of one cell in pixels (16 for 128 / 8)
            for box, label in zip(bxs, labels):
                # 1. the centre of the box, in pixels
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                # 2. the responsible cell
                i = int(cy // cell)                          # row
                j = int(cx // cell)                          # column
                # 3. objectness
                T[i, j, 0] = 1
                # 4. the box, normalised by the image size
                w = box[2] - box[0]
                h = box[3] - box[1]
                T[i, j, 1:5] = [cx / size, cy / size, w / size, h / size]
                # 5. the one-hot class
                T[i, j, 5 + int(label)] = 1
            return T

        tests.check_grid_targets(grid_targets)
        """),
    api("np.ndenumerate"),
    api("np.random.default_rng", brief=True),
    code(r"""
    g_img, g_boxes, g_labels, _ = shapes.make_sample(np.random.default_rng(7), 128)
    T = shapes.grid_targets(g_boxes, g_labels, 8, 128, 3)
    fig = plt.figure(figsize=(16, 5.2))
    ax = fig.add_axes([0.0, 0.05, 0.33, 0.85]); ax.imshow(g_img)
    for k in range(9):
        ax.axhline(k * 16 - 0.5, color="k", lw=0.5, alpha=0.5); ax.axvline(k * 16 - 0.5, color="k", lw=0.5, alpha=0.5)
    for b, l in zip(g_boxes, g_labels):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; i, j = int(cy // 16), int(cx // 16)
        ax.add_patch(Rectangle((j * 16 - 0.5, i * 16 - 0.5), 16, 16, fc="yellow", alpha=0.45, ec="k", lw=2))
        ax.plot(cx, cy, "k+", ms=12, mew=2)
        ax.add_patch(Rectangle(b[:2], *(b[2:] - b[:2]), fill=False, ec="k", lw=1.2, ls="--"))
    ax.set_title("8 x 8 grid: the cell holding each centre (+) is responsible"); ax.axis("off")
    names = ["objectness", "cx", "cy", "w", "h"] + [f"class: {n}" for n in shapes.CLASS_NAMES]
    for c in range(8):
        a = fig.add_axes([0.35 + (c % 4) * 0.165, 0.5 - (c // 4) * 0.47, 0.15, 0.4])
        a.imshow(T[..., c], cmap="viridis" if 1 <= c <= 4 else "Greys", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(T[..., c]):
            if T[i, j, 0] > 0:
                a.text(j, i, f"{v:.2f}" if 1 <= c <= 4 else f"{v:.0f}", ha="center", va="center", fontsize=7, color="red")
        a.set_title(f"channel {c}: {names[c]}", fontsize=9); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("The target tensor (8, 8, 5 + 3): only responsible cells carry a box and a class", y=1.0)
    plt.show()
    """),
    md(r"""
    What to look at: every channel is zero except in the four responsible (yellow) cells, and the box
    channels hold numbers relative to the *whole image*: the circle on the right has `cx = 0.84`
    whatever cell it sits in. The other 60 cells only teach "no object".
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · A mini detector, trained end to end

    `day2.MiniDetector`: four stride-2 stages (128 -> 64 -> 32 -> 16 -> 8, conv-BN-ReLU) and a
    1 x 1 head with 5 + 3 channels. To make the box easy to learn, the model **decodes inside
    `forward`**: for the cell in row $i$, column $j$ of a $G \times G$ grid

    $$c_x = \frac{j + \sigma(t_x)}{G}, \quad c_y = \frac{i + \sigma(t_y)}{G}, \quad w = \sigma(t_w), \quad h = \sigma(t_h),$$

    so a cell can only place a centre *inside itself* (exactly YOLO's parametrisation) and the
    output already lives in the same layout as the target. Channel 0 and the class channels stay
    logits for numerically stable losses.
    """),
    api("torchkit.summarize", brief=True),
    code(r"""
    print(torchkit.summarize(day2.MiniDetector(), (3, 128, 128)))
    """),
    md(r"""
    ### Exercise 2 · the three-term detection loss

    `pred` and `target` are `(N, 5 + C, G, G)`. With `pos = target[:, 0] > 0.5` (responsible cells):

    * **objectness**: `F.binary_cross_entropy_with_logits` over **all** cells (mean) - every cell must learn yes / no;
    * **box**: L1 between `pred[:, 1:5]` and `target[:, 1:5]`, summed over the 4 numbers, averaged over **positive cells only**;
    * **class**: `F.cross_entropy` on the class logits of **positive cells only** (target = argmax of the one-hot);
    * total = `1 * obj + 5 * box + 1 * cls`. Return `(total, {"obj": ..., "box": ..., "cls": ...})`.

    Tip: `pred[:, 1:5].permute(0, 2, 3, 1)[pos]` gives a `(P, 4)` tensor of the positive cells.
    Guard the case "no object in the batch" (box and class loss = 0, not NaN).
    """),
    api("F.binary_cross_entropy_with_logits", title="Functions you will need"),
    api("F.cross_entropy", "Tensor.permute", "Tensor.any", "Tensor.abs", "Tensor.sum", "Tensor.mean", "Tensor.argmax",
        brief=True),
    exercise(
        stub=r"""
        def dense_loss(pred, target, w_obj=1.0, w_box=5.0, w_cls=1.0):
            '''pred, target: (N, 5 + C, G, G); channel 0 objectness, 1-4 the box, 5.. the classes
            (pred holds logits for objectness and classes, target a 0/1 objectness and a one-hot class)
            -> (total, {"obj": l_obj, "box": l_box, "cls": l_cls}).'''
            # 1. the responsible cells
            pos = target[:, 0] > 0.5                                  # (N, G, G) True / False
            # 2. YOUR CODE: l_obj, F.binary_cross_entropy_with_logits of pred[:, 0] against target[:, 0] (all cells)
            if pos.any():
                # 3. YOUR CODE: pred_box, pred[:, 1:5] with the channels moved last (.permute(0, 2, 3, 1))
                #               -> (N, G, G, 4); then pred_box[pos] keeps the positive cells -> (P, 4)
                # YOUR CODE: target_box, the same for target[:, 1:5] -> (P, 4)
                # YOUR CODE: abs_err = |pred_box - target_box| -> (P, 4)
                # YOUR CODE: l1_per_cell, abs_err summed over the 4 numbers (.sum(-1)) -> (P,)
                # YOUR CODE: l_box, the mean of l1_per_cell over the positive cells
                # 4. YOUR CODE: logits, pred[:, 5:] channels last, positive cells only -> (P, C)
                # YOUR CODE: target_onehot, target[:, 5:] channels last, positive cells only -> (P, C)
                # YOUR CODE: target_class, the class index of every positive cell: target_onehot.argmax(-1) -> (P,)
                # YOUR CODE: l_cls = F.cross_entropy(logits, target_class)
                raise NotImplementedError
            else:
                # no object in the batch: box and class loss 0, not NaN (pred.sum() * 0.0 is a 0 that is still a tensor)
                l_box = pred.sum() * 0.0
                l_cls = pred.sum() * 0.0
            # 5. YOUR CODE: total = w_obj * l_obj + w_box * l_box + w_cls * l_cls
            raise NotImplementedError
            return total, {"obj": l_obj, "box": l_box, "cls": l_cls}

        tests.check_dense_loss(dense_loss)
        """,
        solution=r"""
        def dense_loss(pred, target, w_obj=1.0, w_box=5.0, w_cls=1.0):
            '''pred, target: (N, 5 + C, G, G); channel 0 objectness, 1-4 the box, 5.. the classes
            (pred holds logits for objectness and classes, target a 0/1 objectness and a one-hot class)
            -> (total, {"obj": l_obj, "box": l_box, "cls": l_cls}).'''
            # 1. the responsible cells
            pos = target[:, 0] > 0.5                                  # (N, G, G) True / False
            # 2. objectness: binary cross-entropy on every cell
            l_obj = F.binary_cross_entropy_with_logits(pred[:, 0], target[:, 0])
            if pos.any():
                # 3. box: channels last, so that [pos] picks one row of 4 numbers per positive cell
                pred_box = pred[:, 1:5].permute(0, 2, 3, 1)           # (N, G, G, 4)
                pred_box = pred_box[pos]                              # (P, 4), P = number of positive cells
                target_box = target[:, 1:5].permute(0, 2, 3, 1)       # (N, G, G, 4)
                target_box = target_box[pos]                          # (P, 4)
                abs_err = (pred_box - target_box).abs()               # (P, 4)
                l1_per_cell = abs_err.sum(-1)                         # (P,) summed over the 4 numbers
                l_box = l1_per_cell.mean()                            # mean over the positive cells
                # 4. class: cross-entropy on the positive cells
                logits = pred[:, 5:].permute(0, 2, 3, 1)              # (N, G, G, C)
                logits = logits[pos]                                  # (P, C)
                target_onehot = target[:, 5:].permute(0, 2, 3, 1)     # (N, G, G, C)
                target_onehot = target_onehot[pos]                    # (P, C)
                target_class = target_onehot.argmax(-1)               # (P,) class index of every positive cell
                l_cls = F.cross_entropy(logits, target_class)
            else:
                # no object in the batch: box and class loss 0, not NaN (pred.sum() * 0.0 is a 0 that is still a tensor)
                l_box = pred.sum() * 0.0
                l_cls = pred.sum() * 0.0
            # 5. the weighted sum
            total = w_obj * l_obj + w_box * l_box + w_cls * l_cls
            return total, {"obj": l_obj, "box": l_box, "cls": l_cls}

        tests.check_dense_loss(dense_loss)
        """),
    md(r"""
    ### Train (about 2 minutes on a laptop CPU)

    1500 training images, 6 epochs, AdamW with a one-cycle schedule, random horizontal flips
    (the flip must also flip the grid columns and $c_x$ - look at `day2.train_mini_detector`, it is
    20 lines). The loop uses **your** loss if it passed the checker. A time cap stops training
    after 3 minutes on a very slow machine (even 2 epochs reach mAP@0.5 > 0.95).

    > **CPU tip.** Tiny models like this one are dominated by thread-synchronisation overhead: on a
    > busy laptop all cores can be several times *slower* than 1 or 2 threads. The training loop
    > therefore first times one batch with a few thread counts and uses the fastest (first line of
    > the log). Benchmark, do not guess.
    """),
    code(r"""
    loss_fn = dense_loss if tests.check_dense_loss(dense_loss, verbose=False) else day2.dense_loss
    print("using", "your dense_loss" if loss_fn is dense_loss else "the reference day2.dense_loss")
    X_tr, B_tr, L_tr = day2.shapes_tensors(1500, 128, seed=0)
    T_tr = day2.grid_target_tensor(B_tr, L_tr)
    X_va, B_va, L_va = day2.shapes_tensors(300, 128, seed=1)
    print("train", tuple(X_tr.shape), "targets", tuple(T_tr.shape))

    torch.manual_seed(0)
    det = day2.MiniDetector()
    t0 = time.time()
    hist = day2.train_mini_detector(det, X_tr, T_tr, loss_fn=loss_fn, epochs=6, batch_size=32, lr=4e-3, max_seconds=180)
    train_seconds = time.time() - t0
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
    for ax, k, c in zip(axes, ["total", "obj", "box", "cls"], ["k", ORANGE, BLUE, GREEN]):
        ax.plot(range(1, len(hist[k]) + 1), hist[k], "o-", color=c); ax.set_title(f"{k} loss"); ax.set_xlabel("epoch"); ax.grid(alpha=.3)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: the class loss is close to 0 after two epochs (telling circles from squares is
    easy), while the box loss keeps falling until the last epoch: placing a box to the pixel is the
    hard part of detection.
    """),
    md(r"""
    ### Exercise 3 · decode the grid

    Inverse of Exercise 1 for a prediction `(G, G, 5 + C)` whose objectness and classes are already
    probabilities: every cell with objectness $\ge$ threshold becomes a box (scale `cx, cy, w, h` by
    `size`, convert to xyxy), its score is the objectness, its label the argmax of the class part.
    `day2.decode_predictions` then multiplies score by class probability and runs class-wise NMS.
    """),
    api("np.array", "Tensor.reshape", "np.argmax", "np.arange", "Tensor.round", brief=True),
    exercise(
        stub=r"""
        def decode_grid(T, size=128, score_threshold=0.5):
            '''T (G, G, 5 + C): channel 0 objectness, 1-4 (cx, cy, w, h) / size, 5.. class probabilities ->
            (boxes (K, 4) xyxy in pixels, scores (K,), labels (K,)): one box per cell with objectness >= score_threshold.'''
            bxs = []
            scores = []
            labels = []
            G = T.shape[0]
            for i in range(G):                               # row
                for j in range(G):                           # column
                    p = T[i, j]                              # (5 + C,) the numbers of this cell
                    # 1. YOUR CODE: skip this cell (continue) when its objectness p[0] is below score_threshold
                    # 2. YOUR CODE: cx, cy, w, h: p[1], p[2], p[3], p[4], each times size (normalised -> pixels)
                    # 3. YOUR CODE: append the xyxy box [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] to bxs
                    # 4. YOUR CODE: append the score float(p[0]) to scores
                    # 5. YOUR CODE: append the label, the index of the largest class probability
                    #               (int(np.argmax(p[5:]))), to labels
                    raise NotImplementedError
            bxs = np.array(bxs).reshape(-1, 4)               # (K, 4); (0, 4) when no cell passed
            return bxs, np.array(scores), np.array(labels, int)

        tests.check_decode_grid(decode_grid)
        """,
        solution=r"""
        def decode_grid(T, size=128, score_threshold=0.5):
            '''T (G, G, 5 + C): channel 0 objectness, 1-4 (cx, cy, w, h) / size, 5.. class probabilities ->
            (boxes (K, 4) xyxy in pixels, scores (K,), labels (K,)): one box per cell with objectness >= score_threshold.'''
            bxs = []
            scores = []
            labels = []
            G = T.shape[0]
            for i in range(G):                               # row
                for j in range(G):                           # column
                    p = T[i, j]                              # (5 + C,) the numbers of this cell
                    # 1. skip the cells without an object
                    if p[0] < score_threshold:
                        continue
                    # 2. the box in pixels
                    cx = p[1] * size
                    cy = p[2] * size
                    w = p[3] * size
                    h = p[4] * size
                    # 3. as xyxy
                    bxs.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                    # 4. the score
                    scores.append(float(p[0]))
                    # 5. the label: the class with the largest probability
                    labels.append(int(np.argmax(p[5:])))
            bxs = np.array(bxs).reshape(-1, 4)               # (K, 4); (0, 4) when no cell passed
            return bxs, np.array(scores), np.array(labels, int)

        tests.check_decode_grid(decode_grid)
        """),
    code(r"""
    t0 = time.time()
    P_va = day2.predict_batches(det, X_va)                          # (300, 8, 8, 8) raw outputs
    preds = day2.decode_predictions(P_va, 128, score_threshold=0.05, iou_threshold=0.5)
    gts = day2.gt_dicts(B_va, L_va)
    map50, per_class = boxes.mean_average_precision(preds, gts, 3, (0.5,))
    map_coco, _ = boxes.mean_average_precision(preds, gts, 3, tuple(np.arange(0.5, 0.96, 0.05)))
    print(f"mini detector on 300 validation images: mAP@0.5 = {map50:.3f}   mAP@[.5:.95] = {map_coco:.3f}")
    print("AP@0.5 per class:", dict(zip(shapes.CLASS_NAMES, per_class.round(3))), f"| trained {train_seconds:.0f} s, eval {time.time() - t0:.1f} s")

    probs = day2.grid_to_probs(P_va)
    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    for c, i in enumerate(range(6)):
        axes[0, c].imshow(X_va[i].permute(1, 2, 0)); axes[0, c].axis("off")
        viz.draw_boxes(axes[0, c], B_va[i], edgecolor="k", lw=2.5, alpha=0.4)
        k = preds[i]["scores"] > 0.3
        day2.draw_detections(axes[0, c], preds[i]["boxes"][k], preds[i]["labels"][k], preds[i]["scores"][k], shapes.CLASS_NAMES)
        axes[1, c].imshow(probs[i, ..., 0], cmap="magma", vmin=0, vmax=1); axes[1, c].set_xticks([]); axes[1, c].set_yticks([])
        axes[1, c].set_title("objectness (8 x 8)", fontsize=9)
    fig.suptitle("predictions (coloured, score > 0.3) vs ground truth (black) and the objectness grid", y=1.0)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    In the objectness grids (bottom row), some objects light up two neighbouring cells: their centre
    lies near a cell border, and two cells that see almost the same pixels are both fairly confident
    (the border jump of the lab's "Try this" 3). Class-wise NMS then keeps one of the two boxes.

    Expected: **mAP@0.5 above 0.95** after ~6 epochs (well above the 0.9 target), and mAP@[.5:.95]
    around 0.8: the grid detector finds everything, its boxes are good but not pixel-perfect.
    That is a complete detector: ~300 k parameters, one forward pass, decode, NMS.

    What breaks YOLOv1 on real images: **one object per cell** (two small birds in one cell - one is
    lost), **one scale** (a 16 px cell must handle a 16 px and a 500 px object), and objects whose
    centre falls near a cell border flip between cells. The rest of this notebook fixes these.
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · Multi-scale: SSD default boxes and the Feature Pyramid Network

    **SSD** (2016) attaches a detection head to **several feature maps** of decreasing resolution:
    fine maps (stride 8) handle small objects, coarse maps (stride 64+) large ones, each with a few
    "default boxes" (SSD's name for anchors) whose scale grows with depth.
    """),
    api("torchkit.get_pet_datasets", "np.asarray", "np.linspace", "np.sqrt", brief=True),
    code(r"""
    from PIL import Image
    pet_train, pet_val, _ = torchkit.get_pet_datasets(target="detection")
    ssd_img = np.asarray(Image.open(pet_val.image_path(12)).convert("RGB").resize((300, 300))) / 255.0
    maps = [38, 19, 10, 5, 3, 1]
    s_k = np.linspace(0.2, 0.9, 6)                                  # SSD scales (fraction of the image)
    fig, axes = plt.subplots(1, 6, figsize=(17, 3.4))
    for ax, f, s in zip(axes, maps, s_k):
        ax.imshow(ssd_img); step = 300 / f
        for k in range(f + 1):
            ax.axhline(k * step, color="w", lw=0.4, alpha=0.7); ax.axvline(k * step, color="w", lw=0.4, alpha=0.7)
        c = (f // 2 + 0.5) * step
        for r, col in zip([1, 2, 0.5], [RED, ORANGE, BLUE]):
            w, h = 300 * s / np.sqrt(r), 300 * s * np.sqrt(r)
            ax.add_patch(Rectangle((c - w / 2, c - h / 2), w, h, fill=False, ec=col, lw=1.8))
        ax.plot(c, c, "y*", ms=10, mec="k")
        ax.set_xlim(-60, 360); ax.set_ylim(360, -60); ax.axis("off")
        ax.set_title(f"{f} x {f} map, scale {s:.2f}", fontsize=10)
    fig.suptitle("SSD300: default boxes at one location of each of the 6 maps (8732 boxes in total)", y=1.02)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: from left to right the cells grow from 8 px to the whole image and the default
    boxes from 20% to 90% of the image, so each map is responsible for one band of object sizes.

    SSD's problem: the high-resolution maps come from **early** layers, which see edges, not
    objects - so small-object detection stayed weak. The **Feature Pyramid Network** (Lin et al.
    2017) gives every level both resolution *and* semantics:

    1. **bottom-up**: the backbone's stage outputs $C_3, C_4, C_5$ (strides 8, 16, 32);
    2. **lateral** 1 x 1 convs bring each to the same width $d$ (256);
    3. **top-down**: start at the coarsest, upsample x2 (nearest) and **add** to the next finer lateral;
    4. a 3 x 3 conv smooths each merged map (removes upsampling aliasing) -> $P_3, P_4, P_5$;
    5. RetinaNet adds $P_6, P_7$ with stride-2 convs for very large objects.

    Every level now carries the "what" of $C_5$ at its own resolution.
    """),
    code(r"""
    fig, ax = plt.subplots(figsize=(14, 6.4)); ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(-0.7, 6.2)
    C = [("C3", "128 x 32 x 32", 2.2), ("C4", "256 x 16 x 16", 1.6), ("C5", "512 x 8 x 8", 1.1)]
    P = [("P3", "256 x 32 x 32", 2.2), ("P4", "256 x 16 x 16", 1.6), ("P5", "256 x 8 x 8", 1.1)]
    ys = [0.4, 2.4, 4.2]
    ax.add_patch(FancyBboxPatch((1.2, -0.55), 1.4, 0.35, boxstyle="round,pad=0.02", fc="#eee", ec="k"))
    ax.text(2.8, -0.38, "input 3 x 256 x 256 (ResNet-18)", va="center", fontsize=9)
    ax.annotate("", (1.9, 0.4), (1.9, -0.18), arrowprops=dict(arrowstyle="-|>", lw=1.2))
    for (n, s, w), y in zip(C, ys):
        ax.add_patch(FancyBboxPatch((1.6 - w / 2 + 0.3, y), w, 1.2, boxstyle="round,pad=0.02", fc="#c6dbef", ec="k"))
        ax.text(1.9, y + 0.6, f"{n}\n{s}", ha="center", va="center", fontsize=9)
    for (n, s, w), y in zip(P, ys):
        ax.add_patch(FancyBboxPatch((8.0 - w / 2, y), w, 1.2, boxstyle="round,pad=0.02", fc="#c7e9c0", ec="k"))
        ax.text(8.0, y + 0.6, f"{n}\n{s}", ha="center", va="center", fontsize=9)
        ax.annotate("", (8.0 - w / 2 - 0.05, y + 0.6), (3.1, y + 0.6), arrowprops=dict(arrowstyle="-|>", lw=1.5))
        ax.text(3.2, y + 0.72, "1x1 conv (lateral) -> 256", fontsize=8)
        ax.add_patch(Circle((6.2, y + 0.6), 0.18, fc="white", ec="k")); ax.text(6.2, y + 0.6, "+", ha="center", va="center")
        ax.text(6.45, y + 0.72, "3x3 conv", fontsize=8)
    for y0, y1 in [(4.2, 2.4 + 1.2), (2.4, 0.4 + 1.2)]:
        ax.annotate("", (6.2, y1 - 0.42), (6.2, y0 + 0.42), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=RED))
        ax.text(6.35, (y0 + y1) / 2 - 0.1, "upsample x2", color=RED, fontsize=8)
    for y0, y1 in [(0.4 + 1.2, 2.4), (2.4 + 1.2, 4.2)]:
        ax.annotate("", (1.9, y1), (1.9, y0), arrowprops=dict(arrowstyle="-|>", lw=1.2))
    ax.text(1.9, 5.7, "bottom-up\n(backbone)", ha="center", fontsize=10, weight="bold")
    ax.text(6.2, 5.7, "top-down + lateral", ha="center", fontsize=10, weight="bold", color=RED)
    ax.text(8.0, 5.7, "pyramid", ha="center", fontsize=10, weight="bold", color=GREEN)
    ax.text(11.0, 3.1, "the SAME head runs on every P level\n(RetinaNet, FCOS, Faster R-CNN's RPN)", fontsize=10, va="center")
    plt.show()
    """),
    md(r"""
    ### Lab · FPN levels
    **Predict first:** a 16 x 16 px object in a 512 px image. What fraction of one $P_5$ cell (stride 32)
    does it cover, and which level will RetinaNet's anchors assign it to? Use the size presets to
    check, then open the "top-down pathway" view and predict the shapes before you write the
    `FPN` below (6 minutes).
    """),
    code("""
    labs.show("fpn-levels")
    """),
    md(r"""
    ### Exercise 4 · `FPN`

    `FPN(in_channels_list, out_channels)` with `nn.ModuleList`s `lateral` (1x1) and `smooth` (3x3,
    padding 1). `forward(feats)` takes the list `[C3, C4, C5]` (finest first) and returns
    `[P3, P4, P5]`. Upsample with `F.interpolate(top, size=lat.shape[-2:], mode="nearest")`.
    Every $P_i$ has `out_channels` channels and the height and width of its $C_i$: the shapes of the
    lab's pathway view.
    """),
    api("nn.ModuleList", title="Functions you will need"),
    api("nn.Conv2d", "F.interpolate", brief=True),
    exercise(
        stub=r"""
        class FPN(nn.Module):
            '''FPN(in_channels_list, d): forward([C3, C4, C5], finest first, C_i of shape (N, c_i, H_i, W_i))
            -> [P3, P4, P5]: P_i of shape (N, d, H_i, W_i).'''
            def __init__(self, in_channels_list, out_channels=256):
                super().__init__()
                # 1. one lateral 1x1 conv per level, c_i channels -> out_channels
                self.lateral = nn.ModuleList()                   # a list of layers that PyTorch knows about
                for c in in_channels_list:
                    # YOUR CODE: append nn.Conv2d(c, out_channels, 1) to self.lateral
                    raise NotImplementedError
                # 2. one smoothing 3x3 conv per level, out_channels -> out_channels (padding 1 keeps the size)
                self.smooth = nn.ModuleList()
                # YOUR CODE: a loop over the levels (for level in range(len(in_channels_list))) that appends
                #            one such conv per level to self.smooth
                raise NotImplementedError

            def forward(self, feats):
                # 3. YOUR CODE: lat, a list: the lateral conv of every level applied to its feature map,
                #               lat[i] = self.lateral[i](feats[i]) -> (N, d, H_i, W_i)
                outs = [None] * len(feats)                       # P3, P4, P5, filled from the coarsest level down
                # 4. YOUR CODE: the coarsest level: top = lat[-1]; outs[-1] = self.smooth[-1](top)
                # 5. top-down: from the second-coarsest level to the finest
                for i in range(len(feats) - 2, -1, -1):          # i = len - 2, ..., 1, 0
                    # YOUR CODE: upsampled, top enlarged to the size of lat[i]:
                    #            F.interpolate(top, size=lat[i].shape[-2:], mode="nearest") -> (N, d, H_i, W_i)
                    # YOUR CODE: top = lat[i] + upsampled (the merge)
                    # YOUR CODE: outs[i] = self.smooth[i](top)
                    raise NotImplementedError
                return outs

        tests.check_fpn_shapes(FPN)
        """,
        solution=r"""
        class FPN(nn.Module):
            '''FPN(in_channels_list, d): forward([C3, C4, C5], finest first, C_i of shape (N, c_i, H_i, W_i))
            -> [P3, P4, P5]: P_i of shape (N, d, H_i, W_i).'''
            def __init__(self, in_channels_list, out_channels=256):
                super().__init__()
                # 1. one lateral 1x1 conv per level, c_i channels -> out_channels
                self.lateral = nn.ModuleList()                   # a list of layers that PyTorch knows about
                for c in in_channels_list:
                    self.lateral.append(nn.Conv2d(c, out_channels, 1))
                # 2. one smoothing 3x3 conv per level, out_channels -> out_channels (padding 1 keeps the size)
                self.smooth = nn.ModuleList()
                for level in range(len(in_channels_list)):
                    self.smooth.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

            def forward(self, feats):
                # 3. lat: the lateral conv of every level applied to its feature map
                lat = []
                for i in range(len(feats)):
                    lat.append(self.lateral[i](feats[i]))        # (N, d, H_i, W_i)
                outs = [None] * len(feats)                       # P3, P4, P5, filled from the coarsest level down
                # 4. the coarsest level: only smoothed
                top = lat[-1]
                outs[-1] = self.smooth[-1](top)
                # 5. top-down: from the second-coarsest level to the finest
                for i in range(len(feats) - 2, -1, -1):          # i = len - 2, ..., 1, 0
                    upsampled = F.interpolate(top, size=lat[i].shape[-2:], mode="nearest")   # (N, d, H_i, W_i)
                    top = lat[i] + upsampled                     # the merge
                    outs[i] = self.smooth[i](top)                # (N, d, H_i, W_i)
                return outs

        tests.check_fpn_shapes(FPN)
        """),
    api("torchvision.models.feature_extraction.create_feature_extractor"),
    api("torchvision.models.resnet18", "Module.eval", "torch.no_grad", "torch.randn", "torchkit.count_parameters",
        brief=True),
    code(r"""
    # Your FPN on top of a pretrained ResNet-18: the shapes of the figure above
    from torchvision.models.feature_extraction import create_feature_extractor
    fpn_cls = FPN if tests.check_fpn_shapes(FPN, verbose=False) else day2.FPN
    r18 = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
    body = create_feature_extractor(r18, {"layer2": "C3", "layer3": "C4", "layer4": "C5"})
    fpn = fpn_cls([128, 256, 512], 256).eval()
    with torch.no_grad():
        cs = body(torch.randn(1, 3, 256, 256))
        ps = fpn([cs["C3"], cs["C4"], cs["C5"]])
    for (k, v), p_ in zip(cs.items(), ps):
        print(f"{k}: {tuple(v.shape)}  ->  P{k[1]}: {tuple(p_.shape)}")
    print(f"FPN parameters: {torchkit.count_parameters(fpn):,d}")
    """),
    md(r"""
    What do the pyramid levels *see*? Our FPN is untrained, so we look at the trained FPN inside
    torchvision's COCO RetinaNet. Below, the three main principal components of each level's 256
    channels are shown as RGB: $P_3$ has fine detail (fur, edges, eyes), $P_5$ is coarse but each
    of its cells summarises a whole head or body.
    """),
    api("torchvision.models.detection.retinanet_resnet50_fpn", "np.linalg.svd"),
    api("torch.tensor", "Tensor.numpy", "Tensor.min", "Tensor.max", brief=True),
    code(r"""
    from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
    retina = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT).eval()
    pet_img = np.asarray(Image.open(pet_val.image_path(30)).convert("RGB").resize((384, 384))) / 255.0
    xin = (torch.tensor(pet_img, dtype=torch.float32).permute(2, 0, 1) - torch.tensor(torchkit.IMAGENET_MEAN)[:, None, None]) / torch.tensor(torchkit.IMAGENET_STD)[:, None, None]
    with torch.no_grad():
        feats = retina.backbone(xin[None])                       # OrderedDict '0'..'p7' = P3..P7
    def pca_rgb(f):
        f = f[0].reshape(f.shape[1], -1).T.numpy(); f = f - f.mean(0)
        u, s_, vt = np.linalg.svd(f, full_matrices=False)
        z = f @ vt[:3].T; z = (z - z.min(0)) / (z.max(0) - z.min(0) + 1e-8)
        return z.reshape(int(np.sqrt(len(z))), -1, 3)
    names = ["P3", "P4", "P5", "P6", "P7"]
    fig, axes = plt.subplots(1, 6, figsize=(17, 3.3))
    axes[0].imshow(pet_img); axes[0].set_title("input 384 x 384")
    for ax, (k, f), n in zip(axes[1:], feats.items(), names):
        ax.imshow(pca_rgb(f), interpolation="nearest"); ax.set_title(f"{n}: {f.shape[-2]} x {f.shape[-1]} (stride {384 // f.shape[-1]})")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · RetinaNet: one head shared by all levels

    RetinaNet (Lin et al. 2017) = ResNet + FPN ($P_3$-$P_7$) + two small fully-convolutional heads
    that run **with the same weights on every level** (an object is an object at every scale):

    * classification tower: 4 x (3x3 conv 256 + ReLU), then a 3x3 conv with $A \cdot C$ outputs (sigmoid per class, no softmax);
    * box tower: same structure, $A \cdot 4$ outputs (deltas vs anchors, as in D2-02);
    * $A = 9$ anchors per location (3 scales x 3 ratios); anchor base size 32 on $P_3$ up to 512 on $P_7$.

    One subtle detail matters a lot: the final classification bias is initialised to
    $b = -\log\frac{1-\pi}{\pi}$ with $\pi = 0.01$, so that at the start *every* anchor predicts
    "1% object". Without it, the ~100 000 background anchors produce a huge loss in the first
    iterations and training diverges.
    """),
    md(r"""
    ### Exercise 5 · `RetinaHead`

    `RetinaHead(in_channels, num_anchors, num_classes, num_convs=4)`. Build two towers (`nn.Sequential`
    of conv + ReLU), a `cls_logits` conv with `A * C` outputs and a `bbox_pred` conv with `A * 4`
    outputs, initialise the classification bias to the prior, and make `forward(list of feature maps)`
    return `(list of class maps, list of box maps)`: for an input level `(N, 256, H, W)` the class map
    is `(N, A * C, H, W)` and the box map `(N, A * 4, H, W)`, one entry per level.
    """),
    api("nn.init.constant_", title="Functions you will need"),
    api("nn.Sequential", "nn.ReLU", "torch.zeros", "torch.sigmoid", brief=True),
    exercise(
        stub=r"""
        import math
        class RetinaHead(nn.Module):
            '''RetinaHead(C_in, A, K): forward(list of feature maps (N, C_in, H, W), one per level) ->
            (list of class maps (N, A * K, H, W), list of box maps (N, A * 4, H, W)); the same weights on every level.'''
            def __init__(self, in_channels, num_anchors, num_classes, num_convs=4, prior=0.01):
                super().__init__()
                # 1. the classification tower: num_convs times (3x3 conv in_channels -> in_channels, padding 1, then ReLU)
                cls_layers = []
                for _ in range(num_convs):
                    # YOUR CODE: append nn.Conv2d(in_channels, in_channels, 3, padding=1) and then nn.ReLU() to cls_layers
                    raise NotImplementedError
                self.cls_tower = nn.Sequential(*cls_layers)      # runs the layers of the list one after the other
                # 2. YOUR CODE: the box tower, self.box_tower: the same structure, built from its own list box_layers
                # 3. YOUR CODE: self.cls_logits, a 3x3 conv (padding 1), in_channels -> num_anchors * num_classes
                # 4. YOUR CODE: self.bbox_pred, a 3x3 conv (padding 1), in_channels -> num_anchors * 4
                # 5. YOUR CODE: bias_value = -math.log((1 - prior) / prior) (then sigmoid(bias_value) = prior), and
                #               nn.init.constant_(self.cls_logits.bias, bias_value) sets every bias of cls_logits to it
                raise NotImplementedError

            def forward(self, feats):
                cls_maps = []
                box_maps = []
                for f in feats:                                  # f: (N, C_in, H, W), one pyramid level
                    # 6. YOUR CODE: cls_features = self.cls_tower(f); append self.cls_logits(cls_features)
                    #               to cls_maps -> (N, A * K, H, W)
                    # 7. YOUR CODE: the same with self.box_tower and self.bbox_pred for box_maps -> (N, A * 4, H, W)
                    raise NotImplementedError
                return cls_maps, box_maps

        tests.check_retina_head(RetinaHead)
        """,
        solution=r"""
        import math
        class RetinaHead(nn.Module):
            '''RetinaHead(C_in, A, K): forward(list of feature maps (N, C_in, H, W), one per level) ->
            (list of class maps (N, A * K, H, W), list of box maps (N, A * 4, H, W)); the same weights on every level.'''
            def __init__(self, in_channels, num_anchors, num_classes, num_convs=4, prior=0.01):
                super().__init__()
                # 1. the classification tower: num_convs times (3x3 conv in_channels -> in_channels, padding 1, then ReLU)
                cls_layers = []
                for _ in range(num_convs):
                    cls_layers.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
                    cls_layers.append(nn.ReLU())
                self.cls_tower = nn.Sequential(*cls_layers)      # runs the layers of the list one after the other
                # 2. the box tower: the same structure, its own weights
                box_layers = []
                for _ in range(num_convs):
                    box_layers.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
                    box_layers.append(nn.ReLU())
                self.box_tower = nn.Sequential(*box_layers)
                # 3. A * K class logits per location
                self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
                # 4. A * 4 box deltas per location
                self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)
                # 5. the prior: every anchor starts at P(object) = prior
                bias_value = -math.log((1 - prior) / prior)      # sigmoid(bias_value) = prior
                nn.init.constant_(self.cls_logits.bias, bias_value)

            def forward(self, feats):
                cls_maps = []
                box_maps = []
                for f in feats:                                  # f: (N, C_in, H, W), one pyramid level
                    # 6. classification branch
                    cls_features = self.cls_tower(f)             # (N, C_in, H, W)
                    cls_maps.append(self.cls_logits(cls_features))   # (N, A * K, H, W)
                    # 7. box branch
                    box_features = self.box_tower(f)             # (N, C_in, H, W)
                    box_maps.append(self.bbox_pred(box_features))    # (N, A * 4, H, W)
                return cls_maps, box_maps

        tests.check_retina_head(RetinaHead)
        """),
    code(r"""
    # How many predictions does RetinaNet make on a 512 x 512 image? (A = 9, C = 80 COCO classes)
    head = day2.RetinaHead(256, 9, 80)
    levels = {}                                      # level name -> grid size on a 512 px image (stride 2 ** l)
    for l in range(3, 8):
        levels[f"P{l}"] = 512 // 2 ** l
    level_inputs = []                                # one empty feature map (1, 256, s, s) per level
    for s in levels.values():
        level_inputs.append(torch.zeros(1, 256, s, s))
    with torch.no_grad():
        cls_out, box_out = head(level_inputs)
    tot = 0
    print(f"{'level':6}{'grid':>9}{'cls output':>22}{'anchors':>10}")
    for (n, s), c in zip(levels.items(), cls_out):
        tot += s * s * 9
        print(f"{n:6}{f'{s} x {s}':>9}{str(tuple(c.shape[1:])):>22}{s * s * 9:>10,d}")
    print(f"{'total':6}{'':>9}{'':>22}{tot:>10,d}  anchors, x 80 class scores each")
    print(f"head parameters (shared by all 5 levels): {torchkit.count_parameters(head) / 1e6:.2f} M; "
          f"initial P(object) = {torch.sigmoid(cls_out[0]).mean():.3f}")
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · Class imbalance and focal loss

    RetinaNet scores ~50 000 anchors per image; maybe 20 are positive. With plain binary
    cross-entropy, $\mathrm{CE}(p_t) = -\log p_t$ where $p_t$ is the predicted probability of the
    *true* label, each easy negative (say $p_t = 0.99$) contributes only $0.01$ - but there are
    50 000 of them, and together they swamp the gradient of the 20 positives. Two-stage detectors
    avoid this by sampling (D2-02); SSD by hard-negative mining (D2-01). **Focal loss** keeps all
    anchors and down-weights the easy ones with a modulating factor:

    $$\mathrm{FL}(p_t) = -\,\alpha_t\,(1 - p_t)^{\gamma}\,\log p_t .$$

    * $\gamma = 0$ gives (alpha-weighted) cross-entropy; $\gamma = 2$ (the paper's default) makes an example with $p_t = 0.9$ count **100x less** and one with $p_t = 0.99$ **10 000x less**;
    * $\alpha_t = \alpha$ for positives, $1-\alpha$ for negatives ($\alpha = 0.25$): a small extra re-balancing (it *down*-weights positives, because $\gamma$ already removed most negatives).
    """),
    api("np.concatenate", "np.log", "np.clip", brief=True),
    code(r"""
    pt = np.linspace(0.001, 1, 500)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    for g, c in zip([0, 0.5, 1, 2, 5], [GREY, "#9ecae1", BLUE, RED, "#6a3d9a"]):
        axes[0].plot(pt, -(1 - pt) ** g * np.log(pt), color=c, lw=2.5 if g == 2 else 1.8, label=f"gamma = {g}" + (" (CE)" if g == 0 else ""))
    axes[0].axvspan(0.6, 1, color=GREEN, alpha=0.08); axes[0].text(0.75, 3.6, "well-classified\nexamples", ha="center", color=GREEN)
    axes[0].set_xlabel("p_t (probability of the true class)"); axes[0].set_ylabel("loss"); axes[0].set_ylim(0, 5); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[0].set_title("focal loss for several gamma (alpha = 1)")
    # contribution of easy negatives on a simulated image: 50 000 anchors
    rng = np.random.default_rng(0)
    n_easy, n_hard, n_pos = 49_000, 980, 20
    p_neg = np.concatenate([rng.beta(1, 60, n_easy), rng.beta(3, 4, n_hard)])     # predicted P(object) for negatives
    p_pos = rng.beta(4, 3, n_pos)
    def fl(pt_, g, a):
        return -a * (1 - pt_) ** g * np.log(np.clip(pt_, 1e-7, 1))
    rows = []
    for name, g, a in [("CE", 0, 0.5), ("focal (gamma=2, alpha=0.25)", 2, 0.25)]:
        easy = fl(1 - p_neg[:n_easy], g, 1 - a).sum()           # total loss of the easy negatives (p_t = 1 - p)
        hard = fl(1 - p_neg[n_easy:], g, 1 - a).sum()           # of the hard negatives
        pos = fl(p_pos, g, a).sum()                             # of the positives (p_t = p)
        rows.append((name, easy, hard, pos))
    x = np.arange(2)
    tot = np.array([r[1] + r[2] + r[3] for r in rows])
    for k, (lab, c) in enumerate(zip(["49 000 easy negatives", "980 hard negatives", "20 positives"], [GREY, ORANGE, GREEN])):
        vals = np.array([r[1 + k] for r in rows]) / tot
        bottom = np.array([sum(r[1 + j] for j in range(k)) for r in rows]) / tot
        axes[1].bar(x, vals, bottom=bottom, color=c, label=lab, width=0.5)
        for xi, v, b in zip(x, vals, bottom):
            if v > 0.04: axes[1].text(xi, b + v / 2, f"{100 * v:.0f}%", ha="center", va="center", color="white", weight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels([r[0] for r in rows]); axes[1].set_ylabel("share of the total loss")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0)); axes[1].set_title("who drives the gradient? (simulated image, 50 000 anchors)")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    With cross-entropy the 49 000 easy negatives - each individually negligible - make up more than
    half of the loss. With focal loss their share drops below 1%: the loss now comes from the examples
    the network still gets wrong (here mostly hard negatives; the 20 positives are few, and $\alpha = 0.25$
    deliberately down-weights them because $\gamma$ has already removed the competition). Compared with
    hard-negative mining / OHEM (online hard example mining: keep only the highest-loss examples of
    each batch), nothing is thrown away and there is no sorting step; the re-weighting is smooth.
    """),
    md(r"""
    ### Lab · Focal loss
    **Predict first:** 100 000 easy negatives at $p_t = 0.99$ against 20 positives at $p_t = 0.3$.
    Under plain cross-entropy, roughly what share of the *total* loss comes from the easy
    negatives? And with $\gamma = 2$? Check in the right half of the lab, then work through the
    "Try this" list (5 minutes).
    """),
    code("""
    labs.show("focal-loss")
    """),
    md(r"""
    ### Exercise 6 · focal loss in PyTorch

    `focal_loss(logits, targets, gamma=2.0, alpha=0.25)` -> mean over all elements (a 0-d tensor).
    `logits` and `targets` have the same shape; `targets` holds 0. / 1. floats. Compute the
    log term with `F.binary_cross_entropy_with_logits(..., reduction="none")` (stable for large
    logits), $p = \sigma(\text{logits})$, $p_t = p\,t + (1-p)(1-t)$, $\alpha_t = \alpha t + (1-\alpha)(1-t)$.
    The checker compares with `torchvision.ops.sigmoid_focal_loss`. For a single positive with
    $p_t = 0.9$ your function returns the lab's FL for that marker, 2.63e-4.
    """),
    exercise(
        stub=r"""
        def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
            '''logits, targets: tensors of the same shape (targets hold 0. or 1.) -> the mean focal loss, a 0-d tensor.'''
            # 1. YOUR CODE: ce, the log term of every element: F.binary_cross_entropy_with_logits(logits, targets,
            #               reduction="none") (that is -log p_t, computed stably) -> same shape as logits
            # 2. YOUR CODE: p, the predicted probability of "object": torch.sigmoid(logits)
            # 3. YOUR CODE: p_t, the probability of the true label: p where the target is 1, 1 - p where it is 0
            #               (p * targets + (1 - p) * (1 - targets))
            # 4. YOUR CODE: alpha_t: alpha for positives, 1 - alpha for negatives (the same trick with targets)
            # 5. YOUR CODE: loss = alpha_t * (1 - p_t) ** gamma * ce, element by element
            # 6. YOUR CODE: return the mean of loss
            raise NotImplementedError

        tests.check_focal_loss_torch(focal_loss)
        """,
        solution=r"""
        def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
            '''logits, targets: tensors of the same shape (targets hold 0. or 1.) -> the mean focal loss, a 0-d tensor.'''
            # 1. ce, the log term of every element (-log p_t, computed stably)
            ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            # 2. p, the predicted probability of "object"
            p = torch.sigmoid(logits)
            # 3. p_t, the probability of the true label
            p_t = p * targets + (1 - p) * (1 - targets)
            # 4. alpha_t: alpha for positives, 1 - alpha for negatives
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            # 5. the focal loss of every element: (1 - p_t) ** gamma shrinks the easy ones (p_t close to 1)
            loss = alpha_t * (1 - p_t) ** gamma * ce
            # 6. its mean
            return loss.mean()

        tests.check_focal_loss_torch(focal_loss)
        """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · Anchor-free heads: FCOS, CenterNet, modern YOLO

    Anchors bring hyper-parameters (scales, ratios, IoU thresholds) that must be tuned per dataset.
    Anchor-free detectors predict boxes from **points**:

    * **FCOS** (Tian et al. 2019): every feature-map location inside a GT box is a positive and
      regresses the four distances $(l, t, r, b)$ to the box sides (FPN levels are assigned by box
      size). Locations near the box border produce poor boxes, so FCOS adds a **centre-ness**
      branch, $\sqrt{\frac{\min(l,r)}{\max(l,r)} \cdot \frac{\min(t,b)}{\max(t,b)}}$, multiplied into the
      score at test time so that NMS keeps the central predictions.
    * **CenterNet** ("Objects as points", Zhou et al. 2019): predict a **heat map** per class with a
      Gaussian peak at each object centre, plus the box size at the peak. Decoding = local maxima
      (a 3 x 3 max-pool comparison) - no NMS at all.
    * **Modern YOLO heads** (YOLOX, v8, 11): anchor-free, **decoupled** (separate classification and
      box branches, like RetinaNet's towers), box sides predicted as a *distribution* over bins
      (**Distribution Focal Loss**) instead of a single number, and dynamic label assignment
      (positives chosen by a cost that mixes classification score and IoU).
    """),
    md(r"""
    ### Exercise 7 · FCOS centre-ness

    `centerness(ltrb)` for an array `(..., 4)` of positive distances.
    """),
    api("np.minimum", "np.maximum", brief=True),
    exercise(
        stub=r"""
        def centerness(ltrb):
            '''ltrb (..., 4): positive distances (l, t, r, b) from a point to the left, top, right and bottom box sides
            -> centre-ness (...), in [0, 1]: 1 at the box centre, close to 0 near a side.'''
            ltrb = np.asarray(ltrb, float)
            # 1. the four distances, one array each: shape (...)
            l = ltrb[..., 0]
            t = ltrb[..., 1]
            r = ltrb[..., 2]
            b = ltrb[..., 3]
            # 2. YOUR CODE: horizontal, the smaller of l and r divided by the larger, element by element
            #               (np.minimum, np.maximum): 1 when the point is half-way between left and right
            # 3. YOUR CODE: vertical, the same for t and b
            # 4. YOUR CODE: return the square root of horizontal * vertical
            raise NotImplementedError

        tests.check_centerness(centerness)
        """,
        solution=r"""
        def centerness(ltrb):
            '''ltrb (..., 4): positive distances (l, t, r, b) from a point to the left, top, right and bottom box sides
            -> centre-ness (...), in [0, 1]: 1 at the box centre, close to 0 near a side.'''
            ltrb = np.asarray(ltrb, float)
            # 1. the four distances, one array each: shape (...)
            l = ltrb[..., 0]
            t = ltrb[..., 1]
            r = ltrb[..., 2]
            b = ltrb[..., 3]
            # 2. horizontal: 1 when the point is half-way between left and right
            horizontal = np.minimum(l, r) / np.maximum(l, r)
            # 3. vertical: the same for top and bottom
            vertical = np.minimum(t, b) / np.maximum(t, b)
            # 4. the square root of their product
            return np.sqrt(horizontal * vertical)

        tests.check_centerness(centerness)
        """),
    api("np.mgrid", "F.max_pool2d", "np.ma.masked_where"),
    api("np.stack", "Tensor.ravel", "np.exp", "np.where", brief=True),
    code(r"""
    fc_img, fc_boxes, fc_labels, _ = shapes.make_sample(np.random.default_rng(7), 128)
    box = fc_boxes[np.argmax(boxes.box_area(fc_boxes))]
    yy, xx = np.mgrid[0:128, 0:128] + 0.5
    pts = np.stack([xx.ravel(), yy.ravel()], 1)
    ltrb = day2.fcos_ltrb(pts, box)
    cn = day2.centerness(ltrb).reshape(128, 128)

    # CenterNet-style heat map: a Gaussian at every centre (sigma ~ box size / 6), then peaks by 3x3 max-pool
    heat = np.zeros((128, 128))
    for b in fc_boxes:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; sig = (b[2] - b[0]) / 6
        heat = np.maximum(heat, np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2)))
    ht = torch.tensor(heat)[None, None]
    peaks = (ht == F.max_pool2d(ht, 3, 1, 1)) & (ht > 0.5)
    py, px = np.where(peaks[0, 0].numpy())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(fc_img); p = np.array([box[0] + 0.3 * (box[2] - box[0]), box[1] + 0.6 * (box[3] - box[1])])
    l, t, r, b = day2.fcos_ltrb(p[None], box)[0]
    for (dx, dy), lab_, v in [((-l, 0), "l", l), ((0, -t), "t", t), ((r, 0), "r", r), ((0, b), "b", b)]:
        axes[0].annotate("", (p[0] + dx, p[1] + dy), p, arrowprops=dict(arrowstyle="->", lw=2, color="k"))
        axes[0].text(p[0] + dx / 2 + 1, p[1] + dy / 2 - 1, f"{lab_}={v:.0f}", fontsize=9, weight="bold")
    axes[0].plot(*p, "o", color=RED); viz.draw_boxes(axes[0], box, edgecolor="k", lw=2)
    axes[0].set_title("FCOS: a point regresses (l, t, r, b)")
    axes[1].imshow(fc_img); axes[1].imshow(np.ma.masked_where(cn == 0, cn), cmap="magma", alpha=0.8, vmin=0, vmax=1)
    axes[1].set_title("centre-ness of every point inside the box")
    axes[2].imshow(fc_img); axes[2].imshow(heat, cmap="jet", alpha=0.5)
    axes[2].plot(px, py, "w*", ms=14, mec="k"); axes[2].set_title(f"CenterNet heat map: {len(px)} peaks = {len(fc_boxes)} objects")
    for ax in axes: ax.set_xlim(0, 128); ax.set_ylim(128, 0); ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: centre-ness reaches 1 only at the box centre ($l = r$ and $t = b$), stays high
    along the two centre lines (the bright cross) and fades to 0 at the border, so predictions from
    border points are down-weighted at test time.
    The CenterNet map has exactly one local maximum per object: the 3 x 3 max-pool test finds them
    without any NMS.
    """),

    # ------------------------------------------------------------------ 7
    md(r"""
    ## 7 · Library detectors on Pet images (no training)

    torchvision ships RetinaNet and FCOS trained on COCO, which has `cat` and `dog` classes. We run
    both on a few Pet validation images at 512 px. Note what they detect: **whole animals**, while
    our Pet labels are **heads** - the model is only as good as the match between its training labels
    and your task. That mismatch is why we fine-tune (D2-04).
    """),
    api("torchvision.models.detection.fcos_resnet50_fpn"),
    api("torchvision.transforms.v2.Compose", "torchvision.transforms.v2.ToImage", "torchvision.transforms.v2.ToDtype",
        brief=True),
    code(r"""
    from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
    cats = RetinaNet_ResNet50_FPN_Weights.DEFAULT.meta["categories"]
    fcos = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT).eval()
    for m in (retina, fcos):
        m.transform.min_size, m.transform.max_size = (512,), 683
    from torchvision.transforms import v2
    tf = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    idx = [5, 40, 77, 120, 260]
    imgs = []
    for i in idx:
        pil_img = Image.open(pet_val.image_path(i)).convert("RGB")
        imgs.append(tf(pil_img))                     # (3, H, W) float in [0, 1]
    res = {}                                         # model name -> one output dict per image
    for name, m in [("RetinaNet", retina), ("FCOS", fcos)]:
        t0 = time.time()
        res[name] = []
        with torch.no_grad():
            for im in imgs:
                outputs = m([im])                    # torchvision detectors take a list of images ...
                res[name].append(outputs[0])         # ... and return a list of dicts: boxes, scores, labels
        print(f"{name}: {(time.time() - t0) / len(imgs):.2f} s / image on CPU at 512 px")
    fig, axes = plt.subplots(2, len(idx), figsize=(17, 7))
    for r, name in enumerate(res):
        for c, (im, out) in enumerate(zip(imgs, res[name])):
            ax = axes[r, c]; ax.imshow(im.permute(1, 2, 0)); ax.axis("off")
            k = out["scores"] > 0.5
            viz.draw_boxes(ax, out["boxes"][k].numpy(), labels=[cats[l] for l in out["labels"][k]], scores=out["scores"][k].numpy(), edgecolor=ORANGE, lw=2)
            _, tgt = pet_val[idx[c]]
            sx, sy = im.shape[2] / 224, im.shape[1] / 224
            viz.draw_boxes(ax, tgt["boxes"].numpy() * [sx, sy, sx, sy], edgecolor=GREEN, lw=2)
            if c == 0: ax.set_title(f"{name} (COCO weights)", loc="left", fontsize=11)
    fig.suptitle("orange: COCO detections with score > 0.5   green: the Pet *head* annotation", y=1.0)
    plt.tight_layout(); plt.show()
    """),

    # ------------------------------------------------------------------ 8
    md(r"""
    ## 8 · Summary

    * A one-stage detector is a convnet whose last layer outputs $A(1 + 4 + C)$ channels per cell; the
      target tensor says which cell / anchor is responsible for which object.
    * Our mini detector (300 k parameters, 2 minutes of CPU) reaches mAP@0.5 > 0.95 on shapes with
      BCE (objectness, all cells) + L1 (box, positives) + CE (class, positives), decode and NMS.
    * Multi-scale is essential: SSD predicts from several maps; FPN adds a top-down pathway so that
      every level has both resolution and semantics. Heads are shared across levels.
    * Dense detectors face ~1000:1 background imbalance. Focal loss $-\alpha_t(1-p_t)^\gamma\log p_t$
      silences easy negatives; the prior-probability bias init keeps the first iterations stable.
    * Anchor-free heads (FCOS distances + centre-ness, CenterNet peaks, YOLOv8's decoupled DFL head)
      removed anchor hyper-parameters and are today's default.

    ### Self-check
    1. How many output channels does a head need for 3 anchors per cell and 20 classes (YOLOv3-style, with objectness)?
    2. Why does the box loss only use the responsible cells, while the objectness loss uses all cells?
    3. In an FPN, why add the upsampled coarse map instead of just using $C_3$ for small objects?
    4. With $\gamma = 2$, by how much is an example with $p_t = 0.9$ down-weighted relative to CE?
    5. What would happen at the first iteration of RetinaNet if the classification bias were initialised to 0?
    6. Which FCOS predictions does centre-ness suppress, and why does that help NMS?

    ### Go deeper
    * Redmon et al., *You Only Look Once: Unified, Real-Time Object Detection* (YOLOv1, CVPR 2016).
    * Liu et al., *SSD: Single Shot MultiBox Detector* (ECCV 2016).
    * Lin et al., *Feature Pyramid Networks for Object Detection* (CVPR 2017).
    * Lin et al., *Focal Loss for Dense Object Detection* (RetinaNet, ICCV 2017).
    * Tian et al., *FCOS: Fully Convolutional One-Stage Object Detection* (ICCV 2019).
    * Zhou et al., *Objects as Points* (CenterNet, 2019).
    """),
]
