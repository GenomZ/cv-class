from nb import md, code, exercise, header

NAME = "D1-02-training-recipe"

CELLS = [
    header(
        "D1-02 · Training convnets well",
        steps=[
            "The small-data problem: train from scratch on 2 000 Pet images and watch it overfit",
            "The input pipeline: `Dataset` -> `transforms.v2` -> `DataLoader`",
            "Data augmentation: geometric, photometric, automatic policies, Mixup / CutMix",
            "Regularisation: weight decay (AdamW), dropout, BatchNorm train / eval, label smoothing, early stopping",
            "Optimisers and learning-rate schedules; the LR range test",
            "The debugging recipe: loss at init, overfit one batch, gradient norms",
            "Baseline vs full recipe; a 10-line random search",
            "Checkpointing and resuming",
        ],
        topics=[
            "Why a convnet trained from scratch on a small dataset memorises instead of generalising",
            "Build a fast `transforms.v2` augmentation pipeline and know which augmentations preserve the label",
            "Mixup, CutMix and label smoothing as soft-target regularisers",
            "BatchNorm's two modes and the classic `model.eval()` bug",
            "Warm-up + cosine and OneCycle schedules; picking the learning rate with a range test",
            "A repeatable debugging routine you can apply to any new model",
        ],
        minutes=90,
        prereq="D1-01 (convolutions, pooling, the convnet pattern). Neural networks, SGD and backprop.",
    ),
    md(r"""
    ## 0 · Setup

    The data is the **Oxford-IIIT Pet** dataset: 37 cat and dog breeds. We use a class-balanced
    subset of **2 000 training / 500 validation images**, decoded once at 144 px and kept in
    memory as `uint8` tensors (`day1.pet_tensors`; the first call takes a minute or two and
    caches the result in `data/checkpoints/`).

    **Compute budget.** On a GPU the from-scratch runs use 128 px images and 30 epochs. On a
    CPU-only laptop they use **64 px and 10 epochs** (1-2.5 minutes per run) - the curves have
    the same shape, the accuracies are lower. Every run is capped in time and saves a checkpoint,
    so re-running the notebook is fast. Two sections use FashionMNIST (D1-01) as a *fast proxy*
    where a Pet run would take hours.
    """),
    code(r"""
    import os, sys, pathlib
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")   # before importing torch: idle CPU threads sleep, not spin
    for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:   # find the cvcourse package
        if (p / "cvcourse").is_dir():
            sys.path.insert(0, str(p)); break

    import math, time, copy
    import numpy as np
    import matplotlib.pyplot as plt
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2
    from cvcourse import viz, tests, torchkit, day1, labs

    %matplotlib inline
    plt.rcParams["figure.dpi"] = 90
    torch.manual_seed(0)
    device = torchkit.get_device()
    GPU = device.type == "cuda"
    if not GPU:
        day1.pick_cpu_threads()     # on a busy CPU, 1 thread can be faster than 8

    IMG = 128 if GPU else 64        # training resolution of the from-scratch runs
    EPOCHS = 30 if GPU else 10      # per from-scratch run
    MAX_MIN = None if GPU else 2.5  # wall-clock cap per run (minutes)
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · The small-data problem

    ImageNet has ~1 300 images per class. Our subset has **54 per class**, and 37 classes that
    differ in subtle ways (a British Shorthair vs a Russian Blue). A convnet with a few hundred
    thousand parameters can simply *memorise* 2 000 images. Let us look at the data, then train
    the small `PetNet` from scratch with no tricks at all: plain Adam, constant learning rate,
    no augmentation.
    """),
    code(r"""
    data = day1.pet_tensors(144)                 # dict: x_train (2000, 3, 144, 144) uint8, y_train, x_val, y_val, classes
    classes = data["classes"]
    short = [c if len(c) <= 16 else c[:15] + "." for c in classes]
    print("train:", tuple(data["x_train"].shape), data["x_train"].dtype, "| val:", tuple(data["x_val"].shape))
    print("images per class (train):", torch.bincount(data["y_train"]).tolist()[:10], "...")

    show_tf = day1.eval_transform(128)
    idx = torch.randperm(len(data["y_train"]), generator=torch.Generator().manual_seed(1))[:16]
    torchkit.show_batch(torch.stack([show_tf(data["x_train"][i]) for i in idx]), data["y_train"][idx], short, n=16)
    plt.show()
    """),
    code(r"""
    model_plain = day1.PetNet(num_classes=37).to(device)
    prof = torchkit.profile(model_plain, (3, IMG, IMG))
    print(f"PetNet: {prof['params']:,} parameters, {prof['macs'] / 1e6:.0f} M multiply-adds per {IMG} px image")

    hist_plain = day1.load_or_train(
        model_plain, f"d1-02_plain_{IMG}px",
        lambda ck: day1.train_plain(model_plain, data, epochs=EPOCHS, lr=3e-3, size=IMG, max_minutes=MAX_MIN,
                                    checkpoint=ck, device=device))
    """),
    code(r"""
    def plot_history(ax_loss, ax_acc, h, label, color):
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax_loss.plot(ep, h["train_loss"], "--", color=color, label=f"{label}: train")
        ax_loss.plot(ep, h["val_loss"], "-o", ms=3, color=color, label=f"{label}: val")
        ax_acc.plot(ep, np.array(h["val_metric"]) * 100, "-o", ms=3, color=color, label=f"{label}: val")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_history(a1, a2, hist_plain, "plain", viz.PALETTE[3])
    a1.set_title("cross-entropy loss"); a1.set_xlabel("epoch"); a1.legend(); a1.grid(alpha=.3)
    a2.axhline(100 / 37, color="gray", ls=":", label="chance (2.7%)")
    a2.set_title("validation accuracy (%)"); a2.set_xlabel("epoch"); a2.legend(); a2.grid(alpha=.3)
    best = int(np.argmin(hist_plain["val_loss"]))
    a1.annotate("best val loss", (best + 1, hist_plain["val_loss"][best]), xytext=(0, 40),
                textcoords="offset points", ha="center", arrowprops=dict(arrowstyle="->"))
    plt.suptitle("Training from scratch without any regularisation", y=1.02); plt.tight_layout(); plt.show()
    """),
    md(r"""
    **Read the curves.** The training loss keeps falling steadily; the validation loss stops
    improving after about five epochs and only wobbles from then on, so the gap between the two
    keeps opening (about +1.1 nats after 10 epochs; a nat is the unit of a loss computed with the
    natural log). Train longer and the validation loss *rises*.
    The network is learning facts about *these 2 000 images* (backgrounds, poses, individual
    animals) rather than about breeds. On the CPU budget (64 px, 10 epochs) the plain run peaks
    around **12-16 %** (chance: 2.7 %); on a GPU at 128 px with 30 epochs it peaks at
    **~15 %** and the gap keeps widening. The rest of this notebook is about closing that gap. (Spoiler: the real fix for small data is transfer learning, block 4, which
    gets **90 %+** on the same subset.)
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · The input pipeline

    PyTorch splits data loading into three pieces:

    | piece | job | knobs that matter |
    |---|---|---|
    | `Dataset` | `ds[i] -> (image, label)`; knows where the data is | decode once and cache if the data fits in memory |
    | `transforms.v2` | per-sample tensor ops: resize, crop, augment, convert to float, normalise | do cheap ops on `uint8`, convert to float late |
    | `DataLoader` | batches, shuffling, parallel workers | `batch_size`, `shuffle=True` for training, `num_workers`, `pin_memory=True` on GPU, `drop_last` |

    `num_workers > 0` runs `__getitem__` in background processes so the GPU never waits for JPEG
    decoding. On Windows / in notebooks, worker processes must be able to *import* your dataset
    and transforms (no lambdas). Here the images are already decoded `uint8` tensors in memory,
    so `num_workers=0` is fine.

    **Why shuffle?** SGD assumes each mini-batch is a random sample of the data. If the data is
    stored sorted by class (it often is: one folder per class), an unshuffled loader feeds batches
    that contain one or two breeds, and each step pushes the network towards "everything is a
    Bengal".
    """),
    code(r"""
    order = torch.argsort(data["y_train"])                   # the dataset as it sits on disk: sorted by class
    sorted_ds = day1.TensorImageDataset(data["x_train"][order], data["y_train"][order])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), sharey=True)
    for ax, shuffle in zip(axes, (False, True)):
        dl = DataLoader(sorted_ds, batch_size=64, shuffle=shuffle, generator=torch.Generator().manual_seed(0))
        n_cls = [len(torch.unique(y)) for _, y in dl]
        ax.bar(range(len(n_cls)), n_cls, color=viz.PALETTE[0 if shuffle else 3])
        ax.set_title(f"shuffle={shuffle}: distinct breeds per batch of 64 (mean {np.mean(n_cls):.1f})")
        ax.set_xlabel("batch index"); ax.grid(alpha=.3, axis="y")
    axes[0].set_ylabel("# breeds in the batch"); plt.tight_layout(); plt.show()
    """),
    md(r"""
    Left: without shuffling, every batch of 64 holds only two or three breeds (54 images each).
    Right: with shuffling, each batch holds about 30 of the 37 breeds.
    """),
    code(r"""
    # The pipeline used in section 1: deterministic resize -> float -> normalise
    print(show_tf)
    x0 = data["x_train"][0]
    y0 = show_tf(x0)
    print("in :", tuple(x0.shape), x0.dtype, f"range [{int(x0.min())}, {int(x0.max())}]")
    print("out:", tuple(y0.shape), y0.dtype, f"mean {y0.mean():.2f}, std {y0.std():.2f}  (ImageNet-normalised)")

    train_dl = DataLoader(day1.TensorImageDataset(data["x_train"], data["y_train"], show_tf), batch_size=64, shuffle=True)
    t0 = time.perf_counter(); xb, yb = next(iter(train_dl)); dt = time.perf_counter() - t0
    print(f"one batch: images {tuple(xb.shape)}, labels {tuple(yb.shape)}  ({dt * 1000:.0f} ms)")
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · Data augmentation

    Augmentation creates new training images that are *still correctly labelled*: a Bengal cat
    cropped, flipped or slightly darker is still a Bengal cat. It is the single most effective
    regulariser for vision, because it encodes what we know about the problem (invariances)
    directly into the data.

    | family | transforms (`torchvision.transforms.v2`) | invariance it teaches |
    |---|---|---|
    | geometric | `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation`, `RandomAffine` | position, scale, mirror, small tilt |
    | photometric | `ColorJitter`, `RandomGrayscale`, `GaussianBlur` | lighting, white balance, camera |
    | occlusion | `RandomErasing` (a.k.a. Cutout) | partial visibility; do not rely on one part |
    | automatic policies | `RandAugment(num_ops, magnitude)`, `TrivialAugmentWide()`, `AutoAugment` | a random op from a list of ~14, two knobs (or none) instead of a hand-tuned recipe |
    | label-mixing | `MixUp`, `CutMix` (batch-level) | smooth decision boundaries, calibrated confidence |

    One image, eight augmentations, three random draws each:
    """),
    code(r"""
    img_u8 = data["x_train"][11]
    to_f = v2.ToDtype(torch.float32, scale=True)
    gallery = {
        "original": v2.Resize(128),
        "RandomResizedCrop": v2.RandomResizedCrop(128, scale=(0.25, 1.0)),
        "HorizontalFlip": v2.Compose([v2.Resize(128), v2.RandomHorizontalFlip(p=1.0)]),
        "Rotation +-25": v2.Compose([v2.Resize(128), v2.RandomRotation(25)]),
        "ColorJitter": v2.Compose([v2.Resize(128), v2.ColorJitter(0.5, 0.5, 0.5, 0.1)]),
        "Grayscale": v2.Compose([v2.Resize(128), v2.RandomGrayscale(p=1.0)]),
        "RandAugment(2, 9)": v2.Compose([v2.Resize(128), v2.RandAugment(num_ops=2, magnitude=9)]),
        "TrivialAugmentWide": v2.Compose([v2.Resize(128), v2.TrivialAugmentWide()]),
        "RandomErasing": v2.Compose([v2.Resize(128), to_f, v2.RandomErasing(p=1.0, scale=(0.05, 0.25), value=0.5)]),
    }
    def as_np(t):
        t = t if t.is_floating_point() else t.float() / 255
        return t.clamp(0, 1).permute(1, 2, 0).numpy()

    torch.manual_seed(3)
    fig, axes = plt.subplots(3, len(gallery), figsize=(2.0 * len(gallery), 6.3))
    for c, (name, tf) in enumerate(gallery.items()):
        for r in range(3):
            axes[r, c].imshow(as_np(tf(img_u8))); axes[r, c].axis("off")
        axes[0, c].set_title(name, fontsize=9)
    fig.suptitle(f"One training image ({classes[data['y_train'][11]]}), three random draws per augmentation", y=1.0)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Every column is the same source image, every row a new random draw. Compare the columns
    that move pixels (crop, flip, rotation) with those that only change colours, and notice that
    RandAugment and TrivialAugment pick a different operation on every draw.
    """),
    md(r"""
    ### What *not* to do

    An augmentation is only valid if it does not change the label **for your task**. Rotating a
    `6` by 180 degrees makes a `9`; flipping a `b` makes a `d`; flipping a chest X-ray swaps the
    heart to the wrong side; an upside-down dog is not something your test set will ever contain
    (it wastes capacity rather than teaching anything). Always *look at* a batch of augmented
    images before you train.
    """),
    code(r"""
    from PIL import Image, ImageDraw, ImageFont
    def glyph(ch, size=96):
        im = Image.new("L", (size, size), 255)
        try:
            font = ImageFont.load_default(size=int(size * 0.8))
        except TypeError:                                    # Pillow < 10.1
            font = ImageFont.load_default()
        ImageDraw.Draw(im).text((size / 2, size / 2), ch, fill=0, font=font, anchor="mm")
        return torch.from_numpy(np.asarray(im)).unsqueeze(0)

    six, b = glyph("6"), glyph("b")
    pet = v2.Resize(128)(data["x_train"][11])
    panels = [(six, "a '6'"), (v2.functional.rotate(six, 180), "rotated 180 -> a '9'"),
              (b, "a 'b'"), (v2.functional.horizontal_flip(b), "h-flipped -> a 'd'"),
              (pet, "a pet"), (v2.functional.vertical_flip(pet), "v-flipped: never seen at test time")]
    fig, axes = plt.subplots(1, 6, figsize=(13, 2.6))
    for ax, (im, t) in zip(axes, panels):
        ax.imshow(as_np(im).squeeze(), cmap="gray"); ax.set_title(t, fontsize=9); ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Exercise 1 · build the training pipeline

    Complete `build_train_tf(size)`: it returns a `v2.Compose` that takes a `uint8` tensor
    `(3, 144, 144)` and returns a normalised float tensor `(3, size, size)`:

    1. `RandomResizedCrop(size, scale=(0.35, 1.0))` - random crop of 35-100 % of the area, resized to `size`;
    2. `RandomHorizontalFlip()`;
    3. `TrivialAugmentWide()` - one random photometric / geometric op per image, random strength;
    4. `ToDtype(torch.float32, scale=True)` - to float in [0, 1];
    5. `Normalize(IMAGENET_MEAN, IMAGENET_STD)`;
    6. `RandomErasing(p=0.25)` (works on the normalised float tensor).

    Order matters: crops and photometric ops are cheaper and better defined on `uint8`;
    `RandomErasing` fills with zeros, which after normalisation means "average colour".
    """),
    exercise(
        stub=r"""
        IMAGENET_MEAN, IMAGENET_STD = torchkit.IMAGENET_MEAN, torchkit.IMAGENET_STD

        def build_train_tf(size=128):
            # YOUR CODE HERE: return v2.Compose([...the six transforms above, in order...])
            raise NotImplementedError

        ok_aug = tests.check_augment_pipeline(build_train_tf)
        """,
        solution=r"""
        IMAGENET_MEAN, IMAGENET_STD = torchkit.IMAGENET_MEAN, torchkit.IMAGENET_STD

        def build_train_tf(size=128):
            return v2.Compose([
                v2.RandomResizedCrop(size, scale=(0.35, 1.0), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.TrivialAugmentWide(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                v2.RandomErasing(p=0.25),
            ])

        ok_aug = tests.check_augment_pipeline(build_train_tf)
        """),
    md(r"""
    **Visual check.** Sixteen draws of your pipeline on the *same* image. Every tile should still
    obviously be the same animal; if a tile looks unrecognisable to *you*, the augmentation is too
    strong for this dataset.
    """),
    code(r"""
    tf_used = tests.use_or_reference(ok_aug, globals().get("build_train_tf"), day1.recipe_transforms, "build_train_tf")(128)
    torch.manual_seed(0)
    torchkit.show_batch(torch.stack([tf_used(data["x_train"][11]) for _ in range(16)]), n=16,
                        title="16 random draws of the training pipeline (same source image)")
    plt.show()
    """),
    md(r"""
    ### Mixup and CutMix: augmenting the *labels* too

    **Mixup** (Zhang et al., 2018) blends two images *and* their one-hot labels with the same weight:

    $$\tilde x = \lambda x_i + (1-\lambda) x_j, \qquad \tilde y = \lambda y_i + (1-\lambda) y_j, \qquad \lambda \sim \mathrm{Beta}(\alpha, \alpha)$$

    **CutMix** (Yun et al., 2019) pastes a random rectangle of image $j$ into image $i$ and sets
    $\lambda$ to the fraction of area kept. Both turn every batch into a set of soft targets: the
    network is penalised for being over-confident between classes, which acts as a strong
    regulariser and improves calibration. Both are done on the *batch*, after the per-image
    transforms; the loss must accept soft targets (`F.cross_entropy` does, when the target is a
    `(B, C)` float tensor).

    Small $\alpha$ (0.2) gives mostly-unmixed images; $\alpha = 1$ gives uniform $\lambda$:
    """),
    code(r"""
    from scipy import stats
    lam = np.linspace(0.001, 0.999, 400)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    for a, c in zip((0.2, 0.4, 1.0, 2.0), viz.PALETTE):
        ax.plot(lam, stats.beta.pdf(lam, a, a), color=c, label=f"alpha = {a}")
    ax.set_ylim(0, 4); ax.set_xlabel("lambda (weight of image i)"); ax.set_ylabel("density")
    ax.set_title("Beta(alpha, alpha): how strongly Mixup mixes"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Lab · Mixup and CutMix
    **Predict first:** CutMix samples $\lambda = 0.75$ for a 128 px image. Is the pasted square
    32 px or 64 px wide? Switch the lab to CutMix and set $\lambda$ to check, then drag the box
    and work through the "Try this" list (5 minutes).
    """),
    code("""
    labs.show("mixup-cutmix")
    """),
    md(r"""
    ### Exercise 2 · implement `mixup`

    `x` is a batch `(B, 3, H, W)`, `y` integer labels `(B,)`. Return the mixed batch and the soft
    labels `(B, num_classes)`. The scaffolding draws `lam` and a random permutation `perm` (pairing
    each image with another one of the same batch) unless they are given - the checker passes them
    explicitly so it can compare numbers. In the lab's terms, `x` is image i, `x[perm]` image j,
    and each row of the soft labels is the bar under the mixed image.
    """),
    exercise(
        stub=r"""
        def mixup(x, y, alpha=0.2, num_classes=37, lam=None, perm=None):
            if lam is None:
                lam = float(np.random.beta(alpha, alpha))
            if perm is None:
                perm = torch.randperm(x.shape[0])
            # YOUR CODE HERE: one-hot y (F.one_hot(..).float()), mix images and labels with lam and perm
            raise NotImplementedError

        ok_mixup = tests.check_mixup(mixup)
        """,
        solution=r"""
        def mixup(x, y, alpha=0.2, num_classes=37, lam=None, perm=None):
            if lam is None:
                lam = float(np.random.beta(alpha, alpha))
            if perm is None:
                perm = torch.randperm(x.shape[0])
            y1 = F.one_hot(y, num_classes).float()
            x_mix = lam * x + (1 - lam) * x[perm]
            y_mix = lam * y1 + (1 - lam) * y1[perm]
            return x_mix, y_mix

        ok_mixup = tests.check_mixup(mixup)
        """),
    code(r"""
    mix_fn = tests.use_or_reference(ok_mixup, globals().get("mixup"), day1.mixup, "mixup")
    xb = torch.stack([show_tf(data["x_train"][i]) for i in range(0, 2000, 250)])      # 8 different breeds
    yb = data["y_train"][0:2000:250]
    torch.manual_seed(4)
    x_mx, y_mx = mix_fn(xb, yb, alpha=1.0, num_classes=37, lam=0.6, perm=torch.roll(torch.arange(8), 1))  # lam fixed for display
    x_cm, y_cm = v2.CutMix(num_classes=37, alpha=1.0)(xb, yb)

    def soft_title(yrow):
        top = torch.topk(yrow, 2)
        return "\n".join(f"{v:.2f} {short[int(i)]}" for v, i in zip(top.values, top.indices) if v > 0.005)

    fig, axes = plt.subplots(2, 4, figsize=(11, 6.2))
    for k in range(4):
        for r, (xx, yy) in enumerate([(x_mx, y_mx), (x_cm, y_cm)]):
            axes[r, k].imshow(torchkit.denormalize(xx[k])); axes[r, k].axis("off")
            axes[r, k].set_title(soft_title(yy[k]), fontsize=8)
    axes[0, 0].text(-12, 64, "Mixup", rotation=90, va="center", ha="right", fontsize=11)
    axes[1, 0].text(-12, 64, "CutMix", rotation=90, va="center", ha="right", fontsize=11)
    plt.suptitle("Mixed images and their soft labels (the target the loss sees)"); plt.tight_layout(); plt.show()
    """),
    md(r"""
    Top row: both animals are visible everywhere and the titles show 0.60 / 0.40, the fixed
    $\lambda$. Bottom row: `v2.CutMix` drew its own $\lambda$ per batch and a random box; each
    title's weights are the area shares of the two images, clipped boxes included, as in the lab.
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · Regularisation

    | technique | what it does | typical setting for convnets |
    |---|---|---|
    | **weight decay** | pulls weights towards 0 every step: prefers simple functions | AdamW, `weight_decay` 0.05 (0.01-0.1); SGD 1e-4 - 5e-4; none on BN / bias |
    | **dropout** | zeroes random activations at train time: no co-adaptation | 0.1-0.5 before the final linear layer; rarely inside conv stacks |
    | **BatchNorm** | normalises each channel with batch statistics; noise from the batch regularises a bit | after every conv; *train / eval modes differ* |
    | **label smoothing** | target = 0.9 on the true class, rest spread: no infinitely confident logits | eps = 0.1 |
    | **Mixup / CutMix** | soft targets from mixed images | alpha 0.2-1.0, for long schedules |
    | **early stopping** | keep the weights from the epoch with the best validation score | always (`torchkit.fit` checkpoints the best epoch) |

    **AdamW, not Adam + L2.** With plain Adam, an L2 penalty $\frac{\lambda}{2}\|w\|^2$ adds $\lambda w$ to the
    gradient, which is then divided by Adam's per-parameter scale $\sqrt{\hat v}$: weights with large
    gradients get almost no decay. (Here $\lambda$ is the weight-decay coefficient, not Mixup's
    $\lambda$.) AdamW (Loshchilov & Hutter) *decouples* the decay from the gradient statistics:

    $$w \leftarrow w - \eta \left( \frac{\hat m}{\sqrt{\hat v} + \epsilon} + \lambda w \right)$$

    In PyTorch: `torch.optim.AdamW(params, lr, weight_decay=0.05)`. (`torch.optim.Adam(weight_decay=...)` is the coupled L2 version.)
    """),
    md(r"""
    ### BatchNorm has two modes

    In **train** mode a BN layer normalises with the mean / variance *of the current batch* and
    updates exponential running averages (`momentum=0.1`). In **eval** mode it uses those running
    averages. Consequences:

    * Forget `model.eval()` at test time and a prediction depends on *which other images are in the batch*.
    * Tiny batches (1-4 images) give noisy statistics: training gets worse, and eval-mode outputs
      (running stats) differ from what the network saw during training.
    """),
    code(r"""
    torch.manual_seed(0)
    bn = nn.BatchNorm1d(1, momentum=0.1)
    mu_true, sd_true = 3.0, 2.0
    track = []
    bn.train()
    for step in range(40):                                   # 40 training batches of 32 samples
        bn(mu_true + sd_true * torch.randn(32, 1))
        track.append((bn.running_mean.item(), bn.running_var.sqrt().item()))
    track = np.array(track)

    probe = torch.tensor([[4.0]])                            # one fixed sample
    outs_train = []
    for k in range(12):                                      # ...placed in 12 different random batches of 4
        batch = torch.cat([probe, mu_true + sd_true * torch.randn(3, 1)])
        bn.train(); outs_train.append(bn(batch)[0, 0].item())
    bn.eval(); out_eval = bn(probe)[0, 0].item()

    shifted = mu_true + 1.0 + sd_true * torch.randn(256, 1)  # a batch from a slightly shifted distribution
    bn.train(); h_train = bn(shifted).detach().numpy().ravel()
    bn.eval();  h_eval = bn(shifted).detach().numpy().ravel()

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    axes[0].plot(track[:, 0], label="running mean"); axes[0].plot(track[:, 1], label="running std")
    axes[0].axhline(mu_true, ls=":", c="gray"); axes[0].axhline(sd_true, ls=":", c="gray")
    axes[0].set_title("running statistics converge (momentum 0.1)"); axes[0].set_xlabel("training step"); axes[0].legend()
    axes[1].plot(range(12), outs_train, "o", label="train mode: depends on batch-mates")
    axes[1].axhline(out_eval, color=viz.PALETTE[2], lw=2, label="eval mode: fixed")
    axes[1].set_title("BN output for the SAME sample (x = 4.0)"); axes[1].set_xlabel("random batch of 4")
    lo, hi = min(outs_train + [out_eval]), max(outs_train + [out_eval]); axes[1].set_ylim(lo - 0.3, hi + 0.9)
    axes[1].legend(fontsize=8, loc="upper center", ncol=2)
    bins = np.linspace(-4, 4, 40)
    axes[2].hist(h_train, bins, alpha=.6, label=f"train mode: mean {h_train.mean():.2f}")
    axes[2].hist(h_eval, bins, alpha=.6, label=f"eval mode: mean {h_eval.mean():.2f}")
    axes[2].set_title("a shifted batch: train mode hides the shift"); axes[2].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.3)
    plt.tight_layout(); plt.show()
    """),
    code(r"""
    # The same bug in a real network: one image alone vs. in a batch, train vs eval mode
    net = day1.PetNet().to(device)
    with torch.no_grad():
        xb = torch.stack([show_tf(data["x_val"][i]) for i in range(8)]).to(device)
        for mode in ("train", "eval"):
            net.train(mode == "train")
            alone = net(xb[:1])[0]
            in_batch = net(xb)[0]
            print(f"{mode:5s} mode: max |logits(alone) - logits(in a batch of 8)| = {(alone - in_batch).abs().max():.4f}")
    """),
    md(r"""
    ### Label smoothing

    With one-hot targets, cross-entropy is minimised only when $p_{\text{true}} \to 1$, i.e. when the
    true-class logit goes to $+\infty$ relative to the others. The network keeps growing its logits
    on training images it already classifies correctly: over-confidence, and a form of overfitting.
    Label smoothing (Szegedy et al., 2016) replaces the target by

    $$q_k = (1 - \varepsilon)\,[k = y] + \frac{\varepsilon}{C}$$

    so the optimum is a finite logit gap. The loss is still cross-entropy, against $q$:
    $\mathcal{L} = -\sum_k q_k \log p_k$.

    ### Exercise 3 · implement `label_smoothing_ce`

    `logits (B, C)`, integer `target (B,)`. Build `q` with `torch.full_like` + `scatter_` (or
    `F.one_hot`) and return the batch mean. The checker compares with
    `F.cross_entropy(logits, target, label_smoothing=eps)`.
    """),
    exercise(
        stub=r"""
        def label_smoothing_ce(logits, target, eps=0.1):
            C = logits.shape[-1]
            logp = F.log_softmax(logits, dim=-1)
            # YOUR CODE HERE: the smoothed target q (B, C) and the loss
            raise NotImplementedError

        ok_ls = tests.check_label_smoothing(label_smoothing_ce)
        """,
        solution=r"""
        def label_smoothing_ce(logits, target, eps=0.1):
            C = logits.shape[-1]
            logp = F.log_softmax(logits, dim=-1)
            q = torch.full_like(logp, eps / C)
            q.scatter_(-1, target.unsqueeze(-1), 1 - eps + eps / C)
            return -(q * logp).sum(-1).mean()

        ok_ls = tests.check_label_smoothing(label_smoothing_ce)
        """),
    code(r"""
    C = 37
    p = np.linspace(0.02, 0.999, 500)                        # probability assigned to the true class
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    k = np.arange(8)
    for off, eps, c in [(-0.2, 0.0, viz.PALETTE[0]), (0.2, 0.1, viz.PALETTE[1])]:
        q = np.full(8, eps / C); q[2] += 1 - eps
        axes[0].bar(k + off, q, width=0.4, color=c, label=f"eps = {eps}")
    axes[0].set_yscale("log"); axes[0].set_ylim(1e-3, 1.5); axes[0].set_xlabel("class (first 8 of 37; true class = 2)")
    axes[0].set_title("target distribution q (log scale)"); axes[0].legend()
    for eps, c in [(0.0, viz.PALETTE[0]), (0.1, viz.PALETTE[1]), (0.3, viz.PALETTE[2])]:
        qt, qo = 1 - eps + eps / C, eps / C
        loss = -qt * np.log(p) - (C - 1) * qo * np.log((1 - p) / (C - 1))
        axes[1].plot(p, loss, color=c, label=f"eps = {eps}")
        i = np.argmin(loss); axes[1].plot(p[i], loss[i], "o", color=c)
    axes[1].set_ylim(0, 4); axes[1].set_xlabel("p(true class)"); axes[1].set_ylabel("loss")
    axes[1].set_title("minimum moves from p = 1 to p = 1 - eps + eps/C (dots)"); axes[1].legend()
    for ax in axes: ax.grid(alpha=.3)
    plt.tight_layout(); plt.show()
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · Optimisers and learning-rate schedules

    | | SGD + momentum (0.9) | AdamW |
    |---|---|---|
    | update | velocity = running sum of gradients | per-parameter step size from running gradient mean / variance |
    | sensitivity to LR | high; needs warm-up and decay | lower; works out of the box for most nets |
    | typical LR (batch 256) | 0.1 | 1e-3 (convnets), 5e-4 - 1e-3 with warm-up (ViTs) |
    | where it wins | long ImageNet schedules for ResNets (often slightly better final accuracy) | fine-tuning, transformers, quick experiments, small data |

    The picture below runs three optimisers on an elongated quadratic bowl - the typical shape of a
    loss surface near a minimum (curvature differs by 40x between directions).
    """),
    code(r"""
    def bowl(w): return 0.5 * (0.05 * w[0] ** 2 + 2.0 * w[1] ** 2)
    runs = {"SGD lr 0.45": (torch.optim.SGD, dict(lr=0.45)),
            "SGD + momentum 0.9, lr 0.1": (torch.optim.SGD, dict(lr=0.1, momentum=0.9)),
            "AdamW lr 0.3": (torch.optim.AdamW, dict(lr=0.3, weight_decay=0.0))}
    gx, gy = np.meshgrid(np.linspace(-11, 2.5, 200), np.linspace(-3, 3, 200))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.contour(gx, gy, 0.5 * (0.05 * gx ** 2 + 2.0 * gy ** 2), levels=np.geomspace(0.01, 12, 14), colors="lightgray")
    for (name, (cls, kw)), c in zip(runs.items(), viz.PALETTE):
        w = torch.tensor([-10.0, 2.0], requires_grad=True); opt = cls([w], **kw); path = [w.detach().clone()]
        for _ in range(40):
            opt.zero_grad(); bowl(w).backward(); opt.step(); path.append(w.detach().clone())
        path = torch.stack(path).numpy()
        ax.plot(path[:, 0], path[:, 1], "-o", ms=2.5, color=c, label=f"{name}: loss after 40 steps {bowl(torch.tensor(path[-1])):.3f}")
    ax.plot(0, 0, "k*", ms=12); ax.set_title("40 steps on an ill-conditioned bowl"); ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-11, 2.5)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Plain SGD has to keep its LR small enough not to diverge along the steep axis, so it crawls
    along the flat one. Momentum accumulates the consistent direction; Adam rescales each
    coordinate.

    ### Lab · Optimizer playground
    **Predict first:** on this ravine, plain SGD with $\eta = 0.9$ zig-zags but still converges.
    What happens at $\eta = 1.1$: more zig-zag, or something else? And which of the three
    optimisers survives the largest $\eta$? Drag the sliders to check, then work through the
    "Try this" list (6 minutes).
    """),
    code("""
    labs.show("optimizer-playground")
    """),
    md(r"""
    Neither momentum nor Adam removes the need for a good *learning-rate schedule*:

    * **warm-up** (a few hundred to a few thousand steps from ~0 to the peak LR): early gradients
      are large and noisy, BN statistics and Adam's moment estimates are not settled yet;
    * **decay** to ~0 at the end: large steps explore, small steps settle into a minimum. Cosine
      decay is the default today: $\eta_t = \tfrac12 \eta_{\max}\,(1 + \cos(\pi\, t / T))$;
    * **OneCycle** (Smith): warm up to a high peak, anneal to far below the start, with momentum
      cycled inversely. Fast convergence in few epochs.

    ### Lab · Learning-rate schedule designer
    **Predict first:** the D1-01 convnet was trained with `torch.optim.lr_scheduler.OneCycleLR`.
    At step 0, does it start at 0, at `max_lr / 25` or at `max_lr`? The lab opens with the
    schedule of this notebook's recipe run (section 7); switch to one-cycle to check, then work
    through the "Try this" list (4 minutes).
    """),
    code("""
    labs.show("lr-schedule")
    """),
    md(r"""
    ### Exercise 4 · implement `warmup_cosine`

    `warmup_cosine(step, total, warmup, lr_max)`: for `step < warmup` rise linearly,
    `lr_max * (step + 1) / warmup`; afterwards cosine-decay from `lr_max` (at `step == warmup`)
    to 0 (at `step == total`). `step` runs from 0 to `total - 1`; return the learning rate as a
    Python float. With "final LR / peak" at 0, the lab's formula box shows the value your
    function must return for any step.
    """),
    exercise(
        stub=r"""
        def warmup_cosine(step, total, warmup, lr_max):
            # YOUR CODE HERE
            raise NotImplementedError

        ok_sched = tests.check_warmup_cosine(warmup_cosine)
        """,
        solution=r"""
        def warmup_cosine(step, total, warmup, lr_max):
            if step < warmup:
                return lr_max * (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            return 0.5 * lr_max * (1 + math.cos(math.pi * progress))

        ok_sched = tests.check_warmup_cosine(warmup_cosine)
        """),
    code(r"""
    sched_fn = tests.use_or_reference(ok_sched, globals().get("warmup_cosine"), day1.warmup_cosine, "warmup_cosine")
    epochs, spe = 20, 50                                     # 20 epochs x 50 steps
    T = epochs * spe
    steps = np.arange(T)

    def torch_schedule(make):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0, momentum=0.9)
        sch = make(opt); out = []
        for _ in range(T):
            out.append(opt.param_groups[0]["lr"]); opt.step(); sch.step()
        return np.array(out)

    curves = {
        "constant": np.ones(T),
        "step decay (x0.1 at 10, 15)": torch_schedule(lambda o: torch.optim.lr_scheduler.MultiStepLR(o, [10 * spe, 15 * spe], 0.1)),
        "cosine": torch_schedule(lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T)),
        "warm-up (2 ep) + cosine  [yours]": np.array([sched_fn(s, T, 2 * spe, 1.0) for s in steps]),
        "OneCycle": torch_schedule(lambda o: torch.optim.lr_scheduler.OneCycleLR(o, max_lr=1.0, total_steps=T)),
    }
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for (name, cv), c in zip(curves.items(), viz.PALETTE):
        ax.plot(steps / spe, cv, color=c, lw=2 if "yours" in name else 1.4, label=name)
    ax.set_xlabel("epoch"); ax.set_ylabel("LR / peak LR"); ax.set_title("Learning-rate schedules over 20 epochs")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0)); ax.grid(alpha=.3); plt.tight_layout(); plt.show()
    """),
    md(r"""
    Your curve (thick) should rise for two epochs, then decay along a cosine that reaches 0 at
    the same final step as the plain cosine.
    The step schedule drops 10x twice; OneCycle (torch defaults: `pct_start=0.3`, both phases
    cosine-shaped) starts at `max_lr / 25` and ends far below it.
    """),
    md(r"""
    ### The LR range test

    Which peak LR? Leslie Smith's range test trains for a few dozen batches while the LR grows
    exponentially from 1e-5 to 1, and records the (smoothed) loss. The loss first stays flat (LR
    too small to matter), then drops (useful range), then explodes (too large). A good peak LR is
    about **10x below the minimum** of the curve. `torchkit.lr_range_test` restores the weights
    afterwards.

    From scratch on Pet the loss moves so slowly that 100 steps show little more than the
    explosion point, so we run the test on the fast proxy: the D1-01 convnet on FashionMNIST
    (each step takes milliseconds). On your own problem, run it on the real model and data.
    """),
    code(r"""
    fm_train, fm_val = day1.fashion_subsets(n_train=6400, n_val=2000)
    fm_dl = DataLoader(fm_train, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    lrs, losses = torchkit.lr_range_test(day1.FashionNet().to(device), fm_dl, F.cross_entropy, torch.optim.AdamW,
                                         lr_min=1e-5, lr_max=1.0, steps=100)
    i_min = int(np.argmin(losses)); lr_pick = lrs[i_min] / 10
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.semilogx(lrs, losses); ax.axvline(lrs[i_min], ls=":", c="gray", label=f"minimum at {lrs[i_min]:.1e}")
    ax.axvline(lr_pick, color=viz.PALETTE[2], label=f"pick ~ min / 10 = {lr_pick:.1e}")
    ax.set_xlabel("learning rate (log)"); ax.set_ylabel("smoothed training loss")
    ax.set_title("LR range test, D1-01 convnet + AdamW on FashionMNIST"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    The suggestion is a starting point, not gospel: with warm-up + cosine you can usually go a
    bit higher. For PetNet we use a peak of 3e-3 with AdamW.
    """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · The debugging recipe

    Most failed training runs are bugs, not bad hyper-parameters. Karpathy's recipe, condensed:

    1. **Check the loss at initialisation.** With $C$ balanced classes and a fresh network the
       softmax is ~uniform, so the loss should be $\ln C$ ($\ln 37 = 3.61$). Much higher means
       a broken init or a bad last layer; NaN means a data or normalisation bug.
    2. **Look at the data right before it enters the model** (after augmentation, after
       normalisation): `torchkit.show_batch`. Are the labels right? Are the images upside down?
    3. **Overfit one batch.** A correct model + loss + optimiser must drive the loss of a single
       fixed batch to ~0. If it cannot, nothing else matters yet.
    4. **Watch gradient norms per layer** for vanishing / exploding gradients.
    5. **Loss becomes NaN / inf** -> lower the LR (or add warm-up / gradient clipping), check for
       `log(0)`, division by zero, fp16 overflow.
    6. Only then: make it bigger, add regularisation, tune.
    """),
    code(r"""
    torch.manual_seed(0)
    xb, yb = next(iter(DataLoader(day1.TensorImageDataset(data["x_train"], data["y_train"], show_tf), 64, shuffle=True)))
    xb, yb = xb.to(device), yb.to(device)
    net = day1.PetNet().to(device)
    with torch.no_grad():
        print(f"loss at init: {F.cross_entropy(net(xb), yb):.3f}   (ln 37 = {math.log(37):.3f})")
        net.fc.weight.mul_(30)                               # simulate a badly scaled last layer
        print(f"with a 30x last layer: {F.cross_entropy(net(xb), yb):.3f}  <- confident and wrong at step 0: suspicious")
    """),
    md(r"""
    ### Exercise 5 · the training step

    This is the inner loop of `torchkit.fit` (and of every PyTorch training loop you will ever
    write). Put the five lines in the right order: train mode, forward, loss, **zero the old
    gradients**, backward, optimiser step; return the loss as a Python float. The checker runs
    three steps and compares the weights with a reference: forgetting `zero_grad` makes the
    gradients *accumulate* across steps.
    """),
    exercise(
        stub=r"""
        def train_step(model, x, y, loss_fn, optimizer):
            # YOUR CODE HERE
            raise NotImplementedError

        ok_step = tests.check_train_step(train_step)
        """,
        solution=r"""
        def train_step(model, x, y, loss_fn, optimizer):
            model.train()
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.item()

        ok_step = tests.check_train_step(train_step)
        """),
    md(r"""
    ### Exercise 6 · overfit one batch

    Write `overfit_one_batch(model, x, y, steps, lr)`: create an `AdamW` optimiser, call your
    `train_step` `steps` times **on the same batch**, return the list of losses. The checker uses
    a small network; then we run it on `PetNet` with 16 real images (at 64 px to keep it quick).
    """),
    exercise(
        stub=r"""
        step_fn = tests.use_or_reference(ok_step, globals().get("train_step"), day1.train_step, "train_step")

        def overfit_one_batch(model, x, y, steps=100, lr=1e-2):
            # YOUR CODE HERE
            raise NotImplementedError

        ok_overfit = tests.check_overfit_one_batch(overfit_one_batch)
        """,
        solution=r"""
        step_fn = tests.use_or_reference(ok_step, globals().get("train_step"), day1.train_step, "train_step")

        def overfit_one_batch(model, x, y, steps=100, lr=1e-2):
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
            return [step_fn(model, x, y, F.cross_entropy, opt) for _ in range(steps)]

        ok_overfit = tests.check_overfit_one_batch(overfit_one_batch)
        """),
    code(r"""
    tf64 = day1.eval_transform(64)
    x16 = torch.stack([tf64(data["x_train"][i]) for i in range(0, 2000, 125)]).to(device)   # 16 images, 16 breeds
    y16 = data["y_train"][0:2000:125].to(device)
    of_fn = tests.use_or_reference(ok_overfit, globals().get("overfit_one_batch"), day1.overfit_one_batch, "overfit_one_batch")
    torch.manual_seed(0)
    healthy = of_fn(day1.PetNet().to(device), x16, y16, 60, 3e-3)

    # the same loop with a typical data bug: images and labels get out of sync (labels shuffled every step)
    torch.manual_seed(0)
    net_bug = day1.PetNet().to(device); opt_bug = torch.optim.AdamW(net_bug.parameters(), lr=3e-3, weight_decay=0.0)
    buggy = [day1.train_step(net_bug, x16, y16[torch.randperm(16)], F.cross_entropy, opt_bug) for _ in range(60)]

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(healthy, color=viz.PALETTE[2], label=f"correct pipeline: final loss {healthy[-1]:.3f}")
    ax.plot(buggy, color=viz.PALETTE[3], label=f"labels out of sync (bug): final loss {buggy[-1]:.2f}")
    ax.axhline(math.log(37), ls=":", c="gray", label="ln 37 (chance)")
    ax.set_yscale("log"); ax.set_xlabel("step (same 16 images every step)"); ax.set_ylabel("training loss (log)")
    ax.set_title("Overfit one batch: a correct setup reaches ~0, a buggy one cannot"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.show()
    """),
    code(r"""
    # Gradient norm per layer after one backward pass on a fresh PetNet
    net = day1.PetNet().to(device)
    F.cross_entropy(net(xb), yb).backward()
    gn = day1.grad_norms_per_layer(net)
    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.bar(range(len(gn)), list(gn.values()), color=viz.PALETTE[0])
    ax.set_xticks(range(len(gn))); ax.set_xticklabels([k.replace(".weight", "") for k in gn], rotation=30, ha="right", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("||grad|| (log)"); ax.set_title("Gradient norm per weight tensor at init: no layer is starved")
    ax.grid(alpha=.3, axis="y"); plt.tight_layout(); plt.show()
    """),

    # ------------------------------------------------------------------ 7
    md(r"""
    ## 7 · Putting it together: baseline vs. recipe

    The recipe, all in `day1.train_recipe` (read it - it is 15 lines):

    * augmentation: **your** pipeline from exercise 1 (`RandomResizedCrop` + flip + `TrivialAugmentWide` + `RandomErasing`);
    * `AdamW(lr=3e-3, weight_decay=0.05)`, no decay on BN / bias parameters;
    * linear warm-up for 15 % of the steps, then cosine to 0 (**your** `warmup_cosine` via `LambdaLR`);
    * label smoothing 0.1 (**your** `label_smoothing_ce`); dropout 0.2 before the classifier;
    * the checkpoint keeps the best epoch (early stopping).

    Same network, same epochs, same data as section 1.
    """),
    code(r"""
    aug_builder = tests.use_or_reference(ok_aug, globals().get("build_train_tf"), day1.recipe_transforms, "build_train_tf")
    sched_fn = tests.use_or_reference(ok_sched, globals().get("warmup_cosine"), day1.warmup_cosine, "warmup_cosine")
    ls_fn = tests.use_or_reference(ok_ls, globals().get("label_smoothing_ce"), day1.label_smoothing_ce, "label_smoothing_ce")

    model_recipe = day1.PetNet(num_classes=37, dropout=0.2).to(device)
    hist_recipe = day1.load_or_train(
        model_recipe, f"d1-02_recipe_{IMG}px",
        lambda ck: day1.train_recipe(model_recipe, data, epochs=EPOCHS, lr=3e-3, weight_decay=0.05, label_smoothing=0.1,
                                     size=IMG, train_tf=aug_builder(IMG), schedule_fn=sched_fn, smoothing_loss=ls_fn,
                                     max_minutes=MAX_MIN, checkpoint=ck, device=device))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_history(a1, a2, hist_plain, "plain", viz.PALETTE[3])
    plot_history(a1, a2, hist_recipe, "recipe", viz.PALETTE[0])
    a1.set_title("loss (dashed: train, solid: val)"); a1.set_xlabel("epoch"); a1.legend(fontsize=8); a1.grid(alpha=.3)
    a2.set_title("validation accuracy (%)"); a2.set_xlabel("epoch"); a2.legend(fontsize=8); a2.grid(alpha=.3)
    plt.tight_layout(); plt.show()
    gap = lambda h: h["val_loss"][-1] - h["train_loss"][-1]
    print(f"best val acc  plain {max(hist_plain['val_metric']):.3f}   recipe {max(hist_recipe['val_metric']):.3f}")
    print(f"final val-train loss gap  plain {gap(hist_plain):+.2f}   recipe {gap(hist_recipe):+.2f}")
    """),
    md(r"""
    **What to look for.** With the recipe the train loss is *higher* (augmented, smoothed targets
    are harder to fit - and label smoothing puts a floor under the loss) but the validation loss
    keeps following it: the gap stays small, and the validation loss is still falling when the
    schedule ends. After 10 CPU epochs at 64 px the two accuracies are close (~13-16 %): heavy
    regularisation *slows down* fitting, and pays off only with longer schedules. Expected numbers
    on a GPU at 128 px: **~15 % plain vs ~30-35 % with the recipe after 30-40 epochs** (and the
    plain run gets *worse* the longer you train it). With a pretrained backbone (D1-04): 80-90 %+.
    """),
    md(r"""
    ### A 10-line random search

    Which knobs matter? In order: **learning rate**, then **weight decay** and **augmentation
    strength** / dropout, then everything else. Random search beats grid search because each trial
    tests a new value of *every* knob (Bergstra & Bengio, 2012); sample scale parameters
    log-uniformly. Search on a cheap *proxy* (fewer images, lower resolution, a narrower net, fewer
    epochs): proxies preserve the *ranking* of settings reasonably well, not the absolute accuracy.

    A meaningful search on Pet-from-scratch needs hours of GPU time; to see the procedure work in
    about a minute, the proxy here is the FashionMNIST convnet (6 400 images, 1 epoch per trial).
    """),
    code(r"""
    rng = np.random.default_rng(0); results = []; t0 = time.time()
    for trial in range(8):
        cfg = dict(lr=float(10 ** rng.uniform(-4.5, -0.5)), weight_decay=float(10 ** rng.uniform(-4, -0.5)),
                   dropout=float(rng.uniform(0, 0.5)))
        torch.manual_seed(trial)
        net = day1.FashionNet(cfg["dropout"]).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        h = torchkit.fit(net, fm_dl, DataLoader(fm_val, 500), F.cross_entropy, opt, epochs=1, verbose=False, max_minutes=0.5)
        results.append({**cfg, "val_acc": h["val_metric"][-1]})
        print(f"trial {trial}: lr {cfg['lr']:.1e}  wd {cfg['weight_decay']:.1e}  dropout {cfg['dropout']:.2f}"
              f"  ->  val acc {results[-1]['val_acc']:.3f}   ({time.time() - t0:.0f} s)")
    best = max(results, key=lambda r: r["val_acc"])
    print("best:", {k: round(v, 4) for k, v in best.items()})
    """),
    code(r"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, key in zip(axes, ("lr", "weight_decay")):
        sc = ax.scatter([r[key] for r in results], [100 * r["val_acc"] for r in results], c=[r["dropout"] for r in results],
                        cmap="viridis", vmin=0, vmax=0.5, s=70, edgecolor="k")
        ax.set_xscale("log"); ax.set_xlabel(key + " (log)"); ax.grid(alpha=.3)
    axes[0].set_ylabel("val accuracy (%)"); plt.colorbar(sc, ax=axes, label="dropout")
    axes[0].set_title("accuracy depends strongly on the LR..."); axes[1].set_title("...much less on weight decay (at 1 epoch)")
    plt.show()
    """),

        # ------------------------------------------------------------------ 8
    md(r"""
    ## 8 · Checkpointing and resuming

    Long runs crash, get pre-empted, or you want the best epoch back. A *weights-only*
    checkpoint (`torchkit.save_checkpoint`) is enough for inference and fine-tuning. To **resume**
    training exactly you also need the optimiser state (momentum / Adam moments), the scheduler,
    the epoch counter and the random-number-generator state (data order, augmentation, dropout).

    The cell below proves it: run A trains 2 epochs straight; run B trains 1 epoch, "crashes",
    builds *fresh* objects, loads the checkpoint and trains the second epoch. The weights match.
    """),
    code(r"""
    def make_run(seed=0):
        torch.manual_seed(seed)
        model = day1.PetNet(width=16).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2)
        dl, _ = day1.loaders(data, day1.recipe_transforms(64), day1.eval_transform(64), 64, n_train=256)
        return model, opt, sched, dl

    def one_epoch(model, opt, sched, dl):
        for x, y in dl:
            day1.train_step(model, x.to(device), y.to(device), F.cross_entropy, opt)
        sched.step()

    # run A: two epochs, no interruption
    mA, oA, sA, dlA = make_run()
    torch.manual_seed(123); one_epoch(mA, oA, sA, dlA); one_epoch(mA, oA, sA, dlA)

    # run B: one epoch, save EVERYTHING, "crash"
    mB, oB, sB, dlB = make_run()
    torch.manual_seed(123); one_epoch(mB, oB, sB, dlB)
    ck_path = torchkit.CHECKPOINTS / "d1-02_resume_demo.pt"
    torch.save({"model": mB.state_dict(), "optimizer": oB.state_dict(), "scheduler": sB.state_dict(), "epoch": 1,
                "torch_rng": torch.get_rng_state(), "loader_rng": dlB.generator.get_state()}, ck_path)
    del mB, oB, sB, dlB

    # resume into brand-new objects
    mC, oC, sC, dlC = make_run(seed=999)                     # different init on purpose: everything comes from the file
    state = torch.load(ck_path, weights_only=False)
    mC.load_state_dict(state["model"]); oC.load_state_dict(state["optimizer"]); sC.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"]); dlC.generator.set_state(state["loader_rng"])
    one_epoch(mC, oC, sC, dlC)

    diff = max((a - b).abs().max().item() for a, b in zip(mA.state_dict().values(), mC.state_dict().values()))
    print(f"resumed from epoch {state['epoch']}; max |weights(A) - weights(resumed)| = {diff:.2e}")
    ck_path.unlink()
    """),

    # ------------------------------------------------------------------ 9
    md(r"""
    ## 9 · Summary

    * A convnet trained from scratch on ~50 images per class memorises: train loss down, val loss
      up. Always plot both curves.
    * The pipeline: decode once, cheap `uint8` transforms, then float + normalise; shuffle the
      training loader; `num_workers` so the GPU never waits.
    * Augmentation is the strongest regulariser for vision, *if* it preserves the label. Mixup /
      CutMix and label smoothing soften the targets.
    * BatchNorm uses batch statistics in train mode and running averages in eval mode: call
      `model.eval()` before validation / inference.
    * AdamW (decoupled weight decay) + linear warm-up + cosine decay is a robust default; the LR
      range test gives the order of magnitude of the peak LR.
    * Debug first: loss at init = $\ln C$, look at the batch, overfit one batch, watch gradient
      norms, NaN -> lower the LR.
    * Search LR first (log-uniform, random search, on a proxy), then weight decay and augmentation.
    * To resume exactly, checkpoint model + optimiser + scheduler + epoch + RNG state.

    ### Self-check
    1. Your loss at step 0 is 12.4 on a 37-class problem. Name two likely causes.
    2. Why is vertical flip a bad augmentation for Pet but fine for satellite images?
    3. What is the target vector for class 2 of 4 with label smoothing $\varepsilon = 0.2$?
    4. Validation accuracy changes when you change the validation batch size. What did you forget?
    5. Why does Adam with `weight_decay` behave differently from AdamW?
    6. Your loss becomes NaN after 300 steps. List three things to try, in order.

    ### Go deeper
    * A. Karpathy, *A Recipe for Training Neural Networks* (blog, 2019).
    * L. N. Smith, *A disciplined approach to neural network hyper-parameters: Part 1* (2018) - LR range test, OneCycle.
    * E. D. Cubuk et al., *RandAugment: Practical automated data augmentation with a reduced search space* (2020); S. G. Muller & F. Hutter, *TrivialAugment* (2021).
    * H. Zhang et al., *mixup: Beyond Empirical Risk Minimization* (ICLR 2018).
    * S. Yun et al., *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features* (ICCV 2019).
    * I. Loshchilov & F. Hutter, *Decoupled Weight Decay Regularization* (AdamW, ICLR 2019); *SGDR: Stochastic Gradient Descent with Warm Restarts* (cosine schedule, 2017).
    * R. Wightman et al., *ResNet strikes back: An improved training procedure in timm* (2021) - what a modern recipe is worth.
    """),
]
