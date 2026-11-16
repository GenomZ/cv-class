from nb import md, code, exercise, header, api

NAME = "D1-02-training-recipe"

CELLS = [
    header(
        "D1-02 · Training convnets well",
        steps=[
            "The small-data problem: train a convnet from scratch on 2 000 Pet images and watch it overfit; "
            "logits, softmax, cross-entropy, accuracy",
            "The input pipeline: `Dataset` -> `transforms.v2` -> `DataLoader`, and why the training data is shuffled",
            "Data augmentation: geometric, photometric, automatic policies (exercise 1); Mixup / CutMix (exercise 2)",
            "Regularisation: weight decay, dropout, BatchNorm in train and eval mode, label smoothing (exercise 3), "
            "early stopping",
            "Optimisers (SGD, momentum, Adam / AdamW) and learning-rate schedules (exercise 4); the LR range test",
            "The debugging recipe: loss at initialisation, the training step (exercise 5), overfit one batch "
            "(exercise 6), gradient norms",
            "Baseline vs full recipe; a small random search",
            "Checkpointing and resuming",
        ],
        topics=[
            "Why a convnet trained from scratch on a small dataset memorises instead of generalising",
            "Read a cross-entropy loss: what ln 37 = 3.61 means at the first step",
            "Build a fast `transforms.v2` augmentation pipeline and know which augmentations preserve the label",
            "Mixup, CutMix and label smoothing as soft-target regularisers",
            "BatchNorm's two modes and the classic `model.eval()` bug",
            "Warm-up + cosine and OneCycle schedules; picking the learning rate with a range test",
            "A repeatable debugging routine you can apply to any new model",
        ],
        minutes=140,
        setting="in class (guided + hands-on), with all six exercises and the three labs; the **More** boxes are extra",
        prereq="D0-00 (tensors, normalisation, the training loop) and D1-01 (convolutions, pooling, the convnet "
               "pattern). Neural networks, SGD and backpropagation.",
    ),
    md(r"""
    ## How to read this notebook

    The instructor guides you through it in class; the text is written so that you can also follow it
    alone. D0-00 has the long version of these rules.

    * **Run the cells in order** (Shift+Enter). Cells use what earlier cells made.
    * **Six exercises**, each under a heading `### Exercise N`. The text above an exercise gives the
      goal, what goes in and what comes out (with a tiny example), a plan in steps, what the checker
      tests, and a table of common mistakes with the message each one produces.
    * **The exercise cell is a template.** The function, its numbered steps and the bookkeeping lines
      are written. You fill in the lines marked `# YOUR CODE`: the three dots `____` stand for the code
      that is missing, and the comment says what that line must compute. When every blank is filled,
      delete the line `raise NotImplementedError` and run the cell.
    * **Checkers.** The last line of an exercise cell calls a checker, which prints `[PASS]`, or
      `[FAIL]` with what went wrong and a hint.
    * **An unfinished exercise does not stop the notebook.** Its checker prints `[FAIL] ... not
      implemented yet`, and the later cells that need your function use the course's own version
      instead, with the line `note: using the reference implementation of ...`.
    * **Training cells** load a saved result from `data/checkpoints/` when there is one, so they take
      seconds; on a fresh machine each trains for 1 to 3 minutes.
    * **Stuck for more than five minutes?** `notebooks/solutions/D1-02-training-recipe-solution.ipynb`
      is this notebook with every blank filled. Read that one cell, close it, and type it yourself.
    * Closed boxes **Functions used here** explain each library function of the next cell; boxes
      titled **More:** are optional depth, skip them on a first read. New words are **bold** where they
      are explained and collected in "Words from this notebook" at the end.
    """),
    md(r"""
    ## 0 · Setup

    **The data.** The **Oxford-IIIT Pet** dataset (D0-00, section 5): photos of 37 cat and dog breeds.
    We use a **class-balanced** subset, one with (almost) the same number of images for every breed:
    **2 000 training and 500 validation images**. They are decoded once at 144 x 144 pixels and kept in
    memory as `uint8` tensors (`day1.pet_tensors`; the first call takes a minute or two and saves the
    result in `data/checkpoints/`).

    **Compute budget.** Sections 1 and 7 each train a network **from scratch**: starting from random
    weights, not from a network someone else trained (that is D1-04). On a GPU those runs use 128 px
    images and 30 epochs. On a CPU-only laptop they use **64 px and 10 epochs** (1 to 2.5 minutes per
    run): the curves have the same shape, the accuracies are lower. Every run is capped in time and
    saves a checkpoint, so running the notebook again is fast. Two sections use FashionMNIST (D1-01)
    as a **fast proxy**: a smaller problem that behaves alike, used where the same experiment on Pet
    would take hours.

    The setup cell, line by line:
    * PyTorch splits big computations over several **threads**, lines of work the CPU runs side by side.
      `OMP_WAIT_POLICY=PASSIVE` makes idle threads sleep instead of spinning (checking for work again
      and again); it must be set before `import torch`;
    * the `for` loop finds the course package `cvcourse` (as in D0-00);
    * `F` is `torch.nn.functional`: the operations of the layers as plain functions, for example
      `F.cross_entropy` (section 1); `v2` is `torchvision.transforms.v2`, torchvision's image
      transformations (sections 2 and 3);
    * `day1` holds this day's shared code: the dataset, the network `PetNet`, the training functions,
      and the course's own solution of every exercise;
    * `day1.pick_cpu_threads()` times a small workload with 1 and with 4 threads and keeps the faster
      setting (on a busy CPU, 1 thread can win);
    * `IMG`, `EPOCHS` and `MAX_MIN` are the image size, the number of epochs and the time cap (in
      minutes) of the two from-scratch runs.
    """),
    api("torch.manual_seed", "torchkit.get_device", brief=True),
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

    **Why this section.** ImageNet, the collection most vision networks learn from, has about 1 300
    images per class. Our subset has **54 per class**, and the 37 breeds differ in small details (a
    British Shorthair and a Russian Blue are both grey cats). This section trains a small network on it
    with no tricks at all and shows the problem that the rest of the notebook solves.

    **Words.**
    * **Overfitting**: the model learns the training images themselves (their backgrounds, poses, the
      individual animals) instead of what makes a breed a breed. It keeps getting better on the training
      images and stops getting better, or gets worse, on new ones. A network with a few hundred
      thousand parameters can **memorise** 2 000 images: store enough about each one to recognise it.
    * **Generalisation**: doing well on images the model has never seen. In the end it is the only
      thing that counts.
    * **Training set and validation set**: the model learns from the training images only. The
      validation images (D0-00) are kept apart and measure generalisation after every epoch.

    **The numbers we watch.** D1-01 used them; here are the details.
    * The network ends with one raw score per class, the **logits**: any real numbers, larger means
      "more likely this class".
    * **Softmax** turns the logits into probabilities: $p_k = e^{z_k} / \sum_j e^{z_j}$ for the logit
      $z_k$ of class k. Every $p_k$ lies between 0 and 1, and together they add up to 1.
    * The **cross-entropy** loss of one image is $-\ln p_{\text{true}}$: minus the natural logarithm of
      the probability given to the right class. It is 0 when the right class gets probability 1, and
      grows without limit as that probability goes to 0. For a batch, the images' losses are averaged.
      Its unit is the **nat** (the unit of a quantity computed with the natural log).
    * **Accuracy**: the share of images whose largest logit belongs to the right class. Guessing at
      random gives 1/37 = 2.7 %, the **chance level**.

    **Worked example.** Three classes, logits (2, 1, 0) for one image. By hand: $e^2 = 7.389$,
    $e^1 = 2.718$, $e^0 = 1$, together 11.107; the probabilities are 7.389 / 11.107 = 0.665, then 0.245
    and 0.090. If the right class is class 0, the loss is $-\ln 0.665 = 0.41$; if it is class 2,
    $-\ln 0.090 = 2.41$. A network that gives each of the 37 Pet breeds the same probability 1/37 has
    the loss $-\ln(1/37) = \ln 37 = 3.61$. A freshly initialised network starts close to that; section
    6 turns this into a test.
    """),
    api("Tensor.softmax", "F.cross_entropy"),
    api("torch.tensor", "Tensor.sum", "Tensor.item", "Tensor.unsqueeze", brief=True),
    code(r"""
    logits = torch.tensor([2.0, 1.0, 0.0])           # one image, 3 classes: the raw scores
    probs = logits.softmax(dim=-1)                   # along the last axis (dim=-1): e^z / sum of e^z, (3,), adds up to 1
    print("probabilities:", probs, "  sum:", probs.sum().item())
    print("loss if the right class is 0:", -math.log(probs[0].item()))
    print("loss if the right class is 2:", -math.log(probs[2].item()))

    batch_logits = logits.unsqueeze(0)               # (1, 3): F.cross_entropy wants a batch (B, C)
    target = torch.tensor([0])                       # (1,): the right class of every image, as a whole number
    print("F.cross_entropy, right class 0:", F.cross_entropy(batch_logits, target).item())   # softmax, -ln, batch mean
    print("ln 37 =", math.log(37))
    """),
    md(r"""
    Now the data. The next cell loads the subset and shows 16 random training images with their breed.
    `data` is a dict: `x_train` holds the 2 000 images as one uint8 tensor (2000, 3, 144, 144),
    `y_train` their labels (2000,) as class indices 0 to 36, `x_val` and `y_val` the 500 validation
    images, `classes` the 37 breed names (label k is `classes[k]`). `show_tf` resizes an image to
    128 px and normalises it, for display and for evaluation; section 2 looks inside it.
    """),
    api("torch.bincount", "torch.randperm", "torchkit.profile", "np.argmin"),
    api("torch.stack", "Tensor.tolist", "torch.Generator", "torchkit.show_batch", "Tensor.to", "np.arange", "np.array",
        brief=True),
    code(r"""
    data = day1.pet_tensors(144)                 # dict: x_train (2000, 3, 144, 144) uint8, y_train, x_val, y_val, classes
    classes = data["classes"]                    # the 37 breed names: label k is classes[k]
    short = []                                   # the same names cut to 16 characters, for figure titles
    for name in classes:
        if len(name) > 16:
            name = name[:15] + "."
        short.append(name)
    print("train:", tuple(data["x_train"].shape), data["x_train"].dtype, "| val:", tuple(data["x_val"].shape))
    print("images per class (train):", torch.bincount(data["y_train"]).tolist()[:10], "...")

    # 16 random training images, prepared for display (resized to 128 px, normalised)
    show_tf = day1.eval_transform(128)
    idx = torch.randperm(len(data["y_train"]), generator=torch.Generator().manual_seed(1))[:16]
    shown = []
    for i in idx:
        shown.append(show_tf(data["x_train"][i]))        # (3, 128, 128) float
    torchkit.show_batch(torch.stack(shown), data["y_train"][idx], short, n=16)
    plt.show()
    """),
    md(r"""
    **The network.** `day1.PetNet` is a small convnet of the kind built in D1-01: five 3 x 3 convolution
    layers (four of them with stride 2, each halving the image), each followed by BatchNorm (section 4)
    and a ReLU; then global average pooling (one number per channel) and one linear layer from 256
    numbers to 37 logits. `torchkit.profile` counts its parameters and its multiply-adds for one image
    (a **multiply-add** is one multiplication plus one addition, the basic operation of a layer; their
    number measures how much computation one image costs).

    **The plain run.** `day1.train_plain` trains it with no tricks at all: no augmentation, the Adam
    optimiser (section 5) with a constant learning rate of 0.003, batches of 64, `EPOCHS` epochs. After
    every epoch it measures the loss and the accuracy on the validation set, and it saves the weights
    whenever the validation accuracy improves.

    `day1.load_or_train(model, name, train_function)` first looks for `data/checkpoints/<name>.pt`. If
    the file exists, it loads the weights and the history from it (seconds); if not, it calls
    `train_function(path)` to train and save (1 to 2.5 minutes on a CPU). Either way it returns the
    **history**: a dict with one number per epoch under `train_loss`, `val_loss` and `val_metric` (the
    validation accuracy, from 0 to 1).
    """),
    code(r"""
    model_plain = day1.PetNet(num_classes=37).to(device)
    prof = torchkit.profile(model_plain, (3, IMG, IMG))
    print(f"PetNet: {prof['params']:,} parameters, {prof['macs'] / 1e6:.0f} M multiply-adds per {IMG} px image")

    def train_plain_run(checkpoint_path):
        '''Train model_plain with no regularisation at all; the best epoch is saved to checkpoint_path.'''
        return day1.train_plain(model_plain, data, epochs=EPOCHS, lr=3e-3, size=IMG, max_minutes=MAX_MIN,
                                checkpoint=checkpoint_path, device=device)

    hist_plain = day1.load_or_train(model_plain, f"d1-02_plain_{IMG}px", train_plain_run)
    print("history keys:", list(hist_plain))
    """),
    code(r"""
    # Figure: training and validation loss (left), validation accuracy (right), one point per epoch
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
    **Read the curves.** The training loss (dashed) keeps falling steadily; the validation loss (solid)
    stops improving after about five epochs and only wobbles from then on, so the gap between the two
    keeps opening (about +1.1 nats after 10 epochs). That gap is overfitting, measured. Trained for
    longer, the validation loss typically starts to *rise*. The network is learning facts about *these 2 000 images*
    (backgrounds, poses, individual animals) rather than about breeds. On the CPU budget (64 px, 10
    epochs) the plain run peaks around **12-16 %** accuracy (chance: 2.7 %); on a GPU at 128 px with
    30 epochs it peaks at **~15 %** and the gap keeps widening. The rest of this notebook is about
    closing that gap. (Spoiler: the real fix for small data is transfer learning, starting from a network
    pretrained on ImageNet; block 4 (D1-04) gets **90 %+** on the same subset that way.)
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · The input pipeline

    **Why.** Every training run of the course reads its images through the same three pieces. Knowing
    which piece does what tells you where augmentation goes (section 3), where to look when training is
    slow, and where a data bug can hide (section 6).

    | piece | job | knobs that matter |
    |---|---|---|
    | `Dataset` | `ds[i]` returns one example `(image, label)`; knows where the data is | decode once and keep in memory if the data fits |
    | `transforms.v2` | changes one image at a time: resize, crop, augment, convert to float, normalise | do cheap steps on `uint8`, convert to float late |
    | `DataLoader` | groups examples into batches, shuffles, can use several worker processes | `batch_size`, `shuffle=True` for training, `num_workers`, `pin_memory=True` on a GPU, `drop_last` |

    * A **Dataset** is any object with `len(ds)` and `ds[i]`. Here it is
      `day1.TensorImageDataset(x, y, transform)`: `ds[i]` takes image i from the uint8 tensor `x`, runs
      `transform` on it, and returns it with its label `y[i]`.
    * A **transform** is a function from one image to one (changed) image; `v2.Compose([t1, t2, ...])`
      chains several into one, applied in list order.
    * The **DataLoader** (D0-00) asks the dataset for `batch_size` examples at a time and stacks them
      into one batch. `shuffle=True` serves the examples in a new random order every epoch.
      `num_workers > 0` prepares the next batches in background processes while the model computes, so
      that a GPU never waits for JPEG decoding; on Windows and in notebooks those processes must be able
      to import your dataset and transforms (so no `lambda` functions in them). Our images are already
      decoded tensors in memory, so `num_workers=0` (everything in the notebook's own process) is fine.
      `pin_memory=True` speeds up copying batches to a GPU. `drop_last=True` skips the last, incomplete
      batch of an epoch.
    * One **step** (D0-00) is one batch: forward, loss, backward, update. With 2 000 images and batches
      of 64, an epoch is 31 steps (`drop_last` skips the 16 images left over; the next shuffle leaves
      out others).

    **Why shuffle?** Every step should see a random sample of the data, so that its correction is a
    fair estimate of the correction the whole dataset asks for. Data is often stored sorted by class
    (one folder per breed). Without shuffling, a batch then holds one or two breeds, and each step
    pushes the network towards "everything is a Bengal", the next one towards "everything is a Birman".

    **Worked example.** Sorted by breed, the first 55 training images are Abyssinians and the next 55
    American Bulldogs. An unshuffled loader's first batch of 64 is therefore 55 Abyssinians and 9
    American Bulldogs: 2 breeds. The figure counts the breeds in every batch, without and with
    shuffling.
    """),
    api("torch.argsort", "torch.unique", "np.mean"),
    api("torch.utils.data.DataLoader", "Tensor.mean", "Tensor.std", "Tensor.min", "Tensor.max", brief=True),
    code(r"""
    order = torch.argsort(data["y_train"])                   # positions that sort the labels: the data as stored on disk
    sorted_ds = day1.TensorImageDataset(data["x_train"][order], data["y_train"][order])
    # Figure: how many different breeds each batch of 64 holds, without (left) and with (right) shuffling
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), sharey=True)
    for ax, shuffle in zip(axes, (False, True)):
        dl = DataLoader(sorted_ds, batch_size=64, shuffle=shuffle, generator=torch.Generator().manual_seed(0))
        breeds_per_batch = []                                # one number per batch
        for _, y in dl:
            breeds_per_batch.append(len(torch.unique(y)))    # the number of different labels in this batch
        ax.bar(range(len(breeds_per_batch)), breeds_per_batch, color=viz.PALETTE[0 if shuffle else 3])
        ax.set_title(f"shuffle={shuffle}: distinct breeds per batch of 64 (mean {np.mean(breeds_per_batch):.1f})")
        ax.set_xlabel("batch index"); ax.grid(alpha=.3, axis="y")
    axes[0].set_ylabel("# breeds in the batch"); plt.tight_layout(); plt.show()
    """),
    md(r"""
    Left: without shuffling, every batch of 64 holds only two or three breeds (a breed has 54 or 55
    images, so 64 images in a row span two or three breeds). Right: with shuffling, each batch holds
    about 30 of the 37 breeds.

    The next cell prints the evaluation pipeline `show_tf` used so far, runs it on one image, and times
    one batch of a DataLoader built on it. Its four steps: `Resize(128)` scales the image so that its
    shorter side is 128 pixels; `CenterCrop(128)` keeps the middle 128 x 128; `ToDtype(torch.float32,
    scale=True)` converts uint8 0 ... 255 to float32 0 ... 1 (D0-00's `to_chw` divided by 255 for the
    same reason); `Normalize(mean, std)` is D0-00's `normalize` with the ImageNet numbers. Nothing in it
    is random: a validation image must look the same in every epoch.
    """),
    code(r"""
    print(show_tf)                                       # the pipeline: its four steps, in order
    x0 = data["x_train"][0]                              # one image as stored: uint8 (3, 144, 144)
    y0 = show_tf(x0)                                     # after the pipeline: float32 (3, 128, 128)
    print("in :", tuple(x0.shape), x0.dtype, f"range [{int(x0.min())}, {int(x0.max())}]")
    print("out:", tuple(y0.shape), y0.dtype, f"mean {y0.mean():.2f}, std {y0.std():.2f}  (ImageNet-normalised)")

    train_dl = DataLoader(day1.TensorImageDataset(data["x_train"], data["y_train"], show_tf), batch_size=64, shuffle=True)
    t0 = time.perf_counter()
    xb, yb = next(iter(train_dl))                        # the first batch of a new epoch
    dt = time.perf_counter() - t0
    print(f"one batch: images {tuple(xb.shape)}, labels {tuple(yb.shape)}  ({dt * 1000:.0f} ms)")
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · Data augmentation

    **Why.** Section 1's network memorised its 2 000 images. More photos would help, but labelled
    photos are expensive. **Data augmentation** makes new training images from the ones we have, by
    random changes that do *not* change the right answer: a Bengal cat cropped, mirrored or a little
    darker is still a Bengal cat. A new random version of an image is drawn every time it is used, so
    the network never sees exactly the same picture twice, and memorising stops paying off. It is the
    most effective **regulariser** for vision (section 4: anything that makes memorising harder),
    because it builds what we know about the task directly into the data. Such a change that must not
    change the answer is an **invariance** of the task: a breed does not depend on where the animal
    stands, on mirroring, or on the lighting.

    **Geometric** augmentations move pixels (crop, flip, rotate); **photometric** ones change colours
    and brightness, not positions. An automatic augmentation **policy** is a ready-made recipe that
    picks the operations and their strengths for you.

    | family | transform in `torchvision.transforms.v2` | what it does | what it teaches |
    |---|---|---|---|
    | geometric | `RandomResizedCrop(size, scale)` | cuts out a random rectangle covering a random share of the image (between `scale[0]` and `scale[1]` of the area, with a random width-to-height ratio) and resizes it to size x size | position and size do not matter |
    | | `RandomHorizontalFlip(p=0.5)` | mirrors the image left-right, with probability p | mirroring does not matter |
    | | `RandomRotation(degrees)` | rotates by a random angle between -degrees and +degrees; the corners become black | a small tilt does not matter |
    | | `RandomAffine` | rotation, shift, zoom and shear (slanting) in one | the same, combined |
    | photometric | `ColorJitter(brightness, contrast, saturation, hue)` | random changes of brightness (lighter / darker), contrast (spread between dark and light), saturation (how colourful) and hue (all colours shifted around the colour wheel) | lighting and camera do not matter |
    | | `RandomGrayscale(p)` | removes the colour, with probability p | do not rely on colour alone |
    | | `GaussianBlur` | blurs | sharpness does not matter |
    | occlusion | `RandomErasing(p)`, also called Cutout | fills a random rectangle with one value, with probability p | do not rely on one part of the animal |
    | automatic policies | `RandAugment(num_ops, magnitude)` | applies `num_ops` operations picked at random from a list of 14 (rotate, shear, colour, contrast, ...), all with the strength `magnitude` | a strong mix with two knobs instead of a hand-tuned list |
    | | `TrivialAugmentWide()` | applies one operation from a similar list, with a random strength; no knobs at all | the same, with nothing to tune |
    | | `AutoAugment()` | a fixed list of operations found by a large search on ImageNet | the same |
    | label-mixing | `MixUp`, `CutMix` (on a whole batch) | mix two images *and* their labels (later in this section) | less over-confident predictions |

    The next cell applies eight of them to one image, three random draws each. `as_np` turns a
    (C, H, W) image tensor into the (H, W, C) NumPy array `plt.imshow` expects (D0-00).
    """),
    api("torchvision.transforms.v2.Compose", "torchvision.transforms.v2.Resize", "torchvision.transforms.v2.ToDtype",
        "torchvision.transforms.v2.RandomResizedCrop", "torchvision.transforms.v2.RandomHorizontalFlip",
        "torchvision.transforms.v2.RandomRotation", "torchvision.transforms.v2.ColorJitter",
        "torchvision.transforms.v2.RandomGrayscale",
        title="Functions used here: the pipeline, geometric and colour transforms"),
    api("torchvision.transforms.v2.RandAugment", "torchvision.transforms.v2.TrivialAugmentWide",
        "torchvision.transforms.v2.RandomErasing", "Tensor.is_floating_point", "Tensor.clamp",
        title="Functions used here: automatic policies, erasing, display"),
    api("Tensor.float", "Tensor.permute", "Tensor.numpy", brief=True),
    code(r"""
    img_u8 = data["x_train"][11]                         # one training image, uint8 (3, 144, 144)
    to_f = v2.ToDtype(torch.float32, scale=True)         # uint8 0..255 -> float32 0..1
    gallery = {                                          # name -> transform; most start with Resize(128) for a fair comparison
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
        '''(C, H, W) image tensor, uint8 or float in [0, 1] -> (H, W, C) NumPy array for plt.imshow.'''
        if not t.is_floating_point():
            t = t.float() / 255                          # uint8 0..255 -> float 0..1
        t = t.clamp(0, 1)                                # keep the values plt.imshow can show
        return t.permute(1, 2, 0).numpy()

    # Figure: one column per augmentation, three random draws (rows) each
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
    Every column is the same source image, every row a new random draw. Compare the columns that move
    pixels (crop, flip, rotation) with those that only change colours, and notice that RandAugment and
    TrivialAugment pick a different operation on every draw.

    ### What *not* to do

    An augmentation is only valid if it does not change the label **for your task**. Rotating a `6` by
    180 degrees makes a `9`; mirroring a `b` makes a `d`; mirroring a chest X-ray puts the heart on the
    wrong side; an upside-down dog is not something the model will ever be shown in use (it uses up
    the network's limited capacity to store patterns rather than teaching anything). Always *look at* a batch of augmented images
    before you train. The next cell draws the three cases.
    """),
    api("torchvision.transforms.v2.functional.rotate", "torchvision.transforms.v2.functional.horizontal_flip",
        "torchvision.transforms.v2.functional.vertical_flip", "np.asarray", "Tensor.squeeze"),
    api("torch.from_numpy", "Tensor.unsqueeze", brief=True),
    code(r"""
    from PIL import Image, ImageDraw, ImageFont
    def glyph(ch, size=96):
        '''A black character on a white square, as a uint8 tensor (1, size, size).'''
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
    # Figure: six panels, each an image and the label problem an augmentation creates
    fig, axes = plt.subplots(1, 6, figsize=(13, 2.6))
    for ax, (im, t) in zip(axes, panels):
        ax.imshow(as_np(im).squeeze(), cmap="gray"); ax.set_title(t, fontsize=9); ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Exercise 1 · build the training pipeline

    **Goal.** Write `build_train_tf(size)`: it returns the augmentation pipeline that section 7 uses to
    train the recipe model.

    **Why by hand.** torchvision provides every step; the skill is choosing the steps and their order.
    A wrong order either raises an error or silently undoes a step (see the table of mistakes).

    **Input.** `size`: an int, the side of the output image (128 here; section 7 calls
    `build_train_tf(IMG)`, which is 64 on a CPU).

    **Output.** A transform: an object you call on one image. It takes a uint8 tensor (3, 144, 144) with
    values 0 to 255 and returns a float32 tensor (3, size, size), normalised: values from about -2.1 to
    2.6 (a pixel value 0 becomes (0 - 0.485) / 0.229 = -2.12 in the red channel; 255, which is 1.0
    after the conversion to float, becomes (1 - 0.406) / 0.225 = 2.64 in the blue channel). Every call
    gives a different random version.

    **Example.**
    ```
    tf = build_train_tf(128)
    out = tf(data["x_train"][11])     # uint8 (3, 144, 144) -> float32 (3, 128, 128), values about -2.1 ... 2.6
    out2 = tf(data["x_train"][11])    # the same image again: another random draw, so out2 differs from out
    ```

    **Plan.** Six transforms, in this order, then `v2.Compose` to chain them:
    1. `v2.RandomResizedCrop(size, scale=(0.35, 1.0))`: a random crop of 35 % to 100 % of the area,
       resized to size x size;
    2. `v2.RandomHorizontalFlip()`: mirror left-right, half of the time (the default p = 0.5);
    3. `v2.TrivialAugmentWide()`: one random photometric or geometric operation, of random strength;
    4. `v2.ToDtype(torch.float32, scale=True)`: uint8 0 ... 255 to float32 0 ... 1;
    5. `v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)`: D0-00's normalisation;
    6. `v2.RandomErasing(p=0.25)`: a quarter of the time, fill a random rectangle with 0.

    Why this order: steps 1 to 3 work on uint8 pixels, which is cheaper (one byte per number) and is
    what TrivialAugmentWide's colour operations expect; `Normalize` needs float input, so step 4 comes
    before it; `RandomErasing` comes last because it fills with 0, and after normalisation 0 means "the
    average ImageNet colour", a neutral grey-brown, rather than black.

    **What the checker tests.** It calls `build_train_tf(128)`, runs the pipeline 4 times on a random
    uint8 image (3, 144, 144), and checks: a tensor of shape (3, 128, 128); dtype float32; not all values
    inside [0, 1] (so it was normalised); the four outputs are not all equal (so it is random). It does
    not check which augmentations you chose or their settings: the visual check after it is for that.

    **Common mistakes.**

    | you wrote | the checker prints |
    |---|---|
    | no `v2.Normalize` | `values are in [0, 1]: did you forget v2.Normalize(mean, std)?` |
    | no `v2.ToDtype`, or `Normalize` before `ToDtype` | `your code raised TypeError: Input tensor should be a float tensor. Got torch.uint8.` |
    | `TrivialAugmentWide` after `Normalize` | `values are in [0, 1]: did you forget v2.Normalize(mean, std)?`, although `Normalize` is there: TrivialAugmentWide treats a float image as values in [0, 1] and clips the rest, undoing the normalisation |
    | `v2.ToDtype(torch.float32)` without `scale=True` | `[PASS]`: the checker does not notice, but the values go up to about 1 100 instead of 2.6 (255 is normalised as if it were 1.0) |
    | the list `[...]` returned without `v2.Compose` | `your code raised TypeError: 'list' object is not callable` |
    | no `return` | `your code raised TypeError: 'NoneType' object is not callable` |
    | `RandomResizedCrop(128, ...)` instead of `size` | `[PASS]` (the checker asks for 128), but section 7 calls `build_train_tf(64)` on a CPU and would get 128 px images |
    | `raise NotImplementedError` left in the cell | `not implemented yet: fill in the blanks, then delete the raise NotImplementedError line` |
    """),
    api("torchvision.transforms.v2.Compose", "torchvision.transforms.v2.RandomResizedCrop",
        "torchvision.transforms.v2.RandomHorizontalFlip", "torchvision.transforms.v2.TrivialAugmentWide",
        "torchvision.transforms.v2.ToDtype", "torchvision.transforms.v2.Normalize", "torchvision.transforms.v2.RandomErasing",
        title="Functions you will need"),
    exercise(
        stub=r'''
        IMAGENET_MEAN, IMAGENET_STD = torchkit.IMAGENET_MEAN, torchkit.IMAGENET_STD   # D0-00's per-channel numbers

        def build_train_tf(size=128):
            """The training augmentation pipeline.
            Returns a transform: uint8 image (3, H, W) with H, W >= size -> normalised float32 image (3, size, size)."""
            # Part A: random changes, on the uint8 image (values 0..255)
            # 1. A random crop of 35 % to 100 % of the area, resized to size x size.
            crop = ____          # YOUR CODE: a v2.RandomResizedCrop with output size `size` and scale=(0.35, 1.0)
            # 2. A left-right mirror, half of the time.
            flip = ____          # YOUR CODE: a v2.RandomHorizontalFlip with its default probability
            # 3. One random operation (colour, contrast, shear, ...) of random strength.
            trivial = ____       # YOUR CODE: a v2.TrivialAugmentWide
            # Part B: convert
            # 4. uint8 0..255 -> float32 0..1.
            to_float = ____      # YOUR CODE: a v2.ToDtype to torch.float32 that also scales (scale=True)
            # 5. Normalise every channel with the ImageNet mean and std.
            normalize = ____     # YOUR CODE: a v2.Normalize with IMAGENET_MEAN and IMAGENET_STD
            # Part C: on the normalised float image
            # 6. A quarter of the time, fill a random rectangle with 0 (= the average colour after step 5).
            erase = ____         # YOUR CODE: a v2.RandomErasing with probability 0.25
            raise NotImplementedError   # delete this line once the six blanks above are filled

            # 7. Chain the six steps, in this order, into one transform.
            pipeline = v2.Compose([crop, flip, trivial, to_float, normalize, erase])
            return pipeline

        ok_aug = tests.check_augment_pipeline(build_train_tf)
        ''',
        solution=r'''
        IMAGENET_MEAN, IMAGENET_STD = torchkit.IMAGENET_MEAN, torchkit.IMAGENET_STD   # D0-00's per-channel numbers

        def build_train_tf(size=128):
            """The training augmentation pipeline.
            Returns a transform: uint8 image (3, H, W) with H, W >= size -> normalised float32 image (3, size, size)."""
            # Part A: random changes, on the uint8 image (values 0..255)
            # 1. A random crop of 35 % to 100 % of the area, resized to size x size.
            crop = v2.RandomResizedCrop(size, scale=(0.35, 1.0))      # uint8 (3, H, W) -> uint8 (3, size, size)
            # 2. A left-right mirror, half of the time.
            flip = v2.RandomHorizontalFlip()                          # same shape
            # 3. One random operation (colour, contrast, shear, ...) of random strength.
            trivial = v2.TrivialAugmentWide()                         # same shape
            # Part B: convert
            # 4. uint8 0..255 -> float32 0..1.
            to_float = v2.ToDtype(torch.float32, scale=True)          # float32 (3, size, size), values 0..1
            # 5. Normalise every channel with the ImageNet mean and std.
            normalize = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)     # values about -2.1..2.6
            # Part C: on the normalised float image
            # 6. A quarter of the time, fill a random rectangle with 0 (= the average colour after step 5).
            erase = v2.RandomErasing(p=0.25)                          # same shape

            # 7. Chain the six steps, in this order, into one transform.
            pipeline = v2.Compose([crop, flip, trivial, to_float, normalize, erase])
            return pipeline

        ok_aug = tests.check_augment_pipeline(build_train_tf)
        '''),
    md(r"""
    **Did it work?** `[PASS]` means the pipeline has the right output format and is random. Whether its
    augmentations suit the task, only your eyes can say: the next cell shows sixteen draws of it on the
    *same* image. Every tile should still obviously be the same animal; if a tile is unrecognisable to
    *you*, the augmentation is too strong for this dataset.

    The cell starts with `tests.use_or_reference(ok, mine, reference, name)`, which every later cell of
    this notebook uses as well: it returns your function if its checker passed (`ok` is True), and
    otherwise the course's version, printing a note that says so.
    """),
    code(r"""
    make_tf = tests.use_or_reference(ok_aug, globals().get("build_train_tf"), day1.recipe_transforms, "build_train_tf")
    tf_used = make_tf(128)                                   # the pipeline for 128 px images
    torch.manual_seed(0)
    draws = []
    for k in range(16):
        draws.append(tf_used(data["x_train"][11]))           # (3, 128, 128): a new random draw each time
    torchkit.show_batch(torch.stack(draws), n=16, title="16 random draws of the training pipeline (same source image)")
    plt.show()
    """),
    md(r"""
    ### Mixup and CutMix: augmenting the *labels* too

    **Why.** The augmentations so far change an image but keep its label. Mixup and CutMix make images
    that are part one animal and part another, and give them a label that says so. A network trained on
    such images cannot be 100 % sure about anything, which stops it from pushing its logits ever higher
    on the training images (section 4 calls this over-confidence). Both are regularisers for long
    training runs. Section 4's label smoothing reuses the idea of a label that is not all-or-nothing.

    **Words.**
    * A **one-hot label** is the label written as a vector with one entry per class: 1 at the right
      class, 0 everywhere else. Class 2 out of 4 is (0, 0, 1, 0). `F.one_hot(y, num_classes)` turns a
      tensor of class indices into such rows.
    * A **soft label** (or soft target) is any vector of numbers between 0 and 1 that add up to 1, for
      example (0.7, 0, 0.3, 0): "70 % class 0, 30 % class 2". Cross-entropy accepts it: the loss
      becomes $-\sum_k q_k \ln p_k$ for the soft label q, which for a one-hot q is section 1's
      $-\ln p_{\text{true}}$. `F.cross_entropy` does this when its target is a float tensor (B, C)
      instead of class indices (B,).
    * **Mixup** (Zhang et al., 2018) blends two whole images *and* their one-hot labels with the same
      weight λ (lambda), a number between 0 and 1:

    $$\tilde x = \lambda x_i + (1-\lambda) x_j, \qquad \tilde y = \lambda y_i + (1-\lambda) y_j$$

    * λ is drawn at random from a **Beta distribution** Beta(α, α): a way of drawing a random number
      between 0 and 1 whose shape is set by α (alpha). A small α (0.2) gives λ near 0 or near 1 most of
      the time, so most images are barely mixed; α = 1 makes every λ between 0 and 1 equally likely.
    * **CutMix** (Yun et al., 2019) pastes a random rectangle of image j into image i instead of
      blending everything. λ is then the share of the area that is still image i, and the labels are
      mixed with that λ.

    Both work on a whole *batch*, after the per-image transforms: image j is another image of the same
    batch.

    **Worked example.** Image i is a cat (class 0 of 4), image j a dog (class 2), λ = 0.7. A pixel with
    the value 0.2 in image i and 0.8 in image j becomes 0.7 · 0.2 + 0.3 · 0.8 = 0.38. The label becomes
    0.7 · (1, 0, 0, 0) + 0.3 · (0, 0, 1, 0) = (0.7, 0, 0.3, 0). With CutMix on a 10 x 10 image and
    λ = 0.84, the pasted square covers 1 - 0.84 = 16 % of the area: 16 pixels, a 4 x 4 square; the label
    is (0.84, 0, 0.16, 0).

    The figure shows how α shapes the draws of λ: the higher a curve, the more often λ falls there.
    """),
    api("np.linspace", "scipy.stats.beta.pdf"),
    code(r"""
    # Figure: how often each lambda is drawn from Beta(alpha, alpha), for four values of alpha
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
    ### Exercise 2 · `mixup`

    **Goal.** Write `mixup(x, y, ...)`: mix every image of a batch with another image of the same batch,
    and mix their one-hot labels with the same weight.

    **Why by hand.** torchvision has `v2.MixUp`; writing it shows that Mixup is a few lines of
    arithmetic. Two of its pieces come back: one-hot labels in exercise 3, and `x[perm]`, the way to
    pair every element of a batch with another one.

    **Input.**
    * `x`: float tensor (B, 3, H, W), a batch of normalised images.
    * `y`: int64 tensor (B,), the class index of every image, 0 to num_classes - 1.
    * `alpha`: float, the Beta parameter (used only when `lam` is None).
    * `num_classes`: int, the number of classes (37 for Pet; the checker uses 5).
    * `lam`: a float between 0 and 1, or None. `perm`: an int64 tensor (B,) holding 0 ... B-1 in a
      shuffled order, or None. When they are None the function draws them (the first lines of the
      template, already written). The checker passes fixed values so that the result is predictable.

    **Output.** A tuple `(x_mix, y_mix)`: `x_mix` has the shape of x; `y_mix` is float (B, num_classes),
    and every row adds up to 1.

    **What `x[perm]` does.** Indexing with a tensor of positions picks those rows in that order. With
    perm = [1, 2, 0], `x[perm]` is [x[1], x[2], x[0]]: image 0's partner is image 1, image 1's partner
    image 2, image 2's partner image 0.

    **Example.** Three one-pixel grayscale "images" with the values 0, 10 and 20, labels y = [0, 3, 1]
    out of 4 classes, perm = [1, 2, 0], lam = 0.7:
    ```
    x_partner = x[perm]                  = [10, 20, 0]
    x_mix     = 0.7 * x + 0.3 * x_partner = [0.7*0 + 0.3*10, 0.7*10 + 0.3*20, 0.7*20 + 0.3*0] = [3, 13, 14]

    y_onehot = [[1, 0, 0, 0],            y_partner = y_onehot[perm] = [[0, 0, 0, 1],
                [0, 0, 0, 1],                                          [0, 1, 0, 0],
                [0, 1, 0, 0]]                                          [1, 0, 0, 0]]
    y_mix    = [[0.7, 0.0, 0.0, 0.3],
                [0.0, 0.3, 0.0, 0.7],
                [0.3, 0.7, 0.0, 0.0]]
    ```
    The cell after the exercise runs your function on exactly this example.

    **Plan.**
    1. (written) `lam`: drawn from Beta(alpha, alpha) when not given.
    2. (written) `perm`: `torch.randperm(B)` when not given.
    3. One-hot labels: `F.one_hot(y, num_classes)`, then `.float()` so that they can be mixed. (B, num_classes).
    4. `x_partner = x[perm]`, then `x_mix = lam * x + (1 - lam) * x_partner`.
    5. `y_partner = y_onehot[perm]`, then `y_mix = lam * y_onehot + (1 - lam) * y_partner`.
    6. (written) return both.

    **What the checker tests.** Six random images (6, 3, 8, 8) with the labels [0, 3, 1, 4, 2, 3] out of
    5 classes and the fixed perm [3, 0, 5, 1, 2, 4]. With lam = 0.7 and with lam = 0.25 it compares both
    outputs with the formulas. Then it calls your function once with lam and perm left out: the shapes
    must be kept and every soft-label row must add up to 1.

    **Common mistakes.**

    | you wrote | the checker prints |
    |---|---|
    | mixed the integer labels, `lam * y + (1 - lam) * y[perm]` | `soft labels should have shape (B, num_classes) = (6, 5), got (6,)` |
    | `y_onehot` instead of `y_onehot[perm]` in step 5 | `lam=0.7: soft labels differ from lam * onehot(y) + (1 - lam) * onehot(y[perm])` (and the same for lam=0.25) |
    | `x` instead of `x[perm]` in step 4 | `lam=0.7: mixed images differ from lam * x + (1 - lam) * x[perm]` |
    | `lam` and `1 - lam` swapped | both lines, for lam=0.7 and for lam=0.25 |
    | drew a new perm every time, ignoring the argument | both lines: the checker's perm was not used. Draw only when `perm is None`, as the template does |
    | `F.one_hot(y[perm], ...)` in step 3 and `[perm]` again in step 5 | `soft labels differ ...`: the labels were shuffled twice |
    | `F.one_hot(y)` without `num_classes` | `[PASS]` here by luck (the checker's labels contain the largest class, 4), but `F.one_hot(torch.tensor([0, 3, 1]))` has 4 columns, not 37: on a real batch the soft labels would be too short |
    | `return x_mix` only | `your code raised ValueError: too many values to unpack (expected 2)` |
    """),
    api("F.one_hot", "np.random.beta", title="Functions you will need"),
    exercise(
        stub=r'''
        def mixup(x, y, alpha=0.2, num_classes=37, lam=None, perm=None):
            """Mixup on one batch.
            x: (B, 3, H, W) float images; y: (B,) int64 class indices.
            Returns x_mix (B, 3, H, W) and y_mix (B, num_classes) soft labels (every row adds up to 1)."""
            # 1. The mixing weight lam, a number between 0 and 1: drawn at random unless the caller gave one.
            if lam is None:
                lam = float(np.random.beta(alpha, alpha))
            # 2. The partner of every image: a shuffled order of 0 .. B-1, unless the caller gave one.
            if perm is None:
                perm = torch.randperm(x.shape[0])
            # 3. One-hot labels: (B,) class indices -> (B, num_classes) rows of 0.0 and 1.0.
            y_onehot = ____      # YOUR CODE: F.one_hot of y with num_classes columns (whole numbers 0 and 1)
            y_onehot = ____      # YOUR CODE: the same rows converted to float, so that they can be mixed
            # 4. Mixed images: lam of every image plus (1 - lam) of its partner.
            x_partner = ____     # YOUR CODE: the partner images, x indexed with perm
            x_mix = ____         # YOUR CODE: lam times x plus (1 - lam) times x_partner
            # 5. Mixed labels: the same weights on the one-hot rows and on the partners' rows.
            y_partner = ____     # YOUR CODE: the partners' one-hot rows, y_onehot indexed with perm
            y_mix = ____         # YOUR CODE: lam times y_onehot plus (1 - lam) times y_partner
            raise NotImplementedError   # delete this line once the six blanks above are filled

            # 6. Return both.
            return x_mix, y_mix

        ok_mixup = tests.check_mixup(mixup)
        ''',
        solution=r'''
        def mixup(x, y, alpha=0.2, num_classes=37, lam=None, perm=None):
            """Mixup on one batch.
            x: (B, 3, H, W) float images; y: (B,) int64 class indices.
            Returns x_mix (B, 3, H, W) and y_mix (B, num_classes) soft labels (every row adds up to 1)."""
            # 1. The mixing weight lam, a number between 0 and 1: drawn at random unless the caller gave one.
            if lam is None:
                lam = float(np.random.beta(alpha, alpha))
            # 2. The partner of every image: a shuffled order of 0 .. B-1, unless the caller gave one.
            if perm is None:
                perm = torch.randperm(x.shape[0])
            # 3. One-hot labels: (B,) class indices -> (B, num_classes) rows of 0.0 and 1.0.
            y_onehot = F.one_hot(y, num_classes)                  # (B, num_classes) int64
            y_onehot = y_onehot.float()                           # (B, num_classes) float32
            # 4. Mixed images: lam of every image plus (1 - lam) of its partner.
            x_partner = x[perm]                                   # (B, 3, H, W)
            x_mix = lam * x + (1 - lam) * x_partner               # (B, 3, H, W)
            # 5. Mixed labels: the same weights on the one-hot rows and on the partners' rows.
            y_partner = y_onehot[perm]                            # (B, num_classes)
            y_mix = lam * y_onehot + (1 - lam) * y_partner        # (B, num_classes)

            # 6. Return both.
            return x_mix, y_mix

        ok_mixup = tests.check_mixup(mixup)
        '''),
    md(r"""
    **Check by hand.** Your function (or the course's, if yours has not passed yet) on the worked
    example above: `x_mix` must be [3, 13, 14] and `y_mix` the three rows computed there.
    """),
    api("torchvision.transforms.v2.CutMix", "torch.topk", "torch.roll"),
    api("torch.arange", "torchkit.denormalize", "Tensor.view", "Tensor.flatten", brief=True),
    code(r"""
    mix_fn = tests.use_or_reference(ok_mixup, globals().get("mixup"), day1.mixup, "mixup")
    x_tiny = torch.tensor([0.0, 10.0, 20.0]).view(3, 1, 1, 1)   # three 1-pixel grayscale images: (3, 1, 1, 1)
    y_tiny = torch.tensor([0, 3, 1])                             # their labels, out of 4 classes
    perm_tiny = torch.tensor([1, 2, 0])                          # partners: image 1, image 2, image 0
    x_mix_tiny, y_mix_tiny = mix_fn(x_tiny, y_tiny, num_classes=4, lam=0.7, perm=perm_tiny)
    print("x_mix:", x_mix_tiny.flatten().tolist())
    print("y_mix:")
    print(y_mix_tiny)
    """),
    md(r"""
    The figure below uses your function on 8 Pet images (every 250th training image; the four shown
    are four different breeds), with λ fixed at 0.6 and every image mixed with its neighbour (`torch.roll` shifts the positions by one: partner order
    7, 0, 1, ..., 6). Below it, torchvision's `v2.CutMix` on the same batch, with its own random λ and
    box. Each title lists the two largest entries of the image's soft label.
    """),
    code(r"""
    images = []
    for i in range(0, 2000, 250):                                # 8 images, 250 apart (5 different breeds)
        images.append(show_tf(data["x_train"][i]))
    xb = torch.stack(images)                                     # (8, 3, 128, 128)
    yb = data["y_train"][0:2000:250]                             # (8,)
    torch.manual_seed(4)
    partners = torch.roll(torch.arange(8), 1)                    # [7, 0, 1, 2, 3, 4, 5, 6]
    x_mx, y_mx = mix_fn(xb, yb, alpha=1.0, num_classes=37, lam=0.6, perm=partners)   # lam fixed for display
    x_cm, y_cm = v2.CutMix(num_classes=37, alpha=1.0)(xb, yb)    # CutMix draws its own lam and box

    def soft_title(soft_label):
        '''The two largest entries of a soft label (37,), as lines "weight breed".'''
        top = torch.topk(soft_label, 2)                          # the two largest values and their positions
        lines = []
        for weight, cls in zip(top.values, top.indices):
            if weight > 0.005:                                   # skip a second entry that is (almost) 0
                lines.append(f"{weight:.2f} {short[int(cls)]}")
        return "\n".join(lines)

    # Figure: top row Mixup, bottom row CutMix; four of the eight images each
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
    Top row: both animals are visible everywhere and the titles show 0.60 / 0.40, the fixed λ. Bottom
    row: `v2.CutMix` drew its own λ for the batch and a random box; each title's weights are the area
    shares of the two images, clipped boxes included, as in the lab.

    **Did it work?** `[PASS]` means your function makes exactly the batches and soft labels of the
    Mixup paper. Section 7's recipe run does not use Mixup (it pays off only in long runs); torchvision's
    `v2.MixUp` and `v2.CutMix` do the same on a batch when you need them.
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · Regularisation

    **Why.** Section 1 showed the gap: training loss down, validation loss stuck. **Regularisation**
    is any change to training that makes memorising harder or less rewarding, so that the model
    generalises better; the price is usually a somewhat higher training loss. Augmentation (section 3)
    is one kind. This section explains the others; section 7 trains with several of them at once.

    | technique | what it does | typical setting for convnets |
    |---|---|---|
    | **weight decay** | pulls every weight a little towards 0 at every step: prefers simple functions | AdamW `weight_decay` 0.05 (0.01 to 0.1); SGD 1e-4 to 5e-4; none on BatchNorm and bias parameters |
    | **dropout** | sets random numbers inside the network to 0 during training, so that it cannot rely on any single one of them | 0.1 to 0.5 before the final linear layer; rarely inside the convolution layers |
    | **BatchNorm** | normalises each channel with the statistics of the current batch; the batch-to-batch noise regularises a little | after every convolution; *train and eval modes differ* |
    | **label smoothing** | the target is about 0.9 for the right class instead of 1: no infinitely confident logits | ε (epsilon, the share of probability moved off the right class) = 0.1 |
    | **Mixup / CutMix** | soft targets from mixed images (section 3) | α 0.2 to 1.0, for long training runs |
    | **early stopping** | keep the weights from the epoch with the best validation score | always (`torchkit.fit` saves the best epoch) |

    ### Weight decay

    At every step, on top of the gradient step, every weight is shrunk by a small fraction of itself:
    $w \leftarrow w - \eta \cdot \text{wd} \cdot w$, with η the learning rate and wd the weight-decay
    coefficient. **By hand:** with η = 0.01 and wd = 0.05, a weight of 2.0 loses
    0.01 · 0.05 · 2.0 = 0.001 per step and becomes 1.999. A weight the loss does not need decays towards
    0 over thousands of steps; a weight the loss needs is pushed back up by its gradient. Small weights
    make a smoother function, one that cannot give wildly different answers for two very similar images,
    which is what memorising needs.

    In PyTorch it is the `weight_decay` argument of the optimiser:
    `torch.optim.AdamW(params, lr, weight_decay=0.05)`. AdamW is the Adam optimiser (section 5) with
    weight decay done exactly this way; section 5 also explains why `torch.optim.Adam(...,
    weight_decay=...)` is weaker. BatchNorm and bias parameters are usually left out: they only shift
    and scale, and shrinking them does not make the function smoother.

    ### Dropout

    During training, a **dropout** layer sets each number passing through it to 0 with probability p,
    and multiplies the survivors by 1 / (1 - p) so that the average stays the same. In evaluation it
    does nothing. A **unit** (or neuron) is one output number of a layer, for example one channel of a
    convolution. With dropout, a unit cannot depend on one particular other unit (the two would
    "co-adapt"), because that other unit is missing part of the time; the network has to spread what
    it knows. PetNet has a dropout layer before its last linear layer (p = 0 in section 1, p = 0.2 in
    section 7).

    **Train mode and eval mode.** Every network (`nn.Module`) has a switch: `model.train()` puts it, with
    all its layers, in **train mode**; `model.eval()` puts it in **eval mode** (evaluation). In our
    networks only two kinds of layers care: dropout and BatchNorm. `torchkit.fit` calls `model.train()`
    before every training epoch and `model.eval()` for every validation pass.

    **By hand:** with p = 0.5, each of eight 1s comes out as 0 or as 1 / (1 - 0.5) = 2, about half of
    each; in eval mode all eight stay 1.
    """),
    api("nn.Dropout", "Module.train", "Module.eval"),
    api("torch.ones", brief=True),
    code(r"""
    torch.manual_seed(0)
    drop = nn.Dropout(p=0.5)                                 # a dropout layer on its own
    ones = torch.ones(8)
    drop.train()                                             # train mode: drop and rescale
    print("train mode:", drop(ones))
    drop.eval()                                              # eval mode: pass everything through
    print("eval mode: ", drop(ones))
    """),
    md(r"""
    Here 5 of the 8 numbers were dropped and the other 3 doubled: the choice is random, half on average.

    ### BatchNorm has two modes

    **BatchNorm** (batch normalisation; `nn.BatchNorm2d` after a convolution) normalises every channel
    the way D0-00 normalised images, with numbers taken from the current batch: for one channel, it
    takes that channel's values over all images and positions of the batch, subtracts their mean and
    divides by their standard deviation. Then it multiplies by a learned scale γ (gamma, starts at 1)
    and adds a learned shift β (beta, starts at 0). This keeps the numbers inside a deep network on a
    steady scale, which lets it train faster and with larger learning rates.

    * In **train mode** it uses the mean and the **variance** (the square of the standard deviation) *of
      the current batch*, and updates two running averages, `running_mean` and `running_var`:
      new = 0.9 · old + 0.1 · this batch's value (`momentum=0.1` is the weight of the new value; it has
      nothing to do with the optimiser's momentum of section 5).
    * In **eval mode** it uses those running averages instead, so that an image's output does not depend
      on which other images happen to share its batch.

    **By hand:** one channel, a batch of four values 1, 2, 3, 6. The mean is (1 + 2 + 3 + 6) / 4 = 3;
    the squared distances from it are 4, 1, 0, 9, their average 3.5, the standard deviation
    √3.5 = 1.87. Normalised: (1 - 3) / 1.87 = -1.07, then -0.53, 0, 1.60. The running mean moves from its
    start 0 a tenth of the way to 3: 0.9 · 0 + 0.1 · 3 = 0.3.
    """),
    api("nn.BatchNorm1d", "torch.cat", "Tensor.sqrt", "Tensor.ravel", "Tensor.abs"),
    api("torch.randn", "Tensor.detach", "torch.no_grad", brief=True),
    code(r"""
    bn_demo = nn.BatchNorm1d(1)                              # BatchNorm for vectors, 1 channel: gamma = 1, beta = 0
    values = torch.tensor([[1.0], [2.0], [3.0], [6.0]])      # (4, 1): a batch of four, one channel
    bn_demo.train()
    out = bn_demo(values)                                    # (4, 1)
    print("train mode output:", out.detach().flatten())
    print("running mean after this one batch:", bn_demo.running_mean.item())
    """),
    md(r"""
    The next figure makes the two modes visible with a BatchNorm layer for one channel, fed numbers
    drawn from a normal distribution with mean 3 and standard deviation 2:
    * left: 40 training batches of 32 numbers; the running mean and the running std (the square root of
      the running variance) settle at the true 3 and 2;
    * middle: the same number, 4.0, put into 12 different random batches of 4. In train mode its output
      changes with its batch-mates; in eval mode it is always the same;
    * right: a batch from a distribution shifted up by 1. Train mode normalises with this batch's own
      mean, so the shift disappears from the output; eval mode keeps it.

    Two consequences:
    * Forget `model.eval()` before predicting and a prediction depends on *which other images are in
      the batch*.
    * Tiny batches (1 to 4 images) give noisy statistics: training gets worse, and the eval-mode outputs
      (running averages) differ from what the network saw during training.
    """),
    code(r"""
    torch.manual_seed(0)
    bn = nn.BatchNorm1d(1, momentum=0.1)
    mu_true, sd_true = 3.0, 2.0                              # the data: normal numbers with mean 3 and std 2
    track = []                                               # (running mean, running std) after every batch
    bn.train()
    for step in range(40):                                   # 40 training batches of 32 numbers
        bn(mu_true + sd_true * torch.randn(32, 1))
        running_std = bn.running_var.sqrt().item()
        track.append((bn.running_mean.item(), running_std))
    track = np.array(track)                                  # (40, 2)

    probe = torch.tensor([[4.0]])                            # one fixed number
    outs_train = []
    for k in range(12):                                      # ...put into 12 different random batches of 4
        batch = torch.cat([probe, mu_true + sd_true * torch.randn(3, 1)])   # (4, 1): the probe first
        bn.train()
        outs_train.append(bn(batch)[0, 0].item())            # the probe's output in train mode
    bn.eval()
    out_eval = bn(probe)[0, 0].item()                        # the probe's output in eval mode

    shifted = mu_true + 1.0 + sd_true * torch.randn(256, 1)  # a batch from a distribution shifted up by 1
    bn.train()
    h_train = bn(shifted).detach().numpy().ravel()           # (256,) outputs in train mode
    bn.eval()
    h_eval = bn(shifted).detach().numpy().ravel()            # (256,) outputs in eval mode

    # Figure: running statistics (left), one number in 12 batches (middle), a shifted batch (right)
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
    md(r"""
    The same bug in a real network: a fresh PetNet computes the logits of one validation image alone,
    and of the same image inside a batch of 8, once in train mode and once in eval mode.
    """),
    code(r"""
    net = day1.PetNet().to(device)
    images = []
    for i in range(8):
        images.append(show_tf(data["x_val"][i]))             # 8 validation images, (3, 128, 128) each
    with torch.no_grad():                                    # no gradients needed: we only look at outputs
        xb = torch.stack(images).to(device)                  # (8, 3, 128, 128)
        for mode in ("train", "eval"):
            net.train(mode == "train")                       # train(True) = train mode, train(False) = eval mode
            alone = net(xb[:1])[0]                           # logits of image 0 alone in its batch: (37,)
            in_batch = net(xb)[0]                            # logits of image 0 inside the batch of 8: (37,)
            largest_change = (alone - in_batch).abs().max()  # the biggest difference over the 37 logits
            print(f"{mode:5s} mode: max |logits(alone) - logits(in a batch of 8)| = {largest_change:.4f}")
    """),
    md(r"""
    In train mode, the same image gets different logits depending on its batch-mates (here they change
    by up to 0.2043); in eval mode the difference is 0. Always call `model.eval()` before validating or predicting, and `model.train()`
    before training again (exercise 5).

    ### Label smoothing

    **Why.** With a one-hot target, the cross-entropy $-\ln p_{\text{true}}$ (section 1) reaches its
    minimum 0 only when $p_{\text{true}} = 1$, and softmax gives exactly 1 only when the right class's
    logit is infinitely far above the others. So on training images it already classifies correctly,
    the network keeps widening the gap between the logits, step after step. It becomes
    **over-confident**: 99.9 % sure, also when it is wrong on new images. That is a form of overfitting.

    **Label smoothing** (Szegedy et al., 2016) gives the network a target it can reach: take a little
    probability ε (epsilon) away from the right class and spread it evenly over all C classes:

    $$q_k = (1 - \varepsilon)\,[k = y] + \frac{\varepsilon}{C}$$

    where $[k = y]$ is 1 for the right class y and 0 for the others (the one-hot label). The loss is
    cross-entropy against this soft label: $\mathcal{L} = -\sum_k q_k \ln p_k$. Its minimum is at
    $p = q$: a finite gap between the logits.

    **By hand.** C = 4 classes, ε = 0.1, right class 2. Every class gets ε / C = 0.025; the right class
    gets 1 - 0.1 = 0.9 on top: q = (0.025, 0.025, 0.925, 0.025), which adds up to 1. With 37 classes and
    ε = 0.1: 0.9027 for the right class and 0.0027 for each of the other 36.

    Now the loss for the logits (0, 0, 2, 0): softmax gives p = (0.096, 0.096, 0.711, 0.096), so
    ln p = (-2.341, -2.341, -0.341, -2.341). Plain cross-entropy: $-\ln 0.711 = 0.341$. Smoothed:
    0.925 · 0.341 + 3 · 0.025 · 2.341 = 0.315 + 0.176 = 0.491. The smoothed loss is higher, because it
    also asks for a little probability on the wrong classes.

    **Log-softmax.** `F.log_softmax(logits, dim=-1)` computes ln p in one step. It gives the same numbers
    as the log of the softmax, but stays accurate when a probability is tiny (the log of a probability
    that was rounded to 0 would be minus infinity).

    ### Exercise 3 · `label_smoothing_ce`

    **Goal.** Write the cross-entropy loss with label smoothing, for a batch of logits and integer
    labels.

    **Why by hand.** PyTorch has it built in, `F.cross_entropy(logits, target, label_smoothing=eps)`,
    and the checker compares with it. Writing it shows that label smoothing is only a different target
    vector, and that cross-entropy against any soft target (Mixup's too) is "minus the sum of target
    times log-probability". Section 7's recipe trains with your function.

    **Input.** `logits`: float tensor (B, C), raw scores for B images and C classes. `target`: int64
    tensor (B,), the right class of each image, 0 to C - 1. `eps`: a float such as 0.1.

    **Output.** One number: the batch average of $-\sum_k q_k \ln p_k$, as a tensor with no axes (shape
    `()`), so that `.backward()` works on it.

    **Example** (the hand example as a batch of one):
    ```
    label_smoothing_ce(torch.tensor([[0., 0., 2., 0.]]), torch.tensor([2]), eps=0.1)   ->  tensor(0.4908)
    label_smoothing_ce(torch.tensor([[0., 0., 2., 0.]]), torch.tensor([2]), eps=0.0)   ->  tensor(0.3408)   plain cross-entropy
    ```

    **Plan.**
    1. (written) `C` and `logp = F.log_softmax(logits, dim=-1)`, shape (B, C).
    2. One-hot targets: `F.one_hot(target, C)`, then `.float()`. (B, C).
    3. `q = (1 - eps) * onehot + eps / C`. (B, C); every row adds up to 1.
    4. `weighted = q * logp`, element by element. (B, C).
    5. `per_example = -weighted.sum(dim=-1)`: the sum over the classes (the last axis), with a minus
       sign. (B,).
    6. `loss = per_example.mean()`: the average over the batch. Shape ().

    **What the checker tests.** Random logits (16, 37) and random labels, for ε = 0, 0.1 and 0.3. It
    compares your number with `F.cross_entropy(logits, target, label_smoothing=eps)`, to within 0.01 %.
    With ε = 0 your function must give plain cross-entropy.

    **Common mistakes.**

    | you wrote | the checker prints |
    |---|---|
    | `q = onehot + eps / C` (no `1 - eps`) | `eps=0.1: got 7.41940, expected 6.74902`: the rows of q add up to 1 + eps |
    | `eps / (C - 1)` on the other classes only (a definition some papers use) | `eps=0.1: got 6.75027, expected 6.74902`: close, but PyTorch spreads eps over all C classes |
    | no minus sign in step 5 | `eps=0.0: got -6.70386, expected 6.70386` |
    | `.sum()` over everything in step 5 | `eps=0.0: got 107.26168, expected 6.70386`: the sum of 16 examples' losses, not their mean |
    | `.mean()` over everything instead of the sum over the classes | `eps=0.0: got 0.18119, expected 6.70386`: divided by 37 as well |
    | `F.softmax` instead of `F.log_softmax` | `eps=0.0: got -0.01190, expected 6.70386` |
    | `F.log_softmax(logits, dim=0)` | `eps=0.0: got 5.38471, expected 6.70386`: normalised over the batch instead of over the classes |
    | the raw `logits` instead of `logp` | `eps=0.0: got -0.63280, expected 6.70386` |
    """),
    api("F.log_softmax", "torch.full_like", "Tensor.scatter_", "np.full", "np.log"),
    api("F.one_hot", "Tensor.float", "Tensor.sum", "Tensor.mean", title="Functions you will need", brief=True),
    exercise(
        stub=r'''
        def label_smoothing_ce(logits, target, eps=0.1):
            """Cross-entropy against label-smoothed targets.
            logits: (B, C) raw scores; target: (B,) int64 class indices; eps: the smoothing, e.g. 0.1.
            Returns the mean loss over the batch, a tensor of shape ()."""
            # 1. The number of classes C and the log-probabilities ln p.
            C = logits.shape[-1]
            logp = F.log_softmax(logits, dim=-1)                  # (B, C)
            # 2. One-hot targets, as float.
            onehot = ____        # YOUR CODE: F.one_hot of target with C columns
            onehot = ____        # YOUR CODE: the same rows converted to float
            # 3. The smoothed targets q: 1 - eps on the right class, plus eps / C on every class.
            q = ____             # YOUR CODE: (1 - eps) times onehot, plus eps / C. (B, C), every row adds up to 1
            # 4. q times ln p, element by element.
            weighted = ____      # YOUR CODE: (B, C)
            # 5. The loss of every example: minus the sum over the classes (the last axis).
            per_example = ____   # YOUR CODE: (B,)
            # 6. The average over the batch.
            loss = ____          # YOUR CODE: shape ()
            raise NotImplementedError   # delete this line once the six blanks above are filled
            return loss

        ok_ls = tests.check_label_smoothing(label_smoothing_ce)
        ''',
        solution=r'''
        def label_smoothing_ce(logits, target, eps=0.1):
            """Cross-entropy against label-smoothed targets.
            logits: (B, C) raw scores; target: (B,) int64 class indices; eps: the smoothing, e.g. 0.1.
            Returns the mean loss over the batch, a tensor of shape ()."""
            # 1. The number of classes C and the log-probabilities ln p.
            C = logits.shape[-1]
            logp = F.log_softmax(logits, dim=-1)                  # (B, C)
            # 2. One-hot targets, as float.
            onehot = F.one_hot(target, C)                         # (B, C) int64
            onehot = onehot.float()                               # (B, C) float32
            # 3. The smoothed targets q: 1 - eps on the right class, plus eps / C on every class.
            q = (1 - eps) * onehot + eps / C                      # (B, C)
            # 4. q times ln p, element by element.
            weighted = q * logp                                   # (B, C)
            # 5. The loss of every example: minus the sum over the classes (the last axis).
            per_example = -weighted.sum(dim=-1)                   # (B,)
            # 6. The average over the batch.
            loss = per_example.mean()                             # ()
            return loss

        ok_ls = tests.check_label_smoothing(label_smoothing_ce)
        '''),
    md(r"""
    **Check by hand.** Your loss (or the course's) on the hand example: 0.4908 with ε = 0.1, and 0.3408,
    plain cross-entropy, with ε = 0.
    """),
    code(r"""
    ls_fn = tests.use_or_reference(ok_ls, globals().get("label_smoothing_ce"), day1.label_smoothing_ce, "label_smoothing_ce")
    hand_logits = torch.tensor([[0.0, 0.0, 2.0, 0.0]])          # (1, 4): one image, 4 classes
    hand_target = torch.tensor([2])                             # (1,): its right class
    print("eps 0.1:", ls_fn(hand_logits, hand_target, 0.1).item())
    print("eps 0.0:", ls_fn(hand_logits, hand_target, 0.0).item())
    """),
    md(r"""
    <details>
    <summary><b>More: the same targets with <code>scatter_</code></b> (optional, click to open)</summary>

    Many code bases build q without a one-hot matrix: fill a tensor with ε / C, then write
    1 - ε + ε / C into the right column of every row. `q.scatter_(dim, index, value)` writes `value` into
    `q` at the positions `index` along `dim`, in place (the underscore):
    ```
    q = torch.full_like(logp, eps / C)                       # (B, C), every entry eps / C
    q.scatter_(-1, target.unsqueeze(-1), 1 - eps + eps / C)  # row b: column target[b] gets 1 - eps + eps / C
    loss = -(q * logp).sum(-1).mean()
    ```
    `target.unsqueeze(-1)` has the shape (B, 1): one column position per row. The result is the same as
    the one-hot version.

    </details>

    **Did it work?** `[PASS]` means your loss matches PyTorch's built-in label smoothing to 0.01 %.
    Section 7's recipe run uses it. What it changes: the figure shows, left, the target q for ε = 0 and
    ε = 0.1 (log scale, first 8 of 37 classes, right class 2): ε moves a tenth of the probability onto
    the other classes. Right, the loss as a function of the probability the model gives the right class
    (the rest spread evenly over the others): for ε = 0 it falls all the way to p = 1; with ε = 0.1 its
    minimum (dot) is at p = 1 - ε + ε / C = 0.90, with ε = 0.3 at 0.71.
    """),
    code(r"""
    # Figure: the smoothed target (left) and where the smoothed loss has its minimum (right)
    C = 37
    p = np.linspace(0.02, 0.999, 500)                        # probability assigned to the true class
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    k = np.arange(8)
    for off, eps, c in [(-0.2, 0.0, viz.PALETTE[0]), (0.2, 0.1, viz.PALETTE[1])]:
        q = np.full(8, eps / C)                              # eps / C for every class
        q[2] += 1 - eps                                      # plus 1 - eps for the true class 2
        axes[0].bar(k + off, q, width=0.4, color=c, label=f"eps = {eps}")
    axes[0].set_yscale("log"); axes[0].set_ylim(1e-3, 1.5); axes[0].set_xlabel("class (first 8 of 37; true class = 2)")
    axes[0].set_title("target distribution q (log scale)"); axes[0].legend()
    for eps, c in [(0.0, viz.PALETTE[0]), (0.1, viz.PALETTE[1]), (0.3, viz.PALETTE[2])]:
        qt, qo = 1 - eps + eps / C, eps / C                  # target of the true class, of every other class
        loss = -qt * np.log(p) - (C - 1) * qo * np.log((1 - p) / (C - 1))
        axes[1].plot(p, loss, color=c, label=f"eps = {eps}")
        i = np.argmin(loss)                                  # where this loss is lowest
        axes[1].plot(p[i], loss[i], "o", color=c)
    axes[1].set_ylim(0, 4); axes[1].set_xlabel("p(true class)"); axes[1].set_ylabel("loss")
    axes[1].set_title("minimum moves from p = 1 to p = 1 - eps + eps/C (dots)"); axes[1].legend()
    for ax in axes: ax.grid(alpha=.3)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Early stopping

    **Early stopping** means keeping the weights from the epoch with the best validation score, not the
    last ones (and, in long runs, stopping once the validation score has not improved for a while).
    `torchkit.fit` saves a checkpoint every time the validation accuracy improves, and
    `day1.load_or_train` loads that best epoch, so every training run of this notebook stops early in
    this sense.
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · Optimisers and learning-rate schedules

    **Why.** The **optimiser** (D0-00) turns the gradients into a change of the weights. Its update
    rule, and how large its steps are at each moment of training (the **learning rate** and its
    **schedule**), decide whether training is fast, slow, or diverges. Section 7's recipe uses AdamW
    with warm-up and cosine decay; this section builds both.

    **Three update rules.** For one weight w with gradient g and learning rate η (eta):
    * **SGD** (D0-00): $w \leftarrow w - \eta\, g$. The step is proportional to the gradient.
    * **SGD with momentum**: keep a **velocity** v, a running sum of the past gradients in which old ones
      fade: $v \leftarrow \mu v + g$, then $w \leftarrow w - \eta\, v$, with μ (mu) = 0.9 typically.
      Where the gradient keeps pointing the same way, v grows: with g = 1 at every step,
      v = 1, 1.9, 2.71, 3.44, ... towards 1 / (1 - 0.9) = 10, so the steps become up to 10 times larger.
      Where the gradient flips sign from step to step (a zigzag), the contributions cancel.
    * **Adam**: keep, for every weight, a running average of its gradients (m) and of their squares
      (v), and step by $\eta\, m / (\sqrt{v} + \epsilon)$, with ε a tiny number that avoids dividing by
      0 (not the ε of label smoothing). Dividing by the typical size of the gradient makes every weight move by about η per step,
      however large or small its gradient is. **AdamW** is Adam with weight decay done as in section 4.

    **By hand:** two weights with the gradients 100 and 0.01, η = 0.01. SGD moves them by
    0.01 · 100 = 1.0 and by 0.01 · 0.01 = 0.0001: a factor 10 000 apart. Adam's first step moves both
    by 0.01. The next cell checks this.
    """),
    api("torch.optim.Adam"),
    api("torch.optim.SGD", "torch.zeros", "Tensor.backward", "Optimizer.step", brief=True),
    code(r"""
    for name, optimizer_class in [("SGD ", torch.optim.SGD), ("Adam", torch.optim.Adam)]:
        w = torch.zeros(2, requires_grad=True)               # two weights, both 0
        opt = optimizer_class([w], lr=0.01)
        loss = 100.0 * w[0] + 0.01 * w[1]                    # a loss whose gradient is (100, 0.01)
        loss.backward()
        opt.step()                                           # one update
        moved_0 = w[0].item()                                # how far weight 0 moved (it started at 0)
        moved_1 = w[1].item()
        print(f"{name}: weight 0 moved by {moved_0:+.4f}, weight 1 moved by {moved_1:+.6f}")
    """),
    md(r"""
    <details>
    <summary><b>More: AdamW, not Adam + L2</b> (optional, click to open)</summary>

    The older way to decay weights is an **L2 penalty**: add $\frac{\lambda}{2}\|w\|^2$ (λ/2 times the
    sum of the squared weights) to the loss. Its gradient is $\lambda w$. With SGD that is the same as
    weight decay. With plain Adam, the $\lambda w$ is added to the gradient and then divided by Adam's
    per-parameter scale $\sqrt{\hat v}$: weights with large gradients get almost no decay. (Here $\lambda$ is the weight-decay coefficient, not Mixup's
    $\lambda$; $\hat m$ and $\hat v$ are Adam's running averages m and v.) AdamW (Loshchilov & Hutter)
    *decouples* the decay from the gradient statistics:

    $$w \leftarrow w - \eta \left( \frac{\hat m}{\sqrt{\hat v} + \epsilon} + \lambda w \right)$$

    In PyTorch: `torch.optim.AdamW(params, lr, weight_decay=0.05)`. `torch.optim.Adam(weight_decay=...)`
    is the coupled L2 version.

    </details>

    | | SGD + momentum (0.9) | AdamW |
    |---|---|---|
    | update | velocity = running sum of gradients | per-parameter step size from the running gradient mean and its running square |
    | sensitivity to the learning rate | high; needs warm-up and decay | lower; works out of the box for most networks |
    | typical learning rate (batch of 256) | 0.1 | 1e-3 (convnets), 5e-4 to 1e-3 with warm-up (transformers) |
    | where it wins | long ImageNet training of ResNets (often slightly better final accuracy) | fine-tuning, transformers, quick experiments, small data |

    ResNets (D1-03) are the classic deep convnets; transformers are another family of networks, met in
    D1-03 and again on Days 2 and 3; **fine-tuning** (D1-04) means training a pretrained network a
    little further on new data.

    The figure runs three optimisers for 40 steps on a loss with two parameters, shaped like an
    elongated bowl: 40 times steeper in one direction than in the other. Such an **ill-conditioned**
    surface is the typical shape of a loss near a minimum. The grey lines are contour lines (equal
    loss); the star is the minimum; each coloured path shows where the two parameters were after every
    step, and the legend gives the loss after 40 steps.
    """),
    api("np.meshgrid", "np.geomspace", "Tensor.clone"),
    api("torch.optim.AdamW", "Optimizer.zero_grad", brief=True),
    code(r"""
    def bowl(w):
        '''A loss with two parameters, 40 times steeper along w[1] than along w[0]; lowest at (0, 0).'''
        return 0.5 * (0.05 * w[0] ** 2 + 2.0 * w[1] ** 2)

    runs = {"SGD lr 0.45": (torch.optim.SGD, dict(lr=0.45)),          # name -> (optimiser class, its settings)
            "SGD + momentum 0.9, lr 0.1": (torch.optim.SGD, dict(lr=0.1, momentum=0.9)),
            "AdamW lr 0.3": (torch.optim.AdamW, dict(lr=0.3, weight_decay=0.0))}
    gx, gy = np.meshgrid(np.linspace(-11, 2.5, 200), np.linspace(-3, 3, 200))   # a grid of points for the contour lines
    # Figure: contour lines of the bowl and the path of every optimiser
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.contour(gx, gy, 0.5 * (0.05 * gx ** 2 + 2.0 * gy ** 2), levels=np.geomspace(0.01, 12, 14), colors="lightgray")
    for (name, (cls, kw)), c in zip(runs.items(), viz.PALETTE):
        w = torch.tensor([-10.0, 2.0], requires_grad=True)       # the two parameters, starting far from the minimum
        opt = cls([w], **kw)
        path = [w.detach().clone()]                              # where w was before each step
        for _ in range(40):                                      # 40 steps of the training loop
            opt.zero_grad()
            loss = bowl(w)
            loss.backward()
            opt.step()
            path.append(w.detach().clone())
        path = torch.stack(path).numpy()                         # (41, 2)
        final_loss = bowl(torch.tensor(path[-1]))
        ax.plot(path[:, 0], path[:, 1], "-o", ms=2.5, color=c, label=f"{name}: loss after 40 steps {final_loss:.3f}")
    ax.plot(0, 0, "k*", ms=12); ax.set_title("40 steps on an ill-conditioned bowl"); ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-11, 2.5)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Plain SGD has to keep its learning rate small enough not to diverge along the steep axis, so it
    crawls along the flat one. Momentum accumulates the consistent direction; Adam rescales each
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
    ### Learning-rate schedules

    Neither momentum nor Adam removes the need for a good **learning-rate schedule**: a rule that
    changes the learning rate during training, step by step.

    * **Warm-up**: start with a tiny learning rate and raise it in a straight line to the peak over the
      first few hundred to few thousand steps. At the start the weights are random, the gradients are
      large and noisy, and BatchNorm's running statistics and Adam's running averages are not settled
      yet: full-size steps there can throw training off.
    * **Decay**: lower the learning rate towards 0 at the end. Large steps explore; small steps settle
      into a minimum. **Cosine decay** is today's default: from the peak $\eta_{\max}$ down to 0 along
      half a cosine wave, $\eta_t = \tfrac12 \eta_{\max}\,(1 + \cos(\pi\, t / T))$ at step t of T. It is
      flat at the start, steepest in the middle and flat again at the end.
    * **Step decay**, the older way: divide the learning rate by 10 at a few fixed epochs.
    * **OneCycle** (Smith): warm up to a high peak, then **anneal** (lower gradually) to far below the
      starting value, with the momentum moved the opposite way. Fast convergence in few epochs.

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
    ### Exercise 4 · `warmup_cosine`

    **Goal.** A function that gives the learning rate at any step of a run with linear warm-up followed
    by cosine decay. Section 7 trains the recipe model with it.

    **Why by hand.** Most modern recipes use this schedule, and PyTorch has no single built-in for it.
    In section 7, the scheduler `torch.optim.lr_scheduler.LambdaLR` calls your function at every step to
    set the learning rate.

    **Input.** `step`: the step number, an int from 0 to `total`. `total`: the number of steps of the
    whole run. `warmup`: the number of warm-up steps, 0 < warmup < total. `lr_max`: the peak learning
    rate, a float.

    **Output.** The learning rate at `step`, a Python float:
    * warm-up, `step < warmup`: $\eta = \eta_{\max} \cdot (\text{step} + 1) / \text{warmup}$, rising in
      equal increments so that the last warm-up step, warmup - 1, reaches $\eta_{\max}$;
    * cosine, `step >= warmup`: the progress $p = (\text{step} - \text{warmup}) / (\text{total} -
      \text{warmup})$ goes from 0 (at step = warmup) to 1 (at step = total), and
      $\eta = \tfrac12\, \eta_{\max}\, (1 + \cos(\pi p))$ goes from $\eta_{\max}$ down to 0.

    **By hand**, with the checker's numbers total = 100, warmup = 10, lr_max = 0.1:

    | step | phase | computation | learning rate |
    |---|---|---|---|
    | 0 | warm-up | 0.1 · 1 / 10 | 0.01 |
    | 4 | warm-up | 0.1 · 5 / 10 | 0.05 |
    | 9 | warm-up | 0.1 · 10 / 10 | 0.1 |
    | 10 | cosine | p = 0 / 90 = 0, cos 0 = 1: 0.5 · 0.1 · (1 + 1) | 0.1 |
    | 55 | cosine | p = 45 / 90 = 0.5, cos(0.5 π) = 0: 0.5 · 0.1 · (1 + 0) | 0.05 |
    | 85 | cosine | p = 75 / 90 = 0.833, cos(0.833 π) = -0.866: 0.5 · 0.1 · 0.134 | 0.0067 |
    | 100 | cosine | p = 1, cos π = -1: 0.5 · 0.1 · 0 | 0 |

    **Plan.**
    1. If `step < warmup`: `lr = lr_max * (step + 1) / warmup`.
    2. Otherwise: `steps_done = step - warmup`; `cosine_length = total - warmup`;
       `progress = steps_done / cosine_length`; `cosine = math.cos(math.pi * progress)` (angles in
       radians: `math.pi` is π); `lr = 0.5 * lr_max * (1 + cosine)`.
    3. (written) return `lr`.

    **What the checker tests.** It calls your function for every step from 0 to 100 with total = 100,
    warmup = 10 and lr_max = 0.1, and compares with the formulas above (it also accepts the warm-up
    `lr_max * step / warmup`, which starts at exactly 0). Then it checks that step 10 gives 0.1 and step
    100 gives 0. In the lab, with "final LR / peak" at 0, the formula box shows the value your function
    must return for any step.

    **Common mistakes.**

    | you wrote | the checker prints |
    |---|---|
    | `step <= warmup` | `cosine part: at step 10 got 0.11000, expected 0.10000` |
    | `progress = steps_done / total` | `cosine part: at step 71 got 0.03306, expected 0.02350` |
    | `progress = step / total` (warm-up not subtracted) | `cosine part: at step 39 got 0.06694, expected 0.07650` |
    | no `0.5` | `cosine part: at step 10 got 0.20000, expected 0.10000` |
    | `math.cos(progress)`, without π | `cosine part: at step 93 got 0.08020, expected 0.00149` |
    | `math.cos(180 * progress)`, degrees instead of radians | `cosine part: at step 98 got 0.09987, expected 0.00012` |
    | `//` instead of `/` | `cosine part: at step 99 got 0.10000, expected 0.00003`: whole-number division keeps the progress at 0 until the very end |
    | `1 - cosine` | `cosine part: at step 10 got 0.00000, expected 0.10000`: the curve rises instead of falling |
    | no warm-up branch | `warm-up should rise linearly to lr_max: got [0.097  0.0976 ...` |
    """),
    exercise(
        stub=r'''
        def warmup_cosine(step, total, warmup, lr_max):
            """Learning rate at `step` (0 .. total) for a linear warm-up followed by cosine decay.
            step, total, warmup: ints with 0 < warmup < total; lr_max: the peak learning rate.
            Returns a Python float."""
            if step < warmup:
                # Phase 1, warm-up: steps 0 .. warmup-1 rise in equal increments up to lr_max.
                lr = ____                # YOUR CODE: lr_max times (step + 1) / warmup
            else:
                # Phase 2, cosine decay: from lr_max at step == warmup down to 0 at step == total.
                steps_done = ____        # YOUR CODE: how many steps after the end of the warm-up `step` is
                cosine_length = ____     # YOUR CODE: how many steps the cosine phase has in total
                progress = ____          # YOUR CODE: steps_done / cosine_length, from 0.0 to 1.0
                cosine = ____            # YOUR CODE: math.cos of math.pi times progress, from 1.0 down to -1.0
                lr = ____                # YOUR CODE: 0.5 times lr_max times (1 + cosine)
            raise NotImplementedError   # delete this line once the six blanks above are filled
            return lr

        ok_sched = tests.check_warmup_cosine(warmup_cosine)
        ''',
        solution=r'''
        def warmup_cosine(step, total, warmup, lr_max):
            """Learning rate at `step` (0 .. total) for a linear warm-up followed by cosine decay.
            step, total, warmup: ints with 0 < warmup < total; lr_max: the peak learning rate.
            Returns a Python float."""
            if step < warmup:
                # Phase 1, warm-up: steps 0 .. warmup-1 rise in equal increments up to lr_max.
                lr = lr_max * (step + 1) / warmup               # step 0: lr_max / warmup; step warmup-1: lr_max
            else:
                # Phase 2, cosine decay: from lr_max at step == warmup down to 0 at step == total.
                steps_done = step - warmup                      # 0 at the first cosine step
                cosine_length = total - warmup                  # e.g. 100 - 10 = 90
                progress = steps_done / cosine_length           # 0.0 .. 1.0
                cosine = math.cos(math.pi * progress)           # 1.0 .. -1.0
                lr = 0.5 * lr_max * (1 + cosine)                # lr_max .. 0.0
            return lr

        ok_sched = tests.check_warmup_cosine(warmup_cosine)
        '''),
    md(r"""
    **Did it work?** `[PASS]` means your schedule gives the right learning rate at every step. The next
    figure draws it (thick line, peak 1, 2 warm-up epochs) next to the schedules PyTorch provides, over
    20 epochs of 50 steps. In PyTorch a **scheduler** is an object attached to an optimiser that changes
    its learning rate each time you call `scheduler.step()`. `lr_curve` builds an optimiser and a
    scheduler and records the learning rate at every step, the way a training loop would see it.
    """),
    api("torch.optim.lr_scheduler.MultiStepLR", "torch.optim.lr_scheduler.CosineAnnealingLR",
        "Optimizer.param_groups", "np.ones"),
    api("torch.optim.lr_scheduler.OneCycleLR", brief=True),
    code(r"""
    sched_fn = tests.use_or_reference(ok_sched, globals().get("warmup_cosine"), day1.warmup_cosine, "warmup_cosine")
    epochs, spe = 20, 50                                     # 20 epochs x 50 steps per epoch
    T = epochs * spe                                         # 1 000 steps in all
    steps = np.arange(T)

    def lr_curve(scheduler_class, **settings):
        '''The learning rate at each of the T steps under a PyTorch scheduler (peak learning rate 1.0).'''
        dummy = torch.zeros(1, requires_grad=True)           # a scheduler needs an optimiser, an optimiser a parameter
        opt = torch.optim.SGD([dummy], lr=1.0, momentum=0.9)
        sched = scheduler_class(opt, **settings)
        lrs = []
        for _ in range(T):
            lrs.append(opt.param_groups[0]["lr"])            # the learning rate this step uses
            opt.step()
            sched.step()                                     # move the schedule on by one step
        return np.array(lrs)

    yours = []
    for s in range(T):
        yours.append(sched_fn(s, T, 2 * spe, 1.0))           # 2 warm-up epochs, peak 1.0

    curves = {
        "constant": np.ones(T),
        "step decay (x0.1 at 10, 15)": lr_curve(torch.optim.lr_scheduler.MultiStepLR, milestones=[10 * spe, 15 * spe], gamma=0.1),
        "cosine": lr_curve(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=T),
        "warm-up (2 ep) + cosine  [yours]": np.array(yours),
        "OneCycle": lr_curve(torch.optim.lr_scheduler.OneCycleLR, max_lr=1.0, total_steps=T),
    }
    # Figure: the five schedules over 20 epochs
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for (name, cv), c in zip(curves.items(), viz.PALETTE):
        ax.plot(steps / spe, cv, color=c, lw=2 if "yours" in name else 1.4, label=name)
    ax.set_xlabel("epoch"); ax.set_ylabel("LR / peak LR"); ax.set_title("Learning-rate schedules over 20 epochs")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0)); ax.grid(alpha=.3); plt.tight_layout(); plt.show()
    """),
    md(r"""
    Your curve (thick) should rise for two epochs, then decay along a cosine that reaches 0 at the same
    final step as the plain cosine. The step schedule drops 10x twice; OneCycle (torch defaults:
    `pct_start=0.3`, both phases cosine-shaped) starts at `max_lr / 25` and ends far below it.

    ### The LR range test

    Which peak learning rate? Leslie Smith's **LR range test** trains for a few dozen to a hundred
    batches while the learning rate grows **exponentially**, multiplied by the same factor at every step,
    from 1e-5 to 1: in 100 steps that factor is $(1 / 10^{-5})^{1/99} = 1.12$. It records the training
    loss, **smoothed** (a running average, so that the batch-to-batch noise does not hide the trend). The
    curve has three parts: flat (the learning rate is too small to change anything), falling (the useful
    range), exploding (too large: training diverges). A good peak learning rate is about **10 times below
    the lowest point** of the curve. `torchkit.lr_range_test` puts the weights back afterwards.

    From scratch on Pet the loss moves so slowly that 100 steps show little more than the explosion
    point, so we run the test on the fast proxy: the D1-01 convnet (`day1.FashionNet`) on FashionMNIST,
    where each step takes milliseconds. On your own problem, run it on the real model and data.
    """),
    api("torchkit.lr_range_test"),
    code(r"""
    fm_train, fm_val = day1.fashion_subsets(n_train=6400, n_val=2000)     # FashionMNIST: 6 400 training, 2 000 validation images
    fm_dl = DataLoader(fm_train, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    lrs, losses = torchkit.lr_range_test(day1.FashionNet().to(device), fm_dl, F.cross_entropy, torch.optim.AdamW,
                                         lr_min=1e-5, lr_max=1.0, steps=100)
    i_min = int(np.argmin(losses))                           # the step with the lowest smoothed loss
    lr_pick = lrs[i_min] / 10                                # the rule of thumb: 10 times below it
    # Figure: smoothed loss against the learning rate (log scale), with the minimum and the pick
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.semilogx(lrs, losses); ax.axvline(lrs[i_min], ls=":", c="gray", label=f"minimum at {lrs[i_min]:.1e}")
    ax.axvline(lr_pick, color=viz.PALETTE[2], label=f"pick ~ min / 10 = {lr_pick:.1e}")
    ax.set_xlabel("learning rate (log)"); ax.set_ylabel("smoothed training loss")
    ax.set_title("LR range test, D1-01 convnet + AdamW on FashionMNIST"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    The suggestion is a starting point, not a law: with warm-up + cosine you can usually go a bit
    higher. For PetNet we use a peak of 3e-3 with AdamW.
    """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · The debugging recipe

    **Why.** Most failed training runs are bugs, not bad settings: labels out of step with their
    images, a missing `zero_grad`, a wrong normalisation. They rarely raise an error; the loss just goes
    down slowly, or not at all. A few cheap checks, run before any long training, find most of them in
    seconds. Karpathy's recipe, condensed:

    1. **Check the loss at initialisation**, at the start of training, while the weights are still
       random. With C balanced classes a fresh network gives every class about the same probability
       1/C, so the cross-entropy is about $\ln C$ (section 1): $\ln 37 = 3.61$. Much higher means the
       network is confidently wrong before it has learned anything: a bad initialisation or a badly
       scaled last layer. **NaN** ("not a number", the result of 0 / 0 or ∞ - ∞; once it appears it
       spreads to every number it touches) means a bug in the data or the normalisation.
    2. **Look at the data right before it enters the model**, after augmentation and normalisation:
       `torchkit.show_batch`. Are the labels right? Are the images upside down?
    3. **Overfit one batch** (exercise 6). A correct model, loss and optimiser must drive the loss of one
       small fixed batch to about 0. If they cannot, nothing else matters yet.
    4. **Watch the gradient norm per layer.** The **norm** of a gradient tensor is its length: the
       square root of the sum of the squares of all its entries (a gradient (3, 4) has the norm 5).
       Gradients that shrink towards 0 from layer to layer (**vanishing**) leave the first layers
       untrained; gradients that grow (**exploding**) make training blow up.
    5. **The loss becomes NaN or inf** (infinite) during training: lower the learning rate, or add
       warm-up or **gradient clipping** (scaling the gradient down whenever its norm exceeds a limit);
       look for `log(0)`, a division by 0, or an overflow of **fp16** (16-bit floats, half the size of
       float32, used on GPUs for speed).
    6. Only then: make the model bigger, add regularisation, tune the settings.

    The next cell runs check 1 on a fresh PetNet, then multiplies the weights of its last layer by 30,
    a typical bad initialisation, and checks again.
    """),
    api("Tensor.mul_"),
    code(r"""
    torch.manual_seed(0)
    check_loader = DataLoader(day1.TensorImageDataset(data["x_train"], data["y_train"], show_tf), batch_size=64, shuffle=True)
    xb, yb = next(iter(check_loader))                        # one batch: (64, 3, 128, 128) images, (64,) labels
    xb = xb.to(device)
    yb = yb.to(device)
    net = day1.PetNet().to(device)                           # fresh, random weights
    with torch.no_grad():
        loss_init = F.cross_entropy(net(xb), yb)
        print(f"loss at init: {loss_init:.3f}   (ln 37 = {math.log(37):.3f})")
        net.fc.weight.mul_(30)                               # simulate a badly scaled last layer: every weight x 30
        loss_bad = F.cross_entropy(net(xb), yb)
        print(f"with a 30x last layer: {loss_bad:.3f}  <- confident and wrong at step 0: suspicious")
    """),
    md(r"""
    The fresh network starts at 3.630, close to ln 37 = 3.611: healthy. With the badly scaled last layer
    the same batch gives 14.433: the network is very sure of answers that are mostly wrong.

    ### Exercise 5 · the training step

    **Goal.** One training step on one batch: the inside of `torchkit.fit` and of every PyTorch training
    loop you will write.

    **Why again.** You wrote it in D0-00 (exercise 3). One line is new. A real training loop also
    *evaluates*: it switches the model to eval mode for a validation pass after every epoch (section 4).
    If the next training step does not switch it back, dropout stays off and BatchNorm keeps using its
    running averages: training goes on, silently different. So the step starts with `model.train()`.
    Exercise 6 builds on this function.

    **Input.** `model`: a network (an `nn.Module`). `x`: a batch of inputs, (B, ...). `y`: the batch's
    labels. `loss_fn`: a function `loss_fn(logits, y)` that returns one number, for example
    `F.cross_entropy`. `optimizer`: an optimiser over the model's parameters, for example
    `torch.optim.SGD(model.parameters(), lr=0.1)`.

    **Output.** The loss of this batch, computed *before* the update, as a Python float (`loss.item()`).
    As side effects, the model's parameters have moved by one step and the model is in train mode.

    **Example** (D0-00's hand example): a model `nn.Linear(1, 1)` with w = 1 and b = 0, one example
    x = `[[1.]]`, y = `[[3.]]`, `nn.MSELoss()` and SGD with lr 0.1. `train_step` returns 4.0, and
    afterwards w = 1.4 and b = 0.4.

    **Plan.** The order is the whole point:
    1. `model.train()`.
    2. Forward: `logits = model(x)`.
    3. Loss: `loss = loss_fn(logits, y)`: the predictions first, the labels second.
    4. `optimizer.zero_grad()`: clear the gradients of the previous step, which would otherwise add up
       (D0-00, section 6).
    5. `loss.backward()`: fill `p.grad` for every parameter p.
    6. `optimizer.step()`: update every parameter from its gradient.
    7. (written) `return loss.item()`.

    **What the checker tests.** A small network (two linear layers with a ReLU and a dropout layer
    between them), put in eval mode first; 10 random examples with 3 classes; SGD with lr 0.1. It calls
    your function 3 times and compares the 3 returned losses and the final weights with 3 clean steps it
    computes itself, and checks that the model is in train mode afterwards. Three calls, so that a
    missing `zero_grad` shows.

    **Common mistakes.**

    | you wrote | the checker prints |
    |---|---|
    | no `model.train()`, or `model.eval()` | `call model.train() inside the step (dropout / BatchNorm behave differently in eval mode)` |
    | no `optimizer.zero_grad()` | `losses [1.1866, 1.1592, 1.1096] vs expected [1.1866, 1.1592, 1.1344]` and `parameters after 3 steps differ from the reference: ...`: from the second step on, the gradients added up |
    | `zero_grad()` between `backward()` and `step()`; `step()` before `backward()`; no `step()`; or `loss.backward` without `()` | `losses [1.1866, 1.1866, 1.1866] vs expected [1.1866, 1.1592, 1.1344]` and the `parameters ... differ` line: nothing moved |
    | `return loss` | `return loss.item() (a Python float), got Tensor` |
    | no `return` | `return loss.item() (a Python float), got NoneType` |
    | `loss_fn(y, logits)`, arguments swapped | `not implemented yet: fill in the blanks, then delete the raise NotImplementedError line`. Misleading: PyTorch itself raised a `NotImplementedError` ("log_softmax ... not implemented for 'Long'") because it tried to take the softmax of the integer labels. Predictions first, labels second |
    """),
    api("Module.train", "Optimizer.zero_grad", "Tensor.backward", "Optimizer.step", "Tensor.item",
        title="Functions you will need", brief=True),
    exercise(
        stub=r'''
        def train_step(model, x, y, loss_fn, optimizer):
            """One training step on one batch.
            Returns the loss of the batch, computed before the update, as a Python float."""
            # 1. Train mode: dropout drops, BatchNorm uses the statistics of this batch.
            ____                 # YOUR CODE: switch the model to train mode
            # 2. Forward: the model's raw scores for the batch.
            logits = ____        # YOUR CODE: the model applied to x
            # 3. Loss: one number for the whole batch.
            loss = ____          # YOUR CODE: loss_fn of the logits and the labels y, in this order
            # 4. Clear the gradients of the previous step (otherwise they add up).
            ____                 # YOUR CODE: the optimiser's zero_grad
            # 5. Backward: fill p.grad for every parameter p.
            ____                 # YOUR CODE: backward on the loss
            # 6. Update every parameter from its gradient.
            ____                 # YOUR CODE: the optimiser's step
            raise NotImplementedError   # delete this line once the six blanks above are filled

            # 7. The loss as a Python float, for printing and plotting.
            return loss.item()

        ok_step = tests.check_train_step(train_step)
        ''',
        solution=r'''
        def train_step(model, x, y, loss_fn, optimizer):
            """One training step on one batch.
            Returns the loss of the batch, computed before the update, as a Python float."""
            # 1. Train mode: dropout drops, BatchNorm uses the statistics of this batch.
            model.train()
            # 2. Forward: the model's raw scores for the batch.
            logits = model(x)                                     # (B, num_classes)
            # 3. Loss: one number for the whole batch.
            loss = loss_fn(logits, y)                             # shape ()
            # 4. Clear the gradients of the previous step (otherwise they add up).
            optimizer.zero_grad()
            # 5. Backward: fill p.grad for every parameter p.
            loss.backward()
            # 6. Update every parameter from its gradient.
            optimizer.step()

            # 7. The loss as a Python float, for printing and plotting.
            return loss.item()

        ok_step = tests.check_train_step(train_step)
        '''),
    md(r"""
    **Did it work?** `[PASS]` means your step does exactly what `torchkit.fit` does for every batch.

    ### Exercise 6 · overfit one batch

    **Goal.** Write `overfit_one_batch(model, x, y, steps, lr)`: train a fresh model on one fixed batch,
    `steps` times, and return the loss of every step.

    **Why.** Check 3 of the debugging recipe. A network with hundreds of thousands of parameters can
    memorise 16 images in a few dozen steps: the loss must fall from about $\ln 37 = 3.61$ to nearly 0.
    If it does not, something between the data and the update is broken (labels that do not belong to
    the images, a layer that does not learn, a wrong loss, a learning rate far off), and you have found
    out in seconds instead of after an hour of training. Here, memorising is exactly what we want.

    **Input.** `model`: a freshly initialised network for 37 classes. `x`: one batch of images
    (B, 3, H, W). `y`: its labels (B,). `steps`: how many steps, an int. `lr`: the learning rate, a float.

    **Output.** A Python list of `steps` floats: the loss of each step (computed before that step's
    update). It should start near 3.61 and end near 0.

    **Example.** On the checker's network (below), 150 steps with lr 0.01 go from 3.651 at step 0 to
    0.0067 at the end. On PetNet with 16 real images, the figure after the exercise shows the curve.

    **Plan.**
    1. (written) `step_fn` is your `train_step` from exercise 5 if it passed, otherwise the course's.
    2. One optimiser, created *once*, before the loop: `torch.optim.AdamW(model.parameters(), lr=lr,
       weight_decay=0.0)`. No weight decay: here we want pure memorising.
    3. (written) An empty list `losses`.
    4. Repeat `steps` times: `loss = step_fn(model, x, y, F.cross_entropy, optimizer)` on the same x
       and y; append it to `losses` (written).
    5. (written) Return `losses`.

    **What the checker tests.** A small convnet with BatchNorm, 16 random images (3, 32, 32) with random
    labels out of 37, `steps=150`, `lr=0.01`. It prints the loss at step 0 and the lowest of the last
    five, and passes when your function returned at least 10 losses, the first is within 1 of ln 37, and
    the lowest of the last five is below 0.05.

    **Common mistakes.**

    | you wrote | the checker prints |
    |---|---|
    | the optimiser created inside the loop | `[PASS]` (`last = 0.0008`): it passes here, but a new AdamW forgets its running averages (section 5) at every step. Create it once, before the loop |
    | `range(10)` instead of `range(steps)` | `loss: step 0 = 3.651 (ln 37 = 3.611), last = 2.1968` and `the loss did not go to ~0 on a single batch: ...` |
    | SGD instead of AdamW | `last = 3.0662` and the same line |
    | a fixed small learning rate such as `1e-4` instead of the argument `lr` | `last = 3.2693` and the same line |
    | `model.parameters` without `()` | `your code raised TypeError: 'method' object is not iterable` |
    | `nn.MSELoss()` instead of `F.cross_entropy` | `your code raised RuntimeError: The size of tensor a (37) must match the size of tensor b (16) at non-singleton dimension 1` |
    | `return losses` indented inside the loop | `return the list of per-step losses`: the function returned after one step |
    | no `return` | `your code raised TypeError: 'NoneType' object is not iterable` |
    """),
    api("torch.optim.AdamW", "Module.parameters", title="Functions you will need", brief=True),
    exercise(
        stub=r'''
        step_fn = tests.use_or_reference(ok_step, globals().get("train_step"), day1.train_step, "train_step")

        def overfit_one_batch(model, x, y, steps=100, lr=1e-2):
            """Train `model` `steps` times on the same batch (x, y). Returns the list of the `steps` losses."""
            # 1. One optimiser for the whole run, created once: AdamW, learning rate lr, no weight decay.
            optimizer = ____     # YOUR CODE: torch.optim.AdamW over model.parameters() with lr=lr and weight_decay=0.0
            # 2. An empty list for the losses.
            losses = []
            # 3. `steps` training steps on the SAME batch, each with step_fn and F.cross_entropy.
            for step in range(steps):
                loss = ____      # YOUR CODE: one call of step_fn(model, x, y, loss function, optimizer); a float
                losses.append(loss)
            raise NotImplementedError   # delete this line once the two blanks above are filled

            # 4. Return the list of losses.
            return losses

        ok_overfit = tests.check_overfit_one_batch(overfit_one_batch)
        ''',
        solution=r'''
        step_fn = tests.use_or_reference(ok_step, globals().get("train_step"), day1.train_step, "train_step")

        def overfit_one_batch(model, x, y, steps=100, lr=1e-2):
            """Train `model` `steps` times on the same batch (x, y). Returns the list of the `steps` losses."""
            # 1. One optimiser for the whole run, created once: AdamW, learning rate lr, no weight decay.
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
            # 2. An empty list for the losses.
            losses = []
            # 3. `steps` training steps on the SAME batch, each with step_fn and F.cross_entropy.
            for step in range(steps):
                loss = step_fn(model, x, y, F.cross_entropy, optimizer)   # a Python float
                losses.append(loss)

            # 4. Return the list of losses.
            return losses

        ok_overfit = tests.check_overfit_one_batch(overfit_one_batch)
        '''),
    md(r"""
    **Did it work?** `[PASS]` means you can memorise one batch: model, loss and optimiser work together.
    The next cell runs your function on PetNet with 16 Pet images (64 px, 11 different breeds), 60 steps
    at lr 0.003. Then it runs the same loop with a typical data bug: the labels are shuffled at every
    step, so they no longer belong to their images.
    """),
    code(r"""
    tf64 = day1.eval_transform(64)                           # 64 px, no augmentation
    images = []
    for i in range(0, 2000, 125):                            # 16 images, 125 apart (11 different breeds)
        images.append(tf64(data["x_train"][i]))
    x16 = torch.stack(images).to(device)                     # (16, 3, 64, 64)
    y16 = data["y_train"][0:2000:125].to(device)             # (16,)
    of_fn = tests.use_or_reference(ok_overfit, globals().get("overfit_one_batch"), day1.overfit_one_batch, "overfit_one_batch")
    torch.manual_seed(0)
    healthy = of_fn(day1.PetNet().to(device), x16, y16, 60, 3e-3)

    # the same loop with a typical data bug: images and labels get out of step (labels shuffled at every step)
    torch.manual_seed(0)
    net_bug = day1.PetNet().to(device)
    opt_bug = torch.optim.AdamW(net_bug.parameters(), lr=3e-3, weight_decay=0.0)
    buggy = []
    for step in range(60):
        wrong_labels = y16[torch.randperm(16)]               # the 16 labels in a new random order
        buggy.append(day1.train_step(net_bug, x16, wrong_labels, F.cross_entropy, opt_bug))

    # Figure: training loss per step (log scale) for the correct and the buggy run
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(healthy, color=viz.PALETTE[2], label=f"correct pipeline: final loss {healthy[-1]:.3f}")
    ax.plot(buggy, color=viz.PALETTE[3], label=f"labels out of sync (bug): final loss {buggy[-1]:.2f}")
    ax.axhline(math.log(37), ls=":", c="gray", label="ln 37 (chance)")
    ax.set_yscale("log"); ax.set_xlabel("step (same 16 images every step)"); ax.set_ylabel("training loss (log)")
    ax.set_title("Overfit one batch: a correct setup reaches ~0, a buggy one cannot"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: the correct run falls towards 0 (the vertical axis is logarithmic). The buggy one
    stops falling at about 2.5: with labels that change at every step, the most it can learn is how
    often each breed occurs in the batch (the 16 labels cover 11 breeds, 5 of them twice), which is
    worth a loss of about 2.34. It never learns which image is which breed, and no amount of training
    will change that.

    **Check 4, gradient norms.** The next cell runs one backward pass of a fresh PetNet on the batch from
    check 1; `day1.grad_norms_per_layer(net)` returns a dict {weight tensor name: norm of its gradient}. Healthy means no bar is orders of magnitude below the
    others; here all six lie between about 0.3 and 1.
    """),
    code(r"""
    net = day1.PetNet().to(device)
    F.cross_entropy(net(xb), yb).backward()                  # one backward pass: fills p.grad for every parameter
    gn = day1.grad_norms_per_layer(net)                      # {name of a weight tensor: norm of its gradient}
    names = []
    for key in gn:
        names.append(key.replace(".weight", ""))             # shorter labels for the x axis
    # Figure: one bar per weight tensor, log scale
    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.bar(range(len(gn)), list(gn.values()), color=viz.PALETTE[0])
    ax.set_xticks(range(len(gn))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("||grad|| (log)"); ax.set_title("Gradient norm per weight tensor at init: no layer is starved")
    ax.grid(alpha=.3, axis="y"); plt.tight_layout(); plt.show()
    """),

    # ------------------------------------------------------------------ 7
    md(r"""
    ## 7 · Putting it together: baseline vs. recipe

    The recipe, all in `day1.train_recipe` (in `notebooks/cvcourse/day1.py`):

    * augmentation: **your** pipeline from exercise 1 (`RandomResizedCrop` + flip + `TrivialAugmentWide`
      + `RandomErasing`);
    * `AdamW(lr=3e-3, weight_decay=0.05)`, no weight decay on BatchNorm and bias parameters;
    * linear warm-up for the first 15 % of the steps, then cosine decay to 0: **your** `warmup_cosine`.
      `torch.optim.lr_scheduler.LambdaLR` calls it at every step with lr_max = 1, so it returns a factor
      from 0 to 1, and multiplies the optimiser's learning rate 3e-3 by it;
    * label smoothing 0.1 (**your** `label_smoothing_ce`); dropout 0.2 before the last linear layer;
    * the checkpoint keeps the best epoch (early stopping).

    Same network, same number of epochs, same data as section 1. Each of your three functions is used
    only if its checker passed; otherwise the course's version is, with a note. If the checkpoint
    `d1-02_recipe_64px.pt` (`_128px` on a GPU) already exists, it is loaded instead of trained: delete
    it to train with your own functions (up to 2.5 minutes on a CPU).
    """),
    code(r"""
    aug_builder = tests.use_or_reference(ok_aug, globals().get("build_train_tf"), day1.recipe_transforms, "build_train_tf")
    sched_fn = tests.use_or_reference(ok_sched, globals().get("warmup_cosine"), day1.warmup_cosine, "warmup_cosine")
    ls_fn = tests.use_or_reference(ok_ls, globals().get("label_smoothing_ce"), day1.label_smoothing_ce, "label_smoothing_ce")

    model_recipe = day1.PetNet(num_classes=37, dropout=0.2).to(device)

    def train_recipe_run(checkpoint_path):
        '''Train model_recipe with the full recipe and your three functions; the best epoch goes to checkpoint_path.'''
        return day1.train_recipe(model_recipe, data, epochs=EPOCHS, lr=3e-3, weight_decay=0.05, label_smoothing=0.1,
                                 size=IMG, train_tf=aug_builder(IMG), schedule_fn=sched_fn, smoothing_loss=ls_fn,
                                 max_minutes=MAX_MIN, checkpoint=checkpoint_path, device=device)

    hist_recipe = day1.load_or_train(model_recipe, f"d1-02_recipe_{IMG}px", train_recipe_run)

    # Figure: both runs; loss (left, dashed = train, solid = validation) and validation accuracy (right)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_history(a1, a2, hist_plain, "plain", viz.PALETTE[3])
    plot_history(a1, a2, hist_recipe, "recipe", viz.PALETTE[0])
    a1.set_title("loss (dashed: train, solid: val)"); a1.set_xlabel("epoch"); a1.legend(fontsize=8); a1.grid(alpha=.3)
    a2.set_title("validation accuracy (%)"); a2.set_xlabel("epoch"); a2.legend(fontsize=8); a2.grid(alpha=.3)
    plt.tight_layout(); plt.show()

    def final_gap(h):
        '''Validation loss minus training loss at the last epoch: how far the run overfits.'''
        return h["val_loss"][-1] - h["train_loss"][-1]

    print(f"best val acc  plain {max(hist_plain['val_metric']):.3f}   recipe {max(hist_recipe['val_metric']):.3f}")
    print(f"final val-train loss gap  plain {final_gap(hist_plain):+.2f}   recipe {final_gap(hist_recipe):+.2f}")
    """),
    md(r"""
    **What to look for.** With the recipe the training loss is *higher* (augmented images and smoothed
    targets are harder to fit, and label smoothing puts a floor under the loss) but the validation loss
    keeps following it: the gap stays small, and the validation loss is still falling when the schedule
    ends. After 10 CPU epochs at 64 px the two accuracies are close (~13-16 %): heavy regularisation
    *slows down* fitting, and pays off only with longer schedules. Expected numbers on a GPU at 128 px:
    **~15 % plain vs ~30-35 % with the recipe after 30-40 epochs** (and the plain run gets *worse* the
    longer you train it). Starting from a pretrained network instead (D1-04): 80-90 %+.

    ### A small random search

    A **hyper-parameter** is a setting chosen before training rather than learned: the learning rate,
    the weight decay, the dropout probability, the number of epochs. Which ones matter? In order: the
    **learning rate**, then **weight decay** and **augmentation strength** / dropout, then everything
    else.

    * **Grid search** tries every combination of a few values per setting. **Random search** draws every
      setting at random for every trial. Random search wins (Bergstra & Bengio, 2012) because each trial
      tests a new value of *every* setting: with 8 trials, a 2 x 2 x 2 grid tries only 2 learning rates,
      random search 8.
    * Settings that act by orders of magnitude, like the learning rate (1e-4 vs 1e-3 vs 1e-2), are drawn
      **log-uniformly**: draw the exponent uniformly, then take 10 to that power. Every decade (1e-4 to
      1e-3, 1e-3 to 1e-2, ...) is then equally likely. **By hand:** the first draw below is the exponent
      -1.952, so the learning rate is $10^{-1.952} = 0.011$. Drawing the learning rate itself uniformly
      between 0.00003 and 0.3 would put 9 out of 10 trials above 0.03.
    * Search on a cheap **proxy** (fewer images, lower resolution, a narrower network, fewer epochs).
      Proxies keep the *ranking* of settings reasonably well, not the accuracy itself.

    A meaningful search on Pet from scratch needs hours of GPU time; to see the procedure work in about
    a minute, the proxy here is the FashionMNIST convnet (6 400 images, 1 epoch per trial).
    """),
    api("np.random.default_rng", "torchkit.fit", brief=True),
    code(r"""
    rng = np.random.default_rng(0)                   # NumPy's random generator, seeded: the same 8 trials every run
    results = []                                     # one dict per trial: its settings and its validation accuracy
    t0 = time.time()
    for trial in range(8):
        # 1. Draw the settings: learning rate and weight decay log-uniformly, dropout uniformly.
        lr = float(10 ** rng.uniform(-4.5, -0.5))
        weight_decay = float(10 ** rng.uniform(-4, -0.5))
        dropout = float(rng.uniform(0, 0.5))
        # 2. Train a fresh FashionNet for one epoch with these settings.
        torch.manual_seed(trial)
        net = day1.FashionNet(dropout).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        h = torchkit.fit(net, fm_dl, DataLoader(fm_val, 500), F.cross_entropy, opt, epochs=1, verbose=False, max_minutes=0.5)
        # 3. Record the result.
        val_acc = h["val_metric"][-1]
        results.append({"lr": lr, "weight_decay": weight_decay, "dropout": dropout, "val_acc": val_acc})
        print(f"trial {trial}: lr {lr:.1e}  wd {weight_decay:.1e}  dropout {dropout:.2f}"
              f"  ->  val acc {val_acc:.3f}   ({time.time() - t0:.0f} s)")

    best = results[0]                                # the trial with the highest validation accuracy
    for r in results:
        if r["val_acc"] > best["val_acc"]:
            best = r
    print(f"best: lr {best['lr']:.4f}  weight_decay {best['weight_decay']:.4f}  dropout {best['dropout']:.4f}  "
          f"val_acc {best['val_acc']:.4f}")
    """),
    code(r"""
    # Figure: validation accuracy of every trial against its learning rate (left) and weight decay (right); colour = dropout
    accs, dropouts = [], []
    for r in results:
        accs.append(100 * r["val_acc"])
        dropouts.append(r["dropout"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, key in zip(axes, ("lr", "weight_decay")):
        xs = []
        for r in results:
            xs.append(r[key])
        sc = ax.scatter(xs, accs, c=dropouts, cmap="viridis", vmin=0, vmax=0.5, s=70, edgecolor="k")
        ax.set_xscale("log"); ax.set_xlabel(key + " (log)"); ax.grid(alpha=.3)
    axes[0].set_ylabel("val accuracy (%)"); plt.colorbar(sc, ax=axes, label="dropout")
    axes[0].set_title("accuracy depends strongly on the LR..."); axes[1].set_title("...much less on weight decay (at 1 epoch)")
    plt.show()
    """),

    # ------------------------------------------------------------------ 8
    md(r"""
    ## 8 · Checkpointing and resuming

    **Why.** Long runs crash, get stopped (a shared GPU is taken away: the run is "pre-empted"), or you
    want the best epoch back. Every training cell of the course already saves and loads checkpoints;
    this section looks inside one.

    **Words.**
    * A model's **state_dict** is a dict from the name of each parameter (and of each BatchNorm running
      statistic) to its tensor: everything needed to rebuild the trained model, given its class.
      `model.load_state_dict(sd)` copies the tensors back in.
    * An optimiser has a state_dict too: for AdamW, for every parameter, its step count and its two
      running averages (section 5); for SGD with momentum, the velocity. A scheduler's state_dict records
      how many steps it has done.
    * The **random number generator** (RNG) produces every "random" number: the shuffling of the data,
      the augmentation draws, dropout. Its **state** is its current position; saving and restoring it
      makes the random draws continue exactly where they stopped.
    * `torch.save(obj, path)` writes a dict of tensors and numbers to a file; `torch.load(path)` reads it
      back. `weights_only=False` allows the file to hold more than tensors; load only files you trust
      that way, because such a file can run code when it is loaded.

    A *weights-only* checkpoint (`torchkit.save_checkpoint`) is enough to use a model, or to fine-tune it
    later (section 5). To **resume** training exactly, you also need the optimiser state, the scheduler state, the
    epoch counter and the RNG states. First, the smallest state_dict there is:
    """),
    api("Module.state_dict", "Module.load_state_dict", "torch.save", "torch.load", "torch.get_rng_state",
        "torch.set_rng_state"),
    api("nn.Linear", brief=True),
    code(r"""
    torch.manual_seed(0)
    lin = nn.Linear(2, 1)                                    # pred = w1 * x1 + w2 * x2 + b: 2 weights and 1 bias
    sd = lin.state_dict()                                    # {name: tensor}
    for name in sd:
        print(name, tuple(sd[name].shape), sd[name])
    """),
    md(r"""
    The cell below shows that resuming works: run A trains 2 epochs straight; run B trains 1 epoch,
    saves *everything* and "crashes"; run C builds *fresh* objects (with a different random start on
    purpose), loads the checkpoint and trains the second epoch. If the checkpoint is complete, run C
    ends with exactly the weights of run A.
    """),
    code(r"""
    def make_run(seed=0):
        '''Fresh model, optimiser, scheduler and data loader for the demo (a narrow PetNet on 256 images).'''
        torch.manual_seed(seed)
        model = day1.PetNet(width=16).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2)
        dl, _ = day1.loaders(data, day1.recipe_transforms(64), day1.eval_transform(64), 64, n_train=256)
        return model, opt, sched, dl

    def one_epoch(model, opt, sched, dl):
        '''One epoch of training steps, then one step of the (per-epoch) schedule.'''
        for x, y in dl:
            day1.train_step(model, x.to(device), y.to(device), F.cross_entropy, opt)
        sched.step()

    # run A: two epochs, no interruption
    mA, oA, sA, dlA = make_run()
    torch.manual_seed(123)
    one_epoch(mA, oA, sA, dlA)
    one_epoch(mA, oA, sA, dlA)

    # run B: one epoch, save EVERYTHING, "crash"
    mB, oB, sB, dlB = make_run()
    torch.manual_seed(123)
    one_epoch(mB, oB, sB, dlB)
    ck_path = torchkit.CHECKPOINTS / "d1-02_resume_demo.pt"
    checkpoint = {
        "model": mB.state_dict(),                            # every weight and BatchNorm statistic
        "optimizer": oB.state_dict(),                        # AdamW's running averages and step counts
        "scheduler": sB.state_dict(),                        # how far the schedule has got
        "epoch": 1,                                          # epochs finished
        "torch_rng": torch.get_rng_state(),                  # PyTorch's RNG: augmentation, dropout
        "loader_rng": dlB.generator.get_state(),             # the loader's own RNG: the order of the data
    }
    torch.save(checkpoint, ck_path)
    del mB, oB, sB, dlB                                      # the "crash": only the file is left

    # run C: resume into brand-new objects
    mC, oC, sC, dlC = make_run(seed=999)                     # different start on purpose: everything comes from the file
    state = torch.load(ck_path, weights_only=False)
    mC.load_state_dict(state["model"])
    oC.load_state_dict(state["optimizer"])
    sC.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    dlC.generator.set_state(state["loader_rng"])
    one_epoch(mC, oC, sC, dlC)

    # compare: the largest difference between any tensor of run A and the same tensor of run C
    weights_A = mA.state_dict()
    weights_C = mC.state_dict()
    diff = 0.0
    for name in weights_A:
        gap = weights_A[name] - weights_C[name]              # same shape as the tensor
        diff = max(diff, gap.abs().max().item())
    print(f"resumed from epoch {state['epoch']}; max |weights(A) - weights(resumed)| = {diff:.2e}")
    ck_path.unlink()                                         # delete the demo file
    """),

    # ------------------------------------------------------------------ 9
    md(r"""
    ## 9 · Summary

    * A convnet trained from scratch on ~50 images per class memorises: training loss down, validation
      loss stuck or up. Always plot both curves.
    * The pipeline: decode once, cheap `uint8` transforms, then float + normalise; shuffle the training
      loader; `num_workers` so that the GPU never waits.
    * Augmentation is the strongest regulariser for vision, *if* it preserves the label. Mixup / CutMix
      and label smoothing soften the targets.
    * BatchNorm uses batch statistics in train mode and running averages in eval mode: call
      `model.eval()` before validating or predicting, `model.train()` before training.
    * AdamW (decoupled weight decay) + linear warm-up + cosine decay is a robust default; the LR range
      test gives the order of magnitude of the peak learning rate.
    * Debug first: loss at init = $\ln C$, look at the batch, overfit one batch, watch gradient norms,
      NaN -> lower the learning rate.
    * Search the learning rate first (log-uniform, random search, on a proxy), then weight decay and
      augmentation.
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
    md(r"""
    ## Words from this notebook

    **Generalisation and the loss**

    | word | meaning |
    |---|---|
    | overfitting | learning the training images themselves instead of the task: training loss down, validation loss stuck or up |
    | memorise | store enough about each training image to recognise it |
    | generalisation | doing well on images the model has never seen |
    | from scratch | training from random weights, not from a pretrained network |
    | class-balanced | (almost) the same number of examples for every class |
    | logits | the network's raw scores, one per class, any real numbers |
    | softmax | turns logits into probabilities: $e^{z_k} / \sum_j e^{z_j}$ |
    | cross-entropy | the classification loss: $-\ln p_{\text{true}}$, or $-\sum_k q_k \ln p_k$ for a soft label q |
    | log-softmax | the log of the softmax, computed in one accurate step |
    | nat | the unit of a loss computed with the natural log |
    | accuracy | the share of images whose largest logit is the right class |
    | chance level | the accuracy of guessing: 1 / number of classes |
    | history | the dict of per-epoch numbers a training run returns |
    | multiply-add | one multiplication plus one addition; their count measures the computation per image |
    | thread | one line of work the CPU runs; PyTorch splits big computations over several |

    **Data and augmentation**

    | word | meaning |
    |---|---|
    | Dataset | an object with `len(ds)` and `ds[i] -> (image, label)` |
    | transform | a function from one image to a changed image; `v2.Compose` chains several |
    | shuffle | serving the examples in a new random order every epoch |
    | num_workers, pin_memory, drop_last | DataLoader settings: background processes, faster copies to a GPU, skip the incomplete last batch |
    | fast proxy | a smaller problem that behaves alike, used for quick experiments |
    | data augmentation | random changes to training images that keep the right answer |
    | invariance | a change that must not change the answer (position, mirroring, lighting) |
    | geometric / photometric | augmentations that move pixels / that change colours and brightness |
    | RandomResizedCrop, flip, rotation, ColorJitter, grayscale, RandomErasing | the classic augmentations (table in section 3) |
    | policy | a ready-made augmentation recipe that picks the operations and their strengths |
    | RandAugment, TrivialAugmentWide, AutoAugment | automatic policies: random operations from a fixed list |
    | one-hot label | a vector with 1 at the right class and 0 elsewhere |
    | soft label, soft target | a vector of class probabilities that add up to 1 |
    | Mixup | blend two images and their one-hot labels with the weight λ |
    | CutMix | paste a rectangle of one image into another; λ = the share of the area kept |
    | Beta distribution | draws a random λ between 0 and 1; α sets how strongly it mixes |

    **Regularisation**

    | word | meaning |
    |---|---|
    | regularisation | any change to training that makes memorising harder, so that the model generalises better |
    | weight decay | shrinking every weight by a small fraction of itself at every step |
    | L2 penalty | λ/2 times the sum of the squared weights, added to the loss; the older form of weight decay |
    | unit, neuron | one output number of a layer, for example one channel of a convolution |
    | dropout | setting random numbers inside the network to 0 during training (survivors scaled by 1 / (1 - p)) |
    | train mode / eval mode | `model.train()` / `model.eval()`: dropout and BatchNorm behave differently in the two |
    | BatchNorm | normalises every channel with the batch's mean and variance (train mode) or running averages (eval mode), then scales by γ and shifts by β |
    | variance | the square of the standard deviation |
    | running mean / running variance | BatchNorm's averages over the training batches: new = 0.9 · old + 0.1 · batch value |
    | over-confident | far more sure of its answers than its accuracy justifies |
    | label smoothing | target 1 - ε + ε / C for the right class and ε / C for every other class |
    | early stopping | keeping the weights of the epoch with the best validation score |

    **Optimisation**

    | word | meaning |
    |---|---|
    | optimiser | the object that updates the weights from their gradients |
    | SGD | $w \leftarrow w - \eta g$ |
    | momentum, velocity | a running sum of past gradients; steps grow where the gradient keeps its direction |
    | Adam | steps by $\eta\, m / (\sqrt{v} + \epsilon)$: every weight moves about η per step |
    | AdamW | Adam with decoupled weight decay |
    | learning rate (η, lr) | the size of the steps |
    | ill-conditioned | a loss surface much steeper in some directions than in others |
    | learning-rate schedule | a rule that changes the learning rate step by step |
    | scheduler | PyTorch's object that changes an optimiser's learning rate at every `scheduler.step()` |
    | fine-tuning | training a pretrained network a little further on new data (D1-04) |
    | warm-up | raising the learning rate in a straight line from tiny to the peak at the start |
    | cosine decay | lowering it from the peak to 0 along half a cosine wave |
    | step decay | dividing it by 10 at fixed epochs |
    | OneCycle | warm up to a high peak, then anneal far below the start |
    | LR range test | a short run with an exponentially growing learning rate; pick about 10x below the loss minimum |
    | hyper-parameter | a setting chosen before training (learning rate, weight decay, dropout, epochs) |
    | grid search / random search | trying every combination of a few values / drawing every setting at random per trial |
    | log-uniform | drawing the exponent uniformly, so that every decade is equally likely |

    **Debugging and checkpoints**

    | word | meaning |
    |---|---|
    | initialisation | the random starting weights; the loss there should be about ln C |
    | NaN, inf | "not a number" (0 / 0) and infinity: signs of a bug or a learning rate that is too large |
    | overfit one batch | the check that model, loss and optimiser can memorise one small batch |
    | gradient norm | the length of a gradient: square root of the sum of its squared entries |
    | vanishing / exploding gradients | gradients that shrink towards 0 / grow huge from layer to layer |
    | gradient clipping | scaling the gradient down when its norm exceeds a limit |
    | fp16 | 16-bit floating point numbers, used on GPUs for speed |
    | checkpoint | a file with a model's state (and, to resume, the optimiser's, the scheduler's and the RNG's) |
    | state_dict | a dict from parameter names to tensors; optimisers and schedulers have one too |
    | random number generator (RNG), RNG state | what produces "random" numbers, and its current position |
    | resume | continue a stopped run exactly where it stopped |
    """),
]
