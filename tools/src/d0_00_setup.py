from nb import md, code, exercise, header

NAME = "D0-00-setup-and-tensors"

CELLS = [
    header(
        "D0-00 · Setup and tensors (optional pre-course)",
        steps=[
            "Check the environment: Python, PyTorch, torchvision, GPU",
            "Download the course datasets and look at one batch of each",
            "Tensor refresher: shapes, broadcasting, HWC vs CHW, the batch dimension",
            "Autograd in five lines",
            "A ten-line training loop on a linear model",
            "The rhythm of every notebook (figure, lab, exercise, checker) and a check that the interactive labs work",
        ],
        topics=[
            "Confirm your machine runs every notebook of the course (CPU is enough)",
            "Know the four datasets you will meet: FashionMNIST, Oxford-IIIT Pet, Penn-Fudan, synthetic shapes",
            "Move images between the NumPy / Matplotlib layout (H, W, C) and the PyTorch layout (C, H, W)",
            "Use broadcasting to normalise a whole batch in one line",
            "Write the forward -> loss -> zero_grad -> backward -> step loop from memory",
        ],
        minutes=25,
        prereq="Python and NumPy. Neural networks and backpropagation (previous class).",
    ),
    md(r"""
    ## 0 · Setup

    Run this cell first in every notebook of the course. It puts the shared `cvcourse`
    helpers on the import path, loads PyTorch, fixes the random seed and picks a device.
    """),
    code("""
    import sys, pathlib
    for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:   # find the cvcourse package
        if (p / "cvcourse").is_dir():
            sys.path.insert(0, str(p)); break

    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from cvcourse import viz, torchkit, tests

    %matplotlib inline
    plt.rcParams["figure.dpi"] = 90
    torch.manual_seed(0)
    device = torchkit.get_device()
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · Environment check

    The course needs PyTorch 2.x and torchvision. A GPU is welcome but **not** required:
    every training cell is sized to finish in a few minutes on a laptop CPU, and the
    longer fine-tunes load a saved checkpoint instead of training when one exists.
    """),
    code("""
    import platform, matplotlib, torchvision

    print(f"{'python':<14}{platform.python_version()}")
    for mod in (torch, torchvision, np, matplotlib):
        print(f"{mod.__name__:<14}{mod.__version__}")
    print(f"{'CUDA GPU':<14}{torch.cuda.is_available()}" + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print(f"{'CPU threads':<14}{torch.get_num_threads()}")

    for name in ("timm", "transformers", "open_clip"):
        try:
            __import__(name); status = "installed"
        except ImportError:
            status = "not installed (optional: only used in a few demo cells)"
        print(f"{name:<14}{status}")
    """),
    md(r"""
    ## 2 · Datasets

    All data lives in `data/` at the repository root (`torchkit.data_root()` finds it from any
    notebook). The download script fetches everything once (about 850 MB) and skips what is
    already there, so it is safe to run again. Trained weights are saved to
    `data/checkpoints/` and reused when you re-run a notebook.

    | dataset | what | used for |
    |---|---|---|
    | FashionMNIST | 70 000 grayscale 28x28 clothes, 10 classes | Day 1: first convnet |
    | Oxford-IIIT Pet | 7 349 photos, 37 breeds, head boxes, trimap masks | the thread through all three days |
    | Penn-Fudan | 170 street photos with pedestrian instance masks | Faster R-CNN / Mask R-CNN fine-tuning |
    | synthetic shapes | circles, squares, triangles drawn on the fly | every mechanism you must *see* (anchors, matching, U-Net) |
    """),
    code("""
    import subprocess
    REPO = torchkit.data_root().parent
    result = subprocess.run([sys.executable, str(REPO / "tools" / "download_data.py")], capture_output=True, text=True)
    print(result.stdout[-1500:] or result.stderr[-1500:])
    """),
    md(r"""
    Now one batch of each. The helper `torchkit.show_batch` undoes the normalisation, draws
    class names, boxes or masks, and returns a Matplotlib figure. You will use it all week.

    **FashionMNIST.** Small, clean, and fast enough to train a convnet on a CPU in a minute.
    """),
    code("""
    fm_train, fm_test, fm_classes = torchkit.get_fashion_mnist()
    x, y = next(iter(DataLoader(fm_train, batch_size=16, shuffle=True)))
    print("batch:", tuple(x.shape), x.dtype, "| labels:", y[:8].tolist(), "...")
    torchkit.show_batch(x, y, fm_classes, n=16, size=1.6); plt.show()
    """),
    md(r"""
    **Oxford-IIIT Pet** comes with three kinds of labels for the same photos, so we can
    classify (Day 1), detect (Day 2) and segment (Day 3) one dataset. `get_pet_datasets`
    returns `(train, val, classes)` for the label type you ask for.
    """),
    code("""
    pet_cls_train, _, pet_classes = torchkit.get_pet_datasets(size=160, target="category")
    print(len(pet_classes), "breeds, e.g.", pet_classes[:4], "| train images:", len(pet_cls_train))
    x, y = next(iter(DataLoader(pet_cls_train, batch_size=10, shuffle=True)))
    torchkit.show_batch(x, y, pet_classes, n=10, cols=5, size=2.6, title="Pet: breed labels"); plt.show()
    """),
    code("""
    pet_det_train, _, det_classes = torchkit.get_pet_datasets(size=160, target="detection")
    pet_seg_train, _, seg_classes = torchkit.get_pet_datasets(size=160, target="segmentation")
    idx = list(range(0, 1000, 200))
    det_items = [pet_det_train[i] for i in idx]
    seg_items = [pet_seg_train[i] for i in idx]
    print("detection target:", det_items[0][1])
    print("segmentation target:", tuple(seg_items[0][1].shape), seg_items[0][1].dtype, "values", seg_items[0][1].unique().tolist(), "=", seg_classes)
    torchkit.show_batch([im for im, _ in det_items], [t for _, t in det_items], det_classes, cols=5, size=2.6,
                        title="Pet: head boxes (Day 2)"); plt.show()
    torchkit.show_batch([im for im, _ in seg_items], [m for _, m in seg_items], cols=5, size=2.6,
                        title="Pet: trimap (orange = pet, green = border, clear = background) (Day 3)"); plt.show()
    """),
    md(r"""
    **Penn-Fudan** has one mask *per person* (instance segmentation). The images keep their
    original sizes, so a batch is a Python list rather than one stacked tensor; that is
    what `torchkit.detection_collate` does.
    """),
    code("""
    pf = torchkit.get_penn_fudan()
    imgs, targets = torchkit.detection_collate([pf[i] for i in (0, 10, 20, 30)])
    print(len(pf), "images; first one", tuple(imgs[0].shape), "with", len(targets[0]["boxes"]), "people;",
          "target keys:", list(targets[0]))
    torchkit.show_batch(imgs, targets, ["bg", "person"], cols=4, size=3.2, title="Penn-Fudan: one mask and box per person"); plt.show()
    """),
    md(r"""
    **Synthetic shapes** are generated in memory from a seed. Nothing to download, and
    every object is perfectly labelled, which makes them ideal for watching a detector or a
    U-Net learn in under two minutes.
    """),
    code("""
    shapes_det = torchkit.ShapesDataset(n=8, size=128, seed=0, task="detection")
    shapes_seg = torchkit.ShapesDataset(n=8, size=128, seed=0, task="segmentation")
    imgs, targets = torchkit.detection_collate([shapes_det[i] for i in range(8)])
    torchkit.show_batch(imgs, targets, shapes_det.classes, cols=8, size=1.8, title="shapes: boxes"); plt.show()
    masks = [shapes_seg[i][1].numpy() for i in range(8)]
    viz.show_images(masks, cols=8, size=1.8, cmap="tab10", vmin=0, vmax=9,
                    suptitle="shapes: class masks (blue 0 = background, 1 circle, 2 square, 3 triangle)"); plt.show()
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · Tensor refresher

    A `torch.Tensor` is a NumPy array that can live on a GPU and remembers how it was
    computed (for autograd). Three attributes matter all the time: `shape`, `dtype`, `device`.
    """),
    code("""
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print(t)
    print("shape", tuple(t.shape), "| dtype", t.dtype, "| device", t.device)
    print("row 1:", t[1].tolist(), "| column 2:", t[:, 2].tolist(), "| sum over rows:", t.sum(dim=0).tolist())
    print("to NumPy and back:", type(t.numpy()).__name__, type(torch.from_numpy(t.numpy())).__name__)
    """),
    md(r"""
    ### HWC vs CHW

    NumPy, PIL and Matplotlib store an image as **(H, W, C)**: height, width, channels.
    PyTorch layers expect **(C, H, W)**, and a *batch* of images as **(B, C, H, W)**.
    `permute` reorders the axes without copying any pixel; `unsqueeze(0)` adds the batch axis.

    | layout | who uses it | example shape for one 480x640 RGB photo |
    |---|---|---|
    | (H, W, C) | NumPy, PIL, Matplotlib, OpenCV | (480, 640, 3) |
    | (C, H, W) | one image in PyTorch | (3, 480, 640) |
    | (B, C, H, W) | a batch in PyTorch | (1, 3, 480, 640) |
    """),
    code("""
    photo_u8 = (viz.sample_photo() * 255).astype(np.uint8)       # how images arrive from disk: uint8 (H, W, C)
    print("NumPy photo:", photo_u8.shape, photo_u8.dtype)

    chw = torch.from_numpy(photo_u8).permute(2, 0, 1)              # (C, H, W), still uint8, no copy
    batch = chw.unsqueeze(0)                                        # (1, C, H, W)
    print("CHW:", tuple(chw.shape), "| batch:", tuple(batch.shape), "| shares memory:", chw.data_ptr() == torch.from_numpy(photo_u8).data_ptr())
    print("contiguous after permute?", chw.is_contiguous(), "-> .contiguous() makes a real copy if a library needs one")
    """),
    md(r"""
    ### Exercise 1 · `to_chw`

    Write `to_chw(img_hwc)`: take a NumPy image `(H, W, C)` and return a **float32** tensor
    `(C, H, W)` with values in `[0, 1]`. A `uint8` image (0-255) must be divided by 255; a
    float image is already in `[0, 1]` and stays as it is.
    """),
    exercise(
        stub="""
        def to_chw(img_hwc):
            t = torch.from_numpy(img_hwc)
            # YOUR CODE HERE: reorder the axes and convert to float32 in [0, 1]
            raise NotImplementedError

        tests.check_to_chw(to_chw)
        """,
        solution="""
        def to_chw(img_hwc):
            t = torch.from_numpy(img_hwc)
            t = t.permute(2, 0, 1)
            return t.float() / 255 if img_hwc.dtype == np.uint8 else t.float()

        tests.check_to_chw(to_chw)
        """),
    md(r"""
    ### Broadcasting: normalise a whole batch in one line

    When two tensors have different shapes, PyTorch (like NumPy) lines them up **from the
    right** and stretches every axis of size 1. A per-channel mean of shape `(3, 1, 1)`
    therefore broadcasts against an image `(3, H, W)` *and* against a batch `(B, 3, H, W)`.

    $$x_{\text{norm}}[c, i, j] = \frac{x[c, i, j] - \mu_c}{\sigma_c}$$
    """),
    code("""
    a = torch.ones(2, 3, 4, 5)
    for other in [(5,), (4, 1), (3, 1, 1), (1, 3, 1, 1)]:
        print(f"{str(tuple(a.shape)):<16} op {str(other):<14} -> {tuple((a + torch.zeros(other)).shape)}")
    try:
        a + torch.zeros(3)
    except RuntimeError as e:
        print(f"{str(tuple(a.shape)):<16} op {str((3,)):<14} -> error: the 3 lines up with W=5, not with C")
    """),
    md(r"""
    ### Exercise 2 · `normalize`

    Implement `normalize(x, mean, std)` for `x` of shape `(C, H, W)` or `(B, C, H, W)`, with
    `mean` and `std` given as lists of length `C`. Two lines: turn them into tensors of
    shape `(C, 1, 1)`, then let broadcasting do the rest.
    """),
    exercise(
        stub="""
        def normalize(x, mean, std):
            # YOUR CODE HERE
            raise NotImplementedError

        tests.check_normalize(normalize)
        """,
        solution="""
        def normalize(x, mean, std):
            mean = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
            std = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
            return (x - mean) / std

        tests.check_normalize(normalize)
        """),
    code("""
    # Visual self-check: normalising changes the numbers, not the picture (once we undo it for display)
    x = to_chw(photo_u8)
    xn = normalize(x, torchkit.IMAGENET_MEAN, torchkit.IMAGENET_STD)
    print(f"before: mean per channel {x.mean(dim=(1, 2)).numpy().round(2)}   after: {xn.mean(dim=(1, 2)).numpy().round(2)}")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    axes[0].imshow(x.permute(1, 2, 0)); axes[0].set_title("to_chw(photo), shown as HWC")
    axes[1].hist([xn[c].flatten().numpy() for c in range(3)], bins=40, color=["#E45756", "#54A24B", "#4C78A8"],
                 label=["R", "G", "B"], histtype="step", lw=1.5)
    axes[1].set_title("normalised values per channel"); axes[1].legend()
    axes[2].imshow(torchkit.denormalize(xn)); axes[2].set_title("denormalize(normalize(x)) = x")
    axes[0].axis("off"); axes[2].axis("off")
    plt.tight_layout(); plt.show()
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · Autograd in five lines

    Mark a tensor with `requires_grad=True`, compute something from it, call `.backward()`
    on the scalar result, and the gradient appears in `.grad`. That is all a training loop
    needs. For $y = x^2 + 3x$ the derivative is $dy/dx = 2x + 3$, so at $x = 2$ we expect 7.
    """),
    code("""
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2 + 3 * x
    y.backward()
    print("y =", y.item(), "| dy/dx =", x.grad.item(), "(expected 2*2 + 3 = 7)")
    """),
    md(r"""
    Gradients **accumulate**: calling `backward()` again adds to `.grad` instead of replacing
    it. That is why every training step starts with `optimizer.zero_grad()`.
    """),
    code("""
    y = x ** 2 + 3 * x; y.backward()
    print("after a second backward() without zeroing: x.grad =", x.grad.item(), "(7 + 7)")
    x.grad = None
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · A ten-line training loop

    Every model in this course, from a linear layer to Mask R-CNN, is trained with the same
    five operations, in this order:

    1. **forward**: `pred = model(x)`
    2. **loss**: `loss = loss_fn(pred, y)`
    3. **zero the old gradients**: `optimizer.zero_grad()`
    4. **backward**: `loss.backward()` fills `p.grad` for every parameter
    5. **step**: `optimizer.step()` updates the parameters, e.g. $w \leftarrow w - \eta\, \partial L / \partial w$ for SGD

    We fit a line $y = 3x - 1$ hidden in noisy data with `nn.Linear(1, 1)` (one weight, one bias).
    """),
    code("""
    g = torch.Generator().manual_seed(0)
    X = torch.rand(256, 1, generator=g) * 4 - 2                     # 256 points in [-2, 2]
    Y = 3 * X - 1 + 0.4 * torch.randn(256, 1, generator=g)          # y = 3x - 1 + noise
    loader = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True)
    plt.figure(figsize=(4.5, 3.2)); plt.scatter(X, Y, s=8, alpha=0.6); plt.title("data: y = 3x - 1 + noise"); plt.show()
    """),
    md(r"""
    ### Exercise 3 · one training step

    Write `train_step(model, x, y, loss_fn, optimizer)`: the five operations above, returning
    the loss as a Python float (`loss.item()`). The checker runs two steps and compares the
    weights with two clean SGD steps, so the order matters.
    """),
    exercise(
        stub="""
        def train_step(model, x, y, loss_fn, optimizer):
            # YOUR CODE HERE: forward, loss, zero_grad, backward, step; return the loss as a float
            raise NotImplementedError

        tests.check_d0_train_step(train_step)
        """,
        solution="""
        def train_step(model, x, y, loss_fn, optimizer):
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.item()

        tests.check_d0_train_step(train_step)
        """),
    md(r"""
    ### Exercise 4 · the loop

    Put it together: create the model, the loss (`nn.MSELoss`) and an optimiser
    (`torch.optim.SGD` with `lr=0.1`), then loop over epochs and over the batches of `loader`,
    calling `train_step` and recording the mean loss of every epoch in `losses`. About ten lines.
    """),
    exercise(
        stub="""
        torch.manual_seed(0)
        model = nn.Linear(1, 1)
        losses = []
        # YOUR CODE HERE: loss_fn, optimizer, then 30 epochs over `loader`
        raise NotImplementedError

        tests.check_linear_fit(model)
        """,
        solution="""
        torch.manual_seed(0)
        model = nn.Linear(1, 1)
        losses = []
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for epoch in range(30):
            epoch_losses = [train_step(model, xb, yb, loss_fn, optimizer) for xb, yb in loader]
            losses.append(sum(epoch_losses) / len(epoch_losses))

        tests.check_linear_fit(model)
        """),
    code("""
    # Visual self-check: the loss should drop close to the noise level (0.4^2 = 0.16) and the line should sit on the data
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot(range(1, len(losses) + 1), losses, marker="o", ms=3)
    axes[0].axhline(0.16, color="gray", ls="--", lw=1, label="noise variance 0.16")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE"); axes[0].set_yscale("log")
    axes[0].set_yticks([0.16, 0.5, 1, 2, 4], ["0.16", "0.5", "1", "2", "4"]); axes[0].minorticks_off(); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title("training loss")
    xs = torch.linspace(-2, 2, 50).unsqueeze(1)
    with torch.no_grad():
        axes[1].plot(xs, model(xs), color="#E45756", lw=2, label=f"learned: y = {model.weight.item():.2f}x {model.bias.item():+.2f}")
    axes[1].plot(xs, 3 * xs - 1, color="gray", ls="--", lw=1, label="true: y = 3x - 1")
    axes[1].scatter(X, Y, s=6, alpha=0.4); axes[1].legend(); axes[1].set_title("fit")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    From Day 1 on, the loop lives in `torchkit.fit(model, train_loader, val_loader, loss_fn,
    optimizer, epochs, ...)`: the same five lines plus validation, a learning-rate schedule,
    a time budget and checkpointing. It returns a `history` dict you can plot. Here it is
    on the same problem:
    """),
    code("""
    torch.manual_seed(0)
    model2 = nn.Linear(1, 1)
    history = torchkit.fit(model2, loader, loader, nn.MSELoss(), torch.optim.SGD(model2.parameters(), lr=0.1),
                           epochs=3, log_every=0)
    print("history keys:", list(history))
    """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · How every notebook of the course works

    Each mechanism is taught in the same four beats. Knowing the rhythm lets you spend your
    attention on the content:

    | beat | what you see | what you do |
    |---|---|---|
    | **1 · Figure** | a generated picture of the mechanism, with the one formula that matters | read the "what to look at" sentence, find it in the picture |
    | **2 · Lab** | an interactive panel under a `labs.show(...)` cell | answer the **Predict first** question in your head, *then* drag / slide to check, then do the "Try this" list |
    | **3 · Exercise** | a cell with `# YOUR CODE HERE` and `raise NotImplementedError` | write the 3-15 lines that *are* the concept |
    | **4 · Checker** | `tests.check_...` prints `[PASS]`, or `[FAIL]` with a hint | fix until it passes, then move on |

    Work **cell by cell, top to bottom**. An unfinished exercise stops "Run all" at that
    cell on purpose: later cells use your function. If you are stuck for more than five
    minutes, open the same notebook in `notebooks/solutions/`, read that one cell, close it,
    and type the answer yourself.

    The labs are small web pages that ship with the course: no installation, no internet.
    Run the next cell to check that they work on your machine. You should see a photo, an
    edge map, and a 3x3 kernel you can edit. Move the pointer over the photo.
    """),
    code("""
    from cvcourse import labs
    labs.show("conv-playground")
    """),
    md(r"""
    If the panel stays empty (some notebook viewers block embedded pages), open the lab in a
    browser tab instead with `labs.open("conv-playground")`, or double-click
    `notebooks/cvcourse/labs/index.html`, which lists every lab of the course.
    `labs.catalog()` prints the list with the notebook each lab belongs to.
    """),
    code("""
    labs.catalog();
    """),

    # ------------------------------------------------------------------ 7
    md(r"""
    ## 7 · Summary

    * Check your setup once: PyTorch + torchvision imported, data in `data/`, a device picked.
      CPU is fine for this course.
    * Images: `(H, W, C)` uint8 on disk -> `(C, H, W)` float in `[0, 1]` -> normalised ->
      batched `(B, C, H, W)`. `permute`, `float() / 255`, broadcasting with `(C, 1, 1)`, `unsqueeze(0)`.
    * Autograd: `requires_grad` -> compute -> `.backward()` -> `.grad`. Gradients accumulate.
    * Training step: forward, loss, `zero_grad`, `backward`, `step`. Everything else is bookkeeping.

    ### Self-check
    1. A batch has shape `(16, 3, 224, 224)`. Which axis is the channel axis, and what does `x.mean(dim=(0, 2, 3))` compute?
    2. Why does `torch.zeros(2, 3, 4, 5) + torch.zeros(3)` fail while `+ torch.zeros(3, 1, 1)` works?
    3. What goes wrong if you forget `optimizer.zero_grad()`?
    4. Why can a 480x640 photo and a 375x500 photo not be stacked into one tensor, and what do detection data loaders do instead?

    ### Go deeper
    * PyTorch tutorials: "Tensors", "A gentle introduction to torch.autograd", "Optimizing model parameters".
    * NumPy documentation, "Broadcasting" (the same rules apply to PyTorch).
    """),
]
