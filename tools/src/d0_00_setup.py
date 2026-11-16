from nb import md, code, exercise, header, api

NAME = "D0-00-setup-and-tensors"

CELLS = [
    header(
        "D0-00 · Setup and tensors (optional pre-course)",
        steps=[
            "Check that your computer can run the course: Python, PyTorch, torchvision, and a GPU if there is one",
            "Download the course datasets (once, about 850 MB)",
            "Tensors and images: shape, dtype, pixels and channels, the HWC and CHW layouts, batches (exercise 1)",
            "Normalisation and broadcasting: put every colour channel on the scale networks expect (exercise 2)",
            "The four datasets of the course and their labels: classes, boxes, masks",
            "Autograd: PyTorch computes the gradients for you",
            "Gradient descent by hand, then the training loop (exercises 3 and 4)",
            "The rhythm of every notebook (figure, lab, exercise, checker) and a check that the interactive labs work",
        ],
        topics=[
            "Confirm your machine runs every notebook of the course (a CPU is enough)",
            "Read a tensor's shape and dtype, and say what each axis of (B, C, H, W) means",
            "Turn an image from the file / Matplotlib layout (H, W, C) into the PyTorch layout (C, H, W), "
            "and explain why `reshape` cannot do it",
            "Explain what normalisation does to the pixel values and why pretrained networks need it; "
            "normalise a whole batch in one line with broadcasting",
            "Know the four datasets: FashionMNIST, Oxford-IIIT Pet, Penn-Fudan, synthetic shapes, and their labels",
            "Explain what a gradient is, what autograd computes, and why gradients must be zeroed before every step",
            "Write the forward -> loss -> zero_grad -> backward -> step loop from memory",
        ],
        minutes=90,
        setting="on your own, before the course (plus the data download)",
        prereq="Python and basic NumPy (arrays, `.shape`, indexing and slicing). You have heard what a neural "
               "network is. No PyTorch and no image-processing experience needed. The backpropagation lecture of "
               "the previous class helps in sections 6 and 7, but is not required.",
    ),
    md(r"""
    ## How to read this notebook

    This notebook is homework before the course: work through it alone, top to bottom. It takes
    about 90 minutes (half reading, half running cells and the four exercises), plus the one-time
    download in section 2.

    * **Run the cells in order.** Click a code cell and press Shift+Enter. Cells use what earlier
      cells made, so a skipped cell gives errors further down.
    * **Exercises.** Four cells, each under a heading `### Exercise N`, contain `# YOUR CODE HERE`
      and `raise NotImplementedError`. Replace the `raise` line with your code, then run the cell.
      The text above every exercise gives the goal, what goes in and what must come out (with an
      example), a plan in steps, and the common mistakes.
    * **Checkers.** The last line of an exercise cell calls a checker, for example
      `tests.check_to_chw(to_chw)`. It tries your code on inputs of its own and prints `[PASS]`, or
      `[FAIL]` with what went wrong and a hint.
    * **When it says `[FAIL]`:** read the indented lines just above `[FAIL]` (which input failed, what
      came out, what was expected), then the hint, then the "Common mistakes" table under the
      exercise. Fix your code and run the cell again.
    * **Cells after an unfinished exercise** that use your function stop with an error such as
      `NotImplementedError`. That is expected: finish the exercise, then run them again.
    * **Stuck for more than five minutes?** The same notebook with every exercise solved is
      `notebooks/solutions/D0-00-setup-and-tensors-solution.ipynb`. Read that one cell, close it, and
      type the answer yourself.
    * **Fold-out boxes.** Before most code cells there is a closed box **Functions used here** (before
      an exercise: **Functions you will need**). Click it to see what each library function in the
      next cell does, with a small example and its real output. In any cell,
      `from cvcourse import apiref; apiref.show("permute")` prints the same explanation.
    * **New words** are in **bold** where they are explained. All of them are collected in
      "Words from this notebook" at the very end.
    * Boxes titled **More:** are optional depth. Skip them on a first read.
    """),

    # ------------------------------------------------------------------ 0
    md(r"""
    ## 0 · Setup

    Run this cell first, every time you open the notebook. Every notebook of the course starts with
    a cell like it. Line by line:

    * the `for` loop finds the folder `cvcourse`, the course's own helper code (data loading,
      plots, exercise checkers), and lets Python import it;
    * it imports NumPy, Matplotlib (plots) and **PyTorch** (`torch`), the deep-learning library of the
      course, with `torch.nn`, its building blocks for neural networks;
    * `torch.manual_seed(0)` fixes the **random seed**, the starting point of PyTorch's random number
      generator. With the same seed, every "random" number, and so every result, is the same each
      time you run the notebook;
    * `torchkit.get_device()` picks the **device**, the processor that will do the arithmetic, and
      prints its choice.

    A **CPU** is your computer's main processor. A **GPU** (graphics card) has thousands of small cores
    that do many multiplications at the same time, which makes training networks much faster. PyTorch
    uses NVIDIA GPUs through **CUDA**, NVIDIA's toolkit for computing on its cards, so such a GPU is the
    device `"cuda"`; an Apple-silicon Mac's GPU is `"mps"`. You do not need a GPU for this course:
    `device: cpu` is fine.
    """),
    api("torch.manual_seed", "torchkit.get_device"),
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

    The course needs PyTorch 2.x and **torchvision**, PyTorch's companion library for images
    (datasets, image transformations, ready-made networks). Four words you will meet from here on:

    * a **model** is a neural network: a function with many adjustable numbers inside, which turns an
      input (an image) into an output (for example one score per possible answer);
    * those adjustable numbers are its **parameters**, also called **weights**;
    * **training** means changing the parameters a little at a time, so that the model's outputs get
      closer to the right answers on many examples (sections 6 and 7 show exactly how);
    * a **checkpoint** is a file that holds a model's trained parameters, so that a notebook can load
      them instead of training again.

    A GPU is welcome but **not** required: every training cell is sized to finish in a few minutes on
    a laptop CPU, and the longer ones load a saved checkpoint instead of training when one exists.
    """),
    api("torch.cuda.is_available", "torch.cuda.get_device_name", "torch.get_num_threads"),
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
    **What you should see:** `torch 2.something` and a `torchvision` version. `CUDA GPU False` is fine.
    The three optional libraries (timm, transformers, open_clip) provide ready-made networks for a few
    demonstrations on Days 1 to 3; "not installed" is fine. If the setup cell failed at `import torch`,
    the notebook is not using the course environment: see "If something goes wrong" in the README.
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · Download the data

    A **dataset** is a collection of examples. Each example is an input (here an image) and its
    **label**: the right answer the model should learn to give, for example the breed of the dog in
    the photo. The course uses four datasets; section 5 looks at each of them.

    Three of them come from the internet. The next cell downloads them once into `data/` at the
    repository root (about 850 MB) and skips whatever is already there, so it is safe to run again;
    the first run can take several minutes. Trained weights are saved to `data/checkpoints/` and
    reused when you re-run a notebook. `torchkit.data_root()` finds the `data/` folder from any
    notebook. The output lists each dataset with what was done (`already complete` on later runs) and
    ends with the path of the data folder.
    """),
    api("torchkit.data_root"),
    code("""
    import subprocess
    REPO = torchkit.data_root().parent
    result = subprocess.run([sys.executable, str(REPO / "tools" / "download_data.py")], capture_output=True, text=True)
    print(result.stdout[-1500:] or result.stderr[-1500:])
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · Tensors and images

    ### What a tensor is

    A **tensor** is PyTorch's array: a grid of numbers, like a NumPy array. Each direction you can
    move along the grid is an **axis**, also called a **dimension**: a single number has 0 axes, a
    list 1, a table 2, a colour image 3, a stack of colour images 4. A tensor can do two things a
    NumPy array cannot: live on a GPU, and remember how it was computed (section 6 shows why that is
    useful).

    Three attributes describe every tensor. Print them first whenever something does not fit:

    | attribute | what it tells you | example |
    |---|---|---|
    | `shape` | the size along each axis | `(3, 4)`: 3 rows, 4 columns |
    | `dtype` | the data type: what kind of number every element is | `torch.float32` |
    | `device` | where the numbers are stored and computed | `cpu`, or `cuda:0` for the first GPU |

    In the next cell, `torch.arange(12)` makes the numbers 0 to 11 and `reshape(3, 4)` arranges them
    in 3 rows of 4. Indexing works as in NumPy and counts from 0. `t.sum(dim=0)` adds *along* axis 0,
    down the rows: that axis disappears and one sum per column is left. `t.numpy()` and
    `torch.from_numpy(...)` convert between PyTorch and NumPy without copying the numbers.
    """),
    api("Tensor.shape", "Tensor.dtype", "Tensor.device", "Tensor.to", "torch.arange", "Tensor.reshape",
        "Tensor.tolist", "Tensor.sum", "Tensor.numpy", "torch.from_numpy"),
    code("""
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print(t)
    print("shape", tuple(t.shape), "| dtype", t.dtype, "| device", t.device)
    print("row 1:", t[1].tolist(), "| column 2:", t[:, 2].tolist(), "| sum over rows:", t.sum(dim=0).tolist())
    print("to NumPy and back:", type(t.numpy()).__name__, type(torch.from_numpy(t.numpy())).__name__)
    """),
    md(r"""
    Check the third line by hand: column 0 holds 0, 4 and 8, so its sum is 12.

    ### Data types: uint8 and float32

    An image file stores every number as **uint8**, an *unsigned 8-bit integer*: a whole number from 0
    to 255 in one byte. Networks compute in **float32**: decimal numbers ("floating point") in 4 bytes.
    The two behave differently, and mixing them up gives wrong numbers without any error message.

    | dtype | holds | range | used for |
    |---|---|---|---|
    | `torch.uint8` | whole numbers, 1 byte | 0 to 255 | pixel values as stored in image files |
    | `torch.float32` | decimal numbers, 4 bytes | huge, about 7 significant digits | everything a network computes: inputs, weights, outputs |
    | `torch.int64` | whole numbers, 8 bytes | huge | class labels, positions |
    | `torch.bool` | `True` / `False` | | masks, results of comparisons |

    **Predict first:** a uint8 holds nothing above 255. What do you think `250 + 10` gives in uint8?
    """),
    api("torch.tensor", "Tensor.float"),
    code("""
    pix = torch.tensor([100, 200, 250], dtype=torch.uint8)   # three pixel values, as stored in a file
    print("uint8 + 10 :", pix + 10)                          # 250 + 10 = 260 does not fit: it wraps around past 255
    print("float + 10 :", pix.float() + 10)                  # convert to float32 first: 260 is fine
    print("float / 255:", pix.float() / 255)                 # the usual scaling to [0, 1]: 0 -> 0.0, 255 -> 1.0
    """),
    md(r"""
    `250 + 10` became 4: after 255 a uint8 starts again at 0 (260 - 256 = 4). So the first thing to do
    with an image is to convert it to float32. Dividing by 255 then maps 0 ... 255 onto 0 ... 1, the
    range the rest of the course (and section 4's normalisation) assumes.

    ### An image is a tensor

    A digital image is a grid of **pixels** ("picture elements"): the small squares you see when you
    zoom in far enough. A grayscale image stores one number per pixel, its brightness. A colour image
    stores three: how much **r**ed, **g**reen and **b**lue light the pixel has (RGB), each from 0 (none)
    to 255 (full). The red numbers of all pixels together form the red **channel**, a grayscale
    picture of its own; the same for green and blue. So a colour photo has three axes: height (rows
    of pixels), width (columns of pixels) and channels.

    The next cell loads a photo that ships with Matplotlib (Grace Hopper, a pioneer of programming)
    the way an image file arrives: uint8, with the axes (height, width, channels).
    """),
    code("""
    photo_u8 = (viz.sample_photo() * 255).astype(np.uint8)        # as read from a file: uint8, (H, W, C)
    H, W, C = photo_u8.shape
    print("shape", photo_u8.shape, "->", H, "rows,", W, "columns,", C, "channels | dtype", photo_u8.dtype)
    print("pixel at row 300, column 250 (on the face):  R, G, B =", photo_u8[300, 250].tolist())
    print("pixel at row 50, column 450 (the background): R, G, B =", photo_u8[50, 450].tolist())

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    axes[0].imshow(photo_u8); axes[0].set_title("the photo, (H, W, C)")
    for c, name in enumerate(["red", "green", "blue"]):
        axes[c + 1].imshow(photo_u8[:, :, c], cmap="gray", vmin=0, vmax=255)
        axes[c + 1].set_title(f"channel {c} ({name})\\nphoto_u8[:, :, {c}]", fontsize=10)
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: each channel is a grayscale picture, white where that colour is strong. The
    background is blue, so it is bright in the blue channel and darker in the red one; the face is
    brightest in the red channel, as its pixel values (much more red than blue) say.

    ### HWC vs CHW: why every image gets its axes reordered

    The same photo can be stored with its axes in two orders:

    * **HWC** = (height, width, channels): for every row, for every pixel in that row, its three colour
      values side by side. Image files, NumPy, PIL (Pillow, the library that opens image files),
      OpenCV and Matplotlib use this order.
    * **CHW** = (channels, height, width): first the whole red picture, then the whole green one, then
      the whole blue one. PyTorch's image layers use this order.

    | layout | who uses it | this photo |
    |---|---|---|
    | (H, W, C) | image files, NumPy, PIL, OpenCV, Matplotlib | (600, 512, 3) |
    | (C, H, W) | one image in PyTorch | (3, 600, 512) |

    **Why this matters.** A layer does not look at your picture; it reads the axes by position.
    PyTorch's image layers (you meet them on Day 1) take the first axis of an image as its
    channels. Give one this photo in HWC and it sees 600 channels; it stops with
    `expected input ... to have 3 channels, but got 600 channels instead`. Matplotlib has the
    opposite rule: `plt.imshow` on a CHW tensor fails with `Invalid shape (3, 2, 2) for image data`.
    So every image you feed to a network goes HWC to CHW, and every image you plot goes back.

    **Worked example.** A 2 x 2 colour image with numbers chosen so that you can follow each one: the
    tens digit is the channel (1 = red, 2 = green, 3 = blue), the ones digit is the pixel (1 to 4,
    reading row by row). Pixel 1 (top left) is red 11, green 21, blue 31. In HWC the 12 numbers lie in
    memory pixel by pixel, three colours each:

    `11 21 31 | 12 22 32 | 13 23 33 | 14 24 34`

    `permute` reorders whole axes. Its arguments list the *old* axis numbers in their new order: HWC has
    H = axis 0, W = axis 1, C = axis 2, so CHW is `permute(2, 0, 1)`. Written out, the result is three
    2 x 2 pictures: red `[[11, 12], [13, 14]]`, green `[[21, 22], [23, 24]]`, blue `[[31, 32], [33, 34]]`.

    **Predict first:** `reshape(3, 2, 2)` gives the same shape (3, 2, 2). What is in its "red channel"?
    Hint: reshape keeps the memory order above and only cuts it into pieces of the new shape.
    """),
    api("np.array", "Tensor.permute", "Tensor.flatten"),
    code("""
    img_hwc = np.array([[[11, 21, 31], [12, 22, 32]],
                        [[13, 23, 33], [14, 24, 34]]], dtype=np.uint8)      # (H=2, W=2, C=3)
    t = torch.from_numpy(img_hwc)
    print("HWC", tuple(t.shape), "| memory order:", t.flatten().tolist())
    print("pixel (row 0, column 0):", t[0, 0].tolist(), "= its red, green, blue\\n")
    print("permute(2, 0, 1): CHW", tuple(t.permute(2, 0, 1).shape))
    print(t.permute(2, 0, 1), "\\n")
    print("reshape(3, 2, 2): also", tuple(t.reshape(3, 2, 2).shape), "but scrambled")
    print(t.reshape(3, 2, 2))
    """),
    md(r"""
    `reshape` filled its "channel 0" with the first four numbers in memory, `11 21 31 12`: red, green,
    blue and red again. The figure makes this visible: each cell is tinted with the colour its number
    really belongs to. Below it, the same two operations on the photo.
    """),
    code("""
    from matplotlib.colors import to_rgb
    tint = {1: to_rgb("#F4B6B6"), 2: to_rgb("#B9E3B2"), 3: to_rgb("#B6CDF0")}   # tens digit = the true channel
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.8), layout="constrained")
    for row, (how, chw) in enumerate([("permute(2, 0, 1)", t.permute(2, 0, 1)), ("reshape(3, 2, 2)", t.reshape(3, 2, 2))]):
        for c in range(3):
            vals = chw[c].numpy()
            axes[row, c].imshow(np.array([[tint[int(v) // 10] for v in r] for r in vals]))
            viz.annotate_grid(axes[row, c], vals, fontsize=14)
            axes[row, c].set_title(f"{how}\\nchannel {c}", fontsize=10)
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
    fig.suptitle("cell colour = the channel the number really belongs to (1x red, 2x green, 3x blue)", fontsize=10)
    plt.show()

    hwc = torch.from_numpy(photo_u8)                                    # (600, 512, 3)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(photo_u8); axes[0].set_title("the photo, (H, W, C)")
    axes[1].imshow(hwc.permute(2, 0, 1)[0], cmap="gray"); axes[1].set_title("permute(2, 0, 1), channel 0:\\nthe red channel")
    axes[2].imshow(hwc.reshape(C, H, W)[0], cmap="gray"); axes[2].set_title(f"reshape({C}, {H}, {W}), 'channel 0':\\nscrambled")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: in the top row (permute) every channel has a single colour; in the bottom row
    (reshape) every "channel" mixes colours. On the photo, reshape's "channel 0" is not the red
    picture at all: it shows only the top of the photo (cap and flag), stretched and overlaid three
    times, because it is simply the first third of the memory read as if it were one channel. No
    error is raised: the shape is right and the content is garbage. Hence the rule: **move axes with
    `permute`, never with `reshape` or `view`.**

    ### The batch axis

    Networks never take a single image. They take a **batch**: several images stacked along a new
    first axis, giving (B, C, H, W) with B the batch size (often written NCHW, N for "number of
    images"). Two reasons: computing 32 images at once is much faster than computing them one by one,
    and training averages its correction over several examples (section 7). A single image becomes a
    batch of one: `unsqueeze(0)` inserts a new axis of size 1 in front.

    | layout | meaning | this photo |
    |---|---|---|
    | (C, H, W) | one image in PyTorch | (3, 600, 512) |
    | (B, C, H, W) | a batch of B images | (1, 3, 600, 512) |

    ### View or copy

    `permute` and `unsqueeze` copy no pixel. They return a **view**: a new tensor that reads the *same*
    memory in another order or shape. Views are instant even for big images, but they share their
    numbers with the original: change one and the other changes too. A **copy** has memory of its
    own. The next cell checks this with `data_ptr()`, the memory address where a tensor's numbers
    start.
    """),
    api("Tensor.unsqueeze", "Tensor.data_ptr", "Tensor.is_contiguous", "Tensor.contiguous"),
    code("""
    chw = torch.from_numpy(photo_u8).permute(2, 0, 1)              # (C, H, W), still uint8, no copy
    batch = chw.unsqueeze(0)                                        # (1, C, H, W)
    print("HWC:", photo_u8.shape, "| CHW:", tuple(chw.shape), "| batch:", tuple(batch.shape))
    print("shares memory with the NumPy photo:", chw.data_ptr() == torch.from_numpy(photo_u8).data_ptr())
    print("contiguous after permute?", chw.is_contiguous(), "-> .contiguous() makes a real copy if a library needs one")
    """),
    md(r"""
    <details>
    <summary><b>More: why views matter</b> (optional, click to open)</summary>

    Memory is one long row of numbers. A tensor is that memory plus a recipe for reading it: its shape,
    and for every axis how many places to jump to reach the next element. `permute` and `unsqueeze`
    change only the recipe. In the 2 x 2 example the memory stays `11 21 31 12 22 32 ...`; to read the
    red channel of the permuted tensor, PyTorch starts at 11 and jumps 3 places at a time: 11, 12, 13, 14.

    A tensor is **contiguous** when its memory order is the order in which you read it (row by row in
    its current shape). After `permute` it usually is not, which matters in two places:

    * `x.view(...)` never copies, so it only works on contiguous tensors; `x.reshape(...)` copies when it
      has to. After a permute, use `reshape`, or `x.contiguous().view(...)`.
    * A few libraries read the raw memory; `x.contiguous()` makes the copy they need.

    Writing into a view writes into the original: `chw[0, 0, 0] = 0` would also change `photo_u8`.
    When you need an independent tensor, make a copy (`x.clone()`).

    </details>

    ### Exercise 1 · `to_chw`

    **Goal.** Write the function that turns any image as loaded from a file into the tensor a PyTorch
    network expects. Every image of the course goes through these steps.

    **Input.** `img_hwc`, a NumPy array of shape (H, W, C), either
    * `uint8` with values 0 to 255 (as read from a file), or
    * `float32` with values already in [0, 1] (as `viz.sample_photo()` returns).

    **Output.** A `torch.Tensor` of shape (C, H, W), dtype `torch.float32`, values in [0, 1]: uint8
    values divided by 255, float values unchanged.

    **Example.** The 2 x 2 image above (uint8, values 11 to 34) must give:
    ```
    to_chw(img_hwc)   ->   shape (3, 2, 2), torch.float32
    tensor([[[0.0431, 0.0471],       # red:   11/255, 12/255
             [0.0510, 0.0549]],      #        13/255, 14/255
            [[0.0824, 0.0863],       # green: 21/255, ...
             [0.0902, 0.0941]],
            [[0.1216, 0.1255],       # blue:  31/255, ...
             [0.1294, 0.1333]]])
    ```
    A float32 image of shape (2, 2, 3) full of 0.5 gives shape (3, 2, 2), still all 0.5.

    **Plan.**
    1. `torch.from_numpy(img_hwc)` (already in the stub) gives a tensor with the same shape and dtype.
    2. Move the channel axis from position 2 to the front with `permute`.
    3. Convert to float32.
    4. If the *input* was uint8, divide by 255; otherwise leave the values alone. Decide this from
       `img_hwc.dtype` before you convert (afterwards every image looks like a float image).
    5. Return the result.

    **What the checker tests.** It calls your function on a random uint8 image of shape (5, 7, 3) and a
    random float32 image of shape (6, 4, 3), and compares the type, shape, dtype and every value with
    the expected result. Height and width differ on purpose, so that swapped axes show.

    **Common mistakes.**

    | you wrote | the checker prints |
    |---|---|
    | `reshape(C, H, W)` instead of `permute` | `right shape, but the pixels are scrambled` for both images: `reshape` re-read the same numbers in a new shape, as in the figure above. Use `permute(2, 0, 1)` |
    | `permute(2, 1, 0)` | `shape (3, 7, 5), expected (3, 5, 7)`: height and width swapped |
    | divided every image by 255 | `float32 (6, 4, 3): values differ`: float images are already in [0, 1] |
    | `t.dtype == np.uint8` | `uint8 (5, 7, 3): values differ`: a PyTorch dtype never equals a NumPy dtype, so the uint8 image was not divided. Compare `img_hwc.dtype == np.uint8`, or `t.dtype == torch.uint8` |
    | converted to float *before* checking the dtype | `uint8 (5, 7, 3): values differ`, for the same reason |
    | `.double()` instead of `.float()` | `dtype torch.float64, expected torch.float32` |
    | `np.transpose`, or no `return` | `returned ndarray` or `returned NoneType, expected a torch.Tensor` |
    """),
    api("Tensor.permute", "Tensor.float", title="Functions you will need"),
    exercise(
        stub="""
        def to_chw(img_hwc):
            # 1. a tensor with the same shape (H, W, C) and dtype as the NumPy array
            t = torch.from_numpy(img_hwc)
            # 2. YOUR CODE: move the channel axis to the front with permute -> shape (C, H, W)
            # 3. YOUR CODE: convert to float32
            # 4. YOUR CODE: if the input img_hwc is uint8, divide by 255 (float images are already in [0, 1])
            # 5. YOUR CODE: return the result
            raise NotImplementedError

        tests.check_to_chw(to_chw)
        """,
        solution="""
        def to_chw(img_hwc):
            # 1. a tensor with the same shape (H, W, C) and dtype as the NumPy array
            t = torch.from_numpy(img_hwc)
            # 2. move the channel axis to the front with permute
            t = t.permute(2, 0, 1)                 # (C, H, W)
            # 3. convert to float32
            t = t.float()
            # 4. if the input img_hwc is uint8, divide by 255 (float images are already in [0, 1])
            if img_hwc.dtype == np.uint8:
                t = t / 255
            # 5. return the result
            return t

        tests.check_to_chw(to_chw)
        """),
    md(r"""
    **Did it work?** `[PASS]` means your function turns any image array into the format a network
    takes: (C, H, W), float32, values in [0, 1]. It is the first half of every image pipeline of the
    course. From Day 1 on, torchvision does the same in two library steps, `v2.ToImage()` (HWC to CHW)
    and `v2.ToDtype(torch.float32, scale=True)` (to float32, divided by 255): `torchkit.get_pet_datasets`
    runs them on every Pet photo, and D1-02 builds pipelines of such steps. The opposite direction,
    `x.permute(1, 2, 0)`, comes back whenever a notebook shows a tensor with `plt.imshow`.
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · Normalisation and broadcasting

    ### Why normalise?

    After `to_chw` the pixel values are in [0, 1]. Networks get one more preparation step,
    **normalisation**: shift and scale the values of each channel so that, over a large collection of
    photos, they have mean 0 and standard deviation 1.

    * A **pretrained** network, one that someone else already trained (usually on ImageNet, a
      collection of more than a million labelled photos), learned its weights on inputs normalised
      with ImageNet's numbers. Day 1, block 4 (transfer learning) and much of Days 2 and 3 reuse such
      networks. Feed one raw [0, 1] values and every input number is off from what its weights expect:
      nothing raises an error, the network just gives worse answers. Its inputs must be prepared
      exactly as its training inputs were.
    * A network trained from scratch also learns faster and more steadily when every input channel is
      centred on 0 with a similar spread: no channel dominates only because its numbers are bigger.

    ### What it is: mean, standard deviation, normalising

    * The **mean** of a set of numbers is their average.
    * The **standard deviation** (std) measures their spread: the square root of the average squared
      distance from the mean. Roughly: how far a typical number lies from the mean.
    * **Normalising** a number x means computing (x - mean) / std. The result says how many standard
      deviations x lies above (positive) or below (negative) the mean. Done to every number of the
      set, the results have mean 0 and std 1.

    **Worked example.** The numbers 2, 4, 4, 4, 5, 5, 7, 9. By hand: the mean is 40 / 8 = 5; the
    squared distances from 5 add up to 32, their average is 4, its square root is 2. So the std is 2,
    and 9, which lies 4 above the mean, becomes 4 / 2 = 2.
    """),
    api("Tensor.mean", "Tensor.item"),
    code("""
    v = torch.tensor([2., 4., 4., 4., 5., 5., 7., 9.])
    mean = v.mean()
    dist = v - mean                                    # distance of every number from the mean
    std = (dist ** 2).mean() ** 0.5                    # square root of the average squared distance
    print("values            :", v.tolist())
    print("mean              :", mean.item())
    print("distance from mean:", dist.tolist())
    print("squared distance  :", (dist ** 2).tolist(), "-> average", (dist ** 2).mean().item(), "-> square root", std.item())
    z = (v - mean) / std
    print("normalised        :", z.tolist())
    print("normalised mean", z.mean().item(), "and std", ((z - z.mean()) ** 2).mean().item() ** 0.5)
    """),
    md(r"""
    ### Per channel, with fixed numbers

    For images we do not compute the mean and std of each photo. We use fixed numbers, computed once
    over the whole ImageNet training set (pixel values in [0, 1]), one pair **per channel** because the
    colours differ (photos contain more red than blue, on average):

    | channel | mean | std |
    |---|---|---|
    | red | 0.485 | 0.229 |
    | green | 0.456 | 0.224 |
    | blue | 0.406 | 0.225 |

    The whole field uses them; in the course they are `torchkit.IMAGENET_MEAN` and
    `torchkit.IMAGENET_STD`. By hand, for the red values of three pixels: 0.256 becomes
    (0.256 - 0.485) / 0.229 = -1; 0.485 becomes 0; 0.943 becomes (0.943 - 0.485) / 0.229 = 2.
    """),
    code("""
    vals = torch.tensor([0.256, 0.485, 0.943])         # the red value of three pixels, in [0, 1]
    print((vals - 0.485) / 0.229)                      # ImageNet's red mean and std
    """),
    md(r"""
    Now one whole channel of the photo: its red values before and after the same formula.
    """),
    api("Tensor.min", "Tensor.max", "Tensor.std"),
    code("""
    red = torch.from_numpy(photo_u8[:, :, 0]).float() / 255        # the red channel, (600, 512), in [0, 1]
    red_n = (red - 0.485) / 0.229                                   # normalised with ImageNet's red mean and std
    for name, ch in [("before", red), ("after ", red_n)]:
        print(f"{name}: min {ch.min().item():5.2f}   max {ch.max().item():5.2f}   mean {ch.mean().item():5.2f}   std {ch.std().item():4.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, ch, title in [(axes[0], red, "red channel in [0, 1]"), (axes[1], red_n, "after (x - 0.485) / 0.229")]:
        ax.hist(ch.flatten().numpy(), bins=50, color="#E45756")
        ax.axvline(0, color="gray", ls="--", lw=1)
        ax.set_title(title); ax.set_xlabel("value")
    axes[0].set_ylabel("number of pixels")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: the two histograms have exactly the same shape; only the numbers on the
    horizontal axis change. Normalisation shifts and stretches the values, it does not change the
    picture. Here the std grew from 0.31 to 1.36 (it is divided by 0.229) and the mean moved from 0.32
    to -0.71. The mean is not exactly 0 because this photo is darker than an average ImageNet photo;
    that is fine. What matters is that the photo is now on the scale the network was trained on.

    ### Broadcasting: one line for all channels

    To normalise a whole image we subtract a different mean from each channel. A loop over the
    channels would work, but there is a shorter and faster way. **Broadcasting** is how PyTorch (like
    NumPy) combines two tensors of different shapes in one operation:

    1. Line the two shapes up **from the right**.
    2. Each pair of sizes must be equal, or one of them must be 1 (a missing axis on the left counts as 1).
    3. An axis of size 1 is stretched (repeated) to match the other one. No memory is copied for this.

    | x | other | lined up from the right | result |
    |---|---|---|---|
    | (3, 2, 2) one image | (3, 1, 1) one value per channel | 2 with 1, 2 with 1, 3 with 3: fine | (3, 2, 2): each channel's value is repeated over all its pixels |
    | (8, 3, 2, 2) a batch | (3, 1, 1) | 2 with 1, 2 with 1, 3 with 3, 8 with nothing: fine | (8, 3, 2, 2): the same values for every image too |
    | (3, 2, 2) one image | (3,) | 2 with 3: neither is 1 | error |

    The last row is the classic mistake: a vector of shape (3,) lines up with the *width*, not with the
    channels. Here it is on the 2 x 2 image: subtracting (10, 20, 30) channel by channel leaves
    1, 2, 3, 4 in every channel. `.view(3, 1, 1)` gives the same three numbers the shape (3, 1, 1).
    """),
    api("Tensor.view"),
    code("""
    tiny = torch.from_numpy(img_hwc).permute(2, 0, 1).float()       # the 2 x 2 image as CHW floats: 11..14, 21..24, 31..34
    per_channel = torch.tensor([10., 20., 30.])
    print("per_channel.view(3, 1, 1) has shape", tuple(per_channel.view(3, 1, 1).shape))
    print(tiny - per_channel.view(3, 1, 1))                         # red - 10, green - 20, blue - 30
    try:
        tiny - per_channel                                          # shape (3,) lines up with the width 2
    except RuntimeError as e:
        print("with shape (3,):", e)
    """),
    md(r"""
    The next cell adds tensors of other shapes to a batch of 2 images with 3 channels of 4 x 5 pixels.
    Each line shows the two shapes (`op` = the addition) and the shape of the result. Every line
    passes rule 2, except the last.
    """),
    api("torch.ones", "torch.zeros"),
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

    **Goal.** One function that normalises a single image or a whole batch, channel by channel, with
    given means and standard deviations:

    $$x_{\text{norm}}[c, i, j] = \frac{x[c, i, j] - \mu_c}{\sigma_c}$$

    where $\mu_c$ is the mean and $\sigma_c$ the std of channel c (and i, j the row and column).

    **Input.** `x`: a float tensor (C, H, W) or (B, C, H, W). `mean`, `std`: Python lists of C numbers,
    for example `torchkit.IMAGENET_MEAN` and `torchkit.IMAGENET_STD`.

    **Output.** A tensor of the same shape as `x`, normalised by the formula above, for every image of
    a batch.

    **Example.** `tiny` from the cell above, with mean (10, 20, 30) and std (1, 2, 4):
    ```
    normalize(tiny, [10, 20, 30], [1, 2, 4])
    tensor([[[1.0000, 2.0000],       # red:   (11 - 10) / 1, (12 - 10) / 1, ...
             [3.0000, 4.0000]],
            [[0.5000, 1.0000],       # green: (21 - 20) / 2, (22 - 20) / 2, ...
             [1.5000, 2.0000]],
            [[0.2500, 0.5000],       # blue:  (31 - 30) / 4, ...
             [0.7500, 1.0000]]])
    ```

    **Plan.**
    1. Turn `mean` into a tensor with `torch.tensor`, with `dtype=x.dtype` so that it matches `x`.
    2. Give it the shape (C, 1, 1) with `.view(-1, 1, 1)`, so that it lines up with the channel axis
       (-1 means "whatever size is left", here C).
    3. The same for `std`.
    4. Return `(x - mean) / std`. Broadcasting does the rest, for (C, H, W) and (B, C, H, W) alike,
       because shapes are lined up from the right.

    **What the checker tests.** Random inputs of shape (3, 5, 4) and (2, 3, 5, 4), with the ImageNet
    mean and std, compared value by value with the formula.

    **Common mistakes.**

    | you wrote | the checker prints |
    |---|---|
    | `mean` and `std` as tensors of shape (3,) | `raised RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 2`: the 3 lined up with the width 4 |
    | `x - mean` with the Python list itself | `raised TypeError: unsupported operand type(s) for -: 'Tensor' and 'list'` |
    | `.view(1, -1, 1, 1)` | `input (3, 5, 4): wrong result (shape (1, 3, 5, 4))`: fine for a batch, but it adds an axis to a single image |
    | `.view(-1, 1)` | `The size of tensor a (5) must match the size of tensor b (3) at non-singleton dimension 1`: (3, 1) lines up with (H, W), not with C |
    | `x - mean / std` | `wrong result (shape (3, 5, 4))`: the shape is right, but the division happened first; put brackets around `x - mean` |

    Had the image been exactly 3 pixels wide, the shape-(3,) mistake would raise no error and
    silently subtract the wrong numbers. The checker uses width 4 on purpose.
    """),
    api("torch.tensor", "Tensor.view", title="Functions you will need"),
    exercise(
        stub="""
        def normalize(x, mean, std):
            # x: (C, H, W) or (B, C, H, W) float tensor; mean, std: lists of C numbers
            # 1. YOUR CODE: mean as a tensor with torch.tensor, dtype=x.dtype -> shape (C,)
            # 2. YOUR CODE: give it the shape (C, 1, 1) with .view(-1, 1, 1)
            # 3. YOUR CODE: the same two steps for std
            # 4. YOUR CODE: return (x - mean) / std (broadcasting lines (C, 1, 1) up with the channel axis)
            raise NotImplementedError

        tests.check_normalize(normalize)
        """,
        solution="""
        def normalize(x, mean, std):
            # x: (C, H, W) or (B, C, H, W) float tensor; mean, std: lists of C numbers
            # 1. mean as a tensor with torch.tensor, dtype=x.dtype
            mean = torch.tensor(mean, dtype=x.dtype)    # (C,)
            # 2. give it the shape (C, 1, 1) with .view(-1, 1, 1)
            mean = mean.view(-1, 1, 1)                  # (C, 1, 1)
            # 3. the same two steps for std
            std = torch.tensor(std, dtype=x.dtype)      # (C,)
            std = std.view(-1, 1, 1)                    # (C, 1, 1)
            # 4. return (x - mean) / std (broadcasting lines (C, 1, 1) up with the channel axis)
            return (x - mean) / std

        tests.check_normalize(normalize)
        """),
    md(r"""
    The next cell uses both of your functions on the photo. What to look at: left, the photo after
    `to_chw`, turned back to HWC with `permute(1, 2, 0)` for Matplotlib; middle, the normalised values
    of the three channels, now on both sides of 0 instead of between 0 and 1; right, the result of
    `torchkit.denormalize` (which computes `x * std + mean` and turns the tensor back to HWC): the same
    photo again. Normalising changes the numbers, not the picture.
    """),
    api("torchkit.denormalize", "Tensor.round"),
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
    md(r"""
    **Did it work?** `[PASS]` means you can normalise any image or batch in one line. Every network
    input of the course from Day 1 on is normalised like this, by torchvision's
    `v2.Normalize(mean, std)` (the same formula) at the end of each data pipeline, and
    `torchkit.denormalize` undoes it whenever an image is shown. When the colours of a plotted image
    look wrong, or a pretrained network does much worse than expected, check the normalisation first.
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · The four datasets

    Time to look at the real data. Each dataset hands out items `(image, label)`, with the image
    already turned into a float (C, H, W) tensor by steps like your `to_chw` (FashionMNIST and Pet
    images are normalised too). A network trains on batches, and
    building them is the job of a **DataLoader**: given a dataset and a `batch_size`, it takes that
    many items (in a new random order on every pass when `shuffle=True`), stacks their images into one
    tensor (B, C, H, W) and their labels into one tensor. `next(iter(loader))` takes only the first
    batch, to look at it; a training loop runs `for x, y in loader:` over all of them.

    What a label looks like depends on the task, and the three tasks of the course are its three days:

    | task | the model answers | label | day |
    |---|---|---|---|
    | **classification** | which **class** is in the image? A class is one category out of a fixed list, for example "sneaker" | one whole number, the class index | 1 |
    | **detection** | where is each object, and what is it? | per object a **bounding box**, the smallest upright rectangle around it, given by four pixel coordinates x1, y1, x2, y2 (left, top, right, bottom), and a class | 2 |
    | **segmentation** | which class does each pixel belong to? | a **mask**: an image of class numbers, with the same height and width as the photo | 3 |

    | dataset | what | used for |
    |---|---|---|
    | FashionMNIST | 70 000 grayscale 28 x 28 images of clothes, 10 classes | Day 1: the first network you train on images |
    | Oxford-IIIT Pet | 7 349 photos of cats and dogs, 37 breeds, a box around each head, a pet / border / background mask | the thread through all three days |
    | Penn-Fudan | 170 street photos with one mask per pedestrian | Days 2 and 3: training a pretrained detector further, to find pedestrians and their outlines |
    | synthetic shapes | circles, squares, triangles drawn on the fly | Days 2 and 3: watching a detector or a segmentation network learn in under two minutes |

    The course helper `torchkit.show_batch` draws a batch with its class names, boxes or masks, and
    undoes the normalisation for display. You will use it all week.

    **FashionMNIST.** Small, clean, and fast enough to train a network on a CPU in a minute.
    """),
    api("torchkit.get_fashion_mnist", "torch.utils.data.DataLoader", "torchkit.show_batch"),
    code("""
    fm_train, fm_test, fm_classes = torchkit.get_fashion_mnist()
    x, y = next(iter(DataLoader(fm_train, batch_size=16, shuffle=True)))
    print("batch:", tuple(x.shape), x.dtype, "| labels:", y[:8].tolist(), "...")
    print("classes (label 0, 1, 2, ...):", fm_classes)
    torchkit.show_batch(x, y, fm_classes, n=16, size=1.6); plt.show()
    """),
    md(r"""
    Reading the printout: `(16, 1, 28, 28)` is 16 images, 1 channel (grayscale), 28 x 28 pixels. They
    are float32 because the dataset already scaled and normalised them (with FashionMNIST's own mean
    0.286 and std 0.353, one channel). Each label is a class index: label 3 means `fm_classes[3]`,
    "Dress".

    **Oxford-IIIT Pet** comes with three kinds of labels for the same photos, so we can classify
    (Day 1), detect (Day 2) and segment (Day 3) one dataset. `get_pet_datasets(size=160, target=...)`
    returns `(train, val, classes)` for the label type you ask for, with every photo resized to
    160 x 160 and normalised with the ImageNet numbers. `train` is for training; `val`, the
    **validation** set, is kept apart to measure how well a model does on photos it never trained on.
    """),
    api("torchkit.get_pet_datasets", "Tensor.unique"),
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
    print("detection target:", det_items[0][1], "| det_classes:", det_classes)
    print("segmentation target:", tuple(seg_items[0][1].shape), seg_items[0][1].dtype, "values", seg_items[0][1].unique().tolist(), "=", seg_classes)
    torchkit.show_batch([im for im, _ in det_items], [t for _, t in det_items], det_classes, cols=5, size=2.6,
                        title="Pet: head boxes (Day 2)"); plt.show()
    torchkit.show_batch([im for im, _ in seg_items], [m for _, m in seg_items], cols=5, size=2.6,
                        title="Pet: trimap (orange = pet, green = border, clear = background) (Day 3)"); plt.show()
    """),
    md(r"""
    Reading the printout:
    * The detection label is a dict. `boxes` has one row per object, `[x1, y1, x2, y2]` in pixels of
      the 160 x 160 image; `labels` gives each box's class, here 1 = `head`. Class 0, `background`,
      means "no object": torchvision's detectors reserve it.
    * The segmentation label is a mask of shape (160, 160), int64, with the values 0, 1, 2. It is a
      **trimap**, a mask with three classes: background, pet, and border, a thin strip along the
      pet's outline where the true edge is uncertain.

    **Penn-Fudan** has one mask *per person*. That is **instance segmentation**: it tells the
    individual objects apart (person 1, person 2), not only "person" versus "background". The images
    keep their original sizes, and `torch.stack` can only join tensors of identical shape, so a batch
    here is a Python list of images and a list of target dicts. That is what
    `torchkit.detection_collate` does.
    """),
    api("torchkit.get_penn_fudan", "torchkit.detection_collate", "torch.stack"),
    code("""
    pf = torchkit.get_penn_fudan()
    imgs, targets = torchkit.detection_collate([pf[i] for i in (0, 10, 20, 30)])
    print(len(pf), "images; first one", tuple(imgs[0].shape), "with", len(targets[0]["boxes"]), "people;",
          "target keys:", list(targets[0]))
    torchkit.show_batch(imgs, targets, ["bg", "person"], cols=4, size=3.2, title="Penn-Fudan: one mask and box per person"); plt.show()
    """),
    md(r"""
    Reading the printout: the first image is (3, 536, 559), 3 channels of 536 x 559 pixels, with 2
    people. Its target dict holds `boxes`, `labels` and `masks` (one 0 / 1 mask per person), plus three
    bookkeeping entries that the detection models of Day 2 expect.

    **Synthetic shapes** are generated in memory from a seed. Nothing to download, and every object is
    perfectly labelled, which makes them ideal for watching a detector or a segmentation network learn
    in under two minutes.
    """),
    api("torchkit.ShapesDataset"),
    code("""
    shapes_det = torchkit.ShapesDataset(n=8, size=128, seed=0, task="detection")
    shapes_seg = torchkit.ShapesDataset(n=8, size=128, seed=0, task="segmentation")
    imgs, targets = torchkit.detection_collate([shapes_det[i] for i in range(8)])
    torchkit.show_batch(imgs, targets, shapes_det.classes, cols=8, size=1.8, title="shapes: boxes"); plt.show()
    masks = [shapes_seg[i][1].numpy() for i in range(8)]
    viz.show_images(masks, cols=8, size=1.8, cmap="tab10", vmin=0, vmax=9,
                    suptitle="shapes: class masks (blue 0 = background, 1 circle, 2 square, 3 triangle)"); plt.show()
    """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · Autograd: gradients for free

    **Why.** Training (section 7) changes every parameter of a model a little, again and again, in the
    direction that makes the model's error smaller. To know that direction it needs the gradient
    (defined just below). A network has millions of parameters (Day 1's ResNet-18 network has 11.7
    million), so nobody works out
    these derivatives by hand. **Autograd** ("automatic gradients"), the part of PyTorch that computes
    them, does it for every model in this course.

    **Words.**
    * The **derivative** dy/dx says how fast y changes when x changes a little: if dy/dx = 7, moving x
      up by 0.001 moves y up by about 0.007.
    * The **gradient** of one number (in training: the error) with respect to many parameters is the
      collection of all those derivatives, one per parameter, in a tensor shaped like the parameters.
      It points in the direction that increases the number fastest, so a step against it decreases it.
    * `requires_grad=True` marks a tensor whose gradient you want. From then on, PyTorch records every
      operation done with it.
    * That record is the **computation graph**: which operation made which tensor from which inputs.
      Computing the result is the **forward pass**.
    * `y.backward()` runs the **backward pass**: it walks the graph from y back to the inputs, applies
      the chain rule (next subsection) at every operation, and stores dy/dx in `x.grad` for every
      marked tensor x.

    **Worked example.** $y = x^2 + 3x$. By hand: $dy/dx = 2x + 3$, so at $x = 2$: $y = 4 + 6 = 10$ and
    $dy/dx = 2 \cdot 2 + 3 = 7$. The cell computes the same with autograd, shows the operation that
    made y (its `grad_fn`, the last link of the graph), and checks what the 7 means by moving x by
    0.001.
    """),
    api("Tensor.requires_grad", "Tensor.backward", "Tensor.grad"),
    code("""
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2 + 3 * x                     # forward pass: autograd records "x squared", "3 times x", "add"
    y.backward()                           # backward pass: fills x.grad with dy/dx
    print("y =", y.item(), "| dy/dx =", x.grad.item(), "(expected 2*2 + 3 = 7)")
    print("y was made by", type(y.grad_fn).__name__, "from", [type(f).__name__ for f, _ in y.grad_fn.next_functions])
    y_moved = 2.001 ** 2 + 3 * 2.001       # plain Python numbers: x moved up by 0.001
    print(f"x moved by 0.001 -> y moved by {y_moved - 10:.6f}, about 7 x 0.001")
    """),
    md(r"""
    `AddBackward0` from `PowBackward0` and `MulBackward0` is the computation graph of $x^2 + 3x$: an
    addition of a power and a multiplication. The 0.007001 confirms the meaning of the derivative 7.

    ### The chain rule on a two-step graph

    A network is many simple steps in a row, and the backward pass uses the **chain rule**: the
    derivative through two steps is the product of the two steps' own derivatives. Take
    $u = 3x + 1$, then $y = u^2$, at $x = 2$:

    | step | forward value | its own derivative |
    |---|---|---|
    | $u = 3x + 1$ | $u = 7$ | $du/dx = 3$ |
    | $y = u^2$ | $y = 49$ | $dy/du = 2u = 14$ |
    | chain rule | | $dy/dx = dy/du \cdot du/dx = 14 \cdot 3 = 42$ |

    The figure draws this graph: values flow forward left to right, derivatives are multiplied right
    to left. That right-to-left sweep is backpropagation, and `backward()` does it for every parameter
    of a network.
    """),
    code("""
    x = torch.tensor(2.0, requires_grad=True)
    u = 3 * x + 1                          # step 1
    y = u ** 2                             # step 2
    y.backward()
    print("u =", u.item(), "| y =", y.item(), "| dy/dx =", x.grad.item(), "(by hand: 14 * 3 = 42)")

    fig, ax = plt.subplots(figsize=(9, 2.6))
    for px, text in [(0, "$x = 2$"), (4, "$u = 3x + 1 = 7$"), (8, "$y = u^2 = 49$")]:
        ax.text(px, 0, text, ha="center", va="center", fontsize=13, bbox=dict(boxstyle="round,pad=0.5", fc="#EEF3FA", ec="#4C78A8"))
    for a, b, label in [(0, 4, "forward: times 3, plus 1"), (4, 8, "forward: square")]:
        ax.annotate("", xy=(b - 1.5, 0.3), xytext=(a + 1.5, 0.3), arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.text((a + b) / 2, 0.55, label, ha="center", fontsize=10)
    for a, b, label in [(8, 4, "backward: times dy/du = 14"), (4, 0, "backward: times du/dx = 3")]:
        ax.annotate("", xy=(b + 1.5, -0.3), xytext=(a - 1.5, -0.3), arrowprops=dict(arrowstyle="->", lw=1.5, color="#E45756"))
        ax.text((a + b) / 2, -0.6, label, ha="center", fontsize=10, color="#E45756")
    ax.text(8, -1.0, "start: dy/dy = 1", ha="center", fontsize=10, color="#E45756")
    ax.text(0, -1.0, "dy/dx = 1 x 14 x 3 = 42", ha="center", fontsize=11, color="#E45756", fontweight="bold")
    ax.set_xlim(-2, 10); ax.set_ylim(-1.25, 0.8); ax.axis("off")
    ax.set_title("forward: values, left to right; backward: derivatives, right to left")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Gradients accumulate: what `zero_grad` is for

    `backward()` *adds* to `.grad` instead of replacing it. That is on purpose: it lets a gradient be
    summed over several pieces, for example a batch too big for memory processed in two halves. But a
    training step needs the gradient of the *current* batch only, so every step first clears the old
    one; in a training loop that is the line `optimizer.zero_grad()` (section 7 introduces the
    optimiser). Forget it and step 2 uses the sum of the gradients of
    steps 1 and 2, step 3 the sum of three: the steps grow, training goes wrong, and no error message
    tells you. The cell runs the same backward three times.
    """),
    code("""
    x = torch.tensor(2.0, requires_grad=True)
    for i in range(3):
        y = x ** 2 + 3 * x
        y.backward()
        print(f"after backward() number {i + 1}: x.grad = {x.grad.item()}   (7 each time, added up)")
    x.grad = None                          # clear it by hand: what optimizer.zero_grad() does for every parameter
    print("after x.grad = None:", x.grad)
    """),

    # ------------------------------------------------------------------ 7
    md(r"""
    ## 7 · Training: gradient descent and the training loop

    **Why.** All three days of the course train their models with the same five lines of code; only
    the model and the error measure change. This section builds them up from a model with one
    parameter.

    **Words.**
    * The **loss** is one number that says how wrong the model's predictions are on some examples.
      Training makes it small. For predicting numbers, the usual loss is the **mean squared error**
      (MSE): the average of (prediction - target)².
    * **Gradient descent**: compute the gradient of the loss with respect to every parameter, then move
      every parameter a small step against its gradient: $w \leftarrow w - \eta \cdot \partial L / \partial w$.
      Here $\partial L / \partial w$ is the derivative of the loss L with respect to the one parameter
      w (the curly ∂ says "the other parameters are held still"): w's entry in the gradient.
    * The **learning rate** $\eta$ (the Greek letter eta; `lr` in code) sets the size of that step.
    * One such update of the parameters is a **step**.

    ### One step of gradient descent, by hand

    The model is pred = w · x, with one parameter w that starts at 1. One training example: input
    x = 1 with target y = 3, so the best w is 3. Learning rate 0.1.

    | | by hand |
    |---|---|
    | forward | pred = w · x = 1 · 1 = 1 |
    | loss | (pred - y)² = (1 - 3)² = 4 |
    | gradient | dL/dw = 2 · (w · x - y) · x = 2 · (1 - 3) · 1 = -4 |
    | step | w ← w - lr · dL/dw = 1 - 0.1 · (-4) = 1.4 |

    The gradient is negative: increasing w decreases the loss, so the step moves w up, towards 3. The
    cell does the same with autograd. The update itself runs inside `torch.no_grad()`, so that
    autograd does not record it as part of the next forward pass.
    """),
    api("torch.no_grad"),
    code("""
    w = torch.tensor(1.0, requires_grad=True)      # the one parameter
    x, y, lr = 1.0, 3.0, 0.1                       # one example (input 1, target 3) and the learning rate

    pred = w * x                                   # forward
    loss = (pred - y) ** 2                         # loss
    loss.backward()                                # backward: fills w.grad
    print(f"pred {pred.item():.1f} | loss {loss.item():.1f} | w.grad {w.grad.item():.1f}")
    with torch.no_grad():
        w -= lr * w.grad                           # step: 1 - 0.1 * (-4)
    w.grad = None                                  # zero the gradient before the next step
    print(f"w after one step: {w.item():.2f}")
    """),
    md(r"""
    The same step, repeated. `descend(lr)` runs ten steps from w = 1 and returns w before the first and
    after every step. What to look at in the figure: the gray curve is the loss for every possible w,
    lowest at w = 3; the first red dot is the start (w = 1), and each further dot is w after one more
    step. With lr = 0.1 the steps shrink as w nears 3, because the
    slope (the gradient) gets flatter: each step closes 20 % of the remaining gap (2, 1.6, 1.28, ...).
    With lr = 0.9 every step jumps over the minimum and the path zigzags in.
    """),
    api("torch.linspace"),
    code("""
    def descend(lr, steps=10, w0=1.0, x=1.0, y=3.0):
        \"\"\"Gradient descent on loss = (w * x - y) ** 2; returns w before the first and after every step.\"\"\"
        w = torch.tensor(w0, requires_grad=True)
        ws = [w.item()]
        for _ in range(steps):
            loss = (w * x - y) ** 2
            loss.backward()
            with torch.no_grad():
                w -= lr * w.grad
            w.grad = None
            ws.append(w.item())
        return ws

    print("lr 0.1:", [round(v, 2) for v in descend(0.1)])
    wgrid = torch.linspace(0, 5, 101)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    for ax, lr_try in zip(axes, (0.1, 0.9)):
        path = torch.tensor(descend(lr_try))
        ax.plot(wgrid, (wgrid - 3) ** 2, color="gray", lw=1, label="loss = (w - 3)^2")
        ax.plot(path, (path - 3) ** 2, "o-", color="#E45756", ms=4, lw=1, label=f"10 steps, lr = {lr_try}")
        ax.set_xlabel("w"); ax.set_title(f"lr = {lr_try}"); ax.legend(); ax.grid(alpha=0.3)
    axes[0].set_ylabel("loss")
    plt.tight_layout(); plt.show()
    for lr_try in (0.5, 1.0, 1.1):
        print(f"lr {lr_try}:", [round(v, 2) for v in descend(lr_try)])
    """),
    md(r"""
    The last three lines show why the learning rate matters. With lr = 0.5 the first step lands
    exactly on 3 (a lucky value for this one loss). With lr = 1.0, w bounces between 1 and 5 forever.
    With lr = 1.1 every jump is bigger than the last and w runs away: the training **diverges**. Too
    small a learning rate is slow, too large a one diverges; choosing it is one of the first things
    D1-02 teaches.

    ### The training loop: five lines

    A real model has many parameters and learns from many examples, but each step is the one above.
    Every model in this course, from a single linear layer to Mask R-CNN on Day 3, is trained with
    these five operations, in this order:

    1. **forward**: `pred = model(x)`, the model's predictions for a batch x
    2. **loss**: `loss = loss_fn(pred, y)`, one number saying how far they are from the targets y
    3. **zero the old gradients**: `optimizer.zero_grad()`
    4. **backward**: `loss.backward()` fills `p.grad` for every parameter p
    5. **step**: `optimizer.step()` updates every parameter, for SGD $w \leftarrow w - \eta\, \partial L / \partial w$

    Two objects do the bookkeeping:
    * the **loss function** `nn.MSELoss()` computes the mean squared error of a whole batch;
    * the **optimiser** holds the list of parameters and the learning rate; its `step()` does the
      update you just did by hand, for every parameter at once, and its `zero_grad()` clears all their
      `.grad`. `torch.optim.SGD` is plain gradient descent. SGD stands for *stochastic* gradient
      descent: stochastic (random) because each step uses a random batch instead of all the data.

    The model this time is `nn.Linear(1, 1)`, a **linear layer** with one input and one output:
    pred = w · x + b, two parameters (a weight w and a bias b). We fit it to 256 points scattered
    around the line y = 3x - 1, with random **noise** added (normal random numbers with std 0.4), as
    real measurements have. A `TensorDataset` pairs each x with its y, and the DataLoader serves them in
    batches of 32, so one **epoch**, one pass over all the data, is 256 / 32 = 8 steps.
    """),
    api("torch.Generator", "torch.rand", "torch.randn", "torch.utils.data.TensorDataset"),
    code("""
    g = torch.Generator().manual_seed(0)
    X = torch.rand(256, 1, generator=g) * 4 - 2                     # 256 points in [-2, 2]
    Y = 3 * X - 1 + 0.4 * torch.randn(256, 1, generator=g)          # y = 3x - 1 + noise
    loader = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True)
    print("X", tuple(X.shape), "| Y", tuple(Y.shape), "| batches per epoch:", len(loader))
    print("mean squared noise:", round(((Y - (3 * X - 1)) ** 2).mean().item(), 3),
          "(even the true line has this loss: no line can do much better)")
    plt.figure(figsize=(4.5, 3.2)); plt.scatter(X, Y, s=8, alpha=0.6); plt.title("data: y = 3x - 1 + noise"); plt.show()
    """),
    md(r"""
    ### Exercise 3 · one training step

    **Goal.** Write the five operations as a function that performs one training step on one batch,
    for any model, loss function and optimiser.

    **Input.** `model` (for example `nn.Linear(1, 1)`); a batch of inputs `x`, shape (B, 1) here; the
    matching targets `y`, shape (B, 1); `loss_fn` (for example `nn.MSELoss()`); `optimizer` (for
    example `torch.optim.SGD(model.parameters(), lr=0.1)`).

    **Output.** The loss of this batch, computed *before* the update, as a Python float (`loss.item()`).
    As a side effect the model's parameters have moved by one step.

    **Example** (the hand example, now with a bias). A model with w = 1 and b = 0, one example
    x = `[[1.]]`, y = `[[3.]]`, MSE and SGD with lr 0.1: `train_step` returns 4.0, and afterwards
    w = 1.4 and b = 0.4 (the bias gradient is 2 · (1 - 3) = -4 too).

    **Plan.** The five lines of the list above, in that order, then `return loss.item()`. Use the
    function's arguments (`model`, `loss_fn`, `optimizer`), not variables from other cells.

    **What the checker tests.** It makes its own model (`nn.Linear(3, 1)`), 8 random examples and an
    SGD optimiser, calls your function twice, and compares the two returned losses and the final
    weights with two clean SGD steps it computes itself. Two calls, so that a missing `zero_grad` shows.

    **Common mistakes.**

    | you wrote | the checker prints |
    |---|---|
    | no `optimizer.zero_grad()` | `weights after two calls differ from two clean SGD steps (...)`: the second step used the sum of both gradients |
    | `zero_grad()` between `backward()` and `step()` | `returned losses [1.1529 1.1529], expected [1.1529 0.9386]` and `weights ... differ`: the gradient was wiped before it was used, so nothing moved |
    | no `optimizer.step()`, or `step()` before `backward()` | the same two lines: the parameters never changed |
    | `loss.backward` without `()` | the same two lines: the method was named, not called |
    | no `return`, or `return` before `step()` | `return the loss as a Python float: loss.item()`, or the same two lines as above |
    """),
    api("Optimizer.zero_grad", "Tensor.backward", "Optimizer.step", "Tensor.item", title="Functions you will need"),
    exercise(
        stub="""
        def train_step(model, x, y, loss_fn, optimizer):
            # 1. YOUR CODE: forward: pred, the model's predictions for the batch x
            # 2. YOUR CODE: loss: loss_fn of pred and the targets y
            # 3. YOUR CODE: zero the old gradients with the optimizer
            # 4. YOUR CODE: backward: fill .grad of every parameter
            # 5. YOUR CODE: step: let the optimizer update the parameters
            # 6. YOUR CODE: return the loss as a Python float
            raise NotImplementedError

        tests.check_d0_train_step(train_step)
        """,
        solution="""
        def train_step(model, x, y, loss_fn, optimizer):
            # 1. forward: pred, the model's predictions for the batch x
            pred = model(x)
            # 2. loss: loss_fn of pred and the targets y
            loss = loss_fn(pred, y)
            # 3. zero the old gradients with the optimizer
            optimizer.zero_grad()
            # 4. backward: fill .grad of every parameter
            loss.backward()
            # 5. step: let the optimizer update the parameters
            optimizer.step()
            # 6. return the loss as a Python float
            return loss.item()

        tests.check_d0_train_step(train_step)
        """),
    md(r"""
    **Did it work?** `[PASS]` means your function does exactly what a library SGD step does. You will
    write it again in D1-02 (with one more line, `model.train()`), and it is the inside of
    `torchkit.fit`, the loop that trains the models of the course from Day 1 on.

    ### Exercise 4 · the loop

    **Goal.** Train `nn.Linear(1, 1)` on the noisy points for 30 epochs with your `train_step`, and
    record the loss of every epoch.

    **Given** in the cell: `model = nn.Linear(1, 1)` with random starting parameters, an empty list
    `losses`, and `loader` from above (8 batches of 32 points per epoch).

    **You write**, in about eight lines:
    1. `loss_fn`, the mean squared error: `nn.MSELoss()`.
    2. `optimizer`: `torch.optim.SGD` over `model.parameters()`, with `lr=0.1`.
    3. A loop over 30 epochs. Inside it, a loop over the batches, `for xb, yb in loader:`, that calls
       `train_step(model, xb, yb, loss_fn, optimizer)` and keeps the loss it returns.
    4. After each epoch, append the mean of that epoch's 8 batch losses to `losses`, so that `losses`
       ends with 30 numbers.

    **Expected.** The checker prints the learned parameters, `learned w = 3.003 (true 3.0),  b = -0.975
    (true -1.0)`. The epoch loss falls from 4.30 in epoch 1 to 0.19, the noise level printed above.

    **What the checker tests.** Only the trained `model`: w within 0.15 of 3 and b within 0.15 of -1.
    The plot after it uses `losses`.

    **Common mistakes.**

    | you wrote | what you see |
    |---|---|
    | left `raise NotImplementedError` in the cell | `NotImplementedError` before anything trains: delete that line |
    | only `for xb, yb in loader:`, no loop over epochs | `learned w = 2.734 (true 3.0),  b = -0.803 (true -1.0)` and `[FAIL]`: one epoch is not enough |
    | `torch.optim.SGD(model.parameters, ...)` | `TypeError: 'method' object is not iterable`: `parameters` needs its brackets |
    | an optimiser over another model, for example a new `nn.Linear(1, 1)` | `learned w = -0.007 (true 3.0),  b = 0.536 (true -1.0)`: the untouched starting values, nothing trained `model` |
    | appended every batch loss | `[PASS]`, but `losses` has 240 numbers and the plot's x axis counts batches, not epochs |
    | Exercise 3 has not passed yet | errors or `[FAIL]` here too: fix `train_step` first |
    """),
    api("nn.Linear", "nn.MSELoss", "torch.optim.SGD", "Module.parameters", title="Functions you will need"),
    exercise(
        stub="""
        torch.manual_seed(0)
        model = nn.Linear(1, 1)      # pred = w * x + b, random starting w and b
        losses = []                  # one number per epoch: the mean loss of that epoch's batches
        # 1. YOUR CODE: loss_fn, the mean squared error
        # 2. YOUR CODE: optimizer, SGD over model.parameters() with lr=0.1
        # 3. YOUR CODE: a loop over 30 epochs; inside it:
        #      an empty list epoch_losses,
        #      a loop `for xb, yb in loader:` that calls train_step and appends the loss it returns,
        #      and after the batch loop, append the mean of epoch_losses to losses
        raise NotImplementedError

        tests.check_linear_fit(model)
        """,
        solution="""
        torch.manual_seed(0)
        model = nn.Linear(1, 1)      # pred = w * x + b, random starting w and b
        losses = []                  # one number per epoch: the mean loss of that epoch's batches
        # 1. loss_fn, the mean squared error
        loss_fn = nn.MSELoss()
        # 2. optimizer, SGD over model.parameters() with lr=0.1
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        # 3. a loop over 30 epochs
        for epoch in range(30):
            epoch_losses = []                        # the 8 batch losses of this epoch
            for xb, yb in loader:
                batch_loss = train_step(model, xb, yb, loss_fn, optimizer)
                epoch_losses.append(batch_loss)
            # the mean of this epoch's batch losses
            losses.append(sum(epoch_losses) / len(epoch_losses))

        tests.check_linear_fit(model)
        """),
    md(r"""
    What to look at: left, the loss of every epoch on a logarithmic scale: a big drop in the first
    epoch, then flat at about 0.19, a little above the dashed line. The dashed 0.16 = 0.4² is the noise
    level over infinitely many points; these 256 points happen to be a bit noisier (0.19, printed
    above), and no straight line can remove noise. Right, the learned line (red) on top of the true
    one (dashed). Before Exercise 4 runs, the left panel is empty and the red line is the untrained
    model's.
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
    **Did it work?** `[PASS]` means you have written a complete training loop: data in batches, epochs,
    and the five-line step. From Day 1 on, the loop lives in `torchkit.fit(model, train_loader,
    val_loader, loss_fn, optimizer, epochs, ...)`: the same five lines plus a pass over the validation
    data after every epoch, an optional schedule that changes the learning rate during training, a
    time budget, and saving checkpoints. It returns a
    `history` dict you can plot. Here it is on the same problem; `val_loss` is the loss on the
    validation loader (here simply the same data again) and `val_metric` is `nan` because "accuracy"
    only exists for classifiers.
    """),
    api("torchkit.fit"),
    code("""
    torch.manual_seed(0)
    model2 = nn.Linear(1, 1)
    history = torchkit.fit(model2, loader, loader, nn.MSELoss(), torch.optim.SGD(model2.parameters(), lr=0.1),
                           epochs=3, log_every=0)
    print("history keys:", list(history))
    """),

    # ------------------------------------------------------------------ 8
    md(r"""
    ## 8 · How every notebook of the course works

    Each mechanism is taught in the same four beats. You have met beats 3 and 4 in this notebook;
    knowing the rhythm lets you spend your attention on the content:

    | beat | what you see | what you do |
    |---|---|---|
    | **1 · Figure** | a generated picture of the mechanism, with the one formula that matters | read the "what to look at" sentence, find it in the picture |
    | **2 · Lab** | an interactive panel under a `labs.show(...)` cell | answer the **Predict first** question in your head, *then* drag / slide to check, then do the "Try this" list |
    | **3 · Exercise** | a cell with `# YOUR CODE HERE` and `raise NotImplementedError` | write the 3-15 lines that *are* the concept |
    | **4 · Checker** | `tests.check_...` prints `[PASS]`, or `[FAIL]` with a hint | fix until it passes, then move on |

    Work **cell by cell, top to bottom**. Cells after an unfinished exercise either stop with an
    error (as in this notebook) or, in the later notebooks, use the course's reference solution
    instead and say so with a line `note: using the reference implementation ...`. If you are stuck
    for more than five minutes, open the same notebook in `notebooks/solutions/`, read that one cell,
    close it, and type the answer yourself.

    The labs are small web pages that ship with the course: no installation, no internet. Run the
    next cell to check that they work on your machine. You should see a photo, an edge map, and a
    3x3 grid of numbers you can edit (a "kernel": Day 1 explains it). Move the pointer over the photo.
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
    md(r"""
    The fold-out boxes **Functions used here** come from a function reference that you can also open
    from any cell. Functions met in an earlier notebook come back one line each, under *Functions from
    earlier notebooks*:
    """),
    code("""
    from cvcourse import apiref
    apiref.show("permute")        # one entry, opened
    apiref.search("memory")       # every entry whose description mentions a word
    apiref.index()                # all entries, grouped by library
    """),

    # ------------------------------------------------------------------ 9
    md(r"""
    ## 9 · Summary

    * Check your setup once: PyTorch and torchvision imported, data in `data/`, a device picked. A CPU
      is fine for this course.
    * Images: (H, W, C) uint8 in files -> (C, H, W) float32 in [0, 1] (`permute`, `.float() / 255`;
      never `reshape`) -> normalised per channel ((x - mean) / std, broadcasting with shape (C, 1, 1))
      -> batched (B, C, H, W) (`unsqueeze(0)` for one image, a DataLoader for many).
    * Labels: a class index (classification), boxes x1, y1, x2, y2 (detection), a mask (segmentation).
    * Autograd: `requires_grad` -> compute (forward) -> `.backward()` -> `.grad`. Gradients accumulate
      until you clear them.
    * Training step: forward, loss, `zero_grad`, `backward`, `step`. An epoch is one pass over the data.
      Everything else is bookkeeping.

    ### Self-check
    1. A batch has shape `(16, 3, 224, 224)`. Which axis is the channel axis, and what does `x.mean(dim=(0, 2, 3))` compute?
    2. `img.reshape(3, H, W)` on an (H, W, 3) image raises no error. Why is the result still wrong?
    3. Why does `torch.zeros(2, 3, 4, 5) + torch.zeros(3)` fail while `+ torch.zeros(3, 1, 1)` works?
    4. A pretrained network gets inputs in [0, 1] but was trained on ImageNet-normalised inputs. What happens?
    5. What goes wrong if you forget `optimizer.zero_grad()`?
    6. One SGD step with lr 0.1 on loss = (w - 3)², starting from w = 1: what is w afterwards?
    7. Why can a 480x640 photo and a 375x500 photo not be stacked into one tensor, and what do detection data loaders do instead?

    ### Go deeper
    * PyTorch tutorials: "Tensors", "A gentle introduction to torch.autograd", "Optimizing model parameters".
    * NumPy documentation, "Broadcasting" (the same rules apply to PyTorch).
    """),
    md(r"""
    ## Words from this notebook

    **Tensors and images**

    | word | meaning |
    |---|---|
    | tensor | PyTorch's array: a grid of numbers with any number of axes, which can live on a GPU and record how it was computed |
    | axis, dimension | one direction of a tensor; a colour image has 3 (channels, height, width) |
    | shape | the size along each axis, for example (3, 600, 512) |
    | dtype | the type of number in every element: `float32`, `uint8`, `int64`, `bool` |
    | uint8 | whole numbers 0 to 255 in one byte: how image files store pixel values; arithmetic wraps past 255 |
    | float32 | decimal numbers in 4 bytes: what networks compute with |
    | pixel | one point of the image grid; one number (grayscale) or three (red, green, blue) |
    | channel | one colour of an image (all its red values, say), a grayscale picture of its own |
    | HWC, CHW | axis orders (height, width, channels) for files and Matplotlib, (channels, height, width) for PyTorch |
    | batch, (B, C, H, W), NCHW | several images stacked along a new first axis; B (or N) is the number of images |
    | view | a tensor that reads another tensor's memory in a new order or shape; no copy, shared numbers |
    | copy | a tensor with memory of its own |
    | contiguous | memory order equals reading order; usually false after `permute` |
    | broadcasting | combining tensors of different shapes: line up from the right, stretch axes of size 1 |
    | mean | the average of some numbers |
    | standard deviation (std) | the spread: square root of the average squared distance from the mean |
    | normalisation | (x - mean) / std per channel, so that inputs are centred on 0 with spread about 1 |
    | ImageNet mean and std | the per-channel numbers (0.485, 0.456, 0.406) and (0.229, 0.224, 0.225) used to normalise for pretrained networks |
    | pretrained | already trained by someone else, usually on ImageNet, and reused |

    **Data**

    | word | meaning |
    |---|---|
    | dataset | a collection of examples, each an input and its label |
    | label | the right answer for one example |
    | class | one category out of a fixed list, stored as a whole number (its index) |
    | classification | predicting the class of a whole image (Day 1) |
    | bounding box | the smallest upright rectangle around an object: x1, y1, x2, y2 in pixels |
    | detection | finding every object with its box and class (Day 2) |
    | mask | an image of class numbers, one per pixel, the same size as the photo |
    | segmentation | predicting the class of every pixel (Day 3) |
    | trimap | the Pet mask: 0 background, 1 pet, 2 border |
    | instance segmentation | one mask per object, so that two people are two separate masks |
    | DataLoader | serves a dataset in batches, shuffled if asked |
    | validation set | examples kept apart from training, to measure how well the model does on new data |
    | noise | random error in the data that no model can predict |

    **Training**

    | word | meaning |
    |---|---|
    | model | a neural network: a function with adjustable numbers inside |
    | parameter, weight | one of those adjustable numbers; a linear layer has a weight w and a bias b |
    | linear layer | `nn.Linear`: pred = w · x + b |
    | training | adjusting the parameters so that the outputs match the labels |
    | loss | one number that says how wrong the predictions are; MSE is the average squared error |
    | derivative | how fast one number changes when another changes a little |
    | gradient | all derivatives of the loss, one per parameter |
    | autograd | PyTorch's automatic computation of gradients |
    | requires_grad | marks a tensor whose gradient autograd should compute |
    | computation graph | autograd's record of which operation made which tensor |
    | forward pass | computing the outputs (and the loss) from the inputs |
    | backward pass, backpropagation | `loss.backward()`: computing every gradient, multiplying derivatives right to left through the graph |
    | chain rule | the derivative through two steps is the product of the two steps' derivatives |
    | zero_grad | clears old gradients, which otherwise add up |
    | gradient descent | moving every parameter a small step against its gradient |
    | learning rate (lr) | the size of that step; too large and training diverges |
    | diverge | the parameters run away instead of settling, because every step overshoots more |
    | optimiser | the object that does the step for all parameters (`torch.optim.SGD`) |
    | SGD | stochastic gradient descent: gradient descent on random batches |
    | step | one update of the parameters, from one batch |
    | epoch | one pass over the whole training data |
    | checkpoint | a file with a model's trained parameters |

    **Computer**

    | word | meaning |
    |---|---|
    | PyTorch, torchvision | the deep-learning library of the course, and its companion for images |
    | device | the processor that computes: `cpu`, `cuda` or `mps` |
    | CPU | the computer's main processor |
    | GPU | a graphics card: thousands of small cores, much faster for networks |
    | CUDA | NVIDIA's toolkit for computing on its GPUs; PyTorch's name for such a GPU |
    | random seed | the starting point of the random number generator; same seed, same results |
    """),
]
