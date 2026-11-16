from nb import md, code, exercise, header, api

NAME = "D1-03-architectures"

CELLS = [
    header(
        "D1-03 · Architecture zoo",
        steps=[
            "Timeline: one idea per architecture, 1998-2022; accuracy vs. size",
            "VGG: 3x3 convolutions everywhere",
            "GoogLeNet / Inception: parallel branches and 1x1 bottlenecks (exercises 1 and 2)",
            "ResNet: residual and bottleneck blocks; why very deep nets train; residual vs. dense (exercises 3 and 4)",
            "Efficient blocks: depthwise-separable convolutions, squeeze-and-excitation, EfficientNet, ConvNeXt (exercise 5)",
            "Reading a model: summaries, parameters and multiply-adds of torchvision models",
            "Assemble a MiniResNet and train it on the Pet subset (exercise 6)",
        ],
        topics=[
            "The handful of ideas that survived from 2012 to today and the problem each one solved",
            "Write a network block as an `nn.Module` class: layers in `__init__`, data flow in `forward`",
            "Implement Inception, residual (basic + bottleneck) and depthwise-separable blocks in PyTorch",
            "Why residual connections fix the optimisation of deep networks (gradient flow at initialisation)",
            "Residual addition vs. dense concatenation",
            "Parameter / compute trade-offs and how to read a model summary",
        ],
        minutes=120,
        prereq="D1-01 (convolution, padding and stride, the output-size formula, 1x1 convolution, pooling, "
               "global average pooling, receptive field, `nn.Module`) and D1-02 (training recipe, BatchNorm).",
    ),
    md(r"""
    ## How to read this notebook

    The same rhythm as D0-00, whose first section explains it in full: run the cells top to bottom,
    read the "What to look at" sentence under each figure, and answer a lab's **Predict first**
    question before you touch the lab. The six exercises are **code templates**. Above each one: the
    goal, what goes in and comes out (with an example), a plan in numbered steps, a drawing of the
    data flow with the tensor shapes, what the checker tests, and a table of common mistakes with what
    each one prints. In the cell, numbered `# step` comments follow that plan, the bookkeeping lines
    are already written, and every line you write goes under a `# YOUR CODE:` comment that says what
    it must create or compute. When a method's blanks are filled, delete its
    `raise NotImplementedError` line(s) and run the cell. An unfinished exercise does not stop the
    notebook: the cells after it print `note: using the reference implementation ...` and use the
    course's own version. Stuck for five minutes? Read that cell in
    `notebooks/solutions/D1-03-architectures-solution.ipynb`, close it, and type it yourself. New
    words are **bold** where they are explained and are collected in "Words from this notebook" at the
    end. Boxes titled **More:** are optional. Short of time in class: leave Exercise 4 (`Bottleneck`)
    and the **More:** boxes for afterwards.
    """),
    md(r"""
    ## 0 · Setup

    Run this cell first. Besides the usual imports (D0-00 explains them) it loads `day1`, the
    course's Day 1 helper module: the published numbers behind the first figures, small helper
    functions, and the course's own version of every exercise (the **reference implementation**, used
    when yours has not passed yet). On a CPU, `day1.pick_cpu_threads()` times one and four threads and
    keeps the faster. The two small functions at the end, `box` and `arrow`, draw the block diagrams
    of this notebook.
    """),
    api("torch.manual_seed", "torchkit.get_device", brief=True),
    code(r"""
    import os, sys, pathlib
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")   # before importing torch: idle CPU threads sleep, not spin
    for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:   # find the cvcourse package
        if (p / "cvcourse").is_dir():
            sys.path.insert(0, str(p)); break

    import math, time
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle
    import torch, torch.nn as nn, torch.nn.functional as F
    import torchvision
    from cvcourse import viz, tests, torchkit, day1, labs

    %matplotlib inline
    plt.rcParams["figure.dpi"] = 90
    torch.manual_seed(0)
    device = torchkit.get_device()
    if device.type == "cpu":
        day1.pick_cpu_threads()     # on a busy CPU, 1 thread can be faster than 8

    def box(ax, x, y, w, h, text, fc="#cfe0f5", fs=8, ec="k"):
        # a rounded box of width w and height h centred on (x, y), with text inside
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(ax, x0, y0, x1, y1, **kw):
        # an arrow from (x0, y0) to (x1, y1)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.2, **kw))
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · Twenty-five years in one picture

    **Why this notebook.** Day 2's detectors and Day 3's segmentation networks are not built from
    scratch: they take one of the networks of this notebook (most often a ResNet) and put a new head on
    it. D1-04 fine-tunes one for the 37 Pet breeds. This notebook shows what is inside those networks,
    and why each piece is there.

    ### Words to describe a network

    * An **architecture** is the design of a network: which layers, in which order, with how many
      channels, and how they are connected. Two networks with the same architecture differ only in the
      values of their parameters.
    * A **layer** is one operation, usually with parameters of its own: one convolution, one
      BatchNorm (D1-02), one linear layer.
    * A **block** is a small group of layers that is repeated many times. Architectures are designed
      block by block: an Inception block, a residual block, ...
    * A **stage** is a run of blocks that work at the same height and width. From one stage to the
      next, height and width are usually halved and the number of channels doubled.
    * The **stem** is the first few layers, which shrink the image quickly. The **head** is the last
      layers, which turn the features into one score per class (in a ResNet: global average pooling,
      the mean of each channel over all positions (D1-01), then one linear layer).
    * **Depth** is the number of layers with weights; **width** is the number of channels.

    ResNet-18, the network section 6 takes apart, as an example (shapes for one 224 x 224 image; its
    blocks are the residual blocks of section 4):

    ```
    image                                                   (3, 224, 224)
    stem      7x7 conv with stride 2, then max-pool         (64, 56, 56)
    stage 1   2 residual blocks                             (64, 56, 56)
    stage 2   2 residual blocks, the first with stride 2    (128, 28, 28)
    stage 3   2 residual blocks, the first with stride 2    (256, 14, 14)
    stage 4   2 residual blocks, the first with stride 2    (512, 7, 7)
    head      global average pooling, linear layer          (1000,)  one score per ImageNet class
    ```

    ### How networks are compared

    * **ImageNet-1k** is the standard benchmark: about 1.3 million training photos in 1 000 classes
      (the collection D0-00 introduced; "1k" for its 1 000 classes).
    * **Top-1 accuracy**: the percentage of validation photos for which the class with the highest
      score is the right one. (Top-5: the right class is among the five highest scores.)
    * **Parameter count**: how many numbers the network learns. It decides the memory and the file size.
    * **Multiply-adds** (**MACs**, from multiply-accumulate): how many multiplications, each followed by
      an addition, one image costs. It measures the compute. A GMAC is a billion of them. Papers often
      quote **FLOPs** (floating-point operations) instead, about 2 x MACs. Section 6 shows how they are
      counted.

    The two figures use published numbers, hard-coded in `day1.ARCHITECTURES`: one row per network,
    `(name, year, top-1 %, parameters in millions, GMACs, idea)`. Each network is remembered for
    **one idea** that the later ones kept.
    """),
    api("np.linspace", brief=True),
    code(r"""
    # Draws the timeline: one dot per network, with its year, its name and the idea it contributed
    A = day1.ARCHITECTURES          # rows of (name, year, top-1 %, parameters in M, GMACs, idea)
    timeline = []                   # the rows to draw: skip the two bigger siblings and the 2021-recipe row
    for row in A:
        name = row[0]
        if "recipe" in name or name in ("ResNet-152", "EfficientNet-B4"):
            continue
        timeline.append(row)

    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.axhline(0, color="k", lw=1)
    xs = np.linspace(0, 1, len(timeline))          # evenly spaced positions along the line
    for i, (name, year, top1, params, macs, idea) in enumerate(timeline):
        up = 1 if i % 2 == 0 else -1               # labels alternate above (+1) and below (-1) the line
        va = "bottom" if up > 0 else "top"
        ax.plot([xs[i], xs[i]], [0, 0.55 * up], color="gray", lw=0.8)
        ax.plot(xs[i], 0, "o", color=viz.color(i), ms=9)
        ax.text(xs[i], 0.62 * up, f"{year}  {name}", ha="center", va=va, fontsize=10, weight="bold")
        if len(idea) >= 34:                        # break a long idea into two lines at a space
            cut = idea.rfind(" ", 0, 34)
            idea = idea[:cut] + "\n" + idea[cut + 1:]
        ax.text(xs[i], 0.75 * up, idea, ha="center", va=va, fontsize=8, color="#333")
    ax.set_xlim(-0.09, 1.09); ax.set_ylim(-1.45, 1.45); ax.axis("off")
    ax.set_title("ConvNet architectures: the idea each one contributed", fontsize=12)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    The scatter below puts every network with a published ImageNet number at its parameter count
    (horizontal, log scale: equal steps are equal factors) and its top-1 accuracy (vertical). The area
    of a marker grows with the compute per image. The arrow joins two points for the *same* ResNet-50:
    trained in 2015, and trained again in 2021 with a modern recipe.
    """),
    code(r"""
    # Draws top-1 accuracy against parameter count; marker area grows with the GMACs per image
    fig, ax = plt.subplots(figsize=(10, 5.2))
    offsets = {"AlexNet": (8, -12), "VGG-16": (-10, 8), "GoogLeNet": (8, -10), "ResNet-50": (-14, -16),     # label positions,
               "ResNet-152": (8, -2), "DenseNet-121": (-62, 8), "MobileNetV2": (8, -10), "EfficientNet-B0": (-30, 10),
               "EfficientNet-B4": (8, 4), "ViT-B/16": (8, -4), "ConvNeXt-T": (8, 6), "ResNet-50 (2021 recipe)": (8, -12)}
    for name, year, top1, params, macs, idea in A:
        if top1 is None:                           # LeNet-5 was never run on ImageNet
            continue
        ax.scatter(params, top1, s=30 + 25 * macs, color=viz.color(year - 2012), alpha=0.75, edgecolor="k", zorder=3)
        label = name if "recipe" in name else f"{name} ({year})"
        ax.annotate(label, (params, top1), xytext=offsets.get(name, (6, 4)), textcoords="offset points", fontsize=8)

    for row in A:                                  # find the two ResNet-50 rows for the arrow
        if row[0] == "ResNet-50":
            r50_2015 = row
        if row[0] == "ResNet-50 (2021 recipe)":
            r50_2021 = row
    ax.annotate("", xy=(r50_2021[3], r50_2021[2]), xytext=(r50_2015[3], r50_2015[2]),
                arrowprops=dict(arrowstyle="->", color=viz.PALETTE[3], lw=1.5, connectionstyle="arc3,rad=-0.3"))
    ax.text(14.5, 78.2, "same network,\nbetter training: +4.8", color=viz.PALETTE[3], fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("parameters (millions, log scale)"); ax.set_ylabel("ImageNet top-1 (%)")
    ax.set_title("Accuracy vs. size (marker area ~ compute per image)"); ax.grid(alpha=.3, which="both")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Three things to read off the scatter:

    * **Size is not accuracy.** VGG-16 has 26x the parameters of EfficientNet-B0 and is 6 points
      worse. Most of VGG's parameters sit in its head of fully-connected layers (section 2). Later
      networks replaced that head by global average pooling followed by one small linear layer.
    * **Architecture is not everything.** The same ResNet-50 gains 4.8 points from a 2021 training
      recipe: longer training, the AdamW optimiser, strong augmentation (RandAugment, Mixup / CutMix)
      and label smoothing, the D1-02 toolbox. Compare architectures only when they were trained with
      the same recipe.
    * **The last big jumps came from design and recipe together** (EfficientNet, ConvNeXt). ViT-B/16
      is a **vision transformer**: it cuts the image into patches and replaces convolutions by
      attention, a layer that lets every patch look at every other patch (Day 3). ViTs only win with a
      lot of data or pretraining.
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · VGG (2014): 3x3 everywhere

    **Why.** VGG's plan, stacks of 3x3 convolutions with the channels doubling each time the image is
    halved, is the plan of almost every convnet since, including the ResNets of Days 2 and 3.

    AlexNet (2012), the first deep network to win ImageNet, used 11x11 and 5x5 kernels. VGG replaced
    them with stacks of 3x3 convolutions. Recall from D1-01 that the **receptive field** of an output
    pixel is the patch of the input it depends on: two stacked 3x3 convs see 5x5, three see 7x7. A
    stack of small kernels beats one big kernel in two ways:

    * **Fewer parameters.** A conv layer with $C_{\text{in}}$ input channels, $C_{\text{out}}$ output
      channels and a k x k kernel has $C_{\text{in}} \cdot C_{\text{out}} \cdot k \cdot k$ weights, plus
      $C_{\text{out}}$ biases. With C channels in and out, one 7x7 layer has $49\,C^2$ weights and three
      3x3 layers $3 \cdot 9\,C^2 = 27\,C^2$.
    * **More non-linearities.** A ReLU follows each of the three layers instead of one, so the stack
      can compute more complicated functions.

    **Worked example.** A 3x3 conv from 2 to 4 channels: 2 · 4 · 3 · 3 = 72 weights, plus 4 biases =
    76 parameters. At C = 256: one 7x7 layer has 49 · 256² = 3,211,264 weights, three 3x3 layers
    27 · 256² = 1,769,472.

    VGG-16 has 16 layers with weights: 13 conv layers in five groups, each group followed by a 2x2
    max-pool that halves height and width, then three **fully-connected** (fc) layers. A
    fully-connected layer is a linear layer (`nn.Linear`): every output is connected to every input.
    The figure draws VGG-16 to scale: box height ~ spatial size, box width grows with the channels.
    """),
    api("torchvision.models.vgg16", "Tensor.numel"),
    api("Module.parameters", "torchkit.count_parameters", "nn.Conv2d", brief=True),
    code(r"""
    # Draws VGG-16 to scale: one box per conv layer (height = image size, width grows with the channels)
    vgg_cfg = [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]       # (channels, number of 3x3 convs) per group
    fig, ax = plt.subplots(figsize=(13, 3.6)); x = 0; hw = 224
    for b, (c, n) in enumerate(vgg_cfg):
        for k in range(n):
            w = 0.18 + 0.5 * c / 512
            ax.add_patch(Rectangle((x, -hw / 2), w, hw, fc=viz.color(b), ec="k", lw=0.6, alpha=0.85)); x += w + 0.08
        ax.text(x - (n * 0.6) / 2, hw / 2 + 8, f"{n} x conv3x3\n{c} ch @ {hw}x{hw}", ha="center", fontsize=8)
        hw //= 2
        ax.add_patch(Rectangle((x, -hw / 2), 0.12, hw, fc="lightgray", ec="k", lw=0.6)); x += 0.5
    for i, n in enumerate([4096, 4096, 1000]):
        ax.add_patch(Rectangle((x, -40), 0.35, 80, fc="#f8d0b8", ec="k", lw=0.6)); ax.text(x + 0.17, -52, f"fc\n{n}", ha="center", va="top", fontsize=8); x += 0.6
    ax.text(x - 1.0, 55, "124 M of the 138 M\nparameters are here", ha="center", fontsize=8, color=viz.PALETTE[3])
    ax.set_xlim(-0.3, x + 0.3); ax.set_ylim(-140, 150); ax.axis("off"); ax.set_title("VGG-16 to scale (gray = 2x2 max-pool)")
    plt.tight_layout(); plt.show()

    # Count the parameters: all of them, and those of the head (vgg.classifier = the three fc layers)
    vgg = torchvision.models.vgg16(weights=None)          # the architecture with random weights: no download
    fc_params = 0
    for p in vgg.classifier.parameters():                  # every weight and bias tensor of the head
        fc_params += p.numel()                             # numel() = number of numbers in the tensor
    print(f"VGG-16: {torchkit.count_parameters(vgg) / 1e6:.1f} M params, of which the fc head: {fc_params / 1e6:.1f} M")
    first_fc = vgg.classifier[0]
    print(f"the first fc layer alone: {first_fc.in_features:,} inputs x {first_fc.out_features:,} outputs "
          f"= {first_fc.weight.numel():,} weights")

    # The worked examples above
    print("3x3 conv 2 -> 4 channels:", torchkit.count_parameters(nn.Conv2d(2, 4, kernel_size=3)), "parameters (72 weights + 4 biases)")
    C = 256
    print(f"one 7x7 conv {C}->{C}: {49 * C * C:,} weights | three 3x3 convs: {27 * C * C:,} weights (same receptive field)")
    """),
    md(r"""
    What to look at: the conv boxes get shorter (the image shrinks) and wider (more channels) from left
    to right, and the fc layers hold 124 of the 138 million parameters. The first fc layer alone has
    102.8 million: the last max-pool leaves 512 channels of 7 x 7 pixels, and all 512 · 7 · 7 = 25,088
    of those numbers are connected to each of its 4,096 outputs.
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · GoogLeNet / Inception (2014): go wide, cheaply

    **Why.** Two ideas from this block come back all week: a 1x1 convolution that shrinks the channels
    before an expensive layer (the ResNet bottleneck of section 4, the detector heads of Day 2), and
    concatenation along channels to combine feature maps (U-Net, the segmentation network of Day 3,
    combines its early and late feature maps this way).

    **The idea.** Which kernel size is right at a given layer: 1x1, 3x3 or 5x5? The Inception block
    does not choose. It runs four **branches** in parallel, each a short chain of layers, all fed the
    same input, and **concatenates** their outputs along the channel axis: the output holds the
    channels of branch 1, then those of branch 2, and so on. The channel counts add up; height and
    width must be the same in every branch.

    **1x1 convolutions as channel mixers.** Recall from D1-01: a 1x1 conv looks at one pixel at a
    time; at each pixel, each output channel is a weighted sum of the input channels. It changes the
    number of channels without touching height or width, and costs only $C_{\text{in}} \cdot
    C_{\text{out}}$ weights. Placed before an expensive 3x3 or 5x5 conv to cut the channels that conv
    has to read, it is called a **bottleneck** (or reduction), after the narrow neck of a bottle.

    GoogLeNet (2014) stacked nine of these blocks. It also replaced VGG's huge fc head by global
    average pooling (6.6 M parameters in total instead of 138 M), and it attached two **auxiliary
    classifiers**: small extra heads halfway up the network, whose loss (weighted 0.3) pushed gradient
    into the lower layers. They were a fix for the problem that residual connections later solved much
    better (section 4).

    The figure shows block "3a" of GoogLeNet with its channel counts. Follow the numbers: the input has
    192 channels of 28 x 28 pixels; every branch keeps 28 x 28; the four branches deliver 64, 128, 32
    and 32 channels, so the output has 64 + 128 + 32 + 32 = 256.
    """),
    code(r"""
    # Draws Inception block 3a: four branches from the same input, concatenated along the channels
    fig, ax = plt.subplots(figsize=(12, 5)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6.2)
    box(ax, 6, 5.7, 3.0, 0.5, "input  192 x 28 x 28", fc="#e5e5e5", fs=9)
    cols = [(1.5, ["1x1 conv, 64"], "64"), (4.5, ["1x1 conv, 96", "3x3 conv, 128"], "128"),       # (x, layers, channels out)
            (7.5, ["1x1 conv, 16", "5x5 conv, 32"], "32"), (10.5, ["3x3 max-pool", "1x1 conv, 32"], "32")]
    for x, layers, cout in cols:
        arrow(ax, 6, 5.42, x, 4.75)
        for k, name in enumerate(layers):
            fc = "#cde8cf" if name.startswith("1x1") else ("#e5e5e5" if "pool" in name else "#f8d0b8")   # green = 1x1
            box(ax, x, 4.45 - 1.1 * k, 2.3, 0.55, name, fc=fc, fs=9)
            if k: arrow(ax, x, 4.45 - 1.1 * (k - 1) - 0.29, x, 4.45 - 1.1 * k + 0.29)
        ytop = 4.45 - 1.1 * (len(layers) - 1) - 0.29
        arrow(ax, x, ytop, 6, 1.55)
        mx, my = 0.5 * (x + 6), 0.5 * (ytop + 1.55)
        ax.text(mx, my, f"{cout} ch", fontsize=9, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=1))
    box(ax, 6, 1.25, 4.6, 0.55, "concatenate along channels: 64 + 128 + 32 + 32 = 256", fc="#f6e7a6", fs=9)
    arrow(ax, 6, 0.97, 6, 0.45); ax.text(6, 0.25, "output  256 x 28 x 28", ha="center", fontsize=9)
    ax.text(0.1, 0.4, "green = 1x1 bottleneck /\nprojection (channel mixer)", fontsize=8, color="#2e7d32")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: the green boxes are 1x1 convs. Branch 1 is a single 1x1 conv; in branches 2 and 3
    a 1x1 conv shrinks 192 channels to 96 and to 16 before the 3x3 and the 5x5; in branch 4 a 1x1 conv
    after the max-pool shrinks the 192 channels the pool passes on to 32.

    **Worked example: what the 1x1 reductions save.** The weights of the 3x3 and 5x5 branches with and
    without their 1x1 reduction (biases left out), with the conv formula of section 2:

    | branch | without the 1x1 | with the 1x1 |
    |---|---|---|
    | 3x3 | 192 · 128 · 9 = 221,184 | 192 · 96 + 96 · 128 · 9 = 18,432 + 110,592 = 129,024 |
    | 5x5 | 192 · 32 · 25 = 153,600 | 192 · 16 + 16 · 32 · 25 = 3,072 + 12,800 = 15,872 |

    The 5x5 branch shrinks almost 10x, because its expensive conv now reads 16 channels instead of 192.
    The cell checks the sums.
    """),
    code(r"""
    # Weights of the 3x3 and 5x5 branches of block 3a, without and with the 1x1 reduction (no biases)
    naive_3x3 = 192 * 128 * 9                      # 3x3 conv straight from the 192 input channels
    reduced_3x3 = 192 * 96 + 96 * 128 * 9          # 1x1 conv 192 -> 96, then 3x3 conv 96 -> 128
    naive_5x5 = 192 * 32 * 25
    reduced_5x5 = 192 * 16 + 16 * 32 * 25          # 1x1 conv 192 -> 16, then 5x5 conv 16 -> 32
    print(f"3x3 branch: {naive_3x3:,} -> {reduced_3x3:,} weights ({naive_3x3 / reduced_3x3:.1f}x fewer)")
    print(f"5x5 branch: {naive_5x5:,} -> {reduced_5x5:,} weights ({naive_5x5 / reduced_5x5:.1f}x fewer)")
    naive = naive_3x3 + naive_5x5
    reduced = reduced_3x3 + reduced_5x5
    print(f"both: {naive:,} -> {reduced:,} weights ({naive / reduced:.1f}x fewer)")
    """),
    md(r"""
    ### Exercise 1 · Inception output channels (warm-up)

    **Goal.** A function that says how many channels leave an Inception block, given its seven numbers.

    **Why by hand.** Before you build a block you must know the shape it produces: the next layer's
    input channels must equal it. Counting channels is the most common bookkeeping in architectures.

    **Input.** Seven whole numbers, in this order:
    * `c_in`: channels coming into the block;
    * `f1`: filters (= output channels) of branch 1's 1x1 conv;
    * `f3r`, `f3`: filters of branch 2's 1x1 reduction, and of its 3x3 conv;
    * `f5r`, `f5`: the same for branch 3 with its 5x5 conv;
    * `fpool`: filters of the 1x1 conv after branch 4's max-pool.

    **Output.** One whole number: the channels of the block's output.

    **Example.** Block 3a above: `inception_out_channels(192, 64, 96, 128, 16, 32, 32)` returns 256.

    **Plan.**
    1. For each branch, the channels that leave it are the filters of its **last** conv.
    2. Concatenation puts the four branch outputs side by side along the channel axis, so the block's
       output channels are their sum.
    3. `c_in`, `f3r` and `f5r` stay inside the block: they are not part of the answer.

    **What the checker tests.** Block 3a (expects 256) and block 3b,
    `(256, 128, 128, 192, 32, 96, 64)` (expects 480).

    **Common mistakes.** For every wrong number the checker prints the same two lines,
    `[FAIL]  inception output channels` and its hint. Find your mistake by what your function returns
    for block 3a (call `inception_out_channels(192, 64, 96, 128, 16, 32, 32)` in a new cell):

    | you wrote | returns for block 3a |
    |---|---|
    | added `c_in` as well | 448 |
    | added the reductions `f3r` and `f5r` as well | 368 |
    | used `c_in` for branch 4 (the pool keeps 192 channels, but the 1x1 conv after it makes 32) | 416 |
    | deleted the `raise` line but left the `total = ...` line out | the checker prints `your code raised NameError: name 'total' is not defined` |
    """),
    exercise(
        stub=r'''
        def inception_out_channels(c_in, f1, f3r, f3, f5r, f5, fpool):
            """Number of channels that leave an Inception block (see the figure above)."""
            # step 1: the channels leaving each branch = the filters of the branch's LAST conv
            # YOUR CODE: out_b1 = the channels leaving branch 1 (one 1x1 conv)
            # YOUR CODE: out_b2 = the channels leaving branch 2 (1x1 reduction, then 3x3 conv)
            # YOUR CODE: out_b3 = the channels leaving branch 3 (1x1 reduction, then 5x5 conv)
            # YOUR CODE: out_b4 = the channels leaving branch 4 (max-pool, then 1x1 conv)
            # step 2: concatenation puts the four outputs side by side along the channel axis
            # YOUR CODE: total = the channels of the concatenated output
            raise NotImplementedError    # delete this line when steps 1 and 2 are written
            return total

        ok_shapes = tests.check_inception_shapes(inception_out_channels)
        ''',
        solution=r'''
        def inception_out_channels(c_in, f1, f3r, f3, f5r, f5, fpool):
            """Number of channels that leave an Inception block (see the figure above)."""
            # step 1: the channels leaving each branch = the filters of the branch's LAST conv
            out_b1 = f1                  # the 1x1 conv
            out_b2 = f3                  # the 3x3 conv decides; f3r stays inside the branch
            out_b3 = f5                  # the 5x5 conv decides; f5r stays inside the branch
            out_b4 = fpool               # the 1x1 conv after the pool decides, not c_in
            # step 2: concatenation puts the four outputs side by side along the channel axis
            total = out_b1 + out_b2 + out_b3 + out_b4
            return total

        ok_shapes = tests.check_inception_shapes(inception_out_channels)
        '''),
    md(r"""
    ### How a block is written in PyTorch: `nn.Module`

    From now on every block is a **class**. PyTorch's base class for anything that has layers is
    `nn.Module` (D1-01's `TinyConvNet` was one). You write two methods:

    * `__init__(self, ...)` runs once, when you create the block. It creates the layers and stores each
      one as an attribute: `self.conv = nn.Conv2d(...)`. Its first line must be `super().__init__()`,
      which sets up the bookkeeping that lets PyTorch find the layers' parameters.
    * `forward(self, x)` runs every time data goes through the block. It applies the layers to the
      input `x` in the right order and returns the result. You do not call it by name: `block(x)` calls
      `forward(x)` for you.

    **Why a class and not a function?** The layers have parameters that must exist before the first
    input arrives and must survive from one call to the next, so that the optimiser can train them,
    `model.parameters()` can list them and a checkpoint can save them. A class keeps them together as
    attributes. PyTorch finds a parameter only if its layer is stored as an attribute that is itself a
    module (`nn.Conv2d`, `nn.Sequential`, ...): a layer kept in a plain Python list is invisible to it.

    **`nn.Sequential(layer1, layer2, ...)`** is a ready-made module whose `forward` applies its layers
    one after the other. It is handy for a branch: one attribute holds a whole chain of layers. Its
    layers are named by their position: `0`, `1`, `2`, ...

    The cell writes a small block (a 3x3 conv followed by a ReLU) both ways and lists the parameters
    of each: `named_parameters()` gives the name and the tensor of every parameter.
    """),
    api("Module.named_parameters"),
    api("nn.Module", "nn.Conv2d", "nn.ReLU", "nn.Sequential", "torch.randn", brief=True),
    code(r'''
    class ConvReLU(nn.Module):
        """A 3x3 conv (keeping height and width) followed by a ReLU."""

        def __init__(self, c_in, c_out):
            super().__init__()                                              # always first
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)    # a layer with parameters
            self.relu = nn.ReLU()                                           # a layer without parameters

        def forward(self, x):
            x = self.conv(x)             # (B, c_out, H, W)
            x = self.relu(x)             # same shape, negative values set to 0
            return x

    block = ConvReLU(2, 4)
    x = torch.randn(1, 2, 5, 5)          # a batch of 1 image with 2 channels of 5 x 5 pixels
    print("as a class:     block(x) has shape", tuple(block(x).shape))
    for name, p in block.named_parameters():
        print("   ", name, tuple(p.shape))

    same = nn.Sequential(nn.Conv2d(2, 4, kernel_size=3, padding=1), nn.ReLU())
    print("as nn.Sequential: same(x) has shape", tuple(same(x).shape))
    for name, p in same.named_parameters():
        print("   ", name, tuple(p.shape))
    '''),
    md(r"""
    Reading the printout: both versions have a weight of shape (4, 2, 3, 3), that is (out channels, in
    channels, kernel height, kernel width), and 4 biases: 72 + 4 = 76 parameters, the worked example of
    section 2. The ReLU has none. In the class the names come from the attributes (`conv.weight`); in
    the `nn.Sequential` from the positions (`0.weight`).

    ### Two more pieces: keeping the size, and concatenating

    All four branches must deliver the same height and width, or they cannot be concatenated. The
    output-size formula of D1-01, $\lfloor (n + 2p - k)/s \rfloor + 1$, gives the padding p that
    keeps n = 28 with stride s = 1: p = 1 for a 3x3 kernel (28 + 2 - 3 + 1 = 28), p = 2 for a 5x5
    kernel (28 + 4 - 5 + 1 = 28), p = 0 for 1x1. A max-pool has a stride as well, and its **default
    stride is its kernel size**: `nn.MaxPool2d(3)` moves 3 pixels at a time and shrinks 28 to 9.
    Branch 4 needs `nn.MaxPool2d(3, stride=1, padding=1)`, which keeps 28 x 28 and the 192 channels.

    `torch.cat([a, b, ...], dim=1)` joins tensors along axis 1, the channel axis of (B, C, H, W). All
    other axes must match. Its first argument is a **list** of tensors.
    """),
    api("nn.MaxPool2d", "torch.cat", "torch.zeros", "torch.ones", "Tensor.item", brief=True),
    code(r"""
    x = torch.randn(1, 192, 28, 28)                     # the input of block 3a
    r = torch.randn(1, 16, 28, 28)                      # what branch 3's 1x1 reduction produces
    print("1x1 conv 192 -> 16:             ", tuple(nn.Conv2d(192, 16, kernel_size=1)(x).shape))
    print("5x5 conv 16 -> 32, padding=2:   ", tuple(nn.Conv2d(16, 32, kernel_size=5, padding=2)(r).shape))
    print("5x5 conv 16 -> 32, no padding:  ", tuple(nn.Conv2d(16, 32, kernel_size=5)(r).shape), "<- 24 x 24: cannot be concatenated")
    print("max-pool 3, stride=1, padding=1:", tuple(nn.MaxPool2d(3, stride=1, padding=1)(x).shape))
    print("max-pool 3 (default stride 3):  ", tuple(nn.MaxPool2d(3)(x).shape))

    a = torch.zeros(1, 2, 4, 4)                         # 2 channels, all zeros
    b = torch.ones(1, 3, 4, 4)                          # 3 channels, all ones
    both = torch.cat([a, b], dim=1)                     # 2 + 3 = 5 channels
    print("torch.cat([a, b], dim=1):       ", tuple(both.shape),
          "| channel 0 holds", both[0, 0, 0, 0].item(), "(from a), channel 4 holds", both[0, 4, 0, 0].item(), "(from b)")
    """),
    md(r"""
    ### Exercise 2 · `InceptionBlock`

    **Goal.** Write GoogLeNet's Inception block as an `nn.Module`.

    **Why by hand.** It is the smallest network piece with more than one path. Writing it shows that a
    `forward` can send the same input through several chains of layers and combine their results,
    which every architecture from here on does.

    **Input and output.** `InceptionBlock(c_in, f1, f3r, f3, f5r, f5, fpool)` builds the block (the
    seven numbers of exercise 1). Called on `x` of shape (B, c_in, H, W), it returns
    (B, f1 + f3 + f5 + fpool, H, W). Example: `InceptionBlock(192, 64, 96, 128, 16, 32, 32)` turns a
    (2, 192, 28, 28) batch into (2, 256, 28, 28).

    **Plan.** `__init__` creates four attributes, each an `nn.Sequential`. Every conv keeps its bias
    (the default) and is followed by `nn.ReLU()`.

    | attribute | layers, in order | channels out |
    |---|---|---|
    | `b1` (given) | 1x1 conv `c_in -> f1`, ReLU | f1 |
    | `b2` | 1x1 conv `c_in -> f3r`, ReLU, 3x3 conv `f3r -> f3` with `padding=1`, ReLU | f3 |
    | `b3` | 1x1 conv `c_in -> f5r`, ReLU, 5x5 conv `f5r -> f5` with `padding=2`, ReLU | f5 |
    | `b4` | 3x3 max-pool with `stride=1, padding=1`, 1x1 conv `c_in -> fpool`, ReLU | fpool |

    Watch the input channels of each conv: the first layer of every branch reads the block's input
    (`c_in`); the second conv of branches 2 and 3 reads what the 1x1 before it made (`f3r`, `f5r`); the
    max-pool keeps all `c_in` channels, so the conv after it reads `c_in`.

    `forward` sends the same `x` into all four branches, then concatenates:

    ```
                             x  (B, c_in, H, W)
              +--------------+--------------+--------------+
              b1             b2             b3             b4
        (B, f1, H, W)  (B, f3, H, W)  (B, f5, H, W)  (B, fpool, H, W)
              +--------------+------+-------+--------------+
                       torch.cat([...], dim=1)
                    (B, f1 + f3 + f5 + fpool, H, W)
    ```

    **What the checker tests.** Blocks 3a (on a 2 x 192 x 28 x 28 batch, expects 256 channels) and 3b
    (2 x 256 x 14 x 14, expects 480): the output shape, the number of parameters (compared with the
    course's reference block), and that no output is negative (every branch ends with a ReLU).

    **Common mistakes.** A `your code raised ...` message ends with `(line N in ...)`: the line where
    the error happened, in your cell or inside PyTorch.

    | you wrote | the checker prints |
    |---|---|
    | no `padding=1` on the 3x3 conv | `your code raised RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 28 but got size 26 for tensor number 1 in the list.` Counting from 0, tensor number 1 is `out2` |
    | `padding=1` on the 5x5 conv | the same, `... got size 26 for tensor number 2 in the list.` |
    | `nn.MaxPool2d(3, padding=1)`, without `stride=1` | `... Expected size 28 but got size 10 for tensor number 3 in the list.`: the default stride is 3 |
    | `torch.cat(..., dim=0)` | `Sizes of tensors must match except in dimension 0. Expected size 64 but got size 128 ...`: axis 0 is the batch |
    | `torch.cat(out1, out2, out3, out4, dim=1)`, no list | `your code raised TypeError: cat() takes from 1 to 2 positional arguments but 4 were given` |
    | `out1 + out2 + out3 + out4` | `The size of tensor a (64) must match the size of tensor b (128) at non-singleton dimension 1`: adding needs equal channels, concatenating does not |
    | the 3x3 conv reads `c_in` instead of `f3r` | `Given groups=1, weight of size [128, 192, 3, 3], expected input[2, 96, 28, 28] to have 192 channels, but got 96 channels instead` |
    | branch 4 without its 1x1 conv | `output (2, 416, 28, 28), expected (2, 256, 28, 28)`: the max-pool passes on all 192 input channels |
    | no ReLU at the end of a branch | `outputs should be >= 0: every branch ends with a ReLU` |
    | `bias=False` in the convs | `163,328 parameters, expected 163,696 ...`: the 368 biases, one per output channel of each conv |
    | the branches kept in a Python list, `self.branches = [...]` | `0 parameters, expected 163,696 ...`: PyTorch does not look inside a plain list |
    | forgot `super().__init__()` | `your code raised AttributeError: cannot assign module before Module.__init__() call` |
    """),
    api("nn.Sequential", "nn.Conv2d", "nn.ReLU", "nn.MaxPool2d", "torch.cat", title="Functions you will need", brief=True),
    exercise(
        stub=r'''
        class InceptionBlock(nn.Module):
            """GoogLeNet's Inception block: four branches see the same input; their outputs are concatenated."""

            def __init__(self, c_in, f1, f3r, f3, f5r, f5, fpool):
                super().__init__()           # always first: sets up the bookkeeping of nn.Module
                # step 1 (given): branch 1, a 1x1 conv c_in -> f1, then ReLU
                self.b1 = nn.Sequential(
                    nn.Conv2d(c_in, f1, kernel_size=1),
                    nn.ReLU(),
                )
                # step 2: branch 2, four layers: 1x1 conv c_in -> f3r, ReLU, 3x3 conv f3r -> f3 with padding=1, ReLU
                # YOUR CODE: self.b2 = an nn.Sequential of these four layers, in this order
                # step 3: branch 3, the same pattern with a 5x5 conv: 1x1 conv c_in -> f5r, ReLU,
                #         5x5 conv f5r -> f5 with padding=2, ReLU
                # YOUR CODE: self.b3 = an nn.Sequential of these four layers
                # step 4: branch 4: 3x3 max-pool with stride=1 and padding=1, 1x1 conv c_in -> fpool, ReLU
                # YOUR CODE: self.b4 = an nn.Sequential of these three layers
                raise NotImplementedError    # delete this line when b2, b3 and b4 are written

            def forward(self, x):
                # x: (B, c_in, H, W)
                # step 5: send the same x through every branch
                out1 = self.b1(x)            # (B, f1, H, W)
                # YOUR CODE: out2 = branch 2 applied to x                              -> (B, f3, H, W)
                # YOUR CODE: out3 = branch 3 applied to x                              -> (B, f5, H, W)
                # YOUR CODE: out4 = branch 4 applied to x                              -> (B, fpool, H, W)
                # step 6: concatenate the four outputs along the channel axis (dim=1)
                # YOUR CODE: out = torch.cat of a list of the four outputs, with dim=1 -> (B, f1 + f3 + f5 + fpool, H, W)
                raise NotImplementedError    # delete this line when steps 5 and 6 are written
                return out

        ok_inception = tests.check_inception_block(InceptionBlock)
        ''',
        solution=r'''
        class InceptionBlock(nn.Module):
            """GoogLeNet's Inception block: four branches see the same input; their outputs are concatenated."""

            def __init__(self, c_in, f1, f3r, f3, f5r, f5, fpool):
                super().__init__()           # always first: sets up the bookkeeping of nn.Module
                # step 1 (given): branch 1, a 1x1 conv c_in -> f1, then ReLU
                self.b1 = nn.Sequential(
                    nn.Conv2d(c_in, f1, kernel_size=1),
                    nn.ReLU(),
                )
                # step 2: branch 2, four layers: 1x1 conv c_in -> f3r, ReLU, 3x3 conv f3r -> f3 with padding=1, ReLU
                self.b2 = nn.Sequential(
                    nn.Conv2d(c_in, f3r, kernel_size=1),               # reduce: c_in -> f3r channels
                    nn.ReLU(),
                    nn.Conv2d(f3r, f3, kernel_size=3, padding=1),      # padding 1 keeps H x W
                    nn.ReLU(),
                )
                # step 3: branch 3, the same pattern with a 5x5 conv: 1x1 conv c_in -> f5r, ReLU,
                #         5x5 conv f5r -> f5 with padding=2, ReLU
                self.b3 = nn.Sequential(
                    nn.Conv2d(c_in, f5r, kernel_size=1),               # reduce: c_in -> f5r channels
                    nn.ReLU(),
                    nn.Conv2d(f5r, f5, kernel_size=5, padding=2),      # padding 2 keeps H x W
                    nn.ReLU(),
                )
                # step 4: branch 4: 3x3 max-pool with stride=1 and padding=1, 1x1 conv c_in -> fpool, ReLU
                self.b4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # keeps H x W and all c_in channels
                    nn.Conv2d(c_in, fpool, kernel_size=1),
                    nn.ReLU(),
                )

            def forward(self, x):
                # x: (B, c_in, H, W)
                # step 5: send the same x through every branch
                out1 = self.b1(x)            # (B, f1, H, W)
                out2 = self.b2(x)            # (B, f3, H, W)
                out3 = self.b3(x)            # (B, f5, H, W)
                out4 = self.b4(x)            # (B, fpool, H, W)
                # step 6: concatenate the four outputs along the channel axis (dim=1)
                out = torch.cat([out1, out2, out3, out4], dim=1)       # (B, f1 + f3 + f5 + fpool, H, W)
                return out

        ok_inception = tests.check_inception_block(InceptionBlock)
        '''),
    md(r"""
    The next cell builds block 3a from your class (or from the reference, if yours has not passed:
    `tests.use_or_reference` decides) and prints its summary: one row per attribute created in
    `__init__`, with its output shape, parameters and multiply-adds (section 6 explains the columns).
    What to look at: the four output shapes add up to 256 channels, and branch 2 holds 129,248 of the
    163,696 parameters.
    """),
    code(r"""
    Inc = tests.use_or_reference(ok_inception, globals().get("InceptionBlock"), day1.InceptionBlock, "InceptionBlock")
    blk = Inc(192, 64, 96, 128, 16, 32, 32)
    print(day1.summarize(blk, (192, 28, 28), depth=0))
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · ResNet (2015): learn the residual

    **Why.** Residual connections are the reason networks can have 50, 100 or 1 000 layers. You will
    meet them everywhere: the ResNet-50 inside Faster R-CNN and Mask R-CNN (Days 2 and 3), D1-04's
    fine-tuned network, and every transformer of Day 3.

    **The problem.** In 2015, stacking more layers made networks *worse*, and worse on the **training**
    set itself: a 56-layer plain network had a higher training error than a 20-layer one. That is not
    **overfitting** (D1-02: good on the training images, bad on new ones): the deeper network fails to
    fit even its training data. The training itself fails, because the gradient that reaches the first
    layers is far too small or far too large for them to learn (we measure it below). He et al. called
    it the **degradation problem**.

    **The fix.** Let each block learn a *correction* to its input instead of a whole new output:

    $$y = \operatorname{ReLU}\big(F(x) + x\big)$$

    * $F(x)$, the **residual**, is what the block's layers compute (the **main path**);
    * the $+\,x$ arrives through a **shortcut**, also called a **skip connection**: a path that skips
      the layers and carries the input unchanged to the addition.

    Two consequences:

    * "Do nothing" is easy: with $F = 0$ the block passes its input on (apart from the final ReLU), so
      extra blocks cannot make the network much worse at the start of training.
    * The gradient has a direct way back. The derivative of y with respect to x is
      $I + \partial F / \partial x$, where $I$ (the identity) stands for "the gradient arrives
      unchanged through `+ x`". Even when $\partial F / \partial x$ is tiny, the gradient still reaches
      the earlier layers.

    **When the shapes differ.** The addition needs F(x) and x to have the same shape. When the block
    halves height and width (stride 2) or changes the number of channels, the shortcut becomes a
    **projection**: a 1x1 conv with the block's stride (it changes the channels and the size),
    followed by BatchNorm. Otherwise it is the **identity** shortcut, `nn.Identity()`: a module whose
    forward returns its input unchanged, with no parameters.

    **BatchNorm** (BN, D1-02) normalises every channel over the batch, then multiplies it by a learned
    **scale** (the layer's `weight`, often written γ) and adds a learned **shift** (its `bias`, β), one
    pair per channel. ResNets put one after every conv, and the conv before it has no bias
    (`bias=False`): BN's shift does that job. With scale and shift both 0, BN outputs zeros; the
    checkers below use that to switch F off.

    | block | layers in F | used in |
    |---|---|---|
    | **basic** | 3x3 conv (stride) - BN - ReLU - 3x3 conv - BN | ResNet-18 / 34 |
    | **bottleneck** | 1x1 conv (reduce) - BN - ReLU - 3x3 conv (stride) - BN - ReLU - 1x1 conv (expand x4) - BN | ResNet-50 / 101 / 152 |
    """),
    code(r"""
    # Draws a basic block and a bottleneck block, each with its identity shortcut
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, title, layers in [
            (axes[0], "basic block (64 ch)", ["3x3 conv, 64", "BN, ReLU", "3x3 conv, 64", "BN"]),
            (axes[1], "bottleneck block (256 ch in / out)", ["1x1 conv, 64", "BN, ReLU", "3x3 conv, 64", "BN, ReLU", "1x1 conv, 256", "BN"])]:
        ax.axis("off"); ax.set_xlim(0, 6); ax.set_ylim(-0.4, 8.2); ax.set_title(title)
        n = len(layers); ys = np.linspace(6.8, 1.9, n)
        box(ax, 2.5, 7.6, 2.0, 0.5, "x", fc="#e5e5e5", fs=10)
        prev = 7.35
        for y, name in zip(ys, layers):
            fc = "#cde8cf" if "1x1" in name else ("#f8d0b8" if "3x3" in name else "#ffffff")   # green 1x1, orange 3x3
            arrow(ax, 2.5, prev, 2.5, y + 0.25); box(ax, 2.5, y, 2.2, 0.48, name, fc=fc, fs=9); prev = y - 0.25
        box(ax, 2.5, 1.0, 0.6, 0.5, "+", fc="#f6e7a6", fs=13); arrow(ax, 2.5, prev, 2.5, 1.25)
        ax.plot([3.5, 4.3, 4.3], [7.6, 7.6, 1.0], color=viz.PALETTE[3], lw=2)
        ax.annotate("", xy=(2.82, 1.0), xytext=(4.3, 1.0), arrowprops=dict(arrowstyle="->", lw=2, color=viz.PALETTE[3]))
        ax.text(4.45, 4.3, "identity\nshortcut", color=viz.PALETTE[3], fontsize=10)
        arrow(ax, 2.5, 0.75, 2.5, 0.2); ax.text(2.5, -0.15, "ReLU -> y", ha="center", fontsize=10)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: in both blocks the red identity shortcut skips the whole main path and is added at
    `+`, *before* the last ReLU. In the bottleneck the expensive 3x3 conv works on only 64 channels,
    between a cheap 1x1 conv that reduces 256 -> 64 and another that expands back to 256.

    ### Worked example: a residual block with stride 2

    `BasicBlock(64, 128, stride=2)` on a batch x of shape (B, 64, 16, 16). The output size of a conv
    is $\lfloor (n + 2p - k)/s \rfloor + 1$ (D1-01).

    | step | layer | shape after it | by hand |
    |---|---|---|---|
    | | input x | (B, 64, 16, 16) | |
    | 1 | conv1: 3x3, 64 -> 128, stride 2, padding 1 | (B, 128, 8, 8) | ⌊(16 + 2 - 3)/2⌋ + 1 = 8 |
    | 2 | bn1, then ReLU | (B, 128, 8, 8) | BN and ReLU never change the shape |
    | 3 | conv2: 3x3, 128 -> 128, stride 1, padding 1 | (B, 128, 8, 8) | (8 + 2 - 3) + 1 = 8 |
    | 4 | bn2: this is F(x) | (B, 128, 8, 8) | |
    | | F(x) + x? | error | 128 vs 64 channels, 8 vs 16 pixels |
    | 5 | shortcut: 1x1 conv, 64 -> 128, stride 2, then BN | (B, 128, 8, 8) | ⌊(16 - 1)/2⌋ + 1 = 8 |
    | 6 | F(x) + shortcut(x), then ReLU | (B, 128, 8, 8) | |

    Parameters by hand: conv1 64 · 128 · 9 = 73,728; conv2 128 · 128 · 9 = 147,456; projection conv
    64 · 128 · 1 = 8,192; three BatchNorms with a scale and a shift per channel, 3 · 2 · 128 = 768.
    Total 230,144. The cell builds these layers one by one, prints the shape after each step, the error
    of the plain `+ x`, and the parameter count.
    """),
    api("nn.BatchNorm2d"),
    api("F.relu", brief=True),
    code(r"""
    x = torch.randn(2, 64, 16, 16)                                         # B = 2
    conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
    bn1 = nn.BatchNorm2d(128)
    conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
    bn2 = nn.BatchNorm2d(128)
    proj_conv = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)   # the projection shortcut ...
    proj_bn = nn.BatchNorm2d(128)                                          # ... and its BatchNorm

    out = conv1(x)
    print("1  conv1:                 ", tuple(out.shape))
    out = bn1(out)
    out = F.relu(out)
    print("2  bn1, ReLU:             ", tuple(out.shape))
    out = conv2(out)
    print("3  conv2:                 ", tuple(out.shape))
    out = bn2(out)
    print("4  bn2 = F(x):            ", tuple(out.shape))
    try:
        out + x
    except RuntimeError as e:
        print("   F(x) + x:              RuntimeError:", e)
    sc = proj_conv(x)
    sc = proj_bn(sc)
    print("5  shortcut(x):           ", tuple(sc.shape))
    y = F.relu(out + sc)
    print("6  ReLU(F(x) + shortcut): ", tuple(y.shape))

    n_params = 0
    for layer in [conv1, bn1, conv2, bn2, proj_conv, proj_bn]:
        n_params += torchkit.count_parameters(layer)
    print(f"parameters: {n_params:,}")
    """),
    md(r"""
    The error names dimension 3 first (the width, 8 vs 16) because PyTorch compares shapes from the
    right, as in D0-00's broadcasting rules. In the exercise the same message means: the shortcut did
    not change the size.

    ### Exercise 3 · `BasicBlock`

    **Goal.** Complete ResNet-18's basic block: add the shortcut.

    **Why by hand.** The shortcut, and *where* it is added, are the whole idea of ResNet. The rest of
    the block is two convs you know, and they are already written.

    **Input and output.** `BasicBlock(c_in, c_out, stride=1)`. Called on x of shape (B, c_in, H, W) it
    returns (B, c_out, H/stride, W/stride). Examples: `BasicBlock(64, 128, stride=2)` turns
    (2, 64, 16, 16) into (2, 128, 8, 8) and has 230,144 parameters (the worked example);
    `BasicBlock(64, 64)` keeps (2, 64, 16, 16) and has 73,984 (no projection).

    **Plan.** The step numbers are those of the comments in the cell. Already written: step 1, the
    four layers of the main path in `__init__`, and step 3, the main path in `forward`.

    * Step 2 (`__init__`): if the block changes the shape (`stride != 1` or `c_in != c_out`; the `if`
      is written), `self.shortcut` is a projection: an `nn.Sequential` of a 1x1 conv `c_in -> c_out`
      with `stride=stride` and `bias=False`, then `nn.BatchNorm2d(c_out)`. Otherwise
      `self.shortcut = nn.Identity()`. Call it exactly `shortcut`: the checker and the MiniResNet of
      section 7 look for that name.
    * Step 4 (`forward`): apply the shortcut to the block's **input** `x`, not to `out`.
    * Step 5: add it to `out`, *then* apply the ReLU.

    ```
    x (B, c_in, H, W) --------------------- shortcut: identity, or 1x1 conv stride s + BN ------+
      |                                                                                        |
      +-> conv1 3x3 stride s -> bn1 -> ReLU -> conv2 3x3 -> bn2 = F(x) ---------------------> (+) -> ReLU
          (B, c_out, H/s, W/s)                               (B, c_out, H/s, W/s)                    (B, c_out, H/s, W/s)
    ```

    **What the checker tests.** `BasicBlock(64, 64, stride=1)` and `BasicBlock(64, 128, stride=2)` on a
    (2, 64, 16, 16) batch, in eval mode (BN uses its stored averages, D1-02): the output shape and the
    parameter count against the reference. Then it sets bn2's weight and bias to 0, which makes
    F(x) = 0, and checks that the block returns ReLU(shortcut(x)); for the identity block that is
    ReLU(x).

    **Common mistakes.**

    | you wrote | the checker prints |
    |---|---|
    | `nn.Identity()` in both cases | `your code raised RuntimeError: The size of tensor a (8) must match the size of tensor b (16) at non-singleton dimension 3` (the stride-2 test) |
    | the projection conv without `stride=stride` | the same message: the shortcut kept 16 x 16 |
    | a projection in both cases | `BasicBlock(64, 64, stride=1): 78,208 params, expected 73,984 -> identity shortcut (no parameters)` and `identity case: with the residual branch off the block must pass x through (relu(x))` |
    | a projection without BatchNorm | `BasicBlock(64, 128, stride=2): 229,888 params, expected 230,144 -> 1x1 conv (stride) + BN projection shortcut` |
    | a projection conv with a bias | `... 230,272 params, expected 230,144 ...`: 128 biases too many |
    | `F.relu(out) + sc` (ReLU before the addition), no final ReLU, or no addition | `with bn2 zeroed the output should be relu(shortcut(x)) - is the shortcut added BEFORE the final ReLU?` |
    | `self.shortcut(out)` instead of `self.shortcut(x)` | `with bn2 zeroed ...` for the identity block, then `... expected input[2, 128, 8, 8] to have 64 channels, but got 128 channels instead` |
    | the projection only when `c_in != c_out` | `[PASS]`, because both tests change the channels. But `BasicBlock(64, 64, stride=2)` would then crash with the size-8-vs-16 message above: keep both conditions |
    """),
    api("nn.Identity", title="Functions you will need"),
    api("nn.Sequential", "nn.Conv2d", "nn.BatchNorm2d", "F.relu", brief=True),
    exercise(
        stub=r'''
        class BasicBlock(nn.Module):
            """ResNet-18 block: y = ReLU(F(x) + shortcut(x)), with F = 3x3 conv (stride) - BN - ReLU - 3x3 conv - BN."""

            def __init__(self, c_in, c_out, stride=1):
                super().__init__()
                # step 1 (given): the main path F: two 3x3 convs, each followed by BatchNorm
                self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(c_out)
                self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(c_out)
                # step 2: the shortcut. A projection is needed when the block changes the shape of x
                if stride != 1 or c_in != c_out:
                    # YOUR CODE: self.shortcut = an nn.Sequential of a 1x1 conv c_in -> c_out
                    #            (stride=stride, bias=False) and a BatchNorm2d over c_out channels
                    raise NotImplementedError    # replace this line with yours
                else:
                    # YOUR CODE: self.shortcut = a module that returns its input unchanged
                    raise NotImplementedError    # replace this line with yours

            def forward(self, x):
                # x: (B, c_in, H, W)
                # step 3 (given): the main path F(x)
                out = self.conv1(x)          # (B, c_out, H/stride, W/stride)
                out = self.bn1(out)          # same shape
                out = F.relu(out)            # same shape
                out = self.conv2(out)        # (B, c_out, H/stride, W/stride)
                out = self.bn2(out)          # same shape: this is F(x)
                # step 4: the shortcut, applied to the block's INPUT x (not to out)
                # YOUR CODE: sc = the shortcut applied to x               -> (B, c_out, H/stride, W/stride)
                # step 5: add the two paths, THEN the final ReLU
                # YOUR CODE: out = out plus sc                            -> same shape
                # YOUR CODE: out = the ReLU of out                        -> same shape
                raise NotImplementedError    # delete this line when steps 4 and 5 are written
                return out

        ok_basic = tests.check_residual_block(BasicBlock)
        ''',
        solution=r'''
        class BasicBlock(nn.Module):
            """ResNet-18 block: y = ReLU(F(x) + shortcut(x)), with F = 3x3 conv (stride) - BN - ReLU - 3x3 conv - BN."""

            def __init__(self, c_in, c_out, stride=1):
                super().__init__()
                # step 1 (given): the main path F: two 3x3 convs, each followed by BatchNorm
                self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(c_out)
                self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(c_out)
                # step 2: the shortcut. A projection is needed when the block changes the shape of x
                if stride != 1 or c_in != c_out:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),   # new channels and size
                        nn.BatchNorm2d(c_out),
                    )
                else:
                    self.shortcut = nn.Identity()    # returns x unchanged; no parameters

            def forward(self, x):
                # x: (B, c_in, H, W)
                # step 3 (given): the main path F(x)
                out = self.conv1(x)          # (B, c_out, H/stride, W/stride)
                out = self.bn1(out)          # same shape
                out = F.relu(out)            # same shape
                out = self.conv2(out)        # (B, c_out, H/stride, W/stride)
                out = self.bn2(out)          # same shape: this is F(x)
                # step 4: the shortcut, applied to the block's INPUT x (not to out)
                sc = self.shortcut(x)        # (B, c_out, H/stride, W/stride)
                # step 5: add the two paths, THEN the final ReLU
                out = out + sc               # same shape: F(x) + shortcut(x)
                out = F.relu(out)            # same shape
                return out

        ok_basic = tests.check_residual_block(BasicBlock)
        '''),
    md(r"""
    ### Exercise 4 · `Bottleneck`

    **Goal.** ResNet-50's block: a 1x1 conv that reduces the channels, a 3x3 conv, a 1x1 conv that
    expands them again, plus the shortcut.

    **Why.** ResNet-50 is 16 of these blocks, and the detectors and segmentation networks of Days 2 and
    3 run on it. The point is the cost: at 256 channels a basic block has 1,180,672 parameters, a
    bottleneck block 70,400 (the cell after the exercise compares them).

    **Input and output.** `Bottleneck(c_in, c_mid, stride=1)`. `c_mid` is the narrow width inside the
    block; the block outputs `c_out = 4 * c_mid` channels (the class attribute `expansion = 4`). Called
    on x of shape (B, c_in, H, W) it returns (B, 4 · c_mid, H/stride, W/stride). Examples:
    `Bottleneck(256, 64)` keeps (2, 256, 14, 14); `Bottleneck(256, 128, stride=2)` turns
    (2, 256, 14, 14) into (2, 512, 7, 7).

    **Worked example.** The parameters of `Bottleneck(256, 64)`: conv1 256 · 64 = 16,384; conv2
    64 · 64 · 9 = 36,864; conv3 64 · 256 = 16,384; the BatchNorms 2 · 64 + 2 · 64 + 2 · 256 = 768; the
    shortcut is the identity (256 in, 256 out, stride 1), 0. Total 70,400.

    **Plan.** Use the attribute names `conv1, bn1, conv2, bn2, conv3, bn3, shortcut`; every conv has
    `bias=False`.

    1. (given) `conv1`: 1x1 conv `c_in -> c_mid`, and `bn1`.
    2. `conv2`: 3x3 conv `c_mid -> c_mid` with `stride=stride` and `padding=1`, and `bn2`.
    3. `conv3`: 1x1 conv `c_mid -> c_out`, and `bn3` over `c_out` channels.
    4. `shortcut`: as in `BasicBlock`, but compare with `c_out`: a projection (1x1 conv
       `c_in -> c_out` with the stride, then BN) when `stride != 1 or c_in != c_out`, else the identity.
    5. `forward`: conv1, bn1, ReLU (given), conv2, bn2, ReLU, conv3, bn3, add the shortcut of x,
       ReLU. There is no ReLU between bn3 and the addition.

    ```
    x (B, c_in, H, W) ------------------------------ shortcut ------------------------------------+
      |                                                                                           |
      +-> conv1 1x1 -> bn1 -> ReLU -> conv2 3x3 stride s -> bn2 -> ReLU -> conv3 1x1 -> bn3 ---> (+) -> ReLU
          (B, c_mid, H, W)             (B, c_mid, H/s, W/s)                  (B, 4 c_mid, H/s, W/s)
    ```

    **What the checker tests.** `Bottleneck(256, 64)` and `Bottleneck(256, 128, stride=2)` on 14 x 14,
    and `Bottleneck(64, 64)` on 8 x 8 (here c_in = 64 but c_out = 256: a projection without a stride):
    the output shapes, the parameter count against torchvision's own `Bottleneck`, and ReLU(shortcut(x))
    with bn3 zeroed.

    **Common mistakes.**

    | you wrote | the checker prints |
    |---|---|
    | conv3 to `c_mid` instead of `c_out` | `Bottleneck(256, 64, stride=1): output (2, 64, 14, 14), expected (2, 256, 14, 14)` |
    | a projection only when `stride != 1` | `your code raised RuntimeError: The size of tensor a (256) must match the size of tensor b (64) at non-singleton dimension 1` (the `Bottleneck(64, 64)` test) |
    | no `padding=1` on conv2 | `The size of tensor a (12) must match the size of tensor b (14) at non-singleton dimension 3` |
    | convs with a bias | `Bottleneck(256, 64, stride=1): 70,784 params, torchvision's has 70,400` |
    | no final ReLU | `with bn3 zeroed the output should be relu(shortcut(x))` |
    | a ReLU after bn3, before the addition | `[PASS]`: the checker cannot see it (with bn3 zeroed both versions give the same). But F(x) could then never be negative: the block could only add to x, never subtract |
    | the stride on conv1 instead of conv2 | `[PASS]` as well: same shapes and parameters. torchvision puts it on the 3x3 conv, which looks at every input pixel; a 1x1 conv with stride 2 skips three pixels out of four |
    """),
    exercise(
        stub=r'''
        class Bottleneck(nn.Module):
            """ResNet-50 block: 1x1 reduce - 3x3 (stride) - 1x1 expand (x4), BN after each conv, + shortcut, ReLU."""
            expansion = 4                    # the block outputs expansion * c_mid channels

            def __init__(self, c_in, c_mid, stride=1):
                super().__init__()
                c_out = c_mid * self.expansion
                # step 1 (given): 1x1 conv that reduces c_in -> c_mid channels, and its BatchNorm
                self.conv1 = nn.Conv2d(c_in, c_mid, kernel_size=1, bias=False)
                self.bn1 = nn.BatchNorm2d(c_mid)
                # step 2: 3x3 conv c_mid -> c_mid that carries the block's stride (padding 1, no bias), and its BatchNorm
                # YOUR CODE: self.conv2 = ____
                # YOUR CODE: self.bn2 = ____
                # step 3: 1x1 conv that expands c_mid -> c_out channels (no bias), and its BatchNorm
                # YOUR CODE: self.conv3 = ____
                # YOUR CODE: self.bn3 = ____
                # step 4: the shortcut, as in BasicBlock: if stride != 1 or c_in != c_out, a projection
                #         (1x1 conv c_in -> c_out with stride=stride, bias=False, then BatchNorm2d), else nn.Identity()
                # YOUR CODE: the if / else that sets self.shortcut
                raise NotImplementedError    # delete this line when steps 2, 3 and 4 are written

            def forward(self, x):
                # x: (B, c_in, H, W)
                # step 5 (given): reduce
                out = self.conv1(x)          # (B, c_mid, H, W)
                out = self.bn1(out)          # same shape
                out = F.relu(out)            # same shape
                # step 6: 3x3 conv, BN, ReLU
                # YOUR CODE: out = conv2 applied to out                  -> (B, c_mid, H/stride, W/stride)
                # YOUR CODE: out = bn2 applied to out                    -> same shape
                # YOUR CODE: out = the ReLU of out                       -> same shape
                # step 7: expand, BN, and no ReLU here
                # YOUR CODE: out = conv3 applied to out                  -> (B, c_out, H/stride, W/stride)
                # YOUR CODE: out = bn3 applied to out                    -> same shape: this is F(x)
                # step 8: add the shortcut of the INPUT x, then ReLU
                # YOUR CODE: sc = the shortcut applied to x              -> (B, c_out, H/stride, W/stride)
                # YOUR CODE: out = out plus sc
                # YOUR CODE: out = the ReLU of out
                raise NotImplementedError    # delete this line when steps 6, 7 and 8 are written
                return out

        ok_bottleneck = tests.check_bottleneck(Bottleneck)
        ''',
        solution=r'''
        class Bottleneck(nn.Module):
            """ResNet-50 block: 1x1 reduce - 3x3 (stride) - 1x1 expand (x4), BN after each conv, + shortcut, ReLU."""
            expansion = 4                    # the block outputs expansion * c_mid channels

            def __init__(self, c_in, c_mid, stride=1):
                super().__init__()
                c_out = c_mid * self.expansion
                # step 1 (given): 1x1 conv that reduces c_in -> c_mid channels, and its BatchNorm
                self.conv1 = nn.Conv2d(c_in, c_mid, kernel_size=1, bias=False)
                self.bn1 = nn.BatchNorm2d(c_mid)
                # step 2: 3x3 conv c_mid -> c_mid that carries the block's stride (padding 1, no bias), and its BatchNorm
                self.conv2 = nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(c_mid)
                # step 3: 1x1 conv that expands c_mid -> c_out channels (no bias), and its BatchNorm
                self.conv3 = nn.Conv2d(c_mid, c_out, kernel_size=1, bias=False)
                self.bn3 = nn.BatchNorm2d(c_out)
                # step 4: the shortcut, as in BasicBlock: if stride != 1 or c_in != c_out, a projection
                #         (1x1 conv c_in -> c_out with stride=stride, bias=False, then BatchNorm2d), else nn.Identity()
                if stride != 1 or c_in != c_out:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(c_out),
                    )
                else:
                    self.shortcut = nn.Identity()

            def forward(self, x):
                # x: (B, c_in, H, W)
                # step 5 (given): reduce
                out = self.conv1(x)          # (B, c_mid, H, W)
                out = self.bn1(out)          # same shape
                out = F.relu(out)            # same shape
                # step 6: 3x3 conv, BN, ReLU
                out = self.conv2(out)        # (B, c_mid, H/stride, W/stride)
                out = self.bn2(out)          # same shape
                out = F.relu(out)            # same shape
                # step 7: expand, BN, and no ReLU here
                out = self.conv3(out)        # (B, c_out, H/stride, W/stride)
                out = self.bn3(out)          # same shape: this is F(x)
                # step 8: add the shortcut of the INPUT x, then ReLU
                sc = self.shortcut(x)        # (B, c_out, H/stride, W/stride)
                out = out + sc               # same shape
                out = F.relu(out)            # same shape
                return out

        ok_bottleneck = tests.check_bottleneck(Bottleneck)
        '''),
    md(r"""
    The next cell compares the two blocks at 256 channels on a 14 x 14 input. `torchkit.profile` counts
    the parameters and the multiply-adds for one input. Its first two lines store the classes used
    from here on: `Basic` is your `BasicBlock` if it passed its checker, otherwise the reference (the
    same for `Bott`). Exercise 6 builds its network from `Basic`.
    """),
    api("torchkit.profile", brief=True),
    code(r"""
    Basic = tests.use_or_reference(ok_basic, globals().get("BasicBlock"), day1.BasicBlock, "BasicBlock")
    Bott = tests.use_or_reference(ok_bottleneck, globals().get("Bottleneck"), day1.Bottleneck, "Bottleneck")
    blocks_to_compare = [("basic 256->256", Basic(256, 256)), ("bottleneck 256->(64)->256", Bott(256, 64))]
    for name, blk in blocks_to_compare:
        prof = torchkit.profile(blk, (256, 14, 14))           # {"params": ..., "macs": ...} for one 256 x 14 x 14 input
        print(f"{name:28s} params {prof['params']:>9,d}   MACs @14x14 {prof['macs'] / 1e6:6.1f} M")
    print("-> the bottleneck is ~17x cheaper at 256 channels: that is how ResNet-50 affords 50 layers")
    """),
    md(r"""
    ### Why it trains: gradient flow at initialisation

    This experiment shows the degradation problem at its root: how much gradient reaches the first
    layer of plain and residual networks of growing depth. Three words first:

    * **Initialisation**: the random values the weights get when a layer is created, before any
      training. PyTorch's default draws them from a small range that depends on the layer's size.
    * The **norm** of a tensor, written ||·||, is its size as one number: the square root of the sum of
      its squared entries. ||dL/dW|| of the first conv is the size of the gradient of the loss with
      respect to that conv's weights: how far one training step would move them.
    * **Vanishing gradient**: on its way back, the gradient is multiplied by one factor per layer (the
      chain rule of D0-00). If the factors are below 1, after many layers almost nothing is left and
      the first layers cannot learn. **Exploding gradient**: the factors are above 1 and the gradient
      grows out of all proportion.

    The experiment: networks with 4 to 60 layers with weights (3x3 convs with 16 channels, built by
    `day1.make_deep_net`), plain or residual, without and with BatchNorm, at their default
    initialisation. One random batch goes forward, the loss goes backward once, and we record
    ||dL/dW|| of the first conv, averaged over 3 random initialisations. Nothing is trained. The
    vertical axis is logarithmic: equal steps are equal factors.
    """),
    api("torch.randint", "np.maximum"),
    api("torch.randn", "np.mean", brief=True),
    code(r"""
    torch.manual_seed(0)                       # the same random batch every time the cell runs
    depths = [4, 10, 20, 30, 40, 60]
    x_g = torch.randn(16, 3, 16, 16)           # 16 random "images" of 3 x 16 x 16
    y_g = torch.randint(0, 10, (16,))          # 16 random labels out of 10 classes
    res = {}                                   # (bn, residual) -> one gradient norm per depth
    for bn in (False, True):
        for residual in (False, True):
            vals = []
            for d in depths:
                g = []
                for seed in range(3):          # 3 different random initialisations
                    torch.manual_seed(seed)
                    net = day1.make_deep_net(d, residual, bn=bn)
                    g.append(day1.first_layer_grad_norm(net, x_g, y_g))
                vals.append(np.mean(g))
            res[(bn, residual)] = vals

    # Draws the gradient norm against depth: left without BatchNorm, right with
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, bn in zip(axes, (False, True)):
        for residual, c in ((False, viz.PALETTE[3]), (True, viz.PALETTE[0])):
            ax.semilogy(depths, np.maximum(res[(bn, residual)], 1e-30), "-o", color=c, label="residual" if residual else "plain")
        ax.set_title(f"{'with' if bn else 'without'} BatchNorm"); ax.set_xlabel("depth (weight layers)")
        ax.grid(alpha=.3, which="both"); ax.legend()
    axes[0].set_ylabel("||dL/dW|| of the first conv (log)")
    plt.suptitle("Gradient reaching the first layer at initialisation", y=1.02); plt.tight_layout(); plt.show()

    for (bn, residual), vals in res.items():   # the numbers behind the plot: depth:gradient norm
        line = f"BN={bn!s:5}  {'residual' if residual else 'plain   '}"
        for d, g in zip(depths, vals):
            line += f"  {d}:{g:.1e}"
        print(line)
    """),
    md(r"""
    What to look at (red = plain, blue = residual):

    * **Without BN, plain** (left, red): the gradient shrinks by a roughly constant factor per layer, a
      straight line falling on the log axis: 6.1e-03 at 4 layers, 1.7e-09 at 20, 9.8e-18 at 40, and
      exactly 0 at 60 (drawn at 1e-30, the bottom of the plot). The first layer of the deep net
      receives nothing and cannot learn.
    * **Without BN, residual** (left, blue): the gradient grows with depth, from 6.6e-02 to 1.5e+01, a
      factor of about 230 from 4 to 60 layers. The plain net loses a factor of almost 10^15 by 40
      layers. The `+ x` paths carry the gradient back.
    * **With BN, plain** (right, red): BatchNorm does not fix the plain net: now the gradient *grows*
      with depth, from 7.4e-02 to 1.6e+03 (each BN-ReLU-conv layer amplifies it a little; Yang et al.
      2019 analyse why), so deep plain nets with BN still train badly. That is the 2015 paper's
      degradation problem.
    * **With BN, residual** (right, blue): almost flat, from 7.7e-02 to 4.4e-01 at every depth. This
      combination trains with 152 layers, and in the paper even with 1 000.

    Every modern architecture has residual connections: ConvNeXt, ViT and the other transformers,
    many U-Net variants, and the large language models.
    """),
    md(r"""
    ### Lab · Gradient flow, live
    **Predict first:** the figure used PyTorch's default initialisation. With **He** initialisation
    (designed to keep the signal size constant through ReLU layers) and no BatchNorm, does the plain
    net still lose its gradient at depth 60? And does the residual net stay flat? The lab runs the same
    experiment with fully-connected layers, one real backward pass per change; work through its
    "Try this" list (5 minutes).
    """),
    code(r"""
    labs.show("gradient-flow")
    """),
    md(r"""
    ### Residual (add) vs. dense (concatenate)

    DenseNet (2017) connects each layer to **all** earlier ones by *concatenation*: layer $l$ receives
    $[x_0, x_1, \dots, x_{l-1}]$, the block's input and every earlier layer's output stacked along the
    channels, and adds `growth` new channels of its own (here 32). Features are reused instead of
    learned again, which makes DenseNets very parameter-efficient, at the cost of memory: the channel
    count grows inside a block, and every output must be kept until the end of the block. The original
    course labelled a dense block "residual"; here the two are side by side so that you can tell them
    apart by the **shapes**.
    """),
    code(r"""
    # Channel counts through 4 residual blocks and 4 dense layers (random weights: only the shapes matter here)
    x = torch.randn(1, 64, 16, 16)
    res_chs = [64]                       # channels after each step of the residual chain
    dense_chs = [64]                     # the same for the dense chain
    r = x
    d = x
    for _ in range(4):
        r = day1.BasicBlock(64, 64)(r)                     # residual: as many channels out as in
        res_chs.append(r.shape[1])
        d = day1.DenseLayer(d.shape[1], growth=32)(d)      # dense: the input's channels + 32 new ones
        dense_chs.append(d.shape[1])
    print("residual chain:", res_chs)
    print("dense chain:   ", dense_chs)

    # Draws the two chains (left and middle) and their channel counts (right)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for ax, title, op, chs in [(axes[0], "residual: y = F(x) + x", "+", res_chs), (axes[1], "dense: y = concat(x, F(x))", "cat", dense_chs)]:
        ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.set_title(title)
        for i, c in enumerate(chs):
            w = 0.25 + 1.3 * c / 192
            ax.add_patch(Rectangle((0.3 + i * 2.0, 2 - w / 2 * 1.2), 0.5, w * 1.2, fc=viz.color(i), ec="k"))
            ax.text(0.55 + i * 2.0, 0.35, f"{c} ch", ha="center", fontsize=8)
            if i:
                ax.text(-0.45 + i * 2.0, 2.0, op, ha="center", va="center", fontsize=10, weight="bold")
                if op == "+":
                    srcs = [i - 1]               # residual: an arrow from the previous output only
                else:
                    srcs = list(range(i))        # dense: arrows from ALL previous outputs
                for j in srcs:
                    ax.annotate("", xy=(0.55 + i * 2.0, 2.0 + w * 0.6), xytext=(0.55 + j * 2.0, 2.0 + 0.6),
                                arrowprops=dict(arrowstyle="->", lw=0.8, color="gray", connectionstyle="arc3,rad=-0.35"))
    axes[2].plot(range(5), res_chs, "-o", label="4 residual blocks"); axes[2].plot(range(5), dense_chs, "-o", label="4 dense layers (growth 32)")
    axes[2].set_xlabel("layer"); axes[2].set_ylabel("channels of the output"); axes[2].legend(); axes[2].grid(alpha=.3)
    axes[2].set_xticks(range(5))
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    The residual chain keeps 64 channels (every block adds into a tensor of the same shape); each dense
    layer appends 32 new channels, so after 4 layers the tensor is 64 + 4 · 32 = 192 channels wide, and
    every later layer reads all of them.
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · Efficiency: depthwise-separable convolutions, SE, EfficientNet, ConvNeXt

    **Why.** Phones, cameras and real-time systems cannot afford ResNet-50's 4 billion multiply-adds
    per image. The blocks of this section do the same job for a fraction of the compute. Mobile
    networks (MobileNet, EfficientNet) are built from them, and ConvNeXt, at the end of this section,
    uses them with 7x7 kernels.

    **The idea.** A standard 3x3 conv mixes space and channels at once: every output channel looks at a
    3x3 patch of *every* input channel, $C_{\text{in}} \cdot C_{\text{out}} \cdot 9$ weights. MobileNet
    (2017) splits it into two convs:

    1. a **depthwise** conv: one 3x3 filter per input channel, applied to that channel alone. Space
       only: $C_{\text{in}}$ channels in and out, $9\,C_{\text{in}}$ weights;
    2. a **pointwise** conv: a 1x1 conv, which mixes the channels at each pixel. Channels only:
       $C_{\text{in}} \cdot C_{\text{out}}$ weights.

    The pair is a **depthwise-separable** convolution.

    **`groups`.** In PyTorch the depthwise conv is an `nn.Conv2d` with the argument `groups`.
    `groups=g` splits the input and the output channels into g groups and connects each output group
    only to its own input group, so every filter sees $C_{\text{in}} / g$ channels. `groups=1` (the
    default) is the standard conv. `groups=C_in` makes one group per channel: each filter sees a single
    channel, and the weight tensor has shape (C_in, 1, 3, 3).

    **Worked example.** C_in = 4, C_out = 8, 3x3 kernels, no biases:

    | conv | weight shape (out, in per filter, k, k) | weights |
    |---|---|---|
    | standard 3x3 | (8, 4, 3, 3) | 8 · 4 · 9 = 288 |
    | depthwise 3x3, `groups=4` | (4, 1, 3, 3) | 4 · 9 = 36 |
    | pointwise 1x1 | (8, 4, 1, 1) | 8 · 4 = 32 |
    | depthwise + pointwise | | 36 + 32 = 68: 4.2x fewer |

    In general the ratio is $(9\,C_{\text{in}} + C_{\text{in}} C_{\text{out}}) / (9\,C_{\text{in}}
    C_{\text{out}}) = 1/C_{\text{out}} + 1/9$; here 1/8 + 1/9 = 0.236 = 68 / 288. For wide layers
    $1/C_{\text{out}}$ is tiny and the saving approaches 9x (8.7x at 256 channels), for a small loss of
    accuracy. The cell builds the three convs, prints their weight shapes, and checks that the
    depthwise conv keeps the channels apart: wiping input channel 1 changes output channel 1 only.
    """),
    api("torch.equal"),
    api("Tensor.clone", brief=True),
    code(r"""
    standard = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)
    depthwise = nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4, bias=False)    # one filter per input channel
    pointwise = nn.Conv2d(4, 8, kernel_size=1, bias=False)                         # mixes the 4 channels at each pixel
    for name, conv in [("standard 3x3 ", standard), ("depthwise 3x3", depthwise), ("pointwise 1x1", pointwise)]:
        print(f"{name}: weight {tuple(conv.weight.shape)} -> {conv.weight.numel()} weights")
    n_separable = depthwise.weight.numel() + pointwise.weight.numel()
    print(f"depthwise + pointwise: {n_separable} weights, {standard.weight.numel() / n_separable:.1f}x fewer than the standard conv")

    x = torch.randn(1, 4, 5, 5)
    x_wiped = x.clone()
    x_wiped[0, 1] = 0                                   # wipe input channel 1
    before = depthwise(x)
    after = depthwise(x_wiped)
    for c in range(4):
        print(f"depthwise output channel {c} changed: {not torch.equal(before[0, c], after[0, c])}")
    """),
    md(r"""
    ### Lab · What does a block cost?
    **Predict first:** at 512 channels in and out, you switch the kernel from 3x3 to 7x7. How many
    times more weights does a standard conv need? And a depthwise-separable block? The lab puts the
    standard conv, the ResNet bottleneck, the depthwise-separable block and an Inception-style
    1x1 -> kxk branch side by side for the same input, with every formula filled in (5 minutes).
    """),
    code(r"""
    labs.show("block-cost")
    """),
    md(r"""
    ### Exercise 5 · `DepthwiseSeparable`

    **Goal.** MobileNet's block as an `nn.Module`.

    **Why by hand.** It is four layers, but one argument, `groups=c_in`, is the whole idea. Writing it,
    and seeing the parameter count drop, makes it stick.

    **Input and output.** `DepthwiseSeparable(c_in, c_out, stride=1)`. Called on x of shape
    (B, c_in, H, W) it returns (B, c_out, H/stride, W/stride). Example: `DepthwiseSeparable(32, 64)`
    turns (2, 32, 16, 16) into (2, 64, 16, 16) and has 32 · 9 + 2 · 32 + 32 · 64 + 2 · 64 =
    288 + 64 + 2,048 + 128 = 2,528 parameters (the depthwise conv, bn1, the pointwise conv, bn2).

    **Plan.**
    1. `self.depthwise`: a 3x3 conv `c_in -> c_in` with `stride=stride`, `padding=1`, `groups=c_in`,
       `bias=False`.
    2. `self.bn1`: BatchNorm over the `c_in` channels the depthwise conv makes.
    3. `self.pointwise`: a 1x1 conv `c_in -> c_out`, `bias=False` (stride 1: the depthwise conv
       already did the striding).
    4. `self.bn2`: BatchNorm over `c_out` channels.
    5. `forward`: depthwise, bn1, ReLU, pointwise, bn2, ReLU, one line each.

    ```
    x                                          (B, c_in, H, W)
    depthwise 3x3, groups=c_in, stride s   ->  (B, c_in, H/s, W/s)     space: each channel on its own
    bn1, ReLU                              ->  (B, c_in, H/s, W/s)
    pointwise 1x1                          ->  (B, c_out, H/s, W/s)    channels: mixed at each pixel
    bn2, ReLU                              ->  (B, c_out, H/s, W/s)
    ```

    **What the checker tests.** `DepthwiseSeparable(32, 64)` and `DepthwiseSeparable(128, 256,
    stride=2)` on 16 x 16 inputs: the output shape; the parameter count, $9\,C_{\text{in}} +
    C_{\text{in}} C_{\text{out}}$ conv weights plus 2 per channel for each BatchNorm (the lab's numbers
    with "count BatchNorm" on); and that one conv has `groups == c_in`. It also prints, for
    comparison, the weights of a standard 3x3 conv 32 -> 64. In the lab, `depthwise` is the yellow
    segment and `pointwise` the orange one.

    **Common mistakes.**

    | you wrote | the checker prints |
    |---|---|
    | no `groups=c_in` | `DepthwiseSeparable(32, 64): 11,456 params, expected 2,528 = ...` and `no depthwise conv found: the 3x3 conv needs groups=c_in` |
    | convs with a bias | `DepthwiseSeparable(32, 64): 2,624 params, expected 2,528 ...`: 32 + 64 biases too many |
    | no `padding=1` | `DepthwiseSeparable(32, 64, 1): output (2, 64, 14, 14), expected (2, 64, 16, 16)` |
    | `self.bn1 = nn.BatchNorm2d(c_out)` | `your code raised RuntimeError: running_mean should contain 32 elements not 64`: bn1 sees the c_in channels of the depthwise conv |
    | a 3x3 pointwise conv | `DepthwiseSeparable(32, 64): 18,912 params, expected 2,528 ...`: that is a standard conv again |
    | the stride on the pointwise conv | `[PASS]`: same shapes and parameters. But the depthwise conv then computes every position and the 1x1 conv throws three out of four away |
    """),
    api("nn.Conv2d", "nn.BatchNorm2d", "F.relu", title="Functions you will need", brief=True),
    exercise(
        stub=r'''
        class DepthwiseSeparable(nn.Module):
            """MobileNet block: depthwise 3x3 (space) - BN - ReLU - pointwise 1x1 (channels) - BN - ReLU."""

            def __init__(self, c_in, c_out, stride=1):
                super().__init__()
                # step 1: depthwise conv: 3x3, c_in -> c_in, one filter per channel (groups=c_in),
                #         stride=stride, padding=1, no bias
                # YOUR CODE: self.depthwise = ____
                # step 2: BatchNorm over the c_in channels the depthwise conv makes
                # YOUR CODE: self.bn1 = ____
                # step 3: pointwise conv: 1x1, c_in -> c_out, no bias
                # YOUR CODE: self.pointwise = ____
                # step 4: BatchNorm over the c_out channels
                # YOUR CODE: self.bn2 = ____
                raise NotImplementedError    # delete this line when steps 1-4 are written

            def forward(self, x):
                # x: (B, c_in, H, W)
                # step 5: depthwise conv, BN, ReLU
                # YOUR CODE: x = the depthwise conv applied to x          -> (B, c_in, H/stride, W/stride)
                # YOUR CODE: x = bn1 applied to x                         -> same shape
                # YOUR CODE: x = the ReLU of x                            -> same shape
                # step 6: pointwise conv, BN, ReLU
                # YOUR CODE: x = the pointwise conv applied to x          -> (B, c_out, H/stride, W/stride)
                # YOUR CODE: x = bn2 applied to x                         -> same shape
                # YOUR CODE: x = the ReLU of x                            -> same shape
                raise NotImplementedError    # delete this line when steps 5 and 6 are written
                return x

        ok_dw = tests.check_dw_params(DepthwiseSeparable)
        ''',
        solution=r'''
        class DepthwiseSeparable(nn.Module):
            """MobileNet block: depthwise 3x3 (space) - BN - ReLU - pointwise 1x1 (channels) - BN - ReLU."""

            def __init__(self, c_in, c_out, stride=1):
                super().__init__()
                # step 1: depthwise conv: 3x3, c_in -> c_in, one filter per channel (groups=c_in),
                #         stride=stride, padding=1, no bias
                self.depthwise = nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride, padding=1, groups=c_in, bias=False)
                # step 2: BatchNorm over the c_in channels the depthwise conv makes
                self.bn1 = nn.BatchNorm2d(c_in)
                # step 3: pointwise conv: 1x1, c_in -> c_out, no bias
                self.pointwise = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
                # step 4: BatchNorm over the c_out channels
                self.bn2 = nn.BatchNorm2d(c_out)

            def forward(self, x):
                # x: (B, c_in, H, W)
                # step 5: depthwise conv, BN, ReLU
                x = self.depthwise(x)        # (B, c_in, H/stride, W/stride)
                x = self.bn1(x)              # same shape
                x = F.relu(x)                # same shape
                # step 6: pointwise conv, BN, ReLU
                x = self.pointwise(x)        # (B, c_out, H/stride, W/stride)
                x = self.bn2(x)              # same shape
                x = F.relu(x)                # same shape
                return x

        ok_dw = tests.check_dw_params(DepthwiseSeparable)
        '''),
    md(r"""
    The next cell counts the multiply-adds of a standard 3x3 conv and of your block (or the reference)
    for widths from 32 to 512 channels, on a 28 x 28 input, and draws them side by side. The vertical
    axis is logarithmic; the number on each orange bar is how many times fewer multiply-adds the
    depthwise-separable block needs.
    """),
    api("np.arange", brief=True),
    code(r"""
    DW = tests.use_or_reference(ok_dw, globals().get("DepthwiseSeparable"), day1.DepthwiseSeparable, "DepthwiseSeparable")
    widths = [32, 64, 128, 256, 512]
    std_m = []                           # millions of multiply-adds of the standard conv, one per width
    dw_m = []                            # the same for the depthwise-separable block
    for c in widths:
        std_m.append(day1.profile_macs(nn.Conv2d(c, c, 3, padding=1, bias=False), (c, 28, 28)) / 1e6)
        dw_m.append(day1.profile_macs(DW(c, c), (c, 28, 28)) / 1e6)

    # Draws the two bar series and the saving factor on each orange bar
    fig, ax = plt.subplots(figsize=(8, 3.4)); xw = np.arange(len(widths))
    ax.bar(xw - 0.2, std_m, 0.4, label="standard 3x3 conv"); ax.bar(xw + 0.2, dw_m, 0.4, label="depthwise 3x3 + pointwise 1x1")
    for i in range(len(widths)):
        ax.text(xw[i] + 0.2, dw_m[i], f"{std_m[i] / dw_m[i]:.1f}x", ha="center", va="bottom", fontsize=8)
    tick_labels = []
    for c in widths:
        tick_labels.append(f"{c} -> {c}")
    ax.set_xticks(xw); ax.set_xticklabels(tick_labels); ax.set_yscale("log")
    ax.set_ylabel("M multiply-adds @ 28x28 (log)"); ax.set_title("Same input / output shape, a fraction of the compute"); ax.legend(); ax.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: the saving printed on each bar grows with the width (7.0x at 32 channels, 8.8x at
    512) towards its limit $k^2 = 9$: at large widths almost all of the remaining cost is the 1x1
    pointwise conv.

    ### Three more ideas, in brief

    **Squeeze-and-excitation** (SE, 2018) lets a block turn its channels up or down depending on the
    image, a form of **channel attention**: it averages each channel to one number, turns those
    numbers into one weight between 0 and 1 per channel, and multiplies every channel by its weight.
    MobileNetV3 and EfficientNet use it.

    <details>
    <summary><b>More: squeeze-and-excitation and inverted residuals, step by step</b> (optional, click to open)</summary>

    The three steps of SE on a feature map (B, C, H, W):

    1. **squeeze**: global average pooling, one number per channel: (B, C, 1, 1);
    2. **excite**: a tiny **MLP** (two linear layers with a ReLU between them, here written as 1x1
       convs: C -> C/4 -> C), then a **sigmoid**, which squashes any number into (0, 1): one weight per
       channel;
    3. multiply every channel of the feature map by its weight: (B, C, H, W) again.

    It costs two tiny layers per block (`day1.SqueezeExcite` is a version of it). MobileNetV3 and
    EfficientNet use it inside **inverted residual** blocks: a 1x1 conv that *expands* the channels,
    a depthwise 3x3, SE, and a 1x1 conv that projects back to few channels, with the shortcut between
    the two thin ends. A ResNet bottleneck is thin in the middle; an inverted residual is wide in the
    middle.

    </details>

    **EfficientNet** (2019) asked how to make a good small network bigger. Its answer, **compound
    scaling**: grow the depth, the width and the input resolution *together*, with fixed ratios. The
    family runs from B0 (224 px, 5.3 M parameters, 77.7 % top-1) to B7 (600 px, 66 M, 84.4 %).

    <details>
    <summary><b>More: compound scaling in numbers</b> (optional, click to open)</summary>

    Depth is multiplied by 1.2^φ, width by 1.1^φ and the input resolution by 1.15^φ, where the number
    φ (phi) says how much bigger; the compute then grows by about 2^φ. Published numbers:

    | model | resolution | width x | depth x | params | top-1 |
    |---|---|---|---|---|---|
    | B0 | 224 | 1.0 | 1.0 | 5.3 M | 77.7 |
    | B1 | 240 | 1.0 | 1.1 | 7.8 M | 79.2 |
    | B2 | 260 | 1.1 | 1.2 | 9.2 M | 80.3 |
    | B3 | 300 | 1.2 | 1.4 | 12 M | 81.7 |
    | B4 | 380 | 1.4 | 1.8 | 19 M | 83.4 |
    | B7 | 600 | 2.0 | 3.1 | 66 M | 84.4 |

    </details>

    **ConvNeXt** (2022) started from ResNet-50 and applied, one at a time, the design choices of vision
    transformers, measuring the accuracy after each step. The result is a pure convnet that matches
    Swin, a vision transformer of 2021. The chart below shows the accuracy after each step; what
    matters is the size of the steps, not every name.

    <details>
    <summary><b>More: what each ConvNeXt step means</b> (optional, click to open)</summary>

    | step | what it means |
    |---|---|
    | stage ratio 3:3:9:3 | the number of blocks in the four stages (ResNet-50 has 3:4:6:3) |
    | patchify stem 4x4/4 | the stem is one 4x4 conv with stride 4: it cuts the image into non-overlapping 4x4 patches |
    | depthwise conv + width 96 | the 3x3 conv of each block becomes depthwise, and the first stage gets 96 channels |
    | inverted bottleneck | each block widens 4x in the middle instead of narrowing (as in MobileNet) |
    | depthwise conv moved up | the depthwise conv goes first in the block, before the wide part |
    | kernel 3 -> 7 | the depthwise conv sees 7x7 (cheap in a depthwise conv: the lab's first "Try this") |
    | ReLU -> GELU | **GELU** is a smooth version of ReLU, used by transformers |
    | fewer activations, fewer norms | one activation and one normalisation per block instead of three |
    | BN -> LayerNorm | **LayerNorm** normalises the channels of each pixel of each image, instead of each channel over the batch |
    | separate downsampling layers | the halving between stages is done by its own small conv layer |

    </details>
    """),
    code(r"""
    # Draws the ConvNeXt path (published numbers): one bar per step, red = the training recipe alone
    steps = [("ResNet-50, 2015 recipe", 76.1), ("modern training recipe (300 ep, AdamW, aug)", 78.8),
             ("stage ratio 3:3:9:3", 79.4), ("patchify stem 4x4/4", 79.5), ("depthwise conv + width 96", 80.5),
             ("inverted bottleneck", 80.6), ("depthwise conv moved up", 79.9), ("kernel 3 -> 7", 80.6),
             ("ReLU -> GELU", 80.6), ("fewer activations", 81.3), ("fewer norms", 81.4), ("BN -> LayerNorm", 81.5),
             ("separate downsampling layers = ConvNeXt-T", 82.0)]
    names = []
    lengths = []                         # bar lengths: the bars start at 75 %
    colours = []
    for i, (name, top1) in enumerate(steps):
        names.append(name)
        lengths.append(top1 - 75)
        colours.append(viz.PALETTE[3] if i == 1 else viz.PALETTE[0])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh(range(len(steps)), lengths, left=75, color=colours)
    for i, (name, top1) in enumerate(steps):
        ax.text(top1 + 0.05, i, f"{top1:.1f}", va="center", fontsize=8)
    ax.set_yticks(range(len(steps))); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    ax.set_xlim(75, 83); ax.set_xlabel("ImageNet top-1 (%)"); ax.set_title("Modernising a ResNet into ConvNeXt (Liu et al. 2022, Fig. 2); red = training recipe alone")
    ax.grid(alpha=.3, axis="x"); plt.tight_layout(); plt.show()
    """),
    md(r"""
    Read the bars top to bottom: the largest single step (+2.7 points) is the training recipe, with no
    change to the network; the eleven design changes add +3.2 together, most of them a few tenths
    each (one, moving the depthwise conv up, even costs 0.7 before the 7x7 kernel wins it back).

    ## 6 · Reading a model

    **Why.** Before you train, fine-tune or cut up a network (D1-04 replaces ResNet heads; Day 2 uses
    the stages of a ResNet as a detector's **backbone**, the part that computes the features), you
    need to know its stages, their shapes and where its parameters and compute are. A summary table
    tells you.

    `day1.summarize(model, input_size, depth)` (which calls `torchkit.summarize`) runs one dummy image
    of `input_size` = (C, H, W) through the model. A **forward hook**, a small function that PyTorch
    calls every time a given layer runs, records each layer's output shape. The table has one row per
    layer: its name (the attribute path from `__init__`), its type, the output shape for one image,
    its parameters and its multiply-adds. With `depth=0` there is one row per **top-level module**, an
    attribute the model's own `__init__` created; for ResNet-18 those are `conv1, bn1, relu, maxpool`
    (the stem), `layer1` to `layer4` (the four stages, each an `nn.Sequential` of 2 basic blocks),
    `avgpool` and `fc` (the head).

    **How the multiply-adds are counted.** A conv uses each of its weights once per output position:
    MACs = weights x output height x output width. Worked example, ResNet-18's first conv (3 -> 64
    channels, 7x7, output 112 x 112): 3 · 64 · 7 · 7 = 9,408 weights, and 9,408 · 112 · 112 =
    118,013,952 MACs. A linear layer uses each weight once: `fc` (512 -> 1000) has 512 · 1000 + 1000 =
    513,000 parameters but 512,000 MACs (the 1 000 biases are added, not multiplied). BatchNorm, ReLU
    and pooling are counted as 0: they are cheap next to the convs.

    Find both numbers in the first rows of the summary.
    """),
    api("torchvision.models.resnet18"),
    code(r"""
    r18 = torchvision.models.resnet18(weights=None)
    print(day1.summarize(r18, (3, 224, 224), depth=0))
    """),
    md(r"""
    What to look at, stage by stage (`layer1` to `layer4`): from one stage to the next the image
    shrinks from 56 x 56 to 7 x 7 while the channels double, so

    * the **compute per stage stays roughly constant** (462 million MACs in `layer1`, 411 million in
      each of the others): halving H and W divides the output positions by 4, doubling the channels
      multiplies every conv's weights by 4 (both its input and its output channels double);
    * the **parameters grow about 4x per stage** (148 thousand, 526 thousand, 2.1 million, 8.4 million):
      `layer4` holds 72 % of the network's 11.7 million parameters.

    The next cell profiles six torchvision networks (random weights, nothing downloaded) and puts them
    on one chart.
    """),
    code(r"""
    zoo = {"vgg16": torchvision.models.vgg16, "resnet18": torchvision.models.resnet18, "resnet50": torchvision.models.resnet50,
           "mobilenet_v3_large": torchvision.models.mobilenet_v3_large, "efficientnet_b0": torchvision.models.efficientnet_b0,
           "convnext_tiny": torchvision.models.convnext_tiny}
    rows = []                                                  # (name, millions of parameters, billions of MACs)
    for name, ctor in zoo.items():
        prof = torchkit.profile(ctor(weights=None), (3, 224, 224))
        rows.append((name, prof["params"] / 1e6, prof["macs"] / 1e9))
    print(f"{'model':<20}{'params (M)':>12}{'GMACs @224':>12}")
    for name, params_m, gmacs in rows:
        print(f"{name:<20}{params_m:>12.1f}{gmacs:>12.2f}")

    # Draws parameters against compute, both on log axes
    label_offsets = {"mobilenet_v3_large": (-10, 10), "efficientnet_b0": (6, -14), "convnext_tiny": (8, 4), "resnet50": (8, -10)}
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (name, params_m, gmacs) in enumerate(rows):
        ax.scatter(gmacs, params_m, s=80, color=viz.color(i), edgecolor="k", zorder=3)
        ax.annotate(name, (gmacs, params_m), xytext=label_offsets.get(name, (7, -3)), textcoords="offset points", fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("GMACs per 224 px image (log)"); ax.set_ylabel("parameters, M (log)")
    ax.set_xlim(0.15, 40); ax.set_ylim(3.5, 220); ax.set_title("Parameters and compute are different budgets"); ax.grid(alpha=.3, which="both")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: the two budgets do not go together. ConvNeXt-T has a few more parameters than
    ResNet-50 for about the same compute; VGG-16 has 5 times ResNet-50's parameters and 4 times its
    compute; MobileNetV3 and EfficientNet-B0 have about the same parameters but differ almost 2x in
    compute.

    Parameters decide **memory and download size**; MACs roughly decide **compute**; neither is
    **latency**, the time one image takes on a real machine. A depthwise conv does few multiply-adds
    for every number it reads from memory, so it is limited by memory speed rather than arithmetic
    (**memory-bound**), and MobileNet / EfficientNet run less fast on a GPU than their MAC counts
    suggest. Always time the model on the hardware you will deploy on (D2-04).
    """),

    # ------------------------------------------------------------------ 7
    md(r"""
    ## 7 · Assemble a MiniResNet and train it

    **Why.** Everything so far was one block at a time. A whole network is a stem, a few stages of
    blocks, and a head. You assemble one from your `BasicBlock`, then train it with D1-02's recipe and
    compare it with D1-02's plain network.

    ### Exercise 6 · `MiniResNet`

    **Goal.** A small ResNet for the 37 Pet breeds: stem, three stages of residual blocks, global
    average pooling, linear layer.

    **Why by hand.** This is the shape of every classification network of the course (ResNet-18 of
    section 6 follows the same plan with four stages). Building it shows where the image is halved and
    how the channel count is handed from block to block.

    **Input and output.** `MiniResNet(num_classes=37, widths=(32, 64, 128), blocks=(1, 1, 1))`.
    `widths[i]` is the number of channels of stage i, `blocks[i]` the number of blocks in it. Called
    on a batch (B, 3, H, W), it returns (B, num_classes) class scores, for any H and W. Example:
    `MiniResNet()` turns (2, 3, 128, 128) into (2, 37) and has 312,133 parameters.

    **Worked example: the shapes for a 128 x 128 image.**

    ```
    input                                                        (B, 3, 128, 128)
    stem      3x3 conv 3 -> 32 with stride 2, BN, ReLU            (B, 32, 64, 64)
              max-pool 2                                          (B, 32, 32, 32)
    stage 1   Basic(32, 32, stride=1)                             (B, 32, 32, 32)
    stage 2   Basic(32, 64, stride=2)                             (B, 64, 16, 16)
    stage 3   Basic(64, 128, stride=2)                            (B, 128, 8, 8)
    pool      nn.AdaptiveAvgPool2d(1): global average pooling     (B, 128, 1, 1)
    flatten   x.flatten(1): keep axis 0, join all the others      (B, 128)
    fc        nn.Linear(128, 37)                                  (B, 37)
    ```

    With `blocks=(2, 2, 2)`, each stage gets a second block that keeps its shape:
    `Basic(width, width, stride=1)`. `nn.AdaptiveAvgPool2d(1)` is global average pooling as a layer:
    whatever H and W come in, it outputs 1 x 1 per channel. That is why any image size works.

    **Plan.**
    1. (given) `self.stem`.
    2. The stages. Both loops are written: the outer one over the stages (`i = 0, 1, 2`), the inner
       one over the blocks of a stage (`j = 0, ..., blocks[i] - 1`). The variable `c` holds the
       channels coming into the next block: it starts at `widths[0]` (what the stem makes), and the
       given line `c = width` updates it after every block. You write:
       * a. the stride: 2 for the first block (`j == 0`) of stages 2 and 3 (`i > 0`), otherwise 1;
       * b. one block, `Basic(c, width, stride=stride)`, appended to `block_list`. Use `Basic`, not
         `BasicBlock`: `Basic` (made in the cell after exercise 4) is your BasicBlock if it passed,
         otherwise the reference;
       * c. after the inner loop: the stage, `nn.Sequential(*block_list)`, appended to `stage_list`.
         The `*` hands over the list's items as separate arguments: `nn.Sequential(*[a, b])` is
         `nn.Sequential(a, b)`.
    3. (given) `self.stages = nn.Sequential(*stage_list)`: the three stages, in order.
    4. The head: `self.pool = nn.AdaptiveAvgPool2d(1)` and `self.fc = nn.Linear(c, num_classes)`.
       After the loops `c` is the last stage's width, 128.
    5. `forward`: stem (given), stages, pool, `flatten(1)`, fc; one line each.

    Keep the names `stem`, `stages`, `pool`, `fc` and the nesting (an `nn.Sequential` of stages, each an
    `nn.Sequential` of blocks): the training cell below loads saved weights by these names
    (`stages.0.0.conv1.weight`, ...).

    **What the checker tests.** `MiniResNet(num_classes=37)` in eval mode on (2, 3, 128, 128) and
    (2, 3, 96, 96): the output is (2, 37) both times; at least 3 residual blocks (modules with a
    `.shortcut`); 0.1 to 3 million parameters; at least 2 blocks with a stride-2 conv. It prints the
    parameter and block counts.

    **Common mistakes.**

    | you wrote | the checker prints |
    |---|---|
    | stride 1 in every block | `stages 2 and 3 should start with a stride-2 block (downsample + double the channels)` |
    | `Basic(width, width, ...)` instead of `Basic(c, width, ...)` | `your code raised RuntimeError: ... expected input[2, 32, 32, 32] to have 64 channels, but got 32 channels instead`: stage 2's block expects 64 channels but receives the 32 that stage 1 made |
    | `nn.Sequential(block_list)`, without `*` | `your code raised TypeError: list is not a Module subclass` |
    | `self.stages = stage_list` (a plain list) and a loop in `forward` | `found 0 residual blocks (modules with a .shortcut) ...` and `5,701 parameters; expected roughly 0.1 M - 3 M ...`: PyTorch does not see modules inside a plain list |
    | `nn.Linear(widths[0], num_classes)` | `your code raised RuntimeError: addmm: mat1 and mat2 shapes cannot be multiplied (2x128 and 32x37)`: the fc receives 128 features |
    | no pooling | `... (2x8192 and 128x37)`: 128 channels x 8 x 8 positions |
    | no `flatten(1)` | `... (256x1 and 128x37)` |
    | `x.flatten()` without the 1 | `... (1x256 and 128x37)`: the batch axis was flattened too |
    | stride 2 in the first block of *every* stage, stage 1 included | `[PASS]` (313,221 parameters), but the stages run at 16, 8 and 4 px instead of 32, 16 and 8: compare the summary below with the worked example |
    """),
    api("nn.AdaptiveAvgPool2d", title="Functions you will need"),
    api("nn.Sequential", "nn.Linear", "Tensor.flatten", brief=True),
    exercise(
        stub=r'''
        class MiniResNet(nn.Module):
            """stem -> 3 stages of residual blocks -> global average pooling -> linear layer."""

            def __init__(self, num_classes=37, widths=(32, 64, 128), blocks=(1, 1, 1)):
                super().__init__()
                # step 1 (given): the stem. 3 -> widths[0] channels; the stride 2 and the max-pool make the image 4x smaller
                self.stem = nn.Sequential(
                    nn.Conv2d(3, widths[0], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(widths[0]),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                # step 2: the stages. Stage i has blocks[i] residual blocks with widths[i] output channels
                stage_list = []              # one nn.Sequential per stage
                c = widths[0]                # channels coming into the next block: the stem made widths[0]
                for i in range(len(widths)):     # i = 0, 1, 2 for stages 1, 2, 3
                    width = widths[i]
                    block_list = []          # the blocks of this stage
                    for j in range(blocks[i]):   # j = 0 is the first block of the stage
                        # step 2a: the stride: 2 for the first block (j == 0) of stages 2 and 3 (i > 0), else 1
                        # YOUR CODE: an if / else that sets stride (4 lines)
                        # step 2b: one residual block, c -> width channels, with that stride; keep it in block_list
                        # YOUR CODE: block = a Basic block from c to width channels with this stride
                        # YOUR CODE: append block to block_list
                        c = width            # the next block receives `width` channels
                    # step 2c: one module per stage that runs its blocks in order
                    # YOUR CODE: stage = an nn.Sequential of the blocks in block_list (use *block_list)
                    # YOUR CODE: append stage to stage_list
                # step 3 (given): the three stages, in order
                self.stages = nn.Sequential(*stage_list)
                # step 4: the head
                # YOUR CODE: self.pool = global average pooling to 1 x 1
                # YOUR CODE: self.fc = a linear layer from c features (the last width) to num_classes scores
                raise NotImplementedError    # delete this line when steps 2a-4 are written

            def forward(self, x):
                # x: (B, 3, H, W), for example (B, 3, 128, 128)
                # step 5: stem, stages, pool, flatten, fc
                x = self.stem(x)             # (B, 32, H/4, W/4)
                # YOUR CODE: x = the stages applied to x                   -> (B, 128, H/16, W/16)
                # YOUR CODE: x = the pooling applied to x                  -> (B, 128, 1, 1)
                # YOUR CODE: x = x flattened from axis 1 on: x.flatten(1)  -> (B, 128)
                # YOUR CODE: x = the linear layer applied to x             -> (B, num_classes)
                raise NotImplementedError    # delete this line when step 5 is written
                return x

        ok_mini = tests.check_mini_resnet(MiniResNet)
        ''',
        solution=r'''
        class MiniResNet(nn.Module):
            """stem -> 3 stages of residual blocks -> global average pooling -> linear layer."""

            def __init__(self, num_classes=37, widths=(32, 64, 128), blocks=(1, 1, 1)):
                super().__init__()
                # step 1 (given): the stem. 3 -> widths[0] channels; the stride 2 and the max-pool make the image 4x smaller
                self.stem = nn.Sequential(
                    nn.Conv2d(3, widths[0], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(widths[0]),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                # step 2: the stages. Stage i has blocks[i] residual blocks with widths[i] output channels
                stage_list = []              # one nn.Sequential per stage
                c = widths[0]                # channels coming into the next block: the stem made widths[0]
                for i in range(len(widths)):     # i = 0, 1, 2 for stages 1, 2, 3
                    width = widths[i]
                    block_list = []          # the blocks of this stage
                    for j in range(blocks[i]):   # j = 0 is the first block of the stage
                        # step 2a: the stride: 2 for the first block (j == 0) of stages 2 and 3 (i > 0), else 1
                        if i > 0 and j == 0:
                            stride = 2       # halves H and W (and the block doubles the channels)
                        else:
                            stride = 1       # keeps H and W
                        # step 2b: one residual block, c -> width channels, with that stride; keep it in block_list
                        block = Basic(c, width, stride=stride)
                        block_list.append(block)
                        c = width            # the next block receives `width` channels
                    # step 2c: one module per stage that runs its blocks in order
                    stage = nn.Sequential(*block_list)
                    stage_list.append(stage)
                # step 3 (given): the three stages, in order
                self.stages = nn.Sequential(*stage_list)
                # step 4: the head
                self.pool = nn.AdaptiveAvgPool2d(1)      # global average pooling: (B, c, h, w) -> (B, c, 1, 1)
                self.fc = nn.Linear(c, num_classes)      # c = widths[-1] after the loops

            def forward(self, x):
                # x: (B, 3, H, W), for example (B, 3, 128, 128)
                # step 5: stem, stages, pool, flatten, fc
                x = self.stem(x)             # (B, 32, H/4, W/4)
                x = self.stages(x)           # (B, 128, H/16, W/16)
                x = self.pool(x)             # (B, 128, 1, 1)
                x = x.flatten(1)             # (B, 128)
                x = self.fc(x)               # (B, num_classes)
                return x

        ok_mini = tests.check_mini_resnet(MiniResNet)
        '''),
    md(r"""
    The summary of your network (or the reference) for one 128 x 128 image; `depth=1` gives one row per
    layer of the stem and one per stage. Compare the output shapes with the worked example above, and
    the total with 312,133.
    """),
    code(r"""
    Mini = tests.use_or_reference(ok_mini, globals().get("MiniResNet"), day1.MiniResNet, "MiniResNet")
    print(day1.summarize(Mini(), (3, 128, 128), depth=1))
    """),
    md(r"""
    Now train it with **exactly** the D1-02 recipe and compare it with D1-02's `PetNet`, the plain
    convnet of that notebook. Same data (the Pet subset of D1-02: 2,000 training and 500 validation
    photos), same number of epochs, same augmentation (random changes to the training images), same
    optimiser (AdamW), same learning-rate schedule (warm-up, then cosine decay), same label smoothing:
    only the architecture differs.

    How the cell works:

    * On a CPU it trains at 64 x 64 pixels for 10 epochs, at most 2.5 minutes (on a GPU: 128 px,
      30 epochs).
    * `day1.load_or_train(model, name, train_fn)` loads the saved weights `data/checkpoints/<name>.pt`
      if the file exists. Otherwise it calls `train_fn`, here `train_mini`, which runs
      `day1.train_recipe`, the D1-02 recipe, and saves the best epoch. Delete the file to train again.
    * PetNet's training history is loaded from D1-02's checkpoint if you ran D1-02.

    What to look at: left, the loss of both networks on the training data (dashed) and the validation
    data (solid); right, the validation accuracy after each epoch.
    """),
    api("Tensor.to", "torchkit.load_checkpoint", "np.array", brief=True),
    code(r'''
    GPU = device.type == "cuda"
    if GPU:
        IMG, EPOCHS, MAX_MIN = 128, 30, None      # image size, epochs, time limit in minutes
    else:
        IMG, EPOCHS, MAX_MIN = 64, 10, 2.5        # same budget as D1-02
    data = day1.pet_tensors(144)                  # the Pet subset of D1-02, stored at 144 px
    torch.manual_seed(0)
    mini = Mini(num_classes=37).to(device)

    def train_mini(checkpoint_path):
        """Train `mini` with the D1-02 recipe; called only when no saved checkpoint exists."""
        return day1.train_recipe(mini, data, epochs=EPOCHS, lr=3e-3, weight_decay=0.05, label_smoothing=0.1,
                                 size=IMG, max_minutes=MAX_MIN, checkpoint=checkpoint_path, device=device)

    hist_mini = day1.load_or_train(mini, f"d1-03_miniresnet_{IMG}px", train_mini)

    ck_petnet = torchkit.CHECKPOINTS / f"d1-02_recipe_{IMG}px.pt"     # D1-02's network, trained with the recipe
    if ck_petnet.exists():
        hist_petnet = torchkit.load_checkpoint(day1.PetNet(dropout=0.2), ck_petnet)["history"]
    else:
        hist_petnet = None

    # Draws the loss (left) and the validation accuracy (right) of both networks, epoch by epoch
    curves = [("MiniResNet", hist_mini, viz.PALETTE[0])]
    if hist_petnet is not None:
        curves.append(("PetNet (D1-02 recipe)", hist_petnet, viz.PALETTE[1]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    for name, h, c in curves:
        ep = np.arange(1, len(h["val_loss"]) + 1)
        a1.plot(ep, h["train_loss"], "--", color=c, label=f"{name}: train"); a1.plot(ep, h["val_loss"], "-o", ms=3, color=c, label=f"{name}: val")
        a2.plot(ep, 100 * np.array(h["val_metric"]), "-o", ms=3, color=c, label=name)
    a1.set_title("loss"); a2.set_title("validation accuracy (%)")
    for a in (a1, a2):
        a.set_xlabel("epoch"); a.legend(fontsize=8); a.grid(alpha=.3)
    plt.tight_layout(); plt.show()

    for name, h, _ in curves:
        print(f"{name:24s} best val acc {max(h['val_metric']):.3f}")
    if hist_petnet is None:
        print("(run D1-02 first to overlay the PetNet curve)")
    print(f"params: MiniResNet {torchkit.count_parameters(mini):,}  PetNet {torchkit.count_parameters(day1.PetNet()):,}")
    print(f"MACs per {IMG} px image: MiniResNet {torchkit.profile(mini, (3, IMG, IMG))['macs'] / 1e6:.0f} M  "
          f"PetNet {torchkit.profile(day1.PetNet(), (3, IMG, IMG))['macs'] / 1e6:.0f} M")
    '''),
    md(r"""
    With 10 CPU epochs at 64 px both networks are still early in training and reach about the same
    best validation accuracy: 13.6 % for the MiniResNet and 13.2 % for PetNet with the saved
    checkpoints (guessing would give 1/37 = 2.7 %). A difference this small can go either way when you
    train again. The MiniResNet gets there with fewer parameters (312,133 against 435,397) and about
    half the multiply-adds (13 M against 24 M per 64 px image). Training from scratch on 2,000 images
    stays hard for *any* architecture, which is exactly why the next notebook reuses a network that
    has already seen 1.3 million ImageNet photos.

    ## 8 · Summary

    | idea | first in | why it stayed |
    |---|---|---|
    | conv + pool + fc, trained with backprop | LeNet 1998 | weight sharing (one kernel for every position) and translation equivariance (a shifted input gives a shifted output), D1-01 |
    | ReLU, dropout, GPU training, augmentation | AlexNet 2012 | trains deep nets fast |
    | small 3x3 kernels, stacked | VGG 2014 | same receptive field, fewer params, more non-linearity |
    | 1x1 bottlenecks, multi-branch, GAP | GoogLeNet 2014 | cheap width, no giant fc head |
    | residual connections + BN | ResNet 2015 | gradients reach every layer: depth works |
    | feature concatenation | DenseNet 2017 | feature reuse, few params |
    | depthwise-separable convs, SE | MobileNet / SENet 2017-18 | 8-9x cheaper convs, channel attention |
    | compound scaling | EfficientNet 2019 | scale d, w, r together |
    | ViT-era design + training recipe | ConvNeXt 2022 | a convnet that matches transformers |

    In code: a block is an `nn.Module` with its layers created in `__init__` and applied in `forward`;
    parallel branches are joined with `torch.cat(..., dim=1)`; a residual block adds
    `self.shortcut(x)` to the main path before the last ReLU; a depthwise conv is an `nn.Conv2d` with
    `groups=c_in`; a network is a stem, stages of blocks (`nn.Sequential`) and a head (global average
    pooling, `flatten(1)`, `nn.Linear`).

    ### Self-check
    1. An Inception block gets 256 channels and has branches (128, 128->192, 32->96, pool->64). Output channels?
    2. Why does a 1x1 convolution before a 5x5 save compute? Give the ratio for 192 -> 16 -> 32 vs 192 -> 32.
    3. When does a residual block need a projection shortcut?
    4. In the gradient-flow plot, why does the plain net without BN fail at depth 60, and what does `+ x` change in the backward pass?
    5. Residual vs dense: which one keeps the channel count constant? Which one uses more memory?
    6. A depthwise-separable 3x3 conv 256 -> 256: how many weights, vs a standard 3x3 conv?
    7. Why does a layer kept in a plain Python list not get trained, and what do you use instead?

    ### Go deeper
    * Y. LeCun et al., *Gradient-based learning applied to document recognition* (1998).
    * A. Krizhevsky, I. Sutskever, G. Hinton, *ImageNet classification with deep convolutional neural networks* (2012).
    * K. Simonyan & A. Zisserman, *Very deep convolutional networks for large-scale image recognition* (VGG, 2014).
    * C. Szegedy et al., *Going deeper with convolutions* (GoogLeNet, 2014).
    * K. He et al., *Deep residual learning for image recognition* (2015); *Identity mappings in deep residual networks* (2016).
    * G. Huang et al., *Densely connected convolutional networks* (DenseNet, 2017).
    * A. Howard et al., *MobileNets* (2017); J. Hu et al., *Squeeze-and-excitation networks* (2018).
    * M. Tan & Q. Le, *EfficientNet: rethinking model scaling for convolutional neural networks* (2019).
    * Z. Liu et al., *A ConvNet for the 2020s* (ConvNeXt, 2022).
    """),
    md(r"""
    ## Words from this notebook

    **Describing a network**

    | word | meaning |
    |---|---|
    | architecture | the design of a network: which layers, in which order, how many channels, how connected |
    | layer | one operation, usually with parameters: a conv, a BatchNorm, a linear layer |
    | block | a small group of layers that is repeated; architectures are designed block by block |
    | stage | a run of blocks at the same height and width; the next stage halves them and doubles the channels |
    | stem | the first layers, which shrink the image quickly |
    | head | the last layers, which turn features into class scores (here global average pooling + a linear layer) |
    | depth, width | the number of layers with weights; the number of channels |
    | fully-connected (fc) layer | a linear layer: every output connected to every input |
    | top-level module | an attribute created by the model's own `__init__`: one row of a `depth=0` summary |
    | backbone | the part of a network that computes the features; detectors (Day 2) put their own heads on a ResNet backbone |

    **Measuring a network**

    | word | meaning |
    |---|---|
    | ImageNet-1k | the standard benchmark: about 1.3 million training photos in 1 000 classes |
    | top-1 / top-5 accuracy | how often the highest score / one of the five highest scores is the right class |
    | parameter count | how many numbers the network learns; decides memory and file size |
    | multiply-adds (MACs), GMACs | multiplications (each with an addition) per image; a GMAC is a billion. A conv's MACs = its weights x output positions |
    | FLOPs | floating-point operations, about 2 x MACs |
    | latency | the time one image takes on a real machine |
    | memory-bound | limited by how fast numbers come from memory rather than by arithmetic |
    | forward hook | a function PyTorch calls every time a given layer runs; summaries use it to record shapes |
    | training recipe | everything about training except the architecture: epochs, optimiser, schedule, augmentation, label smoothing |

    **Blocks and connections**

    | word | meaning |
    |---|---|
    | receptive field | the input patch an output pixel depends on (D1-01) |
    | branch | one of several parallel chains of layers fed the same input |
    | concatenation along channels | stacking tensors of equal height and width along axis 1; channel counts add up (`torch.cat(..., dim=1)`) |
    | 1x1 convolution | a per-pixel weighted sum of the channels: changes the channel count, not H or W |
    | bottleneck, reduction | a 1x1 conv that cuts the channels before an expensive conv |
    | auxiliary classifier | an extra head halfway up a network, to push gradient into the lower layers (GoogLeNet) |
    | residual, main path | F(x): what a residual block's layers compute |
    | shortcut, skip connection | the path that carries a block's input past its layers to the addition |
    | identity shortcut | the shortcut that passes x unchanged (`nn.Identity()`) |
    | projection shortcut | a 1x1 conv with the block's stride, plus BN, used when the block changes the shape |
    | bottleneck block | ResNet-50's block: 1x1 reduce, 3x3, 1x1 expand x4 |
    | degradation problem | deeper plain networks train worse, even on the training data |
    | BatchNorm scale and shift | the learned `weight` (γ) and `bias` (β) per channel applied after normalising |
    | dense connection | each layer receives the concatenation of all earlier outputs (DenseNet) |
    | growth | the number of new channels each dense layer adds |

    **Gradients**

    | word | meaning |
    |---|---|
    | initialisation | the random starting values of the weights |
    | He initialisation | a rule for those starting values that keeps the signal size constant through ReLU layers |
    | norm, ‖·‖ | the size of a tensor as one number: square root of the sum of its squared entries |
    | vanishing gradient | the gradient shrinks layer by layer on its way back, so the first layers cannot learn |
    | exploding gradient | the gradient grows layer by layer on its way back |

    **Efficient blocks**

    | word | meaning |
    |---|---|
    | depthwise conv | one filter per input channel, applied to that channel alone (`groups=c_in`) |
    | pointwise conv | a 1x1 conv that mixes the channels |
    | depthwise-separable conv | a depthwise conv followed by a pointwise conv: about 1/C_out + 1/9 of the cost of a standard conv |
    | groups | `nn.Conv2d` argument: split the channels into groups and connect each output group only to its input group |
    | squeeze-and-excitation (SE) | global average pooling, a tiny MLP and a sigmoid give one weight per channel; the channels are multiplied by them |
    | channel attention | turning channels up or down depending on the input |
    | MLP | a small stack of linear layers with activations between them |
    | sigmoid | a function that squashes any number into (0, 1) |
    | inverted residual | 1x1 expand, depthwise conv, 1x1 project back, with the shortcut between the thin ends |
    | compound scaling | growing depth, width and resolution together with fixed ratios (EfficientNet) |
    | vision transformer (ViT) | a network that cuts the image into patches and uses attention, a layer that lets every patch look at every other patch, instead of convolutions (Day 3) |
    | patchify stem | a stem that is one conv with kernel = stride, cutting the image into non-overlapping patches |
    | GELU | a smooth version of ReLU |
    | LayerNorm | normalises the channels of each position of each image, instead of each channel over the batch |

    **PyTorch**

    | word | meaning |
    |---|---|
    | `nn.Module` | the base class of every block and model: layers created in `__init__`, applied in `forward`; `block(x)` runs `forward` |
    | `super().__init__()` | the first line of every module's `__init__`: sets up the bookkeeping that registers the layers |
    | `nn.Sequential` | a module that applies its layers one after the other; layers are named 0, 1, 2, ... |
    | `*list` in a call | hands over the list's items as separate arguments: `nn.Sequential(*[a, b])` is `nn.Sequential(a, b)` |
    | `named_parameters()` | the name and tensor of every parameter of a module |
    | reference implementation | the course's own version of an exercise, used when yours has not passed yet |
    """),
]
