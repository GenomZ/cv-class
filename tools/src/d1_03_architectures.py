from nb import md, code, exercise, header

NAME = "D1-03-architectures"

CELLS = [
    header(
        "D1-03 · Architecture zoo",
        steps=[
            "Timeline: one idea per architecture, 1998-2022; accuracy vs. size",
            "VGG: 3x3 convolutions everywhere",
            "GoogLeNet / Inception: parallel branches and 1x1 bottlenecks",
            "ResNet: residual and bottleneck blocks; why very deep nets train; residual vs. dense",
            "Efficient blocks: depthwise-separable convolutions, squeeze-and-excitation, EfficientNet, ConvNeXt",
            "Reading a model: summaries, parameters and multiply-adds of torchvision models",
            "Assemble a MiniResNet and train it on the Pet subset",
        ],
        topics=[
            "The handful of ideas that survived from 2012 to today and the problem each one solved",
            "Implement Inception, residual (basic + bottleneck) and depthwise-separable blocks in PyTorch",
            "Why residual connections fix the optimisation of deep networks (gradient flow at initialisation)",
            "Residual addition vs. dense concatenation",
            "Parameter / compute trade-offs and how to read a model summary",
        ],
        minutes=90,
        prereq="D1-01 (convolution, 1x1 conv, pooling, receptive field) and D1-02 (training recipe).",
    ),
    md(r"""
    ## 0 · Setup
    """),
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
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(ax, x0, y0, x1, y1, **kw):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.2, **kw))
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · Twenty-five years in one picture

    Every architecture below is remembered for **one idea** that the next ones kept. The
    numbers are the published ImageNet-1k top-1 accuracies (single 224 px crop unless noted)
    and parameter counts, hard-coded in `day1.ARCHITECTURES`.
    """),
    code(r"""
    A = day1.ARCHITECTURES
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.axhline(0, color="k", lw=1)
    timeline = [a for a in A if "recipe" not in a[0] and a[0] not in ("ResNet-152", "EfficientNet-B4")]
    years = [a[1] for a in timeline]
    xs = np.linspace(0, 1, len(timeline))
    for i, (name, year, top1, params, macs, idea) in enumerate(timeline):
        up = 1 if i % 2 == 0 else -1
        ax.plot([xs[i], xs[i]], [0, 0.55 * up], color="gray", lw=0.8)
        ax.plot(xs[i], 0, "o", color=viz.color(i), ms=9)
        ax.text(xs[i], 0.62 * up, f"{year}  {name}", ha="center", va="bottom" if up > 0 else "top", fontsize=10, weight="bold")
        wrapped = idea if len(idea) < 34 else idea[:idea.rfind(" ", 0, 34)] + "\n" + idea[idea.rfind(" ", 0, 34) + 1:]
        ax.text(xs[i], (0.62 + 0.13) * up if up > 0 else (0.62 + 0.13) * up, wrapped, ha="center",
                va="bottom" if up > 0 else "top", fontsize=8, color="#333")
    ax.set_xlim(-0.09, 1.09); ax.set_ylim(-1.45, 1.45); ax.axis("off")
    ax.set_title("ConvNet architectures: the idea each one contributed", fontsize=12)
    plt.tight_layout(); plt.show()
    """),
    code(r"""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    offsets = {"AlexNet": (8, -12), "VGG-16": (-10, 8), "GoogLeNet": (8, -10), "ResNet-50": (-14, -16),
               "ResNet-152": (8, -2), "DenseNet-121": (-62, 8), "MobileNetV2": (8, -10), "EfficientNet-B0": (-30, 10),
               "EfficientNet-B4": (8, 4), "ViT-B/16": (8, -4), "ConvNeXt-T": (8, 6), "ResNet-50 (2021 recipe)": (8, -12)}
    for i, (name, year, top1, params, macs, idea) in enumerate(A):
        if top1 is None:
            continue
        ax.scatter(params, top1, s=30 + 25 * macs, color=viz.color(year - 2012), alpha=0.75, edgecolor="k", zorder=3)
        label = name if "recipe" in name else f"{name} ({year})"
        ax.annotate(label, (params, top1), xytext=offsets.get(name, (6, 4)), textcoords="offset points", fontsize=8)
    r50 = [a for a in A if a[0].startswith("ResNet-50")]
    ax.annotate("", xy=(r50[1][3], r50[1][2]), xytext=(r50[0][3], r50[0][2]),
                arrowprops=dict(arrowstyle="->", color=viz.PALETTE[3], lw=1.5, connectionstyle="arc3,rad=-0.3"))
    ax.text(14.5, 78.2, "same network,\nbetter training: +4.8", color=viz.PALETTE[3], fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("parameters (millions, log scale)"); ax.set_ylabel("ImageNet top-1 (%)")
    ax.set_title("Accuracy vs. size (marker area ~ compute per image)"); ax.grid(alpha=.3, which="both")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Three things to read off the scatter:

    * **Size is not accuracy.** VGG-16 has 26x the parameters of EfficientNet-B0 and is 6 points worse.
      Most of VGG's parameters sit in its fully-connected head; later networks replaced that head by
      *global average pooling* (average each channel over all positions, then one small linear layer).
    * **Architecture is not everything.** The same ResNet-50 gains ~5 points from a 2021 training
      recipe (longer schedule, AdamW / LAMB, RandAugment, Mixup / CutMix, label smoothing - the D1-02
      toolbox). Compare architectures only under the same recipe.
    * **The last big jumps came from design + recipe together** (EfficientNet, ConvNeXt), and ViTs
      only win with lots of data or pretraining (Day 3).
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · VGG (2014): 3x3 everywhere

    VGG replaced AlexNet's 11x11 and 5x5 kernels with stacks of **3x3 convolutions**. Two 3x3
    layers see 5x5, three see 7x7 (D1-01), but with fewer parameters and more non-linearities:
    for $C$ channels in and out, one 7x7 layer has $49C^2$ weights, three 3x3 layers $27C^2$. The
    blueprint - blocks of 3x3 convs, then 2x2 max-pool, channels doubling as resolution halves -
    is still the backbone of most convnets. Below, VGG-16 drawn to scale: box height ~ spatial
    size, width ~ channels.
    """),
    code(r"""
    vgg_cfg = [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]
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
    ax.text(x - 1.0, 55, "102 M of the 138 M\nparameters are here", ha="center", fontsize=8, color=viz.PALETTE[3])
    ax.set_xlim(-0.3, x + 0.3); ax.set_ylim(-140, 150); ax.axis("off"); ax.set_title("VGG-16 to scale (gray = 2x2 max-pool)")
    plt.tight_layout(); plt.show()

    vgg = torchvision.models.vgg16(weights=None)
    fc_params = sum(p.numel() for p in vgg.classifier.parameters())
    print(f"VGG-16: {torchkit.count_parameters(vgg) / 1e6:.1f} M params, of which the fc head: {fc_params / 1e6:.1f} M")
    C = 256
    print(f"one 7x7 conv {C}->{C}: {49 * C * C:,} weights | three 3x3 convs: {27 * C * C:,} weights (same receptive field)")
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · GoogLeNet / Inception (2014): go wide, cheaply

    Which kernel size is right at a given layer: 1x1, 3x3 or 5x5? The Inception block does not
    choose: it runs **four branches in parallel** and **concatenates** their outputs along the
    channel axis. To keep that affordable, the expensive 3x3 and 5x5 branches first shrink the
    channel count with a **1x1 convolution** (the "bottleneck"). GoogLeNet stacked nine of these
    blocks, used global average pooling instead of a big fc head (7 M parameters vs VGG's 138 M),
    and attached two **auxiliary classifiers** to intermediate layers (loss weight 0.3) to push
    gradient into the lower layers - a fix for the gradient-flow problem that residual connections
    later solved much better (section 4).

    Block "3a" of GoogLeNet, with its channel counts:
    """),
    code(r"""
    fig, ax = plt.subplots(figsize=(12, 5)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6.2)
    box(ax, 6, 5.7, 3.0, 0.5, "input  192 x 28 x 28", fc="#e5e5e5", fs=9)
    cols = [(1.5, ["1x1 conv, 64"], "64"), (4.5, ["1x1 conv, 96", "3x3 conv, 128"], "128"),
            (7.5, ["1x1 conv, 16", "5x5 conv, 32"], "32"), (10.5, ["3x3 max-pool", "1x1 conv, 32"], "32")]
    for x, layers, cout in cols:
        arrow(ax, 6, 5.42, x, 4.75)
        for k, name in enumerate(layers):
            fc = "#cde8cf" if name.startswith("1x1") else ("#e5e5e5" if "pool" in name else "#f8d0b8")
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

    naive = 192 * 128 * 9 + 192 * 32 * 25
    reduced = 192 * 96 + 96 * 128 * 9 + 192 * 16 + 16 * 32 * 25
    print(f"3x3 + 5x5 branches without 1x1 reductions: {naive:,} weights; with them: {reduced:,} ({naive / reduced:.1f}x fewer)")
    """),
    md(r"""
    ### Exercise 1 · Inception output channels (warm-up)

    `inception_out_channels(c_in, f1, f3r, f3, f5r, f5, fpool)` -> number of output channels. (This is
    the 2-line version of a question from the original course.)
    """),
    exercise(
        stub=r"""
        def inception_out_channels(c_in, f1, f3r, f3, f5r, f5, fpool):
            # YOUR CODE HERE
            raise NotImplementedError

        ok_shapes = tests.check_inception_shapes(inception_out_channels)
        """,
        solution=r"""
        def inception_out_channels(c_in, f1, f3r, f3, f5r, f5, fpool):
            return f1 + f3 + f5 + fpool          # c_in, f3r and f5r do not matter: they are internal

        ok_shapes = tests.check_inception_shapes(inception_out_channels)
        """),
    md(r"""
    ### Exercise 2 · `InceptionBlock`

    Four branches, each conv **with bias** and followed by `nn.ReLU()`, spatial size preserved:

    * `b1`: 1x1 conv `c_in -> f1`
    * `b2`: 1x1 conv `c_in -> f3r`, ReLU, 3x3 conv `f3r -> f3` (padding 1), ReLU
    * `b3`: 1x1 conv `c_in -> f5r`, ReLU, 5x5 conv `f5r -> f5` (padding 2), ReLU
    * `b4`: 3x3 max-pool with stride 1, padding 1, then 1x1 conv `c_in -> fpool`, ReLU

    `forward` concatenates the four outputs along dimension 1 (channels).
    """),
    exercise(
        stub=r"""
        class InceptionBlock(nn.Module):
            def __init__(self, c_in, f1, f3r, f3, f5r, f5, fpool):
                super().__init__()
                self.b1 = nn.Sequential(nn.Conv2d(c_in, f1, 1), nn.ReLU(inplace=True))
                # YOUR CODE HERE: self.b2, self.b3, self.b4
                raise NotImplementedError

            def forward(self, x):
                # YOUR CODE HERE
                raise NotImplementedError

        ok_inception = tests.check_inception_block(InceptionBlock)
        """,
        solution=r"""
        class InceptionBlock(nn.Module):
            def __init__(self, c_in, f1, f3r, f3, f5r, f5, fpool):
                super().__init__()
                self.b1 = nn.Sequential(nn.Conv2d(c_in, f1, 1), nn.ReLU(inplace=True))
                self.b2 = nn.Sequential(nn.Conv2d(c_in, f3r, 1), nn.ReLU(inplace=True),
                                        nn.Conv2d(f3r, f3, 3, padding=1), nn.ReLU(inplace=True))
                self.b3 = nn.Sequential(nn.Conv2d(c_in, f5r, 1), nn.ReLU(inplace=True),
                                        nn.Conv2d(f5r, f5, 5, padding=2), nn.ReLU(inplace=True))
                self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                        nn.Conv2d(c_in, fpool, 1), nn.ReLU(inplace=True))

            def forward(self, x):
                return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

        ok_inception = tests.check_inception_block(InceptionBlock)
        """),
    code(r"""
    Inc = tests.use_or_reference(ok_inception, globals().get("InceptionBlock"), day1.InceptionBlock, "InceptionBlock")
    blk = Inc(192, 64, 96, 128, 16, 32, 32)
    print(day1.summarize(blk, (192, 28, 28), depth=0))
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · ResNet (2015): learn the residual

    In 2015, stacking more layers made networks *worse* - on the **training** set. A 56-layer plain
    net had higher training error than a 20-layer one, which is not overfitting but an
    optimisation failure. He et al.'s fix: let each block learn a correction to its input,

    $$y = \operatorname{ReLU}\big(F(x) + x\big),$$

    so that "do nothing" (identity) is the easy default ($F = 0$) and the gradient has a direct path
    back through every `+ x`: $\partial y / \partial x = I + \partial F / \partial x$. When the
    block changes the shape (stride 2 or more channels), the shortcut becomes a **projection**:
    a 1x1 conv with the same stride, plus BN.

    | block | layers in $F$ | used in |
    |---|---|---|
    | basic | 3x3 (stride) - BN - ReLU - 3x3 - BN | ResNet-18 / 34 |
    | bottleneck | 1x1 reduce - BN - ReLU - 3x3 (stride) - BN - ReLU - 1x1 expand (x4) - BN | ResNet-50 / 101 / 152 |
    """),
    code(r"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, title, layers in [
            (axes[0], "basic block (64 ch)", ["3x3 conv, 64", "BN, ReLU", "3x3 conv, 64", "BN"]),
            (axes[1], "bottleneck block (256 ch in / out)", ["1x1 conv, 64", "BN, ReLU", "3x3 conv, 64", "BN, ReLU", "1x1 conv, 256", "BN"])]:
        ax.axis("off"); ax.set_xlim(0, 6); ax.set_ylim(-0.4, 8.2); ax.set_title(title)
        n = len(layers); ys = np.linspace(6.8, 1.9, n)
        box(ax, 2.5, 7.6, 2.0, 0.5, "x", fc="#e5e5e5", fs=10)
        prev = 7.35
        for y, name in zip(ys, layers):
            fc = "#cde8cf" if "1x1" in name else ("#f8d0b8" if "3x3" in name else "#ffffff")
            arrow(ax, 2.5, prev, 2.5, y + 0.25); box(ax, 2.5, y, 2.2, 0.48, name, fc=fc, fs=9); prev = y - 0.25
        box(ax, 2.5, 1.0, 0.6, 0.5, "+", fc="#f6e7a6", fs=13); arrow(ax, 2.5, prev, 2.5, 1.25)
        ax.plot([3.5, 4.3, 4.3], [7.6, 7.6, 1.0], color=viz.PALETTE[3], lw=2)
        ax.annotate("", xy=(2.82, 1.0), xytext=(4.3, 1.0), arrowprops=dict(arrowstyle="->", lw=2, color=viz.PALETTE[3]))
        ax.text(4.45, 4.3, "identity\nshortcut", color=viz.PALETTE[3], fontsize=10)
        arrow(ax, 2.5, 0.75, 2.5, 0.2); ax.text(2.5, -0.15, "ReLU -> y", ha="center", fontsize=10)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    In both blocks the green identity shortcut skips the whole main path and is added at `+`, before
    the last ReLU. In the bottleneck the expensive 3x3 conv works on only 64 channels, between a
    cheap 1x1 that reduces 256 -> 64 and another that expands back to 256.

    ### Exercise 3 · `BasicBlock`

    The main path is given. Add the shortcut: `nn.Identity()` when the shape does not change,
    otherwise a projection `nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))`.
    In `forward`, add the shortcut **before** the final ReLU.

    The checker also switches the residual branch off (sets the last BN's weight to 0, so
    $F(x) = 0$) and verifies the block then passes its input through.
    """),
    exercise(
        stub=r"""
        class BasicBlock(nn.Module):
            def __init__(self, c_in, c_out, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(c_out)
                self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(c_out)
                # YOUR CODE HERE: self.shortcut = ...
                raise NotImplementedError

            def forward(self, x):
                # YOUR CODE HERE
                raise NotImplementedError

        ok_basic = tests.check_residual_block(BasicBlock)
        """,
        solution=r"""
        class BasicBlock(nn.Module):
            def __init__(self, c_in, c_out, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(c_out)
                self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(c_out)
                if stride != 1 or c_in != c_out:
                    self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
                else:
                    self.shortcut = nn.Identity()

            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                return F.relu(out + self.shortcut(x))

        ok_basic = tests.check_residual_block(BasicBlock)
        """),
    md(r"""
    ### Exercise 4 · `Bottleneck`

    `Bottleneck(c_in, c_mid, stride)` outputs `4 * c_mid` channels: 1x1 `c_in -> c_mid` (BN, ReLU),
    3x3 `c_mid -> c_mid` with the stride (BN, ReLU), 1x1 `c_mid -> 4*c_mid` (BN, **no** ReLU), plus
    the shortcut (projection when `stride != 1` or `c_in != 4*c_mid`), then ReLU. Use the attribute
    names `conv1, bn1, conv2, bn2, conv3, bn3, shortcut`; convs have `bias=False`. The checker
    compares the parameter count with torchvision's own `Bottleneck`.
    """),
    exercise(
        stub=r"""
        class Bottleneck(nn.Module):
            expansion = 4

            def __init__(self, c_in, c_mid, stride=1):
                super().__init__()
                c_out = c_mid * self.expansion
                # YOUR CODE HERE
                raise NotImplementedError

            def forward(self, x):
                # YOUR CODE HERE
                raise NotImplementedError

        ok_bottleneck = tests.check_bottleneck(Bottleneck)
        """,
        solution=r"""
        class Bottleneck(nn.Module):
            expansion = 4

            def __init__(self, c_in, c_mid, stride=1):
                super().__init__()
                c_out = c_mid * self.expansion
                self.conv1 = nn.Conv2d(c_in, c_mid, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(c_mid)
                self.conv2 = nn.Conv2d(c_mid, c_mid, 3, stride, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(c_mid)
                self.conv3 = nn.Conv2d(c_mid, c_out, 1, bias=False)
                self.bn3 = nn.BatchNorm2d(c_out)
                if stride != 1 or c_in != c_out:
                    self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
                else:
                    self.shortcut = nn.Identity()

            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = F.relu(self.bn2(self.conv2(out)))
                out = self.bn3(self.conv3(out))
                return F.relu(out + self.shortcut(x))

        ok_bottleneck = tests.check_bottleneck(Bottleneck)
        """),
    code(r"""
    Basic = tests.use_or_reference(ok_basic, globals().get("BasicBlock"), day1.BasicBlock, "BasicBlock")
    Bott = tests.use_or_reference(ok_bottleneck, globals().get("Bottleneck"), day1.Bottleneck, "Bottleneck")
    for name, blk in [("basic 256->256", Basic(256, 256)), ("bottleneck 256->(64)->256", Bott(256, 64))]:
        prof = torchkit.profile(blk, (256, 14, 14))
        print(f"{name:28s} params {prof['params']:>9,d}   MACs @14x14 {prof['macs'] / 1e6:6.1f} M")
    print("-> the bottleneck is ~17x cheaper at 256 channels: that is how ResNet-50 affords 50 layers")
    """),
    md(r"""
    ### Why it trains: gradient flow at initialisation

    Build plain and residual networks of increasing depth (3x3 convs, 16 channels, PyTorch's
    default initialisation), push one random batch through, and measure the norm of the gradient
    that reaches the **first** layer. With BN and without.
    """),
    code(r"""
    depths = [4, 10, 20, 30, 40, 60]
    x_g = torch.randn(16, 3, 16, 16); y_g = torch.randint(0, 10, (16,))
    res = {}
    for bn in (False, True):
        for residual in (False, True):
            vals = []
            for d in depths:
                g = []
                for seed in range(3):
                    torch.manual_seed(seed)
                    g.append(day1.first_layer_grad_norm(day1.make_deep_net(d, residual, bn=bn), x_g, y_g))
                vals.append(np.mean(g))
            res[(bn, residual)] = vals

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, bn in zip(axes, (False, True)):
        for residual, c in ((False, viz.PALETTE[3]), (True, viz.PALETTE[0])):
            ax.semilogy(depths, np.maximum(res[(bn, residual)], 1e-30), "-o", color=c, label="residual" if residual else "plain")
        ax.set_title(f"{'with' if bn else 'without'} BatchNorm"); ax.set_xlabel("depth (weight layers)")
        ax.grid(alpha=.3, which="both"); ax.legend()
    axes[0].set_ylabel("||dL/dW|| of the first conv (log)")
    plt.suptitle("Gradient reaching the first layer at initialisation", y=1.02); plt.tight_layout(); plt.show()
    for k, v in res.items():
        print(f"BN={k[0]!s:5}  {'residual' if k[1] else 'plain   '}  " + "  ".join(f"{d}:{g:.1e}" for d, g in zip(depths, v)))
    """),
    md(r"""
    Without BN the plain net's gradient shrinks *exponentially* with depth (each layer multiplies
    it by a factor < 1); at 60 layers the first layer receives essentially nothing and cannot
    learn. The residual net's gradient changes only slowly with depth: the `+ x` paths carry it
    back unchanged. Adding BatchNorm to the plain net does not fix it - the gradient now *explodes*
    with depth (each BN-ReLU-conv layer amplifies it a little; Yang et al. 2019 analyse why), so
    deep plain nets with BN still train badly (the 2015 paper's "degradation problem"). Residual +
    BN stays flat at every depth: 152 (and 1 000) layers train fine. Every modern
    architecture - ConvNeXt, ViT, U-Net decoders, LLMs - has residual connections.
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

    DenseNet (2017) connects each layer to **all** previous ones by *concatenation*: layer $l$ gets
    $[x_0, x_1, \dots, x_{l-1}]$ and adds `growth` new channels. Features are reused instead of
    re-learned, which makes DenseNets very parameter-efficient, at the cost of memory (the channel
    count grows inside a block). The original course labelled a dense block "residual"; here they
    are side by side so you can tell them apart by the **shapes**:
    """),
    code(r"""
    x = torch.randn(1, 64, 16, 16)
    res_chs, dense_chs = [64], [64]
    r, d = x, x
    for _ in range(4):
        r = day1.BasicBlock(64, 64)(r); res_chs.append(r.shape[1])
        d = day1.DenseLayer(d.shape[1], growth=32)(d); dense_chs.append(d.shape[1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for ax, title, op, chs in [(axes[0], "residual: y = F(x) + x", "+", res_chs), (axes[1], "dense: y = concat(x, F(x))", "cat", dense_chs)]:
        ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.set_title(title)
        for i, c in enumerate(chs):
            w = 0.25 + 1.3 * c / 192
            ax.add_patch(Rectangle((0.3 + i * 2.0, 2 - w / 2 * 1.2), 0.5, w * 1.2, fc=viz.color(i), ec="k"))
            ax.text(0.55 + i * 2.0, 0.35, f"{c} ch", ha="center", fontsize=8)
            if i:
                ax.text(-0.45 + i * 2.0, 2.0, op, ha="center", va="center", fontsize=10, weight="bold")
                srcs = [i - 1] if op == "+" else list(range(i))          # residual: previous only; dense: ALL previous
                for j in srcs:
                    ax.annotate("", xy=(0.55 + i * 2.0, 2.0 + w * 0.6), xytext=(0.55 + j * 2.0, 2.0 + 0.6),
                                arrowprops=dict(arrowstyle="->", lw=0.8, color="gray", connectionstyle="arc3,rad=-0.35"))
    axes[2].plot(range(5), res_chs, "-o", label="4 residual blocks"); axes[2].plot(range(5), dense_chs, "-o", label="4 dense layers (growth 32)")
    axes[2].set_xlabel("layer"); axes[2].set_ylabel("channels of the output"); axes[2].legend(); axes[2].grid(alpha=.3)
    axes[2].set_xticks(range(5))
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    The residual chain keeps 64 channels (every block adds into the same tensor); each dense layer
    appends 32 new channels, so after 4 layers the tensor is 64 + 4 x 32 = 192 channels wide, and
    every later layer reads all of them.
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · Efficiency: depthwise-separable convolutions, SE, EfficientNet, ConvNeXt

    Residual connections made depth trainable; the remaining question is what each block *costs* in
    parameters and compute. A standard 3x3 conv mixes **space and channels at once**: $C_{\text{in}} \cdot C_{\text{out}} \cdot 9$
    weights. MobileNet (2017) factorises it into

    1. a **depthwise** 3x3 conv: one 3x3 filter *per input channel* (`groups=C_in`), space only: $9\,C_{\text{in}}$ weights;
    2. a **pointwise** 1x1 conv: channels only: $C_{\text{in}} C_{\text{out}}$ weights.

    The cost ratio is $\frac{1}{C_{\text{out}}} + \frac{1}{9}$: about 8-9x fewer parameters and
    multiply-adds for typical widths, for a small accuracy loss.
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

    `depthwise` (3x3, `groups=c_in`, stride, padding 1, no bias) - `bn1` - ReLU - `pointwise`
    (1x1, no bias) - `bn2` - ReLU. The checker counts parameters: $9\,C_{\text{in}} + C_{\text{in}} C_{\text{out}}$
    conv weights plus 2 per channel for each BatchNorm (the lab's numbers with "count BatchNorm" on).
    In the lab, `depthwise` is the yellow segment and `pointwise` the orange one.
    """),
    exercise(
        stub=r"""
        class DepthwiseSeparable(nn.Module):
            def __init__(self, c_in, c_out, stride=1):
                super().__init__()
                # YOUR CODE HERE: self.depthwise, self.bn1, self.pointwise, self.bn2
                raise NotImplementedError

            def forward(self, x):
                # YOUR CODE HERE
                raise NotImplementedError

        ok_dw = tests.check_dw_params(DepthwiseSeparable)
        """,
        solution=r"""
        class DepthwiseSeparable(nn.Module):
            def __init__(self, c_in, c_out, stride=1):
                super().__init__()
                self.depthwise = nn.Conv2d(c_in, c_in, 3, stride, 1, groups=c_in, bias=False)
                self.bn1 = nn.BatchNorm2d(c_in)
                self.pointwise = nn.Conv2d(c_in, c_out, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(c_out)

            def forward(self, x):
                x = F.relu(self.bn1(self.depthwise(x)))
                return F.relu(self.bn2(self.pointwise(x)))

        ok_dw = tests.check_dw_params(DepthwiseSeparable)
        """),
    code(r"""
    DW = tests.use_or_reference(ok_dw, globals().get("DepthwiseSeparable"), day1.DepthwiseSeparable, "DepthwiseSeparable")
    widths = [32, 64, 128, 256, 512]
    std_m, dw_m = [], []
    for c in widths:
        std_m.append(day1.profile_macs(nn.Conv2d(c, c, 3, padding=1, bias=False), (c, 28, 28)) / 1e6)
        dw_m.append(day1.profile_macs(DW(c, c), (c, 28, 28)) / 1e6)
    fig, ax = plt.subplots(figsize=(8, 3.4)); xw = np.arange(len(widths))
    ax.bar(xw - 0.2, std_m, 0.4, label="standard 3x3 conv"); ax.bar(xw + 0.2, dw_m, 0.4, label="depthwise 3x3 + pointwise 1x1")
    for i in range(len(widths)): ax.text(xw[i] + 0.2, dw_m[i], f"{std_m[i] / dw_m[i]:.1f}x", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xw); ax.set_xticklabels([f"{c} -> {c}" for c in widths]); ax.set_yscale("log")
    ax.set_ylabel("M multiply-adds @ 28x28 (log)"); ax.set_title("Same input / output shape, a fraction of the compute"); ax.legend(); ax.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    The saving printed on each bar grows with the width (7x at 32 channels, 8.8x at 512) towards its
    limit $k^2 = 9$: at large widths almost all of the remaining cost is the 1x1 pointwise conv.

    **Squeeze-and-excitation** (SE, 2018) adds cheap *channel attention*: global-average-pool each
    channel to one number ("squeeze"), pass the vector through a tiny 2-layer MLP with a sigmoid
    ("excite"), and multiply each channel by its weight in (0, 1). The block learns to turn
    channels up or down depending on the image, for ~1 % more compute (`day1.SqueezeExcite`).
    MobileNetV3 and EfficientNet use *inverted residual* blocks (1x1 expand - depthwise 3x3 - SE -
    1x1 project, with a residual connection between the thin ends).

    **EfficientNet** (2019) asked how to scale a good small network up. Answer: scale depth $d$,
    width $w$ and input resolution $r$ *together* with fixed ratios, $d = 1.2^\phi$, $w = 1.1^\phi$,
    $r = 1.15^\phi$ (so compute grows by ~$2^\phi$):

    | model | resolution | width x | depth x | params | top-1 |
    |---|---|---|---|---|---|
    | B0 | 224 | 1.0 | 1.0 | 5.3 M | 77.7 |
    | B1 | 240 | 1.0 | 1.1 | 7.8 M | 79.2 |
    | B2 | 260 | 1.1 | 1.2 | 9.2 M | 80.3 |
    | B3 | 300 | 1.2 | 1.4 | 12 M | 81.7 |
    | B4 | 380 | 1.4 | 1.8 | 19 M | 83.4 |
    | B7 | 600 | 2.0 | 3.1 | 66 M | 84.4 |

    **ConvNeXt** (2022) started from ResNet-50 and applied, one at a time, the design choices of
    vision transformers, measuring each step. The result is a pure convnet that matches Swin
    transformers: patchify stem (4x4 stride 4), stage ratio 3:3:9:3, 7x7 depthwise convs,
    inverted bottleneck (1x1 expand x4 - GELU - 1x1), LayerNorm, fewer activations and norms.
    """),
    code(r"""
    steps = [("ResNet-50, 2015 recipe", 76.1), ("modern training recipe (300 ep, AdamW, aug)", 78.8),
             ("stage ratio 3:3:9:3", 79.4), ("patchify stem 4x4/4", 79.5), ("depthwise conv + width 96", 80.5),
             ("inverted bottleneck", 80.6), ("depthwise conv moved up", 79.9), ("kernel 3 -> 7", 80.6),
             ("ReLU -> GELU", 80.6), ("fewer activations", 81.3), ("fewer norms", 81.4), ("BN -> LayerNorm", 81.5),
             ("separate downsampling layers = ConvNeXt-T", 82.0)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    vals = [v for _, v in steps]
    ax.barh(range(len(steps)), [v - 75 for v in vals], left=75, color=[viz.PALETTE[3] if i == 1 else viz.PALETTE[0] for i in range(len(steps))])
    for i, v in enumerate(vals): ax.text(v + 0.05, i, f"{v:.1f}", va="center", fontsize=8)
    ax.set_yticks(range(len(steps))); ax.set_yticklabels([s for s, _ in steps], fontsize=8); ax.invert_yaxis()
    ax.set_xlim(75, 83); ax.set_xlabel("ImageNet top-1 (%)"); ax.set_title("Modernising a ResNet into ConvNeXt (Liu et al. 2022, Fig. 2); red = training recipe alone")
    ax.grid(alpha=.3, axis="x"); plt.tight_layout(); plt.show()
    """),
    md(r"""
    Read the bars top to bottom: the largest single step (+2.7 points) is the training recipe, with no
    change to the network; the eleven design changes add +3.2 together, most of them a few tenths
    each (one, moving the depthwise conv up, even costs 0.7 before the 7x7 kernel wins it back).

    ## 6 · Reading a model

    `torchkit.summarize(model, input_size)` registers forward hooks on every module, runs a dummy
    image through and prints output shapes, parameters and multiply-adds (MACs; one MAC = one
    multiply + one add, so FLOPs ~ 2 x MACs). `day1.summarize(..., depth=0)` groups the rows by
    top-level module (stages) and sums their MACs. Read a summary from top to bottom: the spatial
    size halves at each stage while the channels double, so the **compute per stage stays roughly
    constant** while the **parameters grow ~4x per stage** - most of the weights live in the last
    stage.
    """),
    code(r"""
    r18 = torchvision.models.resnet18(weights=None)
    print(day1.summarize(r18, (3, 224, 224), depth=0))
    """),
    code(r"""
    zoo = {"vgg16": torchvision.models.vgg16, "resnet18": torchvision.models.resnet18, "resnet50": torchvision.models.resnet50,
           "mobilenet_v3_large": torchvision.models.mobilenet_v3_large, "efficientnet_b0": torchvision.models.efficientnet_b0,
           "convnext_tiny": torchvision.models.convnext_tiny}
    rows = []
    for name, ctor in zoo.items():
        prof = torchkit.profile(ctor(weights=None), (3, 224, 224))
        rows.append((name, prof["params"] / 1e6, prof["macs"] / 1e9))
    print(f"{'model':<20}{'params (M)':>12}{'GMACs @224':>12}")
    for n, p_, m_ in rows: print(f"{n:<20}{p_:>12.1f}{m_:>12.2f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (n, p_, m_) in enumerate(rows):
        ax.scatter(m_, p_, s=80, color=viz.color(i), edgecolor="k", zorder=3)
        off = {"mobilenet_v3_large": (-10, 10), "efficientnet_b0": (6, -14), "convnext_tiny": (8, 4), "resnet50": (8, -10)}.get(n, (7, -3))
        ax.annotate(n, (m_, p_), xytext=off, textcoords="offset points", fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("GMACs per 224 px image (log)"); ax.set_ylabel("parameters, M (log)")
    ax.set_xlim(0.15, 40); ax.set_ylim(3.5, 220); ax.set_title("Parameters and compute are different budgets"); ax.grid(alpha=.3, which="both")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Parameters decide **memory and download size**; MACs roughly decide **compute**; neither is
    **latency**. Depthwise convolutions have few MACs but are memory-bound, so MobileNet /
    EfficientNet are less fast on a GPU than their MAC count suggests. Always time the model on
    the hardware you will deploy on (D2-04).

    ## 7 · Assemble a MiniResNet and train it

    ### Exercise 6 · `MiniResNet`

    stem (given) -> three stages of residual blocks with widths `(32, 64, 128)`: stage 1 keeps the
    resolution (stride 1), stages 2 and 3 start with a stride-2 block that also doubles the
    channels -> `nn.AdaptiveAvgPool2d(1)` -> flatten -> `nn.Linear(128, num_classes)`. Use your
    `BasicBlock` (`Basic`, which falls back to the reference if yours has not passed) and
    `blocks[i]` blocks per stage. On a 128 px image the stages run at 32, 16 and 8 px.
    """),
    exercise(
        stub=r"""
        class MiniResNet(nn.Module):
            def __init__(self, num_classes=37, widths=(32, 64, 128), blocks=(1, 1, 1)):
                super().__init__()
                self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 2, 1, bias=False), nn.BatchNorm2d(widths[0]),
                                          nn.ReLU(inplace=True), nn.MaxPool2d(2))
                # YOUR CODE HERE: self.stages (nn.Sequential of nn.Sequential of Basic blocks), self.pool, self.fc
                raise NotImplementedError

            def forward(self, x):
                # YOUR CODE HERE
                raise NotImplementedError

        ok_mini = tests.check_mini_resnet(MiniResNet)
        """,
        solution=r"""
        class MiniResNet(nn.Module):
            def __init__(self, num_classes=37, widths=(32, 64, 128), blocks=(1, 1, 1)):
                super().__init__()
                self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 2, 1, bias=False), nn.BatchNorm2d(widths[0]),
                                          nn.ReLU(inplace=True), nn.MaxPool2d(2))
                stages, c = [], widths[0]
                for i, (w, n) in enumerate(zip(widths, blocks)):
                    layers = []
                    for j in range(n):
                        layers.append(Basic(c, w, stride=2 if (i > 0 and j == 0) else 1))
                        c = w
                    stages.append(nn.Sequential(*layers))
                self.stages = nn.Sequential(*stages)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(c, num_classes)

            def forward(self, x):
                return self.fc(self.pool(self.stages(self.stem(x))).flatten(1))

        ok_mini = tests.check_mini_resnet(MiniResNet)
        """),
    code(r"""
    Mini = tests.use_or_reference(ok_mini, globals().get("MiniResNet"), day1.MiniResNet, "MiniResNet")
    print(day1.summarize(Mini(), (3, 128, 128), depth=1))
    """),
    md(r"""
    Now train it with **exactly** the D1-02 recipe (same data, epochs, augmentation, AdamW,
    warm-up + cosine, label smoothing) and compare with D1-02's `PetNet` (loaded from its
    checkpoint if you ran D1-02). Same budget, different architecture.
    """),
    code(r"""
    GPU = device.type == "cuda"
    IMG, EPOCHS, MAX_MIN = (128, 30, None) if GPU else (64, 10, 2.5)     # same budget as D1-02
    data = day1.pet_tensors(144)
    torch.manual_seed(0)
    mini = Mini(num_classes=37).to(device)
    hist_mini = day1.load_or_train(
        mini, f"d1-03_miniresnet_{IMG}px",
        lambda ck: day1.train_recipe(mini, data, epochs=EPOCHS, lr=3e-3, weight_decay=0.05, label_smoothing=0.1,
                                     size=IMG, max_minutes=MAX_MIN, checkpoint=ck, device=device))

    ck_petnet = torchkit.CHECKPOINTS / f"d1-02_recipe_{IMG}px.pt"
    hist_petnet = torchkit.load_checkpoint(day1.PetNet(dropout=0.2), ck_petnet)["history"] if ck_petnet.exists() else None

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    curves = [("MiniResNet", hist_mini, viz.PALETTE[0])] + ([("PetNet (D1-02 recipe)", hist_petnet, viz.PALETTE[1])] if hist_petnet else [])
    for name, h, c in curves:
        ep = np.arange(1, len(h["val_loss"]) + 1)
        a1.plot(ep, h["train_loss"], "--", color=c, label=f"{name}: train"); a1.plot(ep, h["val_loss"], "-o", ms=3, color=c, label=f"{name}: val")
        a2.plot(ep, 100 * np.array(h["val_metric"]), "-o", ms=3, color=c, label=name)
    a1.set_title("loss"); a2.set_title("validation accuracy (%)")
    for a in (a1, a2): a.set_xlabel("epoch"); a.legend(fontsize=8); a.grid(alpha=.3)
    plt.tight_layout(); plt.show()
    for name, h, _ in curves:
        print(f"{name:24s} best val acc {max(h['val_metric']):.3f}")
    if hist_petnet is None:
        print("(run D1-02 first to overlay the PetNet curve)")
    print(f"params: MiniResNet {torchkit.count_parameters(mini):,}  PetNet {torchkit.count_parameters(day1.PetNet()):,}")
    print(f"MACs per {IMG} px image: MiniResNet {torchkit.profile(mini, (3, IMG, IMG))['macs'] / 1e6:.0f} M  "
          f"PetNet {torchkit.profile(day1.PetNet(), (3, IMG, IMG))['macs'] / 1e6:.0f} M")
    """),
    md(r"""
    With 10 CPU epochs at 64 px both networks are still early in training, so expect differences of
    a few points either way (both ~13 %); on a GPU at 128 px with 40 epochs the MiniResNet (with
    `blocks=(2, 2, 2)`) reaches **~40 %** vs ~33 % for PetNet with the same recipe. From-scratch training on 2 000 images stays
    hard for *any* architecture - which is exactly why the next notebook reuses a network that has
    already seen 1.3 M images.

    ## 8 · Summary

    | idea | first in | why it stayed |
    |---|---|---|
    | conv + pool + fc, backprop | LeNet 1998 | weight sharing, translation equivariance |
    | ReLU, dropout, GPU training, augmentation | AlexNet 2012 | trains deep nets fast |
    | small 3x3 kernels, stacked | VGG 2014 | same receptive field, fewer params, more non-linearity |
    | 1x1 bottlenecks, multi-branch, GAP | GoogLeNet 2014 | cheap width, no giant fc head |
    | residual connections + BN | ResNet 2015 | gradients reach every layer: depth works |
    | feature concatenation | DenseNet 2017 | feature reuse, few params |
    | depthwise-separable convs, SE | MobileNet / SENet 2017-18 | 8-9x cheaper convs, channel attention |
    | compound scaling | EfficientNet 2019 | scale d, w, r together |
    | ViT-era design + training recipe | ConvNeXt 2022 | a convnet that matches transformers |

    ### Self-check
    1. An Inception block gets 256 channels and has branches (128, 128->192, 32->96, pool->64). Output channels?
    2. Why does a 1x1 convolution before a 5x5 save compute? Give the ratio for 192 -> 16 -> 32 vs 192 -> 32.
    3. When does a residual block need a projection shortcut?
    4. In the gradient-flow plot, why does the plain net without BN fail at depth 60, and what does `+ x` change in the backward pass?
    5. Residual vs dense: which one keeps the channel count constant? Which one uses more memory?
    6. A depthwise-separable 3x3 conv 256 -> 256: how many weights, vs a standard 3x3 conv?

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
]
