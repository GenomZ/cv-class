from nb import md, code, exercise, framework, header

NAME = "D1-01-convolution"

CELLS = [
    header(
        "D1-01 · The convolution operation",
        steps=[
            "Look at an image as a tensor",
            "Compute a cross-correlation by hand and animate it",
            "Implement `cross_correlate2d` and apply classic hand-made kernels",
            "Stride, padding, dilation and the output-size formula",
            "Multi-channel convolution, parameter and FLOP counting, `im2col`",
            "Receptive fields: why deep stacks of 3x3 see the whole image",
            "Pooling, global average pooling and 1x1 convolutions",
            "The convnet pattern: shapes through a network; train a tiny convnet in PyTorch and look at its learned kernels",
        ],
        topics=[
            "What a convolution layer computes, exactly, and why it is the right prior for images",
            "Output-size arithmetic for any kernel / stride / padding / dilation",
            "Kernels as pattern detectors; learned filters look like hand-made ones",
            "Receptive field growth; two 3x3 layers = one 5x5 with fewer parameters",
            "Pooling, global average pooling, 1x1 convolutions as channel mixers",
        ],
        minutes=90,
    ),
    md(r"""
    ## 0 · Setup

    The mechanism sections (1-7) use NumPy and Matplotlib only; section 8 trains
    the same network in PyTorch.
    """),
    code("""
    import sys, pathlib
    for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:   # find the cvcourse package
        if (p / "cvcourse").is_dir():
            sys.path.insert(0, str(p)); break

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from cvcourse import viz, convops, tests, labs

    %matplotlib inline
    plt.rcParams["figure.dpi"] = 90
    np.set_printoptions(precision=2, suppress=True)
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · An image is a tensor

    A colour image is a 3-D array: **height x width x channels**. Deep-learning
    frameworks usually store it *channels-first* as `(C, H, W)` (PyTorch) or
    *channels-last* `(H, W, C)` (TensorFlow, Matplotlib, PIL). The numbers are
    integers 0-255 on disk and are converted to floats, typically scaled to `[0, 1]`
    and then standardised with a per-channel mean and standard deviation.
    """),
    code("""
    img = viz.sample_photo()              # ships with Matplotlib, no download
    print("shape (H, W, C):", img.shape, "| dtype:", img.dtype, "| range:", img.min(), "-", img.max())

    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    axes[0].imshow(img); axes[0].set_title("RGB")
    for c, name in enumerate("RGB"):
        axes[c + 1].imshow(img[..., c], cmap="gray", vmin=0, vmax=1)
        axes[c + 1].set_title(f"channel {c} = {name}")
    for ax in axes: ax.axis("off")
    plt.show()

    gray = viz.sample_photo(gray=True)
    print("grayscale shape:", gray.shape)
    print("per-channel mean:", img.reshape(-1, 3).mean(0), "std:", img.reshape(-1, 3).std(0))
    """),
    md(r"""
    > **Why normalise?** Weights are initialised assuming inputs have roughly zero mean and
    > unit variance. Feeding 0-255 integers makes the first layer's pre-activations huge and
    > the optimiser's job unnecessarily hard. Pretrained models expect *their* mean/std
    > (ImageNet: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) - we will meet this
    > again on transfer-learning day.
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · Cross-correlation by hand

    A convolution layer slides a small **kernel** (a.k.a. filter) over the input. At every
    position it multiplies the kernel with the patch underneath, elementwise, and sums the
    result into one output number:

    $$\text{out}[i, j] = \sum_{u=0}^{k_h-1} \sum_{v=0}^{k_w-1} x[i+u,\; j+v] \cdot w[u, v]$$

    Strictly this is a *cross-correlation*; a mathematical *convolution* would flip the kernel
    first. Deep-learning frameworks never flip (the network learns the weights anyway, so a
    flipped kernel is just another kernel) and call the operation "convolution". We will do the same.

    Let's watch it happen on a 5x5 input with a 3x3 kernel.
    """),
    code("""
    x = np.array([[1, 2, 0, 3, 1],
                  [0, 1, 2, 1, 0],
                  [3, 0, 1, 2, 2],
                  [1, 2, 3, 0, 1],
                  [0, 1, 0, 2, 3]], dtype=float)
    k = np.array([[ 1, 0, -1],
                  [ 1, 0, -1],
                  [ 1, 0, -1]], dtype=float)      # a vertical-edge detector

    out = convops.cross_correlate2d(x, k)
    print("input 5x5, kernel 3x3  ->  output", out.shape)
    print(out)
    """),
    code("""
    # Step-by-step figure: the window at every output position
    Ho, Wo = out.shape
    fig, axes = plt.subplots(Ho, Wo * 2, figsize=(2.1 * Wo * 2, 2.1 * Ho))
    for i in range(Ho):
        for j in range(Wo):
            ax_in, ax_out = axes[i, 2 * j], axes[i, 2 * j + 1]
            ax_in.imshow(x, cmap="Blues", vmin=0, vmax=3); viz.annotate_grid(ax_in, x)
            ax_in.add_patch(Rectangle((j - 0.5, i - 0.5), 3, 3, fill=False, lw=2.5, color="#E45756"))
            ax_in.set_title(f"window at ({i},{j})", fontsize=8)
            partial = np.full_like(out, np.nan); partial[:i + 1, :] = out[:i + 1, :]; partial[i, j + 1:] = np.nan
            ax_out.imshow(partial, cmap="Oranges", vmin=out.min(), vmax=out.max())
            viz.annotate_grid(ax_out, np.nan_to_num(partial), fmt="{:.0f}")
            ax_out.set_title(f"out[{i},{j}] = {out[i, j]:.0f}", fontsize=8)
            for ax in (ax_in, ax_out): ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Sliding the 3x3 kernel over the 5x5 input: 9 positions -> 3x3 output", y=1.01)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Each pair of panels is one output position: the red window on the input, and the output
    filled in so far. Check the first one by hand: the kernel's `+1` column sits on `1, 0, 3`
    and its `-1` column on `0, 2, 1`, so `out[0, 0] = (1 + 0 + 3) - (0 + 2 + 1) = 1`.
    """),
    code("""
    # Same thing as an animation (self-contained HTML, no extra packages)
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML

    fig, (ax_in, ax_k, ax_out) = plt.subplots(1, 3, figsize=(9, 3))
    ax_in.imshow(x, cmap="Blues", vmin=0, vmax=3); viz.annotate_grid(ax_in, x); ax_in.set_title("input x")
    ax_k.imshow(k, cmap="RdBu", vmin=-1, vmax=1); viz.annotate_grid(ax_k, k); ax_k.set_title("kernel w")
    rect = Rectangle((-0.5, -0.5), 3, 3, fill=False, lw=2.5, color="#E45756"); ax_in.add_patch(rect)
    shown = np.full_like(out, np.nan)
    im = ax_out.imshow(shown, cmap="Oranges", vmin=out.min(), vmax=out.max()); ax_out.set_title("output")
    texts = [ax_out.text(j, i, "", ha="center", va="center") for i in range(Ho) for j in range(Wo)]
    for ax in (ax_in, ax_k, ax_out): ax.set_xticks([]); ax.set_yticks([])

    def update(frame):
        i, j = divmod(frame, Wo)
        rect.set_xy((j - 0.5, i - 0.5))
        shown[i, j] = out[i, j]; im.set_data(shown)
        texts[frame].set_text(f"{out[i, j]:.0f}")
        ax_out.set_title(f"out[{i},{j}] = sum(patch * w) = {out[i, j]:.0f}")
        return [rect, im, *texts]

    anim = FuncAnimation(fig, update, frames=Ho * Wo, interval=700, blit=False)
    plt.close(fig)
    HTML(anim.to_jshtml())
    """),
    md(r"""
    ### Lab · Convolution playground
    **Predict first:** the kernel above has `+1` in its left column and `-1` in its right column.
    Where the photo goes from dark (left) to bright (right), is its output positive or negative?
    In the lab, pick **sobel x** (its columns are the other way round), hover such an edge to
    check, then work through the "Try this" list under the lab (5 minutes).
    """),
    code("""
    labs.show("conv-playground")
    """),

    md(r"""
    ### Exercise 1 · implement `cross_correlate2d`

    Write the function for a single-channel input `x (H, W)` and kernel `k (kh, kw)`, with
    `stride` and zero-`pad` arguments. Use the formula above; two `for` loops over output
    positions are fine. You may use `convops.pad2d(x, pad)` for the padding. Return a float
    array of shape `(Ho, Wo)`; note that in the stub `H, W` are measured *after* padding, so
    `Ho = (H - kh) // stride + 1` (the output-size formula of section 4).
    """),
    exercise(
        stub="""
        def cross_correlate2d(x, k, stride=1, pad=0):
            x = convops.pad2d(x, pad)
            H, W = x.shape
            kh, kw = k.shape
            # YOUR CODE HERE: compute Ho, Wo, allocate the output and fill it
            raise NotImplementedError

        tests.check_cross_correlate(cross_correlate2d)
        """,
        solution="""
        def cross_correlate2d(x, k, stride=1, pad=0):
            x = convops.pad2d(x, pad)
            H, W = x.shape
            kh, kw = k.shape
            Ho = (H - kh) // stride + 1
            Wo = (W - kw) // stride + 1
            out = np.zeros((Ho, Wo))
            for i in range(Ho):
                for j in range(Wo):
                    patch = x[i * stride:i * stride + kh, j * stride:j * stride + kw]
                    out[i, j] = np.sum(patch * k)
            return out

        tests.check_cross_correlate(cross_correlate2d)
        """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · A kernel is a pattern detector

    Before deep learning, people *designed* kernels by hand. Each one answers a question at
    every pixel: "how much does the neighbourhood look like *this* pattern?" Run the classics
    on the photo. A convnet learns thousands of such kernels from data, and the first layer of
    a trained network looks remarkably like this gallery.
    """),
    code("""
    names = list(convops.KERNELS)
    results = [convops.cross_correlate2d(gray, convops.KERNELS[n], pad=1) for n in names]

    fig, axes = plt.subplots(2, len(names), figsize=(2.2 * len(names), 4.6))
    for c, (n, r) in enumerate(zip(names, results)):
        axes[0, c].imshow(convops.KERNELS[n], cmap="RdBu", vmin=-2, vmax=2)
        viz.annotate_grid(axes[0, c], convops.KERNELS[n], fmt="{:.2f}", fontsize=6)
        axes[0, c].set_title(n, fontsize=9)
        lo, hi = (0, 1) if n in ("identity", "box blur", "gaussian") else (np.percentile(r, 1), np.percentile(r, 99))
        axes[1, c].imshow(r, cmap="gray", vmin=lo, vmax=hi)
        for ax in axes[:, c]: ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Look at *Sobel x* vs *Sobel y*: the same photo, two kernels, two different "where are the
    edges" maps (vertical vs horizontal). The *Laplacian* fires on any sharp change. The
    *Gaussian* removes noise and detail. A convnet's first layer is a learned, data-specific
    version of this gallery; deeper layers combine these responses into corners, textures,
    eyes, wheels, and so on.
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · Stride, padding, dilation and the output size

    Three knobs change the geometry of the output:

    | knob | what it does | typical use |
    |---|---|---|
    | **padding** `p` | add `p` rows/cols of zeros on every side | `p = (k-1)/2` keeps the size ("same" padding) |
    | **stride** `s` | move the kernel `s` pixels at a time | `s = 2` halves the resolution (downsampling) |
    | **dilation** `d` | insert `d-1` zeros between kernel taps | bigger receptive field at the same cost (segmentation) |

    The effective kernel size with dilation is $k_{\text{eff}} = d\,(k - 1) + 1$ and the output size along one axis is

    $$n_{\text{out}} = \left\lfloor \frac{n_{\text{in}} + 2p - k_{\text{eff}}}{s} \right\rfloor + 1 .$$
    """),
    code("""
    # Visual: valid vs same padding, and stride 2
    x8 = np.arange(64, dtype=float).reshape(8, 8) % 7
    k3 = convops.KERNELS["sobel x"]
    cases = [("valid: p=0, s=1", 1, 0), ("same: p=1, s=1", 1, 1), ("stride 2: p=1, s=2", 2, 1)]
    fig, axes = plt.subplots(1, 1 + len(cases), figsize=(13, 3.2))
    axes[0].imshow(x8, cmap="Blues"); axes[0].set_title("input 8x8")
    for ax, (title, s, p) in zip(axes[1:], cases):
        o = convops.cross_correlate2d(x8, k3, s, p)
        ax.imshow(o, cmap="Oranges"); ax.set_title(f"{title} -> {o.shape[0]}x{o.shape[1]}")
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Read the sizes in the titles and check them with the formula: valid
    $\lfloor (8 - 3)/1 \rfloor + 1 = 6$, same $\lfloor (8 + 2 - 3)/1 \rfloor + 1 = 8$, stride 2
    $\lfloor (8 + 2 - 3)/2 \rfloor + 1 = 4$.

    ### Lab · stride, padding and dilation
    **Predict first:** the playground's photo is 64 x 64. With a 3x3 kernel, stride 2 and no
    padding, is the output 32 x 32 or 31 x 31? Then run the cell: the lab starts in exactly that
    setting, with the formula box filled in. Try padding 1 and dilation 2 as well (2 minutes).
    """),
    code("""
    labs.show("conv-playground", stride=2, pad=0)
    """),
    md(r"""
    ### Exercise 2 · output-size formula

    Implement `conv_output_size(n_in, k, stride, pad, dilation)` and pass the checker. Then
    answer by hand: a 28x28 input, 7x7 kernel, stride 1, no padding gives what? (This was a
    quiz question in the old course; now you can derive it.)
    """),
    exercise(
        stub="""
        def conv_output_size(n_in, k, stride=1, pad=0, dilation=1):
            # YOUR CODE HERE
            raise NotImplementedError

        tests.check_output_size(conv_output_size)
        """,
        solution="""
        def conv_output_size(n_in, k, stride=1, pad=0, dilation=1):
            k_eff = dilation * (k - 1) + 1
            return (n_in + 2 * pad - k_eff) // stride + 1

        tests.check_output_size(conv_output_size)
        print("28x28, k=7, s=1, p=0 ->", conv_output_size(28, 7))
        """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · Multi-channel convolution and the cost of a layer

    Real inputs have channels. One kernel is now a small 3-D block `(C_in, kh, kw)` that
    is multiplied with a `(C_in, kh, kw)` patch and summed over **all** of it to give one
    number. A layer has `C_out` such kernels, stacked into a weight tensor

    $$W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k_h \times k_w}, \qquad b \in \mathbb{R}^{C_{\text{out}}}$$

    so **every output channel sees every input channel**. The parameter count is
    $C_{\text{out}} \cdot C_{\text{in}} \cdot k_h \cdot k_w + C_{\text{out}}$ and it does **not**
    depend on the image size - that is the weight sharing that makes convnets cheap and
    translation-equivariant (shift the input and the output shifts the same way). The *compute*
    does depend on the image. It is counted in MACs (multiply-accumulate operations, one
    `a * b + c` each; 1 MAC = 2 FLOPs):
    $\text{MACs} = C_{\text{out}} \cdot H_{\text{out}} \cdot W_{\text{out}} \cdot C_{\text{in}} \cdot k_h \cdot k_w$.
    """),
    code("""
    # Diagram of the weight tensor and one output pixel
    fig, ax = plt.subplots(figsize=(11, 3.4)); ax.axis("off")
    def block(x0, y0, w, h, depth, color, label):
        for d in range(depth - 1, -1, -1):
            ax.add_patch(Rectangle((x0 + 0.12 * d, y0 + 0.12 * d), w, h, fc=color, ec="k", lw=0.8, alpha=0.9))
        ax.text(x0 + w / 2, y0 - 0.35, label, ha="center", fontsize=9)
    block(0.0, 0.5, 2.6, 2.6, 3, "#cfe0f5", "input  (C_in=3, 8, 8)")
    ax.add_patch(Rectangle((0.9, 1.3), 1.0, 1.0, fill=False, ec="#E45756", lw=2.5))
    block(4.0, 1.3, 1.0, 1.0, 3, "#f8d0b8", "kernel 0  (3, 3, 3)")
    block(5.9, 1.3, 1.0, 1.0, 3, "#f8d0b8", "kernel 1")
    ax.text(7.4, 2.55, "...  C_out kernels", va="center", fontsize=9)
    block(9.2, 0.7, 2.0, 2.0, 4, "#cde8cf", "output  (C_out, 6, 6)")
    ax.add_patch(Rectangle((9.55, 1.75), 0.33, 0.33, fc="#E45756", ec="k"))
    ax.annotate("", xy=(4.0, 1.8), xytext=(1.9, 1.8), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(9.5, 1.9), xytext=(5.0, 1.8), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(2.9, 2.05, "patch * kernel,\\nsum over C_in, kh, kw", ha="center", fontsize=8)
    ax.set_xlim(-0.2, 11.8); ax.set_ylim(-0.2, 3.6)
    plt.show()
    """),
    md(r"""
    ### Exercise 3 · multi-channel `conv2d`

    `x` is `(C_in, H, W)`, `w` is `(C_out, C_in, kh, kw)`, `b` is `(C_out,)` or `None`. Return
    `(C_out, Ho, Wo)` with `Ho, Wo` from the output-size formula (no dilation here). Reuse the
    padding helper and loop over output channels and positions; when `b` is given, add `b[o]`
    to every position of output channel `o`.
    """),
    exercise(
        stub="""
        def conv2d(x, w, b=None, stride=1, pad=0):
            C_in, H, W = x.shape
            C_out, _, kh, kw = w.shape
            xp = convops.pad2d(x, pad)
            # YOUR CODE HERE
            raise NotImplementedError

        tests.check_conv2d(conv2d)
        """,
        solution="""
        def conv2d(x, w, b=None, stride=1, pad=0):
            C_in, H, W = x.shape
            C_out, _, kh, kw = w.shape
            xp = convops.pad2d(x, pad)
            Ho = (H + 2 * pad - kh) // stride + 1
            Wo = (W + 2 * pad - kw) // stride + 1
            out = np.zeros((C_out, Ho, Wo))
            for o in range(C_out):
                for i in range(Ho):
                    for j in range(Wo):
                        patch = xp[:, i * stride:i * stride + kh, j * stride:j * stride + kw]
                        out[o, i, j] = np.sum(patch * w[o])
                if b is not None:
                    out[o] += b[o]
            return out

        tests.check_conv2d(conv2d)
        """),
    md(r"""
    ### Exercise 4 · count the parameters

    `count_params(c_in, c_out, k, bias=True)`. Then compare: VGG's first layer
    (3 -> 64, 3x3) vs AlexNet's (3 -> 96, 11x11) vs a ResNet bottleneck's 1x1 (256 -> 64).
    """),
    exercise(
        stub="""
        def count_params(c_in, c_out, k, bias=True):
            # YOUR CODE HERE
            raise NotImplementedError

        tests.check_count_params(count_params)
        """,
        solution="""
        def count_params(c_in, c_out, k, bias=True):
            return c_out * c_in * k * k + (c_out if bias else 0)

        tests.check_count_params(count_params)
        for name, args in [("VGG conv1 3->64 3x3", (3, 64, 3)), ("AlexNet conv1 3->96 11x11", (3, 96, 11)),
                           ("ResNet bottleneck 256->64 1x1", (256, 64, 1)), ("VGG conv 512->512 3x3", (512, 512, 3))]:
            print(f"{name:32s} {count_params(*args):>10,d} params")
        """),
    md(r"""
    ### How frameworks actually compute it: `im2col` + one matrix multiplication

    The loops above are for understanding. GPUs want one big matmul. `im2col` unfolds every
    patch into a row of a matrix (`Ho*Wo` rows, `C_in*kh*kw` columns); the kernels become a
    `(C_out, C_in*kh*kw)` matrix; the product is the whole output. Same numbers, 100x faster
    in NumPy, and cuDNN / Winograd / FFT variants go further.
    """),
    code("""
    import time
    rng = np.random.default_rng(0)
    xb = rng.normal(size=(16, 64, 64)); wb = rng.normal(size=(32, 16, 3, 3)); bb = rng.normal(size=32)
    t0 = time.perf_counter(); o1 = convops.conv2d(xb, wb, bb, 1, 1); t1 = time.perf_counter()
    o2 = convops.conv2d_im2col(xb, wb, bb, 1, 1); t2 = time.perf_counter()
    print(f"loops : {t1 - t0:6.2f} s   im2col+matmul: {t2 - t1:6.3f} s   max |diff| = {np.abs(o1 - o2).max():.1e}")
    cols = convops.im2col(xb, 3, 3, 1, 1)
    print("im2col matrix:", cols.shape, "= (Ho*Wo, C_in*kh*kw);  weights:", wb.reshape(32, -1).shape)
    """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · Receptive fields

    One 3x3 layer lets each output pixel see a 3x3 patch. Stack a second 3x3 layer and each
    of *its* outputs sees a 3x3 patch of 3x3 patches: **5x5** of the input. Pooling or strided
    convolutions multiply the growth. The recursion, layer by layer, is

    $$r_{l} = r_{l-1} + (k_{l} - 1)\, j_{l-1}, \qquad j_{l} = j_{l-1} \cdot s_{l}$$

    where $r$ is the receptive field and $j$ the "jump" (how many input pixels one step in the
    feature map corresponds to). This is *why* a late-layer neuron can respond to "a dog's face"
    although each layer only ever looks at 3x3 neighbourhoods.
    """),
    code("""
    # Two 3x3 layers vs one 5x5 layer: same receptive field, fewer parameters, one extra non-linearity
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, title, layers in zip(axes, ["two 3x3 layers (18 weights)", "one 5x5 layer (25 weights)"],
                                 [[(3, 1, 0), (3, 1, 0)], [(5, 1, 0)]]):
        n = 7
        ax.imshow(np.zeros((n, n)), cmap="Greys", vmin=0, vmax=1)
        ax.add_patch(Rectangle((0.5, 0.5), 5, 5, fc="#cfe0f5", ec="#4C78A8", lw=2))
        if len(layers) == 2:
            ax.add_patch(Rectangle((1.5, 1.5), 3, 3, fc="#f8d0b8", ec="#F58518", lw=2))
        ax.add_patch(Rectangle((2.5, 2.5), 1, 1, fc="#E45756", ec="k"))
        rf = convops.receptive_field(layers)[-1]["rf"]
        ax.set_title(f"{title}\\nreceptive field = {rf}x{rf}"); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    The red pixel is one unit of the last layer. Left: it reads the orange 3x3 window of the
    layer below, and each of those nine units reads its own 3x3 window of the input, which
    together cover the blue 5x5 square. Right: one 5x5 kernel reads the same square directly.

    ### Lab · Receptive field explorer
    **Predict first:** stack conv 3x3, then maxpool 2, then conv 3x3. Is the receptive field of
    one unit of the last layer 7, 8 or 9 pixels wide? Build the stack in the lab (**clear**, then
    add the three layers), click a top unit to check, then work through the "Try this" list (5 minutes).
    """),
    code("""
    labs.show("receptive-field")
    """),
    md(r"""
    ### Exercise 5 · receptive field of a VGG-style stack

    Implement `receptive_field(layers)` where `layers` is a list of `(kernel, stride, pad)`
    tuples. Return a list of dicts with at least the key `"rf"` per layer (the checker reads
    the last one); `rf` and `jump` are the `r` and `j` columns of the lab's table. Padding does
    not change the size of the receptive field, only where it sits, so `p` is unused here. Then
    plot the growth for the stack below.
    """),
    exercise(
        stub="""
        def receptive_field(layers):
            rf, jump = 1, 1
            out = []
            for (k, s, p) in layers:
                # YOUR CODE HERE: update rf and jump, append dict(rf=rf, jump=jump)
                raise NotImplementedError
            return out

        tests.check_receptive_field(receptive_field)
        """,
        solution="""
        def receptive_field(layers):
            rf, jump = 1, 1
            out = []
            for (k, s, p) in layers:
                rf = rf + (k - 1) * jump
                jump = jump * s
                out.append(dict(rf=rf, jump=jump))
            return out

        tests.check_receptive_field(receptive_field)
        """),
    code("""
    vgg_like = [("conv3", (3, 1, 1)), ("conv3", (3, 1, 1)), ("pool", (2, 2, 0)),
                ("conv3", (3, 1, 1)), ("conv3", (3, 1, 1)), ("pool", (2, 2, 0)),
                ("conv3", (3, 1, 1)), ("conv3", (3, 1, 1)), ("conv3", (3, 1, 1)), ("pool", (2, 2, 0)),
                ("conv3", (3, 1, 1)), ("conv3", (3, 1, 1)), ("conv3", (3, 1, 1)), ("pool", (2, 2, 0)),
                ("conv3", (3, 1, 1)), ("conv3", (3, 1, 1)), ("conv3", (3, 1, 1)), ("pool", (2, 2, 0))]
    rfs = convops.receptive_field([l for _, l in vgg_like])
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot([d["rf"] for d in rfs], marker="o")
    ax.set_xticks(range(len(vgg_like))); ax.set_xticklabels([n for n, _ in vgg_like], rotation=60, fontsize=8)
    ax.set_ylabel("receptive field (px)"); ax.set_title("Receptive field through a VGG-16 style stack (input 224)"); ax.grid(alpha=.3)
    for i, d in enumerate(rfs):
        if vgg_like[i][0] == "pool": ax.annotate(f"jump {d['jump']}", (i, d["rf"]), textcoords="offset points", xytext=(0, 8), fontsize=7, ha="center")
    plt.tight_layout(); plt.show()
    print("final receptive field:", rfs[-1]["rf"], "px  (input is 224 px)")
    """),
    md(r"""
    Each pool doubles the jump, so every 3x3 after it adds twice as many pixels as before: the
    slope of the curve doubles after every pool. Thirteen 3x3 layers and five pools end at 212 px,
    so a unit of the last block sees almost the whole 224 px image.
    """),

    # ------------------------------------------------------------------ 7
    md(r"""
    ## 7 · Pooling, global average pooling, 1x1 convolutions

    * **Max / average pooling** reduce resolution with no parameters, giving some translation
      *invariance* ("there is an edge somewhere in this 2x2 region"). Modern nets often use
      stride-2 convolutions instead, but pooling is still everywhere.
    * **Global average pooling** collapses `(C, H, W)` to `(C,)` by averaging each channel over
      space. It replaced the giant `Flatten -> Dense` head of VGG (100M+ parameters) with
      zero parameters, and it lets the same network accept any input size.
    * A **1x1 convolution** does not look at neighbours at all: at each pixel it is a plain
      linear layer across channels. Uses: cheaply change the channel count (bottlenecks in
      Inception and ResNet), mix channels after a depthwise conv (MobileNet), produce
      per-pixel predictions (segmentation heads, detection heads).
    """),
    code("""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    axes[0].imshow(gray, cmap="gray"); axes[0].set_title(f"input {gray.shape}")
    p2 = convops.pool2d(gray, 2, 2, "max"); axes[1].imshow(p2, cmap="gray"); axes[1].set_title(f"max pool 2x2 -> {p2.shape}")
    p8 = convops.pool2d(gray, 8, 8, "max"); axes[2].imshow(p8, cmap="gray"); axes[2].set_title(f"max pool 8x8 -> {p8.shape}")
    a8 = convops.pool2d(gray, 8, 8, "avg"); axes[3].imshow(a8, cmap="gray"); axes[3].set_title(f"avg pool 8x8 -> {a8.shape}")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()
    print("global average pooling of the RGB image -> one number per channel:", img.mean(axis=(0, 1)))
    """),
    md(r"""
    ### Exercise 6 · `pool2d`

    `pool2d(x, k, stride, mode)` for `x` of shape `(..., H, W)`, `mode` in `{"max", "avg"}`.
    """),
    exercise(
        stub="""
        def pool2d(x, k=2, stride=2, mode="max"):
            H, W = x.shape[-2:]
            # YOUR CODE HERE
            raise NotImplementedError

        tests.check_pool2d(pool2d)
        """,
        solution="""
        def pool2d(x, k=2, stride=2, mode="max"):
            H, W = x.shape[-2:]
            Ho = (H - k) // stride + 1
            Wo = (W - k) // stride + 1
            out = np.zeros(x.shape[:-2] + (Ho, Wo))
            red = np.max if mode == "max" else np.mean
            for i in range(Ho):
                for j in range(Wo):
                    out[..., i, j] = red(x[..., i * stride:i * stride + k, j * stride:j * stride + k], axis=(-2, -1))
            return out

        tests.check_pool2d(pool2d)
        """),
    code("""
    # A 1x1 convolution is a per-pixel linear layer across channels: here 3 -> 2 channels
    rng = np.random.default_rng(1)
    x_chw = img.transpose(2, 0, 1)                      # (3, H, W)
    w11 = np.array([[[[0.299]], [[0.587]], [[0.114]]],  # channel 0 = luminance
                    [[[1.0]],   [[-1.0]],  [[0.0]]]])    # channel 1 = red minus green
    y = convops.conv2d_im2col(x_chw, w11, None, 1, 0)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    axes[0].imshow(img); axes[0].set_title("input (3 channels)")
    axes[1].imshow(y[0], cmap="gray"); axes[1].set_title("1x1 conv, out ch 0: 0.3R+0.6G+0.1B")
    axes[2].imshow(y[1], cmap="RdBu"); axes[2].set_title("1x1 conv, out ch 1: R - G")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()
    print("weights shape:", w11.shape, "-> parameters:", w11.size, "(no spatial extent, only channel mixing)")
    """),

    # ------------------------------------------------------------------ 8
    md(r"""
    ## 8 · The convnet pattern

    Stack it all up and you get the shape of every classifier since LeNet: **spatial size
    shrinks, channel count grows**, then a global pool and a linear classifier. Early layers
    detect edges and colours (generic); late layers detect object parts (specific). Let us
    push a random-weight network forward in NumPy just to watch the tensor shapes - this is
    the same arithmetic you will read in every `model.summary()`.
    """),
    code("""
    def describe_convnet(input_hw, channels_in, spec):
        \"\"\"spec: list of ('conv', c_out, k, s, p) | ('pool', k, s) | ('gap',) | ('fc', n). Prints the shape table.\"\"\"
        h, w, c = *input_hw, channels_in
        total = 0
        print(f"{'layer':<18}{'output shape':<22}{'params':>12}")
        print(f"{'input':<18}{str((c, h, w)):<22}{'-':>12}")
        for layer in spec:
            if layer[0] == "conv":
                _, co, k, s, p = layer
                h, w = convops.conv_output_size(h, k, s, p), convops.conv_output_size(w, k, s, p)
                n = convops.conv_params(c, co, k); c = co
                print(f"{f'conv {k}x{k}/{s}':<18}{str((c, h, w)):<22}{n:>12,d}")
            elif layer[0] == "pool":
                _, k, s = layer; h, w = (h - k) // s + 1, (w - k) // s + 1; n = 0
                print(f"{f'maxpool {k}/{s}':<18}{str((c, h, w)):<22}{n:>12,d}")
            elif layer[0] == "gap":
                h = w = 1; n = 0
                print(f"{'global avg pool':<18}{str((c,)):<22}{n:>12,d}")
            elif layer[0] == "fc":
                _, nout = layer; n = c * h * w * nout + nout; c, h, w = nout, 1, 1
                print(f"{f'fc -> {nout}':<18}{str((c,)):<22}{n:>12,d}")
            total += n
        print(f"{'TOTAL':<18}{'':<22}{total:>12,d}")

    print("Tiny convnet for 28x28 grayscale (FashionMNIST):")
    describe_convnet((28, 28), 1, [("conv", 32, 3, 1, 1), ("pool", 2, 2), ("conv", 64, 3, 1, 1), ("pool", 2, 2),
                                   ("conv", 64, 3, 1, 1), ("gap",), ("fc", 10)])
    print()
    print("Same net but with Flatten+Dense instead of global average pooling:")
    describe_convnet((28, 28), 1, [("conv", 32, 3, 1, 1), ("pool", 2, 2), ("conv", 64, 3, 1, 1), ("pool", 2, 2),
                                   ("conv", 64, 3, 1, 1), ("fc", 64), ("fc", 10)])
    """),
    md(r"""
    ### Train it in PyTorch

    Time to build the network from the first table for real and train it on **FashionMNIST**:
    60 000 training and 10 000 test images of clothes, 28x28 grayscale, 10 classes. We load
    the whole dataset into memory as tensors (`as_tensors=True`), which is the fastest
    option on a CPU, and normalise it with the dataset's mean and standard deviation.
    """),
    code("""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from cvcourse import torchkit

    torch.manual_seed(0)
    device = torchkit.get_device()
    fm_train, fm_test, fm_classes = torchkit.get_fashion_mnist(as_tensors=True)
    train_loader = DataLoader(fm_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(fm_test, batch_size=1000)
    xb, yb = next(iter(train_loader))
    print("one batch:", tuple(xb.shape), "labels", tuple(yb.shape), "| train / test:", len(fm_train), "/", len(fm_test))
    torchkit.show_batch(xb, yb, fm_classes, n=16, size=1.5); plt.show()
    """),
    md(r"""
    ### Exercise 7 · the forward pass of the tiny convnet

    The layers are created in `__init__` for you (`nn.Conv2d(c_in, c_out, k, padding=p)` is
    exactly the `W` and `b` of section 5). Write `forward`: conv1 -> ReLU -> pool, conv2 ->
    ReLU -> pool, conv3 -> ReLU, **global average pooling** (mean over the two spatial axes),
    then the linear classifier. Return raw scores (logits); the cross-entropy loss applies
    the softmax itself. The checker compares shapes, the parameter count and the values.
    """),
    exercise(
        stub="""
        class TinyConvNet(nn.Module):
            def __init__(self, n_classes=10):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)     # (B, 1, 28, 28)  -> (B, 32, 28, 28)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)    # after pool: (B, 32, 14, 14) -> (B, 64, 14, 14)
                self.conv3 = nn.Conv2d(64, 64, 3, padding=1)    # after pool: (B, 64, 7, 7)   -> (B, 64, 7, 7)
                self.pool = nn.MaxPool2d(2)
                self.fc = nn.Linear(64, n_classes)              # after GAP: (B, 64) -> (B, 10)

            def forward(self, x):
                # YOUR CODE HERE
                raise NotImplementedError

        tests.check_tiny_convnet(TinyConvNet())
        """,
        solution="""
        class TinyConvNet(nn.Module):
            def __init__(self, n_classes=10):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)     # (B, 1, 28, 28)  -> (B, 32, 28, 28)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)    # after pool: (B, 32, 14, 14) -> (B, 64, 14, 14)
                self.conv3 = nn.Conv2d(64, 64, 3, padding=1)    # after pool: (B, 64, 7, 7)   -> (B, 64, 7, 7)
                self.pool = nn.MaxPool2d(2)
                self.fc = nn.Linear(64, n_classes)              # after GAP: (B, 64) -> (B, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = F.relu(self.conv3(x))
                x = x.mean(dim=(2, 3))                          # global average pooling
                return self.fc(x)

        tests.check_tiny_convnet(TinyConvNet())
        """),
    md(r"""
    `torchkit.summarize` pushes a dummy image through the model with forward hooks and
    records every layer's output. Compare it with the NumPy table at the start of this
    section: same shapes, same 56 394 parameters. The MACs column is the formula from
    section 5.
    """),
    code("""
    print(torchkit.summarize(TinyConvNet(), (1, 28, 28)))
    """),
    md(r"""
    **Training.** Two epochs of AdamW, capped at three minutes. The learning rate follows a
    *one-cycle* schedule: it warms up to a peak and then anneals to almost zero, which gets
    much more out of two epochs than a constant rate (about 90% instead of 81% with a constant
    rate, for this net; the why is in D1-02). The trained weights are saved to `data/checkpoints/`; when the file
    exists, the cell loads it instead of training, so re-running the notebook is instant
    (delete the file to retrain).

    For comparison we train a **dense baseline** that never looks at neighbourhoods: flatten
    the 784 pixels, one hidden layer of 256, then 10 outputs. Expected test accuracy after two
    epochs: about **89-90% for the convnet** and **87% for the dense network**, which has
    3.6x *more* parameters. On a laptop CPU the convnet takes one to two minutes, the dense
    network a few seconds.
    """),
    code("""
    def train_or_load(model, name, max_lr, epochs=2):
        \"\"\"Train with torchkit.fit, or load data/checkpoints/<name>.pt if it exists. Returns the history.\"\"\"
        ck = torchkit.CHECKPOINTS / f"{name}.pt"
        if ck.exists():
            extras = torchkit.load_checkpoint(model, ck)
            print(f"loaded {ck.name} (delete it to retrain)")
            return extras["history"]
        opt = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=epochs * len(train_loader), pct_start=0.15)
        return torchkit.fit(model, train_loader, test_loader, nn.CrossEntropyLoss(), opt, epochs, scheduler=sched,
                            device=device, max_minutes=3, log_every=0, checkpoint=ck, restore_best=True)

    torch.manual_seed(0)
    cnn = TinyConvNet().to(device)
    hist_cnn = train_or_load(cnn, "d1_01_tiny_convnet", max_lr=1e-2)

    torch.manual_seed(0)
    dense = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
    hist_dense = train_or_load(dense, "d1_01_dense_baseline", max_lr=3e-3)
    """),
    code("""
    _, acc_cnn = torchkit.evaluate(cnn, test_loader, nn.CrossEntropyLoss(), torchkit.accuracy, device)
    _, acc_dense = torchkit.evaluate(dense, test_loader, nn.CrossEntropyLoss(), torchkit.accuracy, device)
    rows = [("tiny convnet", torchkit.count_parameters(cnn), acc_cnn, "#4C78A8"),
            ("dense 784-256-10", torchkit.count_parameters(dense), acc_dense, "#BAB0AC")]
    print(f"{'model':<20}{'params':>10}{'test acc':>10}")
    for name, n, acc, _ in rows:
        print(f"{name:<20}{n:>10,d}{acc:>10.1%}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for (name, n, acc, c), h in zip(rows, (hist_cnn, hist_dense)):
        ep = range(1, len(h["val_metric"]) + 1)
        axes[0].plot(ep, h["train_loss"], marker="o", color=c, label=f"{name} train")
        axes[0].plot(ep, h["val_loss"], marker="s", ls="--", color=c, label=f"{name} test")
    axes[0].set_xlabel("epoch"); axes[0].set_xticks([1, 2]); axes[0].set_title("cross-entropy loss"); axes[0].legend(fontsize=8)
    names = [r[0] for r in rows]
    bars = axes[1].bar(names, [r[2] * 100 for r in rows], color=[r[3] for r in rows])
    axes[1].bar_label(bars, fmt="%.1f%%"); axes[1].set_ylim(80, 95); axes[1].set_title("test accuracy (%)")
    bars = axes[2].bar(names, [r[1] / 1000 for r in rows], color=[r[3] for r in rows])
    axes[2].bar_label(bars, fmt="%.0fk"); axes[2].set_title("parameters (thousands)"); axes[2].set_ylim(0, 240)
    for ax in axes: ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    The convnet wins with fewer parameters because its prior is right for images: the same
    small detector is reused at every position (weight sharing), and pooling makes the
    answer tolerant to small shifts. The dense network has to learn "a sleeve at the left"
    and "a sleeve at the right" as two unrelated patterns.

    ### What did the first layer learn?

    `cnn.conv1.weight` has shape `(32, 1, 3, 3)`: 32 kernels of 3x3, exactly like the hand-made
    gallery of section 3. Below, each learned kernel is drawn with its own symmetric colour
    scale (blue positive, red negative, white zero, as in section 3), so only the *pattern* matters.
    """),
    code("""
    W1 = cnn.conv1.weight.detach().cpu()[:, 0].numpy()          # (32, 3, 3)
    hand = {n: k for n, k in convops.KERNELS.items() if n != "identity"}

    fig = plt.figure(figsize=(12, 7.4), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 2.7])
    axs = top.subplots(1, len(hand))
    for ax, (n, k) in zip(axs, hand.items()):
        m = np.abs(k).max(); ax.imshow(k, cmap="RdBu", vmin=-m, vmax=m)
        ax.set_title(n, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    top.suptitle("hand-made kernels (section 3)", fontsize=12)
    axs = bottom.subplots(4, 8)
    for i, ax in enumerate(axs.flat):
        m = np.abs(W1[i]).max(); ax.imshow(W1[i], cmap="RdBu", vmin=-m, vmax=m)
        ax.set_title(f"#{i}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    bottom.suptitle("the 32 kernels conv1 learned on FashionMNIST in two epochs", fontsize=12)
    plt.show()
    """),
    md(r"""
    Look for the families you know: kernels with a positive column next to a negative
    column are vertical-edge detectors (Sobel x), rows of opposite sign are horizontal-edge
    detectors (Sobel y), a strong centre with an opposite-sign ring is a Laplacian / sharpen,
    and kernels with one sign everywhere act as blurs or brightness detectors. Nobody
    designed these: gradient descent found that these patterns are useful for telling
    clothes apart. On a 3x3 grid and two epochs the resemblance is rough; the first layer of
    a big ImageNet network (Day 1, block 4) shows clean edges, colour blobs and Gabor-like stripes.

    ### Feature maps: what the first layer does to one image

    Each of the 32 kernels produces one output channel, a 28x28 map that is bright where the
    image looks like that kernel. This is the `(32, 28, 28)` tensor that `conv2` receives.

    Notice the bright frame around many maps: after normalisation the black background is
    about -0.8, but `padding=1` surrounds the image with zeros, so the border looks like a
    sudden edge to every edge detector. Padding is not free of side effects.
    """),
    code("""
    x0, y0 = fm_test[0]
    with torch.no_grad():
        fmap = F.relu(cnn.conv1(x0.unsqueeze(0).to(device)))[0].cpu().numpy()   # (32, 28, 28)
        pred = cnn(x0.unsqueeze(0).to(device)).argmax(1).item()

    fig = plt.figure(figsize=(13, 5.6), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1, 3.4])
    ax = left.subplots()
    ax.imshow(torchkit.denormalize(x0, torchkit.FASHION_MNIST_MEAN, torchkit.FASHION_MNIST_STD), cmap="gray")
    ax.set_title(f"input: {fm_classes[y0]}\\n(predicted: {fm_classes[pred]})"); ax.axis("off")
    axs = right.subplots(4, 8)
    for i, ax in enumerate(axs.flat):
        ax.imshow(fmap[i], cmap="magma", vmin=0, vmax=max(fmap[i].max(), 1e-6))
        ax.set_title(f"#{i}", fontsize=8); ax.axis("off")
    right.suptitle("ReLU(conv1) output: one map per learned kernel (same numbering as above)")
    plt.show()
    print("maps that stay black: the kernel never matches this image (ReLU clips negative responses to 0)")
    print("dead maps:", [i for i in range(32) if fmap[i].max() == 0])
    """),

    # ------------------------------------------------------------------ 9
    md(r"""
    ## 9 · Summary

    * A convolution layer = `C_out` small kernels, each multiplied-and-summed over every
      `(C_in, k, k)` patch of the input. Parameters do not depend on image size; compute does.
    * Output size: $\lfloor (n + 2p - d(k-1) - 1)/s \rfloor + 1$.
    * Receptive field grows additively with kernel size and multiplicatively with stride;
      two 3x3 layers = one 5x5 layer with 28% fewer weights and an extra non-linearity.
    * Pooling / strides trade resolution for invariance; global average pooling kills the
      giant dense head; 1x1 convolutions mix channels per pixel.
    * The pattern "shrink space, grow channels, pool, classify" is every convnet from 1998 to ConvNeXt.

    ### Self-check (answers in the Day 1 quiz)
    1. Input 32x32, conv 5x5 stride 1 pad 2, then max-pool 2: output size?
    2. Which has more parameters: one 7x7 conv or three stacked 3x3 convs with the same channels?
    3. Why does global average pooling let a classifier accept any input resolution?
    4. Name two things a 1x1 convolution is used for.

    ### Go deeper
    * cs231n notes, "Convolutional Networks" (the pre-read for this course).
    * Dumoulin & Visin, *A guide to convolution arithmetic for deep learning* (the animated GIFs of every padding / stride combination).
    * Araujo et al., *Computing receptive fields of convolutional neural networks* (Distill, 2019).
    """),
]
