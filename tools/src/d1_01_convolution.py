import re
import textwrap

from nb import md, code, exercise, header, api

NAME = "D1-01-convolution"

# ---------------------------------------------------------------------------------------------
# Guided exercises: the stub is DERIVED from the solution, so that the two can never drift apart.
#
# In a solution, a line   `name = expression   # YOUR CODE: what this line computes`   is a blank:
# the student notebook shows `name = ____` there, with the same comment. The line `#@raise` becomes
# `raise NotImplementedError   # delete this line ...` in the student notebook and disappears from
# the solution. Everything else is identical in both notebooks.
# ---------------------------------------------------------------------------------------------
_BLANK = re.compile(r"^(?P<indent>\s*)(?P<lhs>.+?) = (?P<rhs>.+?)(?P<comment>\s+# YOUR CODE:.*)$")
_RAISE_MARK = "#@raise"
_RAISE_LINE = "raise NotImplementedError   # delete this line once every ____ below is filled in"


def guided(solution: str) -> dict:
    stub_lines = []
    solution_lines = []
    for line in textwrap.dedent(solution).strip("\n").splitlines():
        if line.strip() == _RAISE_MARK:
            indent = line[: len(line) - len(line.lstrip())]
            stub_lines.append(indent + _RAISE_LINE)
            continue
        solution_lines.append(line)
        blank = _BLANK.match(line)
        if blank:
            filler = "____".ljust(len(blank.group("rhs")))
            stub_lines.append(blank.group("indent") + blank.group("lhs") + " = " + filler + blank.group("comment"))
        else:
            stub_lines.append(line)
    return exercise(stub="\n".join(stub_lines), solution="\n".join(solution_lines))


CELLS = [
    header(
        "D1-01 · The convolution operation",
        steps=[
            "An image is an array of numbers (a short recap of D0-00)",
            "Cross-correlation by hand, as a figure and as an animation; lab; write `cross_correlate2d` (exercise 1)",
            "Hand-made kernels: what a kernel detects",
            "Stride, padding, dilation and the output-size formula; lab; exercise 2",
            "Many channels: the convolution layer (exercise 3), its parameters (exercise 4) and its cost; `im2col`",
            "Receptive fields: how much of the image one output number sees; lab; exercise 5",
            "Pooling (exercise 6), global average pooling and 1x1 convolutions",
            "The convnet pattern: shapes through a network; the forward pass of a tiny convnet in PyTorch "
            "(exercise 7); train it, compare it with a dense network, and look at what its first layer learned",
        ],
        topics=[
            "Compute a convolution by hand and in code: slide a kernel, multiply element by element, add up",
            "Predict the output size of a layer from its kernel size, stride, padding and dilation",
            "Read a kernel as a pattern detector; see that learned first-layer kernels look like hand-made ones",
            "Count the parameters and the multiply-adds of a convolution layer",
            "Explain how the receptive field grows through a stack of layers, and why two 3x3 layers beat one 5x5",
            "Explain what pooling, global average pooling and 1x1 convolutions are for",
            "Write the forward pass of a small convolutional network in PyTorch",
        ],
        minutes=120,
        setting="in class, guided + hands-on (about 100 min if the instructor does exercises 3 to 5 on the "
                "board and the optional parts are skipped)",
        prereq="D0-00 (tensors, the (C, H, W) layout, the training loop). Python loops, NumPy indexing and "
               "slicing. Neural networks and backpropagation (previous class).",
    ),
    md(r"""
    **How to read this notebook.** It works like D0-00 (its section "How to read this notebook" has the
    details): run the cells in order; a word in **bold** is explained where it first appears and listed
    again in "Words from this notebook" at the end; the closed boxes **Functions used here** explain
    every library function before the cell that uses it; boxes titled **More:** are optional. The seven
    exercises are **code templates**: read the numbered steps, replace every `____` by the code its
    `# YOUR CODE:` comment describes, then delete the line `raise NotImplementedError` at the top of the
    function and run the cell. The text above each exercise gives an example to check by hand, the plan,
    and a table of common mistakes with the message each one produces. Stuck for more than five minutes?
    The solved notebook is `notebooks/solutions/D1-01-convolution-solution.ipynb`.
    """),

    # ------------------------------------------------------------------ 0
    md(r"""
    ## 0 · Setup

    Sections 1 to 7 use only NumPy and Matplotlib: every operation is written as plain loops you can
    read. Section 8 switches to PyTorch to train a network. Besides the usual imports, the setup cell
    loads four course modules:

    * `convops`: the course's own NumPy versions of every operation in this notebook. The demonstrations
      use them, and the checkers compare your code with them;
    * `viz`: small drawing helpers; `tests`: the exercise checkers; `labs`: the interactive labs.

    `np.set_printoptions` makes NumPy print two decimals, so the printed arrays stay readable.
    """),
    api("np.set_printoptions"),
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

    A convolution works on images, so first a reminder of how an image is stored (D0-00, section 3).
    A colour image is a 3-D array of **pixels**: height x width x 3 **channels** (red, green, blue).
    Files store each value as a whole number 0 to 255; we work with floats in [0, 1]. Matplotlib and
    NumPy keep the channels last, (H, W, C); PyTorch puts them first, (C, H, W).

    `viz.sample_photo()` returns the photo of Grace Hopper you met in D0-00 as an (H, W, C) float array.
    With `gray=True` it returns a **grayscale** version, one channel of brightness: a weighted average
    0.299 R + 0.587 G + 0.114 B (green counts most because the eye is most sensitive to it). Sections 2
    to 4 work on one channel; section 5 adds the others.
    """),
    api("Tensor.min", "Tensor.max", "Tensor.std"),
    api("Tensor.reshape", "Tensor.mean", brief=True),
    code("""
    img = viz.sample_photo()              # (H, W, C) float32 in [0, 1]; ships with Matplotlib, no download
    print("shape (H, W, C):", img.shape, "| dtype:", img.dtype, "| range:", img.min(), "-", img.max())

    # Figure: the photo and its three channels, each shown as a grayscale picture
    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    axes[0].imshow(img)
    axes[0].set_title("RGB")
    channel_names = ["R", "G", "B"]
    for c in range(3):
        axes[c + 1].imshow(img[:, :, c], cmap="gray", vmin=0, vmax=1)
        axes[c + 1].set_title(f"channel {c} = {channel_names[c]}")
    for ax in axes:
        ax.axis("off")
    plt.show()

    gray = viz.sample_photo(gray=True)    # (H, W): one channel
    print("grayscale shape:", gray.shape)
    pixels = img.reshape(-1, 3)           # (600 * 512, 3): one row per pixel, one column per channel
    print("per-channel mean:", pixels.mean(axis=0), "std:", pixels.std(axis=0))
    """),
    md(r"""
    The last line is the start of **normalisation** (D0-00, section 4): every channel is shifted and
    scaled to mean 0 and standard deviation 1 before it enters a network. Section 8 normalises
    FashionMNIST with its own mean and std; pretrained networks (D1-04) expect ImageNet's numbers.
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · Cross-correlation by hand

    **Why.** Every network of this course is built from one operation, the **convolution**: the
    classifiers of Day 1, the backbones that turn a photo into features for the detectors of Day 2, the
    U-Net that labels every pixel on Day 3. This section computes it by hand on a 5 x 5 example.

    **Words.**
    * A **kernel**, also called a **filter**, is a small grid of numbers, here 3 x 3. In a network these
      numbers are **weights**: parameters (D0-00) that training adjusts. In sections 2 and 3 we choose
      them by hand.
    * Place the kernel over a 3 x 3 **window** of the input. The input values under it are the **patch**.
    * Multiply every kernel number with the input number under it (**element by element**: top-left with
      top-left, and so on) and add up the 9 products. That sum is one output number.
    * Move the window one pixel to the right and repeat; at the end of a row, go one row down. The
      output numbers form a new grid, the **feature map**: at every position, how strongly the input
      there looks like the kernel's pattern.

    As a formula, with $i, j$ the row and column of the output and $u, v$ the row and column inside the
    kernel ($x$ = input, $w$ = kernel, $k_h \times k_w$ = kernel size):

    $$\text{out}[i, j] = \sum_{u=0}^{k_h-1} \sum_{v=0}^{k_w-1} x[i+u,\; j+v] \cdot w[u, v]$$

    **Cross-correlation or convolution?** Strictly, the formula above is a **cross-correlation**. A
    mathematical **convolution** first flips the kernel upside down and left to right. Deep-learning
    libraries never flip, and call the operation "convolution" anyway: the kernel is learned, so a
    flipped kernel is one more kernel training could have found. This course does the same:
    "convolution" means the formula above.

    **Worked example.** The input `x` (5 x 5) and the kernel `k` (3 x 3) of the next cell. For output
    position (0, 0) the window covers rows 0 to 2 and columns 0 to 2:

    ```
    patch            kernel          products
    1  2  0          1  0  -1        1  0   0
    0  1  2          1  0  -1        0  0  -2
    3  0  1          1  0  -1        3  0  -1
    ```

    Sum of the products: 1 + 0 + 0 + 0 + 0 - 2 + 3 + 0 - 1 = **1**. One step to the right, position
    (0, 1), the patch is columns 1 to 3: its left column 2, 1, 0 is multiplied by +1, its right column
    3, 1, 2 by -1, the middle by 0: (2 + 1 + 0) - (3 + 1 + 2) = **-3**.

    This kernel computes "left column minus right column": large where the left side of the window is
    brighter than the right, 0 where all columns are equal. It detects vertical **edges**, places where
    the brightness changes quickly from left to right.

    **How many positions?** The window's top-left corner can sit at rows 0, 1 and 2 (at row 3 the
    window would hang over the bottom edge), and the same for columns. In general an input of size
    n and a kernel of size k give n - k + 1 positions per axis: 5 - 3 + 1 = 3, so the output is 3 x 3.

    The next cell computes all 9 outputs with the course's version, `convops.cross_correlate2d`. You
    write your own in Exercise 1.
    """),
    api("np.array", "np.full_like"),
    code("""
    x = np.array([[1, 2, 0, 3, 1],
                  [0, 1, 2, 1, 0],
                  [3, 0, 1, 2, 2],
                  [1, 2, 3, 0, 1],
                  [0, 1, 0, 2, 3]], dtype=float)      # (5, 5) input
    k = np.array([[ 1, 0, -1],
                  [ 1, 0, -1],
                  [ 1, 0, -1]], dtype=float)         # (3, 3) kernel: a vertical-edge detector

    out = convops.cross_correlate2d(x, k)
    print("input 5x5, kernel 3x3  ->  output", out.shape)
    print(out)
    """),
    md(r"""
    The first row starts with 1 and -3, as computed by hand. The figure below shows every step: each
    pair of panels is one output position, with the red window on the input (left) and the output
    filled in so far (right).
    """),
    code("""
    # Figure: one pair of panels per output position. Left: the window on the input. Right: the output so far.
    out_height, out_width = out.shape                         # 3, 3
    fig, axes = plt.subplots(out_height, 2 * out_width, figsize=(12.6, 6.3))
    for row in range(out_height):
        for col in range(out_width):
            ax_in = axes[row, 2 * col]
            ax_out = axes[row, 2 * col + 1]
            # left panel: the input, with the 3x3 window of this position in red
            ax_in.imshow(x, cmap="Blues", vmin=0, vmax=3)
            viz.annotate_grid(ax_in, x)
            ax_in.add_patch(Rectangle((col - 0.5, row - 0.5), 3, 3, fill=False, lw=2.5, color="#E45756"))
            ax_in.set_title(f"window at ({row},{col})", fontsize=8)
            # right panel: the outputs computed up to this position; the others stay blank (NaN)
            so_far = np.full_like(out, np.nan)
            so_far[:row, :] = out[:row, :]                     # every earlier row
            so_far[row, :col + 1] = out[row, :col + 1]         # this row, up to this column
            ax_out.imshow(so_far, cmap="Oranges", vmin=out.min(), vmax=out.max())
            for r in range(out_height):
                for c in range(out_width):
                    if r < row or (r == row and c <= col):
                        ax_out.text(c, r, f"{out[r, c]:.0f}", ha="center", va="center", fontsize=8)
            ax_out.set_title(f"out[{row},{col}] = {out[row, col]:.0f}", fontsize=8)
            for ax in (ax_in, ax_out):
                ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Sliding the 3x3 kernel over the 5x5 input: 9 positions -> 3x3 output", y=1.01)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    The same as an animation: the window moves, the output fills in. (Matplotlib's `FuncAnimation`
    calls `update` once per frame; the result is shown as a small web page with play buttons.)
    """),
    code("""
    # Animation: the window slides over the input, the output fills in one number per frame
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML

    fig, (ax_in, ax_k, ax_out) = plt.subplots(1, 3, figsize=(9, 3))
    ax_in.imshow(x, cmap="Blues", vmin=0, vmax=3)
    viz.annotate_grid(ax_in, x)
    ax_in.set_title("input x")
    ax_k.imshow(k, cmap="RdBu", vmin=-1, vmax=1)
    viz.annotate_grid(ax_k, k)
    ax_k.set_title("kernel k")
    window = Rectangle((-0.5, -0.5), 3, 3, fill=False, lw=2.5, color="#E45756")
    ax_in.add_patch(window)
    shown = np.full_like(out, np.nan)                       # the output so far; NaN cells stay blank
    picture = ax_out.imshow(shown, cmap="Oranges", vmin=out.min(), vmax=out.max())
    ax_out.set_title("output")
    labels = {}                                             # one text label per output cell, empty for now
    for row in range(out_height):
        for col in range(out_width):
            labels[(row, col)] = ax_out.text(col, row, "", ha="center", va="center")
    for ax in (ax_in, ax_k, ax_out):
        ax.set_xticks([]); ax.set_yticks([])

    def update(frame):
        row = frame // out_width                            # frames 0, 1, 2 -> row 0; 3, 4, 5 -> row 1; ...
        col = frame % out_width                             # frames 0, 3, 6 -> column 0; ...
        window.set_xy((col - 0.5, row - 0.5))               # move the red square to this position
        shown[row, col] = out[row, col]
        picture.set_data(shown)
        labels[(row, col)].set_text(f"{out[row, col]:.0f}")
        ax_out.set_title(f"out[{row},{col}] = sum(patch * k) = {out[row, col]:.0f}")

    anim = FuncAnimation(fig, update, frames=out_height * out_width, interval=700)
    plt.close(fig)                                          # show only the animation, not an extra still picture
    HTML(anim.to_jshtml())
    """),
    md(r"""
    ### Lab · Convolution playground
    The lab applies a 3 x 3 kernel to a photo; you can pick classic kernels or type your own numbers,
    and hovering any pixel shows the patch, the kernel and the one output number they give.

    **Predict first:** the kernel above has `+1` in its left column and `-1` in its right column.
    Where the photo goes from dark (left) to bright (right), is its output positive or negative?
    In the lab, pick **sobel x** (its columns are the other way round), hover such an edge to
    check, then work through the "Try this" list under the lab (5 minutes).
    """),
    code("""
    labs.show("conv-playground")
    """),

    md(r"""
    ### Two more knobs before the exercise: padding and stride

    Exercise 1 also takes two settings that every convolution layer has. Section 4 comes back to them;
    here is what you need now.

    * **Padding** `pad`: before sliding, add `pad` rows and columns of zeros on every side of the input.
      With `pad=1` the 5 x 5 input becomes 7 x 7, and the output is 7 - 3 + 1 = 5 per axis: the same
      size as the input. Without padding every 3 x 3 layer would shrink the image by 2 pixels.
    * **Stride** `stride`: how many pixels the window moves between two positions. With `stride=2` on
      the 5 x 5 input, the window's top row can be 0 or 2 (at 4 it would hang over the edge), and the
      same for columns: a 2 x 2 output. It holds out[0, 0], out[0, 2], out[2, 0] and out[2, 2] of the
      stride-1 output above: `[[1, 0], [0, -2]]`.

    **Counting positions.** For an input of size n (padding included), a kernel of size k and a stride
    s, the window's first row can be 0, s, 2s, ... as long as the window still fits, that is as long as
    the first row is at most n - k. That gives

    $$\text{number of positions} = (n - k) \;//\; s + 1$$

    where `//` is Python's whole-number division (it rounds down) and the `+ 1` counts position 0.
    Check: stride 1: (5 - 3) // 1 + 1 = 3; stride 2: (5 - 3) // 2 + 1 = 2; `pad=1`, stride 1:
    (7 - 3) // 1 + 1 = 5.

    **Where a window starts.** With stride s, output row `row` reads the input from row `top = row * s`,
    and output column `col` from column `left = col * s`. For a 3 x 3 kernel, `bottom = top + 3` and
    `right = left + 3`, and Python's slice `x[top:bottom, left:right]` gives the 3 x 3 patch: rows top,
    top + 1, top + 2 (a slice stops before its end, so `bottom` is one past the last row). The
    next cell shows padding, both strides and the window of output (1, 1) at stride 2.
    """),
    api("np.sum"),
    code("""
    print("x with pad=1, shape", convops.pad2d(x, 1).shape)
    print(convops.pad2d(x, 1))                                    # a frame of zeros around x
    print("stride 2 -> output", convops.cross_correlate2d(x, k, stride=2).shape)
    print(convops.cross_correlate2d(x, k, stride=2))
    print("pad 1 -> output", convops.cross_correlate2d(x, k, pad=1).shape)
    print(convops.cross_correlate2d(x, k, pad=1))

    row = 1
    col = 1
    stride = 2
    top = row * stride                                            # first input row of this window: 2
    bottom = top + 3                                              # one past its last row: 5
    left = col * stride                                           # first input column: 2
    right = left + 3                                              # one past its last column: 5
    patch = x[top:bottom, left:right]                             # (3, 3)
    print(f"stride 2, output ({row}, {col}): the window starts at input row {top}, column {left}")
    print(patch)
    print("sum of patch * k:", np.sum(patch * k))
    """),
    md(r"""
    The window of output (1, 1) at stride 2 has left column 1, 3, 0 and right column 2, 1, 3:
    (1 + 3 + 0) - (2 + 1 + 3) = -2, the last number of the stride-2 output.

    ### Exercise 1 · `cross_correlate2d`

    **Goal.** Write the convolution of one channel, with stride and zero padding.

    **Why by hand.** PyTorch's `nn.Conv2d` does exactly this, much faster. Writing the loops once means
    that every size, parameter count and receptive field later in the course is something you can
    work out, not something to memorise.

    **Input.**
    * `x`: a NumPy float array (H, W), one channel;
    * `k`: the kernel, a NumPy array (kh, kw), for example 3 x 3;
    * `stride`: a whole number, at least 1 (default 1);
    * `pad`: a whole number, at least 0 (default 0).

    **Output.** A NumPy float array (out_height, out_width), with
    out_height = (H + 2·pad - kh) // stride + 1 and out_width likewise with W and kw.

    **Example.** With `x` and `k` from above:
    ```
    cross_correlate2d(x, k)             ->  [[ 1. -3.  0.]
                                             [-2.  0.  3.]
                                             [ 0. -1. -2.]]
    cross_correlate2d(x, k, stride=2)   ->  [[ 1.  0.]
                                             [ 0. -2.]]
    cross_correlate2d(x, k, pad=1)      ->  5 x 5, first row [-3. -1. -1.  1.  4.]
    ```

    **Plan.** The template already pads, measures, allocates and loops; the steps you write are in bold.
    1. Pad `x` with `convops.pad2d(x, pad)` and read the sizes of the padded input and of the kernel.
    2. **Compute `out_height` and `out_width`** with (n - k) // stride + 1, on the padded sizes.
    3. Make the output array, full of zeros, with `np.zeros`.
    4. Loop over every output `row`, and inside it over every output `col`. For each position:
       **the window's `top` and `bottom` rows and its `left` and `right` columns**, as in the cell
       above but with the kernel's size instead of 3; **the `patch`**, the slice of the padded input
       between them; **the `products`**, `patch * k` element by element; **the output value**, the
       sum of all products (`np.sum`).
    5. Return the output.

    **What the checker tests.** A random 9 x 11 input (9 rows, 11 columns, so that swapped height and
    width show) and a random 3 x 3 kernel, with four settings: stride 1 or 2, pad 0 or 1. It compares
    the shape and every value with `convops.cross_correlate2d`. For every setting that fails it prints
    `stride=..., pad=...: shape (yours) vs (expected), or values differ`: when the two shapes are equal,
    the values are wrong. The last line of the cell prints your result for the stride-2 example.

    **Common mistakes.**

    | you wrote | what you see |
    |---|---|
    | kept the line `raise NotImplementedError` | `NotImplementedError` at the end of the error message: delete that line |
    | left a `____` | `NameError: name '____' is not defined` |
    | `top = row` (no `* stride`) | `stride=2, pad=0: shape (4, 5) vs (4, 5), or values differ`, and the same for `stride=2, pad=1`; stride 1 passes. The stride-2 example gives `[[1, -3], [-2, 0]]`: neighbouring windows instead of every second one |
    | `bottom = top + k_height - 1` (and the same for `right`) | `ValueError: operands could not be broadcast together with shapes (2,2) (3,3)`: the slice stops *before* `bottom`, so the patch lost a row and a column |
    | `in_height - k_height + 1` (no stride in the size) | `ValueError: operands could not be broadcast together with shapes (3,0) (3,3)`: too many positions, so windows run past the edge of the input |
    | forgot the `+ 1` | `stride=1, pad=0: shape (6, 8) vs (7, 9), or values differ` and similar lines for all four settings: one row and one column short |
    | `/` instead of `//` | `TypeError: 'float' object cannot be interpreted as an integer`: `range` and `np.zeros` need whole numbers |
    | `patch @ k` (a matrix product) | `values differ` in all four settings, with equal shapes; on the example every output is 0 |
    | no `np.sum` in the output line | `ValueError: setting an array element with a sequence.`: a 3 x 3 array does not fit in one cell |
    | flipped the kernel first | `values differ` in all four settings; on the example every sign is reversed |
    | no `return out` | `stride=1, pad=0: shape () vs (7, 9), or values differ` and three more such lines |
    """),
    api("np.zeros", "np.sum", title="Functions you will need"),
    guided("""
    def cross_correlate2d(x, k, stride=1, pad=0):
        \"\"\"Slide the kernel k over the one-channel input x; one output number per window position.

        x: (H, W) float array.  k: (kh, kw) kernel.
        stride: how many pixels the window moves between positions.  pad: rows / columns of zeros added on every side.
        Returns a float array (out_height, out_width).
        \"\"\"
        #@raise

        # Step 1. Pad the input with zeros, then read the sizes (done for you).
        x_padded = convops.pad2d(x, pad)                   # (H + 2*pad, W + 2*pad)
        in_height, in_width = x_padded.shape               # measured AFTER padding
        k_height, k_width = k.shape                        # kernel height and width

        # Step 2. How many window positions fit along each axis.
        out_height = (in_height - k_height) // stride + 1  # YOUR CODE: window positions down the padded input
        out_width = (in_width - k_width) // stride + 1     # YOUR CODE: window positions across it

        # Step 3. The output, full of zeros for now (done for you).
        out = np.zeros((out_height, out_width))            # (out_height, out_width)

        # Step 4. Visit every output position, row by row, and fill in its number.
        for row in range(out_height):
            for col in range(out_width):
                top = row * stride                         # YOUR CODE: the first input row of this window
                bottom = top + k_height                    # YOUR CODE: one past its last row
                left = col * stride                        # YOUR CODE: the first input column of this window
                right = left + k_width                     # YOUR CODE: one past its last column
                patch = x_padded[top:bottom, left:right]   # YOUR CODE: the values under the window -> (k_height, k_width)
                products = patch * k                       # YOUR CODE: patch times kernel, element by element
                out[row, col] = np.sum(products)           # YOUR CODE: the sum of all the products
        return out

    tests.check_cross_correlate(cross_correlate2d)
    print(cross_correlate2d(x, k, stride=2))                # the example: [[1, 0], [0, -2]]
    """),
    md(r"""
    **Did it work?** `[PASS]` means your loops compute exactly what a convolution layer computes on one
    channel. The rest of the notebook uses the course's `convops.cross_correlate2d` (the same result),
    so the later cells also run while an exercise is unfinished.
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · A kernel is a pattern detector

    **Why.** Before deep learning, people designed kernels by hand to find edges or remove noise. A
    convolutional network learns its own kernels from data, and the first layer of a trained network
    looks remarkably like the classic ones (section 8 shows it). Knowing the classics tells you what the
    learned ones do.

    **Words.** An **edge** is a place where the brightness changes quickly: the outline of an object,
    the border of a shadow. To **blur** is to replace every pixel by an average of its neighbourhood,
    which removes fine detail and **noise** (random speckle).

    | kernel | its numbers | what its output says at each pixel |
    |---|---|---|
    | identity | 1 in the centre, 0 elsewhere | the pixel itself: output = input |
    | box blur | 1/9 everywhere | the average of the 3 x 3 neighbourhood |
    | gaussian | 1 2 1 / 2 4 2 / 1 2 1, divided by 16 | a weighted average in which nearer pixels count more: a softer blur |
    | sharpen | 5 in the centre, -1 above, below, left and right | the pixel plus how much it differs from its neighbours: edges get stronger |
    | sobel x | left column -1 -2 -1, right column +1 +2 +1 | right side minus left side: vertical edges, positive where it gets brighter to the right |
    | sobel y | top row -1 -2 -1, bottom row +1 +2 +1 | bottom minus top: horizontal edges |
    | laplacian | -4 in the centre, 1 above, below, left and right | neighbours minus centre: any sharp change, in any direction |
    | emboss | negative towards the top left, positive towards the bottom right | a diagonal edge detector; the photo looks carved in relief |

    **Worked example.** Two 3 x 3 patches: a step edge, dark on the left and bright on the right
    (columns 0, 0, 1), and a flat grey patch (0.5 everywhere). By hand, Sobel x on the edge:
    -1·0 + 0·0 + 1·1 for each row, weighted 1, 2, 1: 1 + 2 + 1 = 4. On the flat patch the -1 -2 -1 and
    the +1 +2 +1 cancel: 0. Every edge detector gives 0 where nothing changes; the blur keeps the grey.
    """),
    api("np.full"),
    code("""
    step_edge = np.array([[0, 0, 1],
                          [0, 0, 1],
                          [0, 0, 1]], dtype=float)       # dark on the left, bright on the right
    flat = np.full((3, 3), 0.5)                          # the same grey everywhere
    for name in ["box blur", "sobel x", "sobel y", "laplacian"]:
        kernel = convops.KERNELS[name]
        on_edge = np.sum(step_edge * kernel)
        on_flat = np.sum(flat * kernel)
        print(f"{name:<10} on the edge: {on_edge:5.2f}    on the flat patch: {on_flat:5.2f}")
    """),
    md(r"""
    Now all eight kernels on the whole grayscale photo. Top row: the kernel (blue = positive, red =
    negative). Bottom row: its output, with `pad=1` so it keeps the photo's size. For the edge detectors
    the grey scale runs from the 1st to the 99th **percentile** of the output (the values below which
    1 % and 99 % of the pixels lie), so that a few extreme pixels do not wash out the picture. The cell
    takes about 10 seconds: plain Python loops over 300 000 positions, eight times.
    """),
    api("np.percentile"),
    code("""
    # Figure: every classic kernel (top) and its output on the photo (bottom)
    names = list(convops.KERNELS)                        # the 8 kernel names, in order
    fig, axes = plt.subplots(2, len(names), figsize=(2.2 * len(names), 4.6))
    for col in range(len(names)):
        name = names[col]
        kernel = convops.KERNELS[name]                   # (3, 3)
        response = convops.cross_correlate2d(gray, kernel, pad=1)    # (600, 512): same size as the photo
        axes[0, col].imshow(kernel, cmap="RdBu", vmin=-2, vmax=2)
        viz.annotate_grid(axes[0, col], kernel, fmt="{:.2f}", fontsize=6)
        axes[0, col].set_title(name, fontsize=9)
        if name in ("identity", "box blur", "gaussian"):
            low, high = 0, 1                             # these keep the photo's brightness range
        else:
            low = np.percentile(response, 1)             # edge maps: clip the most extreme 1 % on each side
            high = np.percentile(response, 99)
        axes[1, col].imshow(response, cmap="gray", vmin=low, vmax=high)
        for ax in axes[:, col]:
            ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: *sobel x* and *sobel y* turn the same photo into two different "where are the
    edges" maps, one for vertical and one for horizontal edges (grey = no edge, white and black = the
    two directions of change). The *laplacian* lights up at any sharp change. The *gaussian* removes
    detail. A convolutional network's first layer is a learned version of this gallery; its deeper
    layers combine these responses into corners, textures, and eventually eyes or wheels.
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · Stride, padding, dilation and the output size

    **Why.** Every layer of a network changes the size of the image it receives, according to three
    settings. When a model stops with a shape error, this section's formula tells you why. Day 2's
    detectors read objects off feature maps 8, 16 and 32 times smaller than the photo; Day 3's U-Net
    shrinks the image and grows it back to the same size. Both are this arithmetic.

    You already know two of the settings from Exercise 1. The third:

    * **Dilation** `d` spreads the kernel's numbers `d` pixels apart, skipping `d - 1` pixels between
      them. A 3 x 3 kernel with dilation 2 covers a 5 x 5 area but still has only 9 weights: it sees
      further at the same cost (segmentation networks use this). Its **effective kernel size**, the
      width of the area it covers, is $k_{\text{eff}} = d\,(k - 1) + 1$: for k = 3, d = 2 that is
      2·2 + 1 = 5. With d = 1 (no dilation), k_eff = k.

    | setting | what it does | typical use |
    |---|---|---|
    | **padding** `p` | adds `p` rows / columns of zeros on every side | `p = (k-1)/2` keeps the size: **same padding** (1 for 3x3, 2 for 5x5); `p = 0` is called **valid** |
    | **stride** `s` | moves the kernel `s` pixels at a time | `s = 2` halves height and width: **downsampling**, making the feature map smaller |
    | **dilation** `d` | puts `d - 1` skipped pixels between the kernel's numbers | sees further at the same number of weights |

    The output size along one axis (height or width) is the number of positions from Exercise 1, with
    the padded size n + 2p and the effective kernel size:

    $$n_{\text{out}} = \left\lfloor \frac{n_{\text{in}} + 2p - k_{\text{eff}}}{s} \right\rfloor + 1$$

    The brackets ⌊ ⌋ mean "round down", which is `//` in Python.

    **Worked examples** (the first three are the panels of the next figure):

    | n | k | s | p | d | by hand | output |
    |---|---|---|---|---|---|---|
    | 8 | 3 | 1 | 0 | 1 | (8 + 0 - 3) // 1 + 1 | 6 |
    | 8 | 3 | 1 | 1 | 1 | (8 + 2 - 3) // 1 + 1 | 8 |
    | 8 | 3 | 2 | 1 | 1 | (8 + 2 - 3) // 2 + 1 = 7 // 2 + 1 = 3 + 1 | 4 |
    | 7 | 3 | 1 | 0 | 2 | k_eff = 5; (7 + 0 - 5) // 1 + 1 | 3 |
    """),
    api("np.arange"),
    code("""
    # Figure: an 8x8 input and its Sobel-x output in three settings: valid, same, and stride 2
    x8 = np.arange(64, dtype=float).reshape(8, 8) % 7        # an 8x8 test input with a repeating pattern (values 0 to 6)
    sobel_x = convops.KERNELS["sobel x"]
    settings = [("valid: p=0, s=1", 1, 0),
                ("same: p=1, s=1", 1, 1),
                ("stride 2: p=1, s=2", 2, 1)]                 # (title, stride, pad)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    axes[0].imshow(x8, cmap="Blues")
    axes[0].set_title("input 8x8")
    for i in range(3):
        title, stride, pad = settings[i]
        result = convops.cross_correlate2d(x8, sobel_x, stride, pad)
        axes[i + 1].imshow(result, cmap="Oranges")
        axes[i + 1].set_title(f"{title} -> {result.shape[0]}x{result.shape[1]}")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    The sizes in the titles, 6, 8 and 4, are the first three rows of the table.

    ### Lab · stride, padding and dilation
    **Predict first:** the playground's photo is 64 x 64. With a 3x3 kernel, stride 2 and no
    padding, is the output 32 x 32 or 31 x 31? Then run the cell: the lab starts in exactly that
    setting, with the formula box filled in. Try padding 1 and dilation 2 as well (2 minutes).
    """),
    code("""
    labs.show("conv-playground", stride=2, pad=0)
    """),
    md(r"""
    ### Exercise 2 · `conv_output_size`

    **Goal.** Turn the output-size formula into a function.

    **Why.** You will compute output sizes all week, mostly in your head: `torchkit.summarize` and
    every model's shape table are this function applied layer after layer.

    **Input.** Whole numbers: `n_in` (input height or width), `k` (kernel size), `stride` (default 1),
    `pad` (default 0), `dilation` (default 1).

    **Output.** A whole number (a Python `int`): the output size along that axis.

    **Example.** `conv_output_size(28, 7)` is 22: a 28 x 28 input and a 7 x 7 kernel, stride 1, no
    padding, give (28 - 7) // 1 + 1 = 22 (a quiz question of the old course). `conv_output_size(7, 3,
    dilation=2)` is 3, the last row of the table above.

    **Plan.**
    1. The effective kernel size `k_eff` from `k` and `dilation`.
    2. The padded input size `n_padded`: `pad` zeros on *both* sides.
    3. The number of positions `n_out`: whole steps of `stride` that fit in `n_padded - k_eff`, plus 1.
    4. Return `n_out` (in the template).

    **What the checker tests.** Six cases, `(n_in, k, stride, pad, dilation) -> expected`:
    (28, 7, 1, 0, 1) -> 22, (28, 3, 1, 1, 1) -> 28, (32, 3, 2, 1, 1) -> 16, (64, 5, 1, 0, 2) -> 56,
    (7, 7, 1, 0, 1) -> 1, (224, 7, 2, 3, 1) -> 112. For each wrong case it prints
    `n_in=..., k=..., stride=..., pad=..., dilation=...: got (yours), expected (right)`.

    **Common mistakes.**

    | you wrote | the checker prints |
    |---|---|
    | `k_eff = k` (dilation ignored) | `n_in=64, k=5, stride=1, pad=0, dilation=2: got 60, expected 56` |
    | `k_eff = dilation * k` | `n_in=64, k=5, stride=1, pad=0, dilation=2: got 55, expected 56` |
    | `n_in + pad` (padding on one side only) | `n_in=28, k=3, stride=1, pad=1, dilation=1: got 27, expected 28` and the same for the 224 case (111 instead of 112) |
    | `/` instead of `//` | `n_in=32, k=3, stride=2, pad=1, dilation=1: got 16.5, expected 16` and `got 112.5, expected 112` |
    | forgot the `+ 1` | all six cases one too small, for example `n_in=7, k=7, stride=1, pad=0, dilation=1: got 0, expected 1` |
    | `n_padded - k_eff // stride + 1` (no brackets) | `n_in=32, k=3, stride=2, pad=1, dilation=1: got 34, expected 16`: `//` is computed before `-` |
    """),
    guided("""
    def conv_output_size(n_in, k, stride=1, pad=0, dilation=1):
        \"\"\"Output size along one axis (height or width) of a convolution or pooling layer. Returns an int.\"\"\"
        #@raise

        # Step 1. The width of the area the (dilated) kernel covers.
        k_eff = dilation * (k - 1) + 1          # YOUR CODE: the effective kernel size, from k and dilation

        # Step 2. The input size after padding.
        n_padded = n_in + 2 * pad               # YOUR CODE: n_in plus `pad` zeros on each of the two sides

        # Step 3. How many window positions fit: the first one, plus one per whole stride step.
        n_out = (n_padded - k_eff) // stride + 1    # YOUR CODE: whole steps of `stride` that fit in n_padded - k_eff, plus 1
        return n_out

    tests.check_output_size(conv_output_size)
    print("28x28 input, 7x7 kernel ->", conv_output_size(28, 7))      # the example: 22
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · Multi-channel convolution and the cost of a layer

    **Why.** A colour photo has 3 channels, and inside a network the feature maps have tens to hundreds
    of channels. Every convolution layer of the course (the architectures of D1-03, the backbones of
    Days 2 and 3) combines all of them at once. The parameter count you derive here also explains why
    some of D1-03's architectures are much cheaper than others.

    **Words.**
    * A layer reads **C_in input channels** and writes **C_out output channels**.
    * One kernel of the layer is now a 3-D block of shape (C_in, kh, kw): one kh x kw slice per input
      channel. It is multiplied element by element with the (C_in, kh, kw) patch under the window, and
      **all** the products, over every channel and every position, are added up into one number.
    * The layer has C_out such kernels. Each one slides over the input and makes one output channel, so
      **every output channel sees every input channel**. Together the kernels form the **weight
      tensor** W of shape (C_out, C_in, kh, kw).
    * Each output channel also has a **bias**: one learned number added to every position of that
      channel's map, so that the layer can shift its output up or down. The biases form b of shape (C_out,).
    * The **parameter count** of the layer is therefore C_out · C_in · kh · kw + C_out. It does **not**
      depend on the image size, because the same kernel is used at every position: **weight sharing**.
    * Weight sharing makes the layer **translation equivariant**: shift the input by a few pixels and
      the output map shifts by the same amount, because the same kernel finds the same pattern
      wherever it is.
    * The *computation* does grow with the image. It is counted in **MACs** (multiply-accumulates: one
      multiplication and one addition, `a * b + c`; 1 MAC = 2 **FLOPs**, floating-point operations).
      Every output number needs C_in · kh · kw of them, so a layer costs
      MACs = C_out · H_out · W_out · C_in · kh · kw.

    **Worked example.** An input with 2 channels of 3 x 3, a layer with 2 output channels and 2 x 2
    kernels (the next cell holds the numbers):

    ```
    input channel 0     input channel 1        kernel 0: slice 0   slice 1    bias 0.5
    1  2  0             0  1  1                          1  0      0  1
    0  1  3             2  0  1                          0  1      1  0
    2  1  0             1  1  0                kernel 1: slice 0   slice 1    bias 0
                                                         1  1     -1 -1
                                                         1  1     -1 -1
    ```

    Output channel 0 at position (0, 0): channel 0's patch `1 2 / 0 1` times slice 0 gives 1 + 0 + 0 + 1
    = 2; channel 1's patch `0 1 / 2 0` times slice 1 gives 0 + 1 + 2 + 0 = 3; together 2 + 3 = 5, plus
    the bias 0.5: **5.5**. Output channel 1 at (0, 0): (1 + 2 + 0 + 1) - (0 + 1 + 2 + 0) = **1**. The
    output has shape (2, 2, 2): 2 channels of (3 - 2) // 1 + 1 = 2 positions per axis. The layer has
    2 · 2 · 2 · 2 + 2 = 18 parameters and costs 2 · 2 · 2 · 2 · 2 · 2 = 64 MACs.
    """),
    code("""
    x_multi = np.array([[[1, 2, 0], [0, 1, 3], [2, 1, 0]],          # input channel 0
                        [[0, 1, 1], [2, 0, 1], [1, 1, 0]]], dtype=float)   # input channel 1 -> (C_in=2, 3, 3)
    w_multi = np.array([[[[1, 0], [0, 1]], [[0, 1], [1, 0]]],        # kernel 0: one 2x2 slice per input channel
                        [[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]]], dtype=float)   # kernel 1 -> (C_out=2, C_in=2, 2, 2)
    b_multi = np.array([0.5, 0.0])                                    # one bias per output channel -> (2,)
    print("input", x_multi.shape, "| weights", w_multi.shape, "| bias", b_multi.shape)
    print(convops.conv2d(x_multi, w_multi, b_multi))                  # (C_out=2, 2, 2)
    """),
    md(r"""
    The diagram shows the same thing for a bigger layer: one (3, 3, 3) kernel meets a (3, 3, 3) patch of
    a 3-channel input and gives one number of output channel 0; the C_out kernels give C_out channels.
    """),
    code("""
    # Diagram: input block, the patch (red square), two of the C_out kernels, and the output block with one result pixel
    fig, ax = plt.subplots(figsize=(11, 3.4)); ax.axis("off")
    def block(x0, y0, w, h, depth, color, label):            # a stack of `depth` squares drawn like a 3-D block
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

    **Goal.** Write a whole convolution layer: several input channels, several output channels, bias,
    stride and padding (no dilation).

    **Why by hand.** It is Exercise 1 with one more loop and one more axis in the patch. After it,
    `nn.Conv2d` holds no secrets: its weight has the shape of `w` here and it computes exactly this.

    **Input.**
    * `x`: a NumPy float array (C_in, H, W);
    * `w`: the weights (C_out, C_in, kh, kw), one kernel per output channel;
    * `b`: the biases (C_out,), or `None` for a layer without bias;
    * `stride`, `pad`: as in Exercise 1 (the padding goes around height and width only).

    **Output.** A NumPy float array (C_out, out_height, out_width), with out_height and out_width as in
    Exercise 1.

    **Example.** The worked example: `conv2d(x_multi, w_multi, b_multi)` gives
    ```
    [[[5.5 6.5]
      [2.5 3.5]]

     [[1.  3. ]
      [0.  3. ]]]
    ```

    **Plan.** Steps 1 to 3 are Exercise 1 with a channel axis in front; the bold ones are yours.
    1. Pad `x` (height and width only) and read the sizes of the padded input and of the weights.
    2. **`out_height` and `out_width`**, exactly as in Exercise 1.
    3. Make the output, (C_out, out_height, out_width), full of zeros.
    4. Loop over the output channels. For each one: **its `kernel`**, `w[out_channel]`, of shape
       (C_in, kh, kw). Then loop over rows and columns as in Exercise 1 (the window's `top`, `bottom`,
       `left` and `right` are written for you): **the `patch`** now takes *all* input channels,
       `x_padded[:, rows, columns]`, shape (C_in, kh, kw); **the `products`** with the kernel; **the
       output value**, the sum of all of them.
    5. After the position loops, if there is a bias: **add `b[out_channel]` to the whole output channel**
       (one line; NumPy adds a single number to every element of an array).
    6. Return the output.

    **What the checker tests.** A random input (3, 8, 8), random weights (4, 3, 3, 3) and a random bias
    (4,), with stride 1 / pad 0, stride 1 / pad 1 and stride 2 / pad 1, compared with `convops.conv2d`.
    For each failing setting it prints `stride=..., pad=...: got shape (yours), expected (expected)`.
    **If the two shapes are equal, your values are wrong** (the message does not say so).

    **Common mistakes.**

    | you wrote | what you see |
    |---|---|
    | the patch from one channel only, `x_padded[0, ...]` | `stride=1, pad=0: got shape (4, 6, 6), expected (4, 6, 6)` and two more such lines: equal shapes, wrong values |
    | `kernel = w[:, out_channel]` | `ValueError: operands could not be broadcast together with shapes (3,3,3) (4,3,3)`: the first axis of `w` is the output channel |
    | forgot the bias | equal shapes, wrong values; the example gives 5.0 instead of 5.5 |
    | added the bias inside the row / column loops | equal shapes, wrong values; the example's channel 0 gives `[[7, 7.5], [3, 3.5]]`: the bias was added again at every position |
    | `in_height + 2 * pad - k_height` (padding counted twice) | `ValueError: operands could not be broadcast together with shapes (3,3,2) (3,3,3)`: one position too many, the last window is cut off |
    """),
    guided("""
    def conv2d(x, w, b=None, stride=1, pad=0):
        \"\"\"A convolution layer with several input and output channels (no dilation).

        x: (C_in, H, W) input.  w: (C_out, C_in, kh, kw), one kernel per output channel.
        b: (C_out,) one bias per output channel, or None.
        Returns a float array (C_out, out_height, out_width).
        \"\"\"
        #@raise

        # Step 1. Pad height and width, then read the sizes (done for you).
        x_padded = convops.pad2d(x, pad)                   # (C_in, H + 2*pad, W + 2*pad)
        c_in, in_height, in_width = x_padded.shape
        c_out, _, k_height, k_width = w.shape              # the _ is C_in again

        # Step 2. How many window positions fit, as in Exercise 1.
        out_height = (in_height - k_height) // stride + 1  # YOUR CODE: window positions down the padded input
        out_width = (in_width - k_width) // stride + 1     # YOUR CODE: window positions across it

        # Step 3. The output, full of zeros for now (done for you).
        out = np.zeros((c_out, out_height, out_width))     # (C_out, out_height, out_width)

        # Step 4. One output channel at a time; inside it, every position as in Exercise 1.
        for out_channel in range(c_out):
            kernel = w[out_channel]                                    # YOUR CODE: its kernel -> (C_in, k_height, k_width)
            for row in range(out_height):
                for col in range(out_width):
                    top = row * stride                                 # the window, as in Exercise 1
                    bottom = top + k_height
                    left = col * stride
                    right = left + k_width
                    patch = x_padded[:, top:bottom, left:right]        # YOUR CODE: ALL channels under the window
                    products = patch * kernel                          # YOUR CODE: element by element -> (C_in, k_height, k_width)
                    out[out_channel, row, col] = np.sum(products)      # YOUR CODE: the sum over channels and positions
            # Step 5. The bias: one number added to every position of this output channel.
            if b is not None:
                out[out_channel] = out[out_channel] + b[out_channel]   # YOUR CODE: this channel plus its bias
        return out

    tests.check_conv2d(conv2d)
    print(conv2d(x_multi, w_multi, b_multi))                    # the example: [[[5.5 6.5] [2.5 3.5]] [[1 3] [0 3]]]
    """),
    md(r"""
    **Did it work?** `[PASS]` means your function is a complete convolution layer: given the weights of
    a trained `nn.Conv2d`, it would compute the same numbers.

    ### Exercise 4 · count the parameters

    **Goal.** The number of parameters of `nn.Conv2d(c_in, c_out, k, bias=bias)`: weights plus biases.

    **Why.** The parameter count decides how much memory a model needs and, roughly, how much data it
    needs to train. D1-03 compares architectures with it.

    **Input.** Whole numbers `c_in`, `c_out`, `k` (a square k x k kernel), and `bias`, `True` or `False`.
    **Output.** A whole number.

    **Example.** The worked example's layer: `count_params(2, 2, 2)` = 2 · 2 · 2 · 2 + 2 = 18. The first
    layer of the network in section 8, `count_params(1, 32, 3)` = 32 · 1 · 3 · 3 + 32 = 320.

    **Plan.** 1. The weights of one kernel: c_in slices of k x k. 2. The weights of the layer: c_out
    kernels. 3. The biases: one per output channel if `bias` is true, else 0 (the `if` is written for
    you). 4. Return weights plus biases.

    **What the checker tests.** (3, 64, 7) with bias -> 9 472; (64, 64, 3) with bias -> 36 928;
    (256, 64, 1) without bias -> 16 384. It prints only `[FAIL]  conv parameter count` and a hint, so
    the cell's last line prints your three answers.

    **Common mistakes** (your answers for the three cases):

    | you wrote | your answers instead of 9472, 36928, 16384 |
    |---|---|
    | forgot `c_in` | 3200, 640, 64 |
    | `k` instead of `k * k` | 1408, 12352, 16384 |
    | one bias for the whole layer | 9409, 36865, 16384 |
    | biases counted when `bias` is `False` | 9472, 36928, 16448 |
    """),
    guided("""
    def count_params(c_in, c_out, k, bias=True):
        \"\"\"Number of parameters of nn.Conv2d(c_in, c_out, k, bias=bias): weights plus biases.\"\"\"
        #@raise

        weights_per_kernel = c_in * k * k       # YOUR CODE: one kernel has c_in slices of k x k weights
        n_weights = c_out * weights_per_kernel  # YOUR CODE: the layer has c_out kernels
        if bias:
            n_biases = c_out                    # YOUR CODE: one bias per output channel
        else:
            n_biases = 0
        return n_weights + n_biases

    tests.check_count_params(count_params)
    print(count_params(3, 64, 7), count_params(64, 64, 3), count_params(256, 64, 1, bias=False))   # 9472 36928 16384
    """),
    md(r"""
    The same formula for a few famous first layers. AlexNet (2012), VGG (2014) and ResNet (2015) are
    landmark networks that D1-03 builds; the cell uses the course's `convops.conv_params`, the same
    formula as yours. AlexNet's single 11 x 11 layer has about 20 times the parameters of VGG's 3 x 3
    one. ResNet's 1 x 1 layer turns 256 channels into 64 with 16 448 parameters; a 3 x 3 layer doing
    the same would need 9 times as many weights.
    """),
    code("""
    layers = [("VGG conv1 3->64 3x3", 3, 64, 3),
              ("AlexNet conv1 3->96 11x11", 3, 96, 11),
              ("ResNet bottleneck 256->64 1x1", 256, 64, 1),
              ("VGG conv 512->512 3x3", 512, 512, 3)]          # (name, c_in, c_out, k)
    for layer in layers:
        name, c_in, c_out, k_size = layer
        n_params = convops.conv_params(c_in, c_out, k_size)
        print(f"{name:32s} {n_params:>10,d} params")
    """),
    md(r"""
    ### How libraries compute it: `im2col` and one matrix product (optional)

    The loops above are for understanding; they are slow. Libraries turn the whole layer into **one
    matrix multiplication**, which processors (GPUs above all) are built to do fast. **im2col**
    ("image to columns") copies every patch, flattened, into one row of a big matrix of shape
    (H_out · W_out, C_in · kh · kw). The kernels, flattened, form a (C_out, C_in · kh · kw) matrix. The
    product of the two gives every output number of every channel at once: the same numbers as the
    loops. The next cell times both on a 16-channel 64 x 64 input with 32 kernels of 3 x 3.

    <details>
    <summary><b>More: why the matrix product gives the same numbers</b> (optional, click to open)</summary>

    Row r of the im2col matrix is the patch at output position r, flattened into C_in · kh · kw numbers.
    Row o of the weight matrix is kernel o, flattened the same way. Their dot product (multiply element
    by element, add up) is exactly `np.sum(patch * kernel)`: output channel o at position r. The matrix
    product computes all these dot products in one call, in optimised code instead of Python loops.
    GPU libraries such as NVIDIA's cuDNN go further with other tricks, but the result is the same.

    </details>
    """),
    api("np.random.default_rng", "np.abs"),
    code("""
    import time
    rng = np.random.default_rng(0)
    x_big = rng.normal(size=(16, 64, 64))           # (C_in, H, W)
    w_big = rng.normal(size=(32, 16, 3, 3))         # (C_out, C_in, kh, kw)
    b_big = rng.normal(size=32)                     # (C_out,)

    start = time.perf_counter()
    out_loops = convops.conv2d(x_big, w_big, b_big, 1, 1)           # the loops of Exercise 3
    time_loops = time.perf_counter() - start

    start = time.perf_counter()
    out_matmul = convops.conv2d_im2col(x_big, w_big, b_big, 1, 1)   # im2col + one matrix product
    time_matmul = time.perf_counter() - start

    largest_difference = np.abs(out_loops - out_matmul).max()
    print(f"loops: {time_loops:.2f} s   im2col + matrix product: {time_matmul:.3f} s   largest difference: {largest_difference:.1e}")
    patches = convops.im2col(x_big, 3, 3, 1, 1)
    print("im2col matrix:", patches.shape, "= (H_out * W_out, C_in * kh * kw);  weight matrix:", w_big.reshape(32, -1).shape)
    """),
    md(r"""
    The largest difference, a few times 1e-14, is rounding noise: the same result. The matrix product
    is tens of times faster here, and the gap grows on a GPU.
    """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · Receptive fields

    **Why.** A number in a feature map can only react to the input pixels it depends on. To recognise
    a dog's face, some unit must see the whole face. Day 2's detectors assign objects of different sizes
    to feature maps whose units see that size.

    **Words.**
    * A **unit** (or neuron) is one number of a feature map: one position of one channel.
    * Its **receptive field** is the square of input pixels that can influence it; r is its width.
      After one 3 x 3 layer, r = 3.
    * Between two convolution layers, a network applies an **activation function** to every number.
      The usual one is **ReLU** (rectified linear unit): negative numbers become 0, the others pass
      unchanged, ReLU(x) = max(0, x). Without it, two convolutions in a row could be rewritten as one
      bigger convolution, so stacking layers would add nothing; with it, each layer adds a decision
      ("is this pattern there or not?").
    * **Max pooling** (section 7) keeps the largest value of each 2 x 2 window and halves height and
      width. For the receptive field it counts as a layer with k = 2 and stride 2.
    * The **jump** j of a layer is the distance, in input pixels, between two neighbouring units of its
      output. It starts at 1 and is multiplied by every stride.

    **Two layers.** Stack a second 3 x 3 layer on the first: each of its units reads a 3 x 3 window of
    first-layer units, and each of those reads its own 3 x 3 window of the input. Together they cover
    **5 x 5** input pixels. Layer by layer:

    $$r_{l} = r_{l-1} + (k_{l} - 1)\, j_{l-1}, \qquad j_{l} = j_{l-1} \cdot s_{l}$$

    In words: a layer with kernel k looks at k neighbouring units of the layer below; those are
    $j_{l-1}$ input pixels apart, so the k of them span $(k - 1)\,j_{l-1}$ more pixels than one unit
    alone. Note that the r-update uses the jump *before* this layer's stride.

    **Worked example.** conv 3x3 with stride 2, then conv 3x3 with stride 1:

    | layer | k | s | r = r + (k - 1) · j | j = j · s |
    |---|---|---|---|---|
    | input | | | 1 | 1 |
    | conv 3x3, stride 2 | 3 | 2 | 1 + 2 · 1 = 3 | 1 · 2 = 2 |
    | conv 3x3, stride 1 | 3 | 1 | 3 + 2 · 2 = 7 | 2 · 1 = 2 |

    The stride of the first layer makes the second layer's 3 x 3 kernel reach 7 pixels instead of 5.
    This is *why* a unit in a late layer can respond to a whole face although every layer only looks
    at 3 x 3 neighbourhoods.
    """),
    code("""
    # Figure: two 3x3 layers (left) and one 5x5 layer (right) give the red unit the same 5x5 receptive field
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    stacks = [[(3, 1, 0), (3, 1, 0)], [(5, 1, 0)]]          # (kernel, stride, pad) of each layer
    titles = ["two 3x3 layers (18 weights)", "one 5x5 layer (25 weights)"]
    for i in range(2):
        ax = axes[i]
        ax.imshow(np.zeros((7, 7)), cmap="Greys", vmin=0, vmax=1)                 # a 7x7 input grid
        ax.add_patch(Rectangle((0.5, 0.5), 5, 5, fc="#cfe0f5", ec="#4C78A8", lw=2))   # blue: the receptive field
        if len(stacks[i]) == 2:
            ax.add_patch(Rectangle((1.5, 1.5), 3, 3, fc="#f8d0b8", ec="#F58518", lw=2))   # orange: the 3x3 window of the middle layer
        ax.add_patch(Rectangle((2.5, 2.5), 1, 1, fc="#E45756", ec="k"))           # red: one unit of the last layer
        rf = convops.receptive_field(stacks[i])[-1]["rf"]
        ax.set_title(f"{titles[i]}\\nreceptive field = {rf}x{rf}")
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    The red square is one unit of the last layer. Left: it reads the orange 3 x 3 window of the layer
    below, and each of those nine units reads its own 3 x 3 window of the input; together they cover
    the blue 5 x 5 square. Right: one 5 x 5 kernel reads the same square directly. Same reach, but the
    two 3 x 3 layers have 2 · 9 = 18 weights instead of 25 (for one input and one output channel;
    28 % fewer), and a ReLU between them. That is why modern networks use stacks of 3 x 3 layers.

    ### Lab · Receptive field explorer
    **Predict first:** stack conv 3x3, then maxpool 2, then conv 3x3. Is the receptive field of
    one unit of the last layer 7, 8 or 9 pixels wide? Build the stack in the lab (**clear**, then
    add the three layers), click a top unit to check, then work through the "Try this" list (5 minutes).
    """),
    code("""
    labs.show("receptive-field")
    """),
    md(r"""
    ### Exercise 5 · `receptive_field`

    **Goal.** Compute the receptive field and the jump after every layer of a stack.

    **Why.** The receptive field tells you whether a network can see an object of a given size at all;
    the jump is the stride of a feature map, a number Day 2 uses constantly.

    **Input.** `layers`: a list of `(kernel, stride, pad)` tuples, from the input to the output. Padding
    moves the receptive field but does not change its size, so `pad` is not used.

    **Output.** A list with one dict per layer, `{"rf": r, "jump": j}`: the r and j columns of the
    table above (and of the lab's table).

    **Example.** The lab's stack, conv 3x3, maxpool 2, conv 3x3:
    ```
    receptive_field([(3, 1, 0), (2, 2, 0), (3, 1, 0)])
    -> [{'rf': 3, 'jump': 1}, {'rf': 4, 'jump': 2}, {'rf': 8, 'jump': 2}]
    ```
    By hand: 1 + 2 · 1 = 3; then 3 + 1 · 1 = 4 and the jump becomes 2; then 4 + 2 · 2 = 8.

    **Plan.** Start with r = 1 and j = 1 (in the template). For every layer: **update r** with the
    formula, using the jump of the layer below; **then update j**; append the dict (in the template).

    **What the checker tests.** The stack conv, conv, pool, conv, conv, pool, conv:
    `[(3, 1, 1), (3, 1, 1), (2, 2, 0), (3, 1, 1), (3, 1, 1), (2, 2, 0), (3, 1, 1)]`. It reads `"rf"` of
    the last dict and expects 24. On a wrong answer it prints only
    `[FAIL]  receptive_field` and the hint `expected 24: ...`, so the cell's last line prints your list
    for the lab's stack.

    **Common mistakes.**

    | you wrote | the checker's stack gives | the lab's stack gives |
    |---|---|---|
    | the jump updated before r | 27 | 9 |
    | `rf + (kernel - 1)` (forgot the jump) | 13 | 6 |
    | `rf + kernel * jump` | 37 | 12 |
    | no `append` | `IndexError: list index out of range` | `[]` |
    """),
    guided("""
    def receptive_field(layers):
        \"\"\"Receptive field and jump after every layer of a stack.

        layers: list of (kernel, stride, pad) tuples, from the input to the output.
        Returns a list with one dict {"rf": ..., "jump": ...} per layer.
        \"\"\"
        #@raise

        rf = 1                 # before any layer, a "unit" is one input pixel: it sees 1 pixel
        jump = 1               # and neighbouring pixels are 1 pixel apart
        results = []
        for layer in layers:
            kernel, stride, pad = layer                # pad is not needed: it moves the field, it does not resize it
            rf = rf + (kernel - 1) * jump              # YOUR CODE: this layer adds (kernel - 1) steps of `jump` input pixels
            jump = jump * stride                       # YOUR CODE: the stride multiplies the distance between neighbouring units
            results.append({"rf": rf, "jump": jump})
        return results

    tests.check_receptive_field(receptive_field)
    print(receptive_field([(3, 1, 0), (2, 2, 0), (3, 1, 0)]))     # the lab's stack: rf 3, 4, 8
    """),
    md(r"""
    Now a real network: **VGG-16** (2014) has 13 convolution layers of 3 x 3 in five blocks, each block
    ending with a 2 x 2 max pooling, on 224 x 224 photos. The cell builds its list of layers and plots
    the receptive field after each one (with `convops.receptive_field`, the same as yours).
    """),
    code("""
    # VGG-16's layers as (kernel, stride, pad): blocks of 2, 2, 3, 3 and 3 convolutions, each followed by a pool
    vgg_names = []
    vgg_layers = []
    for n_convs in [2, 2, 3, 3, 3]:
        for _ in range(n_convs):
            vgg_names.append("conv3")
            vgg_layers.append((3, 1, 1))
        vgg_names.append("pool")
        vgg_layers.append((2, 2, 0))
    results = convops.receptive_field(vgg_layers)

    rf_values = []
    for result in results:
        rf_values.append(result["rf"])

    # Figure: the receptive field after every layer, with the jump written at every pool
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(rf_values, marker="o")
    ax.set_xticks(range(len(vgg_names)))
    ax.set_xticklabels(vgg_names, rotation=60, fontsize=8)
    ax.set_ylabel("receptive field (px)")
    ax.set_title("Receptive field through a VGG-16 style stack (input 224)")
    ax.grid(alpha=0.3)
    for i in range(len(vgg_names)):
        if vgg_names[i] == "pool":
            ax.annotate(f"jump {results[i]['jump']}", (i, rf_values[i]), textcoords="offset points",
                        xytext=(0, 8), fontsize=7, ha="center")
    plt.tight_layout(); plt.show()
    print("final receptive field:", rf_values[-1], "px  (input is 224 px)")
    """),
    md(r"""
    Each pool doubles the jump, so every 3 x 3 layer after it adds twice as many pixels as before: the
    slope of the curve doubles after every pool. Thirteen 3 x 3 layers and five pools end at 212 px,
    so a unit of the last block sees almost the whole 224 px image.
    """),

    # ------------------------------------------------------------------ 7
    md(r"""
    ## 7 · Pooling, global average pooling, 1x1 convolutions

    ### Pooling

    **What it is.** **Pooling** replaces every k x k window of a feature map by one number: its largest
    value (**max pooling**) or its mean (**average pooling**). Each channel is pooled on its own, so the
    number of channels stays the same. Pooling has no weights: nothing to learn, zero parameters.

    **Why networks pool.**
    1. **Smaller maps, cheaper layers.** A 2 x 2 pool with stride 2 halves height and width, so the next
       layer has 4 times fewer positions to compute.
    2. **Further reach.** After the pool the jump doubles (section 6), so every later 3 x 3 layer sees
       twice as far: deeper layers can see whole objects.
    3. **Tolerance to small shifts.** Max pooling reports "the pattern is somewhere in this 2 x 2
       window", not exactly where; moving the input by one pixel often leaves the output unchanged. This
       is (local) **translation invariance**: the output stays the same when the input moves a little.
       Compare the convolution's *equivariance* (section 5), where the output moves with the input.

    Many modern networks downsample with stride-2 convolutions instead (their weights are learned), but
    max pooling is still common, for example right at the start of ResNet (D1-03).

    **Output size.** The windows move like a convolution's, so the formula of section 4 applies,
    usually with no padding: (n - k) // stride + 1. With the usual k = 2, stride 2: (n - 2) // 2 + 1,
    which is n / 2 for an even n.

    **Worked example.** A 4 x 4 feature map (one channel), pooled with k = 2, stride 2. The four windows
    do not overlap:

    ```
    1  3 | 2  0          max pooling            average pooling
    4  6 | 1  1          6  2                   (1+3+4+6)/4 = 3.5   (2+0+1+1)/4 = 1.0
    -----+-----          3  8                   (0+2+3+1)/4 = 1.5   (5+7+8+2)/4 = 5.5
    0  2 | 5  7
    3  1 | 8  2
    ```

    Top-left window {1, 3, 4, 6}: max 6, mean 14 / 4 = 3.5. Output size (4 - 2) // 2 + 1 = 2 per axis.
    With k = 3 and stride 1 the windows overlap: (4 - 3) // 1 + 1 = 2 positions per axis, max pooling
    gives `[[6, 7], [8, 8]]`.
    """),
    code("""
    pool_example = np.array([[[1, 3, 2, 0],
                              [4, 6, 1, 1],
                              [0, 2, 5, 7],
                              [3, 1, 8, 2]]], dtype=float)       # (C=1, 4, 4): one channel
    print("max pooling, k=2, stride 2:\\n", convops.pool2d(pool_example, 2, 2, "max"))
    print("average pooling, k=2, stride 2:\\n", convops.pool2d(pool_example, 2, 2, "avg"))
    print("max pooling, k=3, stride 1:\\n", convops.pool2d(pool_example, 3, 1, "max"))
    """),
    md(r"""
    On the photo: a 2 x 2 max pool halves each side and is hard to tell from the original; an 8 x 8
    pool keeps one number per 8 x 8 block. With 8 x 8 windows, max pooling keeps every small bright
    detail and makes it bigger (the stars of the flag, the cap badge, the collar), while average
    pooling gives a smooth, blurred copy.
    """),
    code("""
    # Figure: the grayscale photo, then max pooling 2x2 and 8x8, then average pooling 8x8
    max2 = convops.pool2d(gray, 2, 2, "max")
    max8 = convops.pool2d(gray, 8, 8, "max")
    avg8 = convops.pool2d(gray, 8, 8, "avg")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    axes[0].imshow(gray, cmap="gray"); axes[0].set_title(f"input {gray.shape}")
    axes[1].imshow(max2, cmap="gray"); axes[1].set_title(f"max pool 2x2 -> {max2.shape}")
    axes[2].imshow(max8, cmap="gray"); axes[2].set_title(f"max pool 8x8 -> {max8.shape}")
    axes[3].imshow(avg8, cmap="gray"); axes[3].set_title(f"avg pool 8x8 -> {avg8.shape}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Exercise 6 · `pool2d`

    **Goal.** Max or average pooling of every channel of a stack of feature maps.

    **Why by hand.** It is the convolution loop of Exercise 1 with "multiply by a kernel and add up"
    replaced by "take the largest value" or "take the mean". Writing it shows that pooling has no
    weights at all.

    **Input.**
    * `x`: a NumPy float array (C, H, W), one feature map per channel;
    * `k`: the window size, a whole number (k x k windows; default 2);
    * `stride`: the step between windows (default 2);
    * `mode`: the string `"max"` (largest value) or `"avg"` (mean).

    **Output.** A NumPy float array (C, out_height, out_width), with out_height = (H - k) // stride + 1
    and out_width likewise.

    **Example.** `pool_example` from above, shape (1, 4, 4):
    ```
    pool2d(pool_example, 2, 2, "max")   ->  [[[6. 2.]
                                              [3. 8.]]]
    pool2d(pool_example, 2, 2, "avg")   ->  [[[3.5 1. ]
                                              [1.5 5.5]]]
    ```

    **Plan.** The template reads the sizes, makes the output and writes the three loops.
    1. **`out_height` and `out_width`**: the number of window positions, (n - k) // stride + 1.
    2. For every channel, row and column: **the window's `top`, `bottom`, `left` and `right`**, as in
       Exercise 1 with the window size k; **the `window`**, the k x k values of *this channel* between
       them, `x[channel, rows, columns]`; then, depending on `mode`, **its largest value** (`np.max`)
       or **its mean** (`np.mean`) goes into `out[channel, row, col]`.
    3. Return the output (in the template).

    **What the checker tests.** A random input of shape (2, 8, 8), pooled with k = 2 and stride 2, once
    with `"max"` and once with `"avg"`, compared with `convops.pool2d`. It prints only `[FAIL]  pool2d`
    and the hint, so the cell's last two lines print your result on the example.

    **Common mistakes.**

    | you wrote | what you see |
    |---|---|
    | `top = row` (no `* stride`) | `[FAIL]`; the example's max gives `[[6, 6], [6, 6]]`: overlapping windows instead of the four separate ones |
    | `in_height - k + 1` (no stride in the size) | `ValueError: zero-size array to reduction operation maximum which has no identity`: windows past the edge are empty |
    | `x[:, top:bottom, left:right]` (all channels at once) | `[FAIL]`: every channel gets the maximum over all channels. The one-channel example cannot show it |
    | `if mode == "mean":` | `[FAIL]`: the checker passes `"avg"`, which then falls into the max branch; the example's average gives `[[6, 2], [3, 8]]` |
    | `np.sum` instead of `np.mean` | `[FAIL]`; the example's average gives `[[14, 4], [6, 22]]` |
    | called it on a 2-D array (H, W) | `ValueError: not enough values to unpack (expected 3, got 2)`: give (C, H, W), like `pool_example` |
    """),
    api("np.max", "np.mean", title="Functions you will need"),
    guided("""
    def pool2d(x, k=2, stride=2, mode="max"):
        \"\"\"Max or average pooling of every channel of x.

        x: (C, H, W) feature maps.  k: window size (k x k).  stride: step between windows.
        mode: "max" keeps the largest value of each window, "avg" its mean.
        Returns a float array (C, out_height, out_width).
        \"\"\"
        #@raise

        # Step 1. Read the sizes (done for you). Pooling uses no padding here.
        n_channels, in_height, in_width = x.shape

        # Step 2. How many windows fit along each axis: the same count as for a convolution.
        out_height = (in_height - k) // stride + 1         # YOUR CODE: window positions down the input
        out_width = (in_width - k) // stride + 1           # YOUR CODE: window positions across it

        # Step 3. The output, full of zeros for now (done for you).
        out = np.zeros((n_channels, out_height, out_width))    # (C, out_height, out_width)

        # Step 4. Every channel on its own; in each channel, every output position.
        for channel in range(n_channels):
            for row in range(out_height):
                for col in range(out_width):
                    top = row * stride                         # YOUR CODE: the first input row of this window
                    bottom = top + k                           # YOUR CODE: one past its last row
                    left = col * stride                        # YOUR CODE: the first input column of this window
                    right = left + k                           # YOUR CODE: one past its last column
                    window = x[channel, top:bottom, left:right]    # YOUR CODE: this channel's values under it -> (k, k)
                    if mode == "max":
                        out[channel, row, col] = np.max(window)    # YOUR CODE: the largest value in the window
                    else:
                        out[channel, row, col] = np.mean(window)   # YOUR CODE: the mean of the window
        return out

    tests.check_pool2d(pool2d)
    print(pool2d(pool_example, 2, 2, "max"))       # the example: [[[6 2] [3 8]]]
    print(pool2d(pool_example, 2, 2, "avg"))       # the example: [[[3.5 1] [1.5 5.5]]]
    """),
    md(r"""
    **Did it work?** `[PASS]` means you have written PyTorch's `nn.MaxPool2d` and `nn.AvgPool2d` for one
    image. The library versions also take a batch, (B, C, H, W), and run the same loops for every image.

    ### Global average pooling

    **Global average pooling** (GAP) averages each channel over *all* its positions: (C, H, W) becomes
    (C,), one number per channel. By hand, a channel `[[1, 3], [5, 7]]` becomes (1 + 3 + 5 + 7) / 4 = 4.

    **Why.** At the end of a classifier, the last feature maps must become one vector of numbers for the
    final layer. Older networks **flattened** them, writing all C · H · W numbers one after the other,
    and fed that vector to a **dense layer**, also called a **fully connected layer**: D0-00's linear
    layer (`nn.Linear`) with many inputs and outputs, every output connected to every input. VGG's first
    dense layer alone has 7 · 7 · 512 · 4096 + 4096 = 102 764 544 parameters. GAP has none, and its
    output has C numbers whatever the image size, so the same network accepts photos of any size.
    Section 8 compares both on a small network. The GAP of the colour photo, one mean per channel:
    """),
    code("""
    print("global average pooling of the RGB image -> one number per channel:", img.mean(axis=(0, 1)))
    """),
    md(r"""
    ### 1x1 convolutions

    A **1x1 convolution** has a kernel of size 1 x 1: it does not look at neighbours at all. At every
    pixel it computes, for each output channel, a weighted sum of that pixel's input channels (plus a
    bias): the same small linear layer, C_in -> C_out, applied to every pixel.

    **Uses.** Change the number of channels cheaply (256 -> 64 in ResNet's bottleneck, D1-03; 16 448
    parameters, as computed above); mix channels after a layer that treated them separately (MobileNet,
    D1-03); and make a prediction at every pixel (the last layers of Day 2's detectors and of Day 3's
    segmentation networks).

    **Worked example.** Two output channels on an RGB pixel (R, G, B) = (0.8, 0.4, 0.2), without bias:
    * output channel 0, weights (0.299, 0.587, 0.114), the grayscale formula of section 1:
      0.299 · 0.8 + 0.587 · 0.4 + 0.114 · 0.2 = 0.2392 + 0.2348 + 0.0228 = 0.4968;
    * output channel 1, weights (1, -1, 0): 0.8 - 0.4 = 0.4, "red minus green".

    The layer has 2 · 3 · 1 · 1 = 6 weights. The cell applies it to every pixel of the photo.
    """),
    api("Tensor.transpose"),
    code("""
    # A 1x1 convolution is a per-pixel linear layer across channels: here 3 -> 2 channels
    x_chw = img.transpose(2, 0, 1)                      # (3, H, W): channels first
    w_1x1 = np.array([[[[0.299]], [[0.587]], [[0.114]]],    # output channel 0 = grayscale (luminance)
                      [[[1.0]],   [[-1.0]],  [[0.0]]]])     # output channel 1 = red minus green -> (2, 3, 1, 1)
    mixed = convops.conv2d_im2col(x_chw, w_1x1, None, 1, 0)   # (2, H, W): same height and width

    # Figure: the photo and the two output channels of the 1x1 convolution
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    axes[0].imshow(img); axes[0].set_title("input (3 channels)")
    axes[1].imshow(mixed[0], cmap="gray"); axes[1].set_title("1x1 conv, out ch 0: 0.3R+0.6G+0.1B")
    axes[2].imshow(mixed[1], cmap="RdBu"); axes[2].set_title("1x1 conv, out ch 1: R - G")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout(); plt.show()
    print("weights shape:", w_1x1.shape, "-> parameters:", w_1x1.size, "(no spatial extent, only channel mixing)")
    """),

    # ------------------------------------------------------------------ 8
    md(r"""
    ## 8 · The convnet pattern

    **Why.** Everything so far assembles into a **classifier**: a network that gives one score per class
    (D0-00, section 5). Every convolutional network since LeNet (1998), the first successful one, has the
    same shape: blocks of convolution -> ReLU -> pooling, in which **height and width shrink while the
    number of channels grows**; then global average pooling and a final linear layer (**fc**, "fully
    connected") with one output per class. Early layers detect edges and colours (generic); late layers,
    whose receptive fields are large, detect parts of objects (specific).

    **Worked example.** The tiny network of this section, for 28 x 28 grayscale images with 10 classes,
    by hand. Every convolution is 3 x 3 with padding 1 (the size stays), every pool 2 x 2 with stride 2
    (the size halves):

    | layer | output shape | parameters |
    |---|---|---|
    | input | (1, 28, 28) | |
    | conv 1 -> 32 | (32, 28, 28) | 32 · 1 · 9 + 32 = 320 |
    | max pool | (32, 14, 14) | 0 |
    | conv 32 -> 64 | (64, 14, 14) | 64 · 32 · 9 + 64 = 18 496 |
    | max pool | (64, 7, 7) | 0 |
    | conv 64 -> 64 | (64, 7, 7) | 64 · 64 · 9 + 64 = 36 928 |
    | global average pool | (64,) | 0 |
    | fc 64 -> 10 | (10,) | 64 · 10 + 10 = 650 |

    Total: 56 394 parameters. The next cell computes the same table in plain Python
    (`describe_convnet`), and then the same network with flatten + a dense layer of 64 instead of GAP:
    its first dense layer alone has 64 · 7 · 7 · 64 + 64 = 200 768 parameters.
    """),
    code("""
    def describe_convnet(input_size, channels_in, layers):
        \"\"\"Print the output shape and the parameter count of every layer, like torchkit.summarize.

        input_size: height (= width) of the input image.  channels_in: its number of channels.
        layers: a list of ("conv", c_out, k, stride, pad), ("pool", k, stride), ("gap",) or ("fc", n_out).
        \"\"\"
        channels = channels_in
        height = input_size
        width = input_size
        total = 0
        print(f"{'layer':<18}{'output shape':<22}{'params':>12}")
        print(f"{'input':<18}{str((channels, height, width)):<22}{'-':>12}")
        for layer in layers:
            kind = layer[0]
            if kind == "conv":
                c_out = layer[1]
                k = layer[2]
                stride = layer[3]
                pad = layer[4]
                params = convops.conv_params(channels, c_out, k)          # c_out * channels * k * k + c_out
                height = convops.conv_output_size(height, k, stride, pad)
                width = convops.conv_output_size(width, k, stride, pad)
                channels = c_out
                name = f"conv {k}x{k}/{stride}"
                shape = (channels, height, width)
            elif kind == "pool":
                k = layer[1]
                stride = layer[2]
                params = 0                                                  # pooling has no parameters
                height = convops.conv_output_size(height, k, stride)       # the same formula, no padding
                width = convops.conv_output_size(width, k, stride)
                name = f"maxpool {k}/{stride}"
                shape = (channels, height, width)
            elif kind == "gap":
                params = 0
                height = 1
                width = 1
                name = "global avg pool"
                shape = (channels,)
            elif kind == "fc":
                n_out = layer[1]
                n_in = channels * height * width                            # everything left, flattened
                params = n_in * n_out + n_out                               # weights + biases
                channels = n_out
                height = 1
                width = 1
                name = f"fc -> {n_out}"
                shape = (channels,)
            total = total + params
            print(f"{name:<18}{str(shape):<22}{params:>12,d}")
        print(f"{'TOTAL':<18}{'':<22}{total:>12,d}")

    print("Tiny convnet for 28x28 grayscale (FashionMNIST):")
    describe_convnet(28, 1, [("conv", 32, 3, 1, 1), ("pool", 2, 2),
                             ("conv", 64, 3, 1, 1), ("pool", 2, 2),
                             ("conv", 64, 3, 1, 1), ("gap",), ("fc", 10)])
    print()
    print("Same net but with Flatten+Dense instead of global average pooling:")
    describe_convnet(28, 1, [("conv", 32, 3, 1, 1), ("pool", 2, 2),
                             ("conv", 64, 3, 1, 1), ("pool", 2, 2),
                             ("conv", 64, 3, 1, 1), ("fc", 64), ("fc", 10)])
    """),
    md(r"""
    ### Train it in PyTorch

    Time to build this network for real and train it on **FashionMNIST** (D0-00, section 5): 60 000
    training and 10 000 test images of clothes, 28 x 28 grayscale, 10 classes. The **test set** is kept
    apart: the network never trains on it, so its accuracy there measures how well it does on new
    images. `get_fashion_mnist(as_tensors=True)` loads the whole dataset into memory as tensors (the
    fastest option on a CPU), normalised with the dataset's own mean and standard deviation. The new
    import, `torch.nn.functional as F`, holds layer operations as plain functions, such as `F.relu`.
    """),
    api("torch.manual_seed", "torchkit.get_device", "torchkit.get_fashion_mnist", "torch.utils.data.DataLoader",
        "torchkit.show_batch", brief=True),
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
    xb, yb = next(iter(train_loader))                 # one batch, to look at
    print("one batch:", tuple(xb.shape), "labels", tuple(yb.shape), "| train / test:", len(fm_train), "/", len(fm_test))
    torchkit.show_batch(xb, yb, fm_classes, n=16, size=1.5); plt.show()
    """),
    md(r"""
    ### Exercise 7 · the forward pass of the tiny convnet

    **Words.**
    * A PyTorch model is a class derived from `nn.Module`: its layers are created in `__init__` (done
      for you below) and the computation is written in `forward(self, x)`, which runs when you call
      `model(x)`.
    * `nn.Conv2d(c_in, c_out, k, padding=p)` is exactly the layer of Exercise 3: weight (C_out, C_in, k, k)
      and bias (C_out,). It works on a whole batch at once: (B, C_in, H, W) -> (B, C_out, H_out, W_out),
      where B is the batch size.
    * The network ends with **logits**: the raw scores of the last linear layer, one per class, any real
      numbers. The predicted class is the one with the largest logit.
    * **Softmax** turns logits into probabilities (positive, adding up to 1):
      $p_c = e^{z_c} / \sum_j e^{z_j}$. Logits (2, 0, -1) give e² = 7.39, e⁰ = 1, e⁻¹ = 0.37, together
      8.76, so the probabilities are (0.84, 0.11, 0.04).
    * The **cross-entropy** loss is -log(probability of the correct class): if class 0 is correct,
      -ln(0.84) = 0.17; if class 2 is correct, -ln(0.04) = 3.17. `nn.CrossEntropyLoss` applies the
      softmax itself, so `forward` must return the logits, **without** softmax.

    **Goal.** Write `forward`: the shapes of the table above, for a batch.

    **Input.** `x`, a float tensor (B, 1, 28, 28): a batch of B normalised images.
    **Output.** The logits, a float tensor (B, 10).

    **Plan** (each line is one call; the first is written for you):

    | step | code | shape afterwards |
    |---|---|---|
    | block 1 | `self.conv1(x)`, then `F.relu(x)`, then `self.pool(x)` | (B, 32, 28, 28) -> (B, 32, 28, 28) -> (B, 32, 14, 14) |
    | block 2 | `self.conv2`, ReLU, `self.pool` | (B, 64, 14, 14) -> ... -> (B, 64, 7, 7) |
    | block 3 | `self.conv3`, ReLU (no pooling) | (B, 64, 7, 7) |
    | global average pooling | `x.mean(dim=(2, 3))`: the mean over axes 2 and 3, height and width | (B, 64) |
    | classifier | `self.fc(x)` | (B, 10) |

    **What the checker tests.** It counts the parameters (56 394: keep `__init__` as given), runs a
    random batch of 4 images and checks the output shape (4, 10) and the values against conv-ReLU-pool,
    conv-ReLU-pool, conv-ReLU, GAP, fc. Unfinished, it prints `[FAIL]  TinyConvNet.forward` with the hint
    `write the forward pass`; this checker does not stop the notebook.

    **Common mistakes.**

    | you wrote | the checker prints |
    |---|---|
    | no pooling after conv2, or a pooling after conv3 too | `output shape is right but the values differ from conv-ReLU-pool, conv-ReLU-pool, conv-ReLU, GAP, fc`: GAP hides the size difference, the numbers do not |
    | `F.softmax` at the end | the same line: return logits, the loss applies the softmax |
    | no global average pooling | `forward raised RuntimeError: addmm: mat1 and mat2 shapes cannot be multiplied (1792x7 and 64x10)`: the linear layer got (4, 64, 7, 7) |
    | `x.mean(dim=(1, 2))` | `forward raised RuntimeError: ... (4x7 and 64x10)`: averaged over channels and height instead of height and width |
    | `x.flatten(1)` instead of GAP | `forward raised RuntimeError: ... (4x3136 and 64x10)`: 64 · 7 · 7 = 3136 numbers, but `fc` takes 64 |
    | `self.relu(x)` | `forward raised AttributeError: 'TinyConvNet' object has no attribute 'relu'`: use `F.relu(x)` |
    | no `return` | `AttributeError: 'NoneType' object has no attribute 'shape'` (an error, not a `[FAIL]` line) |
    """),
    api("nn.Module", "nn.Conv2d", "nn.MaxPool2d", "F.relu", title="Functions you will need"),
    api("nn.Linear", "Tensor.mean", brief=True),
    guided("""
    class TinyConvNet(nn.Module):
        def __init__(self, n_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)     # (B, 1, 28, 28)  -> (B, 32, 28, 28)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)    # after pool: (B, 32, 14, 14) -> (B, 64, 14, 14)
            self.conv3 = nn.Conv2d(64, 64, 3, padding=1)    # after pool: (B, 64, 7, 7)   -> (B, 64, 7, 7)
            self.pool = nn.MaxPool2d(2)                     # halves height and width
            self.fc = nn.Linear(64, n_classes)              # after GAP: (B, 64) -> (B, 10)

        def forward(self, x):                               # x: (B, 1, 28, 28)
            #@raise

            # Block 1: convolution, ReLU, max pooling
            x = self.conv1(x)                               # (B, 32, 28, 28)
            x = F.relu(x)                                   # YOUR CODE: ReLU -> (B, 32, 28, 28)
            x = self.pool(x)                                # YOUR CODE: max pooling -> (B, 32, 14, 14)
            # Block 2: the same with conv2
            x = self.conv2(x)                               # YOUR CODE: conv2 -> (B, 64, 14, 14)
            x = F.relu(x)                                   # YOUR CODE: ReLU
            x = self.pool(x)                                # YOUR CODE: max pooling -> (B, 64, 7, 7)
            # Block 3: conv3 and ReLU, no pooling
            x = self.conv3(x)                               # YOUR CODE: conv3 -> (B, 64, 7, 7)
            x = F.relu(x)                                   # YOUR CODE: ReLU
            # Global average pooling: the mean over height (axis 2) and width (axis 3)
            x = x.mean(dim=(2, 3))                          # YOUR CODE: -> (B, 64)
            # Classifier: one logit per class, no softmax
            logits = self.fc(x)                             # YOUR CODE: -> (B, 10)
            return logits

    tests.check_tiny_convnet(TinyConvNet())
    """),
    md(r"""
    `torchkit.summarize` pushes one dummy image through the model and records the output shape of every
    layer as it passes. Compare it with the table at the start of this section: same shapes, same
    56 394 parameters. The MACs column is the formula of section 5.
    """),
    api("torchkit.summarize"),
    code("""
    print(torchkit.summarize(TinyConvNet(), (1, 28, 28)))
    """),
    md(r"""
    **Training.** Two **epochs** (passes over the 60 000 training images, D0-00) with the training loop
    of D0-00 inside `torchkit.fit`, capped at three minutes. Two choices differ from D0-00:
    * the optimiser is **AdamW** instead of SGD: it gives every parameter a step size adapted to the size
      of its own gradients, which needs less tuning;
    * the learning rate follows a **one-cycle schedule**: it rises to a peak (`max_lr`) during the first
      15 % of the steps, then falls to almost zero. This gets much more out of two epochs than a
      constant rate (D1-02 explains why).

    The trained weights are saved as a checkpoint in `data/checkpoints/`; when the file exists, the cell
    loads it instead of training, so re-running the notebook is instant (delete the file to retrain).

    For comparison we train a **dense baseline** that never looks at neighbourhoods: flatten the 784
    pixels, one **hidden layer** of 256 units (a linear layer between input and output, followed by a
    ReLU), then 10 outputs. It has 784 · 256 + 256 + 256 · 10 + 10 = 203 530 parameters, 3.6 times as
    many as the convnet. On a laptop CPU the convnet trains in one to two minutes, the dense network in
    a few seconds.

    <details>
    <summary><b>More: what the arguments of <code>train_or_load</code> do</b> (optional, click to open)</summary>

    * `weight_decay=1e-4`: AdamW also shrinks every weight a little at every step, which keeps the
      weights small and reduces **overfitting** (learning the training images by heart instead of
      what makes a class; D1-02 is about it).
    * `OneCycleLR(..., total_steps=epochs * len(train_loader), pct_start=0.15)`: one learning-rate value
      per batch, rising during the first 15 % of the steps.
    * `torchkit.fit(..., restore_best=True)`: after training, the weights of the epoch with the best
      test accuracy are put back.
    * `max_lr`: 1e-2 for the convnet, 3e-3 for the dense network: values that worked for each.

    </details>
    """),
    api("torch.optim.AdamW", "torch.optim.lr_scheduler.OneCycleLR", "nn.CrossEntropyLoss", "torchkit.load_checkpoint",
        "nn.Sequential", "nn.Flatten", "nn.ReLU"),
    api("torchkit.fit", "Tensor.to", "Module.parameters", brief=True),
    code("""
    def train_or_load(model, name, max_lr, epochs=2):
        \"\"\"Train with torchkit.fit, or load data/checkpoints/<name>.pt if it exists. Returns the history dict.\"\"\"
        checkpoint = torchkit.CHECKPOINTS / f"{name}.pt"
        if checkpoint.exists():
            extras = torchkit.load_checkpoint(model, checkpoint)        # puts the saved weights into model
            print(f"loaded {checkpoint.name} (delete it to retrain)")
            return extras["history"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-4)
        total_steps = epochs * len(train_loader)                        # one step per batch
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.15)
        return torchkit.fit(model, train_loader, test_loader, nn.CrossEntropyLoss(), optimizer, epochs,
                            scheduler=scheduler, device=device, max_minutes=3, log_every=0,
                            checkpoint=checkpoint, restore_best=True)

    torch.manual_seed(0)
    cnn = TinyConvNet().to(device)
    hist_cnn = train_or_load(cnn, "d1_01_tiny_convnet", max_lr=1e-2)

    torch.manual_seed(0)
    dense = nn.Sequential(
        nn.Flatten(),                   # (B, 1, 28, 28) -> (B, 784)
        nn.Linear(28 * 28, 256),        # hidden layer: 784 -> 256
        nn.ReLU(),
        nn.Linear(256, 10),             # 256 -> 10 logits
    ).to(device)
    hist_dense = train_or_load(dense, "d1_01_dense_baseline", max_lr=3e-3)
    """),
    md(r"""
    The next cell measures the **accuracy** of both networks on the test set: the fraction of test
    images whose largest logit is the correct class. With the provided checkpoints: **89.5 %** for the
    convnet and **87.6 %** for the dense network (training again gives slightly different numbers).
    """),
    api("torchkit.evaluate", "torchkit.accuracy", "torchkit.count_parameters"),
    code("""
    loss_fn = nn.CrossEntropyLoss()
    _, acc_cnn = torchkit.evaluate(cnn, test_loader, loss_fn, torchkit.accuracy, device)
    _, acc_dense = torchkit.evaluate(dense, test_loader, loss_fn, torchkit.accuracy, device)
    params_cnn = torchkit.count_parameters(cnn)
    params_dense = torchkit.count_parameters(dense)
    print(f"{'model':<20}{'params':>10}{'test acc':>10}")
    print(f"{'tiny convnet':<20}{params_cnn:>10,d}{acc_cnn:>10.1%}")
    print(f"{'dense 784-256-10':<20}{params_dense:>10,d}{acc_dense:>10.1%}")

    # Figure: loss per epoch, test accuracy and parameter count of the two models
    names = ["tiny convnet", "dense 784-256-10"]
    colors = ["#4C78A8", "#BAB0AC"]
    histories = [hist_cnn, hist_dense]
    accuracies = [acc_cnn * 100, acc_dense * 100]
    thousands = [params_cnn / 1000, params_dense / 1000]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for i in range(2):
        history = histories[i]
        epochs = range(1, len(history["val_metric"]) + 1)
        axes[0].plot(epochs, history["train_loss"], marker="o", color=colors[i], label=f"{names[i]} train")
        axes[0].plot(epochs, history["val_loss"], marker="s", ls="--", color=colors[i], label=f"{names[i]} test")
    axes[0].set_xlabel("epoch"); axes[0].set_xticks([1, 2]); axes[0].set_title("cross-entropy loss"); axes[0].legend(fontsize=8)
    bars = axes[1].bar(names, accuracies, color=colors)
    axes[1].bar_label(bars, fmt="%.1f%%"); axes[1].set_ylim(80, 95); axes[1].set_title("test accuracy (%)")
    bars = axes[2].bar(names, thousands, color=colors)
    axes[2].bar_label(bars, fmt="%.0fk"); axes[2].set_title("parameters (thousands)"); axes[2].set_ylim(0, 240)
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    The convnet wins with fewer parameters because its built-in assumptions suit images: the same small
    detector is reused at every position (weight sharing), and pooling makes the answer tolerant to
    small shifts. The dense network has to learn "a sleeve at the left" and "a sleeve at the right" as
    two unrelated patterns.

    ### What did the first layer learn?

    `cnn.conv1.weight` has shape (32, 1, 3, 3): 32 kernels of 3 x 3, like the hand-made gallery of
    section 3. Below, each learned kernel is drawn with its own symmetric colour scale (blue positive,
    red negative, white zero, as in section 3), so only the *pattern* matters.
    """),
    api("Tensor.detach", "Tensor.cpu"),
    api("Tensor.numpy", brief=True),
    code("""
    learned = cnn.conv1.weight.detach().cpu().numpy()      # (32, 1, 3, 3): 32 kernels, 1 input channel
    learned = learned[:, 0]                                # (32, 3, 3)
    hand_names = []
    for name in convops.KERNELS:
        if name != "identity":                             # the identity kernel detects nothing
            hand_names.append(name)

    # Figure: the hand-made kernels of section 3 (top) and the 32 learned kernels (bottom)
    fig = plt.figure(figsize=(12, 7.4), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 2.7])
    axs = top.subplots(1, len(hand_names))
    for i in range(len(hand_names)):
        kernel = convops.KERNELS[hand_names[i]]
        limit = np.abs(kernel).max()                       # symmetric colour scale: white = 0
        axs[i].imshow(kernel, cmap="RdBu", vmin=-limit, vmax=limit)
        axs[i].set_title(hand_names[i], fontsize=9); axs[i].set_xticks([]); axs[i].set_yticks([])
    top.suptitle("hand-made kernels (section 3)", fontsize=12)
    axs = bottom.subplots(4, 8)
    for i, ax in enumerate(axs.flat):
        limit = np.abs(learned[i]).max()
        ax.imshow(learned[i], cmap="RdBu", vmin=-limit, vmax=limit)
        ax.set_title(f"#{i}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    bottom.suptitle("the 32 kernels conv1 learned on FashionMNIST in two epochs", fontsize=12)
    plt.show()
    """),
    md(r"""
    Look for the families you know: kernels with a positive column next to a negative column are
    vertical-edge detectors (Sobel x), rows of opposite sign are horizontal-edge detectors (Sobel y), a
    strong centre with an opposite-sign ring is a Laplacian / sharpen, and kernels with one sign
    everywhere act as blurs or brightness detectors. Nobody designed these: gradient descent found that
    these patterns help to tell clothes apart. On a 3 x 3 grid and two epochs the resemblance is rough;
    the first layer of a big ImageNet network (D1-04) shows clean edges, colour blobs and stripe patterns.

    ### Feature maps: what the first layer does to one image

    Each of the 32 kernels produces one output channel, a 28 x 28 map that is bright where the image
    looks like that kernel. This is the (32, 28, 28) tensor that `conv2` receives (after the ReLU,
    before the pooling).

    Notice the bright frame around many maps: after normalisation the black background is about -0.8,
    but `padding=1` surrounds the image with zeros, so the border looks like a sudden edge to every edge
    detector. Padding is not free of side effects.
    """),
    api("Tensor.argmax"),
    api("torch.no_grad", "Tensor.unsqueeze", "Tensor.item", "torchkit.denormalize", brief=True),
    code("""
    x0, y0 = fm_test[0]                                    # one test image (1, 28, 28) and its label
    with torch.no_grad():
        batch = x0.unsqueeze(0).to(device)                 # (1, 1, 28, 28): a batch of one
        maps = F.relu(cnn.conv1(batch))                    # (1, 32, 28, 28)
        fmap = maps[0].cpu().numpy()                       # (32, 28, 28)
        logits = cnn(batch)                                # (1, 10)
        pred = logits.argmax(dim=1).item()                 # the predicted class index

    # Figure: the input image (left) and the 32 feature maps of conv1 after ReLU (right)
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

    dead = []                                              # maps that are 0 everywhere for this image
    for i in range(32):
        if fmap[i].max() == 0:
            dead.append(i)
    print("maps that stay black: the kernel never matches this image (ReLU clips negative responses to 0)")
    print("dead maps:", dead)
    """),

    # ------------------------------------------------------------------ 9
    md(r"""
    ## 9 · Summary

    * A convolution layer = C_out small kernels of shape (C_in, k, k); each one is multiplied element by
      element with every (C_in, k, k) patch of the input and summed, plus a bias. Parameters
      (C_out · C_in · k · k + C_out) do not depend on the image size; the computation does.
    * Output size: $\lfloor (n + 2p - k_{\text{eff}})/s \rfloor + 1$ with $k_{\text{eff}} = d(k-1) + 1$.
    * The receptive field grows by (k - 1) · jump per layer, and every stride multiplies the jump.
      Two 3 x 3 layers see as far as one 5 x 5 layer, with 28 % fewer weights and a ReLU in between.
    * Pooling (max or average) shrinks the maps with no parameters and makes the answer tolerant to
      small shifts; global average pooling replaces the giant dense layers at the end of older
      networks; 1 x 1 convolutions mix channels at every pixel.
    * The pattern "shrink space, grow channels, pool, classify" runs from LeNet (1998) to today's
      convnets (D1-03).

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
    md(r"""
    ## Words from this notebook

    Words from D0-00 (tensor, channel, pixel, parameter, epoch, checkpoint, normalisation, ...) are in
    its own list.

    **The convolution**

    | word | meaning |
    |---|---|
    | grayscale | one channel of brightness per pixel |
    | kernel, filter | a small grid of numbers (weights) slid over the input |
    | window | the part of the input the kernel covers at one position |
    | patch | the input values under the window |
    | element by element | multiplying two arrays position by position (top-left with top-left, ...) |
    | feature map | the output grid of one kernel: how much each place looks like the kernel's pattern |
    | cross-correlation | slide, multiply element by element, add up; what deep learning calls convolution |
    | convolution | in mathematics the same with the kernel flipped; in deep learning, the cross-correlation |
    | edge | a place where the brightness changes quickly |
    | blur | averaging every pixel with its neighbours |
    | noise | random speckle in an image |
    | percentile | the value below which a given percentage of the numbers lie |

    **Sizes**

    | word | meaning |
    |---|---|
    | padding | rows and columns of zeros added on every side; "same" keeps the size, "valid" means none |
    | stride | how many pixels the window moves between positions |
    | dilation | spreading the kernel's numbers apart, with skipped pixels between them |
    | effective kernel size | the width a dilated kernel covers: d (k - 1) + 1 |
    | downsampling | making a feature map smaller (stride 2, pooling) |
    | output-size formula | (n + 2p - k_eff) // s + 1 |
    | receptive field | the square of input pixels one unit depends on |
    | jump | the distance in input pixels between neighbouring units of a layer |
    | unit, neuron | one number of a feature map |

    **Layers**

    | word | meaning |
    |---|---|
    | input / output channels | C_in channels read, C_out channels written by a layer |
    | weight tensor | all kernels of a layer, shape (C_out, C_in, kh, kw) |
    | bias | one learned number per output channel, added at every position |
    | weight sharing | the same kernel used at every position |
    | translation equivariance | shift the input, and the output shifts the same way |
    | translation invariance | shift the input a little, and the output stays the same |
    | MAC, FLOP | one multiply-and-add; 1 MAC = 2 floating-point operations |
    | im2col | copying every patch into one row of a matrix, so a layer becomes one matrix product |
    | activation function | a non-linear function applied to every number between layers |
    | ReLU | max(0, x): negative numbers become 0 |
    | pooling | one number per window: the largest (max pooling) or the mean (average pooling) |
    | global average pooling (GAP) | the mean of each channel over all positions: (C, H, W) -> (C,) |
    | flatten | writing all numbers of a tensor one after another, (C, H, W) -> C · H · W |
    | dense / fully connected layer (fc) | a linear layer in which every output depends on every input |
    | hidden layer | a layer between input and output |
    | 1x1 convolution | a per-pixel linear layer across channels |

    **Classification**

    | word | meaning |
    |---|---|
    | classifier | a network that outputs one score per class |
    | logits | the raw scores of the last layer, one per class |
    | softmax | turns logits into probabilities: e^z / sum of e^z |
    | cross-entropy | the classification loss: -log(probability of the correct class) |
    | accuracy | the fraction of examples whose largest logit is the correct class |
    | test set | examples kept apart from training, used only to measure the result |
    | AdamW | an optimiser that adapts every parameter's step size to its gradients, with weight decay |
    | one-cycle schedule | a learning rate that rises to a peak, then falls to almost zero |
    | overfitting | learning the training examples by heart instead of what makes a class (D1-02) |
    | dense baseline | a network without convolutions, for comparison |
    """),
]
