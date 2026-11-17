from nb import md, code, exercise, api

NAME = "P3-googlenet"

CELLS = [
    md(r"""
    # P3 · GoogLeNet step by step: build an Inception network from its blueprint

    **What we do today**

    1. Look at the idea behind **GoogLeNet** (also called Inception v1, Szegedy et al., 2014), the network that won
       the 2014 ImageNet competition: instead of choosing one filter size, run several side by side.
    2. Build GoogLeNet yourself in eight small steps, (a) to (h). Every step gives you a table of layers; you
       type those layers into a prepared cell, a **check** tells you what is right and what is not, and a drawing
       shows what you have built so far.
    3. Look at the finished network: block by block, its size, its cost, and a comparison with VGG-16.
    4. Train a small version of it on colour photos (optional on slow computers).

    **How to use this notebook.** Run the cells from top to bottom with **Shift+Enter**. You type code only in
    the eight cells headed **Your turn**. In them, every `____` (four underscores) is a blank: replace it with
    the layer or the line its `# YOUR CODE:` comment describes, copying the numbers from the table above the
    cell. Each of these cells starts with the line `raise NotImplementedError`, which stops the cell's function
    with the message "not implemented"; delete that line once every blank is filled in.

    You cannot get stuck. If a step is not finished, its check says what is missing, and the next cells
    continue with the course's own **reference** version of that step and say so ("note: using the reference
    ..."). You can even run the whole notebook before typing anything. Come back to a step whenever you like:
    fix your cell, run it, then its check and the cell after it. The finished cells are in
    `solutions/P3-googlenet-solution.ipynb`.

    * `P.something(...)` is a function of the course toolkit (P1 introduced it); `T.something(...)` is a check or
      a helper of this notebook.
    * Closed boxes **Functions used here (click to open)** explain the PyTorch functions of the next cell.
    * **New words** are in bold where they are explained, and collected at the end.
    * Numbers quoted in the text are what the cells print. Times differ from computer to computer.
    """),

    # ------------------------------------------------------------------ 0
    md(r"""
    ## 0 · Setup

    As in P1, this cell finds the course's helper folder `cvcourse` and loads PyTorch (`torch`, with its layers
    in `torch.nn`, short `nn`) and the toolkit `P`. New today:

    * `T`: the checks of this notebook, one per step, and `T.yours_or_reference(...)`, which keeps your version of
      a step if it passes its checks and otherwise hands over the reference version;
    * `reference`: the course's own finished GoogLeNet, so that we can look at the target before building it.

    `P.setup(seed=0)` fixes the **random seed**, so that random numbers repeat from run to run.
    """),
    code("""
    import sys
    import pathlib

    folder = pathlib.Path.cwd()
    while not (folder / "cvcourse").is_dir() and folder != folder.parent:
        folder = folder.parent                      # go up until the folder that contains cvcourse
    sys.path.insert(0, str(folder))

    import torch
    import torch.nn as nn
    import torchvision
    from cvcourse import practical as P
    from cvcourse import ptests_p3 as T
    from cvcourse import pmodels_p3 as reference

    P.setup(seed=0)
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · The idea: four filter sizes side by side

    In P1 every convolution layer had one filter size, for example 3 x 3. But which size is right? A small
    filter sees fine detail (the edge of a whisker), a big one sees larger patterns (the shape of an ear). The
    authors of GoogLeNet did not choose. Their **Inception block** sends the same input through four
    **branches** (short chains of layers) side by side:

    | branch | layers | what it looks at |
    |---|---|---|
    | 1 | 1x1 convolution | each pixel on its own |
    | 2 | 1x1 convolution, then 3x3 convolution | small neighbourhoods |
    | 3 | 1x1 convolution, then 5x5 convolution | larger neighbourhoods |
    | 4 | 3x3 max-pool, then 1x1 convolution | the strongest value around each pixel |

    and then **concatenates** their outputs: it stacks the **feature maps** (the output pictures of the
    filters, also called **maps** or **channels**) of the four branches one after the other along the channel
    axis. If branch 1 makes 64 maps, branch 2 128, branch 3 32 and branch 4 32, the block outputs
    64 + 128 + 32 + 32 = 256 maps. Every branch keeps the height and the width of the input, so that the maps
    can be stacked. Every convolution is followed by a ReLU (P1: it keeps positive numbers and turns negative
    ones into 0).

    GoogLeNet is a **stem** (a few plain layers that shrink the photo), nine Inception blocks and a small
    **head** that turns the maps into scores. The original also had two **auxiliary classifiers**, small extra
    heads half-way up that helped training; we build the main network without them, as is usual today.

    ### Why the 1x1 convolutions?

    A **1x1 convolution** has filters of a single pixel. At every position it takes the numbers of that pixel in
    all input maps (192 of them, say) and computes weighted sums of them, one per output map: it mixes the maps
    and changes how many there are, without changing height or width. That makes it a cheap way to shrink the
    number of maps before an expensive filter. In branches 2 and 3 a 1x1 convolution first **reduces** the 192
    input maps to 96 and to 16, and only then does the 3x3 or 5x5 convolution run. This is why the table of
    GoogLeNet has columns called "3x3 reduce" and "5x5 reduce". (In branch 4, the 1x1 after the pool has the
    same role: it shrinks the 192 maps the pool lets through.)

    A convolution with `c_in` input maps, `c_out` output maps and a `k x k` filter has `c_in x c_out x k x k`
    weights. The cell counts the weights of branches 2 and 3 of the first Inception block (block "3a": 192 maps
    in), without and with the 1x1 reduction (biases left out):
    """),
    code("""
    # branch 2 of block 3a: 192 maps in, 128 maps out of a 3x3 convolution
    without_1x1 = 192 * 128 * 3 * 3                 # a 3x3 convolution straight from the 192 input maps
    with_1x1 = 192 * 96 + 96 * 128 * 3 * 3          # a 1x1 convolution 192 -> 96 maps, then a 3x3 convolution 96 -> 128
    print("branch 2 (3x3):", without_1x1, "weights without the 1x1,", with_1x1, "with it:",
          round(without_1x1 / with_1x1, 1), "times fewer")

    # branch 3 of block 3a: 192 maps in, 32 maps out of a 5x5 convolution
    without_1x1 = 192 * 32 * 5 * 5
    with_1x1 = 192 * 16 + 16 * 32 * 5 * 5           # a 1x1 convolution 192 -> 16 maps, then a 5x5 convolution 16 -> 32
    print("branch 3 (5x5):", without_1x1, "weights without the 1x1,", with_1x1, "with it:",
          round(without_1x1 / with_1x1, 1), "times fewer")
    """),
    md(r"""
    The 5x5 branch shrinks almost 10 times (153 600 weights become 15 872), because its expensive 5x5 filters now
    read 16 maps instead of 192. The work shrinks by the same factor: every weight is used once at each of the
    28 x 28 = 784 positions. Without these reductions GoogLeNet would be several times bigger and slower.

    ### The target

    Here is a finished Inception block 3a, from the course's reference version, drawn by `P.show_model` (P1): the
    input on the left, the four branches one above the other, joined by the red **Concat** box (concatenate).
    Under each box: what the layer does, the shape of its output (maps x height x width) and its parameters. This
    is what you will build in steps (a) to (d).
    """),
    code("""
    P.show_model(reference.Inception(192, 64, 96, 128, 16, 32, 32), (192, 28, 28),
                 title="The target: Inception block 3a (the reference version)")
    """),
    md(r"""
    **Reading the drawing.** All four branches end at 28 x 28: nothing shrinks inside a block. The four outputs
    have 64, 128, 32 and 32 maps, and the Concat box stacks them into 256. Most parameters sit in the 3x3
    convolution of branch 2 (110 720 of the block's 163 696), even after its reduction.
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · Build GoogLeNet in eight steps

    | step | what | you type |
    |---|---|---|
    | (a) | `conv_relu`: a convolution followed by a ReLU, the unit everything is built from | nothing: a worked example |
    | (b) | branches 1 and 2 of the Inception block | 3 layers |
    | (c) | branches 3 and 4 | 4 layers |
    | (d) | the block itself: run the four branches, stack their outputs | 4 lines |
    | (e) | the stem: the first layers, 3 x 224 x 224 photo -> 192 x 28 x 28 | 6 layers |
    | (f) | the nine Inception blocks 3a ... 5b, from the published table | 8 lines |
    | (g) | the head: maps -> 1000 scores | 3 layers |
    | (h) | the whole network: stem, blocks, head | 3 lines |

    Every step has the same rhythm:

    1. a **table** says exactly which layers to write, with all their numbers;
    2. **Your turn**: a prepared cell where you fill in the blanks `____` and delete the `raise` line;
    3. a **check**: one `[PASS]` or `[FAIL]` line per property (the right kind of layer in the right place, its
       channels, filter size, stride, padding, the output shape, the number of parameters, ...). A `[FAIL]` line
       comes with a **hint** that names the layer and the number that is wrong, what it is and what it should be;
    4. a **drawing** of what you have built. Its first line, `x = T.yours_or_reference(x)`, keeps your `x` if it
       passes all its checks ("using your x") and otherwise replaces it by the reference version ("note: using
       the reference x ..."), so that the next steps work either way.

    ### How to read a table row

    Each table row is one layer. The parameters of a convolution (as in P1): `in_channels`, the maps coming in;
    `out_channels`, the filters, so the maps going out; `kernel_size`, the filter size (3 means 3 x 3); `stride`,
    the step of the filter (2 halves height and width); `padding`, rows and columns of zeros added around the
    picture. The output-size rule from P1 tells you what comes out:

    > output size = (n + 2 x padding - kernel_size) / stride + 1, rounded down

    Inside an Inception block every layer must keep the size: a 3x3 filter needs padding 1
    ((28 + 2 - 3) / 1 + 1 = 28), a 5x5 filter padding 2 ((28 + 4 - 5) / 1 + 1 = 28), a 1x1 filter padding 0.

    ### Step (a) · `conv_relu`, a worked example

    GoogLeNet has 57 convolutions, and each is followed by a ReLU. The function `conv_relu` makes such a pair in
    one line: it returns an `nn.Sequential` (P1: layers applied one after the other) of an `nn.Conv2d` and an
    `nn.ReLU`. `stride=1, padding=0` in its first line are **default values**: a call that leaves them out gets
    1 and 0. A table row then becomes one call:

    | layer | in | out | kernel | stride | padding | you write |
    |---|---|---|---|---|---|---|
    | conv_relu | 192 | 64 | 1 | 1 | 0 | `conv_relu(192, 64, kernel_size=1)` |
    | conv_relu | 96 | 128 | 3 | 1 | 1 | `conv_relu(96, 128, kernel_size=3, padding=1)` |

    This cell is complete: run it, then the check and the drawing.
    """),
    api("nn.Sequential", "nn.Conv2d", "nn.ReLU", brief=True),
    code('''
    def conv_relu(in_channels, out_channels, kernel_size, stride=1, padding=0):
        """A convolution followed by a ReLU: the unit GoogLeNet is built from."""
        layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),                      # negative numbers -> 0
        )
        return layers
    '''),
    md(r"""
    **The check.** `T.check_conv_relu(conv_relu)` builds a small `conv_relu`, looks at its layers one by one,
    runs a batch of random pictures through it and prints what it found. This is what every check of this
    notebook looks like.
    """),
    code("""
    T.check_conv_relu(conv_relu)
    """),
    md(r"""
    **The drawing.** `conv_relu(192, 64, kernel_size=1)` is already branch 1 of block 3a: 192 maps in, 64 out,
    each pixel on its own.
    """),
    code("""
    P.show_model(conv_relu(192, 64, kernel_size=1), (192, 28, 28), title="conv_relu(192, 64, kernel_size=1)")
    """),

    # ------------------------------------------------------------------ (b)
    md(r"""
    ### Step (b) · Branches 1 and 2

    An Inception block has seven numbers, written in this order everywhere today (the column names of the
    published table in brackets):

    | name | meaning | block 3a |
    |---|---|---|
    | `in_channels` | maps coming into the block | 192 |
    | `c1` | maps out of branch 1 (#1x1) | 64 |
    | `c3_reduce` | maps out of the 1x1 reduction of branch 2 (#3x3 reduce) | 96 |
    | `c3` | maps out of the 3x3 convolution of branch 2 (#3x3) | 128 |
    | `c5_reduce` | maps out of the 1x1 reduction of branch 3 (#5x5 reduce) | 16 |
    | `c5` | maps out of the 5x5 convolution of branch 3 (#5x5) | 32 |
    | `pool_proj` | maps out of the 1x1 convolution after the pool of branch 4 (pool proj) | 32 |

    Each branch is a function that receives the numbers it needs and returns the branch as an `nn.Sequential`.
    Write the arguments' **names** (`in_channels`, `c1`, ...), not the numbers of block 3a: the same function
    must build all nine blocks.

    **Branch 1**, `make_branch1(in_channels, c1)`:

    | layer | in | out | kernel | stride | padding | block 3a |
    |---|---|---|---|---|---|---|
    | conv_relu | `in_channels` | `c1` | 1 | 1 | 0 | 192 -> 64 |

    **Branch 2**, `make_branch2(in_channels, c3_reduce, c3)`:

    | layer | in | out | kernel | stride | padding | block 3a |
    |---|---|---|---|---|---|---|
    | conv_relu | `in_channels` | `c3_reduce` | 1 | 1 | 0 | 192 -> 96 |
    | conv_relu | `c3_reduce` | `c3` | 3 | 1 | 1 | 96 -> 128 |

    The second layer reads what the first one made: its `in` is the first one's `out`.

    **Your turn.** Replace each `____` by one `conv_relu(...)` call, delete the two `raise` lines, run the cell.
    """),
    exercise(
        stub='''
        def make_branch1(in_channels, c1):
            """Branch 1 of an Inception block: a 1x1 convolution, in_channels -> c1 maps."""
            raise NotImplementedError   # delete this line once every ____ is filled in
            branch = nn.Sequential(
                ____,      # YOUR CODE: conv_relu, a 1x1 convolution from in_channels to c1 maps
            )
            return branch


        def make_branch2(in_channels, c3_reduce, c3):
            """Branch 2: a 1x1 convolution in_channels -> c3_reduce maps, then a 3x3 convolution -> c3 maps."""
            raise NotImplementedError   # delete this line once every ____ is filled in
            branch = nn.Sequential(
                ____,      # YOUR CODE: conv_relu, a 1x1 convolution from in_channels to c3_reduce maps
                ____,      # YOUR CODE: conv_relu, a 3x3 convolution from c3_reduce to c3 maps, padding 1
            )
            return branch
        ''',
        solution='''
        def make_branch1(in_channels, c1):
            """Branch 1 of an Inception block: a 1x1 convolution, in_channels -> c1 maps."""
            branch = nn.Sequential(
                conv_relu(in_channels, c1, kernel_size=1),      # YOUR CODE: conv_relu, a 1x1 convolution from in_channels to c1 maps
            )
            return branch


        def make_branch2(in_channels, c3_reduce, c3):
            """Branch 2: a 1x1 convolution in_channels -> c3_reduce maps, then a 3x3 convolution -> c3 maps."""
            branch = nn.Sequential(
                conv_relu(in_channels, c3_reduce, kernel_size=1),       # YOUR CODE: conv_relu, a 1x1 convolution from in_channels to c3_reduce maps
                conv_relu(c3_reduce, c3, kernel_size=3, padding=1),     # YOUR CODE: conv_relu, a 3x3 convolution from c3_reduce to c3 maps, padding 1
            )
            return branch
        '''),
    md(r"""
    **Check.** Both branches are built with the numbers of block 3a and looked at layer by layer, then run on a
    batch of 2 random "pictures" of 192 x 28 x 28. The last line of each builds the branch again with the numbers
    of block 4b, to catch numbers typed where names belong.
    """),
    code("""
    T.check_branches_1_2(make_branch1, make_branch2)
    """),
    md(r"""
    **What you built.** Branch 2 with the numbers of block 3a: 192 maps are reduced to 96, then the 3x3
    convolution makes 128 maps of the same 28 x 28.
    """),
    code("""
    make_branch1 = T.yours_or_reference(make_branch1)
    make_branch2 = T.yours_or_reference(make_branch2)
    P.show_model(make_branch2(192, 96, 128), (192, 28, 28), title="Branch 2 of block 3a: make_branch2(192, 96, 128)")
    """),

    # ------------------------------------------------------------------ (c)
    md(r"""
    ### Step (c) · Branches 3 and 4

    **Branch 3**, `make_branch3(in_channels, c5_reduce, c5)`: like branch 2, with a 5x5 filter.

    | layer | in | out | kernel | stride | padding | block 3a |
    |---|---|---|---|---|---|---|
    | conv_relu | `in_channels` | `c5_reduce` | 1 | 1 | 0 | 192 -> 16 |
    | conv_relu | `c5_reduce` | `c5` | 5 | 1 | 2 | 16 -> 32 |

    **Branch 4**, `make_branch4(in_channels, pool_proj)`: first a **max-pool** (P1: keeps the largest value of
    each window), then a 1x1 convolution.

    | layer | in | out | kernel | stride | padding | block 3a |
    |---|---|---|---|---|---|---|
    | `nn.MaxPool2d` | | | 3 | 1 | 1 | 192 maps stay 192 |
    | conv_relu | `in_channels` | `pool_proj` | 1 | 1 | 0 | 192 -> 32 |

    This max-pool is unusual: with stride 1 and padding 1 it does not shrink the picture (in P1 every max-pool
    halved it). Each pixel is replaced by the largest value of its 3 x 3 neighbourhood. Write the stride: without
    `stride=...`, `nn.MaxPool2d` steps by its kernel size (here 3) and the picture would shrink to a third. A
    max-pool has no parameters and keeps the number of maps, so the 1x1 convolution after it reads
    `in_channels` maps.
    """),
    api("nn.MaxPool2d", brief=True),
    exercise(
        stub='''
        def make_branch3(in_channels, c5_reduce, c5):
            """Branch 3: a 1x1 convolution in_channels -> c5_reduce maps, then a 5x5 convolution -> c5 maps."""
            raise NotImplementedError   # delete this line once every ____ is filled in
            branch = nn.Sequential(
                ____,      # YOUR CODE: conv_relu, a 1x1 convolution from in_channels to c5_reduce maps
                ____,      # YOUR CODE: conv_relu, a 5x5 convolution from c5_reduce to c5 maps, padding 2
            )
            return branch


        def make_branch4(in_channels, pool_proj):
            """Branch 4: a 3x3 max-pool that keeps the size, then a 1x1 convolution in_channels -> pool_proj maps."""
            raise NotImplementedError   # delete this line once every ____ is filled in
            branch = nn.Sequential(
                ____,      # YOUR CODE: nn.MaxPool2d with kernel_size=3, stride=1, padding=1
                ____,      # YOUR CODE: conv_relu, a 1x1 convolution from in_channels to pool_proj maps
            )
            return branch
        ''',
        solution='''
        def make_branch3(in_channels, c5_reduce, c5):
            """Branch 3: a 1x1 convolution in_channels -> c5_reduce maps, then a 5x5 convolution -> c5 maps."""
            branch = nn.Sequential(
                conv_relu(in_channels, c5_reduce, kernel_size=1),       # YOUR CODE: conv_relu, a 1x1 convolution from in_channels to c5_reduce maps
                conv_relu(c5_reduce, c5, kernel_size=5, padding=2),     # YOUR CODE: conv_relu, a 5x5 convolution from c5_reduce to c5 maps, padding 2
            )
            return branch


        def make_branch4(in_channels, pool_proj):
            """Branch 4: a 3x3 max-pool that keeps the size, then a 1x1 convolution in_channels -> pool_proj maps."""
            branch = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),       # YOUR CODE: nn.MaxPool2d with kernel_size=3, stride=1, padding=1
                conv_relu(in_channels, pool_proj, kernel_size=1),       # YOUR CODE: conv_relu, a 1x1 convolution from in_channels to pool_proj maps
            )
            return branch
        '''),
    code("""
    T.check_branches_3_4(make_branch3, make_branch4)
    """),
    md(r"""
    **What you built.** Branches 3 and 4 with the numbers of block 3a. In branch 3 the 1x1 convolution squeezes
    192 maps into 16 before the 5x5 filters; in branch 4 the max-pool keeps 192 maps of 28 x 28 and the 1x1
    convolution reduces them to 32.
    """),
    code("""
    make_branch3 = T.yours_or_reference(make_branch3)
    make_branch4 = T.yours_or_reference(make_branch4)
    P.show_model(make_branch3(192, 16, 32), (192, 28, 28), title="Branch 3 of block 3a: make_branch3(192, 16, 32)")
    P.show_model(make_branch4(192, 32), (192, 28, 28), title="Branch 4 of block 3a: make_branch4(192, 32)")
    """),

    # ------------------------------------------------------------------ (d)
    md(r"""
    ### Step (d) · The Inception block

    A block whose layers are not one chain cannot be an `nn.Sequential`: we write it as a **class**, a recipe
    for objects, built on PyTorch's `nn.Module`. It has two parts:

    * `__init__` runs once, when you create a block with `Inception(192, 64, ...)`. It builds the four branches
      with your functions and stores them as **attributes** of the block (`self.branch1`, ...), so that PyTorch
      finds their parameters. `super().__init__()` must come first: it prepares the `nn.Module` bookkeeping.
      This part is given.
    * `forward(self, x)` says what happens to an input `x` (a batch of maps): this is your part. You never call
      it by name: `block(x)` runs `forward(x)`.

    `forward` sends the **same** `x` through each branch and stacks the four results along the channel axis,
    axis 1 of a batch of shape (batch, maps, height, width):

    ```
                                x: (batch, in_channels, H, W)
              branch1           branch2           branch3           branch4
          y1: c1 maps       y2: c3 maps       y3: c5 maps       y4: pool_proj maps     (all H x W)
                         out = torch.cat([y1, y2, y3, y4], dim=1)
                         (batch, c1 + c3 + c5 + pool_proj, H, W)
    ```

    `torch.cat` takes **one list** of tensors (P1: PyTorch's arrays of numbers; here the four branch outputs) in
    square brackets, and the axis `dim`. The order of the list is
    the order of the maps in the output: y1 first. Keep it: the real GoogLeNet uses this order, and a network
    whose maps come in a different order is a different network (its trained weights would not fit).

    **Your turn.** The first line (`y1`) is the example; fill in `y2`, `y3`, `y4` and `out`, delete the `raise` line.
    """),
    api("nn.Module", "torch.cat"),
    exercise(
        stub='''
        class Inception(nn.Module):
            """GoogLeNet's Inception block: four branches side by side, their outputs stacked along the maps."""

            def __init__(self, in_channels, c1, c3_reduce, c3, c5_reduce, c5, pool_proj):
                super().__init__()
                self.branch1 = make_branch1(in_channels, c1)                     # step (b)
                self.branch2 = make_branch2(in_channels, c3_reduce, c3)          # step (b)
                self.branch3 = make_branch3(in_channels, c5_reduce, c5)          # step (c)
                self.branch4 = make_branch4(in_channels, pool_proj)              # step (c)

            def forward(self, x):
                raise NotImplementedError   # delete this line once every ____ is filled in
                y1 = self.branch1(x)        # branch 1 applied to x: c1 maps (the example)
                y2 = ____                   # YOUR CODE: branch 2 applied to x: c3 maps
                y3 = ____                   # YOUR CODE: branch 3 applied to x: c5 maps
                y4 = ____                   # YOUR CODE: branch 4 applied to x: pool_proj maps
                out = ____                  # YOUR CODE: torch.cat of the list [y1, y2, y3, y4] along dim=1
                return out
        ''',
        solution='''
        class Inception(nn.Module):
            """GoogLeNet's Inception block: four branches side by side, their outputs stacked along the maps."""

            def __init__(self, in_channels, c1, c3_reduce, c3, c5_reduce, c5, pool_proj):
                super().__init__()
                self.branch1 = make_branch1(in_channels, c1)                     # step (b)
                self.branch2 = make_branch2(in_channels, c3_reduce, c3)          # step (b)
                self.branch3 = make_branch3(in_channels, c5_reduce, c5)          # step (c)
                self.branch4 = make_branch4(in_channels, pool_proj)              # step (c)

            def forward(self, x):
                y1 = self.branch1(x)        # branch 1 applied to x: c1 maps (the example)
                y2 = self.branch2(x)        # YOUR CODE: branch 2 applied to x: c3 maps
                y3 = self.branch3(x)        # YOUR CODE: branch 3 applied to x: c5 maps
                y4 = self.branch4(x)        # YOUR CODE: branch 4 applied to x: pool_proj maps
                out = torch.cat([y1, y2, y3, y4], dim=1)    # YOUR CODE: torch.cat of the list [y1, y2, y3, y4] along dim=1
                return out
        '''),
    md(r"""
    **Check.** Besides the shapes and the parameter count, two checks look inside: "branch order" compares each
    group of output maps with the output of each branch (maps 0 to 63 must be branch 1's, and so on), and the
    next one copies your block's weights into the reference block and compares the two outputs number by number.
    """),
    code("""
    T.check_inception(Inception)
    """),
    md(r"""
    **What you built.** Your block 3a. Compare it with the target in section 1: the same four branches, the
    same shapes, the same 163 696 parameters.
    """),
    code("""
    Inception = T.yours_or_reference(Inception)
    block_3a = Inception(192, 64, 96, 128, 16, 32, 32)
    P.show_model(block_3a, (192, 28, 28), title="Your Inception block 3a")
    """),

    # ------------------------------------------------------------------ (e)
    md(r"""
    ### Step (e) · The stem

    GoogLeNet reads photos of 3 x 224 x 224 (3 colour channels: red, green, blue). Running Inception blocks on
    pictures that big would be very slow, so a few plain layers come first: the **stem** shrinks the picture 8
    times in height and width (224 -> 28) while making 192 maps. `make_stem()` returns these eight layers as one
    `nn.Sequential`:

    | # | layer | in | out | kernel | stride | padding | output for one photo | the size rule |
    |---|---|---|---|---|---|---|---|---|
    | 1 | `nn.Conv2d` | 3 | 64 | 7 | 2 | 3 | 64 x 112 x 112 | (224 + 6 - 7) / 2 + 1 = 112.5 -> 112 |
    | 2 | `nn.ReLU` | | | | | | 64 x 112 x 112 | |
    | 3 | `nn.MaxPool2d` | | | 3 | 2 | 1 | 64 x 56 x 56 | (112 + 2 - 3) / 2 + 1 = 56.5 -> 56 |
    | 4 | `nn.Conv2d` | 64 | 64 | 1 | 1 | 0 | 64 x 56 x 56 | |
    | 5 | `nn.ReLU` | | | | | | 64 x 56 x 56 | |
    | 6 | `nn.Conv2d` | 64 | 192 | 3 | 1 | 1 | 192 x 56 x 56 | (56 + 2 - 3) / 1 + 1 = 56 |
    | 7 | `nn.ReLU` | | | | | | 192 x 56 x 56 | |
    | 8 | `nn.MaxPool2d` | | | 3 | 2 | 1 | 192 x 28 x 28 | (56 + 2 - 3) / 2 + 1 = 28.5 -> 28 |

    Here every layer is written out with `nn.Conv2d` and `nn.ReLU`, as in P1, instead of `conv_relu`: the stem
    is a plain chain, and this way each of its layers gets its own box when we draw the whole network in
    section 3. (The paper also had two "local response normalisation" layers in the stem; later networks
    dropped them, and torchvision leaves them out too, so we do as well.)

    **Your turn.** Layers 1 and 2 are the example; type layers 3 to 8, one per blank, and delete the `raise` line.
    """),
    exercise(
        stub='''
        def make_stem():
            """The stem of GoogLeNet: a 3 x 224 x 224 photo -> 192 maps of 28 x 28."""
            raise NotImplementedError   # delete this line once every ____ is filled in
            stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),     # 1: the example   -> 64 x 112 x 112
                nn.ReLU(),                                                # 2: the example
                ____,      # YOUR CODE: 3, nn.MaxPool2d, kernel_size=3, stride=2, padding=1   -> 64 x 56 x 56
                ____,      # YOUR CODE: 4, nn.Conv2d from 64 to 64 maps, kernel_size=1
                ____,      # YOUR CODE: 5, nn.ReLU()
                ____,      # YOUR CODE: 6, nn.Conv2d from 64 to 192 maps, kernel_size=3, padding=1   -> 192 x 56 x 56
                ____,      # YOUR CODE: 7, nn.ReLU()
                ____,      # YOUR CODE: 8, nn.MaxPool2d, kernel_size=3, stride=2, padding=1   -> 192 x 28 x 28
            )
            return stem
        ''',
        solution='''
        def make_stem():
            """The stem of GoogLeNet: a 3 x 224 x 224 photo -> 192 maps of 28 x 28."""
            stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),     # 1: the example   -> 64 x 112 x 112
                nn.ReLU(),                                                # 2: the example
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),         # YOUR CODE: 3, nn.MaxPool2d, kernel_size=3, stride=2, padding=1   -> 64 x 56 x 56
                nn.Conv2d(64, 64, kernel_size=1),                         # YOUR CODE: 4, nn.Conv2d from 64 to 64 maps, kernel_size=1
                nn.ReLU(),                                                # YOUR CODE: 5, nn.ReLU()
                nn.Conv2d(64, 192, kernel_size=3, padding=1),             # YOUR CODE: 6, nn.Conv2d from 64 to 192 maps, kernel_size=3, padding=1   -> 192 x 56 x 56
                nn.ReLU(),                                                # YOUR CODE: 7, nn.ReLU()
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),         # YOUR CODE: 8, nn.MaxPool2d, kernel_size=3, stride=2, padding=1   -> 192 x 28 x 28
            )
            return stem
        '''),
    code("""
    T.check_stem(make_stem)
    """),
    md(r"""
    **What you built.** The boxes get shorter (224 -> 112 -> 56 -> 28) and wider (3 -> 64 -> 192 maps). The 7x7
    convolution with stride 2 does most of the shrinking in one go.
    """),
    code("""
    make_stem = T.yours_or_reference(make_stem)
    P.show_model(make_stem(), (3, 224, 224), title="Your stem: 3 x 224 x 224 photo -> 192 x 28 x 28")
    """),

    # ------------------------------------------------------------------ (f)
    md(r"""
    ### Step (f) · The nine Inception blocks

    The published table of GoogLeNet (Szegedy et al., 2014, table 1) gives the seven numbers of each block, in
    the order of our `Inception(...)`. The blocks come in three **stages**, named after their place in the
    network (3, 4, 5); between two stages a max-pool with stride 2 halves height and width.

    | block | input size | `in_channels` | `c1` | `c3_reduce` | `c3` | `c5_reduce` | `c5` | `pool_proj` | maps out = c1 + c3 + c5 + pool_proj |
    |---|---|---|---|---|---|---|---|---|---|
    | 3a | 28 x 28 | 192 | 64 | 96 | 128 | 16 | 32 | 32 | 256 |
    | 3b | 28 x 28 | 256 | 128 | 128 | 192 | 32 | 96 | 64 | 480 |
    | pool3 | max-pool 3x3, stride 2, padding 1: 28 x 28 -> 14 x 14 | | | | | | | | 480 |
    | 4a | 14 x 14 | 480 | 192 | 96 | 208 | 16 | 48 | 64 | 512 |
    | 4b | 14 x 14 | 512 | 160 | 112 | 224 | 24 | 64 | 64 | 512 |
    | 4c | 14 x 14 | 512 | 128 | 128 | 256 | 24 | 64 | 64 | 512 |
    | 4d | 14 x 14 | 512 | 112 | 144 | 288 | 32 | 64 | 64 | 528 |
    | 4e | 14 x 14 | 528 | 256 | 160 | 320 | 32 | 128 | 128 | 832 |
    | pool4 | max-pool 3x3, stride 2, padding 1: 14 x 14 -> 7 x 7 | | | | | | | | 832 |
    | 5a | 7 x 7 | 832 | 256 | 160 | 320 | 32 | 128 | 128 | 832 |
    | 5b | 7 x 7 | 832 | 384 | 192 | 384 | 48 | 128 | 128 | 1024 |

    Check the chain as you type: each block's `in_channels` is the maps out of the block before it (a max-pool
    keeps the number of maps). For 3b: 64 + 128 + 32 + 32 = 256 maps come out of 3a.

    The blocks go into an empty `nn.Sequential()` one by one with `blocks.add_module(name, layer)`, which appends
    a layer under a name of your choice; the names ("inception3a", ...) appear in the drawings.

    **Your turn.** Block 3a is the example. Type `Inception(...)` with the seven numbers of each row into the
    eight blanks, and delete the `raise` line.
    """),
    api("Module.add_module"),
    exercise(
        stub='''
        def make_inception_blocks():
            """The nine Inception blocks 3a ... 5b and the two max-pools between them: 192 x 28 x 28 -> 1024 x 7 x 7."""
            raise NotImplementedError   # delete this line once every ____ is filled in
            blocks = nn.Sequential()
            # stage 3: 28 x 28
            blocks.add_module("inception3a", Inception(192, 64, 96, 128, 16, 32, 32))    # the example: -> 256 maps
            blocks.add_module("inception3b", ____)      # YOUR CODE: Inception(...) with the seven numbers of row 3b   -> 480 maps
            blocks.add_module("pool3", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 28 x 28 -> 14 x 14
            # stage 4: 14 x 14
            blocks.add_module("inception4a", ____)      # YOUR CODE: Inception(...) with the numbers of row 4a   -> 512 maps
            blocks.add_module("inception4b", ____)      # YOUR CODE: row 4b   -> 512 maps
            blocks.add_module("inception4c", ____)      # YOUR CODE: row 4c   -> 512 maps
            blocks.add_module("inception4d", ____)      # YOUR CODE: row 4d   -> 528 maps
            blocks.add_module("inception4e", ____)      # YOUR CODE: row 4e   -> 832 maps
            blocks.add_module("pool4", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 14 x 14 -> 7 x 7
            # stage 5: 7 x 7
            blocks.add_module("inception5a", ____)      # YOUR CODE: row 5a   -> 832 maps
            blocks.add_module("inception5b", ____)      # YOUR CODE: row 5b   -> 1024 maps
            return blocks
        ''',
        solution='''
        def make_inception_blocks():
            """The nine Inception blocks 3a ... 5b and the two max-pools between them: 192 x 28 x 28 -> 1024 x 7 x 7."""
            blocks = nn.Sequential()
            # stage 3: 28 x 28
            blocks.add_module("inception3a", Inception(192, 64, 96, 128, 16, 32, 32))    # the example: -> 256 maps
            blocks.add_module("inception3b", Inception(256, 128, 128, 192, 32, 96, 64))   # YOUR CODE: Inception(...) with the seven numbers of row 3b   -> 480 maps
            blocks.add_module("pool3", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 28 x 28 -> 14 x 14
            # stage 4: 14 x 14
            blocks.add_module("inception4a", Inception(480, 192, 96, 208, 16, 48, 64))    # YOUR CODE: Inception(...) with the numbers of row 4a   -> 512 maps
            blocks.add_module("inception4b", Inception(512, 160, 112, 224, 24, 64, 64))   # YOUR CODE: row 4b   -> 512 maps
            blocks.add_module("inception4c", Inception(512, 128, 128, 256, 24, 64, 64))   # YOUR CODE: row 4c   -> 512 maps
            blocks.add_module("inception4d", Inception(512, 112, 144, 288, 32, 64, 64))   # YOUR CODE: row 4d   -> 528 maps
            blocks.add_module("inception4e", Inception(528, 256, 160, 320, 32, 128, 128))  # YOUR CODE: row 4e   -> 832 maps
            blocks.add_module("pool4", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 14 x 14 -> 7 x 7
            # stage 5: 7 x 7
            blocks.add_module("inception5a", Inception(832, 256, 160, 320, 32, 128, 128))  # YOUR CODE: row 5a   -> 832 maps
            blocks.add_module("inception5b", Inception(832, 384, 192, 384, 48, 128, 128))  # YOUR CODE: row 5b   -> 1024 maps
            return blocks
        '''),
    md(r"""
    **Check.** One line per block: the check reads the seven numbers back from the block's six convolutions and
    compares them with the row of the table. If they belong to another row, the hint says which.
    """),
    code("""
    T.check_inception_blocks(make_inception_blocks)
    """),
    md(r"""
    **What you built.** `depth=1` draws each block as one box (a stacked box: a block of layers) instead of
    opening it up. The maps grow from 192 to 1024 while the picture shrinks from 28 x 28 to 7 x 7.
    """),
    code("""
    make_inception_blocks = T.yours_or_reference(make_inception_blocks)
    P.show_model(make_inception_blocks(), (192, 28, 28), depth=1, title="Your nine Inception blocks, 3a to 5b")
    """),

    # ------------------------------------------------------------------ (g)
    md(r"""
    ### Step (g) · The head

    After block 5b there are 1024 maps of 7 x 7. The head turns them into one **score** per class (1000 classes
    for ImageNet), in three steps:

    * **global average pooling**: each map becomes one number, its average over all 7 x 7 positions:
      1024 x 7 x 7 -> 1024 x 1 x 1. `nn.AdaptiveAvgPool2d(1)` does this for maps of any size (output size 1);
    * `nn.Flatten()` (P1) lays the 1024 x 1 x 1 numbers out in a row: 1024 numbers (given);
    * **dropout**, `nn.Dropout(p=0.4)`: during training it sets 40% of these numbers to 0, a different random 40%
      for every picture, so that the network cannot rely on a few of them; when the network is used to predict,
      it does nothing. It has no parameters;
    * `nn.Linear(in_channels, num_classes)` (P1): 1024 numbers in, 1000 scores out.

    | layer | settings | output |
    |---|---|---|
    | `nn.AdaptiveAvgPool2d` | output size 1 | 1024 x 1 x 1 |
    | `nn.Flatten` (given) | | 1024 |
    | `nn.Dropout` | `p=0.4` | 1024 |
    | `nn.Linear` | `in_channels` -> `num_classes` | 1000 |

    This is a big saving. VGG-16 (2014, the network GoogLeNet is compared with in section 3) flattens its last
    512 maps of 7 x 7 into 25 088 numbers and connects every one of them to 4096 neurons: 102.8 million weights in
    a single layer. GoogLeNet averages first, so its only linear layer has 1024 x 1000 weights.

    **Your turn.** Fill in the three layers and delete the `raise` line. Use the names `in_channels` and
    `num_classes` (section 4 builds a head with other numbers).
    """),
    api("nn.AdaptiveAvgPool2d", "nn.Dropout"),
    api("nn.Flatten", "nn.Linear", brief=True),
    exercise(
        stub='''
        def make_head(in_channels, num_classes):
            """The head: one average per map, dropout, then a linear layer that gives num_classes scores."""
            raise NotImplementedError   # delete this line once every ____ is filled in
            head = nn.Sequential(
                ____,             # YOUR CODE: nn.AdaptiveAvgPool2d with output size 1: the average of each map
                nn.Flatten(),     # in_channels x 1 x 1 -> in_channels numbers in a row (given)
                ____,             # YOUR CODE: nn.Dropout with p=0.4
                ____,             # YOUR CODE: nn.Linear from in_channels to num_classes
            )
            return head
        ''',
        solution='''
        def make_head(in_channels, num_classes):
            """The head: one average per map, dropout, then a linear layer that gives num_classes scores."""
            head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),                 # YOUR CODE: nn.AdaptiveAvgPool2d with output size 1: the average of each map
                nn.Flatten(),     # in_channels x 1 x 1 -> in_channels numbers in a row (given)
                nn.Dropout(p=0.4),                       # YOUR CODE: nn.Dropout with p=0.4
                nn.Linear(in_channels, num_classes),     # YOUR CODE: nn.Linear from in_channels to num_classes
            )
            return head
        '''),
    code("""
    T.check_head(make_head)
    """),
    code("""
    make_head = T.yours_or_reference(make_head)
    P.show_model(make_head(1024, 1000), (1024, 7, 7), title="Your head: 1024 maps of 7 x 7 -> 1000 scores")
    """),

    # ------------------------------------------------------------------ (h)
    md(r"""
    ### Step (h) · The whole network

    The last class puts the three parts together. `__init__` (given) builds the stem, the blocks and the head with
    your functions; `forward` sends the photo through them, in that order, each one receiving what the one before
    produced:

    | part | in | out |
    |---|---|---|
    | `self.stem` | 3 x 224 x 224 | 192 x 28 x 28 |
    | `self.blocks` | 192 x 28 x 28 | 1024 x 7 x 7 |
    | `self.head` | 1024 x 7 x 7 | `num_classes` scores |

    **Your turn.** Three lines: each is `x = ` a part applied to `x`, as `y1 = self.branch1(x)` in step (d).
    """),
    exercise(
        stub='''
        class GoogLeNet(nn.Module):
            """GoogLeNet (Inception v1) without the auxiliary classifiers: stem, nine Inception blocks, head."""

            def __init__(self, num_classes=1000):
                super().__init__()
                self.stem = make_stem()                       # step (e): 3 x 224 x 224 -> 192 x 28 x 28
                self.blocks = make_inception_blocks()         # step (f): 192 x 28 x 28 -> 1024 x 7 x 7
                self.head = make_head(1024, num_classes)      # step (g): 1024 x 7 x 7 -> num_classes scores

            def forward(self, x):
                raise NotImplementedError   # delete this line once every ____ is filled in
                x = ____        # YOUR CODE: the stem applied to x
                x = ____        # YOUR CODE: the Inception blocks applied to x
                x = ____        # YOUR CODE: the head applied to x
                return x
        ''',
        solution='''
        class GoogLeNet(nn.Module):
            """GoogLeNet (Inception v1) without the auxiliary classifiers: stem, nine Inception blocks, head."""

            def __init__(self, num_classes=1000):
                super().__init__()
                self.stem = make_stem()                       # step (e): 3 x 224 x 224 -> 192 x 28 x 28
                self.blocks = make_inception_blocks()         # step (f): 192 x 28 x 28 -> 1024 x 7 x 7
                self.head = make_head(1024, num_classes)      # step (g): 1024 x 7 x 7 -> num_classes scores

            def forward(self, x):
                x = self.stem(x)        # YOUR CODE: the stem applied to x
                x = self.blocks(x)      # YOUR CODE: the Inception blocks applied to x
                x = self.head(x)        # YOUR CODE: the head applied to x
                return x
        '''),
    md(r"""
    **Check.** The whole network against the reference, and then against **torchvision**'s GoogLeNet
    (torchvision is PyTorch's library of ready-made vision networks), an implementation written by other
    people. It takes a second or two.
    """),
    code("""
    T.check_googlenet(GoogLeNet)
    """),
    md(r"""
    **Reading the last lines.** Your network has the same 57 convolutions, in the same order and with the same
    numbers of maps, as torchvision's, and the same output. Their parameter counts differ (6 998 552 against
    6 624 904), and the check explains the difference to the last parameter: torchvision made two changes to
    the paper's design.

    * Its "5x5" branches use 3x3 filters (a known slip, kept so that its trained weights still fit): 380 928
      fewer weights.
    * After every convolution it uses **batch normalisation** (a layer that rescales each map to an average of
      0 and a spread of 1, then learns its own scale and shift: 2 parameters per map) instead of a bias (1 per
      map): 7 280 more parameters, one per map of the 57 convolutions.

    The parameter column of the paper's table adds up to about 6.8 million. Counting the layers of that same
    table exactly, as your network does, gives 6 998 552, about 7.0 million: the paper's per-block figures are a
    little lower than exact counts (for block 3a it lists 159 thousand, while the layers of 3a hold 163 696, the
    number your check printed in step (d)).

    **What you built.** `depth=1` shows the three parts of your network, each as one box.
    """),
    code("""
    GoogLeNet = T.yours_or_reference(GoogLeNet)
    net = GoogLeNet()
    P.show_model(net, (3, 224, 224), depth=1, title="Your GoogLeNet: stem, blocks, head")
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · Look at what you built

    ### Block by block

    `depth=2` opens the three parts one level: the eight layers of the stem, the nine blocks with the two
    max-pools between the stages, and the four layers of the head.
    """),
    code("""
    P.show_model(net, (3, 224, 224), depth=2, title="GoogLeNet block by block")
    """),
    md(r"""
    **What to look at.** The picture shrinks in five places only, each time by half: three times in the stem (the
    stride-2 convolution and the two max-pools: 224 -> 112 -> 56 -> 28), then at pool3 (28 -> 14) and pool4
    (14 -> 7). In between, each stage keeps its size while the blocks add maps. The deepest blocks hold the most parameters (5b: 1 444 080), because they read and make
    the most maps.

    One block opened up: block 4e, the block with the most maps going in among the 14 x 14 ones.
    `net.blocks.inception4e` is that block (the name given to `add_module` in step (f)).
    """),
    code("""
    P.show_model(net.blocks.inception4e, (528, 14, 14), title="Block 4e opened up")
    """),
    md(r"""
    `P.summary(model, input_shape, depth=2)` gives the same overview as a table: one row per layer of the stem and
    head and per block, with its output shape, parameters and **multiply-adds** (the multiplications and
    additions it needs for one photo: a measure of its work).
    """),
    code("""
    P.summary(net, (3, 224, 224), depth=2)
    """),
    md(r"""
    `T.show_params_per_block(net)` draws the parameter column of that table: the stem, each block, the head.
    """),
    code("""
    T.show_params_per_block(net)
    """),
    md(r"""
    **What to look at.** The parameters grow block by block, because the blocks deeper in the network read and
    make more maps; 5b alone holds 1 444 080. The head, the part that costs VGG-16 most (see below), holds
    1 025 000: less than block 5b.

    ### What it costs

    `P.cost(model, input_shape)` (P1) measures the network on this computer. `batch_size=8` times the training
    step on 8 photos at a time, to keep the memory small. `epoch_images=1281167` asks how long one **epoch**
    (one pass over the training photos) of ImageNet, the 1.28 million photos GoogLeNet was trained on, would
    take here. **Expected time: a few seconds.**
    """),
    code("""
    P.cost(net, (3, 224, 224), name="GoogLeNet", batch_size=8, epoch_images=1281167)
    """),
    md(r"""
    **Reading the report.** The 6 998 552 parameters take 26.70 MB (4 bytes each). One photo needs 1.58 billion
    multiply-adds; on this computer a prediction takes a few tens of milliseconds. Training is another matter: the
    last line, in minutes, says that one single pass over ImageNet would take roughly a day here, and a network
    like this needs dozens of passes. The authors estimated that GoogLeNet could be trained in about a week on a
    few high-end graphics cards.

    ### Compared with VGG-16

    **VGG-16** (Simonyan and Zisserman, 2014) came second to GoogLeNet in the same competition. It is a single chain of
    13 3x3 convolutions and 3 linear layers: 16 layers with weights. GoogLeNet is 22 such layers deep along its
    longest path (3 in the stem, 2 in each of the 9 blocks, 1 in the head). `torchvision.models.vgg16(weights=None)`
    builds VGG-16 with random weights (nothing is downloaded). First, where its parameters are:
    """),
    api("torchvision.models.vgg16"),
    code("""
    vgg = torchvision.models.vgg16(weights=None)
    P.summary(vgg, (3, 224, 224), depth=1)
    """),
    md(r"""
    VGG-16 has 138 357 544 parameters, and 123 642 856 of them (89%) are in `classifier`, its three linear
    layers at the end. Its convolutions (`features`) hold 14.7 million.

    `P.compare` (P1) measures both networks side by side. It also times a training step, and training VGG-16
    needs a lot of memory: its parameters alone take 528 MB, and a training step needs several times that (the
    changes computed for every weight, and the optimiser's own numbers for every weight). `batch_size=2` keeps the
    step as small as possible; even so, the notebook used about 4 GB of memory during this cell here. **Expected
    time: about 10 seconds** (up to a minute on a slow laptop). On a laptop with 8 GB of memory or less, skip it.
    """),
    code("""
    P.compare({"GoogLeNet": net, "VGG-16": vgg}, (3, 224, 224), batch_size=2)
    """),
    md(r"""
    **What the numbers say.** VGG-16 has about 20 times the parameters of GoogLeNet (138.4 million against
    7.0 million) and about 10 times the multiply-adds (15.47 billion against 1.58 billion per photo). On this
    computer it was several times slower per photo and per training step (the measured times vary from run to
    run). The times do not always shrink as much as the multiply-adds: GoogLeNet's many small layers each cost a
    little time of their own.

    **Why GoogLeNet is so much smaller at a similar depth:**

    * **1x1 reductions.** Every 3x3 and 5x5 convolution reads a reduced number of maps (section 1: the 5x5 branch
      of block 3a needs almost 10 times fewer weights). VGG's 3x3 convolutions read up to 512 maps each.
    * **Global average pooling instead of huge linear layers.** VGG flattens 512 x 7 x 7 = 25 088 numbers into
      linear layers of 4096 neurons (124 million parameters); GoogLeNet averages each of its 1024 maps to one
      number and needs one linear layer of 1 025 000 parameters.
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · Train a small GoogLeNet on CIFAR-10 (optional on slow computers)

    Training the full GoogLeNet on ImageNet takes days even on graphics cards. Here we train a small version
    built from **your** `Inception` class (or the reference one, if step (d) is not finished) on **CIFAR-10**:
    60 000 small colour photos of 32 x 32 pixels in 10 classes, which the dataset names airplane, automobile,
    bird, cat, deer, dog, frog, horse, ship and truck. `P.get_cifar10(...)` loads them (it downloads about 170 MB
    into `data/` the first time);
    we train on a part of the training photos and measure on 2 000 other photos (the **validation** photos).
    """),
    code("""
    train_loader, val_loader, classes = P.get_cifar10(n_train=10000, n_val=2000, batch_size=64)
    P.show_samples(train_loader, classes)
    """),
    md(r"""
    ### What changes for 32 x 32 photos

    The full GoogLeNet would shrink a 32 x 32 photo to 4 x 4 in the stem and to 1 x 1 by stage 5, and it is far
    too slow to train on a laptop. `SmallGoogLeNet` makes four changes:

    1. **A lighter stem**: one 3x3 convolution (3 -> 48 maps, padding 1) with its ReLU and one 3x3 max-pool with
       stride 2: 32 x 32 -> 16 x 16 (instead of the 7x7 convolution, two pools and two more convolutions, which
       shrink 224 to 28).
    2. **Three blocks instead of nine**: 3a and 3b at 16 x 16, a max-pool to 8 x 8, then 4a. Each has the numbers
       of the published table **divided by 4**: 3a is `Inception(48, 16, 24, 32, 4, 8, 8)` instead of
       `Inception(192, 64, 96, 128, 16, 32, 32)`.
    3. **The same head** (your `make_head`), with 128 maps in and 10 classes.
    4. **He initialisation** of the convolutions. A network's weights start as random numbers. PyTorch's default
       random numbers are small; without batch normalisation (which GoogLeNet did not have yet), the signal
       shrinks a little in every convolution, and after 7 of them in a row the network starts learning very
       slowly. `nn.init.kaiming_normal_` draws each convolution's starting weights with the spread that suits a
       layer followed by a ReLU (He et al., 2015). `self.modules()` lists every layer inside the network, and
       `isinstance(layer, nn.Conv2d)` picks the convolutions.
    """),
    api("nn.init.kaiming_normal_", "Module.modules"),
    code('''
    class SmallGoogLeNet(nn.Module):
        """GoogLeNet for 32 x 32 photos: a light stem, blocks 3a, 3b and 4a at a quarter of their width, the same head."""

        def __init__(self, num_classes=10):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 48, kernel_size=3, padding=1),           # 3 x 32 x 32 -> 48 x 32 x 32
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),     # -> 48 x 16 x 16
            )
            self.blocks = nn.Sequential()
            self.blocks.add_module("inception3a", Inception(48, 16, 24, 32, 4, 8, 8))        # -> 64 x 16 x 16
            self.blocks.add_module("inception3b", Inception(64, 32, 32, 48, 8, 24, 16))      # -> 120 x 16 x 16
            self.blocks.add_module("pool3", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))    # -> 120 x 8 x 8
            self.blocks.add_module("inception4a", Inception(120, 48, 24, 52, 4, 12, 16))     # -> 128 x 8 x 8
            self.head = make_head(128, num_classes)                                           # -> 10 scores
            # He initialisation: start every convolution with weights of the size that suits a ReLU network
            for layer in self.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        def forward(self, x):
            x = self.stem(x)
            x = self.blocks(x)
            x = self.head(x)
            return x
    '''),
    md(r"""
    `torch.manual_seed(0)` (P1) makes the random starting weights the same on every run.
    """),
    api("torch.manual_seed", brief=True),
    code("""
    torch.manual_seed(0)
    small_net = SmallGoogLeNet()
    P.show_model(small_net, (3, 32, 32), depth=2, title="SmallGoogLeNet for 32 x 32 photos")
    P.cost(small_net, (3, 32, 32), name="SmallGoogLeNet")
    """),
    md(r"""
    **The small network** has 60 978 parameters (GoogLeNet: 6 998 552) and needs 11.65 million multiply-adds per
    photo (GoogLeNet: 1.58 billion for its 224 x 224 photos). The last line of the cost report estimates the time
    of one epoch over 10 000 photos on your computer.

    ### Try it: the run settings

    The network is fixed; you may change how it is trained. Run this cell, then the training cell.

    * `epochs`: passes over the training photos;
    * `learning_rate`: the size of each weight change (P1);
    * `n_train`: how many training photos (at most 50 000). The time per epoch grows in proportion.
    """),
    code("""
    # ---------------- Try it: run settings ----------------
    epochs = 5                    # passes over the training photos
    learning_rate = 0.001         # size of each weight change
    n_train = 10000               # training photos (at most 50000)
    version_name = "small GoogLeNet"   # the name on the curves and in the leaderboard
    """),
    md(r"""
    `P.train` (P1) trains with the Adam optimiser and draws the loss and accuracy curves after every epoch. The
    starting weights are reset with `torch.manual_seed(0)` and a new `SmallGoogLeNet`, so every run starts from
    the same point. **Expected time: about 2 minutes** with the settings above (about 20 seconds per epoch here;
    up to twice as long on a slow laptop).
    """),
    code("""
    train_loader, val_loader, classes = P.get_cifar10(n_train=n_train, n_val=2000, batch_size=64)
    torch.manual_seed(0)
    small_net = SmallGoogLeNet()
    history = P.train(small_net, train_loader, val_loader, epochs=epochs, lr=learning_rate, name=version_name)
    """),
    md(r"""
    **Reading the curves.** After the first epoch the network gets 33.1% of the validation photos right; after the
    fifth, 46.5%: almost one in two, where guessing would give one in ten. Both accuracy curves still rise at the
    end, so more epochs would help (try it). The validation accuracy is above the training accuracy here, for two
    reasons: the training accuracy is an average over the epoch, while the network was still learning, and it is
    measured with dropout switched on (40% of the head's numbers set to 0); the validation accuracy is measured
    after each epoch, with dropout off. 46.5% is modest, after five passes over 10 000 small photos; the point is
    that a network built from your blocks learns. Run the cell again with the same settings and you get exactly
    the same numbers: the seed fixes the starting weights and the order of the photos.

    `P.show_predictions` (P1) shows the network's answers for the first 16 validation photos.
    """),
    code("""
    P.show_predictions(small_net, val_loader, classes)
    """),
    md(r"""
    Here 6 of the first 16 are right. Some mistakes are understandable (an airplane taken for a ship, both often
    seen against a plain blue or grey background; a dog taken for a cat), others less so (a truck taken for a deer).
    """),
    md(r"""
    `P.remember` and `P.leaderboard` (P1) keep a list of every run: change the settings in **Try it**, run the
    training cell and this one again, and compare.
    """),
    code("""
    P.remember(version_name, small_net, history)
    P.leaderboard()
    """),
    md(r"""
    **Things to try** (predict first, then run):

    * `epochs = 10`: does the validation accuracy keep rising, and does the gap to the training accuracy grow?
    * `n_train = 20000`: twice the photos, so about twice the time per epoch. Is it worth it?
    * `learning_rate = 0.01` and `learning_rate = 0.0001`: too big and too small (P1).
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · What to remember

    * An **Inception block** runs four branches side by side on the same input (1x1; 1x1 then 3x3; 1x1 then 5x5;
      3x3 max-pool then 1x1), all keeping height and width, and **concatenates** their maps:
      out = c1 + c3 + c5 + pool_proj maps.
    * A **1x1 convolution** mixes the maps at each pixel and changes their number. Placed before a 3x3 or 5x5
      convolution it **reduces** the maps that expensive filter reads: almost 10 times fewer weights for the 5x5
      branch of block 3a.
    * GoogLeNet = **stem** (224 -> 28) + nine Inception blocks in three stages, with max-pools halving the size
      between stages + **head** (global average pooling, dropout, one linear layer): 6 998 552 parameters.
    * Global average pooling instead of large linear layers, and the 1x1 reductions, make GoogLeNet about 20 times
      smaller than VGG-16 (138 million parameters), at a similar depth.
    * A network is built from a specification table, layer by layer: each layer's input maps equal the output
      maps of the layer before, and the output-size rule tells you every height and width.

    ### Words from this notebook

    | word | meaning |
    |---|---|
    | 1x1 convolution | a convolution with one-pixel filters: mixes the maps at each position, changes their number |
    | attribute | a value stored in an object, such as `self.branch1` in a block |
    | auxiliary classifier | a small extra head half-way up the original GoogLeNet, to help training; left out here |
    | batch normalisation | a layer that rescales each map to average 0 and spread 1, then learns a scale and a shift |
    | branch | one of the four chains of layers of an Inception block |
    | CIFAR-10 | 60 000 colour photos of 32 x 32 pixels in 10 classes |
    | class (Python) | a recipe for objects; a network block is a class built on `nn.Module` |
    | concatenate | stack tensors along an axis; here the maps of the four branches along the channel axis |
    | default value | the value an argument takes when a call leaves it out, such as `stride=1` |
    | dropout | during training, setting a random part of the numbers to 0; nothing at prediction time |
    | epoch | one pass over all training photos |
    | feature map (map, channel) | one output picture of a convolution, one per filter |
    | forward | the method of a module that says what happens to an input |
    | global average pooling | replacing each map by the average of all its values |
    | GoogLeNet (Inception v1) | the 2014 network built from Inception blocks |
    | He initialisation | random starting weights whose spread suits layers followed by a ReLU |
    | head | the last layers of a classifier, turning maps into one score per class |
    | ImageNet | 1.28 million training photos in 1000 classes, the benchmark of GoogLeNet and VGG |
    | Inception block | four branches side by side, outputs concatenated |
    | multiply-adds | the multiplications and additions a network needs for one photo: a measure of its work |
    | reduce (reduction) | a 1x1 convolution that cuts the number of maps before an expensive convolution |
    | reference version | the course's own finished code, used when a step is not finished |
    | stage | a group of blocks working at the same picture size (3: 28 x 28, 4: 14 x 14, 5: 7 x 7) |
    | stem | the first plain layers, shrinking the photo before the Inception blocks |
    | torchvision | PyTorch's library of ready-made vision networks and datasets |
    | validation photos | photos kept apart from training, to measure the network |
    | VGG-16 | a 2014 network: one chain of 13 3x3 convolutions and 3 large linear layers |
    """),
]
