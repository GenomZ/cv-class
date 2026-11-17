from nb import md, code, exercise, api

NAME = "P4-unet"

CELLS = [
    md(r"""
    # P4 · U-Net: build an encoder-decoder layer by layer

    **What we do today**

    1. Look at a new task, **segmentation**: instead of one answer for the whole picture, one class for
       every pixel.
    2. See why the networks so far cannot do it, and how the **U-Net** does: an encoder that shrinks the
       picture, a decoder that grows it back, and skip connections between them.
    3. Build the U-Net yourself in four small steps, one building block at a time. Every step has a table, a
       code template, a test and a picture.
    4. Measure your network (parameters, time) and compare it with the full-size U-Net of the original paper.
    5. Train it on generated pictures of shapes (predefined, about half a minute) and look at its predictions.
    6. Switch off the skip connections and see what they are for (section 8, **Try it**).

    **How to use this notebook.** Run the cells from top to bottom with **Shift+Enter**. In four places you
    type code into a template:

    * a template contains blanks `____`; replace each one with what the `# YOUR CODE:` comment next to it
      says. Everything else is already written: you only type layers and the lines that use them;
    * the template's first line is `raise NotImplementedError   # delete this line once every ____ is filled
      in`: delete it when you are done;
    * run the template cell (it defines a block and prints nothing), then the **test cell** below it. The test
      prints one line per property: `[PASS]` or `[FAIL]` with a hint. Fix what a `[FAIL]` line names, run the
      template cell again, then the test cell again;
    * **you cannot get stuck**: if a block does not pass its test yet, the cells below use the course's
      reference block instead (the test cell says so), so the rest of the notebook still runs. The complete
      solution is in `solutions/P4-unet-solution.ipynb`.

    The whole notebook computes for about 1.5 minutes on the course computer (8 CPU threads), up to 5 minutes on
    a slower laptop; the two training cells say how long they take.
    `P.` is the course toolkit you know from P1 (`from cvcourse import practical as P`); `R.` holds this
    notebook's data and pictures and the reference blocks, `T.` the tests. Closed boxes **Functions used here**
    explain the PyTorch functions of the next cell. **New words** are in bold where they are explained and
    collected at the end.
    """),

    # ------------------------------------------------------------------ 0
    md(r"""
    ## 0 · Setup

    As in P1: find the folder `cvcourse`, import PyTorch and the toolkit, fix the random seed. Two more
    imports: `pmodels_p4` (short `R`, for reference) and `ptests_p4` (short `T`, for tests).
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
    from cvcourse import practical as P
    from cvcourse import pmodels_p4 as R            # this notebook's data, pictures and reference blocks
    from cvcourse import ptests_p4 as T             # the tests after every step

    P.setup(seed=0)
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · The task: a class for every pixel

    In P1 a network looked at a picture and gave **one** answer: "this is a 7". In **segmentation** it gives an
    answer for **every pixel**: this pixel is background, this one belongs to a circle, this one to a square.
    The answer for a whole picture is a **mask**: a grid of the same height and width as the picture, holding
    one class number per pixel.

    Our data are generated pictures of 64 x 64 pixels with circles, squares and triangles on a light
    background; a computer draws them, so every mask is exactly right and nothing needs downloading. There are
    4 classes: 0 background, 1 circle, 2 square, 3 triangle.

    `R.get_data(...)` makes 1 200 training pictures and 200 other validation pictures, with their masks, and
    returns the two loaders (batches of 16 pictures) and the class names. `R.show_examples(loader)` draws the
    first six validation pictures above their masks, one colour per class.
    """),
    code("""
    train_loader, val_loader, classes = R.get_data("shapes", n_train=1200, n_val=200, batch_size=16)
    R.show_examples(val_loader)
    """),
    md(r"""
    The colours give a shape's class away (circles are drawn red, squares blue, triangles green), so finding
    *which* shape is easy here. The real work is the **outline**: every pixel on the border of a shape must
    get the right class. That is what the network we build today is good at.

    What does one batch look like? `next(iter(train_loader))` takes the first batch.
    """),
    api("Tensor.shape", "Tensor.unique", "Tensor.tolist"),
    code("""
    images, masks = next(iter(train_loader))       # the first batch
    print("images:", images.shape)
    print("masks: ", masks.shape)
    print("the classes in the first mask:", masks[0].unique().tolist())
    """),
    md(r"""
    | tensor | shape | meaning |
    |---|---|---|
    | `images` | `[16, 3, 64, 64]` | 16 pictures, 3 colour channels (red, green, blue), 64 x 64 pixels |
    | `masks` | `[16, 64, 64]` | 16 masks: for each pixel one class number (0 to 3), no channel axis |

    So the network receives `3 x 64 x 64` per picture and must answer with **4 scores for every pixel**: an
    output of shape `4 x 64 x 64`, one map of scores per class. For each pixel, the class with the highest
    score is the network's answer. The four maps are the network's output **channels** (also called
    **maps**): each channel is a 64 x 64 grid of numbers.

    ### Why a classifier cannot do it

    Here is a small classifier in the style of P1, for our 3 x 64 x 64 pictures and 4 classes: convolutions
    and pooling layers, then flatten and a linear layer.
    """),
    api("torch.manual_seed", "nn.Sequential", "nn.Conv2d", "nn.ReLU", "nn.MaxPool2d", "nn.Flatten", "nn.Linear"),
    code("""
    torch.manual_seed(0)
    classifier = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),     # 16 x 64 x 64
        nn.ReLU(),
        nn.MaxPool2d(2),                                # 16 x 32 x 32
        nn.Conv2d(16, 32, kernel_size=3, padding=1),    # 32 x 32 x 32
        nn.ReLU(),
        nn.MaxPool2d(2),                                # 32 x 16 x 16
        nn.Flatten(),                                   # 32 x 16 x 16 = 8192 numbers in a row
        nn.Linear(8192, 4),                             # 4 scores for the whole picture
    )
    P.show_model(classifier, (3, 64, 64), title="a classifier: one answer for the whole picture")
    """),
    md(r"""
    Follow the boxes: the picture shrinks (64, 32, 16 pixels), and at the end 4 numbers come out for the
    **whole** picture. They can say "there is a circle somewhere", not *where* it is.

    Could we drop flatten and the linear layer and stop at the 32 x 16 x 16 maps? They are still pictures, but
    each of their pixels stands for a 4 x 4 square of the original: far too coarse for an outline. The pooling
    layers that let the network see larger patterns (*what* is there) have thrown away the exact positions
    (*where* it is). We need a second half that grows the picture back to 64 x 64.

    ### The U-Net idea

    The **U-Net** (Ronneberger, Fischer and Brox, 2015, made for microscope images of cells) has three parts:

    * the **encoder** shrinks the picture step by step, like a classifier: the height and width halve, the
      number of maps doubles. It learns *what* is in the picture;
    * the **decoder** grows it back step by step: the height and width double, the number of maps halves. It
      puts the *what* back in its place, the *where*;
    * **skip connections** copy the encoder's maps of each size across to the decoder step of the same size.
      The decoder's maps come up from a tiny 4 x 4 picture and are blurred; the copied encoder maps still have
      every sharp edge. Glued together, the decoder gets both.

    The picture below is the map of the network you will build, with the names we give its blocks. Read it
    as a "U": down the left side, through the bottom, up the right side; the dashed red arrows are the skip
    connections. Every box shows the shape of what comes out of it (maps x height x width) for a 64 x 64
    picture and `base = 16` maps in the first block.
    """),
    code("""
    R.draw_unet(base=16, size=64, in_channels=3, n_classes=4)
    """),
    md(r"""
    **Reading the map.**

    * Left side, the encoder: `inc` turns the 3 colour channels into 16 maps (still 64 x 64); `down1` to
      `down4` each halve the size and double the maps: 32, 16, 8, then 4 x 4 pixels with 256 maps. The bottom
      block `down4` is called the **bottleneck**: the smallest picture with the most maps.
    * We keep the result of every encoder block in a variable: `x1` to `x5`. The decoder needs `x1` to `x4`
      again.
    * Right side, the decoder: `up1` to `up4` each double the size, take the skip of the same size (`up1`
      takes `x4`, ..., `up4` takes `x1`), and halve the maps: back to 16 maps of 64 x 64.
    * `outc` turns the 16 maps into 4 maps: 4 scores for every pixel, the output we need.

    `P.show_model` draws the same network from the actual PyTorch code, here the course's reference version.
    `depth=1` draws one box per block (a stacked box = a block of several layers). The blocks are drawn in
    the order the data flows, so the "U" is laid out flat, from left to right over two rows. The skip
    connections are the red arcs; a skip that continues on the next row is marked with a numbered circle at
    both ends.
    """),
    code("""
    P.show_model(R.UNet(in_channels=3, n_classes=4, base=16), (3, 64, 64), depth=1,
                 title="the target: the finished U-Net (the course's reference)")
    """),
    md(r"""
    10 blocks, 1 944 100 parameters. Under each box: its name, the kind of block, the shape of its output and
    its parameters. Each `Up` box is the end of a red arc: there the skip arrives. Most parameters sit at the
    bottom of the U: `down4` (886 272) and `up1` (574 336) hold three quarters of them, because there the
    blocks have the most maps.

    Now we build this network in four steps, from the smallest block up.
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · Step 1: `DoubleConv`, the basic block

    Every box of the U-Net except `outc` contains the same unit: two 3 x 3 convolutions, each followed by
    batch normalisation and a ReLU. We write it once as a block called `DoubleConv`.

    **Batch normalisation** (`nn.BatchNorm2d`) is the one new layer. During training it takes each map of the
    batch, shifts and scales its values so that they average 0 with a spread of 1 (as `P.get_mnist` did for
    the pixels in P1), then multiplies and shifts them by two numbers it learns for that map. Networks with
    many convolutions learn faster and more reliably with it. Its one setting is how many maps come in. It has
    2 learned parameters per map.

    A PyTorch block is a class with two parts (P1's `nn.Sequential` did this for us; here we write it out):

    * `__init__` creates the layers and stores them under names: `self.conv1 = ...`;
    * `forward(self, x)` says what happens to an input `x`: the layers in order, each line working on the
      result of the line before: `x = self.conv1(x)`.

    **The table.** `DoubleConv(in_channels, out_channels)`: `in_channels` maps in, `out_channels` maps out,
    height and width unchanged.

    | name | layer | in maps | out maps | kernel | stride | padding |
    |---|---|---|---|---|---|---|
    | `conv1` | `nn.Conv2d` | in_channels | out_channels | 3 | 1 | 1 |
    | `bn1` | `nn.BatchNorm2d` | out_channels | out_channels | | | |
    | `relu1` | `nn.ReLU` | | | | | |
    | `conv2` | `nn.Conv2d` | out_channels | out_channels | 3 | 1 | 1 |
    | `bn2` | `nn.BatchNorm2d` | out_channels | out_channels | | | |
    | `relu2` | `nn.ReLU` | | | | | |

    Padding 1 with a 3 x 3 kernel keeps the height and width (P1's output-size rule: (64 + 2 - 3) / 1 + 1 =
    64). Stride 1 is the default, so it does not need to be typed. The first layer is written for you as the
    worked example: `nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)`. Write the others the same
    way, using the names `in_channels` and `out_channels`, never fixed numbers: the U-Net uses this block with
    nine different sizes.
    """),
    api("nn.Module", "nn.BatchNorm2d", title="Functions you will need"),
    exercise(
        stub="""
        class DoubleConv(nn.Module):
            def __init__(self, in_channels, out_channels):
                raise NotImplementedError   # delete this line once every ____ is filled in
                super().__init__()
                # the layers, one per row of the table
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)   # given: the worked example
                self.bn1 = ____        # YOUR CODE: nn.BatchNorm2d for out_channels maps
                self.relu1 = ____      # YOUR CODE: nn.ReLU()
                self.conv2 = ____      # YOUR CODE: nn.Conv2d from out_channels to out_channels maps, kernel_size=3, padding=1
                self.bn2 = ____        # YOUR CODE: nn.BatchNorm2d for out_channels maps
                self.relu2 = ____      # YOUR CODE: nn.ReLU()

            def forward(self, x):
                # the layers in the table's order; each line works on the result of the line before
                x = self.conv1(x)      # given: the pattern
                x = ____               # YOUR CODE: self.bn1 applied to x
                x = ____               # YOUR CODE: self.relu1 applied to x
                x = ____               # YOUR CODE: self.conv2 applied to x
                x = ____               # YOUR CODE: self.bn2 applied to x
                x = ____               # YOUR CODE: self.relu2 applied to x
                return x
        """,
        solution="""
        class DoubleConv(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                # the layers, one per row of the table
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)   # given: the worked example
                self.bn1 = nn.BatchNorm2d(out_channels)                                        # nn.BatchNorm2d for out_channels maps
                self.relu1 = nn.ReLU()                                                         # nn.ReLU()
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # nn.Conv2d from out_channels to out_channels maps
                self.bn2 = nn.BatchNorm2d(out_channels)                                        # nn.BatchNorm2d for out_channels maps
                self.relu2 = nn.ReLU()                                                         # nn.ReLU()

            def forward(self, x):
                # the layers in the table's order; each line works on the result of the line before
                x = self.conv1(x)      # given: the pattern
                x = self.bn1(x)        # self.bn1 applied to x
                x = self.relu1(x)      # self.relu1 applied to x
                x = self.conv2(x)      # self.conv2 applied to x
                x = self.bn2(x)        # self.bn2 applied to x
                x = self.relu2(x)      # self.relu2 applied to x
                return x
        """),
    md(r"""
    **Test it.** `T.check_double_conv(DoubleConv)` builds `DoubleConv(3, 16)`, compares every layer with the
    table, runs it on a batch of 2 random pictures of 3 x 64 x 64, checks the output shape, the order of the
    layers in `forward` and the parameter count, and builds a `DoubleConv(16, 32)` too, to catch numbers typed
    in place of `in_channels` and `out_channels`. It returns `True` when every line says `[PASS]`.

    `T.yours_or_reference(...)` then decides which `DoubleConv` the rest of the notebook uses: yours if it
    passed, otherwise the course's reference (with a note).
    """),
    code("""
    ok_double_conv = T.check_double_conv(DoubleConv)
    DoubleConv = T.yours_or_reference(ok_double_conv, DoubleConv, "DoubleConv")
    """),
    md(r"""
    **Look at it.** The same picture as in P1, now of your block: six boxes in the order of `forward`.
    """),
    code("""
    P.show_model(DoubleConv(3, 16), (3, 64, 64), title="DoubleConv(3, 16)")
    """),
    md(r"""
    The thin pink and yellow boxes are the batch normalisations and ReLUs: they change the values, not the
    shape. Every output is `16x64x64`: the picture keeps its size, the 3 colour channels become 16 maps.

    The parameters, by hand: `conv1` has 16 x 3 x 3 x 3 weights + 16 biases = 448; `conv2` 16 x 16 x 3 x 3 + 16
    = 2 320; each batch normalisation 2 x 16 = 32. Together 448 + 32 + 2 320 + 32 = 2 832, the number the test
    printed.
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · Step 2: `Down`, one step down the encoder

    A `Down` block goes one level down the U: a **max-pooling** layer halves the height and width (it keeps the
    largest value of every 2 x 2 square, as in P1), then a `DoubleConv` changes the number of maps.

    **The table.** `Down(in_channels, out_channels)`: `in_channels` maps of H x W in, `out_channels` maps of
    H/2 x W/2 out.

    | name | layer | in maps | out maps | kernel | stride |
    |---|---|---|---|---|---|
    | `pool` | `nn.MaxPool2d` | in_channels | in_channels | 2 | 2 (the default: the same as the kernel) |
    | `conv` | `DoubleConv` (your step 1) | in_channels | out_channels | 3 (inside) | 1 (inside) |

    For example `Down(16, 32)` turns 16 x 64 x 64 into 16 x 32 x 32 (pooling), then into 32 x 32 x 32
    (`DoubleConv(16, 32)`). A block can use another block as a layer: `self.conv = DoubleConv(...)` puts your
    whole step-1 block inside this one.
    """),
    exercise(
        stub="""
        class Down(nn.Module):
            def __init__(self, in_channels, out_channels):
                raise NotImplementedError   # delete this line once every ____ is filled in
                super().__init__()
                self.pool = ____       # YOUR CODE: nn.MaxPool2d with a 2 x 2 window (kernel_size 2)
                self.conv = ____       # YOUR CODE: DoubleConv from in_channels to out_channels maps

            def forward(self, x):
                x = ____               # YOUR CODE: self.pool applied to x: half the height and width
                x = ____               # YOUR CODE: self.conv applied to x: out_channels maps
                return x
        """,
        solution="""
        class Down(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.pool = nn.MaxPool2d(2)                           # nn.MaxPool2d with a 2 x 2 window (kernel_size 2)
                self.conv = DoubleConv(in_channels, out_channels)     # DoubleConv from in_channels to out_channels maps

            def forward(self, x):
                x = self.pool(x)       # self.pool applied to x: half the height and width
                x = self.conv(x)       # self.conv applied to x: out_channels maps
                return x
        """),
    md(r"""
    **Test it.** `T.check_down` builds `Down(16, 32)` and feeds it 2 x 16 x 64 x 64; it expects 2 x 32 x 32 x 32.
    """),
    code("""
    ok_down = T.check_down(Down)
    Down = T.yours_or_reference(ok_down, Down, "Down")
    """),
    code("""
    P.show_model(Down(16, 32), (16, 64, 64), title="Down(16, 32)")
    """),
    md(r"""
    The orange pooling box comes first and halves the picture to `16x32x32`; the six layers of the
    `DoubleConv` follow and give 32 maps. Pooling has no parameters, so `Down(16, 32)` has exactly the
    parameters of `DoubleConv(16, 32)`: 14 016.
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · Step 3: `Up`, one step up the decoder

    An `Up` block goes one level up the U. It receives two tensors: `x`, coming up from the level below
    (small, many maps), and `skip`, the encoder's result of the same level (twice the height and width). It
    does three things:

    1. **Upsample** `x`: twice the height and width, half the maps. We use a **transposed convolution**,
       `nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)`: it turns every input
       pixel into a 2 x 2 patch of output pixels, weighted by weights it learns, so the height and width
       double; like a normal convolution it also sets the number of maps, here half of them (`//` divides and
       rounds down to a whole number). Now `x` has the size of `skip`.
    2. **Glue** the skip and `x` together along the channels: `torch.cat([skip, x], dim=1)`. To
       **concatenate** is to put tensors one after the other along one axis; `dim=1` is the channel axis of
       a batch (batch, channels, height, width). The maps are stacked, nothing is added up: 128 maps and 128
       maps give 256 maps. Height and width must be the same, which is why we upsample first. The skip comes
       first in the list, then `x`.
    3. **Mix** them with a `DoubleConv`: it looks at the sharp encoder maps and the upsampled decoder maps
       together, and gives `out_channels` maps.

    <details>
    <summary><b>The other way to upsample: <code>nn.Upsample</code> and a convolution</b> (optional, click to open)</summary>

    `nn.Upsample(scale_factor=2, mode="bilinear")` doubles the height and width without any weights: every new
    pixel is a blend of its neighbours. It cannot change the number of maps, so a convolution follows it.
    These lines could replace `self.up`:

    ```python
    self.up = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear"),               # twice the height and width, same maps
        nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),    # half the maps
    )
    ```

    The original U-Net uses the transposed convolution, which learns how to upsample; many U-Nets in use today
    take the `nn.Upsample` way, which is cheaper and gives smoother maps. Both give the same shapes, so the
    rest of the block stays the same.
    </details>

    **The channel arithmetic**, worked out for `up1` of the U-Net, `Up(256, 128, 128)`:

    | step | line of `forward` | shape |
    |---|---|---|
    | `x` arrives from below (it is `x5`) | | 256 x 4 x 4 |
    | upsample: `ConvTranspose2d(256, 128, kernel_size=2, stride=2)` | `x = self.up(x)` | 128 x 8 x 8 |
    | `skip` arrives from the encoder (it is `x4`) | | 128 x 8 x 8 |
    | glue, the skip first | `x = torch.cat([skip, x], dim=1)` | 128 + 128 = 256 x 8 x 8 |
    | mix: `DoubleConv(256, 128)` | `x = self.conv(x)` | 128 x 8 x 8 |

    So the `DoubleConv` receives `in_channels // 2 + skip_channels` maps: the upsampled maps plus the skip's.

    **The table.** `Up(in_channels, skip_channels, out_channels)`:

    | name | layer | in maps | out maps | kernel | stride |
    |---|---|---|---|---|---|
    | `up` | `nn.ConvTranspose2d` | in_channels | in_channels // 2 | 2 | 2 |
    | (in `forward`) | `torch.cat([skip, x], dim=1)` | skip_channels + in_channels // 2 | the same | | |
    | `conv` | `DoubleConv` (your step 1) | in_channels // 2 + skip_channels | out_channels | 3 (inside) | 1 (inside) |
    """),
    api("nn.ConvTranspose2d", "torch.cat", "nn.Upsample", title="Functions you will need"),
    exercise(
        stub="""
        class Up(nn.Module):
            def __init__(self, in_channels, skip_channels, out_channels):
                raise NotImplementedError   # delete this line once every ____ is filled in
                super().__init__()
                self.up = ____         # YOUR CODE: nn.ConvTranspose2d from in_channels to in_channels // 2 maps, kernel_size=2, stride=2
                self.conv = ____       # YOUR CODE: DoubleConv from in_channels // 2 + skip_channels to out_channels maps

            def forward(self, x, skip):
                x = ____               # YOUR CODE: self.up applied to x: twice the height and width, half the maps
                x = ____               # YOUR CODE: torch.cat of the list [skip, x] along dim=1: the skip's maps first, then x's
                x = ____               # YOUR CODE: self.conv applied to x
                return x
        """,
        solution="""
        class Up(nn.Module):
            def __init__(self, in_channels, skip_channels, out_channels):
                super().__init__()
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)   # half the maps, twice the size
                self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)                # DoubleConv from in_channels // 2 + skip_channels maps

            def forward(self, x, skip):
                x = self.up(x)                     # self.up applied to x: twice the height and width, half the maps
                x = torch.cat([skip, x], dim=1)    # torch.cat of the list [skip, x] along dim=1: the skip's maps first, then x's
                x = self.conv(x)                   # self.conv applied to x
                return x
        """),
    md(r"""
    **Test it.** `T.check_up` builds `Up(64, 32, 32)` and feeds it `x` of 2 x 64 x 16 x 16 and a skip of
    2 x 32 x 32 x 32. Besides the layers and the shapes it checks the gluing: the order (skip first), and that
    the skip really is used: with `x` set to zeros a different skip must still change the output. It also
    builds `Up(64, 16, 24)`, where the skip has fewer maps than the upsampled `x`, to check the rule
    `in_channels // 2 + skip_channels`.
    """),
    code("""
    ok_up = T.check_up(Up)
    Up = T.yours_or_reference(ok_up, Up, "Up")
    """),
    md(r"""
    **Look at it: the smallest U-Net.** An `Up` block needs a skip, so we draw it inside the smallest possible
    U-Net: one level down and one level up, built from your three blocks. The class is given; read its
    `forward`: it keeps the result of `inc` in `x1` and hands it to `up1` as the skip, after `down1`. Your full
    U-Net in step 4 repeats this pattern on four levels.
    """),
    code("""
    class OneLevelUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inc = DoubleConv(3, 16)                    # 3 x 64 x 64  -> 16 x 64 x 64
            self.down1 = Down(16, 32)                       # 16 x 64 x 64 -> 32 x 32 x 32
            self.up1 = Up(32, 16, 16)                       # 32 x 32 x 32 up to 16 x 64 x 64, + 16 skip maps -> 16 x 64 x 64
            self.outc = nn.Conv2d(16, 4, kernel_size=1)     # 16 x 64 x 64 -> 4 x 64 x 64: 4 scores per pixel

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x = self.up1(x2, x1)                            # x1 is the skip: it jumps over down1
            x = self.outc(x)
            return x

    torch.manual_seed(0)
    one_level = OneLevelUNet()
    P.show_model(one_level, (3, 64, 64), depth=1, title="the smallest U-Net: one level down, one level up")
    """),
    md(r"""
    The red arc runs from `inc` over `down1` straight into `up1`: that is the skip connection. `up1` brings the
    32 x 32 picture back to 64 x 64 and glues on the 16 maps of `x1`. The last box is `outc`, a **1 x 1
    convolution**: a convolution whose kernel covers a single pixel, so it mixes the 16 maps of each pixel
    into 4 scores for that pixel without looking at its neighbours. The output has the input's height and
    width: 4 x 64 x 64.
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · Step 4: the whole U-Net

    The U-Net is ten blocks: `inc`, four `Down`, four `Up`, and `outc`. Its settings are `in_channels` (3 for
    colour pictures), `n_classes` (4 for our shapes) and `base`, the number of maps of the first block; every
    level down doubles it.

    **Step 4a, the table of the blocks** (`__init__`). The last column is the shape of the block's output for
    `base = 16` and a 3 x 64 x 64 picture, with the variable it will be kept in.

    | name | block | in maps | skip maps | out maps | output shape |
    |---|---|---|---|---|---|
    | `inc` | `DoubleConv` | in_channels | | base | 16 x 64 x 64: `x1` |
    | `down1` | `Down` | base | | base * 2 | 32 x 32 x 32: `x2` |
    | `down2` | `Down` | base * 2 | | base * 4 | 64 x 16 x 16: `x3` |
    | `down3` | `Down` | base * 4 | | base * 8 | 128 x 8 x 8: `x4` |
    | `down4` | `Down` | base * 8 | | base * 16 | 256 x 4 x 4: `x5`, the bottleneck |
    | `up1` | `Up` | base * 16 | base * 8 (`x4`) | base * 8 | 128 x 8 x 8 |
    | `up2` | `Up` | base * 8 | base * 4 (`x3`) | base * 4 | 64 x 16 x 16 |
    | `up3` | `Up` | base * 4 | base * 2 (`x2`) | base * 2 | 32 x 32 x 32 |
    | `up4` | `Up` | base * 2 | base (`x1`) | base | 16 x 64 x 64 |
    | `outc` | `nn.Conv2d`, kernel_size=1 | base | | n_classes | 4 x 64 x 64 |

    The arguments of a block are its numbers in the order of the table: `Down(in, out)`, `Up(in, skip, out)`.
    The first `Down` and the first `Up` are written for you as the pattern.

    **Step 4b, the wiring** (`forward`). Down the encoder, each block works on the result of the one before,
    and every result gets its own variable, `x1` to `x5`, because the decoder needs them again. Up the
    decoder, each `Up` takes two arguments: `x`, the result from below, and the skip of the same size: `up1`
    takes `x4`, `up2` takes `x3`, `up3` takes `x2`, `up4` takes `x1`. Last comes `outc`. The comments show each
    result's shape for `base = 16` and a 64 x 64 picture.
    """),
    exercise(
        stub="""
        class UNet(nn.Module):
            def __init__(self, in_channels, n_classes, base):
                raise NotImplementedError   # delete this line once every ____ is filled in
                super().__init__()
                # step 4a: the blocks, one per row of the table
                self.inc = DoubleConv(in_channels, base)          # given
                self.down1 = Down(base, base * 2)                 # given: the pattern of the encoder
                self.down2 = ____      # YOUR CODE: Down from base * 2 to base * 4 maps
                self.down3 = ____      # YOUR CODE: Down from base * 4 to base * 8 maps
                self.down4 = ____      # YOUR CODE: Down from base * 8 to base * 16 maps (the bottleneck)
                self.up1 = Up(base * 16, base * 8, base * 8)      # given: the pattern of the decoder
                self.up2 = ____        # YOUR CODE: Up with in base * 8, skip base * 4, out base * 4 maps
                self.up3 = ____        # YOUR CODE: Up with in base * 4, skip base * 2, out base * 2 maps
                self.up4 = ____        # YOUR CODE: Up with in base * 2, skip base, out base maps
                self.outc = ____       # YOUR CODE: nn.Conv2d from base to n_classes maps, kernel_size=1

            def forward(self, x):
                # step 4b, the encoder: every result in its own variable
                x1 = self.inc(x)       # given: 16 x 64 x 64
                x2 = self.down1(x1)    # given: 32 x 32 x 32
                x3 = ____              # YOUR CODE: self.down2 applied to x2 -> 64 x 16 x 16
                x4 = ____              # YOUR CODE: self.down3 applied to x3 -> 128 x 8 x 8
                x5 = ____              # YOUR CODE: self.down4 applied to x4 -> 256 x 4 x 4
                # the decoder: the result from below and the skip of the same size
                x = self.up1(x5, x4)   # given: 128 x 8 x 8
                x = ____               # YOUR CODE: self.up2 applied to x, with the skip x3 -> 64 x 16 x 16
                x = ____               # YOUR CODE: self.up3 applied to x, with the skip x2 -> 32 x 32 x 32
                x = ____               # YOUR CODE: self.up4 applied to x, with the skip x1 -> 16 x 64 x 64
                x = ____               # YOUR CODE: self.outc applied to x -> 4 x 64 x 64, n_classes scores per pixel
                return x
        """,
        solution="""
        class UNet(nn.Module):
            def __init__(self, in_channels, n_classes, base):
                super().__init__()
                # step 4a: the blocks, one per row of the table
                self.inc = DoubleConv(in_channels, base)          # given
                self.down1 = Down(base, base * 2)                 # given: the pattern of the encoder
                self.down2 = Down(base * 2, base * 4)             # Down from base * 2 to base * 4 maps
                self.down3 = Down(base * 4, base * 8)             # Down from base * 4 to base * 8 maps
                self.down4 = Down(base * 8, base * 16)            # Down from base * 8 to base * 16 maps (the bottleneck)
                self.up1 = Up(base * 16, base * 8, base * 8)      # given: the pattern of the decoder
                self.up2 = Up(base * 8, base * 4, base * 4)       # Up with in base * 8, skip base * 4, out base * 4 maps
                self.up3 = Up(base * 4, base * 2, base * 2)       # Up with in base * 4, skip base * 2, out base * 2 maps
                self.up4 = Up(base * 2, base, base)               # Up with in base * 2, skip base, out base maps
                self.outc = nn.Conv2d(base, n_classes, kernel_size=1)   # nn.Conv2d from base to n_classes maps, kernel_size=1

            def forward(self, x):
                # step 4b, the encoder: every result in its own variable
                x1 = self.inc(x)       # given: 16 x 64 x 64
                x2 = self.down1(x1)    # given: 32 x 32 x 32
                x3 = self.down2(x2)    # self.down2 applied to x2 -> 64 x 16 x 16
                x4 = self.down3(x3)    # self.down3 applied to x3 -> 128 x 8 x 8
                x5 = self.down4(x4)    # self.down4 applied to x4 -> 256 x 4 x 4
                # the decoder: the result from below and the skip of the same size
                x = self.up1(x5, x4)   # given: 128 x 8 x 8
                x = self.up2(x, x3)    # self.up2 applied to x, with the skip x3 -> 64 x 16 x 16
                x = self.up3(x, x2)    # self.up3 applied to x, with the skip x2 -> 32 x 32 x 32
                x = self.up4(x, x1)    # self.up4 applied to x, with the skip x1 -> 16 x 64 x 64
                x = self.outc(x)       # self.outc applied to x -> 4 x 64 x 64, n_classes scores per pixel
                return x
        """),
    md(r"""
    **Test it.** `T.check_unet` builds `UNet(in_channels=3, n_classes=4, base=16)`. First the blocks, one line
    per row of the table, with the parameters of each block next to the reference's; then a second network,
    `UNet(in_channels=1, n_classes=3, base=8)`, to catch fixed numbers; then the total. Then the wiring: it
    runs 2 random pictures of 3 x 64 x 64 through the network and checks the output (4 maps, 64 x 64), the order
    of the blocks, and that every block receives the right tensors, in particular every `Up` the right skip.
    """),
    code("""
    ok_unet = T.check_unet(UNet)
    UNet = T.yours_or_reference(ok_unet, UNet, "UNet")
    """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · Look at your U-Net

    Your network, drawn from your code, one box per block. It should look exactly like the target in section 1.
    """),
    code("""
    torch.manual_seed(0)
    unet = UNet(in_channels=3, n_classes=4, base=16)
    P.show_model(unet, (3, 64, 64), depth=1, title="your U-Net (base 16)")
    """),
    md(r"""
    `depth=2` opens every block one level: you see the pooling layer and the `DoubleConv` inside each `Down`,
    and the transposed convolution, the concatenation (red box, "Concat") and the `DoubleConv` inside each
    `Up`. The red arcs now end at the "Concat" boxes, where the skips are glued on.
    """),
    code("""
    P.show_model(unet, (3, 64, 64), depth=2, title="your U-Net, every block opened one level")
    """),
    md(r"""
    `P.summary(model, input_shape, depth=1)` gives the same as a table: one row per block with its output shape,
    parameters and **multiply-adds** (the multiplications and additions it needs for one picture: a measure
    of its work).
    """),
    code("""
    P.summary(unet, (3, 64, 64), depth=1)
    """),
    md(r"""
    **Reading the table.** Down the encoder (`inc` to `down4`) the height and width halve at each row
    (64, 32, 16, 8, 4) while the maps double (16, 32, 64, 128, 256). Up the decoder (`up1` to `up4`) it is the
    reverse. The parameters pile up at the bottom (`down4` and `up1`), but the work does not: every `Down` block
    does the same 14 155 776 multiply-adds, every `Up` block 30 408 704. One level down, a convolution slides over
    a quarter of the pixels but has four times the weights (twice the maps in, twice out), and the two cancel.
    The `Up` blocks do about twice the work of the `Down` blocks because their `DoubleConv` reads the glued
    maps: twice as many.

    `P.cost` measures what the network costs on this computer: memory, the time to predict one picture, and
    one **training step** (a batch of 32 pictures: predict, work out how to change every weight, change them).
    `epoch_images=1200` makes it estimate one **epoch** (a pass over all 1 200 training pictures).
    """),
    code("""
    P.cost(unet, (3, 64, 64), name="small U-Net (base 16)", epoch_images=1200)
    """),
    md(r"""
    ### The U-Net of the paper

    The original U-Net starts with `base = 64` maps instead of 16. Your class builds it with one number changed:
    `UNet(in_channels=3, n_classes=4, base=64)`. Every convolution then has 4 times as many maps in and 4 times
    as many out, so about 16 times as many weights. (We time it with batches of 8 instead of 32 so that the
    measurement stays short, a few seconds.)
    """),
    code("""
    torch.manual_seed(0)
    unet_paper = UNet(in_channels=3, n_classes=4, base=64)
    P.cost(unet_paper, (3, 64, 64), name="paper-size U-Net (base 64)", batch_size=8, epoch_images=1200)
    """),
    md(r"""
    31 043 716 parameters, the "about 31 million" of the U-Net paper, against 1 944 100 for ours: 16 times as
    many, 118 MB instead of 7 MB. It does 3.01 billion multiply-adds for one picture instead of 0.19 billion.
    On the course computer it took about 35 ms to predict one picture alone (ours: about 6 ms), and one epoch
    over our 1 200 pictures would take about 100 seconds of training steps instead of about 10. (Times change
    from run to run, and grow a lot while the computer is busy with other work.)

    `P.compare` puts the two side by side: one table and one chart.
    """),
    code("""
    P.compare({"small U-Net (base 16)": unet, "paper-size U-Net (base 64)": unet_paper}, (3, 64, 64),
              batch_size=8, epoch_images=1200)
    """),
    md(r"""
    **Why we train the small one.** Every number in the table is about 10 to 16 times bigger for the paper-size
    network: three epochs would take about 5 minutes here instead of half a minute, and much longer on a slower
    laptop. During training PyTorch also keeps a gradient and Adam's two running averages for every weight, so
    its 118 MB of weights become about 470 MB before any picture is stored. And it would buy nothing here: the
    small network already gets every pixel of our shapes right (section 7). The paper's network was made for
    microscope images of 572 x 572 pixels and trained on a graphics card. Choosing a network's size to fit the
    task and the computer is part of the job; your class builds either one with a single number.
    """),

    # ------------------------------------------------------------------ 7
    md(r"""
    ## 7 · Train it (predefined)

    Training works as in P1: `P.train` shows the network the training pictures batch by batch and changes the
    weights a little after each batch. Two things are new for segmentation, and `P.train` handles both:

    * the **loss** (how wrong the network is) is the classification loss of P1, the **cross-entropy**, computed
      for every pixel and averaged over all pixels: each pixel is a small classification problem;
    * the **accuracy** is the **pixel accuracy**: the percentage of all pixels whose class is right.

    The settings are fixed here: 3 epochs, learning rate 0.001. You change them in section 8.

    **Expected time: about 30 seconds** on the course computer (up to 2 minutes on a slower laptop, or when the
    computer is busy). The curves update after every epoch.
    """),
    code("""
    torch.manual_seed(0)                    # the same starting weights every time
    unet = UNet(in_channels=3, n_classes=4, base=16)
    history = P.train(unet, train_loader, val_loader, epochs=3, lr=0.001, name="U-Net (base 16)")
    """),
    md(r"""
    **Reading the curves.** After the first epoch the validation pixel accuracy is already 99.9%, after the
    second 100.0%: the U-Net gets practically every pixel of the 200 validation pictures right. The validation
    loss keeps falling (0.335, 0.129, 0.061): the network becomes more and more sure of its answers. The
    training accuracy of the first epoch is lower (96.0%) because it is averaged over the epoch, while the
    network was still learning. Each epoch took about 10 seconds here.

    The pixel accuracy can mislead: the background covers 84.7% of the validation pixels, so a network that
    paints every pixel as background would already score 84.7%. The usual score for segmentation is the **IoU** (intersection
    over union) of each class: the pixels that are this class in both the prediction and the true mask,
    divided by the pixels that are this class in at least one of them, so 1 is a perfect match and 0 no
    overlap at all. The **mean IoU** averages it over the classes, so a small triangle counts as much as the
    background, and painting everything as background scores badly.

    `R.mean_iou(model, loader)` computes it over all validation pictures.
    """),
    code("""
    scores = R.mean_iou(unet, val_loader, name="U-Net (base 16)")
    """),
    md(r"""
    `R.show_predictions(model, loader)` shows the first six validation pictures: the picture, the true mask,
    the network's mask, and its wrong pixels in dark grey (with their number).
    """),
    code("""
    R.show_predictions(unet, val_loader, title="U-Net (base 16): predictions on validation pictures")
    """),
    md(r"""
    Not one wrong pixel in these six pictures: the outlines of the squares, circles and triangles are exact,
    down to the tip of the small triangle in the first picture. The mean IoU of 1.000 says the same for all 200
    validation pictures.
    """),

    # ------------------------------------------------------------------ 8
    md(r"""
    ## 8 · Try it: what do the skip connections buy?

    The run cell below trains a U-Net with settings you choose in the **Try it** cell. It uses the course's
    reference U-Net, `R.UNet`, which is built exactly like yours but has two more settings: `depth`, the number
    of `Down` (and `Up`) blocks, and `use_skips`, which can switch the skip connections off (then every `Up`
    block ignores its skip and its `DoubleConv` receives only the upsampled maps).

    **The prepared experiment** is already set: the same network, the same data and the same training, with
    the skips switched off (`use_skips = False`). **Predict first:** the network still has its whole encoder and
    decoder; only the copies across the U are gone. What will the masks look like?

    `P.remember(name, model, history)` records each version for `P.leaderboard()`, as in P1. First the run
    from section 7:
    """),
    code("""
    P.remember("U-Net base 16, with skips", unet, history)
    """),
    md(r"""
    **Try it.** Change the numbers, then run this cell and the next one.
    """),
    code("""
    # ---------------- Try it: settings of the U-Net ----------------
    base = 16                     # maps of the first block; every level down doubles them
    depth = 4                     # Down blocks (and as many Up blocks); 64 must be divisible by 2 ** depth
    use_skips = False             # False: the decoder gets no skip connections
    # ---------------- run settings ----------------
    dataset = "shapes"            # or "pets": photos of cats and dogs (Oxford-IIIT Pet), 3 classes
    epochs = 3                    # passes over the training pictures
    learning_rate = 0.001         # size of each weight change
    n_train = 1200                # training pictures (shapes: any number; pets: at most 3680)
    version_name = dataset + ", base " + str(base) + ", depth " + str(depth) + ", skips " + str(use_skips)
    """),
    md(r"""
    **Expected time: about 30 seconds** on the course computer with the settings above (up to 2 minutes on a
    slower laptop). The first run with `dataset = "pets"` also reads the 1 400 photos: about a minute, once (they
    are then kept in the `data/` folder).
    """),
    code("""
    train_loader, val_loader, classes = R.get_data(dataset, n_train=n_train, n_val=200, batch_size=16)
    torch.manual_seed(0)
    unet_try = R.UNet(in_channels=3, n_classes=len(classes), base=base, depth=depth, use_skips=use_skips)
    P.show_model(unet_try, (3, 64, 64), depth=1, title=version_name)
    history_try = P.train(unet_try, train_loader, val_loader, epochs=epochs, lr=learning_rate, name=version_name)
    scores_try = R.mean_iou(unet_try, val_loader, name=version_name)
    R.show_predictions(unet_try, val_loader, title=version_name)
    P.remember(version_name, unet_try, history_try)
    """),
    md(r"""
    <details><summary><b>What we saw with the skips switched off</b> (open after your run)</summary>

    The diagram has no red arcs: nothing crosses the U. The network has 1 748 260 parameters instead of
    1 944 100 (the `DoubleConv` of each `Up` block receives fewer maps); the encoder and the decoder are
    otherwise the same, and so are the data and the training.

    The validation pixel accuracy reaches 96.0% at best (epoch 2) and ends at 93.8%, instead of 100.0%. The
    mean IoU drops from 1.000 to 0.738 (background 0.933, circle 0.688, square 0.703, triangle 0.628). The
    masks show why: the network still finds every shape, in the right place and mostly with the right class,
    but the outlines are soft blobs. Squares lose their corners, triangles their tips, and some shapes get
    patches of a wrong class. The wrong pixels lie almost all along the outlines.

    Without skips every detail has to come up through the 4 x 4 bottleneck, where one pixel stands for a
    16 x 16 square of the picture: the decoder knows *what* is roughly *where*, not where the edge runs. The
    skip connections hand it the sharp maps of every level directly. That is what they buy.
    </details>

    `P.plot_history` draws the run of section 7 and your **Try it** run on top of each other (solid lines:
    validation, dashed: training), and
    `P.leaderboard()` lists every version so far, best first. Its accuracy column is the pixel accuracy; the
    mean IoU of each run is printed under its training cell.
    """),
    code("""
    P.plot_history({"section 7 (with skips)": history, "Try it": history_try})
    P.leaderboard()
    """),
    md(r"""
    ### More experiments

    For each: change the setting in the **Try it** cell, predict, run the two cells, compare, then open the
    fold-out. Put `use_skips = True` back for experiments 2 to 4.

    **2. Fewer maps.** `use_skips = True`, `base = 8`.
    <details><summary>What we saw</summary>

    487 316 parameters, a quarter of base 16 (half the maps everywhere, so a quarter of the weights), and a
    quarter of the multiply-adds (47.9 million). Pixel accuracy 97.0%: almost as good, it seems. The mean IoU
    says otherwise: 0.734, and the per-class line shows why: triangle 0.000. After 3 epochs this smaller
    network has not learned triangles at all and paints them as background; the wrong pixels are whole
    triangles. Triangles cover 3.0% of the pixels, so the pixel accuracy hardly notices; the mean IoU, which
    counts every class the same, does. Training twice as long (`epochs = 6`) did not fix it here: still
    triangle 0.000, mean IoU 0.739. The small network got stuck on an answer that ignores the rarest class,
    while the network with 16 maps found the triangles in its first epoch. Nothing in the pixel accuracy
    warns you: only the per-class IoU does.
    </details>

    **3. A shallower U.** `use_skips = True`, `depth = 2` (only two `Down` and two `Up` blocks).
    <details><summary>What we saw</summary>

    117 732 parameters, 16 times fewer, and 100.6 million multiply-adds instead of 189.7 million: and the mean
    IoU is still 1.000. Its bottleneck is 64 x 16 x 16, so every pixel there sees only a small part of the
    picture, and for these shapes that is enough: a pixel's class can be read from its colour and its near
    neighbours. A deep encoder pays off when the class of a pixel depends on a large part of the picture: a
    patch of fur only becomes "pet" when the network sees the animal around it.
    </details>

    **4. Real photos.** `use_skips = True`, `dataset = "pets"`: photos of cats and dogs reduced to 64 x 64
    pixels, with 3 classes: 0 background, 1 pet, 2 border (a thin band along the pet's outline, where the
    people who drew the masks were not sure).
    <details><summary>What we saw</summary>

    Reading the 1 400 photos took about a minute the first time, the training about 2 minutes. After 3 epochs:
    pixel accuracy 79.7%, mean IoU 0.552 (background 0.777, pet 0.627, border 0.251). The predicted masks put
    the pets in the right places with rough outlines, and the thin border class is the hardest. Real photos
    are much harder than our shapes: colour gives nothing away, and the network has to learn what a cat or a
    dog looks like from 1 200 small photos, from scratch. The validation loss was still falling at epoch 3
    (0.635, 0.606, 0.513), so more epochs (and more photos: `n_train = 3680` uses all of them) should help.
    </details>
    """),

    # ------------------------------------------------------------------ 9
    md(r"""
    ## 9 · What to remember

    * **Segmentation** gives a class to every pixel; the network's output has one map of scores per class, with
      the height and width of the picture.
    * A **U-Net** is an encoder that shrinks the picture and learns *what* is in it, a decoder that grows it back
      to restore *where*, and **skip connections** that copy each encoder level across to the decoder level of
      the same size.
    * It is built from three small blocks: `DoubleConv` (two 3 x 3 convolutions with batch normalisation and
      ReLU), `Down` (max-pooling, then `DoubleConv`) and `Up` (transposed convolution, `torch.cat([skip, x],
      dim=1)`, `DoubleConv`); a 1 x 1 convolution gives the scores.
    * Down the U the size halves and the maps double; up the U the reverse. An `Up` block's `DoubleConv`
      receives `in_channels // 2 + skip_channels` maps.
    * Measure with the mean IoU: without the skips it fell from 1.000 to 0.738 because the outlines went soft,
      and a smaller network scored 97.0% of the pixels right while missing every triangle.

    ### Words from this notebook

    | word | meaning |
    |---|---|
    | 1 x 1 convolution | a convolution whose kernel covers one pixel: it mixes the maps of each pixel, here into the class scores |
    | batch normalisation | a layer that rescales each map to average 0 and spread 1 over the batch, then applies a learned scale and shift |
    | bottleneck | the bottom of the U: the smallest picture with the most maps (here `down4`) |
    | channel (map) | one of the grids of numbers a picture or a layer output consists of |
    | concatenate | put tensors one after the other along one axis; `torch.cat(..., dim=1)` stacks maps |
    | cross-entropy | the classification loss; for segmentation it is averaged over all pixels |
    | decoder | the half of the U that grows the picture back: restores *where* |
    | `DoubleConv` | our basic block: conv 3 x 3, batch normalisation, ReLU, twice |
    | encoder | the half of the U that shrinks the picture: learns *what* |
    | epoch | one pass over all training pictures |
    | IoU (intersection over union) | for one class: pixels of that class in both prediction and truth, divided by those in at least one |
    | mask | a grid with one class number per pixel |
    | max-pooling | keeping the largest value of every 2 x 2 square: halves height and width |
    | mean IoU | the IoU averaged over the classes |
    | multiply-adds | the multiplications and additions a network needs for one picture: a measure of its work |
    | pixel accuracy | the percentage of pixels whose class is right |
    | segmentation | giving every pixel of a picture a class |
    | skip connection | a copy of an encoder level's maps, glued onto the decoder level of the same size |
    | training step | one batch: predict, work out the weight changes, change the weights |
    | transposed convolution | a learned upsampling layer: with kernel 2 and stride 2 every pixel becomes a 2 x 2 patch |
    | U-Net | encoder + decoder + skip connections, drawn as a U |
    | upsample | make a picture or map larger |
    """),
]
