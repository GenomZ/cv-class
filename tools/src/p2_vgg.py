from nb import md, code, api

NAME = "P2-vgg"

CELLS = [
    md(r"""
    # P2 · VGG: a famous deep network, in full and in miniature

    **What we do today**

    1. Learn the idea behind **VGG**, a network from 2014 that is still used today: build a deep network from
       one small brick, the 3 x 3 convolution.
    2. Write out the full **VGG16**, all 16 layers, and look at it from every side: diagram, table, where its
       138 million parameters sit, what it costs on this computer, and why we cannot train it here.
    3. Shrink the same recipe into a **mini-VGG** for small colour photos (CIFAR-10).
    4. Train the mini-VGG (about 2 minutes) and look at what it learned: its answers, its mistakes, its filters
       and its feature maps.
    5. Change the mini-VGG's settings yourself and compare the versions (section 5, **Try it**).

    **How to use this notebook.** As in P1, you do not write any code. Run the cells from top to bottom with
    **Shift+Enter**. Cells marked **Try it** contain settings you may change and run again. The whole notebook
    runs in about 4 minutes here (longer on a slow laptop); every slow cell says how long it takes.

    * `P.something(...)` is a function of the course toolkit (the same as in P1); the text explains each new one
      where it is first used.
    * Closed boxes **Functions used here (click to open)** explain the PyTorch functions in the next cell.
    * **New words** are in bold where they are explained, and collected at the end.
    * Numbers quoted in the text are what the cells print. On your computer the last digit can differ, and
      times can be quite different.
    """),

    # ------------------------------------------------------------------ 0
    md(r"""
    ## 0 · Setup

    The same setup as in P1: find the course helper code, load PyTorch and the toolkit `P`, fix the random seed.
    One addition: `matplotlib.pyplot` (short `plt`) is the library the toolkit draws its figures with; we use it
    once ourselves, for a bar chart in section 2.
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
    import matplotlib.pyplot as plt
    from cvcourse import practical as P

    P.setup(seed=0)
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · The idea of VGG

    In 2014 Karen Simonyan and Andrew Zisserman, of the **Visual Geometry Group** at the University of Oxford
    (hence the name VGG), entered the **ImageNet** competition: 1.28 million training photos in 1 000 classes
    (dog breeds, birds, vehicles, food, tools, ...), each to be recognised from a 224 x 224 pixel colour photo.
    They came second, with a design so regular that it is still the textbook example of a convolutional network.

    The whole recipe fits in five lines:

    1. use **only 3 x 3 convolutions** (with padding 1, so a convolution keeps the size of the picture), each
       followed by ReLU;
    2. stack two or three of them in a row: such a group is a **block**;
    3. after each block, a 2 x 2 max pooling halves height and width;
    4. after each pooling, **double the number of feature maps**: 64 -> 128 -> 256 -> 512 (and stop at 512);
    5. at the end, flatten and three fully connected layers: 4 096 neurons, 4 096 neurons, 1 000 scores.

    **VGG16** is the version with 16 layers that have weights: 13 convolutions and 3 fully connected layers.
    (Pooling, ReLU and flatten have no weights, so they are not counted.)

    | part | layers | feature maps | picture size in -> out |
    |---|---|---|---|
    | block 1 | 2 convolutions + pooling | 64 | 224 x 224 -> 112 x 112 |
    | block 2 | 2 convolutions + pooling | 128 | 112 x 112 -> 56 x 56 |
    | block 3 | 3 convolutions + pooling | 256 | 56 x 56 -> 28 x 28 |
    | block 4 | 3 convolutions + pooling | 512 | 28 x 28 -> 14 x 14 |
    | block 5 | 3 convolutions + pooling | 512 | 14 x 14 -> 7 x 7 |
    | classifier | flatten + 3 fully connected | | 512 x 7 x 7 = 25 088 -> 4 096 -> 4 096 -> 1 000 |

    ### Why two 3 x 3 convolutions instead of one 5 x 5?

    Older networks used big filters (AlexNet, the 2012 winner, started with 11 x 11). VGG's point: a stack of
    small filters sees as much of the picture as one big filter, with fewer weights.

    The part of the input picture that one output number depends on is its **receptive field** (its "view").
    One 3 x 3 convolution has a view of 3 x 3 pixels. Put a second 3 x 3 convolution on top: each of its outputs
    reads 3 x 3 neighbouring outputs of the first one, and each of those saw 3 x 3 pixels. Shown along one row of
    pixels:

    ```
    picture:              p p p p p      5 neighbouring pixels
    first 3x3 layer:        a a a        each a reads 3 neighbouring p's
    second 3x3 layer:         b          b reads 3 neighbouring a's, so 5 p's
    ```

    So two stacked 3 x 3 convolutions have a 5 x 5 view, and three have a 7 x 7 view. The next cell builds both
    options, each taking 64 feature maps of 56 x 56 pixels and giving 64 maps out (sizes that occur inside
    VGG16), and prints their tables with `P.summary` (from P1: one row per layer with its parameters and
    multiply-adds).
    """),
    api("torch.manual_seed", "nn.Sequential", "nn.Conv2d", "nn.ReLU", brief=True),
    code("""
    torch.manual_seed(0)
    one_5x5 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=5, padding=2),     # one 5x5 filter per output map: a 5x5 view
        nn.ReLU(),
    )
    two_3x3 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, padding=1),     # a 3x3 view ...
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),     # ... and a 3x3 view of that: a 5x5 view of the input
        nn.ReLU(),
    )
    P.summary(one_5x5, (64, 56, 56))
    print()
    P.summary(two_3x3, (64, 56, 56))
    """),
    md(r"""
    **What the numbers say.** Same view of the picture, but:

    * **weights:** one 5 x 5 convolution has 64 x 64 x 25 + 64 = 102 464 parameters; two 3 x 3 convolutions have
      2 x (64 x 64 x 9 + 64) = 73 856, 28% fewer (for every pair of input map and output map, 2 x 9 = 18
      weights instead of 25);
    * **work:** 231 million multiply-adds instead of 321 million;
    * **decisions:** two ReLUs instead of one, so the stack can combine patterns in more complex ways.

    For a 7 x 7 view the saving is bigger: three 3 x 3 layers have 3 x 9 = 27 weights per pair of maps instead
    of 49 (45% fewer). That is the whole trick of VGG: small filters, many layers.
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · The full VGG16

    ### 2.1 · Written out, layer by layer

    A network this size is usually written as a **class**: a recipe from which a network is built (this is how
    the PyTorch library torchvision writes its own VGG16). Read the cell below in three parts:

    * `class VGG16(nn.Module):` says "here comes a PyTorch network called VGG16";
    * `__init__` creates the layers, grouped into six named parts: `self.block1` to `self.block5` (the
      convolutions) and `self.classifier` (the fully connected layers). Each part is an `nn.Sequential`, a
      chain of layers as in P1. The first line, `super().__init__()`, prepares PyTorch's bookkeeping and starts
      every network class;
    * `forward(self, x)` says what happens to a batch of pictures `x`: block 1, then block 2, ..., then the
      classifier. Its result is 1 000 scores per picture.

    Every one of the 16 layers with weights is written out and numbered in the comments. One new layer appears
    in the classifier:

    **Dropout** (`nn.Dropout(0.5)`) is active only during training: for every picture it sets a random half of
    the numbers passing through it to 0, a different half each time. The next layer can never rely on one
    particular neuron, which makes it harder for the network to **memorise** its training pictures (learn the
    pictures themselves instead of what their classes look like). When the network
    predicts, dropout is switched off. VGG16 puts it after the first two fully connected layers, where most of
    the parameters are.
    """),
    api("nn.Module", "nn.Dropout"),
    api("nn.MaxPool2d", "nn.Flatten", "nn.Linear", brief=True),
    code("""
    class VGG16(nn.Module):
        def __init__(self):
            super().__init__()
            # block 1: 64 feature maps, the picture stays 224 x 224, then pooling halves it
            self.block1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),       # layer 1: 3 colour channels in, 64 maps out
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),      # layer 2
                nn.ReLU(),
                nn.MaxPool2d(2),                                  # 224 x 224 -> 112 x 112
            )
            # block 2: 128 feature maps (twice as many), 112 x 112
            self.block2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),     # layer 3
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),    # layer 4
                nn.ReLU(),
                nn.MaxPool2d(2),                                  # 112 x 112 -> 56 x 56
            )
            # block 3: 256 feature maps, 56 x 56, now three convolutions
            self.block3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),    # layer 5
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),    # layer 6
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),    # layer 7
                nn.ReLU(),
                nn.MaxPool2d(2),                                  # 56 x 56 -> 28 x 28
            )
            # block 4: 512 feature maps, 28 x 28
            self.block4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),    # layer 8
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),    # layer 9
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),    # layer 10
                nn.ReLU(),
                nn.MaxPool2d(2),                                  # 28 x 28 -> 14 x 14
            )
            # block 5: still 512 feature maps (VGG stops doubling at 512), 14 x 14
            self.block5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),    # layer 11
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),    # layer 12
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),    # layer 13
                nn.ReLU(),
                nn.MaxPool2d(2),                                  # 14 x 14 -> 7 x 7
            )
            # classifier: three fully connected layers turn the 512 maps of 7 x 7 into 1000 scores
            self.classifier = nn.Sequential(
                nn.Flatten(),                                     # 512 x 7 x 7 = 25088 numbers in a row
                nn.Linear(25088, 4096),                           # layer 14
                nn.ReLU(),
                nn.Dropout(0.5),                                  # during training: drop half of the numbers
                nn.Linear(4096, 4096),                            # layer 15
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 1000),                            # layer 16: one score per ImageNet class
            )

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.classifier(x)
            return x


    torch.manual_seed(0)
    vgg16 = VGG16()           # build one network from the recipe, with random starting weights
    """),
    md(r"""
    ### 2.2 · Look at it

    `P.show_model(model, input_shape, depth=1)` draws one box per named part (a stacked box is a block of
    layers); without `depth` it opens every block and draws every layer. A 224 x 224 colour photo has the shape
    `3 x 224 x 224` (3 channels: red, green, blue).
    """),
    code("""
    P.show_model(vgg16, (3, 224, 224), depth=1, title="VGG16: one box per block")
    """),
    md(r"""
    The picture shrinks block by block (224, 112, 56, 28, 14, 7) while the number of maps grows (64, 128, 256,
    512, 512). Look at the parameter counts under the boxes: the classifier alone has far more than all five
    blocks together. Now every layer (the diagram continues over several rows; follow the grey line):
    """),
    code("""
    P.show_model(vgg16, (3, 224, 224), title="VGG16: every layer")
    """),
    md(r"""
    ### 2.3 · The table: shapes through the network

    `P.summary` lists every layer with the shape of what comes out of it for one 224 x 224 photo. The layer
    names say where a layer sits: `block3.4` is item 4 (counting from 0) of block 3, its third convolution.
    """),
    code("""
    P.summary(vgg16, (3, 224, 224))
    """),
    md(r"""
    **Reading the table.**

    * Every convolution keeps the size (padding 1), every pooling halves it: 224 -> 112 -> 56 -> 28 -> 14 -> 7.
    * The first convolution has only 3 x 64 x 9 + 64 = 1 792 parameters, but needs 87 million multiply-adds:
      its 64 filters are applied at every one of the 224 x 224 = 50 176 positions.
    * The first fully connected layer, `classifier.1`, has 25 088 x 4 096 + 4 096 = 102 764 544 parameters,
      about three quarters of all, but needs only 103 million multiply-adds: each weight is used once per
      picture.
    * In total: 138 357 544 parameters (528 MB of numbers) and 15.47 **billion** multiply-adds for one photo.

    ### 2.4 · Where do the parameters sit, and where is the work?

    The next cell takes the same table as a list of rows (`verbose=False`: do not print it again) and draws, for
    the 16 layers with weights, their parameters (left) and their multiply-adds (right), coloured as in the
    diagram.
    """),
    code("""
    rows = P.summary(vgg16, (3, 224, 224), verbose=False)    # the table above, as a list of rows
    names = []
    params = []
    work = []
    colours = []
    for row in rows:
        if row["params"] > 0:                                 # only the 16 layers that have weights
            names.append(row["layer"])
            params.append(row["params"] / 1e6)                # in millions
            work.append(row["mult_adds"] / 1e9)               # in billions
            colours.append(P.KIND_COLORS[row["kind"]])        # blue = convolution, green = linear, as in the diagram

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True, layout="constrained")
    ax_left.barh(names, params, color=colours, height=0.7)
    ax_left.set_title("parameters of each layer (millions)", loc="left")
    ax_right.barh(names, work, color=colours, height=0.7)
    ax_right.set_title("multiply-adds of each layer for one photo (billions)", loc="left")
    for k in range(len(names)):                               # write the value at the end of every bar
        ax_left.text(params[k], k, f"  {params[k]:.2f}", va="center", fontsize=9)
        ax_right.text(work[k], k, f"  {work[k]:.2f}", va="center", fontsize=9)
    ax_left.set_xlim(0, 120)
    ax_right.set_xlim(0, 2.3)
    ax_left.invert_yaxis()                                    # layer 1 at the top
    fig.suptitle("VGG16: blue = convolution layers, green = fully connected layers")
    plt.show()
    """),
    md(r"""
    The two charts are almost mirror images. Per part of the network (`depth=1` gives one row per block):
    """),
    code("""
    block_rows = P.summary(vgg16, (3, 224, 224), depth=1, verbose=False)   # one row per block
    total_params = 0
    total_work = 0
    for row in block_rows:
        total_params = total_params + row["params"]
        total_work = total_work + row["mult_adds"]

    print("part          parameters   share    multiply-adds   share")
    for row in block_rows:
        share_params = 100 * row["params"] / total_params
        share_work = 100 * row["mult_adds"] / total_work
        print(f"{row['layer']:<11} {row['params']:>12,}  {share_params:5.1f}%  {row['mult_adds']:>15,}  {share_work:5.1f}%")
    """),
    md(r"""
    **What the numbers say.** The three fully connected layers hold 89.4% of the parameters (the first one alone
    74%), but do only 0.8% of the work. The 13 convolutions are the opposite: 10.6% of the parameters, 99.2% of
    the multiply-adds. Most of the memory goes into the classifier, most of the time into the convolutions.
    Later networks (ResNet, 2015) replaced the huge fully connected layers with a single small one and
    became far smaller.

    ### 2.5 · Memory for one photo

    The weights are not the only memory a network needs. During training, every layer's output must be kept
    until the weights are updated (the corrections are computed from them, backwards from the last layer).
    The next cell adds up the sizes of all layer outputs in the table, for one photo: each output shape, for
    example 64 x 224 x 224, is that many numbers, and a number takes 4 bytes.
    """),
    code("""
    numbers = 0
    for row in rows:                                   # every layer of the table above
        count = 1
        for size in row["output_shape"]:               # e.g. (64, 224, 224) -> 64 x 224 x 224 numbers
            count = count * size
        numbers = numbers + count
    megabytes = numbers * 4 / 2**20                    # 4 bytes per number; 2**20 bytes = 1 MB
    print(f"layer outputs for one 224 x 224 photo: {numbers:,} numbers = {megabytes:.0f} MB")
    print(f"for a batch of 32 photos: {32 * megabytes / 1024:.1f} GB, on top of the 528 MB of weights")
    """),
    md(r"""
    About 109 MB per photo, 190 times the photo itself (3 x 224 x 224 = 150 528 numbers): a single batch of 32
    photos needs about 3.4 GB while training. Graphics cards used for training have a fixed amount of memory (8 to 80 GB),
    and this is often what limits the batch size.

    ### 2.6 · What it costs on this computer

    `P.cost` (from P1) measures the network on this machine. A training step of VGG16 is slow, so we measure it
    on a batch of 4 photos (`batch_size=4`). **Expected time: about 15 seconds** (up to a minute on a slow
    laptop).
    """),
    code("""
    vgg16_cost = P.cost(vgg16, (3, 224, 224), name="VGG16", batch_size=4)
    """),
    md(r"""
    Between a tenth and a fifth of a second to recognise one photo (110 to 190 milliseconds in our runs,
    depending on what else the computer was doing): fast enough for an app. But one training step for 4 photos
    takes about 2 seconds, and an epoch of only 10 000 photos well over an hour.

    ### 2.7 · Is it really VGG16?

    torchvision, PyTorch's computer-vision library, has its own VGG16. `torchvision.models.vgg16(weights=None)`
    builds it with random weights (it downloads nothing; the trained ImageNet weights would be a 528 MB
    download, which we do not need today). The next cell checks that ours matches it: the same number of
    parameters, every weight tensor with the same shape in the same order, and the same output shape.
    `zip(a, b)` walks through two lists side by side.
    """),
    api("torchvision.models.vgg16"),
    api("Module.parameters", "Tensor.numel", brief=True),
    code("""
    import torchvision

    reference = torchvision.models.vgg16(weights=None)       # PyTorch's own VGG16, random weights

    ours = 0
    for weights in vgg16.parameters():
        ours = ours + weights.numel()
    theirs = 0
    for weights in reference.parameters():
        theirs = theirs + weights.numel()

    same_shapes = True
    for mine, other in zip(vgg16.parameters(), reference.parameters()):
        if mine.shape != other.shape:
            same_shapes = False

    our_output = P.output_shape(vgg16, (3, 224, 224))          # what comes out for one photo
    their_output = P.output_shape(reference, (3, 224, 224))

    print(f"parameters:   ours {ours:,}   torchvision {theirs:,}")
    print(f"every weight tensor has the same shape, in the same order: {same_shapes}")
    print(f"output shape: ours {our_output}   torchvision {their_output}")
    del reference                                             # free its 528 MB again
    """),
    md(r"""
    Identical. torchvision's version has one extra layer without weights between the convolutions and the
    classifier (`AdaptiveAvgPool2d`), which for a 224 x 224 photo changes nothing (the maps are already 7 x 7);
    it lets the network accept photos of other sizes.

    ### 2.8 · Why we do not train it here

    ImageNet has 1 281 167 training photos. At the speed measured above:
    """),
    code("""
    imagenet_photos = 1281167
    seconds_per_photo = vgg16_cost["train_step_ms"] / 1000 / vgg16_cost["batch_size"]   # training step time per photo
    hours_per_epoch = seconds_per_photo * imagenet_photos / 3600     # 3600 seconds in an hour
    days_per_epoch = hours_per_epoch / 24
    days_74_epochs = 74 * days_per_epoch

    print(f"training: {seconds_per_photo:.2f} s per photo on this computer")
    print(f"one epoch of ImageNet: {hours_per_epoch:.0f} hours = {days_per_epoch:.1f} days")
    print(f"74 epochs, as in the VGG paper: {days_74_epochs:.0f} days = {days_74_epochs / 365:.1f} years")
    """),
    md(r"""
    About a week or more for a single pass over ImageNet (7 to 10 days in our runs), and well over a year for
    the 74 passes the authors needed. They trained on four graphics cards, and even so each
    network took two to three weeks. On a laptop processor it is out of the question, and the ImageNet photos
    (about 140 GB) would hardly fit on its disk.

    That is why people almost never train VGG16 themselves: they download weights that someone has already
    trained on ImageNet and reuse them. Today we keep the recipe and make it small enough for a laptop.
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · Mini-VGG: the same recipe for small photos

    ### 3.1 · The data: CIFAR-10

    **CIFAR-10** is a collection of 60 000 small colour photos of 32 x 32 pixels in 10 classes: airplane,
    automobile, bird, cat, deer, dog, frog, horse, ship and truck (collected in 2009 at the University of
    Toronto). It is much harder than MNIST: the objects come in every pose, colour, size and background, and at
    32 x 32 pixels even people sometimes hesitate between a cat and a dog.

    `P.get_cifar10(...)` works like `P.get_mnist` in P1 (it downloads about 170 MB into `data/` the first time).
    We use 20 000 of its 50 000 training photos, so that training takes about 2 minutes, and 2 000 of its
    10 000 test photos for validation.
    """),
    code("""
    train_loader, val_loader, classes = P.get_cifar10(n_train=20000, n_val=2000, batch_size=64)
    P.show_samples(train_loader, classes)
    images, labels = next(iter(train_loader))      # the first batch
    print("images:", images.shape)
    """),
    md(r"""
    A batch is `[64, 3, 32, 32]`: 64 photos, 3 colour channels, 32 x 32 pixels. So the input shape of one photo
    is `3 x 32 x 32`, 49 times fewer pixels than VGG16's 224 x 224.

    ### 3.2 · The recipe, shrunk

    | | VGG16 | mini-VGG |
    |---|---|---|
    | input | 3 x 224 x 224 | 3 x 32 x 32 |
    | blocks | 5 blocks of 2 or 3 convolutions | 3 blocks of 2 convolutions |
    | feature maps | 64, 128, 256, 512, 512 | 32, 64, 128 |
    | after each convolution | ReLU | BatchNorm, then ReLU |
    | picture after the blocks | 512 x 7 x 7 | 128 x 4 x 4 |
    | classifier | 25 088 -> 4 096 -> 4 096 -> 1 000, dropout 0.5 twice | 2 048 -> 128 -> 10, dropout 0.3 once |

    Everything else is the same recipe: only 3 x 3 convolutions with padding 1, max pooling after each block,
    the maps doubling from block to block.

    One layer is new. **Batch normalisation** (`nn.BatchNorm2d`, "BatchNorm") rescales each feature map so that
    over the current batch its values average 0 with a spread of 1, then applies a scale and a shift that it
    learns. This keeps the numbers in a steady range from layer to layer, so a deep stack trains faster and is
    less sensitive to the learning rate. The original VGG had none (BatchNorm was invented a year later, in
    2015): its authors got their deepest networks to train only by starting them from the weights of a
    shallower network they had trained first.
    """),
    api("nn.BatchNorm2d"),
    code("""
    class MiniVGG(nn.Module):
        def __init__(self):
            super().__init__()
            # block 1: 32 feature maps, the photo stays 32 x 32, then pooling halves it
            self.block1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),       # layer 1: 3 colour channels in, 32 maps out
                nn.BatchNorm2d(32),                               # rescale each of the 32 maps
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),      # layer 2
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),                                  # 32 x 32 -> 16 x 16
            )
            # block 2: 64 feature maps (twice as many), 16 x 16
            self.block2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),      # layer 3
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),      # layer 4
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),                                  # 16 x 16 -> 8 x 8
            )
            # block 3: 128 feature maps, 8 x 8
            self.block3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),     # layer 5
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),    # layer 6
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),                                  # 8 x 8 -> 4 x 4
            )
            # classifier: one hidden fully connected layer instead of VGG16's two big ones
            self.classifier = nn.Sequential(
                nn.Flatten(),                                     # 128 x 4 x 4 = 2048 numbers in a row
                nn.Linear(2048, 128),                             # layer 7
                nn.ReLU(),
                nn.Dropout(0.3),                                  # during training: drop 30% of the numbers
                nn.Linear(128, 10),                               # layer 8: one score per CIFAR-10 class
            )

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.classifier(x)
            return x


    torch.manual_seed(0)
    mini_vgg = MiniVGG()
    P.show_model(mini_vgg, (3, 32, 32), title="mini-VGG: every layer")
    """),
    md(r"""
    The same pyramid as VGG16, only smaller: the photo shrinks 32 -> 16 -> 8 -> 4 while the maps grow 32 -> 64
    -> 128. The pink boxes are the BatchNorm layers.
    """),
    code("""
    P.summary(mini_vgg, (3, 32, 32))
    """),
    md(r"""
    A BatchNorm layer has 2 parameters per map (its learned scale and shift): 64 for 32 maps. Here, too, the
    largest layer is the first fully connected one (2 048 x 128 + 128 = 262 272 parameters, almost half of the
    network), and the convolutions do nearly all the multiply-adds.

    `P.cost` with `epoch_images=20000` estimates the time of one epoch over our 20 000 training photos:
    """),
    code("""
    mini_cost = P.cost(mini_vgg, (3, 32, 32), name="mini-VGG", epoch_images=20000)
    """),
    md(r"""
    ### 3.3 · Side by side

    The two networks take photos of different sizes, so we put their two cost reports next to each other (the
    last column: how many times larger VGG16 is).
    """),
    code("""
    def compare_line(label, vgg_value, mini_value):
        print(f"{label:<36}{vgg_value:>14,.1f}{mini_value:>12,.2f}{vgg_value / mini_value:>10,.0f} x")

    print(f"{'':<36}{'VGG16':>14}{'mini-VGG':>12}{'ratio':>12}")
    compare_line("input pixels (thousands)", 224 * 224 / 1000, 32 * 32 / 1000)
    compare_line("parameters (millions)", vgg16_cost["params"] / 1e6, mini_cost["params"] / 1e6)
    compare_line("memory of the weights (MB)", vgg16_cost["size_mb"], mini_cost["size_mb"])
    compare_line("multiply-adds per photo (millions)", vgg16_cost["mult_adds"] / 1e6, mini_cost["mult_adds"] / 1e6)
    compare_line("prediction, ms for one photo", vgg16_cost["ms_per_image_single"], mini_cost["ms_per_image_single"])
    compare_line("training, ms per photo",
                 vgg16_cost["train_step_ms"] / vgg16_cost["batch_size"],
                 mini_cost["train_step_ms"] / mini_cost["batch_size"])
    """),
    md(r"""
    The mini-VGG is about 250 times smaller and needs about 400 times less work per photo: small enough to train
    on a laptop in a couple of minutes.
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · Train the mini-VGG

    `P.train` works as in P1: in every epoch it shows the network all 20 000 training photos in batches of 64,
    then measures loss and accuracy on the 2 000 validation photos, and redraws the curves. Ten classes means
    that guessing would be right 10% of the time.

    **Expected time: about 2 to 3 minutes** (5 epochs of 25 to 30 seconds here; up to 5 minutes on a slow
    laptop).
    """),
    code("""
    history_mini = P.train(mini_vgg, train_loader, val_loader, epochs=5, lr=0.001, name="mini-VGG")
    """),
    md(r"""
    **Reading the curves.** The validation accuracy climbs from 55.8% after the first epoch to 73.0% after the
    fifth (the star): the mini-VGG recognises almost 3 of every 4 photos it has never seen, where guessing would
    get 1 in 10. Both curves are still rising, so more epochs would still help.

    Unlike in P1, the validation curve lies above the training curve for most of the run. That is not a
    mistake: the training accuracy is the average over the whole epoch, while the network is still learning,
    and with dropout switched on; the validation accuracy is measured at the end of the epoch, with dropout
    off. When the training curve climbs clearly above the validation curve, the network starts memorising.

    `P.show_predictions` shows the answers for the first 16 validation photos (green = right, with how sure the
    network is; red = wrong, with the true class).
    """),
    code("""
    P.show_predictions(mini_vgg, val_loader, classes)
    """),
    md(r"""
    ### Which classes get confused?

    `P.confusion(model, loader, classes)` counts, for every true class (row), what the network answered
    (column). The grey diagonal holds the correct answers; every orange cell is a kind of mistake, darker =
    more often. It also prints the five most common mistakes.
    """),
    code("""
    confusion_matrix = P.confusion(mini_vgg, val_loader, classes)
    """),
    md(r"""
    **What to look at.** 541 of the 2 000 photos are wrong (27.1%). The most common mistake by far: a dog taken
    for a cat (62 times), and cats are also often taken for dogs (27). Most of the orange sits in the middle of
    the matrix, among the six animal classes (bird, cat, deer, dog, frog, horse): animals are confused with
    animals. Vehicles are confused with vehicles (an automobile taken for a truck 18 times, an airplane for a
    ship 14 times), and rarely with animals; the exception is airplane and bird (16), often a small shape on a
    plain sky. The column "frog" collects many mistakes (cat 33, bird 25, deer 21 times): at 32 x 32 pixels a
    small brown or green animal on the ground looks much like a frog. The best rows are automobile, frog and
    ship; the worst are bird, cat and dog.

    `only_wrong=True` shows only mistakes. Several of them would make people hesitate too at this size.
    """),
    code("""
    P.show_predictions(mini_vgg, val_loader, classes, only_wrong=True, title="16 of the mini-VGG's mistakes")
    """),
    md(r"""
    ### What did the first layer learn?

    `P.show_filters(model)` draws the 32 filters of the first convolution. They work on colour photos, so each
    filter is drawn as a tiny 3 x 3 colour patch (its weights for red, green and blue, scaled to visible
    colours).
    """),
    code("""
    P.show_filters(mini_vgg)
    """),
    md(r"""
    **What to look at.** Do not expect much. Each patch shows a colour preference (greenish, purple, orange,
    ...) and a change from light to dark somewhere across its 9 pixels, but 3 x 3 pixels are too few to show a
    recognisable pattern. Older networks with big first filters (AlexNet's 11 x 11) show clear stripes and colour
    blobs here; in a VGG-style network the patterns appear only when the small filters are stacked, which is
    what the feature maps below show.

    ### Feature maps: early and late

    `P.show_feature_maps(model, image, layer=...)` runs one photo through the network and shows 16 feature
    maps of one layer: bright where the map responds. We look at what leaves the ReLU at the end of block 1
    (layer `block1.5`, 32 maps of 32 x 32) and at the end of block 3 (layer `block3.5`, 128 maps of 8 x 8).
    """),
    code("""
    image, label = val_loader.dataset[0]
    print("this photo shows:", classes[label])
    P.show_feature_maps(mini_vgg, image, layer="block1.5")
    """),
    md(r"""
    **What to look at (block 1).** The maps still look like the photo, each highlighting something else: some
    trace parts of the puppy's outline (maps 0 and 2), one lights up the puppy and leaves most of the background
    dark (map 3), one does the opposite (map 12: bright grass, dark puppy), and several pick up the fine texture
    of the grass as short stripes (maps 7, 8 and 9). After two 3 x 3 convolutions each pixel of these maps has a
    view of only 5 x 5 pixels: edges, colours and textures.
    """),
    code("""
    P.show_feature_maps(mini_vgg, image, layer="block3.5")
    """),
    md(r"""
    **What to look at (block 3).** The photo is gone: each map is a coarse grid of 8 x 8 cells, lit in a few
    places only. Several maps light up in the middle, where the puppy's head and body are (maps 3, 10, 13 and
    14), others along the top edge, in the background (maps 11 and 12), and some hardly at all for this photo
    (map 2). Six 3 x 3 convolutions and two poolings have widened the view of each cell to 32 x 32 pixels, as
    large as the whole photo: a cell no longer reports "an edge here" but "something like this, around here".
    These are the patterns the classifier turns into the answer "dog".
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · Try it: build your own mini-VGG

    Now you change the recipe. As in P1 there are three cells:

    1. a **builder function** that turns a few settings into a network: plain `for` loops that add layers to a
       list, nothing hidden;
    2. a **Try it** cell with the settings: change values there;
    3. a **run** cell that loads the data, builds the network, draws it, measures its cost, trains it and
       records the result.

    So that you can try many things, the run settings start smaller than in section 4: 5 000 training photos
    and 5 epochs, **about 40 seconds** per run. `P.remember` records every version, `P.leaderboard()` lists them.
    First, record the mini-VGG of section 4:
    """),
    code("""
    P.remember("mini-VGG of section 4 (20000 photos)", mini_vgg, history_mini)
    """),
    md(r"""
    `build_mini_vgg(...)` repeats the recipe: for every entry of `blocks` (the number of maps of that block) it
    adds `convs_per_block` convolutions (each with BatchNorm if switched on, and ReLU), then a pooling layer. It
    keeps track of the picture size (`size`), which pooling halves, to know how many numbers reach the first
    fully connected layer: maps x size x size. With the default settings it builds exactly the network of
    section 3, as one flat `nn.Sequential`.
    """),
    code("""
    def build_mini_vgg(blocks, convs_per_block, use_batchnorm, dropout, dense_size):
        layers = []
        in_channels = 3                                       # colour photos: red, green, blue
        size = 32                                             # height and width of the photo
        for maps in blocks:                                   # one block per entry of the list
            for k in range(convs_per_block):                  # the 3x3 convolutions of this block
                layers.append(nn.Conv2d(in_channels, maps, kernel_size=3, padding=1))
                if use_batchnorm:
                    layers.append(nn.BatchNorm2d(maps))
                layers.append(nn.ReLU())
                in_channels = maps                            # the next convolution receives these maps
            layers.append(nn.MaxPool2d(2))                    # end of the block: halve height and width
            size = size // 2
        layers.append(nn.Flatten())                           # maps x size x size numbers in a row
        layers.append(nn.Linear(in_channels * size * size, dense_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dense_size, 10))              # 10 scores, one per class
        return nn.Sequential(*layers)
    """),
    md(r"""
    **Try it.** Change the settings, then run this cell and the next one. **Data augmentation**
    (`augment = True`): every time a training photo is used, it is shifted by up to 4 pixels and, half of the
    time, mirrored left to right, so the network rarely sees exactly the same photo twice (the validation
    photos stay as they are). With 5 blocks the photo is down to 1 x 1 pixel; a sixth block stops with a
    message from `P` saying that the picture is too small to pool.
    """),
    code("""
    # ---------------- Try it: settings of the mini-VGG ----------------
    blocks = [32, 64, 128]        # feature maps of each block; one entry = one block (VGG16: [64, 128, 256, 512, 512])
    convs_per_block = 2           # 3x3 convolutions in each block (VGG16: 2 or 3)
    use_batchnorm = True          # BatchNorm after every convolution (VGG16: no)
    dropout = 0.3                 # fraction of numbers dropped in the classifier during training (0 = no dropout)
    dense_size = 128              # neurons of the hidden fully connected layer (VGG16: 4096)
    augment = False               # randomly shift and mirror the training photos
    # ---------------- run settings ----------------
    epochs = 5                    # passes over the training photos
    learning_rate = 0.001         # size of each weight change
    n_train = 5000                # training photos (at most 50000)
    batch_size = 64               # photos per batch
    version_name = f"{blocks} x{convs_per_block} bn={use_batchnorm} drop={dropout} dense={dense_size} aug={augment}"
    """),
    md(r"""
    **Expected time: about 40 seconds** with the settings above (more photos, more epochs, more maps or more
    blocks take longer: the cost report tells you how long one epoch takes).
    """),
    code("""
    train_loader, val_loader, classes = P.get_cifar10(n_train=n_train, n_val=2000, batch_size=batch_size, augment=augment)
    torch.manual_seed(0)
    my_vgg = build_mini_vgg(blocks, convs_per_block, use_batchnorm, dropout, dense_size)
    P.show_model(my_vgg, (3, 32, 32), title=version_name)
    P.cost(my_vgg, (3, 32, 32), name=version_name, epoch_images=n_train)
    history = P.train(my_vgg, train_loader, val_loader, epochs=epochs, lr=learning_rate, name=version_name)
    P.remember(version_name, my_vgg, history)
    """),
    md(r"""
    ### 5.1 · The leaderboard

    Run this cell after every experiment: every version so far, best validation accuracy first.
    """),
    code("""
    P.leaderboard()
    """),
    md(r"""
    ### 5.2 · Guided experiments

    For each one: change the setting in the **Try it** cell, **predict** what will happen to the accuracy, the
    parameters and the time, run the two cells, and compare with your prediction. Then open the fold-out to read
    what we saw. Put the setting back afterwards.

    **A word on chance.** With the default settings we got a best validation accuracy of 52.9% (epoch 4). Short
    runs on 5 000 photos are noisy: in that same run the accuracy fell to 49.1% one epoch later. Treat
    differences of less than about 4 points as chance; look for bigger effects, and look at the training
    accuracy and the time too.

    **1. No BatchNorm.** `use_batchnorm = False`.
    <details><summary>What we saw</summary>

    40.3% instead of 52.9%: the biggest loss of all the architecture changes. Without BatchNorm the network
    learns much more slowly: 25.8% after the first epoch instead of 36.0%, and still climbing at the end. It
    has only 896 parameters fewer (550 570), so the difference is not size: it is how easily the deep stack
    trains. This is the problem the VGG authors had in 2014.
    </details>

    **2. One convolution per block.** `convs_per_block = 1`.
    <details><summary>What we saw</summary>

    55.5%: at least as good as two convolutions per block (52.9%; the difference is within chance), with 357 258
    parameters instead of 551 466 and a quarter of the multiply-adds (10.6 million instead of 38.9 million). With
    only 5 000 photos and 5 epochs, the deeper network does not get to profit from its extra layers. Deeper is
    not automatically better.
    </details>

    **3. A fourth block.** `blocks = [32, 64, 128, 256]`.
    <details><summary>What we saw</summary>

    The photo now shrinks to 2 x 2 before the classifier. The new block adds two convolutions with 256 maps, so
    the network grows to 1 306 666 parameters (2.4 times as many), while the first fully connected layer halves
    (256 x 2 x 2 = 1 024 inputs instead of 2 048). Accuracy: 48.0%, no better than three blocks, and each epoch
    takes longer. Again: more layers did not help in a short run on few photos.
    </details>

    **4. Narrower and wider.** `blocks = [16, 32, 64]`, then `blocks = [64, 128, 256]`.
    <details><summary>What we saw</summary>

    Narrower: 205 018 parameters, 10.0 million multiply-adds (a quarter), 51.0%, almost as good as the default.
    Wider: 1 672 906 parameters and 153 million multiply-adds (4 times the work, about 4 times the time per epoch),
    and only 47.7%. Doubling every block's maps quadruples the work of the convolutions (twice the maps in,
    twice the maps out), and in a short run on few photos it does not pay off.
    </details>

    **5. Data augmentation.** `augment = True`.
    <details><summary>What we saw</summary>

    48.3% instead of 52.9%, and the training accuracy drops from 57.9% to 47.9%. Every photo looks a little
    different each time, so the network can no longer memorise the training photos; but in 5 epochs that makes
    learning harder. With `epochs = 10` as well (about 2 minutes), we got 60.4% with augmentation against 64.5%
    without, but look at the training accuracy: without augmentation it reaches 74.1%, 10 to 15 points above
    the validation accuracy (memorising), with augmentation 58.8%, below it, and the validation accuracy was
    still rising at epoch 10. Augmentation is for long trainings, where memorising is the main problem.
    </details>

    **6. More data.** `n_train = 20000` (about 2 minutes).
    <details><summary>What we saw</summary>

    This is the network of section 4 again: 73.0% instead of 52.9%, by far the biggest gain of all experiments,
    without changing the network at all. Four times the photos cost about four times the time per epoch.
    </details>

    **7. Less and more dropout.** `dropout = 0`, then `dropout = 0.5`.
    <details><summary>What we saw</summary>

    Dropout 0: 52.4%, the same as with 0.3, but the training accuracy climbs to 66.3% (instead of 57.9%), 14
    points above the validation accuracy: the network starts memorising its training photos. Dropout 0.5:
    48.9%, with a training accuracy of 45.4%; dropping half of the numbers slows learning down in a short run.
    Dropout does not make a network better at once; it keeps the gap between training and validation small,
    which matters in longer runs.
    </details>

    **8. Learning rate too high.** `learning_rate = 0.1`.
    <details><summary>What we saw</summary>

    10.8% at best, then 10.4%: no better than guessing one of the 10 classes. The weight changes are so big that
    every step destroys what the last one learned. Even BatchNorm cannot rescue a learning rate 100 times too
    large.
    </details>
    """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · What to remember

    * **VGG's recipe:** only 3 x 3 convolutions, stacked in blocks, max pooling after each block, the number of
      feature maps doubling (64 -> 128 -> 256 -> 512), then fully connected layers. Two stacked 3 x 3
      convolutions see as much of the picture as one 5 x 5, with 28% fewer weights and one more ReLU.
    * **VGG16 in numbers:** 138 million parameters (528 MB) and 15.5 billion multiply-adds per 224 x 224 photo.
      The three fully connected layers hold 89% of the parameters; the 13 convolutions do 99% of the work.
    * **Training is the expensive part:** here more than half a second per photo, a week or more per pass over
      ImageNet. Big networks are trained once, on many graphics cards, and everyone else reuses their trained
      weights.
    * **The same recipe, shrunk** to 0.55 million parameters, recognised 73% of CIFAR-10 photos it had never
      seen after about 2 minutes of training on a CPU (guessing: 10%). Its mistakes are mostly between classes
      that look alike.
    * **What mattered in the experiments:** more data helped by far the most; BatchNorm made a deep stack train
      much faster; a learning rate that is too high breaks training completely. Extra layers and maps did not
      help in a short run on few photos, and dropout and augmentation mainly keep the network from memorising.

    ### Words from this notebook

    Words from P1 (layer, convolution, filter, feature map, pooling, padding, parameters, epoch, ...) are in
    P1's list. New here, or used in a new way:

    | word | meaning |
    |---|---|
    | augmentation (data augmentation) | changing each training photo a little every time it is used (here: shifting and mirroring), so the network rarely sees the same photo twice |
    | batch normalisation (BatchNorm) | a layer that rescales each feature map to average 0 and spread 1 over the batch, then applies a learned scale and shift |
    | block | a group of layers repeated as a unit; in VGG: 2 or 3 convolutions with ReLU, then pooling |
    | CIFAR-10 | 60 000 colour photos of 32 x 32 pixels in 10 classes (airplane, ..., truck) |
    | class (Python) | a recipe from which objects are built; a network class has `__init__` (creates the layers) and `forward` (the order the data goes through them) |
    | confusion matrix | a table that counts, for every true class, which class the network answered |
    | dropout | during training, setting a random fraction of the numbers in a layer to 0, so the network cannot rely on any single neuron |
    | forward pass | running pictures through the network, from the input to the scores |
    | ImageNet | 1.28 million training photos in 1 000 classes, the benchmark VGG was built for |
    | memorising | learning the training photos themselves instead of what their classes look like: training accuracy rises, validation accuracy does not |
    | receptive field (view) | the part of the input picture that one output number depends on |
    | torchvision | PyTorch's library for computer vision: datasets, ready-made networks, image tools |
    | VGG, VGG16 | the network family of the Visual Geometry Group (Oxford, 2014); VGG16 has 13 convolutions and 3 fully connected layers |
    """),
]
