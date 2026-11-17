from nb import md, code, api

NAME = "P1-dense-vs-conv-mnist"

CELLS = [
    md(r"""
    # P1 · Two networks read hand-written digits: fully connected vs convolutional

    **What we do today**

    1. Look at MNIST, a collection of small pictures of hand-written digits 0 to 9.
    2. Build and train **network A**, a fully connected network: every pixel is wired to every neuron.
    3. Build and train **network B**, a small convolutional network (CNN): small filters slide over the image.
    4. Put the two side by side: accuracy, size, speed, and the mistakes each one makes.
    5. Change the settings of both networks yourself and watch what happens (section 5, **Try it**).

    **How to use this notebook.** You do not write any code today. Click into the first code cell and press
    **Shift+Enter** to run it; that moves you to the next cell. Run the cells from top to bottom: later cells use
    what earlier cells made. Cells marked **Try it** contain numbers you may change and run again. The whole
    notebook runs in about 1 to 3 minutes on a laptop; every training cell says how long it takes.

    * `P.something(...)` is a function of the course toolkit. The text explains each one in a sentence where it
      is first used: it does the plumbing (loading data, drawing, timing, the training loop) so that you can
      concentrate on the networks.
    * Closed boxes **Functions used here (click to open)** explain the PyTorch functions in the next cell, with
      a small example. Open them when you want the details.
    * **New words** are in bold where they are explained. All of them are collected at the end.
    * Numbers quoted in the text are what the cells print. On your computer the last digit can differ a little.
    """),

    # ------------------------------------------------------------------ 0
    md(r"""
    ## 0 · Setup

    The first cell prepares everything, and must run first every time you open the notebook:

    * the `while` loop finds the folder `cvcourse`, the course's helper code, and lets Python import it;
    * `import torch` loads **PyTorch**, the library we build and train networks with; `torch.nn` (short `nn`)
      holds its building blocks, the **layers**;
    * `from cvcourse import practical as P` loads the course toolkit under the short name `P`;
    * `P.setup(seed=0)` fixes the **random seed**, the starting point of every random number (the starting
      weights of a network, the order of the images), so that every run gives the same results; it also prints
      the PyTorch version and whether it computes on the CPU or on a graphics card (a CPU is fine).
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

    P.setup(seed=0)
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · The data: MNIST

    **MNIST** is a classic collection of 70 000 hand-written digits, each a small grey picture of 28 x 28
    **pixels** (the little squares an image is made of). Every picture comes with its **label**: the digit it
    shows, 0 to 9. Our networks must learn to give the right label for a picture. Choosing one answer out of a
    fixed list (here the 10 digits, called the **classes**) is called **classification**.

    We split the pictures into two groups:

    * the **training set** (10 000 pictures): the network learns from these;
    * the **validation set** (2 000 other pictures): the network never learns from these; we only use them to
      measure how well it does on pictures it has not seen. That is the number that counts.

    `P.get_mnist(...)` loads the pictures (it downloads about 12 MB into the folder `data/` the first time) and
    returns three things: `train_loader` and `val_loader`, which hand out the training and validation pictures
    in **batches** (small groups of pictures processed together, here 64), and `classes`, the names of the 10
    classes. `P.show_samples(...)` draws the first 16 training pictures with their labels.
    """),
    code("""
    train_loader, val_loader, classes = P.get_mnist(n_train=10000, n_val=2000, batch_size=64)
    P.show_samples(train_loader, classes)
    """),
    md(r"""
    What does a batch look like to the network? The next cell takes the first batch out of `train_loader` and
    prints its shape: the size of the data along each direction (each **axis**).
    """),
    api("Tensor.shape", "Tensor.tolist"),
    code("""
    images, labels = next(iter(train_loader))      # the first batch
    print("images:", images.shape)
    print("labels:", labels.shape, "  the first ten:", labels[:10].tolist())
    """),
    md(r"""
    **Reading the shapes.** `images` has the shape `[64, 1, 28, 28]`:

    | axis | size | meaning |
    |---|---|---|
    | 0 | 64 | 64 pictures in this batch |
    | 1 | 1 | 1 **channel**: the pictures are grey (a colour picture has 3 channels: red, green, blue) |
    | 2 | 28 | 28 rows of pixels (height) |
    | 3 | 28 | 28 columns of pixels (width) |

    So one picture is `1 x 28 x 28`: this is the **input shape** we give to every `P.` function below. `labels`
    holds 64 numbers, the digit of each picture. Such multi-dimensional arrays of numbers are called
    **tensors** in PyTorch.

    The pixel values are not 0 to 255 as in an image file: `P.get_mnist` has **normalised** them (shifted and
    scaled so that, over all pictures, they average 0 with a spread of 1), which helps networks learn.
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · Network A: fully connected

    ### The idea

    A **neuron** takes a list of numbers, multiplies each by its own **weight**, adds them up, adds one more
    number (the **bias**) and passes the result on. A **fully connected layer** (also called a **dense** or
    **linear** layer) is a row of neurons that each see *every* number of the input. The weights and biases are
    the network's **parameters**: the numbers that training adjusts.

    Network A has four layers, applied one after the other:

    1. `nn.Flatten()` lays the 28 x 28 picture out as one long row of 784 numbers (28 x 28 = 784). This throws
       the 2-D layout away: after flattening, the network no longer knows which pixels are neighbours. It sees
       784 separate numbers, and anything it learns about one place of the picture it has to learn again for
       every other place.
    2. `nn.Linear(784, 128)`: 128 neurons, each connected to all 784 pixels.
    3. `nn.ReLU()`, an **activation**: it keeps positive numbers and turns negative ones into 0. Without it,
       two linear layers in a row would be no more powerful than one.
    4. `nn.Linear(128, 10)`: 10 neurons, one per digit. Their outputs are the network's 10 **scores**; the digit
       with the highest score is its answer.

    ### The parameters of `nn.Linear`

    | parameter | meaning | network A |
    |---|---|---|
    | `in_features` | how many numbers come in (must equal what the layer before produces) | 784, then 128 |
    | `out_features` | how many neurons, so how many numbers go out | 128, then 10 |
    | `bias` (default `True`) | each neuron also has its own added number | on |

    A linear layer has `in_features x out_features` weights plus `out_features` biases.

    `nn.Sequential(...)` chains layers: the output of each one is the input of the next. `torch.manual_seed(0)`
    makes the random starting weights the same on every run.
    """),
    api("torch.manual_seed", "nn.Sequential", "nn.Flatten", "nn.Linear", "nn.ReLU"),
    code("""
    torch.manual_seed(0)                 # the same random starting weights every time
    dense_net = nn.Sequential(
        nn.Flatten(),                    # 1 x 28 x 28 picture  ->  784 numbers in a row
        nn.Linear(784, 128),             # 784 numbers in  ->  128 neurons  ->  128 numbers out
        nn.ReLU(),                       # keep positive numbers, set negative ones to 0
        nn.Linear(128, 10),              # 128 numbers in  ->  10 scores out, one per digit 0-9
    )
    print(dense_net)
    """),
    md(r"""
    ### Look at it

    `P.show_model(model, input_shape)` draws the network: one box per layer in the order the data flows,
    coloured by the kind of layer, with what the layer does and the shape of what comes out of it written
    underneath. A box's height shows the size of the picture at that point; a tall thin bar is a flat list of
    numbers, and the taller the bar, the longer the list.
    """),
    code("""
    P.show_model(dense_net, (1, 28, 28), title="Network A: fully connected")
    """),
    md(r"""
    The picture enters as `1x28x28`, is flattened into 784 numbers, becomes 128 numbers, then 10 scores.

    `P.summary(model, input_shape)` gives the same information as a table, one row per layer, with the number of
    parameters of every layer and its **multiply-adds** (how many multiplications and additions the layer
    needs for one picture: a measure of the work, so of the time it takes).
    """),
    code("""
    P.summary(dense_net, (1, 28, 28))
    """),
    md(r"""
    Check the parameter counts by hand: the first linear layer has 784 x 128 = 100 352 weights plus 128 biases
    = 100 480; the second 128 x 10 + 10 = 1 290; together 101 770. Flatten and ReLU have no parameters: they
    compute, but learn nothing.

    `P.cost(model, input_shape)` measures what the network costs on this computer: parameters, memory,
    multiply-adds, the time to answer for one picture, the time of one **training step** (one batch: compute the
    scores, work out how to change every weight, change them), and from that the time of one **epoch** (one
    pass over all training pictures).
    """),
    code("""
    P.cost(dense_net, (1, 28, 28), name="Network A")
    """),
    md(r"""
    ### Train it

    `P.train(model, train_loader, val_loader, epochs=..., lr=...)` trains the network. In every epoch it shows
    the network all 10 000 training pictures, batch by batch; after each batch it changes every weight a little
    in the direction that makes the scores better. Then it measures the network on the 2 000 validation
    pictures. Two numbers describe how well the network does:

    * the **loss**: how wrong the scores are, averaged over the pictures (lower is better; training makes it
      smaller);
    * the **accuracy**: the percentage of pictures whose highest score is the right digit (higher is better).

    It prints one line per epoch and draws both numbers as **curves**, updated after every epoch: blue for the
    training pictures, orange for the validation pictures. The star marks the best validation accuracy.

    Two settings of the run:

    * `epochs`: how many passes over the training pictures;
    * `lr`, the **learning rate**: how big each change of the weights is. Too small and learning is slow;
      too big and the weights jump around and nothing is learned. 0.001 is a good default for the **Adam**
      optimiser, the method `P.train` uses to change the weights.

    <details>
    <summary><b>What <code>P.train</code> does inside</b> (optional, click to open)</summary>

    No magic: the heart of `P.train` is this loop. The rest of the function prints the lines and draws the
    curves.

    ```python
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)   # the method that changes the weights
    loss_fn = nn.CrossEntropyLoss()                               # the usual loss for classification
    for epoch in range(epochs):
        model.train()                              # training mode
        for images, labels in train_loader:        # one batch of 64 pictures at a time
            scores = model(images)                 # forward pass: 10 scores per picture
            loss = loss_fn(scores, labels)         # how wrong are the scores?
            optimizer.zero_grad()                  # forget the previous batch's corrections
            loss.backward()                        # work out how each weight should change (backpropagation)
            optimizer.step()                       # change every weight a little
        # then: loss and accuracy on the validation pictures, without changing any weight
    ```
    </details>

    **Expected time: about 2 seconds** (up to 10 seconds on a slow laptop).
    """),
    code("""
    history_dense = P.train(dense_net, train_loader, val_loader, epochs=5, lr=0.001, name="Network A: fully connected")
    """),
    md(r"""
    **Reading the curves.** The validation accuracy climbs from 91.8% after the first epoch to 94.8% after the
    fifth: network A reads about 19 of every 20 digits right. The loss falls on both sides. The training
    accuracy ends a little higher than the validation accuracy (96.9% against 94.8%): the network is slightly
    better on the pictures it learned from than on new ones. That gap is normal while it stays small.

    `P.show_predictions(model, loader, classes)` shows the network's answers for the first 16 validation
    pictures: in green when right (with how sure the network is, in percent), marked **WRONG** in red with the
    true digit when wrong.
    """),
    code("""
    P.show_predictions(dense_net, val_loader, classes)
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · Network B: a small convolutional network

    ### The idea

    A **convolution** layer looks at the picture through small windows. Its **filter** (also called **kernel**)
    is a small grid of weights, for example 3 x 3 = 9 weights. The filter slides over the picture; at every
    position it multiplies its 9 weights with the 9 pixels under it and adds them up (plus a bias). The results,
    one per position, form a new picture called a **feature map**: it is bright where the filter's pattern (an
    edge, a curve, a corner) appears in the picture.

    Two things make this a good fit for pictures:

    * the 2-D layout is kept: a filter always sees neighbouring pixels together;
    * the same 9 weights are used at every position, so a pattern learned in one place is found everywhere,
      with very few parameters.

    A layer has many filters, each learning its own pattern, so it outputs many feature maps (also called
    **channels**). The next convolution layer then combines those maps into more complex patterns.

    A **pooling** layer shrinks the feature maps: `nn.MaxPool2d(2)` keeps the largest value of every 2 x 2
    square, halving height and width. It has no parameters.

    ### The parameters of `nn.Conv2d` and `nn.MaxPool2d`

    | parameter | meaning | network B |
    |---|---|---|
    | `in_channels` | how many maps come in: 1 for a grey picture, else the `out_channels` of the convolution before | 1, then 16 |
    | `out_channels` | how many filters, so how many feature maps come out | 16, then 32 |
    | `kernel_size` | the filter's size: 3 means 3 x 3 pixels | 3 |
    | `stride` (default 1) | the step of the filter: 1 = every position, 2 = every second position (the output is half as big) | 1 |
    | `padding` (default 0) | rows and columns of zeros added around the picture, so the filter can also sit on the border | 1 |
    | `nn.MaxPool2d(kernel_size)` | size of the pooling window; it steps by the same amount, so 2 halves the size | 2 |

    **The output-size rule.** For a picture of size `n` (height or width):

    > output size = (n + 2 x padding - kernel_size) / stride + 1, rounded down

    On network B, layer by layer (written channels x height x width):

    | layer | computation | output shape |
    |---|---|---|
    | input | | 1 x 28 x 28 |
    | `nn.Conv2d(1, 16, kernel_size=3, padding=1)` | (28 + 2 - 3) / 1 + 1 = 28 | 16 x 28 x 28 |
    | `nn.ReLU()` | no change | 16 x 28 x 28 |
    | `nn.MaxPool2d(2)` | 28 / 2 = 14 | 16 x 14 x 14 |
    | `nn.Conv2d(16, 32, kernel_size=3, padding=1)` | (14 + 2 - 3) / 1 + 1 = 14 | 32 x 14 x 14 |
    | `nn.ReLU()` | no change | 32 x 14 x 14 |
    | `nn.MaxPool2d(2)` | 14 / 2 = 7 | 32 x 7 x 7 |
    | `nn.Flatten()` | 32 x 7 x 7 = 1 568 | 1568 |
    | `nn.Linear(1568, 64)`, `nn.ReLU()` | | 64 |
    | `nn.Linear(64, 10)` | | 10 |

    Padding 1 with a 3 x 3 filter keeps the size (the filter "loses" one pixel on each side and the padding
    gives it back). The `in_features` of the first linear layer, 1 568, must equal what comes out of the
    flatten: the table tells you.
    """),
    api("nn.Conv2d", "nn.MaxPool2d"),
    code("""
    torch.manual_seed(0)
    cnn = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),     # 1 grey map in -> 16 feature maps, 3x3 filters: 16 x 28 x 28
        nn.ReLU(),
        nn.MaxPool2d(2),                                # halve height and width: 16 x 14 x 14
        nn.Conv2d(16, 32, kernel_size=3, padding=1),    # 16 maps in -> 32 maps out: 32 x 14 x 14
        nn.ReLU(),
        nn.MaxPool2d(2),                                # 32 x 7 x 7
        nn.Flatten(),                                   # 32 x 7 x 7 = 1568 numbers in a row
        nn.Linear(1568, 64),                            # 1568 numbers in -> 64 neurons
        nn.ReLU(),
        nn.Linear(64, 10),                              # 10 scores, one per digit
    )
    P.show_model(cnn, (1, 28, 28), title="Network B: convolutional")
    """),
    md(r"""
    **What to look at.** The boxes get shorter (the picture shrinks: 28, 14, 7) and wider (more feature maps: 1,
    16, 32). This "pyramid" is the shape of almost every convolutional network: the picture gets smaller while
    the number of patterns grows. At the end, flatten and two linear layers turn the patterns into 10 scores,
    like network A. The diagram continues on a second row (follow the grey line).
    """),
    code("""
    P.summary(cnn, (1, 28, 28))
    """),
    md(r"""
    **Counting parameters.** A convolution has `out_channels x in_channels x kernel x kernel` weights plus one
    bias per filter: 16 x 1 x 3 x 3 + 16 = 160 for the first, 32 x 16 x 3 x 3 + 32 = 4 640 for the second.
    The linear layers have 1 568 x 64 + 64 = 100 416 and 64 x 10 + 10 = 650. Total: 105 866, almost exactly as
    many as network A (101 770). But look at the multiply-adds: 1.12 million against 0.10 million. Every filter
    weight is used at every position of the picture (784 positions in the first layer), so a convolution does a
    lot of work with few parameters.
    """),
    code("""
    P.cost(cnn, (1, 28, 28), name="Network B")
    """),
    md(r"""
    The cost report confirms it: the same memory as network A, but about 5 times the time per training step.

    **Expected time: about 8 seconds** (up to 30 seconds on a slow laptop).
    """),
    code("""
    history_cnn = P.train(cnn, train_loader, val_loader, epochs=5, lr=0.001, name="Network B: convolutional")
    """),
    md(r"""
    **Reading the curves.** Already after the first epoch network B reaches 94.8% on the validation pictures, as
    much as network A after five. Its best is 97.7% (epoch 4): it gets about 39 of every 40 digits right. In
    the last epoch the validation accuracy dips a little (97.5%) while the training accuracy still rises: from
    here on, more epochs would bring little.
    """),
    code("""
    P.show_predictions(cnn, val_loader, classes)
    """),
    md(r"""
    ### What did the filters learn?

    `P.show_filters(model)` draws the 16 filters of the first convolution layer, each as a 3 x 3 grid of its
    weights: red = positive weight, blue = negative weight. A filter with red on one side and blue on the
    other responds to an **edge**: a change from dark to bright in that direction. Nobody programmed these:
    they came out of training.
    """),
    code("""
    P.show_filters(cnn)
    """),
    md(r"""
    `P.show_feature_maps(model, image)` runs one validation picture through the network and shows what the
    convolution layers make of it: the first 8 feature maps of each convolution layer, bright where the filter
    found its pattern, black where it found nothing. `val_loader.dataset[0]` is the first validation picture and
    its label.
    """),
    code("""
    image, label = val_loader.dataset[0]
    P.show_feature_maps(cnn, image)
    """),
    md(r"""
    **What to look at.** First row (layer `0`, 28 x 28): each map is the digit with one kind of edge
    highlighted, for example only the upper edges of the strokes, or only the left ones; one map lights up the
    background instead. Second row (layer `3`, 14 x 14): coarser maps that combine the first layer's edges into
    pieces of strokes. The deeper the layer, the smaller the maps and the more abstract the patterns.
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · Side by side

    `P.compare({name: model, ...}, input_shape, histories={name: history, ...})` measures several networks and
    shows one table and one chart, with the best validation accuracy of each run.
    """),
    code("""
    P.compare({"A: fully connected": dense_net, "B: convolutional": cnn}, (1, 28, 28),
              histories={"A: fully connected": history_dense, "B: convolutional": history_cnn})
    """),
    md(r"""
    `P.plot_history({name: history, ...})` draws the training curves of several runs on top of each other
    (solid lines: validation, dashed: training).
    """),
    code("""
    P.plot_history({"A: fully connected": history_dense, "B: convolutional": history_cnn})
    """),
    md(r"""
    **What the numbers say.**

    * **Accuracy:** network B is better, 97.7% against 94.8%. On 2 000 validation pictures that is about 46
      wrong answers instead of 104: fewer than half the mistakes.
    * **Parameters:** almost the same, about 102 000 against 106 000 (0.4 MB each). Network B is not better
      because it is bigger: it is better because its layers suit pictures (neighbouring pixels stay together,
      one filter is reused everywhere).
    * **Time:** network B costs more: about 11 times the multiply-adds, and about 5 to 6 times the training time
      per epoch on this computer (the table shows the measured numbers). Convolutions trade computation for accuracy.

    **Where each network makes its mistakes.** `show_predictions(..., only_wrong=True)` shows only pictures the
    network got wrong. Many of network B's mistakes are digits that are hard to read for people too; network A
    also misses clearly written ones.
    """),
    code("""
    P.show_predictions(dense_net, val_loader, classes, only_wrong=True, title="Network A: 16 of its mistakes")
    P.show_predictions(cnn, val_loader, classes, only_wrong=True, title="Network B: 16 of its mistakes")
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · Try it: change the networks yourself

    Now you are the network designer. For each network there are three cells:

    1. a **builder function** that turns a few settings into a network. Read it once: it is a plain `for` loop
       that adds layers to a list, nothing hidden;
    2. a **Try it** cell with the settings: change numbers there;
    3. a **run** cell that builds the network, draws it, measures its cost, trains it and records the result.

    After changing a setting, run the **Try it** cell and then the run cell. `P.remember(name, model, history)`
    records every version you train, and `P.leaderboard()` lists all of them, best validation accuracy first,
    with their parameters, time per epoch and run settings. If you keep a name but change a setting, the
    leaderboard adds "(2)", "(3)", ... so nothing is lost.

    First, record the two networks from above:
    """),
    code("""
    P.remember("A: fully connected [128]", dense_net, history_dense)
    P.remember("B: convolutional [16, 32]", cnn, history_cnn)
    """),
    md(r"""
    ### 5.1 · The fully connected network

    `build_dense(hidden_sizes)` builds network A with any number of hidden layers: `hidden_sizes = [128]` is
    network A; `[256, 64]` would be two hidden layers of 256 and 64 neurons. `layers.append(...)` adds a layer
    to the list, and `nn.Sequential(*layers)` chains all layers of the list (the `*` hands over the list's
    items one by one).
    """),
    code("""
    def build_dense(hidden_sizes):
        layers = [nn.Flatten()]                          # 1 x 28 x 28 picture -> 784 numbers
        n_in = 28 * 28                                   # how many numbers the next layer receives
        for size in hidden_sizes:                        # one hidden layer per entry of the list
            layers.append(nn.Linear(n_in, size))         # n_in numbers in -> `size` neurons
            layers.append(nn.ReLU())
            n_in = size                                  # the next layer receives `size` numbers
        layers.append(nn.Linear(n_in, 10))               # the output layer: 10 scores
        return nn.Sequential(*layers)
    """),
    md(r"""
    **Try it.** Change the numbers, then run this cell and the next one.
    """),
    code("""
    # ---------------- Try it: settings of the fully connected network ----------------
    hidden_sizes = [128]          # neurons of each hidden layer, e.g. [32], [512], [256, 64]
    # ---------------- run settings ----------------
    epochs = 5                    # passes over the training pictures
    learning_rate = 0.001         # size of each weight change
    batch_size = 64               # pictures per batch
    n_train = 10000               # training pictures (at most 60000)
    version_name = "dense " + str(hidden_sizes)
    """),
    md(r"""
    **Expected time: a few seconds** with the settings above (bigger layers and more pictures take longer: the
    cost report tells you).
    """),
    code("""
    train_loader, val_loader, classes = P.get_mnist(n_train=n_train, n_val=2000, batch_size=batch_size)
    torch.manual_seed(0)
    dense_try = build_dense(hidden_sizes)
    P.show_model(dense_try, (1, 28, 28), title=version_name)
    P.cost(dense_try, (1, 28, 28), name=version_name, epoch_images=n_train)
    history = P.train(dense_try, train_loader, val_loader, epochs=epochs, lr=learning_rate, name=version_name)
    P.remember(version_name, dense_try, history)
    """),
    md(r"""
    ### 5.2 · The convolutional network

    `build_cnn(...)` builds a CNN from six settings. It adds one convolution (+ ReLU, + pooling if switched on)
    per entry of `conv_channels`, then flatten and two linear layers. The one tricky number is the
    `in_features` of the first linear layer: it depends on all the convolution settings. Instead of computing it
    by hand, the function pushes one black dummy picture (`torch.zeros(1, 1, 28, 28)`: a batch of 1 picture,
    all zeros) through the convolution part and counts the numbers that come out with `.numel()`.
    """),
    api("torch.zeros", "Tensor.numel"),
    code("""
    def build_cnn(conv_channels, kernel_size, padding, stride, use_pooling, dense_size):
        layers = []
        in_channels = 1                                          # MNIST pictures have 1 (grey) channel
        for out_channels in conv_channels:                       # one convolution per entry of the list
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding))
            layers.append(nn.ReLU())
            if use_pooling:
                layers.append(nn.MaxPool2d(2))                   # halve height and width
            in_channels = out_channels                           # the next convolution receives these maps

        # How many numbers come out of the convolution part for one picture? Push a dummy picture through it.
        conv_part = nn.Sequential(*layers)
        dummy_picture = torch.zeros(1, 1, 28, 28)                # a batch of 1 black picture
        n_features = conv_part(dummy_picture).numel()            # maps x height x width

        layers.append(nn.Flatten())
        layers.append(nn.Linear(n_features, dense_size))         # in_features = what the convolutions produce
        layers.append(nn.ReLU())
        layers.append(nn.Linear(dense_size, 10))                 # 10 scores
        return nn.Sequential(*layers)
    """),
    md(r"""
    **Try it.** Change the numbers, then run this cell and the next one. If you make the picture shrink to
    nothing (for example many layers without padding), the next cell stops with "Output size is too small" or a
    message from `P` saying which layer has a picture smaller than its filter: add padding or remove pooling.
    """),
    code("""
    # ---------------- Try it: settings of the convolutional network ----------------
    conv_channels = [16, 32]      # feature maps of each convolution layer; [16, 32] = two layers
    kernel_size = 3               # filter size: 3 means 3 x 3 pixels
    padding = 1                   # rows / columns of zeros around the picture before each convolution
    stride = 1                    # step of the filter: 1 = every position, 2 = every second one
    use_pooling = True            # MaxPool 2 x 2 after each convolution (halves height and width)
    dense_size = 64               # neurons of the hidden fully connected layer
    # ---------------- run settings ----------------
    epochs = 5
    learning_rate = 0.001
    batch_size = 64
    n_train = 10000
    version_name = "cnn " + str(conv_channels)
    """),
    md(r"""
    **Expected time: about 8 seconds** with the settings above (up to 30 seconds on a slow laptop; more or
    bigger layers, and more pictures, take longer).
    """),
    code("""
    train_loader, val_loader, classes = P.get_mnist(n_train=n_train, n_val=2000, batch_size=batch_size)
    torch.manual_seed(0)
    cnn_try = build_cnn(conv_channels, kernel_size, padding, stride, use_pooling, dense_size)
    P.show_model(cnn_try, (1, 28, 28), title=version_name)
    P.cost(cnn_try, (1, 28, 28), name=version_name, epoch_images=n_train)
    history = P.train(cnn_try, train_loader, val_loader, epochs=epochs, lr=learning_rate, name=version_name)
    P.remember(version_name, cnn_try, history)
    """),
    md(r"""
    ### 5.3 · The leaderboard

    Run this cell after every experiment: every version so far, best validation accuracy first.
    """),
    code("""
    P.leaderboard()
    """),
    md(r"""
    ### 5.4 · Guided experiments

    Do them in any order. For each one: change the setting in a **Try it** cell, **predict** what will happen
    to the accuracy, the parameters and the time, run the two cells, and compare with your prediction. Then
    open the fold-out to read what we saw (5 epochs, 10 000 training pictures; your numbers can differ in the
    last digit). Put the setting back afterwards, or keep going from there.

    **1. Fewer feature maps.** CNN: `conv_channels = [4, 8]`.
    <details><summary>What we saw</summary>

    97.7% becomes 95.8%, with 26 138 parameters instead of 105 866 and half the time per epoch. Compare it
    with a fully connected network of the same size, `hidden_sizes = [32]` (25 450 parameters): 93.1%. With the
    same number of parameters, the convolutional network is clearly better.
    </details>

    **2. More feature maps.** CNN: `conv_channels = [32, 64]`.
    <details><summary>What we saw</summary>

    97.4%: no better than [16, 32] (the difference of a few pictures is chance), with twice the parameters
    (220 234) and about twice the time per epoch. More is not always better: for these small digits 16 and 32
    maps are already enough.
    </details>

    **3. No padding.** CNN: `padding = 0`.
    <details><summary>What we saw</summary>

    The picture now shrinks in every convolution: 28 -> 26 -> 13 -> 11 -> 5, so the convolution part ends at
    32 x 5 x 5 = 800 numbers instead of 1 568. The first linear layer halves, and with it the whole network:
    56 714 parameters. Accuracy 97.7%, the same as before: the lost border pixels of MNIST pictures are empty
    anyway.
    </details>

    **4. Stride 2 instead of pooling.** CNN: `stride = 2` and `use_pooling = False`.
    <details><summary>What we saw</summary>

    The filters now step over every second position, so each convolution halves the picture by itself
    (28 -> 14 -> 7), and the final shape is the same 32 x 7 x 7 as with pooling: exactly 105 866 parameters
    again. But each convolution computes only a quarter of the positions: 0.36 million multiply-adds instead of
    1.12 million, and about half the time per epoch. The price: 96.8% instead of 97.7%, since pooling chose the
    strongest of four positions while the stride skips three of them.
    </details>

    **5. A bigger filter.** CNN: `kernel_size = 5` and `padding = 2` (padding 2 keeps the size for a 5 x 5 filter).
    <details><summary>What we saw</summary>

    97.8%, the same as with 3 x 3 filters, but each filter now has 25 weights instead of 9: 2.9 million
    multiply-adds instead of 1.1 million, and slower. Most modern networks use 3 x 3 filters and more layers
    instead of bigger filters.
    </details>

    **6. A third convolution layer.** CNN: `conv_channels = [16, 32, 64]`.
    <details><summary>What we saw</summary>

    With pooling after each layer the picture goes 28 -> 14 -> 7 -> 3, so only 64 x 3 x 3 = 576 numbers reach the
    linear layer: 60 874 parameters, fewer than with two layers although we added one. Accuracy 98.0% at its
    best epoch, the best of these experiments. Deeper networks combine patterns into more complex ones.
    </details>

    **7. A huge fully connected layer.** CNN: `dense_size = 1024`.
    <details><summary>What we saw</summary>

    1 621 706 parameters, 15 times as many, almost all in the first linear layer (1 568 x 1 024 weights).
    Accuracy 97.9%: hardly better. The training accuracy reaches 99.1% while the validation accuracy stays near
    97.7%: the network starts to memorise the training pictures rather than learn more about digits.
    </details>

    **8. Learning rate too high, and too low.** CNN: `learning_rate = 0.1`, then `learning_rate = 0.00001`.
    <details><summary>What we saw</summary>

    With 0.1 the accuracy stays around 10%: one digit in ten, as good as guessing. The weight changes are so big
    that every step throws away what the last one learned. With 0.00001 the network learns, but very slowly:
    35.5% after one epoch, 77.7% after five. The default 0.001 sits in between.
    </details>

    **9. Fewer or more epochs.** CNN: `epochs = 1`, then `epochs = 10`.
    <details><summary>What we saw</summary>

    One epoch: 94.8%. Ten epochs: 97.8% at best, hardly more than after five (97.7%), while the training
    accuracy keeps climbing to 99.5%. After a few epochs, training longer mostly teaches the network the
    training pictures by heart.
    </details>

    **10. Less or more data.** CNN: `n_train = 1000`, then `n_train = 60000` (all of MNIST; about 40 seconds
    here, a few minutes on a slow laptop).
    <details><summary>What we saw</summary>

    With 1 000 training pictures: 90.6%. With all 60 000: 99.0%. Nothing else we changed helped as much as
    more data, and it costs only time: 6 times the pictures, 6 times the seconds per epoch.
    </details>
    """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · Why convolution? A test with shuffled pixels (optional)

    Here is a way to see what the convolutional network relies on. `P.get_mnist(shuffle_pixels=True)` moves
    every pixel of every picture to another place, always the same place for the same pixel. To us the digits
    become noise, but the information is all still there: a pixel that was the top-left corner is now always at
    the same new position.

    **Predict first:** which network suffers? The fully connected one sees 784 separate numbers and never knew
    which ones were neighbours. The convolutional one looks at neighbouring pixels through its filters...
    """),
    code("""
    shuffled_train, shuffled_val, classes = P.get_mnist(n_train=10000, n_val=2000, batch_size=64, shuffle_pixels=True)
    P.show_samples(shuffled_train, classes, title="the same kind of pictures, every pixel moved to a fixed new place")
    """),
    md(r"""
    **Expected time: about 10 seconds** for both trainings (up to 40 seconds on a slow laptop). `live=False`
    draws the curves once at the end instead of after every epoch.
    """),
    code("""
    torch.manual_seed(0)
    dense_shuffled = build_dense([128])
    history_dense_shuffled = P.train(dense_shuffled, shuffled_train, shuffled_val, epochs=5, live=False,
                                     name="A on shuffled pixels")
    torch.manual_seed(0)
    cnn_shuffled = build_cnn([16, 32], 3, 1, 1, True, 64)
    history_cnn_shuffled = P.train(cnn_shuffled, shuffled_train, shuffled_val, epochs=5, live=False,
                                   name="B on shuffled pixels")
    """),
    md(r"""
    **What we saw.** Network A does not care: 94.7% on shuffled pixels against 94.8% on normal ones. For it the
    order of the 784 numbers never meant anything. Network B drops from 97.7% to 93.4%, now *below* network A:
    its filters look for strokes and edges made of neighbouring pixels, and after shuffling, neighbouring pixels
    have nothing to do with each other. That is what convolution is for: it builds in the knowledge that in a
    picture, nearby pixels belong together.
    """),

    # ------------------------------------------------------------------ 7
    md(r"""
    ## 7 · What to remember

    * A **fully connected** network flattens the picture and connects every pixel to every neuron; it ignores
      which pixels are neighbours.
    * A **convolutional** network slides small filters over the picture and reuses the same few weights
      everywhere; it keeps the 2-D layout and finds a pattern wherever it appears.
    * With the same number of parameters (about 100 000), the CNN made fewer than half the mistakes (97.7% against
      94.8%), but needed about 11 times the computation.
    * Each convolution's output size follows (n + 2 x padding - kernel) / stride + 1; pooling halves it. The
      `in_features` of the first linear layer must equal maps x height x width after the last convolution.
    * The settings matter in different ways: the learning rate can make or break training; more data helped
      most; extra parameters (wide layers, many maps) helped little.

    ### Words from this notebook

    | word | meaning |
    |---|---|
    | accuracy | percentage of pictures whose highest score is the right class |
    | activation | a function applied to every number between layers; here ReLU: negative -> 0 |
    | Adam | the method (optimiser) that changes the weights after each batch |
    | batch | a small group of pictures processed together (here 64) |
    | bias | a number a neuron adds to its weighted sum; also learned |
    | channel | one of the maps a picture or layer output consists of (1 for grey, 3 for colour, 16 after a convolution with 16 filters) |
    | class | one of the possible answers (here the digits 0 to 9) |
    | classification | choosing one class for each input |
    | convolution | sliding a small filter over a picture, computing a weighted sum at every position |
    | edge | a place where the picture changes from dark to bright |
    | epoch | one pass over all training pictures |
    | feature map | the output of one filter: bright where its pattern appears |
    | filter (kernel) | the small grid of weights of a convolution |
    | flatten | laying out a picture or maps as one long row of numbers |
    | fully connected (dense, linear) layer | neurons that each see every input number |
    | label | the correct answer for a picture |
    | layer | one building block of a network |
    | learning rate | how big each change of the weights is |
    | loss | how wrong the scores are (lower is better) |
    | multiply-adds | the multiplications and additions a network needs for one picture: a measure of its work |
    | neuron | computes a weighted sum of its inputs plus a bias |
    | normalised | shifted and scaled to average 0 with a spread of 1 |
    | padding | zeros added around a picture before a convolution |
    | parameters (weights) | the numbers a network learns |
    | pixel | one small square of a picture |
    | pooling | shrinking feature maps, here by keeping the largest of each 2 x 2 square |
    | random seed | the starting point of the random numbers, so results repeat |
    | score | one output number per class; the highest is the answer |
    | stride | the step of a filter as it slides |
    | tensor | PyTorch's multi-dimensional array of numbers |
    | training / validation set | pictures the network learns from / pictures kept apart to measure it |
    | training step | one batch: scores, loss, weight changes |
    """),
]
