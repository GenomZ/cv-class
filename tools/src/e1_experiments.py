from nb import md, code, api

NAME = "E1-flatten-branches-and-sobel-filters"

CELLS = [
    md(r"""
    # E1 · Bonus experiments: many ways to flatten, and filters we choose ourselves

    This is a **bonus** notebook: two experiments that start from P1 (fully connected against convolutional
    network on MNIST) and each test one idea. Nobody knows the answer of an experiment before running it,
    and that is the point: we write the idea down as a **hypothesis** (a claim that the experiment can confirm or
    refute), build what is needed to test it, measure, and then read the numbers honestly.

    **Experiment 1: many ways to flatten.** P1's fully connected network flattens the picture in one way, pixel
    by pixel. We build a fully connected network that flattens the same picture in several ways (first
    summarising squares of 2 x 2, 4 x 4 and 7 x 7 pixels with a kernel, then flattening), gives every way its
    own input layer, and glues the results together before it answers.
    *Hypothesis: every way of flattening gives the classifier additional information, so the network gets better.*

    **Experiment 2: filters we choose ourselves.** A CNN's filters start as random numbers. We take two CNNs
    that are the same in everything, and in one of them replace the random first-layer filters with classic
    hand-made edge filters (Sobel x, Sobel y and their relatives).
    *Hypothesis: a network that starts with sensible filters learns faster and ends better.*

    **How to use this notebook.** As in P1 you do not write code: run the cells from top to bottom with
    **Shift+Enter**, read what comes out, and change the numbers in the cells marked **Try it**. Run P1 first:
    the words explained there (layer, flatten, filter, feature map, epoch, accuracy, ...) are used here without
    being explained again. The whole notebook runs in about 5 to 7 minutes on a laptop; every training cell says
    how long it takes.
    """),

    # ------------------------------------------------------------------ 0
    md(r"""
    ## 0 · Setup and data

    The same first cell as in P1, with one addition: `matplotlib.pyplot` (short `plt`), the library that draws
    figures; we use it twice for small pictures of our own. Then the same MNIST pictures as in P1: 10 000 for
    training, 2 000 for validation, in batches of 64.
    """),
    code("""
    import sys
    import pathlib

    folder = pathlib.Path.cwd()
    while not (folder / "cvcourse").is_dir() and folder != folder.parent:
        folder = folder.parent                      # go up until the folder that contains cvcourse
    sys.path.insert(0, str(folder))

    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from cvcourse import practical as P

    P.setup(seed=0)
    train_loader, val_loader, classes = P.get_mnist(n_train=10000, n_val=2000, batch_size=64)
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · Experiment 1: many ways to flatten

    ### 1.1 · The idea

    Network A of P1 was: flatten (28 x 28 picture -> 784 numbers), a linear layer with 128 neurons, ReLU, a linear
    layer with 10 scores. Its flatten is the simplest one possible: every pixel becomes one number.

    Here is another way to turn a picture into a row of numbers: first lay a **kernel** of `k x k` pixels on
    the picture, without overlap, and replace every `k x k` square by **one** number that summarises it; then
    flatten what is left. With the average as the summary this is **average pooling** (`nn.AvgPool2d(k)`); with
    the largest value it is the max pooling of P1 (`nn.MaxPool2d(k)`).

    | kernel | what one number means | picture after the kernel | numbers after flatten |
    |---|---|---|---|
    | 1 x 1 (no kernel) | one pixel | 28 x 28 | 784 |
    | 2 x 2 | the average of 4 pixels | 14 x 14 | 196 |
    | 4 x 4 | the average of 16 pixels | 7 x 7 | 49 |
    | 7 x 7 | the average of 49 pixels | 4 x 4 | 16 |

    (28 divides by 1, 2, 4, 7, 14 and 28, so these kernels fit the picture exactly.)

    A hand-worked example of average pooling with a 2 x 2 kernel on a 4 x 4 picture:

    ```
    picture                 2 x 2 squares             result
     0  2 | 4  4
     2  4 | 4  4            (0+2+2+4)/4 = 2           2  4
    ------+------           (4+4+4+4)/4 = 4
     1  1 | 0  8            (1+1+1+1)/4 = 1           1  2
     1  1 | 0  0            (0+8+0+0)/4 = 2
    ```

    Flattening the result gives the row `2, 4, 1, 2`: 4 numbers instead of 16. Each of them describes a whole
    region instead of one pixel: a **coarser** view of the same picture.

    The next cell shows one validation digit through the four kernels. `val_loader.dataset[0]` is the first
    validation picture with its label; the picture has the shape `[1, 28, 28]` (1 channel, 28 rows, 28 columns).
    """),
    api("nn.AvgPool2d"),
    api("nn.MaxPool2d", "Tensor.shape", brief=True),
    code("""
    image, label = val_loader.dataset[0]                 # one picture: shape [1, 28, 28]
    kernel_sizes = [1, 2, 4, 7]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), layout="constrained")
    for position in range(4):
        k = kernel_sizes[position]
        if k == 1:
            view = image                                 # kernel 1 x 1: the picture itself
        else:
            view = nn.AvgPool2d(k)(image)                # every k x k square -> its average
        side = view.shape[1]                             # height (= width) after the kernel
        axes[position].imshow(view[0], cmap="gray")
        axes[position].set_title(f"kernel {k} x {k}: {side * side} numbers")
        axes[position].axis("off")
    plt.show()
    """),
    md(r"""
    **What to look at.** With the 2 x 2 kernel the digit is still easy to read. With 4 x 4 a person can still
    guess it. With 7 x 7 only 16 numbers are left: they say roughly where the ink is, no more.

    ### 1.2 · The network

    The network of this experiment has one **branch** per way of flattening. A branch is a small chain of layers:

    1. the kernel (`nn.AvgPool2d(k)` or `nn.MaxPool2d(k)`; nothing for `k = 1`),
    2. `nn.Flatten()`,
    3. its own **input layer** `nn.Linear(numbers after flatten, branch_size)`,
    4. `nn.ReLU()`.

    Every branch receives the **same** picture and produces `branch_size` numbers. `torch.cat` then glues the
    outputs of all branches into one long row (**concatenation**, as in the Inception block of P3), and one last
    linear layer, the **classifier**, turns that row into the 10 scores:

    ```
                  +-- (kernel 1) ---------- flatten 784 -- Linear(784, 64) -- ReLU --+
                  |                                                                  |
    picture  -----+-- AvgPool 2 x 2 ------- flatten 196 -- Linear(196, 64) -- ReLU --+-- concatenate -- Linear(256, 10) -- scores
    1 x 28 x 28   |                                                                  |   4 x 64 = 256
                  +-- AvgPool 4 x 4 ------- flatten  49 -- Linear( 49, 64) -- ReLU --+
                  |                                                                  |
                  +-- AvgPool 7 x 7 ------- flatten  16 -- Linear( 16, 64) -- ReLU --+
    ```

    It is still a fully connected network: the kernels are fixed (pooling has no parameters), and everything
    that learns is a linear layer.

    **How the class below is written.** A network with branches cannot be an `nn.Sequential` (that is one
    straight chain), so we write a class built on `nn.Module`, as in P3:

    * `__init__` creates the layers. It runs once, when we write `MultiFlattenNet(...)`. It loops over
      `kernel_sizes` and builds one branch per entry. The branches are kept in an `nn.ModuleList`: a list for
      layers, which PyTorch needs in order to find their parameters (a plain Python list would hide them from
      training).
    * `forward` says how a batch `x` flows through the layers. It runs every time the network is used. It sends
      the same `x` through every branch, collects the outputs in a list, concatenates them along `dim=1` and
      applies the classifier.

    Shapes in `forward` for a batch of 64 pictures and the four kernels above with `branch_size = 64`:

    | step | shape |
    |---|---|
    | `x` | `[64, 1, 28, 28]` |
    | output of each branch | `[64, 64]` (64 pictures, 64 numbers each) |
    | `combined = torch.cat(branch_outputs, dim=1)` | `[64, 256]` (axis 0, the pictures, stays; axis 1 grows: 4 x 64) |
    | `scores = self.classifier(combined)` | `[64, 10]` |
    """),
    api("nn.ModuleList"),
    api("nn.Module", "nn.Sequential", "Module.add_module", "nn.Flatten", "nn.Linear", "nn.ReLU", "torch.cat", brief=True),
    code("""
    class MultiFlattenNet(nn.Module):
        def __init__(self, kernel_sizes, branch_size, pooling):
            super().__init__()
            self.branches = nn.ModuleList()                  # one branch per way of flattening
            for k in kernel_sizes:
                branch = nn.Sequential()
                # 1. the kernel: summarise every k x k square by one number (nothing to do for k = 1)
                if k > 1:
                    if pooling == "avg":
                        branch.add_module("kernel", nn.AvgPool2d(k))     # the average of the square
                    else:
                        branch.add_module("kernel", nn.MaxPool2d(k))     # the largest value of the square
                # 2. flatten what is left into one row of numbers
                branch.add_module("flatten", nn.Flatten())
                side = 28 // k                               # height and width after the kernel
                n_numbers = side * side                      # how many numbers the flatten produces
                # 3. this branch's own input layer, and 4. its ReLU
                branch.add_module("input_layer", nn.Linear(n_numbers, branch_size))
                branch.add_module("relu", nn.ReLU())
                self.branches.append(branch)
            n_combined = len(kernel_sizes) * branch_size     # all branch outputs glued together
            self.classifier = nn.Linear(n_combined, 10)      # 10 scores, one per digit

        def forward(self, x):
            branch_outputs = []
            for branch in self.branches:
                branch_output = branch(x)                    # the same picture goes into every branch
                branch_outputs.append(branch_output)
            combined = torch.cat(branch_outputs, dim=1)      # glue the rows of numbers side by side
            scores = self.classifier(combined)
            return scores
    """),
    md(r"""
    Build it with the four kernels and 64 neurons per branch, and look at it. In the diagram the picture splits
    into four branches that meet again in the red "concatenate" box.
    """),
    api("torch.manual_seed", brief=True),
    code("""
    torch.manual_seed(0)
    multi_net = MultiFlattenNet(kernel_sizes=[1, 2, 4, 7], branch_size=64, pooling="avg")
    P.show_model(multi_net, (1, 28, 28), title="Experiment 1: four ways to flatten, concatenated")
    """),
    code("""
    P.summary(multi_net, (1, 28, 28))
    """),
    md(r"""
    **Counting parameters.** The four input layers have (784 + 196 + 49 + 16) x 64 weights + 4 x 64 biases
    = 67 136, the classifier 256 x 10 + 10 = 2 570: together **69 706**. Almost all of them (50 240) sit in the
    branch without a kernel; the three extra branches together cost only 16 896.

    ### 1.3 · A fair comparison

    To say "the branches help" we need something to compare with: a **baseline**. It must be fair. If the new
    network were simply bigger than the baseline, we could not tell whether the branches helped or the extra
    parameters did. So the baseline is P1's network A with a hidden layer chosen to give the same number of
    parameters: 88 neurons give 784 x 88 + 88 + 88 x 10 + 10 = **69 970**.

    `build_dense` is the builder function of P1 (section 5.1), unchanged.
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


    torch.manual_seed(0)
    baseline_net = build_dense([88])
    P.compare({"baseline: one flatten [88]": baseline_net, "four ways to flatten": multi_net}, (1, 28, 28))
    """),
    md(r"""
    Same parameters, same memory. Now train both in the same way: 5 epochs, learning rate 0.001.

    **Expected time: about 10 seconds** for both (up to 30 seconds on a slow laptop).
    """),
    code("""
    history_baseline = P.train(baseline_net, train_loader, val_loader, epochs=5, lr=0.001,
                               name="baseline: one flatten [88]")
    history_multi = P.train(multi_net, train_loader, val_loader, epochs=5, lr=0.001,
                            name="four ways to flatten")
    """),
    code("""
    P.plot_history({"baseline: one flatten [88]": history_baseline, "four ways to flatten": history_multi})
    """),
    md(r"""
    **Reading the result.** The baseline ends at 94.4%, the network with four ways to flatten at 94.7%. On 2 000
    validation pictures that is 6 pictures more. The curves lie almost on top of each other. One more thing from
    the cost table above: both networks have the same parameters and the same multiply-adds, but a training step
    of the branched network takes about twice as long, because the computer runs four small chains one after the
    other instead of one big layer.

    ### 1.4 · One run is not enough

    The two networks differ by a few validation pictures. Is that the branches, or is it chance? Every run
    depends on random numbers: the starting weights and the order of the training pictures. With another random
    seed the same network ends a little higher or lower. Before believing a small difference, repeat the run
    with several seeds and look at the **average** and at how far the runs are **spread**. A difference that is
    smaller than the spread between seeds means nothing.

    `repeat_runs` does that for any network. It takes a **build function** (a function without arguments that
    returns a new, untrained network), trains one network per seed and prints the best validation accuracy of
    each run, their average and their spread. `P.train(..., verbose=False)` trains silently: no lines, no
    curves; the result is read from the history it returns (`history["best_val_acc"]`).
    """),
    code("""
    def repeat_runs(name, build_function, train_loader, val_loader, seeds, epochs):
        accuracies = []
        text = ""
        for seed in seeds:
            torch.manual_seed(seed)                      # other starting weights for every seed
            model = build_function()
            history = P.train(model, train_loader, val_loader, epochs=epochs, lr=0.001, seed=seed, verbose=False)
            accuracies.append(history["best_val_acc"])
            text = text + f"{history['best_val_acc']:6.2f}"
        average = sum(accuracies) / len(accuracies)
        spread = max(accuracies) - min(accuracies)       # distance between the best and the worst run
        print(f"{name:<34} runs:{text}   average {average:.2f}%   spread {spread:.2f}")
    """),
    md(r"""
    Four networks, three seeds each:

    * the baseline with one flatten, [88];
    * four ways to flatten with **average** kernels;
    * four ways to flatten with **max** kernels;
    * a bigger plain network, [128] (network A of P1, 101 770 parameters), to see what simply adding
      parameters buys.

    **Expected time: about 40 seconds** (up to 2 minutes on a slow laptop).
    """),
    code("""
    def build_baseline():
        return build_dense([88])

    def build_multi_avg():
        return MultiFlattenNet(kernel_sizes=[1, 2, 4, 7], branch_size=64, pooling="avg")

    def build_multi_max():
        return MultiFlattenNet(kernel_sizes=[1, 2, 4, 7], branch_size=64, pooling="max")

    def build_bigger():
        return build_dense([128])

    seeds = [0, 1, 2]
    repeat_runs("baseline: one flatten [88]", build_baseline, train_loader, val_loader, seeds, epochs=5)
    repeat_runs("four ways to flatten, average", build_multi_avg, train_loader, val_loader, seeds, epochs=5)
    repeat_runs("four ways to flatten, max", build_multi_max, train_loader, val_loader, seeds, epochs=5)
    repeat_runs("bigger plain network [128]", build_bigger, train_loader, val_loader, seeds, epochs=5)
    """),
    md(r"""
    **What we saw** (your last digits can differ):

    | network | parameters | three runs | average | spread |
    |---|---|---|---|---|
    | baseline: one flatten [88] | 69 970 | 94.40, 94.80, 94.10 | 94.43% | 0.70 |
    | four ways to flatten, average kernels | 69 706 | 94.65, 94.50, 94.55 | 94.57% | 0.15 |
    | four ways to flatten, max kernels | 69 706 | 94.95, 94.80, 95.25 | 95.00% | 0.45 |
    | bigger plain network [128] | 101 770 | 94.80, 94.60, 94.60 | 94.67% | 0.20 |

    **Reading the table.**

    * **Average kernels: no measurable gain.** 94.57% against 94.43% is a difference of 0.14 points, while the
      baseline alone moves by 0.70 points from one seed to the next. The 6 pictures of section 1.3 were chance:
      with seed 1 the baseline is the better one.
    * **Max kernels: a small gain.** 95.00% is 0.57 points above the baseline, and two of the three runs are above
      the baseline's best run. With the same number of parameters it also beats the bigger plain network [128],
      which has 46% more parameters. Three runs are still few: read this as "probably a little better", not as
      a fact.
    * **Nothing here comes close to the convolutional network** of P1 (97.7% with about 106 000 parameters).

    So the hypothesis "every way of flattening gives additional information" is **not confirmed** for average
    kernels and only **weakly** for max kernels. The next section explains why that is what we should expect.

    ### 1.5 · Why the extra views add so little

    The hypothesis said that every way of flattening gives the classifier **additional information**. Look at
    what an average-kernel branch receives: each of its numbers is the average of some pixels, so it is a
    weighted sum of pixels. But a weighted sum of pixels is exactly what a neuron of the first branch already
    computes, and that neuron may choose its weights freely. The plain flatten branch sees all 784 pixels and
    could work out every one of those averages by itself. The coarse views contain **nothing that the full view
    does not contain**: they are the same information, with less detail.

    What a coarse branch does give is a different **starting point**. A neuron fed with 7 x 7 averages can only
    learn rough "where is the ink" patterns, and it learns them with few weights; a neuron on raw pixels must
    discover such patterns among 784 separate numbers. That can help a little, and it costs little, but it is
    not new information.

    A max kernel is different in kind: the largest value of a square is **not** a weighted sum of its pixels,
    so no single linear layer can compute it. A max branch does hand the classifier something the first input
    layer cannot produce by itself. The repeated runs show how much that is worth here.

    Compare this with what really helped in P1. The convolutional network, with about the same number of
    parameters as network A, reached 97.7%: its filters also look at small squares, but the filters are
    **learned**, they **overlap**, and they are stacked in layers. Fixed summaries of squares are a weak
    relative of that idea.

    ### 1.6 · Try it

    Change the settings, run this cell and the next one, then the leaderboard. Each experiment in the list is a
    question: predict the answer first.

    1. **One coarse view alone**: `kernel_sizes = [2]`, then `[4]`, then `[7]`. How much accuracy is left when
       the network only sees 196, 49 or 16 numbers?
    2. **Max instead of average** for the same lists: `pooling = "max"`. Which summary keeps more of the digit
       when the kernel is large?
    3. **Without the full view**: `kernel_sizes = [2, 4, 7]`. Can three coarse views together replace the pixels?
    4. **Wider branches**: `branch_size = 128`, then `256`.
    5. **The same view four times**: `kernel_sizes = [1, 1, 1, 1]`. If this does as well as `[1, 2, 4, 7]`, the
       gain came from the number of neurons, not from the different views.
    """),
    code("""
    # ---------------- Try it: settings of the network with several ways to flatten ----------------
    kernel_sizes = [1, 2, 4, 7]   # one branch per entry; allowed sizes: 1, 2, 4, 7, 14, 28
    branch_size = 64              # neurons of every branch's input layer
    pooling = "avg"               # the summary of each square: "avg" (average) or "max" (largest value)
    # ---------------- run settings ----------------
    epochs = 5
    learning_rate = 0.001
    version_name = "flatten " + str(kernel_sizes) + " x " + str(branch_size) + " " + pooling
    """),
    md(r"""
    **Expected time: a few seconds** with the settings above.
    """),
    code("""
    torch.manual_seed(0)
    multi_try = MultiFlattenNet(kernel_sizes, branch_size, pooling)
    P.show_model(multi_try, (1, 28, 28), title=version_name)
    history = P.train(multi_try, train_loader, val_loader, epochs=epochs, lr=learning_rate, name=version_name)
    P.remember(version_name, multi_try, history)
    """),
    code("""
    P.remember("baseline: one flatten [88]", baseline_net, history_baseline, verbose=False)
    P.remember("four ways to flatten (section 1.3)", multi_net, history_multi, verbose=False)
    P.leaderboard()
    """),
    md(r"""
    <details><summary><b>What we saw</b> (open after you have tried)</summary>

    One run each, seed 0, 5 epochs, 64 neurons per branch unless said otherwise.

    | setting | parameters | best validation accuracy |
    |---|---|---|
    | `[2]` average / max | 13 258 | 92.6% / 92.1% |
    | `[4]` average / max | 3 850 | 89.4% / 84.5% |
    | `[7]` average / max | 1 738 | 72.8% / 57.5% |
    | `[2, 4, 7]` average / max | 18 826 | 93.3% / 92.6% |
    | `[1, 2, 4, 7]` average, `branch_size = 128` | 139 402 | 95.3% |
    | `[1, 2, 4, 7]` average, `branch_size = 256` | 278 794 | 95.9% |
    | `[1, 1, 1, 1]` average | 203 530 | 95.8% |

    1. A 2 x 2 kernel costs little: 92.6% from a quarter of the numbers and 13 258 parameters. Even the 16 numbers
       of the 7 x 7 kernel give 72.8%: knowing roughly where the ink is already tells a lot about the digit.
    2. Alone, max is **worse** than average, and more so for large kernels (57.5% against 72.8% at 7 x 7): the
       largest value of a 7 x 7 square is almost always "there is ink here", the same for every digit, while the
       average still says how much ink. Max kernels only helped as an addition to the full view.
    3. Three coarse views together (93.3%) are better than any one of them and come within about 1.4 points of
       the network that also sees the pixels, with a quarter of its parameters.
    4. Wider branches help steadily: 94.7%, 95.3%, 95.9%.
    5. The same plain view four times reaches 95.8%, more than `[1, 2, 4, 7]` (94.7%). This network is P1's network
       A with 256 hidden neurons, written in a complicated way, and it has three times the parameters. It shows
       what moves the accuracy of a fully connected network: the number of neurons that see all the pixels, not the
       variety of the views.
    </details>
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · Experiment 2: filters we choose ourselves

    ### 2.1 · The idea

    In P1 we looked at the 16 filters of the first convolution layer **after** training and found edge
    detectors among them: red on one side, blue on the other. Before training the same filters were random
    numbers; training had to find the edges. People designed edge filters by hand long before neural networks.
    The best known are the two **Sobel** filters (Irwin Sobel, 1968):

    ```
    Sobel x                 Sobel y
    -1   0   1              -1  -2  -1
    -2   0   2               0   0   0
    -1   0   1               1   2   1
    ```

    **Sobel x** subtracts the pixels on its left from the pixels on its right. On a flat region (all pixels
    equal) the result is 0. Where the picture gets brighter from left to right, the result is large and
    positive: it finds **vertical edges**. **Sobel y** does the same from top to bottom and finds **horizontal
    edges**.

    A hand-worked example: Sobel x on two 3 x 3 patches of a picture.

    ```
    flat patch          edge patch (dark on the left, bright on the right)
    5  5  5             0  0  9
    5  5  5             0  0  9
    5  5  5             0  0  9

    flat:  (-1)(5) + (0)(5) + (1)(5) + (-2)(5) + (0)(5) + (2)(5) + (-1)(5) + (0)(5) + (1)(5) =  0
    edge:  (-1)(0) + (0)(0) + (1)(9) + (-2)(0) + (0)(0) + (2)(9) + (-1)(0) + (0)(0) + (1)(9) = 36
    ```

    One detail matters in a network: after the convolution comes a ReLU, which turns negative numbers into 0.
    Sobel x gives a positive number for dark -> bright and a **negative** one for bright -> dark, and the ReLU
    would throw the second kind away. So we also use the filter with all signs flipped (`-sobel_x`), which finds
    bright -> dark edges. With the two diagonal directions that makes 8 filters:

    | filters | finds |
    |---|---|
    | Sobel x, minus Sobel x | vertical edges: dark -> bright, bright -> dark |
    | Sobel y, minus Sobel y | horizontal edges, both ways |
    | diagonal 1, minus diagonal 1 | edges along one diagonal, both ways |
    | diagonal 2, minus diagonal 2 | edges along the other diagonal, both ways |

    `torch.tensor(...)` makes a tensor from numbers we type; a minus sign in front of a tensor flips the sign
    of every number in it.
    """),
    api("torch.tensor", brief=True),
    code("""
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                            [-2.0, 0.0, 2.0],
                            [-1.0, 0.0, 1.0]])
    sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                            [ 0.0,  0.0,  0.0],
                            [ 1.0,  2.0,  1.0]])
    diagonal_1 = torch.tensor([[ 0.0,  1.0, 2.0],
                               [-1.0,  0.0, 1.0],
                               [-2.0, -1.0, 0.0]])
    diagonal_2 = torch.tensor([[-2.0, -1.0, 0.0],
                               [-1.0,  0.0, 1.0],
                               [ 0.0,  1.0, 2.0]])

    handmade_filters = [sobel_x, -sobel_x, sobel_y, -sobel_y, diagonal_1, -diagonal_1, diagonal_2, -diagonal_2]
    print("number of hand-made filters:", len(handmade_filters))
    """),
    md(r"""
    ### 2.2 · Two networks that differ in one thing only

    The network is network B of P1 with **8** filters in the first convolution (one per hand-made filter)
    instead of 16; the rest is unchanged:

    | layer | output shape |
    |---|---|
    | `nn.Conv2d(1, 8, kernel_size=3, padding=1)`, ReLU | 8 x 28 x 28 |
    | `nn.MaxPool2d(2)` | 8 x 14 x 14 |
    | `nn.Conv2d(8, 32, kernel_size=3, padding=1)`, ReLU | 32 x 14 x 14 |
    | `nn.MaxPool2d(2)` | 32 x 7 x 7 |
    | `nn.Flatten()` | 1568 |
    | `nn.Linear(1568, 64)`, ReLU | 64 |
    | `nn.Linear(64, 10)` | 10 |

    **Where the filters live.** The weights of a convolution layer are one tensor, `layer.weight`, of shape
    `[out_channels, in_channels, kernel height, kernel width]`: here `[8, 1, 3, 3]`, 8 filters, each with 1
    input channel of 3 x 3 weights. So `layer.weight[2, 0]` is the 3 x 3 grid of filter number 2, and assigning
    to it replaces that filter. `layer.bias` holds one bias per filter, shape `[8]`.

    `set_first_layer(model, filters, scale)` writes our filters into the first convolution, `model[0]` (the
    layers of an `nn.Sequential` are numbered from 0):

    * `with torch.no_grad():` tells PyTorch that the lines inside are not part of training: we set the weights
      by hand, and PyTorch must not try to compute corrections for that.
    * Each filter is multiplied by `scale`. PyTorch's random starting weights of this layer lie between -0.33
      and 0.33, while Sobel's numbers go up to 2. With `scale = 0.25` our filters go up to 0.5: about the size
      training expects. (Section 2.6 lets you try other scales.)
    * The bias of each filter is set to 0.
    """),
    api("nn.Conv2d", "nn.MaxPool2d", "torch.no_grad", brief=True),
    code("""
    def build_cnn():
        return nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),      # 8 filters of 3 x 3: 8 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 8 x 14 x 14
            nn.Conv2d(8, 32, kernel_size=3, padding=1),     # 32 x 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 32 x 7 x 7
            nn.Flatten(),                                   # 1568 numbers
            nn.Linear(1568, 64),
            nn.ReLU(),
            nn.Linear(64, 10),                              # 10 scores
        )


    def set_first_layer(model, filters, scale):
        first_conv = model[0]                               # the first convolution layer
        with torch.no_grad():                               # setting weights by hand is not training
            for number in range(len(filters)):
                first_conv.weight[number, 0] = filters[number] * scale     # replace filter `number`
                first_conv.bias[number] = 0.0                              # and set its bias to 0
    """),
    md(r"""
    Now the two networks. Both are built right after `torch.manual_seed(0)`, so they start with **exactly the
    same** random weights in every layer: they are twins. Then we overwrite the first layer of one twin. After
    that the two networks differ in the 8 first-layer filters and in nothing else, so any difference in
    training comes from those filters. Changing one thing at a time is what makes an experiment readable.

    The last two lines check the twins: `torch.equal(a, b)` is `True` when two tensors hold exactly the same
    numbers. Layer `3` is the second convolution.
    """),
    api("torch.equal"),
    code("""
    torch.manual_seed(0)
    random_cnn = build_cnn()                     # twin 1: all filters random

    torch.manual_seed(0)
    handmade_cnn = build_cnn()                   # twin 2: the same random weights ...
    set_first_layer(handmade_cnn, handmade_filters, scale=0.25)      # ... then our filters in the first layer

    print("first layer the same? ", torch.equal(random_cnn[0].weight, handmade_cnn[0].weight))
    print("second convolution the same?", torch.equal(random_cnn[3].weight, handmade_cnn[3].weight))
    """),
    md(r"""
    Look at the first-layer filters of both twins **before training** (red = positive weight, blue = negative).
    """),
    code("""
    P.show_filters(random_cnn, title="random start: the 8 first-layer filters before training")
    P.show_filters(handmade_cnn, title="hand-made start: Sobel x, -Sobel x, Sobel y, -Sobel y, diagonal 1, -diagonal 1, diagonal 2, -diagonal 2")
    """),
    md(r"""
    And what the filters make of a digit, still before any training. `P.show_feature_maps(model, image,
    layer="0")` shows the feature maps of layer `0`, the first convolution.
    """),
    code("""
    image, label = val_loader.dataset[0]
    P.show_feature_maps(random_cnn, image, layer="0", title="random filters: feature maps of the first layer, before training")
    P.show_feature_maps(handmade_cnn, image, layer="0", title="hand-made filters: feature maps of the first layer, before training")
    """),
    md(r"""
    **What to look at.** The hand-made maps are easy to read: each one lights up one side of the strokes (the
    left edges, the right edges, the upper edges, ...), and a filter and its minus twin light up opposite
    sides. The random maps are smeared copies of the digit, some bright on the background instead of on the
    strokes: whatever a random filter happens to compute. The network has to turn them into something useful
    first.

    ### 2.3 · Train both

    The same data, the same 5 epochs, the same learning rate, the same order of the pictures (`P.train` uses
    seed 0 for that in both runs).

    **Expected time: about 15 seconds** for both (up to 1 minute on a slow laptop).
    """),
    code("""
    history_random = P.train(random_cnn, train_loader, val_loader, epochs=5, lr=0.001, name="random start")
    history_handmade = P.train(handmade_cnn, train_loader, val_loader, epochs=5, lr=0.001, name="hand-made start")
    """),
    code("""
    P.plot_history({"random start": history_random, "hand-made start": history_handmade})
    """),
    md(r"""
    **Reading the curves.** The hand-made start is ahead in every epoch:

    | | after epoch 1 | after epoch 2 | best of 5 epochs |
    |---|---|---|---|
    | random start | 93.2% | 95.8% | 96.8% |
    | hand-made start | 95.5% | 97.2% | 97.8% |

    After two epochs the network with the hand-made filters is already better (97.2%) than the random twin after
    five (96.8%). At the end the twin makes 64 mistakes on the 2 000 validation pictures, the hand-made network 44.
    Remember that the two networks have the same layers, the same number of parameters and the same starting
    weights everywhere except in 8 filters of 9 weights each: 72 numbers out of 103 482.

    ### 2.4 · What happened to our filters?

    The hand-made filters were only the **starting point**: training was free to change them like any other
    weight. Did it?
    """),
    code("""
    P.show_filters(random_cnn, title="random start: the first-layer filters after training")
    P.show_filters(handmade_cnn, title="hand-made start: the first-layer filters after training")
    """),
    md(r"""
    **What to look at.** The hand-made filters are still easy to recognise after training: Sobel x is still blue
    on the left and red on the right. Training changed the numbers a little and left the patterns alone, which
    suggests that these filters were already close to what the network needs. The filters of the random twin have
    changed more, and a few of them now look like edge filters too (red on one side, blue on the other), but less
    cleanly: training was on its way to the kind of filter we handed to the other network for free.

    ### 2.5 · Frozen filters, less data, several seeds

    Three more questions.

    **What if the filters are not allowed to change?** A parameter with `requires_grad` set to `False` is
    **frozen**: training leaves it as it is (P5 freezes a whole pretrained network in this way). `build_frozen`
    below sets the hand-made filters and then freezes the first layer's weights and biases. The first layer is
    then classic image processing, and the network learns only what comes after it.

    **Does the starting point matter more when there is little data?** With 10 000 pictures the network has
    plenty of examples to find its own filters. With 1 000 it has 10 times fewer.

    **Is it the pattern of the filters, or only their size?** Our filters are larger numbers than PyTorch's
    random start: after scaling they have a typical size of 0.29, the random ones of 0.19. Larger starting
    weights alone send a stronger signal into the network and can speed up learning. To separate the two
    explanations we need a **control**: a network that shares the boring property (the size) but not the
    interesting one (the edge pattern). `build_random_same_size` fills the first layer with random numbers of
    typical size 0.29: `torch.randn(3, 3)` gives 3 x 3 random numbers of typical size 1, and
    `set_first_layer` multiplies them by 0.29.

    As in experiment 1, one run proves little, so we use `repeat_runs` (section 1.4) with three seeds. For every
    seed the four variants start from the same random weights outside the first layer.

    **Expected time: about 2 minutes** (up to 6 minutes on a slow laptop).
    """),
    api("Tensor.requires_grad_", "torch.randn", brief=True),
    code("""
    def build_random():
        return build_cnn()

    def build_handmade():
        model = build_cnn()
        set_first_layer(model, handmade_filters, scale=0.25)
        return model

    def build_frozen():
        model = build_cnn()
        set_first_layer(model, handmade_filters, scale=0.25)
        model[0].weight.requires_grad_(False)            # training must not change the filters ...
        model[0].bias.requires_grad_(False)              # ... nor their biases
        return model

    def build_random_same_size():
        model = build_cnn()
        noise_filters = []
        for number in range(8):
            noise_filters.append(torch.randn(3, 3))      # 3 x 3 random numbers, typical size 1
        set_first_layer(model, noise_filters, scale=0.29)
        return model


    seeds = [0, 1, 2]
    print("10 000 training pictures, 5 epochs")
    repeat_runs("random start", build_random, train_loader, val_loader, seeds, epochs=5)
    repeat_runs("hand-made start", build_handmade, train_loader, val_loader, seeds, epochs=5)
    repeat_runs("hand-made, frozen", build_frozen, train_loader, val_loader, seeds, epochs=5)
    repeat_runs("random, same size as hand-made", build_random_same_size, train_loader, val_loader, seeds, epochs=5)

    small_train, small_val, classes = P.get_mnist(n_train=1000, n_val=2000, batch_size=64, verbose=False)
    print()
    print("1 000 training pictures, 5 epochs")
    repeat_runs("random start", build_random, small_train, small_val, seeds, epochs=5)
    repeat_runs("hand-made start", build_handmade, small_train, small_val, seeds, epochs=5)
    repeat_runs("hand-made, frozen", build_frozen, small_train, small_val, seeds, epochs=5)
    repeat_runs("random, same size as hand-made", build_random_same_size, small_train, small_val, seeds, epochs=5)
    """),
    md(r"""
    **What we saw** (average of three runs, best validation accuracy; your last digits can differ):

    | start of the first layer | 10 000 training pictures | 1 000 training pictures |
    |---|---|---|
    | random (PyTorch's default) | 96.83% | 88.52% |
    | hand-made filters | 97.80% | 91.42% |
    | hand-made filters, frozen | 97.70% | 91.20% |
    | random numbers of the same size as the hand-made filters | 97.42% | 89.67% |

    **Reading the table.**

    * **The hand-made start is better, on every seed.** With 10 000 pictures: +1.0 points, and its worst run
      (97.75%) is above the random start's best run (97.20%). The hypothesis holds here.
    * **Less data, bigger advantage.** With 1 000 pictures the gain is +2.9 points (the runs are spread more
      widely there, by 2 to 3 points, yet every hand-made run is above every random run). Finding good filters
      takes examples; a network that receives them needs fewer. This is the idea of **transfer learning** in P5,
      where the filters come from a network trained on a million photos instead of from a textbook.
    * **Frozen is as good as trainable.** 97.70% against 97.80%, 91.20% against 91.42%: differences far inside the
      spread. Training had nothing important to add to these filters, as the pictures of section 2.4 suggested.
    * **Part of the gain is the size of the numbers, not their pattern.** The last row is the control: random
      filters, but as large as ours. It lands between the two: of the 1.0 points, about 0.6 come from simply
      starting with larger weights, and about 0.4 from the Sobel pattern (with 1 000 pictures: 1.2 and 1.8 of the
      2.9 points). Without the control we would have credited Sobel with all of it. How large the starting
      weights should be is a topic of its own: the `nn.init.kaiming_normal_` of P3 exists to get that size right.

    ### 2.6 · Try it

    Change the settings, run this cell and the next one. The run cell trains the random twin and your variant
    on the same data and draws both curves.

    1. **The scale of the filters**: `filter_scale = 1.0` (Sobel's own numbers), then `0.05`. Do too large or
       too small starting weights hurt?
    2. **Frozen filters**: `freeze_first_layer = True`.
    3. **Very little data**: `n_train = 200`. And a lot: `n_train = 60000`.
    4. **One epoch only**: `epochs = 1`. A better start should show most at the very beginning.
    5. **Only Sobel x and Sobel y**: `chosen_filters = [sobel_x, -sobel_x, sobel_y, -sobel_y]`. The other four
       filters of the layer then keep their random start.
    6. **A useless start**: `chosen_filters = [sobel_x, sobel_x, sobel_x, sobel_x, sobel_x, sobel_x, sobel_x,
       sobel_x]`, eight times the same filter. Can training repair it? And with `freeze_first_layer = True`?
    """),
    code("""
    # ---------------- Try it: settings of the hand-made start ----------------
    chosen_filters = handmade_filters     # a list of at most 8 filters of 3 x 3
    filter_scale = 0.25                   # every filter is multiplied by this number
    freeze_first_layer = False            # True: training may not change the first layer
    # ---------------- run settings ----------------
    n_train = 10000                       # training pictures (at most 60000)
    epochs = 5
    learning_rate = 0.001
    """),
    md(r"""
    **Expected time: about 15 seconds** with the settings above.
    """),
    code("""
    try_train, try_val, classes = P.get_mnist(n_train=n_train, n_val=2000, batch_size=64)

    torch.manual_seed(0)
    random_try = build_cnn()

    torch.manual_seed(0)
    handmade_try = build_cnn()
    set_first_layer(handmade_try, chosen_filters, filter_scale)
    if freeze_first_layer:
        handmade_try[0].weight.requires_grad_(False)
        handmade_try[0].bias.requires_grad_(False)

    history_random_try = P.train(random_try, try_train, try_val, epochs=epochs, lr=learning_rate,
                                 verbose=False, name="random start")
    history_handmade_try = P.train(handmade_try, try_train, try_val, epochs=epochs, lr=learning_rate,
                                   verbose=False, name="your start")
    print(f"random start: best validation accuracy {history_random_try['best_val_acc']:.1f}%")
    print(f"your start:   best validation accuracy {history_handmade_try['best_val_acc']:.1f}%")
    P.plot_history({"random start": history_random_try, "your start": history_handmade_try})
    P.show_filters(handmade_try, title="your start: the first-layer filters after training")
    """),
    md(r"""
    <details><summary><b>What we saw</b> (open after you have tried)</summary>

    One run each, seed 0. The random twin reaches 96.8% with 10 000 pictures and 5 epochs.

    | setting | random start | your start |
    |---|---|---|
    | `filter_scale = 1.0` | 96.8% | 97.6% (96.5% after the first epoch) |
    | `filter_scale = 0.05` | 96.8% | 97.0% (93.2% after the first epoch) |
    | `freeze_first_layer = True` | 96.8% | 97.8% |
    | `n_train = 200` | 59.5% | 72.3% |
    | `n_train = 60000` | 98.6% | 98.9% |
    | `epochs = 1` | 93.2% | 95.5% |
    | only the four Sobel filters | 96.8% | 97.5% |
    | eight times Sobel x | 96.8% | 96.5% |
    | eight times Sobel x, frozen | 96.8% | 94.9% |

    1. Larger filters give a faster start (96.5% after one epoch with scale 1.0, 95.5% with 0.25) but no better
       end. Tiny filters (0.05) start exactly like the random twin, 93.2%: the pattern is right, but the signal
       that reaches the next layer is too weak to use until training has made the filters larger.
    2. Freezing changes nothing, as in section 2.5.
    3. The less data, the more the start matters: +12.8 points with 200 pictures, +0.3 with 60 000. With enough
       data the network finds good filters by itself.
    4. The advantage is largest at the beginning: 2.3 points after one epoch, 1.0 after five.
    5. Four Sobel filters with four random ones: 97.5%, most of the gain of all eight.
    6. Eight copies of one filter are a bad start: the layer only sees vertical edges of one kind. If training may
       change them, it repairs most of the damage (96.5%, close to the random twin). Frozen, the network stays at
       94.9%: it has to recognise digits from one kind of edge. A hand-made start helps only if the filters are
       good **and different from each other**.
    </details>
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · What to remember

    * **Experiment 1.** A fully connected network with four ways to flatten (kernels 1, 2, 4, 7) was no better
      than a plain one with the same number of parameters when the kernels average (94.57% against 94.43%, inside
      the spread between seeds), and a little better when they take the maximum (95.00%). An average of pixels
      carries no information that the pixels do not carry; a different view is not the same as new information.
    * **Experiment 2.** The same CNN started from eight hand-made edge filters instead of random ones learned
      faster and ended better (97.8% against 96.8%), and the advantage grew when data was scarce (91.4% against
      88.5% with 1 000 pictures). Frozen filters did as well as trainable ones. A control with random filters of
      the same size showed that more than half of the gain with 10 000 pictures came from the size of the starting
      weights.
    * **How to run an experiment.** Write the hypothesis down first. Change one thing at a time (twins from the
      same seed). Compare with a fair baseline (same parameters, same training). Repeat with several seeds and
      compare the difference with the spread. Add a control for the boring explanation (here: "the weights are
      just bigger").
    * **What is known beyond this notebook.** Starting a network from good filters is what transfer learning does
      (P5). Giving a network several views of its input side by side is what the Inception block does (P3), there
      with learned, overlapping filters instead of fixed summaries.

    ### Words from this notebook

    | word | meaning |
    |---|---|
    | average pooling | replacing every k x k square of a picture by the average of its values |
    | baseline | the simple, known network a new idea is compared with |
    | branch | one of several chains of layers that receive the same input side by side |
    | build function | a function without arguments that returns a new, untrained network |
    | control | an extra run that shares the boring explanation but not the interesting one, to tell the two apart |
    | concatenation | gluing tensors together along one axis (`torch.cat`) |
    | fair comparison | comparing networks that differ in the one thing under test, not also in size or training |
    | frozen | a parameter that training is not allowed to change (`requires_grad` is `False`) |
    | hypothesis | a claim that an experiment can confirm or refute |
    | initialisation | the values the weights have before training starts |
    | seed | the starting point of the random numbers; another seed gives another run of the same experiment |
    | Sobel filter | a hand-made 3 x 3 filter that finds edges in one direction |
    | spread | how far repeated runs of the same experiment lie apart |
    """),
]
