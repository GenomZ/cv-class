from nb import md, code, exercise, api

NAME = "P6-googlenet-auxiliary-classifiers"

CELLS = [
    md(r"""
    # P6 · GoogLeNet's auxiliary classifiers: two extra heads half-way up

    **What we do today**

    In P3 you built GoogLeNet: a stem, nine Inception blocks and a head. The original network of 2014 had two
    more parts, which P3 left out: two **auxiliary classifiers**, small extra heads half-way up the network that
    help it train. Today we add them, and only them: the stem, the blocks and the head are P3's, reused as they
    are.

    1. The idea: why GoogLeNet needed help to train, where the two extra heads sit, and the loss it was trained
       with.
    2. Build them in three small steps, (a) to (c): the extra head from its table; the forward pass of a
       GoogLeNet that gives three sets of scores while it trains and one when it predicts; the loss that combines
       the three.
    3. See what they change: how much stronger the training signal gets in the first blocks.
    4. Train a small version on colour photos twice, with and without the extra heads, and compare.

    **How to use this notebook.** As in P3: run the cells from top to bottom with **Shift+Enter**. You type code
    only in the three cells headed **Your turn**. In them, every `____` (four underscores) is a blank: replace
    it with what its `# YOUR CODE:` comment describes, copying the numbers from the table above the cell. Each
    of these cells starts with the line `raise NotImplementedError`; delete that line once every blank is
    filled in.

    You cannot get stuck. If a step is not finished, its check says what is missing, and the next cells
    continue with the course's own **reference** version of that step and say so ("note: using the reference
    ..."). You can even run the whole notebook before typing anything. The finished cells are in
    `solutions/P6-googlenet-auxiliary-classifiers-solution.ipynb`.

    * `P.something(...)` is a function of the course toolkit (P1); `T.something(...)` is a check or a helper of
      this notebook.
    * Closed boxes **Functions used here (click to open)** explain the PyTorch functions of the next cell.
    * **New words** are in bold where they are explained, and collected at the end.
    * Numbers quoted in the text are what the cells print. Times differ from computer to computer.
    """),

    # ------------------------------------------------------------------ 0
    md(r"""
    ## 0 · Setup

    As in P3, this cell finds the course's helper folder `cvcourse` and loads PyTorch (`torch`, with its layers
    in `torch.nn`, short `nn`) and the toolkit `P`. Also:

    * `T`: the checks of this notebook, one per step, and `T.yours_or_reference(...)`, which keeps your version of
      a step if it passes its checks and otherwise hands over the reference version;
    * `p3`: the course's finished GoogLeNet parts from P3 (`p3.make_stem()`, `p3.make_inception_blocks()`,
      `p3.make_head(...)`, `p3.Inception(...)`): today's network is built from them;
    * `reference`: today's finished network, so that we can look at the target before building it.

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
    from cvcourse import practical as P
    from cvcourse import ptests_p6 as T
    from cvcourse import pmodels_p3 as p3
    from cvcourse import pmodels_p6 as reference

    P.setup(seed=0)
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · The idea: two extra heads half-way up

    ### Why GoogLeNet needed help

    GoogLeNet is 22 layers deep along its longest path (P3): 3 in the stem, 2 in each of the 9 blocks, 1 in the
    head. In 2014 that was very deep, and deep networks were hard to train. During training, `loss.backward()`
    (P1) works out, from the last layer back to the first, how each weight should change: its **gradient**. On
    its way back through many layers the gradient tends to shrink, a little at every layer, so the first layers
    get only a faint signal and learn slowly. GoogLeNet had none of today's two remedies: **batch
    normalisation** (P3: a layer that rescales each map; it came out a few months later) and **skip
    connections** (shortcuts around blocks, in ResNet, 2015).

    The authors' remedy: two small extra classifiers, the **auxiliary classifiers** ("auxiliary" means
    "helping"; we also call them **auxiliary heads**). Each one takes the maps of a block in the middle of the
    network and turns them into its own 1000 scores, as if the network ended there. During training, the
    mistakes of these scores are added to the loss, so the blocks below them get a gradient from a head that is
    only a few layers away.

    ### Where they sit

    | head | attached after | reads | gives |
    |---|---|---|---|
    | `aux1` | block 4a | 512 maps of 14 x 14 | 1000 scores |
    | `aux2` | block 4d | 528 maps of 14 x 14 | 1000 scores |
    | main head (P3) | block 5b | 1024 maps of 7 x 7 | 1000 scores |

    ```
    photo -> stem -> 3a -> 3b -> pool3 -> 4a -> 4b -> 4c -> 4d -> 4e -> pool4 -> 5a -> 5b -> head -> main scores
                                           |                 |
                                          aux1              aux2
                                           |                 |
                                      aux1 scores       aux2 scores            (only while training)
    ```

    ### Three facts

    * **Only while training.** A network has two **modes**. In **training mode** our network returns three sets
      of scores: main, aux1, aux2. In **prediction mode** (PyTorch calls it evaluation mode, `model.eval()`)
      the auxiliary heads do not run, and it returns the main scores only. After training they are thrown away:
      what remains is P3's GoogLeNet.
    * **The loss.** Each set of scores gets its own **cross-entropy** loss (P1: the usual loss for
      classification, small when the right class gets a high score). The two auxiliary losses count 0.3 times
      (the paper, section 5):

      > loss = loss_main + 0.3 x loss_aux1 + 0.3 x loss_aux2

      We call the 0.3 `aux_weight`. It keeps the main head in charge: the auxiliary heads help, they do not
      lead.
    * **They are not small.** Each holds about 3.2 million parameters, almost half as many as the whole
      network of P3 (7.0 million).

    ### The target

    `T.show_in_training_mode(model, input_shape, depth=1)` draws a network like `P.show_model` (P1), but in
    training mode. (`P.show_model` always draws in prediction mode, where the auxiliary heads do not run.) Here
    is today's finished network, the course's reference version: `depth=1` draws each part as one box.
    """),
    code("""
    T.show_in_training_mode(reference.GoogLeNetAux(), (3, 224, 224), depth=1,
                            title="The target: GoogLeNet with its two auxiliary heads, in training mode")
    """),
    md(r"""
    **Reading the drawing.** Follow the main path from the input: the stem, the blocks in three groups, the head.
    After the group that ends with block 4a (512 x 14 x 14) the path splits: one branch goes up through `aux1`
    to the output. After the group that ends with block 4d (528 x 14 x 14) it splits again for `aux2`. The output box collects three sets
    of scores: "1000 | 1000 | 1000". The header line counts 13 378 280 parameters: P3's 6 998 552 plus the two
    heads.
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · Build it in three steps

    | step | what | you type |
    |---|---|---|
    | (a) | `make_aux_head`: one auxiliary head, from its table | 6 layers |
    | (b) | `GoogLeNetAux.forward`: three sets of scores in training mode, one in prediction mode | 5 lines |
    | (c) | `aux_loss`: the loss of the three heads | 3 lines |

    Every step has P3's rhythm: a **table** with all the numbers; **Your turn**, a prepared cell with blanks;
    a **check**, one `[PASS]` or `[FAIL]` line per property, with a **hint** for every `[FAIL]`; then
    `x = T.yours_or_reference(x)`, which keeps your `x` if it passes its checks and otherwise uses the
    reference one, and a drawing of what you have built.

    ### Step (a) · The auxiliary head

    The paper (section 5) describes the auxiliary head layer by layer. For `aux1`, which reads the 512 maps of
    14 x 14 coming out of block 4a:

    | # | layer | settings | output for one photo | the numbers |
    |---|---|---|---|---|
    | 1 | `nn.AvgPool2d` | kernel_size=5, stride=3 | 512 x 4 x 4 | (14 - 5) / 3 + 1 = 4 |
    | 2 | `nn.Conv2d` | `in_channels` -> 128 maps, kernel_size=1 | 128 x 4 x 4 | |
    | 3 | `nn.ReLU` | | 128 x 4 x 4 | |
    | 4 | `nn.Flatten` (given) | | 2048 | 128 x 4 x 4 = 2048 |
    | 5 | `nn.Linear` | 2048 -> 1024 | 1024 | |
    | 6 | `nn.ReLU` | | 1024 | |
    | 7 | `nn.Dropout` | p=0.7 | 1024 | |
    | 8 | `nn.Linear` | 1024 -> `num_classes` | 1000 | |

    New here: the **average pool**, `nn.AvgPool2d`. Like a max-pool (P1), it slides a window over each map, but
    it keeps the **average** of the window instead of the largest value. (P3's global average pooling averaged
    a whole map; this one averages small 5 x 5 windows.) With stride 3 the windows start every 3 pixels, at
    pixels 0, 3, 6 and 9: four windows fit into 14 pixels (the last one covers pixels 9 to 13), so 14 x 14
    becomes 4 x 4. Without `stride=3`, `nn.AvgPool2d` steps by its kernel size, 5, and 14 x 14 would become
    2 x 2.

    The rest you know from P3: a 1x1 convolution that reduces the maps (here to 128), then a small classifier
    of two linear layers. The paper uses a stronger dropout here (0.7: 70% of the 1024 numbers set to 0 while
    training) than in the main head (0.4).

    Write `in_channels` and `num_classes` where the table has them: the same function builds `aux1` (512 maps
    in) and `aux2` (528 maps in). The 2048 stays 2048 for both: whatever comes in, the 1x1 convolution makes 128
    maps of 4 x 4.

    **Your turn.** Layers 1 and 4 are given; type layers 2, 3, 5, 6, 7 and 8, one per blank, and delete the
    `raise` line.
    """),
    api("nn.AvgPool2d"),
    api("nn.Sequential", "nn.Conv2d", "nn.ReLU", "nn.Flatten", "nn.Linear", "nn.Dropout", brief=True),
    exercise(
        stub='''
        def make_aux_head(in_channels, num_classes):
            """An auxiliary head: in_channels maps of 14 x 14 -> num_classes scores (the paper, section 5)."""
            raise NotImplementedError   # delete this line once every ____ is filled in
            head = nn.Sequential(
                nn.AvgPool2d(kernel_size=5, stride=3),     # 1: the example   -> in_channels x 4 x 4
                ____,      # YOUR CODE: 2, nn.Conv2d from in_channels to 128 maps, kernel_size=1   -> 128 x 4 x 4
                ____,      # YOUR CODE: 3, nn.ReLU()
                nn.Flatten(),                              # 4: given   -> 2048 numbers
                ____,      # YOUR CODE: 5, nn.Linear from 2048 to 1024
                ____,      # YOUR CODE: 6, nn.ReLU()
                ____,      # YOUR CODE: 7, nn.Dropout with p=0.7
                ____,      # YOUR CODE: 8, nn.Linear from 1024 to num_classes
            )
            return head
        ''',
        solution='''
        def make_aux_head(in_channels, num_classes):
            """An auxiliary head: in_channels maps of 14 x 14 -> num_classes scores (the paper, section 5)."""
            head = nn.Sequential(
                nn.AvgPool2d(kernel_size=5, stride=3),     # 1: the example   -> in_channels x 4 x 4
                nn.Conv2d(in_channels, 128, kernel_size=1),     # YOUR CODE: 2, nn.Conv2d from in_channels to 128 maps, kernel_size=1   -> 128 x 4 x 4
                nn.ReLU(),                                      # YOUR CODE: 3, nn.ReLU()
                nn.Flatten(),                              # 4: given   -> 2048 numbers
                nn.Linear(2048, 1024),                          # YOUR CODE: 5, nn.Linear from 2048 to 1024
                nn.ReLU(),                                      # YOUR CODE: 6, nn.ReLU()
                nn.Dropout(p=0.7),                              # YOUR CODE: 7, nn.Dropout with p=0.7
                nn.Linear(1024, num_classes),                   # YOUR CODE: 8, nn.Linear from 1024 to num_classes
            )
            return head
        '''),
    md(r"""
    **Check.** The head is built for `aux1` (512 maps in, 1000 classes), looked at layer by layer and run on a
    batch of 2 random "pictures" of 512 x 14 x 14. The last line builds it again for `aux2` with 10 classes, to
    catch numbers typed where names belong.
    """),
    code("""
    T.check_aux_head(make_aux_head)
    """),
    md(r"""
    **What you built.** The pool shrinks the maps from 14 x 14 to 4 x 4, the 1x1 convolution reduces 512 maps to
    128, and Flatten lays out 128 x 4 x 4 = 2048 numbers for the two linear layers. The first linear layer holds
    almost all the parameters: 2 098 176 of 3 188 840.
    """),
    code("""
    make_aux_head = T.yours_or_reference(make_aux_head)
    P.show_model(make_aux_head(512, 1000), (512, 14, 14), title="Your auxiliary head aux1: 512 maps of 14 x 14 -> 1000 scores")
    """),

    # ------------------------------------------------------------------ (b)
    md(r"""
    ### Step (b) · The network with three outputs

    The class `GoogLeNetAux` is P3's GoogLeNet with the two auxiliary heads. Its `__init__` is given: it builds
    the parts with P3's functions and with your `make_aux_head`:

    | part | built with | in | out |
    |---|---|---|---|
    | `self.stem` | `p3.make_stem()` (P3, step e) | 3 x 224 x 224 | 192 x 28 x 28 |
    | `self.blocks_3a_to_4a` | parts 0 to 3 of `p3.make_inception_blocks()`: 3a, 3b, pool3, 4a | 192 x 28 x 28 | 512 x 14 x 14 |
    | `self.blocks_4b_to_4d` | parts 4 to 6: 4b, 4c, 4d | 512 x 14 x 14 | 528 x 14 x 14 |
    | `self.blocks_4e_to_5b` | parts 7 to 10: 4e, pool4, 5a, 5b | 528 x 14 x 14 | 1024 x 7 x 7 |
    | `self.head` | `p3.make_head(1024, num_classes)` (P3, step g) | 1024 x 7 x 7 | `num_classes` scores |
    | `self.aux1` | `make_aux_head(512, num_classes)` (step a) | 512 x 14 x 14 | `num_classes` scores |
    | `self.aux2` | `make_aux_head(528, num_classes)` (step a) | 528 x 14 x 14 | `num_classes` scores |

    In P3, `make_inception_blocks()` returned one `nn.Sequential` of 11 parts (9 blocks and 2 max-pools),
    numbered 0 to 10. `blocks[0:4]` takes parts 0, 1, 2 and 3 as a new `nn.Sequential`: a Python **slice**
    runs from the first number up to, but not including, the second. The chain of blocks is cut into three
    groups so that `forward` can reach the maps after block 4a and after block 4d.

    **Your part is `forward`.** It is P3's forward (stem, blocks, head), with two additions:

    * `self.training` is `True` in training mode and `False` in prediction mode (every `nn.Module` has it).
      `if self.training:` runs the indented line below it only while training, so the auxiliary heads cost
      nothing at prediction time;
    * training mode returns three sets of scores, prediction mode one (the last four lines, given).

    | line | what | output for one photo |
    |---|---|---|
    | `x = self.stem(x)` | the example | 192 x 28 x 28 |
    | `x = self.blocks_3a_to_4a(x)` | the example | 512 x 14 x 14 |
    | `aux1 = ...` | `self.aux1` applied to `x`, the output of block 4a | 1000 scores |
    | `x = ...` | `self.blocks_4b_to_4d` applied to `x` | 528 x 14 x 14 |
    | `aux2 = ...` | `self.aux2` applied to `x`, the output of block 4d | 1000 scores |
    | `x = ...` | `self.blocks_4e_to_5b` applied to `x` | 1024 x 7 x 7 |
    | `main = ...` | `self.head` applied to `x` | 1000 scores |

    Each blank is a part applied to `x`, as `x = self.stem(x)`. Note that `aux1` and `aux2` get their own names:
    `x` must go on through the blocks unchanged.

    **Your turn.** Fill in the five blanks of `forward` and delete the `raise` line.
    """),
    api("Module.training"),
    api("nn.Module", brief=True),
    exercise(
        stub='''
        class GoogLeNetAux(nn.Module):
            """GoogLeNet (P3) with its two auxiliary heads, after blocks 4a and 4d."""

            def __init__(self, num_classes=1000):
                super().__init__()
                self.stem = p3.make_stem()                      # P3, step (e): 3 x 224 x 224 -> 192 x 28 x 28
                blocks = p3.make_inception_blocks()             # P3, step (f): the 11 parts 3a, 3b, pool3, 4a, ..., 5b
                self.blocks_3a_to_4a = blocks[0:4]              # 3a, 3b, pool3, 4a   -> 512 x 14 x 14
                self.blocks_4b_to_4d = blocks[4:7]              # 4b, 4c, 4d          -> 528 x 14 x 14
                self.blocks_4e_to_5b = blocks[7:11]             # 4e, pool4, 5a, 5b   -> 1024 x 7 x 7
                self.head = p3.make_head(1024, num_classes)     # P3, step (g): -> num_classes scores
                self.aux1 = make_aux_head(512, num_classes)     # step (a): reads the 512 maps of block 4a
                self.aux2 = make_aux_head(528, num_classes)     # step (a): reads the 528 maps of block 4d

            def forward(self, x):
                raise NotImplementedError   # delete this line once every ____ is filled in
                x = self.stem(x)                    # the example   -> 192 x 28 x 28
                x = self.blocks_3a_to_4a(x)         # the example   -> 512 x 14 x 14, the output of block 4a
                if self.training:
                    aux1 = ____                     # YOUR CODE: self.aux1 applied to x
                x = ____                            # YOUR CODE: self.blocks_4b_to_4d applied to x   -> 528 x 14 x 14
                if self.training:
                    aux2 = ____                     # YOUR CODE: self.aux2 applied to x
                x = ____                            # YOUR CODE: self.blocks_4e_to_5b applied to x   -> 1024 x 7 x 7
                main = ____                         # YOUR CODE: self.head applied to x   -> num_classes scores
                if self.training:
                    return main, aux1, aux2         # training mode: three sets of scores (given)
                return main                         # prediction mode: the main scores only (given)
        ''',
        solution='''
        class GoogLeNetAux(nn.Module):
            """GoogLeNet (P3) with its two auxiliary heads, after blocks 4a and 4d."""

            def __init__(self, num_classes=1000):
                super().__init__()
                self.stem = p3.make_stem()                      # P3, step (e): 3 x 224 x 224 -> 192 x 28 x 28
                blocks = p3.make_inception_blocks()             # P3, step (f): the 11 parts 3a, 3b, pool3, 4a, ..., 5b
                self.blocks_3a_to_4a = blocks[0:4]              # 3a, 3b, pool3, 4a   -> 512 x 14 x 14
                self.blocks_4b_to_4d = blocks[4:7]              # 4b, 4c, 4d          -> 528 x 14 x 14
                self.blocks_4e_to_5b = blocks[7:11]             # 4e, pool4, 5a, 5b   -> 1024 x 7 x 7
                self.head = p3.make_head(1024, num_classes)     # P3, step (g): -> num_classes scores
                self.aux1 = make_aux_head(512, num_classes)     # step (a): reads the 512 maps of block 4a
                self.aux2 = make_aux_head(528, num_classes)     # step (a): reads the 528 maps of block 4d

            def forward(self, x):
                x = self.stem(x)                    # the example   -> 192 x 28 x 28
                x = self.blocks_3a_to_4a(x)         # the example   -> 512 x 14 x 14, the output of block 4a
                if self.training:
                    aux1 = self.aux1(x)             # YOUR CODE: self.aux1 applied to x
                x = self.blocks_4b_to_4d(x)         # YOUR CODE: self.blocks_4b_to_4d applied to x   -> 528 x 14 x 14
                if self.training:
                    aux2 = self.aux2(x)             # YOUR CODE: self.aux2 applied to x
                x = self.blocks_4e_to_5b(x)         # YOUR CODE: self.blocks_4e_to_5b applied to x   -> 1024 x 7 x 7
                main = self.head(x)                 # YOUR CODE: self.head applied to x   -> num_classes scores
                if self.training:
                    return main, aux1, aux2         # training mode: three sets of scores (given)
                return main                         # prediction mode: the main scores only (given)
        '''),
    md(r"""
    **Check.** The network runs on a batch of 2 random photos, once in training mode and once in prediction
    mode. The check records which part runs when and what it receives, compares the scores with the reference
    network's (with the same weights), and finally with **torchvision**'s GoogLeNet built with
    `aux_logits=True` (P3: torchvision is PyTorch's library of ready-made networks). It takes a few seconds.
    """),
    code("""
    T.check_googlenet_aux(GoogLeNetAux)
    """),
    md(r"""
    **Reading the last lines.** torchvision's GoogLeNet also returns three sets of scores in training mode and
    one in prediction mode, and its auxiliary heads read the same maps as yours. Each of its heads has 128
    parameters more than yours: as in P3, torchvision puts a batch normalisation (2 parameters per map) instead
    of a bias (1 per map) after the 1x1 convolution, which makes 128 maps. (Its heads also average with
    `nn.AdaptiveAvgPool2d((4, 4))` instead of the paper's 5x5 windows with stride 3: the same 4 x 4 output, with
    slightly different windows.)

    **What you built.** The same network drawn twice: in training mode with `T.show_in_training_mode`, and in
    prediction mode with `P.show_model`, where the auxiliary heads are not used and disappear from the drawing.
    """),
    code("""
    GoogLeNetAux = T.yours_or_reference(GoogLeNetAux)
    net = GoogLeNetAux()
    T.show_in_training_mode(net, (3, 224, 224), depth=1, title="Your GoogLeNetAux in training mode: three outputs")
    P.show_model(net, (3, 224, 224), depth=1, title="The same network in prediction mode: one output")
    """),
    md(r"""
    In prediction mode the drawing is P3's GoogLeNet again. Its header still counts 13 378 280 parameters,
    because the auxiliary heads are still stored in the network, but only the 6 998 552 parameters of the boxes
    are used. The multiply-adds count what runs: 1.59 billion per photo in training mode, 1.58 billion in
    prediction mode, P3's number. The auxiliary heads are cheap to run; it is their parameters that are many.

    **Run it yourself.** `net.train()` and `net.eval()` switch the mode. `torch.randn(2, 3, 224, 224)` makes
    2 random "photos" (random numbers, nothing to recognise: enough to see the shapes).
    """),
    api("Module.train"),
    api("Module.eval", "torch.manual_seed", brief=True),
    api("torch.randn"),
    code("""
    torch.manual_seed(0)
    photos = torch.randn(2, 3, 224, 224)            # 2 random "photos"

    net.train()                                     # training mode
    outputs = net(photos)
    print("training mode:", len(outputs), "sets of scores")
    for scores in outputs:
        print("    one set of scores of shape", tuple(scores.shape))

    net.eval()                                      # prediction mode
    scores = net(photos)
    print("prediction mode: one set of scores of shape", tuple(scores.shape))
    """),
    md(r"""
    `T.parameters_per_part(net)` counts the parameters of each part:
    """),
    code("""
    T.parameters_per_part(net)
    """),
    md(r"""
    **What the numbers say.** The two auxiliary heads hold 6 379 728 parameters, almost as many as the main
    network (6 998 552): while it trains, the network is almost twice as big. After training they are thrown
    away, and the network that remains is P3's GoogLeNet.
    """),

    # ------------------------------------------------------------------ (c)
    md(r"""
    ### Step (c) · The loss

    In P1's training loop (the box "What `P.train` does inside") the loss of a batch was one line:

    ```python
    loss_fn = nn.CrossEntropyLoss()           # the usual loss for classification
    loss = loss_fn(scores, labels)            # how wrong are the scores?
    ```

    Now the network gives three sets of scores while training, so there are three cross-entropies, one per head,
    and the loss adds them up with their weights:

    | term | scores | labels | weight |
    |---|---|---|---|
    | `loss_main` | `main` | `labels` | 1 |
    | `loss_aux1` | `aux1` | `labels` | `aux_weight` (0.3) |
    | `loss_aux2` | `aux2` | `labels` | `aux_weight` (0.3) |

    > loss = loss_main + aux_weight x loss_aux1 + aux_weight x loss_aux2

    All three heads must name the same class: the labels are the same for all three.

    **An example by hand.** Say that on one batch the main head's cross-entropy is 2.0, aux1's 2.5 and aux2's
    2.4 (the heads half-way up are usually a little worse than the main head). Then

    > loss = 2.0 + 0.3 x 2.5 + 0.3 x 2.4 = 2.0 + 0.75 + 0.72 = 3.47

    The same in Python:
    """),
    code("""
    loss_main = 2.0
    loss_aux1 = 2.5
    loss_aux2 = 2.4
    aux_weight = 0.3
    loss = loss_main + aux_weight * loss_aux1 + aux_weight * loss_aux2
    print("loss =", round(loss, 2))
    """),
    md(r"""
    The function `aux_loss(outputs, labels, aux_weight=0.3)` does this with real scores. Its first line, given,
    takes `outputs` (the three sets of scores of training mode) apart into `main`, `aux1` and `aux2`, in that
    order. `cross_entropy` is P1's `nn.CrossEntropyLoss()`, made once above the function; `aux_weight=0.3` is a
    default value (P3): a call that leaves it out gets 0.3. Write `aux_weight`, not 0.3, so that other weights
    can be tried (section 4 trains with `aux_weight=0`).

    **Your turn.** `loss_main` is the example; fill in `loss_aux1`, `loss_aux2` and `loss`, and delete the
    `raise` line.
    """),
    api("nn.CrossEntropyLoss"),
    exercise(
        stub='''
        cross_entropy = nn.CrossEntropyLoss()           # P1: the usual loss for classification (given)


        def aux_loss(outputs, labels, aux_weight=0.3):
            """The loss of GoogLeNet with auxiliary heads: loss_main + aux_weight x loss_aux1 + aux_weight x loss_aux2."""
            raise NotImplementedError   # delete this line once every ____ is filled in
            main, aux1, aux2 = outputs                  # the three sets of scores of training mode (given)
            loss_main = cross_entropy(main, labels)     # the example: the main head's loss
            loss_aux1 = ____        # YOUR CODE: cross_entropy of aux1 and labels
            loss_aux2 = ____        # YOUR CODE: cross_entropy of aux2 and labels
            loss = ____             # YOUR CODE: loss_main + aux_weight * loss_aux1 + aux_weight * loss_aux2
            return loss
        ''',
        solution='''
        cross_entropy = nn.CrossEntropyLoss()           # P1: the usual loss for classification (given)


        def aux_loss(outputs, labels, aux_weight=0.3):
            """The loss of GoogLeNet with auxiliary heads: loss_main + aux_weight x loss_aux1 + aux_weight x loss_aux2."""
            main, aux1, aux2 = outputs                  # the three sets of scores of training mode (given)
            loss_main = cross_entropy(main, labels)     # the example: the main head's loss
            loss_aux1 = cross_entropy(aux1, labels)     # YOUR CODE: cross_entropy of aux1 and labels
            loss_aux2 = cross_entropy(aux2, labels)     # YOUR CODE: cross_entropy of aux2 and labels
            loss = loss_main + aux_weight * loss_aux1 + aux_weight * loss_aux2      # YOUR CODE: loss_main + aux_weight * loss_aux1 + aux_weight * loss_aux2
            return loss
        '''),
    md(r"""
    **Check.** Made-up scores of 4 photos for 10 classes go through your function with three different weights.
    The last line runs `loss.backward()` and checks that each set of scores gets its share of the gradient: 1 for
    the main head, 0.3 for each auxiliary head.
    """),
    code("""
    T.check_aux_loss(aux_loss)
    """),
    code("""
    aux_loss = T.yours_or_reference(aux_loss)
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · What the auxiliary heads change: the gradient of the first blocks

    Section 1 said that the gradient shrinks on its way back through the layers. Here we measure it, on the
    untrained network.

    For every weight, `loss.backward()` computes a gradient: how strongly, and in which direction, the loss asks
    that weight to change. `T.gradient_per_part(net, photos, labels, aux_loss)` turns this into one number per
    part of the network, the **size of its gradient** (all the gradient numbers of the part's weights, squared,
    added up, square root: the bigger, the more strongly the part is pushed to change). It measures it twice on
    the same batch:

    * with the main loss alone (`aux_weight=0`: as if the auxiliary heads were not there);
    * with the full loss, `aux_weight=0.3`.

    It changes no weight. `torch.tensor([1, 2, 3, 4])` makes the labels of the 4 random photos: made-up
    classes, enough to compute a loss. **Expected time: a few seconds.**
    """),
    api("torch.tensor"),
    code("""
    torch.manual_seed(0)
    net = GoogLeNetAux()
    photos = torch.randn(4, 3, 224, 224)            # 4 random "photos"
    labels = torch.tensor([1, 2, 3, 4])             # made-up classes
    T.gradient_per_part(net, photos, labels, aux_loss)
    """),
    md(r"""
    **Reading the table.** First the losses: an untrained network guesses, so each head's cross-entropy is about
    the value for a blind guess among 1000 classes (ln 1000 = 6.91; the main head's is 6.903), and the full loss,
    11.055, is about (1 + 0.3 + 0.3) x 6.91 = 11.05.

    Then the column "main loss alone": the gradient shrinks from 0.581 in the head to 0.0000535 in the stem,
    about 11 000 times smaller. This is the problem of section 1, measured: with the main head alone, the first
    blocks hardly learn at the start of training.

    With the auxiliary losses, the parts after block 4d do not change ("times bigger" 1.0): no auxiliary head
    sits above them. The blocks 4b to 4d get 2.6 times more, because `aux2` sits right after them. The blocks 3a
    to 4a get 111.5 times more and the stem 104.4 times more: `aux1` is only a few layers above them. This is
    what the auxiliary heads were for.

    **Later networks do it differently.** Batch normalisation keeps the signal at the same size from layer to
    layer, and skip connections give the gradient a short way back to every block. ResNet (2015) and most
    networks since use these instead and have no auxiliary heads. The authors of Inception v3 (Szegedy et al.,
    2015) also found that the auxiliary heads did not make training faster at the start; the network with them
    pulled slightly ahead only near the end of a long training, so they saw them mainly as a protection against
    learning the training photos by heart.
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · Train a small version with and without the auxiliary heads

    Does all this help a network learn? We train the small GoogLeNet of P3 on **CIFAR-10** (P3: colour photos
    of 32 x 32 pixels in 10 classes) twice, once with the auxiliary losses and once without, and compare.
    """),
    code("""
    train_loader, val_loader, classes = P.get_cifar10(n_train=10000, n_val=2000, batch_size=64)
    """),
    md(r"""
    ### The small network

    `SmallGoogLeNetAux` is P3's `SmallGoogLeNet` (a light stem, blocks 3a, 3b and 4a at a quarter of their width,
    the same head, He initialisation) with two auxiliary heads, one after block 3a and one after block 3b. Its
    heads are smaller than the paper's, so that they stay smaller than the network they help:

    | # | layer | settings | output for one photo (`aux2`) |
    |---|---|---|---|
    | 1 | `nn.AvgPool2d` | kernel_size=5, stride=3 | 120 x 4 x 4: (16 - 5) / 3 + 1 = 4.67 -> 4 |
    | 2 | `nn.Conv2d` | `in_channels` -> 16 maps, kernel_size=1 | 16 x 4 x 4 |
    | 3 | `nn.ReLU` | | 16 x 4 x 4 |
    | 4 | `nn.Flatten` | | 256 |
    | 5 | `nn.Linear` | 256 -> 64 | 64 |
    | 6 | `nn.ReLU` | | 64 |
    | 7 | `nn.Dropout` | p=0.7 | 64 |
    | 8 | `nn.Linear` | 64 -> `num_classes` | 10 |

    Nothing to type here: read the cell and run it. `forward` follows the pattern of your step (b).
    """),
    api("nn.MaxPool2d", "Module.add_module", "nn.init.kaiming_normal_", "Module.modules", brief=True),
    code('''
    def make_small_aux_head(in_channels, num_classes):
        """The paper's auxiliary head, made smaller for the small network."""
        head = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),          # 16 x 16 -> 4 x 4
            nn.Conv2d(in_channels, 16, kernel_size=1),      # -> 16 x 4 x 4
            nn.ReLU(),
            nn.Flatten(),                                   # -> 256 numbers
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(64, num_classes),
        )
        return head


    class SmallGoogLeNetAux(nn.Module):
        """P3's SmallGoogLeNet for 32 x 32 photos, with an auxiliary head after block 3a and one after block 3b."""

        def __init__(self, num_classes=10):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 48, kernel_size=3, padding=1),           # 3 x 32 x 32 -> 48 x 32 x 32
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),     # -> 48 x 16 x 16
            )
            self.block_3a = p3.Inception(48, 16, 24, 32, 4, 8, 8)            # -> 64 x 16 x 16
            self.block_3b = p3.Inception(64, 32, 32, 48, 8, 24, 16)          # -> 120 x 16 x 16
            self.pool_and_4a = nn.Sequential()
            self.pool_and_4a.add_module("pool3", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))    # -> 120 x 8 x 8
            self.pool_and_4a.add_module("inception4a", p3.Inception(120, 48, 24, 52, 4, 12, 16))     # -> 128 x 8 x 8
            self.head = p3.make_head(128, num_classes)                        # -> 10 scores
            self.aux1 = make_small_aux_head(64, num_classes)                  # reads the 64 maps of block 3a
            self.aux2 = make_small_aux_head(120, num_classes)                 # reads the 120 maps of block 3b
            # He initialisation (P3): start every convolution with weights of the size that suits a ReLU network
            for layer in self.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        def forward(self, x):
            x = self.stem(x)
            x = self.block_3a(x)
            if self.training:
                aux1 = self.aux1(x)
            x = self.block_3b(x)
            if self.training:
                aux2 = self.aux2(x)
            x = self.pool_and_4a(x)
            main = self.head(x)
            if self.training:
                return main, aux1, aux2
            return main
    '''),
    code("""
    torch.manual_seed(0)
    small_net = SmallGoogLeNetAux()
    T.show_in_training_mode(small_net, (3, 32, 32), depth=1, title="SmallGoogLeNetAux in training mode")
    T.parameters_per_part(small_net)
    """),
    md(r"""
    The main network is P3's small GoogLeNet, 60 978 parameters; the two small auxiliary heads add 37 172.

    ### Does the small network need the help?

    The same measurement as in section 3, on the small untrained network and a real batch: the first 64
    training photos with their labels (`next(iter(train_loader))`, P1, takes the first batch). The small
    network's parts have other names, so the cell lists them.
    """),
    code("""
    images, labels = next(iter(train_loader))       # the first batch: 64 photos and their labels
    T.gradient_per_part(small_net, images, labels, aux_loss, parts=["stem", "block_3a", "block_3b", "pool_and_4a", "head"])
    """),
    md(r"""
    **Reading the table.** Here the stem's gradient from the main loss alone is 0.611, about 8.5 times smaller
    than the head's (5.22), not 11 000 times as in the big untrained GoogLeNet: through 8 layers, with He
    initialisation, the gradient survives the way back. The auxiliary losses make the stem's gradient only 1.3
    times bigger. The small network hardly needs the help; let us see whether it still makes a difference.

    ### How the two runs differ

    Both runs train the same `SmallGoogLeNetAux` from the same starting weights (`torch.manual_seed(seed)`), on
    the same photos in the same order, with the same dropout. The only difference is `aux_weight`:

    * **without**: `aux_weight=0`. The auxiliary heads still run, but their losses count 0 times, so they change
      nothing: this is the network without auxiliary heads;
    * **with**: `aux_weight=0.3`, the paper's loss.

    `T.train_with_aux(model, train_loader, val_loader, aux_loss, aux_weight=..., epochs=..., lr=...)` is
    `P.train` (P1) for a network with auxiliary heads: in every training step it calls your `aux_loss` on the
    three sets of scores. (`P.train` expects one set of scores, so it cannot be used here.) The loss and the
    accuracy it prints and draws are the **main head's**, in both runs, so that the two runs are measured the
    same way; in the run with the auxiliary heads it also prints the full loss, in brackets.

    ### Try it: the run settings

    Run this cell, then the two training cells.

    * `epochs`: passes over the training photos;
    * `learning_rate`: the size of each weight change (P1);
    * `n_train`: how many training photos (at most 50 000);
    * `aux_weight`: the weight of the auxiliary losses in the second run;
    * `seed`: the random starting point of both runs (starting weights, order of the photos, dropout).
    """),
    code("""
    # ---------------- Try it: run settings ----------------
    epochs = 5                    # passes over the training photos
    learning_rate = 0.001         # size of each weight change
    n_train = 10000               # training photos (at most 50000)
    aux_weight = 0.3              # weight of the auxiliary losses in the run "with"
    seed = 0                      # random starting point of both runs: try 1 and 2
    """),
    md(r"""
    **Run 1: without the auxiliary losses.** **Expected time: about 3 minutes** with the settings above (P3
    trained the same network in about 3 minutes; up to twice as long on a slow laptop).
    """),
    code("""
    train_loader, val_loader, classes = P.get_cifar10(n_train=n_train, n_val=2000, batch_size=64)
    torch.manual_seed(seed)
    small_without = SmallGoogLeNetAux()
    history_without = T.train_with_aux(small_without, train_loader, val_loader, aux_loss, aux_weight=0.0,
                                       epochs=epochs, lr=learning_rate, seed=seed, name="without auxiliary losses")
    """),
    md(r"""
    **Run 2: with the auxiliary losses.** Same network, same starting weights, same photos. **Expected time:
    about 3 minutes.**
    """),
    code("""
    torch.manual_seed(seed)
    small_with = SmallGoogLeNetAux()
    history_with = T.train_with_aux(small_with, train_loader, val_loader, aux_loss, aux_weight=aux_weight,
                                    epochs=epochs, lr=learning_rate, seed=seed, name="with auxiliary losses")
    """),
    md(r"""
    ### The two runs side by side

    `P.plot_history` (P1) draws several runs in one figure: dashed lines for the training photos, solid lines for
    the validation photos. `T.compare_runs` lists the validation accuracy of both runs, epoch by epoch.
    """),
    code("""
    P.plot_history({"without": history_without, "with": history_with})
    T.compare_runs(history_without, history_with)
    """),
    md(r"""
    **Reading the result.** The curves of the two runs lie almost on top of each other. For three epochs the
    two runs are almost the same: 32.0% against 32.1% of the
    validation photos right after epoch 1, 38.5% against 38.8% after epoch 2, 44.4% against 44.7% after epoch 3.
    The main head's training loss is almost equal all along (2.078 and 2.076 in epoch 1, 1.461 and 1.466 in
    epoch 5). In epochs 4 and 5 the run with the auxiliary losses is ahead, and it ends at 47.5% against 46.1%:
    1.3 points, about 26 of the 2 000 validation photos.

    Is that the auxiliary heads, or chance? One pair of runs cannot tell. A network that is trained even a little
    differently ends a little differently, and the other measure, the validation loss, ends almost equal, even a
    little lower without (1.392 against 1.398). The honest summary: on this small network, the auxiliary heads
    made little or no difference. To know more, train both versions with a few seeds (below) and compare.

    **Why so little difference?** The gradient table above says it: this small network is 8 layers deep on its
    longest path, not 22, and it starts with He initialisation (P3), so its gradient reaches the first blocks
    well without any help, and the auxiliary losses add little to it. The paper's auxiliary heads were made for a
    deep network without these aids; even there, the authors of Inception v3 found later that they did not help
    at the start of training (section 3).

    **Things to try** (predict first, then run):

    * `seed = 1`, then `seed = 2`: how much does each run move? Is the difference between "with" and
      "without" bigger than that?
    * `aux_weight = 1.0`: the auxiliary losses count as much as the main one. Better or worse for the main head?
    * `epochs = 10`: does a difference appear later in training, as the Inception v3 authors saw?
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · What to remember

    * The original GoogLeNet has two **auxiliary classifiers**: small extra heads after blocks 4a (512 x 14 x 14
      maps) and 4d (528 x 14 x 14), each giving its own 1000 scores. Each holds about 3.2 million parameters.
    * They exist only while training. In **training mode** the network returns three sets of scores (main, aux1,
      aux2); in **prediction mode** only the main scores, and the auxiliary heads do not run. After training they
      are thrown away. In `forward`, `if self.training:` makes the difference.
    * The loss: **loss = loss_main + 0.3 x loss_aux1 + 0.3 x loss_aux2**, three cross-entropies with the same
      labels.
    * Why: in a deep network without batch normalisation or skip connections, the **gradient** shrinks on its way
      back to the first layers. In the untrained GoogLeNet it was about 11 000 times smaller in the stem than in
      the head; the auxiliary losses made it about 100 times bigger in the stem and the first blocks.
    * Today, batch normalisation and skip connections solve the same problem better, and networks rarely have
      auxiliary heads. On our small CIFAR-10 network, which does not lose its gradient, they made little or no
      difference (47.5% against 46.1% after 5 epochs, in one pair of runs).

    ### Words from this notebook

    | word | meaning |
    |---|---|
    | auxiliary classifier (auxiliary head) | a small extra head half-way up a network, used only while training to help the blocks below it learn |
    | average pool | a layer that replaces each window of a map by the average of its values, `nn.AvgPool2d` |
    | batch normalisation | a layer that rescales each map to average 0 and spread 1, then learns a scale and a shift (P3) |
    | cross-entropy | the usual loss for classification: small when the right class gets a high score (P1) |
    | gradient | for each weight, how strongly and in which direction the loss asks it to change; `loss.backward()` computes it |
    | prediction mode (evaluation mode) | the mode set by `model.eval()`: dropout off, auxiliary heads not used |
    | size of a gradient | all its numbers squared, added up, square root: how strongly a part is pushed to change |
    | skip connection | a shortcut that carries the input of a block around it (ResNet, 2015) |
    | slice | a part of a sequence: `blocks[0:4]` takes items 0, 1, 2, 3 |
    | training mode | the mode set by `model.train()`: dropout on, auxiliary heads used; `self.training` is True |
    | `aux_weight` | how much the auxiliary losses count in the total loss: 0.3 in the paper |
    """),
]
