from nb import md, code, exercise, api

NAME = "P7-object-detection-boxes"

CELLS = [
    md(r"""
    # P7 · Object detection: find every object and draw a box around it

    **What we do today**

    1. See what **object detection** is: a classifier gives one answer for a whole picture; a detector finds
       every object in it and draws a **box** around each one.
    2. Measure how well two boxes match with one number, the **IoU**. You write it yourself: a few lines of
       plain arithmetic (step 1).
    3. Build the simplest detector: pictures with exactly one object, and a small network with two **heads**,
       one that says what the object is and one that says where its box is. You type the layers of the two
       heads (step 2), then train the network for about a minute.
    4. Find several objects per picture with a small **grid detector** (given and trained for you), and clean
       up its answers with a **score threshold** and **non-maximum suppression**.
    5. Use a real detector, trained by others on 118 000 photos, on street photos and pet photos.

    **How to use this notebook.** Run the cells from top to bottom with **Shift+Enter**. You type code only in
    the two cells headed **Your turn**. In them, every `____` (four underscores) is a blank: replace it with
    what its `# YOUR CODE:` comment describes. Each of these cells contains the line `raise NotImplementedError`,
    which stops the function with the message "not implemented"; delete that line once every blank is filled in.

    You cannot get stuck. If a step is not finished, its check says what is missing, and the next cells
    continue with the course's own **reference** version of that step and say so ("note: using the reference
    ..."). You can even run the whole notebook before typing anything. The finished cells are in
    `solutions/P7-object-detection-boxes-solution.ipynb`. Cells marked **Try it** hold settings you may change.
    The whole notebook runs in about 3 to 6 minutes on a laptop; every slow cell says how long it takes.

    * `P.something(...)` is a function of the course toolkit (P1 introduced it); `D.something(...)` is a helper
      of today's notebook (the data, the drawings, the training loops); `T.something(...)` is one of today's
      checks.
    * Closed boxes **Functions used here (click to open)** explain the PyTorch functions of the next cell.
    * **New words** are in bold where they are explained, and collected at the end.
    * Numbers quoted in the text are what the cells print. Times differ from computer to computer.
    """),

    # ------------------------------------------------------------------ 0
    md(r"""
    ## 0 · Setup

    As in P1, this cell finds the course's helper folder `cvcourse` and loads PyTorch (`torch`, with its layers
    in `torch.nn`, short `nn`), `torchvision` (PyTorch's library for images) and the toolkit `P`. New today:

    * `D`: today's helpers: pictures with shapes, drawings of boxes, the training loops of the two detectors, and
      the pretrained detector of section 5;
    * `T`: the checks of the two steps, and `T.yours_or_reference(...)`, which keeps your version of a step if it
      passes its checks and otherwise hands over the course's reference version.

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
    from cvcourse import pkit_p7 as D
    from cvcourse import ptests_p7 as T

    P.setup(seed=0)
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · From classification to detection

    In P1 to P5 every network was a **classifier**: one picture in, one answer out ("7", "pug"). A photo of a
    street holds many things at once: people, cars, a dog. **Object detection** finds every **object** in a
    picture, also called an **instance** (one particular person, one particular car: two people are two
    instances of the class "person"), and gives three things for each one:

    * **what** it is: its **class** ("person", "car", ...);
    * **where** it is: a **box** around it (also called a **bounding box**);
    * **how sure** the detector is: a **score** between 0 and 1.

    | | classification | detection |
    |---|---|---|
    | input | one picture | one picture |
    | output | one class for the whole picture | a list with one row per object: (box, class, score) |
    | how many answers | always one | as many as there are objects: 0, 1, 2, ... |

    ### A box is four numbers

    A position in a picture is counted in **pixels** from its top-left corner: **x** to the right and **y
    downwards** (not upwards, as in graphs at school). A box is written as four numbers

    > (x1, y1, x2, y2): (x1, y1) is its top-left corner, (x2, y2) its bottom-right corner.

    So **width** = x2 - x1, **height** = y2 - y1 and **area** = width x height (in pixels). This way of writing
    boxes is called **xyxy**; torchvision uses it too.

    `D.example_picture()` makes a picture of 64 x 64 pixels with three shapes and returns it with their true
    boxes and their **labels** (class numbers: 0 for ellipse, 1 for rectangle, 2 for triangle; `D.CLASSES` holds
    the names). `D.show_boxes(picture, boxes, labels)` draws the boxes; the numbers along the top and the left
    side are the x and y positions in pixels.
    """),
    code("""
    picture, boxes, labels = D.example_picture()
    print("boxes:", boxes)
    print("labels:", labels, "   the class names, numbers 0, 1, 2:", D.CLASSES)
    D.show_boxes(picture, boxes, labels, title="three objects, three boxes")
    """),
    md(r"""
    Read the first box off the drawing: the triangle's box starts at x = 6 and y = 4 (its top-left corner) and
    ends at x = 35 and y = 27 (its bottom-right corner). The cell below computes width, height and area of every
    box, exactly as you would by hand.
    """),
    code("""
    for k in range(len(boxes)):
        x1, y1, x2, y2 = boxes[k]                       # the four numbers of box k
        width = x2 - x1
        height = y2 - y1
        area = width * height
        name = D.CLASSES[labels[k]]
        print(f"{name:<10} box {boxes[k]}: width {x2} - {x1} = {width}, height {y2} - {y1} = {height}, "
              f"area {width} x {height} = {area} pixels")
    """),
    md(r"""
    The three boxes have different shapes: the triangle's is wider than high (29 x 23 pixels), the rectangle's
    higher than wide (21 x 29). A detector for this picture should answer with these three rows (box, class),
    each with a score. The rest of the notebook builds detectors that come close.
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · IoU: how well do two boxes match?

    A detector's box is never exactly on the true box: it may be a few pixels too far left, or a bit too big.
    We need one number that says how good a match is, to measure a detector (and, in section 4, to remove
    duplicate boxes). The usual number is **IoU**, **intersection over union**:

    * the **intersection** of two boxes is the area they share (their overlap);
    * their **union** is the area covered by at least one of them;
    * **IoU = area of the intersection / area of the union**.

    IoU is 1 when the two boxes are the same and 0 when they do not overlap at all. The common rule: a predicted
    box with IoU **at least 0.5** with the true box (and the right class) counts as **found**.

    ### A worked example

    Box A = (2, 2, 10, 8) and box B = (4, 4, 12, 10).

    1. Areas: A is (10 - 2) x (8 - 2) = 8 x 6 = 48 pixels; B is (12 - 4) x (10 - 4) = 8 x 6 = 48 pixels.
    2. The intersection is itself a box. Its left edge is the right-most of the two left edges,
       max(2, 4) = 4; its top edge the lower of the two top edges, max(2, 4) = 4 (y grows downwards); its right
       edge the left-most of the two right edges, min(10, 12) = 10; its bottom edge min(8, 10) = 8. So the
       intersection is (4, 4, 10, 8): width 10 - 4 = 6, height 8 - 4 = 4, area 6 x 4 = 24 pixels.
    3. The union: adding the two areas counts the overlap twice (it is inside both boxes), so subtract it once:
       48 + 48 - 24 = 72 pixels.
    4. IoU = 24 / 72 = 0.333.

    `D.show_iou(box_a, box_b)` draws it on a grid of pixels: the intersection on the left, the union on the right.
    """),
    code("""
    D.show_iou((2, 2, 10, 8), (4, 4, 12, 10))
    """),
    md(r"""
    In the union picture the overlap is darker: it is covered by both boxes, which is why adding the two areas
    counts it twice.

    **Boxes that do not overlap.** Take A = (0, 0, 10, 10) and B = (20, 0, 30, 10), side by side. The recipe
    gives left = max(0, 20) = 20 and right = min(10, 30) = 10: a width of 10 - 20 = -10, which makes no sense.
    A negative width (or height) means "no overlap", so we replace it by 0 with `max(0, right - left)`: the
    intersection is 0 and so is the IoU. Python's `max(a, b)` and `min(a, b)` give the larger and the smaller
    of two numbers: `max(0, -10)` is 0, `max(0, 6)` is 6.

    ### Step 1 · Your turn: `box_iou_one`

    `box_iou_one(box_a, box_b)` receives two boxes, each four numbers (x1, y1, x2, y2), and returns their IoU, one
    number between 0 and 1. ("one" because it compares one pair of boxes.) The plan, in the order of the cell:

    1. take the four numbers of each box apart (given);
    2. the edges of the intersection: `left` and `top` are the **larger** of the two (max), `right` and
       `bottom` the **smaller** of the two (min);
    3. its width and height, kept at 0 or more with `max(0, ...)`, and its area `intersection`;
    4. the areas of the two boxes, and the `union`: both areas minus the intersection;
    5. `iou` = intersection / union.

    **Your turn.** The lines marked "the example" show the pattern; fill in the seven blanks and delete the
    `raise` line.
    """),
    exercise(
        stub='''
        def box_iou_one(box_a, box_b):
            """IoU of two boxes, each (x1, y1, x2, y2) in pixels: from 0 (no overlap) to 1 (the same box)."""
            raise NotImplementedError   # delete this line once every ____ is filled in
            ax1, ay1, ax2, ay2 = box_a          # 1. the four numbers of box a (given)
            bx1, by1, bx2, by2 = box_b          #    and of box b

            # 2. the edges of the intersection
            left = max(ax1, bx1)                # the example: the larger of the two left edges
            top = ____                          # YOUR CODE: the larger of the two top edges, ay1 and by1
            right = ____                        # YOUR CODE: the smaller of the two right edges, ax2 and bx2
            bottom = ____                       # YOUR CODE: the smaller of the two bottom edges, ay2 and by2
            # 3. its width, height and area (0 if the boxes do not overlap)
            width = max(0, right - left)        # the example
            height = ____                       # YOUR CODE: like width, with bottom and top
            intersection = width * height

            # 4. the two areas and the union
            area_a = (ax2 - ax1) * (ay2 - ay1)  # the example: width x height of box a
            area_b = ____                       # YOUR CODE: the same for box b
            union = ____                        # YOUR CODE: area_a + area_b - intersection

            # 5. the ratio
            iou = ____                          # YOUR CODE: intersection divided by union
            return iou
        ''',
        solution='''
        def box_iou_one(box_a, box_b):
            """IoU of two boxes, each (x1, y1, x2, y2) in pixels: from 0 (no overlap) to 1 (the same box)."""
            ax1, ay1, ax2, ay2 = box_a          # 1. the four numbers of box a (given)
            bx1, by1, bx2, by2 = box_b          #    and of box b

            # 2. the edges of the intersection
            left = max(ax1, bx1)                # the example: the larger of the two left edges
            top = max(ay1, by1)                 # YOUR CODE: the larger of the two top edges, ay1 and by1
            right = min(ax2, bx2)               # YOUR CODE: the smaller of the two right edges, ax2 and bx2
            bottom = min(ay2, by2)              # YOUR CODE: the smaller of the two bottom edges, ay2 and by2
            # 3. its width, height and area (0 if the boxes do not overlap)
            width = max(0, right - left)        # the example
            height = max(0, bottom - top)       # YOUR CODE: like width, with bottom and top
            intersection = width * height

            # 4. the two areas and the union
            area_a = (ax2 - ax1) * (ay2 - ay1)  # the example: width x height of box a
            area_b = (bx2 - bx1) * (by2 - by1)  # YOUR CODE: the same for box b
            union = area_a + area_b - intersection      # YOUR CODE: area_a + area_b - intersection

            # 5. the ratio
            iou = intersection / union          # YOUR CODE: intersection divided by union
            return iou
        '''),
    md(r"""
    **Check.** `T.check_box_iou_one(box_iou_one)` tries your function on the example above and on seven other
    pairs: the same box twice, a box moved by half its width, a small box inside a big one, three pairs that do
    not overlap, and a pair with decimal numbers. A `[FAIL]` line says which pair went wrong, what your function
    gave, and which part of the plan to look at.
    """),
    code("""
    T.check_box_iou_one(box_iou_one)
    """),
    md(r"""
    **What you built.** `box_iou_one = T.yours_or_reference(box_iou_one)` keeps your function if it passes (and
    otherwise uses the reference, with a note). `D.show_iou_pairs(box_iou_one)` draws four pairs of boxes, each
    with the IoU your function computes (green when right).
    """),
    code("""
    box_iou_one = T.yours_or_reference(box_iou_one)
    D.show_iou_pairs(box_iou_one)
    """),
    md(r"""
    **What to look at.** A box moved by only half its width already falls to IoU 0.33, below the 0.5 rule:
    IoU is strict. A small box inside a big one scores 0.11 although it lies completely inside: the union is the
    whole big box. IoU rewards boxes that have both the right place and the right size.
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · The simplest detector: one object per picture

    The hard part of detection is that the number of answers changes from picture to picture, while a network
    always gives the same number of outputs. So we start where that problem does not exist: pictures with
    **exactly one** object. Then the answer always has the same size: one class and one box. A network like the
    ones of P1 to P5 can give that, with two **heads** (the last layers, which turn what the network has read
    into its answer) instead of one:

    * the **class head** gives 3 **class scores**, one per class, as a classifier does;
    * the **box head** gives 4 numbers: the box.

    Both heads read the same numbers, made by the **body** (the convolution layers that read the picture).

    ### The data

    `D.one_object_set(n, seed)` makes `n` pictures of 64 x 64 pixels, each with one shape: an ellipse, a
    rectangle or a triangle, of random size (10 to 40 pixels), position and colour, on a grey background with a
    little noise. The colour says nothing about the class: the network has to look at the shape. Each picture
    comes with its label and its true box. We make 4 000 training pictures and 500 validation pictures
    (P1: pictures kept apart, to measure the network on pictures it has not learned from); `seed` makes them
    different from each other and the same on every run. `x.tolist()` turns a tensor into plain Python numbers,
    for printing. **Expected time: a few seconds.**
    """),
    api("Tensor.tolist"),
    code("""
    train_set = D.one_object_set(4000, seed=0)
    val_set = D.one_object_set(500, seed=1)
    print("training set:", train_set)
    print("validation set:", val_set)
    print("images:", tuple(train_set.images.shape), " labels:", tuple(train_set.labels.shape),
          " boxes:", tuple(train_set.boxes.shape))
    print("picture 0:", D.CLASSES[train_set.labels[0]], "with the box", train_set.boxes[0].tolist())
    D.show_set(train_set, n=16, title="the first 16 training pictures with their true boxes")
    """),
    md(r"""
    The pictures are a **tensor** (PyTorch's array of numbers) of shape 4000 x 3 x 64 x 64 (pictures x colour
    channels x height x width); the labels hold one
    class number per picture and the boxes four numbers per picture, in pixels. Picture 0 is an ellipse in the
    box (3, 29, 35, 57).

    **The box as fractions.** The network does not predict pixels directly but the box divided by the size of the
    picture, 64: four numbers between 0 and 1. The box (16, 8, 48, 40), for example, becomes
    (16/64, 8/64, 48/64, 40/64) = (0.25, 0.125, 0.75, 0.625). Small numbers in a fixed range are easier to learn,
    and the box head can be made to stay inside that range (step 2). To draw the box we multiply by 64 again.

    ### The body (given)

    The body is a small convolutional network like network B of P1: four blocks of a 3x3 convolution, a ReLU and
    a 2 x 2 max-pool, each halving height and width (64 -> 32 -> 16 -> 8 -> 4) while the number of **feature
    maps** (the output pictures of the filters) grows from 3 (the colours) to 8, 16, 32 and 64. `nn.Flatten()`
    then lays the 64 x 4 x 4 numbers out in a row: 1 024 numbers that describe the picture.

    Why `nn.Flatten` and not the global average pooling of P3 and P5? Averaging each map over the whole picture
    throws away **where** things are, and the box head needs exactly that. After `nn.Flatten`, each of the 4 x 4
    positions of every map keeps its own place in the row.
    """),
    api("nn.Sequential", "nn.Conv2d", "nn.ReLU", "nn.MaxPool2d", "nn.Flatten", brief=True),
    code('''
    def make_body():
        """The body: four convolution blocks, then Flatten. 3 x 64 x 64 picture -> 1024 numbers."""
        body = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),        # 8 x 64 x 64
            nn.ReLU(),
            nn.MaxPool2d(2),                                  # 8 x 32 x 32
            nn.Conv2d(8, 16, kernel_size=3, padding=1),       # 16 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(2),                                  # 16 x 16 x 16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),      # 32 x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2),                                  # 32 x 8 x 8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),      # 64 x 8 x 8
            nn.ReLU(),
            nn.MaxPool2d(2),                                  # 64 x 4 x 4
            nn.Flatten(),                                     # 1024 numbers in a row
        )
        return body


    P.show_model(make_body(), (3, 64, 64), title="the body: a 3 x 64 x 64 picture -> 1024 numbers")
    '''),

    # ------------------------------------------------------------------ step 2
    md(r"""
    ### Step 2 · Your turn: the two heads

    Each head is a small fully connected network (P1) that reads the 1 024 numbers of the body. For a batch of N
    pictures:

    | head | in | out |
    |---|---|---|
    | class head | N x 1024 numbers | N x 3 class scores (one per class) |
    | box head | N x 1024 numbers | N x 4 box numbers between 0 and 1 (x1/64, y1/64, x2/64, y2/64) |

    **The class head**, `make_class_head(in_features, num_classes)`:

    | # | layer | in | out |
    |---|---|---|---|
    | 1 | `nn.Linear` | `in_features` | 64 |
    | 2 | `nn.ReLU` | | |
    | 3 | `nn.Linear` | 64 | `num_classes` |

    It ends with the scores themselves, as every classifier of P1 to P5: no softmax (the loss deals with that).

    **The box head**, `make_box_head(in_features)`:

    | # | layer | in | out |
    |---|---|---|---|
    | 1 | `nn.Linear` | `in_features` | 64 |
    | 2 | `nn.ReLU` | | |
    | 3 | `nn.Linear` | 64 | 4 |
    | 4 | `nn.Sigmoid` | | |

    **The sigmoid** is new. `nn.Sigmoid()` squeezes every number into the range 0 to 1:
    sigmoid(x) = 1 / (1 + e<sup>-x</sup>). A large negative number becomes almost 0, 0 becomes exactly 0.5, a
    large positive number becomes almost 1: sigmoid(-4) = 0.018, sigmoid(0) = 0.5, sigmoid(4) = 0.982. Our box
    numbers are fractions of the picture's size, between 0 and 1, so the box head ends with a sigmoid: whatever
    the layers before it compute, the box stays inside the picture.

    Write the names `in_features` and `num_classes`, not the numbers 1024 and 3: the check also builds the heads
    with other numbers. For a table row such as `nn.Linear` from `in_features` to 64 you write
    `nn.Linear(in_features, 64)`.

    **Your turn.** The first layer of the class head is the example; fill in the six blanks and delete the two
    `raise` lines.
    """),
    api("nn.Linear", brief=True),
    api("nn.Sigmoid"),
    exercise(
        stub='''
        def make_class_head(in_features, num_classes):
            """The class head: in_features numbers -> one score per class."""
            raise NotImplementedError   # delete this line once every ____ is filled in
            head = nn.Sequential(
                nn.Linear(in_features, 64),      # 1: the example
                ____,                            # YOUR CODE: 2, nn.ReLU()
                ____,                            # YOUR CODE: 3, nn.Linear from 64 to num_classes
            )
            return head


        def make_box_head(in_features):
            """The box head: in_features numbers -> 4 numbers between 0 and 1 (x1, y1, x2, y2 divided by 64)."""
            raise NotImplementedError   # delete this line once every ____ is filled in
            head = nn.Sequential(
                ____,                            # YOUR CODE: 1, nn.Linear from in_features to 64
                ____,                            # YOUR CODE: 2, nn.ReLU()
                ____,                            # YOUR CODE: 3, nn.Linear from 64 to 4
                ____,                            # YOUR CODE: 4, nn.Sigmoid(): keeps the 4 numbers between 0 and 1
            )
            return head
        ''',
        solution='''
        def make_class_head(in_features, num_classes):
            """The class head: in_features numbers -> one score per class."""
            head = nn.Sequential(
                nn.Linear(in_features, 64),      # 1: the example
                nn.ReLU(),                       # YOUR CODE: 2, nn.ReLU()
                nn.Linear(64, num_classes),      # YOUR CODE: 3, nn.Linear from 64 to num_classes
            )
            return head


        def make_box_head(in_features):
            """The box head: in_features numbers -> 4 numbers between 0 and 1 (x1, y1, x2, y2 divided by 64)."""
            head = nn.Sequential(
                nn.Linear(in_features, 64),      # YOUR CODE: 1, nn.Linear from in_features to 64
                nn.ReLU(),                       # YOUR CODE: 2, nn.ReLU()
                nn.Linear(64, 4),                # YOUR CODE: 3, nn.Linear from 64 to 4
                nn.Sigmoid(),                    # YOUR CODE: 4, nn.Sigmoid(): keeps the 4 numbers between 0 and 1
            )
            return head
        '''),
    md(r"""
    **Check.** Both heads are built with the numbers of our detector (1 024 numbers in, 3 classes) and looked at
    layer by layer, then run on 2 rows of random numbers. The box check uses very large numbers, which only a
    sigmoid keeps between 0 and 1. The last line of each builds the head again with other numbers, to catch
    numbers typed where names belong.
    """),
    code("""
    T.check_heads(make_class_head, make_box_head)
    """),
    md(r"""
    **What you built.** The two heads, drawn with the 1 024 numbers of the body as their input.
    """),
    code("""
    make_class_head = T.yours_or_reference(make_class_head)
    make_box_head = T.yours_or_reference(make_box_head)
    P.show_model(make_class_head(1024, 3), (1024,), title="your class head: 1024 numbers -> 3 class scores")
    P.show_model(make_box_head(1024), (1024,), title="your box head: 1024 numbers -> 4 box numbers between 0 and 1")
    """),

    # ------------------------------------------------------------------ the whole detector
    md(r"""
    ### The whole detector (given)

    `TinyDetector` puts the body and your two heads together, as a class like the Inception block of P3:
    `__init__` builds the parts, `forward` says what happens to a batch of pictures `x`. The body reads the
    pictures once; both heads read its 1 024 numbers; `forward` returns **two** results, the class scores and
    the box.
    """),
    api("nn.Module", "torch.manual_seed", brief=True),
    code('''
    class TinyDetector(nn.Module):
        """One object per picture: a body that reads the picture, a class head and a box head."""

        def __init__(self, num_classes=3):
            super().__init__()
            self.body = make_body()                                   # 3 x 64 x 64 -> 1024 numbers
            self.class_head = make_class_head(1024, num_classes)      # 1024 numbers -> 3 class scores (step 2)
            self.box_head = make_box_head(1024)                       # 1024 numbers -> 4 box numbers (step 2)

        def forward(self, x):
            features = self.body(x)
            class_scores = self.class_head(features)
            box = self.box_head(features)
            return class_scores, box


    torch.manual_seed(0)                    # the same random starting weights on every run
    detector = TinyDetector()
    class_scores, box = detector(train_set.images[:2])        # the first two training pictures
    print("class scores:", tuple(class_scores.shape), " box:", tuple(box.shape))
    print("box of picture 0 before training, as fractions:", box[0].tolist())
    P.show_model(detector, (3, 64, 64), title="TinyDetector: one body, two heads")
    '''),
    md(r"""
    **Reading the drawing.** After `Flatten` the path splits in two: the class head on top, the box head below.
    Both get the same 1 024 numbers. (The output box at the end shows the size of the first result only, the 3
    class scores; the network returns both.) The detector has 156 183 parameters, more than four fifths of them
    in the two heads (65 795 + 65 860): the first layer of each head connects all 1 024 numbers to 64 neurons.

    Before training, the four box numbers of picture 0 are all close to 0.5 (0.478, 0.533, 0.489, 0.509): the
    sigmoid of numbers close to 0. As pixels that is a box of almost no size in the middle of the picture. The
    network does not know anything yet.

    ### Training (given)

    Training works as in P1 (an **epoch** is one pass over all training pictures, in batches of 64; after each
    batch every weight changes a little), with one difference: there are two answers, so there are two
    **losses** (numbers that say how wrong an answer is), added together:

    * the **class loss**: cross-entropy, the loss of every classifier of P1 to P5;
    * the **box loss**: the **L1 loss**, the average distance between the 4 predicted numbers and the 4 true
      ones. For example, predicted (0.30, 0.10, 0.70, 0.60) against true (0.25, 0.125, 0.75, 0.625): the
      distances are 0.05, 0.025, 0.05 and 0.025, their average 0.0375, which is 0.0375 x 64 = 2.4 pixels per edge.

    `loss = class loss + box_weight x box loss`. The box loss is a small number (2.4 pixels is only 0.0375), so
    `box_weight = 5` makes it count about as much as the class loss.

    <details>
    <summary><b>What <code>D.train_detector</code> does inside</b> (optional, click to open)</summary>

    The same loop as `P.train` in P1, with two losses:

    ```python
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        for images, labels, boxes in batches_of_64:           # boxes in pixels
            class_scores, predicted_boxes = model(images)
            class_loss = F.cross_entropy(class_scores, labels)
            box_loss = F.l1_loss(predicted_boxes, boxes / 64)  # both as fractions of the picture
            loss = class_loss + box_weight * box_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # then: measure the validation pictures
    ```
    </details>

    After every epoch it measures the 500 validation pictures: the percentage with the right class, and the
    **mean IoU**, the average IoU between predicted and true box.

    ### Try it: the run settings

    Run this cell, then the training cell. Change them later and run both again.
    """),
    code("""
    # ---------------- Try it: run settings ----------------
    epochs = 8                  # passes over the training pictures
    learning_rate = 0.002       # size of each weight change
    box_weight = 5              # how much the box loss counts against the class loss
    """),
    md(r"""
    `D.train_detector(...)` starts from new random weights (`torch.manual_seed(0)` and a new `TinyDetector`), so
    every run starts from the same point. It prints one line per epoch (the seconds since the start at the end of
    each line) and draws the curves at the end. **Expected time: about a minute** (from about 15 to 45 seconds
    on our computer, depending on what else was running; up to twice as long on a slow laptop).
    """),
    code("""
    torch.manual_seed(0)
    detector = TinyDetector()
    history = D.train_detector(detector, train_set, val_set, epochs=epochs, lr=learning_rate, box_weight=box_weight)
    """),
    md(r"""
    **Reading the curves.** The box loss falls first: from 0.120 to 0.050 in two epochs, so the network learns
    early roughly where the shape is. The class loss stays high for two epochs (65.4% right after epoch 2, where
    guessing gives 33%), then drops: in epoch 3 the accuracy jumps to 95.2%, and from epoch 4 on almost every
    picture gets the right class (99.8% after epoch 8). The mean IoU climbs more slowly and less evenly, from
    0.545 after the first epoch to 0.831 after the eighth, and is still rising at the end: more epochs would make
    the boxes a little tighter. A box loss of 0.0179 in the last epoch means that the predicted edges of the
    training pictures are on average 0.0179 x 64 = about 1 pixel away from the true ones.

    `D.evaluate_detector(model, dataset)` measures the validation pictures once more and adds a third number,
    **found**: the right class and an IoU of at least 0.5, the rule of section 2. `D.show_detector_predictions`
    draws the true box (black, dashed) and the predicted box (red) on the first 12 validation pictures, with the
    predicted class, how sure the network is, and the IoU.
    """),
    code("""
    D.evaluate_detector(detector, val_set)
    D.show_detector_predictions(detector, val_set, n=12)
    """),
    md(r"""
    **What to look at.** Of the 500 validation pictures, 99.8% get the right class and 99.4% count as found
    (right class and IoU at least 0.5): only 3 pictures are not found. All 12 pictures above have the right
    class, each at 100%, and the red boxes sit close to the dashed true ones, with IoU between 0.80 and 0.93.
    Where they differ, it is a pixel or two on one or two edges; the worst is the first picture (IoU 0.80), a
    triangle whose predicted box sits a little too low and misses the thin tip. That is what a mean IoU of 0.831
    looks like: good boxes, not perfect ones.

    **Things to try** (change the **Try it** cell, predict first, then run the training cell and these two):

    * `box_weight = 0`: the box loss no longer counts. What happens to the class accuracy, and to the mean IoU?
    * `box_weight = 50`: the box loss counts ten times more. Do the boxes get better, and the classes worse?
    * `epochs = 15`: is it worth the time?

    <details>
    <summary><b>What we saw with <code>box_weight</code> 0 and 50</b> (try first, then click to open)</summary>

    * `box_weight = 0`: the classes are learned faster (99.4% right after epoch 3, against 95.2%) and end at
      99.8%, because the network spends all its effort on them. The boxes are not learned at all: mean IoU
      0.000, and not one of the 500 pictures counts as found. The box head gets no feedback, so its four numbers
      never come to mean anything. A detector needs both losses.
    * `box_weight = 50`: the boxes get tighter (mean IoU 0.868 after 8 epochs) but the classes come slowly
      (74.4% right after epoch 4, 98.6% at the end), so fewer pictures count as found (98.4%). The weight decides
      what the network works on first.
    </details>
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · Several objects per picture: a grid detector

    With several objects the number of answers changes from picture to picture, but a network always gives
    the same number of outputs. The trick of many real detectors (YOLO, SSD, FCOS, ...): cut the picture into a
    **grid** of **cells** and let **every cell** give a fixed set of numbers. Ours cuts the 64 x 64 picture into
    8 x 8 cells of 8 x 8 pixels, and every cell gives 8 numbers:

    * 1 **object score**: "is this cell on an object?", turned into a number from 0 to 1 by a sigmoid;
    * 3 class scores: "which shape?";
    * 4 **distances** from the centre of the cell to the left, top, right and bottom edges of the object's box
      (as fractions of 64, through a sigmoid, like the box head of step 2).

    So the network always gives 8 x 8 cells x 8 numbers, that is 64 **candidate boxes**, one per cell, each with
    a score. Most cells lie on the background and should give a low score.

    **Which cells must find an object?** During training, a cell is **responsible** for an object when the
    centre of the cell lies inside the object's box. Its object score should be 1, its class that of the object,
    and its four distances those to the edges of the object's box. Every other cell should give an object score
    of 0. The drawing shows this for the picture of section 1: on the left the responsible cells (shaded), on
    the right one cell with its four distances.
    """),
    code("""
    D.show_grid_idea(picture, boxes, labels)
    """),
    md(r"""
    A box from a cell's numbers: with the cell's centre at (cx, cy), the box is
    (cx - left, cy - top, cx + right, cy + bottom). For the cell on the right: (20 - 14, 12 - 8, 20 + 15, 12 + 15)
    = (6, 4, 35, 27), the box of the triangle. Note that **all** the shaded cells of an object describe the same
    box, each from its own centre: one object, several candidate boxes. We come back to this below.

    ### The data and the network

    `D.many_objects_set(n, seed)` makes pictures with 1, 2 or 3 shapes each (10 to 28 pixels), which never
    overlap. **Expected time: a few seconds.**
    """),
    code("""
    many_train = D.many_objects_set(4000, seed=0)
    many_val = D.many_objects_set(500, seed=1)
    print("training set:", many_train)
    print("validation set:", many_val)
    D.show_set(many_train, n=16, title="pictures with 1 to 3 shapes, with their true boxes")
    """),
    md(r"""
    The network is given. It starts like the body of section 3, but stops shrinking at 8 x 8, which is our grid:
    each of the 8 x 8 positions of its last maps belongs to one cell. Two more 3x3 convolutions at 8 x 8 let
    every cell see far enough around itself to judge a whole shape. The last layer is a **1x1 convolution**
    (P3: it mixes the maps at each position) with 8 output maps: the 8 numbers of every cell. There is no
    `Flatten` and no linear layer: the answers stay arranged as the grid.
    """),
    code('''
    def make_grid_net():
        """The grid detector: a 3 x 64 x 64 picture -> 8 numbers for each of the 8 x 8 cells."""
        net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),        # 8 x 64 x 64
            nn.ReLU(),
            nn.MaxPool2d(2),                                  # 8 x 32 x 32
            nn.Conv2d(8, 16, kernel_size=3, padding=1),       # 16 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(2),                                  # 16 x 16 x 16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),      # 32 x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2),                                  # 32 x 8 x 8: one position per cell
            nn.Conv2d(32, 64, kernel_size=3, padding=1),      # 64 x 8 x 8
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),      # 64 x 8 x 8
            nn.ReLU(),
            nn.Conv2d(64, 8, kernel_size=1),                  # 8 x 8 x 8: 8 numbers for each of the 8 x 8 cells
        )
        return net


    P.show_model(make_grid_net(), (3, 64, 64), title="the grid detector: 8 numbers for each of the 8 x 8 cells")
    '''),
    md(r"""
    ### Training (given)

    The loss has three parts, added up:

    * the **object loss**, for every cell: is there an object here, yes or no? (binary cross-entropy: the loss
      for yes-or-no answers);
    * the class loss, for the responsible cells only: cross-entropy, as in section 3;
    * the box loss, for the responsible cells only: the L1 loss of the 4 distances, times 5 as in section 3.

    After every epoch, `D.train_grid_detector` counts on the 500 validation pictures how many shapes were
    **found** (a box of the right class with IoU at least 0.5, after the clean-up explained below) and how many
    **extra** boxes remain (boxes that match no shape). Training starts from new random weights, and the curves
    are drawn at the end. **Expected time: about a minute** (from about 30 to 50 seconds on our computer; each
    line ends with the seconds since the start).
    """),
    code("""
    # ---------------- Try it: run settings ----------------
    grid_epochs = 10
    grid_learning_rate = 0.002

    torch.manual_seed(0)                          # the same random starting weights on every run
    grid_net = make_grid_net()
    grid_history = D.train_grid_detector(grid_net, many_train, many_val, epochs=grid_epochs, lr=grid_learning_rate)
    """),
    md(r"""
    **Reading the curves.** After the first epoch the network finds nothing at all (0 of 982 shapes, and not
    one box): it has first learned that most cells are background, the easiest way to lower the object loss. In
    epoch 2 it finds a third of the shapes (322) but also gives 886 extra boxes; from then on both numbers get
    better. After 10 epochs it finds 977 of the 982 shapes (99.5%), with 34 extra boxes on 500 pictures. Epoch 8
    is a small step back (972 found after 974): training does not improve every single epoch.

    ### From 64 candidates to the final boxes: score threshold and NMS

    Take one validation picture and look at what the network says for it, before any clean-up.
    `D.grid_candidates(grid_net, picture)` runs the network and turns the numbers of every cell into a box:
    64 boxes (one per cell, row by row, in pixels), 64 scores and 64 labels. To print the scores as the grid,
    `cand_scores.reshape(8, 8)` arranges the 64 numbers in 8 rows of 8.
    """),
    api("Tensor.reshape"),
    code("""
    picture_index = 4
    picture = many_val.images[picture_index]
    cand_boxes, cand_scores, cand_labels = D.grid_candidates(grid_net, picture)
    print("boxes:", tuple(cand_boxes.shape), " scores:", tuple(cand_scores.shape), " labels:", tuple(cand_labels.shape))
    print("the 64 scores, one row of the grid per line:")
    grid_scores = cand_scores.reshape(8, 8)
    for row in range(8):
        line = ""
        for col in range(8):
            line = line + f"{grid_scores[row, col]:6.2f}"
        print(line)
    """),
    md(r"""
    Most scores are 0.00: background cells. Three groups of high scores are the three shapes of this picture:
    a big triangle (rows 2 and 3, columns 1 to 3, counting from 0), a small triangle (rows 2 to 4, columns 5
    and 6) and a rectangle (rows 5 and 6, columns 3 to 6). Some cells at the edge of a shape are unsure (0.68,
    0.49, 0.43, 0.38).

    **Step A: the score threshold.** Keep only the boxes whose score is above a **score threshold**, here 0.5.
    `cand_scores > score_threshold` gives one True or False per box; indexing a tensor with it,
    `cand_boxes[keep]`, keeps the rows where it is True.
    """),
    code("""
    score_threshold = 0.5
    keep = cand_scores > score_threshold                 # True for the boxes we keep
    sure_boxes = cand_boxes[keep]
    sure_scores = cand_scores[keep]
    sure_labels = cand_labels[keep]
    print(len(sure_boxes), "of 64 boxes have a score above", score_threshold)
    D.show_before_after(picture, cand_boxes, cand_labels, cand_scores, keep,
                        "all 64 candidates (faint = low score)", "score above 0.5")
    """),
    md(r"""
    20 of the 64 boxes remain and the background boxes are gone, but each shape still has several boxes: every
    cell on it gave one. They are **duplicates**: several boxes for one object.

    **Step B: non-maximum suppression (NMS)** removes them, using the IoU of section 2:

    1. sort the boxes by score, best first;
    2. keep the best box;
    3. remove every other box whose IoU with it is above an **IoU threshold** (here 0.5): it describes the same
       object;
    4. repeat with the boxes that remain, until none is left.

    The name says what it does: everything that is not the maximum (the best box) of its group is suppressed.
    `torchvision.ops.nms(boxes, scores, iou_threshold)` does exactly this and returns the positions of the kept
    boxes, best first.
    """),
    api("torchvision.ops.nms"),
    code("""
    iou_threshold = 0.5
    kept = torchvision.ops.nms(sure_boxes, sure_scores, iou_threshold)
    print("NMS keeps the boxes at positions", kept.tolist(), "of", len(sure_boxes))
    final_boxes = sure_boxes[kept]
    final_labels = sure_labels[kept]
    final_scores = sure_scores[kept]
    for k in range(len(final_boxes)):
        x1, y1, x2, y2 = final_boxes[k].tolist()
        name = D.CLASSES[final_labels[k]]
        print(f"{name:<10} score {final_scores[k]:.2f}  box ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
    print("the true boxes:", many_val.boxes[picture_index].tolist())
    D.show_before_after(picture, sure_boxes, sure_labels, sure_scores, kept, "before NMS", "after NMS")
    """),
    md(r"""
    **The result**: 3 boxes out of 20, one per shape, the best first: the list a detector should return. NMS kept
    the boxes at positions 2, 14 and 8 of the 20 and removed the other 17 as duplicates. Every edge of the three
    boxes lies within 2 pixels of the true box (for example the small triangle: (36, 15, 55, 40) against the
    true (38, 16, 54, 39)).

    ### Try it: the two thresholds

    `D.show_grid_results(net, dataset, score_threshold, iou_threshold)` does steps A and B for the first 8
    validation pictures and draws the final boxes (dashed black: the true boxes). `D.evaluate_grid` counts over all
    500 validation pictures. Change the two thresholds and run the cell again:

    * `score_threshold = 0.1`: more boxes survive step A. Do extra boxes appear?
    * `score_threshold = 0.9`: only very sure boxes. Are shapes missed?
    * `iou_threshold = 0.9`: NMS removes only near-identical boxes. Do duplicates stay?
    """),
    code("""
    # ---------------- Try it: the thresholds ----------------
    score_threshold = 0.5      # step A: keep the boxes with a score above this
    iou_threshold = 0.5        # step B (NMS): remove boxes that overlap a better one by more than this

    D.show_grid_results(grid_net, many_val, score_threshold=score_threshold, iou_threshold=iou_threshold)
    D.evaluate_grid(grid_net, many_val, score_threshold=score_threshold, iou_threshold=iou_threshold)
    """),
    md(r"""
    **What we saw.** With both thresholds at 0.5, every shape of the 8 pictures is found, with no extra box.
    Over all 500 validation pictures: 977 of the 982 shapes found, 34 extra boxes. An extra box matches no
    shape: it is a second box on a shape that NMS kept (its IoU with the better box stayed below 0.5), a box
    with the wrong class, or a box too far off (IoU below 0.5).

    <details>
    <summary><b>What we saw with other thresholds</b> (try first, then click to open)</summary>

    | `score_threshold` | `iou_threshold` | shapes found (of 982) | extra boxes |
    |---|---|---|---|
    | 0.5 | 0.5 | 977 | 34 |
    | 0.1 | 0.5 | 979 | 90 |
    | 0.9 | 0.5 | 970 | 18 |
    | 0.5 | 0.9 | 979 | 2 497 |
    | 0.5 | 0.3 | 975 | 11 |

    A low score threshold finds 2 more shapes but gives almost three times as many extra boxes; a high one
    halves the extra boxes but misses 7 more shapes. An IoU threshold of 0.9 almost switches NMS off: the
    duplicates stay (2 497 extra boxes, 5 per picture). 0.3 removes duplicates more strictly, which works here
    because our shapes never overlap; in photos where objects do overlap (people in a crowd), too low an IoU
    threshold also removes the boxes of different objects. Choosing the two thresholds is a trade-off between
    missed objects and extra boxes.
    </details>
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · A real detector on real photos

    Real detectors work on the same ideas: a body that reads the photo, heads that give classes and boxes, a
    score for every box, a score threshold and NMS at the end. They are bigger and trained much longer.
    **Faster R-CNN** (Ren et al., 2015, here in an improved version of torchvision) is one of the best known.
    torchvision has it with weights trained on **COCO**: about 118 000 photos in which people marked every object
    of 80 kinds (person, bicycle, car, dog, cat, chair, bottle, ...) with a box. Training it takes days on
    graphics cards; using it costs one line.

    `D.load_coco_detector()` builds it with these weights from the file in PyTorch's cache folder
    (`fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`, 167 MB; if it is missing, the cell says how to get it and the
    rest of this section is skipped). Inside, these are the two torchvision lines:

    ```python
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    detector = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
    ```

    It also returns `categories`, the class names by label number: `categories[1]` is "person", `categories[18]`
    "dog". COCO numbers its 80 kinds from 1 to 90 and leaves some numbers unused (their name is "N/A"); 0 means
    background. **Expected time: a few seconds.**
    """),
    api("torchvision.models.detection.fasterrcnn_resnet50_fpn_v2",
        "torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights"),
    code("""
    detector, categories = D.load_coco_detector()
    """),
    md(r"""
    **The photos.** `D.load_photo("street", index)` reads one of the 170 street photos of the **Penn-Fudan**
    collection (people walking in a city), and returns it as a tensor of 3 x height x width with values from 0 to
    1, together with the boxes of the people that the authors of the collection marked by hand.
    `D.load_photo("pets", index)` reads one of the 7 390 cat and dog photos of P5 (no boxes).

    **Running the detector.** `D.detect(detector, photo)` runs it on one photo. Inside, it is three lines, which
    is all you need in your own code:

    ```python
    detector.eval()                          # evaluation mode: predict, do not train
    with torch.no_grad():                    # only answering, no training
        outputs = detector([photo])          # a LIST of photos in, one result per photo out
    ```

    The detector resizes the photo itself, so photos of any size work. **Expected time: a few seconds** for one
    photo on a CPU (about 3 seconds on our computer, up to about 20 seconds on a slow or busy one; a graphics
    card does it in a fraction of a second). The 43.7 million parameters of the detector, against the 156 183 of our
    `TinyDetector`, are the reason.
    """),
    code("""
    photo, people = D.load_photo("street", 3)
    result = D.detect(detector, photo)
    """),
    md(r"""
    **The raw result** is a dictionary (Python's `dict`: values looked up by name) with three tensors, one row per
    box, best score first: `"boxes"` (x1, y1, x2, y2 in pixels of the photo), `"labels"` (class numbers, COCO's
    numbering) and `"scores"`. `D.print_result` prints the first rows, with the name of each label.
    `result["boxes"][:8]` would print the first 8 boxes alone, as a tensor.
    """),
    code("""
    D.print_result(result, categories, n=8)
    """),
    md(r"""
    31 boxes for one photo, best first: three people with a score of 0.995 or more, then handbags (label 31),
    another person at 0.779, and 23 more rows with lower scores. The detector has already applied its own NMS
    and a very low score threshold (0.05): it returns everything it finds at least a little likely, and leaves
    the choice to us. `D.show_photo_detections` draws the boxes with a score of at least `score_threshold`, next
    to the boxes the collection marked by hand.
    """),
    code("""
    D.show_photo_detections(photo, result, categories, score_threshold=0.5, true_boxes=people)
    """),
    md(r"""
    **What to look at.** 9 boxes have a score of at least 0.5. The detector finds the woman in front (person,
    1.00) and the two people at the left edge (1.00 and 0.99), where the collection marked only one. Behind the
    woman walks a second person, mostly hidden; the detector gives two weaker boxes there (0.78 and 0.58), the
    second one around only a part of that person. It also finds four handbags (0.96, 0.90, 0.86 and 0.74):
    handbag is one of COCO's 80 kinds, while the collection marks only people. A detector finds what it was
    trained to find, and the boxes people mark by hand are not complete either.

    ### Try it: other photos, other thresholds

    * `photo_set`: `"street"` (170 photos) or `"pets"` (7 390 photos);
    * `photo_index`: which photo (0, 1, 2, ...);
    * `score_threshold`: try 0.9 (only very sure boxes) and 0.2 (more boxes, more mistakes).

    Each run takes as long as the cell above.
    """),
    code("""
    # ---------------- Try it: a photo and a threshold ----------------
    photo_set = "street"         # "street" or "pets"
    photo_index = 10             # which photo
    score_threshold = 0.5        # draw the boxes with at least this score

    photo, people = D.load_photo(photo_set, photo_index)
    result = D.detect(detector, photo)
    D.print_result(result, categories, n=5)
    D.show_photo_detections(photo, result, categories, score_threshold=score_threshold, true_boxes=people)
    """),
    md(r"""
    **What we saw.** Street photo 10: the woman (person, 1.00), her bag twice (handbag 0.99 for the whole bag
    with its strap, 0.51 for its lower part: two boxes on one object that NMS kept, because they overlap too
    little) and a small "cell phone" (0.65) near her hand.

    <details>
    <summary><b>Two pet photos we tried</b> (try first, then click to open)</summary>

    * `photo_set = "pets"`, `photo_index = 0`, an Abyssinian cat lying on a couch: cat 0.99, couch 0.78 and
      0.53. With `score_threshold = 0.2` a fourth box appears: "dog" 0.26, around the same cat.
    * `photo_set = "pets"`, `photo_index = 1000`, a spotted Bengal cat standing on its hind legs: cat 0.98, and
      "giraffe" 0.91 around the same animal, a very confident mistake (a tall animal with spots). With
      `score_threshold = 0.9` exactly these two boxes remain: a high threshold removes doubtful boxes, not
      confident mistakes.
    </details>
    """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · What to remember

    * **Object detection** finds every object (instance) in a picture: for each one a **box**, a **class** and a
      **score**. The answer is a list whose length changes from picture to picture.
    * A box is four numbers **(x1, y1, x2, y2)** in pixels: top-left and bottom-right corner, y downwards;
      width = x2 - x1, height = y2 - y1.
    * **IoU** = intersection / union says how well two boxes match: 1 for the same box, 0 for no overlap. A
      predicted box with IoU at least 0.5 and the right class counts as found.
    * With one object per picture, a detector is a classifier with a second head: the **box head** gives 4
      numbers between 0 and 1 (a **sigmoid** at the end), trained with an **L1 loss** next to the class loss.
    * With several objects, a **grid detector** lets every cell give an object score, class scores and a box:
      always the same number of outputs, whatever the number of objects.
    * Many candidate boxes become the final answer in two steps: a **score threshold** removes unlikely boxes,
      **non-maximum suppression** removes duplicates (boxes with a high IoU with a better box).
    * Real detectors such as **Faster R-CNN** work on the same ideas; pretrained on **COCO** they find 80 kinds of
      objects in any photo: `detector.eval()`, `torch.no_grad()`, `detector([photo])`.

    ### Words from this notebook

    | word | meaning |
    |---|---|
    | 1x1 convolution | a convolution with one-pixel filters: mixes the maps at each position (P3) |
    | area | width x height of a box, in pixels |
    | body | the layers that read the picture, shared by the heads |
    | box (bounding box) | the smallest rectangle around an object: (x1, y1, x2, y2) |
    | candidate boxes | all the boxes a detector proposes before the clean-up |
    | cell | one square of the grid; here 8 x 8 pixels |
    | class head | the head that gives one score per class |
    | box head | the head that gives the 4 numbers of the box |
    | COCO | about 118 000 photos with boxes around every object of 80 kinds |
    | detector | a network that does object detection |
    | duplicates | several boxes for the same object |
    | extra box | a final box that matches no true object |
    | Faster R-CNN | a well-known detector (2015); torchvision has it pretrained on COCO |
    | found | a predicted box with the right class and IoU at least 0.5 with a true box |
    | grid detector | a detector in which every cell of a grid gives a score, classes and a box |
    | head | the last layers of a network, which turn what it has read into an answer |
    | instance | one particular object: two people are two instances of the class "person" |
    | intersection | the area two boxes share |
    | IoU | intersection over union: how well two boxes match, 0 to 1 |
    | IoU threshold | in NMS, the IoU above which a box counts as a duplicate of a better one |
    | L1 loss | the average distance between predicted and true numbers |
    | label | the class number of an object |
    | mean IoU | the average IoU between predicted and true boxes |
    | NMS (non-maximum suppression) | keep the best box, remove the boxes that overlap it too much, repeat |
    | object detection | finding every object in a picture: box, class and score for each |
    | object score | a cell's answer to "is there an object here?", 0 to 1 |
    | Penn-Fudan | 170 street photos with the people marked by hand |
    | responsible cell | a cell whose centre lies inside an object's box: it must find that object |
    | score | how sure the detector is about a box, 0 to 1 |
    | score threshold | the score below which boxes are thrown away |
    | sigmoid | squeezes any number into 0 to 1: 1 / (1 + e<sup>-x</sup>) |
    | union | the area covered by at least one of two boxes |
    | xyxy | writing a box as its two corners (x1, y1, x2, y2) |
    """),
]
