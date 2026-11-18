from nb import md, code, exercise, api

NAME = "P8-vision-transformer"

CELLS = [
    md(r"""
    # P8 · Vision transformers: cut the picture into patches and let the patches talk

    **What we do today**

    1. Look at the idea of a **vision transformer** (ViT, Dosovitskiy et al., 2020): instead of sliding small
       filters over the picture, cut it into patches and let every patch exchange information with every other
       patch.
    2. Turn one hand-written digit into a table of numbers, step by step, with a picture for every step. Then
       write the layer that does it yourself (step (a)).
    3. Meet **attention**, the heart of every transformer: first by hand with three tokens of two numbers, then
       with PyTorch's attention layer on the digit.
    4. Assemble a tiny vision transformer from ready-made blocks (step (b)).
    5. Train it on MNIST (about a minute), compare it with the small CNN of P1, and look inside: where does it
       look?
    6. Optional: a real vision transformer, trained by others on 1.28 million photos, names the breed of a few
       pet photos.

    **How to use this notebook.** Run the cells from top to bottom with **Shift+Enter**. You type code only in
    the two cells headed **Your turn**. In them, every `____` (four underscores) is a blank: replace it with
    what its `# YOUR CODE:` comment describes, copying from the table above the cell. These cells contain lines
    `raise NotImplementedError`, which stop the code with the message "not implemented"; delete them once every
    blank is filled in.

    You cannot get stuck. If a step is not finished, its check says what is missing, and the next cells
    continue with the course's own **reference** version of that step and say so ("note: using the reference
    ..."). You can even run the whole notebook before typing anything. The finished cells are in
    `solutions/P8-vision-transformer-solution.ipynb`.

    * `P.something(...)` is a function of the course toolkit (P1 introduced it). `V.something(...)` draws a
      figure of this notebook or does a small job for it. `T.something(...)` is a check.
    * Closed boxes **Functions used here (click to open)** explain the PyTorch functions of the next cell.
    * **New words** are in bold where they are explained, and collected at the end.
    * Numbers quoted in the text are what the cells print. On your computer the last digit can differ a little,
      and times differ from computer to computer.
    """),

    # ------------------------------------------------------------------ 0
    md(r"""
    ## 0 · Setup

    As in P1, this cell finds the course's helper folder `cvcourse` and loads PyTorch (`torch`, with its layers
    in `torch.nn`, short `nn`) and the toolkit `P`. New today:

    * `V`: the figures of this notebook;
    * `T`: the checks of the two steps, and `T.yours_or_reference(...)`, which keeps your version of a step if
      it passes its checks and otherwise hands over the reference version;
    * `reference`: the course's own finished code.

    `P.setup(seed=0)` fixes the **random seed**, so that random numbers repeat from run to run. Then we load
    MNIST, the hand-written digits of P1 (10 000 training and 2 000 validation pictures of 28 x 28 pixels), and
    take one digit to work with: validation picture number 4, a 3.
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
    from cvcourse import pkit_p8 as V
    from cvcourse import ptests_p8 as T
    from cvcourse import pmodels_p8 as reference

    P.setup(seed=0)
    train_loader, val_loader, classes = P.get_mnist(n_train=10000, n_val=2000, batch_size=64)
    image, label = val_loader.dataset[4]            # one digit: a picture and its label
    print("the digit:", image.shape, " its label:", int(label))
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · The idea in one picture

    In P1 to P5 every network was a **convolutional network** (CNN): small filters of 3 x 3 pixels slide over
    the picture. One filter sees only a small neighbourhood. The network sees the whole digit only after several
    layers and poolings, each widening the view a little.

    A **vision transformer** (**ViT**) reads a picture the way a language model (such as the ones behind chat
    assistants) reads a sentence. A language model cuts a sentence into words, turns every word into a list of
    numbers, and lets every word look at every other word to work out what the sentence means. A ViT does the
    same with a picture:

    1. it cuts the picture into **patches**, small squares, here 7 x 7 pixels: a 28 x 28 digit gives
       4 x 4 = 16 patches. The patches are the "words" of the picture;
    2. it turns every patch into a list of numbers of a fixed length, its **embedding**. One such list is
       called a **token**;
    3. **attention** layers let every token look at every other token, anywhere in the picture, and take in
       information from the ones that matter to it. This is repeated a few times;
    4. one extra token, the **class token**, collects what the network needs for its answer, and a linear layer
       turns it into 10 scores, one per digit.

    `V.show_idea(image)` draws the difference between the two kinds of network.
    """),
    code("""
    V.show_idea(image)
    """),
    md(r"""
    **Reading the picture.** Left: a 3 x 3 filter of P1 sees 9 neighbouring pixels at a time. Right: the same
    digit cut into 16 patches; patch 5 (orange frame) is connected to all 15 other patches. In a single
    attention layer it can take in information from the far corner as easily as from its neighbour.

    The plan for today, with the shape of the data after each step, for one picture:

    ```
    picture                                                                       1 x 28 x 28
     -> cut into 16 patches of 7 x 7; each patch laid out as a row of 49 numbers  16 x 49     (section 2.1)
     -> a linear layer turns each row of 49 numbers into 32: the patch tokens     16 x 32     (2.2 and 2.3)
     -> put the class token in front, add a position row to every token           17 x 32     (2.4, step (a))
     -> transformer blocks: attention, then a small network, a few times          17 x 32     (3 and 4)
     -> take row 0, the class token                                               32
     -> a linear layer                                                            10 scores   (step (b))
    ```
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · From a picture to tokens

    ### 2.1 · Cut the picture into patches

    `image` has the shape `1 x 28 x 28` (1 grey channel, 28 rows, 28 columns), so `image[0]` is the 28 x 28
    table of its pixel values. `picture[a:b, c:d]` cuts out rows `a` to `b - 1` and columns `c` to `d - 1`: for
    example `picture[0:7, 7:14]` is the second patch of the top row.

    The cell goes over the patches row by row, from left to right, like reading a page. `patch.flatten()` lays
    a 7 x 7 patch out as one row of 49 numbers (the 7 pixel rows one after the other), `rows.append(...)` adds
    that row to a Python list, and `torch.stack(rows)` piles the 16 rows into one table of 16 x 49 numbers.
    `V.show_patches` draws the patches with their numbers 0 to 15.
    """),
    api("Tensor.flatten"),
    api("torch.stack", brief=True),
    code("""
    picture = image[0]                  # the 28 x 28 pixel values (image is 1 x 28 x 28: one grey channel)
    patch_size = 7
    rows = []
    for top in range(0, 28, patch_size):            # 0, 7, 14, 21: the first pixel row of each band of patches
        for left in range(0, 28, patch_size):       # 0, 7, 14, 21: the first pixel column of each patch
            patch = picture[top:top + patch_size, left:left + patch_size]    # 7 x 7 pixels
            rows.append(patch.flatten())            # 49 numbers in a row
    patch_table = torch.stack(rows)                 # 16 rows of 49 numbers
    print("one patch:", patch.shape, "  one row:", rows[0].shape, "  the table of all patches:", patch_table.shape)
    V.show_patches(image, patch_size=7)
    """),
    md(r"""
    The 3 lies in the middle patches; the six patches 0, 3, 8, 9, 12 and 15 hold only background.

    `V.show_table(...)` draws the table itself: one row per patch, its 49 pixel values from left to right
    (black = background, white = ink).
    """),
    code("""
    V.show_table(patch_table, "the patch table: 16 rows (one per patch) of 49 pixel values", column_word="pixel values",
                 kind="pixels")
    """),
    md(r"""
    The empty patches are black rows. In the other rows the ink comes back every 7 columns or so: the 7 pixel
    rows of a patch follow each other in its row of the table.

    ### 2.2 · Each patch becomes a token: a linear layer

    49 raw pixel values are not yet a useful description of a patch. A linear layer (P1), `nn.Linear(49, 32)`,
    turns each row of 49 numbers into 32 numbers: 32 neurons, each a weighted sum of the 49 pixel values plus a
    bias. The **same** layer, with the same weights, is used for every patch, just as a convolution uses the
    same filter at every position. What comes out for one patch is its **patch embedding**: its token.

    The length of a token, here 32, is a setting called `dim` (short for dimension). A linear layer given a
    table works on each row on its own: 16 x 49 in, 16 x 32 out. Its weights are random for now; training will
    choose them, so the numbers below do not mean anything yet.
    """),
    api("torch.manual_seed", "nn.Linear", brief=True),
    code("""
    torch.manual_seed(0)
    to_vector = nn.Linear(49, 32)               # 49 pixel values in -> 32 numbers out: 49 x 32 + 32 = 1,600 parameters
    patch_tokens = to_vector(patch_table)       # the same layer on each of the 16 rows
    print("the patch tokens:", patch_tokens.shape)
    V.show_table(patch_tokens, "the 16 patch tokens: 16 rows of 32 numbers (random weights for now)")
    """),
    md(r"""
    Blue cells are negative numbers, red cells positive ones. The six empty patches (0, 3, 8, 9, 12, 15) have
    exactly the same token, because their 49 pixel values are the same; the patches with much ink (4, 5, 6, 10,
    13, 14) stand out.

    ### 2.3 · A convolution does all of this in one go

    Cutting, flattening and the linear layer together are exactly one convolution (P1),
    `nn.Conv2d(1, 32, kernel_size=7, stride=7)`:

    * `kernel_size=7`: each filter covers one 7 x 7 patch;
    * `stride=7`: the filter jumps 7 pixels at a time, from one patch to the next, so it visits every patch
      exactly once, without overlap: 4 x 4 positions;
    * 32 filters, each a weighted sum of the 49 pixels of the patch plus a bias: the 32 neurons of the linear
      layer.

    The output is 32 maps of 4 x 4: 32 numbers for each of the 16 patches. Two steps rearrange them into the
    table of tokens, without changing any number:

    * `x.flatten(2)` flattens from axis 2 on: each 4 x 4 map becomes a row of 16 numbers (`1 x 32 x 16`);
    * `x.transpose(1, 2)` swaps axes 1 and 2, so that each patch gets one row of 32 numbers (`1 x 16 x 32`).

    To prove that it is the same computation, `V.copy_linear_into_conv` gives the convolution the weights of the
    linear layer of 2.2, and `torch.allclose(a, b)` tells whether two tensors hold the same numbers (up to tiny
    rounding differences). A convolution works on a batch of pictures: `image.unsqueeze(0)` adds the batch axis
    in front, `1 x 1 x 28 x 28`. Every vision transformer computes its patch tokens with such a convolution,
    because it is fast.
    """),
    api("nn.Conv2d", "Tensor.unsqueeze", brief=True),
    api("Tensor.transpose", "torch.allclose"),
    code("""
    to_tokens = nn.Conv2d(1, 32, kernel_size=7, stride=7)   # a 7 x 7 filter that jumps from patch to patch
    V.copy_linear_into_conv(to_vector, to_tokens)           # the weights of the linear layer, to compare
    batch = image.unsqueeze(0)                              # a batch of one picture: 1 x 1 x 28 x 28
    x = to_tokens(batch)
    print("after the convolution: ", x.shape)               # 32 maps of 4 x 4: 32 numbers for each patch
    x = x.flatten(2)
    print("after flatten(2):      ", x.shape)               # each 4 x 4 map as a row of 16 numbers
    x = x.transpose(1, 2)
    print("after transpose(1, 2): ", x.shape)               # one row of 32 numbers per patch
    print("the same numbers as the linear layer:", torch.allclose(x[0], patch_tokens, atol=1e-5))
    """),
    md(r"""
    The same 16 x 32 numbers, from one layer with the same 1 600 parameters.

    ### 2.4 · Two more pieces: the class token and the position rows

    **The class token.** One extra row of `dim` numbers, put in front of the 16 patch tokens, so that every
    picture has 17 tokens. It is the same for every picture and belongs to no patch. Its numbers are
    **learned**: they are parameters of the network, like the weights, and training changes them. Through
    attention it gathers information from all patches, and at the very end the network reads its answer from
    this row only. `nn.Parameter(tensor)` turns a tensor into such learned numbers. Our class token starts as
    zeros: `nn.Parameter(torch.zeros(1, 1, dim))`.

    **The position rows.** Attention (section 3) compares tokens with each other, but it does not know where in
    the picture a token comes from. A test: shuffle the 16 patches of the digit. To us the picture is ruined.
    But its table of patches holds the same 16 rows, only in another order. `V.shuffle_patches` moves the
    patches to other places, `V.patch_table(picture, patch_size)` does the loop of section 2.1, and
    `V.same_rows` compares the two tables.
    """),
    code("""
    shuffled = V.shuffle_patches(image, patch_size=7, seed=0)
    V.show_pictures([image, shuffled], ["the digit", "the same 16 patches, shuffled"])
    V.same_rows(V.patch_table(image, patch_size=7), V.patch_table(shuffled, patch_size=7))
    """),
    md(r"""
    The same rows, in another order. Section 3.3 shows that attention gives the class token exactly the same
    result for both pictures: without help, a transformer cannot tell them apart.

    The help is the **position embedding**: one learned row of `dim` numbers for each place in the list of
    tokens (17 places: the class token's and the 16 patches'), **added** to the token at that place. The token
    of a patch at place 5 then differs from the token of the same patch at place 12. Our position rows start as
    random numbers (`torch.randn`), so that every place is different from the start; training adjusts them.

    For a batch of 64 digits and `dim = 32`, the embedding then goes like this:

    | step | code | shape |
    |---|---|---|
    | patch tokens: the convolution, then flatten and transpose | `self.to_tokens(images)`, ... | 64 x 16 x 32 |
    | the class token, one copy per picture | `self.class_token.expand(batch_size, -1, -1)` | 64 x 1 x 32 |
    | the class token in front of the patch tokens | `torch.cat([class_tokens, x], dim=1)` | 64 x 17 x 32 |
    | the position rows added | `x + self.position` | 64 x 17 x 32 |

    * `expand(batch_size, -1, -1)` makes the 1 x 1 x 32 class token look like 64 copies, 64 x 1 x 32, without
      copying any number; `-1` means "keep this size".
    * `torch.cat` (P3) joins tensors along an axis: here axis 1, the axis of the tokens, with the class token
      first.
    * `self.position` is `1 x 17 x 32`, `x` is `64 x 17 x 32`: when the sizes of an axis are 1 and 64, PyTorch
      adds the same position table to every picture of the batch.
    """),

    # ------------------------------------------------------------------ (a)
    md(r"""
    ### Step (a) · Your turn: the `PatchEmbedding` layer

    A layer that is not one chain of other layers is written as a **class** built on `nn.Module` (P3): its
    `__init__` makes the parts once, its `forward` says what happens to a batch of pictures. Our layer has four
    settings: `image_size` (28), `patch_size` (7), `in_channels` (1 for grey pictures, 3 for colour) and `dim`
    (32). Write the names of these settings in your code, not the numbers: the same class must also work for
    other pictures.

    **`__init__`**: the first two lines (given) count the patches, 28 // 7 = 4 per side, 4 x 4 = 16 in all
    (`//` divides and drops the remainder). Then:

    | part | what it is | you write |
    |---|---|---|
    | `self.to_tokens` | the convolution of 2.3: `in_channels` -> `dim` maps, a `patch_size` filter that jumps `patch_size` pixels | `nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)` |
    | `self.class_token` | the class token: learned, 1 x 1 x `dim`, starts as zeros | `nn.Parameter(torch.zeros(1, 1, dim))` |
    | `self.position` | the position rows: learned, 1 x (`number_of_patches` + 1) x `dim`, starts random | `nn.Parameter(torch.randn(1, number_of_patches + 1, dim))` |

    The `+ 1` is the class token's place. The first 1 of each shape is a batch of one: `expand` and the
    addition make it fit any batch.

    **`forward(self, images)`**: `images` is a batch, `batch_size x in_channels x 28 x 28`. The shapes are for
    our digits with `dim = 32`:

    | line | what it does | shape |
    |---|---|---|
    | `x = self.to_tokens(images)` | **yours**: the convolution | batch x 32 x 4 x 4 |
    | `x = x.flatten(2)` | given: each map as a row | batch x 32 x 16 |
    | `x = x.transpose(1, 2)` | given: one row per patch | batch x 16 x 32 |
    | `class_tokens = self.class_token.expand(batch_size, -1, -1)` | given: one class token per picture | batch x 1 x 32 |
    | `x = torch.cat([class_tokens, x], dim=1)` | **yours**: the class token in front | batch x 17 x 32 |
    | `x = x + self.position` | **yours**: add the position rows | batch x 17 x 32 |

    **Your turn.** Fill in the six blanks, three in `__init__` and three in `forward`, and delete the two
    `raise` lines.
    """),
    api("nn.Module", "torch.zeros", "torch.cat", brief=True),
    api("nn.Parameter", "torch.randn", "Tensor.expand"),
    exercise(
        stub='''
        class PatchEmbedding(nn.Module):
            """A picture -> a table of tokens: the class token, then one token per patch, each dim numbers long."""

            def __init__(self, image_size=28, patch_size=7, in_channels=1, dim=32):
                super().__init__()
                raise NotImplementedError   # delete this line once every ____ is filled in
                patches_per_side = image_size // patch_size                 # 28 // 7 = 4
                number_of_patches = patches_per_side * patches_per_side     # 4 x 4 = 16
                self.to_tokens = ____       # YOUR CODE: nn.Conv2d from in_channels to dim maps, kernel_size=patch_size, stride=patch_size
                self.class_token = ____     # YOUR CODE: nn.Parameter of torch.zeros(1, 1, dim)
                self.position = ____        # YOUR CODE: nn.Parameter of torch.randn(1, number_of_patches + 1, dim)

            def forward(self, images):
                raise NotImplementedError   # delete this line once every ____ is filled in
                x = ____                    # YOUR CODE: self.to_tokens applied to images      -> batch x dim x 4 x 4
                x = x.flatten(2)            # given: each map as a row of 16 numbers             -> batch x dim x 16
                x = x.transpose(1, 2)       # given: one row per patch                           -> batch x 16 x dim
                batch_size = x.shape[0]
                class_tokens = self.class_token.expand(batch_size, -1, -1)     # given: one per picture -> batch x 1 x dim
                x = ____                    # YOUR CODE: torch.cat of the list [class_tokens, x] along dim=1   -> batch x 17 x dim
                x = ____                    # YOUR CODE: x plus self.position                    -> batch x 17 x dim
                return x
        ''',
        solution='''
        class PatchEmbedding(nn.Module):
            """A picture -> a table of tokens: the class token, then one token per patch, each dim numbers long."""

            def __init__(self, image_size=28, patch_size=7, in_channels=1, dim=32):
                super().__init__()
                patches_per_side = image_size // patch_size                 # 28 // 7 = 4
                number_of_patches = patches_per_side * patches_per_side     # 4 x 4 = 16
                self.to_tokens = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)   # YOUR CODE: nn.Conv2d from in_channels to dim maps, kernel_size=patch_size, stride=patch_size
                self.class_token = nn.Parameter(torch.zeros(1, 1, dim))                                 # YOUR CODE: nn.Parameter of torch.zeros(1, 1, dim)
                self.position = nn.Parameter(torch.randn(1, number_of_patches + 1, dim))                # YOUR CODE: nn.Parameter of torch.randn(1, number_of_patches + 1, dim)

            def forward(self, images):
                x = self.to_tokens(images)  # YOUR CODE: self.to_tokens applied to images      -> batch x dim x 4 x 4
                x = x.flatten(2)            # given: each map as a row of 16 numbers             -> batch x dim x 16
                x = x.transpose(1, 2)       # given: one row per patch                           -> batch x 16 x dim
                batch_size = x.shape[0]
                class_tokens = self.class_token.expand(batch_size, -1, -1)     # given: one per picture -> batch x 1 x dim
                x = torch.cat([class_tokens, x], dim=1)     # YOUR CODE: torch.cat of the list [class_tokens, x] along dim=1   -> batch x 17 x dim
                x = x + self.position       # YOUR CODE: x plus self.position                    -> batch x 17 x dim
                return x
        '''),
    md(r"""
    **Check.** `T.check_patch_embedding` builds your layer with the numbers of our digits, looks at its parts,
    runs a batch of 2 random pictures through it and compares every number of the output with what the table
    describes. The last line builds it again for colour pictures of 32 x 32 with patches of 4 x 4, to catch
    numbers typed where names belong.
    """),
    code("""
    T.check_patch_embedding(PatchEmbedding)
    """),
    md(r"""
    **What you built.** Your layer on the digit: 17 tokens of 32 numbers. The class token starts at zero, so
    its row (green frame) shows only its position row. Every row has a random position row added, so even the
    empty patches no longer look alike: the network has not learned anything yet.
    """),
    code("""
    PatchEmbedding = T.yours_or_reference(PatchEmbedding)
    torch.manual_seed(0)
    embedding = PatchEmbedding(image_size=28, patch_size=7, in_channels=1, dim=32)
    tokens = embedding(image.unsqueeze(0))      # a batch of one picture -> 1 x 17 x 32
    print("tokens:", tokens.shape)
    V.show_tokens(tokens)
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · Attention: the tokens exchange information

    So far each token knows only its own patch. **Attention** lets every token take in information from the
    other tokens: from the ones that are similar to it, a lot; from the others, little. It works in three
    steps, for every token:

    1. **scores**: compare the token with every token (itself included). The comparison is the **dot
       product**: multiply the two lists number by number and add up. It is large when the two tokens point in
       the same direction, and negative when they point in opposite directions;
    2. **softmax** (P5): turn the scores into **weights** that are positive and add up to 1; the largest score
       gets the largest weight. For scores s1, s2, s3: weight 1 = e^s1 / (e^s1 + e^s2 + e^s3), and so on
       (e = 2.718...);
    3. **weighted average**: the token's new value is weight 1 x token 1 + weight 2 x token 2 + ...

    ### 3.1 · By hand: three tokens of two numbers

    Three tokens: A = [1, 1], B = [1, 0], C = [-1, 0]. A and B point roughly the same way, C the other way.
    Here is the computation for token A:

    1. scores: A · A = 1 x 1 + 1 x 1 = **2**; A · B = 1 x 1 + 1 x 0 = **1**; A · C = 1 x (-1) + 1 x 0 = **-1**;
    2. softmax: e^2 = 7.389, e^1 = 2.718, e^-1 = 0.368, together 10.475; the weights are
       7.389 / 10.475 = **0.705**, 2.718 / 10.475 = **0.259** and 0.368 / 10.475 = **0.035** (they add up to 1);
    3. weighted average: new A = 0.705 x [1, 1] + 0.259 x [1, 0] + 0.035 x [-1, 0]
       = [0.705 + 0.259 - 0.035, 0.705] = **[0.93, 0.71]** (rounded).

    A has taken in a good part of B, its look-alike, and almost nothing of C. The cell does the same for all
    three tokens, with plain loops: `(tokens[i] * tokens[j]).sum()` multiplies two tokens number by number and
    adds up, `scores.softmax(dim=1)` applies the softmax to each row. `V.show_toy_attention` draws the tokens
    before and after.
    """),
    api("torch.tensor", "Tensor.softmax", brief=True),
    api("Tensor.sum"),
    code("""
    tokens_abc = torch.tensor([[1.0, 1.0],          # A
                               [1.0, 0.0],          # B
                               [-1.0, 0.0]])        # C

    # 1. scores: the dot product of every token (row i) with every token (column j)
    scores = torch.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            scores[i, j] = (tokens_abc[i] * tokens_abc[j]).sum()

    # 2. softmax of each row: weights that are positive and add up to 1
    weights_abc = scores.softmax(dim=1)

    # 3. each new token is the weighted average of all tokens
    new_abc = torch.zeros(3, 2)
    for i in range(3):
        for j in range(3):
            new_abc[i] = new_abc[i] + weights_abc[i, j] * tokens_abc[j]

    print("scores:\\n", scores)
    print("weights:\\n", weights_abc)
    print("new tokens:\\n", new_abc)
    V.show_toy_attention(tokens_abc, weights_abc, new_abc, ["A", "B", "C"])
    """),
    md(r"""
    **Reading it.** Row A of the weights is 0.705, 0.259, 0.035, as computed by hand, and the new A is
    [0.9298, 0.7054]. A and B pull each other closer. C is unlike both, so it gives the largest weight to
    itself (0.787). That is all attention does: every token becomes a weighted average of the tokens, with
    large weights for the similar ones.

    ### 3.2 · A real attention layer, on the digit

    A real attention layer adds three things to the hand-worked version:

    * **query, key and value.** Three linear layers make three versions of every token. The scores compare the
      **query** of one token ("what am I looking for?") with the **key** of another ("what do I contain?"),
      and the weighted average is taken over the **values** ("what do I pass on?"). In the toy example all
      three were the token itself; here the network learns them. A fourth linear layer mixes the result.
    * The scores are divided by the square root of the number of numbers they compare, so that the softmax
      does not become too sharp.
    * **Heads.** Several attentions can run side by side, each on a slice of the numbers, with their own
      queries, keys and values; their results are joined, as the branches of an Inception block (P3). Each
      head can look for something else.

    `nn.MultiheadAttention(embed_dim=32, num_heads=1, batch_first=True)` is such a layer for tokens of 32
    numbers, with 1 head. Called with the same tokens three times (for the queries, the keys and the values) it
    computes **self-attention**: the tokens of the digit attend to each other. It returns the new tokens and the
    17 x 17 table of attention weights. The layer's own parameters are random: it has not learned what to look
    for yet.
    """),
    api("nn.MultiheadAttention"),
    api("Module.parameters", "Tensor.numel", brief=True),
    code("""
    torch.manual_seed(0)
    attention = nn.MultiheadAttention(embed_dim=32, num_heads=1, batch_first=True)
    new_tokens, weights = attention(tokens, tokens, tokens)    # self-attention: every token looks at every token
    print("tokens in:", tokens.shape, "  tokens out:", new_tokens.shape, "  weights:", weights.shape)
    count = 0
    for p in attention.parameters():
        count += p.numel()
    print("parameters of the attention layer:", count, "= 4 x 32 x 32 weights + 4 x 32 biases")
    V.show_weights(weights, title="the untrained layer on the digit: 17 x 17 attention weights")
    """),
    md(r"""
    **Reading the table.** Row 0 tells how the class token spreads its attention over the 17 tokens, row 6 how
    the token of patch 6 does, and so on; each row adds up to 1. An even spread would give every token
    1 / 17 = 0.059. Here most weights are below 0.1 and a few are much larger: patch 9 gives 0.29 of its
    attention to patch 7. The pattern means nothing yet: it comes from the random parameters of the
    layer and the random position rows. After training (section 5) we look again at the class token's row.

    ### 3.3 · The shuffle test: why the position rows are needed

    `V.position_test` runs this attention layer on the tokens of the digit and of its shuffled version (2.4),
    first without the position rows, then with them, and prints the first 4 of the 32 numbers that come out for
    the class token.
    """),
    code("""
    V.position_test(embedding, attention, image, shuffled)
    """),
    md(r"""
    Without the position rows, the class token gets exactly the same numbers for the digit and for the
    shuffled mess: attention treats the tokens as a bag, not as a picture. With the position rows the two
    results differ, so the network can learn that a stroke at the top means something else than the same stroke
    at the bottom. Compare a CNN: its filters see neighbouring pixels together, so the layout is built in (P1,
    section 6); a ViT has to learn it from the position rows.
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · The whole tiny vision transformer

    ### 4.1 · One transformer block

    A **transformer block** is attention followed by a small network that works on each token on its own:

    ```
    x -> layer norm -> attention ----------> + x      (skip connection)
      -> layer norm -> MLP: Linear, GELU, Linear -> + x      (skip connection)
    ```

    * **Layer norm** rescales the numbers of each token to an average of 0 and a spread of 1, then applies a
      learned scale and shift; it keeps the numbers in a good range (like batch normalisation, P2, but token by
      token).
    * The **MLP** (multi-layer perceptron: a small fully connected network, like network A of P1) is
      `Linear(dim, 2 x dim)`, an activation, `Linear(2 x dim, dim)`, applied to every token separately.
      Attention is the step where tokens talk to each other; the MLP is the step where each token works on what
      it has heard. The activation **GELU** is a smooth version of ReLU.
    * A **skip connection** adds the input of a step to its output, so that each step only has to learn a
      change of the tokens. This makes deep networks easier to train; ResNet, the network of P5, is built the
      same way. (The skip connections of the U-Net in P4 were different: they joined maps with `torch.cat`.)

    A block keeps the shape: 17 tokens of `dim` numbers in, 17 tokens of `dim` numbers out, so blocks can be
    stacked. PyTorch has it ready made: `nn.TransformerEncoderLayer`. The function `transformer_block(dim,
    heads)` sets it up the way a ViT uses it; the comments explain every setting. This cell is complete.
    """),
    api("nn.TransformerEncoderLayer", "nn.LayerNorm"),
    code('''
    def transformer_block(dim, heads):
        """One transformer block: layer norm, attention, skip connection; layer norm, MLP, skip connection."""
        block = nn.TransformerEncoderLayer(
            d_model=dim,                    # numbers per token
            nhead=heads,                    # attention heads (dim must be a multiple of heads)
            dim_feedforward=2 * dim,        # neurons in the hidden layer of the MLP
            dropout=0.0,                    # no dropout
            activation="gelu",              # the activation of the MLP
            batch_first=True,               # tensors are batch x tokens x numbers
            norm_first=True,                # layer norm before the attention and before the MLP, as in ViT
        )
        return block


    block = transformer_block(32, 4)
    print(block)
    print("tokens in:", tokens.shape, "  tokens out:", block(tokens).shape)
    count = 0
    for p in block.parameters():
        count += p.numel()
    print("parameters of one block:", count)
    '''),
    md(r"""
    **Reading the printout.** `self_attn` is the attention layer (with its output layer `out_proj`), `linear1`
    and `linear2` are the MLP (32 -> 64 -> 32 numbers), `norm1` and `norm2` the two layer norms; the dropout
    layers do nothing with `dropout=0.0`. The 8 544 parameters are 4 224 for the attention (4 x 32 x 32 + 4 x 32),
    4 192 for the MLP (32 x 64 + 64 + 64 x 32 + 32) and 128 for the two layer norms (2 x 2 x 32).

    ### Step (b) · Your turn: `TinyViT`

    The whole network: your patch embedding, `depth` transformer blocks one after the other, then the class
    token, a layer norm, and a linear layer from `dim` numbers to one score per class. The settings: `image_size`
    (28), `patch_size` (7), `dim` (32), `depth` (2 blocks), `heads` (4) and `num_classes` (10).

    **`__init__`**:

    | part | what it is | you write |
    |---|---|---|
    | `self.embedding` | your layer of step (a) | `PatchEmbedding(image_size, patch_size, 1, dim)` |
    | `self.blocks` | given: an empty `nn.Sequential` (P1); the loop adds `depth` blocks with `add_module` (P3) | inside the loop: `transformer_block(dim, heads)` |
    | `self.norm` | given: a layer norm for the class token | `nn.LayerNorm(dim)` |
    | `self.head` | a linear layer: the class token's `dim` numbers -> `num_classes` scores | `nn.Linear(dim, num_classes)` |

    **`forward(self, images)`**, with the shapes for a batch of our digits:

    | line | what it does | shape |
    |---|---|---|
    | `tokens = self.embedding(images)` | **yours**: picture -> tokens | batch x 17 x 32 |
    | `tokens = self.blocks(tokens)` | **yours**: all transformer blocks | batch x 17 x 32 |
    | `class_token = tokens[:, 0]` | **yours**: row 0 of every picture | batch x 32 |
    | `class_token = self.norm(class_token)` | given | batch x 32 |
    | `scores = self.head(class_token)` | **yours**: 10 scores | batch x 10 |

    `tokens[:, 0]` picks, for every picture of the batch (`:` means "all of this axis"), the token number 0:
    the class token. Careful: `tokens[0]` would be something else, all 17 tokens of the first picture only.

    **Your turn.** Fill in the seven blanks and delete the two `raise` lines.
    """),
    api("nn.Sequential", "Module.add_module", brief=True),
    exercise(
        stub='''
        class TinyViT(nn.Module):
            """A tiny vision transformer: patch embedding, depth transformer blocks, the class token -> scores."""

            def __init__(self, image_size=28, patch_size=7, dim=32, depth=2, heads=4, num_classes=10):
                super().__init__()
                raise NotImplementedError   # delete this line once every ____ is filled in
                self.embedding = ____                   # YOUR CODE: PatchEmbedding(image_size, patch_size, 1, dim)
                self.blocks = nn.Sequential()           # given: an empty chain, filled by the loop
                for k in range(depth):
                    self.blocks.add_module("block" + str(k + 1), ____)     # YOUR CODE: transformer_block(dim, heads)
                self.norm = nn.LayerNorm(dim)           # given: a layer norm for the class token
                self.head = ____                        # YOUR CODE: nn.Linear from dim to num_classes

            def forward(self, images):
                raise NotImplementedError   # delete this line once every ____ is filled in
                tokens = ____               # YOUR CODE: self.embedding applied to images    -> batch x 17 x dim
                tokens = ____               # YOUR CODE: self.blocks applied to tokens       -> batch x 17 x dim
                class_token = ____          # YOUR CODE: row 0 of every picture, tokens[:, 0] -> batch x dim
                class_token = self.norm(class_token)    # given                              -> batch x dim
                scores = ____               # YOUR CODE: self.head applied to class_token    -> batch x num_classes
                return scores
        ''',
        solution='''
        class TinyViT(nn.Module):
            """A tiny vision transformer: patch embedding, depth transformer blocks, the class token -> scores."""

            def __init__(self, image_size=28, patch_size=7, dim=32, depth=2, heads=4, num_classes=10):
                super().__init__()
                self.embedding = PatchEmbedding(image_size, patch_size, 1, dim)     # YOUR CODE: PatchEmbedding(image_size, patch_size, 1, dim)
                self.blocks = nn.Sequential()           # given: an empty chain, filled by the loop
                for k in range(depth):
                    self.blocks.add_module("block" + str(k + 1), transformer_block(dim, heads))     # YOUR CODE: transformer_block(dim, heads)
                self.norm = nn.LayerNorm(dim)           # given: a layer norm for the class token
                self.head = nn.Linear(dim, num_classes)     # YOUR CODE: nn.Linear from dim to num_classes

            def forward(self, images):
                tokens = self.embedding(images)     # YOUR CODE: self.embedding applied to images    -> batch x 17 x dim
                tokens = self.blocks(tokens)        # YOUR CODE: self.blocks applied to tokens       -> batch x 17 x dim
                class_token = tokens[:, 0]          # YOUR CODE: row 0 of every picture, tokens[:, 0] -> batch x dim
                class_token = self.norm(class_token)    # given                              -> batch x dim
                scores = self.head(class_token)     # YOUR CODE: self.head applied to class_token    -> batch x num_classes
                return scores
        '''),
    md(r"""
    **Check.** The network with its default settings, part by part; then a batch of 2 pictures, the order in
    which `forward` uses the parts, and whether the scores come from the class token. The last line builds it
    with other settings.
    """),
    code("""
    T.check_tiny_vit(TinyViT)
    """),
    md(r"""
    **What you built.** `P.show_model(..., depth=1)` (P1, P3) draws each part as one box.
    """),
    code("""
    TinyViT = T.yours_or_reference(TinyViT)
    torch.manual_seed(0)
    vit = TinyViT()
    P.show_model(vit, (1, 28, 28), depth=1, title="Your TinyViT with its default settings")
    """),
    md(r"""
    **Reading the drawing.** The embedding turns the 1 x 28 x 28 picture into 17 x 32 tokens (2 176
    parameters); `blocks`, the two transformer blocks drawn as one stacked box, keeps that shape; **Slice** is
    `tokens[:, 0]`, which keeps the class token's 32 numbers; then the layer norm and the linear layer give 10
    scores.

    `P.summary` lists the same parts layer by layer: inside the embedding you can find the convolution, the
    flatten (a "reshape") and the transpose of step (a), and the concatenation of the class token.
    """),
    code("""
    P.summary(vit, (1, 28, 28))
    """),
    md(r"""
    **Reading the table.** 19 658 parameters, fewer than a fifth of the P1 CNN's 105 866. The two blocks hold
    most of them (8 544 each); the class token and the position rows (32 + 17 x 32 = 576) are in the total but
    have no row of their own. One warning about the last column: for a transformer block, `P` counts only the
    multiply-adds of the MLP. The attention's own linear layers are hidden inside `nn.MultiheadAttention`, where
    `P` cannot see them, so the real work of a block is roughly twice what is shown. The measured times of
    `P.cost` and `P.train` are not affected.
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · Train it on MNIST

    ### Try it: the run settings

    The network is your `TinyViT`; these settings choose its size and the training. Run this cell, then the
    training cell.

    * `dim`: numbers per token;
    * `depth`: transformer blocks;
    * `heads`: attention heads per block (`dim` must be a multiple of `heads`);
    * `patch_size`: 7 gives 16 patches of 7 x 7; 4 gives 49 patches (slower), 14 gives 4 patches. It must
      divide 28;
    * `epochs` and `learning_rate`: as in P1;
    * `n_train`: training pictures (at most 60 000); the time per epoch grows in proportion.
    """),
    code("""
    # ---------------- Try it: run settings ----------------
    dim = 64                      # numbers per token
    depth = 2                     # transformer blocks
    heads = 4                     # attention heads per block (dim must be a multiple of heads)
    patch_size = 7                # 7 -> 16 patches of 7 x 7; also 4 (49 patches) or 14 (4 patches)
    epochs = 10                   # passes over the training pictures
    learning_rate = 0.001         # size of each weight change
    n_train = 10000               # training pictures (at most 60000)
    version_name = "tiny ViT"     # the name on the curves and in the leaderboard
    """),
    md(r"""
    `P.cost` (P1) first measures the network; `P.train` (P1) then trains it with the Adam optimiser and draws the
    loss and accuracy curves after every epoch. The starting weights are reset with `torch.manual_seed(0)`, so
    every run with the same settings gives the same numbers. **Expected time: under a minute** with the
    settings above (up to 2 minutes on a slow laptop; the last line of the cost report estimates one epoch on
    your computer).
    """),
    code("""
    train_loader, val_loader, classes = P.get_mnist(n_train=n_train, n_val=2000, batch_size=64)
    torch.manual_seed(0)
    vit = TinyViT(patch_size=patch_size, dim=dim, depth=depth, heads=heads)
    P.cost(vit, (1, 28, 28), name=version_name, epoch_images=n_train)
    history_vit = P.train(vit, train_loader, val_loader, epochs=epochs, lr=learning_rate, name=version_name)
    """),
    md(r"""
    **Reading the curves.** The tiny ViT learns: 88.7% of the validation digits right after the first epoch,
    94.5% after the fourth, and its best is 95.4% in epoch 9 (95.0% after the last). The way up is bumpy: in
    epoch 5 it drops to 92.5% and then climbs back. From about epoch 6 on, the training accuracy keeps rising
    (97.8% at the end) while the validation accuracy stays around 95% and the validation loss creeps up again
    (0.153 in epoch 6, 0.168 at the end): the network starts to learn the training digits by heart (P1).

    `P.show_predictions` (P1) shows its answers for the first 16 validation pictures.
    """),
    code("""
    P.show_predictions(vit, val_loader, classes)
    """),
    md(r"""
    14 of the 16 are right. The network takes a hastily written 4 for an 8 and a thick 6 for a 0, and it is
    only 75% sure about our 3.

    ### Compared with the small CNN of P1

    Network B of P1, trained in the same way on the same pictures, for the same number of epochs (`live=False`
    draws its curves once at the end). **Expected time: about 20 seconds** (up to a minute on a slow laptop).
    """),
    api("nn.ReLU", "nn.MaxPool2d", "nn.Flatten", brief=True),
    code("""
    torch.manual_seed(0)
    cnn = nn.Sequential(                                # network B of P1
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(1568, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    history_cnn = P.train(cnn, train_loader, val_loader, epochs=epochs, lr=learning_rate, name="CNN of P1",
                          live=False)
    """),
    md(r"""
    `P.compare` (P1) puts the two side by side, and `P.plot_history` draws their curves on top of each other
    (solid: validation, dashed: training). Remember that the multiply-adds of the ViT are undercounted (section
    4).
    """),
    code("""
    P.compare({version_name: vit, "CNN of P1": cnn}, (1, 28, 28),
              histories={version_name: history_vit, "CNN of P1": history_cnn})
    P.plot_history({version_name: history_vit, "CNN of P1": history_cnn})
    """),
    md(r"""
    **What the numbers say.**

    * **Accuracy:** the CNN is better, 97.8% against 95.4% at their best epochs. On 2 000 validation digits that
      is about 44 mistakes against about 92: the ViT makes about twice as many.
    * **Size:** the ViT is the smaller network, 72 074 parameters against 105 866. Its lower accuracy is not a
      question of size.
    * **Work and time:** `P` counts 0.61 million multiply-adds for the ViT, but it misses the attention part
      (section 4); the real number is about twice that, roughly as much as the CNN's 1.12 million. In the
      `train step ms` column of `P.compare`, which times both networks in the same way one after the other, the
      ViT needs more time per training step than the CNN: it is made of many small steps, each with a little
      overhead of its own. (How much more differs from computer to computer, and the times measured during
      training jump around when the computer is busy with other work.)
    * **The curves:** the CNN is ahead from the first epoch on (94.8% against 88.7%) and stays ahead.

    **Why does the CNN win here?** A CNN has the layout of pictures built in: its filters always see
    neighbouring pixels together, and a pattern learned in one place is found everywhere (P1). The ViT starts
    without that knowledge: every patch may look at every other patch, and it must learn from the data which
    ones matter, and even where each patch is (section 3.3). On 10 000 small digits, the built-in knowledge is
    worth more. The ViT paper found the same on ImageNet: trained on its 1.28 million photos alone, ViTs were a
    little worse than CNNs of the same size; pretrained on 14 million or 300 million photos, they were better.
    Vision transformers shine with lots of data or pretraining; on a small dataset, a CNN (or a pretrained
    network, P5 and section 6) is usually the better choice.
    """),
    md(r"""
    ### Look inside: where does the class token look?

    `V.show_class_attention(model, val_loader)` runs the first 8 validation pictures through the trained network,
    reads the attention weights of the class token in every block (its row of the 17 x 17 table, without its
    own column: 16 weights, one per patch), averages them over the heads and the blocks, and draws them as a
    4 x 4 map in the places of the patches.
    """),
    code("""
    V.show_class_attention(vit, val_loader, n=8)
    """),
    md(r"""
    **What to look at.** In every picture the class token gives most of its attention to a few patches in the
    middle, where the strokes are, and almost none to the empty patches at the border. Nobody told it where to
    look: it learned from the labels alone that the border carries no information. The maps also differ from
    digit to digit: attention depends on the picture, while a convolution applies the same filter everywhere.
    (7 of the 8 answers are right; the 4 is taken for an 8.)

    ### Look inside: the learned position rows

    `V.show_position_similarity(model)` compares the learned position rows with each other: for every place,
    a small 4 x 4 map shows how similar its row is to the rows of all 16 places (red = similar, blue =
    opposite). The black square marks the place itself, which is always fully similar to itself.
    """),
    code("""
    V.show_position_similarity(vit)
    """),
    md(r"""
    **What to look at.** Apart from the place itself, almost everything is close to 0 (white): each place's
    position row is about as similar to its neighbours as to places far away. The rows started as random
    numbers, which already makes every place different from every other, and 10 epochs on 10 000 digits have not
    taught them which places are neighbours. In large vision transformers trained on millions of photos,
    neighbouring places do end up with similar rows (the ViT paper shows maps like these, with a bright patch
    around each place and along its row and column): one more piece of knowledge a CNN has built in and a ViT
    must learn from data.

    ### Keep a record

    `P.remember` and `P.leaderboard` (P1) keep a list of every run: change the settings in **Try it**, run the
    training cell and this one again, and compare.
    """),
    code("""
    P.remember(version_name, vit, history_vit)
    P.remember("CNN of P1", cnn, history_cnn)
    P.leaderboard()
    """),
    md(r"""
    **Things to try** (predict first, then run the **Try it** cell, the training cell, the CNN cell and this
    one):

    * `depth = 4`: twice the blocks. Better? How much slower?
    * `patch_size = 4` (49 patches of 4 x 4) and `patch_size = 14` (4 patches of 14 x 14): finer or coarser
      patches. Which one learns faster, and what happens to the time per epoch?
    * `heads = 1` and `heads = 8`: the same numbers per token, split into fewer or more attentions.
    * `n_train = 30000` with `epochs = 4`: three times the pictures. Vision transformers are known to need more
      data than CNNs. Does the gap to the CNN shrink?
    * `learning_rate = 0.003` and `learning_rate = 0.0003` (P1: too big, too small?).
    """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · A real vision transformer (optional)

    **ViT-B/16** ("base" size, patches of 16 x 16 pixels) is the middle-sized network of the ViT paper.
    torchvision has it, with weights trained on ImageNet (1.28 million photos in 1 000 classes, P3 and P5), and
    these weights (a file of about 330 MB) may already be on this computer, in PyTorch's cache folder (P5).
    `V.load_pretrained_vit()` loads them if the file is there and never downloads anything; without the file it
    builds the same network with random weights, so that its sizes can still be shown.
    """),
    api("torchvision.models.vit_b_16"),
    code("""
    vit_b16, vit_weights = V.load_pretrained_vit()
    """),
    md(r"""
    Its parts have the names of ours. The cell prints the patch embedding (a convolution, as in 2.3), the shapes
    of the class token and of the position rows, the number of transformer blocks, and the parameters.
    """),
    code("""
    print("patch embedding:", vit_b16.conv_proj)
    print("class token:    ", vit_b16.class_token.shape)
    print("position rows:  ", vit_b16.encoder.pos_embedding.shape)
    print("blocks:         ", len(vit_b16.encoder.layers))
    count = 0
    for p in vit_b16.parameters():
        count += p.numel()
    print("parameters:     ", count)
    """),
    md(r"""
    **Reading it.** The patch embedding is a convolution from 3 colour channels to 768 maps with a 16 x 16
    filter and stride 16: our `to_tokens` with other numbers. A 224 x 224 photo has 14 x 14 = 196 patches, so
    with the class token there are 197 position rows of 768 numbers each. 12 blocks, and 86 567 656 parameters:
    about 1 200 times our tiny ViT.

    | | our tiny ViT | ViT-B/16 |
    |---|---|---|
    | picture | 1 x 28 x 28 | 3 x 224 x 224 |
    | patches | 4 x 4 = 16 of 7 x 7 | 14 x 14 = 196 of 16 x 16 |
    | tokens | 16 + 1 = 17 | 196 + 1 = 197 |
    | numbers per token (`dim`) | 64 | 768 |
    | blocks (`depth`) | 2 | 12 |
    | heads | 4 | 12 |
    | parameters | 72 074 | 86 567 656 |
    | training pictures | 10 000 | 1.28 million (ImageNet) |

    The ViT paper found that trained on ImageNet alone, vision transformers did a little worse than CNNs of the
    same size; only after pretraining on 14 million or even 300 million photos did they win. A later recipe
    (DeiT, 2021) made them train well on ImageNet alone with strong data augmentation (random changes of every
    photo, every epoch) and long training; torchvision's weights were trained that way.

    `V.classify_photos` prepares four photos of Oxford-IIIT Pet (P5) the way the weights expect (224 x 224,
    ImageNet normalisation), runs them through ViT-B/16 and shows its three most likely ImageNet classes for
    each. **Expected time: a few seconds.**
    """),
    code("""
    V.classify_photos(vit_b16, vit_weights, ["pug_12", "Siamese_100", "samoyed_10", "beagle_20"])
    """),
    md(r"""
    **All four are right**, each with 85% to 89% (pug 85.2%, Siamese cat 89.4%, Samoyed 85.9%, beagle 88.1%),
    and every second guess gets less than 1%. These four breeds are among ImageNet's 1 000 classes, so the
    network needed no extra training. For classes it does not know, a pretrained ViT is adapted exactly as the
    pretrained ResNet of P5: replace its last layer (here `vit_b16.heads`) and train that on your own photos.
    """),

    # ------------------------------------------------------------------ 7
    md(r"""
    ## 7 · What to remember

    * A **vision transformer** cuts the picture into **patches** and turns each patch into a **token** (its
      **embedding**) with one linear layer. In practice that layer is a convolution whose kernel size and stride
      are the patch size: `nn.Conv2d(1, dim, kernel_size=7, stride=7)`, then `flatten(2)` and `transpose(1, 2)`.
    * A learned **class token** goes in front and a learned **position row** is added to every token: a
      28 x 28 digit becomes 17 tokens of `dim` numbers. Without the position rows a transformer cannot tell a
      picture from its shuffled patches.
    * **Attention**: scores (dot products) -> softmax -> weighted average. In one step every token can take in
      information from every other token, near or far; a convolution sees only a small neighbourhood.
    * A **transformer block** is layer norm, attention, skip connection; layer norm, MLP, skip connection. It
      keeps the shape, so blocks can be stacked. The answer is read from the class token:
      `head(norm(tokens[:, 0]))`.
    * On 10 000 small digits the tiny ViT learned (95.4% at best), but the small CNN of P1 did better (97.8%),
      although it has more parameters: a CNN has the layout of pictures built in, a ViT must learn it from the
      data. With much more data (millions of photos) or pretraining, vision transformers are among the best
      networks there are; ViT-B/16 named all four pet photos right.

    ### Words from this notebook

    | word | meaning |
    |---|---|
    | attention | every token becomes a weighted average of the tokens, with large weights for similar ones |
    | attention weights | how much each token takes from each token; each row adds up to 1 |
    | class token | an extra learned token in front of the patch tokens; the network reads its answer from it |
    | dot product | multiply two lists number by number and add up; large when they point the same way |
    | embedding (patch embedding) | the list of numbers that stands for a patch |
    | GELU | a smooth version of ReLU, the activation of the MLP in a transformer block |
    | head (attention head) | one of several attentions side by side, each on a slice of the numbers |
    | layer norm | rescales the numbers of each token to average 0 and spread 1, then a learned scale and shift |
    | learned numbers, `nn.Parameter` | numbers that training changes: weights, the class token, the position rows |
    | MLP | a small fully connected network; in a transformer block it works on each token separately |
    | patch | a small square of the picture, here 7 x 7 pixels |
    | position embedding (position rows) | one learned row per place, added to the tokens, so that the network knows where each patch was |
    | pretrained | trained by someone else, on a large collection of pictures |
    | query, key, value | three learned versions of each token: what it looks for, what it is compared by, what it passes on |
    | self-attention | attention in which the queries, keys and values all come from the same tokens |
    | skip connection | adding the input of a step to its output |
    | softmax | turns scores into positive weights that add up to 1 |
    | token | one row of numbers that a transformer works on: here one patch, or the class token |
    | transformer block | layer norm, attention, skip connection; layer norm, MLP, skip connection |
    | vision transformer (ViT) | a network that cuts a picture into patches and processes them with transformer blocks |
    | ViT-B/16 | the base ViT of the paper: patches of 16 x 16, 768 numbers per token, 12 blocks |
    """),
]
