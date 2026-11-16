from nb import md, code, exercise, header, api

NAME = "D1-04-transfer-learning"

CELLS = [
    header(
        "D1-04 · Transfer learning and what the network sees",
        steps=[
            "Model zoos: where pretrained networks come from (torchvision, timm, Hugging Face), and why the "
            "preprocessing must match their weights",
            "Regime A: run the frozen backbone once, cache the features, train a linear probe (exercise 1)",
            "Regimes B and C: freeze stages (exercise 2), give each stage its own learning rate (exercise 3), "
            "train with augmentation; choose a regime for four situations (exercise 4, written)",
            "How much data do you need? Accuracy against training-set size",
            "What the network sees: first-layer filters, feature maps through the stages (hooks), the feature space (PCA)",
            "Grad-CAM with hooks (exercise 5), and a check for shortcut learning",
            "Choosing and reading metrics (ML Yearning): a mini case study on Pet",
            "Modern note: self-supervised features (DINOv2, optional)",
        ],
        topics=[
            "Reuse an ImageNet backbone: the preprocessing must match its weights, the head must be replaced",
            "Linear probe vs. frozen backbone vs. fine-tuning: cost, accuracy, and when to use which",
            "Freeze stages with `requires_grad`; keep BatchNorm in evaluation mode; give earlier stages smaller "
            "learning rates with optimiser parameter groups",
            "Accuracy grows roughly linearly in log(training-set size): measure it before buying labels",
            "Read a network's insides with forward hooks and tensor hooks; compute Grad-CAM; check for shortcut learning",
            "Single-number metric, satisficing metrics, dev / test sets from the same distribution",
        ],
        minutes=120,
        setting="in class (guided + hands-on); about 90 if the instructor demonstrates sections 4, 7 and 8",
        prereq="D0-00 (tensors, autograd, the training loop), D1-01 (convolution, feature maps, global average "
               "pooling), D1-02 (training recipe: augmentation, AdamW, BatchNorm, label smoothing, learning-rate "
               "schedules) and D1-03 (ResNet and its stages). The Pet subset of D1-02, cached in `data/checkpoints/`.",
    ),
    md(r"""
    ## How to read this notebook

    Run the cells in order with Shift+Enter (D0-00 has the long version of these notes). The five
    exercises sit under headings `### Exercise N`. Each code exercise is a **template**: the function's
    first line, its docstring, numbered steps and the bookkeeping lines are written for you; you replace
    each `____` under a comment `# YOUR CODE:` with one or two lines. First delete the line
    `raise NotImplementedError`: it marks the exercise as not done (when a checker's hint says "replace
    `raise NotImplementedError` with your code", it means: delete that line and fill in the `____`). The
    text above every exercise gives the goal, an example small enough to check by hand, a plan in steps,
    what the checker tests, and a table of common mistakes with what the checker then prints. The checker
    prints `[PASS]`, or `[FAIL]` with a hint. An unfinished exercise does not stop the notebook: the cells
    after it print `note: using the reference implementation ...` and carry on with the course's own
    solution. Stuck for more than five minutes? Read that one cell in
    `notebooks/solutions/D1-04-transfer-learning-solution.ipynb`, close it, and type it yourself. The
    closed boxes **Functions used here** explain each library function before its first use; boxes
    titled **More:** are optional depth. New words are in **bold** where they are explained, and
    collected in "Words from this notebook" at the end.
    """),
    md(r"""
    ## 0 · Setup

    **Why this notebook.** Training a network from scratch needs many labelled images. D1-02's PetNet,
    trained from scratch on 2 000 Pet photos of 37 breeds, reached about 13 % validation accuracy.
    **Transfer learning** reuses a network that someone else already trained on more than a million
    photos, and adapts it to a new task. On the same 2 000 photos it reaches about 80 % in this
    notebook, after seconds to minutes of training on a laptop CPU. Almost every practical vision model
    is built this way, and the idea comes back all week. On Day 2, D2-02 adapts a pretrained detection
    network to new data and D2-03 builds a detector on a pretrained ResNet-18. On Day 3, D3-02 puts a
    pretrained ResNet-18 inside a segmentation network and trains that part with a smaller learning
    rate than the rest, D3-03 freezes the pretrained part of an instance-segmentation network, and
    D3-04 compares linear probes on several pretrained networks. This notebook explains every one of
    those ideas: pretrained parts, freezing, per-part learning rates, linear probes.

    **The setup cell**, line by line:

    * it finds the course helpers and imports them, as in every notebook; `day1` holds Day 1's helpers
      and the reference solutions of the exercises, `labs` the interactive labs;
    * `OMP_WAIT_POLICY` and `day1.pick_cpu_threads()` only make PyTorch faster on a busy CPU;
    * it chooses the **backbone** (the pretrained network we reuse; section 1 explains the word) and the
      image size from your hardware. The course standard is **ResNet-50** at 224 x 224 pixels on a GPU.
      On a CPU the same cells use **ResNet-18 at 160 px**: 0.9 instead of 4.1 billion multiply-adds per
      image (one multiply-add is one multiplication plus one addition, the basic step of a network), so
      the whole notebook runs in a few minutes;
    * `N_FT` and `FT_EPOCHS` set how many training images and epochs regimes B and C use (section 3).

    The numbers quoted in the text come from the CPU run (ResNet-18, 160 px); where a GPU run gives
    clearly different numbers, those are given too. Expensive results (cached features, trained weights)
    are saved as files in `data/checkpoints/` and loaded when you run the notebook again.
    """),
    api("torch.manual_seed", "torchkit.get_device", brief=True),
    code(r"""
    import os, sys, pathlib
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")   # before importing torch: idle CPU threads sleep, not spin
    for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:   # find the cvcourse package
        if (p / "cvcourse").is_dir():
            sys.path.insert(0, str(p)); break

    import math, time
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import torchvision
    from torchvision.transforms import v2
    from torchvision.models import get_model, get_model_weights
    from cvcourse import viz, tests, torchkit, day1, labs

    %matplotlib inline
    plt.rcParams["figure.dpi"] = 90
    torch.manual_seed(0)
    device = torchkit.get_device()
    if device.type == "cpu":
        day1.pick_cpu_threads()     # on a busy CPU, 1 thread can be faster than 8

    GPU = device.type == "cuda"
    if GPU:
        BACKBONE = "resnet50"
        IMG = 224                      # image size in pixels (height = width)
        FT_EPOCHS = 3                  # epochs for the augmented regimes B and C
    else:
        BACKBONE = "resnet18"
        IMG = 160
        FT_EPOCHS = 1
    N_FT = 2000                        # images used for the augmented regimes B and C (all of the subset)
    print(f"backbone {BACKBONE} at {IMG} px; regimes B / C on {N_FT} images x {FT_EPOCHS} epoch(s)")
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · Model zoos

    ### Words first

    * **ImageNet** is the dataset most pretrained vision networks learned from: 1.28 million training
      photos, each labelled with one of 1 000 classes (about 120 dog breeds and a few cat breeds among
      them, plus vehicles, food, tools, animals, ...).
    * A **pretrained** network (D0-00) is one that somebody already trained, here on ImageNet. Its
      **weights** (D0-00: its parameters) are the result of that training; you download them as a file.
    * A **model zoo** is a library of network architectures together with such downloadable weights.
    * A classifier has two parts. The **backbone** is everything except the last layer: it turns an
      image into a vector of numbers, the **features** of the image (also called its **embedding**). The
      **head** is the last layer: here a linear layer that turns the features into one score per class.
      In a torchvision ResNet the head is called `fc` ("fully connected"); ResNet-18's head turns 512
      features into 1 000 ImageNet scores.

    **Why the backbone transfers.** Training on ImageNet costs thousands of GPU-hours. In that training
    the early layers learned to detect edges, colours and textures, the later layers object parts such as
    eyes, ears and fur patterns (section 5 shows them). Those are useful for almost any task on ordinary
    photos, so we keep the backbone and replace the head with one for our 37 breeds.

    | library | one line that makes a pretrained model | notes |
    |---|---|---|
    | torchvision | `get_model("resnet50", weights="IMAGENET1K_V2")` | about 80 classic models; the weights object also knows its preprocessing and its accuracy |
    | timm | `timm.create_model("convnext_tiny.fb_in22k", pretrained=True, num_classes=37)` | more than 1 000 models, including the newest; replaces the head for you |
    | Hugging Face | `AutoModel.from_pretrained("facebook/dinov2-small")` | transformers (another family of networks, section 8 and Day 3), and models that combine images with text: Day 3 |

    ### The preprocessing must match the weights

    **Preprocessing** is what is done to an image before the network sees it: resize, crop, convert to
    float in [0, 1], normalise with a mean and std per channel (D0-00). The weights were learned on
    images prepared one particular way. Prepare yours differently (another mean and std, or values 0 to
    255 instead of 0 to 1) and every input number is off from what the weights expect: nothing raises an
    error, the accuracy just drops. The weights object tells you its recipe. The cell prints it for our
    backbone, then a table of six torchvision models:

    * **top-1** is the **top-1 accuracy** on ImageNet's validation photos, in %: the fraction of photos
      whose highest-scoring class is the right one;
    * **params (M)** is the number of parameters in millions;
    * **GFLOPs** is the compute for one 224 x 224 image, in billions of operations (torchvision counts
      one multiply-add as one operation).
    """),
    api("torchvision.models.get_model_weights", "torchvision.models.get_model"),
    code(r"""
    weights = get_model_weights(BACKBONE).DEFAULT          # the recommended ImageNet weights of our backbone
    print(repr(weights))
    print("preprocessing:", weights.transforms())          # the recipe these weights were trained with

    # One row per model: its weights, ImageNet top-1, size and compute (metadata only: nothing is downloaded here)
    print(f"\n{'model':<20}{'weights':<16}{'top-1':>7}{'params (M)':>12}{'GFLOPs':>8}")
    for name in ["resnet18", "resnet50", "mobilenet_v3_large", "efficientnet_b0", "convnext_tiny", "vit_b_16"]:
        w = get_model_weights(name).DEFAULT
        weights_name = str(w).split(".")[-1]               # "ResNet18_Weights.IMAGENET1K_V1" -> "IMAGENET1K_V1"
        top1 = w.meta["_metrics"]["ImageNet-1K"]["acc@1"]   # ImageNet top-1 accuracy, in %
        params_m = w.meta["num_params"] / 1e6              # number of parameters, in millions
        gflops = w.meta.get("_ops", float("nan"))          # billions of operations per 224 px image
        print(f"{name:<20}{weights_name:<16}{top1:>7.1f}{params_m:>12.1f}{gflops:>8.2f}")
    """),
    md(r"""
    **Reading the printout.** `resize_size=[256]` and `crop_size=[224]`: make the shorter side 256
    pixels, then cut out the central 224 x 224 square. `mean` and `std` are the ImageNet numbers of
    D0-00. On a CPU we feed 160 x 160 images instead: a convolutional network accepts other sizes, at the
    cost of some accuracy. The mean and std, on the other hand, are always the ImageNet ones.
    `day1.eval_transform(IMG)` (next cell) applies exactly this recipe at our size: resize, centre crop,
    convert to float in [0, 1], normalise.

    In the table, bigger networks are more accurate and more expensive: ResNet-50 has 2.2 times the
    parameters of ResNet-18 and 2.3 times the compute, for 11 points more top-1 (80.9 % against 69.8 %).

    ### Before any training: the ImageNet head on Pet photos

    The ImageNet classifier already knows a lot about pets: about 25 of the 37 Pet breeds are also
    ImageNet classes. The next cell loads the Pet subset of D1-02 (2 000 training and 500 validation
    photos of 37 breeds), takes six validation photos, and shows the ImageNet model's three most likely
    classes for each. Three words:

    * the **logits** are the network's raw outputs, one score per class (here 1 000);
    * **softmax** (D1-02) turns them into probabilities: positive numbers that add up to 1;
    * `torch.topk(p, 3)` returns the 3 largest values of `p` and their positions, here the class indices;
      `weights.meta["categories"]` holds the 1 000 class names.
    """),
    api("Tensor.softmax"),
    api("Module.eval", "Tensor.to", "torch.no_grad", "torch.stack", "Tensor.cpu", "torch.topk", "torchkit.denormalize",
        brief=True),
    code(r"""
    data = day1.pet_tensors(176)                    # same 2 000 / 500 subset as D1-02, uint8 images decoded at 176 px
    classes = data["classes"]                       # the 37 breed names
    short = []                                      # the breed names cut to 16 characters, for plot titles
    for breed in classes:
        if len(breed) <= 16:
            short.append(breed)
        else:
            short.append(breed[:15] + ".")
    eval_tf = day1.eval_transform(IMG)              # resize + centre crop + float in [0, 1] + ImageNet mean / std

    imagenet_model = get_model(BACKBONE, weights=weights).eval().to(device)   # the whole ImageNet classifier, head included
    imagenet_classes = weights.meta["categories"]   # the 1 000 ImageNet class names
    idx = [3, 60, 130, 222, 301, 444]               # six validation photos
    images = []
    for i in idx:
        images.append(eval_tf(data["x_val"][i]))    # (3, IMG, IMG), normalised
    batch = torch.stack(images).to(device)          # (6, 3, IMG, IMG)
    with torch.no_grad():
        logits = imagenet_model(batch)              # (6, 1000): one score per ImageNet class
    probs = logits.softmax(1).cpu()                 # (6, 1000): probabilities, every row adds up to 1

    # Draw each photo with its true breed and the ImageNet model's three most likely classes
    fig, axes = plt.subplots(1, len(idx), figsize=(15, 3.6))
    for ax, i, p in zip(axes, idx, probs):
        ax.imshow(torchkit.denormalize(eval_tf(data["x_val"][i]))); ax.axis("off")
        top = torch.topk(p, 3)                      # the 3 largest probabilities and their class indices
        title = f"Pet: {short[data['y_val'][i]]}"
        for prob, k in zip(top.values, top.indices):
            title += f"\n{prob:.2f} {imagenet_classes[int(k)][:18]}"
        ax.set_title(title, fontsize=8, loc="left")
    plt.suptitle("Zero training: the ImageNet head's top-3 guesses", y=1.03); plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: with no training on Pet at all, the ImageNet model names the right breed for four of
    the six photos (beagle 0.67, Great Pyrenees 0.93, miniature pinscher 0.86, Siamese cat 0.98). The
    British Shorthair is not an ImageNet class, and the model picks the closest one it has, "Persian cat";
    the upside-down Abyssinian becomes a Chihuahua. The backbone's features are clearly good; what is
    missing is a head with *our* 37 classes. That is what we train next.

    ## 2 · Regime A: the linear probe

    **Why.** This notebook compares three **regimes**, three ways of setting up transfer learning that
    train more and more of the network: A (this section), B and C (section 3). Regime A is the cheapest
    way to reuse a backbone, and the baseline every other method must beat. D3-04 uses the same method
    on Day 3 to compare modern backbones.

    **Words.**
    * **Feature extraction**: running images through a backbone and keeping its output vectors, the
      features. For ResNet-18 that is 512 numbers per image (2 048 for ResNet-50): the 512 channels that
      its last convolution layers output, each averaged over the whole image by **global average
      pooling** (D1-01).
    * A **frozen** parameter is one that training does not change. A frozen backbone is used as it is.
    * A **linear probe** is a single linear layer (`nn.Linear(512, 37)`, D0-00) trained on the features of
      a frozen backbone. It can only draw straight boundaries between the classes in feature space, so
      its accuracy measures how much the features already "know" about the task.
    * **Caching** means computing something once and storing it so that it is never recomputed: here the
      features of all 2 500 images, in a file.

    **The plan.** Cut off the ImageNet head: `backbone.fc = nn.Identity()` replaces the last layer by one
    that returns its input unchanged, so the backbone now outputs its 512 features. Run every image
    through it **once**, store the features, and train the linear probe on them. Training then takes
    about a second, because the expensive backbone never runs again. The price: no **data augmentation**
    (D1-02: random crops, flips and colour changes that show the model a slightly different version of
    every training image in every epoch), because each image has exactly one cached feature vector.

    ### Exercise 1 · `extract_features`

    **Goal.** Run a backbone over every batch of a DataLoader and return all the features and all the
    labels, as two tensors.

    **Why by hand.** It is the core of every frozen-feature method (regime A here, regime B in section 3,
    DINOv2 in section 8, D3-04 on Day 3), and two details in it change the numbers without any error:

    * **`backbone.eval()`** switches every layer to **evaluation mode**. Two kinds of layer behave
      differently in training and in evaluation (D1-02). **BatchNorm** normalises each channel: in
      training mode with the mean and std of the current batch, in evaluation mode with averages stored
      during training, its **running statistics**. **Dropout** sets random activations to 0 in training
      mode and does nothing in evaluation mode. In training mode the same image would get features that
      depend on the other images in its batch, plus random zeros.
    * **`with torch.no_grad():`** (D0-00): autograd records nothing inside this block. We never call
      `backward()` here, so the record would only cost time and memory, and the features would stay
      attached to it.

    **Input.**
    * `backbone`: a network whose output for a batch of B images is (B, D), or (B, D, 1, 1), which
      `.flatten(1)` turns into (B, D) (it keeps axis 0 and joins all the others);
    * `loader`: a DataLoader that yields batches `(x, y)`: images x (B, 3, H, W) and labels y (B,); the
      last batch can be smaller;
    * `device`: where the backbone's parameters are (`cpu` here).

    **Output.** A pair `(features, labels)`: `features` a float tensor (N, D) on the CPU, one row per
    image in loader order, and `labels` a tensor (N,), where N is the number of images.

    **Example, by hand.** Five tiny images of 3 channels and 2 x 2 pixels: image i has the value
    10 i + c at every pixel of channel c. Their labels are 7, 3, 3, 0, 1. The "backbone" is
    `nn.AdaptiveAvgPool2d(1)`, global average pooling alone: one number per channel, the channel's mean.
    So image i gives the feature row [10 i, 10 i + 1, 10 i + 2]. The loader serves batches of 2, so
    2 + 2 + 1 images. The next cell prints the three batches; your function must join them into
    ```
    features = tensor([[ 0.,  1.,  2.],
                       [10., 11., 12.],
                       [20., 21., 22.],
                       [30., 31., 32.],
                       [40., 41., 42.]])        # (5, 3)
    labels   = tensor([7, 3, 3, 0, 1])          # (5,)
    ```
    """),
    api("torch.zeros", "torch.tensor", "torch.utils.data.TensorDataset", "torch.utils.data.DataLoader",
        "nn.AdaptiveAvgPool2d", "Tensor.flatten", "Tensor.tolist", brief=True),
    code(r"""
    # The worked example: 5 tiny images, a "backbone" that only averages each channel, batches of 2
    toy_images = torch.zeros(5, 3, 2, 2)                  # (N=5, C=3, H=2, W=2)
    for i in range(5):
        for c in range(3):
            toy_images[i, c] = 10 * i + c                 # image i, channel c: every pixel is 10 i + c
    toy_labels = torch.tensor([7, 3, 3, 0, 1])
    toy_loader = DataLoader(TensorDataset(toy_images, toy_labels), batch_size=2)
    toy_backbone = nn.AdaptiveAvgPool2d(1)                # global average pooling: (B, 3, 2, 2) -> (B, 3, 1, 1)

    for x, y in toy_loader:
        out = toy_backbone(x)                             # (B, 3, 1, 1)
        feats = out.flatten(1)                            # (B, 3)
        print("x", tuple(x.shape), "y", y.tolist(), "-> backbone", tuple(out.shape),
              "-> flatten(1)", tuple(feats.shape), feats.tolist())
    """),
    md(r"""
    Three batches: (2, 3), (2, 3) and (1, 3) features. Joining them along axis 0 gives the (5, 3) above.

    **Plan.** The template below has these steps; the lines marked `# YOUR CODE:` are yours.
    1. Switch the backbone to evaluation mode.
    2. Two empty lists, one for the feature batches and one for the label batches (written).
    3. Inside `with torch.no_grad():`, loop over the loader (written). For each batch: move x to the
       device (written); compute the backbone's output and flatten it to (B, D); move it to the CPU with
       `.cpu()` and append it to the list; append y (written).
    4. After the loop, join the list of feature batches into one tensor with `torch.cat`, which joins
       tensors along an axis that already exists (axis 0 unless you say otherwise): (2, 3), (2, 3) and
       (1, 3) become (5, 3). The labels the same way (written).
    5. Return the pair (written).

    **What the checker tests.** A small random backbone with a BatchNorm and a Dropout layer, left in
    training mode, and 10 random images in batches of 4, 4 and 2. It compares your features with
    `backbone(x)` in evaluation mode, checks that they do not require gradients (they would if
    `no_grad` were missing), and that the labels come back in loader order.

    **Common mistakes.**

    | you wrote | the checker prints |
    |---|---|
    | no `backbone.eval()`, or only after the loop | `features differ from backbone(x) in eval mode: call backbone.eval() (BatchNorm / Dropout!)` |
    | no `with torch.no_grad():` | `features still require grad: wrap the loop in torch.no_grad() (saves memory and time)` |
    | `torch.stack` instead of `torch.cat` | `your code raised RuntimeError: stack expects each tensor to be equal size, but got [4, 6] at entry 0 and [2, 6] at entry 2 (...)`: `stack` makes a new axis and needs equal shapes, and the last batch is smaller |
    | `torch.cat(feature_batches, dim=1)` | `your code raised RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 4 but got size 2 for tensor number 2 in the list. (...)` |
    | the empty lists created inside the loop | `got features (2, 6) and labels (2,), expected (10, 6) and (10,)`: only the last batch survived |
    | returned the lists, without `torch.cat` | `your code raised AttributeError: 'list' object has no attribute 'shape' (...)` |

    Forgetting `.flatten(1)` passes this checker (its backbone ends with a flatten layer), but the toy
    backbone above would then give features of shape (5, 3, 1, 1).
    """),
    api("Module.eval", "torch.no_grad", "Tensor.to", "Tensor.flatten", "Tensor.cpu", "torch.cat",
        title="Functions you will need", brief=True),
    exercise(
        stub=r"""
        def extract_features(backbone, loader, device=device):
            '''Run `backbone` over every batch of `loader` without training it.

            Returns (features, labels): features (N, D) float on the CPU, labels (N,), in loader order.'''
            raise NotImplementedError   # delete this line, then replace every ____ below

            # 1. Evaluation mode: BatchNorm uses its running statistics, Dropout is off.
            # YOUR CODE: switch the backbone to evaluation mode
            ____

            # 2. One list for the feature batches, one for the label batches.
            feature_batches = []
            label_batches = []

            # 3. No autograd record inside this block: faster, much less memory.
            with torch.no_grad():
                for x, y in loader:                         # x: (B, 3, H, W) images, y: (B,) labels
                    x = x.to(device)                        # the images go where the backbone is
                    # YOUR CODE: the backbone's output for x, flattened to (B, D) with .flatten(1)
                    features = ____                          # (B, D)
                    # YOUR CODE: append the features, moved to the CPU with .cpu(), to feature_batches
                    ____
                    label_batches.append(y)                 # (B,)

            # 4. Join the batches along axis 0: (B1, D), (B2, D), ... -> (N, D).
            # YOUR CODE: join feature_batches into one tensor with torch.cat
            features = ____                                  # (N, D)
            labels = torch.cat(label_batches)               # (N,)

            # 5. The pair.
            return features, labels

        ok_feat = tests.check_extract_features(extract_features)
        """,
        solution=r"""
        def extract_features(backbone, loader, device=device):
            '''Run `backbone` over every batch of `loader` without training it.

            Returns (features, labels): features (N, D) float on the CPU, labels (N,), in loader order.'''

            # 1. Evaluation mode: BatchNorm uses its running statistics, Dropout is off.
            # YOUR CODE: switch the backbone to evaluation mode
            backbone.eval()

            # 2. One list for the feature batches, one for the label batches.
            feature_batches = []
            label_batches = []

            # 3. No autograd record inside this block: faster, much less memory.
            with torch.no_grad():
                for x, y in loader:                         # x: (B, 3, H, W) images, y: (B,) labels
                    x = x.to(device)                        # the images go where the backbone is
                    # YOUR CODE: the backbone's output for x, flattened to (B, D) with .flatten(1)
                    features = backbone(x).flatten(1)       # (B, D)
                    # YOUR CODE: append the features, moved to the CPU with .cpu(), to feature_batches
                    feature_batches.append(features.cpu())
                    label_batches.append(y)                 # (B,)

            # 4. Join the batches along axis 0: (B1, D), (B2, D), ... -> (N, D).
            # YOUR CODE: join feature_batches into one tensor with torch.cat
            features = torch.cat(feature_batches)           # (N, D)
            labels = torch.cat(label_batches)               # (N,)

            # 5. The pair.
            return features, labels

        ok_feat = tests.check_extract_features(extract_features)
        """),
    md(r"""
    **Did it work?** `[PASS]` means your function returns exactly the features of the backbone in
    evaluation mode. The next cell uses it (or the reference, if yours has not passed yet) on the real
    backbone: our ImageNet ResNet with its head replaced by `nn.Identity()`. `day1.TensorImageDataset`
    pairs each stored image with its label and applies `eval_tf` when the image is read. The first run
    extracts the features of all 2 500 images and caches them in `data/checkpoints/`; later runs load the
    file. Expect `train features (2000, 512) | val features (500, 512)`: one vector of 512 numbers per
    image. (`F_` is the dict that holds the four cached tensors.)
    """),
    api("nn.Identity", "torch.save", "torch.load", brief=True),
    code(r"""
    extract = tests.use_or_reference(ok_feat, globals().get("extract_features"), day1.extract_features, "extract_features")

    backbone = get_model(BACKBONE, weights=weights)         # the ImageNet ResNet, head included
    feat_dim = backbone.fc.in_features                      # 512 for ResNet-18 (2 048 for ResNet-50): what the head reads
    backbone.fc = nn.Identity()                             # remove the head: the network now outputs its features
    backbone = backbone.to(device)

    feat_path = torchkit.CHECKPOINTS / f"d1-04_feats_{BACKBONE}_{IMG}.pt"
    if feat_path.exists():
        F_ = torch.load(feat_path)                          # {"f_train", "y_train", "f_val", "y_val"}
        print(f"features loaded from {feat_path.name}")
    else:
        t0 = time.time()
        F_ = {}
        for split in ("train", "val"):
            dataset = day1.TensorImageDataset(data[f"x_{split}"], data[f"y_{split}"], eval_tf)
            dl = DataLoader(dataset, batch_size=50)
            F_[f"f_{split}"], F_[f"y_{split}"] = extract(backbone, dl)
        torch.save(F_, feat_path)
        print(f"features extracted in {time.time() - t0:.0f} s and cached to {feat_path.name}")
    print("train features", tuple(F_["f_train"].shape), "| val features", tuple(F_["f_val"].shape))
    """),
    md(r"""
    Now the probe itself. `day1.train_linear_probe` trains `nn.Linear(512, 37)` on the 2 000 cached
    training features and returns three things: the trained layer, the **validation accuracy** (the
    fraction of the 500 validation images whose highest score is the right breed) and a training history.
    Each of its 100 epochs is a few matrix multiplications on 2 000 x 512 numbers, so it takes about a
    second.

    <details>
    <summary><b>More: what <code>day1.train_linear_probe</code> does inside</b> (optional, click to open)</summary>

    1. It **standardises** each of the 512 features: subtracts its mean over the training images and
       divides by its standard deviation (D0-00's normalisation, per feature instead of per colour
       channel), so that all features are on the same scale.
    2. It trains `nn.Linear(512, 37)` for 100 epochs in batches of 256 with AdamW (learning rate 0.01,
       weight decay 0.001), cross-entropy with label smoothing 0.1 (section 3 recalls both), and a cosine
       schedule that lowers the learning rate to 0.
    3. It folds the standardisation into the layer's weights and bias, so that the returned layer works
       on raw features. That is why regimes B and C can load it as their starting head.

    </details>
    """),
    code(r"""
    t0 = time.time()
    probe_head, acc_a, _ = day1.train_linear_probe(F_["f_train"], F_["y_train"], F_["f_val"], F_["y_val"], epochs=100)
    print(f"Regime A - linear probe on cached features: val accuracy {acc_a:.3f}   ({time.time() - t0:.1f} s of training)")
    """),
    md(r"""
    About **79 %** on 37 breeds with ResNet-18 at 160 px, after about one second of training on a CPU
    (ResNet-50 at 224 px: about 88-90 %). The probe starts from random weights, so the third digit can
    differ between runs (0.788 with every exercise solved, 0.786 with Exercise 1 still open). D1-02's PetNet, trained from scratch on the same 2 000
    images, reached about 13 %. That gap is the whole argument for transfer learning on small data.

    ## 3 · Regimes B and C: training through the backbone

    The linear probe never changes the backbone and never sees an augmented image. The two more
    expensive regimes remove those limits:

    | regime | what is trained | augmentation | cost per epoch | use it when |
    |---|---|---|---|---|
    | A linear probe | the head only, on cached features | no | almost 0 (the features are computed once) | very little data, a quick baseline, many experiments |
    | B frozen backbone + head | the head only | yes | one forward pass of the backbone per image | little data, photos similar to ImageNet's |
    | C fine-tune | the last stages and the head (C': everything) | yes | a forward and a backward pass | more data, or images far from ImageNet (medical scans, satellite images) |

    **Fine-tuning** means training a pretrained network further on your data, with a small learning
    rate, so that its weights move a little instead of being learned from scratch.

    **The parts of a ResNet.** A torchvision ResNet (D1-03) is a chain of six parts, and we freeze or train
    whole parts at a time:
    * the **stem**: the first convolution `conv1` and its BatchNorm `bn1` (plus a ReLU and a max-pool,
      which have no parameters); it shrinks the image 4 times;
    * four **stages** `layer1` to `layer4`: groups of residual blocks; each stage after the first halves
      the height and width and doubles the number of channels;
    * the head `fc`.

    Early stages compute generic patterns (edges, colours), late stages task-specific ones (section 5
    shows this). So the late stages are the ones that need to adapt.

    **Practical rules for regime C.**
    1. **Train the head first.** A new head starts with random weights. Its large, random errors would
       flow back as large gradients into the good pretrained features and damage them. So C starts from
       the probe's trained head.
    2. **A small learning rate**: 1e-4 to 1e-3 with AdamW, about 10 times below training from scratch.
       **AdamW** (D1-02) is the optimiser of the course: like SGD it moves every parameter against its
       gradient, but it adapts each parameter's step size from the history of its gradients, and it
       applies **weight decay** separately (the "W"): every step pulls each weight a little towards 0,
       which discourages extreme weights.
    3. **Discriminative learning rates**: a different learning rate per stage, smaller for earlier stages
       (generic features: change them little), larger for later ones. Exercise 3 builds them.
    4. **BatchNorm in evaluation mode**: with little data and batches of 32 images, the mean and std of
       each batch are noisy; the running statistics learned on ImageNet are better. BatchNorm's two
       learned parameters per channel (a scale and a shift) still train.
    5. **Unfreeze from the top**: when you train more than the head, start with layer4, then layer3, ...

    The figure draws the four regimes as rows of the six parts: orange parts are trained, blue ones frozen.
    """),
    code(r"""
    # Diagram: one row per regime, one box per ResNet part (orange = trained, blue = frozen), and the
    # discriminative learning rates of C' underneath
    stages = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]
    regimes = {                                     # 1 = trained, 0 = frozen, for the six parts
        "A  linear probe":         [0, 0, 0, 0, 0, 1],
        "B  frozen + head (aug)":  [0, 0, 0, 0, 0, 1],
        "C  fine-tune top stages": [0, 0, 0, 1, 1, 1],
        "C' full fine-tune":       [1, 1, 1, 1, 1, 1],
    }
    lr_text = ["3e-4 x 0.3^5\n= 7.3e-7", "3e-4 x 0.3^4\n= 2.4e-6", "3e-4 x 0.3^3\n= 8.1e-6",
               "3e-4 x 0.3^2\n= 2.7e-5", "3e-4 x 0.3\n= 9e-5", "3e-4"]
    fig, ax = plt.subplots(figsize=(12, 3.9)); ax.axis("off"); ax.set_xlim(-2.6, 6.2); ax.set_ylim(-1.4, 4.3)
    row = 0
    for name, trained in regimes.items():
        y = 3.4 - row
        ax.text(-2.5, y, name, va="center", fontsize=10)
        for k in range(len(stages)):
            if trained[k]:
                colour = "#f8d0b8"                  # orange: trained
            else:
                colour = "#d9e3ef"                  # blue: frozen
            ax.add_patch(FancyBboxPatch((k - 0.42, y - 0.3), 0.84, 0.6, boxstyle="round,pad=0.02", fc=colour, ec="k", lw=0.8))
            ax.text(k, y, stages[k], ha="center", va="center", fontsize=8)
        row += 1
    for k in range(len(stages)):
        ax.text(k, -0.95, lr_text[k], ha="center", fontsize=8, color=viz.PALETTE[3])
    ax.text(-2.5, -0.75, "LR in C' (discriminative)", fontsize=8, color=viz.PALETTE[3])
    ax.text(5.6, 3.4, "orange = trainable\nblue = frozen", fontsize=8, va="center")
    ax.set_title("Which layers to unfreeze: generic early features stay, task-specific late features adapt")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: A and B train the same part, the head. They differ only in whether the features are
    cached once (A) or recomputed from augmented images in every epoch (B). The red numbers under C' are
    its discriminative learning rates for a base learning rate of 3e-4 and a decay of 0.3: each stage gets
    0.3 times the learning rate of the stage above it, so the stem's 7.3e-7 is about 400 times smaller
    than the head's 3e-4.

    ### Lab · Freeze, unfreeze, discriminative LRs
    **Predict first:** in regime C you freeze `stem`, `layer1` and `layer2` of ResNet-50. Roughly what
    fraction of its parameters is still trained: a quarter, a half, or more? The lab shows the six stages
    with their true parameter counts; click to freeze / unfreeze and move the LR sliders (4 minutes).
    """),
    code(r"""
    labs.show("freeze-unfreeze")
    """),
    md(r"""
    ### Parameters have names

    To freeze a stage we must know which parameters belong to it. `model.named_parameters()` yields every
    parameter tensor together with its **name**: the path of attribute names that leads to it in the
    model. `layer3.1.conv2.weight` is the weight of `conv2` in block 1 of `layer3`. The stage is the part
    of the name before the first dot, except that the stem's parameters start with `conv1` or `bn1`. The
    next cell defines two helpers that Exercises 2 and 3 use:

    * `ORDER`, the six parts from input to output; a part's position in this list (0 for the stem, 5 for
      `fc`) is how we compare "earlier" and "later";
    * `stage_of(name)`, the part a parameter belongs to.

    It then walks through the parameters of the ImageNet ResNet of section 1, prints the first few (and
    the head's), and adds up the parameters of each part.
    """),
    api("Module.named_parameters"),
    api("Tensor.numel", brief=True),
    code(r"""
    ORDER = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]     # the six parts of a ResNet, input to output


    def stage_of(param_name):
        '''The part a parameter belongs to: "layer3.1.conv2.weight" -> "layer3"; "conv1.weight", "bn1.bias" -> "stem".'''
        top = param_name.split(".")[0]            # the name up to the first dot
        if top in ("conv1", "bn1"):
            return "stem"
        return top


    # Walk through the parameters of the ImageNet ResNet of section 1: print a few, add up each part
    params_per_stage = {}
    for stage in ORDER:
        params_per_stage[stage] = 0
    n_tensors = 0
    for name, param in imagenet_model.named_parameters():
        stage = stage_of(name)
        if n_tensors < 5 or stage == "fc":
            print(f"{name:<24}{str(tuple(param.shape)):<17}{param.numel():>9,} numbers  -> {stage}")
        params_per_stage[stage] += param.numel()  # numel(): the number of elements of the tensor
        n_tensors += 1

    total = sum(params_per_stage.values())
    print(f"\n{n_tensors} parameter tensors, {total:,} parameters in all:")
    for stage in ORDER:
        share = params_per_stage[stage] / total * 100
        print(f"   {stage:<7}{params_per_stage[stage]:>12,}  {share:5.1f} %")
    """),
    md(r"""
    What to look at: ResNet-18 has 62 parameter tensors with 11,689,512 parameters in all. layer4 alone
    holds 71.8 % of them, the stem 0.1 %. Freezing everything up to layer3 still leaves three quarters of
    the network to train.

    **The switch is `requires_grad`** (D0-00). A parameter with `requires_grad = False` gets no gradient in
    `backward()`, so the optimiser never changes it: it is frozen. With `True` it trains.

    ### Exercise 2 · `freeze(model, until)`

    **Goal.** Make every parameter of the parts up to and including `until` frozen, and every parameter
    of the later parts trainable.

    **Why by hand.** It is one loop, but it is the tool of regimes B and C, and D3-03 freezes the
    backbone of Mask R-CNN the same way on Day 3. (`until` is the lab's "freeze until" selector:
    everything up to the dashed line.)

    **Input.** `model`: a torchvision ResNet; `until`: one of the names in `ORDER`, for example `"layer3"`.

    **Output.** The same `model`, changed in place, with `requires_grad` set on every parameter.

    **Example, by hand.** `until="layer3"`. Its position in `ORDER` is 3 (counting from 0); call it `cut`.

    | parameter name | `stage_of(name)` | position in `ORDER` | position > cut? | `requires_grad` |
    |---|---|---|---|---|
    | `conv1.weight` | stem | 0 | 0 > 3: no | False |
    | `layer1.0.conv1.weight` | layer1 | 1 | 1 > 3: no | False |
    | `layer3.1.bn2.bias` | layer3 | 3 | 3 > 3: no | False |
    | `layer4.0.conv1.weight` | layer4 | 4 | 4 > 3: yes | True |
    | `fc.bias` | fc | 5 | 5 > 3: yes | True |

    On ResNet-18 with its 1 000-class head this leaves layer4 and fc trainable: 8,393,728 + 513,000 =
    8,906,728 parameters.

    **Plan.**
    1. `cut = ORDER.index(until)` (written): the position of `until` in the list.
    2. Loop over `model.named_parameters()` (written).
    3. For each parameter, the position of its part: `stage_of(name)` gives the part's name,
       `ORDER.index(...)` its position.
    4. Set `param.requires_grad` to the comparison `position > cut`. A comparison already is `True` or
       `False`, so no `if` is needed.
    5. Return the model (written).

    Set `True` as well as `False`: the same model is often frozen several times (first only the head is
    trained, later more stages), and each call must leave exactly the parts after `until` trainable.

    **What the checker tests.** A fresh ResNet-18 for each of `until` = `"layer3"`, `"layer4"`, `"stem"`
    and `"layer1"`: it counts the trainable parameters and compares with the parts after `until`. Then it
    freezes every parameter itself, calls `freeze(model, "layer2")`, and checks that layer3, layer4 and fc
    became trainable again.

    **Common mistakes.**

    | you wrote | the checker prints |
    |---|---|
    | `position >= cut` | `freeze(model, until='layer3'): 11,006,440 trainable params, expected 8,906,728 (only layer4, fc trainable)` and three more such lines: the stage `until` itself stayed trainable |
    | only `if position <= cut: param.requires_grad = False` | `freeze must also UNfreeze the later stages (set requires_grad=True there), so it can be called repeatedly` |
    | `position < cut` | `freeze(model, until='layer3'): 683,072 trainable params, expected 8,906,728 (...)`: the early parts trained, the late ones frozen |
    | `ORDER.index(name.split(".")[0])` without `stage_of` | `your code raised ValueError: list.index(x): x not in list (...)`: `conv1` and `bn1` are not in `ORDER`; `stage_of` turns them into `stem` |
    | `stage_of(name) > until`, comparing the names | `freeze(model, until='layer3'): 8,403,264 trainable params, expected 8,906,728 (...)`: words compare alphabetically ("stem" comes after "layer3", "fc" before it); compare positions |
    """),
    api("Tensor.requires_grad_", title="Functions you will need"),
    api("Module.named_parameters", "Tensor.requires_grad", title="Functions you will need (explained above)", brief=True),
    exercise(
        stub=r"""
        def freeze(model, until="layer3"):
            '''Freeze every parameter of the parts up to and including `until`; make every later part trainable.

            model: a torchvision ResNet; until: a name from ORDER. Returns the same model.'''
            raise NotImplementedError   # delete this line, then replace every ____ below

            # 1. The position of `until` in ORDER: parts at this position or before are frozen.
            cut = ORDER.index(until)                        # e.g. "layer3" -> 3

            # 2. Every parameter tensor of the model, with its name.
            for name, param in model.named_parameters():
                # 3. YOUR CODE: the position in ORDER of this parameter's part (use stage_of(name))
                position = ____                              # 0 (stem) ... 5 (fc)
                # 4. YOUR CODE: True (trainable) if position comes after cut, else False (frozen)
                param.requires_grad = ____

            # 5. The same model, now partly frozen.
            return model

        ok_freeze = tests.check_frozen(freeze)
        """,
        solution=r"""
        def freeze(model, until="layer3"):
            '''Freeze every parameter of the parts up to and including `until`; make every later part trainable.

            model: a torchvision ResNet; until: a name from ORDER. Returns the same model.'''

            # 1. The position of `until` in ORDER: parts at this position or before are frozen.
            cut = ORDER.index(until)                        # e.g. "layer3" -> 3

            # 2. Every parameter tensor of the model, with its name.
            for name, param in model.named_parameters():
                # 3. YOUR CODE: the position in ORDER of this parameter's part (use stage_of(name))
                position = ORDER.index(stage_of(name))      # 0 (stem) ... 5 (fc)
                # 4. YOUR CODE: True (trainable) if position comes after cut, else False (frozen)
                param.requires_grad = position > cut

            # 5. The same model, now partly frozen.
            return model

        ok_freeze = tests.check_frozen(freeze)
        """),
    md(r"""
    **Did it work?** `[PASS]` means `freeze` leaves exactly the parts after `until` trainable, however
    the model was frozen before. Regime C below calls it as `freeze_fn(ft, "layer2")`.

    ### Exercise 3 · `param_groups_with_lrs(model, base_lr, decay)`

    **Words.**
    * An optimiser can treat different parameters differently. A **parameter group** is a dict with the
      key `"params"` (a list of parameter tensors) and the settings for those parameters, for example
      `"lr"`. `torch.optim.AdamW(groups, weight_decay=1e-4)` accepts a list of such dicts; a setting
      missing from a group (here `weight_decay`) comes from the keyword arguments. Extra keys such as
      `"name"` are kept and ignored, which is handy for printing.
    * **Discriminative learning rates** (also called per-layer or layer-wise learning rates): each part
      gets its own learning rate, smaller the earlier the part. Here part number i of `ORDER` (0 = stem,
      ..., 5 = fc) gets `base_lr * decay ** (5 - i)`. The head gets `base_lr`, and each part below it gets
      `decay` times the learning rate of the part above it.

    **Goal.** Build that list of parameter groups for a ResNet: one group per part that has trainable
    parameters.

    **Why by hand.** It is a dozen lines, and it is how Day 3 trains a U-Net on a pretrained encoder
    (D3-02 gives the encoder a smaller learning rate). Writing it once shows that an optimiser's parameter
    groups are ordinary Python dicts.

    **Example, by hand.** Regime C trains layer3, layer4 and fc (the parts after `until="layer2"`), with
    `base_lr` = 3e-4 and `decay` = 0.3:

    | part | i | 5 - i | lr = 3e-4 x 0.3^(5 - i) |
    |---|---|---|---|
    | layer3 | 3 | 2 | 3e-4 x 0.09 = 2.7e-5 |
    | layer4 | 4 | 1 | 3e-4 x 0.3 = 9e-5 |
    | fc | 5 | 0 | 3e-4 x 1 = 3e-4 |

    The stem, layer1 and layer2 are frozen: they get no group at all. For ResNet-18 with a 37-class head,
    frozen up to layer2, the function must return three dicts:
    ```
    [{"params": [15 tensors of layer3], "lr": 2.7e-05, "name": "layer3"},    # 2,099,712 parameters
     {"params": [15 tensors of layer4], "lr": 9.0e-05, "name": "layer4"},    # 8,393,728
     {"params": [fc.weight, fc.bias],   "lr": 3.0e-04, "name": "fc"}]        #    18,981
    ```
    (Python prints 3e-4 * 0.3 ** 2 as 2.6999999999999996e-05: decimal fractions are stored in binary, so
    the last digit is off. The checker allows for that.)

    **Plan.** The template below has these steps; the lines marked `# YOUR CODE:` are yours.
    1. A dict with one empty list per part (written).
    2. Loop over `model.named_parameters()` (written). Skip a frozen parameter: `if not
       param.requires_grad:` then `continue`, which jumps to the next parameter. Append every other one
       to the list of its part.
    3. Loop over the parts with their position i (written). Skip a part whose list is empty. Compute its
       learning rate. Build the dict and append it (written).
    4. Return the list (written).

    **What the checker tests.** A fresh ResNet-18 (every parameter trainable) with `base_lr` = 1e-3 and
    `decay` = 0.3: every parameter must be in exactly one group, with the learning rate of its part. Then
    the same model frozen up to layer2: no frozen parameter and no empty group may appear.

    **Common mistakes.**

    | you wrote | the checker prints |
    |---|---|
    | no `continue` for frozen parameters | `frozen parameters (requires_grad=False) should not be passed to the optimiser` |
    | kept the empty lists as groups | `drop empty groups` |
    | `decay ** i` | `parameters of stem have lr 1.00e-03, expected 2.43e-06`, then one line per part: the order is reversed, the stem learns fastest |
    | `decay ** (len(ORDER) - i)` | `parameters of stem have lr 7.29e-07, expected 2.43e-06` ... `parameters of fc have lr 3.00e-04, expected 1.00e-03`: one power too many; the last position is `len(ORDER) - 1` = 5 |
    | `decay * (last - i)` | `parameters of stem have lr 1.50e-03, expected 2.43e-06` ...: multiplied instead of raised to a power |
    | the key `"learning_rate"` instead of `"lr"` | `your code raised KeyError: 'lr' (...)`: the optimiser (and the checker) look for `"lr"` |
    """),
    api("Optimizer.param_groups", "torch.optim.AdamW", title="Functions you will need", brief=True),
    exercise(
        stub=r"""
        def param_groups_with_lrs(model, base_lr, decay=0.3):
            '''One optimiser parameter group per part that has trainable parameters, from stem to fc.

            Part i of ORDER (0 = stem ... 5 = fc) gets lr = base_lr * decay ** (5 - i).
            Returns a list of dicts {"params": [tensors], "lr": number, "name": part}.'''
            raise NotImplementedError   # delete this line, then replace every ____ below

            # 1. One empty list per part: {"stem": [], "layer1": [], ..., "fc": []}.
            params_of_stage = {}
            for stage in ORDER:
                params_of_stage[stage] = []

            # 2. Put every trainable parameter into the list of its part.
            for name, param in model.named_parameters():
                # YOUR CODE: if param is frozen (param.requires_grad is False), skip it with `continue`
                ____
                # YOUR CODE: append param to the list of its part, params_of_stage[stage_of(name)]
                ____

            # 3. One group per part that has trainable parameters.
            last = len(ORDER) - 1                           # 5: the position of fc
            groups = []
            for i, stage in enumerate(ORDER):               # i = 0 (stem), 1 (layer1), ..., 5 (fc)
                stage_params = params_of_stage[stage]
                # YOUR CODE: if stage_params is empty (the whole part is frozen), skip it with `continue`
                ____
                # YOUR CODE: this part's learning rate: base_lr times decay to the power (last - i)
                lr = ____
                group = {"params": stage_params, "lr": lr, "name": stage}
                groups.append(group)

            # 4. The list of dicts, ready for torch.optim.AdamW(groups, ...).
            return groups

        ok_groups = tests.check_param_groups(param_groups_with_lrs)
        """,
        solution=r"""
        def param_groups_with_lrs(model, base_lr, decay=0.3):
            '''One optimiser parameter group per part that has trainable parameters, from stem to fc.

            Part i of ORDER (0 = stem ... 5 = fc) gets lr = base_lr * decay ** (5 - i).
            Returns a list of dicts {"params": [tensors], "lr": number, "name": part}.'''

            # 1. One empty list per part: {"stem": [], "layer1": [], ..., "fc": []}.
            params_of_stage = {}
            for stage in ORDER:
                params_of_stage[stage] = []

            # 2. Put every trainable parameter into the list of its part.
            for name, param in model.named_parameters():
                # YOUR CODE: if param is frozen (param.requires_grad is False), skip it with `continue`
                if not param.requires_grad:
                    continue
                # YOUR CODE: append param to the list of its part, params_of_stage[stage_of(name)]
                params_of_stage[stage_of(name)].append(param)

            # 3. One group per part that has trainable parameters.
            last = len(ORDER) - 1                           # 5: the position of fc
            groups = []
            for i, stage in enumerate(ORDER):               # i = 0 (stem), 1 (layer1), ..., 5 (fc)
                stage_params = params_of_stage[stage]
                # YOUR CODE: if stage_params is empty (the whole part is frozen), skip it with `continue`
                if len(stage_params) == 0:
                    continue
                # YOUR CODE: this part's learning rate: base_lr times decay to the power (last - i)
                lr = base_lr * decay ** (last - i)
                group = {"params": stage_params, "lr": lr, "name": stage}
                groups.append(group)

            # 4. The list of dicts, ready for torch.optim.AdamW(groups, ...).
            return groups

        ok_groups = tests.check_param_groups(param_groups_with_lrs)
        """),
    md(r"""
    **Did it work?** `[PASS]` means each trainable part has its own learning rate and nothing frozen
    reaches the optimiser. The lab's learning-rate panel draws exactly these `lr` values, one bar per
    returned group; regime C below prints the groups your function builds for it.

    ### Regime B: frozen backbone, head trained with augmentation

    The backbone runs only forward, under `no_grad`, but on new augmented images in every epoch, so the
    features cannot be cached. The cell first picks your `freeze` and `param_groups_with_lrs` for the
    next cells (`tests.use_or_reference`: yours if it passed its checker, otherwise the course's, with a
    note). Then it builds:
    * `aug_tf`, the augmentation pipeline (D1-02): `RandomResizedCrop` cuts out a random part covering 50
      to 100 % of the image and resizes it to IMG x IMG, `RandomHorizontalFlip` mirrors half of the
      images, then the usual conversion to float and ImageNet normalisation;
    * `aug_dl`, a shuffled DataLoader over the 2 000 training images with that pipeline, 32 per batch
      (`torchkit._balanced_subset` picks `N_FT` image indices with as many images per breed; here that is
      all 2 000);
    * `head_b`, a new `nn.Linear(512, 37)` that starts from the probe's trained weights (rule 1: train the
      head first).

    It then runs the five-line training step of D0-00 for `FT_EPOCHS` epochs, on the head only. The loss
    is **cross-entropy** (D1-02: the classification loss, small when the true class gets a high
    probability) with **label smoothing** 0.1 (D1-02: the target puts 0.9 of the probability on the true
    class and spreads the remaining 0.1 evenly over all 37 classes, so the model is not pushed towards
    infinitely confident scores). Because the backbone is frozen and in evaluation mode, the validation
    features are exactly the cached ones, so evaluating the head costs nothing. The trained head is saved
    as a checkpoint; later runs load it instead of training.
    """),
    api("torchkit.save_checkpoint"),
    api("torchvision.transforms.v2.Compose", "torchvision.transforms.v2.RandomResizedCrop",
        "torchvision.transforms.v2.RandomHorizontalFlip", "torchvision.transforms.v2.ToDtype",
        "torchvision.transforms.v2.Normalize", "torch.Generator", "Tensor.tolist", "nn.Linear", "Module.state_dict", "Module.load_state_dict",
        "torchkit.load_checkpoint", brief=True,
        title="Functions from earlier notebooks, used again here: the augmented loader, the head, checkpoints"),
    api("torch.optim.AdamW", "Module.parameters", "F.cross_entropy", "Optimizer.zero_grad", "Tensor.backward",
        "Optimizer.step", "Tensor.argmax", "Tensor.float", "Tensor.mean", "Tensor.item", brief=True,
        title="Functions from earlier notebooks, used again here: the training step and the accuracy"),
    code(r"""
    freeze_fn = tests.use_or_reference(ok_freeze, globals().get("freeze"), day1.freeze, "freeze")
    pg_fn = tests.use_or_reference(ok_groups, globals().get("param_groups_with_lrs"), day1.param_groups_with_lrs, "param_groups_with_lrs")

    # The augmented training data: a random crop (50-100 % of the image) and a flip, then float + ImageNet normalisation
    aug_tf = v2.Compose([
        v2.RandomResizedCrop(IMG, scale=(0.5, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(torchkit.IMAGENET_MEAN, torchkit.IMAGENET_STD),
    ])
    sub = torchkit._balanced_subset(data["y_train"].tolist(), N_FT, seed=0)   # N_FT training images, as many per breed
    aug_dataset = day1.TensorImageDataset(data["x_train"][sub], data["y_train"][sub], aug_tf)
    aug_dl = DataLoader(aug_dataset, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(0))

    head_b = nn.Linear(feat_dim, 37).to(device)
    head_b.load_state_dict(probe_head.state_dict())         # rule 1, train the head first: start from the probe's head
    ck_b = torchkit.CHECKPOINTS / f"d1-04_frozen_head_{BACKBONE}_{IMG}.pt"
    if ck_b.exists():
        extras = torchkit.load_checkpoint(head_b, ck_b)
        print(f"loaded {ck_b.name}")
    else:
        opt = torch.optim.AdamW(head_b.parameters(), lr=3e-4, weight_decay=1e-4)   # only the head's parameters
        backbone.eval()
        t0 = time.time()
        for epoch in range(FT_EPOCHS):
            for x, y in aug_dl:
                with torch.no_grad():
                    f = backbone(x.to(device))              # (32, 512): features of the augmented images (frozen backbone)
                logits = head_b(f)                          # (32, 37)
                loss = F.cross_entropy(logits, y.to(device), label_smoothing=0.1)
                opt.zero_grad()
                loss.backward()
                opt.step()
        torchkit.save_checkpoint(head_b, ck_b)
        print(f"regime B trained in {time.time() - t0:.0f} s")

    # Validation: the frozen backbone gives exactly the cached features, so only the head runs
    with torch.no_grad():
        val_logits_b = head_b(F_["f_val"].to(device))       # (500, 37)
    pred_b = val_logits_b.argmax(1).cpu()                   # (500,): the predicted breed of every validation image
    acc_b = (pred_b == F_["y_val"]).float().mean().item()   # the fraction of correct predictions
    print(f"Regime B - frozen backbone + head with augmentation: val accuracy {acc_b:.3f}")
    """),
    md(r"""
    ### Regime C: fine-tune the top stages with discriminative learning rates

    The cell builds the fine-tuned model `ft` in five steps:
    1. a fresh ImageNet ResNet with a new 37-class head `fc` that starts from the probe's weights (rule 1);
    2. `freeze_fn(ft, "layer2")`, your `freeze`: stem, layer1 and layer2 frozen; layer3, layer4 and fc
       trained (10,512,421 of 11,195,493 parameters for ResNet-18);
    3. `day1.keep_bn_eval(ft)`: from now on `ft.train()` leaves every BatchNorm layer in evaluation mode
       (rule 4), so the running statistics stay ImageNet's; the scale and shift of the BatchNorm layers in
       layer3 and layer4 still train;
    4. the optimiser: `pg_fn(ft, base_lr=3e-4, decay=0.3)`, your `param_groups_with_lrs`, gives layer3,
       layer4 and fc the learning rates 2.7e-5, 9e-5 and 3e-4 of Exercise 3's example (the cell prints
       them), for AdamW with weight decay 1e-4;
    5. a **warm-up + cosine schedule** (D1-02): the learning rate rises from almost 0 over the first
       eighth of the steps, then falls to 0 along a cosine curve. `torchkit.fit` (D0-00) runs the training
       loop, for 4 minutes at most.

    After training, one validation pass stores the 500 x 37 validation logits in the checkpoint; Grad-CAM
    and the metrics below reuse them. Later runs load the checkpoint.

    A warning from the original experiments: with `base_lr=1e-3` and BatchNorm left in training mode, the
    fine-tuned model ends up *worse* than the linear probe on this small subset (about 61-69 % against
    79 %). A few hundred noisy steps are enough to damage good features.
    """),
    api("torchkit.count_parameters", "torchkit.fit", "np.arange", "np.array", brief=True),
    code(r"""
    ft = get_model(BACKBONE, weights=weights)                # 1. the ImageNet weights ...
    ft.fc = nn.Linear(feat_dim, 37)                          #    ... with a new 37-class head ...
    ft.fc.load_state_dict(probe_head.state_dict())           #    ... that starts from the probe's head
    freeze_fn(ft, "layer2")                                  # 2. stem, layer1, layer2 frozen
    day1.keep_bn_eval(ft)                                    # 3. BatchNorm stays in evaluation mode
    ft = ft.to(device)
    n_trainable = torchkit.count_parameters(ft, trainable_only=True)
    n_total = torchkit.count_parameters(ft)
    print(f"trainable: {n_trainable:,} of {n_total:,} parameters")

    groups = pg_fn(ft, base_lr=3e-4, decay=0.3)              # 4. one parameter group per trained part
    for g in groups:
        n_params = 0
        for p in g["params"]:
            n_params += p.numel()
        print(f"  {g['name']:<7} lr {g['lr']:.1e}  ({n_params:,} params)")

    val_dl = DataLoader(day1.TensorImageDataset(data["x_val"], data["y_val"], eval_tf), batch_size=50)


    def smoothed_cross_entropy(logits, targets):            # the loss: cross-entropy with label smoothing 0.1
        return F.cross_entropy(logits, targets, label_smoothing=0.1)


    ck_c = torchkit.CHECKPOINTS / f"d1-04_finetune_{BACKBONE}_{IMG}.pt"
    if ck_c.exists():
        extras = torchkit.load_checkpoint(ft, ck_c)
        logits_val = extras["logits_val"]                    # (500, 37), saved together with the weights
        print(f"loaded {ck_c.name}")
    else:
        opt = torch.optim.AdamW(groups, weight_decay=1e-4)
        total = FT_EPOCHS * len(aug_dl)                      # the number of training steps
        sched = day1.make_warmup_cosine_scheduler(opt, total, max(1, total // 8))   # 5. warm-up, then cosine
        torchkit.fit(ft, aug_dl, None, smoothed_cross_entropy, opt, FT_EPOCHS,
                     scheduler=sched, max_minutes=4, log_every=4, device=device)
        ft.eval()
        logit_batches = []
        with torch.no_grad():                                # one validation pass, reused below for Grad-CAM and metrics
            for x, _ in val_dl:
                logit_batches.append(ft(x.to(device)).cpu())
        logits_val = torch.cat(logit_batches)                # (500, 37)
        torchkit.save_checkpoint(ft, ck_c, logits_val=logits_val)
    pred_c = logits_val.argmax(1)                            # (500,)
    acc_c = (pred_c == F_["y_val"]).float().mean().item()
    print(f"Regime C - fine-tuned layer3 + layer4 + fc: val accuracy {acc_c:.3f}")
    """),
    md(r"""
    The bar chart puts the regimes side by side: blue, this run; gray, what the full recipe reaches on a
    GPU (ResNet-50, 224 px, 3 epochs). The first pair is D1-02's PetNet trained from scratch, when its
    checkpoint is in `data/checkpoints/`.
    """),
    code(r"""
    # Bar chart: validation accuracy of each regime in this run, next to the expected GPU numbers
    names = ["A linear probe", "B frozen + head\n(augmented)", "C fine-tune\nlayer3-4 + fc"]
    accs = [acc_a, acc_b, acc_c]
    gpu_expected = [0.88, 0.90, 0.93]                       # ResNet-50, 224 px, GPU, full recipe
    if GPU:
        scratch_px = 128
    else:
        scratch_px = 64
    ck_scratch = torchkit.CHECKPOINTS / f"d1-02_recipe_{scratch_px}px.pt"    # D1-02's PetNet, trained from scratch
    if ck_scratch.exists():
        scratch_history = torchkit.load_checkpoint(day1.PetNet(dropout=0.2), ck_scratch)["history"]
        names.insert(0, "from scratch\n(D1-02 PetNet)")
        accs.insert(0, max(scratch_history["val_metric"]))  # its best validation accuracy
        gpu_expected.insert(0, 0.33)
    fig, ax = plt.subplots(figsize=(9, 3.8)); xs = np.arange(len(names))
    ax.bar(xs - 0.2, np.array(accs) * 100, 0.4, color=viz.PALETTE[0], label=f"this run ({BACKBONE}, {IMG} px, CPU budget)")
    ax.bar(xs + 0.2, np.array(gpu_expected) * 100, 0.4, color="lightgray", label="expected: resnet50, 224 px, GPU, full recipe")
    for x_, a in zip(xs, accs):
        ax.text(x_ - 0.2, a * 100 + 1, f"{a * 100:.1f}", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9); ax.set_ylabel("val accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_title("Same 2 000 / 500 Pet images, four ways"); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: from scratch 13.2 %, then about 79 % (A), 80.2 % (B) and 81.6 % (C). On the CPU budget
    (ResNet-18 at 160 px, one augmented epoch) B and C add 1 to 3 points to the probe. With a GPU, 3
    epochs and ResNet-50 at 224 px the expected ladder is about **88 % (A) -> 90 % (B) -> 93 % (C)**. The
    ranking is the lesson: each step buys a little accuracy for a lot more compute, and on small datasets
    A or B is often the right answer.

    ### Exercise 4 · choose and justify an unfreezing schedule (written)

    **Goal.** For four situations, pick a regime (A, B, C or C') and the parts to train, and give the
    reason in one or two sentences each: (1) 300 photos of 5 kinds of pastry; (2) 40 000 chest X-rays, 14
    findings; (3) 2 000 satellite tiles, 10 land-use classes; (4) you must try 50 different heads / losses
    by tomorrow.

    **How to decide.** Three questions settle most cases:
    1. **How many labelled images per class?** A few dozen: A or B. Training millions of parameters on
       them would **overfit** (D1-02: learn the training images by heart instead of what carries over to
       new images). Thousands: C or C'.
    2. **How close are the images to ImageNet photos?** Ordinary photos of objects and animals: the
       pretrained features already fit, train little. Far from them (grey-scale X-rays, satellite images
       seen from above, microscope images): the late stages, perhaps all, must adapt.
    3. **What are the compute and time budgets?** Many experiments, or only a laptop: A, because the
       backbone runs once.

    **What a good answer contains:** the regime, which parts are trained (and that the head is trained
    first), and a reason that names the amount of data, the distance from ImageNet, or the budget. A model
    answer for a fifth situation, not in the list:

    > (0) 500 phone photos of 10 bird species: A first (seconds, a strong baseline), then B. Only 50
    > images per class and ordinary photos, so the frozen ImageNet features fit; training layer4's
    > 8 million parameters on 500 images would overfit.

    There is no checker: compare with your neighbour, then with the solution notebook.
    """),
    exercise(
        stub=r"""
        my_answer = '''
        (1) pastry, 300 images:
        (2) chest X-rays, 40 000 images:
        (3) satellite tiles, 2 000 images:
        (4) 50 experiments by tomorrow:
        '''
        # YOUR CODE: after each colon above, write the regime, the parts you train, and the reason
        # (no checker: discuss your answers with your neighbour)
        print(my_answer)
        """,
        solution=r"""
        my_answer = '''
        (1) pastry, 300 images: A, then B. Tiny data, ImageNet-like photos: the frozen features are already good;
            fine-tuning 11-25 M parameters on 300 images would overfit. B adds augmentation if A is not enough.
        (2) chest X-rays, 40 000 images: C' (full fine-tune, discriminative LRs, head first). Lots of data and a domain
            far from ImageNet (grey-scale, textures, no objects): the late features must change, even early ones a bit.
        (3) satellite tiles, 2 000 images: C on layer3-4 + fc. Medium data, a different viewpoint (top-down, scale):
            keep generic early filters, adapt the later stages; vertical flips / 90-degree rotations ARE valid here.
        (4) 50 experiments: A. Extract features once, then each experiment costs seconds instead of an hour.
        '''
        # YOUR CODE: after each colon above, write the regime, the parts you train, and the reason
        # (no checker: discuss your answers with your neighbour)
        print(my_answer)
        """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · How much data do you need?

    **Why.** Labelling images costs time and money. Before paying for more labels, measure how accuracy
    grows with data. The cached features make this cheap: the cell re-trains the probe on 2, 5, 10, 20,
    30 and 54 images per breed (74 to 2 000 images, always as many per breed; `torchkit._balanced_subset`
    picks them) and plots validation accuracy against training-set size.

    The horizontal axis is on a **log scale**: equal distances mean equal *factors* (74 -> 148 -> 296 are
    equal steps), so every doubling of the data is one step to the right. The empirical rule (Sun et al.
    2017, Hestness et al. 2017): accuracy grows roughly *linearly in log(data)*, so every doubling buys
    about the same number of points. The dotted line is the straight line that fits the points best
    (`np.polyfit` of degree 1, on log2 of the sizes): its slope is the number of points per doubling.
    """),
    api("np.polyfit", "np.log2"),
    code(r"""
    sizes = [74, 185, 370, 740, 1110, 2000]          # 2, 5, 10, 20, 30 and 54 images per breed
    curve = []                                       # validation accuracy for each size
    for n in sizes:
        idx = torchkit._balanced_subset(F_["y_train"].tolist(), n, seed=0)    # n training images, as many per breed
        _, a, _ = day1.train_linear_probe(F_["f_train"][idx], F_["y_train"][idx], F_["f_val"], F_["y_val"], epochs=100)
        curve.append(a)
        print(f"{n:5d} train images ({n / 37:4.1f} / class): val acc {a:.3f}")

    # Accuracy against size on a log axis, with the best straight line in log2(size)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(sizes, np.array(curve) * 100, "-o", color=viz.PALETTE[0], label=f"linear probe on {BACKBONE} features")
    k, b = np.polyfit(np.log2(sizes), np.array(curve) * 100, 1)   # slope k = points per doubling
    xx = np.array([sizes[0], 3680])
    ax.semilogx(xx, k * np.log2(xx) + b, ":", color="gray", label=f"fit: +{k:.1f} points per doubling (extrapolated to 3 680)")
    if ck_scratch.exists():
        ax.plot(2000, accs[0] * 100, "*", ms=14, color=viz.PALETTE[3], label="from scratch, 2 000 images (D1-02)")
    ax.set_xticks(sizes); ax.set_xticklabels(sizes); ax.set_xlabel("training images (log scale)"); ax.set_ylabel("val accuracy (%)")
    ax.set_title("Accuracy vs. training-set size"); ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: 2 images per breed already give 49 % (guessing would give 1 in 37 = 2.7 %), 20 per
    breed give 77 %. The fit says about +6.4 points per doubling, but the last points flatten: 30 per
    breed give 79.6 %, all 54 per breed 78.6 % (one point is 5 images out of 500, within the noise of
    this small validation set). Doubling to 3 680 images would buy a few points at most. (These are the
    numbers of the solution notebook. Every probe starts from random weights, so your run can differ by
    a point or two per size; the shape of the curve stays the same.) Plot this curve
    on your own data before you label more: it tells you whether more images or a better model is the
    cheaper way up.

    ## 5 · What the network sees

    **Why.** Section 1 claimed that the backbone's features are generic early and specific late, and the
    linear probe showed that its features already separate the breeds. This section looks at both, and
    introduces **hooks**, the tool Grad-CAM needs in section 6.

    ### First-layer filters

    The first layer of the ResNet, `conv1`, has 64 **filters** (kernels, D1-01), each of shape
    3 x 7 x 7: one 7 x 7 grid of weights for each colour channel. A filter responds most strongly to an
    image patch that looks like itself, so each one can be drawn as a tiny 7 x 7 colour picture. The cell
    stretches each filter's values to [0, 1] for display (its smallest weight black, its largest white).
    """),
    api("Tensor.detach", "Tensor.min", "Tensor.max", "Tensor.permute", "Tensor.numpy", brief=True),
    code(r"""
    w1 = get_model(BACKBONE, weights=weights).conv1.weight.detach().cpu()        # (64, 3, 7, 7)

    # The 64 filters as 7 x 7 colour pictures, 4 rows of 16
    fig, axes = plt.subplots(4, 16, figsize=(14, 3.8))
    for k, ax in enumerate(axes.flat):
        f = w1[k]                                                               # (3, 7, 7): filter k
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)                          # stretch to [0, 1] for display
        ax.imshow(f.permute(1, 2, 0).numpy(), interpolation="nearest")         # CHW -> HWC for Matplotlib
        ax.axis("off")
    plt.suptitle(f"The 64 first-layer filters of {BACKBONE} (ImageNet weights)"); plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: compare with the hand-made gallery of D1-01. There are edges at many angles (like the
    Sobel filters), colour contrasts (red against green, blue against yellow), blobs and stripes. Nobody
    designed them: ImageNet training produced them, and every convolutional network trained on ordinary
    photos learns similar ones. That is why the early layers transfer to almost anything.

    ### Feature maps through the stages, read with a hook

    A **feature map** (D1-01) is the output of a convolutional layer for one image: one 2-D grid per
    channel, each value an **activation**, how strongly that channel's pattern is present at that place.
    We want the outputs of layer1 to layer4, which the model computes and then throws away.

    A **hook** is a function you attach to a layer; PyTorch calls it every time the layer runs.
    `layer.register_forward_hook(fn)` makes PyTorch call `fn(module, inputs, output)` right after the
    layer's forward pass, so `fn` can store `output`. It returns a **handle**: `handle.remove()` takes the
    hook off again. Always remove hooks when you are done, otherwise they keep running on every forward
    pass and keep their tensors in memory.

    The cell attaches the same hook function to the four stages, runs one image through the model, and
    shows the 6 most active channels of each stage (the largest mean activation). `x_img[None]` adds a
    batch axis of size 1, like `unsqueeze(0)`.
    """),
    api("Module.register_forward_hook", "Tensor.argsort"),
    code(r"""
    probe_model = get_model(BACKBONE, weights=weights).eval().to(device)
    STAGES = ["layer1", "layer2", "layer3", "layer4"]
    outputs = {}                                      # filled by the hook: {layer: its output}


    def save_output(module, inputs, output):          # PyTorch calls this right after `module` has run
        outputs[module] = output.detach().cpu()       # keep a copy without autograd history, on the CPU


    handles = []
    for stage in STAGES:
        layer = getattr(probe_model, stage)           # probe_model.layer1, probe_model.layer2, ...
        handles.append(layer.register_forward_hook(save_output))

    img_i = 130                                       # one validation photo
    x_img = eval_tf(data["x_val"][img_i])             # (3, IMG, IMG)
    with torch.no_grad():
        probe_model(x_img[None].to(device))           # one forward pass of a batch of 1; the hooks fill `outputs`
    for handle in handles:
        handle.remove()                               # take the hooks off again

    acts = {}                                         # {stage name: its feature maps for this image}
    for stage in STAGES:
        acts[stage] = outputs[getattr(probe_model, stage)][0]   # (C, h, w): the batch axis dropped

    # One row per stage: the image, then the 6 channels with the largest mean activation
    fig, axes = plt.subplots(4, 7, figsize=(14, 8.4))
    for r, stage in enumerate(STAGES):
        a = acts[stage]                               # (C, h, w)
        axes[r, 0].imshow(torchkit.denormalize(x_img)); axes[r, 0].set_title(f"{stage}: {tuple(a.shape)}", fontsize=9)
        channel_means = a.mean((1, 2))                # (C,): the mean activation of each channel
        top = channel_means.argsort(descending=True)[:6]   # the 6 channels with the largest mean
        for c, ch in enumerate(top):
            axes[r, c + 1].imshow(a[ch], cmap="magma"); axes[r, c + 1].set_title(f"ch {int(ch)}", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: the resolution halves at every stage while the number of channels doubles, from
    (64, 40, 40) at layer1 to (512, 5, 5) at layer4 for a 160 px image. layer1's maps follow edges and fur
    texture at almost every pixel; layer4's are 5 x 5 blobs that light up around the cat's face. The maps
    go from "an edge at this pixel" to "a cat-face-like pattern somewhere around here".

    ### The feature space

    Each validation image's features are a point in a space with 512 dimensions, which we cannot draw.
    **PCA** (principal component analysis) finds the directions in that space along which the points
    spread out most; the first two are the **principal components** PC 1 and PC 2. Projecting every
    feature vector onto them gives two numbers per image: a point we can plot. The projection is a matrix
    product, `@` in PyTorch: features (500, 512) times directions (512, 2) gives (500, 2).
    `torch.pca_lowrank` computes the directions. We compute them on the training features after
    subtracting their mean, because PCA describes the spread around the mean.
    """),
    api("torch.pca_lowrank"),
    api("np.zeros", brief=True),
    code(r"""
    CAT_BREEDS = {"Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau", "Maine Coon",
                  "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx"}    # the 12 cat breeds; the other 25 are dogs
    train_mean = F_["f_train"].mean(0)                        # (512,): the mean feature vector of the training images
    _, _, V = torch.pca_lowrank(F_["f_train"] - train_mean, q=2, center=False)   # V: (512, 2), the two main directions
    fv = F_["f_val"] - train_mean                             # (500, 512): validation features, centred the same way
    Z = (fv @ V).numpy()                                      # (500, 2): every image's position along PC 1 and PC 2
    yv = F_["y_val"].numpy()                                  # (500,): the breed of every image
    is_cat = np.zeros(len(yv), dtype=bool)                    # (500,): True for the cat photos
    for k in range(len(yv)):
        is_cat[k] = classes[yv[k]] in CAT_BREEDS

    # Left: cats against dogs; right: eight breeds in colour, the rest in gray
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(Z[~is_cat, 0], Z[~is_cat, 1], s=10, alpha=.6, label="dogs (25 breeds)", color=viz.PALETTE[1])
    axes[0].scatter(Z[is_cat, 0], Z[is_cat, 1], s=10, alpha=.6, label="cats (12 breeds)", color=viz.PALETTE[0])
    axes[0].set_title("PCA of penultimate features: species"); axes[0].legend()
    show_breeds = ["Abyssinian", "Bombay", "Siamese", "Sphynx", "Pug", "Samoyed", "Saint Bernard", "Yorkshire Terrier"]
    axes[1].scatter(Z[:, 0], Z[:, 1], s=6, color="lightgray")
    for k, bname in enumerate(show_breeds):
        if bname in classes:
            m = yv == classes.index(bname)                    # True for the images of this breed
            axes[1].scatter(Z[m, 0], Z[m, 1], s=18, color=viz.color(k), label=bname)
    axes[1].set_title("8 breeds highlighted (gray = the other 29)"); axes[1].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.grid(alpha=.3)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: the backbone never saw a Pet label, yet cats (blue) and dogs (orange) fall on
    different sides along PC 1, and single breeds form clusters (the Samoyeds at the top, the Sphynx cats
    at the bottom right). If two of the 512 dimensions already separate this much, a linear layer that
    uses all 512 can separate much more: that is why the linear probe works. ("Penultimate" means "the
    last but one": these are the features just before the head.)
    """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · Grad-CAM: where does the network look?

    **Why.** A model can be right for the wrong reason (end of this section). **Grad-CAM** (Selvaraju et
    al., 2017; "gradient-weighted class activation map") draws a **heat map** over the image, warm where
    the evidence for one class was found. It is the most used way to check what a classifier looks at,
    and the hooks you write for it work on any PyTorch model.

    **The idea.** layer4 outputs K feature maps (512 for ResNet-18), each 5 x 5 for a 160 px image (7 x 7
    at 224 px: up to layer4 the image shrinks 32 times). Each map detects one pattern. For the class we
    want to explain, measure how much each map raised that class's score, and add up the maps, each
    multiplied by that amount. Places where helpful maps are active get high values.

    **Words and symbols.**
    * $A^k$: feature map number k of layer4, a grid of $h \times w$ activations $A^k_{ij}$ (row i,
      column j).
    * $y^c$: the score (logit) of class c, one number.
    * The **gradient with respect to a feature map**, $\partial y^c / \partial A^k_{ij}$, one number per
      position of every map: how much the class score would rise if that one activation rose a little
      (D0-00's derivative). Autograd computes it in `backward()`, as it does for parameters. It has the
      shape of the feature maps.
    * $\alpha_k^c$: the weight of map k, its gradient averaged over the $h \times w$ positions. Positive:
      more of this pattern raises the class score; negative: it lowers it.
    * **ReLU** (D1-01) sets negative numbers to 0 and keeps positive ones.

    **The formula.**

    $$\alpha_k^c = \frac{1}{hw}\sum_{i,j} \frac{\partial y^c}{\partial A^k_{ij}}, \qquad
    L^c_{\text{Grad-CAM}} = \operatorname{ReLU}\Big(\sum_k \alpha_k^c A^k\Big)$$

    then **upsample** $L^c$ (h x w) to the image size, that is, enlarge it by interpolating between its
    values, and rescale it to [0, 1]. The ReLU drops the places with negative totals: they are evidence
    *against* the class.
    """),
    md(r"""
    ### Lab · Grad-CAM by hand
    **Predict first:** what goes wrong if you leave out the ReLU: does the heat map just get a little
    noisier, or can it light up regions that have nothing to do with the class? The lab builds the
    weighted sum from eight hand-made feature maps of a photo, so every $\alpha_k$, $A^k$, the sum,
    the ReLU and the upsampling are visible (4 minutes).
    """),
    code(r"""
    labs.show("gradcam-by-hand")
    """),
    md(r"""
    ### Grad-CAM by hand: a 2 x 2 example

    K = 2 feature maps of 2 x 2 activations, and their gradients:

    | | map 1 | map 2 |
    |---|---|---|
    | activations $A^k$ | [[2, 0], [0, 1]] | [[0, 1], [3, 0]] |
    | gradients $\partial y^c / \partial A^k$ | [[0.4, 0.2], [0.2, 0.0]] | [[-0.2, 0.0], [-0.2, 0.0]] |
    | weight $\alpha_k$ = mean of the 4 gradients | 0.8 / 4 = 0.2 | -0.4 / 4 = -0.1 |
    | $\alpha_k A^k$ | [[0.4, 0], [0, 0.2]] | [[0, -0.1], [-0.3, 0]] |

    1. Sum of the two weighted maps: [[0.4, -0.1], [-0.3, 0.2]].
    2. ReLU: [[0.4, 0], [0, 0.2]].
    3. Rescale to [0, 1]: subtract the minimum (0), divide by the range (0.4 - 0 = 0.4): [[1, 0], [0, 0.5]].

    Map 2 counts *against* the class (negative weight), so where it is active (bottom left) the heat map
    is cold. The next cell does the same in PyTorch, with the shapes the exercise uses: a batch axis of
    size 1 in front, (1, K, h, w). `keepdim=True` keeps the averaged or summed axes with size 1, so that
    the results still line up with the maps for broadcasting (D0-00): alpha (1, 2, 1, 1) times A
    (1, 2, 2, 2) multiplies every map by its own weight.
    """),
    api("F.interpolate"),
    api("Tensor.mean", "Tensor.sum", "F.relu", "Tensor.min", "Tensor.max", brief=True),
    code(r"""
    # The worked example in PyTorch: (batch 1, K = 2 maps, h = 2, w = 2)
    A = torch.tensor([[[[2.0, 0.0], [0.0, 1.0]],                # map 1
                       [[0.0, 1.0], [3.0, 0.0]]]])              # map 2            -> (1, 2, 2, 2)
    G = torch.tensor([[[[0.4, 0.2], [0.2, 0.0]],                # gradient of map 1
                       [[-0.2, 0.0], [-0.2, 0.0]]]])            # gradient of map 2 -> (1, 2, 2, 2)

    alpha = G.mean(dim=(2, 3), keepdim=True)                    # (1, 2, 1, 1): one weight per map
    weighted = alpha * A                                        # (1, 2, 2, 2): every map times its weight
    summed = weighted.sum(dim=1, keepdim=True)                  # (1, 1, 2, 2): the maps added up
    cam = F.relu(summed)                                        # (1, 1, 2, 2): negative evidence dropped
    scaled = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) # (1, 1, 2, 2): smallest -> 0, largest -> 1
    print("alpha:", alpha.flatten())
    print("sum:", summed[0, 0])
    print("ReLU:", cam[0, 0])
    print("rescaled:", scaled[0, 0])

    upsampled = F.interpolate(cam, size=(4, 4), mode="bilinear", align_corners=False)   # (1, 1, 4, 4)
    print("ReLU map upsampled to 4 x 4:", upsampled[0, 0])
    """),
    md(r"""
    The upsampled map keeps 0.4 in the top-left corner and 0.2 in the bottom-right one, and blends
    smoothly in between: **bilinear** interpolation (`mode="bilinear"`) mixes the four nearest values,
    weighted by distance. That is why real Grad-CAM maps look like smooth blobs.

    ### Hooks for the two tensors

    Inside a real model, A is layer4's output, and its gradient only exists for a moment during
    `backward()`. Two hooks catch them:
    * a **forward hook** on the layer (section 5) stores its output A;
    * a **tensor hook**: `tensor.register_hook(fn)` asks autograd to call `fn(grad)` when it computes the
      gradient with respect to that tensor during `backward()`. Registered on A inside the forward hook,
      it hands us $\partial y^c / \partial A$.

    A small check by hand: b = 2a for a = [1, 2, 3], and score = 10 b0 + 0 b1 - 1 b2 = 20 + 0 - 6 = 14.
    The gradient of the score with respect to b is [10, 0, -1], the three factors. The tensor hook on b
    prints exactly that during `backward()`.
    """),
    api("Tensor.register_hook"),
    api("Tensor.backward", "Tensor.item", brief=True),
    code(r"""
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = 2 * a                                         # an intermediate tensor, like a feature map inside a model


    def show_grad(grad):                              # autograd calls this during backward()
        print("gradient of the score with respect to b:", grad)


    b.register_hook(show_grad)
    score = (b * torch.tensor([10.0, 0.0, -1.0])).sum()   # 10 * b0 + 0 * b1 - 1 * b2
    print("score:", score.item())
    score.backward()                                  # the hook runs here
    """),
    md(r"""
    ### Exercise 5 · `gradcam(model, target_layer, x, class_idx)`

    **Goal.** Compute the Grad-CAM heat map of one image for one class.

    **Why by hand.** It puts hooks, autograd and broadcasting together on a real model, and gives you a
    tool to inspect any classifier you train.

    **Input.**
    * `model`: a classifier; for a batch of one image it outputs logits (1, number of classes);
    * `target_layer`: the layer whose feature maps are weighted, for example `ft.layer4`;
    * `x`: one image, a tensor (3, H, W), normalised as the model expects;
    * `class_idx`: the class to explain; `None` means the predicted class (the largest logit).

    **Output.** A NumPy array (H, W), the size of the image, with values in [0, 1].

    **Shapes on the way** (ResNet-18 at 160 px): batch (1, 3, 160, 160) -> logits (1, 37); A and its
    gradient G (1, 512, 5, 5); alpha (1, 512, 1, 1); weighted maps (1, 512, 5, 5); sum and ReLU
    (1, 1, 5, 5) (the lab's "ReLU of the sum" panel); upsampled (1, 1, 160, 160); the map (160, 160).

    **Plan.** The numbers are those of the comments in the template; the lines marked `# YOUR CODE:` are
    yours.
    1. The two hooks (written): `fwd_hook` stores A in `store["act"]` and registers `save_grad` on it,
       which stores the gradient in `store["grad"]`. A `try` / `finally` block removes the forward hook
       even if something fails in between.
    2. Forward pass: `logits = model(batch)`, where `batch = x[None]` is the image as a batch of one.
    3. The class: if `class_idx` is `None`, the index of the largest logit, `int(logits.argmax(1))`.
    4. Backward from that one score: `score.backward()` makes autograd fill `store["grad"]`.
    5. alpha: G averaged over axes 2 and 3 (h and w), with `keepdim=True` so it stays (1, K, 1, 1).
    6. Multiply every map by its weight; sum over axis 1 (the maps) with `keepdim=True`; ReLU.
    7. Upsample to the image size and take the map out of the batch (written).
    8. Rescale to [0, 1]: (cam - min) / (max - min + 1e-8). The 1e-8 avoids a division by 0 for a flat map.

    Steps 5, 6 and 8 are the worked example above.

    **What the checker tests.** A small random convolutional network, a random 32 x 32 image, and as
    `target_layer` its second convolution (16 maps of 16 x 16), with `class_idx` = None, 2 and 4. It
    compares your map with the course's reference map (largest allowed difference 0.0001), and checks that
    no hook is left on the model.

    **Common mistakes** (you wrote: the checker prints):
    * **no ReLU**: `class_idx=None: max |diff| = 0.354 vs the reference map`, and the same for class 2 and 4;
    * **no rescaling to [0, 1]**: `class_idx=None: max |diff| = 0.999 vs the reference map` ...;
    * **`logits.sum().backward()`** instead of the one class score: `class_idx=None: max |diff| = 0.949 vs
      the reference map` ...: that is the gradient of all classes together;
    * **`G.mean(dim=1, keepdim=True)`**, averaged over the maps instead of the positions:
      `class_idx=None: max |diff| = 0.947 vs the reference map` ...;
    * **`A.mean(...)` instead of `G.mean(...)`** for alpha: `class_idx=None: max |diff| = 0.855 vs the
      reference map` ...: the weights must come from the gradients, which carry the class;
    * **`G.mean(dim=(2, 3))` without `keepdim=True`**: `class_idx=None: max |diff| = 0.946 vs the
      reference map` ...: alpha is (1, K) and lines up with the width, not with the maps. In the checker
      K = w = 16, so there is no error; on ResNet-18 the same line raises `The size of tensor a (512) must
      match the size of tensor b (5) at non-singleton dimension 3`;
    * **`.sum(dim=1)` without `keepdim=True`**: `your code raised ValueError: Input and output must have
      the same number of spatial dimensions, but got input with spatial dimensions of [16] and output size
      of torch.Size([32, 32]). ...`: `F.interpolate` needs (1, 1, h, w);
    * **deleted the `try` / `finally`**: `3 hook(s) left registered on the model: call handle.remove() in
      a finally block`.
    """),
    api("Module.register_forward_hook", "Tensor.register_hook", "F.interpolate", "Tensor.argmax", "Tensor.mean",
        "Tensor.sum", "F.relu", title="Functions you will need (explained above)", brief=True),
    exercise(
        stub=r"""
        def gradcam(model, target_layer, x, class_idx=None):
            '''Grad-CAM heat map of one image x (3, H, W) for class class_idx (None: the predicted class).

            target_layer: the layer whose feature maps are weighted, e.g. model.layer4.
            Returns a NumPy array (H, W) with values in [0, 1].'''
            raise NotImplementedError   # delete this line, then replace every ____ below

            # 1. Two hooks fill `store`: the forward hook keeps A, the tensor hook keeps dScore / dA.
            store = {}

            def save_grad(grad):                        # autograd calls this during backward()
                store["grad"] = grad                    # (1, K, h, w)

            def fwd_hook(module, inp, out):             # PyTorch calls this right after target_layer has run
                store["act"] = out                      # A: (1, K, h, w)
                out.register_hook(save_grad)            # ask for the gradient with respect to this tensor

            handle = target_layer.register_forward_hook(fwd_hook)
            model.eval()
            model_device = next(model.parameters()).device
            try:
                # 2. Forward pass on a batch of one image.
                batch = x[None].to(model_device)        # (1, 3, H, W)
                # YOUR CODE: the model's output for batch
                logits = ____                            # (1, number of classes)

                # 3. The class to explain.
                if class_idx is None:
                    # YOUR CODE: the index of the largest logit, as a Python int
                    c = ____
                else:
                    c = int(class_idx)

                # 4. Backward from this one class score: the tensor hook fills store["grad"].
                model.zero_grad()                       # clear old parameter gradients (D0-00: they add up)
                score = logits[0, c]                    # one number: y^c
                # YOUR CODE: run the backward pass from score
                ____
            finally:
                handle.remove()                         # always take the hook off, even after an error

            A = store["act"].detach()                   # (1, K, h, w): the feature maps, without autograd history
            G = store["grad"].detach()                  # (1, K, h, w): their gradients

            # 5. One weight per map: its gradient averaged over all positions (axes 2 and 3).
            # YOUR CODE: G averaged over the axes 2 and 3, with keepdim=True
            alpha = ____                                 # (1, K, 1, 1)

            # 6. Weighted sum of the maps, then ReLU.
            # YOUR CODE: every map multiplied by its weight (broadcasting does it in one multiplication)
            weighted = ____                              # (1, K, h, w)
            # YOUR CODE: add the maps up: the sum over axis 1, with keepdim=True
            cam = ____                                   # (1, 1, h, w)
            # YOUR CODE: keep only the positive evidence
            cam = ____                                   # (1, 1, h, w)

            # 7. Upsample to the image size, then take the single map out of the batch.
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)   # (1, 1, H, W)
            cam = cam[0, 0]                             # (H, W)

            # 8. Rescale to [0, 1]: the smallest value becomes 0, the largest 1.
            # YOUR CODE: (cam minus its minimum) divided by (its maximum minus its minimum + 1e-8)
            cam = ____                                   # (H, W)
            return cam.cpu().numpy()

        ok_cam = tests.check_gradcam(gradcam)
        """,
        solution=r"""
        def gradcam(model, target_layer, x, class_idx=None):
            '''Grad-CAM heat map of one image x (3, H, W) for class class_idx (None: the predicted class).

            target_layer: the layer whose feature maps are weighted, e.g. model.layer4.
            Returns a NumPy array (H, W) with values in [0, 1].'''

            # 1. Two hooks fill `store`: the forward hook keeps A, the tensor hook keeps dScore / dA.
            store = {}

            def save_grad(grad):                        # autograd calls this during backward()
                store["grad"] = grad                    # (1, K, h, w)

            def fwd_hook(module, inp, out):             # PyTorch calls this right after target_layer has run
                store["act"] = out                      # A: (1, K, h, w)
                out.register_hook(save_grad)            # ask for the gradient with respect to this tensor

            handle = target_layer.register_forward_hook(fwd_hook)
            model.eval()
            model_device = next(model.parameters()).device
            try:
                # 2. Forward pass on a batch of one image.
                batch = x[None].to(model_device)        # (1, 3, H, W)
                # YOUR CODE: the model's output for batch
                logits = model(batch)                   # (1, number of classes)

                # 3. The class to explain.
                if class_idx is None:
                    # YOUR CODE: the index of the largest logit, as a Python int
                    c = int(logits.argmax(1))
                else:
                    c = int(class_idx)

                # 4. Backward from this one class score: the tensor hook fills store["grad"].
                model.zero_grad()                       # clear old parameter gradients (D0-00: they add up)
                score = logits[0, c]                    # one number: y^c
                # YOUR CODE: run the backward pass from score
                score.backward()
            finally:
                handle.remove()                         # always take the hook off, even after an error

            A = store["act"].detach()                   # (1, K, h, w): the feature maps, without autograd history
            G = store["grad"].detach()                  # (1, K, h, w): their gradients

            # 5. One weight per map: its gradient averaged over all positions (axes 2 and 3).
            # YOUR CODE: G averaged over the axes 2 and 3, with keepdim=True
            alpha = G.mean(dim=(2, 3), keepdim=True)    # (1, K, 1, 1)

            # 6. Weighted sum of the maps, then ReLU.
            # YOUR CODE: every map multiplied by its weight (broadcasting does it in one multiplication)
            weighted = alpha * A                        # (1, K, h, w)
            # YOUR CODE: add the maps up: the sum over axis 1, with keepdim=True
            cam = weighted.sum(dim=1, keepdim=True)     # (1, 1, h, w)
            # YOUR CODE: keep only the positive evidence
            cam = F.relu(cam)                           # (1, 1, h, w)

            # 7. Upsample to the image size, then take the single map out of the batch.
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)   # (1, 1, H, W)
            cam = cam[0, 0]                             # (H, W)

            # 8. Rescale to [0, 1]: the smallest value becomes 0, the largest 1.
            # YOUR CODE: (cam minus its minimum) divided by (its maximum minus its minimum + 1e-8)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)   # (H, W)
            return cam.cpu().numpy()

        ok_cam = tests.check_gradcam(gradcam)
        """),
    md(r"""
    **Did it work?** `[PASS]` means your map equals the reference map on every pixel. The next cell runs
    it (or the reference) on the fine-tuned model of regime C, on four validation photos the model gets
    right (top row: OK) and four it gets wrong (WRONG), each with its heat map underneath. The loop over
    `clf.parameters()` makes every parameter require gradients again: the tensor hook needs layer4's
    output to be part of autograd's record, and that output is only recorded when something it depends
    on requires a gradient. `viz.heatmap_overlay` draws the map in colour over the photo.
    """),
    api("torch.randperm", brief=True),
    code(r"""
    cam_fn = tests.use_or_reference(ok_cam, globals().get("gradcam"), day1.gradcam, "gradcam")
    clf = ft.eval()                                   # the fine-tuned model of regime C
    for p in clf.parameters():
        p.requires_grad_(True)                        # Grad-CAM needs gradients to flow back to layer4
    pred_val = logits_val.argmax(1)                   # (500,): the predicted breed of every validation image
    yv_t = F_["y_val"]                                # (500,): the true breed

    # The validation set is sorted by breed: shuffle the order once, then take the first 4 right and 4 wrong
    shuffled = torch.randperm(len(yv_t), generator=torch.Generator().manual_seed(3)).tolist()
    right = []
    wrong = []
    for i in shuffled:
        if pred_val[i] == yv_t[i] and len(right) < 4:
            right.append(i)
        if pred_val[i] != yv_t[i] and len(wrong) < 4:
            wrong.append(i)

    # Top row: the photo with its true and predicted breed; bottom row: the Grad-CAM heat map on top of it
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.8))
    for k, i in enumerate(right + wrong):
        x = eval_tf(data["x_val"][i])                 # (3, IMG, IMG)
        img = torchkit.denormalize(x)
        cam = cam_fn(clf, clf.layer4, x)              # (IMG, IMG) in [0, 1]
        axes[0, k].imshow(img); axes[0, k].axis("off")
        viz.heatmap_overlay(axes[1, k], img, cam, alpha=0.5)
        if i in right:
            verdict, colour = "OK", "black"
        else:
            verdict, colour = "WRONG", viz.PALETTE[3]
        axes[0, k].set_title(f"{verdict}\ntrue: {short[yv_t[i]]}\npred: {short[pred_val[i]]}", fontsize=8, color=colour)
    plt.suptitle("Grad-CAM of the predicted class (top: image, bottom: heat map)", y=1.02); plt.tight_layout(); plt.show()
    """),
    md(r"""
    Look at where the heat sits: on most correct predictions it covers the pet's head or face; on the
    wrong ones it is more often spread out or partly on the background (the Staffordshire Bull Terrier on
    the left edge, the Beagle in a band through the grass). A few pictures are an impression, not
    evidence: the next cell turns the question into two measurements.

    ### Is it looking at the pet? A check for shortcut learning

    **Shortcut learning**: a model learns something that happens to go with the label in the training
    data, instead of the object itself. Known examples: snow -> husky (the huskies were photographed in
    snow), a sofa -> cat, a hospital's text marker on an X-ray -> pneumonia (one hospital had more sick
    patients). Such a model looks accurate on data like its training data and fails elsewhere.

    Pet comes with a mask of the animal (the trimap of D0-00), so we can measure where the evidence lies.
    Two checks on 100 random validation photos:
    1. **Evidence on the pet.** For 40 of them: the share of the heat map's total (its sum over all
       pixels) that falls on pet pixels (`day1.cam_on_mask_fraction`), against the share of the image the
       pet covers. A model that ignored the pet would spread its evidence anywhere, and the points would
       lie on the diagonal. Points above it mean that the evidence concentrates on the animal.
    2. **Background only.** Grey out the pet (set its pixels to 0, which after normalisation is the
       average ImageNet colour) and classify again. **Chance level**, the accuracy of random guessing, is
       1 in 37 = 2.7 %. Far above it would mean the background alone predicts the breed: a shortcut.
    """),
    api("torch.from_numpy", "np.mean", brief=True),
    code(r"""
    chk = torch.randperm(len(yv_t), generator=torch.Generator().manual_seed(0))[:100]      # 100 random val images
    chk_paths = []
    for i in chk.tolist():
        chk_paths.append(data["paths_val"][i])
    masks = day1.pet_masks_for(chk_paths, IMG)            # (100, IMG, IMG) NumPy: 1 = pet (with its border), 0 = background

    # Check 1: the share of the Grad-CAM evidence that lies on the pet, for the first 40 images
    on_pet = []
    area = []
    for k in range(40):
        cam = cam_fn(clf, clf.layer4, eval_tf(data["x_val"][chk[k]]))    # (IMG, IMG) in [0, 1]
        on_pet.append(day1.cam_on_mask_fraction(cam, masks[k]))           # sum of cam on the pet / sum of cam
        area.append(masks[k].mean())                                       # the share of pixels that are pet

    # Check 2: classify the same 100 images with the pet greyed out
    images = []
    for i in chk.tolist():
        images.append(eval_tf(data["x_val"][i]))
    x_bg = torch.stack(images)                            # (100, 3, IMG, IMG)
    m_t = torch.from_numpy(masks).float()[:, None]        # (100, 1, IMG, IMG): 1 on the pet
    x_bg = x_bg * (1 - m_t)                               # pet pixels -> 0 = the mean colour after normalisation
    with torch.no_grad():
        acc_orig = (logits_val[chk].argmax(1) == yv_t[chk]).float().mean().item()
        bg_logits = clf(x_bg.to(device))                  # (100, 37)
        acc_bg = (bg_logits.argmax(1).cpu() == yv_t[chk]).float().mean().item()

    # Left: evidence on the pet against pet area; middle: one image with the pet removed; right: the accuracies
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    axes[0].scatter(area, on_pet, color=viz.PALETTE[0]); axes[0].plot([0, 1], [0, 1], ":", color="gray", label="no preference")
    axes[0].set_xlabel("fraction of the image that is pet"); axes[0].set_ylabel("fraction of Grad-CAM mass on the pet")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[0].set_title(f"mean: {np.mean(on_pet):.2f} of the evidence on {np.mean(area):.2f} of the pixels")
    axes[1].imshow(torchkit.denormalize(x_bg[5])); axes[1].axis("off"); axes[1].set_title("pet removed (background only)")
    axes[2].bar(["original", "background only", "chance"], [acc_orig * 100, acc_bg * 100, 100 / 37],
                color=[viz.PALETTE[0], viz.PALETTE[3], "lightgray"])
    axes[2].set_ylabel("accuracy (%) on 100 val images"); axes[2].grid(alpha=.3, axis="y")
    for k, v in enumerate([acc_orig, acc_bg, 1 / 37]):
        axes[2].text(k, v * 100 + 1, f"{v * 100:.1f}", ha="center")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    What to look at: on the left almost every point lies above the diagonal: on average 70 % of the
    evidence lies on the 51 % of the pixels that are pet. On the right, with the pet removed, accuracy
    falls from 85 % to 2 %, below chance: no background shortcut in this dataset. Run the same two checks
    on your own data, especially when the images of different classes come from different sources
    (scanners, cameras, websites, hospitals).

    ## 7 · Choosing and reading metrics (ML Yearning, ch. 4-11)

    **Why.** A **metric** is the number you use to decide which model is better. With the wrong one you
    pick the wrong model, however well you train. Andrew Ng's *Machine Learning Yearning* gives four rules:

    * **Dev and test sets from the distribution you care about.** The **dev set** (development set: our
      validation set) is where you compare models while you work; the **test set** is used once, at the
      end, for the final number. Both must look like the data the model will meet in use. Pet's official
      split is fine; "train on web photos, test on phone photos" is a different problem.
    * **One single-number evaluation metric** to compare models quickly, for example accuracy, mean recall
      or F1 (defined below). With several numbers there is often no clear winner.
    * **Satisficing metrics** are constraints, not goals: "maximise accuracy *subject to* less than 50 ms
      per image on a phone and less than 30 MB". Among the models that satisfy them, pick the best by the
      single-number metric.
    * **Change the metric when it stops matching what users need.**

    **Metrics for a classifier.**
    * **Top-1 accuracy**: the fraction of images whose highest-scoring class is the true one. **Top-5
      accuracy**: the fraction whose true class is among the five highest scores.
    * **Recall** of a class c: of all images whose true class is c, the fraction predicted as c. **Mean
      recall** averages it over the classes, so every class counts the same, however many images it has.
    * **Precision** of a class c (not used below): of all images predicted as c, the fraction that really
      are c. **F1** combines the two into one number: 2 x precision x recall / (precision + recall).
    * A **confusion matrix** is a table of counts: row t, column p holds how many images of true class t
      were predicted as class p. The diagonal holds the correct predictions; every number off the
      diagonal is a confusion.

    **Mini case.** A pet-insurance app shows the breed of a photographed pet to the user. Which metric?
    The cell computes the candidates for the fine-tuned model of regime C on the 500 validation images:
    top-1, top-5, per-breed recall, and the most confused pairs. In it, `yv_t[:, None]` gives the true
    breeds the shape (500, 1), which broadcasts against the (500, 5) top-5 predictions.
    """),
    api("np.argsort", "Tensor.fill_diagonal_", "Tensor.any"),
    api("torch.topk", "torch.zeros", "Tensor.clone", brief=True),
    code(r"""
    top5_idx = torch.topk(logits_val, 5, 1).indices           # (500, 5): the 5 highest-scoring breeds of every image
    hit = (top5_idx == yv_t[:, None]).any(1)                  # (500,): True if one of the 5 is the true breed
    top5 = hit.float().mean().item()
    top1 = (pred_val == yv_t).float().mean().item()

    recall = np.zeros(37)                                     # recall of every breed
    for c in range(37):
        of_breed_c = yv_t == c                                # (500,): True for the images of breed c
        recall[c] = (pred_val[of_breed_c] == c).float().mean().item()
    order = np.argsort(recall)                                # breeds from the lowest to the highest recall
    print(f"top-1 {top1:.3f} | mean recall {recall.mean():.3f} | lowest recall {recall.min():.3f} "
          f"({classes[order[0]]}) | top-5 {top5:.3f}")

    # Bar chart: recall of every breed, sorted; cats in blue, dogs in orange
    bar_colours = []
    bar_labels = []
    for c in order:
        if classes[c] in CAT_BREEDS:
            bar_colours.append(viz.PALETTE[0])
        else:
            bar_colours.append(viz.PALETTE[1])
        bar_labels.append(short[c])
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.bar(range(37), recall[order] * 100, color=bar_colours)
    ax.set_xticks(range(37)); ax.set_xticklabels(bar_labels, rotation=70, ha="right", fontsize=7)
    ax.axhline(top1 * 100, ls="--", color="k", lw=1, label=f"top-1 = {top1 * 100:.1f}%, about the mean recall (balanced val set)")
    ax.set_ylabel("recall (%)"); ax.set_title(f"Per-breed recall (blue = cats, orange = dogs); top-5 accuracy = {top5 * 100:.1f}%")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3, axis="y"); plt.tight_layout(); plt.show()

    # Confusion matrix: row = true breed, column = predicted breed
    conf = torch.zeros(37, 37, dtype=torch.long)
    for t, p_ in zip(yv_t, pred_val):
        conf[t, p_] += 1
    off = conf.clone().fill_diagonal_(0)                      # the same table without the correct predictions
    print("most confused pairs (true -> predicted: count):")
    for k in torch.topk(off.flatten(), 5).indices:            # the 5 largest counts off the diagonal
        t, p_ = divmod(int(k), 37)                            # position in the flattened table -> (row, column)
        print(f"   {classes[t]:<28s} -> {classes[p_]:<28s} {int(off[t, p_])}")
    """),
    md(r"""
    Reading the case:

    * **Top-1** is the single-number metric if the app shows one breed; **top-5** if it shows a short list
      to pick from. Top-5 is much higher (95.4 % against 81.6 %), so the product decision changes the
      metric.
    * **Mean recall** is almost the same as top-1 here (81.7 % against 81.6 %) only because the
      validation set is balanced (13 or 14 images per breed). On a real, imbalanced stream (many more Labradors than Sphynx cats) report it
      separately, and track the **lowest recall** of any breed as a satisficing metric if every breed must
      work: here 46 %, for the Staffordshire Bull Terrier.
    * The confusions are between look-alike breeds (Staffordshire Bull Terrier and American Pit Bull
      Terrier, Beagle and Basset Hound, Maine Coon and Ragdoll). No metric fixes that; more data for those
      pairs, a higher resolution, or a better backbone does.
    * Measure the **latency** (the time from image to answer) and the size of the deployed model as
      satisficing metrics. The next cell times 5 single-image predictions after one warm-up run (the
      first call is always slower), and computes the size: float32 (`fp32`) stores 4 bytes per parameter,
      and 2**20 bytes are 1 MB.
    """),
    code(r"""
    x1 = eval_tf(data["x_val"][0])[None].to(device)      # a batch of one image
    with torch.no_grad():
        clf(x1)                                          # warm-up run
        t0 = time.perf_counter()
        for _ in range(5):
            clf(x1)
    ms = (time.perf_counter() - t0) / 5 * 1000           # milliseconds per image, averaged over 5 runs
    n_params = torchkit.count_parameters(clf)
    size_mb = n_params * 4 / 2**20                       # float32: 4 bytes per parameter
    print(f"{BACKBONE} @ {IMG} px: {n_params / 1e6:.1f} M params ({size_mb:.0f} MB fp32), "
          f"{ms:.0f} ms per image on this {device.type.upper()}")
    """),

    # ------------------------------------------------------------------ 8
    md(r"""
    ## 8 · Modern note: self-supervised backbones (optional)

    ImageNet-*supervised* features, learned from ImageNet's labels, lean towards the 1 000 ImageNet
    classes. **Self-supervised** learning trains a backbone without any labels, on a task made from the
    images themselves. For **DINOv2** (Oquab et al., 2023; 142 million curated images) the task is roughly:
    two differently cropped and coloured versions of the same photo must get similar features. Its
    features transfer better: a linear probe on DINOv2 ViT-S/14 features reaches about 95 % on Pet,
    against about 88-90 % for supervised ResNet-50. **ViT-S/14** is a small **vision transformer**: a
    network that cuts the image into patches of 14 x 14 pixels and relates every patch to every other one
    (Day 3 explains it). 126 px = 9 x 14, so here an image is 9 x 9 patches.

    The cell runs only if `timm` is installed (`pip install timm`; the first run downloads about 90 MB of
    weights). It extracts DINOv2 features with your `extract_features` (or the reference), trains the same
    linear probe, and compares with our ResNet's features on the same training images (740 on a CPU,
    20 per breed). In the CPU run: DINOv2 0.848 against ResNet-18 0.774, at a lower resolution. D3-04 goes
    further: a **k-NN classifier** (each image gets the most common label among its k most similar
    training images in feature space) that trains no layer at all, and CLIP, a network trained on photos
    with their captions, which needs no Pet labels.
    """),
    api("timm.create_model", "timm.data.resolve_data_config", "torchvision.transforms.v2.CenterCrop"),
    api("torchvision.transforms.v2.Resize", "torchvision.transforms.v2.Compose", "torchvision.transforms.v2.ToDtype",
        "torchvision.transforms.v2.Normalize", brief=True),
    code(r"""
    try:
        import timm
    except ImportError:
        timm = None
        print("timm is not installed - skipping the DINOv2 demo (pip install timm to run it). "
              "Expected: linear probe on DINOv2 ViT-S/14 features ~95 % on Pet vs ~88-90 % for supervised ResNet-50.")
    if timm is not None:
        # DINOv2 ViT-S/14 without a head (num_classes=0): it outputs features; its own mean / std for preprocessing
        dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=126).eval().to(device)
        cfg = timm.data.resolve_data_config({}, model=dino)
        dino_tf = v2.Compose([v2.Resize(126, antialias=True), v2.CenterCrop(126), v2.ToDtype(torch.float32, scale=True),
                              v2.Normalize(cfg["mean"], cfg["std"])])
        if GPU:
            n_tr = 2000
        else:
            n_tr = 740                               # 20 training images per breed on a CPU
        sub_d = torchkit._balanced_subset(data["y_train"].tolist(), n_tr, seed=0)
        dl_tr = DataLoader(day1.TensorImageDataset(data["x_train"][sub_d], data["y_train"][sub_d], dino_tf), batch_size=50)
        dl_va = DataLoader(day1.TensorImageDataset(data["x_val"], data["y_val"], dino_tf), batch_size=50)
        t0 = time.time()
        fd_tr, yd_tr = extract(dino, dl_tr)          # DINOv2 features of the training images
        fd_va, yd_va = extract(dino, dl_va)          # ... and of the validation images
        _, acc_dino, _ = day1.train_linear_probe(fd_tr, yd_tr, fd_va, yd_va, epochs=100)
        # The same probe on our ResNet's cached features of the same training images
        idx = torchkit._balanced_subset(F_["y_train"].tolist(), n_tr, seed=0)
        _, acc_sup, _ = day1.train_linear_probe(F_["f_train"][idx], F_["y_train"][idx], F_["f_val"], F_["y_val"], epochs=100)
        print(f"linear probe with {n_tr} training images: DINOv2 ViT-S/14 @126 px {acc_dino:.3f} vs {BACKBONE} @{IMG} px {acc_sup:.3f}"
              f"  ({time.time() - t0:.0f} s)")
    """),

    # ------------------------------------------------------------------ 9
    md(r"""
    ## 9 · Summary

    * On small data, start from a pretrained backbone. Match its preprocessing (`weights.transforms()`:
      the same resize, crop, mean and std) and replace its head with one for your classes.
    * **A** linear probe on cached features: seconds of training, no augmentation, a strong baseline.
      **B** frozen backbone + head: adds augmentation for the cost of one forward pass per image and epoch.
      **C** fine-tuning: train the head first, use a small learning rate, discriminative learning rates
      (earlier = smaller), BatchNorm in evaluation mode, and unfreeze from the top.
    * Freezing is `requires_grad = False` on the parameters of the early parts; discriminative learning
      rates are one optimiser parameter group (a dict with `"params"` and `"lr"`) per part.
    * Accuracy grows roughly linearly with log(training-set size): plot that curve before buying labels.
    * Early filters are generic (edges, colours), late features task-specific: the early ones transfer,
      the late ones must adapt.
    * Hooks (`register_forward_hook` on a layer, `register_hook` on a tensor) expose activations and
      gradients. Grad-CAM = ReLU(sum of the feature maps, each weighted by its average gradient). Check
      that the model looks at the object, and that the background alone does not predict the label.
    * One single-number metric, satisficing constraints, dev and test sets from the target distribution;
      change the metric when it stops matching the product.

    ### Self-check
    1. You fine-tune a ResNet with BatchNorm on batches of 4 images. Why might freezing the BatchNorm statistics help?
    2. Why train the head before unfreezing the backbone?
    3. Your linear probe gets 70 % with 500 images and 76 % with 1 000. Roughly what do you expect with 4 000?
    4. With `base_lr` 1e-3 and `decay` 0.5, what learning rate does layer2 get from `param_groups_with_lrs`?
    5. What two tensors does Grad-CAM need, and how do you get them in PyTorch?
    6. The model is 99 % accurate on the test set but Grad-CAM always highlights the bottom-left corner. What do you check next?
    7. Give one example of a satisficing metric for a phone app.

    ### Go deeper
    * J. Yosinski et al., *How transferable are features in deep neural networks?* (NeurIPS 2014).
    * R. R. Selvaraju et al., *Grad-CAM: Visual explanations from deep networks via gradient-based localization* (ICCV 2017).
    * A. Ng, *Machine Learning Yearning*, chapters 4-11 (dev / test sets, single-number and satisficing metrics).
    * M. Oquab et al., *DINOv2: Learning robust visual features without supervision* (2023).
    * A. Kumar et al., *Fine-tuning can distort pretrained features and underperform out-of-distribution* (ICLR 2022) - why "head first" (LP-FT) works.
    * torchvision docs, "Models and pre-trained weights"; timm docs, "Feature extraction" and "Fine-tuning".
    """),
    md(r"""
    ## Words from this notebook

    **Transfer learning**

    | word | meaning |
    |---|---|
    | transfer learning | reusing a network trained on one task (usually ImageNet classification) for another task |
    | ImageNet | 1.28 million training photos in 1 000 classes; most pretrained vision networks learned from it |
    | pretrained, weights | already trained by someone else; the trained parameters, downloaded as a file |
    | model zoo | a library of architectures with downloadable pretrained weights (torchvision, timm, Hugging Face) |
    | backbone | the network without its last layer: turns an image into features |
    | head | the last layer, here `fc`: turns the features into one score per class |
    | features, embedding | the vector of numbers a backbone outputs for one image (512 for ResNet-18) |
    | preprocessing | resize, crop, convert to float, normalise: must match what the weights were trained with |
    | feature extraction | running images through a frozen backbone and keeping the features |
    | caching | computing something once and storing it in a file, to reuse it |
    | linear probe | one linear layer trained on the features of a frozen backbone |
    | frozen | not changed by training: `requires_grad = False` |
    | fine-tuning | training a pretrained network further on new data, with a small learning rate |
    | regime | a way of setting up transfer: A linear probe, B frozen backbone + head, C fine-tune the top, C' everything |
    | stem, stage | a ResNet's first convolution with its BatchNorm; the groups of residual blocks `layer1` ... `layer4` |
    | evaluation mode | `model.eval()`: BatchNorm uses its running statistics, Dropout is off |
    | running statistics | the per-channel mean and std a BatchNorm layer averaged during training |
    | logits | a network's raw output scores, one per class |
    | softmax | turns logits into probabilities that add up to 1 |

    **Training choices**

    | word | meaning |
    |---|---|
    | parameter group | a dict `{"params": [...], "lr": ...}`: parameters that share optimiser settings |
    | discriminative learning rates | one learning rate per part, smaller for earlier parts: `base_lr * decay ** (5 - i)` |
    | AdamW | the optimiser of the course: per-parameter step sizes, weight decay applied separately |
    | weight decay | every step pulls each weight a little towards 0 |
    | data augmentation | random crops, flips, colour changes: a slightly different version of each image in every epoch |
    | label smoothing | the target puts 0.9 on the true class and spreads 0.1 over all classes |
    | warm-up + cosine schedule | the learning rate rises from about 0, then falls to 0 along a cosine curve |
    | overfitting | learning the training images by heart instead of what carries over to new images |
    | multiply-add, GFLOPs | one multiplication plus one addition; billions of operations per image |
    | log scale | an axis on which equal distances are equal factors, e.g. every doubling |

    **Looking inside**

    | word | meaning |
    |---|---|
    | filter | the weights of one output channel of a convolution, drawn as a tiny picture |
    | feature map, activation | a layer's output for one image, one grid per channel; one value of it |
    | hook | a function PyTorch calls when a layer runs (forward hook) or a gradient is computed (tensor hook) |
    | handle | what `register_forward_hook` returns; `handle.remove()` takes the hook off |
    | PCA, principal component | the directions along which data spread out most; projecting onto two of them gives a 2-D picture |
    | gradient with respect to a feature map | how much a class score would change if one activation rose a little |
    | Grad-CAM | heat map = ReLU(sum of feature maps, each weighted by its average gradient), upsampled to the image |
    | upsampling, bilinear | enlarging a grid by interpolating between its values; bilinear mixes the four nearest values |
    | shortcut learning | a model using something that happens to go with the label (background, watermark) instead of the object |
    | chance level | the accuracy of random guessing: 1 / 37 = 2.7 % for 37 classes |

    **Metrics**

    | word | meaning |
    |---|---|
    | metric | the number used to decide which model is better |
    | dev set, test set | the set models are compared on while working (our validation set); the set used once at the end |
    | single-number metric | one number to rank models by |
    | satisficing metric | a constraint the model must meet (latency, size), not a goal to maximise |
    | top-1, top-5 accuracy | true class has the highest score; true class is among the 5 highest |
    | recall, mean recall | of the images of a class, the fraction predicted as that class; its average over classes |
    | precision, F1 | of the images predicted as a class, the fraction that are; 2PR / (P + R) |
    | confusion matrix | counts of (true class, predicted class) pairs; the diagonal is correct |
    | latency | the time from an input to the model's answer |

    **Modern backbones**

    | word | meaning |
    |---|---|
    | self-supervised | trained without labels, on a task made from the images themselves |
    | DINOv2 | a self-supervised vision transformer whose features transfer very well |
    | vision transformer (ViT-S/14) | a network that cuts the image into 14 x 14-pixel patches and relates every patch to every other |
    | k-NN classifier | gives an image the most common label among its k most similar training images |
    """),
]
