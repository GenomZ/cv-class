from nb import md, code, exercise, api

NAME = "P5-transfer-learning"

CELLS = [
    md(r"""
    # P5 · Transfer learning: adapt a pretrained network to your own photos

    **What we do today**

    1. The idea: reuse a network that others have trained on more than a million photos.
    2. The data: photos of 5 breeds of cats and dogs, only 60 per breed.
    3. A baseline: a small network trained from nothing on the same photos (it does poorly).
    4. The recipe, in five steps: download a pretrained network, freeze it, give it a new classifier, train the
       classifier, then unfreeze its last block and train a little more.
    5. Use the adapted network: name the breed in a photo file, save the network to a file and load it again.
    6. Change the settings yourself and run the whole recipe again (section 10, **Try it**).

    **How to use this notebook.** Click into the first code cell and press **Shift+Enter** to run it; that moves
    you to the next cell. Run the cells from top to bottom: later cells use what earlier cells made. You type
    three short lines of code today, in steps 2, 3 and 5. Each is marked `____` (four underscores) with a comment
    `# YOUR CODE:` that says exactly what to write. The cell after it checks your line and prints `[PASS]` or
    `[FAIL]` with a hint. If you leave a blank as it is, the check does the step for you and says so ("done for
    you so you can continue"), so the notebook always runs to the end. Cells marked **Try it** hold settings you
    may change. The whole notebook runs in about 2 to 5 minutes on a laptop; every training cell says how long it
    takes.

    * `P.something(...)` is a function of the course toolkit (P1 introduced it): it does the plumbing (drawing,
      timing, the training loop) so that you can concentrate on the recipe. Each one is explained in a sentence
      where it is first used.
    * `T.something(...)` is one of today's checks.
    * Closed boxes **Functions used here (click to open)** explain the PyTorch functions in the next cell, with
      a small example. Open them when you want the details.
    * **New words** are in bold where they are explained. All of them are collected at the end.
    * Numbers quoted in the text are what the cells print. On your computer the last digit can differ a little,
      and times depend on how fast your computer is and on what else it is doing.
    """),

    # ------------------------------------------------------------------ 0
    md(r"""
    ## 0 · Setup

    The first cell prepares everything, and must run first every time you open the notebook:

    * the `while` loop finds the folder `cvcourse`, the course's helper code, and lets Python import it;
    * `torch` is **PyTorch**, `torch.nn` (short `nn`) its building blocks (the layers), and `torchvision` the
      PyTorch library for images: it holds the pretrained networks we download;
    * `P` is the course toolkit and `T` today's checks;
    * `P.setup(seed=0)` fixes the **random seed**, so that every run gives the same results, and prints whether
      PyTorch computes on the CPU or on a graphics card (a CPU is fine);
    * `____ = None` creates the blank you will replace three times. Until you replace it, a blank does nothing.
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
    from torchvision.transforms import v2
    from PIL import Image
    from cvcourse import torchkit
    from cvcourse import practical as P
    from cvcourse import ptests_p5 as T

    P.setup(seed=0)
    ____ = None                                     # the blank: replace it where a cell says YOUR CODE
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · The idea

    Training a good image network from nothing needs a lot of photos. **ResNet-18**, the network we use today,
    learned from **ImageNet**: 1.28 million photos sorted into 1 000 classes (animals, vehicles, food, tools,
    ...), for days on graphics cards. What it learned is not only about those 1 000 classes. Its first layers
    respond to edges and colours, the next ones to textures (fur, grass, metal), the deepest ones to patterns
    and object parts (eyes, ears, noses, wheels). That knowledge is useful for almost any photo.

    **Transfer learning** reuses it. A classification network has two parts:

    * the **feature extractor**: all the layers except the last. It turns a photo into a list of numbers that
      describe it, its **features** (for ResNet-18: 512 numbers);
    * the **classifier** (also called the **head**): the last layer. It turns the 512 features into one score
      per class (for ImageNet: 1 000 scores).

    Keep the feature extractor as it is and replace only the classifier by a new one for *your* classes. The new
    layer is the only part that has to learn, and it learns from a few hundred photos in about a minute.
    """),
    code("""
    T.show_idea()
    """),
    md(r"""
    The light blue blocks are **frozen**: copied as they are, training does not change them. Only the new green
    classifier learns. The dashed block, `layer4`, is the optional last step: once the classifier works, let the
    last block of the feature extractor adapt a little too.
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · The data: 5 breeds, 60 photos each

    **Oxford-IIIT Pet** is a collection of 7 390 photos of 37 breeds of cats and dogs, about 200 per breed, taken
    by their owners: all poses, backgrounds and lighting. The photos are already in the folder
    `data/oxford-iiit-pet/images`. Every file name starts with the breed, for example `pug_12.jpg` or
    `Siamese_7.jpg` (cat breeds start with a capital letter, dog breeds do not).

    We take 5 breeds that look different and only **60 training photos per breed, 300 in total**. That is very
    little: ImageNet has about 1 300 photos per class, and P1 trained on 10 000 digits. The **validation photos**
    are 40 other photos of each breed (200 in total): never used for training, only to measure how well the
    network does on photos it has not seen. The breeds are the **classes**: the possible answers.

    The loader below is plain code that you can copy into your own project:

    * `preparation(image_size)` lists the steps that turn a photo into the numbers a network gets: scale it and
      cut out a square of `image_size` x `image_size` pixels, turn it into a **tensor** (PyTorch's array of
      numbers) of shape 3 x height x width with values 0 to 1, then **normalise** each colour channel: subtract
      the ImageNet average and divide by the ImageNet spread (step 1 explains why exactly these numbers).
      `v2.Compose([...])` chains the steps in order.
    * `load_pets(...)` finds the files of each breed, takes the first ones for training and the last ones for
      validation, reads and prepares every photo, and returns two loaders. It takes the breeds in turn (photo 1 of
      every breed, then photo 2, ...), so that the breeds are mixed in the pictures below.
    * `P.ImageSet(images, labels, classes, mean, std)` keeps the prepared photos and their **labels** (the breed
      number of each photo: 0 for the first breed, 1 for the second, ...) in memory; `P.loaders(...)` hands them
      out in **batches** of 32. The training loader shuffles them in every **epoch** (one pass over all training
      photos).
    """),
    api("torchkit.data_root", "PIL.Image.open", "torchvision.transforms.v2.Compose", "torchvision.transforms.v2.Resize",
        "torchvision.transforms.v2.CenterCrop", "torchvision.transforms.v2.ToImage",
        "torchvision.transforms.v2.ToDtype", "torchvision.transforms.v2.Normalize", "torch.stack", "torch.tensor"),
    code("""
    IMAGENET_MEAN = [0.485, 0.456, 0.406]       # the average red, green and blue of the ImageNet photos (0 to 1)
    IMAGENET_STD = [0.229, 0.224, 0.225]        # and their spread (standard deviation)


    def preparation(image_size):
        \"\"\"The steps that turn one photo into what an ImageNet network expects.\"\"\"
        return v2.Compose([
            v2.Resize(image_size),                      # scale the photo: its shorter side becomes image_size pixels
            v2.CenterCrop(image_size),                  # cut out the central square, image_size x image_size
            v2.ToImage(),                               # photo -> tensor 3 x height x width, values 0 to 255
            v2.ToDtype(torch.float32, scale=True),      # -> decimal numbers 0 to 1
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # -> (value - mean) / std, per colour channel
        ])


    def load_pets(breeds, photos_per_class, val_per_class, image_size):
        \"\"\"Training and validation loaders for a few Oxford-IIIT Pet breeds.\"\"\"
        folder = torchkit.data_root() / "oxford-iiit-pet" / "images"
        prepare = preparation(image_size)
        train_files = []                                # train_files[label] = the training files of that breed
        val_files = []
        for breed in breeds:
            found = sorted(folder.glob(breed + "_*.jpg"))         # every photo of this breed, e.g. pug_1.jpg ...
            if len(found) < photos_per_class + val_per_class:
                raise ValueError(f"'{breed}': {len(found)} photos found, {photos_per_class + val_per_class} needed. "
                                 f"Check the spelling (section 10 lists the breeds) or ask for fewer photos.")
            train_files.append(found[:photos_per_class])          # the first photos: training
            val_files.append(found[-val_per_class:])              # the last photos: validation, never trained on

        train_images, train_labels = [], []
        for k in range(photos_per_class):               # photo k of every breed in turn, so the breeds are mixed
            for label in range(len(breeds)):
                photo = Image.open(train_files[label][k]).convert("RGB")
                train_images.append(prepare(photo))
                train_labels.append(label)
        val_images, val_labels = [], []
        for k in range(val_per_class):
            for label in range(len(breeds)):
                photo = Image.open(val_files[label][k]).convert("RGB")
                val_images.append(prepare(photo))
                val_labels.append(label)

        train_set = P.ImageSet(torch.stack(train_images), torch.tensor(train_labels), breeds,
                               IMAGENET_MEAN, IMAGENET_STD, name="Oxford-IIIT Pet")
        val_set = P.ImageSet(torch.stack(val_images), torch.tensor(val_labels), breeds,
                             IMAGENET_MEAN, IMAGENET_STD, name="Oxford-IIIT Pet")
        print(f"{len(breeds)} breeds: {len(train_set)} training photos, {len(val_set)} validation photos, "
              f"each 3 x {image_size} x {image_size}")
        return P.loaders(train_set, val_set, batch_size=32)
    """),
    md(r"""
    Now load 5 breeds at 128 x 128 pixels and look at the first 20 training photos. `number_of_classes` is the
    number of breeds: we need it for the classifiers.

    **Expected time: a few seconds.**
    """),
    code("""
    breeds = ["Bombay", "Siamese", "pug", "samoyed", "beagle"]   # the classes: how their file names start
    image_size = 128
    train_loader, val_loader = load_pets(breeds, photos_per_class=60, val_per_class=40, image_size=image_size)
    number_of_classes = len(breeds)
    P.show_samples(train_loader, breeds, n=20, cols=10)
    """),
    md(r"""
    A Bombay is a black cat, a Siamese a cream cat with dark ears and face, a pug a small dog with a flat black
    face, a samoyed a big white fluffy dog, a beagle a brown, black and white hound. Easy for us, but the network
    has to learn it from 60 photos per breed, all different in pose and background.

    <details>
    <summary><b>Your own photos</b> (click to open)</summary>

    Sort them into one folder per class, for example `my_photos/train/cat/*.jpg` and `my_photos/train/dog/*.jpg`,
    and a second set of other photos in `my_photos/val/cat` and `my_photos/val/dog`. Then
    `torchvision.datasets.ImageFolder` does the work of `load_pets` (it reads the photos as the loader asks for
    them; the class names are the folder names, in alphabetical order):

    ```python
    train_set = torchvision.datasets.ImageFolder("my_photos/train", transform=preparation(128))
    val_set = torchvision.datasets.ImageFolder("my_photos/val", transform=preparation(128))
    train_loader, val_loader = P.loaders(train_set, val_set, batch_size=32)
    ```

    `train_set.classes` lists the class names; use it wherever this notebook uses `breeds`. The rest of the
    recipe stays the same.
    </details>
    """),
    api("torchvision.datasets.ImageFolder"),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · A baseline: a small network trained from scratch

    To see what transfer learning brings, first train a network **from scratch**, that is, starting from random
    weights, on the same 300 photos. It is a small convolutional network like network B of P1, made for colour
    photos: four convolutions (3 colour channels in, 16, 32, 64, 64 feature maps), each followed by ReLU, the
    first three by max pooling (128 -> 64 -> 32 -> 16 pixels).

    One new layer: `nn.AdaptiveAvgPool2d(1)` averages each of the 64 feature maps over the whole picture into one
    number (**global average pooling**), so 64 numbers reach the final linear layer whatever the picture size.
    ResNet-18 ends in the same way.

    `P.train(model, train_loader, val_loader, epochs=..., lr=...)` trains the network (P1 explains it): one line
    per epoch, and the curves of the **loss** (how wrong the scores are, lower is better) and the **accuracy**
    (the percentage of photos whose highest score is the right breed) for the training photos (blue) and the
    validation photos (orange). `lr` is the **learning rate**: how big each change of the weights is.

    **Expected time: about 15 seconds** (up to a minute on a slow laptop).
    """),
    api("torch.manual_seed", "nn.Sequential", "nn.Conv2d", "nn.ReLU", "nn.MaxPool2d", "nn.AdaptiveAvgPool2d",
        "nn.Flatten", "nn.Linear"),
    code("""
    torch.manual_seed(0)                                     # the same random starting weights every time
    scratch_net = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),          # 3 colour channels in -> 16 maps: 16 x 128 x 128
        nn.ReLU(),
        nn.MaxPool2d(2),                                     # 16 x 64 x 64
        nn.Conv2d(16, 32, kernel_size=3, padding=1),         # 32 x 64 x 64
        nn.ReLU(),
        nn.MaxPool2d(2),                                     # 32 x 32 x 32
        nn.Conv2d(32, 64, kernel_size=3, padding=1),         # 64 x 32 x 32
        nn.ReLU(),
        nn.MaxPool2d(2),                                     # 64 x 16 x 16
        nn.Conv2d(64, 64, kernel_size=3, padding=1),         # 64 x 16 x 16
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),                             # average of each map: 64 x 1 x 1
        nn.Flatten(),                                        # 64 numbers
        nn.Linear(64, number_of_classes),                    # one score per breed
    )
    history_scratch = P.train(scratch_net, train_loader, val_loader, epochs=10, lr=0.001,
                              name="from scratch: small CNN")
    """),
    md(r"""
    **Reading the curves.** The validation accuracy creeps from 27.0% after the first epoch to its best, 42.0%, in
    epoch 9, and jumps up and down on the way (31.0% in epoch 8). That is better than guessing (1 breed in 5 is
    20%), but more than half of the photos are wrong. The training accuracy is no better (41.0% at the end): the
    network has not even learned the photos it trains on. 300 photos are far too few to learn from nothing what a
    pug or a Siamese looks like.

    `P.remember(name, model, history)` records this result for the leaderboard at the end of the notebook.
    """),
    code("""
    P.remember("from scratch: small CNN", scratch_net, history_scratch)
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · Step 1: download a pretrained network

    A **pretrained** network is one that somebody else has already trained. Its **weights** (all its learned
    numbers) come as a file that you download. `torchvision.models` holds dozens of such networks.
    `torchvision.models.resnet18(weights=...)` builds ResNet-18 and fills it with the weights named by
    `ResNet18_Weights.IMAGENET1K_V1`: the ones learned on ImageNet.

    **The first time**, this line downloads the weights file, about 45 MB, into PyTorch's cache folder (on
    Windows `C:\Users\<you>\.cache\torch\hub\checkpoints`, on Linux and macOS `~/.cache/torch/hub/checkpoints`).
    Every later time it loads the file from there in about a second. Without an internet connection, copy the
    file `resnet18-f37072fd.pth` from a computer that has it into that folder.

    `print(model.fc)` shows the last layer, the classifier.
    """),
    api("torchvision.models.resnet18", "torchvision.models.ResNet18_Weights"),
    code("""
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    print(model.fc)
    """),
    md(r"""
    The classifier `fc` is a linear layer: 512 numbers in, 1 000 scores out, one per ImageNet class.

    `P.show_model(model, input_shape, depth=1)` draws the network one block per box (P1 explains the drawing;
    `depth=1` keeps each of the blocks `layer1` to `layer4`, a stack of 4 or 5 convolution layers with BatchNorm
    and ReLU, as one stacked box).
    """),
    code("""
    P.show_model(model, (3, 128, 128), depth=1, title="ResNet-18 as downloaded")
    """),
    md(r"""
    **What to look at.** The photo goes through a first convolution (`conv1`, with `bn1`, a **BatchNorm** layer
    that keeps the numbers in a good range) and pooling, then the four blocks `layer1` to `layer4`. The picture
    shrinks (64 x 32 x 32 after the pooling, 512 x 4 x 4 after `layer4`) while the number of feature maps grows
    (64 to 512). `AvgPool` (global average pooling) turns the 512 maps into 512 numbers: the features. Then `fc`.

    So the **feature extractor** is everything up to the pooling, and the **classifier** is `fc`. `P.summary`
    lists the same blocks with their parameters:
    """),
    code("""
    P.summary(model, (3, 128, 128), depth=1)
    """),
    md(r"""
    The feature extractor holds 11 176 512 parameters (all rows except `fc`), the classifier 512 x 1 000 + 1 000
    = 513 000. `layer4` alone has 8 393 728, 72% of the feature extractor: the deepest block has the most
    feature maps (512) and so the most filter weights. `P.cost` measures what the network costs on this computer
    (P1 explains the report):
    """),
    code("""
    P.cost(model, (3, 128, 128), name="ResNet-18", epoch_images=300)
    """),
    md(r"""
    ResNet-18 has 192 times the parameters of our small network (11 689 512 against 60 837) and needs 11 times
    the multiply-adds per photo (593 million against 54 million). Training all of it would take about 5.3 s per
    epoch of 300 photos on our computer (yours prints its own time). We will train much less of it.

    **What has it learned?** `P.show_filters(model)` draws the 64 filters of the first convolution `conv1` as
    small colour pictures (each 7 x 7 pixels). Nobody designed them: they came out of the ImageNet training.
    """),
    code("""
    P.show_filters(model, title="the 64 filters of conv1, learned on ImageNet")
    """),
    md(r"""
    Many filters are grey lines and edges at different angles, a dark band next to a light one (for example
    filters 14, 20, 41 and 44); others respond to colours or to one colour next to another (orange against blue
    in filters 5, 23 and 31, magenta against green in 6 and 16). Each of them finds its pattern anywhere in a
    photo, and the deeper blocks combine them into textures and parts. This is the knowledge we transfer.

    **The preparation the weights expect.** The weights were learned on photos with values 0 to 1, each colour
    channel normalised with the ImageNet mean (0.485, 0.456, 0.406) and spread (0.229, 0.224, 0.225): that is why
    `preparation` uses exactly these numbers. Photos prepared another way give no error, only worse answers,
    because every number is off from what the weights are used to. The size is more flexible: ImageNet photos
    were 224 x 224 pixels, and thanks to the global average pooling the network accepts our 128 x 128 too
    (faster on a laptop, a little less accurate).
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · Step 2: freeze it

    Every parameter tensor (a weight tensor or a bias tensor of one layer) has a switch, `requires_grad`. When it
    is `True`, training computes how this tensor should change and changes it: the tensor is **trainable**. When
    it is `False`, training leaves the tensor as it is: it is **frozen**. Freezing the whole network protects
    what it learned on ImageNet. (The new classifier of step 3 will be trainable again.)

    `count_trainable(model)` adds up the numbers in all trainable tensors: `model.parameters()` goes through every
    parameter tensor of the network, and `param.numel()` is how many numbers one tensor holds.
    """),
    api("Module.parameters", "Tensor.requires_grad", "Tensor.numel"),
    code("""
    def count_trainable(model):
        \"\"\"How many numbers (weights and biases) training may change.\"\"\"
        total = 0
        for param in model.parameters():                 # every weight tensor and bias tensor of the network
            if param.requires_grad:                      # trainable?
                total = total + param.numel()            # add the number of values in this tensor
        return total
    """),
    md(r"""
    **Your line (1 of 3).** Inside the loop, replace `____` by `param.requires_grad = False`. Then run the cell,
    and the check after it.
    """),
    exercise(
        stub="""
        print(f"trainable parameters before freezing: {count_trainable(model):,}")
        for param in model.parameters():                 # every parameter tensor of the network, one after the other
            ____                                         # YOUR CODE: replace ____ by   param.requires_grad = False
        print(f"trainable parameters after freezing:  {count_trainable(model):,}")
        """,
        solution="""
        print(f"trainable parameters before freezing: {count_trainable(model):,}")
        for param in model.parameters():                 # every parameter tensor of the network, one after the other
            param.requires_grad = False                  # frozen: training will not change this tensor
        print(f"trainable parameters after freezing:  {count_trainable(model):,}")
        """),
    code("""
    T.check_frozen(model)
    """),
    md(r"""
    All 11 689 512 parameters are now frozen: trained as it is, the network would not change at all.
    """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · Step 3: a new classifier

    Replace the ImageNet classifier by a new linear layer for our breeds: 512 features in (what the feature
    extractor gives), one score per breed out. Assigning to `model.fc` swaps the layer; the rest of the network
    stays as it is. A new layer starts with random weights and is trainable (`requires_grad` is `True`), so
    after this step exactly its parameters are trainable: 512 x 5 weights + 5 biases = 2 565.

    **Your line (2 of 3).** Replace `____` by `nn.Linear(512, number_of_classes)`.
    """),
    exercise(
        stub="""
        torch.manual_seed(0)                             # the same random starting weights for the new layer every time
        model.fc = ____                                  # YOUR CODE: replace ____ by   nn.Linear(512, number_of_classes)
        """,
        solution="""
        torch.manual_seed(0)                             # the same random starting weights for the new layer every time
        model.fc = nn.Linear(512, number_of_classes)     # 512 features in -> one score per breed out
        """),
    code("""
    T.check_new_head(model, number_of_classes)
    """),
    code("""
    P.show_model(model, (3, 128, 128), depth=1, title="ResNet-18 with a new classifier for 5 breeds")
    """),
    md(r"""
    The same network, except the last box: `Linear 512 -> 5`, 2 565 parameters. Of the 11 179 077 parameters,
    only those 2 565 will learn.

    **Before training**, the new classifier has random weights. `P.evaluate(model, loader)` measures the loss
    and the accuracy on all photos of a loader without changing anything. With 5 breeds, guessing gets 1 photo in
    5 right: 20%, the **chance level**.
    """),
    code("""
    loss, accuracy = P.evaluate(model, val_loader)
    print(f"before training: validation accuracy {accuracy:.1f}%")
    """),
    md(r"""
    12.5%: no better than guessing (here even a little worse, by chance). The features are good, but the random
    classifier does not know yet which features belong to which breed. That is what it learns next.
    """),

    # ------------------------------------------------------------------ 7
    md(r"""
    ## 7 · Step 4: train only the new classifier

    The **optimiser** is the method that changes the trainable numbers after every batch; here **Adam**, the one
    `P.train` uses (P1). It changes exactly the tensors you hand to it. `trainable_parameters(model)` collects the
    trainable tensors (only `fc.weight` and `fc.bias` now), and `T.check_optimizer` confirms that the optimiser got
    them and nothing else. `P.train` normally builds the optimiser itself; with `optimizer=` it uses ours.
    """),
    api("torch.optim.Adam"),
    code("""
    def trainable_parameters(model):
        \"\"\"The parameter tensors that training may change: what the optimiser gets.\"\"\"
        found = []
        for param in model.parameters():
            if param.requires_grad:
                found.append(param)
        return found


    optimizer = torch.optim.Adam(trainable_parameters(model), lr=0.001)
    T.check_optimizer(optimizer, model)
    """),
    md(r"""
    Now train for 5 epochs with learning rate 0.001. Every photo still goes through the whole network, but the
    training step only has to work out the changes for the 2 565 numbers of the classifier, so it is fast.

    **Expected time: about 17 seconds** (up to a minute on a slow laptop).
    """),
    code("""
    history_head = P.train(model, train_loader, val_loader, epochs=5, lr=0.001, optimizer=optimizer,
                           name="ResNet-18: new classifier only")
    P.remember("ResNet-18: new classifier only", model, history_head)
    """),
    md(r"""
    **Reading the curves.** After one epoch the validation accuracy is already 77.5%; after four it reaches
    92.5%, its best, and stays there. The network trained from scratch never passed 42.0% in 10 epochs. Same
    photos, same number of training steps per epoch: the difference is everything ResNet-18 learned on ImageNet.
    (The training accuracy starts lower than the validation accuracy because it is averaged over the epoch while
    the classifier is still learning; the validation accuracy is measured at the end of the epoch.)

    `P.show_predictions(model, loader, classes)` shows the network's answer for the first 16 validation photos:
    green when right, with how sure it is (in percent), red and marked WRONG when wrong.
    """),
    code("""
    P.show_predictions(model, val_loader, breeds)
    """),
    md(r"""
    15 of the 16 photos are right. The one mistake is a pug lying on a striped blanket, called a Bombay (a black
    cat). The percentages vary: the network is sure about clear photos (the beagle by the sea, 100%) and unsure
    about unusual ones (the samoyed puppy, 36%, although right).

    `P.confusion(model, loader, classes)` counts, for every true breed (row), which breed the network answered
    (column): the grey diagonal holds the correct answers, orange cells the mistakes. It also returns the counts
    as a table of numbers; we keep them in `counts`.
    """),
    code("""
    counts = P.confusion(model, val_loader, breeds, title="rows: true breed, columns: answer")
    """),
    md(r"""
    15 of the 200 validation photos are wrong (7.5%), at most 2 in any one cell. Pugs are the hardest: 34 of 40
    right, the others called beagle, samoyed, Bombay or Siamese.

    **Side by side with the network trained from scratch.** `P.compare` puts their size, speed and best validation
    accuracy in one table and chart, and `P.plot_history` draws their curves on top of each other (solid lines:
    validation, dashed: training).
    """),
    code("""
    P.compare({"from scratch": scratch_net, "ResNet-18, new classifier": model}, (3, 128, 128),
              histories={"from scratch": history_scratch, "ResNet-18, new classifier": history_head},
              epoch_images=300)
    P.plot_history({"from scratch": history_scratch, "ResNet-18, new classifier": history_head})
    """),
    md(r"""
    **What the numbers say.** The pretrained network is much bigger: 184 times the parameters (11 179 077
    against 60 837) and 11 times the multiply-adds, so it takes several times as long per photo (on our
    computer, with nothing else running: 5.51 ms against 1.29 ms). Its training step is much cheaper than with
    the whole network trainable (there: 210 ms for 32 photos, against the 570 ms that `P.cost` measured before
    freezing): for frozen layers there is nothing to work out backwards. For that price the best validation
    accuracy goes from 42.0% to 92.5%. Times depend on your computer and on what else runs on it; parameters,
    multiply-adds and accuracies do not.
    """),

    # ------------------------------------------------------------------ 8
    md(r"""
    ## 8 · Step 5: unfreeze the last block and train a little more

    The feature extractor was made for ImageNet's 1 000 classes. Its last block, `layer4`, holds the most
    specific patterns (object parts); letting it adapt to our breeds can make the features fit better. This is
    called **fine-tuning**. We unfreeze only `layer4` and keep the earlier blocks frozen: edges and textures are
    useful as they are.

    Then train again, with a **smaller learning rate** (0.0001 instead of 0.001). `layer4` already holds good
    weights; big changes, driven by only 300 photos, would destroy what it learned on ImageNet instead of
    adjusting it.

    **Your line (3 of 3).** Inside the loop, replace `____` by `param.requires_grad = True`.
    """),
    exercise(
        stub="""
        for param in model.layer4.parameters():          # only the parameter tensors of the last block, layer4
            ____                                         # YOUR CODE: replace ____ by   param.requires_grad = True
        print(f"trainable parameters now: {count_trainable(model):,}")
        """,
        solution="""
        for param in model.layer4.parameters():          # only the parameter tensors of the last block, layer4
            param.requires_grad = True                   # trainable again
        print(f"trainable parameters now: {count_trainable(model):,}")
        """),
    code("""
    T.check_unfrozen(model, ["layer4"])
    """),
    md(r"""
    Now 8 396 293 numbers are trainable (8 393 728 in `layer4` + 2 565 in `fc`). An optimiser only changes the
    tensors it was given, and the one of step 4 knows only `fc`: build a **new** optimiser after unfreezing.

    **Expected time: about 20 seconds** (up to a minute and a half on a slow laptop): the training step now also
    works out the changes for `layer4`.
    """),
    code("""
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=0.0001)     # a new optimiser, 10 times smaller steps
    T.check_optimizer(optimizer, model)
    history_finetune = P.train(model, train_loader, val_loader, epochs=5, lr=0.0001, optimizer=optimizer,
                               name="ResNet-18: + layer4 unfrozen")
    P.remember("ResNet-18: + layer4 unfrozen", model, history_finetune)
    """),
    md(r"""
    **Reading the curves.** The run starts from the trained classifier, so its first epoch is already at 96.5%;
    the second reaches 98.0%, the best, and the last three stay at 98.0% and 97.5%. The training accuracy is
    100.0% from epoch 2 on and the training loss falls to 0.002, while the validation loss stays near 0.105: with
    8.4 million trainable numbers and only 300 photos, the network now knows its training photos by heart. More
    epochs would add little.

    The two runs of the ResNet side by side, and the mistakes that are left:
    """),
    code("""
    P.plot_history({"new classifier only": history_head, "+ layer4 unfrozen": history_finetune})
    counts = P.confusion(model, val_loader, breeds, title="after unfreezing layer4")
    """),
    md(r"""
    **Is unfreezing worth it?** Here, yes: 5 mistakes out of 200 instead of 15 (97.5% at the end, 98.0% at best,
    against 92.5%), for 20 more seconds of training. Read such numbers with care, though: with 200 validation
    photos, one photo is 0.5%, so a difference of 2 or 3 photos can be luck. The experiments of section 10 show
    the limits: unfreezing `layer3` as well adds time but no accuracy here, and too big a learning rate makes the
    network worse than before unfreezing. Unfreezing pays off most when
    you have more photos (thousands rather than hundreds) and when your photos look unlike ImageNet's (X-rays,
    satellite images, microscope slides): then the ImageNet features fit less well and need to adapt. With few
    photos of everyday things, training only the classifier is often enough, and it is faster and safer.
    """),

    # ------------------------------------------------------------------ 9
    md(r"""
    ## 9 · Use it

    ### Name the breed in a photo file

    `predict_photo` is what you would use in an application: read one photo file, prepare it exactly like the
    training photos, and let the network answer. Three new things in it:

    * `x.unsqueeze(0)` turns one photo (3 x 128 x 128) into a batch of one photo (1 x 3 x 128 x 128): networks
      always take batches;
    * `model.eval()` switches the network to **evaluation mode**: the BatchNorm layers then use the averages they
      stored during training instead of those of the current batch, so the answer for a photo does not depend on
      the other photos (`P.train` and `P.evaluate` do this for you); `torch.no_grad()` tells PyTorch not to
      prepare for training, which saves time and memory;
    * `scores.softmax(dim=1)` turns the 5 scores into 5 **probabilities**: numbers between 0 and 1 that add up to
      1. The highest one is the network's **confidence** in its answer.
    """),
    api("Tensor.unsqueeze", "Module.eval", "torch.no_grad", "Tensor.softmax", "Tensor.argmax"),
    code("""
    def predict_photo(model, path, breeds, image_size):
        \"\"\"The breed the model sees in one photo file, and how sure it is (0 to 1).\"\"\"
        photo = Image.open(path).convert("RGB")
        x = preparation(image_size)(photo)              # the same preparation as the training photos: 3 x H x W
        batch = x.unsqueeze(0)                          # a batch of one photo: 1 x 3 x H x W
        model.eval()                                    # evaluation mode
        with torch.no_grad():                           # only answering, no training
            scores = model(batch)                       # 1 x 5 scores
        probabilities = scores.softmax(dim=1)[0]        # 5 numbers between 0 and 1 that add up to 1
        best = int(probabilities.argmax())              # the position of the highest probability
        return breeds[best], float(probabilities[best])


    photo_folder = torchkit.data_root() / "oxford-iiit-pet" / "images"
    for breed in breeds:
        files = sorted(photo_folder.glob(breed + "_*.jpg"))
        path = files[-1]                                # the last photo of this breed: one of the validation photos
        predicted, confidence = predict_photo(model, path, breeds, image_size)
        print(f"{path.name:<16} the network says: {predicted:<8} ({100 * confidence:.1f}% sure)")
    """),
    md(r"""
    All five right, four of them at 98.8% or more and the pug at 96.8%. To use the network on your own photo,
    pass its file name as `path`.

    ### Save it and load it again

    `model.state_dict()` holds all the numbers of the network, by name; `torch.save` writes them to a file. To use
    the network later (in another notebook or program), build the same network with `weights=None` (random
    weights, nothing downloaded), give it the same new classifier, and load the saved numbers into it with
    `load_state_dict`. The file goes into the folder of this notebook.
    """),
    api("Module.state_dict", "torch.save", "torch.load", "Module.load_state_dict"),
    code("""
    torch.save(model.state_dict(), "pets_resnet18.pt")                 # save: all the numbers of the adapted network
    size_mb = pathlib.Path("pets_resnet18.pt").stat().st_size / 1_000_000
    print(f"saved pets_resnet18.pt: {size_mb:.1f} MB")

    reloaded = torchvision.models.resnet18(weights=None)               # load: the same network, random weights,
    reloaded.fc = nn.Linear(512, number_of_classes)                    # with the same new classifier,
    reloaded.load_state_dict(torch.load("pets_resnet18.pt"))           # then the saved numbers put in

    predicted, confidence = predict_photo(reloaded, path, breeds, image_size)
    print(f"the reloaded network on {path.name}: {predicted} ({100 * confidence:.1f}% sure)")
    """),
    md(r"""
    The file is 44.8 MB: about 4 bytes for each of the 11.2 million numbers. The reloaded network gives the same
    answer as the one we trained (beagle, 99.9%): the file holds everything it learned.
    """),

    # ------------------------------------------------------------------ 10
    md(r"""
    ## 10 · Try it: the whole recipe with your settings

    `run_recipe(...)` repeats the data loading and steps 1 to 5 with the settings you give it, in plain code (the
    same lines as above, without the checks). Two details:

    * `getattr(model, "layer4")` is the same as `model.layer4`: it lets the block names come from the list
      `unfreeze`;
    * `model.fc.in_features` is the number of features the old classifier received: 512 for ResNet-18, 2 048 for
      ResNet-50, so the new classifier fits either network.

    It records each training in the leaderboard (`P.remember`) and ends by printing the validation accuracy of the
    network you keep.
    """),
    api("torchvision.models.resnet50"),
    code("""
    def run_recipe(breeds, photos_per_class, image_size, backbone, epochs_head, learning_rate_head,
                   unfreeze, epochs_finetune, learning_rate_finetune):
        \"\"\"Load the photos, then steps 1 to 5 of the recipe. Returns the adapted network.\"\"\"
        train_loader, val_loader = load_pets(breeds, photos_per_class, 40, image_size)
        number_of_classes = len(breeds)
        name = f"{backbone} {photos_per_class}/breed {image_size}px"

        # step 1: a pretrained network
        if backbone == "resnet50":
            model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

        # step 2: freeze every parameter
        for param in model.parameters():
            param.requires_grad = False

        # step 3: a new classifier (trainable)
        torch.manual_seed(0)
        model.fc = nn.Linear(model.fc.in_features, number_of_classes)

        # step 4: train only the new classifier
        if epochs_head > 0:
            optimizer = torch.optim.Adam(trainable_parameters(model), lr=learning_rate_head)
            history = P.train(model, train_loader, val_loader, epochs=epochs_head, lr=learning_rate_head,
                              optimizer=optimizer, name=name + ": classifier only")
            P.remember(name + ": classifier only", model, history)

        # step 5: unfreeze the chosen blocks and train on with a smaller learning rate
        if len(unfreeze) > 0 and epochs_finetune > 0:
            for block_name in unfreeze:
                block = getattr(model, block_name)          # e.g. model.layer4
                for param in block.parameters():
                    param.requires_grad = True
            optimizer = torch.optim.Adam(trainable_parameters(model), lr=learning_rate_finetune)
            history = P.train(model, train_loader, val_loader, epochs=epochs_finetune, lr=learning_rate_finetune,
                              optimizer=optimizer, name=name + ": + " + str(unfreeze))
            P.remember(name + ": + " + str(unfreeze), model, history)

        loss, accuracy = P.evaluate(model, val_loader)
        print(f"the network you keep: validation accuracy {accuracy:.1f}% "
              f"({count_trainable(model):,} parameters were trainable in the last step)")
        return model
    """),
    md(r"""
    **Try it.** Change the settings, then run this cell and the next one. The breeds are the starts of the file
    names: cats `Abyssinian`, `Bengal`, `Birman`, `Bombay`, `British_Shorthair`, `Egyptian_Mau`, `Maine_Coon`,
    `Persian`, `Ragdoll`, `Russian_Blue`, `Siamese`, `Sphynx`; dogs `american_bulldog`,
    `american_pit_bull_terrier`, `basset_hound`, `beagle`, `boxer`, `chihuahua`, `english_cocker_spaniel`,
    `english_setter`, `german_shorthaired`, `great_pyrenees`, `havanese`, `japanese_chin`, `keeshond`,
    `leonberger`, `miniature_pinscher`, `newfoundland`, `pomeranian`, `pug`, `saint_bernard`, `samoyed`,
    `scottish_terrier`, `shiba_inu`, `staffordshire_bull_terrier`, `wheaten_terrier`, `yorkshire_terrier`.
    """),
    code("""
    # ---------------- Try it: the data ----------------
    breeds = ["Bombay", "Siamese", "pug", "samoyed", "beagle"]   # 2 to 37 breeds (names above)
    photos_per_class = 60            # training photos per breed (at most 150)
    image_size = 128                 # height = width in pixels, e.g. 96, 128, 160, 224
    # ---------------- Try it: the recipe ----------------
    backbone = "resnet18"            # or "resnet50": bigger and slower (about 100 MB to download the first time)
    epochs_head = 5                  # step 4: epochs that train only the new classifier (0 = no training)
    learning_rate_head = 0.001
    unfreeze = ["layer4"]            # step 5: blocks to unfreeze: [] (none), ["layer4"] or ["layer3", "layer4"]
    epochs_finetune = 5              # step 5: epochs after unfreezing (0 = no step 5)
    learning_rate_finetune = 0.0001  # step 5: usually 10 times smaller than learning_rate_head
    """),
    md(r"""
    **Expected time: about 40 seconds** with the settings above (up to 2 minutes on a slow laptop; more photos,
    bigger images, more unfrozen blocks and ResNet-50 take longer).
    """),
    code("""
    adapted = run_recipe(breeds, photos_per_class, image_size, backbone, epochs_head, learning_rate_head,
                         unfreeze, epochs_finetune, learning_rate_finetune)
    """),
    md(r"""
    **The leaderboard.** Run this cell after every experiment: every version so far, best validation accuracy
    first. With the settings above, the two rows of `run_recipe` repeat steps 4 and 5 exactly.
    """),
    code("""
    P.leaderboard()
    """),
    md(r"""
    ### Guided experiments

    Do them in any order. For each one: change the setting in the **Try it** cell, **predict** what will happen,
    run the two cells, and compare. Then open the fold-out to read what we saw (the other settings as above; your
    numbers can differ in the last digit). Put the setting back afterwards.

    **1. Fewer photos.** `photos_per_class = 20` (100 training photos in total).
    <details><summary>What we saw</summary>

    The new classifier alone reaches 88.0% at its best (epoch 4; 84.0% after epoch 5), against 92.5% with 60
    photos per breed; unfreezing `layer4` then takes it to 95.5%. Even with 20 photos per breed the pretrained
    network is far ahead of the network trained from scratch on 60 per breed (42.0%). This is where transfer
    learning shines: when photos are few.
    </details>

    **2. How much to unfreeze.** `unfreeze = []` (stop after step 4), then `unfreeze = ["layer3", "layer4"]`.
    <details><summary>What we saw</summary>

    With `[]` the recipe ends after the new classifier: 92.5%, as in step 4. With `["layer3", "layer4"]`,
    10 496 005 numbers are trainable (2.1 million more than with `layer4` alone) and each epoch takes longer
    (about 5 s instead of about 4 s here). The best validation accuracy is 98.0%, the same as with `layer4` alone
    (it ends at 98.0% instead of 97.5%: one photo). On this small set, unfreezing more blocks costs time and
    brings no clear gain.
    </details>

    **3. Fine-tuning with too big a learning rate.** `learning_rate_finetune = 0.01` (100 times the default, 10
    times the learning rate of the classifier).
    <details><summary>What we saw</summary>

    The classifier reaches 92.5% as before. Then the first fine-tuning epoch throws the network back to 68.0%,
    with a validation loss of 12.129 (it was 0.281): the big steps wreck what `layer4` learned on ImageNet. It
    partly recovers (85.5%, 82.5%, 91.0%), but jumps around and ends at 87.0%, worse than before unfreezing. With
    the default 0.0001 the same step took it to 98.0%. That is why fine-tuning uses a small learning rate.
    </details>

    **4. Image size.** `image_size = 96`, then `image_size = 160`.
    <details><summary>What we saw</summary>

    At 96 x 96 pixels: 82.5% with the new classifier alone, 90.0% after unfreezing `layer4`: fur and faces lose
    their detail. At 160 x 160: 98.0% with the classifier alone and 99.5% after unfreezing (1 photo wrong out of
    200). But a 160-pixel photo costs 925 million multiply-adds instead of 592 million, and the epochs took 2 to 3
    times as long here (7.1 s and 11.7 s per epoch). Of all the settings, the photo size helps most, closer to
    the 224 pixels the network was trained on; the price is time.
    </details>

    **5. No training at all.** `epochs_head = 0` and `epochs_finetune = 0`.
    <details><summary>What we saw</summary>

    The pretrained network with its random new classifier: 12.5% on the validation photos, the same as before
    step 4, no better than guessing (20%). The feature extractor alone does not know our breeds: the classifier
    has to learn which features mean which breed. That takes seconds, but it cannot be skipped.
    </details>

    **6. Breeds that look alike.** `breeds = ["american_pit_bull_terrier", "staffordshire_bull_terrier",
    "american_bulldog", "boxer", "miniature_pinscher"]`.
    <details><summary>What we saw</summary>

    Five short-haired, muscular dog breeds that people confuse too. The new classifier alone reaches only 66.0%;
    unfreezing `layer4` adds 7 points (73.0% at its best, epoch 1, and 71.5% at the end), while the training
    accuracy reaches 100%: with 60 photos per breed the network learns its training photos by heart rather than
    the fine differences between the breeds. Breeds this close need many more photos per breed, and larger
    photos; the recipe stays the same.
    </details>
    """),

    # ------------------------------------------------------------------ 11
    md(r"""
    ## 11 · What to remember

    * **Transfer learning** reuses a network trained on a huge collection (ImageNet: 1.28 million photos) for your
      own classes. Its feature extractor already knows edges, textures and object parts.
    * The recipe: **freeze, replace the head, train the head, then maybe unfreeze the last block with a small
      learning rate.** In code: `param.requires_grad = False` in a loop, `model.fc = nn.Linear(512, classes)`,
      train, `param.requires_grad = True` for `model.layer4`, a new optimiser, train again.
    * With 300 photos, the network trained from scratch reached 42.0% at best; the pretrained one with a new
      classifier 92.5% after 17 seconds of training, by training only 2 565 of its 11 million numbers, and 98.0%
      after unfreezing `layer4`.
    * Prepare your photos the way the weights expect (ImageNet mean and std), and build a new optimiser after
      changing what is frozen: an optimiser only changes the tensors it was given.
    * To use the result: `model.eval()`, `torch.no_grad()`, softmax for the confidence; `torch.save(model.state_dict(),
      path)` to keep it, and `load_state_dict` into the same network to load it.

    ### Words from this notebook

    | word | meaning |
    |---|---|
    | accuracy | percentage of photos whose highest score is the right class |
    | batch | a small group of photos processed together (here 32) |
    | BatchNorm | a layer that keeps the numbers in a good range; it behaves differently in training and evaluation mode |
    | chance level | the accuracy of guessing: 1 / number of classes (20% for 5 breeds) |
    | class | one of the possible answers (here a breed) |
    | classifier (head) | the last layer: features in, one score per class out |
    | confidence | the probability the network gives its answer |
    | epoch | one pass over all training photos |
    | evaluation mode | `model.eval()`: BatchNorm uses its stored averages, so one photo's answer does not depend on the others |
    | feature extractor | all layers before the classifier: photo in, features out |
    | feature map | the output of one filter of a convolution: bright where its pattern appears |
    | features | the numbers that describe a photo (512 for ResNet-18) |
    | fine-tuning | training (part of) a pretrained feature extractor a little more, with a small learning rate |
    | from scratch | starting from random weights |
    | frozen | a parameter that training does not change (`requires_grad = False`) |
    | global average pooling | averaging each feature map over the whole picture into one number |
    | ImageNet | 1.28 million photos in 1 000 classes; most pretrained image networks learned from it |
    | label | the correct class of a photo, as a number |
    | learning rate | how big each change of the weights is |
    | loss | how wrong the scores are (lower is better) |
    | multiply-adds | the multiplications and additions a network needs for one photo: a measure of its work |
    | normalise | subtract a mean and divide by a spread, per colour channel |
    | optimiser | the method that changes the trainable numbers after each batch (here Adam) |
    | parameters | the numbers a network learns: the weights and biases of its layers, held in tensors |
    | pretrained | already trained by someone else; its weights come as a file |
    | probability | a number between 0 and 1; the 5 probabilities of a photo add up to 1 (softmax) |
    | random seed | the starting point of the random numbers, so results repeat |
    | score | one output number per class; the highest is the answer |
    | tensor | PyTorch's array of numbers |
    | trainable | a parameter that training changes (`requires_grad = True`) |
    | transfer learning | reusing a network trained on one task for another |
    | unfreeze | make frozen parameters trainable again |
    | training / validation set | photos the network learns from / photos kept apart to measure it |
    | weights | the learned numbers of a network (its parameters) |
    """),
]
