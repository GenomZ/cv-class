"""Reference entries for functions first used in D1-02 (training recipe), D1-03 (architectures)
and D1-04 (transfer learning)."""
from . import E

ENTRIES = [
    # ================================================================== D1-02 · training recipe
    # ------------------------------------------------------------------ section 1: data, profile
    E("torch.bincount", "torch.bincount(labels, minlength=0)",
      """Counts how often each non-negative integer occurs in a 1-D integer tensor: entry k of the
      result is the number of elements equal to k, so the result has max(labels) + 1 entries.
      torch.bincount(labels) counts the images per class.""",
      """
      >>> y = torch.tensor([0, 2, 2, 3, 0, 2])       # labels of six images
      >>> torch.bincount(y)                          # two 0s, no 1, three 2s, one 3
      tensor([2, 0, 3, 1])
      >>> torch.bincount(y, minlength=5)             # room for class 4, which never occurs
      tensor([2, 0, 3, 1, 0])
      """,
      note="""The length follows the largest label present: pass minlength=num_classes when you need
      exactly one count per class.""",
      see=["torch.unique"]),

    E("torch.randperm", "torch.randperm(n, generator=None)",
      """A random ordering of the integers 0 .. n-1, each exactly once (an int64 tensor). Its first k
      entries are k distinct random indices: the way to pick a random sample of a dataset without
      repeats. A shuffling DataLoader orders the data the same way.""",
      """
      >>> g = torch.Generator().manual_seed(1)       # own generator: the same draw on every run
      >>> idx = torch.randperm(10, generator=g)
      >>> idx
      tensor([5, 6, 1, 2, 0, 8, 9, 3, 7, 4])
      >>> idx[:4]                                    # 4 distinct random indices out of 10
      tensor([5, 6, 1, 2])
      >>> sorted(idx.tolist()) == list(range(10))    # every index exactly once
      True
      """,
      see=["torch.randint", "torch.Generator"]),

    # torch.stack: first called in D1-02, but D0-00 already explains it (entry in _day0.py)

    E("torchkit.profile", "torchkit.profile(model, input_size)",
      """Course helper. Runs one all-zero dummy image of input_size (C, H, W) through the model (in eval
      mode, without gradients, restoring the previous mode afterwards) and returns a dict
      {"params": number of parameters, "macs": multiply-adds for one image}. Only convolution and
      linear layers are counted; one MAC is one multiplication plus one addition.""",
      """
      >>> conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)    # 8 x 3 x 3 x 3 weights + 8 biases
      >>> torchkit.profile(conv, (3, 32, 32))                 # 32 x 32 x 8 outputs, 3 x 3 x 3 MACs each
      {'params': 224, 'macs': 221184}
      """,
      see=["torchkit.summarize", "torchkit.count_parameters"]),

    E("np.argmin", "np.argmin(a, axis=None)",
      """The position of the smallest value (the first one if several are equal). Without axis the
      array is treated as one flat list; with axis=k you get one position per row / column. Typical use:
      the epoch with the lowest validation loss.""",
      """
      >>> val_loss = [2.3, 1.9, 1.7, 1.8, 2.0]
      >>> int(np.argmin(val_loss))                   # index 2, i.e. the third epoch
      2
      >>> np.argmin(np.array([[3, 1], [0, 5]]), axis=1)     # per row
      array([1, 0])
      """,
      note="""It returns a position, not the value: the value is val_loss[np.argmin(val_loss)] (or
      np.min(val_loss)). The result is a NumPy integer; int(...) turns it into a plain Python int.""",
      see=["np.argmax", "np.argsort"]),

    # ------------------------------------------------------------------ section 2: input pipeline
    E("torch.argsort", "torch.argsort(x, dim=-1, descending=False)",
      """The indices that would sort a tensor: x[torch.argsort(x)] is x in ascending order
      (descending=True: largest first). Indexing several tensors with the same indices reorders
      them together, e.g. images and labels, to sort a dataset by class.""",
      """
      >>> y = torch.tensor([30, 10, 20])
      >>> order = torch.argsort(y)
      >>> order
      tensor([1, 2, 0])
      >>> y[order]
      tensor([10, 20, 30])
      >>> torch.argsort(y, descending=True)
      tensor([0, 2, 1])
      """,
      note="Equal values may come out in any order; pass stable=True to keep their original order.",
      see=["Tensor.argsort", "torch.topk"]),

    E("torch.unique", "torch.unique(x, sorted=True, return_counts=False)",
      """The distinct values of a tensor, sorted. With return_counts=True it also returns how often
      each value occurs. len(torch.unique(y)) counts how many different classes a batch of labels
      contains.""",
      """
      >>> y = torch.tensor([3, 1, 3, 2, 1, 3])
      >>> torch.unique(y)
      tensor([1, 2, 3])
      >>> torch.unique(y, return_counts=True)
      (tensor([1, 2, 3]), tensor([2, 1, 3]))
      >>> len(torch.unique(y))                       # 3 different classes in this batch
      3
      """,
      see=["torch.bincount", "Tensor.unique"]),

    E("np.mean", "np.mean(a, axis=None)",
      """The average of an array or of a plain Python list of numbers; with axis=k only along that axis.
      Arrays also have it as a method: a.mean(axis=k).""",
      """
      >>> float(np.mean([1, 2, 3, 4]))
      2.5
      >>> a = np.array([[1, 2], [3, 4]])
      >>> np.mean(a, axis=0)                         # down each column
      array([2., 3.])
      >>> a.mean(axis=1)                             # along each row
      array([1.5, 3.5])
      """,
      see=["Tensor.mean"]),

    # ------------------------------------------------------------------ section 3: augmentation
    E("torchvision.transforms.v2.Compose", "v2.Compose([transform1, transform2, ...])",
      """Chains transforms into one: calling the result applies them in the listed order, each on the
      output of the previous one. A whole per-image pipeline (resize, augment, convert to float,
      normalise) becomes one callable that a Dataset applies to every sample.""",
      """
      >>> from torchvision.transforms import v2
      >>> tf = v2.Compose([v2.Resize(16), v2.ToDtype(torch.float32, scale=True)])
      >>> img = torch.randint(0, 256, (3, 32, 48), dtype=torch.uint8)
      >>> out = tf(img)
      >>> out.shape, out.dtype, 0 <= float(out.min()) <= float(out.max()) <= 1
      (torch.Size([3, 16, 24]), torch.float32, True)
      """,
      note="""Order matters: Normalize needs a float tensor, so ToDtype(torch.float32, scale=True) must
      come before it; the cheap geometric and colour transforms go first, on uint8.""",
      see=["torchvision.transforms.v2.ToDtype", "torchvision.transforms.v2.Normalize"]),

    E("torchvision.transforms.v2.Resize", "v2.Resize(size, antialias=True)",
      """Resizes an image tensor (C, H, W) or a batch (N, C, H, W). With one number the shorter side
      becomes that size and the aspect ratio is kept; with a pair (h, w) the output is exactly h x w.
      The dtype is kept: uint8 in, uint8 out.""",
      """
      >>> from torchvision.transforms import v2
      >>> img = torch.randint(0, 256, (3, 32, 48), dtype=torch.uint8)    # H 32, W 48
      >>> v2.Resize(16)(img).shape                   # shorter side -> 16, ratio kept
      torch.Size([3, 16, 24])
      >>> v2.Resize((16, 16))(img).shape             # exact size: the image is squashed
      torch.Size([3, 16, 16])
      >>> v2.Resize(16)(img).dtype
      torch.uint8
      """,
      note="""Resize(128) gives a square image only if the input is square; follow it with
      CenterCrop(128) (or use RandomResizedCrop) when every image must have the same size.""",
      see=["torchvision.transforms.v2.CenterCrop", "torchvision.transforms.v2.RandomResizedCrop", "F.interpolate"]),

    E("torchvision.transforms.v2.RandomResizedCrop", "v2.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3/4, 4/3))",
      """Cuts out a random rectangle covering a random fraction of the image area (drawn from scale)
      with a random aspect ratio (drawn from ratio), and resizes it to size x size. The output size is
      always the same; the content is a random zoom into the image. The standard training
      augmentation for classification.""",
      """
      >>> from torchvision.transforms import v2
      >>> img = torch.randint(0, 256, (3, 32, 48), dtype=torch.uint8)
      >>> crop = v2.RandomResizedCrop(24, scale=(0.25, 1.0))      # 25 % to 100 % of the area
      >>> [tuple(crop(img).shape) for _ in range(3)]              # random content, fixed size
      [(3, 24, 24), (3, 24, 24), (3, 24, 24)]
      >>> crop(img).dtype
      torch.uint8
      """,
      note="""The default scale=(0.08, 1.0) can keep as little as 8 % of the image, too aggressive for a
      small dataset; this course uses scale=(0.35, 1.0) or (0.5, 1.0).""",
      see=["torchvision.transforms.v2.CenterCrop", "torchvision.transforms.v2.Resize"]),

    E("torchvision.transforms.v2.RandomHorizontalFlip", "v2.RandomHorizontalFlip(p=0.5)",
      """Mirrors the image left-right with probability p, otherwise returns it unchanged. Valid
      whenever a mirrored image keeps its label: pets yes, letters and digits no.""",
      """
      >>> from torchvision.transforms import v2
      >>> x = torch.tensor([[[1, 2, 3],
      ...                    [4, 5, 6]]], dtype=torch.uint8)      # 1 channel, 2 x 3 pixels
      >>> v2.RandomHorizontalFlip(p=1.0)(x)                       # p=1: always flip
      tensor([[[3, 2, 1],
               [6, 5, 4]]], dtype=torch.uint8)
      """,
      see=["torchvision.transforms.v2.functional.horizontal_flip"]),

    E("torchvision.transforms.v2.RandomRotation", "v2.RandomRotation(degrees, fill=0)",
      """Rotates the image by a random angle drawn uniformly from [-degrees, +degrees] (or from a
      (min, max) pair). The output keeps the input size, so the corners of the rotated image are cut
      off and the uncovered corners are filled with fill (0 = black).""",
      """
      >>> from torchvision.transforms import v2
      >>> white = torch.full((3, 32, 32), 255, dtype=torch.uint8)
      >>> out = v2.RandomRotation(degrees=(30, 30))(white)     # min = max: exactly 30 degrees
      >>> out.shape, out.dtype
      (torch.Size([3, 32, 32]), torch.uint8)
      >>> round(float((out[0] == 0).float().mean()), 2)        # share of black corner pixels
      0.15
      """,
      note="""Test images never have black corners: keep the angle small (10 to 25 degrees) or choose a
      neutral fill.""",
      see=["torchvision.transforms.v2.functional.rotate"]),

    E("torchvision.transforms.v2.ColorJitter", "v2.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)",
      """Randomly changes brightness, contrast, saturation and hue, in a random order. A number b means a
      factor drawn from [1 - b, 1 + b] (for hue: a colour-wheel shift drawn from [-h, h], h at most
      0.5); a (min, max) pair sets the range directly. Shape and dtype stay the same; uint8 values are
      clipped to 0 .. 255.""",
      """
      >>> from torchvision.transforms import v2
      >>> grey = torch.full((3, 4, 4), 100, dtype=torch.uint8)
      >>> v2.ColorJitter(brightness=(1.5, 1.5))(grey).unique()      # factor exactly 1.5
      tensor([150], dtype=torch.uint8)
      >>> light = torch.full((3, 4, 4), 200, dtype=torch.uint8)
      >>> v2.ColorJitter(brightness=(1.5, 1.5))(light).unique()     # 300 is clipped to 255
      tensor([255], dtype=torch.uint8)
      >>> v2.ColorJitter(0.5, 0.5, 0.5, 0.1)(grey).shape            # the notebook's random version
      torch.Size([3, 4, 4])
      """,
      note="""A float image is assumed to be in [0, 1] and is clipped to that range: applied after
      Normalize it destroys the normalised values. Use it on uint8, before ToDtype and Normalize.""",
      see=["torchvision.transforms.v2.RandomGrayscale"]),

    E("torchvision.transforms.v2.RandomGrayscale", "v2.RandomGrayscale(p=0.1)",
      """With probability p converts the image to grey, but keeps 3 (identical) channels so the shape
      does not change; grey = 0.299 R + 0.587 G + 0.114 B. Teaches the network not to rely on colour
      alone.""",
      """
      >>> from torchvision.transforms import v2
      >>> red = torch.zeros(3, 2, 2, dtype=torch.uint8); red[0] = 255     # a pure red image
      >>> out = v2.RandomGrayscale(p=1.0)(red)
      >>> out.shape
      torch.Size([3, 2, 2])
      >>> out[:, 0, 0]                               # all three channels: 0.299 x 255 = 76
      tensor([76, 76, 76], dtype=torch.uint8)
      """,
      see=["torchvision.transforms.v2.ColorJitter"]),

    E("torchvision.transforms.v2.RandAugment", "v2.RandAugment(num_ops=2, magnitude=9)",
      """An automatic augmentation policy: for every image it applies num_ops operations, each picked at
      random from a fixed list of 14 (identity, shear, shift, rotate, brightness, colour, contrast,
      sharpness, posterize, solarize, auto-contrast, equalize), all at the same strength magnitude
      (0 to 30). Two knobs instead of a hand-made list of transforms. Shape and dtype are unchanged.""",
      """
      >>> from torchvision.transforms import v2
      >>> img = torch.randint(0, 256, (3, 32, 48), dtype=torch.uint8)
      >>> out = v2.RandAugment(num_ops=2, magnitude=9)(img)
      >>> out.shape, out.dtype                       # same shape and dtype, changed pixels
      (torch.Size([3, 32, 48]), torch.uint8)
      """,
      note="""Like ColorJitter it treats a float image as values in [0, 1] and clips: apply it to uint8
      images, before ToDtype and Normalize.""",
      see=["torchvision.transforms.v2.TrivialAugmentWide"]),

    E("torchvision.transforms.v2.TrivialAugmentWide", "v2.TrivialAugmentWide()",
      """An automatic augmentation policy with no knobs at all: for every image it picks one of the same 14
      operations as RandAugment at random, with a random strength from wider ranges ("Wide": rotations
      up to 135 degrees, for example). Shape and dtype are unchanged. A strong default for training
      classifiers.""",
      """
      >>> from torchvision.transforms import v2
      >>> img = torch.randint(0, 256, (3, 32, 48), dtype=torch.uint8)
      >>> out = v2.TrivialAugmentWide()(img)
      >>> out.shape, out.dtype
      (torch.Size([3, 32, 48]), torch.uint8)
      """,
      note="Apply it to uint8 images, before ToDtype and Normalize (float images are clipped to [0, 1]).",
      see=["torchvision.transforms.v2.RandAugment"]),

    E("torchvision.transforms.v2.RandomErasing", "v2.RandomErasing(p=0.5, scale=(0.02, 0.33), value=0)",
      """With probability p, fills one random rectangle with value; its area is a random fraction of the
      image area, drawn from scale (2 % to 33 % by default). Hiding part of the object stops the
      network from relying on a single detail. It is usually the last transform, applied to the
      normalised float tensor, where 0 is the average colour.""",
      """
      >>> from torchvision.transforms import v2
      >>> x = torch.ones(3, 32, 32)
      >>> erase = v2.RandomErasing(p=1.0, scale=(0.25, 0.25), ratio=(1, 1), value=0)   # a square, 25 % of the area
      >>> out = erase(x)
      >>> out.shape
      torch.Size([3, 32, 32])
      >>> float((out[0] == 0).float().mean())        # share of erased pixels
      0.25
      """),

    E("torchvision.transforms.v2.ToDtype", "v2.ToDtype(torch.float32, scale=True)",
      """Converts the image to another dtype. With scale=True it also rescales the values to the range
      of the new type: uint8 0 .. 255 becomes float 0.0 .. 1.0. The step between the cheap uint8
      transforms and Normalize.""",
      """
      >>> from torchvision.transforms import v2
      >>> img = torch.tensor([[[0, 128, 255]]], dtype=torch.uint8)
      >>> v2.ToDtype(torch.float32, scale=True)(img)
      tensor([[[0.0000, 0.5020, 1.0000]]])
      >>> v2.ToDtype(torch.float32)(img)             # scale=False: same numbers, now float
      tensor([[[  0., 128., 255.]]])
      """,
      note="""Without scale=True the values stay 0 .. 255, and Normalize with the ImageNet mean / std
      (made for 0 .. 1) then gives values around 1000: the network sees nonsense, and nothing raises
      an error.""",
      see=["torchvision.transforms.v2.Normalize", "Tensor.float"]),

    E("Tensor.is_floating_point", "x.is_floating_point()",
      """True if the tensor holds floating-point numbers (float16, bfloat16, float32, float64), False for
      integer types such as uint8 and int64 and for bool. Used to tell a raw uint8 image (0 .. 255)
      from a converted float image (0 .. 1).""",
      """
      >>> torch.zeros(2, dtype=torch.uint8).is_floating_point()
      False
      >>> torch.zeros(2).is_floating_point()         # the default dtype is float32
      True
      """),

    E("Tensor.clamp", "x.clamp(min=None, max=None)",
      """Limits every value to [min, max]: smaller values become min, larger ones max; either bound can
      be left out. Used to keep an image in [0, 1] before plotting it (plt.imshow clips and warns
      otherwise).""",
      """
      >>> x = torch.tensor([-0.5, 0.2, 1.7])
      >>> x.clamp(0, 1)
      tensor([0.0000, 0.2000, 1.0000])
      >>> x.clamp(min=0)                             # only a lower bound (what ReLU does)
      tensor([0.0000, 0.2000, 1.7000])
      """,
      note="x.clamp(...) returns a new tensor and leaves x as it is; x.clamp_(...) changes x in place.",
      see=["Tensor.clip"]),

    E("torchvision.transforms.v2.functional.rotate", "v2.functional.rotate(img, angle, fill=0)",
      """Rotates an image by exactly angle degrees, counter-clockwise for a positive angle. v2.functional
      holds the fixed, non-random versions of the transforms; this is the one behind RandomRotation.
      The output has the input's size; uncovered corners are filled with fill.""",
      """
      >>> from torchvision.transforms import v2
      >>> x = torch.arange(9, dtype=torch.uint8).reshape(1, 3, 3)
      >>> x
      tensor([[[0, 1, 2],
               [3, 4, 5],
               [6, 7, 8]]], dtype=torch.uint8)
      >>> v2.functional.rotate(x, 180)              # upside down: a "6" becomes a "9"
      tensor([[[8, 7, 6],
               [5, 4, 3],
               [2, 1, 0]]], dtype=torch.uint8)
      >>> v2.functional.rotate(x, 90)               # a quarter turn counter-clockwise
      tensor([[[2, 5, 8],
               [1, 4, 7],
               [0, 3, 6]]], dtype=torch.uint8)
      """,
      see=["torchvision.transforms.v2.RandomRotation"]),

    E("torchvision.transforms.v2.functional.horizontal_flip", "v2.functional.horizontal_flip(img)",
      """Mirrors an image left-right, every time: the fixed version of RandomHorizontalFlip. A flipped
      "b" reads as a "d", which is why the flip is not a valid augmentation for text.""",
      """
      >>> from torchvision.transforms import v2
      >>> x = torch.arange(9, dtype=torch.uint8).reshape(1, 3, 3)
      >>> v2.functional.horizontal_flip(x)          # columns reversed
      tensor([[[2, 1, 0],
               [5, 4, 3],
               [8, 7, 6]]], dtype=torch.uint8)
      """,
      see=["torchvision.transforms.v2.RandomHorizontalFlip", "torchvision.transforms.v2.functional.vertical_flip"]),

    E("torchvision.transforms.v2.functional.vertical_flip", "v2.functional.vertical_flip(img)",
      """Flips an image upside down, every time. As a random augmentation (v2.RandomVerticalFlip) it only
      makes sense when upside-down images occur at test time: satellite or microscope images yes,
      photos of pets no.""",
      """
      >>> from torchvision.transforms import v2
      >>> x = torch.arange(9, dtype=torch.uint8).reshape(1, 3, 3)
      >>> v2.functional.vertical_flip(x)            # rows reversed
      tensor([[[6, 7, 8],
               [3, 4, 5],
               [0, 1, 2]]], dtype=torch.uint8)
      """,
      see=["torchvision.transforms.v2.functional.horizontal_flip"]),

    E("Tensor.squeeze", "x.squeeze(dim=None)",
      """Removes dimensions of size 1: x.squeeze() removes all of them, x.squeeze(d) only dimension d
      (if its size is 1). Typical use: a one-channel image (1, H, W) becomes (H, W) for
      plt.imshow(..., cmap="gray"). NumPy arrays have the same method; D1-02 calls it on the array
      returned by as_np.""",
      """
      >>> torch.zeros(1, 96, 96).squeeze().shape      # one-channel image -> plain 2-D map
      torch.Size([96, 96])
      >>> torch.zeros(1, 3, 1, 5).squeeze(0).shape    # only the first dimension
      torch.Size([3, 1, 5])
      >>> np.zeros((96, 96, 1)).squeeze().shape       # the NumPy version
      (96, 96)
      """,
      note="""squeeze() with no argument also removes a batch dimension of size 1: a batch of one grey
      image (1, 1, H, W) becomes (H, W). Name the dimension when that matters.""",
      see=["Tensor.unsqueeze"]),

    E("np.asarray", "np.asarray(obj)",
      """Converts its argument (a list, a PIL image, a CPU tensor, ...) to a NumPy array; an array is
      returned as it is, without a copy. A grey PIL image becomes a (H, W) uint8 array, a colour one
      (H, W, 3).""",
      """
      >>> from PIL import Image
      >>> im = Image.new("L", (40, 30), 255)          # PIL sizes are (width, height)
      >>> a = np.asarray(im)
      >>> a.shape, a.dtype                            # NumPy shapes are (height, width)
      ((30, 40), dtype('uint8'))
      >>> np.asarray([[1, 2], [3, 4]]).shape
      (2, 2)
      """,
      note="""The array made from a PIL image is read-only, so torch.from_numpy on it prints a "NumPy
      array is not writable" warning (harmless if you do not write into it); np.array(im) makes a
      writable copy.""",
      see=["np.array", "torch.from_numpy"]),

    E("torchvision.transforms.v2.Normalize", "v2.Normalize(mean, std)",
      """Standardises each channel c: out[c] = (x[c] - mean[c]) / std[c]. With the ImageNet statistics
      (torchkit.IMAGENET_MEAN, torchkit.IMAGENET_STD) the pixel values go from [0, 1] to about
      [-2.1, 2.6], centred near 0, which is what ImageNet-trained networks expect. Needs a float
      tensor.""",
      """
      >>> from torchvision.transforms import v2
      >>> norm = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      >>> norm(torch.ones(3, 1, 1)).flatten()                     # a white pixel
      tensor([2.2489, 2.4286, 2.6400])
      >>> norm(torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)).flatten()    # the mean colour -> 0
      tensor([0., 0., 0.])
      """,
      note="""On a uint8 image it raises TypeError ("Input tensor should be a float tensor"): put
      ToDtype(torch.float32, scale=True) before it. To display a normalised image, undo it with
      torchkit.denormalize.""",
      see=["torchvision.transforms.v2.ToDtype", "torchkit.denormalize"]),

    # ------------------------------------------------------------------ Mixup / CutMix
    E("np.linspace", "np.linspace(start, stop, num=50)",
      """num evenly spaced numbers from start to stop, both ends included. Used for plot axes and
      grids.""",
      """
      >>> np.linspace(0, 1, 5)
      array([0.  , 0.25, 0.5 , 0.75, 1.  ])
      """,
      note="""The third argument is the number of points, not the step (np.arange takes a step and
      leaves out stop).""",
      see=["np.arange", "np.geomspace"]),

    E("scipy.stats.beta.pdf", "stats.beta.pdf(x, a, b)",
      """The probability density of the Beta(a, b) distribution at x, a curve over [0, 1]
      (from scipy import stats). Mixup draws its mixing weight lambda from Beta(alpha, alpha); the
      curve shows how strongly it mixes: alpha below 1 puts most weight near 0 and 1 (mostly one
      image), alpha = 1 is flat (every mix equally likely).""",
      """
      >>> from scipy import stats
      >>> stats.beta.pdf([0.05, 0.5, 0.95], 1.0, 1.0)            # alpha = 1: flat
      array([1., 1., 1.])
      >>> stats.beta.pdf([0.05, 0.5, 0.95], 0.2, 0.2).round(2)   # alpha = 0.2: high at the ends
      array([1.2 , 0.32, 1.2 ])
      """,
      see=["np.random.beta"]),

    E("np.random.beta", "np.random.beta(a, b, size=None)",
      """A random number (or an array of size of them) from the Beta(a, b) distribution, always between
      0 and 1. Mixup draws its mixing weight lambda this way. It uses NumPy's global random
      generator, which torch.manual_seed does not touch.""",
      """
      >>> np.random.seed(0)                          # NumPy's own seed
      >>> lam = np.random.beta(0.2, 0.2, size=10000)
      >>> bool(lam.min() >= 0 and lam.max() <= 1)
      True
      >>> round(float(((lam < 0.1) | (lam > 0.9)).mean()), 2)   # alpha 0.2: most draws are near 0 or 1
      0.69
      """,
      note="""For reproducible runs seed NumPy as well (np.random.seed, or a np.random.default_rng
      generator): torch.manual_seed only seeds PyTorch.""",
      see=["scipy.stats.beta.pdf", "np.random.default_rng"]),

    E("F.one_hot", "F.one_hot(labels, num_classes)",
      """Turns integer class labels (B,) into one-hot rows (B, num_classes): 1 in the column of the label,
      0 everywhere else. The result is int64: call .float() before mixing or multiplying it with float
      tensors.""",
      """
      >>> y = torch.tensor([0, 2, 1])
      >>> F.one_hot(y, num_classes=4)
      tensor([[1, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 1, 0, 0]])
      >>> y1 = F.one_hot(y, 4).float()
      >>> 0.7 * y1[0] + 0.3 * y1[1]                  # a Mixup target: 70 % class 0, 30 % class 2
      tensor([0.7000, 0.0000, 0.3000, 0.0000])
      """,
      note="""Without num_classes the width is max(labels) + 1, so a batch that happens to lack the last
      classes gets too few columns. labels must be an integer (int64) tensor.""",
      see=["Tensor.scatter_"]),

    E("torchvision.transforms.v2.CutMix", "v2.CutMix(num_classes, alpha=1.0)",
      """A batch-level augmentation: it takes a whole batch, images (B, C, H, W) and integer labels (B,),
      and returns mixed images and soft labels (B, num_classes). Every image gets a rectangle pasted
      in from its neighbour in the batch (image i receives part of image i - 1), and its label becomes
      the two labels weighted by area. The box size comes from a lambda drawn from Beta(alpha, alpha),
      once per batch. v2.MixUp(num_classes, alpha) is the blending version.""",
      """
      >>> from torchvision.transforms import v2
      >>> _ = torch.manual_seed(2)
      >>> xb, yb = torch.rand(4, 3, 8, 8), torch.tensor([0, 1, 2, 3])
      >>> x_mix, y_mix = v2.CutMix(num_classes=5, alpha=1.0)(xb, yb)
      >>> x_mix.shape, y_mix.shape
      (torch.Size([4, 3, 8, 8]), torch.Size([4, 5]))
      >>> y_mix[0]                         # image 0: 75 % itself, a box of 25 % from image 3
      tensor([0.7500, 0.0000, 0.0000, 0.2500, 0.0000])
      >>> (x_mix[0] != xb[0]).any(0).float().mean()      # the pasted box covers 25 % of the pixels
      tensor(0.2500)
      >>> y_mix.sum(1)                     # every soft label still sums to 1
      tensor([1., 1., 1., 1.])
      """,
      note="""The loss must accept soft targets (F.cross_entropy does, for a (B, C) float target) and
      accuracy needs the original integer labels. Apply it to batches, after the DataLoader, not in the
      per-image transform.""",
      see=["F.cross_entropy", "F.one_hot"]),

    E("torch.topk", "torch.topk(x, k, dim=-1, largest=True)",
      """The k largest values along a dimension (default: the last), sorted from the largest, together with
      their positions: the result has .values and .indices. Used for an ImageNet model's top-3
      guesses, for top-5 accuracy, and for the largest entries of a confusion matrix.""",
      """
      >>> p = torch.tensor([0.10, 0.60, 0.05, 0.25])
      >>> top = torch.topk(p, 2)
      >>> top.values
      tensor([0.6000, 0.2500])
      >>> top.indices                                # class 1, then class 3
      tensor([1, 3])
      >>> torch.topk(torch.tensor([[1., 5., 3.], [9., 0., 2.]]), 1, dim=1).indices   # per row
      tensor([[1],
              [0]])
      """,
      see=["Tensor.argsort", "Tensor.argmax"]),

    E("torch.roll", "torch.roll(x, shifts, dims=None)",
      """Shifts the elements along a dimension by shifts positions; what falls off one end comes back at
      the other. torch.roll(torch.arange(n), 1) is a fixed permutation that pairs every element with
      its neighbour: a deterministic alternative to torch.randperm for pairing images in Mixup.""",
      """
      >>> torch.roll(torch.arange(5), 1)
      tensor([4, 0, 1, 2, 3])
      >>> torch.roll(torch.arange(5), -1)
      tensor([1, 2, 3, 4, 0])
      >>> torch.roll(torch.tensor([[1, 2, 3], [4, 5, 6]]), 1, dims=1)     # along each row
      tensor([[3, 1, 2],
              [6, 4, 5]])
      """,
      note="Without dims the tensor is flattened, shifted and put back into its shape.",
      see=["torch.randperm"]),

    # ------------------------------------------------------------------ section 4: regularisation
    E("nn.BatchNorm1d", "nn.BatchNorm1d(num_features, momentum=0.1)",
      """A layer that normalises each of num_features features over the batch, for inputs (B, num_features).
      In train mode it uses the mean and variance of the current batch and updates its running averages
      running_mean and running_var (new = (1 - momentum) x old + momentum x batch value); in eval mode
      it uses those running averages instead. A learnable scale (weight) and shift (bias) per feature
      follow.""",
      """
      >>> bn = nn.BatchNorm1d(1, momentum=0.1)        # one feature
      >>> x = torch.tensor([[1.0], [3.0]])            # a batch of two samples, mean 2
      >>> bn(x).detach().flatten()                    # train mode (the default): batch statistics
      tensor([-1.0000,  1.0000])
      >>> bn.running_mean, bn.running_var             # 0.9 x 0 + 0.1 x 2 and 0.9 x 1 + 0.1 x 2
      (tensor([0.2000]), tensor([1.1000]))
      >>> _ = bn.eval()
      >>> bn(x).detach().flatten()                    # eval mode: the running averages
      tensor([0.7628, 2.6697])
      """,
      note="""In train mode the output for one sample depends on the other samples of the batch: call
      model.eval() before validation and inference. A training batch of a single sample raises
      "Expected more than 1 value per channel when training".""",
      see=["nn.BatchNorm2d", "Module.eval", "Module.train"]),

    E("Module.train", "model.train(mode=True)",
      """Puts the model and all its layers into training mode: dropout is active and BatchNorm uses batch
      statistics (and updates its running averages). model.train(False) is the same as model.eval().
      It does not train anything or change any weight; it sets the .training flag of every layer and
      returns the model.""",
      """
      >>> net = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
      >>> _ = net.eval()
      >>> net.training, net[1].training
      (False, False)
      >>> _ = net.train()
      >>> net.training, net[1].training
      (True, True)
      """,
      note="""Call model.train() at the start of every training epoch and model.eval() before evaluating;
      torchkit.fit and torchkit.evaluate do both for you.""",
      see=["Module.eval"]),

    E("Module.eval", "model.eval()",
      """Puts the model and all its layers into evaluation mode: dropout is switched off and BatchNorm uses
      the running averages collected during training, so a prediction no longer depends on the rest of
      the batch. Returns the model, so get_model(...).eval() works in one line.""",
      """
      >>> drop = nn.Dropout(p=0.5)
      >>> x = torch.ones(1, 6)
      >>> _ = torch.manual_seed(0)
      >>> drop(x)                         # train mode (the default): random zeros, the rest x 2
      tensor([[0., 0., 2., 0., 0., 0.]])
      >>> _ = drop.eval()
      >>> drop(x)                         # eval mode: unchanged
      tensor([[1., 1., 1., 1., 1., 1.]])
      """,
      note="""eval() does not stop gradient tracking: wrap inference in torch.no_grad() as well. The mode
      stays until you change it: call model.train() before training again.""",
      see=["Module.train", "torch.no_grad"]),

    E("Tensor.ravel", "x.ravel()",
      """Returns all values as one flat 1-D tensor, row after row: the same as x.flatten() or
      x.reshape(-1). NumPy arrays have the same method; D1-02 calls it on an array (after .numpy())
      to get a flat list of values for a histogram.""",
      """
      >>> torch.tensor([[1, 2], [3, 4]]).ravel()
      tensor([1, 2, 3, 4])
      >>> np.array([[1, 2], [3, 4]]).ravel()
      array([1, 2, 3, 4])
      """,
      see=["Tensor.flatten", "Tensor.reshape"]),

    E("torch.cat", "torch.cat(tensors, dim=0)",
      """Joins a list of tensors along an existing dimension dim; all other dimensions must match.
      dim=0 appends samples (collect per-batch results into one tensor); dim=1 on (N, C, H, W) tensors
      puts channels side by side, as an Inception block does with its branches.""",
      """
      >>> a, b = torch.zeros(2, 3), torch.ones(1, 3)
      >>> torch.cat([a, b]).shape                    # 2 + 1 rows
      torch.Size([3, 3])
      >>> f1, f2 = torch.zeros(1, 64, 8, 8), torch.zeros(1, 32, 8, 8)
      >>> torch.cat([f1, f2], dim=1).shape           # 64 + 32 channels
      torch.Size([1, 96, 8, 8])
      """,
      note="""It takes a list (or tuple) of tensors: torch.cat([a, b]), not torch.cat(a, b). To create a
      new dimension instead, use torch.stack.""",
      see=["torch.stack"]),

    E("Tensor.sqrt", "x.sqrt()",
      """The square root of every element. E.g. BatchNorm's running variance becomes a standard
      deviation.""",
      """
      >>> torch.tensor([4.0, 9.0, 2.0]).sqrt()
      tensor([2.0000, 3.0000, 1.4142])
      """,
      note="The square root of a negative number is nan, not an error."),

    E("Tensor.abs", "x.abs()",
      """The absolute value of every element. (a - b).abs().max() is the largest difference between two
      tensors: the usual check that two results agree.""",
      """
      >>> a = torch.tensor([1.0, -2.0, 3.0])
      >>> a.abs()
      tensor([1., 2., 3.])
      >>> (a - torch.tensor([1.0, -2.5, 3.1])).abs().max()
      tensor(0.5000)
      """,
      see=["torch.allclose"]),

    # ------------------------------------------------------------------ label smoothing
    E("F.log_softmax", "F.log_softmax(logits, dim)",
      """The logarithm of the softmax along dim: log(exp(x_k) / sum_j exp(x_j)) for every entry, computed
      without forming the huge exponentials (numerically stable). It turns raw scores (logits) into
      log-probabilities; their exp() sums to 1 along dim.""",
      """
      >>> logp = F.log_softmax(torch.tensor([[2.0, 1.0, 0.1]]), dim=-1)
      >>> logp
      tensor([[-0.4170, -1.4170, -2.3170]])
      >>> logp.exp().sum()                            # probabilities again
      tensor(1.0000)
      >>> F.log_softmax(torch.zeros(3), dim=0)        # equal scores: log(1/3) each
      tensor([-1.0986, -1.0986, -1.0986])
      """,
      note="""F.cross_entropy already applies log_softmax: give it the raw logits. Computing
      torch.log(x.softmax(dim)) in two steps can give -inf for very negative scores.""",
      see=["Tensor.softmax", "F.cross_entropy"]),

    E("torch.full_like", "torch.full_like(x, value)",
      """A new tensor with the same shape, dtype and device as x, every element set to value, e.g. a
      smoothed target q that starts as eps / C everywhere.""",
      """
      >>> logp = torch.zeros(2, 4)
      >>> torch.full_like(logp, 0.025)
      tensor([[0.0250, 0.0250, 0.0250, 0.0250],
              [0.0250, 0.0250, 0.0250, 0.0250]])
      >>> torch.full_like(torch.zeros(3, dtype=torch.long), 7)     # the dtype follows x
      tensor([7, 7, 7])
      """,
      note="""The dtype is copied from x: torch.full_like(integer_tensor, 0.5) silently gives zeros. Pass
      dtype=torch.float32 to change it.""",
      see=["torch.zeros", "torch.ones"]),

    E("Tensor.scatter_", "x.scatter_(dim, index, value)",
      """In place, writes value into x at the positions listed in index along dimension dim; for a 2-D x
      and dim=1 that is x[i, index[i, j]] = value. The classic use: put a value in the column of each
      sample's label (one-hot or smoothed targets). index must be an int64 tensor with as many
      dimensions as x, so a label vector (B,) becomes (B, 1) with unsqueeze(1).""",
      """
      >>> q = torch.zeros(2, 4)
      >>> labels = torch.tensor([2, 0])
      >>> q.scatter_(1, labels.unsqueeze(1), 0.9)    # row 0: column 2, row 1: column 0
      tensor([[0.0000, 0.0000, 0.9000, 0.0000],
              [0.9000, 0.0000, 0.0000, 0.0000]])
      >>> q[0]                                       # q itself was changed
      tensor([0.0000, 0.0000, 0.9000, 0.0000])
      """,
      note="""The trailing underscore means in place: q changes (and is also returned). q.scatter(...)
      without the underscore returns a changed copy.""",
      see=["F.one_hot", "torch.full_like"]),

    E("np.full", "np.full(shape, value)",
      """An array of the given shape (a number or a tuple) with every element set to value.""",
      """
      >>> np.full(4, 0.25)
      array([0.25, 0.25, 0.25, 0.25])
      >>> np.full((2, 3), 7)
      array([[7, 7, 7],
             [7, 7, 7]])
      """,
      see=["np.ones", "np.zeros"]),

    E("np.log", "np.log(x)",
      """The natural logarithm (base e) of every element. ln C is the cross-entropy loss of a classifier
      that guesses uniformly among C classes: ln 37 = 3.61.""",
      """
      >>> np.log([1, np.e, np.e ** 2])
      array([0., 1., 2.])
      >>> round(float(np.log(37)), 4)
      3.6109
      """,
      note="np.log(0) is -inf (with a RuntimeWarning) and the log of a negative number is nan.",
      see=["np.log2"]),

    # ------------------------------------------------------------------ section 5: optimisers, schedules
    E("np.meshgrid", "X, Y = np.meshgrid(xs, ys)",
      """From a vector of x values and a vector of y values, builds two 2-D arrays X and Y with the x- and
      the y-coordinate of every point of the grid, both of shape (len(ys), len(xs)). A formula
      evaluated on X and Y gives its value on the whole grid, ready for ax.contour or plt.imshow.""",
      """
      >>> X, Y = np.meshgrid([0, 1, 2], [10, 20])
      >>> X
      array([[0, 1, 2],
             [0, 1, 2]])
      >>> Y
      array([[10, 10, 10],
             [20, 20, 20]])
      >>> X + Y                                      # f(x, y) = x + y at every grid point
      array([[10, 11, 12],
             [20, 21, 22]])
      """,
      see=["torch.meshgrid"]),

    E("np.geomspace", "np.geomspace(start, stop, num=50)",
      """num numbers from start to stop with a constant ratio between neighbours (evenly spaced on a log
      scale), both ends included. Good for contour levels or learning rates that span several orders
      of magnitude.""",
      """
      >>> np.geomspace(1, 1000, 4)
      array([   1.,   10.,  100., 1000.])
      """,
      see=["np.linspace", "np.logspace"]),

    E("Tensor.clone", "x.clone()",
      """A copy of the tensor with its own memory: later changes to either one do not affect the other.
      Plain assignment (b = a) copies nothing, it gives the same tensor a second name.
      w.detach().clone() is the usual way to take a snapshot of a tensor that keeps being trained.""",
      """
      >>> a = torch.zeros(3)
      >>> b = a                                      # the same tensor, a second name
      >>> c = a.clone()                              # an independent copy
      >>> a[0] = 5
      >>> b
      tensor([5., 0., 0.])
      >>> c
      tensor([0., 0., 0.])
      """,
      note="""clone() keeps the autograd connection (gradients flow through it); add .detach() when you
      only want the values.""",
      see=["Tensor.detach"]),

    E("torch.optim.lr_scheduler.MultiStepLR", "torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)",
      """A learning-rate scheduler: multiplies the learning rate by gamma each time the number of
      scheduler.step() calls reaches one of the milestones. This is "step decay": constant, then 10x
      lower, then 10x lower again. Call scheduler.step() after optimizer.step(), once per epoch or once
      per batch: the milestones count those calls.""",
      """
      >>> opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
      >>> sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2, 4], gamma=0.1)
      >>> lrs = []
      >>> for step in range(6):
      ...     lrs.append(round(opt.param_groups[0]["lr"], 4))
      ...     opt.step(); sched.step()
      >>> lrs
      [1.0, 1.0, 0.1, 0.1, 0.01, 0.01]
      """,
      see=["torch.optim.lr_scheduler.CosineAnnealingLR", "Optimizer.param_groups"]),

    E("torch.optim.lr_scheduler.CosineAnnealingLR", "torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0)",
      """A learning-rate scheduler that lowers the learning rate from its initial value to eta_min along
      half a cosine wave over T_max calls of scheduler.step(): slowly at first, fastest in the middle,
      slowly again at the end. Call scheduler.step() after optimizer.step().""",
      """
      >>> opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
      >>> sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
      >>> lrs = []
      >>> for step in range(5):
      ...     lrs.append(round(opt.param_groups[0]["lr"], 4))
      ...     opt.step(); sched.step()
      >>> lrs
      [1.0, 0.8536, 0.5, 0.1464, 0.0]
      """,
      note="""Set T_max to the total number of scheduler.step() calls (epochs, or epochs x batches when
      you step every batch): after T_max steps the cosine climbs back up.""",
      see=["torch.optim.lr_scheduler.OneCycleLR", "torch.optim.lr_scheduler.MultiStepLR", "Optimizer.param_groups"]),

    E("np.ones", "np.ones(shape)",
      """An array of the given shape (a number or a tuple) filled with 1.0, e.g. a constant learning-rate
      schedule: learning rate / peak = 1 at every step.""",
      """
      >>> np.ones(3)
      array([1., 1., 1.])
      >>> np.ones((2, 3)).shape
      (2, 3)
      """,
      see=["np.zeros", "np.full"]),

    E("Optimizer.param_groups", "optimizer.param_groups",
      """An attribute, not a call: the optimizer's list of parameter groups, one dict per group with its
      "params" and its own settings ("lr", "weight_decay", ...). Schedulers work by changing "lr" in
      these dicts, so optimizer.param_groups[0]["lr"] is the current learning rate. Passing a list of
      dicts to the optimizer creates several groups, each with its own learning rate (D1-04);
      extra keys such as "name" are kept.""",
      """
      >>> w_old, w_new = torch.zeros(3, requires_grad=True), torch.zeros(2, requires_grad=True)
      >>> opt = torch.optim.AdamW([{"params": [w_old], "lr": 1e-4, "name": "layer4"},  # its own LR
      ...                          {"params": [w_new]}], lr=1e-3)                     # the default LR
      >>> [g["lr"] for g in opt.param_groups]
      [0.0001, 0.001]
      >>> opt.param_groups[0]["name"], opt.param_groups[1]["weight_decay"]    # AdamW's default decay
      ('layer4', 0.01)
      """,
      see=["torch.optim.AdamW", "torch.optim.lr_scheduler.CosineAnnealingLR"]),

    E("torchkit.lr_range_test",
      "lrs, losses = torchkit.lr_range_test(model, loader, loss_fn, optimizer_cls, lr_min=1e-6, lr_max=1.0, steps=100)",
      """Course helper for Leslie Smith's learning-rate range test. It creates
      optimizer_cls(model.parameters(), lr=lr_min) and trains for up to steps batches from loader while
      the learning rate grows exponentially from lr_min to lr_max, recording an exponentially smoothed
      training loss. It stops early once the loss is inf / nan or more than 4x the best loss so far, and
      restores the model's original weights at the end. Returns two equally long lists, the learning
      rates and the smoothed losses; a good peak learning rate is about 10x below the one at the
      minimum of the losses.""",
      """
      >>> net = nn.Linear(4, 3); before = net.weight.clone()
      >>> batches = [(torch.randn(8, 4), torch.randint(0, 3, (8,)))] * 5      # any iterable of (x, y)
      >>> lrs, losses = torchkit.lr_range_test(net, batches, F.cross_entropy, torch.optim.SGD,
      ...                                      lr_min=1e-3, lr_max=1e-1, steps=5)
      >>> [f"{lr:.0e}" for lr in lrs], len(losses)
      (['1e-03', '3e-03', '1e-02', '3e-02', '1e-01'], 5)
      >>> torch.equal(net.weight, before)                                     # weights restored
      True
      """,
      note="""Extra keyword arguments go to the optimizer (e.g. weight_decay=0.0). The model is left in
      train mode.""",
      see=["torch.optim.lr_scheduler.OneCycleLR"]),

    # ------------------------------------------------------------------ section 6: debugging
    E("Tensor.mul_", "x.mul_(k)",
      """Multiplies the tensor by k (a number or a tensor) in place and returns it; a trailing underscore
      always marks an in-place method. On model weights it must run inside torch.no_grad(), as in D1-02,
      which scales the last layer by 30 to imitate a badly initialised network.""",
      """
      >>> w = torch.ones(3)
      >>> w.mul_(30)
      tensor([30., 30., 30.])
      >>> w                                          # w itself was changed
      tensor([30., 30., 30.])
      >>> lin = nn.Linear(2, 2)
      >>> with torch.no_grad():
      ...     _ = lin.weight.mul_(0)
      >>> torch.equal(lin.weight.detach(), torch.zeros(2, 2))
      True
      """,
      note="""Outside torch.no_grad() an in-place change of a parameter raises "a leaf Variable that
      requires grad is being used in an in-place operation".""",
      see=["Tensor.clone", "torch.no_grad"]),

    E("F.cross_entropy", "F.cross_entropy(logits, target, label_smoothing=0.0)",
      """The classification loss: logits (B, C) are raw scores, target is either the class indices (B,)
      as int64 or a (B, C) float tensor of probabilities (soft targets, e.g. from Mixup or CutMix). It
      applies log_softmax itself and returns the mean of -log p(correct class) over the batch.
      label_smoothing=eps trains against (1 - eps) on the true class plus eps / C spread over all
      classes.""",
      """
      >>> logits = torch.zeros(4, 37)                # knows nothing: uniform softmax
      >>> y = torch.tensor([0, 5, 9, 36])
      >>> F.cross_entropy(logits, y)                 # = ln 37, the loss at initialisation
      tensor(3.6109)
      >>> good = torch.tensor([[4.0, 0.0, 0.0]])
      >>> F.cross_entropy(good, torch.tensor([0])), F.cross_entropy(good, torch.tensor([0]), label_smoothing=0.1)
      (tensor(0.0360), tensor(0.3026))
      >>> F.cross_entropy(good, torch.tensor([[0.7, 0.3, 0.0]]))   # a soft (Mixup) target
      tensor(1.2360)
      """,
      note="""Pass the raw logits, not the softmax output (softmax would be applied twice), and integer
      labels, not one-hot (a one-hot target must be float and is then treated as probabilities).
      nn.CrossEntropyLoss() is the same loss as a layer object.""",
      see=["nn.CrossEntropyLoss", "F.log_softmax"]),

    # ------------------------------------------------------------------ section 8: checkpoints
    E("torch.save", "torch.save(obj, path)",
      """Writes a Python object that contains tensors (a tensor, a state_dict, a dict of several
      state_dicts and numbers, ...) to a file path or an open file object, in Python's pickle format.
      The usual checkpoint is a dict: {"model": model.state_dict(), "optimizer": opt.state_dict(),
      "epoch": epoch}.""",
      """
      >>> import io
      >>> buf = io.BytesIO()               # an in-memory file; in practice a path such as "ckpt.pt"
      >>> torch.save({"w": torch.tensor([1.0, 2.0]), "epoch": 3}, buf)
      >>> _ = buf.seek(0)                  # rewind before reading
      >>> torch.load(buf)
      {'w': tensor([1., 2.]), 'epoch': 3}
      """,
      note="""Save state_dict()s rather than whole model objects: a pickled model only loads again where
      its class definition can be imported, unchanged, under the same name.""",
      see=["torch.load", "Module.state_dict", "torchkit.save_checkpoint"]),

    E("torch.load", "torch.load(path, map_location=None, weights_only=True)",
      """Reads back what torch.save wrote. By default (weights_only=True) it only accepts tensors, numbers,
      strings and lists / tuples / dicts of them: a safety measure, because a pickle file can run code.
      weights_only=False loads any object, for files you trust. map_location="cpu" puts tensors that
      were saved on a GPU onto the CPU.""",
      """
      >>> import io
      >>> buf = io.BytesIO()
      >>> torch.save({"model": nn.Linear(2, 1).state_dict(), "epoch": 7}, buf)
      >>> _ = buf.seek(0)
      >>> ck = torch.load(buf, map_location="cpu")
      >>> sorted(ck), ck["epoch"], list(ck["model"])
      (['epoch', 'model'], 7, ['weight', 'bias'])
      """,
      note="""A file that also holds other objects (a NumPy array, an instance of your own class) fails with
      "Weights only load failed" under the default; pass weights_only=False only for files you made
      yourself.""",
      see=["torch.save", "Module.load_state_dict", "torchkit.load_checkpoint"]),

    E("Module.state_dict", "model.state_dict()",
      """An ordered dict from names to tensors holding everything that defines the model's state: all
      parameters (weights, biases) and buffers (BatchNorm's running_mean and running_var). Names follow
      the module structure, e.g. "layer1.0.conv1.weight". Optimizers and schedulers have a state_dict()
      too (momentum, Adam's moment estimates, the step counter): to resume training, save all three.""",
      """
      >>> net = nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2))
      >>> {k: tuple(v.shape) for k, v in net.state_dict().items()}
      {'0.weight': (2, 3), '0.bias': (2,), '1.weight': (2,), '1.bias': (2,), '1.running_mean': (2,), '1.running_var': (2,), '1.num_batches_tracked': ()}
      """,
      note="""The tensors share memory with the model, so further training changes them: to keep a
      snapshot in memory use copy.deepcopy(model.state_dict()); torch.save writes the current values to
      disk.""",
      see=["Module.load_state_dict", "torch.save", "Module.named_parameters"]),

    E("Module.load_state_dict", "model.load_state_dict(state_dict, strict=True)",
      """Copies the tensors of a state_dict into an existing model's parameters and buffers, matched by
      name; names and shapes must agree. Returns a report of missing and unexpected names. Optimizers and
      schedulers have load_state_dict too.""",
      """
      >>> a, b = nn.Linear(3, 2), nn.Linear(3, 2)    # two different random initialisations
      >>> torch.equal(a.weight, b.weight)
      False
      >>> b.load_state_dict(a.state_dict())
      <All keys matched successfully>
      >>> torch.equal(a.weight, b.weight)
      True
      """,
      note="""strict=False skips missing or extra names (handy after replacing a head), but a shape
      mismatch is still an error. It fills a model you have already built; it does not build one.""",
      see=["Module.state_dict", "torch.load", "torchkit.load_checkpoint"]),

    E("torch.get_rng_state", "torch.get_rng_state()",
      """A snapshot of PyTorch's CPU random-number generator, as a uint8 tensor. Restoring it later with
      torch.set_rng_state makes the random numbers that follow (dropout masks, augmentations, shuffles)
      exactly the same as after the snapshot: what you need to resume a run exactly.""",
      """
      >>> state = torch.get_rng_state()
      >>> a = torch.rand(3)
      >>> torch.set_rng_state(state)                 # rewind the generator
      >>> torch.equal(a, torch.rand(3))
      True
      >>> state.dtype
      torch.uint8
      """,
      note="""This is the CPU generator only. GPUs have their own (torch.cuda.get_rng_state()), a
      torch.Generator given to a DataLoader has its own (g.get_state() / g.set_state(), as in
      D1-02), and NumPy and Python's random module are separate as well.""",
      see=["torch.set_rng_state", "torch.manual_seed", "torch.Generator"]),

    E("torch.set_rng_state", "torch.set_rng_state(state)",
      """Restores PyTorch's CPU random-number generator to a state saved earlier with
      torch.get_rng_state(), so the random numbers drawn afterwards repeat exactly.""",
      """
      >>> saved = torch.get_rng_state()
      >>> first = torch.randint(0, 100, (4,))
      >>> torch.set_rng_state(saved)
      >>> torch.equal(first, torch.randint(0, 100, (4,)))     # the same four numbers again
      True
      """,
      see=["torch.get_rng_state", "torch.manual_seed"]),

    # ================================================================== D1-03 · architectures
    E("torchvision.models.vgg16", "torchvision.models.vgg16(weights=None)",
      """Builds a VGG-16 network: 13 3x3 convolutions in 5 blocks with max-pooling (model.features), then
      three fully-connected layers 4096 - 4096 - 1000 (model.classifier); 138 M parameters, most of
      them in the first fc layer. weights=None gives random weights and downloads nothing;
      weights="DEFAULT" downloads the ImageNet weights. Input (N, 3, 224, 224), output (N, 1000) logits.""",
      """
      >>> vgg = torchvision.models.vgg16(weights=None)
      >>> sum(p.numel() for p in vgg.parameters())
      138357544
      >>> sum(isinstance(m, nn.Conv2d) for m in vgg.features)
      13
      >>> vgg.classifier[0]                          # the first fc layer: 512 x 7 x 7 inputs
      Linear(in_features=25088, out_features=4096, bias=True)
      """,
      see=["torchvision.models.resnet18", "torchvision.models.get_model"]),

    E("Tensor.numel", "x.numel()",
      """The total number of elements of a tensor: the product of its shape. Summed over
      model.parameters() it gives the number of parameters of a model.""",
      """
      >>> torch.zeros(64, 3, 7, 7).numel()           # the weight of a 7x7 conv, 3 -> 64 channels
      9408
      >>> lin = nn.Linear(512, 37)
      >>> sum(p.numel() for p in lin.parameters())   # 512 x 37 weights + 37 biases
      18981
      """,
      see=["torchkit.count_parameters", "Tensor.shape"]),

    E("nn.BatchNorm2d", "nn.BatchNorm2d(num_channels)",
      """BatchNorm for image tensors (N, C, H, W): normalises each of the C channels with the mean and
      variance over the batch and all positions (N x H x W values), then applies a learnable scale and
      shift per channel (2 x C parameters). Train and eval mode behave as in nn.BatchNorm1d: batch
      statistics in train mode, running averages in eval mode. It usually follows a convolution, which
      then needs no bias (bias=False): BatchNorm's shift does that job.""",
      """
      >>> bn = nn.BatchNorm2d(8)
      >>> x = 3 * torch.randn(4, 8, 5, 5) + 2           # 8 channels with mean ~2, std ~3
      >>> y = bn(x).detach()                            # train mode; detach: just the values
      >>> y.shape
      torch.Size([4, 8, 5, 5])
      >>> round(abs(float(y.mean())), 3), round(float(y.std()), 2)
      (0.0, 1.0)
      >>> sum(p.numel() for p in bn.parameters())       # a scale and a shift per channel
      16
      """,
      note="Forget model.eval() at test time and every prediction depends on the other images in its batch.",
      see=["nn.BatchNorm1d", "Module.eval"]),

    E("nn.Identity", "nn.Identity()",
      """A layer that returns its input unchanged. Two uses: the "do nothing" shortcut of a residual block
      whose shape does not change, and removing part of a network without rewriting it:
      model.fc = nn.Identity() turns a classifier into a feature extractor.""",
      """
      >>> x = torch.randn(2, 5)
      >>> torch.equal(nn.Identity()(x), x)
      True
      >>> net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 10))
      >>> net[2] = nn.Identity()                     # drop the last layer
      >>> net(torch.zeros(1, 4)).shape               # the network now outputs its 8 features
      torch.Size([1, 8])
      """),

    E("torch.randint", "torch.randint(low, high, size)",
      """Random integers drawn uniformly from low (included) to high (excluded), in a tensor of the given
      size (a tuple), e.g. random class labels for a dummy batch.""",
      """
      >>> _ = torch.manual_seed(0)
      >>> torch.randint(0, 10, (5,))                 # five labels from 0 .. 9
      tensor([4, 9, 3, 0, 3])
      >>> torch.randint(0, 2, (2, 3)).shape
      torch.Size([2, 3])
      """,
      note="""high is never drawn: randint(0, 10, ...) gives 0 .. 9. The size must be a tuple, (16,), not
      16.""",
      see=["torch.randperm", "torch.rand"]),

    E("np.maximum", "np.maximum(a, b)",
      """The element-wise maximum of two arrays, or of an array and a number. np.maximum(values, 1e-30)
      replaces values below 1e-30 (including 0) so a log-scale plot does not break on log(0). Not to be
      confused with np.max, which returns the single largest value.""",
      """
      >>> np.maximum([0.0, 0.5, 3.0], 1e-30)
      array([1.e-30, 5.e-01, 3.e+00])
      >>> np.maximum([1, 5, 2], [4, 3, 6])
      array([4, 5, 6])
      """,
      see=["np.minimum", "np.clip"]),

    E("torchvision.models.resnet18", "torchvision.models.resnet18(weights=None, num_classes=1000)",
      """Builds a ResNet-18: a stem (conv1 7x7 stride 2, bn1, relu, maxpool), four stages layer1 .. layer4
      of two basic residual blocks each (64, 128, 256, 512 channels), global average pooling (avgpool)
      and a linear head fc (512 -> num_classes); 11.7 M parameters. weights=None gives random weights and
      downloads nothing; weights="DEFAULT" downloads the ImageNet weights. Thanks to the global pooling it
      accepts other input sizes than 224 px. Every torchvision model works this way: resnet50,
      mobilenet_v3_large, efficientnet_b0, convnext_tiny, ...""",
      """
      >>> r18 = torchvision.models.resnet18(weights=None)
      >>> [name for name, _ in r18.named_children()]
      ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
      >>> r18.fc
      Linear(in_features=512, out_features=1000, bias=True)
      >>> r18(torch.zeros(2, 3, 64, 64)).shape       # a 64 px input works too
      torch.Size([2, 1000])
      """,
      see=["torchvision.models.get_model", "torchvision.models.vgg16"]),

    E("nn.AdaptiveAvgPool2d", "nn.AdaptiveAvgPool2d(output_size)",
      """Average-pools a (N, C, H, W) tensor down to a fixed output size, whatever H and W are: you give
      the output size, it chooses the windows. With output size 1 it is global average pooling: every
      channel is averaged over all positions, (N, C, H, W) -> (N, C, 1, 1); .flatten(1) then gives (N, C)
      for a linear layer.""",
      """
      >>> x = torch.arange(8.0).view(1, 2, 2, 2)              # 2 channels of 2 x 2
      >>> nn.AdaptiveAvgPool2d(1)(x).flatten(1)               # the mean of each channel
      tensor([[1.5000, 5.5000]])
      >>> nn.AdaptiveAvgPool2d(1)(torch.zeros(2, 128, 8, 8)).shape
      torch.Size([2, 128, 1, 1])
      >>> nn.AdaptiveAvgPool2d(1)(torch.zeros(2, 128, 5, 7)).shape    # any input size
      torch.Size([2, 128, 1, 1])
      """,
      see=["nn.MaxPool2d", "Tensor.flatten"]),

    # ================================================================== D1-04 · transfer learning
    E("torchvision.models.get_model_weights", "get_model_weights(name)",
      """Returns the weights enum of a torchvision model, by name: get_model_weights("resnet18") is
      ResNet18_Weights, and its .DEFAULT member the best available ImageNet weights. A weights object
      carries metadata (.meta: the 1 000 category names, the parameter count, the ImageNet accuracy) and
      the preprocessing the model was trained with (.transforms()). Nothing is downloaded until a model
      is built with it.""",
      """
      >>> from torchvision.models import get_model_weights
      >>> w = get_model_weights("resnet18").DEFAULT
      >>> w
      ResNet18_Weights.IMAGENET1K_V1
      >>> w.meta["num_params"], len(w.meta["categories"]), w.meta["categories"][:2]
      (11689512, 1000, ['tench', 'goldfish'])
      >>> w.meta["_metrics"]["ImageNet-1K"]["acc@1"]           # top-1 accuracy in %
      69.758
      >>> list(get_model_weights("resnet50"))                  # all available versions
      [ResNet50_Weights.IMAGENET1K_V1, ResNet50_Weights.IMAGENET1K_V2]
      """,
      note="""Preprocess your images the way the weights expect: print(w.transforms()) shows the resize,
      crop, mean and std.""",
      see=["torchvision.models.get_model"]),

    E("torchvision.models.get_model", "get_model(name, weights=None, **kwargs)",
      """Builds any torchvision model by its name. With weights=None the weights are random and nothing is
      downloaded; with a weights object (or "DEFAULT") the pretrained weights are downloaded once into
      the torch cache folder and loaded. Other keyword arguments go to the model's constructor, e.g.
      num_classes.""",
      """
      >>> from torchvision.models import get_model
      >>> m = get_model("resnet18", weights=None, num_classes=37)
      >>> m.fc
      Linear(in_features=512, out_features=37, bias=True)
      >>> m(torch.zeros(1, 3, 64, 64)).shape
      torch.Size([1, 37])
      """,
      note="""A freshly built model is in train mode: call .eval() before predicting. With pretrained
      weights num_classes must stay 1000 (anything else raises ValueError); replace the head afterwards
      instead: model.fc = nn.Linear(model.fc.in_features, 37).""",
      see=["torchvision.models.get_model_weights", "torchvision.models.resnet18", "timm.create_model"]),

    E("Tensor.softmax", "logits.softmax(dim)",
      """Turns scores (logits) into probabilities along dimension dim: exp(x) / sum(exp(x)). Every value
      ends up between 0 and 1 and they sum to 1 along dim; the order of the scores is kept, so the argmax
      does not change.""",
      """
      >>> logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
      >>> logits.softmax(1)                          # one probability vector per row
      tensor([[0.0900, 0.2447, 0.6652],
              [0.3333, 0.3333, 0.3333]])
      >>> logits.softmax(1).sum(1)
      tensor([1., 1.])
      """,
      note="""Always give dim (1 for a batch of logits (N, C)). For training, pass the raw logits to
      F.cross_entropy, not the softmax output.""",
      see=["F.log_softmax", "torch.topk"]),

    E("Module.named_parameters", "model.named_parameters()",
      """Iterates over the model's parameters as (name, tensor) pairs. The name is the attribute path
      inside the model, e.g. "layer3.1.conv2.weight" or "fc.bias", so its first part tells which stage a
      parameter belongs to: the basis for freezing stages or giving them their own learning rates.""",
      """
      >>> net = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))
      >>> for name, p in net.named_parameters():
      ...     print(name, tuple(p.shape), p.requires_grad)
      0.weight (2, 3) True
      0.bias (2,) True
      2.weight (1, 2) True
      2.bias (1,) True
      """,
      note="""Layers without parameters (ReLU, pooling) do not appear. BatchNorm's running statistics are
      buffers, not parameters: they are in state_dict(), not here.""",
      see=["Module.parameters", "Module.state_dict", "Tensor.requires_grad_"]),

    E("Tensor.requires_grad_", "p.requires_grad_(flag=True)",
      """Sets, in place, whether autograd computes a gradient for this tensor, and returns the tensor;
      p.requires_grad_(False) does the same as p.requires_grad = False. Freezing a layer means this for
      all its parameters: backward() then computes no .grad for them, so the optimizer leaves them
      unchanged.""",
      """
      >>> lin = nn.Linear(4, 2)                      # 8 weights + 2 biases
      >>> for p in lin.parameters():
      ...     _ = p.requires_grad_(False)            # freeze
      >>> torchkit.count_parameters(lin), torchkit.count_parameters(lin, trainable_only=True)
      (10, 0)
      >>> lin.weight.requires_grad = True            # the attribute form: unfreeze the weight
      >>> torchkit.count_parameters(lin, trainable_only=True)
      8
      """,
      note="""Freezing does not freeze BatchNorm's running statistics: in train mode they keep updating.
      To keep them, leave those BatchNorm layers in eval mode (day1.keep_bn_eval).""",
      see=["Module.named_parameters", "torchkit.count_parameters"]),

    E("torchkit.save_checkpoint", "torchkit.save_checkpoint(model, path, **extra)",
      """Course helper. Saves {"model": model.state_dict(), **extra} to path with torch.save, creating
      missing folders, and returns the path. Extra keyword arguments (epoch=..., history=...,
      logits_val=...) are stored next to the weights; NumPy numbers and arrays among them are turned into
      plain Python numbers and lists first, so that the file loads with the safe default
      weights_only=True. Read it back with torchkit.load_checkpoint(model, path), which fills the model
      and returns the extras as a dict.""",
      see=["torchkit.load_checkpoint", "torch.save"]),

    E("np.polyfit", "np.polyfit(x, y, deg)",
      """Least-squares fit of a polynomial of degree deg through the points (x, y); returns the
      coefficients, highest power first. With deg=1 that is a straight line: (slope, intercept).
      Fitting accuracy against log2(training-set size) makes the slope the accuracy points gained per
      doubling of the data.""",
      """
      >>> slope, intercept = np.polyfit([0, 1, 2, 3], [1, 3, 5, 7], 1)    # y = 2x + 1
      >>> round(float(slope), 6), round(float(intercept), 6)
      (2.0, 1.0)
      >>> sizes, acc = np.array([100, 200, 400, 800]), np.array([60, 65, 70, 75])
      >>> round(float(np.polyfit(np.log2(sizes), acc, 1)[0]), 6)          # +5 points per doubling
      5.0
      """,
      see=["np.log2"]),

    E("np.log2", "np.log2(x)",
      """The base-2 logarithm of every element. On a log2 axis every doubling is one step: log2 of
      1, 2, 4, 8 is 0, 1, 2, 3.""",
      """
      >>> np.log2([1, 2, 8, 1024])
      array([ 0.,  1.,  3., 10.])
      """,
      see=["np.log"]),

    E("Module.register_forward_hook", "handle = module.register_forward_hook(fn)",
      """Makes the layer call fn(module, input, output) right after each of its forward passes: the way to
      read an intermediate activation (a stage's feature maps) without changing the model's code. input
      is a tuple of the layer's inputs. Returns a handle; handle.remove() takes the hook off again.""",
      """
      >>> net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
      >>> acts = {}
      >>> def keep(module, inp, out):
      ...     acts["relu"] = out.detach()
      >>> handle = net[1].register_forward_hook(keep)
      >>> net(torch.randn(5, 4)).shape, acts["relu"].shape     # the output, and the captured activation
      (torch.Size([5, 2]), torch.Size([5, 8]))
      >>> handle.remove()
      """,
      note="""Remove hooks when done: otherwise they run on every later forward pass and keep tensors
      alive. If fn returns something other than None, that value replaces the layer's output (dict
      assignment returns None, which is why D1-04's one-line lambda is safe).""",
      see=["Tensor.register_hook"]),

    E("Tensor.argsort", "x.argsort(dim=-1, descending=False)",
      """The method form of torch.argsort: the indices that would sort the tensor along dim.
      x.argsort(descending=True)[:k] are the positions of the k largest values, the same as
      torch.topk(x, k).indices.""",
      """
      >>> act = torch.tensor([0.2, 0.9, 0.1, 0.5])   # mean activation of 4 channels
      >>> act.argsort(descending=True)
      tensor([1, 3, 0, 2])
      >>> act.argsort(descending=True)[:2]           # the 2 most active channels
      tensor([1, 3])
      """,
      see=["torch.argsort", "torch.topk"]),

    E("torch.pca_lowrank", "U, S, V = torch.pca_lowrank(A, q=6, center=True)",
      """Principal component analysis of a data matrix A (N samples x D features). The q columns of V
      (D, q) are the directions in which the data varies most, strongest first; S holds the matching
      singular values (larger = more variance along that direction).
      Projecting the centred data, (A - A.mean(0)) @ V, gives q coordinates per sample, e.g. 2 for a
      scatter plot. It uses a fast randomised algorithm meant for q much smaller than D.""",
      """
      >>> _ = torch.manual_seed(0)
      >>> t = torch.randn(100, 1)
      >>> A = torch.cat([t, 2 * t, 0.01 * torch.randn(100, 1)], dim=1)   # 100 points near the line (1, 2, 0)
      >>> U, S, V = torch.pca_lowrank(A, q=2)
      >>> V.shape
      torch.Size([3, 2])
      >>> V[:, 0].abs().round(decimals=2)            # first direction: (1, 2, 0) / sqrt(5)
      tensor([0.4500, 0.8900, 0.0000])
      >>> ((A - A.mean(0)) @ V).shape                # 2 coordinates per point
      torch.Size([100, 2])
      """,
      note="""center=True (the default) subtracts the column means first; D1-04 centres the data itself and
      passes center=False. The sign of each column of V is arbitrary: a mirrored scatter plot
      is the same result.""",
      see=["torch.linalg.svd"]),

    E("Tensor.register_hook", "handle = t.register_hook(fn)",
      """Makes autograd call fn(grad) when backward() computes the gradient of the loss with respect to
      this tensor. Parameters keep their gradient in .grad anyway; intermediate results such as a
      layer's output do not, so a hook is the way to read the gradient of an activation, which Grad-CAM
      needs. Registered inside a forward hook, it captures a layer's output and its gradient. Returns a
      handle with .remove().""",
      """
      >>> x = torch.tensor([1.0, 2.0], requires_grad=True)
      >>> y = 3 * x                                  # an intermediate tensor
      >>> grads = {}
      >>> _ = y.register_hook(lambda g: grads.__setitem__("y", g))
      >>> (y ** 2).sum().backward()                  # d(sum y^2)/dy = 2y
      >>> grads["y"]
      tensor([ 6., 12.])
      """,
      note="""The tensor must be part of a graph that requires gradients (not computed under
      torch.no_grad()). If fn returns a tensor, it replaces the gradient that flows on.""",
      see=["Module.register_forward_hook", "Tensor.backward"]),

    E("F.interpolate", "F.interpolate(x, size=None, scale_factor=None, mode=\"nearest\", align_corners=None)",
      """Resizes a batch of maps (N, C, H, W) to size=(h, w), or by scale_factor. mode="nearest" repeats
      pixels, mode="bilinear" blends neighbouring pixels for a smooth result (with align_corners=False,
      the usual convention for images). E.g. to upsample a coarse 5 x 5 or 7 x 7 Grad-CAM map to the
      image size.""",
      """
      >>> x = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])   # (1, 1, 2, 2)
      >>> F.interpolate(x, size=(4, 4), mode="nearest")
      tensor([[[[0., 0., 1., 1.],
                [0., 0., 1., 1.],
                [2., 2., 3., 3.],
                [2., 2., 3., 3.]]]])
      >>> F.interpolate(x, size=(4, 4), mode="bilinear", align_corners=False)[0, 0, 0]   # first row
      tensor([0.0000, 0.2500, 0.7500, 1.0000])
      >>> F.interpolate(torch.zeros(1, 1, 5, 5), size=(160, 160), mode="bilinear", align_corners=False).shape
      torch.Size([1, 1, 160, 160])
      """,
      note="""The input needs batch and channel dimensions: a single (h, w) map must become (1, 1, h, w)
      first (cam[None, None]), otherwise it raises an error about spatial dimensions. size is (height,
      width), in the order of x.shape[-2:].""",
      see=["torchvision.transforms.v2.Resize"]),

    E("np.argsort", "np.argsort(a)",
      """The indices that would sort an array in ascending order; [::-1] reverses them for descending.
      a[np.argsort(a)] is sorted, and indexing other arrays with the same order keeps them aligned
      (class names, colours).""",
      """
      >>> recall = np.array([0.9, 0.2, 0.5])
      >>> order = np.argsort(recall)
      >>> order
      array([1, 2, 0])
      >>> recall[order]
      array([0.2, 0.5, 0.9])
      >>> order[::-1]                                # best first
      array([0, 2, 1])
      """,
      see=["torch.argsort", "np.argmin"]),

    E("Tensor.fill_diagonal_", "x.fill_diagonal_(value)",
      """In place, sets the main diagonal of a 2-D tensor (the entries [i, i]) to value and returns the
      tensor. In a confusion matrix the diagonal holds the correct predictions; zeroing it leaves only
      the mistakes, whose largest entries are the most confused pairs.""",
      """
      >>> conf = torch.tensor([[5, 1], [2, 7]])      # rows: true class, columns: predicted
      >>> off = conf.clone().fill_diagonal_(0)
      >>> off                                        # only the mistakes
      tensor([[0, 1],
              [2, 0]])
      >>> conf                                       # unchanged, thanks to clone()
      tensor([[5, 1],
              [2, 7]])
      """,
      note="In place (trailing underscore): without .clone() the confusion matrix itself would lose its diagonal.",
      see=["Tensor.clone"]),

    E("Tensor.any", "x.any(dim=None)",
      """True if at least one element is True (non-zero): over the whole tensor, or along dim (one answer
      per row with dim=1). Top-k accuracy: compare the k guesses of each image with its label, then ask
      whether any of them matched.""",
      """
      >>> top3 = torch.tensor([[3, 7, 1], [0, 2, 4]])     # 3 guessed classes for each of 2 images
      >>> labels = torch.tensor([7, 5])
      >>> hits = top3 == labels[:, None]
      >>> hits.any(1)                                # is the label among the guesses?
      tensor([ True, False])
      >>> hits.any(1).float().mean()                 # top-3 accuracy
      tensor(0.5000)
      """,
      see=["Tensor.all", "torch.topk"]),

    E("timm.create_model", "timm.create_model(name, pretrained=False, num_classes=None, **kwargs)",
      """Builds any of the 1 000+ models of the timm library by name (import timm). pretrained=True
      downloads the weights from the Hugging Face Hub on first use (internet needed); pretrained=False
      gives random weights. num_classes=k replaces the head with a new one for k classes, num_classes=0
      removes it so the model returns its pooled feature vector; ViTs also take img_size.
      timm.list_models("*dinov2*") finds names.""",
      """
      >>> import timm
      >>> m = timm.create_model("resnet18", pretrained=False, num_classes=0)   # no head
      >>> m(torch.zeros(1, 3, 64, 64)).shape                                  # the 512-d features
      torch.Size([1, 512])
      >>> timm.create_model("resnet18", pretrained=False, num_classes=37)(torch.zeros(1, 3, 64, 64)).shape
      torch.Size([1, 37])
      """,
      note="""Like torchvision models, a timm model is created in train mode: call .eval() before
      extracting features.""",
      see=["timm.data.resolve_data_config", "torchvision.models.get_model"]),

    E("timm.data.resolve_data_config", "timm.data.resolve_data_config({}, model=model)",
      """The preprocessing a timm model expects, read from its pretrained configuration, as a dict:
      input_size, interpolation, mean, std, crop_pct (the fraction of the resized image the centre crop
      keeps) and crop_mode. D1-04 takes mean and std from it for its Normalize.""",
      """
      >>> import timm
      >>> cfg = timm.data.resolve_data_config({}, model=timm.create_model("resnet18", pretrained=False))
      >>> sorted(cfg)
      ['crop_mode', 'crop_pct', 'input_size', 'interpolation', 'mean', 'std']
      >>> cfg["mean"], cfg["std"]
      ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
      """,
      note="""input_size is the size of the pretrained configuration (518 px for the DINOv2 model of
      D1-04), not the img_size passed to create_model.""",
      see=["timm.create_model", "torchvision.transforms.v2.Normalize"]),

    E("torchvision.transforms.v2.CenterCrop", "v2.CenterCrop(size)",
      """Cuts out the central size x size region (or (h, w)): no resizing, no randomness. The evaluation
      counterpart of RandomResizedCrop: Resize(s) followed by CenterCrop(s) gives a square image whose
      side is s.""",
      """
      >>> from torchvision.transforms import v2
      >>> img = torch.randint(0, 256, (3, 32, 48), dtype=torch.uint8)
      >>> v2.CenterCrop(32)(img).shape               # 8 columns cut off on each side
      torch.Size([3, 32, 32])
      >>> torch.equal(v2.CenterCrop(32)(img), img[:, :, 8:40])
      True
      >>> v2.Compose([v2.Resize(16), v2.CenterCrop(16)])(img).shape
      torch.Size([3, 16, 16])
      """,
      see=["torchvision.transforms.v2.Resize", "torchvision.transforms.v2.RandomResizedCrop"]),
]
