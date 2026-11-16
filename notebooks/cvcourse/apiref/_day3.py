"""Reference entries for functions first used on Day 3 (D3-01 .. D3-04: segmentation, U-Net, Mask R-CNN, foundation models)."""
from . import E

ENTRIES = [
    # ------------------------------------------------------------------ D3-01 · segmentation basics
    E("scipy.ndimage.label", "labels, n = ndimage.label(mask)",
      """Finds the connected regions of a binary mask and numbers them. Returns an integer array of the
      mask's shape (0 = background, 1..n = one number per region) and the number of regions n. Pixels
      are connected through their 4 side neighbours, not through diagonals. D3-01 uses it to split a
      pet mask into one region per cat.""",
      """
      >>> from scipy import ndimage
      >>> m = np.array([[1, 1, 0, 0],
      ...               [0, 0, 0, 1],
      ...               [1, 0, 1, 1]])
      >>> labels, n = ndimage.label(m)
      >>> n
      3
      >>> labels
      array([[1, 1, 0, 0],
             [0, 0, 0, 2],
             [3, 0, 2, 2]], dtype=int32)
      """,
      note="""Touching objects become one region: connected components separate instances only when they
      do not touch (two people side by side need Mask R-CNN, D3-03). structure=np.ones((3, 3)) also
      connects diagonal neighbours.""",
      see=["scipy.ndimage.binary_opening", "np.unique"]),

    E("scipy.ndimage.binary_opening", "ndimage.binary_opening(mask, iterations=1)",
      """Removes small specks and thin strands from a binary mask: it first shrinks every True region
      (erosion, see binary_erosion), then grows what is left back by the same amount. Parts narrower
      than about 2 * iterations + 1 pixels disappear; larger regions keep their shape, with slightly
      rounded corners. D3-01 opens the pet mask before ndimage.label so that stray pixels do not
      count as extra instances.""",
      """
      >>> from scipy import ndimage
      >>> m = np.zeros((8, 8), bool)
      >>> m[2:6, 1:5] = True; m[0, 7] = True           # a 4x4 blob and one stray pixel
      >>> ndimage.label(m)[1]                          # number of regions
      2
      >>> ndimage.label(ndimage.binary_opening(m))[1]  # the stray pixel is gone
      1
      """,
      see=["scipy.ndimage.binary_erosion", "scipy.ndimage.label"]),

    E("np.unique", "np.unique(a, return_counts=False)",
      """The distinct values of an array, sorted (the array is flattened first). The quickest check of
      what a label mask contains: a raw Pet trimap gives [1 2 3]. With return_counts=True it also
      returns how often each value occurs, e.g. the number of pixels per class.""",
      """
      >>> mask = np.array([[0, 0, 2],
      ...                  [1, 2, 2]])
      >>> np.unique(mask)
      array([0, 1, 2])
      >>> values, counts = np.unique(mask, return_counts=True)
      >>> counts                                       # two 0s, one 1, three 2s
      array([2, 1, 3])
      """,
      see=["torch.unique", "np.bincount"]),

    E("np.dstack", "np.dstack([a, b, ...])",
      """Stacks arrays along a third axis ("depth"): two (H, W) arrays give (H, W, 2); an (H, W, 3) RGB
      image plus an (H, W) array give (H, W, 4). D3-01 uses it to add an alpha (transparency) channel to
      a colour overlay, because plt.imshow draws an (H, W, 4) array as RGBA.""",
      """
      >>> rgb = np.zeros((2, 3, 3))                    # H=2, W=3, RGB
      >>> alpha = np.ones((2, 3))                      # one opacity value per pixel
      >>> np.dstack([rgb, alpha]).shape                # RGBA
      (2, 3, 4)
      >>> np.dstack([np.zeros((2, 3)), np.ones((2, 3))]).shape
      (2, 3, 2)
      """,
      see=["np.stack", "np.concatenate"]),

    E("np.shape", "np.shape(a)",
      """The shape of anything array-like, as a tuple: it works on NumPy arrays and on nested Python
      lists, which have no .shape attribute. The RLE exercise uses it so that rle_encode accepts a list
      as well as an array.""",
      """
      >>> np.shape([[0, 1, 1], [0, 1, 0]])             # a nested list: 2 rows, 3 columns
      (2, 3)
      >>> np.shape(np.zeros((4, 5)))
      (4, 5)
      """,
      see=["Tensor.shape"]),

    E("np.zeros_like", "np.zeros_like(a, dtype=None)",
      """A new array of zeros with the same shape and dtype as a (dtype= overrides the type). The easy
      way to start an empty mask or an accumulator that matches an existing image.""",
      """
      >>> gt = np.array([[0, 1], [1, 1]])
      >>> np.zeros_like(gt)
      array([[0, 0],
             [0, 0]])
      >>> np.zeros_like(gt, bool)                      # same shape, boolean
      array([[False, False],
             [False, False]])
      """,
      note="""It copies the dtype too: zeros_like of an integer mask is an integer array, so writing 0.5
      into it stores 0.""",
      see=["np.zeros", "torch.zeros_like"]),

    E("np.nanmean", "np.nanmean(a)",
      """The mean of an array that skips NaN ("not a number") values. mIoU is averaged this way: a class
      absent from both prediction and target has an undefined IoU (NaN) and is left out, instead of
      turning the whole mean into NaN.""",
      """
      >>> iou = np.array([0.9, np.nan, 0.5])           # class 1 appears nowhere
      >>> float(np.mean(iou))
      nan
      >>> float(np.nanmean(iou))                       # (0.9 + 0.5) / 2
      0.7
      """,
      see=["np.isnan", "np.mean"]),

    E("np.logical_and", "np.logical_and(a, b)",
      """Element-wise AND of two boolean (or 0/1) arrays of the same shape: True where both are true. On
      a predicted and a true mask it is their intersection, and .sum() counts its pixels; the same as
      a & b for boolean arrays.""",
      """
      >>> pred = np.array([[1, 1, 0], [0, 1, 0]], bool)
      >>> gt   = np.array([[0, 1, 1], [0, 1, 0]], bool)
      >>> np.logical_and(pred, gt)
      array([[False,  True, False],
             [False,  True, False]])
      >>> int(np.logical_and(pred, gt).sum())          # intersection: 2 pixels
      2
      """,
      note="""It treats every non-zero value as True. a & b on integer arrays is a bitwise AND instead:
      2 & 1 is 0, so convert masks to bool first.""",
      see=["np.logical_or"]),

    E("scipy.ndimage.binary_erosion", "ndimage.binary_erosion(mask, iterations=1)",
      """Shrinks the True regions of a binary mask: a pixel stays True only if it and its 4 side
      neighbours are all True, so each iteration peels one pixel off every border. D3-01 uses it to make
      a deliberately too small "prediction"; mask & ~binary_erosion(mask) is the mask's outline.""",
      """
      >>> from scipy import ndimage
      >>> m = np.zeros((5, 5), bool); m[1:4, 1:4] = True       # a 3x3 square
      >>> ndimage.binary_erosion(m).astype(int)                # only the centre survives
      array([[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]])
      >>> int((m & ~ndimage.binary_erosion(m)).sum())          # the outline: 8 pixels
      8
      """,
      note="Pixels outside the array count as False, so a mask that touches the image border also shrinks from there.",
      see=["scipy.ndimage.binary_opening"]),

    E("np.roll", "np.roll(a, shift, axis=None)",
      """Shifts the elements of an array by shift positions along an axis; what falls off one end comes
      back at the other (it wraps around). With tuples, np.roll(mask, (4, 6), (0, 1)) moves a mask 4
      pixels down and 6 pixels to the right.""",
      """
      >>> np.roll(np.array([1, 2, 3, 4, 5]), 2)
      array([4, 5, 1, 2, 3])
      >>> m = np.array([[1, 0, 0],
      ...               [0, 0, 0]])
      >>> np.roll(m, (1, 1), axis=(0, 1))              # one row down, one column right
      array([[0, 0, 0],
             [0, 1, 0]])
      """,
      note="Without axis the array is flattened, shifted and reshaped back, which mixes rows."),

    E("np.log1p", "np.log1p(x)",
      """log(1 + x), element-wise (natural logarithm), accurate even for tiny x. Handy to display counts
      that range from 0 to thousands, such as a confusion matrix: log1p(0) is 0, so empty cells stay 0
      instead of becoming -inf.""",
      """
      >>> cm = np.array([[0, 9], [99, 9999]])
      >>> np.log1p(cm).round(2)                        # log(1), log(10), log(100), log(10000)
      array([[0.  , 2.3 ],
             [4.61, 9.21]])
      """,
      see=["np.log10", "torch.log"]),

    E("np.isnan", "np.isnan(a)",
      """True where a value is NaN ("not a number", e.g. the result of 0 / 0), element-wise. You need it
      because NaN is not equal to anything, not even to NaN, so a comparison with == never finds one. In
      D3-01 the IoU of a class that appears nowhere is NaN.""",
      """
      >>> iou = np.array([0.9, np.nan, 0.5])
      >>> np.isnan(iou)
      array([False,  True, False])
      >>> iou == np.nan                                # never works
      array([False, False, False])
      """,
      see=["np.nanmean", "np.nan_to_num"]),

    E("np.errstate", 'with np.errstate(divide="ignore", invalid="ignore"):',
      """Used in a with statement: inside the block, NumPy changes how it reacts to floating-point problems.
      "ignore" silences the RuntimeWarnings for division by zero (1 / 0 gives inf) and invalid results
      (0 / 0 gives nan). Use it where such values are expected and handled afterwards, like the IoU of a
      class that appears nowhere.""",
      """
      >>> a = np.array([1., 0.]); b = np.array([0., 0.])
      >>> with np.errstate(divide="ignore", invalid="ignore"):
      ...     r = a / b                                # no warning printed
      >>> r
      array([inf, nan])
      """,
      note="It only hides the warning: the result still contains inf and nan, and you must deal with them (np.where, np.nanmean).",
      see=["np.where", "np.nanmean"]),

    E("torch.where", "torch.where(cond, a, b)  /  torch.where(cond)",
      """With three arguments: an element-wise choice, a where cond is True and b where it is False (a and
      b broadcast, so they can be scalars or single-value tensors). With only cond: the indices of the
      True elements, one tensor per dimension (rows, then columns), like np.nonzero.""",
      """
      >>> t = torch.tensor([1., 0., 0., 1.])                         # 1 = object pixel
      >>> torch.where(t > 0, torch.tensor(0.3), torch.tensor(0.1))   # 0.3 on the object, 0.1 elsewhere
      tensor([0.3000, 0.1000, 0.1000, 0.3000])
      >>> ys, xs = torch.where(torch.tensor([[0, 1], [1, 1]]) > 0)
      >>> ys, xs
      (tensor([0, 1, 1]), tensor([1, 0, 1]))
      """,
      note="""Gradients flow only into the chosen elements, but a NaN produced in the branch that was not
      chosen (log(0), 0 / 0) still turns the gradient into NaN: make both branches safe.""",
      see=["np.where", "np.nonzero"]),

    E("F.binary_cross_entropy", "F.binary_cross_entropy(prob, target)",
      """Binary cross-entropy for probabilities: prob (values in (0, 1)) and target (0s and 1s, as floats)
      have the same shape; the result is the mean of -[t log p + (1 - t) log(1 - p)] over all elements.
      D3-01 uses it to ask how big the gradient is with respect to each probability. To train a model,
      use F.binary_cross_entropy_with_logits on the raw outputs instead.""",
      """
      >>> p = torch.tensor([0.9, 0.2, 0.5])
      >>> t = torch.tensor([1., 0., 1.])
      >>> F.binary_cross_entropy(p, t)                 # mean of -log 0.9, -log 0.8, -log 0.5
      tensor(0.3406)
      """,
      note="""It wants probabilities, not logits (values outside [0, 1] raise an error), and a float target.
      A sigmoid followed by this loss is less stable than F.binary_cross_entropy_with_logits.""",
      see=["F.binary_cross_entropy_with_logits", "torch.sigmoid"]),

    E("torch.log", "torch.log(x)",
      """Natural logarithm, element-wise. log(0) is -inf and the log of a negative number is nan, so a
      loss written by hand (the focal loss in D3-01) must keep its probabilities strictly above 0.""",
      """
      >>> torch.log(torch.tensor([1.0, 0.5, 0.0]))
      tensor([ 0.0000, -0.6931,    -inf])
      """,
      note="""-torch.log(torch.sigmoid(z)) is unstable for very negative z; the built-in losses
      (F.binary_cross_entropy_with_logits, F.cross_entropy) compute the same quantity safely.""",
      see=["np.log1p", "F.binary_cross_entropy"]),

    E("np.logspace", "np.logspace(start, stop, num)",
      """num numbers from 10**start to 10**stop, evenly spaced on a log scale: each one is the previous
      times a constant factor. Use it to sweep a quantity over several orders of magnitude, such as
      object sizes from 4 to 2048 pixels.""",
      """
      >>> np.logspace(0, 3, 4)                         # 10**0 ... 10**3
      array([   1.,   10.,  100., 1000.])
      >>> np.logspace(np.log10(4), np.log10(64), 5)    # from 4 to 64: times 2 at each step
      array([ 4.,  8., 16., 32., 64.])
      """,
      note="""start and stop are exponents, not values: np.logspace(4, 2048, 30) means 10**4 to 10**2048.
      np.geomspace(4, 2048, 30) takes the end values directly.""",
      see=["np.linspace", "np.log10"]),

    E("np.log10", "np.log10(x)",
      """Base-10 logarithm, element-wise: log10(1000) is 3. D3-01 uses it to turn sizes into the exponents
      np.logspace expects.""",
      """
      >>> np.log10(np.array([1., 10., 1000.]))
      array([0., 1., 3.])
      """,
      see=["np.logspace", "np.log1p"]),

    E("F.adaptive_avg_pool2d", "F.adaptive_avg_pool2d(x, output_size)",
      """Average-pools a batch (N, C, H, W) down to the output size (h, w) you ask for; the window size
      and stride are worked out for you. D3-01 shrinks a mask by a factor s with it, to imitate a
      backbone of stride s. With output_size=1 it is global average pooling: one number per channel.""",
      """
      >>> x = torch.arange(16.).reshape(1, 1, 4, 4)
      >>> F.adaptive_avg_pool2d(x, (2, 2))             # the mean of each 2x2 block
      tensor([[[[ 2.5000,  4.5000],
                [10.5000, 12.5000]]]])
      >>> F.adaptive_avg_pool2d(torch.zeros(8, 512, 7, 7), 1).shape
      torch.Size([8, 512, 1, 1])
      """,
      see=["nn.AdaptiveAvgPool2d", "F.interpolate"]),

    E("F.conv_transpose2d", "F.conv_transpose2d(x, weight, stride=1, padding=0, groups=1)",
      """Transposed convolution as a function: every input pixel adds a copy of the kernel, multiplied by
      the pixel's value, into the output at position (i * stride, j * stride); where copies overlap they
      add up. x is (N, C_in, H, W) and weight is (C_in, C_out / groups, kH, kW); the output side is
      (H - 1) * stride + kH - 2 * padding.""",
      """
      >>> x = torch.tensor([[[[1., 2.],
      ...                     [3., 4.]]]])             # (1, 1, 2, 2)
      >>> k = torch.ones(1, 1, 3, 3)                   # all-ones 3x3 kernel
      >>> F.conv_transpose2d(x, k, stride=2)           # (2 - 1) * 2 + 3 = 5: four stamps, overlaps add
      tensor([[[[ 1.,  1.,  3.,  2.,  2.],
                [ 1.,  1.,  3.,  2.,  2.],
                [ 4.,  4., 10.,  6.,  6.],
                [ 3.,  3.,  7.,  4.,  4.],
                [ 3.,  3.,  7.,  4.,  4.]]]])
      """,
      note="""The weight is (in_channels, out_channels, kH, kW), the reverse of F.conv2d. Despite the old
      name "deconvolution" it does not undo a convolution.""",
      see=["nn.ConvTranspose2d", "nn.Conv2d"]),

    E("np.pad", 'np.pad(a, pad_width, mode="constant")',
      """Adds a border around an array. pad_width=2 adds 2 elements before and after every axis; a pair
      per axis, ((top, bottom), (left, right)), sets them separately. The default fills the border with
      zeros: the zero padding a convolution applies at the image edges.""",
      """
      >>> np.pad(np.array([[1, 2], [3, 4]]), 1)
      array([[0, 0, 0, 0],
             [0, 1, 2, 0],
             [0, 3, 4, 0],
             [0, 0, 0, 0]])
      >>> np.pad(np.ones((2, 2)), ((0, 0), (1, 2))).shape     # rows untouched; 1 column left, 2 right
      (2, 5)
      """,
      note='mode="edge" repeats the border pixels and mode="reflect" mirrors the image instead of adding zeros.'),

    E("np.outer", "np.outer(a, b)",
      """Outer product of two 1-D vectors: the matrix M[i, j] = a[i] * b[j], of shape (len(a), len(b)).
      np.outer(k1, k1) turns a 1-D smoothing kernel into the matching 2-D kernel.""",
      """
      >>> np.outer([1, 2, 1], [1, 2, 1])               # 1-D blur [1 2 1] -> 3x3 blur kernel
      array([[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]])
      """),

    E("Tensor.expand", "x.expand(*sizes)",
      """Repeats a tensor along dimensions of size 1 (or new leading dimensions) up to the given sizes,
      without copying: every copy reads the same numbers in memory; -1 keeps a size as it is. It is
      broadcasting made explicit. D3-01 gives one (k, k) kernel to 3 channels with
      kk.expand(3, 1, k, k).""",
      """
      >>> k = torch.tensor([[1., 2.], [3., 4.]])
      >>> k.expand(3, 1, 2, 2).shape                   # the same 2x2 kernel for 3 channels
      torch.Size([3, 1, 2, 2])
      >>> torch.tensor([[1], [2]]).expand(-1, 3)       # (2, 1) -> (2, 3)
      tensor([[1, 1, 1],
              [2, 2, 2]])
      """,
      note="""Only size-1 dimensions can be expanded (repeat tiles real data). The copies share memory, so
      never write into an expanded tensor; call .contiguous() or .clone() first to get real copies.""",
      see=["Tensor.contiguous", "Tensor.unsqueeze"]),

    # Tensor.contiguous (first called in D3-01) is written in _day0.py, which explains it with permute.

    E("nn.ConvTranspose2d", "nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=1, padding=0, output_padding=0)",
      """A learnable upsampling layer: the transposed convolution of F.conv_transpose2d (each input pixel
      paints a scaled copy of the kernel into the output; overlaps add up). Output side =
      (H - 1) * stride - 2 * padding + kernel_size + output_padding. kernel_size=2, stride=2 doubles H
      and W with no overlap at all: the upsampler of the U-Net decoder.""",
      """
      >>> x = torch.zeros(1, 8, 16, 16)
      >>> nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)(x).shape       # (16-1)*2 + 2 = 32
      torch.Size([1, 4, 32, 32])
      >>> nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1)(x).shape        # (16-1)*2 - 2 + 3 = 31
      torch.Size([1, 4, 31, 31])
      >>> nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1)(x).shape
      torch.Size([1, 4, 32, 32])
      >>> nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1)(x).shape        # (16-1)*2 - 2 + 4 = 32
      torch.Size([1, 4, 32, 32])
      """,
      note="""When kernel_size is not divisible by stride (3 with stride 2) neighbouring output pixels
      receive different numbers of overlapping stamps: the checkerboard artefact. Use kernel 2 or 4 with
      stride 2, or nn.Upsample followed by nn.Conv2d.""",
      see=["F.conv_transpose2d", "nn.Upsample", "nn.PixelShuffle"]),

    E("nn.Upsample", 'nn.Upsample(scale_factor=None, size=None, mode="nearest", align_corners=None)',
      """A layer without weights that resizes (N, C, H, W) feature maps by a factor, or to a fixed size,
      by calling F.interpolate. nn.Upsample(scale_factor=2, mode="bilinear") followed by a 3x3 nn.Conv2d
      is the artefact-free alternative to a transposed convolution.""",
      """
      >>> x = torch.tensor([[[[1., 2.],
      ...                     [3., 4.]]]])
      >>> nn.Upsample(scale_factor=2, mode="nearest")(x)       # every pixel copied into a 2x2 block
      tensor([[[[1., 1., 2., 2.],
                [1., 1., 2., 2.],
                [3., 3., 4., 4.],
                [3., 3., 4., 4.]]]])
      >>> nn.Upsample(scale_factor=2, mode="bilinear")(x)[0, 0, 0]      # first row: blends neighbours
      tensor([1.0000, 1.2500, 1.7500, 2.0000])
      """,
      note="It has no parameters: on its own it can only copy or blend pixels, so a learned layer must follow it.",
      see=["F.interpolate", "nn.ConvTranspose2d"]),

    E("nn.PixelShuffle", "nn.PixelShuffle(r)",
      """Rearranges (N, C * r * r, H, W) into (N, C, H * r, W * r): each group of r * r channels becomes
      the r x r block of pixels at the same position. It has no weights; the convolution in front of it
      predicts the r * r sub-pixels at low resolution ("sub-pixel convolution").""",
      """
      >>> x = torch.tensor([1., 2., 3., 4.]).reshape(1, 4, 1, 1)   # 4 channels, 1 pixel
      >>> nn.PixelShuffle(2)(x)                                    # 1 channel, 2x2 pixels
      tensor([[[[1., 2.],
                [3., 4.]]]])
      >>> nn.PixelShuffle(2)(torch.zeros(1, 12, 5, 5)).shape      # 12 = 3 channels x 2 x 2
      torch.Size([1, 3, 10, 10])
      """,
      note="The number of channels must be divisible by r * r. nn.PixelUnshuffle does the reverse.",
      see=["nn.ConvTranspose2d", "nn.Upsample"]),

    E("Module.modules", "model.modules()",
      """Iterates over the model itself and every layer inside it, at any depth (layers inside Sequentials
      inside blocks...). The usual way to visit every layer of one type, e.g. to re-initialise the
      weights of all Conv2d layers.""",
      """
      >>> net = nn.Sequential(nn.Conv2d(1, 4, 3), nn.Sequential(nn.ReLU(), nn.Conv2d(4, 2, 1)))
      >>> [type(m).__name__ for m in net.modules()]
      ['Sequential', 'Conv2d', 'Sequential', 'ReLU', 'Conv2d']
      >>> sum(isinstance(m, nn.Conv2d) for m in net.modules())
      2
      """,
      note="The first item is the model itself. children() gives only the direct sub-modules, one level down.",
      see=["Module.children"]),

    E("nn.init.uniform_", "nn.init.uniform_(tensor, a=0.0, b=1.0)",
      """Overwrites a tensor in place with random numbers drawn uniformly between a and b, and returns it.
      Used on a layer's weight to replace the default initialisation; D3-01 sets all weights positive so
      that the overlapping stamps of a transposed convolution become visible.""",
      """
      >>> _ = torch.manual_seed(0)
      >>> w = torch.empty(2, 3)
      >>> _ = nn.init.uniform_(w, 0, 1)                # w itself now holds the numbers
      >>> w
      tensor([[0.4963, 0.7682, 0.0885],
              [0.1320, 0.3074, 0.6341]])
      >>> conv = nn.Conv2d(8, 8, 3)
      >>> bool(nn.init.uniform_(conv.weight, 0, 1).min() >= 0)
      True
      """,
      see=["nn.init.zeros_", "nn.init.constant_"]),

    E("nn.init.zeros_", "nn.init.zeros_(tensor)",
      """Fills a tensor with zeros in place, and returns it. Typically used on a layer's bias after
      initialising its weights: nn.init.zeros_(conv.bias).""",
      """
      >>> lin = nn.Linear(3, 2)
      >>> nn.init.zeros_(lin.bias)
      Parameter containing:
      tensor([0., 0.], requires_grad=True)
      """,
      see=["nn.init.uniform_", "nn.init.constant_"]),

    E("np.floor", "np.floor(x)",
      """Rounds every element down to the nearest whole number (towards minus infinity) and keeps the float
      dtype; add .astype(int) to use the result as an index.""",
      """
      >>> np.floor(np.array([1.7, 2.0, -0.5]))
      array([ 1.,  2., -1.])
      >>> np.floor(np.array([3.4, 0.9])).astype(int)
      array([3, 0])
      """,
      note="Negative numbers go down: floor(-0.5) is -1, while int(-0.5) and .astype(int) cut towards 0 and give 0.",
      see=["torch.round"]),

    E("np.nonzero", "rows, cols = np.nonzero(a)",
      """The indices of the non-zero (or True) elements, as one array per dimension: for a 2-D mask, the
      row indices and the column indices. Same as np.where(a) with one argument. ys.min(), xs.max() etc.
      then give the mask's bounding box.""",
      """
      >>> m = np.array([[0, 0, 0],
      ...               [0, 1, 1],
      ...               [0, 1, 0]])
      >>> ys, xs = np.nonzero(m)                       # rows first, then columns
      >>> ys, xs
      (array([1, 1, 2]), array([1, 2, 1]))
      >>> int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())    # box x1, y1, x2, y2
      (1, 1, 2, 2)
      """,
      note="The rows come first: ys, xs = np.nonzero(mask), not xs, ys.",
      see=["np.where", "torch.where"]),

    E("F.mse_loss", "F.mse_loss(pred, target)",
      """Mean squared error: the mean of (pred - target)**2 over all elements of two tensors of the same
      shape. The function form of nn.MSELoss; D3-01 uses it to score how well the autoencoder
      reconstructs the test images.""",
      """
      >>> F.mse_loss(torch.tensor([1., 2., 3.]), torch.tensor([1., 0., 6.]))   # (0 + 4 + 9) / 3
      tensor(4.3333)
      """,
      note="""Shapes must match exactly: pred (N, 1) with target (N,) is broadcast to (N, N) and gives a
      wrong number, with only a warning.""",
      see=["nn.MSELoss"]),

    E("scipy.ndimage.sobel", "ndimage.sobel(image, axis=-1)",
      """The Sobel derivative of a 2-D image along one axis: large where the brightness changes along
      that axis (axis=1: left-right change, i.e. vertical edges; axis=0: top-bottom change). D3-01
      measures edge strength as np.hypot(sobel(im, 0), sobel(im, 1)).""",
      """
      >>> from scipy import ndimage
      >>> im = np.zeros((4, 4)); im[:, 2:] = 1.0       # dark left half, bright right half
      >>> ndimage.sobel(im, axis=1)                    # change from left to right: the edge
      array([[0., 4., 4., 0.],
             [0., 4., 4., 0.],
             [0., 4., 4., 0.],
             [0., 4., 4., 0.]])
      >>> float(np.abs(ndimage.sobel(im, axis=0)).max())      # nothing changes from top to bottom
      0.0
      """,
      note="""The result has the input's dtype: on a uint8 image the negative and large values wrap around
      (255 * 4 becomes 252). Convert to float first.""",
      see=["np.hypot"]),

    E("np.hypot", "np.hypot(a, b)",
      """Element-wise sqrt(a**2 + b**2): the length of the vector (a, b). Applied to the horizontal and
      vertical Sobel derivatives of an image, it gives the edge strength at every pixel.""",
      """
      >>> np.hypot(np.array([3., 0., 5.]), np.array([4., 2., 12.]))
      array([ 5.,  2., 13.])
      """,
      see=["scipy.ndimage.sobel"]),

    E("np.corrcoef", "np.corrcoef(x, y)",
      """Pearson correlation between two 1-D arrays of the same length: +1 when one grows exactly in step
      with the other (on a straight line), 0 when they are unrelated, -1 when one falls as the other
      rises. Returns the 2x2 matrix [[1, r], [r, 1]]; the number you want is [0, 1].""",
      """
      >>> x = np.array([1., 2., 3., 4.])
      >>> round(float(np.corrcoef(x, 2 * x + 1)[0, 1]), 3)      # a straight line
      1.0
      >>> round(float(np.corrcoef(x, [4., 3., 2., 1.])[0, 1]), 3)
      -1.0
      >>> round(float(np.corrcoef(x, [1., 3., 2., 4.])[0, 1]), 3)
      0.8
      """,
      note="Flatten images first (a.ravel()): with 2-D inputs every row counts as a separate variable.",
      see=["Tensor.ravel"]),

    # ------------------------------------------------------------------ D3-02 · U-Net
    E("Tensor.flip", "x.flip(dims)",
      """Reverses the order of the elements along the given dimension(s) and returns a copy. On an
      (N, C, H, W) batch, x.flip(-1) mirrors every image left-right (the W axis). Flip test-time
      augmentation predicts on the flipped image and flips the prediction back.""",
      """
      >>> x = torch.tensor([[1, 2, 3],
      ...                   [4, 5, 6]])
      >>> x.flip(-1)                                   # last axis: left-right mirror
      tensor([[3, 2, 1],
              [6, 5, 4]])
      >>> x.flip(0)                                    # first axis: upside down
      tensor([[4, 5, 6],
              [1, 2, 3]])
      """,
      note="Tensors do not support negative slicing steps: x[..., ::-1] raises an error, x.flip(-1) is the way.",
      see=["torchvision.transforms.v2.RandomHorizontalFlip"]),

    E("np.logical_or", "np.logical_or(a, b)",
      """Element-wise OR of two boolean (or 0/1) arrays of the same shape: True where at least one is
      true. On a predicted and a true mask it is their union, the denominator of IoU.""",
      """
      >>> pred = np.array([1, 1, 0, 0], bool)
      >>> gt   = np.array([0, 1, 1, 0], bool)
      >>> np.logical_or(pred, gt)
      array([ True,  True,  True, False])
      >>> int(np.logical_and(pred, gt).sum()) / int(np.logical_or(pred, gt).sum())   # IoU = 1 / 3
      0.3333333333333333
      """,
      see=["np.logical_and"]),

    # ------------------------------------------------------------------ D3-03 · instance segmentation
    E("torch.as_tensor", "torch.as_tensor(data, dtype=None)",
      """Turns data (a NumPy array, a list, a number or a tensor) into a tensor. It shares memory with a
      NumPy array instead of copying when it can, and returns a tensor unchanged, which makes it the
      cheap "accept anything" conversion at the top of a function.""",
      """
      >>> a = np.array([1., 2., 3.])
      >>> t = torch.as_tensor(a)
      >>> a[0] = 10.                                   # t shares memory with a
      >>> t
      tensor([10.,  2.,  3.], dtype=torch.float64)
      >>> torch.as_tensor(t) is t                      # already a tensor: returned as it is
      True
      """,
      note="""NumPy floats are float64: add .float() (or dtype=torch.float32) before feeding a model. Use
      torch.tensor(a) when you want an independent copy.""",
      see=["torch.tensor", "torch.from_numpy"]),

    E("torch.round", "torch.round(x, decimals=0)",
      """Rounds every element to the nearest whole number (or to the given number of decimals) and keeps the
      float dtype. Exact halves go to the nearest even number (0.5 -> 0, 1.5 -> 2, 2.5 -> 2), as in NumPy.
      D3-03 snaps box coordinates to the stride-16 grid with torch.round(b / 16) * 16, as RoIPool does.""",
      """
      >>> torch.round(torch.tensor([0.4, 0.5, 1.5, 2.5, -1.7]))
      tensor([ 0.,  0.,  2.,  2., -2.])
      >>> b = torch.tensor([37., 70., 121., 250.])
      >>> torch.round(b / 16) * 16                     # nearest multiple of 16
      tensor([ 32.,  64., 128., 256.])
      """,
      note="Halves round to even, not up; the result is still a float tensor (add .long() for indices).",
      see=["np.floor", "np.round"]),

    E("np.fill_diagonal", "np.fill_diagonal(a, value)",
      """Writes value into the main diagonal a[i, i] of a 2-D array, in place. D3-03 applies it to the IoU
      matrix of a set of boxes with themselves, so that each box's IoU of 1 with itself does not win when
      looking for the most-overlapping pair.""",
      """
      >>> iou = np.array([[1.0, 0.3, 0.6],
      ...                 [0.3, 1.0, 0.1],
      ...                 [0.6, 0.1, 1.0]])
      >>> np.fill_diagonal(iou, 0)
      >>> iou
      array([[0. , 0.3, 0.6],
             [0.3, 0. , 0.1],
             [0.6, 0.1, 0. ]])
      """,
      note="It changes the array and returns None: iou = np.fill_diagonal(iou, 0) throws your matrix away.",
      see=["np.unravel_index"]),

    E("np.unravel_index", "np.unravel_index(flat_index, shape)",
      """Converts a position in the flattened array (what a.argmax() returns without an axis) back into
      (row, column, ...) coordinates for the given shape. Together they answer "where is the largest
      value of this matrix?".""",
      """
      >>> iou = np.array([[0. , 0.3, 0.6],
      ...                 [0.3, 0. , 0.1]])
      >>> int(iou.argmax())                            # position in the flattened array
      2
      >>> r, c = np.unravel_index(iou.argmax(), iou.shape)
      >>> int(r), int(c)                               # row 0, column 2
      (0, 2)
      """,
      see=["np.argmax", "np.fill_diagonal"]),

    E("torch.allclose", "torch.allclose(a, b, rtol=1e-05, atol=1e-08)",
      """True if two float tensors are equal up to small rounding differences: every element satisfies
      |a - b| <= atol + rtol * |b|. The right way to check that two computations agree (a loop against a
      library call, a convolution against unfold + matrix product).""",
      """
      >>> a = torch.tensor([1.0, 2.0])
      >>> torch.allclose(a, a + 1e-6), torch.equal(a, a + 1e-6)
      (True, False)
      >>> torch.allclose(a, a + 0.01), torch.allclose(a, a + 0.01, atol=0.1)
      (False, True)
      """,
      note="""Both tensors must have the same dtype (float32 against float64 raises an error). float32
      results computed in a different order can differ by 1e-5 or more: pass a larger atol.""",
      see=["torch.equal", "np.allclose"]),

    E("torch.equal", "torch.equal(a, b)",
      """True if two tensors have the same shape and exactly the same values, as one Python bool. Right
      for integer and boolean data such as masks and labels; for floats use torch.allclose.""",
      """
      >>> m = torch.tensor([[0, 1], [0, 1]], dtype=torch.uint8)
      >>> torch.equal(m, m.clone()), torch.equal(m, m.T)
      (True, False)
      >>> m == m.T                                     # element-wise: a tensor, not one answer
      tensor([[ True, False],
              [False,  True]])
      """,
      see=["torch.allclose"]),

    E("torchvision.models.detection.mask_rcnn.MaskRCNNPredictor", "MaskRCNNPredictor(in_channels, dim_reduced, num_classes)",
      """The last layers of Mask R-CNN's mask branch: a 2x2 stride-2 transposed convolution (conv5_mask,
      in_channels -> dim_reduced), a ReLU, and a 1x1 convolution (mask_fcn_logits) to num_classes. It
      turns the (R, in_channels, 14, 14) features of R regions into (R, num_classes, 28, 28) mask logits,
      one map per class. To fine-tune on your own classes, replace model.roi_heads.mask_predictor with a
      new one; num_classes counts the background.""",
      """
      >>> from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
      >>> head = MaskRCNNPredictor(256, 256, 2)        # background + person
      >>> head(torch.zeros(5, 256, 14, 14)).shape      # 5 regions -> one 28x28 map per class
      torch.Size([5, 2, 28, 28])
      """,
      note="""Replace the box predictor (FastRCNNPredictor) too, with the same num_classes: the two heads
      must agree on the class list.""",
      see=["torchvision.models.detection.faster_rcnn.FastRCNNPredictor", "nn.ConvTranspose2d"]),

    E("torch.utils.data.Subset", "Subset(dataset, indices)",
      """A dataset restricted to the given indices: subset[i] returns dataset[indices[i]] and len(subset)
      is len(indices). Nothing is copied or loaded, so it is the cheap way to train on a few images or to
      split one dataset into training and validation parts.""",
      """
      >>> from torch.utils.data import Subset, TensorDataset
      >>> ds = TensorDataset(torch.arange(10) * 10)    # items 0, 10, ..., 90
      >>> sub = Subset(ds, [2, 5, 7])
      >>> len(sub), sub[0], sub[2]
      (3, (tensor(20),), (tensor(70),))
      """,
      note="""A Subset keeps the transforms of the dataset it wraps. To train with augmentation and validate
      without, build two datasets and take a Subset of each (as D3-03 does with pf_train and pf).""",
      see=["torch.utils.data.DataLoader"]),

    # ------------------------------------------------------------------ D3-04 · foundation models
    E("Tensor.unfold", "x.unfold(dim, size, step)",
      """Cuts dimension dim into windows of length size, one every step elements, and puts the window as a
      new last dimension (a view, no copy). With size == step the windows do not overlap:
      img.unfold(2, p, p).unfold(3, p, p) turns (B, C, H, W) into (B, C, H/p, W/p, p, p), the grid of
      p x p patches a Vision Transformer starts from.""",
      """
      >>> x = torch.arange(6)
      >>> x.unfold(0, 2, 2)                            # windows of 2, every 2: no overlap
      tensor([[0, 1],
              [2, 3],
              [4, 5]])
      >>> x.unfold(0, 3, 1)                            # windows of 3, every 1: overlapping
      tensor([[0, 1, 2],
              [1, 2, 3],
              [2, 3, 4],
              [3, 4, 5]])
      >>> torch.zeros(1, 3, 224, 224).unfold(2, 16, 16).unfold(3, 16, 16).shape
      torch.Size([1, 3, 14, 14, 16, 16])
      """,
      note="""Tensor.unfold is not F.unfold / nn.Unfold, which extracts sliding blocks for convolutions and
      returns (N, C * k * k, L).""",
      see=["Tensor.permute", "Tensor.reshape"]),

    E("torchvision.models.vit_b_16", "vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)",
      """torchvision's Vision Transformer Base/16 (86.6 M parameters). With these weights the ImageNet-trained
      model is downloaded on first use and cached. Input (N, 3, 224, 224) images normalised with the
      ImageNet mean and std (ViT_B_16_Weights.IMAGENET1K_V1.transforms() resizes, crops and normalises),
      output (N, 1000) class logits; ViT_B_16_Weights.IMAGENET1K_V1.meta["categories"] has the 1000
      names. Parts used in D3-04: vit.conv_proj (the patch embedding, Conv2d(3, 768, kernel_size=16,
      stride=16)), vit.encoder.pos_embedding (1, 197, 768) and the 12 blocks in vit.encoder.layers.""",
      note="""The input must be exactly 224 x 224: there are 196 + 1 position embeddings. weights=None builds
      the same network with random weights and downloads nothing.""",
      see=["torchvision.models.resnet18", "Module.eval"]),

    E("F.normalize", "F.normalize(x, dim=1)",
      """Divides vectors by their length (L2 norm) along dim, so every vector has length 1 and only its
      direction is left. After that, a @ b.T between two sets of normalised vectors is the matrix of
      their cosine similarities, which CLIP zero-shot and k-NN on features are built on.""",
      """
      >>> v = torch.tensor([[3., 4.], [0., 2.]])
      >>> u = F.normalize(v, dim=1)                    # each row divided by its length (5 and 2)
      >>> u
      tensor([[0.6000, 0.8000],
              [0.0000, 1.0000]])
      >>> u @ u.T                                      # cosine similarity of every pair of rows
      tensor([[1.0000, 0.8000],
              [0.8000, 1.0000]])
      """,
      note="""The default dim is 1. For (..., D) features say dim=-1. It is not the mean / std
      normalisation of images (that is v2.Normalize).""",
      see=["torchvision.transforms.v2.Normalize"]),

    E("Tensor.cumsum", "x.cumsum(dim)",
      """Running total along dim: element i is the sum of elements 0 to i. On weights sorted from large
      to small it tells, for every k at once, how much the top k hold; D3-04 keeps the patches that hold
      the top 60% of the attention.""",
      """
      >>> torch.tensor([1, 2, 3, 4]).cumsum(0)
      tensor([ 1,  3,  6, 10])
      >>> w = torch.tensor([0.5, 0.25, 0.125, 0.125])  # sorted, sums to 1
      >>> w.cumsum(0) <= 0.8                           # the top entries holding up to 80% of the mass
      tensor([ True,  True, False, False])
      """,
      see=["np.cumsum", "Tensor.argsort"]),

    E("torch.zeros_like", "torch.zeros_like(x, dtype=None)",
      """A new tensor of zeros with the same shape, dtype and device as x; dtype= overrides the type, e.g.
      torch.bool for an empty mask of the same size.""",
      """
      >>> h = torch.tensor([0.2, 0.7, 0.1])
      >>> torch.zeros_like(h)
      tensor([0., 0., 0.])
      >>> sel = torch.zeros_like(h, dtype=torch.bool); sel[1] = True
      >>> sel
      tensor([False,  True, False])
      """,
      see=["torch.zeros", "np.zeros_like"]),

    E("torch.linalg.svd", "U, S, Vh = torch.linalg.svd(X, full_matrices=False)",
      """Singular value decomposition X = U @ diag(S) @ Vh of an (n, d) matrix; with k = min(n, d), U is
      (n, k), S (k,) and Vh (k, d). S holds the singular values from largest to smallest and the rows of
      Vh are the matching directions. For centred data those rows are the principal components (PCA), so
      X @ Vh[:m].T projects every row onto the top m components.""",
      """
      >>> X = torch.tensor([[2., 0.], [-2., 0.], [0., 1.], [0., -1.]])   # centred: wide in x, narrow in y
      >>> U, S, Vh = torch.linalg.svd(X, full_matrices=False)
      >>> S                                            # largest first
      tensor([2.8284, 1.4142])
      >>> Vh.abs()                                     # rows: the x direction, then the y direction
      tensor([[1., 0.],
              [0., 1.]])
      >>> (X @ Vh[:1].T).shape                         # every point projected on the 1st component
      torch.Size([4, 1])
      """,
      note="""The third output is V transposed: its rows, not its columns, are the components. Their signs are
      arbitrary (PCA colours can flip between runs). Centre the data first (X - X.mean(0)), or the first
      component just points at the mean.""",
      see=["Tensor.mean"]),

    E("open_clip.create_model_and_transforms", 'model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=tag)',
      """Builds a CLIP model from the open_clip package and downloads its weights, e.g. name "ViT-B-32"
      with pretrained "laion2b_s34b_b79k". Returns the model, the training image transform and the
      evaluation image transform. preprocess(pil_image) gives a normalised (3, 224, 224) tensor;
      model.encode_image(images (N, 3, 224, 224)) returns (N, 512) image embeddings and
      model.encode_text(tokens (C, 77)) returns (C, 512) text embeddings, neither normalised.""",
      note="The model comes back in training mode: call model.eval() before using it, and F.normalize the embeddings before comparing them.",
      see=["open_clip.get_tokenizer", "F.normalize"]),

    E("open_clip.get_tokenizer", "tok = open_clip.get_tokenizer(name)",
      """Returns the text tokenizer that belongs to an open_clip model name. tok(list of C strings) gives a
      (C, 77) integer tensor of token ids (every prompt padded or cut to 77 tokens), the input of
      model.encode_text.""",
      see=["open_clip.create_model_and_transforms"]),

    E("transformers.CLIPModel.from_pretrained", 'CLIPModel.from_pretrained("openai/clip-vit-base-patch32")',
      """Downloads once from the Hugging Face Hub and builds OpenAI's CLIP: an image encoder and a text
      encoder. model.get_image_features(pixel_values=...) turns (N, 3, 224, 224) images into (N, 512)
      embeddings; model.get_text_features(input_ids=..., attention_mask=...) turns C tokenised prompts into
      (C, 512) embeddings; neither is normalised. The inputs come from CLIPProcessor, passed with **.""",
      see=["transformers.CLIPProcessor.from_pretrained", "F.normalize"]),

    E("transformers.CLIPProcessor.from_pretrained", 'CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")',
      """Downloads the preprocessing that matches a CLIP checkpoint: an image processor (resize, centre crop
      to 224, normalise) and a tokenizer in one object. proc(images=list_of_PIL_images, return_tensors="pt")
      returns a dict with "pixel_values" (N, 3, 224, 224); proc(text=list_of_strings, return_tensors="pt",
      padding=True) returns "input_ids" and "attention_mask", both (C, L) with L the longest prompt.""",
      see=["transformers.CLIPModel.from_pretrained"]),

    E("transformers.SamModel.from_pretrained", 'SamModel.from_pretrained("facebook/sam-vit-base")',
      """Downloads and builds Segment Anything with the ViT-B image encoder. model(**inputs) takes the
      dict from SamProcessor (pixel_values (1, 3, 1024, 1024) plus the prompt points or boxes) and returns
      pred_masks (1, P, 3, 256, 256): three candidate mask logits per prompt, at low resolution, and
      iou_scores (1, P, 3): the model's own estimate of each candidate's quality (P = number of separate
      prompts; 1 for a single click).""",
      note="""pred_masks are 256 x 256 logits for the padded 1024 x 1024 image: use
      processor.image_processor.post_process_masks to get boolean masks at the original image size.""",
      see=["transformers.SamProcessor.from_pretrained"]),

    E("transformers.SamProcessor.from_pretrained", 'SamProcessor.from_pretrained("facebook/sam-vit-base")',
      """Downloads SAM's preprocessing. proc(pil_image, input_points=[[[x, y]]], return_tensors="pt") resizes
      the image so its longer side is 1024, normalises and pads it, rescales the click (x, y in pixels of
      the original image) the same way, and returns a dict with "pixel_values" (1, 3, 1024, 1024),
      "input_points", "original_sizes" and "reshaped_input_sizes". After the model,
      proc.image_processor.post_process_masks(out.pred_masks, inputs["original_sizes"],
      inputs["reshaped_input_sizes"]) returns one boolean (P, 3, H, W) tensor per image.""",
      note="input_points is nested three deep: images, then points, then (x, y). [[[cx, cy]]] is one click on one image; x comes first, unlike NumPy's (row, col).",
      see=["transformers.SamModel.from_pretrained"]),
]
