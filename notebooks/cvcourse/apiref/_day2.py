"""Reference entries for functions first used on Day 2 (D2-01 .. D2-04: object detection)."""
from . import E

ENTRIES = [
    # ------------------------------------------------------------------ D2-01: boxes, IoU, mAP, NMS
    E("torchvision.transforms.v2.ToImage", "v2.ToImage()",
      """Transform that turns a PIL image or an (H, W, C) NumPy array into an image tensor of shape
      (C, H, W). Values and dtype are kept (uint8 stays 0..255), so it is almost always followed by
      v2.ToDtype(torch.float32, scale=True) to get floats in [0, 1].""",
      """
      >>> from torchvision.transforms import v2
      >>> from PIL import Image
      >>> img = v2.ToImage()(Image.new("RGB", (48, 32)))       # PIL sizes are (width, height)
      >>> img.shape, img.dtype                                 # C, H, W; still uint8
      (torch.Size([3, 32, 48]), torch.uint8)
      >>> v2.ToImage()(np.zeros((32, 48, 3), np.uint8)).shape  # an H, W, C array
      torch.Size([3, 32, 48])
      """,
      see=["torchvision.transforms.v2.ToDtype", "torchvision.transforms.v2.Compose"]),

    E("np.stack", "np.stack(arrays, axis=0)",
      """Joins arrays of the same shape along a NEW axis: k arrays of shape (N,) give (k, N) with axis=0,
      or (N, k) with axis=-1. The box conversions use it to put four (N,) coordinate columns back together
      into one (N, 4) array.""",
      """
      >>> x1, y1 = np.array([10, 30, 50]), np.array([20, 40, 60])   # one value per box, 3 boxes
      >>> np.stack([x1, y1]).shape                                  # new axis first
      (2, 3)
      >>> np.stack([x1, y1], -1)                                    # new axis last: one row per box
      array([[10, 20],
             [30, 40],
             [50, 60]])
      """,
      note="""np.concatenate joins along an EXISTING axis ((2, 3) + (2, 3) -> (4, 3)); np.stack creates a new
      one ((2, 3) + (2, 3) -> (2, 2, 3)).""",
      see=["np.concatenate", "torch.stack"]),

    E("np.minimum", "np.minimum(a, b)",
      """Element-wise minimum of two arrays (broadcast against each other): out[i] = min(a[i], b[i]). For two
      xyxy boxes, the bottom-right corner of their intersection is np.minimum of the two bottom-right
      corners, and the top-left corner is np.maximum of the two top-left corners.""",
      """
      >>> np.minimum(np.array([1, 5, 3]), np.array([4, 2, 3]))
      array([1, 2, 3])
      >>> a, b = np.array([10, 10, 50, 50]), np.array([30, 25, 70, 65])   # two xyxy boxes
      >>> np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])             # corners of the intersection
      (array([30, 25]), array([50, 50]))
      """,
      note="np.min(a) is the smallest element of ONE array; np.minimum(a, b) compares TWO arrays element by element.",
      see=["np.maximum", "np.clip"]),

    E("Tensor.all", "x.all(dim=None)",
      """True if every element is True (non-zero). With dim, the test runs along that dimension and gives one
      result per remaining position. NumPy arrays have the same method (with axis=).""",
      """
      >>> x = torch.tensor([3., 5., 1.])
      >>> (x > 0).all(), (x > 2).all()
      (tensor(True), tensor(False))
      >>> torch.tensor([[1, 1], [0, 1]]).all(dim=1)        # one answer per row
      tensor([ True, False])
      """,
      see=["Tensor.any"]),

    E("Tensor.clip", "x.clip(min=None, max=None)",
      """Limits every value to [min, max] (another name for clamp); with only min, values below min become
      min. NumPy arrays have the same method: (x2 - x1).clip(0) makes the overlap width of two boxes that do
      not overlap 0 instead of negative.""",
      """
      >>> x = torch.tensor([-3., 0.5, 7.])
      >>> x.clip(0)                           # lower bound only
      tensor([0.0000, 0.5000, 7.0000])
      >>> x.clip(0, 1)
      tensor([0.0000, 0.5000, 1.0000])
      >>> np.array([-20., 15.]).clip(0)       # the same on a NumPy array
      array([ 0., 15.])
      """,
      see=["Tensor.clamp", "np.clip"]),

    E("torchvision.ops.box_iou", "torchvision.ops.box_iou(boxes1, boxes2)",
      """IoU of every box in boxes1 (N, 4) with every box in boxes2 (M, 4), returned as an (N, M) tensor.
      Boxes are float tensors in xyxy format (x1, y1, x2, y2). The library version of Exercise 2: same maths,
      runs on the GPU, used inside every torchvision detector.""",
      """
      >>> a = torch.tensor([[0., 0., 10., 10.]])               # one 10 x 10 box
      >>> b = torch.tensor([[0., 0., 10., 10.],                # the same box
      ...                   [5., 0., 15., 10.],                # shifted by half its width
      ...                   [20., 20., 30., 30.]])             # far away
      >>> torchvision.ops.box_iou(a, b)                        # (1, 3); middle: 50 / (100 + 100 - 50)
      tensor([[1.0000, 0.3333, 0.0000]])
      """,
      note="""It needs tensors (torch.tensor(array) first) and assumes xyxy unless you pass fmt="xywh" or
      fmt="cxcywh": boxes in another format give wrong numbers, not an error.""",
      see=["torchvision.ops.generalized_box_iou", "torchvision.ops.nms"]),

    E("np.concatenate", "np.concatenate(arrays, axis=0)",
      """Joins arrays end to end along an EXISTING axis; all other dimensions must match. axis=0 appends rows;
      axis=1 puts columns side by side, e.g. top-left corners (N, 2) and bottom-right corners (N, 2) into
      xyxy boxes (N, 4). The AP code uses it to pad a curve with one value at each end.""",
      """
      >>> lt = np.array([[0, 0], [5, 5]]); rb = np.array([[10, 20], [8, 9]])
      >>> np.concatenate([lt, rb], 1)                     # (2, 2) and (2, 2) side by side -> (2, 4)
      array([[ 0,  0, 10, 20],
             [ 5,  5,  8,  9]])
      >>> np.concatenate([[0.0], np.array([0.5, 0.7]), [1.0]])
      array([0. , 0.5, 0.7, 1. ])
      """,
      see=["np.stack", "torch.cat"]),

    E("Tensor.prod", "x.prod(dim=None)",
      """Multiplies the elements: all of them, or along dim. On an (N, 2) array of widths and heights, prod(-1)
      gives the N areas. NumPy arrays have the same method (with axis=).""",
      """
      >>> wh = torch.tensor([[4., 5.], [2., 0.], [3., 3.]])    # (w, h) of three boxes
      >>> wh.prod(-1)                                          # their areas
      tensor([20.,  0.,  9.])
      """,
      see=["Tensor.sum"]),

    E("np.repeat", "np.repeat(a, repeats, axis=None)",
      """Repeats each element (with axis: each slice along that axis) `repeats` times, next to itself. D2-01 uses
      it to copy one GT box into N identical rows, and its method form a.repeat(3, 2) to turn a grey
      (H, W, 1) image into (H, W, 3).""",
      """
      >>> np.repeat(np.array([1, 2]), 3)
      array([1, 1, 1, 2, 2, 2])
      >>> gt = np.array([80, 40, 120, 80])
      >>> np.repeat(gt[None], 3, 0)                  # (1, 4) -> (3, 4): the same box three times
      array([[ 80,  40, 120,  80],
             [ 80,  40, 120,  80],
             [ 80,  40, 120,  80]])
      >>> np.zeros((2, 2, 1)).repeat(3, 2).shape     # method form, along axis 2
      (2, 2, 3)
      """,
      note="""Tensor.repeat is NOT the same: torch.tensor([1, 2]).repeat(3) tiles the whole tensor, giving
      [1, 2, 1, 2, 1, 2]. The PyTorch twin of np.repeat is torch.repeat_interleave.""",
      see=["np.concatenate"]),

    E("np.diag", "np.diag(a)",
      """With a 2-D array: returns its diagonal a[0, 0], a[1, 1], ... as a 1-D array. With a 1-D array: builds a
      square matrix with those values on the diagonal. D2-01 takes the diagonal of an (N, N) IoU matrix to
      get the IoU of each aligned pair (prediction i with ground truth i).""",
      """
      >>> iou = np.array([[0.9, 0.1, 0.0],
      ...                 [0.2, 0.7, 0.3],
      ...                 [0.0, 0.4, 0.5]])
      >>> np.diag(iou)                         # pair i with pair i
      array([0.9, 0.7, 0.5])
      >>> np.diag([1, 2])
      array([[1, 0],
             [0, 2]])
      """,
      see=["Tensor.diag"]),

    E("torch.optim.Adam", "torch.optim.Adam(params, lr=0.001)",
      """The Adam optimizer. Like SGD it follows the gradient, but it divides each parameter's step by a running
      estimate of that gradient's size, so a step moves about lr whatever the scale of the gradient. D2-01 uses
      it to move four box coordinates with the same step length for every loss; D2-04 to train free queries.""",
      """
      >>> p = torch.tensor([5.0], requires_grad=True)
      >>> opt = torch.optim.Adam([p], lr=1.0)
      >>> loss = (p ** 2).sum()                  # gradient 2 * 5 = 10
      >>> opt.zero_grad(); loss.backward(); opt.step()
      >>> p.detach()                             # moved by lr = 1 (SGD would move by lr * 10)
      tensor([4.])
      """,
      note="""params must be an iterable: torch.optim.Adam([p]) for single tensors, model.parameters() for a
      model. Adam(p) with a bare tensor raises a TypeError.""",
      see=["torch.optim.AdamW", "torch.optim.SGD"]),

    E("Tensor.diag", "x.diag()",
      """The tensor version of np.diag: the diagonal of a 2-D tensor as a 1-D tensor (or a diagonal matrix from a
      1-D tensor). 1 - box_iou(p, g).diag() is the IoU loss of the aligned pairs p[i], g[i].""",
      """
      >>> iou = torch.tensor([[0.9, 0.1], [0.2, 0.7]])
      >>> iou.diag()
      tensor([0.9000, 0.7000])
      """,
      see=["np.diag", "torchvision.ops.box_iou"]),

    E("np.argmax", "np.argmax(a, axis=None)",
      """Index of the largest value: of the flattened array, or one per row / column along axis. On a boolean
      array it is the index of the first True, e.g. the first step at which IoU reached 0.5.""",
      """
      >>> print(np.argmax(np.array([0.2, 0.9, 0.4])))
      1
      >>> ious = np.array([0.1, 0.3, 0.6, 0.8])
      >>> print(np.argmax(ious >= 0.5))                          # the first True
      2
      >>> print(np.argmax(np.array([[1, 7], [9, 2]]), axis=1))   # one per row
      [1 0]
      """,
      note="If no element is True, np.argmax(mask) returns 0, as if the first one were: check mask.any() first.",
      see=["Tensor.argmax", "np.argsort"]),

    E("torchvision.ops.generalized_box_iou_loss", "generalized_box_iou_loss(boxes1, boxes2, reduction='none')",
      """1 - GIoU for ALIGNED pairs: boxes1[i] against boxes2[i], both (N, 4) xyxy, giving N losses in [0, 2]
      (reduction="mean" or "sum" for one number). Unlike 1 - IoU it keeps growing when boxes that do not
      overlap move apart, so it has a gradient. The box loss of DETR (together with L1).""",
      """
      >>> from torchvision.ops import generalized_box_iou_loss
      >>> p = torch.tensor([[0., 0., 10., 10.], [20., 0., 30., 10.]])
      >>> g = torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]])
      >>> generalized_box_iou_loss(p, g)       # perfect pair: 0; disjoint pair: 1 - 0 + (300 - 200) / 300
      tensor([0.0000, 1.3333])
      """,
      see=["torchvision.ops.generalized_box_iou", "torchvision.ops.distance_box_iou_loss",
           "torchvision.ops.complete_box_iou_loss"]),

    E("torchvision.ops.distance_box_iou_loss", "distance_box_iou_loss(boxes1, boxes2, reduction='none')",
      """1 - DIoU for aligned pairs (N, 4) xyxy: 1 - IoU plus the squared distance between the two box centres,
      divided by the squared diagonal of the smallest box enclosing both. Its gradient pulls the centre of the
      prediction straight towards the target.""",
      """
      >>> from torchvision.ops import distance_box_iou_loss
      >>> p = torch.tensor([[20., 0., 30., 10.]]); g = torch.tensor([[0., 0., 10., 10.]])
      >>> distance_box_iou_loss(p, g)          # 1 - 0 + 20**2 / (30**2 + 10**2)
      tensor([1.4000])
      """,
      see=["torchvision.ops.generalized_box_iou_loss", "torchvision.ops.complete_box_iou_loss"]),

    E("torchvision.ops.complete_box_iou_loss", "complete_box_iou_loss(boxes1, boxes2, reduction='none')",
      """1 - CIoU for aligned pairs (N, 4) xyxy: the DIoU loss plus a term that grows when the two boxes have
      different aspect ratios (width / height). The box loss of the YOLOv4 / v5 generation.""",
      """
      >>> from torchvision.ops import complete_box_iou_loss, distance_box_iou_loss
      >>> p = torch.tensor([[0., 0., 10., 20.]]); g = torch.tensor([[0., 0., 10., 10.]])   # tall vs square
      >>> distance_box_iou_loss(p, g), complete_box_iou_loss(p, g)     # CIoU adds the shape mismatch
      (tensor([0.5500]), tensor([0.5532]))
      """,
      see=["torchvision.ops.distance_box_iou_loss", "torchvision.ops.generalized_box_iou_loss"]),

    E("np.cumsum", "np.cumsum(a, axis=None)",
      """Running sum: element k of the result is a[0] + ... + a[k]. On the true-positive flags of predictions
      sorted by score it counts the TPs among the top k + 1, the numerator of precision and recall.""",
      """
      >>> tp = np.array([True, True, False, True])
      >>> np.cumsum(tp)               # TPs so far
      array([1, 2, 2, 3])
      >>> np.cumsum(~tp)              # FPs so far (~ flips True and False)
      array([0, 0, 1, 1])
      """,
      see=["np.maximum.accumulate", "Tensor.cumsum"]),

    E("np.maximum.accumulate", "np.maximum.accumulate(a)",
      """Running maximum: element k is max(a[0], ..., a[k]). Applied to the reversed precision array and reversed
      back, it gives at every recall the best precision at that recall or higher: the monotone envelope
      that AP integrates.""",
      """
      >>> p = np.array([1.0, 0.5, 0.67, 0.75, 0.6])      # raw precision, in score order
      >>> np.maximum.accumulate(p)                        # running max from the left
      array([1., 1., 1., 1., 1.])
      >>> np.maximum.accumulate(p[::-1])[::-1]            # from the right: the envelope
      array([1.  , 0.75, 0.75, 0.75, 0.6 ])
      """,
      see=["np.cumsum", "np.maximum"]),

    E("np.where", "np.where(condition)  or  np.where(condition, x, y)",
      """With one argument: the indices where condition is True, as a tuple with one index array per dimension
      (take [0] for a 1-D array). With three: a new array that takes x where condition is True and y elsewhere,
      element by element.""",
      """
      >>> r = np.array([0.0, 0.25, 0.25, 0.5])
      >>> np.where(r[1:] != r[:-1])              # a tuple: (indices,)
      (array([0, 2]),)
      >>> np.where(r[1:] != r[:-1])[0]           # where recall changes
      array([0, 2])
      >>> np.where(np.array([0.2, 0.9, 0.6]) > 0.5, 1, 0)
      array([0, 1, 1])
      """,
      note="The one-argument form returns a tuple, not an array: forgetting the [0] gives confusing errors later.",
      see=["torch.where", "np.argmax"]),

    E("np.round", "np.round(a, decimals=0)",
      """Rounds every element to the given number of decimals. D2-01 cleans up the IoU thresholds
      0.5, 0.55, ..., 0.95 made by np.arange with it: the floating-point steps of arange are not exact.""",
      """
      >>> t = np.arange(0.5, 0.96, 0.05)
      >>> float(t[2])                     # not exactly 0.6
      0.6000000000000001
      >>> np.round(t, 2)
      array([0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95])
      """,
      note="Halves go to the nearest EVEN number: np.round(0.5) is 0, np.round(1.5) and np.round(2.5) are both 2.",
      see=["Tensor.round"]),

    E("np.clip", "np.clip(a, a_min, a_max)",
      """Limits every element to [a_min, a_max]: smaller values become a_min, larger ones a_max. Keeps simulated
      scores inside [0, 1], and probabilities away from 0 before a log.""",
      """
      >>> np.clip(np.array([-0.2, 0.5, 1.3]), 0, 1)
      array([0. , 0.5, 1. ])
      >>> np.log(np.clip(np.array([0.0, 0.5]), 1e-7, 1))      # no -inf
      array([-16.11809565,  -0.69314718])
      """,
      see=["Tensor.clip", "np.minimum"]),

    E("torchvision.models.resnet50", "torchvision.models.resnet50(weights=None)",
      """Builds a ResNet-50, the 50-layer ResNet made of the bottleneck blocks of D1-03 (25.6 M parameters,
      2048 features before the classifier). weights="IMAGENET1K_V2" downloads the ImageNet-trained weights
      (about 100 MB, once); weights=None gives random ones. Input (N, 3, H, W) normalised images, output
      (N, 1000) ImageNet logits.""",
      """
      >>> net = torchvision.models.resnet50(weights=None).eval()      # random weights, no download
      >>> net(torch.zeros(1, 3, 224, 224)).shape
      torch.Size([1, 1000])
      >>> [name for name, _ in net.named_children()][-3:]             # what comes after layer4
      ['layer4', 'avgpool', 'fc']
      >>> net.fc
      Linear(in_features=2048, out_features=1000, bias=True)
      """,
      see=["torchvision.models.resnet18", "Module.children"]),

    E("Module.children", "model.children()",
      """Iterates over the direct sub-modules of a model in the order they were defined (not recursively).
      list(net.children())[:-2] drops a ResNet's last two children (average pooling and fc), and
      nn.Sequential(*...) glues the rest back together into a feature extractor.""",
      """
      >>> net = torchvision.models.resnet18(weights=None)
      >>> list(net.children())[-2:]
      [AdaptiveAvgPool2d(output_size=(1, 1)), Linear(in_features=512, out_features=1000, bias=True)]
      >>> body = nn.Sequential(*list(net.children())[:-2])
      >>> body(torch.zeros(1, 3, 224, 224)).shape          # a 7 x 7 feature map instead of 1000 logits
      torch.Size([1, 512, 7, 7])
      """,
      note="""Rebuilding with nn.Sequential only works when forward() just calls the children one after the
      other. It is fine for ResNet up to layer4; a model that reshapes or adds branches inside forward() breaks.""",
      see=["Module.modules", "nn.Sequential"]),

    E("torchvision.ops.nms", "torchvision.ops.nms(boxes, scores, iou_threshold)",
      """Greedy non-maximum suppression: repeatedly keeps the highest-scoring remaining box and discards every box
      whose IoU with it is above iou_threshold. boxes (N, 4) float xyxy, scores (N,); returns the int64 indices of
      the kept boxes, highest score first.""",
      """
      >>> boxes = torch.tensor([[0., 0., 10., 10.],         # 0
      ...                       [1., 0., 11., 10.],         # 1: almost box 0 (IoU 90 / 110 = 0.82)
      ...                       [20., 20., 30., 30.]])      # 2: somewhere else
      >>> scores = torch.tensor([0.6, 0.9, 0.7])
      >>> torchvision.ops.nms(boxes, scores, 0.5)           # 1 is kept and removes 0
      tensor([1, 2])
      """,
      note="It ignores classes: a dog box can suppress an overlapping cat box. batched_nms runs it per class.",
      see=["torchvision.ops.batched_nms", "torchvision.ops.box_iou"]),

    E("np.array_equal", "np.array_equal(a, b)",
      """True if the two arrays have the same shape and exactly the same elements, as one Python bool
      (a == b gives an array of element-wise answers instead). For floats with rounding errors use np.allclose.""",
      """
      >>> np.array_equal(np.array([1, 2, 3]), [1, 2, 3])
      True
      >>> np.array_equal(np.array([1, 2]), np.array([1, 2, 3]))     # other shape: False, no error
      False
      >>> np.array([1, 2, 3]) == np.array([1, 0, 3])
      array([ True, False,  True])
      """,
      see=["np.allclose", "torch.equal"]),

    E("np.sort", "np.sort(a, axis=-1)",
      """Returns a sorted copy, smallest first ([::-1] reverses it). D2-01 sorts the indices kept by two NMS
      implementations before comparing them, so that only the set of kept boxes counts, not the order.""",
      """
      >>> a = np.array([3, 1, 2])
      >>> np.sort(a), a                    # a itself is unchanged
      (array([1, 2, 3]), array([3, 1, 2]))
      >>> np.sort(a)[::-1]
      array([3, 2, 1])
      """,
      note="a.sort() sorts in place and returns None; np.argsort(a) gives the order (indices) instead of the values.",
      see=["np.argsort", "torch.argsort"]),

    E("np.exp", "np.exp(x)",
      """e = 2.718... to the power of every element. Soft-NMS multiplies scores by exp(-IoU^2 / sigma): 1 for no
      overlap, close to 0 for a heavy duplicate. It also decodes log-encoded box sizes (D2-02).""",
      """
      >>> np.exp(np.array([0.0, 1.0, -1.0]))
      array([1.        , 2.71828183, 0.36787944])
      >>> iou = np.array([0.0, 0.5, 0.9])
      >>> np.exp(-iou ** 2 / 0.5)                   # Soft-NMS decay factors
      array([1.        , 0.60653066, 0.1978987 ])
      """,
      see=["np.log"]),

    E("torchvision.ops.batched_nms", "torchvision.ops.batched_nms(boxes, scores, idxs, iou_threshold)",
      """NMS run separately per category: a box can only suppress boxes with the same value in idxs (the class
      labels, an int tensor (N,)). Same inputs and output as nms otherwise: kept indices, highest score first.
      The label-offset trick shown in D2-01 is how torchvision does it for small inputs.""",
      """
      >>> boxes = torch.tensor([[0., 0., 10., 10.], [1., 0., 11., 10.], [20., 20., 30., 30.]])
      >>> scores = torch.tensor([0.6, 0.9, 0.7])
      >>> torchvision.ops.nms(boxes, scores, 0.5)                                   # class-agnostic
      tensor([1, 2])
      >>> torchvision.ops.batched_nms(boxes, scores, torch.tensor([0, 1, 0]), 0.5)  # 0 and 1: other classes
      tensor([1, 2, 0])
      """,
      see=["torchvision.ops.nms"]),

    # ------------------------------------------------------------------ D2-02: anchors, RPN, RoIAlign
    E("np.sqrt", "np.sqrt(x)",
      """Square root of every element. Anchors use it to change the shape at a fixed area: for scale s and aspect
      ratio r = h / w, w = s / sqrt(r) and h = s * sqrt(r), so w * h = s^2 for every ratio.""",
      """
      >>> np.sqrt(np.array([4.0, 2.0, 0.25]))
      array([2.        , 1.41421356, 0.5       ])
      >>> r = np.array([0.5, 1.0, 2.0]); w, h = 64 / np.sqrt(r), 64 * np.sqrt(r)
      >>> w.round(1), h.round(1), (w * h).round()     # three shapes, one area: 64 * 64
      (array([90.5, 64. , 45.3]), array([45.3, 64. , 90.5]), array([4096., 4096., 4096.]))
      """),

    E("np.allclose", "np.allclose(a, b, rtol=1e-05, atol=1e-08)",
      """True if every element of a is close to the matching one of b (|a - b| <= atol + rtol * |b|), as one
      bool. The right way to compare floats, which rarely match exactly after a computation and its inverse.""",
      """
      >>> x = np.array([0.1, 0.2]) * 3
      >>> x == np.array([0.3, 0.6]), np.allclose(x, [0.3, 0.6])
      (array([False, False]), True)
      """,
      see=["np.array_equal", "torch.allclose"]),

    E("torch.sigmoid", "torch.sigmoid(x)",
      """Squashes every value (a logit) into (0, 1): sigmoid(x) = 1 / (1 + exp(-x)). Turns an objectness logit into
      a probability; D2-03 and D2-04 also use it to keep predicted box coordinates between 0 and 1.""",
      """
      >>> torch.sigmoid(torch.tensor([-4.6, 0.0, 2.0]))
      tensor([0.0100, 0.5000, 0.8808])
      """,
      note="Losses such as F.binary_cross_entropy_with_logits take the raw logits: do not apply sigmoid first.",
      see=["F.binary_cross_entropy_with_logits", "Tensor.softmax"]),

    E("torchvision.ops.roi_align",
      "torchvision.ops.roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False)",
      """Cuts a fixed-size P x P tile (P = output_size) out of a feature map for every box, by bilinear sampling at
      exact, unrounded positions and averaging sampling_ratio x sampling_ratio samples per bin. input is
      (N, C, H, W); boxes is a list with one (L, 4) xyxy tensor per image, in image pixels; spatial_scale =
      1 / stride converts them to feature-map cells. The output is (K, C, P, P) for K boxes in total.""",
      """
      >>> feat = torch.arange(16.).reshape(1, 1, 4, 4)       # one image, one channel, 4 x 4
      >>> feat[0, 0]
      tensor([[ 0.,  1.,  2.,  3.],
              [ 4.,  5.,  6.,  7.],
              [ 8.,  9., 10., 11.],
              [12., 13., 14., 15.]])
      >>> rois = [torch.tensor([[0., 0., 4., 4.]])]          # one box: the whole map
      >>> torchvision.ops.roi_align(feat, rois, 2, spatial_scale=1.0, sampling_ratio=2, aligned=True)
      tensor([[[[ 2.5000,  4.5000],
                [10.5000, 12.5000]]]])
      """,
      note="""aligned=True treats pixel centres as sitting at +0.5 (Detectron2, Exercise 5); the default False is
      a legacy version shifted by half a pixel, which torchvision's own detectors still use.""",
      see=["torchvision.ops.box_iou"]),

    E("np.add.outer", "np.add.outer(a, b)",
      """All pairwise sums: out[i, j] = a[i] + b[j], with shape (len(a), len(b)); the same as a[:, None] + b[None, :].
      D2-02 uses it to build small test feature maps with a visible pattern.""",
      """
      >>> np.add.outer(np.arange(3), np.array([0, 10, 20, 30]))
      array([[ 0, 10, 20, 30],
             [ 1, 11, 21, 31],
             [ 2, 12, 22, 32]])
      """),

    E("np.sin", "np.sin(x)",
      """Sine of every element, x in radians (np.pi = 180 degrees). D2-02 builds a smooth test feature map from
      sines and cosines of the row and column index.""",
      """
      >>> np.sin(np.array([0.0, np.pi / 6, np.pi / 2]))
      array([0. , 0.5, 1. ])
      """,
      see=["np.cos"]),

    E("np.cos", "np.cos(x)",
      """Cosine of every element, x in radians (np.pi = 180 degrees).""",
      """
      >>> np.cos(np.array([0.0, np.pi / 3, np.pi]))
      array([ 1. ,  0.5, -1. ])
      """,
      see=["np.sin"]),

    E("Tensor.floor", "x.floor()",
      """Rounds every element DOWN to a whole number (towards minus infinity); the result is still a float tensor.
      In bilinear sampling, y.floor() is the index of the grid row at or just above a sample point, and
      y - y.floor() the sample's fractional distance below that row.""",
      """
      >>> y = torch.tensor([1.75, 2.0, -0.25])
      >>> y.floor()
      tensor([ 1.,  2., -1.])
      >>> y - y.floor()                   # fractional part, always in [0, 1)
      tensor([0.7500, 0.0000, 0.7500])
      """,
      see=["Tensor.long", "Tensor.round"]),

    E("Tensor.long", "x.long()",
      """Converts to int64 (torch.long), the dtype PyTorch wants for indices and class labels. Floats are cut
      towards zero (1.7 -> 1, -1.7 -> -1), so floor first when values can be negative.""",
      """
      >>> x = torch.tensor([1.7, -1.7, 3.0])
      >>> x.long()
      tensor([ 1, -1,  3])
      >>> x.floor().long()
      tensor([ 1, -2,  3])
      """,
      see=["Tensor.floor", "Tensor.float", "Tensor.to"]),

    E("Tensor.new_zeros", "x.new_zeros(size)",
      """A tensor of zeros of the given size with the SAME dtype and device as x. Handy inside a function: the
      output buffer is on the GPU whenever the input is, without passing device= around.""",
      """
      >>> feat = torch.ones(3, 8, 8, dtype=torch.float64)
      >>> out = feat.new_zeros((2, 3, 4, 4))
      >>> out.shape, out.dtype
      (torch.Size([2, 3, 4, 4]), torch.float64)
      """,
      see=["torch.zeros", "torch.zeros_like"]),

    E("torch.meshgrid", "torch.meshgrid(ys, xs, indexing='ij')",
      """From a 1-D tensor of row coordinates (length H) and one of column coordinates (length W), builds two
      (H, W) tensors that together list every (row, column) point of the grid. With indexing="ij" the first
      output changes down the rows, the second along the columns. RoIAlign uses it for its sample points.""",
      """
      >>> yy, xx = torch.meshgrid(torch.tensor([0., 1.]), torch.tensor([10., 20., 30.]), indexing="ij")
      >>> yy
      tensor([[0., 0., 0.],
              [1., 1., 1.]])
      >>> xx
      tensor([[10., 20., 30.],
              [10., 20., 30.]])
      """,
      note="""Always pass indexing: np.meshgrid defaults to "xy" (x first, outputs of shape (len(y), len(x))),
      torch.meshgrid to "ij"; mixing them up transposes the grid.""",
      see=["np.meshgrid", "np.mgrid"]),

    E("torchvision.models.detection.fasterrcnn_resnet50_fpn_v2",
      "fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=None, min_size=800, max_size=1333, ...)",
      """Builds torchvision's Faster R-CNN (ResNet-50 + FPN, RPN, RoIAlign, box head; improved "v2" recipe, 43.7 M
      parameters). It takes a LIST of (3, H, W) float images in [0, 1] and resizes them itself (min_size,
      max_size). Train mode needs targets, a list of dicts with "boxes" (K, 4) xyxy pixels and "labels" (K,) int64
      starting at 1 (0 means background), and returns a dict of the four losses; eval mode returns one dict per
      image with "boxes", "labels", "scores".""",
      """
      >>> from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
      >>> m = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=2, min_size=128, max_size=128)
      >>> imgs = [torch.rand(3, 128, 128)]
      >>> targets = [{"boxes": torch.tensor([[20., 30., 90., 100.]]), "labels": torch.tensor([1])}]
      >>> sorted(m.train()(imgs, targets))        # train mode: the losses
      ['loss_box_reg', 'loss_classifier', 'loss_objectness', 'loss_rpn_box_reg']
      >>> sorted(m.eval()(imgs)[0])               # eval mode: detections of image 0
      ['boxes', 'labels', 'scores']
      """,
      note="""weights="DEFAULT" downloads the COCO weights (91 classes including background, about 170 MB, once);
      weights=None gives a random model. num_classes counts the background: a person detector has 2.""",
      see=["torchvision.models.detection.faster_rcnn.FastRCNNPredictor",
           "torchvision.models.detection.fcos_resnet50_fpn"]),

    E("torchvision.models.detection.faster_rcnn.FastRCNNPredictor", "FastRCNNPredictor(in_channels, num_classes)",
      """The last layer of Faster R-CNN's box head: two linear layers on each RoI's feature vector, cls_score
      (num_classes logits) and bbox_pred (4 * num_classes deltas, one box per class). Installing a new one as
      model.roi_heads.box_predictor adapts a pretrained detector to your classes; num_classes counts the
      background, in_channels is the length of the RoI feature vector (cls_score.in_features of the old one).""",
      """
      >>> from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
      >>> head = FastRCNNPredictor(1024, 2)                   # background + 1 class
      >>> scores, deltas = head(torch.zeros(5, 1024))         # features of 5 RoIs
      >>> scores.shape, deltas.shape
      (torch.Size([5, 2]), torch.Size([5, 8]))
      """,
      see=["torchvision.models.detection.fasterrcnn_resnet50_fpn_v2"]),

    E("np.convolve", "np.convolve(a, v, mode='full')",
      """Slides v along a and sums the products at every position (a 1-D convolution). With v = np.ones(k) / k it
      is a moving average over k values; mode="valid" keeps only the positions where v fits entirely, so
      len(a) - k + 1 values come out. D2-02 smooths noisy loss curves this way.""",
      """
      >>> loss = np.array([4., 2., 3., 1., 2., 0.])
      >>> np.convolve(loss, np.ones(3) / 3, "valid")      # mean of every window of 3
      array([3., 2., 2., 1.])
      """),

    # ------------------------------------------------------------------ D2-03: one-stage detectors
    E("np.ndenumerate", "for index, value in np.ndenumerate(a):",
      """Loops over every element of an array of any shape, giving its index tuple together with its value
      (enumerate for N-D arrays). D2-03 uses it to write each grid cell's value on a plot.""",
      """
      >>> for (i, j), v in np.ndenumerate(np.array([[1, 2], [3, 4]])):
      ...     print(i, j, v)
      0 0 1
      0 1 2
      1 0 3
      1 1 4
      """),

    E("F.binary_cross_entropy_with_logits", "F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')",
      """Binary cross-entropy for yes / no outputs, computed from raw logits z: per element
      -[t log sigmoid(z) + (1 - t) log(1 - sigmoid(z))], then the mean (reduction="none" keeps one loss per
      element, "sum" adds them up). targets are floats 0. / 1. of the same shape. Objectness and focal loss use it.""",
      """
      >>> logits = torch.tensor([2.0, -1.0, 0.0])
      >>> targets = torch.tensor([1.0, 0.0, 1.0])
      >>> F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
      tensor([0.1269, 0.3133, 0.6931])
      >>> F.binary_cross_entropy_with_logits(logits, targets)          # the mean
      tensor(0.3778)
      """,
      note="""Give it logits, not probabilities (it applies the sigmoid itself, in a numerically stable way), and
      float targets: integer targets raise an error.""",
      see=["torch.sigmoid", "F.binary_cross_entropy", "F.cross_entropy"]),

    E("nn.ModuleList", "nn.ModuleList([module, ...])",
      """A list of layers that the model knows about: their parameters appear in model.parameters() (so the
      optimizer trains them) and move with model.to(device). Use it for a variable number of layers that your
      forward() calls in its own way, e.g. one lateral conv per pyramid level; unlike nn.Sequential it has no
      forward of its own.""",
      """
      >>> class Lateral(nn.Module):
      ...     def __init__(self):
      ...         super().__init__()
      ...         self.convs = nn.ModuleList([nn.Conv2d(c, 16, 1) for c in (32, 64)])
      >>> m = Lateral()
      >>> len(m.convs), m.convs[1].in_channels
      (2, 64)
      >>> sum(p.numel() for p in m.parameters())      # (32 * 16 + 16) + (64 * 16 + 16)
      1568
      """,
      note="A plain Python list of layers looks the same but is invisible to the model: not in parameters(), not trained, not moved to the GPU.",
      see=["nn.Sequential"]),

    E("torchvision.models.feature_extraction.create_feature_extractor",
      "create_feature_extractor(model, return_nodes={'layer2': 'C3', ...})",
      """Wraps a model so that calling it returns a dict of intermediate outputs instead of the final one:
      return_nodes maps names of internal layers ("layer2", "layer3", ... for a ResNet) to the keys you want.
      The usual way to get the C3 / C4 / C5 maps of a backbone for an FPN.""",
      """
      >>> from torchvision.models.feature_extraction import create_feature_extractor
      >>> body = create_feature_extractor(torchvision.models.resnet18(weights=None), {"layer2": "C3", "layer4": "C5"})
      >>> out = body(torch.zeros(1, 3, 64, 64))
      >>> {k: tuple(v.shape) for k, v in out.items()}       # strides 8 and 32
      {'C3': (1, 128, 8, 8), 'C5': (1, 512, 2, 2)}
      """,
      note="""Valid names are listed by torchvision.models.feature_extraction.get_graph_node_names(model);
      a misspelt name raises an error.""",
      see=["Module.register_forward_hook", "Module.children"]),

    E("torchvision.models.detection.retinanet_resnet50_fpn",
      "retinanet_resnet50_fpn(weights=None, weights_backbone=None, num_classes=None)",
      """Builds torchvision's RetinaNet: ResNet-50 + FPN (P3 to P7), classification and box heads shared by all
      levels, focal loss. Same interface as every torchvision detector: a list of (3, H, W) images in [0, 1];
      eval mode returns one dict per image with "boxes" (xyxy pixels), "labels", "scores", train mode with targets
      a dict of losses. model.backbone(x) on a normalised (N, 3, H, W) batch returns just the pyramid.""",
      """
      >>> from torchvision.models.detection import retinanet_resnet50_fpn
      >>> m = retinanet_resnet50_fpn(weights=None, weights_backbone=None).eval()   # random, no download
      >>> feats = m.backbone(torch.zeros(1, 3, 128, 128))                          # the FPN levels P3 .. P7
      >>> {k: tuple(v.shape[1:]) for k, v in feats.items()}
      {'0': (256, 16, 16), '1': (256, 8, 8), '2': (256, 4, 4), 'p6': (256, 2, 2), 'p7': (256, 1, 1)}
      """,
      note="""weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT downloads the COCO weights; their
      .meta["categories"] lists the 91 class names (index = label). Even with weights=None the default
      weights_backbone downloads ImageNet weights: pass weights_backbone=None for no download at all.""",
      see=["torchvision.models.detection.fcos_resnet50_fpn",
           "torchvision.models.detection.fasterrcnn_resnet50_fpn_v2"]),

    E("np.linalg.svd", "U, S, Vt = np.linalg.svd(X, full_matrices=False)",
      """Singular value decomposition: X = U @ np.diag(S) @ Vt, with S sorted from largest to smallest. If the
      columns of X (n, d) are centred (mean subtracted), the rows of Vt are the principal directions (PCA), and
      X @ Vt[:3].T projects every row onto the 3 most important ones.""",
      """
      >>> X = np.array([[2., 0.], [0., 1.], [-2., 0.], [0., -1.]])     # spread mostly along x
      >>> U, S, Vt = np.linalg.svd(X, full_matrices=False)
      >>> U.shape, S, Vt.shape
      ((4, 2), array([2.82842712, 1.41421356]), (2, 2))
      >>> np.allclose(U @ np.diag(S) @ Vt, X)
      True
      """,
      note="""full_matrices=False gives the small U of shape (n, k) instead of (n, n): with n = thousands of
      pixels the full one would need gigabytes.""",
      see=["torch.pca_lowrank", "torch.linalg.svd"]),

    E("nn.init.constant_", "nn.init.constant_(tensor, value)",
      """Fills a parameter with one value, in place (the trailing _). Detectors set the bias of the last
      classification conv to -log((1 - pi) / pi) with pi = 0.01, so that at the start every anchor predicts
      "1% object" and the many background anchors do not blow up the first losses.""",
      """
      >>> import math
      >>> conv = nn.Conv2d(256, 3, 3, padding=1)
      >>> _ = nn.init.constant_(conv.bias, -math.log((1 - 0.01) / 0.01))
      >>> conv.bias.detach()
      tensor([-4.5951, -4.5951, -4.5951])
      >>> torch.sigmoid(conv.bias.detach())               # the prior probability
      tensor([0.0100, 0.0100, 0.0100])
      """,
      see=["nn.init.normal_", "nn.init.zeros_"]),

    E("np.mgrid", "yy, xx = np.mgrid[0:H, 0:W]",
      """Not a function call but indexing with slices: gives the row index and the column index of every position
      of an H x W grid as two (H, W) arrays (like np.meshgrid with indexing="ij"). Adding 0.5 gives pixel
      centres.""",
      """
      >>> yy, xx = np.mgrid[0:2, 0:3]
      >>> yy
      array([[0, 0, 0],
             [1, 1, 1]])
      >>> xx
      array([[0, 1, 2],
             [0, 1, 2]])
      """,
      see=["np.meshgrid", "torch.meshgrid"]),

    E("F.max_pool2d", "F.max_pool2d(x, kernel_size, stride=None, padding=0)",
      """The function form of nn.MaxPool2d: each kernel_size x kernel_size window of an (N, C, H, W) tensor is
      replaced by its maximum (stride defaults to kernel_size). With kernel 3, stride 1, padding 1 the size is
      kept and each value becomes the max of its 3 x 3 neighbourhood, so x == F.max_pool2d(x, 3, 1, 1) marks the
      local maxima: CenterNet's peak finding without NMS.""",
      """
      >>> x = torch.arange(16.).reshape(1, 1, 4, 4)
      >>> F.max_pool2d(x, 2)[0, 0]                          # 2 x 2 windows: H and W halved
      tensor([[ 5.,  7.],
              [13., 15.]])
      >>> h = torch.tensor([[0., 1., 0.], [0., 0., 0.], [0., 0., 2.]])[None, None]
      >>> ((h == F.max_pool2d(h, 3, 1, 1)) & (h > 0))[0, 0]    # local maxima
      tensor([[False,  True, False],
              [False, False, False],
              [False, False,  True]])
      """,
      see=["nn.MaxPool2d"]),

    E("np.ma.masked_where", "np.ma.masked_where(condition, a)",
      """Returns a masked array: a copy of a in which the elements where condition is True are marked as missing.
      Matplotlib draws masked pixels transparent, so imshow(np.ma.masked_where(m == 0, m)) overlays only the
      non-zero part of a map on an image.""",
      """
      >>> m = np.array([0.0, 0.5, 0.0, 1.0])
      >>> np.ma.masked_where(m == 0, m)
      masked_array(data=[--, 0.5, --, 1.0],
                   mask=[ True, False,  True, False],
             fill_value=1e+20)
      """),

    E("torchvision.models.detection.fcos_resnet50_fpn",
      "fcos_resnet50_fpn(weights=None, weights_backbone=None, num_classes=None, min_size=800, max_size=1333)",
      """Builds torchvision's FCOS: anchor-free (one prediction per feature-map location, so num_anchors = 1),
      ResNet-50 + FPN, focal loss and centre-ness. Same interface as the other torchvision detectors: a list of
      (3, H, W) images in [0, 1]; eval mode returns per image a dict with "boxes", "labels", "scores", train mode
      with targets a dict of losses. Its class layer is model.head.classification_head.cls_logits.""",
      """
      >>> from torchvision.models.detection import fcos_resnet50_fpn
      >>> m = fcos_resnet50_fpn(weights=None, weights_backbone=None)            # random, no download
      >>> head = m.head.classification_head
      >>> head.num_anchors, head.num_classes, head.cls_logits.out_channels       # 91 COCO classes
      (1, 91, 91)
      >>> m.transform.min_size, m.transform.max_size                             # inputs are resized to this
      ((800,), 1333)
      """,
      note="""weights="DEFAULT" (FCOS_ResNet50_FPN_Weights.DEFAULT) downloads the COCO weights; the default
      weights_backbone downloads ImageNet weights even with weights=None, so pass weights_backbone=None too
      for no download at all.""",
      see=["torchvision.models.detection.retinanet_resnet50_fpn", "nn.init.normal_", "nn.init.constant_"]),

    # ------------------------------------------------------------------ D2-04: DETR, fine-tuning, deployment
    E("scipy.optimize.linear_sum_assignment", "rows, cols = linear_sum_assignment(cost)",
      """The Hungarian algorithm. Given a cost matrix (Q, G), it picks min(Q, G) pairs, using each row and each
      column at most once, with the smallest total cost, and returns their row and column indices. With more
      rows (queries) than columns (objects), every object gets exactly one query and the rest stay unmatched.""",
      """
      >>> from scipy.optimize import linear_sum_assignment
      >>> cost = np.array([[4., 1.],             # 3 queries (rows) x 2 objects (columns)
      ...                  [2., 3.],
      ...                  [5., 6.]])
      >>> rows, cols = linear_sum_assignment(cost)
      >>> rows, cols                             # query 0 -> object 1, query 1 -> object 0
      (array([0, 1]), array([1, 0]))
      >>> float(cost[rows, cols].sum())          # 1 + 2; query 2 is left over
      3.0
      """,
      note="""It works on NumPy arrays: convert a cost computed by a model with .detach().cpu().numpy() first.
      maximize=True finds the largest total instead (for scores rather than costs).""",
      see=["torch.cdist", "torchvision.ops.generalized_box_iou"]),

    E("torch.is_tensor", "torch.is_tensor(obj)",
      """True if obj is a PyTorch tensor (also an nn.Parameter or a torchvision image), False for NumPy arrays,
      lists and numbers. Lets a helper accept both: tensors are converted with .detach().cpu().numpy(),
      everything else with np.asarray.""",
      """
      >>> torch.is_tensor(torch.zeros(2)), torch.is_tensor(np.zeros(2)), torch.is_tensor([1, 2])
      (True, False, False)
      """),

    E("Tensor.argmin", "x.argmin(dim=None)",
      """Index of the smallest value: of the flattened tensor without dim, otherwise one index per row or column.
      NumPy arrays have the same method (axis=); D2-04 uses it on one column of a cost matrix to find the
      cheapest query for an object.""",
      """
      >>> cost = torch.tensor([[4., 1.], [2., 3.], [5., 0.5]])
      >>> cost[:, 0].argmin()              # cheapest query for object 0
      tensor(1)
      >>> cost.argmin(dim=0)               # for every object (column)
      tensor([1, 2])
      >>> cost.argmin()                    # no dim: an index into the flattened tensor
      tensor(5)
      """,
      see=["Tensor.argmax", "Tensor.min"]),

    E("np.delete", "np.delete(a, index, axis=None)",
      """Returns a copy of a without the given element(s); with axis, whole rows (axis=0) or columns (axis=1) are
      removed. The original array is not changed.""",
      """
      >>> np.delete(np.arange(5), 2)
      array([0, 1, 3, 4])
      >>> np.delete(np.array([[1, 2], [3, 4], [5, 6]]), 0, axis=0)      # drop row 0
      array([[3, 4],
             [5, 6]])
      """,
      note="Without axis the array is flattened first: np.delete(m, 0) on a 2-D m returns a 1-D array.",
      see=["np.where"]),

    E("torch.cdist", "torch.cdist(a, b, p=2.0)",
      """Distances between every row of a (N, D) and every row of b (M, D), as an (N, M) tensor. p=2 is the
      Euclidean distance, p=1 the L1 distance (sum of absolute differences): DETR's box cost between Q predicted
      and G ground-truth boxes of 4 numbers each.""",
      """
      >>> a = torch.tensor([[0., 0.], [1., 1.]])
      >>> b = torch.tensor([[1., 0.], [3., 4.], [0., 0.]])
      >>> torch.cdist(a, b)                  # Euclidean, (2, 3)
      tensor([[1.0000, 5.0000, 0.0000],
              [1.0000, 3.6056, 1.4142]])
      >>> torch.cdist(a, b, p=1)             # L1: |dx| + |dy|
      tensor([[1., 7., 0.],
              [1., 5., 2.]])
      """,
      see=["torchvision.ops.generalized_box_iou"]),

    E("torchvision.ops.generalized_box_iou", "torchvision.ops.generalized_box_iou(boxes1, boxes2)",
      """Pairwise GIoU, like box_iou: every box of boxes1 (N, 4) against every box of boxes2 (M, 4), an (N, M)
      tensor, xyxy format. GIoU = IoU - (empty part of the smallest enclosing box) / (area of that box), in
      [-1, 1]: unlike IoU it keeps decreasing as boxes that do not overlap move apart.""",
      """
      >>> a = torch.tensor([[0., 0., 10., 10.]])
      >>> b = torch.tensor([[0., 0., 10., 10.], [20., 0., 30., 10.]])    # the same box; a disjoint one
      >>> torchvision.ops.generalized_box_iou(a, b)                       # right: 0 - (300 - 200) / 300
      tensor([[ 1.0000, -0.3333]])
      """,
      note="""It needs xyxy, while DETR predicts cxcywh: convert first (day2.box_cxcywh_to_xyxy_t or
      torchvision.ops.box_convert). For aligned pairs take .diag() of the result.""",
      see=["torchvision.ops.box_iou", "torchvision.ops.generalized_box_iou_loss"]),

    E("nn.Parameter", "nn.Parameter(tensor)",
      """Marks a tensor as a learnable parameter: requires_grad is switched on, and when it is assigned as an
      attribute of an nn.Module it is registered, so model.parameters() (and thus the optimizer) includes it.
      DETR's learned object queries are such parameters; D2-04 also hands two of them straight to an optimizer.""",
      """
      >>> q = nn.Parameter(torch.zeros(6, 4))
      >>> q.requires_grad
      True
      >>> class Queries(nn.Module):
      ...     def __init__(self):
      ...         super().__init__()
      ...         self.q = nn.Parameter(torch.zeros(6, 4))
      >>> [(name, tuple(p.shape)) for name, p in Queries().named_parameters()]
      [('q', (6, 4))]
      """,
      see=["Module.parameters", "Tensor.requires_grad_"]),

    E("torch.full", "torch.full(size, fill_value)",
      """A new tensor of the given size with every element equal to fill_value. The dtype follows the value: an
      int gives int64 (what F.cross_entropy wants as class targets), a float gives float32. D2-04 starts every
      query's target as "no object" this way and then overwrites the matched ones.""",
      """
      >>> t = torch.full((5,), 3)                          # 5 queries, class 3 = "no object"
      >>> t[torch.tensor([0, 2])] = torch.tensor([1, 0])   # two matched queries get real classes
      >>> t, t.dtype
      (tensor([1, 3, 0, 3, 3]), torch.int64)
      >>> torch.full((2, 2), 0.5)
      tensor([[0.5000, 0.5000],
              [0.5000, 0.5000]])
      """,
      see=["torch.zeros", "torch.full_like"]),

    E("transformers.pipeline", 'pipeline(task, model="org/name")',
      """Hugging Face's one-line inference API (pip install transformers): returns a ready-to-call object that
      downloads the named model from the Hugging Face Hub on first use (cached afterwards) and does its own pre-
      and post-processing. For task="zero-shot-object-detection", ov(pil_image, candidate_labels=[...],
      threshold=0.1) returns a list of dicts {"score", "label", "box": {"xmin", "ymin", "xmax", "ymax"}} in pixels.""",
      note="The first call downloads hundreds of MB and needs internet access: wrap it in try / except, as D2-04 does.",
      see=["torchvision.models.detection.fcos_resnet50_fpn"]),

    E("nn.init.normal_", "nn.init.normal_(tensor, mean=0.0, std=1.0)",
      """Fills a parameter in place with random values from a normal distribution. A new classification layer in
      RetinaNet / FCOS gets weights with std=0.01 (so small that the output is set by the bias) and its bias
      from nn.init.constant_.""",
      """
      >>> conv = nn.Conv2d(256, 2, 3, padding=1)
      >>> _ = nn.init.normal_(conv.weight, std=0.01)
      >>> round(conv.weight.std().item(), 3)
      0.01
      """,
      see=["nn.init.constant_", "nn.init.zeros_"]),

    E("np.median", "np.median(a, axis=None)",
      """The middle value of the sorted data (the mean of the two middle values for an even count). A few slow
      outliers, such as a run disturbed by another program, shift the mean a lot but the median hardly at all:
      the right summary for repeated timings.""",
      """
      >>> ms = np.array([241., 238., 950., 244.])       # one run was disturbed
      >>> float(np.median(ms)), float(np.mean(ms))
      (242.5, 418.25)
      """,
      see=["np.mean", "np.percentile"]),

    E("torch.onnx.export", "torch.onnx.export(model, args, f, input_names=None, output_names=None, dynamic_axes=None)",
      """Converts a model to ONNX, a framework-neutral format that ONNX Runtime, TensorRT, OpenVINO or CoreML can
      load and optimise: it records the operations the model performs on the example input args (a tensor or a
      tuple of tensors) and writes graph and weights to f (a file name or a file-like object such as io.BytesIO).
      input_names / output_names name the graph's inputs and outputs; dynamic_axes marks dimensions, such as the
      batch, that may change at run time.""",
      note="""Needs the onnx package (and onnxscript for the default exporter of recent PyTorch versions); call
      model.eval() first. Detection post-processing such as NMS is often the part that fails to export.""",
      see=["Module.eval"]),

    E("np.bincount", "np.bincount(x)",
      """Counts how often each non-negative integer occurs: result[k] = how many times k appears in x, for
      k = 0 ... max(x). D2-04 turns "detections per image" into a histogram with it.""",
      """
      >>> np.bincount([0, 2, 2, 3, 2])           # 0 once, 1 never, 2 three times, 3 once
      array([1, 0, 3, 1])
      """,
      see=["torch.bincount"]),
]
