"""Reference entries for functions first used in D0-00 (setup and tensors)."""
from . import E

ENTRIES = [
    # ------------------------------------------------------------------ setup and environment
    E("torch.manual_seed", "torch.manual_seed(seed)",
      """Fixes PyTorch's random number generator, so random weights, shuffles and augmentations come
      out the same on every run. The notebooks call it in the setup cell and again right before
      creating a model, so that its initial weights are repeatable.""",
      """
      >>> _ = torch.manual_seed(0); a = torch.rand(2)
      >>> _ = torch.manual_seed(0); b = torch.rand(2)
      >>> torch.equal(a, b)
      True
      """),

    E("torchkit.get_device", "device = torchkit.get_device()",
      """Course helper. Returns torch.device("cuda") when PyTorch can use an NVIDIA GPU,
      torch.device("mps") on an Apple-silicon Mac, otherwise torch.device("cpu"), and prints which
      one it picked (for the CPU, with the number of threads PyTorch uses). Models and batches are
      then moved there with .to(device)."""),

    E("torch.cuda.is_available", "torch.cuda.is_available()",
      """True when PyTorch can run on an NVIDIA GPU (CUDA), False otherwise. The usual pattern: pick
      the device once at the top of a notebook (torchkit.get_device does it for you), then create
      or move tensors and models there.""",
      """
      >>> device = "cuda" if torch.cuda.is_available() else "cpu"
      >>> x = torch.zeros(2, 3, device=device)     # created directly on the GPU when there is one
      >>> x.shape
      torch.Size([2, 3])
      """,
      note="""It is also False on a machine with an NVIDIA GPU when the installed PyTorch is a CPU-only
      build (torch.__version__ ends in "+cpu"), and on a Mac, whose GPU is "mps", not CUDA.""",
      see=["torchkit.get_device", "Tensor.to"]),

    E("torch.cuda.get_device_name", "torch.cuda.get_device_name(0)",
      """The name of GPU number 0 as a string, for example "NVIDIA GeForce RTX 3070". Only call it
      after torch.cuda.is_available() returned True: without a CUDA GPU it raises an error.""",
      """
      >>> name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA GPU"
      >>> type(name)
      <class 'str'>
      """,
      see=["torch.cuda.is_available"]),

    E("torch.get_num_threads", "torch.get_num_threads()",
      """How many CPU threads PyTorch uses to run one operation (a matrix product, a convolution) in
      parallel. The default comes from your CPU's core count; torch.set_num_threads(n) changes it.""",
      """
      >>> n = torch.get_num_threads()
      >>> torch.set_num_threads(2); torch.get_num_threads()
      2
      >>> torch.set_num_threads(n)                 # back to what it was
      """),

    # ------------------------------------------------------------------ datasets and loaders
    E("torchkit.data_root", "torchkit.data_root()",
      """Course helper. Returns the folder where all datasets and checkpoints live, as a
      pathlib.Path: data/ at the repository root (or the folder named in the CVCOURSE_DATA
      environment variable, if set), created if it is missing. torchkit.CHECKPOINTS is
      data_root() / "checkpoints"."""),

    E("torchkit.get_fashion_mnist", "train, test, classes = torchkit.get_fashion_mnist(normalize=True, as_tensors=False)",
      """Course helper. Returns the 60 000 training and 10 000 test images of FashionMNIST and the
      list of its 10 class names, downloading the data into data/ on first use. An item ds[i] is
      (image, label): a float tensor (1, 28, 28), normalised with mean 0.286 and std 0.353 unless
      normalize=False (then in [0, 1]), and the class index. as_tensors=True holds everything in
      memory as a TensorDataset, which is much faster on a CPU.""",
      see=["torch.utils.data.DataLoader", "torchkit.show_batch"]),

    E("torch.utils.data.DataLoader", "DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=None)",
      """Turns a dataset (anything with len(ds) and ds[i]) into batches you can loop over: each step
      takes batch_size items and stacks them, so images (C, H, W) become one tensor (B, C, H, W)
      and integer labels a tensor (B,). shuffle=True draws a new random order every epoch (for
      training; evaluation does not need it).""",
      """
      >>> ds = torch.utils.data.TensorDataset(torch.arange(10.).unsqueeze(1), torch.arange(10))
      >>> loader = torch.utils.data.DataLoader(ds, batch_size=4)
      >>> len(loader), [tuple(xb.shape) for xb, yb in loader]      # the last batch is smaller
      (3, [(4, 1), (4, 1), (2, 1)])
      >>> xb, yb = next(iter(loader))            # only the first batch, e.g. to look at it
      >>> yb
      tensor([0, 1, 2, 3])
      """,
      note="""Items of different sizes (photos of different resolutions, a different number of boxes
      per image) cannot be stacked and raise an error; detection loaders pass
      collate_fn=torchkit.detection_collate, which keeps them as lists.""",
      see=["torch.utils.data.TensorDataset", "torch.stack", "torchkit.detection_collate"]),

    E("torchkit.show_batch", "torchkit.show_batch(images, labels=None, classes=None, n=16, cols=None, size=2.2, preds=None, title=None)",
      """Course helper. Draws up to n images in a grid and returns the Matplotlib figure (follow it
      with plt.show()). images is a batch (B, C, H, W) or a list of (C, H, W) tensors, and the
      normalisation is undone for display. A label can be a class index (shown as the title, named
      with classes), an (H, W) mask (overlaid in colour) or a target dict whose "boxes" are drawn
      and "masks" overlaid. With preds (predicted class ids), the title of every wrong prediction
      turns red and names the predicted class.""",
      note="""It guesses the normalisation: values already in [0, 1] are shown as they are, otherwise
      it assumes FashionMNIST statistics for 1 channel and ImageNet statistics for 3. For other
      statistics pass mean= and std=, or the colours come out wrong.""",
      see=["torchkit.denormalize"]),

    E("Tensor.shape", "x.shape",
      """The size of every dimension, as a tuple-like torch.Size. An attribute, not a call.
      The first thing to print when something does not fit: images are (C, H, W), batches (N, C, H, W).""",
      """
      >>> x = torch.zeros(8, 3, 32, 48)        # a batch of 8 RGB images, 32 high, 48 wide
      >>> x.shape
      torch.Size([8, 3, 32, 48])
      >>> x.shape[0], x.shape[-2:]             # batch size; height and width
      (8, torch.Size([32, 48]))
      """,
      see=["Tensor.ndim", "Tensor.numel"]),

    E("Tensor.dtype", "x.dtype",
      """The type of the elements, an attribute. The ones you will meet: torch.float32 (the default
      for floats, and what layers compute in), torch.int64 (also called long: class labels,
      indices), torch.uint8 (images as stored in files, 0 to 255) and torch.bool (masks, the
      results of comparisons).""",
      """
      >>> torch.zeros(2).dtype, torch.tensor([1, 2]).dtype
      (torch.float32, torch.int64)
      >>> (torch.tensor([1, 2]) > 1).dtype
      torch.bool
      >>> torch.zeros(2, dtype=torch.uint8).dtype      # most creation functions take dtype=
      torch.uint8
      """,
      note="""Layers only accept the dtype of their weights (float32): a uint8 image fed to nn.Conv2d
      fails with "Input type (unsigned char) and bias type (float) should be the same", a float64
      one with "(double)". Convert with .float() first.""",
      see=["Tensor.float", "Tensor.to"]),

    E("Tensor.tolist", "x.tolist()",
      """Converts a tensor into (nested) Python lists of plain Python numbers, for printing a few
      values readably. A 0-dimensional tensor gives a single number, like .item().""",
      """
      >>> torch.tensor([[1, 2], [3, 4]]).tolist()
      [[1, 2], [3, 4]]
      >>> y = torch.tensor([3, 0, 7, 1, 9])
      >>> y[:3].tolist()                         # slicing works as in NumPy: the first three
      [3, 0, 7]
      """,
      see=["Tensor.item", "Tensor.numpy"]),

    E("torchkit.get_pet_datasets", 'train, val, classes = torchkit.get_pet_datasets(size=224, target="category", subset=None)',
      """Course helper. Returns (train, val, classes) for Oxford-IIIT Pet (run tools/download_data.py
      first); images are float (3, size, size), resized and normalised with the ImageNet mean and
      std. target chooses the label: "category" gives (image, breed index 0..36) and the 37 breed
      names; "segmentation" gives (image, int64 mask (H, W)) with 0 background, 1 pet, 2 border;
      "detection" gives (image, {"boxes": (N, 4) as x1, y1, x2, y2 in pixels, "labels": (N,)}) for
      the pet's head. subset=N keeps a class-balanced N training (and N // 4 validation) images
      for quick experiments.""",
      see=["torchkit.show_batch"]),

    E("Tensor.unique", "x.unique(return_counts=False)",
      """The distinct values of a tensor, sorted, as a 1-D tensor; with return_counts=True also how
      often each one occurs. Handy to check which class ids a segmentation mask contains.""",
      """
      >>> mask = torch.tensor([[0, 0, 2], [1, 1, 0]])
      >>> mask.unique()
      tensor([0, 1, 2])
      >>> mask.unique(return_counts=True)        # three 0s, two 1s, one 2
      (tensor([0, 1, 2]), tensor([3, 2, 1]))
      """),

    E("torchkit.get_penn_fudan", "pf = torchkit.get_penn_fudan()",
      """Course helper. Returns the Penn-Fudan pedestrian dataset (170 street photos), downloading
      it into data/ if needed. An item is (image, target): a float image (3, H, W) in [0, 1] at its
      original size, and a dict in the torchvision detection format: "boxes" (N, 4) as x1, y1, x2,
      y2 pixels, "labels" (N,) all 1 (= person), "masks" (N, H, W) of 0 / 1, one per person, plus
      "image_id", "area" and "iscrowd".""",
      see=["torchkit.detection_collate"]),

    E("torchkit.detection_collate", "DataLoader(ds, batch_size=4, collate_fn=torchkit.detection_collate)",
      """Course helper. Turns a list of (image, target) pairs into (list of images, list of targets)
      instead of stacking them into one tensor. Needed for detection and instance segmentation,
      where images have different sizes and a different number of objects; torchvision's
      detection models take exactly these two lists.""",
      """
      >>> a = (torch.zeros(3, 4, 6), {"boxes": torch.zeros(1, 4)})      # one object
      >>> b = (torch.zeros(3, 5, 5), {"boxes": torch.zeros(2, 4)})      # another size, two objects
      >>> imgs, targets = torchkit.detection_collate([a, b])
      >>> [tuple(im.shape) for im in imgs], [len(t["boxes"]) for t in targets]
      ([(3, 4, 6), (3, 5, 5)], [1, 2])
      """,
      see=["torch.stack", "torch.utils.data.DataLoader"]),

    E("torch.stack", "torch.stack(tensors, dim=0)",
      """Joins a list of tensors of identical shape along a new dimension: N tensors of shape
      (C, H, W) give one tensor (N, C, H, W). This is what a DataLoader does with the items of a
      batch.""",
      """
      >>> imgs = [torch.zeros(3, 4, 4), torch.ones(3, 4, 4)]
      >>> torch.stack(imgs).shape
      torch.Size([2, 3, 4, 4])
      >>> torch.stack(imgs, dim=1).shape         # the new axis can go anywhere
      torch.Size([3, 2, 4, 4])
      >>> torch.stack([torch.zeros(3, 4, 6), torch.zeros(3, 5, 5)])
      Traceback (most recent call last):
      ...
      RuntimeError: stack expects each tensor to be equal size, but got [3, 4, 6] at entry 0 and [3, 5, 5] at entry 1
      """,
      note="""torch.cat joins along an existing dimension instead: two (2, 3) tensors give (4, 3) with
      cat and (2, 2, 3) with stack.""",
      see=["torch.cat", "torchkit.detection_collate"]),

    E("torchkit.ShapesDataset", 'torchkit.ShapesDataset(n=512, size=128, seed=0, task="detection")',
      """Course helper. n synthetic images of circles, squares and triangles, generated in memory
      from seed (no download, the same images every time). Items are (image, target) with a float
      image (3, size, size) in [0, 1]; task="detection" gives {"boxes": (N, 4) as x1, y1, x2, y2,
      "labels": (N,)} with 0 circle, 1 square, 2 triangle; task="segmentation" gives an int64 mask
      (size, size) with 0 background, 1 circle, 2 square, 3 triangle. The names are in ds.classes.""",
      """
      >>> ds = torchkit.ShapesDataset(n=4, size=64, seed=0, task="segmentation")
      >>> img, mask = ds[0]
      >>> img.shape, mask.shape, mask.dtype
      (torch.Size([3, 64, 64]), torch.Size([64, 64]), torch.int64)
      >>> ds.classes
      ['circle', 'square', 'triangle']
      """,
      note="""torchvision's detection models reserve label 0 for "background": for them create the
      dataset with label_offset=1, so the classes become 1, 2, 3.""",
      see=["torchkit.detection_collate", "torchkit.show_batch"]),

    E("Tensor.numpy", "x.numpy()",
      """Returns the tensor's data as a NumPy array without copying: both share the same memory, so
      changing one changes the other. Used to hand results to Matplotlib or to NumPy code.""",
      """
      >>> t = torch.tensor([1., 2., 3.])
      >>> a = t.numpy()
      >>> a[0] = 10.; t                          # same memory
      tensor([10.,  2.,  3.])
      """,
      note="""Only for CPU tensors that autograd does not track. A model weight or a GPU result needs
      x.detach().cpu().numpy().""",
      see=["torch.from_numpy", "Tensor.detach", "Tensor.cpu"]),

    # ------------------------------------------------------------------ tensor refresher
    E("torch.arange", "torch.arange(end) / torch.arange(start, end, step, dtype=None)",
      """Like Python's range, as a 1-D tensor: from start (default 0) up to but not including end.
      Integer arguments give an int64 tensor; a float argument or dtype=torch.float32 gives floats.
      With reshape, the quickest way to build a small tensor whose values are easy to follow.""",
      """
      >>> torch.arange(5)
      tensor([0, 1, 2, 3, 4])
      >>> torch.arange(1, 2, 0.25)
      tensor([1.0000, 1.2500, 1.5000, 1.7500])
      >>> torch.arange(6, dtype=torch.float32).reshape(2, 3)
      tensor([[0., 1., 2.],
              [3., 4., 5.]])
      """,
      see=["torch.linspace", "Tensor.reshape"]),

    E("Tensor.reshape", "x.reshape(*shape)",
      """Returns the same elements in a new shape, read in row order (the last dimension changes
      fastest). One size may be -1: it is computed from the others. NumPy arrays have the same
      method.""",
      """
      >>> t = torch.arange(6).reshape(2, 3)
      >>> t
      tensor([[0, 1, 2],
              [3, 4, 5]])
      >>> t[1], t[:, 2]                          # a row; a column (indexing as in NumPy)
      (tensor([3, 4, 5]), tensor([2, 5]))
      >>> t.reshape(-1).shape, t.reshape(3, -1).shape
      (torch.Size([6]), torch.Size([3, 2]))
      """,
      note="""reshape keeps the order of the elements; it does not swap axes. An (H, W, C) image
      reshaped to (C, H, W) comes out scrambled: use permute for that. It returns a view (no copy)
      when it can and a copy otherwise.""",
      see=["Tensor.view", "Tensor.permute", "Tensor.flatten"]),

    E("Tensor.sum", "x.sum(dim=None, keepdim=False)",
      """Adds up all elements (giving a 0-dimensional tensor), or only along dim, which then
      disappears from the shape: for a (2, 3) tensor, sum(dim=0) adds the two rows together and
      gives one value per column, shape (3,). dim can be a tuple; keepdim=True keeps the summed
      dimension with size 1.""",
      """
      >>> t = torch.tensor([[1, 2, 3], [4, 5, 6]])
      >>> t.sum()
      tensor(21)
      >>> t.sum(dim=0), t.sum(dim=1)             # down each column; along each row
      (tensor([5, 7, 9]), tensor([ 6, 15]))
      >>> t.sum(dim=1, keepdim=True).shape
      torch.Size([2, 1])
      """,
      note="""dim names the axis that is summed away, not the one that is kept. NumPy calls the same
      argument axis= (PyTorch accepts both names).""",
      see=["Tensor.mean"]),

    E("torch.from_numpy", "torch.from_numpy(arr)",
      """Wraps a NumPy array as a tensor without copying: both share memory, and the dtype is kept
      (a uint8 image stays uint8, NumPy's default float64 stays float64). The usual first step for
      an image loaded with NumPy or PIL.""",
      """
      >>> a = np.zeros(3, dtype=np.uint8)
      >>> t = torch.from_numpy(a)
      >>> a[0] = 7; t                            # the tensor sees the change
      tensor([7, 0, 0], dtype=torch.uint8)
      >>> torch.from_numpy(np.zeros(2)).dtype
      torch.float64
      """,
      note="""Layers want float32: follow it with .float() (and / 255 for a uint8 image).
      torch.tensor(arr) makes a copy instead of sharing.""",
      see=["Tensor.numpy", "torch.tensor", "Tensor.float"]),

    E("Tensor.device", "x.device",
      """Where the tensor's data lives, an attribute: device(type='cpu') in main memory,
      device(type='cuda', index=0) on the first GPU. An operation needs all its tensors on the same
      device, so a model's input must be where the model's weights are.""",
      """
      >>> torch.zeros(2).device
      device(type='cpu')
      >>> torch.zeros(2).device.type             # "cpu", "cuda" or "mps"
      'cpu'
      """,
      note="""Mixing devices fails with "Expected all tensors to be on the same device"; move one of
      them with .to(device).""",
      see=["Tensor.to", "torchkit.get_device"]),

    E("Tensor.to", "x.to(device) / x.to(dtype) / model.to(device)",
      """Returns the tensor converted to another device (x.to("cuda")), another dtype
      (x.to(torch.float32)) or both; if nothing changes, x itself. On a model, model.to(device)
      moves all its weights and returns the model. The course moves every model and every batch to
      the device picked by torchkit.get_device.""",
      """
      >>> x = torch.tensor([0, 128, 255], dtype=torch.uint8)
      >>> x.to(torch.float32)
      tensor([  0., 128., 255.])
      >>> x.dtype                                # x itself is unchanged
      torch.uint8
      >>> model = nn.Linear(2, 1)
      >>> model.to("cpu") is model               # a model is moved in place and returned
      True
      """,
      note="""For a tensor, x.to(device) on a line of its own does nothing to x: write
      x = x.to(device). For a model, model.to(device) on its own is enough.""",
      see=["Tensor.device", "Tensor.float", "Tensor.cpu"]),

    E("Tensor.ndim", "x.ndim",
      """The number of dimensions, the same as len(x.shape): 1 for a vector, 3 for one image
      (C, H, W), 4 for a batch (B, C, H, W). An attribute; x.dim() is the same as a method.""",
      """
      >>> torch.zeros(3, 32, 32).ndim, torch.zeros(8, 3, 32, 32).ndim
      (3, 4)
      """,
      see=["Tensor.shape"]),

    # ------------------------------------------------------------------ HWC vs CHW
    E("Tensor.permute", "x.permute(*dims)",
      """Reorders the dimensions: the argument lists the old positions in their new order. No data is
      copied. Matplotlib and image files use (H, W, C); PyTorch layers use (C, H, W); permute converts.""",
      """
      >>> hwc = torch.zeros(32, 48, 3)         # H, W, C  (as loaded from a file)
      >>> hwc.permute(2, 0, 1).shape           # C, H, W  (what nn.Conv2d wants)
      torch.Size([3, 32, 48])
      >>> chw = hwc.permute(2, 0, 1)
      >>> chw.permute(1, 2, 0).shape           # and back, for plt.imshow
      torch.Size([32, 48, 3])
      """,
      note="""permute moves whole axes; reshape or view with the same target shape would scramble the
      pixels instead. After permute, call .contiguous() before .view().""",
      see=["Tensor.transpose", "Tensor.reshape"]),

    E("Tensor.unsqueeze", "x.unsqueeze(dim)",
      """Inserts a new dimension of size 1 at position dim; the data is untouched. Its main use:
      a model expects a batch (N, C, H, W), you have one image (C, H, W), so add the batch dimension.""",
      """
      >>> img = torch.zeros(3, 32, 48)         # C, H, W
      >>> img.unsqueeze(0).shape               # 1, C, H, W: a batch of one
      torch.Size([1, 3, 32, 48])
      >>> v = torch.tensor([1., 2., 3.])
      >>> v.unsqueeze(1).shape                 # a column: (3,) -> (3, 1)
      torch.Size([3, 1])
      >>> v[None].shape                        # indexing with None does the same as unsqueeze(0)
      torch.Size([1, 3])
      """,
      note="unsqueeze returns a new view; x itself keeps its shape unless you write x = x.unsqueeze(0).",
      see=["Tensor.squeeze", "Tensor.view"]),

    E("Tensor.data_ptr", "x.data_ptr()",
      """The memory address of the tensor's first element, as an int. Two tensors with the same
      address share their data: the course uses it to show that permute and from_numpy do not
      copy anything.""",
      """
      >>> a = torch.zeros(2, 3)
      >>> a.permute(1, 0).data_ptr() == a.data_ptr()                 # same memory, new shape
      True
      >>> a.permute(1, 0).contiguous().data_ptr() == a.data_ptr()    # a real copy
      False
      """,
      see=["Tensor.contiguous"]),

    E("Tensor.is_contiguous", "x.is_contiguous()",
      """True if the elements lie in memory in the order of the tensor's current shape, row by row.
      A freshly created tensor is contiguous; after permute or transpose it usually is not, because
      only the way the memory is read has changed.""",
      """
      >>> x = torch.zeros(4, 5, 3)
      >>> x.is_contiguous(), x.permute(2, 0, 1).is_contiguous()
      (True, False)
      """,
      see=["Tensor.contiguous", "Tensor.view"]),

    E("Tensor.contiguous", "x.contiguous()",
      """Returns a tensor with the same values laid out in memory in the order of its shape: a copy
      if x is not contiguous (after permute or transpose), x itself if it already is. Needed before
      .view() and by some libraries that read the raw memory.""",
      """
      >>> x = torch.arange(6).reshape(2, 3).permute(1, 0)       # (3, 2), not contiguous
      >>> x.contiguous().is_contiguous()
      True
      >>> x.contiguous().view(6)                                # x.view(6) would raise an error
      tensor([0, 3, 1, 4, 2, 5])
      """,
      see=["Tensor.is_contiguous", "Tensor.view", "Tensor.reshape"]),

    E("Tensor.float", "x.float()",
      """Returns the tensor converted to float32, the dtype layers and losses work in (the same as
      x.to(torch.float32)); x itself is not changed. A uint8 image becomes 0.0 to 255.0, so divide
      by 255 to get [0, 1].""",
      """
      >>> img = torch.tensor([0, 128, 255], dtype=torch.uint8)
      >>> img.float() / 255
      tensor([0.0000, 0.5020, 1.0000])
      >>> img + 10                               # uint8 arithmetic wraps around past 255
      tensor([ 10, 138,   9], dtype=torch.uint8)
      """,
      note="""Convert before doing arithmetic on a uint8 image, as the last line shows. .long()
      (int64) and .double() (float64) work the same way.""",
      see=["Tensor.to", "Tensor.dtype"]),

    # ------------------------------------------------------------------ broadcasting, normalisation
    E("torch.zeros", "torch.zeros(*size, dtype=None, device=None)",
      """A new tensor of the given shape filled with 0 (float32 unless dtype says otherwise). Used
      to allocate outputs and to build test inputs of a given shape.""",
      """
      >>> torch.zeros(2, 3)
      tensor([[0., 0., 0.],
              [0., 0., 0.]])
      >>> torch.zeros((2, 3)).shape, torch.zeros(4, dtype=torch.int64).dtype     # a tuple works too
      (torch.Size([2, 3]), torch.int64)
      """,
      see=["torch.ones", "np.zeros"]),

    E("torch.ones", "torch.ones(*size, dtype=None, device=None)",
      """A new tensor of the given shape filled with 1 (float32 unless dtype says otherwise).""",
      """
      >>> torch.ones(2, 3)
      tensor([[1., 1., 1.],
              [1., 1., 1.]])
      >>> (torch.ones(2, 3, 4, 5) + torch.zeros(3, 1, 1)).shape       # broadcasting, as in this notebook
      torch.Size([2, 3, 4, 5])
      """,
      see=["torch.zeros"]),

    E("torch.tensor", "torch.tensor(data, dtype=None, device=None)",
      """Builds a new tensor from a Python number, a (nested) list or a NumPy array, always copying
      the data. The dtype is inferred: integers give int64, floats give float32; pass dtype= to
      choose.""",
      """
      >>> torch.tensor([1, 2, 3]).dtype, torch.tensor([1., 2., 3.]).dtype
      (torch.int64, torch.float32)
      >>> torch.tensor(2.0)                      # one number: a 0-dimensional tensor, shape ()
      tensor(2.)
      >>> torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).shape   # per-channel values, ready to broadcast
      torch.Size([3, 1, 1])
      """,
      note="""An integer tensor cannot be averaged: torch.tensor([1, 2]).mean() raises an error. Write
      the numbers as floats (1.) or pass dtype=torch.float32.""",
      see=["torch.from_numpy", "Tensor.view"]),

    E("Tensor.view", "x.view(*shape)",
      """Like reshape, but guaranteed not to copy: the result shares memory with x. One size may be
      -1. The course uses v.view(-1, 1, 1) to turn C per-channel values into shape (C, 1, 1), which
      broadcasts against (C, H, W) images; indexing with None, v[:, None, None], does the same.""",
      """
      >>> mean = torch.tensor([0.5, 0.4, 0.3])
      >>> mean.view(-1, 1, 1).shape
      torch.Size([3, 1, 1])
      >>> mean[:, None, None].shape              # the same by indexing: each None adds an axis of size 1
      torch.Size([3, 1, 1])
      >>> torch.zeros(2, 3).permute(1, 0).view(6)
      Traceback (most recent call last):
      ...
      RuntimeError: view size is not compatible with input tensor's size and stride ...
      """,
      note="""view fails on a non-contiguous tensor (after permute or transpose), as the last line
      shows; use reshape, or .contiguous().view(...).""",
      see=["Tensor.reshape", "Tensor.contiguous", "Tensor.unsqueeze"]),

    E("torchkit.denormalize", "torchkit.denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD)",
      """Course helper. Undoes the normalisation (x * std + mean, per channel) and returns a NumPy
      array ready for plt.imshow, clipped to [0, 1]: (C, H, W) becomes (H, W, C), a batch
      (B, C, H, W) becomes (B, H, W, C), a 1-channel image becomes (H, W). The default statistics
      are ImageNet's (torchkit.IMAGENET_MEAN, IMAGENET_STD); for FashionMNIST pass
      torchkit.FASHION_MNIST_MEAN, FASHION_MNIST_STD.""",
      """
      >>> x = torch.zeros(3, 2, 2)               # a normalised image equal to the ImageNet mean colour
      >>> torchkit.denormalize(x).shape
      (2, 2, 3)
      >>> torchkit.denormalize(x)[0, 0]
      array([0.485, 0.456, 0.406], dtype=float32)
      """,
      see=["torchkit.show_batch"]),

    E("Tensor.mean", "x.mean(dim=None, keepdim=False)",
      """The average of all elements, or along dim (a number or a tuple), which then disappears from
      the shape. For a batch (B, C, H, W), mean(dim=(2, 3)) averages over the pixels and gives
      (B, C): global average pooling. NumPy arrays have the same method, with axis= for dim=.""",
      """
      >>> x = torch.arange(24.).reshape(2, 3, 2, 2)                 # B=2, C=3, H=W=2
      >>> x.mean()
      tensor(11.5000)
      >>> x.mean(dim=(2, 3))                     # one value per image and channel
      tensor([[ 1.5000,  5.5000,  9.5000],
              [13.5000, 17.5000, 21.5000]])
      >>> x.mean(dim=(0, 2, 3))                  # per channel, over the whole batch
      tensor([ 7.5000, 11.5000, 15.5000])
      """,
      note="Only for float tensors: on an int64 tensor it raises an error; call .float() first.",
      see=["Tensor.sum", "Tensor.std"]),

    E("Tensor.flatten", "x.flatten(start_dim=0, end_dim=-1)",
      """Returns the tensor with the dimensions from start_dim to end_dim merged into one; by default
      all of them, giving a 1-D tensor of every element. x.flatten(1) keeps the batch dimension:
      (B, C, H, W) becomes (B, C*H*W).""",
      """
      >>> x = torch.zeros(2, 3, 4)
      >>> x.flatten().shape, x.flatten(1).shape
      (torch.Size([24]), torch.Size([2, 12]))
      """,
      note="The layer nn.Flatten() starts at dimension 1 by default (it keeps the batch); x.flatten() starts at 0.",
      see=["Tensor.reshape", "nn.Flatten"]),

    E("Tensor.round", "x.round(decimals=0)",
      """Rounds every element to the nearest integer, or to decimals digits after the point; exact
      halves go to the even neighbour (0.5 -> 0, 1.5 -> 2). NumPy arrays have the same method: the
      course uses arr.round(2) to print short numbers.""",
      """
      >>> x = torch.tensor([0.5, 1.5, 2.567])
      >>> x.round()
      tensor([0., 2., 3.])
      >>> x.round(decimals=1)
      tensor([0.5000, 1.5000, 2.6000])
      """),

    # ------------------------------------------------------------------ autograd
    E("Tensor.requires_grad", "x.requires_grad",
      """An attribute: True if autograd records the operations done with this tensor, so that
      backward() can later compute its .grad. The weights of nn layers have it, data tensors do
      not, and a result computed from at least one such tensor has it too. Set it when creating a
      tensor: torch.tensor(2.0, requires_grad=True).""",
      """
      >>> torch.ones(2).requires_grad, nn.Linear(2, 1).weight.requires_grad
      (False, True)
      >>> w = torch.tensor([1., 2.], requires_grad=True)
      >>> (w * 3).requires_grad                  # computed from w, so tracked too
      True
      """,
      see=["Tensor.grad", "Tensor.backward", "torch.no_grad"]),

    E("Tensor.backward", "loss.backward()",
      """Computes the gradient of a single number (usually the loss) with respect to every tensor it
      was computed from that has requires_grad=True, and adds it to that tensor's .grad. One call
      fills the gradients of all the parameters of a model.""",
      """
      >>> x = torch.tensor(3.0, requires_grad=True)
      >>> y = x ** 2                             # dy/dx = 2x = 6
      >>> y.backward()
      >>> x.grad
      tensor(6.)
      """,
      note="""Gradients add up over calls, which is why each training step starts with
      optimizer.zero_grad(). It works on a one-element tensor; on a tensor with more elements it
      raises "grad can be implicitly created only for scalar outputs".""",
      see=["Tensor.grad", "Tensor.requires_grad", "Optimizer.zero_grad"]),

    E("Tensor.grad", "x.grad",
      """An attribute: after backward(), the gradient of the loss with respect to x, a tensor of x's
      shape. It is None before the first backward() and after optimizer.zero_grad(); every further
      backward() adds to it.""",
      """
      >>> w = torch.tensor([1., 2.], requires_grad=True)
      >>> print(w.grad)
      None
      >>> (w ** 2).sum().backward(); w.grad      # d/dw of w1^2 + w2^2 is 2w
      tensor([2., 4.])
      >>> (w ** 2).sum().backward(); w.grad      # a second backward adds up
      tensor([4., 8.])
      >>> w.grad = None                          # reset by hand: what optimizer.zero_grad() does
      """,
      see=["Tensor.backward", "Optimizer.zero_grad"]),

    E("Tensor.item", "x.item()",
      """Returns the value of a one-element tensor as a plain Python number (float, int or bool).
      Used to print or store a loss: loss.item(). The number is no longer tracked by autograd and
      no longer on the GPU.""",
      """
      >>> loss = torch.tensor(0.25)
      >>> loss.item()
      0.25
      >>> torch.tensor([[7]]).item()             # any shape, as long as there is one element
      7
      """,
      note="On a tensor with more than one element it raises an error; use .tolist() for those.",
      see=["Tensor.tolist"]),

    # ------------------------------------------------------------------ the training loop
    E("torch.Generator", "g = torch.Generator().manual_seed(seed)",
      """A random number generator of your own. Passed as generator=g to torch.rand, torch.randn,
      torch.randperm or a DataLoader, it makes exactly those numbers repeatable, without touching
      (or being disturbed by) the global seed of torch.manual_seed.""",
      """
      >>> g = torch.Generator().manual_seed(0)
      >>> a = torch.rand(3, generator=g)
      >>> g = torch.Generator().manual_seed(0)
      >>> torch.equal(a, torch.rand(3, generator=g))
      True
      """,
      see=["torch.manual_seed", "torch.rand"]),

    E("torch.rand", "torch.rand(*size, generator=None)",
      """A tensor of the given shape filled with random numbers drawn uniformly from [0, 1). Scale
      and shift it for another range: torch.rand(n) * 4 - 2 is uniform in [-2, 2).""",
      """
      >>> _ = torch.manual_seed(0)
      >>> torch.rand(2, 3)
      tensor([[0.4963, 0.7682, 0.0885],
              [0.1320, 0.3074, 0.6341]])
      """,
      see=["torch.randn", "torch.Generator"]),

    E("torch.randn", "torch.randn(*size, generator=None)",
      """A tensor of the given shape filled with random numbers from the standard normal distribution
      (mean 0, standard deviation 1); 0.4 * torch.randn(n) has standard deviation 0.4. The course
      uses it for noise and for random test inputs.""",
      """
      >>> _ = torch.manual_seed(0)
      >>> x = torch.randn(100_000)
      >>> abs(x.mean().item()) < 0.01, round(x.std().item(), 2)      # mean about 0, std about 1
      (True, 1.0)
      """,
      see=["torch.rand"]),

    E("torch.utils.data.TensorDataset", "TensorDataset(X, Y)",
      """Wraps tensors that have the same first dimension into a dataset: item i is (X[i], Y[i]) and
      len() is that first size. The simplest way to feed data that is already in memory to a
      DataLoader.""",
      """
      >>> X, Y = torch.arange(6.).reshape(3, 2), torch.tensor([0, 1, 0])
      >>> ds = torch.utils.data.TensorDataset(X, Y)
      >>> len(ds), ds[1]
      (3, (tensor([2., 3.]), tensor(1)))
      """,
      see=["torch.utils.data.DataLoader"]),

    E("Optimizer.zero_grad", "optimizer.zero_grad()",
      """Clears the .grad of every parameter the optimiser updates (sets it to None), so that the next
      backward() starts from zero. Call it in every training step, before loss.backward().""",
      """
      >>> w = torch.tensor([1.0], requires_grad=True)
      >>> opt = torch.optim.SGD([w], lr=0.1)
      >>> (w ** 2).sum().backward(); w.grad
      tensor([2.])
      >>> opt.zero_grad(); print(w.grad)
      None
      """,
      note="""Forgetting it raises no error: the gradients of successive batches add up, and every
      step uses the sum of all past gradients.""",
      see=["Optimizer.step", "Tensor.backward"]),

    E("Optimizer.step", "optimizer.step()",
      """Updates every parameter the optimiser was given, using the gradients now in .grad; for plain
      SGD, w <- w - lr * w.grad. Call it after loss.backward().""",
      """
      >>> w = torch.tensor([1.0], requires_grad=True)
      >>> opt = torch.optim.SGD([w], lr=0.1)
      >>> (w ** 2).sum().backward()              # w.grad = 2w = 2
      >>> opt.step(); w                          # 1 - 0.1 * 2
      tensor([0.8000], requires_grad=True)
      """,
      see=["Optimizer.zero_grad", "torch.optim.SGD"]),

    E("torch.optim.SGD", "torch.optim.SGD(model.parameters(), lr, momentum=0, weight_decay=0)",
      """Stochastic gradient descent. You create it once with the parameters to update and the
      learning rate lr; every optimizer.step() then moves each parameter against its gradient,
      w <- w - lr * w.grad. With momentum=0.9 each step also keeps 0.9 of the previous step's
      direction, which smooths and speeds up training.""",
      """
      >>> model = nn.Linear(1, 1)
      >>> opt = torch.optim.SGD(model.parameters(), lr=0.1)
      >>> opt.param_groups[0]["lr"]
      0.1
      """,
      see=["Optimizer.step", "Optimizer.zero_grad", "torch.optim.AdamW"]),

    E("Module.parameters", "model.parameters()",
      """All learnable tensors of a model (the weights and biases of every layer inside it), one
      after the other. What you hand to an optimiser: torch.optim.SGD(model.parameters(), lr=0.1).""",
      """
      >>> model = nn.Linear(3, 2)
      >>> [tuple(p.shape) for p in model.parameters()]       # weight (out, in), bias (out,)
      [(2, 3), (2,)]
      >>> sum(p.numel() for p in model.parameters())         # the number of parameters
      8
      """,
      note="It returns a generator, which can be read only once: wrap it in list(...) if you need it twice.",
      see=["torchkit.count_parameters"]),

    E("nn.Linear", "nn.Linear(in_features, out_features, bias=True)",
      """A fully connected layer: y = x @ W.T + b, with a learned weight W of shape
      (out_features, in_features) and a bias b of shape (out_features,). It acts on the last
      dimension, which must have size in_features, and keeps the others: (B, in) -> (B, out).""",
      """
      >>> lin = nn.Linear(3, 2)
      >>> lin.weight.shape, lin.bias.shape
      (torch.Size([2, 3]), torch.Size([2]))
      >>> lin(torch.zeros(8, 3)).shape           # a batch of 8 vectors of length 3
      torch.Size([8, 2])
      """,
      note="""An image batch (B, C, H, W) must first become (B, features), with nn.Flatten or a global
      average pool; a wrong size fails with "mat1 and mat2 shapes cannot be multiplied".""",
      see=["nn.Flatten", "Module.parameters"]),

    E("nn.MSELoss", "loss_fn = nn.MSELoss(); loss = loss_fn(pred, target)",
      """Mean squared error: the average of (pred - target) ** 2 over all elements, as a
      0-dimensional tensor. The standard loss for regression (predicting numbers rather than
      classes).""",
      """
      >>> loss_fn = nn.MSELoss()
      >>> loss_fn(torch.tensor([1., 2., 3.]), torch.tensor([1., 2., 6.]))    # (0 + 0 + 9) / 3
      tensor(3.)
      """,
      note="""pred and target must have the same shape. A prediction (N, 1) against a target (N,)
      broadcasts to (N, N) and gives a wrong loss, with only a warning.""",
      see=["nn.CrossEntropyLoss"]),

    E("torch.no_grad", "with torch.no_grad(): ...",
      """A with block in which autograd records nothing: results have requires_grad=False, and
      computing them takes less memory and time. Use it when you only evaluate a model (testing,
      plotting its predictions), never around the training step itself.""",
      """
      >>> w = torch.ones(2, requires_grad=True)
      >>> (w * 2).requires_grad
      True
      >>> with torch.no_grad():
      ...     y = w * 2
      >>> y.requires_grad
      False
      """,
      note="""It does not switch layers such as dropout and batch norm to evaluation mode: that is
      model.eval(), and evaluation code uses both.""",
      see=["Tensor.requires_grad", "Module.eval"]),

    E("torch.linspace", "torch.linspace(start, end, steps)",
      """steps evenly spaced numbers from start to end, both included, as a float tensor. Used to
      sample an x-axis for plotting a function or a model's predictions.""",
      """
      >>> torch.linspace(-2, 2, 5)
      tensor([-2., -1.,  0.,  1.,  2.])
      >>> torch.linspace(-2, 2, 50).unsqueeze(1).shape    # a column of 50 inputs for nn.Linear(1, 1)
      torch.Size([50, 1])
      """,
      note="arange takes a step size and leaves out the end; linspace takes a number of points and includes the end.",
      see=["torch.arange"]),

    E("torchkit.fit", "history = torchkit.fit(model, train_loader, val_loader, loss_fn, optimizer, epochs, scheduler=None, device=None, max_minutes=None, checkpoint=None, restore_best=False, log_every=50)",
      """Course helper: the training loop. It moves the model and every batch to device (default:
      where the model's weights already are) and every epoch runs, batch by batch, forward, loss,
      optimizer.zero_grad(), backward(), optimizer.step() (plus scheduler.step(): after each batch
      for OneCycleLR, after each epoch for most other schedules), then evaluates on val_loader and
      prints one line. It returns a history dict of per-epoch lists, "train_loss", "val_loss",
      "val_metric" (accuracy for classifiers, NaN otherwise), "lr" and "time", plus "best_epoch"
      and "stopped_early".""",
      """
      >>> X = torch.linspace(-1, 1, 64).unsqueeze(1)
      >>> loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, 3 * X - 1), batch_size=16)
      >>> model = nn.Linear(1, 1)
      >>> history = torchkit.fit(model, loader, loader, nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=0.1),
      ...                        epochs=2, verbose=False)
      >>> list(history), len(history["train_loss"])
      (['train_loss', 'val_loss', 'val_metric', 'lr', 'time', 'stopped_early', 'best_epoch'], 2)
      """,
      note="""With checkpoint=path the file keeps the weights of the best epoch, but the model in
      memory ends with the last epoch's weights unless restore_best=True. max_minutes stops
      training once the time budget is used up (history["stopped_early"] is then True).""",
      see=["torchkit.evaluate", "torchkit.load_checkpoint"]),
]
