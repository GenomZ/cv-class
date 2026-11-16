"""Reference entries for functions first used in D1-01 (the convolution operation)."""
from . import E

ENTRIES = [
    # ------------------------------------------------------------------ NumPy, sections 0-7
    E("np.set_printoptions", "np.set_printoptions(precision=2, suppress=True)",
      """Changes how NumPy prints arrays for the rest of the session: precision=2 shows two decimals,
      suppress=True writes tiny numbers as 0. instead of in scientific notation (2e-05). Only the
      display changes, never the values.""",
      """
      >>> a = np.array([1 / 3, 2e-5, 12.5])
      >>> print(a)
      [3.33333333e-01 2.00000000e-05 1.25000000e+01]
      >>> np.set_printoptions(precision=2, suppress=True); print(a)
      [ 0.33  0.   12.5 ]
      >>> np.set_printoptions(precision=8, suppress=False)      # NumPy's defaults again
      """),

    E("Tensor.min", "x.min() / x.min(dim)",
      """x.min() is the smallest element (a 0-dimensional tensor); x.min(dim) works along one
      dimension and returns two tensors, the minimum values and their positions. NumPy arrays have
      the same method: arr.min() and arr.min(axis=...), which returns only the values.""",
      """
      >>> x = torch.tensor([[4, 1, 6], [0, 9, 2]])
      >>> x.min()
      tensor(0)
      >>> values, idx = x.min(dim=1)             # per row: the smallest value and where it is
      >>> values, idx
      (tensor([1, 0]), tensor([1, 0]))
      """,
      note="With dim the result is a pair (values, indices), not a tensor; see Tensor.max.",
      see=["Tensor.max", "Tensor.argmax"]),

    E("Tensor.max", "x.max() / x.max(dim)",
      """x.max() is the largest element (a 0-dimensional tensor); x.max(dim) works along one
      dimension and returns two tensors, the maximum values and their positions (indices). NumPy
      arrays have the same method: arr.max() and arr.max(axis=...), which returns only the values.""",
      """
      >>> x = torch.tensor([[1, 5, 2], [7, 0, 3]])
      >>> x.max()
      tensor(7)
      >>> values, idx = x.max(dim=1)             # per row: the largest value and where it is
      >>> values, idx
      (tensor([5, 7]), tensor([1, 0]))
      >>> np.array([[1, 5, 2], [7, 0, 3]]).max(axis=1)
      array([5, 7])
      """,
      note="""With dim the result is a pair, not a tensor, so x.max(dim=1) + 1 fails: take
      x.max(dim=1).values, or use x.amax(dim) for the values only and x.argmax(dim) for the
      positions.""",
      see=["Tensor.argmax", "Tensor.min", "np.max"]),

    E("Tensor.std", "x.std(dim=None, correction=1)",
      """The standard deviation of all elements, or along dim. PyTorch divides by N - 1 by default
      (the sample standard deviation), while NumPy's arr.std(axis) divides by N. Used for the
      per-channel statistics of a normalisation.""",
      """
      >>> x = torch.tensor([1., 2., 3., 4.])
      >>> x.std()
      tensor(1.2910)
      >>> x.std(correction=0)                    # divide by N, like NumPy
      tensor(1.1180)
      >>> float(np.array([1., 2., 3., 4.]).std())
      1.118033988749895
      """,
      note="""PyTorch and NumPy give different numbers for the same data unless you pass correction=0
      (PyTorch) or ddof=1 (NumPy). The difference only matters for small N, not for the millions of
      pixels of a dataset.""",
      see=["Tensor.mean"]),

    E("np.array", "np.array(data, dtype=None)",
      """Creates a NumPy array from a (nested) list, copying the data. The dtype is inferred:
      integers give int64, any float gives float64; dtype=float gives floats from integer
      literals.""",
      """
      >>> x = np.array([[1, 2, 0], [0, 1, 2]], dtype=float)
      >>> x.shape, x.dtype
      ((2, 3), dtype('float64'))
      >>> a = np.array([1, 2, 3])                # integers: int64
      >>> a[0] = 0.7; a                          # a float stored into an int array is truncated
      array([0, 2, 3])
      """,
      see=["torch.tensor", "np.zeros"]),

    E("np.full_like", "np.full_like(a, fill_value)",
      """A new array with the same shape and dtype as a, with every element set to fill_value. The
      course fills one with np.nan to mean "not computed yet": Matplotlib's imshow leaves NaN pixels
      blank.""",
      """
      >>> out = np.zeros((2, 3))
      >>> np.full_like(out, np.nan)
      array([[nan, nan, nan],
             [nan, nan, nan]])
      >>> np.full_like(out, 7).dtype             # the dtype follows a, not the fill value
      dtype('float64')
      """,
      note="Because the dtype is copied, the full_like of an integer array cannot hold NaN: pass dtype=float.",
      see=["np.zeros", "np.nan_to_num"]),

    E("np.nan_to_num", "np.nan_to_num(a, nan=0.0)",
      """Returns a copy of a with every NaN replaced by 0 (or by the value given as nan=), and
      infinities by very large finite numbers. Used before code that cannot handle NaN, such as
      writing the values into a figure.""",
      """
      >>> np.nan_to_num(np.array([1.5, np.nan, 3.0]))
      array([1.5, 0. , 3. ])
      >>> np.nan_to_num(np.array([np.nan]), nan=-1.0)
      array([-1.])
      """),

    E("np.zeros", "np.zeros(shape, dtype=float)",
      """A new array of the given shape filled with 0.0 (float64). The standard way to allocate an
      output array before filling it in a loop.""",
      """
      >>> out = np.zeros((2, 3))
      >>> out[0, 1] = 5; out
      array([[0., 5., 0.],
             [0., 0., 0.]])
      >>> np.zeros(2, 3)
      Traceback (most recent call last):
      ...
      TypeError: Cannot interpret '3' as a data type
      """,
      note="""The shape is one tuple, np.zeros((Ho, Wo)), unlike torch.zeros(Ho, Wo), which takes the
      sizes as separate arguments.""",
      see=["torch.zeros", "np.full_like"]),

    E("np.sum", "np.sum(a, axis=None)",
      """Adds up all elements of an array (one number), or only along axis. In the convolution
      exercises, np.sum(patch * k) multiplies a patch with the kernel element by element and adds
      everything up: one output value.""",
      """
      >>> patch = np.array([[1, 2], [3, 4]])
      >>> k = np.array([[1, 0], [0, -1]])
      >>> int(np.sum(patch * k))                 # 1*1 + 2*0 + 3*0 + 4*(-1)
      -3
      >>> np.sum(patch, axis=0)                  # down each column
      array([4, 6])
      """,
      note="patch * k is the element-wise product; patch @ k would be a matrix product, a different thing.",
      see=["Tensor.sum", "np.max"]),

    E("np.percentile", "np.percentile(a, q)",
      """The value below which q percent of the elements lie (q from 0 to 100; q=50 is the median).
      The course uses the 1st and 99th percentiles as the colour range of imshow, so that a few
      extreme pixels do not wash out the picture.""",
      """
      >>> a = np.arange(101)                     # 0, 1, ..., 100
      >>> np.percentile(a, [1, 50, 99])
      array([ 1., 50., 99.])
      """),

    E("np.arange", "np.arange(stop) / np.arange(start, stop, step, dtype=None)",
      """Like Python's range, as a NumPy array: from start (default 0) up to but not including stop.
      With reshape, the quickest way to build a small test array with distinct values.""",
      """
      >>> np.arange(5)
      array([0, 1, 2, 3, 4])
      >>> np.arange(6, dtype=float).reshape(2, 3)
      array([[0., 1., 2.],
             [3., 4., 5.]])
      """,
      see=["torch.arange"]),

    E("np.random.default_rng", "rng = np.random.default_rng(seed)",
      """Creates a NumPy random number generator; the same seed gives the same numbers on every run.
      Draw from it with rng.normal(size=...) (normal distribution), rng.random(size) (uniform in
      [0, 1)), rng.integers(low, high, size) or rng.permutation(n).""",
      """
      >>> rng = np.random.default_rng(0)
      >>> rng.normal(size=(16, 64, 64)).shape
      (16, 64, 64)
      >>> np.random.default_rng(0).integers(0, 10, size=5)      # same seed, same numbers
      array([8, 6, 5, 2, 3])
      """,
      note="""It is independent of torch.manual_seed: seeding PyTorch does not make NumPy's numbers
      repeatable, nor the other way round.""",
      see=["torch.manual_seed"]),

    E("np.abs", "np.abs(a)",
      """The absolute value of every element. np.abs(a - b).max() is the largest difference between
      two arrays: the course's check that two implementations compute the same numbers (up to tiny
      rounding differences).""",
      """
      >>> a, b = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 2.0])
      >>> np.abs(a - b)
      array([0. , 0.5, 1. ])
      >>> float(np.abs(a - b).max())
      1.0
      """),

    E("np.max", "np.max(a, axis=None)",
      """The largest element of an array, or the largest along axis, which may be a tuple:
      axis=(-2, -1) reduces the last two dimensions, for example each k x k window of a pooling
      layer. np.mean, np.sum and np.min take the same axis argument.""",
      """
      >>> x = np.arange(8).reshape(2, 2, 2)      # two 2x2 windows
      >>> np.max(x, axis=(-2, -1))
      array([3, 7])
      >>> np.mean(x, axis=(-2, -1))
      array([1.5, 5.5])
      """,
      see=["Tensor.max", "np.sum"]),

    E("Tensor.transpose", "x.transpose(dim0, dim1)",
      """Swaps two dimensions of a tensor without copying (permute for exactly two axes); x.T swaps
      the two axes of a matrix. NumPy arrays have a method of the same name that takes the whole
      new order instead, like permute: img.transpose(2, 0, 1) turns an (H, W, C) array into
      (C, H, W).""",
      """
      >>> x = torch.zeros(2, 3, 4)
      >>> x.transpose(0, 2).shape
      torch.Size([4, 3, 2])
      >>> img = np.zeros((32, 48, 3))            # a NumPy image (H, W, C)
      >>> img.transpose(2, 0, 1).shape           # NumPy: the full new order
      (3, 32, 48)
      """,
      note="On a tensor, x.transpose(2, 0, 1) is an error (it takes exactly two dimensions): use x.permute(2, 0, 1).",
      see=["Tensor.permute"]),

    # ------------------------------------------------------------------ PyTorch, section 8
    E("nn.Module", "class Net(nn.Module): __init__ and forward",
      """The base class of every model and layer. Write a model as a subclass: create its layers in
      __init__ (after super().__init__()) as attributes, which registers their parameters, and
      write the computation in forward(self, x). Then call model(x), which runs forward.""",
      """
      >>> class Tiny(nn.Module):
      ...     def __init__(self):
      ...         super().__init__()
      ...         self.fc = nn.Linear(4, 2)
      ...     def forward(self, x):
      ...         return F.relu(self.fc(x))
      >>> model = Tiny()
      >>> model(torch.zeros(3, 4)).shape
      torch.Size([3, 2])
      >>> sum(p.numel() for p in model.parameters())     # the weights of self.fc: 4 * 2 + 2
      10
      """,
      note="""Call model(x), not model.forward(x): the call also runs the hooks that tools such as
      torchkit.summarize rely on. Layers kept in a plain Python list are not registered (they are
      not trained, not moved by .to(device)); use nn.Sequential or nn.ModuleList.""",
      see=["nn.Sequential", "Module.parameters"]),

    E("nn.Conv2d", "nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True)",
      """A convolution layer: out_channels learned kernels of size (in_channels, k, k), each slid over
      the input exactly like this notebook's cross-correlation (no kernel flip). Input
      (B, C_in, H, W) -> output (B, C_out, H_out, W_out), with H_out from the output-size formula.
      The weight has shape (C_out, C_in, k, k), the bias (C_out,).""",
      """
      >>> conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
      >>> conv.weight.shape, conv.bias.shape
      (torch.Size([8, 3, 3, 3]), torch.Size([8]))
      >>> conv(torch.zeros(2, 3, 28, 28)).shape                 # padding 1 keeps 28 x 28
      torch.Size([2, 8, 28, 28])
      >>> nn.Conv2d(3, 8, 3, stride=2)(torch.zeros(2, 3, 28, 28)).shape     # (28 - 3) // 2 + 1 = 13
      torch.Size([2, 8, 13, 13])
      """,
      note="""The input must be channels-first, with in_channels channels: a (B, H, W, C) batch fails
      with "expected input ... to have 3 channels, but got 28 channels instead"; permute it first.""",
      see=["nn.MaxPool2d", "Tensor.permute"]),

    E("nn.MaxPool2d", "nn.MaxPool2d(kernel_size, stride=None)",
      """Max pooling: keeps the largest value of every kernel_size x kernel_size window, separately
      for each channel; it has no parameters. The stride defaults to kernel_size, so
      nn.MaxPool2d(2) halves the height and width: (B, C, 28, 28) -> (B, C, 14, 14).""",
      """
      >>> x = torch.arange(16.).reshape(1, 1, 4, 4)
      >>> nn.MaxPool2d(2)(x)
      tensor([[[[ 5.,  7.],
                [13., 15.]]]])
      >>> nn.MaxPool2d(2)(torch.zeros(1, 64, 7, 7)).shape       # odd sizes are rounded down
      torch.Size([1, 64, 3, 3])
      """,
      see=["nn.Conv2d"]),

    E("F.relu", "F.relu(x)",
      """The rectified linear unit, element by element: negative values become 0, the others pass
      unchanged. The function form, for use inside forward(); nn.ReLU() is the same operation as a
      layer, for nn.Sequential.""",
      """
      >>> F.relu(torch.tensor([-2., -0.5, 0., 1.5]))
      tensor([0.0000, 0.0000, 0.0000, 1.5000])
      """,
      see=["nn.ReLU"]),

    E("torchkit.summarize", "print(torchkit.summarize(model, input_size))",
      """Course helper. Pushes one dummy (all-zero) image of input_size, (C, H, W), through the model
      and returns a table as a string: one row per layer with its name, type, output shape, number
      of parameters and MACs (multiply-accumulates, counted for conv and linear layers), then the
      totals. The model's train / eval mode is restored afterwards.""",
      """
      >>> net = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
      ...                     nn.Flatten(), nn.Linear(4 * 14 * 14, 10))
      >>> print(torchkit.summarize(net, (1, 28, 28)))
      layer  type       output shape          params            MACs
      --------------------------------------------------------------
      0      Conv2d     (1, 4, 28, 28)            40          28,224
      1      ReLU       (1, 4, 28, 28)             0               0
      2      MaxPool2d  (1, 4, 14, 14)             0               0
      3      Flatten    (1, 784)                   0               0
      4      Linear     (1, 10)                7,850           7,840
      --------------------------------------------------------------
      total params: 7,890   trainable: 7,890   MACs per image (conv + linear): 36,064 (~0.00 G)
      """,
      see=["torchkit.count_parameters"]),

    E("torch.optim.AdamW", "torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)",
      """Adam with decoupled weight decay, the default optimiser of modern training. Adam divides
      each parameter's step by a running average of its own gradient size, so parameters move at a
      similar pace whatever the scale of their gradients; AdamW also shrinks every weight by the
      fraction lr * weight_decay at each step, which keeps the weights small and reduces
      overfitting. Used exactly like SGD.""",
      """
      >>> w = torch.tensor([1.0, 1.0], requires_grad=True)
      >>> opt = torch.optim.AdamW([w], lr=0.1, weight_decay=0.0)
      >>> (w * torch.tensor([100.0, 0.01])).sum().backward()   # gradients 100 and 0.01
      >>> opt.step(); w                          # the first step is about lr for both
      tensor([0.9000, 0.9000], requires_grad=True)
      """,
      see=["torch.optim.SGD", "torch.optim.lr_scheduler.OneCycleLR"]),

    E("torch.optim.lr_scheduler.OneCycleLR", "OneCycleLR(optimizer, max_lr, total_steps, pct_start=0.3)",
      """A learning-rate schedule for the whole run: the rate starts at max_lr / 25, rises to max_lr
      during the first pct_start of total_steps, then falls along a cosine to almost zero. It sets
      the optimiser's lr itself and must be stepped after every batch (scheduler.step();
      torchkit.fit does that), so total_steps = epochs * len(train_loader).""",
      """
      >>> opt = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-3)
      >>> sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.0, total_steps=10, pct_start=0.3)
      >>> lrs = []
      >>> for step in range(10):
      ...     lrs.append(round(opt.param_groups[0]["lr"], 2))
      ...     opt.step(); sched.step()
      >>> lrs
      [0.04, 0.52, 1.0, 0.95, 0.81, 0.61, 0.39, 0.19, 0.05, 0.0]
      """,
      note="""Stepping it more often than total_steps raises an error, so total_steps must match the
      number of batches you train on.""",
      see=["torch.optim.AdamW", "torchkit.fit"]),

    E("torchkit.load_checkpoint", "extras = torchkit.load_checkpoint(model, path)",
      """Course helper. Loads the weights saved in path (by torchkit.fit(checkpoint=path) or
      torchkit.save_checkpoint) into model, in place, and returns everything else stored in the
      file as a dict; for a fit checkpoint "epoch", "val_loss", "val_metric" and the whole
      "history". The course keeps checkpoints in torchkit.CHECKPOINTS, which is data/checkpoints/.""",
      note="""The model must be built with exactly the same layers as the one that was saved;
      otherwise loading fails with an error that lists the missing, unexpected or wrongly sized
      weights.""",
      see=["torchkit.fit"]),

    E("nn.CrossEntropyLoss", "loss_fn = nn.CrossEntropyLoss(); loss = loss_fn(logits, targets)",
      """The classification loss. logits are the raw scores of the model, (B, num_classes); targets
      the correct class index of every sample, an int64 tensor (B,). It applies the softmax itself
      and returns the mean of -log(probability of the correct class). Segmentation uses it per
      pixel: logits (B, C, H, W), targets (B, H, W).""",
      """
      >>> logits = torch.tensor([[2.0, 0.0, -1.0],     # sample 0: class 0 scores highest
      ...                        [0.0, 0.0, 0.0]])     # sample 1: no preference
      >>> targets = torch.tensor([0, 2])
      >>> nn.CrossEntropyLoss(reduction="none")(logits, targets)   # per sample; a uniform guess costs log(3)
      tensor([0.1698, 1.0986])
      >>> nn.CrossEntropyLoss()(logits, targets)                   # the default: their mean
      tensor(0.6342)
      """,
      note="""Give it raw logits, not softmax outputs: applying softmax twice raises no error but trains
      badly. The model should therefore end with a Linear layer, without softmax.""",
      see=["Tensor.argmax", "nn.MSELoss"]),

    E("nn.Sequential", "nn.Sequential(layer1, layer2, ...)",
      """A model made of layers applied one after the other, the output of each being the input of
      the next. Saves writing a class when forward() would only chain layers. The layers can be read
      back by position: model[0].""",
      """
      >>> model = nn.Sequential(nn.Flatten(), nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
      >>> model(torch.zeros(5, 2, 2)).shape      # (5, 2, 2) -> (5, 4) -> (5, 3) -> (5, 2)
      torch.Size([5, 2])
      >>> model[1]
      Linear(in_features=4, out_features=3, bias=True)
      """,
      see=["nn.Module"]),

    E("nn.Flatten", "nn.Flatten(start_dim=1)",
      """A layer that flattens everything except the batch dimension: (B, C, H, W) -> (B, C*H*W).
      Goes between convolutional layers and an nn.Linear, or in front of a dense network that reads
      raw pixels.""",
      """
      >>> nn.Flatten()(torch.zeros(8, 1, 28, 28)).shape
      torch.Size([8, 784])
      """,
      see=["Tensor.flatten", "nn.Linear"]),

    E("nn.ReLU", "nn.ReLU()",
      """The ReLU activation as a layer (a module): max(x, 0) element by element, no parameters. Used
      inside nn.Sequential; in a forward() you can write F.relu(x) instead.""",
      """
      >>> nn.ReLU()(torch.tensor([-2., 0., 3.]))
      tensor([0., 0., 3.])
      """,
      see=["F.relu", "nn.Sequential"]),

    E("torchkit.evaluate", "loss, metric = torchkit.evaluate(model, loader, loss_fn=None, metric_fn=None, device=None)",
      """Course helper. Runs the model over every batch of loader in evaluation mode and without
      gradients, and returns two floats: the average loss and the average metric, weighted by batch
      size. metric_fn(outputs, targets) defaults to accuracy when the outputs are class scores with
      integer targets (NaN otherwise); the model's train / eval mode is restored afterwards.""",
      """
      >>> model = nn.Linear(2, 2)
      >>> with torch.no_grad():
      ...     _ = model.weight.copy_(torch.eye(2)); _ = model.bias.zero_()     # logits = inputs
      >>> X, Y = torch.tensor([[2., 0.], [0., 2.], [2., 1.]]), torch.tensor([0, 1, 1])
      >>> loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=2)
      >>> loss, acc = torchkit.evaluate(model, loader, nn.CrossEntropyLoss(), torchkit.accuracy)
      >>> round(loss, 3), round(acc, 3)          # the third sample is predicted as class 0
      (0.522, 0.667)
      """,
      see=["torchkit.accuracy", "torchkit.fit"]),

    E("torchkit.count_parameters", "torchkit.count_parameters(model, trainable_only=False)",
      """Course helper. The total number of parameters of a model, sum(p.numel() for p in
      model.parameters()); trainable_only=True counts only those with requires_grad (useful once
      layers are frozen for fine-tuning).""",
      """
      >>> torchkit.count_parameters(nn.Linear(3, 2))        # 3 * 2 weights + 2 biases
      8
      >>> torchkit.count_parameters(nn.Conv2d(1, 32, 3))    # 32 * 1 * 3 * 3 + 32
      320
      """,
      see=["Module.parameters", "torchkit.summarize"]),

    E("torchkit.accuracy", "torchkit.accuracy(logits, targets)",
      """Course helper. The fraction of samples whose highest-scoring class, logits.argmax(1), equals
      the target, as a Python float. Works for logits (B, C) with targets (B,), and per pixel for
      (B, C, H, W) with (B, H, W); passed to torchkit.evaluate as the metric.""",
      """
      >>> logits = torch.tensor([[2., 1.], [0., 3.], [1., 0.]])     # predicted classes 0, 1, 0
      >>> torchkit.accuracy(logits, torch.tensor([0, 1, 1]))
      0.6666666865348816
      """,
      see=["Tensor.argmax", "torchkit.evaluate"]),

    E("Tensor.detach", "x.detach()",
      """Returns the same data as a tensor that autograd no longer tracks (requires_grad=False),
      without copying. Needed to turn a weight or a model output into NumPy, or to keep it without
      keeping its whole computation alive.""",
      """
      >>> w = nn.Linear(2, 1).weight             # a parameter: requires_grad=True
      >>> w.numpy()
      Traceback (most recent call last):
      ...
      RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
      >>> w.detach().numpy().shape
      (1, 2)
      """,
      note="The result shares memory with x: changing it in place changes x too. For a separate copy add .clone().",
      see=["Tensor.cpu", "Tensor.numpy", "torch.no_grad"]),

    E("Tensor.cpu", "x.cpu()",
      """Returns a copy of the tensor in main memory (the CPU), or the tensor itself if it is already
      there. Needed before .numpy() or plotting when the result was computed on a GPU:
      x.detach().cpu().numpy().""",
      """
      >>> x = torch.ones(2)                      # on a GPU machine: torch.ones(2, device="cuda")
      >>> x.cpu().device
      device(type='cpu')
      >>> x.cpu() is x                           # already on the CPU: nothing to copy
      True
      """,
      see=["Tensor.to", "Tensor.numpy", "Tensor.detach"]),

    E("Tensor.argmax", "x.argmax(dim)",
      """The position of the largest value along dim. For logits (B, num_classes), x.argmax(dim=1)
      is the predicted class of every sample, shape (B,). Without dim it gives the position in the
      flattened tensor.""",
      """
      >>> logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 0.5]])     # B=2 samples, 3 classes
      >>> logits.argmax(dim=1)                   # predicted class per sample
      tensor([1, 0])
      >>> logits.argmax()                        # no dim: index into the 6 flattened values
      tensor(3)
      """,
      note="Softmax does not change which value is largest, so predictions are read off the logits directly.",
      see=["Tensor.max", "torchkit.accuracy"]),
]
