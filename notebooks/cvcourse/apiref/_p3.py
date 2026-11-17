"""Reference entries for functions first used in P3 (GoogLeNet step by step).

Same format as `_day0.py`. Entries whose key already exists in an earlier file are ignored (the earliest
wording wins), so only functions no earlier notebook explains are here.
"""
from . import E

ENTRIES = [
    E("Module.add_module", 'model.add_module(name, layer)',
      """Adds a layer (or a whole block) to a model under a name of your choice, as if you had written
      self.name = layer. On an empty nn.Sequential() it appends the layer at the end, so a network can be built
      line by line with readable names; the name then shows up in print(model), in drawings and in
      model.name.""",
      """
      >>> blocks = nn.Sequential()
      >>> blocks.add_module("conv", nn.Conv2d(3, 8, kernel_size=3))
      >>> blocks.add_module("relu", nn.ReLU())
      >>> blocks
      Sequential(
        (conv): Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1))
        (relu): ReLU()
      )
      >>> blocks.conv.out_channels
      8
      """,
      note="""The name must be new in that model and must not contain a dot. Layers run in the order they were
      added.""",
      see=["nn.Sequential"]),

    E("torchvision.models.googlenet", "torchvision.models.googlenet(weights=None, aux_logits=False, init_weights=False)",
      """Builds torchvision's GoogLeNet (Inception v1): input (N, 3, 224, 224), output (N, 1000) scores.
      weights=None gives random weights and downloads nothing; aux_logits=False leaves out the two auxiliary
      classifiers; init_weights=False keeps PyTorch's default starting weights (and avoids a warning). It differs
      from the paper in two details: a batch normalisation after every convolution, and 3x3 kernels where the paper
      has 5x5.""",
      """
      >>> net = torchvision.models.googlenet(weights=None, aux_logits=False, init_weights=False)
      >>> net(torch.zeros(1, 3, 224, 224)).shape
      torch.Size([1, 1000])
      >>> sum(p.numel() for p in net.parameters())
      6624904
      """,
      see=["torchvision.models.vgg16"]),

    E("nn.init.kaiming_normal_", 'nn.init.kaiming_normal_(tensor, nonlinearity="relu")',
      """Fills a weight tensor in place with random numbers whose spread suits a layer followed by a ReLU ("He
      initialisation", He et al. 2015): standard deviation sqrt(2 / fan_in), where fan_in = in channels x kernel
      height x kernel width. Without batch normalisation, PyTorch's default starting weights are smaller, and the
      signal shrinks from layer to layer, which makes a deep network start learning slowly.""",
      """
      >>> conv = nn.Conv2d(64, 32, kernel_size=3)       # fan_in = 64 x 3 x 3 = 576
      >>> _ = torch.manual_seed(0)
      >>> _ = nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
      >>> round(conv.weight.std().item(), 3), round((2 / 576) ** 0.5, 3)
      (0.059, 0.059)
      """,
      note="""The trailing underscore means "in place": the tensor you pass is changed, nothing new is created.
      Call it before training, right after building the model.""",
      see=["nn.init.normal_", "Module.modules"]),
]
