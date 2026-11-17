"""Reference entries for functions first used in P7 (object detection with boxes).

Same format as `_day0.py`. Entries whose key already exists in an earlier file are ignored (the earliest
wording wins), so only functions no earlier notebook explains are here. torchvision.ops.nms and
fasterrcnn_resnet50_fpn_v2 are explained in `_day2.py` and reused from there.
"""
from . import E

ENTRIES = [
    E("nn.Sigmoid", "nn.Sigmoid()",
      """A layer without parameters that squeezes every number into the range 0 to 1:
      sigmoid(x) = 1 / (1 + e^-x). Large negative numbers become almost 0, 0 becomes 0.5, large positive numbers
      almost 1. At the end of a box head it keeps the predicted box numbers (fractions of the picture's width and
      height) inside the picture, whatever the layers before it compute.""",
      """
      >>> squash = nn.Sigmoid()
      >>> squash(torch.tensor([-4.0, 0.0, 4.0]))
      tensor([0.0180, 0.5000, 0.9820])
      >>> head = nn.Sequential(nn.Linear(8, 4), nn.Sigmoid())
      >>> out = head(torch.randn(2, 8) * 100)                # even very large inputs
      >>> out.shape, bool(((out >= 0) & (out <= 1)).all())
      (torch.Size([2, 4]), True)
      """,
      note="""Not for the class scores of a classifier: cross-entropy wants the raw scores of the last
      nn.Linear. torch.sigmoid(x) is the same computation as a function instead of a layer.""",
      see=["torch.sigmoid", "nn.ReLU"]),

    E("torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights",
      "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
      """Describes the pretrained COCO weights of torchvision's Faster R-CNN (ResNet-50 FPN v2): pass it as
      weights=... to fasterrcnn_resnet50_fpn_v2 to build the detector with them (downloaded once, about 167 MB,
      into PyTorch's cache folder). Its .meta["categories"] lists the 91 class names, index = label number of the
      detector's output (0 is the background, "N/A" marks numbers COCO does not use).""",
      """
      >>> from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
      >>> weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
      >>> categories = weights.meta["categories"]
      >>> len(categories), categories[1], categories[18]
      (91, 'person', 'dog')
      >>> weights.url.split("/")[-1]                          # the file name in the cache folder
      'fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'
      """,
      note="""Looking at the weights object downloads nothing; only building the detector with weights=... reads
      (or downloads) the file.""",
      see=["torchvision.models.detection.fasterrcnn_resnet50_fpn_v2"]),
]
