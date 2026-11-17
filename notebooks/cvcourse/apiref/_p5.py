"""Reference entries for functions first used in P5 (transfer learning on a few pet breeds).

Same format as `_day0.py`. Entries whose key already exists in an earlier file are ignored (the earliest
wording wins), so P5 reuses the D1-02 / D1-04 entries for resnet18, state_dict, torch.save, ... and adds
only what no earlier notebook explains.
"""
from . import E

ENTRIES = [
    E("PIL.Image.open", 'photo = Image.open(path).convert("RGB")',
      """Opens an image file (JPEG, PNG, ...) as a PIL image; `from PIL import Image` first. PIL (the
      "Pillow" library) is what torchvision uses to read photos. .convert("RGB") makes sure the photo has
      exactly 3 colour channels: a few photos are stored in grey (1 channel) or with a transparency
      channel (4), and a network trained on colour photos needs 3. photo.size is (width, height).""",
      """
      >>> from PIL import Image
      >>> import io
      >>> file = io.BytesIO()                                # an in-memory file; in practice a path "pug_1.jpg"
      >>> Image.new("L", (40, 30)).save(file, format="PNG")  # a grey 40 x 30 picture
      >>> photo = Image.open(file)
      >>> photo.mode, photo.size                             # 1 grey channel; (width, height)
      ('L', (40, 30))
      >>> photo.convert("RGB").mode                          # now 3 colour channels
      'RGB'
      """,
      note="""PIL sizes are (width, height), tensors are (channels, height, width): v2.ToImage() does the
      conversion.""",
      see=["torchvision.transforms.v2.ToImage", "torchvision.datasets.ImageFolder"]),

    E("torchvision.models.ResNet18_Weights", "torchvision.models.ResNet18_Weights.IMAGENET1K_V1",
      """Names one set of trained weights for ResNet-18: IMAGENET1K_V1 are the weights learned on ImageNet
      (1.28 million photos, 1 000 classes); .DEFAULT is the recommended set (here the same one). Passing it
      to torchvision.models.resnet18(weights=...) downloads the file once (about 45 MB) into PyTorch's
      cache folder and loads it. The object also knows the preparation the weights expect
      (.transforms()) and the 1 000 class names (.meta["categories"]); reading those downloads nothing.
      Every torchvision model has such a class: ResNet50_Weights, MobileNet_V3_Large_Weights, ...""",
      """
      >>> w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
      >>> len(w.meta["categories"]), w.meta["categories"][:3]
      (1000, ['tench', 'goldfish', 'great white shark'])
      >>> w.transforms().mean, w.transforms().std        # the normalisation the weights expect
      ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      >>> torchvision.models.ResNet18_Weights.DEFAULT == w
      True
      """,
      note="""weights=None gives the same network with random weights and downloads nothing: use it to rebuild
      a network whose trained weights you load yourself with load_state_dict.""",
      see=["torchvision.models.resnet18", "Module.load_state_dict"]),

    E("torchvision.datasets.ImageFolder", 'ds = torchvision.datasets.ImageFolder(root, transform=None)',
      """A dataset of your own photos, sorted into one sub-folder per class: root/cat/*.jpg, root/dog/*.jpg,
      ... ds.classes lists the class names (the folder names, in alphabetical order); ds[i] is (image,
      class index), with `transform` (for example a v2.Compose of resize, ToImage, ToDtype, Normalize)
      applied to the image as it is read. Put training and validation photos in two separate root folders,
      each with the same class sub-folders, and wrap each dataset in a DataLoader.""",
      """
      >>> import pathlib, tempfile
      >>> from PIL import Image
      >>> root = pathlib.Path(tempfile.mkdtemp())
      >>> for name in ["dog", "cat"]:
      ...     (root / name).mkdir()
      ...     Image.new("RGB", (8, 6)).save(root / name / "photo1.jpg")
      >>> ds = torchvision.datasets.ImageFolder(root)
      >>> ds.classes, len(ds)                 # alphabetical: cat = class 0, dog = class 1
      (['cat', 'dog'], 2)
      >>> image, label = ds[1]
      >>> image.size, label                   # a PIL image (no transform given) and its class index
      ((8, 6), 1)
      >>> import shutil; shutil.rmtree(root)  # tidy up the example folder
      """,
      see=["torch.utils.data.DataLoader", "torchvision.transforms.v2.Compose", "PIL.Image.open"]),
]
