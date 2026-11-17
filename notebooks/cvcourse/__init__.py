"""Shared helpers for the Deep Learning for Computer Vision course notebooks.

Everything here is NumPy / Matplotlib only, so it runs regardless of the
deep-learning framework chosen for the model-training cells.

Modules
-------
viz      : plotting helpers (image grids, boxes, masks, overlays)
convops  : reference convolution / pooling / receptive-field maths
boxes    : box formats, IoU family, NMS, anchors, matching, encoding, AP
seg      : segmentation metrics and losses (Dice, IoU), upsampling references
shapes   : synthetic "shapes" dataset with boxes and masks (no download)
tests    : checkers for the exercises (`tests.check_iou(my_iou)` etc.)
labs     : interactive browser labs embedded in the notebooks (`labs.show("conv-playground")`)
practical: the practical track's toolkit (PyTorch): `from cvcourse import practical as P`
"""

from . import viz, convops, boxes, seg, shapes, tests, labs, apiref, practical  # noqa: F401

__all__ = ["viz", "convops", "boxes", "seg", "shapes", "tests", "labs", "apiref", "practical"]
