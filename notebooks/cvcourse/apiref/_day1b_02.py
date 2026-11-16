"""Reference entries for functions first used in D1-02 (training recipe) that `_day1b.py` does not cover.

Same format as `_day1b.py`; registered in `apiref._MODULES` after `_day1b`, so an entry with the same key
in an earlier file wins.
"""
from . import E

ENTRIES = [
    E("nn.Dropout", "nn.Dropout(p=0.5)",
      """A layer that, in train mode, sets each number passing through it to 0 with probability p and
      multiplies the survivors by 1 / (1 - p), so that the average stays the same. In eval mode
      (model.eval()) it passes everything through unchanged. It has no parameters. D1-02 puts one before
      PetNet's last linear layer as a regulariser.""",
      """
      >>> _ = torch.manual_seed(0)
      >>> drop = nn.Dropout(p=0.5)
      >>> drop.train()(torch.ones(8))             # train mode: 0, or 1 / (1 - 0.5) = 2
      tensor([0., 0., 2., 0., 0., 0., 2., 2.])
      >>> drop.eval()(torch.ones(8))              # eval mode: unchanged
      tensor([1., 1., 1., 1., 1., 1., 1., 1.])
      """,
      note="""Which numbers are dropped is random, so a model in train mode gives different outputs for the
      same input. Call model.eval() before validating or predicting.""",
      see=["Module.train", "Module.eval", "nn.BatchNorm2d"]),
]
