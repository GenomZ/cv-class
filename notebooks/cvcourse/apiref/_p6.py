"""Reference entries for functions first used in P6 (GoogLeNet with auxiliary classifiers).

Same format as `_day0.py`. Entries whose key already exists in an earlier file are ignored (the earliest
wording wins), so only functions no earlier notebook explains are here.
"""
from . import E

ENTRIES = [
    E("Module.training", "model.training   (inside forward: self.training)",
      """True or False: is the network in training mode? A new network starts in training mode (True);
      model.eval() sets it to False in the network and in every layer inside it, model.train() back to True.
      Layers such as nn.Dropout read it to decide what to do. Your own forward can read it too, as
      self.training, to do something only while training: GoogLeNet computes its auxiliary classifiers only
      when self.training is True.""",
      """
      >>> net = nn.Sequential(nn.Linear(4, 2), nn.Dropout(p=0.5))
      >>> net.training                       # a new network starts in training mode
      True
      >>> _ = net.eval()
      >>> net.training, net[1].training      # the network and every layer inside it
      (False, False)
      >>> _ = net.train()
      >>> net.training
      True
      """,
      note="""It is an attribute, not a function: write self.training, without brackets. It does not change
      by itself: P.train and P.evaluate switch it for you; in your own code call model.train() before training and
      model.eval() before predicting.""",
      see=["Module.train", "Module.eval", "nn.Dropout"]),
]
