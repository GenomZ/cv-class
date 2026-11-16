# Function reference and "Functions used here" boxes

**Why.** Students meet `unsqueeze`, `permute`, `F.interpolate`, `torchkit.fit` in a code cell and
have to guess what they do. Every library function a notebook calls is now explained in a
fold-out box placed before the cell that first uses it: what it does, a runnable example,
the classic mistake. Closed by default, so it does not get in the way of students who know it.

## The pieces

| Piece | Where |
|---|---|
| Entries (one per function, written once) | `notebooks/cvcourse/apiref/_day0.py`, `_day1a.py` (D1-01), `_day1b.py` (D1-02..04), `_day2.py`, `_day3.py` |
| Box in a notebook | `api("Tensor.permute", "torch.stack", ...)` in `tools/src/*.py` (`from nb import api`) |
| Same entries at run time | `apiref.show("permute")`, `apiref.search("dimension")`, `apiref.index()` |
| Checker | `python tools/check_api.py [pattern] [--lines] [--inventory] [--examples day2]` |

`check_api.py --inventory` lists, per notebook, the functions that notebook is the first to use:
that notebook's day file owns their entries. If two files define the same key, the earlier day wins.

## Writing an entry

```python
E("Tensor.permute", "x.permute(*dims)",
  """What it does in one to three plain sentences, and what the course uses it for.""",
  """
  >>> hwc = torch.zeros(32, 48, 3)         # H, W, C
  >>> hwc.permute(2, 0, 1).shape           # C, H, W
  torch.Size([3, 32, 48])
  """,
  note="The classic mistake, if there is one.", see=["Tensor.transpose"]),
```

* **Key** exactly as the checker prints it: `Tensor.x`, `torch.x`, `nn.X`, `F.x`, `Module.x`,
  `Optimizer.x`, `np.x`, `torchvision.ops.nms`, `torchkit.x`, ...
* **Signature** as a student types it, with the arguments that matter (`F.interpolate(x, size=None,
  scale_factor=None, mode="nearest")`), not the full official signature.
* **What**: say what goes in and what comes out, with shapes where shapes are the point. Write for
  someone who knows Python and NumPy basics but has never seen the function. No "simply", no jargon
  that is not explained elsewhere in the course.
* **Example**: a doctest of 2 to 8 lines with the real output; `torch`, `nn`, `F`, `np`,
  `torchvision`, `torchkit` are available; anything else must be imported inside the example. Show
  the shape or the values that make the behaviour visible; small tensors with easy numbers. It must
  be deterministic (`torch.manual_seed`, or avoid randomness), fast, and need no data or download.
  For models and anything that would download weights, show construction with `weights=None` or
  leave the example out and describe inputs and outputs in the text.
* **Note** only when there is a real trap (in-place vs returned copy, logits vs probabilities,
  `dim` semantics, train vs eval).
* Course helpers (`torchkit.*`): what it does for the student, what it returns; an example is optional.
* Tensor attributes that are not calls (`.shape`, `.dtype`, `.device`, `.grad`, `.requires_grad`,
  `.T`) are not found by the checker; add entries for those a notebook relies on and box them by hand.

## Placing boxes

* One box right before the first code cell or exercise of a section that uses functions not yet
  explained **in this notebook**. Title defaults to "Functions used here"; before an exercise use
  `title="Functions you will need"` when the box lists what the student should reach for.
* Functions whose first use in the course was an earlier notebook go into a second, one-line-each
  box: `api(..., brief=True)` ("Functions from earlier notebooks, used again here"). New functions
  get the full box. A notebook therefore explains everything it uses, but stays short.
* Keep boxes small (3 to 12 functions): one per section, not one giant box at the top. The setup
  cell's functions go in a box right after the setup cell's markdown intro, before the code.
* Do not change code cells, exercises, checkers or any other markdown.
* False matches of the checker (a Matplotlib `ax.scatter` reported as `Tensor.scatter`) go into
  your `tools/api_ignore_<day>.txt` with a reason; private helpers (`torchkit._x`) too.

## Done means

* `python tools/build_notebooks.py <pattern>` prints no "names without a reference entry" for
  entries you own.
* `python tools/check_api.py --examples <your day file>`: 0 failing examples.
* `python tools/check_api.py <notebook>`: `[PASS]`, or the only remaining findings are entries owned
  by another day that is being written at the same time.
* Notebook code is unchanged, so the notebooks do not need to be re-executed for this.
