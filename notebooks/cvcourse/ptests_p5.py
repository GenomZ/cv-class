"""Friendly checks for P5 (transfer learning), and the notebook's opening picture.

Import it in the notebook as::

    from cvcourse import ptests_p5 as T

    T.show_idea()                               # the picture of section 1
    T.check_frozen(model)                       # after step 2: every parameter frozen
    T.check_new_head(model, number_of_classes)  # after step 3: the new classifier
    T.check_optimizer(optimizer, model)         # the optimiser holds exactly the trainable parameters
    T.check_unfrozen(model, ["layer4"])         # after step 5: layer4 and fc trainable, everything else frozen

Every check prints one ``[PASS]`` or ``[FAIL]`` line per property, with a plain hint under a FAIL, and
never raises an error. When a step is not done yet (its blank ``____`` is still there) or is done wrongly,
the check does the step itself and says so ("done for you so you can continue"), so that the rest of the
notebook still runs. Each check returns a Verdict: true when everything passed.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["Verdict", "check_frozen", "check_new_head", "check_optimizer", "check_unfrozen", "show_idea", "idea_figure"]

_PASS = "[PASS]"
_FAIL = "[FAIL]"
_FEATURES = 512          # what ResNet-18's pooling hands to the classifier, per photo


class Verdict:
    """What a check returns: true when every property passed. Shows nothing under the cell: the check has printed."""

    def __init__(self, ok: bool):
        self.ok = bool(ok)

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        return "all checks passed" if self.ok else "some checks failed"

    def _ipython_display_(self) -> None:
        return None


def _line(ok: bool, text: str, hint: str = "") -> bool:
    print(f"{_PASS if ok else _FAIL}  {text}")
    if hint and not ok:
        print(f"        hint: {hint}")
    return ok


def _numbers(params) -> int:
    total = 0
    for p in params:
        total += p.numel()
    return total


def _trainable_numbers(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()
    return total


def _in_blocks(name: str, blocks) -> bool:
    for b in blocks:
        if name == b or name.startswith(b + "."):
            return True
    return False


def _could_not_run(err: BaseException) -> Verdict:
    print(f"{_FAIL}  the check could not run: {type(err).__name__}: {str(err).splitlines()[0] if str(err) else ''}")
    print("        hint: run the cells of this notebook from the top, in order (the model must exist)")
    return Verdict(False)


def _done_for_you(what: str) -> None:
    print(f"\ndone for you so you can continue: {what}")
    print("(fill in or correct the blank above, then run that cell and this one again to check your own line)")


# ================================================================================================ step 2
def check_frozen(model: nn.Module) -> Verdict:
    """After step 2: every parameter of the network has requires_grad = False."""
    try:
        params = list(model.named_parameters())
        total = _numbers(p for _, p in params)
        still = []
        for name, p in params:
            if p.requires_grad:
                still.append(name)
        if len(still) == len(params):
            hint = "the blank is not filled in yet: inside the loop, replace ____ by   param.requires_grad = False"
        else:
            hint = ("the line must be inside the for loop over model.parameters(), so that it runs for every "
                    "parameter:   param.requires_grad = False")
        ok1 = _line(not still,
                    f"every parameter of the network is frozen (requires_grad = False): {len(params) - len(still)} "
                    f"of {len(params)} parameter tensors"
                    + ("" if not still else f"; still trainable: '{still[0]}'"
                       + (f" and {len(still) - 1} more" if len(still) > 1 else "")),
                    hint)
        n = _trainable_numbers(model)
        ok2 = _line(n == 0, f"trainable parameters: {n:,} of {total:,} (expected 0)")
        if ok1 and ok2:
            return Verdict(True)
        for _, p in params:
            p.requires_grad = False
        _done_for_you(f"every parameter is now frozen; trainable parameters now: {_trainable_numbers(model):,}")
        return Verdict(False)
    except Exception as err:  # noqa: BLE001
        return _could_not_run(err)


# ================================================================================================ step 3
def check_new_head(model: nn.Module, number_of_classes: int) -> Verdict:
    """After step 3: model.fc is a new trainable nn.Linear(512, number_of_classes); everything else is frozen."""
    try:
        n = int(number_of_classes)
        head = getattr(model, "fc", None)
        oks = []
        if head is None:
            hint = f"the blank is not filled in yet: replace ____ by   nn.Linear(512, number_of_classes)"
        else:
            hint = (f"model.fc is a {type(head).__name__}, not a linear layer: write   "
                    f"model.fc = nn.Linear(512, number_of_classes)")
        is_linear = isinstance(head, nn.Linear)
        oks.append(_line(is_linear, "model.fc is a linear layer (nn.Linear)", hint))
        if is_linear:
            oks.append(_line(head.in_features == _FEATURES,
                             f"it takes the {_FEATURES} numbers the feature extractor gives per photo "
                             f"(in_features = {head.in_features})",
                             f"the first number must be {_FEATURES}: ResNet-18's pooling hands {_FEATURES} numbers "
                             f"to the classifier"))
            if head.out_features == 1000:
                extra = "1000 is the size of the ImageNet classifier: "
            else:
                extra = ""
            oks.append(_line(head.out_features == n,
                             f"it gives one score per class: {head.out_features} scores for {n} classes "
                             f"(out_features = {head.out_features})",
                             f"{extra}the second number must be number_of_classes, here {n}"))
            trainable_head = True
            for p in head.parameters():
                if not p.requires_grad:
                    trainable_head = False
            oks.append(_line(trainable_head, "the new classifier is trainable (requires_grad = True)",
                             "a new layer is trainable by itself: make sure the new layer is created after "
                             "freezing (run step 2, then step 3)"))
        outside = []
        for name, p in model.named_parameters():
            if p.requires_grad and not _in_blocks(name, ["fc"]):
                outside.append(name)
        oks.append(_line(not outside, "the feature extractor (everything before fc) is still frozen",
                         (f"'{outside[0]}' is trainable: run the freezing cell of step 2 again, then this step"
                          if outside else "")))
        expected = _FEATURES * n + n
        got = _trainable_numbers(model)
        oks.append(_line(got == expected,
                         f"trainable parameters: {got:,} (expected {_FEATURES} x {n} weights + {n} biases = "
                         f"{expected:,})"))
        if all(oks):
            return Verdict(True)
        # repair: the reference step 3 (with the same random start as the solution)
        for p in model.parameters():
            p.requires_grad = False
        head_ok = is_linear and head.in_features == _FEATURES and head.out_features == n
        if head_ok:
            for p in head.parameters():
                p.requires_grad = True
            what = "kept your new classifier, made it trainable and froze everything else"
        else:
            torch.manual_seed(0)
            model.fc = nn.Linear(_FEATURES, n)
            what = f"model.fc = nn.Linear({_FEATURES}, {n}), everything else frozen"
        _done_for_you(f"{what}; trainable parameters now: {_trainable_numbers(model):,}")
        return Verdict(False)
    except Exception as err:  # noqa: BLE001
        return _could_not_run(err)


# ================================================================================================ optimiser
def check_optimizer(optimizer, model: nn.Module) -> Verdict:
    """The optimiser holds every trainable parameter of the model and no frozen one."""
    try:
        held = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                held.add(id(p))
        missing, frozen_held = [], []
        n_trainable = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                n_trainable += 1
                if id(p) not in held:
                    missing.append(name)
            elif id(p) in held:
                frozen_held.append(name)
        ok1 = _line(not missing,
                    f"the optimiser holds every trainable parameter ({n_trainable} tensors)",
                    (f"'{missing[0]}' is trainable but not in the optimiser"
                     + (f" ({len(missing)} tensors are missing)" if len(missing) > 1 else "")
                     + ", so training would never change it: build a new optimiser after freezing or unfreezing, "
                       "from trainable_parameters(model)") if missing else "")
        ok2 = _line(not frozen_held, "the optimiser holds no frozen parameter",
                    (f"it also holds {len(frozen_held)} frozen tensors (the first is '{frozen_held[0]}'): give it "
                     f"trainable_parameters(model), not model.parameters()") if frozen_held else "")
        numbers = 0
        for group in optimizer.param_groups:
            numbers += _numbers(group["params"])
        lr = optimizer.param_groups[0].get("lr")
        print(f"        the optimiser was given {numbers:,} numbers to train, learning rate {lr:g}")
        return Verdict(ok1 and ok2)
    except Exception as err:  # noqa: BLE001
        return _could_not_run(err)


# ================================================================================================ step 5
def check_unfrozen(model: nn.Module, blocks=("layer4",)) -> Verdict:
    """After step 5: the given blocks and fc are trainable, everything else is frozen."""
    try:
        blocks = [str(b) for b in blocks]
        names = " and ".join(blocks)
        known = []
        for child, _ in model.named_children():
            known.append(child)
        unknown = []
        for b in blocks:
            if b not in known:
                unknown.append(b)
        if unknown:
            print(f"{_FAIL}  the model has no block called '{unknown[0]}'")
            print(f"        hint: the blocks of a ResNet are {', '.join(known)}")
            return Verdict(False)
        frozen_in, block_params = [], []
        head_ok = True
        others, other_children = [], []
        for name, p in model.named_parameters():
            if _in_blocks(name, blocks):
                block_params.append(p)
                if not p.requires_grad:
                    frozen_in.append(name)
            elif _in_blocks(name, ["fc"]):
                if not p.requires_grad:
                    head_ok = False
            else:
                child = name.split(".")[0]
                if child not in other_children:
                    other_children.append(child)
                if p.requires_grad:
                    others.append(name)
        n_block = _numbers(block_params)
        n_fc = _numbers(model.fc.parameters())
        oks = []
        if len(frozen_in) == len(block_params):
            hint = "the blank is not filled in yet: inside the loop, replace ____ by   param.requires_grad = True"
        else:
            hint = f"'{frozen_in[0] if frozen_in else ''}' is still frozen: the loop must set every parameter of " \
                   f"{names} to   param.requires_grad = True"
        oks.append(_line(not frozen_in,
                         f"every parameter of {names} is trainable: {len(block_params) - len(frozen_in)} of "
                         f"{len(block_params)} tensors ({n_block:,} numbers)", hint))
        oks.append(_line(head_ok, "the new classifier fc is still trainable",
                         "run step 3 again (a new classifier), or set requires_grad = True for model.fc.parameters()"))
        oks.append(_line(not others,
                         f"everything else is still frozen ({', '.join(other_children)})",
                         (f"'{others[0]}'" + (f" and {len(others) - 1} more" if len(others) > 1 else "")
                          + f" became trainable: loop over the parameters of {' and '.join('model.' + b for b in blocks)}"
                            f" only, not over model.parameters()") if others else ""))
        expected = n_block + n_fc
        got = _trainable_numbers(model)
        oks.append(_line(got == expected,
                         f"trainable parameters: {got:,} (expected {n_block:,} in {names} + {n_fc:,} in fc = "
                         f"{expected:,})"))
        if all(oks):
            return Verdict(True)
        for name, p in model.named_parameters():
            p.requires_grad = _in_blocks(name, blocks) or _in_blocks(name, ["fc"])
        _done_for_you(f"{names} and fc are now trainable, everything else frozen; trainable parameters now: "
                      f"{_trainable_numbers(model):,}")
        return Verdict(False)
    except Exception as err:  # noqa: BLE001
        return _could_not_run(err)


# ================================================================================================ the picture
_BLUE = "#2a78d6"        # convolution blocks (same colours as P.show_model)
_ORANGE = "#eb6834"      # pooling
_GREEN = "#1baf7a"       # linear layers
_GREY = "#e6e5e0"
_INK = "#0b0b0b"
_INK2 = "#52514e"
_BLUE_LIGHT = "#cadcf5"   # the same colours, light: frozen
_ORANGE_LIGHT = "#fad9cc"
_GREEN_DARK = "#0e7a53"


def idea_figure():
    """The recipe in one picture: the ImageNet network above, the adapted network below. Returns the figure."""
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig = Figure(figsize=(11.6, 5.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 116)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def box(x, y, w, h, face, title, sub, ink="white", edge=None, dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", facecolor=face,
                                    edgecolor=edge or face, linewidth=2.0 if dashed else 1.0,
                                    linestyle=(0, (4, 2)) if dashed else "-"))
        ax.text(x + w / 2, y + h / 2 + 1.4, title, ha="center", va="center", fontsize=10, color=ink,
                fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 1.7, sub, ha="center", va="center", fontsize=8.5, color=ink)

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=11, color=_INK2, lw=1.1))

    blocks = [("conv1 + layer1", "edges, colours", 12, 14), ("layer2", "textures", 27.5, 11.5),
              ("layer3", "patterns, parts", 40.5, 11.5), ("layer4", "object parts", 53.5, 11.5)]
    h = 8.5

    def row(y, frozen):
        box(1, y, 8.5, h, _GREY, "photo", "3 x 128 x 128", ink=_INK)
        arrow(9.5, 12, y + h / 2)
        for title, sub, x, w in blocks:
            if frozen:
                box(x, y, w, h, _BLUE_LIGHT, title, sub, ink=_INK, edge=_INK if title == "layer4" else _BLUE,
                    dashed=title == "layer4")
            else:
                box(x, y, w, h, _BLUE, title, sub)
            arrow(x + w, x + w + 1.5, y + h / 2)
        if frozen:
            box(66.5, y, 9, h, _ORANGE_LIGHT, "avgpool", "512 numbers", ink=_INK, edge=_ORANGE)
        else:
            box(66.5, y, 9, h, _ORANGE, "avgpool", "512 numbers")
        arrow(75.5, 78, y + h / 2)

    # ---- row 1: the network as it is downloaded
    y1 = 33
    ax.text(1, 47, "1 · Downloaded: ResNet-18, trained by others on ImageNet (1.28 million photos, 1 000 classes)",
            fontsize=12, color=_INK, fontweight="bold", va="center")
    row(y1, frozen=False)
    box(78, y1, 11.5, h, _GREEN, "fc", "512 -> 1 000")
    arrow(89.5, 92, y1 + h / 2)
    ax.text(92.5, y1 + h / 2, "1 000 ImageNet scores\n(tabby cat, pug, beagle,\nfire engine, ...)", fontsize=9.5,
            color=_INK, va="center")
    ax.plot([12, 75.5], [y1 - 1.6, y1 - 1.6], color=_INK2, lw=1)
    ax.plot([78, 89.5], [y1 - 1.6, y1 - 1.6], color=_INK2, lw=1)
    ax.text(43.75, y1 - 3.6, "the feature extractor: turns a photo into 512 numbers that describe it",
            ha="center", va="center", fontsize=9.5, color=_INK2)
    ax.text(83.75, y1 - 3.6, "the classifier", ha="center", va="center", fontsize=9.5, color=_INK2)

    # ---- row 2: the adapted network
    y2 = 6.5
    ax.text(1, 23.5, "2 · Yours: keep the feature extractor, replace the classifier, train it on a few hundred photos",
            fontsize=12, color=_INK, fontweight="bold", va="center")
    row(y2, frozen=True)
    box(78, y2, 11.5, h, _GREEN, "new fc", "512 -> 5", edge=_INK)
    arrow(89.5, 92, y2 + h / 2)
    ax.text(92.5, y2 + h / 2, "5 scores, one per breed:\nBombay, Siamese, pug,\nsamoyed, beagle", fontsize=9.5,
            color=_INK, va="center")
    ax.text(43.75, y2 + h + 2.0, "frozen: copied from above, training does not change it", ha="center",
            va="center", fontsize=9.5, color=_BLUE, fontweight="bold")
    ax.text(83.75, y2 + h + 2.0, "new: trained", ha="center", va="center", fontsize=9.5, color=_GREEN_DARK,
            fontweight="bold")
    ax.text(59.25, y2 - 2.6, "step 5 (optional): unfreeze layer4 (dashed) and train it a little, with a small "
            "learning rate", ha="center", va="center", fontsize=9.5, color=_INK)
    return fig


def show_idea() -> None:
    """Draw the picture of section 1 under the cell."""
    import io

    fig = idea_figure()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
        if get_ipython() is not None:
            display(Image(data=buf.getvalue()))
            return None
    except ImportError:
        pass
    print("(the picture is shown in a notebook)")
    return None
