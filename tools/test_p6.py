"""Tests for the checks and helpers of P6 (notebooks/cvcourse/ptests_p6.py). Plain asserts, no pytest needed.

    python tools/test_p6.py            # about a minute on a laptop CPU (CIFAR-10 must be downloaded once)

The reference passes every check; typical wrong answers fail with a hint that names the problem; unfinished
cells (the raise line still there) count as "not written yet"; yours_or_reference falls back to the reference.
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from cvcourse import pmodels_p3 as p3  # noqa: E402
from cvcourse import pmodels_p6 as R  # noqa: E402
from cvcourse import ptests_p6 as T  # noqa: E402

RESULTS: list = []
T0 = time.time()


def check(name):
    def wrap(fn):
        t0 = time.time()
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:400]}")
            return fn
        RESULTS.append((name, True, ""))
        print(f"[PASS] {name} ({time.time() - t0:.1f} s)")
        return fn
    return wrap


def run(fn, *args):
    """(result, printed text) of a check."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args)
    return result, buf.getvalue()


def failed_with(fn, thing, *words):
    """The check fails, and its output contains every word."""
    result, text = run(fn, thing)
    assert not result, f"expected a failure:\n{text}"
    for w in words:
        assert w in text, f"{w!r} not in:\n{text}"
    text.encode("ascii")
    return text


# ------------------------------------------------------------------ step (a): wrong versions of make_aux_head
def aux_stub(in_channels, num_classes):
    raise NotImplementedError


def aux_blank(in_channels, num_classes):
    return nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=3), ____)  # noqa: F821  (a blank left)


def aux_no_stride(in_channels, num_classes):
    return nn.Sequential(nn.AvgPool2d(kernel_size=5), nn.Conv2d(in_channels, 128, kernel_size=1), nn.ReLU(),
                         nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.7),
                         nn.Linear(1024, num_classes))


def aux_maxpool(in_channels, num_classes):
    return nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=3), nn.Conv2d(in_channels, 128, kernel_size=1), nn.ReLU(),
                         nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.7),
                         nn.Linear(1024, num_classes))


def aux_adaptive(in_channels, num_classes):
    return nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)), nn.Conv2d(in_channels, 128, kernel_size=1), nn.ReLU(),
                         nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.7),
                         nn.Linear(1024, num_classes))


def aux_numbers(in_channels, num_classes):
    return nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=3), nn.Conv2d(512, 128, kernel_size=1), nn.ReLU(),
                         nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.7), nn.Linear(1024, 1000))


def aux_no_relu(in_channels, num_classes):
    return nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=3), nn.Conv2d(in_channels, 128, kernel_size=1),
                         nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.7),
                         nn.Linear(1024, num_classes))


def aux_dropout(in_channels, num_classes):
    return nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=3), nn.Conv2d(in_channels, 128, kernel_size=1), nn.ReLU(),
                         nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.4),
                         nn.Linear(1024, num_classes))


def aux_conv_64(in_channels, num_classes):
    return nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=3), nn.Conv2d(in_channels, 64, kernel_size=1), nn.ReLU(),
                         nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.7),
                         nn.Linear(1024, num_classes))


def make_aux_head(in_channels, num_classes):       # a correct copy, written as in the solution
    head = nn.Sequential(
        nn.AvgPool2d(kernel_size=5, stride=3),
        nn.Conv2d(in_channels, 128, kernel_size=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.7),
        nn.Linear(1024, num_classes),
    )
    return head


@check("step (a): the reference and a correct copy pass")
def _():
    result, text = run(T.check_aux_head, R.make_aux_head)
    assert result, text
    assert "3,188,840 parameters" in text and "=> all" in text, text
    text.encode("ascii")
    result, text = run(T.check_aux_head, make_aux_head)
    assert result, text


@check("step (a): unfinished cells say 'not written yet' or name the blank")
def _():
    failed_with(T.check_aux_head, aux_stub, "not written yet")
    failed_with(T.check_aux_head, aux_blank, "a ____ blank is still empty")


@check("step (a): typical mistakes get a hint that names them")
def _():
    failed_with(T.check_aux_head, aux_no_stride, "stride is 5, expected 3", "steps by its kernel size")
    failed_with(T.check_aux_head, aux_maxpool, "MaxPool2d", "averages")
    failed_with(T.check_aux_head, aux_adaptive, "AdaptiveAvgPool2d", "torchvision")
    failed_with(T.check_aux_head, aux_numbers, "make_aux_head(528, 10)", "in_channels and num_classes")
    failed_with(T.check_aux_head, aux_no_relu, "no ReLU after the 1x1 convolution")
    failed_with(T.check_aux_head, aux_dropout, "p is 0.4, expected 0.7")
    failed_with(T.check_aux_head, aux_conv_64, "out channels is 64, expected 128", "mat1 and mat2",
                "the 1x1 convolution must make 128 maps")
    failed_with(T.check_aux_head, aux_no_stride, "Flatten receives 128 x 2 x 2", "(14 - 5) / 3 + 1 = 4")


# ------------------------------------------------------------------ step (b): wrong versions of GoogLeNetAux
class AuxBase(nn.Module):
    """The given __init__ of the notebook (with the reference auxiliary head)."""

    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = p3.make_stem()
        blocks = p3.make_inception_blocks()
        self.blocks_3a_to_4a = blocks[0:4]
        self.blocks_4b_to_4d = blocks[4:7]
        self.blocks_4e_to_5b = blocks[7:11]
        self.head = p3.make_head(1024, num_classes)
        self.aux1 = R.make_aux_head(512, num_classes)
        self.aux2 = R.make_aux_head(528, num_classes)


class GoogLeNetAux(AuxBase):                         # a correct copy, written as in the solution
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks_3a_to_4a(x)
        if self.training:
            aux1 = self.aux1(x)
        x = self.blocks_4b_to_4d(x)
        if self.training:
            aux2 = self.aux2(x)
        x = self.blocks_4e_to_5b(x)
        main = self.head(x)
        if self.training:
            return main, aux1, aux2
        return main


class AuxStub(AuxBase):
    def forward(self, x):
        raise NotImplementedError


class AuxAlways(AuxBase):                            # the auxiliary heads run in prediction mode too
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks_3a_to_4a(x)
        aux1 = self.aux1(x)
        x = self.blocks_4b_to_4d(x)
        aux2 = self.aux2(x)
        x = self.blocks_4e_to_5b(x)
        main = self.head(x)
        if self.training:
            return main, aux1, aux2
        return main


class AuxSwapped(AuxBase):                           # aux1 and aux2 exchanged
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks_3a_to_4a(x)
        if self.training:
            aux1 = self.aux2(x)
        x = self.blocks_4b_to_4d(x)
        if self.training:
            aux2 = self.aux1(x)
        x = self.blocks_4e_to_5b(x)
        main = self.head(x)
        if self.training:
            return main, aux1, aux2
        return main


class AuxOrder(GoogLeNetAux):                        # returns main, aux2, aux1 (torchvision's order)
    def forward(self, x):
        out = super().forward(x)
        if self.training:
            return out[0], out[2], out[1]
        return out


class AuxNoCall(AuxBase):                            # main = self.head, without (x)
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks_3a_to_4a(x)
        if self.training:
            aux1 = self.aux1(x)
        x = self.blocks_4b_to_4d(x)
        if self.training:
            aux2 = self.aux2(x)
        x = self.blocks_4e_to_5b(x)
        main = self.head
        if self.training:
            return main, aux1, aux2
        return main


class AuxSkip(AuxBase):                              # blocks 4b to 4d forgotten
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks_3a_to_4a(x)
        if self.training:
            aux1 = self.aux1(x)
        if self.training:
            aux2 = self.aux2(x)
        x = self.blocks_4e_to_5b(x)
        main = self.head(x)
        if self.training:
            return main, aux1, aux2
        return main


class AuxLate(AuxBase):                              # aux2 reads the output of 4e to 5b
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks_3a_to_4a(x)
        if self.training:
            aux1 = self.aux1(x)
        x = self.blocks_4b_to_4d(x)
        x = self.blocks_4e_to_5b(x)
        if self.training:
            aux2 = self.aux2(x)
        main = self.head(x)
        if self.training:
            return main, aux1, aux2
        return main


class AuxOne(AuxBase):                               # always returns the main scores only
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks_3a_to_4a(x)
        x = self.blocks_4b_to_4d(x)
        x = self.blocks_4e_to_5b(x)
        return self.head(x)


@check("step (b): the reference and a correct copy pass (with torchvision)")
def _():
    result, text = run(T.check_googlenet_aux, R.GoogLeNetAux)
    assert result, text
    assert "13,378,280 parameters" in text and "3,188,968" in text and "=> all" in text, text
    text.encode("ascii")
    result, text = run(T.check_googlenet_aux, GoogLeNetAux)
    assert result, text


@check("step (b): typical mistakes get a hint that names them")
def _():
    failed_with(T.check_googlenet_aux, AuxStub, "not written yet")
    failed_with(T.check_googlenet_aux, AuxAlways, "ran in prediction mode", "if self.training")
    failed_with(T.check_googlenet_aux, AuxSwapped, "aux2 (built for the 528 maps of block 4d) received 512 maps")
    failed_with(T.check_googlenet_aux, AuxOrder, "output 2 should be aux1's scores, it is aux2's",
                "return main, aux1, aux2")
    failed_with(T.check_googlenet_aux, AuxNoCall, "main is a Sequential, not a tensor")
    failed_with(T.check_googlenet_aux, AuxSkip, "aux2 (built for the 528 maps of block 4d)")
    failed_with(T.check_googlenet_aux, AuxLate, "aux2 (built for the 528 maps of block 4d)",
                "right after self.blocks_4b_to_4d")
    failed_with(T.check_googlenet_aux, AuxOne, "it returns Tensor")


# ------------------------------------------------------------------ step (c): wrong versions of aux_loss
ce = nn.CrossEntropyLoss()


def loss_stub(outputs, labels, aux_weight=0.3):
    raise NotImplementedError


def loss_main_only(outputs, labels, aux_weight=0.3):
    main, aux1, aux2 = outputs
    return ce(main, labels)


def loss_no_weight(outputs, labels, aux_weight=0.3):
    main, aux1, aux2 = outputs
    return ce(main, labels) + ce(aux1, labels) + ce(aux2, labels)


def loss_literal(outputs, labels, aux_weight=0.3):
    main, aux1, aux2 = outputs
    return ce(main, labels) + 0.3 * ce(aux1, labels) + 0.3 * ce(aux2, labels)


def loss_weight_on_main(outputs, labels, aux_weight=0.3):
    main, aux1, aux2 = outputs
    return aux_weight * ce(main, labels) + ce(aux1, labels) + ce(aux2, labels)


def loss_aux1_twice(outputs, labels, aux_weight=0.3):
    main, aux1, aux2 = outputs
    return ce(main, labels) + aux_weight * ce(aux1, labels) + aux_weight * ce(aux1, labels)


def loss_item(outputs, labels, aux_weight=0.3):
    main, aux1, aux2 = outputs
    loss = ce(main, labels) + aux_weight * ce(aux1, labels) + aux_weight * ce(aux2, labels)
    return loss.item()


def loss_per_photo(outputs, labels, aux_weight=0.3):
    each = nn.CrossEntropyLoss(reduction="none")
    main, aux1, aux2 = outputs
    return each(main, labels) + aux_weight * each(aux1, labels) + aux_weight * each(aux2, labels)


def aux_loss(outputs, labels, aux_weight=0.3):     # a correct copy, written as in the solution
    main, aux1, aux2 = outputs
    loss_main = ce(main, labels)
    loss_aux1 = ce(aux1, labels)
    loss_aux2 = ce(aux2, labels)
    loss = loss_main + aux_weight * loss_aux1 + aux_weight * loss_aux2
    return loss


@check("step (c): the reference and a correct copy pass")
def _():
    result, text = run(T.check_aux_loss, R.aux_loss)
    assert result, text
    text.encode("ascii")
    result, text = run(T.check_aux_loss, aux_loss)
    assert result, text


@check("step (c): typical mistakes get a hint that names them")
def _():
    failed_with(T.check_aux_loss, loss_stub, "not written yet")
    failed_with(T.check_aux_loss, loss_main_only, "only the main loss")
    failed_with(T.check_aux_loss, loss_no_weight, "without their weight")
    failed_with(T.check_aux_loss, loss_literal, "the number 0.3 is written instead of the argument aux_weight")
    failed_with(T.check_aux_loss, loss_weight_on_main, "aux_weight multiplies the main loss")
    failed_with(T.check_aux_loss, loss_aux1_twice, "loss_aux1 is counted twice", "gradient of aux1, aux2")
    failed_with(T.check_aux_loss, loss_item, "it returns a float", "no .item()")
    failed_with(T.check_aux_loss, loss_per_photo, "not one per photo")


# ------------------------------------------------------------------ never stuck
@check("yours_or_reference: yours when it passes, the reference otherwise, with a note")
def _():
    got, text = run(T.yours_or_reference, make_aux_head)
    assert got is make_aux_head and "using your make_aux_head" in text, text
    got, text = run(lambda: T.yours_or_reference(aux_stub, "make_aux_head"))
    assert got is R.make_aux_head and "note: using the reference make_aux_head" in text, text
    got, text = run(lambda: T.yours_or_reference(aux_no_stride, "make_aux_head"))
    assert got is R.make_aux_head and "note: using the reference make_aux_head" in text, text
    got, text = run(lambda: T.yours_or_reference(AuxStub, "GoogLeNetAux"))
    assert got is R.GoogLeNetAux and "note: using the reference GoogLeNetAux" in text, text
    got, text = run(T.yours_or_reference, GoogLeNetAux)
    assert got is GoogLeNetAux and "using your GoogLeNetAux" in text, text
    got, text = run(lambda: T.yours_or_reference(loss_stub, "aux_loss"))
    assert got is R.aux_loss and "note: using the reference aux_loss" in text, text
    got, text = run(T.yours_or_reference, aux_loss)
    assert got is aux_loss, text
    got, text = run(T.yours_or_reference, R.aux_loss)
    assert got is R.aux_loss and "still using the reference" in text, text
    text.encode("ascii")


@check("checks do not change PyTorch's random numbers")
def _():
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)
    run(T.check_aux_head, R.make_aux_head)
    run(T.check_aux_loss, loss_main_only)
    run(T.yours_or_reference, aux_loss)
    assert torch.equal(torch.rand(3), expected)


# ------------------------------------------------------------------ helpers
@check("show_in_training_mode: draws three outputs and restores the mode")
def _():
    torch.manual_seed(0)
    small = R.SmallGoogLeNetAux()
    small.eval()
    fig = T.show_in_training_mode(small, (3, 32, 32), depth=1, title="test")
    assert fig is not None and not small.training and "train" not in small.__dict__
    from cvcourse import practical as P
    g = P._graph(small, (3, 32, 32), 1)
    assert g.nodes[-1].shapes == [(10,)]                     # prediction mode: one output


@check("parameters_per_part and gradient_per_part print the right numbers")
def _():
    net = R.GoogLeNetAux()
    _, text = run(T.parameters_per_part, net)
    assert "6,998,552" in text and "6,379,728" in text and "13,378,280" in text, text
    text.encode("ascii")
    torch.manual_seed(0)
    small = R.SmallGoogLeNetAux()
    before = {}
    for k, v in small.state_dict().items():
        before[k] = v.clone()
    images = torch.randn(8, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    parts = ("stem", "block_3a", "block_3b", "pool_and_4a", "head")
    _, text = run(lambda: T.gradient_per_part(small, images, labels, R.aux_loss, parts=parts))
    lines = text.splitlines()
    assert "times bigger" in text and len(lines) == 8, text
    assert lines[-1].split()[-1] == "1.0" and lines[-2].split()[-1] == "1.0", text     # above the heads: no change
    for k, v in small.state_dict().items():
        assert torch.equal(v, before[k]), k
    text.encode("ascii")


@check("train_with_aux: history, the main head's numbers, determinism, aux_weight matters, compare_runs")
def _():
    from cvcourse import practical as P
    tl, vl, _ = P.get_cifar10(n_train=256, n_val=128, batch_size=64, verbose=False)
    hists = []
    for w in (0.0, 0.0, 0.3):
        torch.manual_seed(0)
        small = R.SmallGoogLeNetAux()
        h = T.train_with_aux(small, tl, vl, R.aux_loss, aux_weight=w, epochs=2, live=False, verbose=False,
                             name=f"w={w}")
        hists.append(h)
    h = hists[0]
    for key in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds", "best_val_acc", "best_epoch",
                "train_total_loss", "aux_weight", "input_shape"):
        assert key in h, key
    assert h["epoch"] == [1, 2] and h["input_shape"] == (3, 32, 32)
    assert h["train_loss"] == h["train_total_loss"], "with aux_weight=0 the total loss is the main loss"
    assert hists[0]["val_acc"] == hists[1]["val_acc"], "same seed, different result"
    assert hists[2]["train_total_loss"][0] > hists[2]["train_loss"][0]
    assert hists[2]["val_loss"] != hists[0]["val_loss"], "aux_weight changed nothing"
    _, text = run(T.compare_runs, hists[0], hists[2])
    assert "difference" in text and "best" in text and len(text.splitlines()) == 5, text
    text.encode("ascii")
    fig = P.plot_history({"without": hists[0], "with": hists[2]})
    assert len(fig.axes) == 2


if __name__ == "__main__":
    failed = []
    for r in RESULTS:
        if not r[1]:
            failed.append(r)
    print(f"\n{len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed in {time.time() - T0:.0f} s")
    for name, _, err in failed:
        print(f"  failed: {name}: {err[:200]}")
    sys.exit(1 if failed else 0)
