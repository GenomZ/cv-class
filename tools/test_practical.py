"""Tests for the practical-track toolkit (notebooks/cvcourse/practical.py). Plain asserts, no pytest needed.

    python tools/test_practical.py            # about a minute on a laptop CPU; downloads MNIST / CIFAR-10 once

Covers every public function on tiny models, including branches joined by torch.cat, residual additions,
the course U-Net, a model torch.fx cannot trace, and the beginner-facing error messages.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from cvcourse import practical as P  # noqa: E402
from cvcourse.day3 import UNet  # noqa: E402

RESULTS: list = []
T0 = time.time()


def check(name):
    def wrap(fn):
        t0 = time.time()
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:300]}")
            return fn
        RESULTS.append((name, True, ""))
        print(f"[PASS] {name} ({time.time() - t0:.1f} s)")
        return fn
    return wrap


# ------------------------------------------------------------------ tiny models of every shape
def mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10))


def cnn():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(16 * 7 * 7, 32), nn.ReLU(), nn.Linear(32, 10))


class Branchy(nn.Module):
    """Inception-like: three branches joined by torch.cat, written as a class with forward()."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 8, 3, padding=1)
        self.b1 = nn.Conv2d(8, 4, 1)
        self.b2 = nn.Sequential(nn.Conv2d(8, 4, 1), nn.ReLU(), nn.Conv2d(4, 6, 3, padding=1))
        self.b3 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(8, 2, 1))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(12, 10))

    def forward(self, x):
        x = torch.relu(self.stem(x))
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.head(x)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return torch.relu(x + self.conv2(torch.relu(self.conv1(x))))


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 8, 3, padding=1)
        self.blocks = nn.Sequential(Block(8), Block(8))
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return self.fc(x.mean((2, 3)))


class Untraceable(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv(x)
        if x.sum() > -1e9:                      # data-dependent branch: torch.fx cannot trace it
            x = torch.relu(x)
        return self.fc(x.flatten(1))


class TwoOutputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 28 * 28, 10)

    def forward(self, x):
        f = self.conv(x)
        return self.fc(f.flatten(1)), f


class Shared(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1)
        self.inp = nn.Conv2d(1, 4, 1)
        self.fc = nn.Linear(4 * 28 * 28, 10)

    def forward(self, x):
        x = self.inp(x)
        x = self.conv(self.conv(x))             # the same layer twice
        return self.fc(x.flatten(1))


MNIST = (1, 28, 28)


# ------------------------------------------------------------------ data
@check("get_mnist: subsets, shapes, normalisation, determinism, full set")
def _():
    tl, vl, classes = P.get_mnist(n_train=256, n_val=128, batch_size=32, verbose=False)
    assert classes == [str(i) for i in range(10)]
    x, y = next(iter(tl))
    assert x.shape == (32, 1, 28, 28) and y.dtype == torch.int64 and y.shape == (32,)
    assert len(tl.dataset) == 256 and len(vl.dataset) == 128
    assert abs(float(tl.dataset.x.mean()) - 0.0) < 0.2 and 0.8 < float(tl.dataset.x.std()) < 1.2
    tl2, _, _ = P.get_mnist(n_train=256, n_val=128, batch_size=32, verbose=False)
    assert torch.equal(tl.dataset.y, tl2.dataset.y)
    tl3, _, _ = P.get_mnist(n_train=256, n_val=128, seed=1, verbose=False)
    assert not torch.equal(tl.dataset.y, tl3.dataset.y)
    full, _, _ = P.get_mnist(n_train=None, n_val=None, verbose=False)
    assert len(full.dataset) == 60000
    sh, sv, _ = P.get_mnist(n_train=256, n_val=128, shuffle_pixels=True, verbose=False)
    a, b = tl.dataset.x[0].flatten(), sh.dataset.x[0].flatten()
    assert torch.equal(a.sort().values, b.sort().values) and not torch.equal(a, b)


@check("get_cifar10: shapes, augmentation, resizing, ImageNet normalisation")
def _():
    tl, vl, classes = P.get_cifar10(n_train=128, n_val=64, batch_size=16, verbose=False)
    assert len(classes) == 10 and classes[0] == "airplane"
    x, y = next(iter(tl))
    assert x.shape == (16, 3, 32, 32)
    tl, vl, _ = P.get_cifar10(n_train=64, n_val=32, batch_size=8, augment=True, size=40, normalize="imagenet",
                              verbose=False)
    x, y = next(iter(tl))
    assert x.shape == (8, 3, 40, 40) and next(iter(vl))[0].shape[-1] == 40
    assert tl.dataset.mean == P.IMAGENET_MEAN


@check("show_samples: grey and colour")
def _():
    tl, _, classes = P.get_mnist(n_train=64, n_val=32, verbose=False)
    fig = P.show_samples(tl, classes, n=10)
    assert len([a for a in fig.axes if a.images]) == 10
    tc, _, cc = P.get_cifar10(n_train=32, n_val=16, verbose=False)
    fig = P.show_samples(tc, cc, n=8)
    assert fig.axes[0].images[0].get_array().shape == (32, 32, 3)


# ------------------------------------------------------------------ graph, show_model, summary
def kinds(model, shape=MNIST, depth=None):
    g = P._graph(model, shape, depth)
    return [n.kind for n in g.nodes], g


@check("graph: MLP and CNN are chains with the right kinds and shapes")
def _():
    k, g = kinds(mlp())
    assert k == ["input", "flatten", "linear", "activation", "linear", "output"], k
    assert g.nodes[1].shape == (784,) and g.nodes[-1].shape == (10,)
    k, g = kinds(cnn())
    assert k.count("convolution") == 2 and k.count("pooling") == 2 and k.count("linear") == 2
    conv2 = [n for n in g.nodes if n.kind == "convolution"][1]
    assert conv2.shape == (16, 14, 14) and conv2.params == 8 * 16 * 9 + 16
    assert conv2.macs == 16 * 14 * 14 * 8 * 9
    items, arcs, edges, sp = P._structure(g)
    assert sp and not arcs and all(kind == "box" for kind, _ in items)


@check("graph: branches joined by torch.cat become a parallel group")
def _():
    k, g = kinds(Branchy())
    assert "merge" in k
    cat = [n for n in g.nodes if n.kind == "merge"][0]
    assert cat.title == "Concat" and len(cat.inputs) == 3 and cat.shape == (12, 28, 28)
    items, arcs, edges, sp = P._structure(g)
    pars = [v for kind, v in items if kind == "par"]
    assert sp and len(pars) == 1 and len(pars[0]) == 3, items


@check("graph: residual additions become skip arcs; depth collapses blocks")
def _():
    k, g = kinds(Residual())
    adds = [n for n in g.nodes if n.kind == "merge" and n.op == "add"]
    assert len(adds) == 2
    items, arcs, edges, sp = P._structure(g)
    assert sp and len(arcs) == 2
    k1, g1 = kinds(Residual(), depth=1)
    assert [n.name for n in g1.nodes][1:4] == ["stem", "blocks", "fc"] or "blocks" in [n.name for n in g1.nodes]
    blocks = [n for n in g1.nodes if n.block]
    assert len(blocks) == 1 and blocks[0].params == sum(p.numel() for p in Residual().blocks.parameters())
    k2, g2 = kinds(Residual(), depth=2)
    assert [n.name for n in g2.nodes if n.block] == ["blocks.0", "blocks.1"]
    # the mean over height and width counts as pooling
    assert "pooling" in k


@check("graph: course U-Net at depth 1 and full depth (skips, tuple outputs, ModuleList)")
def _():
    net = UNet(in_ch=3, num_classes=3, base=4, depth=3)
    g1 = P._graph(net, (3, 32, 32), 1)
    names = [n.name for n in g1.nodes]
    assert names[1:4] == ["encoders.0", "encoders.1", "encoders.2"], names
    enc0 = g1.nodes[1]
    assert enc0.shapes == [(4, 16, 16), (4, 32, 32)]          # pooled output and the skip
    items, arcs, edges, sp = P._structure(g1)
    assert sp and len(arcs) == 3
    g = P._graph(net, (3, 32, 32))
    items, arcs, edges, sp = P._structure(g)
    assert sp and len(arcs) == 3 and sum(1 for n in g.nodes if n.title == "Concat") == 3


@check("graph: untraceable model falls back to the execution order")
def _():
    g = P._graph(Untraceable(), MNIST)
    assert not g.traced and [n.kind for n in g.nodes] == ["input", "convolution", "linear", "output"]


@check("graph: tuple output, shared layer, single layer as model")
def _():
    g = P._graph(TwoOutputs(), MNIST)
    assert g.nodes[-1].shapes == [(10,), (4, 28, 28)]
    g = P._graph(Shared(), MNIST)
    convs = [n for n in g.nodes if n.name == "conv"]
    assert len(convs) == 2 and convs[1].shared and convs[1].params == 0
    g = P._graph(nn.Conv2d(1, 4, 3), MNIST)
    assert [n.kind for n in g.nodes] == ["input", "convolution", "output"]


@check("show_model: every model draws and returns a figure")
def _():
    import torchvision
    for model, shape, depth in [(mlp(), MNIST, None), (cnn(), MNIST, None), (Branchy(), MNIST, None),
                                (Residual(), MNIST, None), (Residual(), MNIST, 1), (Untraceable(), MNIST, None),
                                (TwoOutputs(), MNIST, None), (UNet(3, 3, 4, 3), (3, 32, 32), None),
                                (UNet(3, 3, 4, 3), (3, 32, 32), 1),
                                (torchvision.models.resnet18(weights=None), (3, 64, 64), 2)]:
        fig = P.show_model(model, shape, depth=depth, title="test")
        w, h = fig.get_size_inches()
        assert 6 <= w <= 14 and h > 2, (type(model).__name__, w, h)
        texts = [t.get_fontsize() for t in fig.axes[0].texts]
        assert min(texts) >= 8.5, min(texts)


@check("summary: rows, totals, ASCII output")
def _():
    rows = P.summary(cnn(), MNIST, verbose=False)
    assert [r["kind"] for r in rows][:3] == ["convolution", "activation", "pooling"]
    assert sum(r["params"] for r in rows) == sum(p.numel() for p in cnn().parameters())
    assert rows[0]["settings"] == "Conv 3x3, 8 maps, stride 1, pad 1" and rows[0]["output_shape"] == (8, 28, 28)
    rows.text.encode("ascii")
    rows = P.summary(Branchy(), MNIST, verbose=False)
    assert any(r["kind"] == "merge" for r in rows)


@check("output_shape")
def _():
    assert P.output_shape(cnn(), MNIST) == (10,)
    assert P.output_shape(TwoOutputs(), MNIST) == ((10,), (4, 28, 28))


# ------------------------------------------------------------------ error messages
def error_of(fn):
    try:
        fn()
    except P.ModelShapeError as e:
        return str(e)
    raise AssertionError("no ModelShapeError raised")


@check("errors: wrong in_features, missing Flatten, wrong channels, image too small, bad input_shape")
def _():
    bad = nn.Sequential(nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(8 * 14 * 14, 10))
    msg = error_of(lambda: P.show_model(bad, MNIST))
    assert "Layer '4' (Linear)" in msg and "in_features=1568" in msg and "nn.Linear(1352, 10)" in msg, msg
    # a Linear layer on a 4 x 26 x 26 block runs without a PyTorch error: the toolkit must say so
    no_flatten = nn.Sequential(nn.Conv2d(1, 4, 3), nn.Linear(26, 10))
    rows = P.summary(no_flatten, MNIST, verbose=False)
    assert "nn.Flatten() before" in rows.text, rows.text
    tl, vl, _ = P.get_mnist(n_train=64, n_val=32, verbose=False)
    msg = error_of(lambda: P.train(no_flatten, tl, vl, epochs=1, live=False))
    assert "block of numbers per image" in msg and "nn.Flatten()" in msg, msg
    msg = error_of(lambda: P.cost(nn.Sequential(nn.Conv2d(3, 4, 3)), MNIST))
    assert "in_channels=3" in msg and "Fix: in_channels=1" in msg, msg
    tiny = nn.Sequential(nn.Conv2d(1, 4, 5), nn.MaxPool2d(2), nn.Conv2d(4, 4, 5), nn.MaxPool2d(2), nn.Conv2d(4, 4, 5))
    msg = error_of(lambda: P.output_shape(tiny, MNIST))
    assert "smaller than the filter" in msg and "Layer '4'" in msg, msg
    msg = error_of(lambda: P.show_model(cnn(), (28, 28)))
    assert "(1, 28, 28)" in msg, msg
    msg = error_of(lambda: P.show_model(cnn(), (1, 1, 28, 28)))
    assert "batch" in msg, msg
    msg = error_of(lambda: P.show_model(Branchy(), (3, 28, 28)))
    assert "Layer 'stem'" in msg, msg


@check("errors: label count does not match the last layer")
def _():
    tl, vl, _ = P.get_mnist(n_train=64, n_val=32, verbose=False)
    torch.manual_seed(0)
    five = nn.Sequential(nn.Flatten(), nn.Linear(784, 5))
    msg = error_of(lambda: P.train(five, tl, vl, epochs=1, live=False))
    assert "10 classes" in msg and "nn.Linear(..., 10)" in msg, msg


# ------------------------------------------------------------------ cost, compare
@check("cost: report keys, weights and random state untouched")
def _():
    model = cnn()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    rep = P.cost(model, MNIST, name="cnn", verbose=False)
    assert torch.equal(torch.rand(3), expected), "cost() changed the random number generator"
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), f"cost() changed {k}"
    for key in ("params", "trainable_params", "size_mb", "mult_adds", "ms_per_image_single", "ms_per_image_batch",
                "train_step_ms", "epoch_seconds"):
        assert key in rep and rep[key] >= 0, key
    assert rep["params"] == sum(p.numel() for p in model.parameters())
    assert rep["mult_adds"] == 8 * 28 * 28 * 9 + 16 * 14 * 14 * 8 * 9 + 32 * 784 + 10 * 32
    rep.text.encode("ascii")


@check("compare: table rows and bar chart")
def _():
    h = {"cnn": {"best_val_acc": 97.0}}
    rows = P.compare({"mlp": mlp(), "cnn": cnn()}, MNIST, histories=h)
    assert [r["name"] for r in rows] == ["mlp", "cnn"] and rows[1]["best_val_acc"] == 97.0
    assert rows[0]["best_val_acc"] is None and len(rows.figure.axes) == 4


# ------------------------------------------------------------------ training and results
@check("train: history, determinism, NaN stop, max_minutes, plot_history")
def _():
    tl, vl, classes = P.get_mnist(n_train=512, n_val=256, batch_size=64, verbose=False)
    trained = cnn()
    h1 = P.train(trained, tl, vl, epochs=2, live=False, name="a")
    for key in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds", "best_val_acc", "best_epoch",
                "input_shape", "params"):
        assert key in h1, key
    assert h1["epoch"] == [1, 2] and h1["input_shape"] == MNIST and 10 < h1["val_acc"][-1] <= 100
    P.get_mnist(n_train=512, n_val=256, batch_size=64, verbose=False)      # a call in between must not matter
    h2 = P.train(cnn(), tl, vl, epochs=2, live=False, name="b")
    assert h1["train_loss"] == h2["train_loss"] and h1["val_acc"] == h2["val_acc"], "same seed, different result"
    h3 = P.train(cnn(), tl, vl, epochs=2, live=False, optimizer="sgd", lr=1e5, name="too high")
    assert h3["stopped_early"] and len(h3["epoch"]) == 1
    h4 = P.train(mlp(), tl, None, epochs=3, live=False, max_minutes=0.0, name="budget")
    assert h4["stopped_early"] and h4["val_acc"] == [None]
    assert "best validation accuracy" in repr(h1)
    fig = P.plot_history(h1)
    assert len(fig.axes) == 2
    fig = P.plot_history({"a": h1, "b": h3})
    assert len(fig.axes[0].lines) == 4
    loss, acc = P.evaluate(trained, vl)
    assert loss > 0 and abs(acc - h1["val_acc"][-1]) < 1e-9


@check("train: per-pixel classification (segmentation-style outputs)")
def _():
    x = torch.randn(64, 1, 16, 16)
    y = (x[:, 0] > 0).long()
    ds = P.ImageSet(x, y, ["background", "object"], (0.0,), (1.0,))
    tl, vl = P.loaders(ds, ds, batch_size=16)
    torch.manual_seed(0)
    seg = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 2, 1))
    h = P.train(seg, tl, vl, epochs=3, lr=1e-2, live=False)
    assert h["val_acc"][-1] > 90, h["val_acc"]


@check("results: predict, show_predictions, confusion, show_filters, show_feature_maps")
def _():
    tl, vl, classes = P.get_mnist(n_train=512, n_val=256, verbose=False)
    model = cnn()
    P.train(model, tl, vl, epochs=1, live=False)
    image, label = vl.dataset[0]
    assert P.predict(model, image).dim() == 0 and P.predict(model, vl.dataset.x[:5]).shape == (5,)
    fig = P.show_predictions(model, vl, classes, n=12)
    assert len([a for a in fig.axes if a.images]) == 12
    fig = P.show_predictions(model, vl, classes, n=6, only_wrong=True)
    assert fig is None or len([a for a in fig.axes if a.images]) <= 6
    m = P.confusion(model, vl, classes)
    assert m.shape == (10, 10) and m.sum() == 256
    fig = P.show_filters(model)
    assert len([a for a in fig.axes if a.images]) == 8
    fig = P.show_filters(model, layer="3")
    assert len([a for a in fig.axes if a.images]) == 16
    fig = P.show_feature_maps(model, image)
    assert len([a for a in fig.axes if a.images]) == 2 * (1 + 8)
    fig = P.show_feature_maps(model, image, layer="0", n=4)
    assert len([a for a in fig.axes if a.images]) == 2 * (1 + 4)
    try:
        P.show_filters(model, layer="nope")
        raise AssertionError("no error for a wrong layer name")
    except ValueError as e:
        assert "'0'" in str(e) and "'3'" in str(e)
    tc, vc, cc = P.get_cifar10(n_train=64, n_val=32, verbose=False)
    torch.manual_seed(0)
    colour = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 32 * 32, 10))
    fig = P.show_filters(colour)
    assert fig.axes[0].images[0].get_array().shape == (3, 3, 3)


@check("remember, leaderboard, forget")
def _():
    P.forget()
    tl, vl, _ = P.get_mnist(n_train=256, n_val=128, verbose=False)
    a, b = mlp(), cnn()
    ha = P.train(a, tl, vl, epochs=1, live=False, name="mlp")
    hb = P.train(b, tl, vl, epochs=1, live=False, name="cnn")
    P.remember("mlp", a, ha, verbose=False)
    P.remember("cnn", b, hb, verbose=False)
    P.remember("cnn", b, hb, verbose=False)                         # same name: replaced, not added
    rows = P.leaderboard(chart=True)
    assert len(rows) == 2 and rows[0]["best_val_acc"] >= rows[1]["best_val_acc"]
    assert rows[0]["params"] > 0 and "model" not in rows[0]
    rows = P.leaderboard(sort="params")
    assert rows[0]["params"] <= rows[1]["params"]
    P.remember("untrained", mlp(), input_shape=MNIST, verbose=False)
    assert len(P.leaderboard()) == 3
    P.forget("untrained")
    assert len(P.leaderboard()) == 2
    P.forget()
    assert len(P.leaderboard()) == 0


@check("notebook plumbing: figures and results are not shown twice")
def _():
    fig = P.show_model(mlp(), MNIST)
    assert hasattr(fig, "_ipython_display_")
    rows = P.summary(mlp(), MNIST, verbose=False)
    assert hasattr(rows, "_ipython_display_") and repr(rows) == rows.text


if __name__ == "__main__":
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed in {time.time() - T0:.0f} s")
    for name, _, err in failed:
        print(f"  failed: {name}: {err[:200]}")
    sys.exit(1 if failed else 0)
