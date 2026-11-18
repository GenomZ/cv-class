"""Tests for the P8 helpers: checks (ptests_p8), reference models (pmodels_p8) and figures (pkit_p8).

    python tools/test_p8.py            # about a minute on a laptop CPU; needs MNIST in data/ (P1 downloads it)

Plain asserts, no pytest needed. Covers: the reference passes both checks; typical wrong answers fail with a
hint that names the problem; unfinished stubs are reported as "not written yet"; yours_or_reference falls back;
every figure helper runs; a few training steps of the tiny ViT reduce the loss; the pretrained-ViT helpers
degrade gracefully when the weights file is missing.
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

from cvcourse import pkit_p8 as V  # noqa: E402
from cvcourse import pmodels_p8 as R  # noqa: E402
from cvcourse import practical as P  # noqa: E402
from cvcourse import ptests_p8 as T  # noqa: E402

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


def run(fn, *args):
    """(result, printed text) of a check."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args)
    return result, buf.getvalue()


def assert_ascii(text: str):
    text.encode("ascii")                         # the Windows console is cp1250: checker output must be ASCII


# ------------------------------------------------------------------ wrong versions of step (a)
class EmbStub(nn.Module):
    """The untouched stub: raise NotImplementedError."""

    def __init__(self, image_size=28, patch_size=7, in_channels=1, dim=32):
        super().__init__()
        raise NotImplementedError


class EmbBlank(nn.Module):
    """raise deleted but a blank left."""

    def __init__(self, image_size=28, patch_size=7, in_channels=1, dim=32):
        super().__init__()
        self.to_tokens = ____  # noqa: F821


EmbStub.__name__ = "PatchEmbedding"           # the notebook's class names: yours_or_reference reads them
EmbBlank.__name__ = "PatchEmbedding"


def make_emb(no_stride=False, plain_tensor=False, position_rows=None, class_last=False, no_position=False,
             hard_coded=False, flat_class=False):
    class Emb(nn.Module):
        def __init__(self, image_size=28, patch_size=7, in_channels=1, dim=32):
            super().__init__()
            if hard_coded:
                image_size, patch_size, in_channels, dim = 28, 7, 1, 32
            per_side = image_size // patch_size
            n = per_side * per_side
            stride = 1 if no_stride else patch_size
            self.to_tokens = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=stride)
            if plain_tensor:
                self.class_token = torch.zeros(1, 1, dim)
            elif flat_class:
                self.class_token = nn.Parameter(torch.zeros(1, dim))
            else:
                self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            rows = n + 1 if position_rows is None else position_rows
            self.position = nn.Parameter(torch.randn(1, rows, dim))

        def forward(self, images):
            x = self.to_tokens(images)
            x = x.flatten(2)
            x = x.transpose(1, 2)
            cls = self.class_token.expand(x.shape[0], -1, -1)
            if class_last:
                x = torch.cat([x, cls], dim=1)
            else:
                x = torch.cat([cls, x], dim=1)
            if not no_position:
                x = x + self.position
            return x
    Emb.__name__ = "PatchEmbedding"
    return Emb


@check("step (a): the reference PatchEmbedding passes, with ASCII output")
def _():
    result, text = run(T.check_patch_embedding, R.PatchEmbedding)
    assert result, text
    assert "[FAIL]" not in text and "all" in text and "step (a) is done" in text
    assert_ascii(text)


@check("step (a): the untouched stub and a left blank are 'not written yet' / 'blank'")
def _():
    result, text = run(T.check_patch_embedding, EmbStub)
    assert not result and "not written yet" in text, text
    assert "Traceback" not in text
    result, text = run(T.check_patch_embedding, EmbBlank)
    assert not result and "____ blank is still empty" in text, text


@check("step (a): typical mistakes fail with a hint that names them")
def _():
    cases = [
        (dict(no_stride=True), "stride is 1"),
        (dict(plain_tensor=True), "plain tensor"),
        (dict(position_rows=16), "+ 1"),
        (dict(class_last=True), "put it first"),
        (dict(no_position=True), "position rows are not added"),
        (dict(hard_coded=True), "not the numbers"),
        (dict(flat_class=True), "torch.zeros(1, 1, dim)"),
    ]
    for kwargs, words in cases:
        result, text = run(T.check_patch_embedding, make_emb(**kwargs))
        assert not result, (kwargs, text)
        assert words in text, (kwargs, words, text)
        assert_ascii(text)


# ------------------------------------------------------------------ wrong versions of step (b)
def make_vit(read="row0", no_norm=False, head_in=None, numbers=False, depth_fixed=None):
    class Vit(nn.Module):
        def __init__(self, image_size=28, patch_size=7, dim=32, depth=2, heads=4, num_classes=10):
            super().__init__()
            if numbers:
                image_size, patch_size, dim, depth, heads, num_classes = 28, 7, 32, 2, 4, 10
            self.embedding = R.PatchEmbedding(image_size, patch_size, 1, dim)
            self.blocks = nn.Sequential()
            for k in range(depth_fixed or depth):
                self.blocks.add_module("block" + str(k + 1), R.transformer_block(dim, heads))
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(head_in or dim, num_classes)

        def forward(self, images):
            tokens = self.embedding(images)
            tokens = self.blocks(tokens)
            if read == "row0":
                c = tokens[:, 0]
            elif read == "last":
                c = tokens[:, -1]
            elif read == "mean":
                c = tokens.mean(dim=1)
            elif read == "first_picture":
                c = tokens[0]
            else:
                c = tokens
            if not no_norm:
                c = self.norm(c)
            return self.head(c)
    Vit.__name__ = "TinyViT"
    return Vit


class VitStub(nn.Module):
    def __init__(self, image_size=28, patch_size=7, dim=32, depth=2, heads=4, num_classes=10):
        super().__init__()
        raise NotImplementedError


VitStub.__name__ = "TinyViT"


@check("step (b): the reference TinyViT passes; the stub is 'not written yet'")
def _():
    result, text = run(T.check_tiny_vit, R.TinyViT)
    assert result, text
    assert "step (b) is done" in text
    assert_ascii(text)
    result, text = run(T.check_tiny_vit, VitStub)
    assert not result and "not written yet" in text, text


@check("step (b): typical mistakes fail with a hint that names them")
def _():
    cases = [
        (dict(read="last"), "last row"),
        (dict(read="mean"), "average all tokens"),
        (dict(read="first_picture"), "tokens[:, 0]"),
        (dict(read="all"), "give it only the class token"),
        (dict(no_norm=True), "self.norm(class_token) is missing"),
        (dict(numbers=True), "not numbers"),
        (dict(depth_fixed=1), "found 1 block"),
    ]
    for kwargs, words in cases:
        result, text = run(T.check_tiny_vit, make_vit(**kwargs))
        assert not result, (kwargs, text)
        assert words in text, (kwargs, words, text)
        assert_ascii(text)
    result, text = run(T.check_tiny_vit, make_vit())
    assert result, text


@check("yours_or_reference: keeps a passing version, falls back otherwise, never raises")
def _():
    good = make_emb()
    out, text = run(T.yours_or_reference, good)
    assert out is good and "using your PatchEmbedding" in text
    out, text = run(T.yours_or_reference, EmbStub)
    assert out is R.PatchEmbedding and "note: using the reference PatchEmbedding" in text, text
    out, text = run(T.yours_or_reference, VitStub)
    assert out is R.TinyViT and "note: using the reference TinyViT" in text
    out, text = run(T.yours_or_reference, R.TinyViT)
    assert out is R.TinyViT and "still using the reference" in text
    state = torch.get_rng_state()
    run(T.check_tiny_vit, R.TinyViT)
    assert torch.equal(state, torch.get_rng_state()), "a check changed the random numbers"


@check("reference models: shapes, parameter counts, the conv = cut + flatten + linear")
def _():
    torch.manual_seed(0)
    emb = R.PatchEmbedding()
    x = torch.randn(3, 1, 28, 28)
    assert emb(x).shape == (3, 17, 32)
    assert sum(p.numel() for p in emb.parameters()) == 2176
    net = R.TinyViT()
    assert net(x).shape == (3, 10)
    assert sum(p.numel() for p in net.parameters()) == 19658
    assert sum(p.numel() for p in R.transformer_block(32, 4).parameters()) == 8544
    assert sum(p.numel() for p in R.small_cnn().parameters()) == 105866
    linear = nn.Linear(49, 32)
    conv = nn.Conv2d(1, 32, kernel_size=7, stride=7)
    _, text = run(V.copy_linear_into_conv, linear, conv)
    table = V.patch_table(x[0], 7)
    by_conv = conv(x[:1]).flatten(2).transpose(1, 2)[0]
    assert torch.allclose(by_conv, linear(table), atol=1e-5)


@check("figure helpers run on a digit and on a trained-looking model")
def _():
    tl, vl, classes = P.get_mnist(n_train=256, n_val=64, verbose=False)
    image, _ = vl.dataset[4]
    assert len(V.show_idea(image).axes) == 2
    fig = V.show_patches(image, patch_size=7)
    assert len(fig.axes) == 1 + 16
    table = V.patch_table(image, 7)
    assert table.shape == (16, 49)
    V.show_table(table, "pixels", kind="pixels")
    torch.manual_seed(0)
    emb = R.PatchEmbedding()
    tokens = emb(image.unsqueeze(0))
    V.show_tokens(tokens)
    shuffled = V.shuffle_patches(image, 7, seed=0)
    assert not torch.equal(shuffled, image)
    found, text = run(V.same_rows, table, V.patch_table(shuffled, 7))
    assert found and "True" in text and "same order: False" in text
    V.show_pictures([image, shuffled], ["a", "b"])
    toy = torch.tensor([[1.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
    w = (toy @ toy.T).softmax(dim=1)
    V.show_toy_attention(toy, w, w @ toy, ["A", "B", "C"])
    attention = nn.MultiheadAttention(embed_dim=32, num_heads=1, batch_first=True)
    _, weights = attention(tokens, tokens, tokens)
    V.show_weights(weights)
    _, text = run(V.position_test, emb, attention, image, shuffled)
    assert "the same numbers" in text and "different" in text, text
    net = R.TinyViT(dim=16, depth=2, heads=2)
    maps = V.attention_maps(net, torch.stack([image, shuffled]))
    assert len(maps) == 2 and maps[0].shape == (2, 17, 17)
    assert torch.allclose(maps[0].sum(dim=2), torch.ones(2, 17), atol=1e-5)
    fig = V.show_class_attention(net, vl, n=4)
    assert len(fig.axes) == 8
    V.show_class_attention(net, vl, n=3, block=-1)
    V.show_position_similarity(net)
    V.show_position_similarity(R.TinyViT(patch_size=4, dim=16, depth=1, heads=2))


@check("a few training steps of the tiny ViT reduce the loss; P.summary / P.cost cope with it")
def _():
    tl, vl, _ = P.get_mnist(n_train=512, n_val=128, batch_size=64, verbose=False)
    torch.manual_seed(0)
    net = R.TinyViT()
    h = P.train(net, tl, vl, epochs=3, lr=1e-3, live=False, verbose=False)
    assert h["train_loss"][-1] < h["train_loss"][0], h["train_loss"]
    assert h["val_acc"][-1] > 20, h["val_acc"]
    rows = P.summary(net, (1, 28, 28), verbose=False)
    assert sum(r["params"] for r in rows) <= 19658
    rep = P.cost(net, (1, 28, 28), verbose=False)
    assert rep["params"] == 19658
    P.show_model(net, (1, 28, 28), depth=1)


@check("pretrained ViT helpers: load or degrade gracefully, never download")
def _():
    import torchvision

    folder = Path(torch.hub.get_dir()) / "checkpoints"
    have = (folder / V.VIT_FILE).exists()
    calls = []
    original = torch.hub.load_state_dict_from_url

    def no_download(*args, **kwargs):
        calls.append(args)
        raise AssertionError("tried to download")
    torch.hub.load_state_dict_from_url = no_download
    try:
        (vit, weights), text = run(V.load_pretrained_vit)
        assert isinstance(vit, torchvision.models.VisionTransformer)
        if have:
            assert weights is not None and "loaded" in text
        else:
            assert weights is None and "not on this computer" in text
        fig, text = run(V.classify_photos, vit, None, ["pug_12"])
        assert fig is None and "skipping" in text
    finally:
        torch.hub.load_state_dict_from_url = original
    assert not calls
    assert V.pet_photo("pug_12").name == "pug_12.jpg"
    if have:
        fig, text = run(V.classify_photos, vit, weights, ["pug_12", "no_such_photo_1"])
        assert fig is not None and "pug_12:" in text and "photo not found" in text, text


if __name__ == "__main__":
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed in {time.time() - T0:.0f} s")
    for name, _, err in failed:
        print(f"  failed: {name}: {err[:200]}")
    sys.exit(1 if failed else 0)
