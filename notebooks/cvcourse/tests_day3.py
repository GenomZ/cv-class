"""Day-3 exercise checkers (segmentation, instance masks, foundation models).

Re-exported by `cvcourse.tests`, so notebooks call `tests.check_rle(...)` etc.
Every checker prints an ASCII [PASS] / [FAIL] line with a hint and returns a bool.
"""
from __future__ import annotations

import numpy as np

from . import seg

__all__ = ["check_rle", "check_soft_dice_torch", "check_dilate_kernel", "check_unet_blocks",
           "check_unet_forward", "check_bce_dice", "check_sliding_window", "check_threshold_sweep",
           "check_paste_mask", "check_mask_iou", "check_penn_fudan_target", "check_maskrcnn_heads",
           "check_mask_ap", "check_patchify", "check_attention", "check_zero_shot", "check_prompt_ensemble"]

_PASS, _FAIL = "[PASS]", "[FAIL]"


def _report(name, ok, hint=""):
    print(f"{_PASS if ok else _FAIL}  {name}" + (f"\n      hint: {hint}" if (hint and not ok) else ""))
    return bool(ok)


def _np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _run(name, fn, hint):
    """Call fn(); turn NotImplementedError / exceptions into a FAIL line."""
    try:
        return fn()
    except NotImplementedError:
        _report(name, False, "not implemented yet: fill in the blanks, then delete the `raise NotImplementedError` line")
        return None
    except Exception as e:  # noqa: BLE001
        _report(name, False, f"{type(e).__name__}: {e}  |  {hint}")
        return None


# ----------------------------------------------------------------- D3-01 ---
def check_rle(encode, decode):
    from .day3 import rle_encode as ref
    name = "rle_encode / rle_decode"

    def go():
        rng = np.random.default_rng(20)
        cases = [np.zeros((3, 4), np.uint8), np.ones((3, 4), np.uint8),
                 np.array([[0, 1, 1], [0, 1, 0]], np.uint8), (rng.uniform(size=(13, 17)) > 0.6).astype(np.uint8)]
        blob = np.zeros((20, 30), np.uint8); blob[5:12, 8:20] = 1; cases.append(blob)
        for m in cases:
            got = encode(m)
            want = ref(m)
            if list(got["counts"]) != want["counts"] or list(got["size"]) != want["size"]:
                print(f"   mask {m.shape}: counts {list(got['counts'])[:8]}... expected {want['counts'][:8]}...")
                return _report(name, False, "read the mask column by column (order='F'); the first count is the number of leading ZEROS (0 if it starts with 1)")
            back = _np(decode(want))
            if back.shape != m.shape or not np.array_equal(back.astype(bool), m.astype(bool)):
                return _report(name, False, "decode: alternate runs of 0s and 1s starting with 0, then reshape((H, W), order='F')")
        return _report(name, True)
    return _run(name, go, "encode must return a dict with 'size' and 'counts'")


def check_soft_dice_torch(fn):
    import torch
    name = "soft_dice_loss (torch)"

    def go():
        g = torch.Generator().manual_seed(0)
        p = torch.rand(2, 1, 16, 16, generator=g, requires_grad=True)
        t = (torch.rand(2, 1, 16, 16, generator=g) > 0.7).float()
        loss = fn(p, t)
        if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
            return _report(name, False, "return a scalar tensor (0-dim), not a float or a per-pixel map")
        want = seg.soft_dice_loss(p.detach().numpy(), t.numpy())
        ok = abs(float(loss.detach()) - want) < 1e-4
        perfect = float(fn(t.clone(), t))
        ok &= abs(perfect) < 1e-4
        loss.backward()
        ok &= p.grad is not None and float(p.grad.abs().sum()) > 0
        return _report(name, ok, f"expected {want:.4f}: 1 - (2 * sum(p*t) + eps) / (sum(p) + sum(t) + eps), over the whole batch, with torch ops only")
    return _run(name, go, "use torch operations so autograd works")


def check_dilate_kernel(fn):
    name = "dilate_kernel"

    def go():
        rng = np.random.default_rng(21)
        for k in [rng.normal(size=(3, 3)), rng.normal(size=(2, 3)), np.ones((1, 1))]:
            for d in (1, 2, 3):
                got = _np(fn(k, d)); want = seg.dilate_kernel(k, d)
                if got.shape != want.shape or not np.allclose(got, want):
                    print(f"   kernel {k.shape}, dilation {d}: got shape {got.shape}, expected {want.shape}")
                    return _report(name, False, "size d*(k-1)+1; place the taps at out[::d, ::d]")
        return _report(name, True)
    return _run(name, go, "")


# ----------------------------------------------------------------- D3-02 ---
def check_unet_blocks(EncoderBlock, DecoderBlock):
    import torch
    name = "EncoderBlock / DecoderBlock"

    def go():
        torch.manual_seed(0)
        enc = EncoderBlock(3, 16)
        x = torch.randn(2, 3, 64, 64)
        out = enc(x)
        if not (isinstance(out, (tuple, list)) and len(out) == 2):
            return _report(name, False, "EncoderBlock.forward must return (pooled, skip)")
        pooled, skip = out
        if tuple(pooled.shape) != (2, 16, 32, 32) or tuple(skip.shape) != (2, 16, 64, 64):
            return _report(name, False, f"encoder: got pooled {tuple(pooled.shape)}, skip {tuple(skip.shape)}; expected (2,16,32,32) and (2,16,64,64)")
        dec = DecoderBlock(32, 16, 16)
        xd = torch.randn(2, 32, 16, 16); sk = torch.randn(2, 16, 32, 32)
        y = dec(xd, sk)
        if tuple(y.shape) != (2, 16, 32, 32):
            return _report(name, False, f"decoder: got {tuple(y.shape)}, expected (2, 16, 32, 32): upsample x2, concat skip, double conv")
        dec.eval()
        y1 = dec(xd, sk); y2 = dec(xd, sk + 1.0)
        if torch.allclose(y1, y2):
            return _report(name, False, "the decoder output does not depend on the skip: did you torch.cat it?")
        dec_ns = DecoderBlock(32, 16, 16, use_skip=False)
        y3 = dec_ns(xd, None)
        if tuple(y3.shape) != (2, 16, 32, 32):
            return _report(name, False, "with use_skip=False the decoder must work with skip=None")
        n_skip = sum(p.numel() for p in dec.parameters()); n_ns = sum(p.numel() for p in dec_ns.parameters())
        if n_skip <= n_ns:
            return _report(name, False, "the conv after the concat must take out_ch + skip_ch input channels")
        return _report(name, True)
    return _run(name, go, "check the shapes in the comments")


def check_unet_forward(UNet):
    import torch
    from .day3 import UNet as RefUNet
    name = "UNet forward"

    def go():
        torch.manual_seed(0)
        for (in_ch, nc, base, depth, size) in [(3, 4, 16, 4, 128), (1, 2, 8, 3, 64)]:
            m = UNet(in_ch=in_ch, num_classes=nc, base=base, depth=depth)
            y = m(torch.randn(2, in_ch, size, size))
            if tuple(y.shape) != (2, nc, size, size):
                return _report(name, False, f"UNet(in_ch={in_ch}, num_classes={nc}, base={base}, depth={depth}) on {size}px gave {tuple(y.shape)}, expected {(2, nc, size, size)}")
        m = UNet(in_ch=3, num_classes=4, base=16, depth=4)
        y = m(torch.randn(1, 3, 64, 64)); y.sum().backward()
        first = next(m.parameters())
        if first.grad is None or float(first.grad.abs().sum()) == 0:
            return _report(name, False, "no gradient reaches the first layer")
        m_ns = UNet(in_ch=3, num_classes=4, base=16, depth=4, use_skips=False)
        if tuple(m_ns(torch.randn(1, 3, 64, 64)).shape) != (1, 4, 64, 64):
            return _report(name, False, "use_skips=False must still produce full-resolution logits")
        n, n_ref = sum(p.numel() for p in m.parameters()), sum(p.numel() for p in RefUNet(3, 4, 16, 4).parameters())
        if n != n_ref:
            print(f"   note: {n:,} parameters vs {n_ref:,} in the reference design (fine if you chose a different block layout)")
        return _report(name, True)
    return _run(name, go, "encoders -> bottleneck -> decoders (with reversed skips) -> 1x1 head")


def check_bce_dice(fn):
    import torch
    import torch.nn.functional as F
    name = "bce_dice_loss"

    def go():
        g = torch.Generator().manual_seed(1)
        logits = torch.randn(2, 1, 12, 12, generator=g, requires_grad=True)
        t = (torch.rand(2, 1, 12, 12, generator=g) > 0.8).float()
        for w in (0.5, 0.2, 1.0):
            got = fn(logits, t, bce_weight=w)
            bce = F.binary_cross_entropy_with_logits(logits, t)
            p = torch.sigmoid(logits)
            dice = 1 - (2 * (p * t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6)
            want = w * bce + (1 - w) * dice
            if abs(float(got.detach()) - float(want.detach())) > 1e-4:
                return _report(name, False, f"bce_weight={w}: got {float(got.detach()):.4f}, expected {float(want.detach()):.4f}: "
                                            "w * BCE-with-logits + (1-w) * soft Dice of sigmoid(logits)")
        got.backward()
        return _report(name, logits.grad is not None, "the loss must be differentiable (torch ops only)")
    return _run(name, go, "")


def check_sliding_window(fn):
    import torch
    name = "sliding_window_predict"

    def go():
        g = torch.Generator().manual_seed(2)
        for (H, W, win, stride) in [(64, 64, 32, 32), (100, 80, 48, 32), (37, 53, 32, 16)]:
            img = torch.randn(3, H, W, generator=g)
            out = fn(lambda x: x, img, win, stride)
            if tuple(out.shape) != (3, H, W) or not torch.allclose(out, img, atol=1e-5):
                return _report(name, False, f"identity model on a {H}x{W} image, window {win}, stride {stride}: "
                                            "output should equal the input. Did you cover the right/bottom border "
                                            "(add a last window ending exactly at H / W) and divide by the overlap count?")
        img = torch.randn(3, 50, 70, generator=g)
        out = fn(lambda x: torch.cat([2 * x, x[:, :1]], 1), img, 32, 20)
        if tuple(out.shape) != (4, 50, 70) or not torch.allclose(out[:3], 2 * img, atol=1e-5):
            return _report(name, False, "the output channel count comes from the model (K), not from the image (C)")
        return _report(name, True)
    return _run(name, go, "")


def check_threshold_sweep(fn):
    from .day3 import best_threshold
    name = "threshold sweep"

    def go():
        rng = np.random.default_rng(22)
        t = rng.uniform(size=(40, 40)) > 0.7
        prob = np.clip(t * 0.35 + rng.uniform(0, 0.6, t.shape), 0, 1)
        ths = np.linspace(0.05, 0.95, 19)
        bt, ious = fn(prob, t, ths)
        wt, wious = best_threshold(prob, t, ths)
        ok = np.allclose(np.asarray(ious, float), wious, atol=1e-6) and abs(float(bt) - wt) < 1e-9
        return _report(name, ok, f"for each threshold: IoU of (prob >= th) with the target; best = argmax (expected {wt:.2f})")
    return _run(name, go, "")


# ----------------------------------------------------------------- D3-03 ---
def check_paste_mask(fn):
    from .day3 import paste_mask as ref, mask_iou
    name = "paste_mask"

    def go():
        rng = np.random.default_rng(23)
        m = np.zeros((28, 28), np.float32); m[4:24, 8:20] = 1.0
        cases = [((10.0, 20.0, 50.0, 90.0), (120, 100)), ((60.3, 5.7, 99.2, 40.1), (64, 128)),
                 ((-10.0, 30.0, 40.0, 70.0), (80, 80)), ((70.0, 70.0, 130.0, 110.0), (100, 100))]
        for box, hw in cases:
            got = _np(fn(m, box, hw)).astype(bool)
            want = ref(m, box, hw)
            if got.shape != tuple(hw):
                return _report(name, False, f"output must be the full image size {hw}, got {got.shape}")
            iou = mask_iou(got[None], want[None])[0, 0]
            if iou < 0.95:
                return _report(name, False, f"box {box} in image {hw}: IoU with reference {iou:.2f}. Resize the 28x28 map "
                                            "to the box size (bilinear), threshold at 0.5, write it at the box position, clip at the image border")
        full = _np(fn(np.ones((28, 28), np.float32), (10, 10, 30, 40), (50, 50))).astype(bool)
        if abs(int(full.sum()) - 600) > 60:
            return _report(name, False, f"an all-ones mask should fill its 20x30 box (600 px), got {int(full.sum())}")
        return _report(name, True)
    return _run(name, go, "")


def check_mask_iou(fn):
    from .day3 import mask_iou as ref
    name = "mask_iou (pairwise)"

    def go():
        rng = np.random.default_rng(24)
        a = rng.uniform(size=(4, 20, 24)) > 0.5; b = rng.uniform(size=(3, 20, 24)) > 0.6
        b[2] = False                                # an empty mask
        got = _np(fn(a, b))
        want = ref(a, b)
        if got.shape != want.shape:
            return _report(name, False, f"expected shape {want.shape} (N, M), got {got.shape}")
        ok = np.allclose(got, want, atol=1e-6)
        ok &= np.allclose(_np(fn(a, a)).diagonal(), 1.0)
        return _report(name, ok, "flatten to (N, H*W); intersection = A @ B.T; union = |A| + |B| - inter; empty union -> 0")
    return _run(name, go, "")


def check_penn_fudan_target(fn):
    import torch
    from .day3 import penn_fudan_target as ref
    name = "Penn-Fudan target conversion"

    def go():
        m = np.zeros((30, 40), np.uint8)
        m[2:10, 3:9] = 1; m[12:28, 20:35] = 3; m[5:8, 30:38] = 2       # ids 1, 2, 3 (not in reading order)
        got = fn(m); want = ref(m)
        for k in ("boxes", "labels", "masks"):
            if k not in got:
                return _report(name, False, f"the target dict needs the key '{k}'")
        gb, gl, gm = (torch.as_tensor(got[k]) for k in ("boxes", "labels", "masks"))
        if gb.dtype != torch.float32 or gl.dtype != torch.int64:
            return _report(name, False, f"dtypes: boxes float32 (got {gb.dtype}), labels int64 (got {gl.dtype})")
        if tuple(gm.shape) != (3, 30, 40):
            return _report(name, False, f"masks should be (N, H, W) = (3, 30, 40), got {tuple(gm.shape)}")
        ok = torch.allclose(gb, want["boxes"]) and torch.equal(gl, want["labels"]) and torch.equal(gm.to(torch.uint8), want["masks"])
        return _report(name, ok, f"expected boxes {want['boxes'].tolist()}: one mask per id != 0 (sorted), box = "
                                  "(x_min, y_min, x_max, y_max) of its pixels, label 1 for every person")
    return _run(name, go, "")


def check_maskrcnn_heads(model, num_classes=2):
    import torch
    name = "Mask R-CNN predictors replaced"

    def go():
        rh = model.roi_heads
        ok = rh.box_predictor.cls_score.out_features == num_classes
        ok &= rh.box_predictor.bbox_pred.out_features == 4 * num_classes
        ok &= rh.mask_predictor.mask_fcn_logits.out_channels == num_classes
        return _report(name, ok, f"box predictor: FastRCNNPredictor(in_features, {num_classes}); "
                                  f"mask predictor: MaskRCNNPredictor(in_channels, 256, {num_classes})")
    return _run(name, go, "")


def check_mask_ap(fn):
    from .day3 import mask_average_precision as ref
    name = "mask AP (instance)"

    def go():
        rng = np.random.default_rng(25)
        preds, gts = [], []
        for _ in range(4):
            H, W = 32, 32
            g = np.zeros((2, H, W), bool); g[0, 2:14, 2:14] = True; g[1, 18:30, 15:28] = True
            p = np.zeros((4, H, W), bool)
            p[0] = g[0]; p[1, 18:30, 17:30] = True; p[2, 0:6, 20:31] = True; p[3] = g[0]
            p[1] ^= rng.uniform(size=(H, W)) > 0.97
            preds.append({"masks": p, "scores": rng.uniform(0.3, 1.0, 4)}); gts.append({"masks": g})
        ok = True
        for thr in (0.5, 0.75):
            got, want = float(fn(preds, gts, thr)), ref(preds, gts, thr)
            if abs(got - want) > 1e-3:
                ok = False
                print(f"   IoU threshold {thr}: got {got:.3f}, expected {want:.3f}")
        return _report(name, ok, "sort predictions by score; a prediction is TP if its best still-unmatched GT has mask IoU >= thr; "
                                  "pool (score, tp) over images, then precision/recall -> boxes.average_precision(..., 'interp101')")
    return _run(name, go, "")


# ----------------------------------------------------------------- D3-04 ---
def check_patchify(fn):
    import torch
    import torch.nn as nn
    from .day3 import patchify as ref
    name = "patchify == strided conv"

    def go():
        torch.manual_seed(0)
        x = torch.randn(2, 3, 32, 48)
        got = fn(x, 16)
        want = ref(x, 16)
        if tuple(got.shape) != tuple(want.shape):
            return _report(name, False, f"expected shape {tuple(want.shape)} = (B, num_patches, C*p*p), got {tuple(got.shape)}")
        if not torch.allclose(got, want):
            return _report(name, False, "patches must be in row-major order and each flattened as (C, p, p), "
                                        "the same order as conv.weight.reshape(D, -1)")
        conv = nn.Conv2d(3, 8, 16, 16)
        a = conv(x).flatten(2).transpose(1, 2)
        b = got @ conv.weight.reshape(8, -1).T + conv.bias
        return _report(name, torch.allclose(a, b, atol=1e-5), "the linear projection of your patches must equal the conv output")
    return _run(name, go, "x.unfold(2, p, p).unfold(3, p, p) -> permute -> reshape")


def check_attention(fn):
    import torch
    import torch.nn.functional as F
    name = "scaled dot-product attention"

    def go():
        torch.manual_seed(0)
        q, k, v = torch.randn(2, 4, 5, 8), torch.randn(2, 4, 7, 8), torch.randn(2, 4, 7, 6)
        out = fn(q, k, v)
        if not (isinstance(out, (tuple, list)) and len(out) == 2):
            return _report(name, False, "return (output, weights)")
        o, w = out
        want = F.scaled_dot_product_attention(q, k, v)
        ok = tuple(w.shape) == (2, 4, 5, 7) and torch.allclose(w.sum(-1), torch.ones(2, 4, 5), atol=1e-5)
        ok &= torch.allclose(o, want, atol=1e-5)
        return _report(name, ok, "weights = softmax(q @ k^T / sqrt(d), dim=-1) (rows sum to 1); output = weights @ v")
    return _run(name, go, "")


def check_zero_shot(fn):
    import torch
    from .day3 import zero_shot_classify as ref
    name = "zero_shot_classify"

    def go():
        torch.manual_seed(0)
        img = torch.randn(10, 16) * torch.linspace(0.5, 3, 10)[:, None]     # NOT normalised on purpose
        txt = torch.randn(4, 16) * torch.tensor([[1.0], [5.0], [0.2], [2.0]])
        pred, probs = fn(img, txt)
        wp, wpr = ref(img, txt)
        ok = torch.equal(torch.as_tensor(pred).long(), wp) and torch.allclose(torch.as_tensor(probs).float(), wpr, atol=1e-4)
        return _report(name, ok, "L2-normalise BOTH image and text features first (otherwise long text vectors win), "
                                  "cos = img @ txt.T, probs = softmax(logit_scale * cos)")
    return _run(name, go, "")


def check_prompt_ensemble(fn):
    import torch
    import torch.nn.functional as F
    name = "prompt ensemble"

    def go():
        torch.manual_seed(0)
        t = F.normalize(torch.randn(5, 3, 8), dim=-1)          # (templates, classes, D)
        got = fn(t)
        want = F.normalize(F.normalize(t, dim=-1).mean(0), dim=-1)
        ok = tuple(got.shape) == (3, 8) and torch.allclose(got, want, atol=1e-5)
        return _report(name, ok, "normalise each prompt embedding, average over templates (dim 0), normalise again")
    return _run(name, go, "")
