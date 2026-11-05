"""Day-2 (object detection) exercise checkers. Re-exported by `cvcourse.tests`.

Every checker prints an ASCII `[PASS]` / `[FAIL]` line with a hint and returns a bool, and
never crashes on a stub: a `NotImplementedError` (or any exception) becomes a `[FAIL]`.
Pass `verbose=False` to only get the bool (used to fall back to reference code).
"""
from __future__ import annotations

import math
import traceback

import numpy as np

from . import boxes, shapes

_PASS = "[PASS]"
_FAIL = "[FAIL]"

__all__ = [
    "check_giou", "check_precision_recall", "check_soft_nms", "check_batched_nms",
    "check_rpn_head", "check_roi_align", "check_box_predictor",
    "check_dense_loss", "check_decode_grid", "check_fpn_shapes", "check_retina_head",
    "check_focal_loss_torch", "check_centerness",
    "check_hungarian", "check_detr_cost",
]


def _report(name, ok, hint="", verbose=True):
    if verbose:
        print(f"{_PASS if ok else _FAIL}  {name}" + (f"\n      hint: {hint}" if (hint and not ok) else ""))
    return bool(ok)


def _np(x):
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().double().numpy()
    except ImportError:  # pragma: no cover
        pass
    return np.asarray(x, dtype=float)


def _close(a, b, tol=1e-5):
    a = _np(a); b = _np(b)
    return a.shape == b.shape and np.allclose(a, b, atol=tol, rtol=tol)


def _guard(name, verbose, fn):
    """Run fn(); turn exceptions into a FAIL with a short message."""
    try:
        return fn()
    except NotImplementedError:
        return _report(name, False, "not implemented yet (the stub still raises NotImplementedError)", verbose)
    except Exception as e:  # noqa: BLE001
        if verbose:
            tb = traceback.format_exception_only(type(e), e)[-1].strip()
            print(f"   your code raised: {tb}")
        return _report(name, False, "fix the exception above", verbose)


# ------------------------------------------------------------------ D2-01 ---
def check_giou(fn, verbose=True):
    """fn(a, b) -> GIoU for aligned pairs (N, 4) x (N, 4) -> (N,)."""
    name = "generalized IoU"

    def run():
        a = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10], [2, 2, 8, 8], [0, 0, 4, 4]], float)
        b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [10, 0, 20, 10], [20, 20, 30, 30], [0, 0, 10, 10], [6, 0, 10, 4]], float)
        want = boxes.generalized_iou(a, b)
        got = _np(fn(a, b)).reshape(-1)
        ok = _close(got, want, 1e-6)
        if not ok and verbose and got.shape == want.shape:
            for i in range(len(a)):
                if not np.isclose(got[i], want[i], atol=1e-6):
                    print(f"   GIoU({a[i].tolist()}, {b[i].tolist()}) = {got[i]:.4f}, expected {want[i]:.4f}")
        return _report(name, ok, "GIoU = IoU - (area(C) - union) / area(C), C = smallest box enclosing both; "
                                 "it is negative when the boxes do not overlap", verbose)
    return _guard(name, verbose, run)


def check_precision_recall(fn, verbose=True):
    """fn(pred_boxes, pred_scores, gt_boxes, iou_threshold) -> (precision, recall, ...)."""
    name = "precision / recall by greedy matching"

    def run():
        rng = np.random.default_rng(21)
        ok = True
        for trial in range(6):
            gt = rng.uniform(0, 80, (4, 2)); gt = np.concatenate([gt, gt + rng.uniform(15, 40, (4, 2))], 1)
            pr = np.concatenate([gt + rng.normal(0, 4, gt.shape), gt[:2] + rng.normal(0, 6, (2, 4)),
                                 rng.uniform(0, 60, (3, 2)) @ np.array([[1, 0, 1, 0], [0, 1, 0, 1]]) + [0, 0, 20, 20]])
            sc = rng.uniform(0, 1, len(pr))
            wp, wr, _ = boxes.precision_recall(pr, sc, gt, 0.5)
            out = fn(pr, sc, gt, 0.5)
            gp, gr = out[0], out[1]
            if not (_close(gp, wp) and _close(gr, wr)):
                ok = False
        return _report(name, ok, "sort by score; a prediction is TP if its best *unmatched* GT has IoU >= thr; "
                                 "precision = cumTP / (cumTP + cumFP), recall = cumTP / #GT", verbose)
    return _guard(name, verbose, run)


def check_soft_nms(fn, verbose=True):
    """fn(boxes, scores, sigma, score_threshold) -> (keep indices, kept scores)."""
    name = "Soft-NMS (Gaussian)"

    def run():
        rng = np.random.default_rng(22)
        ok = True
        for _ in range(4):
            c = rng.uniform(0, 60, (3, 2)); bx = []
            for cc in c:
                for _ in range(4):
                    j = rng.normal(0, 4, 2); bx.append(np.concatenate([cc + j, cc + j + 30]))
            bx = np.array(bx); sc = rng.uniform(0.1, 1.0, len(bx))
            wk, ws = boxes.soft_nms(bx, sc, 0.5, 0.001)
            gk, gs = fn(bx.copy(), sc.copy(), 0.5, 0.001)
            gk = np.asarray(gk).reshape(-1); gs = _np(gs).reshape(-1)
            if not (np.array_equal(gk, wk) and _close(gs, ws, 1e-6)):
                ok = False
                if verbose:
                    print(f"   kept {gk.tolist()[:8]}... vs expected {wk.tolist()[:8]}...")
                break
        return _report(name, ok, "pick the max, then multiply the *other* scores by exp(-IoU^2 / sigma); "
                                 "drop scores below the threshold; repeat (do not remove overlapping boxes outright)", verbose)
    return _guard(name, verbose, run)


def check_batched_nms(fn, verbose=True):
    """fn(boxes, scores, labels, iou_threshold) -> kept indices (class-wise NMS)."""
    name = "class-wise (batched) NMS"

    def run():
        rng = np.random.default_rng(23)
        bx = rng.uniform(0, 40, (12, 2)); bx = np.concatenate([bx, bx + 30], 1)
        sc = rng.uniform(0, 1, 12); lb = rng.integers(0, 3, 12)
        want = boxes.batched_nms(bx, sc, lb, 0.5)
        got = np.asarray(fn(bx, sc, lb, 0.5)).reshape(-1)
        return _report(name, np.array_equal(np.sort(got), np.sort(want)), "run NMS separately per class (or offset boxes by label)", verbose)
    return _guard(name, verbose, run)


# ------------------------------------------------------------------ D2-02 ---
def check_rpn_head(cls, verbose=True):
    """cls(in_channels, num_anchors) -> module; forward(x) -> (objectness (N, A, H, W), deltas (N, 4A, H, W))."""
    name = "RPNHead"

    def run():
        import torch
        torch.manual_seed(0)
        head = cls(64, 6)
        x = torch.randn(2, 64, 16, 20, requires_grad=True)
        obj, d = head(x)
        ok_shape = tuple(obj.shape) == (2, 6, 16, 20) and tuple(d.shape) == (2, 24, 16, 20)
        n = sum(p.numel() for p in head.parameters())
        want_n = 64 * 64 * 9 + 64 + 64 * 6 + 6 + 64 * 24 + 24
        (obj.sum() + d.sum()).backward()
        ok_grad = x.grad is not None and float(x.grad.abs().sum()) > 0
        if verbose and not ok_shape:
            print(f"   got objectness {tuple(obj.shape)}, deltas {tuple(d.shape)}; expected (2, 6, 16, 20), (2, 24, 16, 20)")
        if verbose and n != want_n:
            print(f"   {n:,d} parameters, expected {want_n:,d} (3x3 conv C->C, 1x1 C->A, 1x1 C->4A, all with bias)")
        return _report(name, ok_shape and n == want_n and ok_grad,
                       "3x3 conv (same padding) + ReLU, then two sibling 1x1 convs with A and 4A output channels", verbose)
    return _guard(name, verbose, run)


def check_roi_align(fn, verbose=True):
    """fn(feature (C, H, W), rois (K, 4), output_size, spatial_scale, sampling_ratio) vs torchvision.ops.roi_align(aligned=True)."""
    name = "RoIAlign vs torchvision.ops.roi_align"

    def run():
        import torch
        from torchvision.ops import roi_align as tv_roi_align
        g = torch.Generator().manual_seed(0)
        feat = torch.randn(3, 16, 16, generator=g)
        rois = torch.tensor([[8.0, 8.0, 40.0, 40.0], [3.3, 10.7, 60.2, 31.9], [20.0, 20.0, 27.0, 50.0], [1.0, 1.0, 62.0, 62.0]])
        ok = True
        for out, scale, sr in [(7, 0.25, 2), (3, 0.25, 2), (5, 0.25, 1)]:
            want = tv_roi_align(feat[None], [rois], output_size=out, spatial_scale=scale, sampling_ratio=sr, aligned=True)
            got = fn(feat, rois, out, scale, sr)
            if not _close(got, want, 1e-4):
                ok = False
                if verbose:
                    gs = tuple(getattr(got, "shape", ()))
                    print(f"   output_size={out}, sampling_ratio={sr}: shape {gs} vs {tuple(want.shape)}"
                          + ("" if gs != tuple(want.shape) else f", max |diff| = {float((torch.as_tensor(got) - want).abs().max()):.4f}"))
        return _report(name, ok, "scale the box, shift by -0.5 (aligned), split into P x P bins, sample "
                                 "sampling_ratio^2 regular points per bin with bilinear interpolation, average", verbose)
    return _guard(name, verbose, run)


def check_box_predictor(model, num_classes=2, verbose=True):
    """The detector's last layers must output `num_classes` classes (Faster R-CNN, RetinaNet or FCOS)."""
    name = f"detector head replaced for {num_classes} classes"

    def run():
        ok = False; msg = ""
        if hasattr(model, "roi_heads"):                       # Faster / Mask R-CNN
            p = model.roi_heads.box_predictor
            ok = p.cls_score.out_features == num_classes and p.bbox_pred.out_features == 4 * num_classes
            msg = f"cls_score -> {p.cls_score.out_features}, bbox_pred -> {p.bbox_pred.out_features}"
        elif hasattr(model, "head") and hasattr(model.head, "classification_head"):   # RetinaNet / FCOS
            h = model.head.classification_head
            A = h.num_anchors
            ok = h.num_classes == num_classes and h.cls_logits.out_channels == A * num_classes
            msg = f"num_classes={h.num_classes}, cls_logits out_channels={h.cls_logits.out_channels}"
        if verbose:
            print("   " + msg)
        return _report(name, ok, "torchvision counts the background as a class: 1 object class -> num_classes=2", verbose)
    return _guard(name, verbose, run)


# ------------------------------------------------------------------ D2-03 ---
def check_dense_loss(fn, verbose=True):
    """fn(pred, target) -> (total, dict(obj, box, cls)); compared to `day2.dense_loss` (weights 1, 5, 1)."""
    name = "three-term dense detection loss"

    def run():
        import torch
        from . import day2
        g = torch.Generator().manual_seed(0)
        _, bx, lb = day2.shapes_tensors(4, 128, seed=3)
        t = day2.grid_target_tensor(bx, lb)
        pred = torch.randn(t.shape, generator=g)
        pred[:, 1:5] = torch.rand(pred[:, 1:5].shape, generator=g)
        ok = True
        for target in (t, torch.zeros_like(t)):                # a normal batch, and one with no objects
            want_t, want_p = day2.dense_loss(pred, target)
            got_t, got_p = fn(pred.clone(), target.clone())
            if not (torch.isfinite(torch.as_tensor(got_t)) and abs(float(got_t) - float(want_t)) < 1e-4):
                ok = False
                if verbose:
                    print(f"   {'empty batch' if target.sum() == 0 else 'batch'}: total {float(got_t):.4f}, expected {float(want_t):.4f}"
                          + "".join(f" | {k}: {float(got_p.get(k, float('nan'))):.4f} vs {float(want_p[k]):.4f}" for k in want_p))
        p2 = pred.clone().requires_grad_(True)
        tot, _ = fn(p2, t)
        tot.backward()
        neg = t[:, 0] < 0.5
        box_grad_on_neg = float(p2.grad[:, 1:5].permute(0, 2, 3, 1)[neg].abs().sum())
        if box_grad_on_neg > 1e-8:
            ok = False
            if verbose:
                print("   the box loss produces gradients on cells without an object")
        return _report(name, ok, "obj: BCE-with-logits on every cell; box: L1 summed over 4 numbers, mean over positive "
                                 "cells; cls: cross-entropy on positive cells; total = 1*obj + 5*box + 1*cls; "
                                 "return 0 (not NaN) for box/cls when there are no positives", verbose)
    return _guard(name, verbose, run)


def check_decode_grid(fn, verbose=True):
    """fn(T (G, G, 5 + C), size, score_threshold) -> (boxes, scores, labels) like shapes.decode_grid."""
    name = "decode the grid"

    def run():
        rng = np.random.default_rng(24)
        img, bx, lb, _ = shapes.make_sample(rng, 128)
        T = shapes.grid_targets(bx, lb, 8, 128, 3)
        gb, gs, gl = fn(T, 128, 0.5)
        order = np.argsort(np.asarray(gb)[:, 0]); worder = np.argsort(bx[:, 0])
        ok = len(gb) == len(bx) and _close(np.asarray(gb)[order], bx[worder], 1e-4) and np.array_equal(np.asarray(gl)[order], lb[worder])
        return _report(name, ok, "for every cell with objectness >= threshold: (cx, cy, w, h) * size -> xyxy; label = argmax of the class part", verbose)
    return _guard(name, verbose, run)


def check_fpn_shapes(cls, verbose=True):
    """cls(in_channels_list, out_channels); forward(list finest-first) -> list finest-first."""
    name = "FPN shapes and top-down pathway"

    def run():
        import torch
        torch.manual_seed(0)
        fpn = cls([64, 128, 256], 32).eval()
        feats = [torch.randn(1, 64, 32, 32), torch.randn(1, 128, 16, 16), torch.randn(1, 256, 8, 8)]
        with torch.no_grad():
            outs = fpn(feats)
            shapes_ok = [tuple(o.shape) for o in outs] == [(1, 32, 32, 32), (1, 32, 16, 16), (1, 32, 8, 8)]
            feats2 = [f.clone() for f in feats]; feats2[2] = feats2[2] + 1.0
            outs2 = fpn(feats2)
            topdown = float((outs2[0] - outs[0]).abs().max()) > 1e-6       # changing C5 must change P3
            feats3 = [f.clone() for f in feats]; feats3[0] = feats3[0] + 1.0
            outs3 = fpn(feats3)
            no_bottomup = float((outs3[2] - outs[2]).abs().max()) < 1e-6   # changing C3 must not change P5
        if verbose and not shapes_ok:
            print(f"   got {[tuple(o.shape) for o in outs]}")
        if verbose and not topdown:
            print("   P3 does not depend on C5: did you add the upsampled coarser level?")
        if verbose and not no_bottomup:
            print("   P5 depends on C3: the pathway should be top-down only")
        return _report(name, shapes_ok and topdown and no_bottomup,
                       "lateral 1x1 to out_channels; from the coarsest level down: up = interpolate(prev, size), "
                       "merged = lateral + up; then a 3x3 conv on every merged map", verbose)
    return _guard(name, verbose, run)


def check_retina_head(cls, verbose=True):
    """cls(in_channels, num_anchors, num_classes) shared over levels -> (cls outputs, box outputs)."""
    name = "RetinaNet head"

    def run():
        import torch
        torch.manual_seed(0)
        head = cls(32, 9, 3)
        feats = [torch.randn(1, 32, 16, 16), torch.randn(1, 32, 8, 8), torch.randn(1, 32, 4, 4)]
        cls_out, box_out = head(feats)
        ok_shapes = [tuple(c.shape) for c in cls_out] == [(1, 27, 16, 16), (1, 27, 8, 8), (1, 27, 4, 4)] and \
                    [tuple(b.shape) for b in box_out] == [(1, 36, 16, 16), (1, 36, 8, 8), (1, 36, 4, 4)]
        n_params = sum(p.numel() for p in head.parameters())
        head2 = cls(32, 9, 3)
        _ = head2(feats[:1])
        ok_shared = sum(p.numel() for p in head2.parameters()) == n_params
        last_cls = [m for m in head.modules() if isinstance(m, torch.nn.Conv2d) and m.out_channels == 27]
        ok_prior = bool(last_cls) and last_cls[-1].bias is not None and \
            abs(float(torch.sigmoid(last_cls[-1].bias.detach()).mean()) - 0.01) < 2e-3
        if verbose and not ok_shapes:
            print(f"   cls {[tuple(c.shape) for c in cls_out]}, box {[tuple(b.shape) for b in box_out]}")
        if verbose and ok_shapes and not ok_prior:
            print("   classification bias should start at -log((1 - 0.01) / 0.01) so every anchor predicts p = 0.01")
        return _report(name, ok_shapes and ok_shared and ok_prior,
                       "the same towers for every level; cls output A*C channels, box output A*4; "
                       "bias of the cls output initialised to the prior p=0.01", verbose)
    return _guard(name, verbose, run)


def check_focal_loss_torch(fn, verbose=True):
    """fn(logits, targets, gamma, alpha) -> scalar mean; compared to torchvision.ops.sigmoid_focal_loss."""
    name = "focal loss (torch, on logits)"

    def run():
        import torch
        from torchvision.ops import sigmoid_focal_loss
        g = torch.Generator().manual_seed(0)
        x = torch.randn(500, generator=g) * 3; t = (torch.rand(500, generator=g) < 0.1).float()
        ok = True
        for gamma, alpha in [(2.0, 0.25), (0.0, 0.25), (5.0, 0.5)]:
            want = sigmoid_focal_loss(x, t, alpha=alpha, gamma=gamma, reduction="mean")
            got = fn(x, t, gamma, alpha)
            if abs(float(got) - float(want)) > 1e-5:
                ok = False
                if verbose:
                    print(f"   gamma={gamma}, alpha={alpha}: got {float(got):.5f}, expected {float(want):.5f}")
        xl = torch.tensor([60.0, -60.0]); tl = torch.tensor([0.0, 1.0])
        if not torch.isfinite(torch.as_tensor(fn(xl, tl, 2.0, 0.25))):
            ok = False
            if verbose:
                print("   not finite for large logits: use F.binary_cross_entropy_with_logits for the log term")
        return _report(name, ok, "ce = BCE-with-logits (no reduction); p_t = p*t + (1-p)*(1-t); "
                                 "loss = alpha_t * (1 - p_t)^gamma * ce, alpha_t = alpha*t + (1-alpha)*(1-t); mean", verbose)
    return _guard(name, verbose, run)


def check_centerness(fn, verbose=True):
    """fn(ltrb (..., 4)) -> centre-ness in [0, 1]."""
    name = "FCOS centre-ness"

    def run():
        from . import day2
        rng = np.random.default_rng(25)
        ltrb = rng.uniform(0.5, 30, (50, 4))
        ok = _close(fn(ltrb), day2.centerness(ltrb), 1e-6) and np.isclose(float(_np(fn(np.array([[5., 5., 5., 5.]]))).reshape(-1)[0]), 1.0)
        return _report(name, ok, "sqrt( min(l,r)/max(l,r) * min(t,b)/max(t,b) )", verbose)
    return _guard(name, verbose, run)


# ------------------------------------------------------------------ D2-04 ---
def check_hungarian(fn, verbose=True):
    """fn(cost (Q, G)) -> (query_idx, gt_idx): an optimal one-to-one assignment covering every GT."""
    name = "Hungarian matching"

    def run():
        from scipy.optimize import linear_sum_assignment
        rng = np.random.default_rng(26)
        ok = True
        for Q, G in [(3, 2), (6, 4), (10, 3), (5, 5)]:
            c = rng.uniform(0, 1, (Q, G))
            r, k = fn(c)
            r = np.asarray(r).reshape(-1); k = np.asarray(k).reshape(-1)
            wr, wk = linear_sum_assignment(c)
            valid = len(r) == G and len(set(r.tolist())) == G and sorted(k.tolist()) == list(range(G))
            if not valid or not np.isclose(c[r, k].sum(), c[wr, wk].sum()):
                ok = False
                if verbose:
                    print(f"   Q={Q}, G={G}: total cost {c[r, k].sum() if valid else float('nan'):.3f}, optimum {c[wr, wk].sum():.3f}"
                          + ("" if valid else " (not a one-to-one matching of every GT)"))
        return _report(name, ok, "scipy.optimize.linear_sum_assignment(cost) returns (rows, cols) of the cheapest matching", verbose)
    return _guard(name, verbose, run)


def check_detr_cost(fn, verbose=True):
    """fn(pred_logits, pred_boxes, gt_labels, gt_boxes) -> (Q, G) cost, weights (1, 5, 2) like DETR."""
    name = "DETR matching cost"

    def run():
        import torch
        from . import day2
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(6, 4, generator=g)
        pb = torch.rand(6, 4, generator=g) * 0.5 + 0.25
        gl = torch.tensor([0, 2]); gb = torch.tensor([[0.3, 0.4, 0.2, 0.3], [0.7, 0.6, 0.3, 0.2]])
        want = day2.detr_cost(logits, pb, gl, gb)
        got = fn(logits, pb, gl, gb)
        ok = _close(got, want, 1e-5)
        if verbose and not ok:
            print(f"   got shape {tuple(np.shape(_np(got)))}, expected {tuple(want.shape)}")
        return _report(name, ok, "cost = -1 * softmax(logits)[:, gt_labels] + 5 * cdist(boxes, p=1) "
                                 "- 2 * GIoU(cxcywh -> xyxy)", verbose)
    return _guard(name, verbose, run)
