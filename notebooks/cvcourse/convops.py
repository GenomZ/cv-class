"""Reference implementations of the convolution-family operations (NumPy).

These are deliberately simple and readable, not fast. They are the ground truth
the exercise checkers compare student code against.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Shape arithmetic
# ---------------------------------------------------------------------------
def conv_output_size(n_in: int, k: int, stride: int = 1, pad: int = 0, dilation: int = 1) -> int:
    """Spatial output size of a convolution along one axis."""
    k_eff = dilation * (k - 1) + 1
    return (n_in + 2 * pad - k_eff) // stride + 1


def transposed_conv_output_size(n_in: int, k: int, stride: int = 1, pad: int = 0, output_pad: int = 0) -> int:
    return (n_in - 1) * stride - 2 * pad + k + output_pad


def receptive_field(layers):
    """Receptive field, effective stride (jump) and start offset after a stack of layers.

    layers: list of (kernel, stride, padding[, dilation]) tuples, input to output.
    Returns a list of dicts, one per layer, following the standard RF recursion:
        jump_out = jump_in * stride
        rf_out   = rf_in + (k_eff - 1) * jump_in
        start_out = start_in + ((k_eff - 1) / 2 - pad) * jump_in
    """
    rf, jump, start = 1, 1, 0.5
    out = []
    for layer in layers:
        k, s, p = layer[:3]
        d = layer[3] if len(layer) > 3 else 1
        k_eff = d * (k - 1) + 1
        rf = rf + (k_eff - 1) * jump
        start = start + ((k_eff - 1) / 2 - p) * jump
        jump = jump * s
        out.append(dict(rf=rf, jump=jump, start=start))
    return out


# ---------------------------------------------------------------------------
# Padding helper
# ---------------------------------------------------------------------------
def pad2d(x: np.ndarray, pad: int, value: float = 0.0) -> np.ndarray:
    """Zero-pad the last two axes of x by `pad` on every side."""
    if pad == 0:
        return x
    width = [(0, 0)] * (x.ndim - 2) + [(pad, pad), (pad, pad)]
    return np.pad(x, width, mode="constant", constant_values=value)


# ---------------------------------------------------------------------------
# Convolutions
# ---------------------------------------------------------------------------
def cross_correlate2d(x: np.ndarray, k: np.ndarray, stride: int = 1, pad: int = 0) -> np.ndarray:
    """Single-channel 2-D cross-correlation (what DL frameworks call 'convolution').

    x: (H, W), k: (kh, kw). Returns (Ho, Wo).
    """
    x = pad2d(x, pad)
    H, W = x.shape
    kh, kw = k.shape
    Ho = (H - kh) // stride + 1
    Wo = (W - kw) // stride + 1
    out = np.zeros((Ho, Wo), dtype=np.result_type(x, k))
    for i in range(Ho):
        for j in range(Wo):
            patch = x[i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[i, j] = np.sum(patch * k)
    return out


def convolve2d_true(x: np.ndarray, k: np.ndarray, stride: int = 1, pad: int = 0) -> np.ndarray:
    """Mathematical convolution = cross-correlation with the kernel flipped."""
    return cross_correlate2d(x, k[::-1, ::-1], stride, pad)


def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None,
           stride: int = 1, pad: int = 0) -> np.ndarray:
    """Multi-channel convolution layer, framework convention (channels first).

    x: (C_in, H, W)
    w: (C_out, C_in, kh, kw)
    b: (C_out,) or None
    returns (C_out, Ho, Wo)
    """
    C_in, H, W = x.shape
    C_out, C_in_w, kh, kw = w.shape
    assert C_in == C_in_w, "input channels must match kernel channels"
    xp = pad2d(x, pad)
    Ho = (H + 2 * pad - kh) // stride + 1
    Wo = (W + 2 * pad - kw) // stride + 1
    out = np.zeros((C_out, Ho, Wo), dtype=np.result_type(x, w))
    for o in range(C_out):
        for i in range(Ho):
            for j in range(Wo):
                patch = xp[:, i * stride:i * stride + kh, j * stride:j * stride + kw]
                out[o, i, j] = np.sum(patch * w[o])
        if b is not None:
            out[o] += b[o]
    return out


def im2col(x: np.ndarray, kh: int, kw: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """Unfold (C, H, W) into a matrix of patches: (Ho*Wo, C*kh*kw)."""
    C, H, W = x.shape
    xp = pad2d(x, pad)
    Ho = (H + 2 * pad - kh) // stride + 1
    Wo = (W + 2 * pad - kw) // stride + 1
    cols = np.empty((Ho * Wo, C * kh * kw), dtype=x.dtype)
    idx = 0
    for i in range(Ho):
        for j in range(Wo):
            patch = xp[:, i * stride:i * stride + kh, j * stride:j * stride + kw]
            cols[idx] = patch.reshape(-1)
            idx += 1
    return cols


def conv2d_im2col(x: np.ndarray, w: np.ndarray, b=None, stride: int = 1, pad: int = 0) -> np.ndarray:
    """Same result as conv2d, computed as ONE matrix multiplication."""
    C_out, C_in, kh, kw = w.shape
    cols = im2col(x, kh, kw, stride, pad)                  # (Ho*Wo, C_in*kh*kw)
    Wm = w.reshape(C_out, -1)                              # (C_out, C_in*kh*kw)
    out = cols @ Wm.T                                      # (Ho*Wo, C_out)
    if b is not None:
        out = out + b
    Ho = (x.shape[1] + 2 * pad - kh) // stride + 1
    Wo = (x.shape[2] + 2 * pad - kw) // stride + 1
    return out.T.reshape(C_out, Ho, Wo)


def conv_params(c_in: int, c_out: int, k: int, bias: bool = True) -> int:
    return c_out * c_in * k * k + (c_out if bias else 0)


def conv_flops(c_in: int, c_out: int, k: int, h_out: int, w_out: int) -> int:
    """Multiply-accumulates of one conv layer (bias ignored)."""
    return c_out * h_out * w_out * c_in * k * k


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------
def pool2d(x: np.ndarray, k: int = 2, stride: int | None = None, mode: str = "max") -> np.ndarray:
    """Pooling over the last two axes of x (..., H, W)."""
    stride = stride or k
    H, W = x.shape[-2:]
    Ho = (H - k) // stride + 1
    Wo = (W - k) // stride + 1
    out = np.zeros(x.shape[:-2] + (Ho, Wo), dtype=x.dtype)
    red = np.max if mode == "max" else np.mean
    for i in range(Ho):
        for j in range(Wo):
            out[..., i, j] = red(x[..., i * stride:i * stride + k, j * stride:j * stride + k], axis=(-2, -1))
    return out


# ---------------------------------------------------------------------------
# Transposed convolution (used in Day 3)
# ---------------------------------------------------------------------------
def transposed_conv2d(x: np.ndarray, k: np.ndarray, stride: int = 1, pad: int = 0) -> np.ndarray:
    """Single-channel transposed convolution by 'scatter-add' of the kernel.

    Every input pixel paints a copy of the kernel, scaled by its value, at
    position (i*stride, j*stride) in the output. Overlaps add up, which is why
    stride < kernel gives the famous checkerboard artefact.
    """
    H, W = x.shape
    kh, kw = k.shape
    Ho = (H - 1) * stride + kh
    Wo = (W - 1) * stride + kw
    out = np.zeros((Ho, Wo), dtype=np.result_type(x, k))
    for i in range(H):
        for j in range(W):
            out[i * stride:i * stride + kh, j * stride:j * stride + kw] += x[i, j] * k
    if pad:
        out = out[pad:Ho - pad, pad:Wo - pad]
    return out


def bilinear_upsample(x: np.ndarray, factor: int = 2) -> np.ndarray:
    """Bilinear upsampling of (H, W) by an integer factor (align_corners=False)."""
    H, W = x.shape
    Ho, Wo = H * factor, W * factor
    ys = (np.arange(Ho) + 0.5) / factor - 0.5
    xs = (np.arange(Wo) + 0.5) / factor - 0.5
    ys = np.clip(ys, 0, H - 1)
    xs = np.clip(xs, 0, W - 1)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, H - 1); wy = (ys - y0)[:, None]
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, W - 1); wx = (xs - x0)[None, :]
    top = x[y0][:, x0] * (1 - wx) + x[y0][:, x1] * wx
    bot = x[y1][:, x0] * (1 - wx) + x[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


# ---------------------------------------------------------------------------
# Classic hand-made kernels for the Day 1 demo
# ---------------------------------------------------------------------------
KERNELS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
    "box blur": np.full((3, 3), 1 / 9.0),
    "gaussian": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float) / 16,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
    "sobel x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float),
    "sobel y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float),
    "laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float),
    "emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=float),
}
