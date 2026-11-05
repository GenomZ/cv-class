# Day 1 quiz - solutions

| Q | correct | topic (notebook) |
|---|---|---|
| 1 | A, B, C, E | output size and parameter count (D1-01) |
| 2 | A, B, C, E | receptive fields (D1-01, D1-03 VGG) |
| 3 | A, C, D, E | 1x1 convolutions (D1-01, D1-03) |
| 4 | A, B, C, E, F | regularisation (D1-02) |
| 5 | A, C, D, E | optimisers, schedules, LR range test (D1-02) |
| 6 | A, B, C, D | debugging recipe (D1-02) |
| 7 | A, C, D | augmentation, TTA (D1-02) |
| 8 | A, B, D, E | residual / dense / depthwise / Inception (D1-03) |
| 9 | A, B, C, E | transfer learning, data scaling (D1-04) |
| 10 | A, B, D, E | filters, Grad-CAM, shortcut learning, hooks (D1-04) |

---

**1. A, B, C, E.** "Same" padding p = (k - 1) / 2 = 2 keeps 32 x 32; 16 filters give 16 channels (A).
Pooling 2 x 2 / 2 halves the size: 16 x 16 x 16 (B). Parameters: 5 * 5 * 3 * 16 weights + 16
biases = 1 216 (C). Pooling has **no** parameters (D is false). Without padding:
(32 - 5) / 1 + 1 = 28 (E).

**2. A, B, C, E.** Receptive field recursion r += (k - 1) * jump: 1 -> 3 -> 5 (A, D false). Weights
2 * 9 C^2 = 18 C^2 vs 25 C^2 (B), and one more ReLU between them (C). A stride multiplies the
"jump", so every later layer adds (k - 1) * jump > k - 1 pixels (E).

**3. A, C, D, E.** A 1 x 1 conv is a linear layer applied at every pixel across channels. It does
not look at neighbours, so it does **not** enlarge the receptive field (B false).

**4. A, B, C, E, F.** Training longer makes overfitting worse (D false; *early stopping* would
be the regulariser). BatchNorm's main role is optimisation, but the noise of batch statistics has
a mild regularising effect (F, as in the original course's quiz).

**5. A, C, D, E.** B is false: `Adam(weight_decay=...)` adds an L2 term to the gradient, which is then
rescaled by Adam's per-parameter step size, so weights with large gradients are barely decayed;
AdamW decays the weights directly ("decoupled"), which works better.

**6. A, B, C, D.** E is false: PyTorch **accumulates** gradients in `.grad` across `backward()`
calls; without `zero_grad()` every step uses the sum of all previous gradients (exercise 5 of D1-02).

**7. A, C, D.** Pets are never upside down in the test set (B false). TTA only changes the *inputs*
at inference and averages the outputs (E false). A rotated 6 is a 9 (F false): an augmentation
must preserve the label for your task.

**8. A, B, D, E.** Residual connections were introduced to make very deep networks *trainable*
(gradient flow, the degradation problem), not to save parameters (C false). D: cost ratio
1/C_out + 1/9. E: concatenation: 64 + 128 + 32 + 32 = 256.

**9. A, B, C, E.** D is reversed: early layers (edges, colours) are generic and transfer well; late
layers are task-specific and are the first to unfreeze. E is the empirical log-linear data
scaling seen in the D1-04 accuracy-vs-size plot.

**10. A, B, D, E.** Grad-CAM works on any trained convnet with one forward and one backward pass;
no re-training (C false). D is the "background only" shortcut check of D1-04.
