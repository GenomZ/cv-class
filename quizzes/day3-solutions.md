# Day 3 quiz - solutions

| Q | correct | topic (notebook) |
|---|---|---|
| 1 | A, B, C | pixel accuracy, IoU, mIoU (D3-01) |
| 2 | A, B, C, D | Dice vs IoU (D3-01) |
| 3 | A, B, C, E | transposed convolution, checkerboard, resize + conv (D3-01) |
| 4 | A, B, C, E | dilation, receptive field, ASPP (D3-01) |
| 5 | A, B, C | U-Net skips, divisibility, ablation, diffusion (D3-02) |
| 6 | A, C, D, E | multi-label outputs, BCE vs Dice, class weights (D3-01, D3-02) |
| 7 | A, C, D | Mask R-CNN mask head, RoIAlign, fine-tuning (D3-03) |
| 8 | A, B, C, D | mask AP, panoptic quality (D3-03) |
| 9 | A, B, C, E | ViT, patch embedding, attention cost, Swin (D3-04) |
| 10 | A, B, E, F | DINOv2, CLIP, SAM, Grounding DINO (D3-04) |

---

**1. A, B, C.** 95% of the pixels are background and predicted as background (A). The object
has TP = 0, so IoU = 0 (B). Background IoU = 0.95 N / (0.95 N + 0.05 N) = 0.95 (the 5% object
pixels are false positives for background), so mIoU = (0.95 + 0) / 2 = 0.475 (C). Dice of the
object is 2 * 0 / (0 + 0.05 N) = 0 (D false). A class absent from both prediction and target
has 0 / 0 = undefined IoU; it is skipped (`np.nanmean`), not counted as 0 (E false) - counting
it as 0 would punish a model for correctly predicting nothing.

**2. A, B, C, D.** Dice = 2 IoU / (1 + IoU) = 1.2 / 1.6 = 0.75 (A). Since 1 + IoU <= 2,
Dice >= IoU, with equality at 0 and 1 (B, C). The function is monotonically increasing, so it
preserves the order on a single image (D); averaged over a dataset the rankings can differ
slightly. The soft Dice loss is computed on probabilities - that is the whole point of "soft"
(E false).

**3. A, B, C, E.** Output size (n - 1) s - 2p + k = 15 * 2 - 2 + 4 = 32 (A). With k = 3,
s = 2 the kernel is not divisible by the stride: odd and even output positions are covered a
different number of times (B; the D3-01 overlap-count figure). The scatter-add equals a
convolution of the zero-stuffed (and padded) input with the flipped kernel (C). It is *not* an
inverse: it only restores the shape, which is why "deconvolution" is a misnomer (D false).
Resize + conv has learnable weights in the conv and uniform coverage (E).

**4. A, B, C, E.** Effective kernel size d (k - 1) + 1 = 5, still 9 weights (A). Receptive
field 1 + 2 * (1 + 2 + 4) = 15 (B). With equal rates the taps never land on odd offsets: holes
(C; the autograd receptive-field figure). Dilation keeps the resolution - that is its purpose
in DeepLab (D false). ASPP = parallel branches with rates 1 (1x1), 6, 12, 18 + image pooling,
concatenated (E).

**5. A, B, C.** A is the definition. 100 -> 50 -> 25 -> 12 (floor) -> 6; going up 6 -> 12 ->
24, which does not match the 25 x 25 skip (B). The ablation in D3-02 shows exactly C. A 2x2
kernel with stride 2 has no overlap at all, hence no checkerboard (D false). The diffusion
U-Net predicts the *noise* (same shape as the image), conditioned on the timestep (E false).

**6. A, C, D, E.** Overlapping classes are independent yes/no decisions per pixel: sigmoid +
BCE per channel (A). Softmax forces the classes to compete and sum to 1 (B false). With mean
BCE every pixel has the same voice, so 99% background pixels dominate (C; the gradient-share
figure). Dice divides by |P| + |T|, which rescales a small object's gradient (D). Class weights
in the cross-entropy are the classic fix for rare classes (E; median-frequency balancing in
D3-01).

**7. A, C, D.** A describes the mask branch (4 convs, a 2x2 deconv, a 1x1 conv to C channels).
The loss is a per-pixel **sigmoid** BCE on the channel of the ground-truth class only; class
competition is left to the box branch (B false; this decoupling was worth ~5 AP in the paper).
RoIPool rounds the RoI to the stride-16 grid: the D3-03 round-trip experiment loses ~0.1 mask
IoU (C). D is the inference procedure (`paste_mask`). Both predictors have per-class outputs:
the box predictor (4 classes incl. background, 16 box deltas) and the mask predictor
(4 x 28 x 28) must be replaced (E false).

**8. A, B, C, D.** A is how COCO `segm` AP works (and the `mask_ap` exercise). A mask must be
right at every pixel, a box only at four edges: mask AP is lower (B). PQ = SQ x RQ, so
RQ = 0.6 / 0.8 = 0.75 (C). RQ = TP / (TP + FP/2 + FN/2) is the F1 score (D). Two people side by
side can have heavily overlapping boxes and almost disjoint masks (E false; the D3-03 example
has box IoU 0.5 and mask IoU 0.0).

**9. A, B, C, E.** 224 / 16 = 14, 14^2 = 196 patches + [class] (A). Patchify + linear is
exactly a 16x16 stride-16 convolution - the exercise proved it on the real `vit_b_16` weights
(B). 384 / 16 = 24, 24^2 = 576; the position embeddings learned for 14 x 14 are interpolated
to 24 x 24 (C). Attention compares every token with every token: quadratic cost (D false) -
the reason Swin uses windows (E).

**10. A, B, E, F.** DINO / DINOv2 are self-supervised; in D3-04 a k-NN on DINOv2 features
beat a linear probe on ImageNet-supervised ResNet-18 features (A). B is the zero-shot recipe
(`zero_shot_classify`). CLIP needs no Pet images at all (C false; ~87-93% zero-shot on Pets in
the CLIP paper). SAM is class-agnostic: masks and predicted IoUs, no labels (D false). E is the
open-vocabulary pipeline. F: one click is ambiguous (shirt or person?), so SAM outputs three
masks with IoU predictions and you take the best or let the user choose.
