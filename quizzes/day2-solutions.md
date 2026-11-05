# Day 2 quiz - solutions

| Q | correct | topic (notebook) |
|---|---|---|
| 1 | A, B, C, E | box formats, normalisation, flips (D2-01) |
| 2 | A, B, C | IoU and its corner cases (D2-01) |
| 3 | A, B, C, E | IoU-family losses (D2-01) |
| 4 | A, B, C | precision / recall, AP (D2-01) |
| 5 | A, B, C, E | anchors, matching rule, imbalance (D2-02) |
| 6 | A, B, C, E | RPN, encoding, RoIAlign, four-term loss (D2-02) |
| 7 | A, B, C | NMS, Soft-NMS, class-wise NMS (D2-01) |
| 8 | A, B, C, D | SSD, FPN (D2-03) |
| 9 | A, B, D, E | focal loss, prior initialisation (D2-03) |
| 10 | A, B, C, E, F | DETR, FCOS, deployment (D2-03, D2-04) |

---

**1. A, B, C, E.** xywh keeps the corner and stores w = 50 - 10 = 40, h = 80 - 20 = 60 (A).
The centre is (30, 50) (B). Area = 40 * 60 = 2 400 (C). Normalising divides x by the **width**
(100) and y by the **height** (200): (0.1, 0.1, 0.5, 0.4) - D divided everything by 100, the
classic bug. A horizontal flip maps x to W - x and swaps the corners: (100 - 50, 20, 100 - 10, 80)
(E).

**2. A, B, C.** Intersection 30 * 30 = 900, union 1 600 + 1 600 - 900 = 2 300, IoU = 0.391 (A).
Touching boxes have zero intersection area (B). If B is inside A the intersection is B and the
union is A (C). IoU is a ratio of areas, so it is **scale invariant** (D false), and the same
pixel shift is much less harmful for a big box (E false; 400 x 400 boxes shifted by 10 px right and down have IoU
0.91) - the reason COCO reports AP for small / medium / large objects separately.

**3. A, B, C, E.** Without overlap IoU = 0 everywhere nearby: flat loss, zero gradient (A).
GIoU = IoU - (|C| - |union|) / |C| is negative for disjoint boxes and shrinks as the boxes approach
(B). DIoU subtracts rho^2 / c^2 (C); CIoU adds alpha * v for the aspect ratio (E). L1 on corners
is **not** scale invariant: a 5 px error costs the same on a tiny and on a huge box (D false);
modern YOLOs use CIoU (plus DFL) for exactly that reason.

**4. A, B, C.** Precision after each prediction: 1, 1/2, 2/3, 3/4, 3/5; recall: 1/3, 1/3, 2/3, 1, 1.
The envelope is 1 on [0, 1/3] and 0.75 on [1/3, 1]: AP = 1/3 * 1 + 2/3 * 0.75 = 0.833 (A).
All 3 objects are found (B); after 3 predictions 2 are TP (C). The last FP comes after recall
reached 1, so it does not change the area under the envelope (D false) - AP never punishes
low-scored false positives after the last TP. Duplicates are false positives (E false).

**5. A, B, C, E.** 512 / 16 = 32 cells per side, 32 * 32 * 9 = 9 216 (A). IoU between 0.3 and 0.7 is
the ignore zone (B), unless the low-quality rule makes it the best anchor of a GT (E). Negatives
must be below 0.3 for all GTs (C: the rule uses the *max* IoU). Almost all anchors are negative,
not positive (D false); the RPN samples 256 anchors with at most 50% positives.

**6. A, B, C, E.** The RPN answers "object or not" for each anchor (A); classes come in stage 2.
RoI pooling quantises twice (box and bins), RoIAlign never rounds (B). log(2) = +0.69 and
log(1/2) = -0.69 (C). The second stage regresses deltas **relative to the proposal**, encoded like
the RPN deltas relative to anchors (D false). The four loss keys are `loss_objectness`,
`loss_rpn_box_reg`, `loss_classifier`, `loss_box_reg` (E).

**7. A, B, C.** A lower threshold suppresses anything that overlaps a little (A); a very high one
keeps duplicates, which become false positives (B). Overlapping people are the classic NMS failure;
Soft-NMS multiplies scores by exp(-IoU^2 / sigma) instead of deleting (C). torchvision uses
`batched_nms`, i.e. NMS **per class** (D false). NMS is a fixed post-processing algorithm, not
learned (E false) - NMS-free detectors (DETR, YOLOv10) remove it by training a one-to-one assignment.

**8. A, B, C, D.** That is the FPN recipe (A). High-resolution levels have small strides and fine
detail, so small objects go to P3 and large ones to P5-P7 (B). Equal width (256) is what makes a
shared RetinaNet / FCOS head possible (C). SSD's high-resolution maps came from shallow layers (D).
FPN does not add downsampling; it adds a top-down pathway that brings semantics to the existing
high-resolution maps (E false; RetinaNet's extra P6 / P7 are for *large* objects).

**9. A, B, D, E.** gamma = 0 removes the modulating factor; alpha_t = 0.5 for both classes gives
0.5 * BCE (A). (1 - 0.9)^2 = 0.01 (B). Focal loss **decreases** the loss of easy examples, most of
which are background (C false). With a bias of -log(99) every anchor starts at p = 0.01, so the
initial loss on ~100 000 background anchors is small (D). That is the point of RetinaNet: dense
training on all anchors (E).

**10. A, B, C, E, F.** One-to-one matching means each object has exactly one query responsible
for it, so duplicates are penalised during training and NMS is unnecessary (A, B). The cost is
-p(class) + 5 * L1 + 2 * (-GIoU) (C). DETR converged **much slower** (~500 epochs on COCO) - fixed
by Deformable DETR, DINO and RT-DETR (D false). E describes FCOS. Convolution cost is
proportional to the number of pixels: 2x side = 4x pixels (F).
