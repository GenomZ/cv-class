# Instructor run-of-show

How to drive each block. The notebooks carry the content; this page is the stage
direction: what to show live, where students usually get stuck, and what to cut if
the clock runs away. Times assume the 09:00-17:00 timetable in `02-syllabus-3-day.md`.

## Rhythm of a block (75-90 min)

1. **Hook (5 min).** One image, one question. "Which of these two kernels found the edges?" "Why did the detector draw three boxes on one cat?"
2. **Guided walk (20 min).** Run the figure cells of the notebook on the projector, top to bottom, stopping at each figure. Do not type code live; the figures are the slides. At each `labs.show(...)` cell, read the **Predict first** question aloud, take a show of hands, then settle it on the projector by dragging (30-60 s per lab). Do not work through the lab's "Try this" list yourself.
3. **Hands-on (45-50 min).** Students work top to bottom: for each lab the first two "Try this" items, then the exercise that follows it. Circulate. The `[PASS]/[FAIL]` checkers mean you only get called for real confusion.
4. **Debrief (10-15 min).** Show the solution for the hardest exercise, run the training cell, discuss the numbers. End with the block's one-sentence takeaway.

Cut list: every notebook's last section before "Summary" is optional and marked as such in the spec (hyperparameter search in D1-02, efficiency table in D1-03, ML Yearning case in D1-04, Selective Search demo in D2-01, second-stage loss walk-through in D2-02, anchor-free heads in D2-03, deployment in D2-04, autoencoder in D3-01, variants gallery in D3-02, alternatives in D3-03, capstone in D3-04). Cut from the end, never the exercises.

## Interactive labs: time budget

The labs were added after the blocks were timed, and done in full they do not fit: the last
column is what every "Try this" item would cost. The plan above (prediction on the projector,
two items per lab during hands-on) costs roughly a third of that. The **core** lab of each
block is the one to protect; the others can be left as homework, since they run offline in
any browser from `notebooks/cvcourse/labs/index.html`.

| Block | Labs (core in bold) | The moment to show on the projector | Full "Try this" |
|---|---|---|---|
| D1-01 | **conv-playground**, receptive-field | Type a kernel that shifts the image; stride 2 shrinks the output. | 12 min |
| D1-02 | **optimizer-playground**, lr-schedule, mixup-cutmix | Push SGD's learning rate past 1 on the ravine: it diverges while AdamW rattles. | 15 min |
| D1-03 | **gradient-flow**, block-cost | At depth 40 the plain net's first-layer gradient is about 1e-16; the residual net's is about 20. | 10 min |
| D1-04 | **freeze-unfreeze**, gradcam-by-hand | Freeze up to layer2: 94% of ResNet-50 still trains. | 8 min |
| D2-01 | **iou-playground**, pr-curve, nms-playground | Drag the boxes apart: the IoU-loss arrows vanish, the GIoU arrows do not. | 22 min |
| D2-02 | **anchor-matching**, roi-align, box-encoding | 2304 anchors, 6 positives; untick low-quality matches and the small box has none. | 24 min |
| D2-03 | **focal-loss**, grid-assignment, fpn-levels | 100 000 easy negatives carry 97.7% of the loss under cross-entropy, 2.5% under focal loss. | 17 min |
| D2-04 | **hungarian-matching** | Tick "compare with greedy": greedy is worse by 1.0. | 6 min |
| D3-01 | **seg-metrics**, transposed-conv | Small object, press "clear": 97% pixel accuracy, IoU 0. | 10 min |
| D3-02 | **unet-shapes**, sliding-window | Input size 100: the concat that fails turns red (24 vs 25). | 10 min |
| D3-03 | **mask-paste** | 32 px box vs 340 px box: mask IoU barely moves, boundary IoU drops from 0.96 to 0.58. | 5 min |
| D3-04 | **patch-attention**, clip-zero-shot | Shuffle the patches with position strength 0: the attention weights do not change. | 11 min |

D2-01 and D2-02 are the tight blocks. If either runs late, keep the core lab and one more,
and apply that block's "If running late" cut below.

For projecting, `labs.open("name")` opens a lab full-window in the browser; a state can be
bookmarked through the URL hash (for example `iou-playground.html#preset=disjoint&loss=iou`).

## Day 1

| Block | Live demo moment | Typical stuck point | If running late |
|---|---|---|---|
| D1-01 | The sliding-kernel animation; then the Sobel gallery on the photo. | Off-by-one in the output size; forgetting to sum over input channels in `conv2d`. | Skip `im2col` timing cell. |
| D1-02 | Show an augmented batch *before* training; then the overfit-one-batch curve hitting zero. | `model.eval()` forgotten before validation (BN / dropout); scheduler stepped per epoch instead of per batch. | Skip random search. |
| D1-03 | The plain-vs-residual gradient-norm plot: 40 layers, gradients vanish, residual does not. | Shortcut shape mismatch when stride 2 or channel change (needs projection). | Skip depthwise / SE; keep ConvNeXt as one slide. |
| D1-04 | Grad-CAM on a wrong prediction: it looks at the sofa. | Preprocessing not matching the pretrained weights (mean / std, resolution); fine-tuning with the head's LR. | Run the fine-tune from the checkpoint only. |

Day 1 takeaway: *a convnet is a learned Sobel gallery stacked 50 deep; reuse one somebody already trained.*

## Day 2

| Block | Live demo moment | Typical stuck point | If running late |
|---|---|---|---|
| D2-01 | The IoU sweep (box sliding across another) and the PR curve being built point by point. | Coordinates in `(x, y)` vs `(row, col)`; AP needs precision made monotone first. | Skip Selective Search and GIoU/DIoU/CIoU landscape. |
| D2-02 | Anchors coloured by match on a shapes image; the max-IoU histogram (99% negatives). | `encode` / `decode` not inverse (log vs exp, centre vs corner); RoIAlign sampling-point coordinates. | Faster R-CNN from checkpoint; skip the tiny RPN training. |
| D2-03 | The mini detector going from noise to boxes in 90 s; focal-loss curves for gamma 0 / 2 / 5. | Loss masked to positive cells only for box and class terms; decoding `cxcywh` back to pixels. | Skip FCOS / CenterNet section; library inference only. |
| D2-04 | Hungarian matching on a 3x2 cost matrix by hand, then with scipy. | None technical; the discussion is the point. | Skip ONNX export. |

Day 2 takeaway: *every detector is "dense proposals + matching + a classification loss that survives 99% negatives + NMS (or a set loss instead)".*

## Day 3

| Block | Live demo moment | Typical stuck point | If running late |
|---|---|---|---|
| D3-01 | Checkerboard artefact from a stride-2 kernel-3 transposed conv vs kernel 4; pixel accuracy 97% with an all-background prediction. | Dice on probabilities vs on thresholded masks; transposed-conv output size. | Skip the autoencoder. |
| D3-02 | Skip-connection ablation side by side. | Concatenation order and channel counts in the decoder; forgetting to crop / pad when sizes are odd. | Pet trimaps from checkpoint; skip variants. |
| D3-03 | Instances coloured on a Penn-Fudan image; a 28x28 mask pasted into its box. | Target dict format for torchvision detection models; masks as `uint8 (N, H, W)`. | Run from checkpoint. |
| D3-04 | Prove patchify == strided conv (max abs diff 0.0); DINOv2 PCA colours segmenting a cat with zero labels. | Normalisation before the dot product in zero-shot classification. | Skip SAM / Grounding DINO cells if offline; keep the map-of-the-field figure and the capstone. |

Day 3 takeaway: *segmentation is classification per pixel with a decoder that gets "where" back from skip connections; foundation models give you features, boxes and masks before you have labelled anything.*

## Hardware fallback

Every training cell has a `max_minutes` cap and loads `data/checkpoints/<name>.pt` when present. Ship the checkpoints with the course material (about 1 GB). CPU-only students run inference and all figure cells; the from-scratch shapes trainings (D2-03 detector, D3-02 U-Net) run on CPU in about two minutes and are the moments where everybody trains something real.
