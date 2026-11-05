# Day 2 quiz - Object detection

10 questions, 15 minutes. **Several answers may be correct**: tick every correct option.
Solutions with explanations: `day2-solutions.md`.

---

**1. Box formats.** A box is given in xyxy pixel coordinates as `(10, 20, 50, 80)` on an image
that is 100 px wide and 200 px high. Which statements are true?

- [ ] A. In xywh it is `(10, 20, 40, 60)`.
- [ ] B. In cxcywh it is `(30, 50, 40, 60)`.
- [ ] C. Its area is 2 400 px^2.
- [ ] D. Normalised by the image size (xyxy) it is `(0.1, 0.2, 0.5, 0.8)`.
- [ ] E. After a horizontal flip of the image the box becomes `(50, 20, 90, 80)`.

---

**2. IoU.** Which statements about Intersection over Union are true?

- [ ] A. Two 40 x 40 boxes, one shifted 10 px right and 10 px down from the other, have IoU of about 0.39.
- [ ] B. Two boxes that only share an edge have IoU 0.
- [ ] C. If box B lies entirely inside box A, IoU = area(B) / area(A).
- [ ] D. Scaling both boxes (and the image) by a factor 2 halves their IoU.
- [ ] E. A 10 px shift costs the same IoU for 40 x 40 boxes as for 400 x 400 boxes.

---

**3. Box losses.** A predicted box does **not** overlap its ground-truth box at all.

- [ ] A. The loss 1 - IoU is constant here, so its gradient with respect to the box is zero.
- [ ] B. GIoU is negative here and still gives a gradient that moves the prediction.
- [ ] C. DIoU adds a penalty proportional to the squared distance between the box centres (normalised by the enclosing box diagonal).
- [ ] D. An L1 loss on the corner coordinates is scale invariant, which is why YOLO prefers it to IoU losses.
- [ ] E. CIoU adds an aspect-ratio consistency term on top of DIoU.

---

**4. Average precision.** An image has 3 ground-truth objects of one class. The detector
returns 5 predictions; sorted by score they are: TP, FP, TP, TP, FP (at IoU >= 0.5).

- [ ] A. The all-point interpolated AP is 0.833.
- [ ] B. The final recall is 1.0.
- [ ] C. After the third prediction, precision is 2/3.
- [ ] D. Deleting the last prediction (a FP with the lowest score) increases the AP.
- [ ] E. A second detection of an already-matched object counts as a true positive as long as its IoU is >= 0.5.

---

**5. Anchors and matching.** A 512 x 512 image, one feature level with stride 16, 3 scales x
3 aspect ratios per cell, Faster R-CNN's RPN rule (positive >= 0.7, negative < 0.3).

- [ ] A. The level has 9 216 anchors.
- [ ] B. An anchor whose best IoU with any GT is 0.5 is ignored in the loss (unless it is the best anchor of some GT).
- [ ] C. A negative anchor has IoU < 0.3 with *every* GT box.
- [ ] D. In a typical image most anchors are positive, so the RPN samples extra negatives.
- [ ] E. The "low-quality match" rule guarantees every GT at least one positive anchor.

---

**6. Two-stage detectors.** Which statements about Faster R-CNN are true?

- [ ] A. The RPN's objectness is class-agnostic ("object vs background").
- [ ] B. RoI pooling rounds the box to the feature grid; RoIAlign samples with bilinear interpolation at exact positions.
- [ ] C. Box widths are encoded as log(g_w / a_w), so that doubling and halving give targets of equal magnitude.
- [ ] D. The second stage predicts its boxes from scratch, ignoring the proposal coordinates.
- [ ] E. The training loss has four terms: RPN objectness, RPN box regression, RoI classification, RoI box regression.

---

**7. Non-maximum suppression.**

- [ ] A. Greedy NMS with an IoU threshold of 0.3 removes more boxes than with 0.7.
- [ ] B. With a threshold of 0.9 many duplicates survive, which lowers precision.
- [ ] C. In crowded scenes NMS can delete correct detections of overlapping objects; Soft-NMS decays their scores instead.
- [ ] D. torchvision's detectors run NMS class-agnostically, so a person and the horse they ride suppress each other.
- [ ] E. NMS is a differentiable layer whose threshold is learned during training.

---

**8. Multi-scale features.**

- [ ] A. An FPN upsamples coarse maps top-down and adds them to 1 x 1 lateral projections of finer backbone maps.
- [ ] B. Small objects are best handled on the high-resolution levels (e.g. P3, stride 8).
- [ ] C. All FPN levels have the same number of channels, which lets one detection head be shared by every level.
- [ ] D. SSD predicted small objects from early, high-resolution layers that carry weak semantics, which FPN fixes.
- [ ] E. An FPN adds extra downsampling stages to the backbone, which is why it improves small-object detection.

---

**9. Focal loss.** FL(p_t) = -alpha_t (1 - p_t)^gamma log(p_t).

- [ ] A. With gamma = 0 and alpha = 0.5 it equals half the binary cross-entropy.
- [ ] B. With gamma = 2 an example with p_t = 0.9 is weighted 100 times less than with cross-entropy.
- [ ] C. It increases the loss of easy negatives so that the network pays more attention to the background.
- [ ] D. Initialising the classification bias so that every anchor predicts p = 0.01 keeps the many background anchors from dominating the first iterations.
- [ ] E. It lets one-stage detectors train on all anchors without sampling or hard-negative mining.

---

**10. Modern detectors.**

- [ ] A. DETR assigns ground-truth objects to predictions with Hungarian matching (one-to-one), so it does not need NMS.
- [ ] B. DETR queries that are not matched to any object are trained to predict "no object".
- [ ] C. DETR's matching cost combines the class probability, an L1 box distance and GIoU.
- [ ] D. DETR converges faster than Faster R-CNN, which is why it replaced it.
- [ ] E. FCOS is anchor-free: each location inside a box regresses (l, t, r, b) and a centre-ness score down-weights locations near the border.
- [ ] F. Doubling the input side length of a convolutional detector roughly quadruples its convolution compute.
