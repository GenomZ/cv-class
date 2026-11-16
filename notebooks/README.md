# Notebooks

Work through them in order; each one is a 75-90 minute block.

| | Notebook | You build |
|---|---|---|
| Before the course | `D0-00-setup-and-tensors` | installation check, tensors, a ten-line training loop |
| Day 1 | `D1-01-convolution` | convolution by hand, output sizes, receptive fields, a tiny convnet |
| | `D1-02-training-recipe` | augmentation, Mixup, warmup + cosine schedule, the training step |
| | `D1-03-architectures` | Inception, residual and bottleneck blocks, a MiniResNet |
| | `D1-04-transfer-learning` | feature extraction, fine-tuning with per-layer learning rates, Grad-CAM |
| Day 2 | `D2-01-boxes-iou-map` | IoU, GIoU, precision-recall, average precision, NMS |
| | `D2-02-two-stage` | anchors, matching, box encoding, an RPN head, RoIAlign |
| | `D2-03-one-stage` | grid targets, the dense loss, FPN, focal loss |
| | `D2-04-modern-detectors` | Hungarian matching, the DETR cost, a comparison of detectors |
| Day 3 | `D3-01-segmentation-basics` | Dice, mean IoU, transposed convolution, pixel shuffle |
| | `D3-02-unet` | a U-Net from blocks, BCE + Dice loss, sliding-window inference |
| | `D3-03-instance-segmentation` | mask paste-back, mask IoU and AP, Mask R-CNN heads |
| | `D3-04-foundation-models` | patchify, attention, CLIP zero-shot classification |

* Run the first cell of a notebook first, then go cell by cell.
* **Exercise** cells contain `raise NotImplementedError`: replace it with your code. The next
  cell prints `[PASS]` or `[FAIL]` with a hint.
* **Functions used here (click to open)** lines explain every library function right before
  its first use; `from cvcourse import apiref; apiref.show("permute")` shows any of them again.
* `labs.show("...")` cells open an interactive lab; all of them are also in
  `cvcourse/labs/index.html`.
* `solutions/` has every notebook with the exercises filled in.
* `cvcourse/` is the helper package the notebooks import; `assets/` holds three fallback images.
