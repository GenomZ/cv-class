# Deep Learning for Computer Vision - 3-day syllabus (2026 refresh)

**Audience.** Students who already know how neural networks and backpropagation work (covered in a previous class). No prior computer-vision knowledge assumed.

**Format.** 3 days x 8 hours. 1h lunch, 15-minute breaks between blocks, most of each block is students working in notebooks. Four 75-90 minute blocks per day, each split roughly 25 min guided explanation with visual aids / 50 min hands-on / 15 min debrief.

**Framework.** PyTorch 2.x + torchvision (decided 2026-11-04; TensorFlow 2 was the legacy choice of the source repos). Everything in the notebooks that is *conceptual* (visualisations, NumPy reference implementations of IoU, NMS, anchors, convolutions, focal loss, Dice) is framework-independent and already written. Model-training cells are written in PyTorch with deliberate gaps for students to fill.

**Thread.** One dataset follows the students through all three days: **Oxford-IIIT Pet** (37 breeds, 7 349 images, downloadable directly, no account). It ships with class labels (Day 1 classification), head bounding boxes (Day 2 detection) and trimap masks (Day 3 segmentation). Synthetic "shapes" data generated inside the notebooks is used wherever a mechanism must be visible at a glance and train in under two minutes on CPU (anchors, matching, NMS, mini-detector, mini-U-Net). Penn-Fudan pedestrians (170 images) is used for Faster R-CNN / Mask R-CNN fine-tuning.

---

## Daily timetable

| Time | Slot |
|---|---|
| 09:00 - 09:15 | Warm-up: recap of yesterday, one puzzle on the board |
| 09:15 - 10:45 | **Block 1** |
| 10:45 - 11:00 | Break |
| 11:00 - 12:30 | **Block 2** |
| 12:30 - 13:30 | Lunch |
| 13:30 - 15:00 | **Block 3** |
| 15:00 - 15:15 | Break |
| 15:15 - 16:30 | **Block 4** |
| 16:30 - 17:00 | Quiz (10 MC questions), open Q&A, pointer to tomorrow |

---

## Day 1 - ConvNets: from pixels to transfer learning

Goal of the day: understand what a convolution computes, why stacks of them see the whole image, which architectural ideas survived from 2012 to today, and how to get a strong classifier on a small dataset by reusing a pretrained backbone.

### D1-B1 The convolution operation (notebook `D1-01-convolution.ipynb`)

* Images as tensors (H, W, C), channels-first vs channels-last, normalisation.
* Cross-correlation vs convolution, kernel, stride, padding (valid / same), dilation; output-size formula.
* Hand-made kernels (edge, blur, sharpen) applied to a real image: the "a kernel is a pattern detector" intuition.
* Multi-channel convolution: one kernel = one output channel; parameter count; FLOPs.
* Receptive field: how it grows with depth, stride and dilation; two 3x3 = one 5x5 with fewer parameters.
* Pooling (max / average), global average pooling; 1x1 convolutions as per-pixel linear layers.
* Hands-on: implement `conv2d` in NumPy (loops, then `im2col`), verify against the framework; compute output shapes and receptive fields for a given net; train the FashionMNIST tiny convnet and look at its learned first-layer kernels.

Visual aids: animated sliding kernel with the running dot product; output-size calculator; receptive-field growth diagram; learned-filter mosaic; feature-map grid per layer.

### D1-B2 Training convnets well (notebook `D1-02-training-recipe.ipynb`)

* The small-data problem: train from scratch on 2 000 Pet images, watch it overfit (train/val curves).
* Data pipeline: Dataset -> transforms -> DataLoader, shuffling, workers, prefetching.
* Augmentation: geometric and photometric; RandAugment / TrivialAugment; Mixup / CutMix; random erasing; test-time augmentation; what *not* to augment (vertical flip on digits).
* Regularisation: dropout, weight decay (AdamW), batch normalisation and its train / eval modes, label smoothing, early stopping.
* Optimisers and schedules: SGD + momentum vs Adam / AdamW; linear warm-up + cosine, OneCycle; LR range test; mixed precision; checkpoint / resume.
* Debugging recipe: overfit a single batch; loss at initialisation should be log(C); visualise augmented batches; gradient norms; NaN -> lower LR.
* Hyperparameter search: which knobs matter (LR, weight decay, augmentation strength, schedule length) and a 10-line random search.
* Hands-on: build the pipeline, visualise augmentations, train with and without augmentation + schedule, compare curves, checkpoint and reload.

Visual aids: augmentation gallery; LR-schedule shapes; overfitting curves side by side; BN train/eval behaviour plot.

### D1-B3 Architecture zoo (notebook `D1-03-architectures.ipynb`)

* Timeline with the one idea each architecture contributed: LeNet (conv + pool), AlexNet (ReLU, dropout, GPUs), VGG (3x3 everywhere), GoogLeNet / Inception (multi-branch, 1x1 bottlenecks, auxiliary heads, GAP), ResNet (residual connections, BN), DenseNet (concatenation), MobileNet (depthwise-separable), EfficientNet (compound scaling), ConvNeXt (a ResNet modernised with ViT training tricks).
* Residual vs dense connections, side by side; why residuals fix optimisation of deep nets (the source repo's "residual block" was actually a dense block - we fix that).
* Parameter / FLOP / latency trade-offs; reading a model summary.
* Hands-on: implement an Inception block, a residual bottleneck block, a depthwise-separable block; assemble a mini-ResNet; count parameters; run the `tests.py` checkers; train the mini-ResNet on Pet for a few epochs.

Visual aids: architecture timeline; block diagrams drawn to scale; gradient-flow plot plain vs residual; parameter-count bar chart.

### D1-B4 Transfer learning and what the network sees (notebook `D1-04-transfer-learning.ipynb`)

* Pretrained backbones (torchvision / timm): ResNet-50, ConvNeXt-T, EfficientNet, ViT, DINOv2 features.
* Three regimes: linear probe on frozen features, frozen backbone + trainable head with augmentation, fine-tuning (which layers, discriminative LRs, small LR, train head first).
* Preprocessing must match the backbone (mean / std, resolution).
* Interpretability: first-layer filters, feature maps, Grad-CAM; sanity checks for shortcut learning.
* Practical gotchas (ML Yearning): dev / test from the same distribution, single-number metric, when to change the metric.
* Hands-on: linear probe -> fine-tune on Pet; reach > 90% on 37 breeds in minutes; Grad-CAM on successes and failures.

Visual aids: "which layers to unfreeze" diagram; accuracy vs. training-set-size (log) curve; Grad-CAM overlays; feature-map evolution through the network.

**Day 1 quiz** refreshes the source MC items (output size, receptive field, 1x1 conv, regularisers, NaN loss, valid augmentations, TTA, which layers to fine-tune, data-scaling law).

---

## Day 2 - Object detection

Goal of the day: go from "classify a crop" to a full detector, understand the components every modern detector is built from (boxes, IoU, matching, anchors or their absence, NMS, multi-scale features, focal loss), evaluate with mAP, and fine-tune a library detector.

### D2-B1 The detection problem, boxes, IoU, mAP, and the region-based lineage (notebook `D2-01-boxes-iou-map.ipynb`)

* Task definition; box formats (xyxy, xywh, cxcywh), normalised vs pixel coordinates, conversions.
* IoU (vectorised, NumPy), GIoU / DIoU / CIoU and why they are better *losses* than L1.
* Evaluation: TP / FP / FN at an IoU threshold, precision-recall curve, AP, mAP@0.5, COCO mAP@[.5:.95], small / medium / large.
* History as a chain of fixes: sliding window -> Selective Search (R-CNN) -> shared features + RoI pooling (Fast R-CNN) -> learned proposals (Faster R-CNN) -> one stage (SSD / YOLO / RetinaNet) -> anchor-free (FCOS, CenterNet) -> set prediction (DETR).
* Non-maximum suppression: greedy NMS, Soft-NMS, class-wise NMS; hard-negative mining.
* The fully-convolutional trick: a classifier applied to a larger image *is* a sliding window.
* Hands-on: implement IoU, NMS and AP from scratch with `tests.py` checks; run the pretrained Pet classifier fully-convolutionally on a large image and see a heat map; compute mAP for a toy prediction set.

Visual aids: box-format diagram; IoU interactive sliders; PR curve construction animation; NMS step-by-step; sliding window vs. fully-convolutional heat map.

### D2-B2 Faster R-CNN internals: anchors, matching, RPN, RoIAlign (notebook `D2-02-two-stage.ipynb`)

* Anchors: grid of centres on a feature map, scales x ratios, stride; how many anchors an image gets.
* Matching anchors to ground truth (IoU >= 0.7 positive, < 0.3 negative, ignore in between), class imbalance.
* Box encoding / decoding (tx, ty, tw, th), why log-space for size; smooth-L1.
* RPN head: 3x3 conv, 1x1 objectness (k) and regression (4k) heads; proposals -> NMS -> top-N.
* RoI pooling vs RoIAlign (quantisation, bilinear sampling); the second stage (class + box refinement).
* Hands-on: generate and draw anchors; run matching on the synthetic shapes set and colour anchors by label; implement encode / decode and verify round-trip; implement RoIAlign with bilinear sampling in NumPy; fine-tune a library Faster R-CNN on Penn-Fudan (or Pet heads) and evaluate mAP.

Visual aids: anchor grid overlay with scale / ratio families; matching heat map; encode / decode diagram; RoI pooling vs RoIAlign grid drawing.

### D2-B3 One-stage detectors: SSD, RetinaNet, FPN, focal loss, anchor-free (notebook `D2-03-one-stage.ipynb`)

* Dense prediction on a grid: output tensor `[H, W, A, (4 + 1 + C)]`; one-cell-per-object target encoding (YOLOv1 / the source repo's SSD).
* Multi-scale default boxes (SSD) -> Feature Pyramid Network (lateral 1x1, top-down upsample + add, 3x3 smoothing, P3-P7).
* Shared heads across pyramid levels; number of output channels.
* Focal loss derived: the modulating factor, alpha and gamma, what it does to easy negatives; comparison to hard-negative mining.
* Anchor-free heads (FCOS centre-ness, CenterNet heat maps) and the modern YOLO head (decoupled, distribution focal loss).
* Hands-on: build an FPN on a ResNet backbone and inspect the pyramid shapes; implement focal loss and plot it; train the mini YOLOv1-style shapes detector end to end with a custom training loop (confidence loss + box loss); decode and NMS its output.

Visual aids: FPN diagram with tensor shapes; focal-loss curves for several gamma; dense-grid target encoding drawing; training-loss curves.

### D2-B4 Modern detectors in practice: YOLO, DETR family, fine-tuning and deployment (notebook `D2-04-modern-detectors.ipynb`)

* YOLO lineage in one page (v1 grid -> anchors -> anchor-free, decoupled heads, NMS-free training in v10, what "YOLO11/12" changed).
* DETR: set prediction, bipartite (Hungarian) matching, no anchors and no NMS; RT-DETR as the real-time version; Deformable attention mentioned only.
* Open-vocabulary detection (Grounding DINO, OWL-ViT): text prompts instead of class lists.
* Fine-tune a modern detector on a small custom dataset (Pet heads or shapes); mAP comparison against the morning's Faster R-CNN.
* Deployment notes: input resolution vs latency, batched inference, ONNX / TensorRT export, quantisation.
* Hands-on: fine-tune, evaluate, visualise failures, export.

Visual aids: two-stage vs one-stage vs DETR block diagram; speed-accuracy scatter plot of current models; Hungarian matching cartoon.

**Day 2 quiz**: box formats, IoU corner cases, AP computation, anchor matching rule, why focal loss, NMS threshold effect, FPN level choice, two-stage vs one-stage trade-offs.

---

## Day 3 - Segmentation and the frontier

Goal of the day: predict at the pixel level with encoder-decoder networks, understand U-Net deeply (it is also the backbone of diffusion models), extend detection to instance masks, and know what the 2024-2026 foundation models change.

### D3-B1 The segmentation problem, upsampling, losses (notebook `D3-01-segmentation-basics.ipynb`)

* Semantic vs instance vs panoptic segmentation; label formats (masks, trimaps, polygons, RLE).
* Metrics: pixel accuracy, per-class IoU, mIoU, Dice; why accuracy lies on imbalanced masks.
* Losses: BCE / cross-entropy, Dice, focal, combined BCE+Dice; class weights.
* Getting resolution back: transposed convolution (and its checkerboard artefact), bilinear upsample + conv, pixel shuffle; dilated (atrous) convolutions and ASPP.
* Autoencoders as the ancestor; the bottleneck loses "where"; skip connections put it back.
* Hands-on: implement transposed conv in NumPy and see the checkerboard; implement Dice and mIoU with `tests.py`; build a Pet trimap target from the raw annotations.

Visual aids: three segmentation types on one image; Dice vs IoU geometry; transposed-conv animation; dilation receptive-field diagram.

### D3-B2 U-Net (notebook `D3-02-unet.ipynb`)

* Encoder block (conv-BN-ReLU x2, pool, return skip), decoder block (upsample, concatenate skip, conv x2), 1x1 output head.
* Channel schedule, number of levels, output activation and loss for binary / multi-class.
* Variants: residual encoders, pretrained encoders (segmentation_models_pytorch), U-Net++ / Attention U-Net, DeepLabv3+ (ASPP), SegFormer and Mask2Former as the transformer answers; U-Net inside diffusion models.
* Hands-on: implement the blocks with `tests.py` checks; assemble U-Net; train on synthetic shapes (2 minutes) then on Pet trimaps with a pretrained ResNet encoder; EarlyStopping / checkpointing; threshold and visualise; compute mIoU.

Visual aids: U-Net drawn with tensor shapes; skip-connection ablation (with vs without); prediction grid image / mask / prediction / error.

### D3-B3 Instance segmentation: Mask R-CNN and friends (notebook `D3-03-instance-segmentation.ipynb`)

* Mask R-CNN = Faster R-CNN + RoIAlign + a small FCN mask head per RoI; losses; mask resolution.
* Instance mask metrics (mask mAP); panoptic quality.
* Fine-tune a library Mask R-CNN on Penn-Fudan; inspect masks; compare with the Day 2 boxes.
* Mask2Former / query-based segmentation in one slide; YOLO-seg heads.
* Hands-on: fine-tune, evaluate mask mAP, visualise.

Visual aids: mask-head diagram; RoIAlign revisited; qualitative result grid.

### D3-B4 Beyond convnets: transformers and foundation models (notebook `D3-04-foundation-models.ipynb`)

* Vision Transformer: patch embedding is a 16x16 stride-16 convolution; attention vs convolution inductive bias; Swin and hierarchical ViTs; ConvNeXt vs ViT today.
* Self-supervised backbones: DINOv2 / v3 features, k-NN and linear probe without labels; PCA of patch features as a segmentation.
* CLIP: zero-shot classification with text prompts; SigLIP.
* Promptable segmentation: SAM / SAM 2 (video) / SAM 3 (concept prompts); open-vocabulary detection (Grounding DINO) + SAM = "segment anything you can name".
* Where to go next: 3D (NeRF / Gaussian splatting), video, multimodal LLMs, dataset curation and labelling with foundation models.
* Hands-on: zero-shot Pet breed classification with CLIP vs the Day 1 fine-tuned ResNet; DINOv2 patch-feature PCA on Pet images; SAM with point / box prompts on the same images.

Visual aids: ViT diagram with the "patchify = conv" identity; attention map overlays; DINOv2 PCA colour maps; SAM prompt-to-mask examples; a one-page map of the field in 2026.

**Day 3 quiz** and a 30-minute capstone discussion: students pick a task (classification / detection / segmentation) and a dataset, and sketch the pipeline (backbone, head, loss, metric, augmentation, schedule) on paper.

---

## Pre-course material (sent one week before)

* Setup: Python 3.11+, the course environment, GPU check (Colab fallback), dataset download script (Pet, Penn-Fudan, pretrained checkpoints, about 2 GB).
* Pre-read: cs231n Module 2 "Convolutional Networks" (replaces the 2-hour in-class reading of the old course).
* Optional: the 20-minute refresher notebook `D0-00-setup-and-tensors.ipynb`.

## Time budget check

| Day | Guided | Hands-on | Debrief / quiz | Total |
|---|---|---|---|---|
| 1 | 4 x 25 min | 4 x 50 min | 4 x 15 min + 30 min | ~7h with breaks |
| 2 | same | same | same | ~7h |
| 3 | same | same (B4 is demo-heavy) | same + capstone | ~7h |

Heavy training (full Pet fine-tune, Faster / Mask R-CNN on Penn-Fudan) is capped at ~10 minutes on a modern GPU; pre-trained checkpoints are provided so CPU-only students can run the inference and analysis cells.
