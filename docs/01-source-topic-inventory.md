# Source topic inventory

Analysis of the three repositories that the new course replaces:

| Repo | Batch | Framework | Content |
|---|---|---|---|
| `GenomZ/DL-Image-B42` | DSR Batch 42 (uploaded May 2025, material from 2024) | TensorFlow 2.3 / Keras, KerasTuner | "Deep Learning for Computer Vision Fundamentals": 3 modules + MC quiz |
| `GenomZ/Advanced-CV` | DSR Feb 2024 | TensorFlow 2.1 / Keras, OpenCV contrib, `focal-loss` | "Advanced Computer Vision": 6 modules, exercise + solution notebooks with `tests.py` checkers |
| `GenomZ/Advanced_Computer_Vision` | DSR June 2024 | same | Near-identical copy of `Advanced-CV` (README dates differ, two notebooks differ trivially, `data/` placeholders missing) |

**Dedup result:** the two "Advanced" repos are one course. The Fundamentals repo is its prerequisite. Together they form one continuous track, which is what the new 3-day course covers.

---

## A. Fundamentals repo (`DL-Image-B42`)

Source: Chollet's *Deep Learning with Python* notebooks 5.1 to 5.4 (first edition), plus pointers to cs231n, d2l.ai, ML Yearning and the Chekmenev debugging repo.

### Module 1 - ConvNets from scratch

| # | Topic | Status in new course |
|---|---|---|
| A1 | What a convnet is: stack of Conv2D + MaxPool, feature maps shrink spatially and grow in channels, Flatten + Dense head | Keep, but teach the convolution *operation* properly first (kernels, stride, padding, output size, receptive field). The original delegated this to "read cs231n Module 2 for 1-2 hours". |
| A2 | MNIST baseline: convnet 99.3% vs dense 97.8% | Keep as a 5-minute demo (FashionMNIST is a less saturated choice). |
| A3 | Small-data regime: cats vs dogs, 2 000 training images | Keep the *concept*. Replace the Kaggle-login dataset with one that downloads directly (Oxford-IIIT Pet). |
| A4 | Input pipelines (`image_dataset_from_directory`, `tf.data` batch/shuffle/prefetch/map) | Keep concept; implementation pending the PyTorch decision (`Dataset` / `DataLoader` / `transforms.v2`). |
| A5 | Overfitting diagnosis from train/val curves | Keep. |
| A6 | Data augmentation (flip, rotation, zoom); augmentation inactive at inference | Keep and extend: RandAugment / TrivialAugment, Mixup, CutMix, random erasing; test-time augmentation (was only a quiz question). |
| A7 | Dropout before the classifier | Keep. |
| A8 | ModelCheckpoint, save-best-only | Keep as a "training loop hygiene" item. |

### Module 2 - Pretrained ConvNet

| # | Topic | Status |
|---|---|---|
| A9 | MiniVGGNet (CONV-RELU-BN x2, POOL, dropout); history of kernel sizes 11x11 -> 7x7 -> 3x3 | Keep as part of an architecture timeline. |
| A10 | VGG16 pretrained on ImageNet, `include_top=False` | Keep (VGG is still the clearest teaching backbone), but ResNet is the default backbone from Day 1 onward. |
| A11 | Feature extraction, fast variant (precompute features, no augmentation) | Keep; this is a "linear probe". |
| A12 | Feature extraction with a frozen base end to end (allows augmentation) | Keep. |
| A13 | Fine-tuning: unfreeze top block, tiny LR, train head first | Keep; add discriminative (layer-wise) learning rates and "full vs partial fine-tune" guidance. |
| A14 | Why reuse the conv base and not the dense head; early layers generic, late layers specific | Keep, with filter and feature-map visualisations. |
| A15 | Model zoo mention (Xception, Inception, ResNet, MobileNet, EfficientNet, DenseNet) | Replace with a modern zoo walkthrough (torchvision / timm / Hugging Face): ResNet, ConvNeXt, EfficientNet, MobileNet, ViT, DINOv2. |
| A16 | ML Yearning chapters 4-11 (dev/test sets, same distribution, single-number metric, satisficing metrics) | Keep as a short "practical gotchas" segment. |

### Module 3 - ConvNet Parameters

| # | Topic | Status |
|---|---|---|
| A17 | Momentum, RMSprop vs Adam | Keep briefly; add AdamW and decoupled weight decay as the modern default. |
| A18 | Hyperparameter optimisation with KerasTuner (Bayesian); what is tunable in a convnet | Keep the concept with Optuna or a plain random search; focus on *which* hyperparameters matter. |
| A19 | Saving / checkpointing / loading | Keep. |
| A20 | LR schedulers: square-root, cosine, cosine with warmup | Keep; add OneCycle and linear warmup + cosine as the modern default. |
| A21 | Debugging DL models (Chekmenev notebooks, ~2h reading) | Replace with an in-class "debugging recipe" (overfit one batch, check loss at init, inspect augmentation visually, LR range test). |
| A22 | MC quiz: BN/dropout/L2 regularise; Adam = momentum + adaptive; conv output size; receptive field of two 3x3 = one 5x5; 1x1 conv uses; argmax not differentiable; NaN loss -> lower LR; valid augmentations; TTA; which layers to fine-tune; performance scales with log(data); Grad-CAM and filter visualisation | Keep and refresh as end-of-day quizzes. Grad-CAM becomes a hands-on exercise rather than a quiz item. |

---

## B. Advanced repo (`Advanced-CV` == `Advanced_Computer_Vision`)

### Module 1 - GoogLeNet project

| # | Topic | Status |
|---|---|---|
| B1 | Inception block: 1x1 / 1x1->3x3 / 1x1->5x5 / pool->1x1, channel concat | Keep (great exercise). |
| B2 | 1x1 convolutions as bottlenecks | Keep, link to A22. |
| B3 | Auxiliary classifiers; multi-output model with weighted losses (1, 0.3, 0.3) | Keep as a historical note on gradient flow; residual connections solved the same problem better. |
| B4 | Global average pooling instead of Flatten | Keep. |
| B5 | Full GoogLeNet from scratch on cats vs dogs, `ImageDataGenerator.flow_from_directory`, `fit_generator` | Replace: build the *block*, assemble a mini version; do not train the 224x224 GoogLeNet in class (too slow, deprecated API). |

### Module 2 - R-CNN

| # | Topic | Status |
|---|---|---|
| B6 | Object detection problem statement; box formats (xyxy, xywh, cxcywh) | Keep and formalise the three box formats and conversions. |
| B7 | IoU implementation (with the `+1` pixel convention) | Keep; drop the `+1` convention (continuous coordinates are standard now); add GIoU / DIoU / CIoU as *losses*. |
| B8 | Naive sliding window detection with a frozen VGG16 patch classifier (planes in satellite imagery) | Keep as a short motivating demo; the fully-convolutional trick (a conv net *is* a sliding window) becomes the punchline. |
| B9 | Selective Search via OpenCV contrib | Keep as a 10-minute historical demo only. |
| B10 | R-CNN: proposals -> warp -> CNN classifier; positive / negative sampling at IoU 0.7 / 0.3 | Keep conceptually; the sampling rule reappears in anchor matching. |
| B11 | Non-maximum suppression | Keep (hands-on); add Soft-NMS and class-wise NMS; mention NMS-free detectors (DETR, YOLOv10). |
| B12 | Hard negative mining | Keep; link to focal loss (which largely replaced it). |
| B13 | Fast R-CNN (README only) | Teach explicitly: shared feature map + RoI pooling. |
| B14 | **Missing:** detection *evaluation* (precision / recall, AP, mAP@0.5, COCO mAP@[.5:.95]) | Add. This is the biggest gap in the source material. |

### Module 3 - Faster R-CNN

| # | Topic | Status |
|---|---|---|
| B15 | Anchors: grid of centres, sizes x ratios, d2l-style generation | Keep (hands-on with visualisation). |
| B16 | Region Proposal Network: 3x3 conv, 1x1 objectness head (k), 1x1 box head (4k), BCE + MSE loss | Keep; teach box *encoding* (deltas) and smooth-L1, which the original skipped. |
| B17 | RoI pooling via `crop_and_resize` | Keep; teach RoIAlign (what Mask R-CNN actually uses) and the quantisation problem. |
| B18 | Full model deferred to external blogs ("several days / weeks") | Replace with fine-tuning a library Faster R-CNN on a small dataset in class. |
| B19 | Two-stage vs single-stage comparison | Keep. |

### Module 4 - SSD (face detection on FDDB)

| # | Topic | Status |
|---|---|---|
| B20 | Single-stage idea: predict a dense grid of (objectness, box) directly | Keep. |
| B21 | Ellipse-to-box label conversion; FDDB dataset | Drop the dataset (manual download, ellipses); keep the idea of target encoding on a grid. |
| B22 | Conv block (conv + BN + dropout); "residual" block implemented as *concatenation* (actually DenseNet-style) | Fix: teach true residual addition vs dense concatenation, side by side. |
| B23 | Fully convolutional output `[H, W, classes + 4]`, 4x4 grid, one anchor per cell | Keep as the "YOLOv1-style" minimal detector students train end to end. |
| B24 | Confidence loss (BCE) + location loss (L1, weight 10), masked to positive cells | Keep. |
| B25 | Custom training loop with `GradientTape` | Keep the concept (a plain PyTorch loop is the same thing). |
| B26 | Multi-scale default boxes of real SSD (reading links only) | Teach explicitly as the bridge to FPN. |

### Module 5 - RetinaNet

| # | Topic | Status |
|---|---|---|
| B27 | ResNet50 backbone, multi-output (C3, C4, C5) | Keep. |
| B28 | Feature Pyramid Network: 1x1 lateral, upsample + add, 3x3 smoothing, P6 / P7 | Keep (hands-on). |
| B29 | Shared classification / regression heads (4x conv256 + output), `num_anchors * num_classes` | Keep. |
| B30 | Focal loss (via `focal-loss` pip package, formula never explained) | Keep and *derive* it with a plot of the modulating factor. |
| B31 | Anchor matching for RetinaNet (reading only) | Teach with the same matching code as Faster R-CNN. |

### Module 6 - Segmentation with U-Net

| # | Topic | Status |
|---|---|---|
| B32 | Segmentation problem statement; pixel-level prediction | Keep; add semantic vs instance vs panoptic, and metrics (mIoU, Dice). |
| B33 | Autoencoders as the ancestor of encoder-decoder nets | Keep, briefly. |
| B34 | DSB-2018 nuclei dataset; merging per-instance masks with OTSU thresholding | Replace dataset (Kaggle login) with Oxford-IIIT Pet trimaps; keep "building a target mask" as a short exercise. |
| B35 | Encoder block (conv-dropout-conv-pool, returns skip) | Keep. |
| B36 | Decoder block (transposed conv, concat skip, conv-dropout-conv) | Keep; add bilinear-upsample + conv alternative and the checkerboard artefact. |
| B37 | U-Net assembly (16-32-64-128-256), 1x1 sigmoid output, BCE | Keep; add Dice and BCE+Dice loss. |
| B38 | EarlyStopping + ModelCheckpoint callbacks | Keep. |
| B39 | Binarise predictions at 0.5, visualise | Keep. |
| B40 | Mask R-CNN as reading homework | Teach it: RoIAlign + mask head; fine-tune a library model. |
| B41 | README items never implemented: "Advancements in detection and segmentation", "Trends in CV", "Where to go from here" | Becomes the Day 3 afternoon block (ViT, DETR, DINOv2, CLIP, SAM). |

---

## C. Pedagogical patterns worth carrying over

* Every exercise notebook ships with a `tests.py` that students run to check their block (`TEST_INCEPTIONBLOCK`, `IoUTest`, `test_anchors`, ...). Keep this pattern: immediate feedback without instructor help.
* Each notebook opens with Steps / Topics / Time estimates. Keep.
* Stated student time in the Advanced repo alone is roughly 22h (IoU 2h, R-CNN 1h, SSD 5h, RetinaNet 3.5h, U-Net 5h, GoogLeNet 3h, plus smaller items), before the Fundamentals. A 3-day course has about 18h of contact time, so reading / video time moves to pre-reads and heavy training runs move to pre-built checkpoints.
* YouTube IFrames and third-party blog links carry most of the theory. The new notebooks explain each mechanism inline with generated figures so the material is self-contained.

## D. Broken or dated items in the sources (do not carry over)

* TensorFlow 2.1 / 2.3, `kerastuner`, `fit_generator`, `ImageDataGenerator`, `layers.experimental.preprocessing`: all deprecated or removed.
* `np.bool` (removed in NumPy 1.24).
* Hard-coded Windows paths (`C:/Users/kbuzar/...`) in Fundamentals 1.2.
* Three datasets require a Kaggle account or manual download (dogs-vs-cats, planes, DSB-2018) and one is a research site (FDDB).
* Several explanations outsource the mechanism to a video (focal loss, FPN, RPN, U-Net).
* `Untitled.ipynb` (empty) in Fundamentals Module 2.
