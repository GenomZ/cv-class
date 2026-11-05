# Notebook build specifications

The contract between the course design and the code. Every notebook is generated from a
module in `tools/src/` (DSL in `tools/nb.py`) into a **student** version (`notebooks/`,
exercise cells are stubs with `# YOUR CODE HERE`) and a **solution** version
(`notebooks/solutions/`). `python tools/run_notebooks.py <pattern>` must execute the
solution end to end on CPU in under 10 minutes per notebook.

Exemplar: `tools/src/d1_01_convolution.py` -> `notebooks/D1-01-convolution.ipynb`. Match its
voice, density of figures, header block, and the exercise -> checker rhythm.

## 0. Conventions (apply to every notebook)

* **Framework: PyTorch 2.x + torchvision** (decided). `timm` and `transformers` only in D1-04 / D2-04 / D3-04, guarded with a friendly ImportError message.
* First code cell: the `sys.path` snippet from the exemplar, imports, `%matplotlib inline`, `torch.manual_seed(0)`, `device = torchkit.get_device()`.
* Shared helpers live in `notebooks/cvcourse/`. NumPy reference implementations (`convops`, `boxes`, `seg`, `shapes`) and checkers (`tests`) already exist; PyTorch helpers go in `cvcourse/torchkit.py` (see section 1). Do not duplicate helpers inside notebooks.
* **Exercises are the point.** Each notebook has 5-8 exercises. A good exercise removes the 3-15 lines that *are* the concept (the matching rule, the loss, the skip connection), keeps the scaffolding (signatures, shapes in comments, data loading), and is followed by a `tests.check_*` cell or a visual self-check ("the overlay should show green anchors only on the shapes"). Add new checkers to `cvcourse/tests.py` as needed (NumPy or torch inputs, ASCII `[PASS]`/`[FAIL]` output).
* Exercise difficulty ramps inside a notebook: first one is 2 lines, last one is "assemble the model".
* **Every mechanism gets a figure generated in the notebook.** No YouTube embeds. External links go in a final "Go deeper" list.
* Training in class is capped: synthetic shapes trains in < 2 min CPU; Pet / Penn-Fudan fine-tunes run <= 3 epochs with `torchkit.fit(..., max_minutes=...)` and load a provided checkpoint when `torchkit.CHECKPOINTS / name` exists. Always print expected numbers in the markdown so CPU-only students know what they would have seen.
* Data under `data/` at the repo root; `torchkit.data_root()` finds it. Datasets: FashionMNIST (torchvision), Oxford-IIIT Pet (torchvision `OxfordIIITPet`, `target_types=["category", "segmentation"]`; head boxes parsed from the `annotations/xmls`), Penn-Fudan (zip from the torchvision tutorial URL), synthetic shapes (`cvcourse.shapes`).
* Markdown style: short paragraphs, one idea each; tables for comparisons; LaTeX for the one formula that matters per section; a "Summary" and "Self-check" at the end.
* Each notebook's header lists Steps / Topics / Time / Prerequisites (use `nb.header`).

## 1. `cvcourse/torchkit.py` (shared PyTorch helpers, build first)

```
get_device() -> torch.device               # cuda > mps > cpu, prints what it picked
seed_everything(seed=0)
data_root() -> Path                        # <repo>/data, created on demand
CHECKPOINTS: Path                          # <repo>/data/checkpoints
count_parameters(model) -> int
summarize(model, input_size) -> str        # layer name, output shape, params (hooks; no torchinfo dependency)
fit(model, train_loader, val_loader, loss_fn, optimizer, epochs, scheduler=None, device=None,
    metric_fn=None, max_minutes=None, amp=False, log_every=50, checkpoint=None) -> dict history
    # history keys: train_loss, val_loss, val_metric, lr ; prints a one-line epoch summary; tqdm optional
evaluate(model, loader, loss_fn, metric_fn, device) -> (loss, metric)
accuracy(logits, targets) -> float
save_checkpoint(model, path, **extra) / load_checkpoint(model, path, map_location="cpu")
lr_range_test(model, loader, loss_fn, optimizer_cls, lr_min=1e-6, lr_max=1, steps=100) -> (lrs, losses)
IMAGENET_MEAN / IMAGENET_STD ; denormalize(tensor) -> numpy HWC in [0,1]
show_batch(images, labels=None, classes=None, n=16)            # grid of a batch (denormalised)
get_pet_datasets(size=224, train_tf=None, eval_tf=None, subset=None, target="category"|"segmentation"|"detection")
    # returns (train_ds, val_ds, classes); 'detection' yields (image, {"boxes": xyxy, "labels"}) with head boxes
get_fashion_mnist(root=None) -> (train_ds, test_ds, classes)
get_penn_fudan(root=None, download=True) -> Dataset yielding (image, target dict with boxes, labels, masks)
ShapesDataset(n, size, seed, task="detection"|"segmentation"|"grid", grid=8)   # wraps cvcourse.shapes
detection_collate(batch) -> (list of images, list of targets)
gradcam(model, target_layer, image_tensor, class_idx=None) -> (H, W) heat map in [0,1]
```

All functions typed and docstringed; a `if __name__ == "__main__":` smoke test that builds ShapesDataset and runs `fit` for one epoch on a 3-layer convnet in < 30 s on CPU.

## 2. Notebook specifications

Time = minutes in class. "Figures" = generated in the notebook. "Exercises" = what students fill in (checker in brackets). "PyTorch" = the training / model cells.

### D0-00 · Setup and tensors (20 min, optional pre-course)
Steps: environment check (torch, torchvision, GPU), download script for all datasets + checkpoints, tensor refresher (shapes, broadcasting, `permute` between HWC and CHW, `unsqueeze` batch dim), autograd in 5 lines, a 10-line training loop on a linear model.
Figures: none beyond a batch grid of each dataset (FashionMNIST, Pet with boxes and trimap, Penn-Fudan with masks, shapes).
Exercises: `to_chw(img_hwc) -> tensor` [check by shape]; write the 10-line loop.

### D1-01 · The convolution operation (90 min) - DONE except the framework cell
Fill the single `framework(...)` cell with PyTorch: FashionMNIST, tiny convnet (conv32-pool-conv64-pool-conv64-GAP-fc10), 2 epochs AdamW, dense baseline, plot learned 3x3 kernels vs the hand-made gallery, 32 feature maps of one image. Expected: convnet ~90% vs dense ~87%.

### D1-02 · Training convnets well (90 min)
1. The small-data problem: Pet subset (2 000 train / 500 val, 37 classes, 128 px) trained from scratch with a small convnet -> overfits by epoch 5. Figure: train / val curves side by side.
2. Pipeline: `Dataset` -> `transforms.v2` -> `DataLoader`; `num_workers`, `pin_memory`, why shuffling matters. Figure: batch grid.
3. Augmentation: RandomResizedCrop, flips, ColorJitter, RandAugment / TrivialAugmentWide, RandomErasing; Mixup / CutMix (`transforms.v2.MixUp`, `CutMix`); what not to do (vertical flip on text). Figure: augmentation gallery, one image x 8 augmentations; Mixup / CutMix examples with their soft labels.
4. Regularisation: dropout, weight decay (AdamW, decoupled), BatchNorm train / eval modes (show running stats), label smoothing, early stopping. Figure: BN output distribution in train vs eval on a tiny batch.
5. Optimiser + schedule: SGD+momentum vs AdamW; linear warm-up + cosine; OneCycle; LR range test. Figure: schedule shapes over 20 epochs; LR range test curve.
6. Debugging recipe: overfit one batch; loss at init = ln(37); visualise an augmented batch; gradient norm per layer; NaN -> lower LR. Figure: "overfit one batch" curve going to zero.
7. Hyperparameter search: 10-line random search over lr / wd / augmentation strength on the subset (3 epochs each, capped), results table.
8. Checkpointing / resume.
Exercises: build the augmentation pipeline (visual check); implement `mixup(x, y, alpha)` [`check_mixup`]; implement `warmup_cosine(step, total, warmup, lr_max)` [`check_warmup_cosine`]; implement `label_smoothing_ce(logits, target, eps)` [`check_label_smoothing`]; complete the training step inside `fit` (forward, loss, backward, step, zero_grad in the right order) [runs]; write the "overfit one batch" test.
PyTorch: full from-scratch run with and without the recipe (3 epochs each, 128 px). Expected val acc: ~15% plain vs ~30% with augmentation + schedule (37 classes, tiny data; the point is the curve shape, Day 1 B4 gets 90%+ with transfer).

### D1-03 · Architecture zoo (90 min)
1. Timeline figure: 1998 LeNet -> 2012 AlexNet -> 2014 VGG / GoogLeNet -> 2015 ResNet -> 2017 MobileNet / DenseNet -> 2019 EfficientNet -> 2020 ViT -> 2022 ConvNeXt, one contribution each; ImageNet top-1 vs parameters scatter (hard-coded table of published numbers).
2. VGG: 3x3 everywhere; why. MiniVGG block.
3. Inception block: four branches, 1x1 bottlenecks, channel concat; auxiliary classifiers as a history note. Figure: block diagram with channel counts.
4. ResNet: residual block, identity vs projection shortcut, bottleneck (1x1-3x3-1x1), BN placement; why it trains at 100+ layers. Figure: gradient norm at the first layer vs depth for plain vs residual (random-init nets, 10 / 20 / 40 layers) - the "vanishing" plot. Residual (add) vs DenseNet (concat) side by side - fixes the source repo's mislabelled block.
5. Efficiency: depthwise-separable conv (MobileNet), parameters and FLOPs vs standard conv; squeeze-and-excitation; EfficientNet compound scaling (one table). ConvNeXt: the modernised ResNet (7x7 depthwise, LayerNorm, GELU, fewer activations).
6. Reading a model: `torchkit.summarize` on torchvision resnet18 / convnext_tiny / mobilenet_v3; parameter and FLOP table.
Exercises: `InceptionBlock(nn.Module)` [`check_inception_block` on shapes]; `BasicBlock` with the shortcut [`check_residual_block`: output shape and that output != conv path alone]; `Bottleneck`; `DepthwiseSeparable` [`check_dw_params`: parameter count]; assemble `MiniResNet` (stem + 3 stages + GAP + fc) [`check_mini_resnet`]; compute the inception output channels [`check_inception_shapes`, exists].
PyTorch: train MiniResNet on Pet subset 3 epochs (compare with D1-02 plain convnet curve); gradient-flow experiment.

### D1-04 · Transfer learning and what the network sees (90 min)
1. Model zoos: torchvision weights API (`ResNet50_Weights.IMAGENET1K_V2`, `weights.transforms()`), timm one-liner, Hugging Face. Table of backbones with top-1 / params / speed.
2. Three regimes with the same backbone (resnet50): (a) linear probe on cached features (fast, no augmentation), (b) frozen backbone + head trained end to end with augmentation, (c) fine-tune last stage(s) with 10x smaller LR, then full fine-tune with discriminative LRs. Figure: bar chart of val accuracy per regime; "which layers to unfreeze" diagram; accuracy vs training-set size (200 / 500 / 1000 / 2000 images) on a log axis.
3. Feature hierarchy: first-layer filters of the pretrained net (match D1-01 gallery); feature maps at layer1 / 2 / 3 / 4 for one image; t-SNE or PCA of penultimate features coloured by breed (37 classes).
4. Grad-CAM: implement with forward / backward hooks; overlay on correct and wrong predictions; a shortcut-learning check (does it look at the cat or the sofa?).
5. ML Yearning gotchas: dev / test from the same distribution, single-number metric, satisficing metrics, when to change the metric. Mini case study with Pet: "top-1 vs top-5 vs per-breed recall".
6. Modern note: DINOv2 features + linear probe (timm, optional) beat ImageNet-supervised features; keep as a demo cell that is skipped if timm is missing.
Exercises: `freeze(model, until="layer3")` [`check_frozen`: counts trainable params]; `param_groups_with_lrs(model, base_lr, decay=0.3)` [`check_param_groups`]; `gradcam(...)` hooks [visual]; `extract_features(model, loader)` for the linear probe [shape check]; choose and justify the unfreezing schedule (short written answer cell).
PyTorch: linear probe (1 epoch, cached features), head-only (2 epochs), fine-tune (2 epochs); expected ~85 / 88 / 92% val on 37 breeds at 224 px with resnet50 (provide checkpoints).

### D2-01 · Boxes, IoU, mAP and the region-based lineage (90 min)
1. Problem statement with Pet head boxes drawn on a batch; three box formats and conversions; normalised vs pixel coordinates. Figure: one box in all three formats.
2. IoU: vectorised NumPy; corner cases (touching, containing, no overlap). Figure: 2x3 grid of box pairs with IoU values; an interactive-style sweep: a box sliding across another with IoU plotted underneath.
3. IoU-family losses: L1 on corners vs IoU, GIoU, DIoU, CIoU; the "no gradient when boxes do not overlap" problem. Figure: loss landscape for a box moving away from a fixed GT.
4. Evaluation: match predictions to GT by score order, TP / FP / FN; PR curve; AP (all-points, 11-pt, 101-pt); mAP over classes; COCO mAP@[.5:.95]; small / medium / large. Figure: PR curve built step by step from 8 predictions; AP as the area.
5. Lineage figure: sliding window -> Selective Search + CNN (R-CNN) -> shared conv features + RoI pooling (Fast R-CNN) -> RPN (Faster R-CNN) -> one-stage -> anchor-free -> DETR; one line per step on what got faster or better.
6. The fully-convolutional trick: a classifier with GAP applied to a larger image gives a class heat map (run the D1-04 fine-tuned Pet model on a 448 px image with GAP removed). Figure: heat map overlay. Then Selective Search as a 10-line OpenCV demo if `opencv-contrib` is present (skip otherwise, with a static figure).
7. NMS: greedy, Soft-NMS, class-wise; hard-negative mining; NMS-free detectors (one paragraph).
Exercises: box conversions [`check_box_conversions`]; `box_iou` pairwise [`check_box_iou_matrix`]; `giou` [`check_giou`, add]; `precision_recall` + `average_precision` [`check_average_precision`]; `nms` [`check_nms`]; `soft_nms` [`check_soft_nms`, add].
PyTorch: the fully-convolutional heat map; `torchvision.ops.box_iou` / `nms` / `generalized_box_iou_loss` as the library equivalents to compare against.

### D2-02 · Two-stage detectors: anchors, matching, RPN, RoIAlign (90 min)
1. Anchors: feature-map stride, centres, scales x ratios; count for a 512 px image at strides 8 / 16 / 32. Figure: anchor families drawn at one cell; full grid at stride 32.
2. Matching: IoU matrix anchors x GT; positive >= 0.7, negative < 0.3, ignore between; low-quality matches; the 1:3 sampling of RPN minibatches. Figure: anchors on a shapes image coloured green / red / grey; histogram of max-IoU per anchor (the imbalance).
3. Encoding: (tx, ty, tw, th), why log for size, smooth-L1 and its beta; decode is the exact inverse. Figure: an anchor, its GT, the deltas.
4. RPN head: 3x3 conv, 1x1 objectness (A) and box (4A); proposal generation: decode, clip, NMS, top-N. Figure: RPN output tensor shapes on a (C, 32, 32) feature map; proposals after each filtering step.
5. RoI pooling vs RoIAlign: quantisation error drawn on a 3x3 grid; bilinear sampling points. Figure: both on the same RoI.
6. Second stage: per-RoI class + box refinement, the full Faster R-CNN loss (4 terms).
7. Library model: `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2` fine-tuned on Penn-Fudan (2 classes), 2 epochs or checkpoint; mAP with the D2-01 code (or torchmetrics if installed).
Exercises: `make_anchors` [`check_make_anchors`]; `match_anchors` [`check_match_anchors`]; `encode_boxes` / `decode_boxes` [`check_encode_decode`]; `RPNHead(nn.Module)` forward [`check_rpn_head` shapes, add]; `roi_align(feature, boxes, output_size)` with bilinear sampling in torch [`check_roi_align` vs `torchvision.ops.roi_align`, add]; replace the predictor head for 2 classes [runs].
PyTorch: tiny RPN trained on shapes (objectness + deltas, 1 min CPU) to show proposals appearing; Faster R-CNN fine-tune on Penn-Fudan; expected mAP@0.5 ~0.9 after 2 epochs.

### D2-03 · One-stage detectors: dense grids, SSD, FPN, RetinaNet, focal loss, anchor-free (90 min)
1. Dense prediction: output tensor `[A*(5+C), H, W]`; YOLOv1-style one-cell-per-object encoding (`shapes.grid_targets`). Figure: the grid over a shapes image with the responsible cells highlighted; the target tensor as an image.
2. Mini detector trained end to end on shapes: backbone (4 conv blocks with stride 2) -> 8x8 grid head; loss = BCE(objectness) + L1(box, positives only) + CE(class, positives only), weights; custom loop; decode + NMS. Expected: mAP@0.5 > 0.9 after ~90 s CPU. Figure: predictions vs GT grid; loss curves per term.
3. Multi-scale: why one grid fails for small + large objects; SSD default boxes on 6 maps; FPN: lateral 1x1, top-down upsample + add, 3x3 smoothing, P3-P7. Figure: FPN with tensor shapes on a 256 px input; feature-map visualisations of P3 vs P5.
4. RetinaNet: shared heads (4 x conv256 + output), A x C and A x 4 output channels, anchors per level. Figure: head diagram.
5. Class imbalance and focal loss: derive from CE, plot for gamma in {0, 0.5, 1, 2, 5}, alpha; compare to hard-negative mining and OHEM. Figure: focal loss curves; contribution of easy negatives before / after.
6. Anchor-free: FCOS (per-pixel l, t, r, b + centre-ness), CenterNet (heat-map peaks); the modern YOLO head (decoupled, DFL) in one paragraph.
7. Library: `torchvision.models.detection.retinanet_resnet50_fpn` and `fcos_resnet50_fpn` inference on Pet images with COCO weights (cats and dogs are COCO classes) - no training needed.
Exercises: `grid_targets` [`check_grid_targets`]; the three-term detection loss [`check_dense_loss`, add]; `decode_grid` + NMS [visual + mAP]; `FPN(nn.Module)` forward [`check_fpn_shapes`, add]; `focal_loss` [`check_focal_loss`]; `RetinaHead` output channels [`check_retina_head`, add].
PyTorch: mini detector training; FPN on resnet18 features; library inference.

### D2-04 · Modern detectors in practice: YOLO, DETR, fine-tuning, deployment (75 min)
1. YOLO lineage figure (v1 2016 -> v3 anchors + FPN -> v5/v8 anchor-free decoupled heads -> v10 NMS-free -> 11/12); what each generation changed, in one table.
2. DETR: CNN features -> transformer encoder / decoder -> N object queries -> set prediction; Hungarian matching cost; no anchors, no NMS; slow convergence; Deformable / RT-DETR fixes. Figure: Hungarian matching cartoon (3 queries x 2 GT cost matrix); block diagram.
3. Open-vocabulary detection: Grounding DINO / OWL-ViT with text prompts (demo cell guarded by `transformers` import; static figure fallback).
4. Fine-tune a modern detector on Pet heads (single class "head") or shapes: use `torchvision` FCOS (BSD) by default; an optional Ultralytics YOLO cell (AGPL note) for those who have it; an RT-DETR cell via `transformers`. Compare mAP, latency and parameters against the D2-02 Faster R-CNN in a table.
5. Deployment: input size vs latency plot (batch of 1, 320 / 480 / 640 px, CPU timings); `torch.onnx.export`; quantisation in one paragraph; what to log in production (score histograms, drift).
Exercises: implement `hungarian_match(cost)` with `scipy.optimize.linear_sum_assignment` and the DETR matching cost [`check_hungarian`, add]; write the latency benchmark; fill the comparison table; a written "which detector would you pick for X" cell with three scenarios.
PyTorch: FCOS fine-tune (2 epochs / checkpoint), inference timings, ONNX export.

### D3-01 · Segmentation basics: tasks, metrics, losses, upsampling (90 min)
1. Semantic vs instance vs panoptic on one Pet image (trimap + head box + per-instance). Label formats: masks, trimaps, polygons, RLE (encode / decode a mask). Figure: the three tasks side by side.
2. Metrics: pixel accuracy lies (97% by predicting background); confusion matrix -> per-class IoU -> mIoU; Dice; relation Dice = 2 IoU / (1 + IoU). Figure: Dice vs IoU curve; a worked confusion matrix.
3. Losses: CE / BCE, Dice loss (soft), focal, BCE + Dice; class weights. Figure: gradients of BCE vs Dice as a function of predicted probability on a small object.
4. Getting resolution back: transposed convolution (scatter-add; checkerboard with stride 2 / kernel 3 vs kernel 4), bilinear upsample + 3x3 conv, pixel shuffle; dilated convolution and ASPP for context without losing resolution. Figure: transposed-conv animation; checkerboard artefact image; dilation receptive-field drawing; pixel-shuffle rearrangement.
5. Autoencoder as the ancestor: train a 3-level conv autoencoder on FashionMNIST for 1 epoch, show reconstructions; the bottleneck loses "where", which motivates skip connections.
6. Build the Pet trimap target (3 classes: pet / background / border) and a binary pet mask; class-frequency bar chart.
Exercises: `rle_encode` / `rle_decode` [`check_rle`, add]; `mean_iou` [`check_mean_iou`]; `dice_coefficient` [`check_dice`]; `soft_dice_loss` in torch [`check_soft_dice_torch`, add]; `transposed_conv2d` [`check_transposed_conv`]; `pixel_shuffle` [`check_pixel_shuffle`]; `dilate_kernel` [`check_dilate_kernel`, add].
PyTorch: autoencoder; `nn.ConvTranspose2d` vs `nn.Upsample` + conv outputs on the same feature map; `nn.PixelShuffle`.

### D3-02 · U-Net (90 min)
1. U-Net anatomy with tensor shapes for a 128 px input, 4 levels, 16-32-64-128-256 channels. Figure: the U drawn with shapes and arrows (skip connections highlighted).
2. Encoder block (conv-BN-ReLU x2, return skip, pool), decoder block (upsample, concat, conv x2), bottleneck, 1x1 head; output activation and loss for binary vs multi-class.
3. Train on synthetic shapes (3 classes + background, 2 min CPU): mIoU > 0.9. Then ablation: same net without skip connections. Figure: image / GT / with skips / without skips; mIoU bars.
4. Pet trimaps with a pretrained ResNet-18 encoder (U-Net with torchvision encoder, written by hand; `segmentation_models_pytorch` as the library alternative in a guarded cell): 3 epochs at 128 px or checkpoint; expected mIoU ~0.75. Figure: prediction grid with error map (FP red / FN blue).
5. Variants in one figure each: U-Net++ (nested skips), Attention U-Net (gated skips), DeepLabv3+ (ASPP + light decoder), SegFormer (transformer encoder, MLP decoder), Mask2Former (queries). U-Net in diffusion models: same block, time embedding added (one paragraph).
6. Inference tricks: sliding-window inference for big images, TTA flips, CRF mention, threshold tuning on the validation set.
Exercises: `unet_shapes(input, depth, base)` [`check_unet_shapes`]; `EncoderBlock` / `DecoderBlock` [`check_unet_blocks`, add: shapes with and without skip]; assemble `UNet` [`check_unet_forward`, add]; the BCE + Dice combined loss [`check_bce_dice`, add]; `sliding_window_predict` [`check_sliding_window`, add: reconstructs identity]; threshold sweep on validation.
PyTorch: both trainings; ablation; checkpoint loading.

### D3-03 · Instance segmentation: Mask R-CNN and friends (75 min)
1. From boxes to masks: Mask R-CNN = Faster R-CNN + a mask branch per RoI (RoIAlign 14x14 -> 4 convs -> deconv -> 28x28 x C); per-pixel BCE only on the GT class; why RoIAlign (not pooling) matters for masks. Figure: mask-head diagram; a 28x28 mask pasted back into the box.
2. Mask mAP and panoptic quality in one table each.
3. Fine-tune `maskrcnn_resnet50_fpn_v2` on Penn-Fudan (2 epochs or checkpoint); box and mask mAP; visualise instances with colours; failure cases (overlapping people).
4. Alternatives: YOLO-seg (prototype masks + coefficients), Mask2Former / query-based unified segmentation, SAM as a class-agnostic mask generator (bridge to D3-04).
Exercises: `paste_mask(mask28, box, image_hw)` [`check_paste_mask`, add]; `mask_iou` [`check_mask_iou`, add]; replace both heads (box predictor and mask predictor) for 2 classes [runs]; the data-to-target conversion for Penn-Fudan (instance masks -> boxes + labels + masks tensor) [`check_penn_fudan_target`, add]; instance mAP via the D2-01 code with mask IoU.
PyTorch: fine-tune; inference; visualisation.

### D3-04 · Beyond convnets: ViT, DINOv2, CLIP, SAM (75 min)
1. ViT: patchify = a 16x16 stride-16 convolution (prove it: compare `nn.Conv2d(3, D, 16, 16)` with unfold + linear); cls token, position embeddings, attention; inductive bias vs data; Swin windows; ConvNeXt vs ViT today. Figure: patch grid on a Pet image; attention map of the cls token overlaid (from timm `vit_small_patch16_224.dino` or torchvision `vit_b_16`).
2. Self-supervised backbones: DINOv2 patch features -> PCA to 3 colours = unsupervised segmentation; k-NN classification on Pet with no training. Figure: PCA colour maps; k-NN accuracy vs linear probe vs D1-04 fine-tune.
3. CLIP / SigLIP: zero-shot classification of 37 breeds with prompts ("a photo of a {breed}, a type of pet"); prompt engineering effect; compare with the fine-tuned ResNet. Figure: accuracy bars; a confusion example.
4. SAM / SAM 2 / SAM 3: point and box prompts -> masks; automatic mask generation; combine Grounding DINO (text -> boxes) + SAM (boxes -> masks). Figure: prompt-to-mask examples on Pet images.
5. Map of the field in 2026 (one figure): tasks x model families with the recommended starting point for each; a "where to go next" list (3D, video, multimodal LLMs, data engines).
6. Capstone: students sketch a pipeline for a task of their choice (template cell with the questions: data, backbone, head, loss, metric, augmentation, schedule, evaluation protocol, deployment constraint).
Exercises: `patchify(img, p)` and the proof that it equals the strided conv [`check_patchify`, add]; `attention(q, k, v)` in 5 lines [`check_attention`, add]; `zero_shot_classify(image_feats, text_feats)` [`check_zero_shot`: argmax of normalised dot products, add]; PCA of patch features [visual]; prompt-engineering experiment table.
PyTorch: all model-loading cells guarded by import checks with static fallback figures saved under `notebooks/assets/` so the notebook still reads well offline. Models: torchvision `vit_b_16`, timm `vit_small_patch14_dinov2`, `open_clip` or HF `openai/clip-vit-base-patch32`, HF `facebook/sam-vit-base`.

## 3. Quizzes

`quizzes/day1.md`, `day2.md`, `day3.md` with 10 multiple-choice items each (several answers may be correct, as in the source course) and `*-solutions.md`. Reuse and refresh the source MC items (listed in `01-source-topic-inventory.md` A22) and add detection / segmentation items.

## 4. Reading list (per notebook "Go deeper" section, and `docs/05-reading-list.md`)

* D1-01: cs231n ConvNets notes; Dumoulin & Visin conv arithmetic; Araujo et al. receptive fields (Distill 2019).
* D1-02: Karpathy, "A recipe for training neural networks" (2019); Smith, "A disciplined approach to neural network hyper-parameters" (OneCycle); Cubuk et al. RandAugment; Zhang et al. Mixup; Yun et al. CutMix; Loshchilov & Hutter AdamW.
* D1-03: LeCun 1998; Krizhevsky 2012; Simonyan & Zisserman VGG; Szegedy GoogLeNet; He et al. ResNet; Huang DenseNet; Howard MobileNet; Tan & Le EfficientNet; Liu et al. ConvNeXt (2022).
* D1-04: Yosinski et al. "How transferable are features" (2014); Selvaraju Grad-CAM; Ng, ML Yearning ch. 4-11; Oquab et al. DINOv2.
* D2-01: Girshick R-CNN / Fast R-CNN; Uijlings Selective Search; Rezatofighi GIoU; Zheng DIoU / CIoU; COCO evaluation docs; Padilla et al. survey on detection metrics.
* D2-02: Ren et al. Faster R-CNN; He et al. Mask R-CNN (RoIAlign); torchvision detection tutorial.
* D2-03: Liu et al. SSD; Lin et al. FPN; Lin et al. RetinaNet (focal loss); Tian et al. FCOS; Zhou et al. CenterNet; Redmon YOLOv1.
* D2-04: Carion et al. DETR; Zhu Deformable DETR; Lv RT-DETR; Wang YOLOv10; Liu Grounding DINO; Ultralytics docs.
* D3-01: Long et al. FCN; Odena et al. "Deconvolution and checkerboard artifacts" (Distill); Chen DeepLab / ASPP; Shi et al. pixel shuffle; Milletari V-Net (Dice loss).
* D3-02: Ronneberger U-Net; Zhou U-Net++; Oktay Attention U-Net; Chen DeepLabv3+; Xie SegFormer; Cheng Mask2Former; Ho et al. DDPM (U-Net in diffusion).
* D3-03: He et al. Mask R-CNN; Kirillov panoptic segmentation; Bolya YOLACT (prototype masks).
* D3-04: Dosovitskiy ViT; Liu Swin; Caron DINO; Oquab DINOv2; Radford CLIP; Zhai SigLIP; Kirillov SAM; Ravi SAM 2; Liu Grounding DINO.
