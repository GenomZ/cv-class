# Course map: notebooks, sections, exercises, figures

One row per notebook: what students see, what they must write, and how it is checked.
All figures are generated inside the notebook (no external images except three static
fallbacks in D3-04 for machines without `timm`). Counts from the student notebooks.

| Notebook | Cells | Figures | Exercises | Checkers |
|---|---|---|---|---|
| D0-00 Setup and tensors | 41 | 8 | 4 | `check_to_chw`, `check_normalize`, `check_d0_train_step`, `check_linear_fit` |
| D1-01 The convolution operation | 53 | 14 | 7 | `check_cross_correlate`, `check_output_size`, `check_conv2d`, `check_count_params`, `check_receptive_field`, `check_pool2d`, `check_tiny_convnet` |
| D1-02 Training convnets well | 56 | 17 | 6 | `check_augment_pipeline`, `check_mixup`, `check_label_smoothing`, `check_warmup_cosine`, `check_train_step`, `check_overfit_one_batch` |
| D1-03 Architecture zoo | 41 | 11 | 6 | `check_inception_shapes`, `check_inception_block`, `check_residual_block`, `check_bottleneck`, `check_dw_params`, `check_mini_resnet` |
| D1-04 Transfer learning | 44 | 10 | 5 (1 written) | `check_extract_features`, `check_frozen`, `check_param_groups`, `check_gradcam` |
| D2-01 Boxes, IoU, mAP | 56 | 17 | 7 | `check_box_conversions`, `check_iou`, `check_box_iou_matrix`, `check_giou`, `check_precision_recall`, `check_average_precision`, `check_nms`, `check_soft_nms` |
| D2-02 Two-stage detectors | 43 | 12 | 6 | `check_make_anchors`, `check_match_anchors`, `check_encode_decode`, `check_rpn_head`, `check_roi_align`, `check_box_predictor` |
| D2-03 One-stage detectors | 41 | 9 | 7 | `check_grid_targets`, `check_dense_loss`, `check_decode_grid`, `check_fpn_shapes`, `check_retina_head`, `check_focal_loss_torch`, `check_centerness` |
| D2-04 Modern detectors | 34 | 8 | 6 (3 written / table) | `check_hungarian`, `check_detr_cost`, `check_box_predictor` |
| D3-01 Segmentation basics | 59 | 18 | 7 | `check_rle`, `check_dice`, `check_mean_iou`, `check_soft_dice_torch`, `check_transposed_conv`, `check_pixel_shuffle`, `check_dilate_kernel` |
| D3-02 U-Net | 38 | 6 | 6 | `check_unet_shapes`, `check_unet_blocks`, `check_unet_forward`, `check_bce_dice`, `check_sliding_window`, `check_threshold_sweep` |
| D3-03 Instance segmentation | 40 | 12 | 5 | `check_paste_mask`, `check_mask_iou`, `check_mask_ap`, `check_penn_fudan_target`, `check_maskrcnn_heads` |
| D3-04 Beyond convnets | 44 | 13 | 5 (1 visual) | `check_patchify`, `check_attention`, `check_zero_shot`, `check_prompt_ensemble` |
| **Total** | **590** | **155** | **77** | **74 checkers** |

## Sections per notebook (the visual guide, in order)

**D0-00** Setup · Environment check · Datasets (grids of FashionMNIST, Pet with boxes and trimaps, Penn-Fudan with masks, shapes) · Tensor refresher · Autograd in five lines · A ten-line training loop.

**D1-01** An image is a tensor · Cross-correlation by hand (step grid + animation) · A kernel is a pattern detector (hand-made kernel gallery on a photo) · Stride, padding, dilation and the output size · Multi-channel convolution and the cost of a layer (weight-tensor diagram, `im2col` timing) · Receptive fields (two 3x3 vs one 5x5; growth through a VGG stack) · Pooling, GAP, 1x1 convolutions · The convnet pattern (shape table; FashionMNIST training; learned kernels vs hand-made; feature maps).

**D1-02** The small-data problem (overfitting curves) · The input pipeline (batch grid) · Data augmentation (gallery; Mixup / CutMix with soft labels) · Regularisation (BN train vs eval) · Optimisers and LR schedules (schedule shapes, LR range test) · The debugging recipe (overfit one batch; buggy-labels control) · Baseline vs recipe · Checkpointing and resuming (bit-exact resume).

**D1-03** Twenty-five years in one picture (timeline; top-1 vs params) · VGG · GoogLeNet / Inception (block diagram with channels) · ResNet (gradient norm vs depth: plain / plain+BN / residual; add vs concat) · Efficiency (depthwise-separable, SE, EfficientNet, ConvNeXt) · Reading a model (summaries) · Assemble a MiniResNet and train it.

**D1-04** Model zoos · Linear probe · Frozen backbone and fine-tuning with discriminative LRs (bar chart of regimes) · How much data do you need (log-axis curve) · What the network sees (filters, feature maps, PCA of embeddings) · Grad-CAM (overlays; shortcut-learning test) · Choosing and reading metrics · Self-supervised backbones (DINOv2 vs ImageNet features).

**D2-01** The detection problem (three box formats) · IoU (corner cases; sliding sweep; IoU vs object size) · GIoU / DIoU / CIoU (1-D loss landscapes; gradient-descent race) · Precision, recall, AP, mAP (PR curve built step by step; AP variants; mAP by threshold / class / size) · Ten years of detectors as a chain of fixes · The fully-convolutional trick (class heat maps vs sliding window) · NMS, Soft-NMS, class-wise NMS, hard negatives.

**D2-02** Anchors (families at one cell; full grid) · Matching (coloured anchors; max-IoU histogram) · Encoding (deltas; smooth-L1) · The RPN (tensor shapes; proposals after each filter; tiny RPN trained on shapes) · RoI pooling vs RoIAlign (grid drawing; sub-cell-shift plot) · The second stage and the four-term loss · Faster R-CNN fine-tuned on Penn-Fudan with mAP vs the COCO baseline.

**D2-03** Dense prediction on a grid (responsible cells; target tensor as image) · Mini detector trained end to end (loss curves per term; predictions vs GT) · SSD default boxes and FPN (diagram with shapes; P3 vs P5 features) · RetinaNet head · Focal loss (curves for gamma; loss share of easy negatives) · FCOS / CenterNet / modern YOLO · Library RetinaNet / FCOS on Pet images.

**D2-04** The YOLO family in one page (lineage table; speed-accuracy scatter) · DETR (block diagram; Hungarian cartoon; 6-query toy converging on 3 objects) · Open-vocabulary detection (mechanism diagram) · Fine-tune FCOS and compare with Faster R-CNN (table) · Deployment (latency vs input size; score histogram) · Which detector would you pick.

**D3-01** Three segmentation tasks on one image · Pixel accuracy lies (confusion matrix; Dice vs IoU curve) · Losses (BCE vs Dice gradients) · Getting the resolution back (transposed-conv animation; checkerboard with kernel 3 vs 4; bilinear + conv; pixel shuffle) · Dilated convolution and ASPP · A convolutional autoencoder (reconstructions; the bottleneck loses "where") · The Pet trimap target (class frequencies).

**D3-02** Anatomy (the U with tensor shapes) · The building blocks · Train on synthetic shapes (skip-connection ablation: mIoU 0.999 vs 0.35) · Pet trimaps with a pretrained ResNet-18 encoder (error map FP red / FN blue) · The U-Net family · Inference tricks (sliding window, TTA, threshold sweep).

**D3-03** From boxes to masks (mask-head diagram; 28x28 mask pasted back) · Instance metrics (mask AP; panoptic quality) · From instance PNGs to Mask R-CNN targets · Fine-tune Mask R-CNN on pedestrians (fine-tuned vs COCO: false positives drop 45 to 7) · Alternatives (YOLO-seg, Mask2Former, SAM).

**D3-04** The Vision Transformer (patchify == strided conv, proven numerically; class-token attention, rollout, attention distance, position-embedding similarity) · DINO and DINOv2 (attention heads; PCA colour maps; k-NN 85% / linear probe 91% with no fine-tuning) · CLIP (schematic; offline zero-shot and prompt-ensemble exercises) · SAM (schematic; DINOv2 prompt-to-mask imitation) · The map of the field (2026) · Capstone template.

## Measured on this CPU-only machine (checkpoints present)

Execution times and metrics are in each notebook's markdown; summary from the build:

| Notebook | Run time (s), full suite on an idle CPU, 2026-11-05 | Key number |
|---|---|---|
| D0-00 | 23 | environment and dataset previews |
| D1-01 | 45 (330 when training from scratch) | FashionMNIST 89.5% convnet vs 87.6% dense with 3.6x fewer parameters |
| D1-02 | 76 (260 first run) | overfit-one-batch loss 0.001; resume diff 0.0 |
| D1-03 | 29 (96 first run) | gradient norm at depth 60: plain 1e-29, residual flat |
| D1-04 | 73 (240 first run) | ResNet-18 @160 px: probe 78% -> fine-tune 82%; DINOv2 probe 85% |
| D2-01 | 32 | all-point AP from a toy PR curve; FCN heat map |
| D2-02 | 59 | Penn-Fudan mAP@0.5 0.99 (both COCO and fine-tuned) |
| D2-03 | 56 | mini detector mAP@0.5 0.996 after 26 s of training |
| D2-04 | 55 | FCOS 263 ms vs Faster R-CNN 553 ms at 320 px, same mAP |
| D3-01 | 38 | checkerboard artefact reproduced; Dice vs IoU |
| D3-02 | 198 | shapes mIoU 0.999 with skips vs 0.35 without; Pet mIoU 0.735 |
| D3-03 | 54 | mask AP 0.73; false positives 45 -> 7 after fine-tuning |
| D3-04 | 28 | DINOv2 k-NN 85%, linear probe 91%; DINO attention IoU with pet mask 0.50 |
| **Total** | **~13 min** | |

GPU numbers for the full-size runs (224 px, resnet50, full subsets) are stated in the
markdown of each notebook next to the CPU numbers.
