# Refresh map: old modules -> new blocks

How every topic of the three source repos lands in the new course, and what was added to bring it to the 2026 state of the art. Topic IDs refer to `01-source-topic-inventory.md`.

## 1. Old -> new mapping

| Old module (repo) | Source topics | New block(s) |
|---|---|---|
| Fundamentals M1 ConvNets from scratch | A1-A8 | D1-B1 (convolution), D1-B2 (training recipe) |
| Fundamentals M2 Pretrained ConvNet | A9-A16 | D1-B3 (architecture zoo), D1-B4 (transfer learning) |
| Fundamentals M3 ConvNet Parameters | A17-A22 | D1-B2 (optimisers, schedules, search, debugging) + daily quizzes |
| Advanced M1 GoogLeNet | B1-B5 | D1-B3 (Inception block exercise, GAP, aux heads as history) |
| Advanced M2 R-CNN | B6-B14 | D2-B1 (boxes, IoU, NMS, hard negatives, region-based lineage, **mAP added**) |
| Advanced M3 Faster R-CNN | B15-B19 | D2-B2 (anchors, matching, encoding, RPN, RoIAlign, library fine-tune) |
| Advanced M4 SSD | B20-B26 | D2-B3 (dense grid target encoding, mini detector with custom loop, multi-scale boxes) and D1-B3 (true residual vs dense block) |
| Advanced M5 RetinaNet | B27-B31 | D2-B3 (FPN, shared heads, focal loss derived) |
| Advanced M6 U-Net | B32-B40 | D3-B1 (problem, upsampling, losses, metrics), D3-B2 (U-Net), D3-B3 (Mask R-CNN taught, not just read) |
| Advanced README "trends / where to go" | B41 | D2-B4 (modern detectors), D3-B4 (transformers and foundation models) |

Nothing from the sources is dropped as a *concept*. What is dropped: the specific TF/Keras APIs, Kaggle / FDDB datasets, YouTube-only explanations, and training the full 224x224 GoogLeNet in class.

## 2. What is new (state-of-the-art refresh)

| Area | 2024 course | 2026 course |
|---|---|---|
| Framework | TensorFlow 2.1 / 2.3, Keras, KerasTuner | PyTorch 2.x (decided), torchvision, timm, `transforms.v2`, `torch.compile`, mixed precision |
| Default backbone | VGG16 | ResNet-50 for teaching, ConvNeXt / ViT / DINOv2 as drop-ins |
| Architectures | VGG, GoogLeNet, ResNet (only via RetinaNet) | + DenseNet, MobileNet (depthwise-separable), EfficientNet, ConvNeXt, ViT, Swin |
| Augmentation | flip / rotate / zoom | + RandAugment, TrivialAugment, Mixup, CutMix, random erasing, TTA |
| Optimisation | RMSprop, momentum, sqrt / cosine schedules | + AdamW, warm-up + cosine, OneCycle, LR range test, label smoothing, EMA |
| Debugging | external 2h reading | in-class recipe (overfit one batch, loss at init, inspect batches) |
| Interpretability | quiz question only | Grad-CAM, filter / feature-map visualisation hands-on |
| Detection metrics | none | precision / recall, AP, mAP@0.5, COCO mAP@[.5:.95] |
| Box losses | MSE / L1 | smooth-L1, GIoU / DIoU / CIoU, distribution focal loss (mention) |
| Proposals | Selective Search, RPN | + anchor-free (FCOS, CenterNet), set prediction (DETR), NMS-free (YOLOv10) |
| RoI features | RoI pooling (`crop_and_resize`) | RoIAlign explained and implemented |
| Detectors in practice | none trained end to end | library Faster R-CNN fine-tune; YOLO / RT-DETR fine-tune; mini YOLOv1-style detector trained from scratch |
| Open vocabulary | none | Grounding DINO / OWL-ViT demo |
| Segmentation metrics / losses | BCE + accuracy | mIoU, Dice, BCE+Dice, focal, class weights |
| Upsampling | transposed conv | + bilinear + conv, pixel shuffle, dilated conv / ASPP, checkerboard artefact |
| Segmentation models | U-Net | + pretrained encoders, U-Net++ / Attention U-Net, DeepLabv3+, SegFormer, Mask2Former, U-Net in diffusion |
| Instance segmentation | Mask R-CNN reading | Mask R-CNN fine-tuned; mask mAP; YOLO-seg |
| Foundation models | none | DINOv2 / v3, CLIP / SigLIP, SAM 2 / 3, Grounding DINO |
| Datasets | dogs-vs-cats (Kaggle), planes (Kaggle), FDDB (manual), DSB-2018 (Kaggle) | Oxford-IIIT Pet (direct), Penn-Fudan (direct), in-notebook synthetic shapes, FashionMNIST |

## 3. Dataset strategy

| Dataset | Size | Used in | Why |
|---|---|---|---|
| FashionMNIST | 30 MB | D1-B1 | instant convnet demo, learned-filter visualisation |
| Oxford-IIIT Pet | ~800 MB | D1-B2, D1-B4, D2-B1 (heads), D2-B4, D3-B1, D3-B2, D3-B4 | one download; classification + head boxes + trimaps; cats and dogs keep the spirit of the old course |
| Penn-Fudan pedestrians | 53 MB | D2-B2, D3-B3 | the standard small dataset for Faster / Mask R-CNN fine-tuning |
| Synthetic shapes (generated) | 0 | D2-B2, D2-B3, D3-B2 | every mechanism (anchors, matching, NMS, dense targets, masks) visible at a glance; trains in minutes on CPU; deterministic |
| Pretrained checkpoints (provided) | ~1 GB | all days | CPU-only students can run inference and analysis |

## 4. Notebook conventions (carried over and tightened)

* Header block: Steps / Topics and learning objectives / Time estimate / Prerequisites.
* Exercise cells marked `# YOUR CODE HERE`, each followed by a `tests.<name>(...)` checker cell (pattern from the Advanced repo).
* A `solutions/` folder with the completed notebooks.
* Every mechanism gets a generated figure (NumPy + Matplotlib) in the notebook itself rather than a linked video; links are "go deeper" extras.
* Mechanism figures and reference implementations are NumPy / Matplotlib; model code is PyTorch.
* Quizzes live in `quizzes/dayN.md` with a separate `dayN-solutions.md`.

## 5. Open decisions

1. ~~PyTorch vs TensorFlow~~ **Decided: PyTorch** (2026-11-04). Rationale kept for the record: torchvision ships Faster / Mask R-CNN, RetinaNet, FCOS, SSD and RoIAlign; timm and Hugging Face provide every backbone and foundation model used on Day 3; the Keras Applications / TF Hub equivalents are thinner and TF-Keras 3 split the ecosystem. The old `focal-loss` and `kerastuner` packages are unmaintained.
2. **Modern detector for D2-B4.** Ultralytics YOLO is the easiest to fine-tune but AGPL-licensed; RT-DETR via Hugging Face `transformers` or torchvision FCOS avoid that. Decide based on what students will use at work.
3. **GPU availability.** The plan assumes a GPU per student (local or Colab). If CPU-only, D1-B4 and D2-B2 / D2-B4 / D3-B3 fine-tunes become "run inference with the provided checkpoint".
