# Reading list

Organised by notebook. **Pre-read** = send before the course. **Core** = the paper the block
is built on; students should at least read the abstract and figure 1. **Deeper** = for
after the course. arXiv ids are given so links stay stable (`https://arxiv.org/abs/<id>`).

## Pre-course

* cs231n, *Convolutional Networks* notes: https://cs231n.github.io/convolutional-networks/ (replaces the 2h in-class reading of the old course).
* Andrew Ng, *Machine Learning Yearning*, chapters 4-11 (dev / test sets, metrics). Free PDF.
* PyTorch 60-minute blitz (tensors, autograd, `nn.Module`, a training loop): https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

## Day 1

### D1-01 The convolution operation
* Core: Dumoulin & Visin, *A guide to convolution arithmetic for deep learning*, 1603.07285 (every padding / stride / dilation combination, with animations).
* Core: Araujo, Norris & Sim, *Computing receptive fields of convolutional neural networks*, Distill 2019, https://distill.pub/2019/computing-receptive-fields/
* Deeper: Lin, Chen & Yan, *Network in Network*, 1312.4400 (1x1 convolutions and global average pooling).

### D1-02 Training convnets well
* Core: Karpathy, *A recipe for training neural networks* (2019), https://karpathy.github.io/2019/04/25/recipe/ (the debugging recipe block is this post).
* Core: Loshchilov & Hutter, *Decoupled weight decay regularization* (AdamW), 1711.05101.
* Core: Smith, *A disciplined approach to neural network hyper-parameters* (OneCycle, LR range test), 1803.09820.
* Augmentation: Zhang et al., *mixup*, 1710.09412; Yun et al., *CutMix*, 1905.04899; Cubuk et al., *RandAugment*, 1909.13719; Mueller & Hutter, *TrivialAugment*, 2103.10158.
* Deeper: Ioffe & Szegedy, *Batch Normalization*, 1502.03167; Szegedy et al., *Rethinking the Inception architecture* (label smoothing), 1512.00567.

### D1-03 Architecture zoo
* LeCun et al., *Gradient-based learning applied to document recognition* (LeNet), 1998.
* Krizhevsky, Sutskever & Hinton, *ImageNet classification with deep convolutional neural networks* (AlexNet), NeurIPS 2012.
* Simonyan & Zisserman, *Very deep convolutional networks* (VGG), 1409.1556.
* Szegedy et al., *Going deeper with convolutions* (GoogLeNet / Inception), 1409.4842.
* **Core:** He et al., *Deep residual learning for image recognition* (ResNet), 1512.03385.
* Huang et al., *Densely connected convolutional networks* (DenseNet), 1608.06993.
* Howard et al., *MobileNets*, 1704.04861; Sandler et al., *MobileNetV2*, 1801.04381.
* Hu, Shen & Sun, *Squeeze-and-Excitation networks*, 1709.01507.
* Tan & Le, *EfficientNet*, 1905.11946.
* **Core:** Liu et al., *A ConvNet for the 2020s* (ConvNeXt), 2201.03545.

### D1-04 Transfer learning and interpretability
* Core: Yosinski et al., *How transferable are features in deep neural networks?*, 1411.1792.
* Core: Selvaraju et al., *Grad-CAM*, 1610.02391.
* Zeiler & Fergus, *Visualizing and understanding convolutional networks*, 1311.2901.
* Geirhos et al., *Shortcut learning in deep neural networks*, 2004.07780.
* Oquab et al., *DINOv2: Learning robust visual features without supervision*, 2304.07193.
* Docs: torchvision models and pretrained weights API, https://pytorch.org/vision/stable/models.html ; timm, https://huggingface.co/docs/timm

## Day 2

### D2-01 Boxes, IoU, mAP, the region-based lineage
* Girshick et al., *Rich feature hierarchies* (R-CNN), 1311.2524.
* Girshick, *Fast R-CNN*, 1504.08083.
* Uijlings et al., *Selective Search for object recognition*, IJCV 2013.
* Rezatofighi et al., *Generalized Intersection over Union*, 1902.09630.
* Zheng et al., *Distance-IoU loss* (DIoU, CIoU), 1911.08287.
* Padilla et al., *A comparative analysis of object detection metrics with a companion open-source toolkit*, Electronics 2021 (AP variants explained).
* COCO detection evaluation: https://cocodataset.org/#detection-eval
* Bodla et al., *Soft-NMS*, 1704.04503.

### D2-02 Two-stage detectors
* **Core:** Ren et al., *Faster R-CNN: Towards real-time object detection with region proposal networks*, 1506.01497.
* He et al., *Mask R-CNN* (RoIAlign), 1703.06870.
* Tutorial: torchvision object detection finetuning (Penn-Fudan), https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
* Deeper: Cai & Vasconcelos, *Cascade R-CNN*, 1712.00726.

### D2-03 One-stage detectors
* Redmon et al., *You Only Look Once*, 1506.02640.
* Liu et al., *SSD: Single shot multibox detector*, 1512.02325.
* **Core:** Lin et al., *Feature Pyramid Networks for object detection*, 1612.03144.
* **Core:** Lin et al., *Focal loss for dense object detection* (RetinaNet), 1708.02002.
* Tian et al., *FCOS: Fully convolutional one-stage object detection*, 1904.01355.
* Zhou, Wang & Krähenbühl, *Objects as points* (CenterNet), 1904.07850.
* Li et al., *Generalized Focal Loss* (distribution focal loss), 2006.04388.

### D2-04 Modern detectors in practice
* **Core:** Carion et al., *End-to-end object detection with transformers* (DETR), 2005.12872.
* Zhu et al., *Deformable DETR*, 2010.04159.
* Lv et al., *DETRs beat YOLOs on real-time object detection* (RT-DETR), 2304.08069.
* Wang et al., *YOLOv10: Real-time end-to-end object detection*, 2405.14458.
* Liu et al., *Grounding DINO*, 2303.05499; Minderer et al., *Simple open-vocabulary object detection with vision transformers* (OWL-ViT), 2205.06230.
* Ultralytics docs (licence note: AGPL-3.0), https://docs.ultralytics.com
* ONNX Runtime and TensorRT deployment guides (vendor docs).

## Day 3

### D3-01 Segmentation basics
* Long, Shelhamer & Darrell, *Fully convolutional networks for semantic segmentation*, 1411.4038.
* **Core:** Odena, Dumoulin & Olah, *Deconvolution and checkerboard artifacts*, Distill 2016, https://distill.pub/2016/deconv-checkerboard/
* Shi et al., *Real-time single image and video super-resolution using an efficient sub-pixel CNN* (pixel shuffle), 1609.05158.
* Chen et al., *DeepLab* (atrous convolution, ASPP), 1606.00915.
* Milletari, Navab & Ahmadi, *V-Net* (Dice loss), 1606.04797.
* Kirillov et al., *Panoptic segmentation*, 1801.00868.

### D3-02 U-Net
* **Core:** Ronneberger, Fischer & Brox, *U-Net: Convolutional networks for biomedical image segmentation*, 1505.04597.
* Zhou et al., *UNet++*, 1807.10165.
* Oktay et al., *Attention U-Net*, 1804.03999.
* Chen et al., *Encoder-decoder with atrous separable convolution* (DeepLabv3+), 1802.02611.
* Xie et al., *SegFormer*, 2105.15203.
* Cheng et al., *Masked-attention mask transformer* (Mask2Former), 2112.01527.
* Ho, Jain & Abbeel, *Denoising diffusion probabilistic models* (U-Net as the diffusion backbone), 2006.11239.
* Library: segmentation_models_pytorch, https://github.com/qubvel-org/segmentation_models.pytorch

### D3-03 Instance segmentation
* **Core:** He et al., *Mask R-CNN*, 1703.06870.
* Bolya et al., *YOLACT* (prototype masks, the idea behind YOLO-seg), 1904.02689.
* Kirillov et al., *Panoptic segmentation*, 1801.00868 (panoptic quality).

### D3-04 Beyond convnets
* **Core:** Dosovitskiy et al., *An image is worth 16x16 words* (ViT), 2010.11929.
* Liu et al., *Swin Transformer*, 2103.14030.
* Caron et al., *Emerging properties in self-supervised vision transformers* (DINO), 2104.14294.
* Oquab et al., *DINOv2*, 2304.07193.
* Radford et al., *Learning transferable visual models from natural language supervision* (CLIP), 2103.00020.
* Zhai et al., *Sigmoid loss for language-image pre-training* (SigLIP), 2303.15343.
* Kirillov et al., *Segment Anything* (SAM), 2304.02643; Ravi et al., *SAM 2*, 2408.00714; SAM 3 (Meta, 2025) project page.
* Liu et al., *Grounding DINO*, 2303.05499.
* Survey for "where next": Kerbl et al., *3D Gaussian splatting*, 2308.04079; any current multimodal-LLM vision survey.

## Books (optional)

* Chollet, *Deep Learning with Python*, 2nd ed. (the source course was built on the 1st edition's chapter 5; chapter 8-9 of the 2nd edition are the modern equivalent).
* Zhang, Lipton, Li & Smola, *Dive into Deep Learning*, chapters 7-8 and 14 (free, with PyTorch code): https://d2l.ai
* Prince, *Understanding Deep Learning* (free PDF), chapter 10 (convnets) and 12 (transformers).
