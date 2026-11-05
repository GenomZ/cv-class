# Day 3 quiz - Segmentation and foundation models

10 questions, 15 minutes. **Several answers may be correct**: tick every correct option.
Solutions with explanations: `day3-solutions.md`.

---

**1. Pixel accuracy vs IoU.** In an image an object covers 5% of the pixels. A model predicts
"background" for every pixel. Which statements are true?

- [ ] A. Its pixel accuracy is 95%.
- [ ] B. The IoU of the object class is 0.
- [ ] C. The mIoU over the two classes (background, object) is 0.475.
- [ ] D. The Dice score of the object class is 0.05.
- [ ] E. A class that is absent from both the prediction and the ground truth should count as IoU 0 when averaging mIoU.

---

**2. Dice and IoU.** Which statements are true?

- [ ] A. A prediction with IoU = 0.6 has Dice = 0.75.
- [ ] B. For any pair of masks, Dice >= IoU.
- [ ] C. Dice = 1 exactly when IoU = 1.
- [ ] D. On a single image, ranking two predictions by Dice gives the same order as ranking them by IoU.
- [ ] E. The soft Dice loss needs thresholded (0/1) predictions to be computed.

---

**3. Transposed convolution.** Consider `nn.ConvTranspose2d(C, C, kernel_size=4, stride=2, padding=1)`.

- [ ] A. On a 16 x 16 input it produces a 32 x 32 output.
- [ ] B. With kernel 3 and stride 2 instead, output pixels receive contributions from 1, 2 or 4 input pixels depending on their position, which creates a checkerboard pattern.
- [ ] C. A transposed convolution equals a normal convolution of the zero-stuffed input with the flipped kernel.
- [ ] D. A transposed convolution is the mathematical inverse of the convolution with the same kernel.
- [ ] E. Bilinear upsampling followed by a 3x3 convolution is also learnable and avoids the uneven overlap.

---

**4. Dilated convolutions and ASPP.**

- [ ] A. A 3x3 convolution with dilation 2 covers a 5x5 neighbourhood with 9 weights.
- [ ] B. Three stacked 3x3 convolutions with dilations 1, 2, 4 have a receptive field of 15 x 15.
- [ ] C. Three stacked 3x3 convolutions with dilations 2, 2, 2 leave input pixels inside the receptive field that never influence the output ("gridding").
- [ ] D. A dilated convolution halves the spatial resolution, like a stride-2 convolution.
- [ ] E. ASPP runs several dilated convolutions with different rates in parallel and concatenates their outputs.

---

**5. U-Net.**

- [ ] A. A skip connection concatenates the encoder feature map of the same resolution to the upsampled decoder features.
- [ ] B. A 4-level U-Net (four 2x2 poolings) fed a 100 x 100 image fails at a concatenation unless the input is padded (or the skip cropped).
- [ ] C. Without skip connections, boundary positions must pass through the bottleneck, and the predicted masks become blobby and misplaced.
- [ ] D. The decoder's `ConvTranspose2d(k=2, s=2)` overlaps neighbouring output pixels and therefore produces a checkerboard.
- [ ] E. In a diffusion model the U-Net predicts one class per pixel.

---

**6. Segmentation losses and outputs.**

- [ ] A. For masks where several classes may overlap at the same pixel (multi-label), use C output channels, a sigmoid per channel and BCE per channel.
- [ ] B. For the same multi-label problem, a softmax over the C channels with cross-entropy is the right choice.
- [ ] C. With a mean BCE loss and a tiny object, most of the gradient comes from background pixels.
- [ ] D. The soft Dice loss normalises by the size of the predicted and target regions, so a small object still gets a large share of the gradient.
- [ ] E. Class weights in `F.cross_entropy(..., weight=w)` can up-weight a rare class such as the thin Pet "border" class.

---

**7. Mask R-CNN.**

- [ ] A. For every RoI, the mask head turns 14x14 RoIAlign features into a 28x28 mask for every class.
- [ ] B. The mask loss is a per-pixel softmax cross-entropy across all classes.
- [ ] C. RoIAlign matters for masks because rounding the RoI to the feature grid (RoIPool) misaligns the crop by up to half a stride.
- [ ] D. At inference the 28x28 mask of the predicted class is resized to the predicted box, pasted into the image and thresholded at 0.5.
- [ ] E. To fine-tune a COCO Mask R-CNN on 3 new classes it is enough to replace the box predictor.

---

**8. Instance and panoptic metrics.**

- [ ] A. Mask AP uses the same score-ordered greedy matching as box AP, with mask IoU instead of box IoU.
- [ ] B. For the same model, mask AP is usually lower than box AP.
- [ ] C. If PQ = 0.6 and SQ = 0.8, then RQ = 0.75.
- [ ] D. RQ is an F1 score over matched segments (TP / (TP + FP/2 + FN/2)).
- [ ] E. Two people whose boxes have IoU 0.5 necessarily have masks with IoU of at least 0.5.

---

**9. Vision Transformers.**

- [ ] A. ViT-B/16 on a 224 x 224 image processes 196 patch tokens plus one class token.
- [ ] B. The patch embedding is exactly a `Conv2d(3, 768, kernel_size=16, stride=16)`.
- [ ] C. On a 384 x 384 image ViT-B/16 sees 576 patches, and the learned position embeddings must be interpolated.
- [ ] D. The cost of global self-attention grows linearly with the number of tokens.
- [ ] E. Swin Transformer computes attention inside local windows and shifts the windows between consecutive layers.

---

**10. Foundation models.**

- [ ] A. DINO / DINOv2 are trained without labels, yet a k-NN classifier on their features is a strong baseline.
- [ ] B. CLIP zero-shot classification compares the image embedding with the text embeddings of one prompt per class (cosine similarity, argmax).
- [ ] C. CLIP must be fine-tuned on a few Pet images before it can classify breeds.
- [ ] D. SAM returns a class label with every mask.
- [ ] E. Grounding DINO + SAM segments objects named in a text prompt: text -> boxes -> masks.
- [ ] F. For one ambiguous click, SAM returns several candidate masks (e.g. whole, part, sub-part) with a predicted IoU for each.
