# Day 1 quiz - ConvNets: from pixels to transfer learning

10 questions, 15 minutes. **Several answers may be correct**: tick every correct option.
Solutions with explanations: `day1-solutions.md`.

---

**1. Shapes and parameters.** An RGB image of 32 x 32 goes through a convolution with 16
filters of size 5 x 5, stride 1, padding 2, followed by a 2 x 2 max-pool with stride 2. Which
statements are true?

- [ ] A. The convolution output is 32 x 32 x 16.
- [ ] B. The final output (after pooling) is 16 x 16 x 16.
- [ ] C. The convolution has 1 216 trainable parameters (weights + biases).
- [ ] D. The max-pool layer has 16 x 4 = 64 trainable parameters.
- [ ] E. Without padding the convolution output would be 28 x 28 x 16.

---

**2. Receptive fields.** Compare two stacked 3 x 3 convolutions (stride 1) with one 5 x 5
convolution, all with C input and C output channels.

- [ ] A. Both see a 5 x 5 region of the input.
- [ ] B. The two 3 x 3 layers have fewer weights (18 C^2 vs 25 C^2).
- [ ] C. The two 3 x 3 layers add an extra non-linearity.
- [ ] D. Two 3 x 3 layers see a 9 x 9 region of the input.
- [ ] E. Adding a stride-2 layer early in a network makes the receptive field of later layers grow faster.

---

**3. 1 x 1 convolutions.** What are they used for?

- [ ] A. Reducing (or increasing) the number of channels cheaply, e.g. before a 3 x 3 or 5 x 5 conv in an Inception block.
- [ ] B. Enlarging the spatial receptive field.
- [ ] C. Mixing information across channels at each pixel (a per-pixel linear layer).
- [ ] D. Producing per-pixel class scores in a segmentation or detection head.
- [ ] E. Mixing channels after a depthwise convolution (MobileNet).

---

**4. Regularisation.** You train from scratch on 2 000 images and the validation loss rises
after epoch 5 while the training loss keeps falling. Which of these act as regularisers?

- [ ] A. Data augmentation (random crops, flips, colour jitter).
- [ ] B. Weight decay (e.g. AdamW with `weight_decay=0.05`).
- [ ] C. Dropout before the classifier.
- [ ] D. Training for twice as many epochs.
- [ ] E. Label smoothing.
- [ ] F. Batch normalisation.

---

**5. Optimisers and schedules.** Which statements are true?

- [ ] A. Adam combines momentum (a running mean of the gradients) with a per-parameter step size (from a running mean of the squared gradients).
- [ ] B. `torch.optim.Adam(..., weight_decay=wd)` and `torch.optim.AdamW(..., weight_decay=wd)` do the same thing.
- [ ] C. A linear warm-up helps because early gradients, BatchNorm statistics and Adam's moment estimates are noisy at the start.
- [ ] D. The LR range test increases the learning rate exponentially and records the loss; a good peak LR is about 10x below the loss minimum.
- [ ] E. With a cosine schedule the learning rate ends near zero.

---

**6. Debugging.** Which statements are true?

- [ ] A. For a 37-class problem, a freshly initialised network should have a cross-entropy loss close to ln 37 = 3.6.
- [ ] B. If the loss becomes NaN after a few hundred steps, a first thing to try is a lower learning rate (and / or warm-up, gradient clipping).
- [ ] C. If a model cannot drive the loss of a single fixed batch close to zero, there is probably a bug in the model, the loss or the training step.
- [ ] D. We train with cross-entropy rather than accuracy because the argmax in accuracy is not differentiable.
- [ ] E. Forgetting `optimizer.zero_grad()` has no effect because PyTorch overwrites gradients at every `backward()`.

---

**7. Augmentation and test-time augmentation.** For classifying the 37 Pet breeds:

- [ ] A. Random horizontal flips are a valid augmentation.
- [ ] B. Random vertical flips are a valid augmentation.
- [ ] C. `RandomResizedCrop` with a minimum scale of ~0.35 is a valid augmentation.
- [ ] D. Test-time augmentation (TTA) averages the predictions for several augmented copies (e.g. flips, crops) of each test image.
- [ ] E. TTA changes the model weights at test time.
- [ ] F. For handwritten digits, a 180-degree rotation is a valid augmentation.

---

**8. Architectures.** Which statements are true?

- [ ] A. A residual block computes `y = ReLU(F(x) + x)`; when F changes the shape, the shortcut becomes a 1 x 1 convolution (projection) with the same stride.
- [ ] B. A DenseNet layer concatenates its output to its input, so the number of channels grows inside a dense block.
- [ ] C. Residual connections were introduced mainly to reduce the number of parameters.
- [ ] D. A depthwise-separable 3 x 3 convolution (depthwise + pointwise) needs roughly 8-9 x fewer weights than a standard 3 x 3 convolution with the same input and output channels.
- [ ] E. An Inception block with branches producing 64, 128, 32 and 32 channels outputs 256 channels.

---

**9. Transfer learning.** You have 300 labelled photos of 5 pastry types and an ImageNet-pretrained ResNet-50.

- [ ] A. A linear probe (or frozen backbone + new head) is a sensible first step.
- [ ] B. When fine-tuning, early layers should get a *smaller* learning rate than the new head.
- [ ] C. The input preprocessing (resolution, mean / std normalisation) must match the one the backbone was trained with.
- [ ] D. Early layers are more task-specific than late layers, so they should be re-trained first.
- [ ] E. Going from 300 to 600 images will typically give a similar accuracy gain as going from 600 to 1 200: accuracy grows roughly linearly in log(data).

---

**10. What the network sees.** Which statements are true?

- [ ] A. First-layer filters of a convnet trained on natural images look like oriented edges and colour blobs, similar to hand-designed Sobel / Gabor filters.
- [ ] B. Grad-CAM weights each feature map of a late conv layer by the spatial average of the gradient of the class score with respect to it, sums them and keeps the positive part.
- [ ] C. Grad-CAM requires re-training the network with a special loss.
- [ ] D. If a classifier still predicts the right class when the object is masked out, it may be using a shortcut (e.g. the background).
- [ ] E. In PyTorch, forward hooks (`register_forward_hook`) can capture intermediate activations without changing the model code.
