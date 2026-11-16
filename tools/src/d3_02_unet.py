from nb import md, code, exercise, header, api

NAME = "D3-02-unet"

CELLS = [
    header(
        "D3-02 · U-Net",
        steps=[
            "U-Net anatomy with tensor shapes",
            "Write the encoder block, the decoder block and assemble the U-Net",
            "Output activations and losses for binary vs multi-class masks (BCE + Dice)",
            "Train on synthetic shapes in ~2 minutes; ablate the skip connections",
            "Pet trimaps with a pretrained ResNet-18 encoder; read the errors",
            "The family: U-Net++, Attention U-Net, DeepLabv3+, SegFormer, Mask2Former, diffusion U-Nets",
            "Inference tricks: sliding windows, test-time flips, threshold tuning",
        ],
        topics=[
            "Encoder (what) + decoder (where) + skips (fine detail) = U-Net",
            "Shape bookkeeping through 4 levels; why input sizes must be divisible by 2^depth",
            "Skip connections measured: same network, same budget, with and without",
            "Reusing an ImageNet encoder for dense prediction (discriminative learning rates)",
            "Error maps (false positives vs false negatives) instead of a single number",
            "Practical inference for big images and for binary decisions",
        ],
        minutes=90,
        prereq="D3-01 (metrics, Dice, transposed convolution), Day 1 transfer learning.",
    ),
    md(r"""
    ## 0 · Setup
    """),
    api("np.set_printoptions", "torch.manual_seed", "torchkit.get_device", brief=True),
    code("""
    import sys, pathlib
    for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:   # find the cvcourse package
        if (p / "cvcourse").is_dir():
            sys.path.insert(0, str(p)); break

    import time, inspect
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from cvcourse import viz, seg, shapes, tests, torchkit, day3, labs

    %matplotlib inline
    plt.rcParams["figure.dpi"] = 90
    np.set_printoptions(precision=3, suppress=True)
    torch.manual_seed(0)
    device = torchkit.get_device()
    if device.type == "cpu":
        day3.tune_cpu_threads()
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · Anatomy

    U-Net (Ronneberger, Fischer & Brox, 2015) was designed for biomedical images with a few
    dozen labelled examples. It is the D3-01 autoencoder plus one idea: **at every
    resolution, copy the encoder features across and concatenate them to the decoder**.

    * **Encoder** (contracting path): `[conv3x3-BN-ReLU] x 2`, then `maxpool 2x2`. Resolution halves,
      channels double: it learns *what* is in the image.
    * **Bottleneck**: the coarsest, most semantic features (stride 16 for 4 levels).
    * **Decoder** (expanding path): `2x2 transposed conv` (x2, no overlap, no checkerboard),
      **concatenate the skip** of the same resolution, `[conv3x3-BN-ReLU] x 2`.
    * **Head**: a `1x1` conv to `num_classes` logits per pixel.

    For a 128 px RGB input, 4 levels and 16 base channels:
    """),
    code("""
    fig, ax = plt.subplots(figsize=(15, 6.2))
    day3.draw_unet(ax, input_size=128, depth=4, base=16, in_ch=3, num_classes=4)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Follow the left column down: every step halves the side and doubles the channels; every orange skip
    hands an encoder map to the decoder block of the **same** size.

    ### Lab · U-Net shapes
    **Predict first:** this network has about 1.9 M parameters. Roughly what share of them sits in the
    bottleneck alone, and which level needs the most activation memory during training? Read the two bar
    charts, then work through the "Try this" list under the lab (5 minutes).
    """),
    code("""
    labs.show("unet-shapes")
    """),
    md(r"""
    ### Exercise 1 · the shape schedule

    Return `[(channels, size)]` for each encoder level and the bottleneck (`depth + 1` tuples of
    ints): level `i` has `base * 2**i` channels at `input_size // 2**i` (integer division, as
    max-pooling floors). (The original paper used 64 base channels, unpadded convolutions and a
    572 px input, the lab's "original 2015" preset; "same" padding makes the bookkeeping trivial.)
    """),
    exercise(
        stub="""
        def unet_shapes(input_size, depth, base):
            '''input_size (px), depth (number of poolings), base (channels of level 0) ->
            a list of depth + 1 tuples (channels, size): the encoder levels 0 .. depth-1, then the bottleneck.'''
            schedule = []
            for i in range(depth + 1):                     # level i; the last one (i = depth) is the bottleneck
                # YOUR CODE: channels = base * 2 ** i (they double at every level)
                # YOUR CODE: size = input_size // 2 ** i (the side halves; integer division, as max-pooling floors)
                # YOUR CODE: append the tuple (channels, size) to schedule
                raise NotImplementedError
            return schedule

        tests.check_unet_shapes(unet_shapes)
        """,
        solution="""
        def unet_shapes(input_size, depth, base):
            '''input_size (px), depth (number of poolings), base (channels of level 0) ->
            a list of depth + 1 tuples (channels, size): the encoder levels 0 .. depth-1, then the bottleneck.'''
            schedule = []
            for i in range(depth + 1):                     # level i; the last one (i = depth) is the bottleneck
                channels = base * 2 ** i                   # the channels double at every level
                size = input_size // 2 ** i                # the side halves (max-pooling floors)
                schedule.append((channels, size))
            return schedule

        tests.check_unet_shapes(unet_shapes)
        """),
    md(r"""
    > **Divisibility.** Four poolings need a side divisible by $2^4 = 16$; otherwise the
    > upsampled decoder map is one pixel smaller than its skip and `torch.cat` fails (the red
    > concat you saw in the lab at 100 px: 24 vs 25). Pad the input (or crop the skip) when
    > sizes do not divide.

    ## 2 · The building blocks

    `day3.double_conv(in_ch, out_ch)` gives you the `[conv3x3-BN-ReLU] x 2` unit
    (convolutions without bias, because BatchNorm adds its own shift).
    """),
    code("""
    print(day3.double_conv(3, 16))
    """),
    md(r"""
    ### Exercise 2 · `EncoderBlock` and `DecoderBlock`

    * `EncoderBlock(in_ch, out_ch)`: `forward(x)` returns **two** tensors, `(pooled, skip)`:
      `skip = double_conv(x)` at full resolution and `pooled = maxpool2x2(skip)`.
    * `DecoderBlock(in_ch, skip_ch, out_ch, use_skip=True)`: upsample `x` with
      `nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)`, concatenate the skip **on the channel
      axis**, then `double_conv(out_ch + skip_ch, out_ch)`. With `use_skip=False` the
      concatenation is skipped (the ablation in section 4) and the conv takes `out_ch` channels.
    """),
    api("nn.MaxPool2d", "nn.ConvTranspose2d", "torch.cat", brief=True, title="Functions you will need"),
    exercise(
        stub="""
        class EncoderBlock(nn.Module):
            '''EncoderBlock(in_ch, out_ch): forward(x (N, in_ch, H, W)) -> (pooled (N, out_ch, H/2, W/2), skip (N, out_ch, H, W)).'''
            def __init__(self, in_ch, out_ch):
                super().__init__()
                # 1. YOUR CODE: self.conv = day3.double_conv(in_ch, out_ch)
                # 2. YOUR CODE: self.pool, a 2x2 max-pool (nn.MaxPool2d(2))
                raise NotImplementedError

            def forward(self, x):                      # x: (N, in_ch, H, W)
                # 3. YOUR CODE: skip = self.conv(x) -> (N, out_ch, H, W), kept for the decoder
                # 4. YOUR CODE: pooled = self.pool(skip) -> (N, out_ch, H/2, W/2), goes down to the next level
                # 5. YOUR CODE: return pooled, skip
                raise NotImplementedError


        class DecoderBlock(nn.Module):
            '''DecoderBlock(in_ch, skip_ch, out_ch): forward(x (N, in_ch, h, w), skip (N, skip_ch, 2h, 2w)) -> (N, out_ch, 2h, 2w).'''
            def __init__(self, in_ch, skip_ch, out_ch, use_skip=True):
                super().__init__()
                self.use_skip = use_skip
                # 1. YOUR CODE: self.up, a 2x2 transposed conv with stride 2 from in_ch to out_ch (doubles the size)
                # 2. YOUR CODE: conv_in, the channels after the concatenation: out_ch + skip_ch with the skip,
                #               out_ch without it (if use_skip: ... else: ...)
                # 3. YOUR CODE: self.conv = day3.double_conv(conv_in, out_ch)
                raise NotImplementedError

            def forward(self, x, skip=None):           # x: (N, in_ch, h, w), skip: (N, skip_ch, 2h, 2w)
                # 4. YOUR CODE: x = self.up(x) -> (N, out_ch, 2h, 2w)
                # 5. YOUR CODE: if self.use_skip: x = torch.cat([x, skip], dim=1) -> (N, out_ch + skip_ch, 2h, 2w)
                # 6. YOUR CODE: return self.conv(x) -> (N, out_ch, 2h, 2w)
                raise NotImplementedError

        tests.check_unet_blocks(EncoderBlock, DecoderBlock)
        """,
        solution="""
        class EncoderBlock(nn.Module):
            '''EncoderBlock(in_ch, out_ch): forward(x (N, in_ch, H, W)) -> (pooled (N, out_ch, H/2, W/2), skip (N, out_ch, H, W)).'''
            def __init__(self, in_ch, out_ch):
                super().__init__()
                # 1. two 3x3 conv-BN-ReLU
                self.conv = day3.double_conv(in_ch, out_ch)
                # 2. 2x2 max-pool
                self.pool = nn.MaxPool2d(2)

            def forward(self, x):                      # x: (N, in_ch, H, W)
                # 3. full resolution, kept for the decoder
                skip = self.conv(x)                    # (N, out_ch, H, W)
                # 4. half resolution, goes down to the next level
                pooled = self.pool(skip)               # (N, out_ch, H/2, W/2)
                # 5. both
                return pooled, skip


        class DecoderBlock(nn.Module):
            '''DecoderBlock(in_ch, skip_ch, out_ch): forward(x (N, in_ch, h, w), skip (N, skip_ch, 2h, 2w)) -> (N, out_ch, 2h, 2w).'''
            def __init__(self, in_ch, skip_ch, out_ch, use_skip=True):
                super().__init__()
                self.use_skip = use_skip
                # 1. x2 upsampling, in_ch -> out_ch
                self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
                # 2. the channels after the concatenation
                if use_skip:
                    conv_in = out_ch + skip_ch
                else:
                    conv_in = out_ch
                # 3. two 3x3 conv-BN-ReLU
                self.conv = day3.double_conv(conv_in, out_ch)

            def forward(self, x, skip=None):           # x: (N, in_ch, h, w), skip: (N, skip_ch, 2h, 2w)
                # 4. upsample
                x = self.up(x)                         # (N, out_ch, 2h, 2w)
                # 5. glue the skip to it, on the channel axis
                if self.use_skip:
                    x = torch.cat([x, skip], dim=1)    # (N, out_ch + skip_ch, 2h, 2w)
                # 6. mix them
                return self.conv(x)                    # (N, out_ch, 2h, 2w)

        tests.check_unet_blocks(EncoderBlock, DecoderBlock)
        """),
    md(r"""
    ### Exercise 3 · assemble the U-Net

    Encoders with channels `base * 2**i` (`i < depth`), a bottleneck `double_conv` to
    `base * 2**depth`, decoders in reverse order each fed the **matching skip**, and a 1x1
    head. Keep the skips in a list on the way down and walk it backwards on the way up.
    `forward(x)` takes `(N, in_ch, H, W)` and returns logits `(N, num_classes, H, W)`.
    """),
    api("nn.ModuleList", "nn.Conv2d", "torchkit.summarize", brief=True),
    exercise(
        stub="""
        class UNet(nn.Module):
            '''UNet(in_ch, num_classes, base, depth): forward(x (N, in_ch, H, W)) -> logits (N, num_classes, H, W);
            H and W must be divisible by 2 ** depth.'''
            def __init__(self, in_ch=3, num_classes=4, base=16, depth=4, use_skips=True):
                super().__init__()
                # 1. the channels of the encoder levels, e.g. [16, 32, 64, 128]
                chs = []
                for i in range(depth):
                    chs.append(base * 2 ** i)
                # 2. YOUR CODE: self.encoders = nn.ModuleList(); c_prev = in_ch; for every c in chs: append
                #               EncoderBlock(c_prev, c), then c_prev = c
                # 3. YOUR CODE: self.bottleneck = day3.double_conv(chs[-1], chs[-1] * 2)
                # 4. YOUR CODE: self.decoders = nn.ModuleList(); for every c in reversed(chs) (128, 64, 32, 16):
                #               append DecoderBlock(c * 2, c, c, use_skips)
                # 5. YOUR CODE: self.head, a 1x1 conv from base to num_classes channels
                raise NotImplementedError

            def forward(self, x):
                # 6. YOUR CODE: down: skips = []; for every encoder: x, skip = encoder(x); append skip to skips
                # 7. YOUR CODE: x = self.bottleneck(x)
                # 8. YOUR CODE: up: for decoder, skip in zip(self.decoders, reversed(skips)): x = decoder(x, skip)
                #               (the coarsest skip goes to the first decoder)
                # 9. YOUR CODE: return self.head(x) -> (N, num_classes, H, W)
                raise NotImplementedError

        tests.check_unet_forward(UNet)
        """,
        solution="""
        class UNet(nn.Module):
            '''UNet(in_ch, num_classes, base, depth): forward(x (N, in_ch, H, W)) -> logits (N, num_classes, H, W);
            H and W must be divisible by 2 ** depth.'''
            def __init__(self, in_ch=3, num_classes=4, base=16, depth=4, use_skips=True):
                super().__init__()
                # 1. the channels of the encoder levels, e.g. [16, 32, 64, 128]
                chs = []
                for i in range(depth):
                    chs.append(base * 2 ** i)
                # 2. the encoders, finest first
                self.encoders = nn.ModuleList()
                c_prev = in_ch
                for c in chs:
                    self.encoders.append(EncoderBlock(c_prev, c))
                    c_prev = c
                # 3. the bottleneck doubles the channels once more
                self.bottleneck = day3.double_conv(chs[-1], chs[-1] * 2)
                # 4. the decoders, coarsest first
                self.decoders = nn.ModuleList()
                for c in reversed(chs):                          # 128, 64, 32, 16
                    self.decoders.append(DecoderBlock(c * 2, c, c, use_skips))
                # 5. one logit per class and pixel
                self.head = nn.Conv2d(base, num_classes, kernel_size=1)

            def forward(self, x):
                # 6. down: keep every skip
                skips = []
                for encoder in self.encoders:
                    x, skip = encoder(x)                         # x: half the size; skip: full size
                    skips.append(skip)
                # 7. the bottleneck
                x = self.bottleneck(x)
                # 8. up: the coarsest skip goes to the first decoder
                for decoder, skip in zip(self.decoders, reversed(skips)):
                    x = decoder(x, skip)                         # twice the size
                # 9. the logits
                return self.head(x)                              # (N, num_classes, H, W)

        tests.check_unet_forward(UNet)
        """),
    code("""
    net = UNet(in_ch=3, num_classes=4, base=16, depth=4)
    print(torchkit.summarize(net, (1, 3, 128, 128), depth=2))
    """),
    md(r"""
    Check the total against the lab (1,942,628 for this configuration) and find the bottleneck and the
    deepest decoder block in the table: together they hold most of the weights.

    ### Output activation and loss

    | task | head channels | activation at inference | loss | prediction |
    |---|---|---|---|---|
    | binary (pet vs not) | 1 | sigmoid | `BCEWithLogits` + Dice | `prob > threshold` (tune it, section 6) |
    | multi-class (C classes, one per pixel) | C | softmax | `CrossEntropy` (+ class weights) + multi-class Dice | `argmax` |
    | multi-label (overlapping masks) | C | sigmoid per channel | BCE per channel + Dice | `prob_c > threshold_c` |

    Never put the sigmoid / softmax inside the model when training: the `...WithLogits`
    and `cross_entropy` losses apply it in a numerically stable way (log-sum-exp).

    ### Exercise 4 · BCE + Dice

    `bce_dice_loss(logits, target, bce_weight=0.5)` for binary masks:
    `bce_weight * BCEWithLogits + (1 - bce_weight) * soft Dice of sigmoid(logits)`, with the
    soft Dice of D3-01 (`eps = 1e-6`, over the whole batch).
    """),
    api("F.binary_cross_entropy_with_logits", "torch.sigmoid", "Tensor.float", "Tensor.sum", brief=True,
        title="Functions you will need"),
    exercise(
        stub="""
        def bce_dice_loss(logits, target, bce_weight=0.5):
            '''logits, target (0 / 1): same shape, one channel -> bce_weight * BCE + (1 - bce_weight) * soft Dice loss, a 0-d tensor.'''
            target = target.float()
            # 1. YOUR CODE: bce = F.binary_cross_entropy_with_logits(logits, target)
            # 2. YOUR CODE: p, the probabilities: torch.sigmoid(logits)
            # 3. YOUR CODE: inter, the soft intersection: the sum of p * target
            # 4. YOUR CODE: dice_loss = 1 - (2 * inter + 1e-6) / (p.sum() + target.sum() + 1e-6)  (as in D3-01)
            # 5. YOUR CODE: return bce_weight * bce + (1 - bce_weight) * dice_loss
            raise NotImplementedError

        tests.check_bce_dice(bce_dice_loss)
        """,
        solution="""
        def bce_dice_loss(logits, target, bce_weight=0.5):
            '''logits, target (0 / 1): same shape, one channel -> bce_weight * BCE + (1 - bce_weight) * soft Dice loss, a 0-d tensor.'''
            target = target.float()
            # 1. BCE on the logits (stable)
            bce = F.binary_cross_entropy_with_logits(logits, target)
            # 2. the probabilities
            p = torch.sigmoid(logits)
            # 3. the soft intersection
            inter = (p * target).sum()
            # 4. the soft Dice loss of D3-01
            dice_loss = 1 - (2 * inter + 1e-6) / (p.sum() + target.sum() + 1e-6)
            # 5. the weighted sum
            return bce_weight * bce + (1 - bce_weight) * dice_loss

        tests.check_bce_dice(bce_dice_loss)
        """),
    md(r"""
    For our multi-class problems we use the same idea with softmax:
    `day3.CEDiceLoss(dice_weight=0.5)` = `0.5 * cross_entropy + 0.5 * multi-class soft Dice`.

    ## 3 · Train on synthetic shapes (~2 minutes on a CPU)

    We now have the network and the loss; what is missing is data and a training run, first on
    synthetic shapes where the answer is known exactly.

    64 px images with circles, squares and triangles (4 classes with background). Colour
    already gives the class away here, so the real challenge is **where the boundary is** -
    exactly what the skip connections are for. If your `UNet` did not pass the checker, use
    `UNet = day3.UNet`.
    """),
    api("torchkit.ShapesDataset", "torch.utils.data.DataLoader", "Tensor.unique", "Tensor.tolist", "Tensor.to",
        "torch.optim.AdamW", "Module.parameters", "torchkit.count_parameters", "np.round", brief=True),
    code("""
    kw = dict(min_r=5, max_r=14)                       # object radius for 64 px images
    shp_tr = torchkit.ShapesDataset(1200, 64, seed=0, task="segmentation", **kw)
    shp_va = torchkit.ShapesDataset(200, 64, seed=1, task="segmentation", **kw)
    shp_tl = DataLoader(shp_tr, batch_size=16, shuffle=True)
    shp_vl = DataLoader(shp_va, batch_size=50)
    x0, y0 = shp_va[0]
    print("image", tuple(x0.shape), "mask", tuple(y0.shape), "classes", y0.unique().tolist())

    results = {}
    for name, use_skips in [("with skips", True), ("without skips", False)]:
        torch.manual_seed(0)
        model = UNet(in_ch=3, num_classes=4, base=16, depth=4, use_skips=use_skips).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        print(f"--- {name}: {torchkit.count_parameters(model):,} parameters")
        hist = day3.train_segmentation(model, shp_tl, day3.CEDiceLoss(0.5), opt, epochs=10, max_minutes=1.25,
                                       val_loader=shp_vl, num_classes=4, log_every=0)
        iou, miou = day3.iou_from_confusion(day3.confusion_over_loader(model, shp_vl, 4))
        results[name] = dict(model=model, hist=hist, iou=iou, miou=miou)
        print(f"{name}: val mIoU {miou:.3f}   per class {np.round(iou, 3)}")
    """),
    md(r"""
    > **Expected** (CPU, 1.25 min each): with skips mIoU > 0.95 after one epoch (often 0.99+);
    > without skips well below that, with the circles and triangles hit hardest. On a GPU
    > both runs take a few seconds per epoch.
    """),
    api("torch.no_grad", "Module.eval", "Tensor.argmax", "Tensor.cpu", "Tensor.permute", "Tensor.numpy", "np.arange",
        "np.nan_to_num", "np.convolve", "np.ones", brief=True),
    code("""
    fig = plt.figure(figsize=(16, 7.5))
    gs = fig.add_gridspec(4, 6, width_ratios=[1, 1, 1, 1, 0.15, 2.6])
    xv, yv = next(iter(DataLoader(shp_va, batch_size=4)))
    preds = {}                                          # "with skips" / "without skips" -> predicted classes (4, 64, 64)
    with torch.no_grad():
        for name, r in results.items():
            model = r["model"].eval()
            logits = model(xv.to(device))               # (4, 4 classes, 64, 64)
            preds[name] = logits.argmax(1).cpu()        # the most likely class of every pixel
    for i in range(4):
        row = [xv[i].permute(1, 2, 0).numpy(), day3.colorize(yv[i]), day3.colorize(preds["with skips"][i]),
               day3.colorize(preds["without skips"][i])]
        for j, im in enumerate(row):
            ax = fig.add_subplot(gs[i, j]); ax.imshow(im, interpolation="nearest"); ax.axis("off")
            if i == 0: ax.set_title(["image", "ground truth", "U-Net with skips", "U-Net WITHOUT skips"][j], fontsize=10)
    ax = fig.add_subplot(gs[:2, 5])
    names4 = ["background"] + shapes.CLASS_NAMES
    xk = np.arange(4)
    for k, (name, r) in enumerate(results.items()):
        ax.bar(xk + (k - 0.5) * 0.38, np.nan_to_num(r["iou"]), 0.38, label=f"{name} (mIoU {r['miou']:.2f})",
               color=["#4C78A8", "#E45756"][k])
    ax.set_xticks(xk); ax.set_xticklabels(names4); ax.set_ylim(0, 1.15); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("per-class IoU on 200 validation images")
    ax = fig.add_subplot(gs[2:, 5])
    for k, (name, r) in enumerate(results.items()):
        l = np.convolve(r["hist"]["step_loss"], np.ones(10) / 10, mode="valid")
        ax.plot(l, label=name, color=["#4C78A8", "#E45756"][k])
    ax.set_xlabel("step"); ax.set_ylabel("training loss (CE + Dice, smoothed)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Same encoder, same decoder, same budget. Without the skips the decoder must re-invent
    every boundary from a **4x4** bottleneck (64 px / 16): it knows *that* there is a
    circle in the top-left, not where its outline runs - blobby, misplaced masks, small
    objects vanish. With the skips the 64x64 encoder features (edges, colours) are handed
    straight to the last decoder block.

    ## 4 · Real data: Pet trimaps with a pretrained ResNet-18 encoder

    Small data + a pretrained encoder is the default recipe (Day 1). We take torchvision's
    ImageNet ResNet-18 and use its stages as the U-Net encoder: the stem gives a stride-2
    skip, `layer1..layer3` give stride 4/8/16 skips and `layer4` (stride 32) is the
    bottleneck. The decoder reuses the `DecoderBlock` from `day3`. Read it:
    """),
    api("Tensor.numel", "torch.zeros", brief=True),
    code("""
    print(inspect.getsource(day3.ResNetUNet.forward))
    m_tmp = day3.ResNetUNet(num_classes=3, pretrained=False)
    n_encoder = 0                                       # parameters of the ResNet-18 encoder
    for p in m_tmp.encoder_parameters():
        n_encoder += p.numel()
    n_decoder = 0                                       # parameters of the decoder
    for p in m_tmp.decoder_parameters():
        n_decoder += p.numel()
    print(f"encoder {n_encoder:,} params (ImageNet), decoder {n_decoder:,} params (random init)")
    with torch.no_grad():
        print("output:", tuple(m_tmp(torch.zeros(1, 3, 128, 128)).shape))
    """),
    code("""
    try:
        import segmentation_models_pytorch as smp
        lib = smp.Unet("resnet18", encoder_weights=None, classes=3)
        print("segmentation_models_pytorch is installed: smp.Unet('resnet18', encoder_weights='imagenet', classes=3) "
              f"would replace our hand-written model ({torchkit.count_parameters(lib):,} params). "
              "It also offers U-Net++, DeepLabv3+, FPN, PSPNet, SegFormer heads with 100+ encoders.")
    except ImportError:
        print("segmentation_models_pytorch is not installed (optional). With it, one line replaces ResNetUNet:\\n"
              "    smp.Unet('resnet18', encoder_weights='imagenet', classes=3)\\n"
              "and the same API gives U-Net++, DeepLabv3+, FPN, SegFormer heads on 100+ encoders.")
    """),
    md(r"""
    **Training recipe**: 128 px, random resized crops + horizontal flips applied to image and
    mask together (`torchvision.transforms.v2` understands `Mask` targets), CE + Dice loss,
    AdamW with a 5x smaller learning rate on the pretrained encoder.

    The cell trains for at most 3 minutes (1 200 images) when no checkpoint exists. The
    course provides `data/checkpoints/d3_pet_resnet_unet.pt`, trained the same way on
    2 000 training images for 3 epochs (OneCycle schedule, ~40 min on a CPU). Set
    `FORCE_TRAIN = True` to train yourself.

    > **Expected numbers.** Checkpoint: mIoU ~0.73 on 300 Pet test-split images at 128 px
    > (background ~0.89, pet ~0.81, border ~0.50). A 3-minute CPU run on 1 200 images:
    > mIoU ~0.6. On a GPU, 10 epochs on all 3 680 images at 224 px take ~5 minutes and reach ~0.80.
    """),
    api("torchkit.get_pet_datasets", "torchkit.fit", "torchkit.load_checkpoint", "torchkit.denormalize", "torch.Generator",
        brief=True),
    code("""
    FORCE_TRAIN = False
    CKPT = torchkit.CHECKPOINTS / "d3_pet_resnet_unet.pt"
    train_tf, eval_tf = day3.pet_seg_transforms(128)
    pet_tr, pet_va, pet_classes = torchkit.get_pet_datasets(size=128, target="segmentation", subset=(1200, 300),
                                                            train_tf=train_tf, eval_tf=eval_tf)
    pet_tl = DataLoader(pet_tr, batch_size=16, shuffle=True)
    pet_vl = DataLoader(pet_va, batch_size=32)

    torch.manual_seed(0)
    model_pet = day3.ResNetUNet(num_classes=3).to(device)
    if CKPT.exists() and not FORCE_TRAIN:
        extra = torchkit.load_checkpoint(model_pet, CKPT)
        print(f"loaded {CKPT.name} (trained {extra.get('epoch', '?')} epochs; stored val metric {extra.get('val_metric', float('nan')):.3f})")
    else:
        opt = day3.pet_unet_optimizer(model_pet)
        hist_pet = torchkit.fit(model_pet, pet_tl, pet_vl, day3.CEDiceLoss(0.5), opt, epochs=2,
                                metric_fn=day3.miou_metric(3), max_minutes=3, log_every=20, checkpoint=CKPT)
        torchkit.load_checkpoint(model_pet, CKPT)
    model_pet.eval()
    cm_pet = day3.confusion_over_loader(model_pet, pet_vl, 3)
    iou_pet, miou_pet = day3.iou_from_confusion(cm_pet)
    print(f"Pet validation ({len(pet_va)} test-split images, 128 px): mIoU {miou_pet:.3f}  " +
          "  ".join(f"{c} {v:.3f}" for c, v in zip(pet_classes, iou_pet)))
    """),
    code("""
    xb, yb = next(iter(DataLoader(pet_va, batch_size=6, shuffle=True, generator=torch.Generator().manual_seed(3))))
    with torch.no_grad():
        logits = model_pet(xb.to(device)).cpu()
    pred = logits.argmax(1)
    fig, axes = plt.subplots(4, 6, figsize=(15, 10.5))
    for j in range(6):
        im = torchkit.denormalize(xb[j])
        axes[0, j].imshow(im)
        axes[1, j].imshow(day3.colorize(yb[j]))
        axes[2, j].imshow(day3.colorize(pred[j]))
        fg_p, fg_t = (pred[j] > 0).numpy().astype(int), (yb[j] > 0).numpy().astype(int)
        axes[3, j].imshow(day3.error_map(fg_p, fg_t, 1))
        iou_j = (fg_p & fg_t).sum() / max((fg_p | fg_t).sum(), 1)
        axes[3, j].set_title(f"foreground IoU {iou_j:.2f}", fontsize=9)
    for r, lab in enumerate(["image", "ground truth", "prediction", "errors"]):
        axes[r, 0].set_ylabel(lab, fontsize=11)
    for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
    fig.legend(handles=[Patch(color=day3.SEG_COLORS[i], label=c) for i, c in enumerate(pet_classes)] +
               [Patch(color=(0.75, 0.75, 0.75), label="correct foreground"), Patch(color=(0.9, 0.15, 0.15), label="false positive"),
                Patch(color=(0.15, 0.35, 0.95), label="false negative")], loc="lower center", ncol=6, fontsize=9)
    plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.show()
    """),
    md(r"""
    Read the error row, not the score: **blue** (missed pet) concentrates on thin parts -
    ears, tails, legs - and on dark fur against dark backgrounds; **red** (hallucinated pet)
    appears on blankets, cushions and people's arms, i.e. texture that co-occurs with pets
    in the training set. The border class is the hardest by construction (it is a 2-3 pixel
    band at 128 px, and annotators disagree on it).

    ## 5 · The U-Net family

    Every modern segmentation network is "an encoder for *what*, a decoder for *where*,
    and some way to bring fine detail back". They differ in the encoder (CNN or
    transformer), how they gather context (ASPP, attention) and how they fuse scales.
    """),
    code("""
    day3.draw_variants(plt.figure(figsize=(16, 9.5)))
    plt.show()
    """),
    md(r"""
    | model | year | idea | when to use |
    |---|---|---|---|
    | U-Net | 2015 | symmetric encoder-decoder + skips | small data, medical, the default baseline |
    | U-Net++ | 2018 | nested dense skips, deep supervision | when boundaries matter and compute is cheap |
    | Attention U-Net | 2018 | gates on the skips | cluttered backgrounds (medical) |
    | DeepLabv3+ | 2018 | dilated encoder + ASPP + one skip | natural scenes, strong CNN baseline |
    | SegFormer | 2021 | hierarchical ViT encoder + MLP decoder | best accuracy/compute on ADE20K / Cityscapes class |
    | Mask2Former | 2022 | queries -> (class, mask) pairs | one model for semantic, instance and panoptic |
    | SAM / SAM 2 | 2023-24 | promptable, class-agnostic masks | D3-04: labelling, interactive tools |

    **U-Net in diffusion models.** The denoiser of DDPM / Stable Diffusion is a U-Net: it
    takes a noisy image $x_t$ and predicts the noise, *same size in, same size out* - a
    dense prediction task like segmentation. Two additions: the timestep $t$ is embedded
    (sinusoid -> MLP) and **added** to the features inside every residual block, and
    self-attention / cross-attention (to the text prompt) layers sit at the low-resolution
    levels. The skip connections are there for the same reason as here: fine detail.

    ## 6 · Inference tricks

    ### Sliding windows for big images

    Satellite tiles, whole-slide pathology images and 4K frames do not fit in memory at
    once. Predict on overlapping windows and **average the logits** where windows overlap
    (the overlap hides the weaker predictions near window borders, where the network saw
    less context).

    ### Lab · sliding-window inference
    **Predict first:** a 200 x 200 image, 64 px windows, stride 48. How many forward passes does
    `sliding_window_predict` run, and how many with stride 16? Check in the lab, then work through the
    "Try this" list (4-5 minutes).
    """),
    code("""
    labs.show("sliding-window")
    """),
    md(r"""
    ### Exercise 5 · `sliding_window_predict`

    `image (C, H, W)`; `model_fn` maps `(1, C, window, window)` to `(1, K, window, window)`.
    Window positions `0, stride, 2*stride, ...`; if the last window does not end exactly at
    the border, **add one more window ending at the border** (the dashed, shifted tiles in the
    lab). Accumulate the outputs and a count map (the lab's "tiles per pixel"), return
    `out / count` of shape `(K, H, W)`, with `K` taken from the model's output.
    """),
    exercise(
        stub="""
        def sliding_window_predict(model_fn, image, window, stride):
            '''model_fn: (1, C, window, window) -> (1, K, window, window); image (C, H, W) ->
            (K, H, W): every pixel is the average of the predictions of all windows that cover it.'''
            C, H, W = image.shape
            # 1. the top rows and left columns of the windows: 0, stride, 2 * stride, ...
            ys = list(range(0, max(H - window, 0) + 1, stride))
            xs = list(range(0, max(W - window, 0) + 1, stride))
            # 2. YOUR CODE: if the last window stops short of the bottom border (ys[-1] + window < H), append
            #               one more top row, H - window, so that a window ends exactly at the border
            # YOUR CODE: the same for xs and the right border W
            # 3. the sums of the predictions (created at the first window, once K is known), and how many windows cover each pixel
            out = None
            count = torch.zeros(1, H, W)
            for y in ys:
                for x in xs:
                    # YOUR CODE: crop = image[:, y:y + window, x:x + window] -> (C, window, window)
                    # YOUR CODE: pred = model_fn(crop[None])[0]: add a batch axis, run, remove it -> (K, window, window)
                    # YOUR CODE: if out is None: out = torch.zeros(pred.shape[0], H, W, dtype=pred.dtype)
                    # YOUR CODE: add pred into out[:, y:y + window, x:x + window], and 1 into the same window of count
                    raise NotImplementedError
            # 4. YOUR CODE: return out / count, the average (count (1, H, W) broadcasts over the K channels)
            raise NotImplementedError

        tests.check_sliding_window(sliding_window_predict)
        """,
        solution="""
        def sliding_window_predict(model_fn, image, window, stride):
            '''model_fn: (1, C, window, window) -> (1, K, window, window); image (C, H, W) ->
            (K, H, W): every pixel is the average of the predictions of all windows that cover it.'''
            C, H, W = image.shape
            # 1. the top rows and left columns of the windows: 0, stride, 2 * stride, ...
            ys = list(range(0, max(H - window, 0) + 1, stride))
            xs = list(range(0, max(W - window, 0) + 1, stride))
            # 2. one more window that ends exactly at the bottom / right border, if needed
            if ys[-1] + window < H:
                ys.append(H - window)
            if xs[-1] + window < W:
                xs.append(W - window)
            # 3. the sums of the predictions (created at the first window, once K is known), and how many windows cover each pixel
            out = None
            count = torch.zeros(1, H, W)
            for y in ys:
                for x in xs:
                    crop = image[:, y:y + window, x:x + window]       # (C, window, window)
                    pred = model_fn(crop[None])[0]                    # (K, window, window)
                    if out is None:
                        out = torch.zeros(pred.shape[0], H, W, dtype=pred.dtype)   # (K, H, W)
                    out[:, y:y + window, x:x + window] += pred
                    count[:, y:y + window, x:x + window] += 1
            # 4. the average over the windows
            return out / count                                        # (K, H, W)

        tests.check_sliding_window(sliding_window_predict)
        """),
    api("np.random.default_rng", "torch.tensor", brief=True),
    code("""
    rng = np.random.default_rng(5)
    big_img, _, _, big_mask = shapes.make_sample(rng, 200, max_objects=14, min_r=5, max_r=14)   # 200 px, not a multiple of 16
    big = torch.tensor(big_img).permute(2, 0, 1)
    shp_model = results["with skips"]["model"].eval()

    def shapes_model_fn(window_batch):
        # one window (1, 3, 64, 64) -> logits (1, 4, 64, 64), back on the CPU
        return shp_model(window_batch.to(device)).cpu()

    with torch.no_grad():
        logits_sw = sliding_window_predict(shapes_model_fn, big, window=64, stride=48)
    pred_sw = logits_sw.argmax(0).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    axes[0].imshow(big_img)
    for y in [0, 48, 96, 136]:
        for x in [0, 48, 96, 136]:
            axes[0].add_patch(Rectangle((x, y), 64, 64, fill=False, ec=viz.color((x + y) // 48), lw=1, alpha=0.8))
    axes[0].set_title("200x200 image, 16 windows of 64 px (stride 48)")
    axes[1].imshow(day3.colorize(big_mask)); axes[1].set_title("ground truth")
    axes[2].imshow(day3.colorize(pred_sw)); axes[2].set_title(f"sliding-window prediction, mIoU {seg.mean_iou(pred_sw, big_mask, 4):.3f}")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    These are the 16 windows of the lab's default state. The stitched prediction shows no tile seams
    although no single window saw the whole image; a Gaussian weighting of each window (the lab's
    "Gaussian" blend, used by nnU-Net and MONAI) would down-weight the window borders further.

    ### Test-time augmentation (TTA)

    Predict on the image and its horizontal flip, flip the second prediction back, average.
    Cheap, typically +0.5-1 mIoU point. (Older pipelines added a **CRF** on top to snap
    boundaries to image edges; modern decoders made it mostly unnecessary.)
    """),
    api("Tensor.flip"),
    api("Tensor.softmax", "np.zeros", "np.concatenate", brief=True),
    code("""
    @torch.no_grad()                                    # no gradients inside this function
    def predict_tta(model, x):
        prob = model(x).softmax(1)                      # (N, K, H, W) on the images
        prob_mirror = model(x.flip(-1)).softmax(1)      # on the mirrored images (flip the width axis)
        prob_back = prob_mirror.flip(-1)                # mirror those predictions back
        return 0.5 * (prob + prob_back)                 # the average of the two

    cm_tta = np.zeros((3, 3), np.int64)
    probs_fg = []                                       # P(foreground) = 1 - P(background), per batch
    gts_fg = []                                         # the true foreground, per batch
    for xb_, yb_ in pet_vl:
        p = predict_tta(model_pet, xb_.to(device)).cpu()
        cm_tta += seg.confusion_matrix(p.argmax(1).numpy(), yb_.numpy(), 3)
        probs_fg.append(1 - p[:, 0].numpy())
        gts_fg.append((yb_ > 0).numpy())
    probs_fg = np.concatenate(probs_fg)                 # (300, 128, 128)
    gts_fg = np.concatenate(gts_fg)
    print(f"mIoU without TTA {miou_pet:.4f}  ->  with flip TTA {day3.iou_from_confusion(cm_tta)[1]:.4f}")
    """),
    md(r"""
    ### Exercise 6 · tune the threshold on the validation set

    Many applications only need **pet vs not pet**. The foreground probability is
    `1 - p(background)`; `argmax` implicitly thresholds it near 0.5, but the IoU-optimal
    threshold is rarely 0.5. Implement `threshold_sweep(prob, target, thresholds)` returning
    `(best_threshold, ious)` where `ious[k]` is the IoU of `prob >= thresholds[k]` with the
    target (over all pixels of all images). In a real project you would tune on validation
    and report on a separate test set.
    """),
    api("np.logical_or", title="Functions you will need"),
    api("np.logical_and", "np.asarray", "np.array", "np.linspace", "np.argmin", "np.abs", "Tensor.max", brief=True),
    exercise(
        stub="""
        def threshold_sweep(prob, target, thresholds):
            '''prob: foreground probabilities, target: 0 / 1, same shape (any); thresholds: a list of numbers ->
            (best_threshold, ious): the threshold with the highest IoU (a float), and the IoU of every threshold (array).'''
            prob = np.asarray(prob)
            t = np.asarray(target).astype(bool)                 # the true foreground
            ious = []
            for th in thresholds:
                # 1. YOUR CODE: p, the predicted foreground at this threshold: prob >= th (True / False)
                # 2. YOUR CODE: union and inter, the number of pixels in p or t, and in p and t
                #               (np.logical_or / np.logical_and, then .sum())
                # 3. YOUR CODE: append inter / union to ious, or 1.0 when union is 0 (both empty: perfect agreement)
                raise NotImplementedError
            ious = np.array(ious)                               # (number of thresholds,)
            # 4. YOUR CODE: best, the index of the largest IoU (ious.argmax())
            # 5. YOUR CODE: return float(thresholds[best]) and ious
            raise NotImplementedError

        tests.check_threshold_sweep(threshold_sweep)
        """,
        solution="""
        def threshold_sweep(prob, target, thresholds):
            '''prob: foreground probabilities, target: 0 / 1, same shape (any); thresholds: a list of numbers ->
            (best_threshold, ious): the threshold with the highest IoU (a float), and the IoU of every threshold (array).'''
            prob = np.asarray(prob)
            t = np.asarray(target).astype(bool)                 # the true foreground
            ious = []
            for th in thresholds:
                # 1. the predicted foreground at this threshold
                p = prob >= th
                # 2. union and intersection, in pixels
                union = np.logical_or(p, t).sum()
                inter = np.logical_and(p, t).sum()
                # 3. the IoU (both empty: perfect agreement)
                if union > 0:
                    ious.append(inter / union)
                else:
                    ious.append(1.0)
            ious = np.array(ious)                               # (number of thresholds,)
            # 4. the index of the largest IoU
            best = ious.argmax()
            # 5. that threshold, and all the IoUs
            return float(thresholds[best]), ious

        tests.check_threshold_sweep(threshold_sweep)
        """),
    code("""
    ths = np.linspace(0.05, 0.95, 37)
    best_t, ious_t = threshold_sweep(probs_fg, gts_fg, ths)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(ths, ious_t, marker="o", ms=3)
    ax.axvline(0.5, ls=":", color="grey", label=f"0.5: IoU {ious_t[np.argmin(np.abs(ths - 0.5))]:.3f}")
    ax.axvline(best_t, color="#E45756", label=f"best {best_t:.2f}: IoU {ious_t.max():.3f}")
    ax.set_xlabel("threshold on P(foreground)"); ax.set_ylabel("foreground IoU (validation)"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ## 7 · Summary

    * U-Net = encoder (double conv + pool) -> bottleneck -> decoder (x2 up + **concat skip** + double conv) -> 1x1 head.
    * Channels double while the resolution halves; the input side must be divisible by $2^{\text{depth}}$.
    * Binary: 1 channel + sigmoid + BCE + Dice; multi-class: C channels + softmax + CE (+ Dice, + class weights).
    * Removing the skips (same budget) destroys the boundaries: the bottleneck knows *what*, the skips know *where*.
    * Pretrained encoders + a light decoder are the default on small real datasets; read error maps, not just mIoU.
    * Inference: sliding windows with averaged overlaps, flip TTA, thresholds tuned on validation.

    ### Self-check (answers in the Day 3 quiz)
    1. A 4-level U-Net receives a 100x100 image. What goes wrong, and how do you fix it?
    2. Why is the decoder's `ConvTranspose2d(k=2, s=2)` free of checkerboard artefacts?
    3. Which loss and output activation for a mask where several classes can overlap at the same pixel?
    4. Your U-Net predicts blobby, slightly shifted masks and misses small objects. Which component would you check first?
    5. What does the diffusion U-Net add to the segmentation U-Net?

    ### Go deeper
    * Ronneberger, Fischer & Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation* (MICCAI 2015).
    * Zhou et al., *UNet++: A Nested U-Net Architecture* (2018); Oktay et al., *Attention U-Net* (2018).
    * Chen et al., *Encoder-Decoder with Atrous Separable Convolution* (DeepLabv3+, ECCV 2018).
    * Xie et al., *SegFormer* (NeurIPS 2021); Cheng et al., *Masked-attention Mask Transformer* (Mask2Former, CVPR 2022).
    * Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models* (2020): the U-Net as a denoiser.
    * `segmentation_models_pytorch` documentation (encoders, losses, metrics).
    """),
]
