from nb import md, code, exercise, header, api

NAME = "D3-04-foundation-models"

CELLS = [
    header(
        "D3-04 · Beyond convnets: ViT, DINOv2, CLIP, SAM",
        steps=[
            "Vision Transformer: patchify is a strided convolution; attention in 5 lines",
            "What a ViT looks at: class-token attention, attention distance, position embeddings",
            "Self-supervised features: DINO attention heads, DINOv2 PCA colours, k-NN without training",
            "CLIP: zero-shot classification and prompt engineering",
            "SAM: promptable segmentation (and a 10-line imitation with DINOv2 features)",
            "The map of the field in 2026, and your capstone pipeline",
        ],
        topics=[
            "ViT = patch embedding + transformer encoder; inductive bias vs data",
            "Swin windows and ConvNeXt: where convnets and transformers met",
            "Self-supervised backbones give segmentation and retrieval for free",
            "Joint image-text embedding spaces and zero-shot transfer",
            "Promptable, class-agnostic segmentation and open-vocabulary pipelines",
            "Choosing a starting point for a new vision problem",
        ],
        minutes=75,
        prereq="Day 1 (convnets, transfer learning), D3-01..03; attention is introduced here from scratch.",
    ),
    md(r"""
    ## 0 · Setup

    torchvision's `vit_b_16` (ImageNet weights, downloaded once) runs everywhere. The
    self-supervised models come from `timm` (`pip install timm`); CLIP and SAM need
    `open_clip_torch` or `transformers`. Every cell that needs an optional package checks
    for it and falls back to a pre-rendered figure from `notebooks/assets/` or to an
    offline experiment, so the notebook always runs top to bottom.
    """),
    api("np.set_printoptions", "torch.manual_seed", "torchkit.get_device", "torchkit.get_pet_datasets",
        "torch.utils.data.DataLoader", "torch.Generator", "torchkit.denormalize", brief=True),
    code("""
    import sys, pathlib, os
    for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:   # find the cvcourse package
        if (p / "cvcourse").is_dir():
            sys.path.insert(0, str(p)); break
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    import time, math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from cvcourse import viz, tests, torchkit, day3, labs

    %matplotlib inline
    plt.rcParams["figure.dpi"] = 90
    np.set_printoptions(precision=3, suppress=True)
    torch.manual_seed(0)
    device = torchkit.get_device()
    if device.type == "cpu":
        day3.tune_cpu_threads()

    try:
        import timm
        HAS_TIMM = True
        print("timm", timm.__version__, "available: DINO / DINOv2 cells will run")
    except ImportError:
        HAS_TIMM = False
        print("timm is not installed (pip install timm): DINO / DINOv2 cells will show pre-rendered figures")

    # a few Pet test-split images at 224 px with their trimap masks (0 bg, 1 pet, 2 border)
    _, pet_va, _ = torchkit.get_pet_datasets(size=224, target="segmentation", subset=(37, 37))
    demo_x, demo_m = next(iter(DataLoader(pet_va, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(1))))
    demo_img = [torchkit.denormalize(t) for t in demo_x]
    print("demo batch:", tuple(demo_x.shape))
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · The Vision Transformer (ViT)

    Dosovitskiy et al. (2020) asked: what if we feed an image to a plain NLP transformer?
    A transformer eats a *sequence of tokens*, so:

    1. cut the 224x224 image into a 14x14 grid of **16x16 patches** (196 patches);
    2. flatten each patch (3 x 16 x 16 = 768 numbers) and project it linearly to $D$ dims;
    3. prepend a learnable **[class] token**, add learnable **position embeddings**;
    4. run $L$ transformer blocks (multi-head self-attention + MLP, LayerNorm, residuals);
    5. classify from the final [class] token.
    """),
    api("np.ones", brief=True),
    code("""
    img0 = demo_img[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1, 2.2]))
    axes[0].imshow(img0)
    for k in range(0, 225, 16):
        axes[0].axhline(k - 0.5, color="white", lw=0.6); axes[0].axvline(k - 0.5, color="white", lw=0.6)
    axes[0].set_title("224 x 224 image = 14 x 14 patches of 16 x 16"); axes[0].axis("off")
    strip = np.ones((16, 16 * 12 + 11 * 4, 3))
    for j, idx in enumerate([0, 1, 2, 13, 14, 15, 90, 91, 104, 105, 194, 195]):
        r, c = divmod(idx, 14)
        strip[:, j * 20:j * 20 + 16] = img0[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16]
    axes[1].imshow(strip); axes[1].set_yticks([])
    axes[1].set_xticks([j * 20 + 8 for j in range(12)]); axes[1].set_xticklabels([0, 1, 2, 13, 14, 15, 90, 91, 104, 105, 194, 195], fontsize=8)
    axes[1].set_title("... a sequence of 196 tokens (patch index below), each 3*16*16 = 768 numbers -> linear -> D dims")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Lab · Patches and attention
    Steps 1-2 turn the image into tokens; the first transformer layer then lets every token take a
    weighted average over **all** tokens (self-attention; Exercise 2 writes it in five lines).
    **Predict first:** click a patch of the dark uniform. Will the patches that get most of its
    attention weight be its neighbours, as in a 3x3 convolution, or anywhere in the image?
    Check, then work through the lab's "Try this" list (6 minutes). The lab's patch embeddings are
    hand-made (colour, contrast, edges); a trained ViT's attention maps follow below.
    """),
    code("""
    labs.show("patch-attention")
    """),
    md(r"""
    ### Exercise 1 · patchify, and the proof that it is a convolution

    Implement `patchify(img (B, C, H, W), p) -> (B, N, C*p*p)` with patches in row-major
    order and each patch flattened as `(C, p, p)`. Then a linear layer on those vectors is
    **exactly** a `Conv2d(C, D, kernel_size=p, stride=p)`: the kernel never overlaps,
    every patch meets the same weights. (That is how every ViT implements its "patch
    embedding".) Hint: `x.unfold(2, p, p).unfold(3, p, p)` gives `(B, C, H/p, W/p, p, p)`.
    """),
    api("Tensor.unfold", title="Functions you will need"),
    api("Tensor.permute", "Tensor.reshape", brief=True, title="Functions you will need (met before)"),
    exercise(
        stub="""
        def patchify(img, p):
            '''img (B, C, H, W), patch size p -> (B, N, C*p*p): the N = (H/p) * (W/p) patches in row-major order,
            each flattened in the order (C, p, p).'''
            B, C, H, W = img.shape
            # 1. YOUR CODE: x = img.unfold(2, p, p): cut the height into pieces of p rows -> (B, C, H/p, W, p)
            # 2. YOUR CODE: x = x.unfold(3, p, p): the same for the width -> (B, C, H/p, W/p, p, p)
            # 3. YOUR CODE: put the grid axes first and the channels next to the pixels of a patch:
            #               x.permute(0, 2, 3, 1, 4, 5) -> (B, H/p, W/p, C, p, p)
            # 4. YOUR CODE: return x reshaped to (B, (H // p) * (W // p), C * p * p)
            raise NotImplementedError

        tests.check_patchify(patchify)
        """,
        solution="""
        def patchify(img, p):
            '''img (B, C, H, W), patch size p -> (B, N, C*p*p): the N = (H/p) * (W/p) patches in row-major order,
            each flattened in the order (C, p, p).'''
            B, C, H, W = img.shape
            # 1. cut the height into pieces of p rows
            x = img.unfold(2, p, p)                            # (B, C, H/p, W, p)
            # 2. the same for the width
            x = x.unfold(3, p, p)                              # (B, C, H/p, W/p, p, p)
            # 3. grid axes first, then the channels and the pixels of one patch
            x = x.permute(0, 2, 3, 1, 4, 5)                    # (B, H/p, W/p, C, p, p)
            # 4. one row of C*p*p numbers per patch
            return x.reshape(B, (H // p) * (W // p), C * p * p)

        tests.check_patchify(patchify)
        """),
    api("torchvision.models.vit_b_16"),
    api("Module.eval", "Tensor.to", "torch.no_grad", "Tensor.flatten", "Tensor.transpose", "torch.allclose", "Tensor.abs",
        "Tensor.max", "torchkit.count_parameters", brief=True),
    code("""
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    try:
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval().to(device)
        HAS_VIT = True
    except Exception as e:                                       # no internet on first use
        vit, HAS_VIT = None, False
        print("could not load vit_b_16 weights:", e)

    if HAS_VIT:
        x = demo_x[:2].to(device)
        with torch.no_grad():
            by_conv = vit.conv_proj(x).flatten(2).transpose(1, 2)                         # (B, 196, 768)
            Wp, bp = vit.conv_proj.weight, vit.conv_proj.bias                             # (768, 3, 16, 16)
            by_linear = patchify(x, 16) @ Wp.reshape(768, -1).T + bp
        print("vit_b_16 patch embedding: conv", tuple(by_conv.shape), "== unfold + linear:",
              torch.allclose(by_conv, by_linear, atol=1e-4), f"(max diff {(by_conv - by_linear).abs().max():.1e})")
        print(f"vit_b_16: {torchkit.count_parameters(vit) / 1e6:.1f} M parameters, 12 layers, 12 heads, D = 768")
    """),
    md(r"""
    ### Exercise 2 · attention in five lines

    Every token makes a **query** $q$, a **key** $k$ and a **value** $v$ (three linear maps).
    Token $i$ looks at every token $j$ with weight $\text{softmax}_j(q_i \cdot k_j / \sqrt{d})$ and
    returns the weighted sum of the values:

    $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

    Implement it for tensors of shape `(..., N, d)`; return `(output, weights)`. Shapes:
    `q (..., Nq, d)`, `k (..., Nk, d)`, `v (..., Nk, dv)` -> `output (..., Nq, dv)` and
    `weights (..., Nq, Nk)`, each row summing to 1 (the checker uses Nq != Nk). In the lab,
    q = k = a hand-made patch embedding and v = the patch pixels; in a ViT all three are learned
    linear maps of each token. Multi-head attention runs $h$ of these in parallel on
    $D/h$-dimensional slices and concatenates.
    """),
    api("Tensor.transpose", "Tensor.softmax", brief=True, title="Functions you will need"),
    exercise(
        stub="""
        def attention(q, k, v):
            '''q (..., Nq, d), k (..., Nk, d), v (..., Nk, dv) -> (output (..., Nq, dv), weights (..., Nq, Nk)).'''
            d = q.shape[-1]
            # 1. YOUR CODE: scores, every query against every key: q @ k.transpose(-2, -1) -> (..., Nq, Nk)
            # 2. YOUR CODE: divide scores by math.sqrt(d)
            # 3. YOUR CODE: weights, the softmax over the keys (dim=-1): every row sums to 1 -> (..., Nq, Nk)
            # 4. YOUR CODE: output, the weighted sum of the values: weights @ v -> (..., Nq, dv)
            # 5. YOUR CODE: return output, weights
            raise NotImplementedError

        tests.check_attention(attention)
        """,
        solution="""
        def attention(q, k, v):
            '''q (..., Nq, d), k (..., Nk, d), v (..., Nk, dv) -> (output (..., Nq, dv), weights (..., Nq, Nk)).'''
            d = q.shape[-1]
            # 1. every query against every key: dot products
            scores = q @ k.transpose(-2, -1)                   # (..., Nq, Nk)
            # 2. scaled, so that the softmax does not saturate for large d
            scores = scores / math.sqrt(d)
            # 3. one probability distribution over the keys per query
            weights = scores.softmax(dim=-1)                   # (..., Nq, Nk), rows sum to 1
            # 4. the weighted sum of the values
            output = weights @ v                               # (..., Nq, dv)
            # 5. both
            return output, weights

        tests.check_attention(attention)
        """),
    md(r"""
    **Convolution vs attention.** A 3x3 conv mixes each pixel with a fixed neighbourhood
    using weights that depend only on the *relative position*: locality and translation
    equivariance are built in (an *inductive bias*). Attention mixes each token with
    **all** tokens using weights computed from the *content*: nothing is built in, so the
    ViT must learn locality from data. It does not even know where a token is unless position
    embeddings are added: the lab's "shuffle the patches" toggle with position strength 0 left
    every weight unchanged. Consequence (ViT paper): trained on ImageNet-1k
    alone, ViT-B/16 reaches ~78% top-1 (below a ResNet of similar size); pre-trained on
    14M-300M images it overtakes convnets (~84-88%). DeiT later reached ~82% on ImageNet-1k
    alone with heavy augmentation and distillation.

    ### What does the class token look at?

    torchvision's ViT does not return attention weights, so `day3.torchvision_vit_all_attentions`
    hooks every block and recomputes them. Two views: the **last layer's** attention from
    the class token to the 196 patches, and **attention rollout** (Abnar & Zuidema 2020),
    which multiplies the attention matrices of all layers (plus the residual identity) to
    trace information back to the input.
    """),
    api("Tensor.argmax", "Tensor.mean", "Tensor.cpu", "Tensor.numpy", brief=True),
    code("""
    if HAS_VIT:
        with torch.no_grad():
            atts = day3.torchvision_vit_all_attentions(vit, demo_x[:6].to(device))
            pred = vit(demo_x[:6].to(device)).argmax(1).cpu()
        roll = day3.attention_rollout([a.cpu() for a in atts])
        last = atts[-1].mean(1)[:, 0, 1:].cpu()                        # head-averaged, class token -> patches
        cats = ViT_B_16_Weights.IMAGENET1K_V1.meta["categories"]
        fig, axes = plt.subplots(3, 6, figsize=(16, 8.3))
        for j in range(6):
            axes[0, j].imshow(demo_img[j]); axes[0, j].set_title(f"ImageNet: {cats[pred[j]][:22]}", fontsize=8)
            viz.heatmap_overlay(axes[1, j], demo_img[j], last[j].reshape(14, 14).numpy(), alpha=0.6)
            viz.heatmap_overlay(axes[2, j], demo_img[j], roll[j, 0, 1:].reshape(14, 14).numpy(), alpha=0.6)
        axes[1, 0].set_title("last-layer [class] attention", fontsize=9, loc="left")
        axes[2, 0].set_title("attention rollout (all 12 layers)", fontsize=9, loc="left")
        for ax in axes[0]: ax.axis("off")
        plt.tight_layout(); plt.show()
    else:
        day3.show_asset("d3-04-vit-attention.png", "vit_b_16 class-token attention (pre-rendered)")
    """),
    md(r"""
    The supervised ViT's attention is noisy: it was only asked for a label, and the last
    layer spreads attention over a few "register-like" background tokens (low-information
    patches that large ViTs repurpose as scratch space for global information; Darcet et al.,
    Go deeper). Rollout is
    cleaner and usually covers the animal. We will see that **self-supervised** ViTs (DINO)
    produce far crisper maps.

    ### Local early, global late: the mean attention distance

    For each layer and head, how far (in pixels) does a patch look on average? Early layers
    have both local heads (like a conv) and global heads; late layers are all global. This
    is the ViT paper's Figure 7, recomputed on our images.
    """),
    api("F.normalize"),
    api("Tensor.detach", "np.arange", brief=True),
    code("""
    if HAS_VIT:
        mad = day3.mean_attention_distance([a.cpu() for a in atts], grid=14, patch=16)
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw=dict(width_ratios=[1.2, 1]))
        for h in range(mad.shape[1]):
            axes[0].scatter(np.arange(1, 13), mad[:, h], s=14, alpha=0.7)
        axes[0].set_xlabel("layer"); axes[0].set_ylabel("mean attention distance (px)"); axes[0].grid(alpha=0.3)
        axes[0].set_title("12 heads per layer (dots); a 3x3 conv sees ~1.5 px, the image is 224 px")
        pe = vit.encoder.pos_embedding[0, 1:].detach().cpu()                       # (196, 768) learned
        sim = F.normalize(pe, dim=1) @ F.normalize(pe, dim=1).T
        grid = np.ones((7 * 15, 7 * 15)) * np.nan
        for a, r in enumerate(range(0, 14, 2)):
            for b, c in enumerate(range(0, 14, 2)):
                grid[a * 15:a * 15 + 14, b * 15:b * 15 + 14] = sim[r * 14 + c].reshape(14, 14).numpy()
        axes[1].imshow(grid, cmap="viridis"); axes[1].axis("off")
        axes[1].set_title("learned position embeddings: cosine similarity of\\npatch (row, col) to all others (every 2nd patch)")
        plt.tight_layout(); plt.show()
    """),
    md(r"""
    Right panel: the position embeddings were learned from 1-D indices with no notion of
    2-D, yet each one is most similar to its spatial neighbours, and rows / columns emerge:
    the model *learned* the image grid.

    ### Swin and ConvNeXt: the two ends meet

    Plain ViT attention costs $O(N^2)$ in the number of tokens - fine for 196 tokens, painful
    for segmentation at high resolution. **Swin Transformer** (2021) restricts attention to
    local windows and shifts the windows in alternate layers, and builds a convnet-like
    pyramid (stride 4, 8, 16, 32) - so it plugs into FPN / U-Net decoders. **ConvNeXt**
    (2022) went the other way: a ResNet modernised with ViT training recipes (7x7 depthwise
    convs, LayerNorm, GELU, fewer activations) matches Swin. Today: convnets and ViTs are
    on par at equal compute and data; ViTs win at scale and in multimodal models,
    convnets on small data and edge devices.
    """),
    code("""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    day3.draw_swin(axes)
    plt.tight_layout(); plt.show()
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · Self-supervised backbones: DINO and DINOv2

    **DINO** (Caron et al., 2021) trains a ViT with *no labels*: a student network must
    match the output of a teacher (an exponential moving average of itself) on different
    crops of the same image. **DINOv2** (Oquab et al., 2023) scales this to 142M curated
    images. The surprise: the features know where objects are. The last-layer attention
    heads of DINO ViT-S/16 segment the animal although no mask was ever seen:
    """),
    api("Tensor.cumsum", "torch.zeros_like"),
    api("timm.create_model", "Tensor.argsort", "Tensor.sum", "Tensor.float", "F.interpolate", "np.mean", brief=True),
    code("""
    if HAS_TIMM:
        dino = timm.create_model("vit_small_patch16_224.dino", pretrained=True).eval().to(device)
        with torch.no_grad():
            a_dino = day3.timm_last_attention(dino, demo_x[:4].to(device)).cpu()     # (4, 6 heads, 197, 197)
        fig, axes = plt.subplots(4, 7, figsize=(16, 9.6))
        for i in range(4):
            axes[i, 0].imshow(demo_img[i]); axes[i, 0].axis("off")
            for h in range(6):
                axes[i, h + 1].imshow(a_dino[i, h, 0, 1:].reshape(14, 14), cmap="inferno"); axes[i, h + 1].axis("off")
                if i == 0: axes[i, h + 1].set_title(f"head {h}", fontsize=9)
        plt.suptitle("DINO ViT-S/16: last-layer [class]-token attention of the 6 heads (no labels used in training)", y=1.0)
        plt.tight_layout(); plt.show()
    else:
        day3.show_asset("d3-04-dino-attention.png", "DINO ViT-S/16 attention heads (pre-rendered)", figsize=(14, 8.5))
    """),
    code("""
    def fg_iou(heat, mask, keep=0.6):
        \"\"\"Threshold a (14, 14) heat map to keep `keep` of its mass, upsample, IoU with the pet (+border) mask.\"\"\"
        h = heat.flatten()                                       # (196,) attention of every patch
        order = h.argsort(descending=True)                       # patches, strongest first
        cum = h[order].cumsum(0) / h.sum()                       # share of the total mass covered so far
        sel = torch.zeros_like(h, dtype=torch.bool)
        sel[order[cum <= keep]] = True                           # the strongest patches that hold `keep` of the mass
        up = F.interpolate(sel.reshape(1, 1, 14, 14).float(), size=mask.shape, mode="nearest")[0, 0] > 0.5
        gt = mask > 0
        return float((up & gt).sum() / (up | gt).sum())

    rows = []
    if HAS_VIT:
        ious_last = []                                           # one IoU per image
        ious_roll = []
        for i in range(4):
            ious_last.append(fg_iou(last[i], demo_m[i]))
            ious_roll.append(fg_iou(roll[i, 0, 1:], demo_m[i]))
        rows.append(("supervised ViT-B/16, last layer", np.mean(ious_last)))
        rows.append(("supervised ViT-B/16, rollout", np.mean(ious_roll)))
    if HAS_TIMM:
        ious_best = []                                           # per image: the best of the 6 heads
        ious_mean = []                                           # per image: the average map of the 6 heads
        for i in range(4):
            head_ious = []
            for h in range(6):
                head_ious.append(fg_iou(a_dino[i, h, 0, 1:], demo_m[i]))
            ious_best.append(max(head_ious))
            ious_mean.append(fg_iou(a_dino[i, :, 0, 1:].mean(0), demo_m[i]))
        rows.append(("DINO ViT-S/16, best head", np.mean(ious_best)))
        rows.append(("DINO ViT-S/16, mean of heads", np.mean(ious_mean)))
    for name, v in rows:
        print(f"{name:34s} IoU with the pet mask (60% of attention mass): {v:.3f}")
    """),
    md(r"""
    What the numbers mean: keep the patches that carry the top 60% of a map's attention and
    measure their IoU with the pet (1 = exactly the pet). The supervised last layer scores
    lowest (its attention sits on a few background tokens); rollout and, above all, the DINO
    heads, which never saw a label, overlap the pet far better.

    ### DINOv2 patch features -> PCA -> colours

    Take the 16x16 = 256 patch features (384-D) of a few images, compute their principal
    components *jointly*, and map the top 3 to RGB. Patches of the same part (heads, legs,
    background) get the same colour across images - semantic segmentation and part
    correspondence for free. The recipe from the DINOv2 paper: (1) PCA on all patches,
    threshold the 1st component to separate foreground from background, (2) PCA again on
    the foreground patches only, top-3 components -> RGB.

    ### Exercise 3 · PCA to RGB

    `pca_rgb(feats (P, D), n=3)`: centre the features, take the top-`n` right singular
    vectors (`torch.linalg.svd(X, full_matrices=False)`), project, and min-max scale each
    component to `[0, 1]`. Visual check: the colours below should follow object parts.
    """),
    api("torch.linalg.svd", title="Functions you will need"),
    api("Tensor.min", "Tensor.max", "torch.randn", "torch.zeros", "torch.cat", brief=True),
    exercise(
        stub="""
        def pca_rgb(feats, n=3):
            '''feats (P, D): P feature vectors -> (P, n): their top-n principal components, each scaled to [0, 1].'''
            X = feats.float()
            # 1. YOUR CODE: Xc, the centred features: X minus its mean over the P vectors (X.mean(0, keepdim=True)) -> (P, D)
            # 2. YOUR CODE: _, _, Vh = torch.linalg.svd(Xc, full_matrices=False): the rows of Vh are the principal
            #               directions, the most important first
            # 3. YOUR CODE: proj, the coordinates along the top n directions: Xc @ Vh[:n].T -> (P, n)
            # 4. YOUR CODE: lo and hi, the smallest and the largest value of every component:
            #               proj.min(0, keepdim=True).values and proj.max(0, keepdim=True).values -> (1, n) each
            # 5. YOUR CODE: return (proj - lo) / (hi - lo + 1e-8): every component scaled to [0, 1]
            raise NotImplementedError

        print("shape check:", tuple(pca_rgb(torch.randn(50, 8)).shape), "(expected (50, 3))")
        """,
        solution="""
        def pca_rgb(feats, n=3):
            '''feats (P, D): P feature vectors -> (P, n): their top-n principal components, each scaled to [0, 1].'''
            X = feats.float()
            # 1. centre the features
            Xc = X - X.mean(0, keepdim=True)                   # (P, D)
            # 2. the principal directions: the rows of Vh, the most important first
            _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
            # 3. the coordinates along the top n directions
            proj = Xc @ Vh[:n].T                               # (P, n)
            # 4. the range of every component
            lo = proj.min(0, keepdim=True).values              # (1, n)
            hi = proj.max(0, keepdim=True).values              # (1, n)
            # 5. every component scaled to [0, 1]
            return (proj - lo) / (hi - lo + 1e-8)

        print("shape check:", tuple(pca_rgb(torch.randn(50, 8)).shape), "(expected (50, 3))")
        """),
    code("""
    if HAS_TIMM:
        dinov2 = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=224).eval().to(device)
        def patch_features(x):                                  # (B, 3, 224, 224) -> (B, 16, 16, 384)
            with torch.no_grad():
                t = dinov2.forward_features(x.to(device))[:, dinov2.num_prefix_tokens:]
            return t.reshape(x.shape[0], 16, 16, -1).cpu()
        FEAT_NAME = "DINOv2 ViT-S/14"
    elif HAS_VIT:
        def patch_features(x):                                  # fallback: supervised ViT-B/16 tokens, 14 x 14
            return day3.torchvision_vit_tokens(vit, x.to(device))[:, 1:].reshape(x.shape[0], 14, 14, -1).cpu()
        FEAT_NAME = "supervised ViT-B/16 (fallback)"

    pf = patch_features(demo_x[:4])
    B_, g_, _, D_ = pf.shape
    flat = pf.reshape(-1, D_)
    first = pca_rgb(flat, n=1)[:, 0]                             # 1st component, scaled to [0, 1]
    f2 = first.reshape(B_, g_, g_)
    fg = torch.zeros(B_, g_, g_, dtype=torch.bool)
    for i in range(B_):                                          # per image: split at its mean value; PCA signs are
        t_i = f2[i].mean()                                       # arbitrary, so call 'background' the side that
        border = torch.cat([f2[i, 0], f2[i, -1], f2[i, :, 0], f2[i, :, -1]]).mean()   # touches the image border
        fg[i] = f2[i] < t_i if border > t_i else f2[i] > t_i
    fg = fg.reshape(-1)
    rgb = torch.zeros(len(flat), 3)
    rgb[fg] = pca_rgb(flat[fg], n=3)
    rgb = rgb.reshape(B_, g_, g_, 3)
    fig, axes = plt.subplots(3, 4, figsize=(13, 9.6))
    for i in range(4):
        axes[0, i].imshow(demo_img[i])
        axes[1, i].imshow(first.reshape(B_, g_, g_)[i], cmap="RdBu_r")
        axes[2, i].imshow(rgb[i].numpy(), interpolation="nearest")
        for ax in axes[:, i]: ax.axis("off")
    axes[1, 0].set_title("1st principal component (fg / bg)", fontsize=9, loc="left")
    axes[2, 0].set_title("PCA of foreground patches -> RGB", fontsize=9, loc="left")
    plt.suptitle(f"{FEAT_NAME} patch features, PCA computed jointly over the 4 images", y=1.0)
    plt.tight_layout(); plt.show()
    if not HAS_TIMM:
        day3.show_asset("d3-04-dinov2-pca.png", "the same with DINOv2 features (pre-rendered)", figsize=(12, 9))
    """),
    md(r"""
    ### k-NN classification with no training at all

    If features cluster by breed, the nearest training images of a test image tell its
    breed: **k-NN** (cosine similarity, k = 10, similarity-weighted votes as in the DINO
    paper) needs no gradient step. A **linear probe** (logistic regression on frozen
    features) is the next cheapest thing. We compare DINOv2 (self-supervised) with an
    ImageNet-supervised ResNet-18, on a small balanced Pet subset (6 training images per
    breed, 4 test images per breed, 37 breeds). Features are cached in
    `data/checkpoints/d3_pet_features.pt` after the first run (~3 min on a CPU).

    > **Expected** (this tiny subset, 224 px): DINOv2 ViT-S/14 k-NN ~85%, linear probe
    > ~90%; ResNet-18 ImageNet k-NN and linear probe ~70-75%. The D1-04 fine-tuned
    > ResNet-50 (full training set) reaches ~92%. With the full Pet training set DINOv2
    > ViT-S/14 k-NN gives ~90%+ and a linear probe ~93-95% - without fine-tuning.
    """),
    api("torch.load", "torch.save", "torchvision.models.resnet18", "nn.Identity", "Tensor.item", brief=True),
    code("""
    FEATS = torchkit.CHECKPOINTS / "d3_pet_features.pt"
    from torchvision.models import resnet18, ResNet18_Weights

    if FEATS.exists():
        feats = torch.load(FEATS, weights_only=True)
        print("loaded cached features:", {k: tuple(v.shape) for k, v in feats.items()})
    else:
        tr_cls, va_cls, breeds = torchkit.get_pet_datasets(size=224, subset=(6 * 37, 4 * 37))
        tl_cls = DataLoader(tr_cls, batch_size=32)
        vl_cls = DataLoader(va_cls, batch_size=32)
        r18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
        r18.fc = nn.Identity()                                             # no classifier: 512-D features

        def resnet18_features(x):
            return r18(x.to(device))

        def dinov2_features(x):
            return dinov2(x.to(device))                                    # timm: pooled [class] token, 384-D

        extractors = {"resnet18": resnet18_features}                       # name -> function images -> features
        if HAS_TIMM:
            extractors["dinov2"] = dinov2_features
        feats = {}
        t0 = time.time()
        for name, fn in extractors.items():
            for split, loader in [("train", tl_cls), ("val", vl_cls)]:
                f, y = day3.extract_features(fn, loader)                   # features (n, D), labels (n,)
                feats[f"{name}_{split}"] = f
                feats[f"y_{split}"] = y
            print(f"{name}: features extracted ({time.time() - t0:.0f} s)")
        torch.save(feats, FEATS)

    results = {}
    for name in ["resnet18", "dinov2"]:
        if f"{name}_train" not in feats: continue
        ytr, yva = feats["y_train"], feats["y_val"]
        knn = day3.knn_predict(feats[f"{name}_train"], ytr, feats[f"{name}_val"], k=10)
        lin = day3.linear_probe(feats[f"{name}_train"], ytr, feats[f"{name}_val"])
        results[name] = ((knn == yva).float().mean().item(), (lin == yva).float().mean().item())
        print(f"{name:9s} k-NN {results[name][0]:.1%}   linear probe {results[name][1]:.1%}   ({len(ytr)} train / {len(yva)} test images)")
    """),
    code("""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    labels_, vals, cols = [], [], []
    for name, pretty, col in [("resnet18", "ResNet-18 (ImageNet labels)", "#BAB0AC"), ("dinov2", "DINOv2 ViT-S/14 (no labels)", "#4C78A8")]:
        if name in results:
            labels_ += [f"{pretty}\\nk-NN", f"{pretty}\\nlinear probe"]; vals += list(results[name]); cols += [col, col]
    labels_.append("D1-04 ResNet-50\\nfine-tuned (reported)"); vals.append(0.92); cols.append("#F58518")
    bars = ax.bar(range(len(vals)), vals, color=cols)
    for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels_, fontsize=8); ax.set_ylim(0, 1.05); ax.set_ylabel("Pet test accuracy (37 breeds)")
    ax.set_title("frozen features, 6 labelled images per breed")
    plt.tight_layout(); plt.show()
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · CLIP: classification without training

    k-NN and the linear probe still needed a few labelled images per breed. CLIP needs none:
    the class names themselves become the classifier.

    **CLIP** (Radford et al., 2021) trains an image encoder and a text encoder together on
    400M (image, caption) pairs from the web, with a contrastive loss: in a batch of $N$
    pairs, the $N$ matching (image, text) embeddings must have a higher cosine similarity
    than the $N^2 - N$ mismatched ones. The result is a **shared embedding space**. To
    classify, write one prompt per class, embed the prompts, and pick the class whose text
    embedding is closest to the image embedding. **SigLIP** (2023) replaces the softmax
    over the batch by a pairwise sigmoid loss (better at small batch sizes); SigLIP 2 and
    the CLIP-like towers inside VLMs are today's default.
    """),
    code("""
    fig, ax = plt.subplots(figsize=(15, 4.6))
    day3.draw_clip(ax)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Lab · CLIP zero-shot geometry
    **Predict first:** an image embedding has cosine similarity 0.998 with "a photo of a beagle"
    and 0.970 with "a photo of a pug". With CLIP's logit scale of 100, is the prediction about
    50%, about 75% or over 90% sure? Check in the lab (a 2-D toy of the embedding space), then
    work through its "Try this" list (5 minutes).
    """),
    code("""
    labs.show("clip-zero-shot")
    """),
    md(r"""
    ### Exercise 4 · `zero_shot_classify`

    `image_feats (N, D)` and `text_feats (C, D)` come straight out of the two encoders and
    are **not** normalised (the lab's "raw, no norm" setting showed what goes wrong then).
    L2-normalise both, compute the cosine similarities, scale by `logit_scale` (CLIP learns
    ~100) and softmax over classes. Return `(pred, probs)`: `pred (N,)` class indices and
    `probs (N, C)`, each row summing to 1.
    """),
    api("F.normalize", "torch.as_tensor", "Tensor.softmax", "Tensor.argmax", brief=True, title="Functions you will need"),
    exercise(
        stub="""
        def zero_shot_classify(image_feats, text_feats, logit_scale=100.0):
            '''image_feats (N, D), text_feats (C, D), not normalised -> (pred (N,) class indices, probs (N, C), rows sum to 1).'''
            # 1. both as float tensors
            img = torch.as_tensor(image_feats).float()         # (N, D)
            txt = torch.as_tensor(text_feats).float()          # (C, D)
            # 2. YOUR CODE: L2-normalise both along the last axis (F.normalize(..., dim=-1)): unit vectors
            # 3. YOUR CODE: logits, the scaled cosine similarity of every image with every class:
            #               (logit_scale * img) @ txt.T -> (N, C)
            # 4. YOUR CODE: probs, the softmax of logits over the classes (dim=-1) -> (N, C)
            # 5. YOUR CODE: pred, the most likely class of every image (argmax over dim=-1); return pred, probs
            raise NotImplementedError

        tests.check_zero_shot(zero_shot_classify)
        """,
        solution="""
        def zero_shot_classify(image_feats, text_feats, logit_scale=100.0):
            '''image_feats (N, D), text_feats (C, D), not normalised -> (pred (N,) class indices, probs (N, C), rows sum to 1).'''
            # 1. both as float tensors
            img = torch.as_tensor(image_feats).float()         # (N, D)
            txt = torch.as_tensor(text_feats).float()          # (C, D)
            # 2. unit vectors: the dot product of two of them is their cosine similarity
            img = F.normalize(img, dim=-1)
            txt = F.normalize(txt, dim=-1)
            # 3. the scaled cosine similarity of every image with every class
            logits = (logit_scale * img) @ txt.T               # (N, C)
            # 4. probabilities over the classes
            probs = logits.softmax(dim=-1)                     # (N, C)
            # 5. the most likely class
            pred = probs.argmax(dim=-1)                        # (N,)
            return pred, probs

        tests.check_zero_shot(zero_shot_classify)
        """),
    md(r"""
    ### Prompt engineering, offline

    CLIP is sensitive to the wording: "a photo of a {breed}, a type of pet." beats the bare
    breed name by several points on Oxford Pets, and averaging the embeddings of many
    templates (**prompt ensembling**) adds more. We study the effect in a *toy embedding
    space* (`day3.toy_clip_space`): every breed has a hidden concept vector; images are noisy
    versions of it; each prompt template drifts the text embedding away from the concept
    by a template-specific amount, on top of an error all wordings share. Same maths as
    CLIP, no download. (In the toy space the templates' errors are independent, so
    ensembling gains ~10 points; with real CLIP, where templates are correlated, expect
    +1-5 points.)

    ### Exercise 5 · prompt ensembling

    `text (T, C, D)` holds the embeddings of `T` templates for `C` classes. Normalise each
    embedding, average over the templates, normalise again -> `(C, D)`. (The lab's "prompt
    ensembling" toggle does exactly this with 4 wordings per class in 2-D.)
    """),
    exercise(
        stub="""
        def prompt_ensemble(text):
            '''text (T, C, D): the embeddings of T templates for C classes -> (C, D): one unit vector per class.'''
            # 1. YOUR CODE: unit, every embedding L2-normalised (F.normalize(text, dim=-1)) -> (T, C, D)
            # 2. YOUR CODE: mean, the average over the T templates (.mean(0)) -> (C, D)
            # 3. YOUR CODE: return mean normalised again -> (C, D)
            raise NotImplementedError

        tests.check_prompt_ensemble(prompt_ensemble)
        """,
        solution="""
        def prompt_ensemble(text):
            '''text (T, C, D): the embeddings of T templates for C classes -> (C, D): one unit vector per class.'''
            # 1. every embedding as a unit vector, so that every template has the same say
            unit = F.normalize(text, dim=-1)                   # (T, C, D)
            # 2. the average over the templates
            mean = unit.mean(0)                                # (C, D)
            # 3. back to unit length
            return F.normalize(mean, dim=-1)                   # (C, D)

        tests.check_prompt_ensemble(prompt_ensemble)
        """),
    api("np.argsort", "np.linspace", "np.array", brief=True),
    code("""
    templates = ["{}", "a photo of a {}.", "a photo of a {}, a type of pet.", "a blurry photo of a {}.",
                 "a close-up photo of a {}.", "a drawing of a {}.", "itap of a {}.", "a low resolution photo of a {}."]
    drift = [1.6, 1.0, 0.75, 1.15, 1.0, 1.4, 1.2, 1.25]          # how far each wording drifts from the concept
    toy = day3.toy_clip_space(n_classes=37, dim=64, n_per_class=20, image_noise=1.4, template_noise=drift, shared_noise=1.0, seed=0)
    accs = []
    for t_, name in enumerate(templates):
        pred, _ = zero_shot_classify(toy["img"], toy["text"][t_])
        accs.append((pred == toy["labels"]).float().mean().item())
    pred_e, probs_e = zero_shot_classify(toy["img"], prompt_ensemble(toy["text"]))
    acc_e = (pred_e == toy["labels"]).float().mean().item()

    print(f"{'prompt template':38s} zero-shot accuracy (toy space, 37 classes)")
    best_first = np.argsort(-np.array(accs), kind="stable")             # template indices, highest accuracy first
    for t_ in best_first:
        print(f"{templates[t_]:38s} {accs[t_]:6.1%}")
    print(f"{'ENSEMBLE of all 8 templates':38s} {acc_e:6.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
    order = np.argsort(accs)
    axes[0].barh([templates[i] for i in order] + ["ensemble (mean of 8)"], [accs[i] for i in order] + [acc_e],
                 color=["#BAB0AC"] * len(order) + ["#4C78A8"])
    axes[0].set_xlabel("accuracy"); axes[0].set_title("prompt wording matters; ensembling beats the best single prompt")
    noise = np.linspace(0.2, 2.5, 12)
    curve = []
    for nz in noise:
        tt = day3.toy_clip_space(37, 64, 20, image_noise=nz, template_noise=drift, shared_noise=1.0, seed=1)
        pred_bare, _ = zero_shot_classify(tt["img"], tt["text"][0])                    # the bare class name
        pred_ens, _ = zero_shot_classify(tt["img"], prompt_ensemble(tt["text"]))      # all 8 templates
        acc_bare = (pred_bare == tt["labels"]).float().mean().item()
        acc_ens = (pred_ens == tt["labels"]).float().mean().item()
        curve.append((acc_bare, acc_ens))
    curve = np.array(curve)
    axes[1].plot(noise, curve[:, 0], "o-", label="bare class name '{}'"); axes[1].plot(noise, curve[:, 1], "o-", label="prompt ensemble")
    axes[1].set_xlabel("image noise (how far photos are from the concept)"); axes[1].set_ylabel("accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Real CLIP on Oxford Pets (needs `open_clip_torch` or `transformers`)

    Published zero-shot numbers on Oxford-IIIT Pet (Radford et al., 2021): roughly **~87%**
    for CLIP ViT-B/32 and **~93%** for ViT-L/14 with the prompt "a photo of a {}, a type of
    pet." - comparable to the D1-04 fine-tuned ResNet-50 (~92%) *without a single Pet
    training image*. The cell below reproduces it on 148 test images if a CLIP package is
    installed.
    """),
    api("open_clip.create_model_and_transforms", "open_clip.get_tokenizer", "transformers.CLIPModel.from_pretrained",
        "transformers.CLIPProcessor.from_pretrained"),
    api("torch.tensor", "torch.stack", brief=True),
    code("""
    clip_backend = None
    try:
        import open_clip
        clip_backend = "open_clip"
    except ImportError:
        try:
            from transformers import CLIPModel, CLIPProcessor
            clip_backend = "transformers"
        except ImportError:
            pass

    if clip_backend is None:
        print("No CLIP package installed (pip install open_clip_torch  or  pip install transformers).\\n"
              "Skipping the real-CLIP experiment; the toy experiment above shows the same mechanics.")
    else:
        from PIL import Image
        _, va_cls, breeds = torchkit.get_pet_datasets(size=224, subset=(37, 4 * 37))
        names = []                                               # "basset_hound" -> "basset hound"
        for b in breeds:
            names.append(b.replace("_", " "))
        pil = []                                                 # the test images, as PIL images
        labels_true = []                                         # and their breed indices
        for i in range(len(va_cls)):
            pil.append(Image.open(va_cls.image_path(i)).convert("RGB"))
            labels_true.append(va_cls.label(i))
        y_true = torch.tensor(labels_true)
        with torch.no_grad():
            if clip_backend == "open_clip":
                cm_, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
                tok = open_clip.get_tokenizer("ViT-B-32")
                cm_.eval()
                batch_feats = []                                 # image embeddings, 32 images at a time
                for i in range(0, len(pil), 32):
                    batch = []
                    for p in pil[i:i + 32]:
                        batch.append(prep(p))                    # (3, 224, 224) preprocessed image
                    batch_feats.append(cm_.encode_image(torch.stack(batch)))
                img_f = torch.cat(batch_feats)                   # (number of images, D)

                def enc_text(prompts):                           # list of C strings -> (C, D)
                    return cm_.encode_text(tok(prompts))
            else:
                cm_ = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
                proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                batch_feats = []                                 # image embeddings, 32 images at a time
                for i in range(0, len(pil), 32):
                    inputs = proc(images=pil[i:i + 32], return_tensors="pt")
                    batch_feats.append(cm_.get_image_features(**inputs))
                img_f = torch.cat(batch_feats)                   # (number of images, D)

                def enc_text(prompts):                           # list of C strings -> (C, D)
                    inputs = proc(text=prompts, return_tensors="pt", padding=True)
                    return cm_.get_text_features(**inputs)
            text_per_template = []                               # one (37, D) block per template
            for t in templates:
                prompts = []
                for n in names:
                    prompts.append(t.format(n))                  # e.g. "a photo of a beagle."
                text_per_template.append(enc_text(prompts))
            text_all = torch.stack(text_per_template)            # (T, 37, D)
        print(f"real CLIP ({clip_backend}) zero-shot on {len(pil)} Pet test images:")
        for t_, name in enumerate(templates):
            print(f"  {name:38s} {(zero_shot_classify(img_f, text_all[t_])[0] == y_true).float().mean():6.1%}")
        print(f"  {'ENSEMBLE':38s} {(zero_shot_classify(img_f, prompt_ensemble(text_all))[0] == y_true).float().mean():6.1%}")
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · SAM: segment anything

    **SAM** (Kirillov et al., 2023) is a promptable, class-agnostic segmenter: a heavy ViT
    image encoder runs once per image; a tiny prompt encoder + mask decoder turns each
    prompt (a click, a box, a rough mask) into 3 candidate masks with a predicted IoU, in
    milliseconds. It was trained on SA-1B (1.1 billion masks) collected with a *data engine*:
    the model helped annotators, the annotations retrained the model. **SAM 2** (2024) adds a
    memory to track masks through video; **SAM 3** (2025) accepts short noun phrases
    ("striped cat") and returns all matching instances. **Grounding DINO + SAM**: an
    open-vocabulary detector turns text into boxes, SAM turns boxes into masks.
    """),
    api("transformers.SamModel.from_pretrained", "transformers.SamProcessor.from_pretrained"),
    api("np.nonzero", "np.median", "np.ma.masked_where", brief=True),
    code("""
    fig, ax = plt.subplots(figsize=(15, 4.6))
    day3.draw_sam(ax)
    plt.tight_layout(); plt.show()
    """),
    code("""
    sam_backend = None
    try:
        from transformers import SamModel, SamProcessor
        sam_backend = "transformers"
    except ImportError:
        pass

    if sam_backend is None:
        print("transformers is not installed (pip install transformers): skipping real SAM.\\n"
              "Below: a 10-line imitation of point prompts with DINOv2 features, to see the idea.")
    else:
        from PIL import Image
        sam = SamModel.from_pretrained("facebook/sam-vit-base").eval()
        sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
        for i, ax in enumerate(axes):
            im = (demo_img[i] * 255).astype(np.uint8)
            ys, xs = np.nonzero(demo_m[i].numpy() == 1)                 # the pixels of the pet
            cy = int(np.median(ys))                                     # a simulated click in its middle
            cx = int(np.median(xs))
            inputs = sam_proc(Image.fromarray(im), input_points=[[[cx, cy]]], return_tensors="pt")
            with torch.no_grad():
                out = sam(**inputs)
            masks = sam_proc.image_processor.post_process_masks(out.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])[0][0]
            best = int(out.iou_scores[0, 0].argmax())
            ax.imshow(im); ax.imshow(np.ma.masked_where(~masks[best].numpy(), masks[best].numpy()), alpha=0.5, cmap="autumn")
            ax.plot(cx, cy, "*", ms=16, color="cyan", mec="k"); ax.axis("off")
            ax.set_title(f"SAM, 1 click: predicted IoU {out.iou_scores[0, 0, best]:.2f}", fontsize=9)
        plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### A poor man's promptable segmenter (not SAM!)

    To feel what "a prompt turns into a mask" means without downloading SAM: take the
    patch features of the image (DINOv2 if available), take the feature of the **clicked**
    patch, and keep every patch whose cosine similarity to it is high; with a **box**
    prompt, also discard everything outside the box. No training, ten lines. It works
    because self-supervised features group object parts together; it is far from SAM's
    quality (16x16 resolution, no boundary refinement, one object type at a time; the
    similarity threshold 0.2 was picked by eye on a few images).
    """),
    api("np.clip", brief=True),
    code("""
    def point_prompt_mask(feats, click_rc, box=None, thr=0.2, out_size=224):
        \"\"\"feats (g, g, D); click_rc = (row, col) in pixels; box = (x1, y1, x2, y2) or None.\"\"\"
        g = feats.shape[0]                                       # patches per side
        # 1. every patch feature as a unit vector
        f = feats.reshape(-1, feats.shape[-1])                   # (g*g, D)
        f = F.normalize(f, dim=1)
        # 2. the patch under the click
        r = int(click_rc[0] * g / out_size)
        c = int(click_rc[1] * g / out_size)
        # 3. cosine similarity of every patch to the clicked one, enlarged to the image size
        sim = f @ f[r * g + c]                                   # (g*g,)
        sim = sim.reshape(1, 1, g, g)
        sim = F.interpolate(sim, size=(out_size, out_size), mode="bilinear", align_corners=False)[0, 0]
        # 4. the mask: every pixel similar enough to the click
        mask = sim > thr
        # 5. a box prompt also removes everything outside the box
        if box is not None:
            x1 = int(np.clip(box[0], 0, out_size))
            y1 = int(np.clip(box[1], 0, out_size))
            x2 = int(np.clip(box[2], 0, out_size))
            y2 = int(np.clip(box[3], 0, out_size))
            inside = torch.zeros_like(mask)
            inside[y1:y2, x1:x2] = True
            mask &= inside
        return mask.numpy(), sim.numpy()

    def iou_with(m, gt):
        return (m & gt).sum() / max((m | gt).sum(), 1)

    pfeat = patch_features(demo_x[:4])
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    for i in range(4):
        gt = demo_m[i].numpy() > 0
        ys, xs = np.nonzero(demo_m[i].numpy() == 1)                 # the pixels of the pet
        click = (int(np.median(ys)), int(np.median(xs)))            # a simulated user click in its middle
        bx = (xs.min() - 5, ys.min() - 5, xs.max() + 5, ys.max() + 5)
        m_pt, sim = point_prompt_mask(pfeat[i], click)
        m_box, _ = point_prompt_mask(pfeat[i], click, box=bx)
        axes[0, i].imshow(demo_img[i]); axes[0, i].imshow(sim, cmap="inferno", alpha=0.55)
        axes[0, i].plot(click[1], click[0], "*", ms=16, color="cyan", mec="k"); axes[0, i].set_title("similarity to the clicked patch", fontsize=9)
        axes[1, i].imshow(demo_img[i]); axes[1, i].imshow(np.ma.masked_where(~m_pt, m_pt), cmap="autumn", alpha=0.55)
        axes[1, i].plot(click[1], click[0], "*", ms=16, color="cyan", mec="k"); axes[1, i].set_title(f"point prompt: IoU {iou_with(m_pt, gt):.2f}", fontsize=9)
        axes[2, i].imshow(demo_img[i]); axes[2, i].imshow(np.ma.masked_where(~m_box, m_box), cmap="autumn", alpha=0.55)
        axes[2, i].add_patch(Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, ec="cyan", lw=2))
        axes[2, i].set_title(f"point + box prompt: IoU {iou_with(m_box, gt):.2f}", fontsize=9)
        for ax in axes[:, i]: ax.axis("off")
    plt.suptitle(f"prompt -> mask with {FEAT_NAME} patch features (no training; SAM does this at pixel precision)", y=1.0)
    plt.tight_layout(); plt.show()
    """),

    # ------------------------------------------------------------------ 5
    md(r"""
    ## 5 · The map of the field (2026)

    Where each task stands, and what to try first on a new problem. The default answer is
    almost never "train from scratch": start from the strongest pretrained model that fits
    your latency budget, measure, then fine-tune.
    """),
    code("""
    fig, ax = plt.subplots(figsize=(16, 8.2))
    day3.draw_field_map(ax)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    **Where to go next**

    * **3D**: NeRF and 3D Gaussian splatting (novel views from photos); DUSt3R / VGGT (3D
      reconstruction from image pairs with one network); Depth Anything for monocular depth.
    * **Video**: SAM 2 tracking, video transformers (VideoMAE, V-JEPA), action recognition.
    * **Multimodal LLMs**: vision-language models answer questions about images, read
      documents, ground objects by text; for many "classify / extract" tasks a VLM API is
      the first baseline, a distilled small model the deployed one.
    * **Data engines**: label with foundation models (Grounding DINO + SAM, VLMs), have
      humans verify, train a small task model, find its failures, repeat. The data, not the
      architecture, is usually the bottleneck.

    ## 6 · Capstone: sketch your pipeline

    Pick a task you care about (classification, detection or segmentation) and a dataset,
    and fill in the template. Be concrete: numbers, model names, metric definitions. You
    will present it in 3 minutes during the capstone discussion.
    """),
    code("""
    capstone = {
        "problem": "e.g. segment weeds vs crop in drone images of a field, 4K tiles",
        "task type": "semantic / instance / detection / classification / retrieval ...",
        "data": "how many images, how labelled, label format, train/val/test split rule (by field? by date?)",
        "first baseline (no training)": "e.g. Grounding DINO + SAM, CLIP zero-shot, DINOv2 k-NN",
        "backbone": "e.g. ConvNeXt-T / DINOv2 ViT-S / ResNet-50, pretrained on ...",
        "head": "e.g. U-Net decoder, Mask R-CNN heads, linear probe",
        "loss": "e.g. CE + Dice with class weights; focal; Hungarian set loss",
        "metric (single number) + satisficing metrics": "e.g. mIoU on the test fields; latency < 50 ms on a Jetson",
        "augmentation": "e.g. flips, rotations (aerial: any angle), colour jitter, mosaic",
        "schedule": "e.g. AdamW, lr 1e-4 (encoder) / 1e-3 (decoder), 40 epochs cosine, warm-up 1 epoch",
        "evaluation protocol": "e.g. 5 held-out fields, report per-class IoU + error maps, failure gallery",
        "deployment constraint": "e.g. sliding windows on 4K tiles, ONNX + INT8, batch 1",
        "biggest risk and how you would detect it": "e.g. domain shift between seasons -> monitor prediction histograms",
    }
    width = max(map(len, capstone))
    for k, v in capstone.items():
        print(f"{k:>{width}} : {v}")
    """),
    md(r"""
    ## 7 · Summary

    * A ViT is a transformer on a sequence of patch embeddings; patchify + linear == a stride-16 16x16 convolution.
    * Attention = softmax(QK^T / sqrt(d)) V: content-dependent, global mixing; no locality prior, so ViTs need more data (or pretraining).
    * Self-supervised ViTs (DINO, DINOv2) learn features that segment objects and cluster classes without labels: PCA, k-NN and linear probes are strong baselines.
    * CLIP / SigLIP put images and text in one space: zero-shot classification via prompts; wording and ensembling matter.
    * SAM turns prompts (points, boxes, text in SAM 3) into masks; combined with an open-vocabulary detector it segments anything you can name.
    * For a new problem: start from a foundation model baseline, measure, then fine-tune the smallest model that meets the requirement.

    ### Self-check (answers in the Day 3 quiz)
    1. How many tokens does a ViT-B/16 process for a 384x384 image? What happens to the position embeddings?
    2. Write the shapes of Q, K, V and of the attention matrix for 197 tokens, 12 heads, D = 768.
    3. Why might a ViT trained from scratch on 5 000 images lose to a ResNet-50?
    4. CLIP zero-shot accuracy jumps from 80% to 86% when you replace "{}" by "a photo of a {}, a type of pet." Why?
    5. SAM returns three masks for one click. Why three, and how do you choose?

    ### Go deeper
    * Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT, ICLR 2021); Touvron et al., *DeiT* (2021).
    * Liu et al., *Swin Transformer* (ICCV 2021); Liu et al., *A ConvNet for the 2020s* (ConvNeXt, CVPR 2022).
    * Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO, ICCV 2021).
    * Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* (2023); Darcet et al., *Vision Transformers Need Registers* (2024).
    * Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP, 2021); Zhai et al., *Sigmoid Loss for Language Image Pre-Training* (SigLIP, 2023).
    * Kirillov et al., *Segment Anything* (2023); Ravi et al., *SAM 2: Segment Anything in Images and Videos* (2024).
    * Liu et al., *Grounding DINO* (2023); Abnar & Zuidema, *Quantifying Attention Flow in Transformers* (attention rollout, 2020).
    """),
]
