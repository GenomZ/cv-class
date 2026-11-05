from nb import md, code, exercise, header

NAME = "D1-04-transfer-learning"

CELLS = [
    header(
        "D1-04 · Transfer learning and what the network sees",
        steps=[
            "Model zoos: torchvision's weights API, timm, Hugging Face",
            "Regime A: cache features once, train a linear probe",
            "Regimes B and C: frozen backbone + head with augmentation; fine-tuning with discriminative learning rates",
            "How much data do you need? Accuracy vs. training-set size",
            "What the network sees: first-layer filters, feature maps, the feature space",
            "Grad-CAM with hooks, and a check for shortcut learning",
            "Choosing and reading metrics (ML Yearning): a mini case study on Pet",
            "Modern note: self-supervised features (DINOv2, optional)",
        ],
        topics=[
            "Reuse an ImageNet backbone: preprocessing must match, the head must be replaced",
            "Linear probe vs. frozen backbone vs. fine-tuning: cost, accuracy, when to use which",
            "Freezing, BatchNorm in frozen layers, discriminative learning rates",
            "Accuracy grows roughly linearly in log(data)",
            "Forward / backward hooks; Grad-CAM; sanity checks for shortcut learning",
            "Single-number metric, satisficing metrics, dev / test from the same distribution",
        ],
        minutes=90,
        prereq="D1-02 (training recipe) and D1-03 (ResNet). The Pet subset from D1-02.",
    ),
    md(r"""
    ## 0 · Setup

    The course standard backbone is **ResNet-50** at 224 px on a GPU. On a CPU-only laptop the
    same cells use **ResNet-18 at 160 px** (about 8x cheaper) so the whole notebook runs in a few
    minutes; every number in the text is given for both. Expensive results (features, fine-tuned
    weights) are cached in `data/checkpoints/` and reused when you re-run the notebook.
    """),
    code(r"""
    import os, sys, pathlib
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")   # before importing torch: idle CPU threads sleep, not spin
    for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:   # find the cvcourse package
        if (p / "cvcourse").is_dir():
            sys.path.insert(0, str(p)); break

    import math, time
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader
    import torchvision
    from torchvision.transforms import v2
    from torchvision.models import get_model, get_model_weights
    from cvcourse import viz, tests, torchkit, day1

    %matplotlib inline
    plt.rcParams["figure.dpi"] = 90
    torch.manual_seed(0)
    device = torchkit.get_device()
    if device.type == "cpu":
        day1.pick_cpu_threads()     # on a busy CPU, 1 thread can be faster than 8

    GPU = device.type == "cuda"
    BACKBONE = "resnet50" if GPU else "resnet18"
    IMG = 224 if GPU else 160
    N_FT = 2000                        # images used for the augmented regimes B and C (all of the subset)
    FT_EPOCHS = 3 if GPU else 1
    print(f"backbone {BACKBONE} at {IMG} px; regimes B / C on {N_FT} images x {FT_EPOCHS} epoch(s)")
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · Model zoos

    Somebody already spent thousands of GPU-hours training a backbone on ImageNet (1.3 M images,
    1 000 classes). Its early layers detect edges, colours and textures, its late layers object
    parts: features that are useful for *any* natural-image task. Three places to get one:

    | library | one-liner | notes |
    |---|---|---|
    | torchvision | `get_model("resnet50", weights="IMAGENET1K_V2")` | ~80 classic models; weights enum carries preprocessing + metadata |
    | timm | `timm.create_model("convnext_tiny.fb_in22k", pretrained=True, num_classes=37)` | 1 000+ models incl. the latest; replaces the head for you |
    | Hugging Face | `AutoModel.from_pretrained("facebook/dinov2-small")` | transformers, multimodal (CLIP, SAM), Day 3 |

    **The preprocessing must match the backbone**: same resize / crop, same mean / std. The weights
    enum tells you exactly what it was trained with:
    """),
    code(r"""
    weights = get_model_weights(BACKBONE).DEFAULT
    print(repr(weights))
    print("preprocessing:", weights.transforms())
    rows = []
    for name in ["resnet18", "resnet50", "mobilenet_v3_large", "efficientnet_b0", "convnext_tiny", "vit_b_16"]:
        w = get_model_weights(name).DEFAULT                  # metadata only: nothing is downloaded here
        rows.append((name, str(w).split(".")[-1], w.meta["_metrics"]["ImageNet-1K"]["acc@1"], w.meta["num_params"] / 1e6,
                     w.meta.get("_ops", float("nan"))))
    print(f"\n{'model':<20}{'weights':<16}{'top-1':>7}{'params (M)':>12}{'GFLOPs':>8}")
    for r in rows: print(f"{r[0]:<20}{r[1]:<16}{r[2]:>7.1f}{r[3]:>12.1f}{r[4]:>8.2f}")
    """),
    md(r"""
    The ImageNet classifier *already* knows a lot about pets: 25 of the 37 Pet breeds are
    ImageNet classes. Before any training, here is what it says about a few validation images:
    """),
    code(r"""
    data = day1.pet_tensors(176)                    # same 2 000 / 500 subset as D1-02, decoded at 176 px
    classes = data["classes"]
    short = [c if len(c) <= 16 else c[:15] + "." for c in classes]
    eval_tf = day1.eval_transform(IMG)              # resize + centre crop + ImageNet mean / std

    imagenet_model = get_model(BACKBONE, weights=weights).eval().to(device)
    cats = weights.meta["categories"]
    idx = [3, 60, 130, 222, 301, 444]
    with torch.no_grad():
        probs = imagenet_model(torch.stack([eval_tf(data["x_val"][i]) for i in idx]).to(device)).softmax(1).cpu()
    fig, axes = plt.subplots(1, len(idx), figsize=(15, 3.6))
    for ax, i, p in zip(axes, idx, probs):
        ax.imshow(torchkit.denormalize(eval_tf(data["x_val"][i]))); ax.axis("off")
        top = torch.topk(p, 3)
        ax.set_title(f"Pet: {short[data['y_val'][i]]}\n" + "\n".join(f"{v:.2f} {cats[int(k)][:18]}" for v, k in zip(top.values, top.indices)),
                     fontsize=8, loc="left")
    plt.suptitle("Zero training: the ImageNet head's top-3 guesses", y=1.03); plt.tight_layout(); plt.show()
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · Regime A: the linear probe

    Cheapest possible transfer: chop off the ImageNet head (`model.fc = nn.Identity()`), run every
    image through the frozen backbone **once**, store the 512-d (ResNet-18) or 2048-d (ResNet-50)
    feature vectors, and train a logistic-regression head on them. Training takes seconds because
    the backbone is never run again - the price is no data augmentation.

    ### Exercise 1 · `extract_features`

    Run `backbone` over `loader` and return `(features (N, D), labels (N,))` on the CPU. Two
    details matter: **`backbone.eval()`** (BatchNorm must use its running statistics, dropout must
    be off) and **`torch.no_grad()`** (no autograd graph: faster, far less memory).
    """),
    exercise(
        stub=r"""
        def extract_features(backbone, loader, device=device):
            # YOUR CODE HERE
            raise NotImplementedError

        ok_feat = tests.check_extract_features(extract_features)
        """,
        solution=r"""
        def extract_features(backbone, loader, device=device):
            backbone.eval()
            feats, labels = [], []
            with torch.no_grad():
                for x, y in loader:
                    feats.append(backbone(x.to(device)).flatten(1).cpu())
                    labels.append(y)
            return torch.cat(feats), torch.cat(labels)

        ok_feat = tests.check_extract_features(extract_features)
        """),
    code(r"""
    extract = tests.use_or_reference(ok_feat, globals().get("extract_features"), day1.extract_features, "extract_features")
    backbone = get_model(BACKBONE, weights=weights)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()                     # the network now outputs the pooled features
    backbone = backbone.to(device)

    feat_path = torchkit.CHECKPOINTS / f"d1-04_feats_{BACKBONE}_{IMG}.pt"
    if feat_path.exists():
        F_ = torch.load(feat_path)
        print(f"features loaded from {feat_path.name}")
    else:
        t0 = time.time(); F_ = {}
        for split in ("train", "val"):
            dl = DataLoader(day1.TensorImageDataset(data[f"x_{split}"], data[f"y_{split}"], eval_tf), batch_size=50)
            F_[f"f_{split}"], F_[f"y_{split}"] = extract(backbone, dl)
        torch.save(F_, feat_path)
        print(f"features extracted in {time.time() - t0:.0f} s and cached to {feat_path.name}")
    print("train features", tuple(F_["f_train"].shape), "| val features", tuple(F_["f_val"].shape))
    """),
    code(r"""
    t0 = time.time()
    probe_head, acc_a, _ = day1.train_linear_probe(F_["f_train"], F_["y_train"], F_["f_val"], F_["y_val"], epochs=100)
    print(f"Regime A - linear probe on cached features: val accuracy {acc_a:.3f}   ({time.time() - t0:.1f} s of training)")
    """),
    md(r"""
    About **79 %** on 37 breeds with ResNet-18 at 160 px, after seconds of training on a CPU
    (ResNet-50 at 224 px: **~88-90 %**). The from-scratch PetNet of D1-02 reached ~10-30 % on the
    same images. That gap is the whole argument for transfer learning on small data.

    ## 3 · Regimes B and C: training through the backbone

    | regime | trainable | augmentation | cost per epoch | use when |
    |---|---|---|---|---|
    | A linear probe | head only, on cached features | no | ~0 (features once) | tiny data, quick baseline, many experiments |
    | B frozen backbone + head | head | yes | 1 forward pass of the backbone | small data, a domain close to ImageNet |
    | C fine-tune | last stages + head (or all) | yes | forward + backward | more data, or a domain far from ImageNet (medical, satellite) |

    Practical rules for C: **train the head first** (a random head sends large, random gradients
    into good features - we start C from the probe's head), use a **small LR** (1e-4 - 1e-3 with
    AdamW, about 10x below training from scratch) and **discriminative LRs** (earlier = more
    generic = smaller LR), keep **BatchNorm in eval mode** (with small data and small batches the
    ImageNet running statistics are better than re-estimated ones), and unfreeze from the top.
    """),
    code(r"""
    stages = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]
    regimes = {"A  linear probe": [0, 0, 0, 0, 0, 1], "B  frozen + head (aug)": [0, 0, 0, 0, 0, 1],
               "C  fine-tune top stages": [0, 0, 0, 1, 1, 1], "C' full fine-tune": [1, 1, 1, 1, 1, 1]}
    lr_text = ["3e-4 x 0.3^5", "x 0.3^4", "x 0.3^3", "x 0.3^2", "x 0.3", "3e-4"]
    fig, ax = plt.subplots(figsize=(12, 3.6)); ax.axis("off"); ax.set_xlim(-2.6, 6.2); ax.set_ylim(-1.0, 4.3)
    for r, (name, train) in enumerate(regimes.items()):
        y = 3.4 - r
        ax.text(-2.5, y, name, va="center", fontsize=10)
        for k, (s, t) in enumerate(zip(stages, train)):
            fc = "#f8d0b8" if t else "#d9e3ef"
            ax.add_patch(FancyBboxPatch((k - 0.42, y - 0.3), 0.84, 0.6, boxstyle="round,pad=0.02", fc=fc, ec="k", lw=0.8))
            ax.text(k, y, s, ha="center", va="center", fontsize=8)
    for k, t in enumerate(lr_text): ax.text(k, -0.55, t, ha="center", fontsize=8, color=viz.PALETTE[3])
    ax.text(-2.5, -0.55, "LR in C' (discriminative)", fontsize=8, color=viz.PALETTE[3])
    ax.text(5.6, 3.4, "orange = trainable\nblue = frozen", fontsize=8, va="center")
    ax.set_title("Which layers to unfreeze: generic early features stay, task-specific late features adapt")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Exercise 2 · `freeze(model, until)`

    For a torchvision ResNet, set `requires_grad = False` for every parameter of the stages up to
    and including `until` and `True` for the later ones. Stage of a parameter = first part of its
    name, with `conv1` / `bn1` meaning `"stem"`. Order: `stem, layer1, layer2, layer3, layer4, fc`.
    Return the model.
    """),
    exercise(
        stub=r"""
        ORDER = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]

        def stage_of(param_name):
            top = param_name.split(".")[0]
            return "stem" if top in ("conv1", "bn1") else top

        def freeze(model, until="layer3"):
            # YOUR CODE HERE
            raise NotImplementedError

        ok_freeze = tests.check_frozen(freeze)
        """,
        solution=r"""
        ORDER = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]

        def stage_of(param_name):
            top = param_name.split(".")[0]
            return "stem" if top in ("conv1", "bn1") else top

        def freeze(model, until="layer3"):
            cut = ORDER.index(until)
            for name, p in model.named_parameters():
                p.requires_grad = ORDER.index(stage_of(name)) > cut
            return model

        ok_freeze = tests.check_frozen(freeze)
        """),
    md(r"""
    ### Exercise 3 · `param_groups_with_lrs(model, base_lr, decay)`

    Return a list of optimiser parameter groups, one per stage that has trainable parameters, with
    `lr = base_lr * decay ** (5 - i)` where `i` is the stage index in `ORDER` (so `fc` gets
    `base_lr`, `layer4` `base_lr * decay`, ...). Skip frozen parameters and empty groups. Each
    group is a dict `{"params": [...], "lr": ..., "name": stage}` - `torch.optim.AdamW` accepts
    such a list directly.
    """),
    exercise(
        stub=r"""
        def param_groups_with_lrs(model, base_lr, decay=0.3):
            # YOUR CODE HERE
            raise NotImplementedError

        ok_groups = tests.check_param_groups(param_groups_with_lrs)
        """,
        solution=r"""
        def param_groups_with_lrs(model, base_lr, decay=0.3):
            groups = {s: [] for s in ORDER}
            for name, p in model.named_parameters():
                if p.requires_grad:
                    groups[stage_of(name)].append(p)
            return [{"params": ps, "lr": base_lr * decay ** (len(ORDER) - 1 - i), "name": s}
                    for i, (s, ps) in enumerate(groups.items()) if ps]

        ok_groups = tests.check_param_groups(param_groups_with_lrs)
        """),
    md(r"""
    ### Regime B: frozen backbone, head trained with augmentation

    The backbone only runs forward (under `no_grad`), so this is cheap - but every epoch costs a
    full forward pass because augmented images are new images. Because the backbone is frozen and
    in eval mode, its validation features are exactly the cached ones: evaluation is free.
    """),
    code(r"""
    freeze_fn = tests.use_or_reference(ok_freeze, globals().get("freeze"), day1.freeze, "freeze")
    pg_fn = tests.use_or_reference(ok_groups, globals().get("param_groups_with_lrs"), day1.param_groups_with_lrs, "param_groups_with_lrs")

    aug_tf = v2.Compose([v2.RandomResizedCrop(IMG, scale=(0.5, 1.0), antialias=True), v2.RandomHorizontalFlip(),
                         v2.ToDtype(torch.float32, scale=True), v2.Normalize(torchkit.IMAGENET_MEAN, torchkit.IMAGENET_STD)])
    sub = torchkit._balanced_subset(data["y_train"].tolist(), N_FT, seed=0)
    aug_dl = DataLoader(day1.TensorImageDataset(data["x_train"][sub], data["y_train"][sub], aug_tf), batch_size=32,
                        shuffle=True, generator=torch.Generator().manual_seed(0))

    head_b = nn.Linear(feat_dim, 37).to(device)
    head_b.load_state_dict(probe_head.state_dict())             # train the head first: start from the probe
    ck_b = torchkit.CHECKPOINTS / f"d1-04_frozen_head_{BACKBONE}_{IMG}.pt"
    if ck_b.exists():
        extras = torchkit.load_checkpoint(head_b, ck_b); print(f"loaded {ck_b.name}")
    else:
        opt = torch.optim.AdamW(head_b.parameters(), lr=3e-4, weight_decay=1e-4)
        backbone.eval(); t0 = time.time()
        for epoch in range(FT_EPOCHS):
            for x, y in aug_dl:
                with torch.no_grad():
                    f = backbone(x.to(device))                  # frozen feature extractor
                loss = F.cross_entropy(head_b(f), y.to(device), label_smoothing=0.1)
                opt.zero_grad(); loss.backward(); opt.step()
        torchkit.save_checkpoint(head_b, ck_b)
        print(f"regime B trained in {time.time() - t0:.0f} s")
    with torch.no_grad():
        acc_b = (head_b(F_["f_val"].to(device)).argmax(1).cpu() == F_["y_val"]).float().mean().item()
    print(f"Regime B - frozen backbone + head with augmentation: val accuracy {acc_b:.3f}")
    """),
    md(r"""
    ### Regime C: fine-tune the top stages with discriminative learning rates

    Start from the ImageNet weights plus the probe head, freeze `stem` .. `layer2`, train `layer3`,
    `layer4` and `fc` with LRs `2.7e-5`, `9e-5`, `3e-4` (your `param_groups_with_lrs` with
    `base_lr=3e-4`), warm-up + cosine. `day1.keep_bn_eval` makes `model.train()` leave **every**
    BatchNorm layer in eval mode, so the running statistics stay ImageNet's (BN weights and biases
    of the unfrozen stages are still trained).

    Try it: with `base_lr=1e-3` and BatchNorm in train mode the fine-tuned model ends up *worse*
    than the linear probe on this small subset (~61-69 % vs ~79 %) - the classic way to destroy
    good features with a few hundred noisy steps.
    """),
    code(r"""
    ft = get_model(BACKBONE, weights=weights)
    ft.fc = nn.Linear(feat_dim, 37)
    ft.fc.load_state_dict(probe_head.state_dict())
    freeze_fn(ft, "layer2")
    day1.keep_bn_eval(ft)
    ft = ft.to(device)
    print(f"trainable: {torchkit.count_parameters(ft, trainable_only=True):,} of {torchkit.count_parameters(ft):,} parameters")

    val_dl = DataLoader(day1.TensorImageDataset(data["x_val"], data["y_val"], eval_tf), batch_size=50)
    ck_c = torchkit.CHECKPOINTS / f"d1-04_finetune_{BACKBONE}_{IMG}.pt"
    if ck_c.exists():
        extras = torchkit.load_checkpoint(ft, ck_c); logits_val = extras["logits_val"]; print(f"loaded {ck_c.name}")
    else:
        groups = pg_fn(ft, base_lr=3e-4, decay=0.3)
        for g in groups: print(f"  {g['name']:<7} lr {g['lr']:.1e}  ({sum(p.numel() for p in g['params']):,} params)")
        opt = torch.optim.AdamW(groups, weight_decay=1e-4)
        total = FT_EPOCHS * len(aug_dl)
        sched = day1.make_warmup_cosine_scheduler(opt, total, max(1, total // 8))
        torchkit.fit(ft, aug_dl, None, lambda o, y: F.cross_entropy(o, y, label_smoothing=0.1), opt, FT_EPOCHS,
                     scheduler=sched, max_minutes=4, log_every=4, device=device)
        ft.eval()
        with torch.no_grad():                         # one validation pass, reused below for Grad-CAM and metrics
            logits_val = torch.cat([ft(x.to(device)).cpu() for x, _ in val_dl])
        torchkit.save_checkpoint(ft, ck_c, logits_val=logits_val)
    acc_c = (logits_val.argmax(1) == F_["y_val"]).float().mean().item()
    print(f"Regime C - fine-tuned layer3 + layer4 + fc: val accuracy {acc_c:.3f}")
    """),
    code(r"""
    names = ["A linear probe", "B frozen + head\n(augmented)", "C fine-tune\nlayer3-4 + fc"]
    accs = [acc_a, acc_b, acc_c]
    gpu_expected = [0.88, 0.90, 0.93]
    ck_scratch = torchkit.CHECKPOINTS / f"d1-02_recipe_{128 if GPU else 64}px.pt"
    if ck_scratch.exists():
        names.insert(0, "from scratch\n(D1-02 PetNet)")
        accs.insert(0, max(torchkit.load_checkpoint(day1.PetNet(dropout=0.2), ck_scratch)["history"]["val_metric"]))
        gpu_expected.insert(0, 0.33)
    fig, ax = plt.subplots(figsize=(9, 3.8)); xs = np.arange(len(names))
    ax.bar(xs - 0.2, np.array(accs) * 100, 0.4, color=viz.PALETTE[0], label=f"this run ({BACKBONE}, {IMG} px, CPU budget)")
    ax.bar(xs + 0.2, np.array(gpu_expected) * 100, 0.4, color="lightgray", label="expected: resnet50, 224 px, GPU, full recipe")
    for x_, a in zip(xs, accs): ax.text(x_ - 0.2, a * 100 + 1, f"{a * 100:.1f}", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9); ax.set_ylabel("val accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_title("Same 2 000 / 500 Pet images, four ways"); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    On the CPU budget (ResNet-18 at 160 px, one augmented epoch) B and C add 1-3 points to the
    probe (~79 % -> ~81 %); with a GPU, 3+ epochs and ResNet-50 at 224 px the expected ladder is
    **~88 % (A) -> ~90 % (B) -> ~93 % (C)**. The ranking is the lesson: each step buys a little
    accuracy for a lot more compute, and on tiny datasets A or B is often the right answer.

    ### Exercise 4 · choose and justify an unfreezing schedule (written)

    For each scenario, say which regime (A / B / C / C') you would use and which stages you would
    unfreeze, in one or two sentences each: (1) 300 photos of 5 kinds of pastry; (2) 40 000
    chest X-rays, 14 findings; (3) 2 000 satellite tiles, 10 land-use classes; (4) you must try 50
    different heads / losses by tomorrow.
    """),
    exercise(
        stub=r"""
        my_answer = '''
        (1) pastry, 300 images:
        (2) chest X-rays, 40 000 images:
        (3) satellite tiles, 2 000 images:
        (4) 50 experiments by tomorrow:
        '''
        # YOUR CODE HERE: replace the blanks above with your answers (no checker - discuss with your neighbour)
        print(my_answer)
        """,
        solution=r"""
        my_answer = '''
        (1) pastry, 300 images: A, then B. Tiny data, ImageNet-like photos: the frozen features are already good;
            fine-tuning 11-25 M parameters on 300 images would overfit. B adds augmentation if A is not enough.
        (2) chest X-rays, 40 000 images: C' (full fine-tune, discriminative LRs, head first). Lots of data and a domain
            far from ImageNet (grey-scale, textures, no objects): the late features must change, even early ones a bit.
        (3) satellite tiles, 2 000 images: C on layer3-4 + fc. Medium data, a different viewpoint (top-down, scale):
            keep generic early filters, adapt the later stages; vertical flips / 90-degree rotations ARE valid here.
        (4) 50 experiments: A. Extract features once, then each experiment costs seconds instead of an hour.
        '''
        print(my_answer)
        """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · How much data do you need?

    With cached features we can afford to re-train the probe on 2, 5, 10, 20, 30 and 54 images per
    class and plot the validation accuracy against the training-set size on a **log axis**. The
    empirical rule (Sun et al. 2017, Hestness et al. 2017): accuracy grows roughly *linearly in
    log(data)* - every doubling of the data buys about the same number of points.
    """),
    code(r"""
    sizes = [74, 185, 370, 740, 1110, 2000]
    curve = []
    for n in sizes:
        idx = torchkit._balanced_subset(F_["y_train"].tolist(), n, seed=0)
        _, a, _ = day1.train_linear_probe(F_["f_train"][idx], F_["y_train"][idx], F_["f_val"], F_["y_val"], epochs=100)
        curve.append(a)
        print(f"{n:5d} train images ({n / 37:4.1f} / class): val acc {a:.3f}")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(sizes, np.array(curve) * 100, "-o", color=viz.PALETTE[0], label=f"linear probe on {BACKBONE} features")
    k, b = np.polyfit(np.log2(sizes), np.array(curve) * 100, 1)
    xx = np.array([sizes[0], 3680]); ax.semilogx(xx, k * np.log2(xx) + b, ":", color="gray",
                                                 label=f"fit: +{k:.1f} points per doubling (extrapolated to 3 680)")
    if ck_scratch.exists():
        ax.plot(2000, accs[0] * 100, "*", ms=14, color=viz.PALETTE[3], label="from scratch, 2 000 images (D1-02)")
    ax.set_xticks(sizes); ax.set_xticklabels(sizes); ax.set_xlabel("training images (log scale)"); ax.set_ylabel("val accuracy (%)")
    ax.set_title("Accuracy vs. training-set size"); ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Two images per class already give a usable classifier; going from 2 000 to 4 000 images would
    buy only a few more points. When you plan a data-collection budget, plot this curve on your
    own data first: it tells you whether to label more images, or to improve the model.

    ## 5 · What the network sees

    ### First-layer filters

    The 64 filters of the first conv (7x7x3) are directly interpretable as tiny RGB images. Compare
    with the hand-made gallery of D1-01: oriented edges at every angle (Sobel-like), colour
    opponents (red-green, blue-yellow), blobs and checkerboards (Gabor-like). Every convnet trained
    on natural images learns these - this is why the early layers transfer to almost anything.
    """),
    code(r"""
    w1 = get_model(BACKBONE, weights=weights).conv1.weight.detach().cpu()        # (64, 3, 7, 7)
    fig, axes = plt.subplots(4, 16, figsize=(14, 3.8))
    for k, ax in enumerate(axes.flat):
        f = w1[k]; f = (f - f.min()) / (f.max() - f.min() + 1e-8)                  # per-filter contrast stretch
        ax.imshow(f.permute(1, 2, 0).numpy(), interpolation="nearest"); ax.axis("off")
    plt.suptitle(f"The 64 first-layer filters of {BACKBONE} (ImageNet weights)"); plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Feature maps through the stages

    One image, the 6 most active channels at the output of each stage. Resolution halves at every
    stage while the channels grow; the maps go from "edges at this pixel" to "dog-face-like
    texture somewhere around here".
    """),
    code(r"""
    probe_model = get_model(BACKBONE, weights=weights).eval().to(device)
    acts = {}
    hooks = [getattr(probe_model, s).register_forward_hook(lambda m, i, o, s=s: acts.__setitem__(s, o.detach().cpu()))
             for s in ("layer1", "layer2", "layer3", "layer4")]
    img_i = 130
    x_img = eval_tf(data["x_val"][img_i])
    with torch.no_grad():
        probe_model(x_img[None].to(device))
    for h in hooks: h.remove()

    fig, axes = plt.subplots(4, 7, figsize=(14, 8.4))
    for r, s in enumerate(("layer1", "layer2", "layer3", "layer4")):
        a = acts[s][0]
        axes[r, 0].imshow(torchkit.denormalize(x_img)); axes[r, 0].set_title(f"{s}: {tuple(a.shape)}", fontsize=9)
        top = a.mean((1, 2)).argsort(descending=True)[:6]
        for c, ch in enumerate(top):
            axes[r, c + 1].imshow(a[ch], cmap="magma"); axes[r, c + 1].set_title(f"ch {int(ch)}", fontsize=8)
    for ax in axes.flat: ax.axis("off")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### The feature space

    Project the cached validation features (one 512-d / 2048-d vector per image) onto their first
    two principal components. Without any training on Pet, cats and dogs already separate, and
    individual breeds form clusters - this is why a *linear* head is enough.
    """),
    code(r"""
    CAT_BREEDS = {"Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau", "Maine Coon",
                  "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx"}
    fv = F_["f_val"] - F_["f_train"].mean(0)
    _, _, V = torch.pca_lowrank(F_["f_train"] - F_["f_train"].mean(0), q=2, center=False)
    Z = (fv @ V).numpy(); yv = F_["y_val"].numpy()
    is_cat = np.array([classes[c] in CAT_BREEDS for c in yv])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(Z[~is_cat, 0], Z[~is_cat, 1], s=10, alpha=.6, label="dogs (25 breeds)", color=viz.PALETTE[1])
    axes[0].scatter(Z[is_cat, 0], Z[is_cat, 1], s=10, alpha=.6, label="cats (12 breeds)", color=viz.PALETTE[0])
    axes[0].set_title("PCA of penultimate features: species"); axes[0].legend()
    show_breeds = ["Abyssinian", "Bombay", "Siamese", "Sphynx", "Pug", "Samoyed", "Saint Bernard", "Yorkshire Terrier"]
    axes[1].scatter(Z[:, 0], Z[:, 1], s=6, color="lightgray")
    for k, bname in enumerate(show_breeds):
        if bname in classes:
            m = yv == classes.index(bname)
            axes[1].scatter(Z[m, 0], Z[m, 1], s=18, color=viz.color(k), label=bname)
    axes[1].set_title("8 breeds highlighted (gray = the other 29)"); axes[1].legend(fontsize=8, ncol=2)
    for ax in axes: ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.grid(alpha=.3)
    plt.tight_layout(); plt.show()
    """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · Grad-CAM: where does the network look?

    Grad-CAM (Selvaraju et al., 2017) answers "which regions made the network say class $c$?"
    using the last convolutional feature maps $A^k$ (here `layer4`, 512 or 2048 channels at
    $\tfrac{1}{32}$ resolution). Weight each channel by how much the class score $y^c$ increases
    with it, averaged over space, keep the positive evidence:

    $$\alpha_k^c = \frac{1}{HW}\sum_{i,j} \frac{\partial y^c}{\partial A^k_{ij}}, \qquad
    L^c_{\text{Grad-CAM}} = \operatorname{ReLU}\Big(\sum_k \alpha_k^c A^k\Big)$$

    then upsample $L^c$ to the image size. In PyTorch you get $A$ and $\partial y^c / \partial A$
    with **hooks**: `layer.register_forward_hook(fn)` calls `fn(module, input, output)` on every
    forward pass; inside it, `output.register_hook(g)` calls `g(grad)` when the gradient w.r.t.
    that tensor is computed. Always `remove()` hooks when done.

    ### Exercise 5 · `gradcam(model, target_layer, x, class_idx)`

    `x` is one image `(3, H, W)`. Register the forward hook, run the model in eval mode, pick
    `class_idx` (the argmax if `None`), backpropagate `logits[0, class_idx]`, compute the map,
    upsample with `F.interpolate(..., size=x.shape[-2:], mode="bilinear", align_corners=False)`,
    min-max scale to [0, 1], return a NumPy `(H, W)` array.
    """),
    exercise(
        stub=r"""
        def gradcam(model, target_layer, x, class_idx=None):
            store = {}

            def fwd_hook(module, inp, out):
                store["act"] = out
                out.register_hook(lambda g: store.__setitem__("grad", g))

            handle = target_layer.register_forward_hook(fwd_hook)
            model.eval()
            try:
                # YOUR CODE HERE: forward x[None], choose the class, zero grads, backward the class score
                raise NotImplementedError
            finally:
                handle.remove()
            # YOUR CODE HERE: weights, weighted sum, ReLU, upsample, min-max -> numpy (H, W)
            raise NotImplementedError

        ok_cam = tests.check_gradcam(gradcam)
        """,
        solution=r"""
        def gradcam(model, target_layer, x, class_idx=None):
            store = {}

            def fwd_hook(module, inp, out):
                store["act"] = out
                out.register_hook(lambda g: store.__setitem__("grad", g))

            handle = target_layer.register_forward_hook(fwd_hook)
            model.eval()
            try:
                logits = model(x[None].to(next(model.parameters()).device))
                c = int(logits.argmax(1)) if class_idx is None else int(class_idx)
                model.zero_grad()
                logits[0, c].backward()
            finally:
                handle.remove()
            A, G = store["act"].detach(), store["grad"].detach()          # (1, K, h, w)
            w = G.mean(dim=(2, 3), keepdim=True)                           # alpha_k
            cam = F.relu((w * A).sum(1, keepdim=True))                     # (1, 1, h, w)
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam.cpu().numpy()

        ok_cam = tests.check_gradcam(gradcam)
        """),
    code(r"""
    cam_fn = tests.use_or_reference(ok_cam, globals().get("gradcam"), day1.gradcam, "gradcam")
    clf = ft.eval()                                   # the fine-tuned model of regime C
    for p in clf.parameters(): p.requires_grad_(True) # Grad-CAM needs gradients to flow back to layer4
    pred_val = logits_val.argmax(1); yv_t = F_["y_val"]
    shuffled = torch.randperm(len(yv_t), generator=torch.Generator().manual_seed(3)).tolist()   # val is sorted by breed
    right = [i for i in shuffled if pred_val[i] == yv_t[i]][:4]
    wrong = [i for i in shuffled if pred_val[i] != yv_t[i]][:4]

    fig, axes = plt.subplots(2, 8, figsize=(16, 4.8))
    for k, i in enumerate(right + wrong):
        x = eval_tf(data["x_val"][i]); img = torchkit.denormalize(x)
        cam = cam_fn(clf, clf.layer4, x)
        axes[0, k].imshow(img); axes[0, k].axis("off")
        viz.heatmap_overlay(axes[1, k], img, cam, alpha=0.5)
        ok_ = i in right
        axes[0, k].set_title(f"{'OK' if ok_ else 'WRONG'}\ntrue: {short[yv_t[i]]}\npred: {short[pred_val[i]]}",
                             fontsize=8, color="black" if ok_ else viz.PALETTE[3])
    plt.suptitle("Grad-CAM of the predicted class (top: image, bottom: heat map)", y=1.02); plt.tight_layout(); plt.show()
    """),
    md(r"""
    ### Is it looking at the pet? A shortcut-learning check

    A model can be right for the wrong reason: snow -> husky, a sofa -> cat, a hospital's
    watermark -> pneumonia. Pet ships a pixel mask of the animal (the trimap), so we can measure it:

    1. for 40 validation images, the fraction of the Grad-CAM mass that falls **on the pet**, versus
       the fraction of the image the pet occupies (a model that ignored the pet would sit on the diagonal);
    2. accuracy when the **pet is greyed out** and only the background is left. Far above chance
       (2.7 %) would mean the background alone predicts the breed: a shortcut.
    """),
    code(r"""
    chk = torch.randperm(len(yv_t), generator=torch.Generator().manual_seed(0))[:100]      # 100 random val images
    masks = day1.pet_masks_for([data["paths_val"][i] for i in chk], IMG)  # 1 = pet (incl. border), 0 = background
    on_pet, area = [], []
    for k in range(40):
        cam = cam_fn(clf, clf.layer4, eval_tf(data["x_val"][chk[k]]))
        on_pet.append(day1.cam_on_mask_fraction(cam, masks[k])); area.append(masks[k].mean())

    x_bg = torch.stack([eval_tf(data["x_val"][i]) for i in chk])
    m_t = torch.from_numpy(masks).float()[:, None]
    x_bg = x_bg * (1 - m_t)                                             # 0 after normalisation = the mean colour
    with torch.no_grad():
        acc_orig = (logits_val[chk].argmax(1) == yv_t[chk]).float().mean().item()
        acc_bg = (clf(x_bg.to(device)).argmax(1).cpu() == yv_t[chk]).float().mean().item()

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    axes[0].scatter(area, on_pet, color=viz.PALETTE[0]); axes[0].plot([0, 1], [0, 1], ":", color="gray", label="no preference")
    axes[0].set_xlabel("fraction of the image that is pet"); axes[0].set_ylabel("fraction of Grad-CAM mass on the pet")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[0].set_title(f"mean: {np.mean(on_pet):.2f} of the evidence on {np.mean(area):.2f} of the pixels")
    axes[1].imshow(torchkit.denormalize(x_bg[5])); axes[1].axis("off"); axes[1].set_title("pet removed (background only)")
    axes[2].bar(["original", "background only", "chance"], [acc_orig * 100, acc_bg * 100, 100 / 37],
                color=[viz.PALETTE[0], viz.PALETTE[3], "lightgray"])
    axes[2].set_ylabel("accuracy (%) on 100 val images"); axes[2].grid(alpha=.3, axis="y")
    for k, v in enumerate([acc_orig, acc_bg, 1 / 37]): axes[2].text(k, v * 100 + 1, f"{v * 100:.1f}", ha="center")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Most points lie above the diagonal: the evidence is concentrated on the animal (often the
    head) more than its area alone would explain. With the pet removed, accuracy collapses to a
    few percent, near chance: no strong background shortcut in this dataset. Run the same two checks on your own data - especially when the images of
    different classes come from different sources (scanners, cameras, websites, hospitals).

    ## 7 · Choosing and reading metrics (ML Yearning, ch. 4-11)

    * **Dev and test sets from the distribution you care about** - the same one. Pet's official
      split is fine; "train on web photos, test on phone photos" is a different problem.
    * **One single-number evaluation metric** to compare models quickly (accuracy, mean recall,
      F1). Several numbers -> no clear winner.
    * **Satisficing metrics** are constraints, not goals: "maximise accuracy *subject to* < 50 ms per
      image on a phone and < 30 MB". Pick the best model that satisfies them.
    * **Change the metric when it stops matching what users need.** Mini case: a pet-insurance app
      shows the breed to the user. Which metric?
    """),
    code(r"""
    top5 = (torch.topk(logits_val, 5, 1).indices == yv_t[:, None]).any(1).float().mean().item()
    top1 = (pred_val == yv_t).float().mean().item()
    recall = np.array([(pred_val[yv_t == c] == c).float().mean().item() for c in range(37)])
    order = np.argsort(recall)

    fig, ax = plt.subplots(figsize=(14, 3.8))
    cols = [viz.PALETTE[0] if classes[c] in CAT_BREEDS else viz.PALETTE[1] for c in order]
    ax.bar(range(37), recall[order] * 100, color=cols)
    ax.set_xticks(range(37)); ax.set_xticklabels([short[c] for c in order], rotation=70, ha="right", fontsize=7)
    ax.axhline(top1 * 100, ls="--", color="k", lw=1, label=f"top-1 = mean recall = {top1 * 100:.1f}% (balanced val set)")
    ax.set_ylabel("recall (%)"); ax.set_title(f"Per-breed recall (blue = cats, orange = dogs); top-5 accuracy = {top5 * 100:.1f}%")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3, axis="y"); plt.tight_layout(); plt.show()

    conf = torch.zeros(37, 37, dtype=torch.long)
    for t, p_ in zip(yv_t, pred_val): conf[t, p_] += 1
    off = conf.clone().fill_diagonal_(0)
    print("most confused pairs (true -> predicted: count):")
    for k in torch.topk(off.flatten(), 5).indices:
        t, p_ = divmod(int(k), 37); print(f"   {classes[t]:<28s} -> {classes[p_]:<28s} {int(off[t, p_])}")
    """),
    md(r"""
    Reading the case:

    * **Top-1** is the single-number metric if the app shows one breed; **top-5** if it shows a
      short list to pick from (it is much higher, so the product decision changes the metric).
    * **Mean per-class recall** equals top-1 here only because the validation set is balanced; on
      a real, imbalanced stream (many more Labradors than Sphynx cats) report it separately, and
      track the **worst-breed recall** as a satisficing metric if every breed must work.
    * The confusions are between look-alike breeds (e.g. terriers, or Ragdoll / Birman): no
      metric fixes that; more data for those pairs, higher resolution, or a better backbone does.
    * Measure latency / size of the deployed model as satisficing metrics:
    """),
    code(r"""
    x1 = eval_tf(data["x_val"][0])[None].to(device)
    with torch.no_grad():
        clf(x1); t0 = time.perf_counter()
        for _ in range(5): clf(x1)
    ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"{BACKBONE} @ {IMG} px: {torchkit.count_parameters(clf) / 1e6:.1f} M params "
          f"({torchkit.count_parameters(clf) * 4 / 2**20:.0f} MB fp32), {ms:.0f} ms per image on this {device.type.upper()}")
    """),

    # ------------------------------------------------------------------ 8
    md(r"""
    ## 8 · Modern note: self-supervised backbones (optional)

    ImageNet-*supervised* features are biased towards the 1 000 ImageNet classes. Self-supervised
    backbones such as **DINOv2** (Oquab et al., 2023; trained on 142 M curated images without
    labels) give features that transfer better: a linear probe on DINOv2 ViT-S/14 features reaches
    **~95 %** on Pet (vs ~88-90 % for supervised ResNet-50). The cell runs only if `timm` is
    installed (`pip install timm`; first use downloads ~90 MB); Day 3 covers these models in depth.
    """),
    code(r"""
    try:
        import timm
    except ImportError:
        timm = None
        print("timm is not installed - skipping the DINOv2 demo (pip install timm to run it). "
              "Expected: linear probe on DINOv2 ViT-S/14 features ~95 % on Pet vs ~88-90 % for supervised ResNet-50.")
    if timm is not None:
        dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=126).eval().to(device)
        cfg = timm.data.resolve_data_config({}, model=dino)
        dino_tf = v2.Compose([v2.Resize(126, antialias=True), v2.CenterCrop(126), v2.ToDtype(torch.float32, scale=True),
                              v2.Normalize(cfg["mean"], cfg["std"])])
        n_tr = 740 if not GPU else 2000
        sub_d = torchkit._balanced_subset(data["y_train"].tolist(), n_tr, seed=0)
        dl_tr = DataLoader(day1.TensorImageDataset(data["x_train"][sub_d], data["y_train"][sub_d], dino_tf), batch_size=50)
        dl_va = DataLoader(day1.TensorImageDataset(data["x_val"], data["y_val"], dino_tf), batch_size=50)
        t0 = time.time()
        fd_tr, yd_tr = extract(dino, dl_tr); fd_va, yd_va = extract(dino, dl_va)
        _, acc_dino, _ = day1.train_linear_probe(fd_tr, yd_tr, fd_va, yd_va, epochs=100)
        idx = torchkit._balanced_subset(F_["y_train"].tolist(), n_tr, seed=0)
        _, acc_sup, _ = day1.train_linear_probe(F_["f_train"][idx], F_["y_train"][idx], F_["f_val"], F_["y_val"], epochs=100)
        print(f"linear probe with {n_tr} training images: DINOv2 ViT-S/14 @126 px {acc_dino:.3f} vs {BACKBONE} @{IMG} px {acc_sup:.3f}"
              f"  ({time.time() - t0:.0f} s)")
    """),

    # ------------------------------------------------------------------ 9
    md(r"""
    ## 9 · Summary

    * On small data, start from a pretrained backbone. Match its preprocessing (`weights.transforms()`),
      replace the head.
    * **A** linear probe on cached features: seconds, no augmentation, a strong baseline.
      **B** frozen backbone + head: adds augmentation for the cost of a forward pass per epoch.
      **C** fine-tuning: train the head first, small LR, discriminative LRs (earlier = smaller),
      frozen BN in eval mode, unfreeze from the top.
    * Accuracy grows roughly linearly with log(training-set size): plot that curve before buying labels.
    * Early filters are generic (edges, colours), late features are task-specific: that is what
      transfers and what must adapt.
    * Hooks (`register_forward_hook`, `Tensor.register_hook`) expose activations and gradients;
      Grad-CAM = ReLU(sum of feature maps weighted by their average gradient). Check that the model
      looks at the object, and that the background alone does not predict the label.
    * One single-number metric, satisficing constraints, dev / test from the target distribution;
      change the metric when it stops matching the product.

    ### Self-check
    1. You fine-tune a ResNet with BatchNorm on batches of 4 images. Why might freezing BN statistics help?
    2. Why train the head before unfreezing the backbone?
    3. Your linear probe gets 70 % with 500 images and 76 % with 1 000. Roughly what do you expect with 4 000?
    4. What two tensors does Grad-CAM need, and how do you get them in PyTorch?
    5. The model is 99 % accurate on the test set but Grad-CAM always highlights the bottom-left corner. What do you check next?
    6. Give one example of a satisficing metric for a phone app.

    ### Go deeper
    * J. Yosinski et al., *How transferable are features in deep neural networks?* (NeurIPS 2014).
    * R. R. Selvaraju et al., *Grad-CAM: Visual explanations from deep networks via gradient-based localization* (ICCV 2017).
    * A. Ng, *Machine Learning Yearning*, chapters 4-11 (dev / test sets, single-number and satisficing metrics).
    * M. Oquab et al., *DINOv2: Learning robust visual features without supervision* (2023).
    * A. Kumar et al., *Fine-tuning can distort pretrained features and underperform out-of-distribution* (ICLR 2022) - why "head first" (LP-FT) works.
    * torchvision docs, "Models and pre-trained weights"; timm docs, "Feature extraction" and "Fine-tuning".
    """),
]
