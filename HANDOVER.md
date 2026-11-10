# Handover: 3-day Deep Learning for Computer Vision course (PyTorch, 2026 refresh)

Written 2026-11-09 for whoever continues this work (human or agent). Read this file, then
`docs/04-notebook-specs.md` (the contract every notebook follows), then open one notebook
source in `tools/src/` next to its generated `.ipynb` to see the pattern.

## 1. What this repository is

A refreshed, PyTorch-based replacement for three TensorFlow course repos from 2024
(`GenomZ/DL-Image-B42`, `GenomZ/Advanced-CV`, `GenomZ/Advanced_Computer_Vision`). Students
already know neural networks and backprop. Three days x 8 hours, four 75-90 min blocks per
day, each block = one notebook: short guided walk through generated figures, then students
fill in the "missing parts" (exercise cells) and get immediate `[PASS]`/`[FAIL]` feedback.

| Day | Notebooks |
|---|---|
| 0 (pre-course) | `D0-00` setup and tensors |
| 1 ConvNets | `D1-01` convolution · `D1-02` training recipe · `D1-03` architecture zoo · `D1-04` transfer learning + Grad-CAM |
| 2 Detection | `D2-01` boxes, IoU, mAP, lineage · `D2-02` anchors, RPN, RoIAlign, Faster R-CNN · `D2-03` dense grids, FPN, focal loss · `D2-04` YOLO, DETR, deployment |
| 3 Segmentation + frontier | `D3-01` metrics, losses, upsampling · `D3-02` U-Net · `D3-03` Mask R-CNN · `D3-04` ViT, DINOv2, CLIP, SAM |

## 2. Current state (verified 2026-11-05, commit `0f7d537`)

* All 13 **solution** notebooks execute end to end on a CPU-only Windows machine via
  `python tools/run_notebooks.py` in about 13 minutes total (with checkpoints present);
  every checker prints `[PASS]`. Per-notebook times and key numbers: `docs/07-course-map.md`.
* 590 cells, 155 generated figures, 77 exercise cells (76 code stubs + written answers),
  74 checkers, 3 quizzes with solutions. Every figure was looked at by its author.
* Datasets are on disk under `data/` (ignored by git): Oxford-IIIT Pet (1.6 GB), Penn-Fudan
  (104 MB), FashionMNIST, plus the in-notebook synthetic "shapes" set. 16 checkpoints
  (1.1 GB) in `data/checkpoints/` make re-runs fast.
* Environment: `.venv` (Python 3.14) with CPU torch 2.15, torchvision, timm 1.0.31, numpy,
  scipy, matplotlib, pillow, nbformat, nbconvert, ipykernel, jupyterlab. **Not** installed:
  `transformers`, `open_clip_torch`, `onnx`, `opencv-contrib-python`,
  `segmentation-models-pytorch`, `torchmetrics` (all optional, all guarded in the notebooks).
* No remote, no CI.
* **Interactive labs (added 2026-11-10).** 26 self-contained HTML labs in
  `notebooks/cvcourse/labs/`, embedded in the notebooks with `labs.show("name")` (29 lab cells
  across the 13 notebooks, each preceded by a "Predict first" question). Every number a lab
  displays was compared against the Python reference (`cvcourse.*`, torch, torchvision) by the
  agent that built it. Verified only in headless Chrome: `python tools/check_labs.py` passes for
  all 26. **Not yet** opened in JupyterLab / VS Code / Colab or dragged with a real mouse (T10).
  The same pass added "what to look at" sentences after figures, shapes and return types to
  exercise instructions, and definitions at first use (about 100 small edits in `tools/src`).

## 3. How the pieces fit

```
docs/                 the design: 01 inventory of the old repos -> 02 syllabus -> 03 refresh map
                      -> 04 notebook specs (THE CONTRACT) -> 05 reading list -> 06 run-of-show -> 07 course map
tools/nb.py           notebook DSL: md() code() exercise(stub, solution) header() build()
tools/src/*.py        one module per notebook: NAME + CELLS; exercise() cells differ between variants
tools/build_notebooks.py [pattern]   -> notebooks/<NAME>.ipynb (student) + notebooks/solutions/<NAME>-solution.ipynb
tools/run_notebooks.py   [pattern]   -> executes solutions headlessly into notebooks/solutions/_executed/ (ignored)
tools/download_data.py [fashion|pet|pennfudan]   idempotent dataset fetch, writes .complete markers
notebooks/cvcourse/   shared package imported by every notebook (sys.path snippet in cell 1):
    viz.py convops.py boxes.py seg.py shapes.py   NumPy/Matplotlib reference implementations + synthetic data
    tests.py + tests_day0..3.py                  checkers; tests.py re-exports the day modules (names must stay unique)
    torchkit.py                                   PyTorch plumbing: get_device, fit, evaluate, summarize, profile,
                                                  tune_cpu_threads, checkpoints, lr_range_test, show_batch, gradcam,
                                                  get_fashion_mnist, get_pet_datasets(target=category|segmentation|detection),
                                                  get_penn_fudan, ShapesDataset, detection_collate
    day1.py day2.py day3.py                       per-day models and reference solutions (MiniResNet, mini detector,
                                                  tiny RPN, FPN, U-Net, Mask R-CNN glue, ViT helpers, schematics)
    labs/                                         interactive labs: lab.css + lab.js (shared toolkit), one <name>.html per lab,
                                                  generated index.html, __init__.py (labs.show / open / catalog), README.md (the contract)
tools/check_labs.py [pattern]        -> loads labs in headless Chrome / Edge, fails on script errors, screenshots to labs/_shots/ (ignored)
tools/build_labs_index.py            -> regenerates labs/index.html from each lab's <meta> tags
notebooks/assets/     three static fallback PNGs for D3-04 when timm / weights are missing
quizzes/dayN.md + dayN-solutions.md
```

**Rules that keep it consistent**

1. Never hand-edit a `.ipynb`; edit `tools/src/<module>.py`, rebuild, re-run.
2. An exercise removes the 3-15 lines that *are* the concept and is followed by a
   `tests.check_*` cell. Student stubs `raise NotImplementedError`. Day 1 notebooks
   additionally wrap later cells with `tests.use_or_reference(ok, mine, reference)` so
   "Run all" continues with the reference implementation; Day 2/3 student notebooks stop at
   the first unfinished stub (students work cell by cell). Either is acceptable; see task T6.
3. Checkers print ASCII `[PASS]` / `[FAIL]` + hint (Windows console is cp1250).
4. Training cells have a `max_minutes` cap and load `data/checkpoints/<name>.pt` if present.
   Markdown states both the CPU numbers measured here and the expected GPU / full-size numbers.
5. Figures are generated in the notebook; external links only under "Go deeper".
6. The Bash tool used in the authoring sessions mangles heredocs containing apostrophes;
   write helper scripts to a file and run them instead.
7. A mechanism with a geometry gets a lab, placed after its static figure and before the
   exercise that implements it, introduced by a markdown cell with one **Predict first**
   question. Labs follow `notebooks/cvcourse/labs/README.md` (one mechanism, live numbers
   equal to the Python reference, a "Try this" list with hidden answers, colours by role,
   no network). `docs/08-interactive-labs-brief.md` is the build brief.
8. Do not run several notebook executions at once on a laptop: cells with a wall-clock cap
   (`max_minutes`) train for fewer steps under load and the latency benchmarks inflate.

Quick start for a continuing agent:

```bash
.venv/Scripts/python tools/build_notebooks.py          # regenerate all notebooks from tools/src
.venv/Scripts/python tools/run_notebooks.py D2-0       # execute the Day 2 solutions
.venv/Scripts/python notebooks/cvcourse/torchkit.py    # toolkit smoke test (~5 s)
```

## 4. Open work, staged for implementation

Ordered by value for the first delivery of the course. Each task is self-contained; "Done
when" is the acceptance test. Keep commits per task.

### T1 · Run the never-executed optional branches (highest value, ~2 h)
The real CLIP, SAM, OWL-ViT / Grounding DINO and ONNX cells are guarded by import checks and
have **never run**. Install into `.venv`: `transformers`, `open_clip_torch`, `onnx`,
`onnxscript`, `opencv-contrib-python`. Then run D2-01 (Selective Search branch), D2-04
(OWL-ViT demo, ONNX export) and D3-04 (CLIP zero-shot on Pet, SAM point/box prompts) and fix
whatever breaks. Keep the static fallbacks for machines without the packages.
Files: `tools/src/d2_01_boxes_iou_map.py`, `d2_04_modern_detectors.py`, `d3_04_foundation_models.py`, `notebooks/cvcourse/day3.py`.
Done when: those three solution notebooks pass `run_notebooks.py` with the packages installed, the CLIP accuracy printed replaces the "~87%" placeholder in the D3-04 markdown, and a SAM mask figure is produced from real SAM weights.

### T2 · GPU pass and full-size checkpoints (~half a day on one GPU)
Nothing has run on a GPU. On a CUDA machine: run the whole suite once (`run_notebooks.py`),
confirm the `device == "cuda"` paths, and record the real full-size numbers that the markdown
currently *states* (resnet50 @224 px fine-tune on 2 000 Pet images, Faster / Mask R-CNN on full
Penn-Fudan, Pet U-Net at 128-224 px, FCOS on Pet heads). Replace the stated numbers in the
markdown where they differ. Save the full-size checkpoints under new names (e.g. `*_gpu_full.pt`)
and make the loaders prefer them when present.
Done when: every "expected on GPU" sentence in the notebooks cites a measured number, and `docs/07-course-map.md` has a GPU column.

### T3 · Host the checkpoints and finish `download_data.py` (~1 h + upload)
`tools/download_data.py` only creates `data/checkpoints/`; there is no URL. Decide hosting
(Hugging Face Hub dataset repo, S3, or a release asset), upload `data/checkpoints/` (1.1 GB now,
more after T2) and add a `checkpoints` task with checksum verification. Also add the Oxford-IIIT
Pet, Penn-Fudan, FashionMNIST and photo (Matplotlib's Grace Hopper sample) attribution to a new
`docs/08-sources-and-licenses.md` (the old course had one; Pet is CC BY-SA 4.0).
Done when: a fresh clone + `pip install -r requirements.txt` + `python tools/download_data.py` + `run_notebooks.py` passes on a clean machine.

### T4 · Fresh-environment install test (~1 h)
`.venv` was built incrementally. Create a fresh venv from `requirements.txt` on Python 3.11/3.12
(the versions most students will have), on Windows and on Linux/macOS, and fix pins.
Also try the Colab path: open `D0-00` in Colab, run the dataset download, run `D2-03` (CPU) and
`D1-04` (GPU). Write the result into `docs/02-syllabus-3-day.md` "Pre-course material".
Done when: the install works on two OSes and Colab, and `requirements.txt` is pinned to tested versions.

### T5 · Unify CPU thread tuning (~30 min)
Three implementations exist: `torchkit.tune_cpu_threads` (canonical, added last),
`day1.pick_cpu_threads`, `day3.tune_cpu_threads`, plus `day2.fastest_num_threads` (per-loop
benchmark, keep). Make `day1`/`day3` delegate to `torchkit.tune_cpu_threads` and call it once in
every notebook's setup cell (D0, D1-01, D2-xx currently do not).
Done when: `grep -n "cpu_threads" tools/src/*.py` shows one call per notebook, all into torchkit, and the suite still passes.

### T6 · Decide and document the student "Run all" behaviour (~1 h)
Day 1 uses `tests.use_or_reference` so student notebooks run to the end with reference
fallbacks; Day 2/3 stop at the first stub. Pick one (recommendation: the Day 1 pattern, it lets
students see the whole block's figures before attempting the exercises) and apply it to the
Day 2/3 sources, or document "work cell by cell" in `D0-00` and the README.
Done when: all 13 student notebooks either run end to end headlessly or the README states the cell-by-cell convention and `D0-00` shows the `[FAIL]` -> fix -> `[PASS]` loop once.

**Status 2026-11-10:** resolved by documentation. `D0-00` section 6 explains the
figure -> lab -> exercise -> checker rhythm and the cell-by-cell convention; Day 2/3 still
stop at the first unfinished stub.

### T10 · Try the labs in the real front ends (~1 h, needs a human)
The labs were only exercised in headless Chrome with synthetic pointer events. Open `D2-01`
and `D3-01` in JupyterLab, in VS Code and in Colab: check that each `labs.show` cell renders,
resizes itself, follows the light / dark theme, and that dragging (boxes), painting
(seg-metrics) and touch input work. Known cosmetic leftovers: several labs are 950-1100 px
tall at full width (guideline was 900); label chips overlap in `hungarian-matching` and in
random `nms-playground` scenes when boxes crowd; `pr-curve` wraps its plot under the list
below ~980 px; `--hl` equals `--c2` (see the labs README).
Done when: the three front ends are confirmed (or `labs.show` falls back to a link where
embedding is blocked) and the run-of-show time budget has been tried on one real block.

### T7 · Instructor review pass of content (~half a day, instructor)
Read each notebook top to bottom as a student. Known soft spots to look at first:
* D1-02: on CPU at 64 px the "recipe" run scores lower than the plain run (13.2% vs 16.0%); the
  markdown explains the point is the train/val gap, but after T2 the GPU numbers should show the
  recipe winning outright. Update the text.
* D2-02: done 2026-11-10. The text now says 0.7 / 0.3 is the paper's rule for ~800 px images and
  the notebook uses 0.5 / 0.3 at 128 px; quiz Q5 agrees. (At 0.7 the demo has 5 positives for
  4 objects, two of them forced matches.)
* D3-02: the "without skips" ablation mIoU varied between 0.42 and 0.73 across runs because the
  training stops on a wall-clock cap; the text only claims "well below". Consider a step cap.
* Block timings: the labs add 5-24 min per block if done in full; `docs/06` has the budget and
  the notebook headers still state the original minutes.
* D2-04: FCOS on Pet heads reaches only mAP@0.5 0.64 on CPU with 48 images; after T2 replace.
* D3-03: Mask R-CNN fine-tuning barely moves AP (COCO already has "person"); the lesson is the
  false-positive drop (45 -> 7). Keep, but say it on the slide.
* Checker cells also echo `True` as the cell's last value (they return a bool). Cosmetic; either
  end those cells with `;` in the sources or make `_report` return `None`.
Done when: the instructor has a list of edits applied via `tools/src` and the suite passes.

### T8 · Slides (deferred by the course owner)
Explicitly out of scope so far. When wanted: one deck per day built from the notebooks'
figures (export via `tools/`-side script that dumps PNGs from `_executed/`), 10-15 slides per
block, following `docs/06-instructor-run-of-show.md`. The figures are already the slides.

### T9 · Nice-to-haves (only after T1-T7)
* `torchmetrics` cross-check of the course's own mAP code in D2-01 (one guarded cell exists in spirit; add the comparison).
* Ultralytics YOLO and RT-DETR fine-tune cells in D2-04 (licence note already in the markdown).
* `segmentation_models_pytorch` comparison cell in D3-02.
* A `tools/export_figures.py` that writes every executed figure to `docs/figures/<notebook>/` for the slides and the course website.
* Convert quizzes to a form (Google Forms / Moodle) with the same items.

## 5. Things a continuing agent should not do

* Do not switch frameworks or re-add TensorFlow material; the decision is final.
* Do not edit `.ipynb` files directly or commit `notebooks/solutions/_executed/`.
* Do not commit `data/` (datasets, checkpoints) or `.venv/`.
* Do not rename checkers; notebooks reference them by name and `tests.py` re-exports all `tests_day*` modules, so a duplicate name silently shadows another.
* Do not lower the "every exercise has a checker or a visual self-check" bar to save time.

## 6. Provenance

* Design, infrastructure, D1-01 and the shared NumPy package: main session (Claude Fable 5.1), 2026-11-04/05.
* `torchkit.py`, `D0-00`, D1-01 training cells; Day 1 (D1-02..04); Day 2 (D2-01..04); Day 3 (D3-01..04): four Opus 5.5 agents, 2026-11-04/05, each verified by executing and inspecting every figure.
* Interactive labs, 2026-11-10: toolkit, `conv-playground` exemplar, loader, check / index tools and
  D0-00 integration by the main session (Claude Fable 5); the other 25 labs, their notebook
  integration and the clarity pass by seven Opus 5.5 agents, one per notebook pair.
* Source material licences: Chollet's DLwP notebooks (MIT, permission for DSR teaching), d2l.ai (CC BY-SA 4.0) were only read for the inventory; no text was copied.
