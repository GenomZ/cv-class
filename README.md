# Deep Learning for Computer Vision - 3-day course (2026 refresh)

A refreshed, PyTorch-based version of the DSR "Deep Learning for Computer Vision
Fundamentals" and "Advanced Computer Vision" courses. Three days, eight hours each,
most of it hands-on in notebooks that explain every mechanism with generated figures
and leave the key lines for students to write.

| Day | Theme | Notebooks |
|---|---|---|
| 1 | ConvNets: from pixels to transfer learning | `D1-01` convolution · `D1-02` training recipe · `D1-03` architecture zoo · `D1-04` transfer learning & Grad-CAM |
| 2 | Object detection | `D2-01` boxes, IoU, mAP · `D2-02` two-stage (anchors, RPN, RoIAlign) · `D2-03` one-stage (FPN, focal loss) · `D2-04` modern detectors |
| 3 | Segmentation and the frontier | `D3-01` segmentation basics · `D3-02` U-Net · `D3-03` Mask R-CNN · `D3-04` ViT, DINOv2, CLIP, SAM |

## Layout

```
docs/        course design: topic inventory of the old repos, 3-day syllabus, refresh map, notebook specs
notebooks/   student notebooks (D1-01 ... D3-04) and notebooks/solutions/
notebooks/cvcourse/   shared helpers: NumPy reference implementations, checkers, synthetic data, PyTorch utilities
tools/       notebook sources (tools/src/*.py) and the build / run scripts
quizzes/     end-of-day multiple-choice quizzes with solutions
data/        datasets and checkpoints (not committed; see setup)
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows   (source .venv/bin/activate on macOS / Linux)
pip install -r requirements.txt
python tools/download_data.py     # Oxford-IIIT Pet, Penn-Fudan, FashionMNIST, checkpoints
jupyter lab notebooks
```

## Working on the notebooks

Notebooks are generated. Edit the source module in `tools/src/`, then:

```bash
python tools/build_notebooks.py          # writes notebooks/*.ipynb and notebooks/solutions/*-solution.ipynb
python tools/run_notebooks.py D1-01      # executes the solution notebook(s) headlessly to verify them
```

Exercise cells are stubs in the student notebooks and complete in the solutions; every
exercise has a `tests.check_*` cell that prints `[PASS]` / `[FAIL]` with a hint.

## Sources

The original material (TensorFlow 2, 2024) lives in `GenomZ/DL-Image-B42`,
`GenomZ/Advanced-CV` and `GenomZ/Advanced_Computer_Vision`. The topic-by-topic mapping is
in `docs/01-source-topic-inventory.md` and `docs/03-refresh-map.md`.
