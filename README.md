# Deep Learning for Computer Vision: a 3-day hands-on course (PyTorch)

Three days, from the convolution operation to foundation models. Each block is one notebook:
generated figures explain the mechanism, an interactive lab lets you play with it, and you
write the few lines that *are* the concept. A checker tells you at once whether they are right.

| Day | Theme | Notebooks |
|---|---|---|
| 0 | Before the course (about 90 min, on your own) | `D0-00` setup check and tensor refresher |
| 1 | ConvNets: from pixels to transfer learning | `D1-01` convolution · `D1-02` training recipe · `D1-03` architecture zoo · `D1-04` transfer learning and Grad-CAM |
| 2 | Object detection | `D2-01` boxes, IoU, mAP · `D2-02` two-stage (anchors, RPN, RoIAlign) · `D2-03` one-stage (FPN, focal loss) · `D2-04` modern detectors |
| 3 | Segmentation and the frontier | `D3-01` segmentation basics · `D3-02` U-Net · `D3-03` Mask R-CNN · `D3-04` ViT, DINOv2, CLIP, SAM |

You need: Python 3.10 or newer, about 6 GB of free disk, and no GPU (everything runs on a
laptop CPU; a GPU only makes the training cells faster). Prerequisites: Python, NumPy, and
the basics of neural networks and backpropagation.

## Practical track: eight short, run-first lessons

A second, simpler way through the material. You run ready-made networks, change their
settings and watch what happens; where you do build something, you type layers from a table
and a check tells you in plain words whether they are right. Start here if you are new to
deep learning.

| Lesson | Notebook | What you do |
|---|---|---|
| 1 | `P1-dense-vs-conv-mnist` | Run a fully connected network and a small CNN on handwritten digits, compare them, then change layers, feature maps, kernel size, padding, stride. No code to write. |
| 2 | `P2-vgg` | See all 16 layers of VGG16 and what they cost, then train a mini-VGG on CIFAR-10 photos. No code to write. |
| 3 | `P3-googlenet` | Build GoogLeNet from Inception blocks, step by step from a table, with a check and a diagram after every step. |
| 4 | `P4-unet` | Assemble a U-Net (encoder, decoder, skip connections) block by block, then train a small one. |
| 5 | `P5-transfer-learning` | Take a pretrained network, give it a new classifier, train it on 300 pet photos. |
| 6 | `P6-googlenet-auxiliary-classifiers` | Add the two auxiliary classifiers that lesson 3 left out of GoogLeNet: build the small extra head, return three outputs, combine three losses, then train with and without them. |
| 7 | `P7-object-detection-boxes` | Find objects with boxes: what a box and IoU are, a tiny detector for one shape per picture, a grid detector with a score threshold and NMS for several shapes, then a pretrained detector on street photos. |
| 8 | `P8-vision-transformer` | Cut a picture into patches, turn them into embeddings, watch attention at work, then assemble and train a tiny vision transformer. |
| bonus | `E1-flatten-branches-and-sobel-filters` | Two experiments after P1: a fully connected network that flattens the picture in several ways at once, and a CNN that starts from hand-made Sobel filters instead of random ones. No code to write. |
| bonus | `E2-fft-instead-of-convolution` | One experiment after P1 and E1: a layer that multiplies the picture's spectrum (FFT) by learned weights instead of sliding a filter, put in the place of every convolution and compared with the same CNN. No code to write. |

Every lesson uses the same small toolkit (`from cvcourse import practical as P`): a diagram
of any model you build (`P.show_model`), its cost in parameters and time (`P.cost`), a
predefined training run with live curves (`P.train`) and a leaderboard of the versions you
tried (`P.leaderboard`). MNIST and CIFAR-10 download themselves on first use; lessons 4, 5 and 8
need the Pet photos and lesson 7 the street photos from `python tools/download_data.py`.

## Students: start here

**1. Get the code and install** (once, before day 1; about 10 minutes plus downloads)

```bash
git clone <repository-url> cv-class
cd cv-class

python -m venv .venv
.venv\Scripts\activate              # Windows
source .venv/bin/activate           # macOS / Linux

pip install -r requirements.txt
python tools/download_data.py       # datasets, about 850 MB, safe to re-run
```

No GPU? Install the smaller CPU build of PyTorch first, then the rest:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

**2. Open the notebooks**

```bash
jupyter lab notebooks
```

VS Code works too: open the folder and pick the `.venv` interpreter as the kernel.

**3. Run `D0-00-setup-and-tensors.ipynb` from top to bottom before the course.** It checks
your installation and the datasets, shows that the interactive labs render on your machine,
and explains how every notebook works. If a cell fails there, fix it before day 1.

**4. During the course**, work through the notebooks in order, one cell at a time:

* Run the first cell of a notebook first (it sets everything up).
* A cell marked **Exercise** has a `raise NotImplementedError` where your code goes. The cell
  after it prints `[PASS]` or `[FAIL]` with a hint.
* A folded line **Functions used here (click to open)** sits before the first cell that uses a
  library function (`unsqueeze`, `permute`, `F.interpolate`, ...). Open it for what the function
  does, a runnable example and the classic mistake. The same text any time:
  `apiref.show("permute")`, `apiref.search("dimension")`, `apiref.index()`.
* A cell with `labs.show("...")` opens an interactive lab. Answer its **Predict first**
  question before you touch it, then work through its "Try this" list.
* Stuck for more than ten minutes? The finished version of every notebook is in
  `notebooks/solutions/`. Read the one cell you need, close it, and write it yourself.

**5. After each day**, take the quiz in `quizzes/` (answers are in the `-solutions` files)
and pick something from `docs/05-reading-list.md`.

### Where things are

```
notebooks/                 the course: D0-00, D1-01 ... D3-04, in order
notebooks/solutions/       the same notebooks with every exercise filled in
notebooks/cvcourse/        helper package the notebooks import (plots, checkers, data loading)
notebooks/cvcourse/labs/   the interactive labs; open index.html in a browser to see them all
quizzes/                   one quiz per day, with solutions
docs/05-reading-list.md    papers and posts for after the course
data/                      datasets and saved models; created by download_data.py, not in git
requirements.txt           what `pip install` reads
tools/                     scripts: download_data.py (you need this one), the rest builds the course
docs/                      course design and instructor material
```

### If something goes wrong

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: cvcourse` | Run the first cell of the notebook; start Jupyter from the repository (`jupyter lab notebooks`). |
| `... not found; run python tools/download_data.py` | Do that. It skips what is already downloaded. |
| `ModuleNotFoundError: torch` in the notebook | The notebook is not using the `.venv` kernel: activate the environment before `jupyter lab`, or select `.venv` in VS Code. |
| A lab shows an empty box | Run the cell again; if it stays empty, open `notebooks/cvcourse/labs/index.html` in a browser and use the lab there. |
| A training cell is slow | Each one stops by itself after a time limit. On a CPU the numbers will be a little lower than in the text; the conclusions hold. |
| `transformers`, `open_clip` or `onnx` will not install | They are optional. The notebooks that use them (the last block of each day) skip those cells and say so. |

## Interactive labs

Every mechanism that has a geometry (a kernel sliding, two boxes overlapping, anchors being
matched, a mask being pasted) has a lab: a small web page where you drag, paint and slide,
and the numbers update with the notebook's formula filled in. They need no packages and no
network.

| Day | Labs |
|---|---|
| 1 | convolution playground · receptive field · Mixup / CutMix · optimizer playground · LR schedule · block cost · gradient flow · freeze / unfreeze · Grad-CAM by hand |
| 2 | IoU playground · PR curve and AP · NMS · anchor matching · box encoding · RoI pooling vs RoIAlign · grid assignment · FPN levels · focal loss · Hungarian matching |
| 3 | segmentation metrics · transposed convolution · U-Net shapes · sliding window · mask paste-back · patches and attention · CLIP zero-shot |

```python
from cvcourse import labs
labs.show("iou-playground")     # embedded under the notebook cell
labs.open("index")              # all labs in a browser tab
labs.catalog()                  # the list, with the notebook each one belongs to
```

## Instructors and maintainers

* **Teaching it:** `docs/02-syllabus-3-day.md` (timetable), `docs/06-instructor-run-of-show.md`
  (what to show live in each block, where students get stuck, what to cut, the time budget
  for the labs), `docs/07-course-map.md` (every section, exercise and checker).
* **Saved models:** the training cells save to and load from `data/checkpoints/` (about 1 GB
  after a full run, not in git). Hand that folder to students with slow machines so the long
  trainings load instead of running.
* **Changing a notebook:** notebooks are generated; never edit an `.ipynb` by hand. Edit the
  source in `tools/src/`, then

  ```bash
  python tools/build_notebooks.py          # writes notebooks/*.ipynb and notebooks/solutions/*
  python tools/run_notebooks.py D1-01      # executes the solution notebook(s) to verify them
  ```
* **Function hint boxes:** the fold-out boxes come from one reference
  (`notebooks/cvcourse/apiref/`, about 300 entries) through `api(...)` cells in `tools/src/`.
  `python tools/check_api.py` runs every example and fails if a notebook calls a library
  function that is not in a box before its first use. Contract: `docs/09-function-reference-brief.md`.
* **Changing a lab:** read `notebooks/cvcourse/labs/README.md`; `python tools/check_labs.py`
  smoke-tests all labs in headless Chrome / Edge, `python tools/build_labs_index.py`
  regenerates their index.
* **Continuing the development:** [docs/HANDOVER.md](docs/HANDOVER.md) has the current state,
  the conventions and the open tasks.

The course is a PyTorch rewrite and refresh of three TensorFlow courses from 2024
(`GenomZ/DL-Image-B42`, `GenomZ/Advanced-CV`, `GenomZ/Advanced_Computer_Vision`); the
topic-by-topic mapping is in `docs/01-source-topic-inventory.md` and `docs/03-refresh-map.md`.

### Running on Google Colab

When opening a notebook directly from GitHub in Google Colab, Colab only loads the `.ipynb` file without the helper package (`notebooks/cvcourse`) or datasets. To set up the full environment in Colab:

1. Open any notebook from `notebooks/` in [Google Colab](https://google.com).
2. *(Optional)* Switch to a GPU runtime via **Runtime > Change runtime type > T4 GPU**.
3. Insert a new code cell at the **very top** of the notebook (above the first import cell) and run:

```python
%cd /content
!git clone https://github.com/GenomZ/cv-class

%cd /content/DSR_cv-class
!pip install -q -r requirements.txt
!python tools/download_data.py

%cd /content/DSR_cv-class/notebooks
```

4. Run the rest of the notebook cells from top to bottom. *(Note: Colab resets temporary storage when a runtime disconnects, so re-run this top setup cell whenever you start a new session.)*
