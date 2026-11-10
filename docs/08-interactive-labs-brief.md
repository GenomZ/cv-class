# Interactive labs: build brief

Working brief for whoever builds or extends the interactive labs. The authoring contract
and toolkit reference are in `notebooks/cvcourse/labs/README.md`; the exemplar is
`notebooks/cvcourse/labs/conv-playground.html`. This file says *why* the labs exist, how
they are wired into the notebooks, and what "done" means.

## Why

The notebooks explain every mechanism with a generated figure, but a figure shows one
case. Students understand a mechanism when they can predict what happens if they change
something, and are then proven right or wrong within a second. Each lab is that loop for
one mechanism: **predict -> manipulate -> see the numbers -> read why**.

## How a lab is wired into a notebook

Sources are `tools/src/dX_YY_*.py` (never edit `.ipynb`). For each lab:

1. Add `labs` to the `from cvcourse import ...` line of the setup cell.
2. Place the lab **after** the static figure that introduces the mechanism and **before**
   the exercise in which the student implements it, as two cells:

   ```python
   md(r"""
   ### Lab · IoU playground
   **Predict first:** two boxes of the same size overlap by half of their width. Is the IoU 0.5?
   Drag the boxes to check, then work through the "Try this" list under the lab (3-4 minutes).
   """),
   code("""
   labs.show("iou-playground")
   """),
   ```

   The markdown names the lab, asks one concrete prediction question whose answer is
   surprising or commonly wrong, and gives a time box. Do not explain the answer in the
   notebook; the lab's "Try this" reveals it.
3. If the following exercise or text can refer back to the lab ("the green anchors you saw
   in the lab are the `labels == 1` entries"), add that sentence.
4. Rebuild and run only your notebooks:
   `.venv/Scripts/python tools/build_notebooks.py d2_01` and
   `.venv/Scripts/python tools/run_notebooks.py D2-01` (every checker must still print `[PASS]`).

## Clarity pass (bounded)

While you are in a notebook source, read it top to bottom as a student who knows neural
networks and backprop but has never done computer vision. Fix, with minimal edits in
`tools/src`:

* a term used before it is defined (add the half-sentence definition where it first appears);
* a figure with no "what to look at" sentence (add one directly after the figure cell);
* an exercise whose instructions leave the expected shapes / return type ambiguous;
* a missing bridge between sections ("we now have X; the missing piece is Y").

Do not restructure sections, rename or weaken checkers, change exercise solutions, or
alter numbers that came from measured training runs. At most about ten such edits per
notebook; list them in your report.

## Done means

* `python tools/check_labs.py <name>` prints `[PASS]`, and you have **looked at** screenshots
  in light and dark theme, at 1100 px and 700 px width, in the default state and at least two
  other states (`--state`), with no overlap, clipping, unreadable text or empty panel.
* Every number shown for at least one state was compared against the Python reference
  implementation in `notebooks/cvcourse/` (or `torch` / `torchvision`), and they agree.
* Drag / paint logic was exercised, e.g. with a throwaway wrapper page that dispatches
  synthetic `PointerEvent`s on the canvas before the screenshot, or by reasoning through
  the handlers with a state set via `--state`.
* The student and solution notebooks were rebuilt, the solution notebooks ran, all checkers pass.
