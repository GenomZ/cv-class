"""Execute the solution notebooks headlessly to verify they run end to end.

    python tools/run_notebooks.py [pattern]

Executed copies are written to notebooks/solutions/_executed/ so the originals
stay output-free (small diffs in git).
"""
from __future__ import annotations

import os
import sys
import time

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOL = os.path.join(ROOT, "notebooks", "solutions")
OUT = os.path.join(SOL, "_executed")
os.makedirs(OUT, exist_ok=True)
pattern = sys.argv[1] if len(sys.argv) > 1 else ""

failed = []
for fname in sorted(os.listdir(SOL)):
    if not fname.endswith(".ipynb") or pattern not in fname:
        continue
    path = os.path.join(SOL, fname)
    nb = nbformat.read(path, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    t0 = time.time()
    try:
        ep.preprocess(nb, {"metadata": {"path": os.path.join(ROOT, "notebooks")}})
        nbformat.write(nb, os.path.join(OUT, fname))
        print(f"OK    {fname}  ({time.time() - t0:.1f}s)")
    except Exception as e:  # noqa: BLE001
        failed.append(fname)
        msg = str(e).splitlines()
        print(f"FAIL  {fname}\n      " + "\n      ".join(msg[:25]))

if failed:
    print("\nFailed:", ", ".join(failed))
    sys.exit(1)
print("\nAll notebooks executed.")
