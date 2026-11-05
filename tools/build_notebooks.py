"""Build every notebook: `python tools/build_notebooks.py [pattern]`.

Each module in tools/src/ defines CELLS and NAME; this script writes the
student notebook to notebooks/ and the solution to notebooks/solutions/.
"""
from __future__ import annotations

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "src"))

from nb import build  # noqa: E402

SRC = os.path.join(HERE, "src")
pattern = sys.argv[1] if len(sys.argv) > 1 else ""

for fname in sorted(os.listdir(SRC)):
    if not fname.endswith(".py") or fname.startswith("_") or pattern not in fname:
        continue
    mod = importlib.import_module(fname[:-3])
    student, solution = build(mod.NAME, mod.CELLS)
    print(f"built {os.path.relpath(student)}  (+ solution, {len(mod.CELLS)} cells)")
