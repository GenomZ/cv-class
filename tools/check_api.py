"""Check the function reference and the "Functions used here" boxes.

    python tools/check_api.py                 # run every reference example, then check every notebook
    python tools/check_api.py D1-01           # only notebooks whose name contains D1-01
    python tools/check_api.py --inventory     # which notebook uses each function first (who writes its entry)
    python tools/check_api.py D0 --lines      # also print the source line of every finding
    python tools/check_api.py --examples day2 # only run the examples in apiref/_day2.py

Two checks:

1. Every entry in notebooks/cvcourse/apiref has an example that runs and prints exactly what the
   entry says (doctest). Course helpers (torchkit.*) may have no example.
2. Every library function a notebook calls in a code cell is explained in a box (`api(...)` in
   tools/src) that comes BEFORE that cell in the same notebook. "Library" = torch, torch.nn,
   torch.nn.functional, tensor / module / optimizer methods, torchvision, numpy, scipy, timm,
   einops and cvcourse.torchkit. Matplotlib, the standard library and the other cvcourse modules
   (viz, tests, labs, day1..3: figures, checkers, reference solutions) are out of scope.

A method call is attributed to tensors by its name alone (`.view(` -> Tensor.view), so a few
matches are wrong (`ax.scatter` is Matplotlib). Silence those in tools/api_ignore*.txt, one per
line: `D1-01 Tensor.scatter   # ax.scatter, Matplotlib` (`*` as the notebook = everywhere).
"""
from __future__ import annotations

import argparse
import ast
import doctest
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

SHORT = [("torch.nn.functional.", "F."), ("torch.nn.", "nn."), ("numpy.", "np."), ("cvcourse.torchkit.", "torchkit.")]
SCOPE = ("torch", "numpy", "torchvision", "scipy", "timm", "einops", "open_clip", "transformers", "cvcourse.torchkit")
# methods of things that are not tensors, by name
METHODS = {
    "eval": "Module.eval", "train": "Module.train", "parameters": "Module.parameters", "named_parameters": "Module.named_parameters",
    "state_dict": "Module.state_dict", "load_state_dict": "Module.load_state_dict", "children": "Module.children",
    "named_children": "Module.named_children", "modules": "Module.modules", "named_modules": "Module.named_modules",
    "register_forward_hook": "Module.register_forward_hook", "register_full_backward_hook": "Module.register_full_backward_hook",
    "requires_grad_": "Tensor.requires_grad_", "zero_grad": "Optimizer.zero_grad", "step": "Optimizer.step",
}
# names that tensors share with str / list / dict / file / Matplotlib objects: never guessed as tensor methods
NOT_TENSOR = set("""split index count copy items keys values get pop append extend sort format join strip replace lower upper
    startswith endswith add update clear remove insert reverse read write close type apply size fill fill_ map bar hist
    scatter where is_integer real imag""".split())
RECEIVER_SKIP = re.compile(r"^(ax\w*|axes|axs|fig\w*|plt|self|sys|os|pathlib|Path|json|re|time|math|random|viz|tests|labs|day[0-3])$")


def tensor_methods() -> set[str]:
    import torch
    return {n for n in dir(torch.Tensor) if not n.startswith("__")} - NOT_TENSOR


def shorten(full: str) -> str:
    for a, b in SHORT:
        if full.startswith(a):
            return b + full[len(a):]
    return full


def dotted(node) -> list[str] | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id, *reversed(parts)]
    return None


def calls_in(src: str, aliases: dict[str, str], tmeth: set[str]) -> list[tuple[str, int]]:
    """(key, line number) for every in-scope call in one code cell; also learns the cell's imports."""
    src = "\n".join("pass" if line.lstrip().startswith(("%", "!")) else line for line in src.splitlines())
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        d = dotted(node.func)
        if d and d[0] in aliases:
            full = ".".join([aliases[d[0]], *d[1:]])
            if full.startswith(SCOPE) and full not in SCOPE:
                # method on a module-level object, e.g. torch.cuda.is_available: keep the whole path
                out.append((shorten(full), node.lineno))
                continue
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
            base = d[0] if d else ""
            if d and (RECEIVER_SKIP.match(base) or base in aliases and not aliases[base].startswith(SCOPE)):
                continue
            if name in METHODS:
                out.append((METHODS[name], node.lineno))
            elif name in tmeth:
                out.append(("Tensor." + name, node.lineno))
    return out


def load_ignore() -> set[tuple[str, str]]:
    out = set()
    for f in sorted((ROOT / "tools").glob("api_ignore*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split("#")[0].split()
            if len(parts) == 2:
                out.add((parts[0], parts[1]))
    return out


def notebooks(pattern: str) -> list[Path]:
    return [p for p in sorted((ROOT / "notebooks" / "solutions").glob("*-solution.ipynb")) if pattern in p.name]


def scan(path: Path, tmeth: set[str]):
    """Yield (key, cell index, line text, boxed-before?) for the first call of every key in a notebook."""
    from cvcourse import apiref
    nb = nbformat.read(path, as_version=4)
    aliases: dict[str, str] = {}
    boxed: set[str] = set()
    seen: set[str] = set()
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown":
            for m in re.finditer(r'data-api="([^"]*)"', cell.source):
                boxed.update(k for k in m.group(1).split(",") if k)
        elif cell.cell_type == "code":
            lines = cell.source.splitlines()
            for key, ln in calls_in(cell.source, aliases, tmeth):
                if key in seen:
                    continue
                seen.add(key)
                yield key, i, lines[ln - 1].strip() if 0 < ln <= len(lines) else "", key in boxed, apiref.has(key) and apiref.get(key).key == key


def run_examples(only: str = "") -> int:
    import numpy as np
    import torch
    import torchvision
    from cvcourse import apiref, torchkit
    failed = 0
    parser, runner = doctest.DocTestParser(), doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS, verbose=False)
    no_example = []
    for key in apiref.keys():
        e = apiref.get(key)
        if only and e.day != only.lstrip("_"):
            continue
        if not e.example:
            if not key.startswith("torchkit."):
                no_example.append(key)
            continue
        globs = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional, "np": np, "torchvision": torchvision, "torchkit": torchkit}
        torch.manual_seed(0)
        test = parser.get_doctest(e.example, globs, key, None, 0)
        before = runner.failures
        runner.run(test, out=sys.__stdout__.write)
        failed += runner.failures > before
    print(f"reference: {len(apiref.keys())} entries, {failed} with a failing example" + (f", {len(no_example)} without example: {', '.join(no_example)}" if no_example else ""))
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", nargs="?", default="")
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--lines", action="store_true")
    ap.add_argument("--examples", metavar="DAY", help="only run the reference examples of one file, e.g. day1b, and stop")
    a = ap.parse_args()
    tmeth, ignore = tensor_methods(), load_ignore()

    if a.inventory:
        first: dict[str, str] = {}
        for path in notebooks(""):
            nbname = path.name[:5]
            for key, *_ in scan(path, tmeth):
                if (nbname, key) not in ignore and ("*", key) not in ignore:
                    first.setdefault(key, nbname)
        by: dict[str, list[str]] = {}
        for k, n in first.items():
            by.setdefault(n, []).append(k)
        for n in sorted(by):
            print(f"{n} ({len(by[n])}): " + ", ".join(sorted(by[n])))
        return 0

    if a.examples:
        return 1 if run_examples(a.examples) else 0
    bad = run_examples() if not a.pattern else 0
    for path in notebooks(a.pattern):
        nbname = path.name[:5]
        no_entry, no_box = [], []
        total = 0
        for key, cell, line, boxed, has in scan(path, tmeth):
            if (nbname, key) in ignore or ("*", key) in ignore:
                continue
            total += 1
            if not has:
                no_entry.append((key, cell, line))
            elif not boxed:
                no_box.append((key, cell, line))
        ok = not no_entry and not no_box
        bad += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {path.name[:-15]}: {total} functions, {len(no_entry)} without entry, {len(no_box)} not in a box before first use")
        for label, items in (("no entry", no_entry), ("no box before first use", no_box)):
            if items and not a.lines:
                print(f"    {label}: " + ", ".join(k for k, _, _ in items))
            for k, cell, line in items if a.lines else []:
                print(f"    {label}: {k:<34} cell {cell:<3} {line[:110]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
