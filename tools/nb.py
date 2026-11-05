"""Tiny notebook DSL: build student and solution .ipynb files from Python.

Usage inside a notebook source module (tools/src/*.py):

    from nb import md, code, exercise, framework, build

    CELLS = [
        md("# Title"),
        code("import numpy as np"),
        exercise(stub="def f():\n    # YOUR CODE HERE\n    raise NotImplementedError",
                 solution="def f():\n    return 1"),
        framework("model = ...  # PyTorch / TF model definition goes here"),
    ]
    build("D1-01-convolution", CELLS)

`exercise` cells differ between the student notebook and the solution notebook.
`framework` cells are placeholders for the deep-learning framework (pending the
PyTorch decision); they render as a commented code cell so "Run all" never fails.
"""
from __future__ import annotations

import os
import textwrap

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_DIR = os.path.join(ROOT, "notebooks")
SOLUTION_DIR = os.path.join(NOTEBOOK_DIR, "solutions")

FRAMEWORK_BANNER = (
    "# FRAMEWORK CELL - filled in once the framework decision (PyTorch vs TensorFlow) is final.\n"
    "# What this cell will do:\n"
)


def _dedent(s: str) -> str:
    return textwrap.dedent(s).strip("\n") + "\n"


def md(text: str) -> dict:
    return {"type": "md", "src": _dedent(text)}


def code(text: str, hide_output: bool = False) -> dict:
    return {"type": "code", "src": _dedent(text)}


def exercise(stub: str, solution: str, title: str | None = None) -> dict:
    return {"type": "exercise", "stub": _dedent(stub), "solution": _dedent(solution)}


def framework(description: str, fallback: str = "") -> dict:
    """A placeholder for framework-specific code. `fallback` runs today (optional)."""
    desc = "\n".join("#   " + line for line in _dedent(description).splitlines())
    src = FRAMEWORK_BANNER + desc + "\n"
    if fallback:
        src += "\n" + _dedent(fallback)
    return {"type": "code", "src": src}


def _header(title: str, steps, topics, minutes: int, prereq: str) -> str:
    s = f"# {title}\n\n"
    s += "### Steps\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(steps)) + "\n\n"
    s += "### Topics covered and learning objectives\n" + "\n".join(f"- {x}" for x in topics) + "\n\n"
    s += f"### Time estimate\n~{minutes} min in class (guided + hands-on)\n\n"
    s += f"### Prerequisites\n{prereq}\n"
    return s


def header(title, steps, topics, minutes, prereq="Neural networks and backpropagation (previous class)."):
    return md(_header(title, steps, topics, minutes, prereq))


def _to_nb(cells, solution: bool):
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for c in cells:
        if c["type"] == "md":
            nb.cells.append(new_markdown_cell(c["src"].rstrip("\n")))
        elif c["type"] == "code":
            nb.cells.append(new_code_cell(c["src"].rstrip("\n")))
        elif c["type"] == "exercise":
            src = c["solution"] if solution else c["stub"]
            cell = new_code_cell(src.rstrip("\n"))
            cell.metadata["tags"] = ["solution" if solution else "exercise"]
            nb.cells.append(cell)
        else:
            raise ValueError(c["type"])
    return nb


def build(name: str, cells) -> tuple[str, str]:
    os.makedirs(NOTEBOOK_DIR, exist_ok=True)
    os.makedirs(SOLUTION_DIR, exist_ok=True)
    student = os.path.join(NOTEBOOK_DIR, name + ".ipynb")
    solution = os.path.join(SOLUTION_DIR, name + "-solution.ipynb")
    nbformat.write(_to_nb(cells, False), student)
    nbformat.write(_to_nb(cells, True), solution)
    return student, solution
