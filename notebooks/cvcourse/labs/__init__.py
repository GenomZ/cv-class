"""Interactive labs: small browser widgets that let you *play* with one mechanism.

Each lab is a self-contained HTML page in this folder (plain JavaScript, no packages,
works offline). In a notebook:

    from cvcourse import labs
    labs.show("conv-playground")            # embeds the lab under the cell
    labs.show("iou-playground", mode="giou")  # start it in a given state
    labs.catalog()                          # list every lab with its notebook
    labs.open("conv-playground")            # full window in your browser (good for projecting)

The same pages can be opened directly: double-click `notebooks/cvcourse/labs/index.html`.
"""
from __future__ import annotations

import html as _html
import json
import re
import webbrowser
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
_META = re.compile(r'<meta\s+name="lab-([a-z]+)"\s+content="([^"]*)"', re.I)


def names() -> list[str]:
    """Identifiers of all labs, in file order."""
    return sorted(p.stem for p in LAB_DIR.glob("*.html") if p.stem != "index")


def meta(name: str) -> dict:
    """Metadata declared in the lab's <meta name="lab-..."> tags: notebook, summary, height; plus title."""
    text = (LAB_DIR / f"{name}.html").read_text(encoding="utf-8")
    out = {k.lower(): _html.unescape(v) for k, v in _META.findall(text)}
    m = re.search(r"<title>(.*?)</title>", text, re.S)
    out["title"] = _html.unescape(m.group(1).strip()) if m else name
    out["name"] = name
    return out


def catalog(verbose: bool = True) -> list[dict]:
    """All labs sorted by notebook. Prints a table unless verbose=False."""
    rows = sorted((meta(n) for n in names()), key=lambda r: (r.get("notebook", "~"), r["name"]))
    if verbose:
        for r in rows:
            print(f"{r.get('notebook', ''):<7} {r['name']:<24} {r['title']}")
    return rows


def _compose(name: str, state: dict | None = None) -> str:
    """The lab's HTML with lab.css / lab.js inlined, so it works inside an iframe with no base URL."""
    text = (LAB_DIR / f"{name}.html").read_text(encoding="utf-8")
    css = (LAB_DIR / "lab.css").read_text(encoding="utf-8")
    js = (LAB_DIR / "lab.js").read_text(encoding="utf-8")
    init = f"<script>window.LAB_STATE = {json.dumps(state or {})};</script>\n"
    text = text.replace('<link rel="stylesheet" href="lab.css">', f"<style>\n{css}\n</style>")
    text = text.replace('<script src="lab.js"></script>', f"{init}<script>\n{js}\n</script>")
    return text


def show(name: str, height: int | None = None, **state):
    """Embed a lab below the current cell. Keyword arguments set its initial state."""
    from IPython.display import HTML

    if not (LAB_DIR / f"{name}.html").exists():
        return HTML(f"<div><b>Lab '{_html.escape(name)}' not found.</b> Available: {', '.join(names())}</div>")
    info = meta(name)
    h = int(height or info.get("height") or 620)
    doc = _html.escape(_compose(name, state), quote=True)
    return HTML(
        f'<div class="cv-lab"><iframe srcdoc="{doc}" title="{_html.escape(info["title"], quote=True)}" '
        f'style="width:100%;height:{h}px;border:1px solid rgba(128,128,128,.35);border-radius:10px" '
        f'loading="lazy"></iframe></div>'
    )


def open(name: str = "index") -> str:  # noqa: A001 - mirrors webbrowser.open
    """Open a lab (or the index of all labs) in the default browser. Returns the file URL."""
    url = (LAB_DIR / f"{name}.html").as_uri()
    webbrowser.open(url)
    return url
