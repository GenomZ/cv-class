"""Function reference for the course: every library function the notebooks call, explained once.

In a notebook:

    from cvcourse import apiref
    apiref.show("permute", "unsqueeze")     # what they do, with a runnable example
    apiref.search("dimension")              # every entry that mentions a word
    apiref.index()                          # all entries, grouped by library

The notebooks' fold-out "Functions used here" boxes are generated from the same entries
(`tools/nb.py: api(...)`), so a function is explained with the same words everywhere.

Entries live in `_day0.py`, `_day1a.py`, `_day1b.py`, `_day2.py`, `_day3.py`: the file of the day that uses the
function first:

    E("Tensor.permute", "x.permute(*dims)",
      "Reorders the dimensions; no data is copied. ...",
      '''
      >>> x = torch.zeros(32, 48, 3)       # H, W, C
      >>> x.permute(2, 0, 1).shape         # C, H, W
      torch.Size([3, 32, 48])
      ''',
      note="...", see=["Tensor.transpose"])

The example is a doctest: `python tools/check_api.py` runs every one of them.
"""
from __future__ import annotations

import html as _html
import importlib
import textwrap
from dataclasses import dataclass, field

__all__ = ["E", "Entry", "get", "has", "keys", "box", "show", "search", "index"]

_MODULES = ["_day0", "_day1a", "_day1b", "_day1b_02", "_day1b_03", "_day1b_04", "_day2", "_day3"]   # _day1a = D1-01, _day1b = D1-02 .. D1-04
# key prefix -> heading used by index()
_GROUPS = [
    ("Tensor.", "Tensor methods and attributes"), ("torch.", "torch"), ("nn.", "torch.nn (layers)"),
    ("F.", "torch.nn.functional"), ("Module.", "nn.Module methods"), ("Optimizer.", "optimizers and schedulers"),
    ("torchvision.", "torchvision"), ("np.", "NumPy"), ("ndarray.", "NumPy"), ("torchkit.", "course helpers (cvcourse.torchkit)"),
]


@dataclass
class Entry:
    key: str                 # canonical name: "Tensor.permute", "torch.stack", "nn.Conv2d", "F.cross_entropy", "np.pad"
    sig: str                 # how you call it, as you would type it: "x.permute(*dims)"
    what: str                # one to three sentences: what it does, and what it is for in this course
    example: str = ""        # doctest (">>> " lines and their real output); may be empty for course helpers
    note: str = ""           # the classic mistake, if there is one
    see: list = field(default_factory=list)
    day: str = ""


def E(key, sig, what, example="", note="", see=()):
    return Entry(key, sig, " ".join(what.split()), textwrap.dedent(example).strip("\n"), " ".join(note.split()), list(see))


_REG: dict[str, Entry] | None = None


def _registry() -> dict[str, Entry]:
    global _REG
    if _REG is None:
        reg: dict[str, Entry] = {}
        for name in _MODULES:
            try:
                mod = importlib.import_module(f"{__name__}.{name}")
            except ModuleNotFoundError:
                continue
            for e in getattr(mod, "ENTRIES", []):
                e.day = name.lstrip("_")
                reg.setdefault(e.key, e)        # the earliest day's wording wins
        _REG = reg
    return _REG


def keys() -> list[str]:
    return list(_registry())


def _resolve(name: str) -> str | None:
    reg = _registry()
    if name in reg:
        return name
    short = [k for k in reg if k.split(".")[-1] == name]
    if len(short) == 1:
        return short[0]
    for prefix in ("Tensor.", "torch.", "nn.", "F.", "np."):       # "permute" -> Tensor.permute before torch.permute
        if prefix + name in reg:
            return prefix + name
    return None


def has(name: str) -> bool:
    return _resolve(name) is not None


def get(name: str) -> Entry:
    k = _resolve(name)
    if k is None:
        raise KeyError(f"no reference entry for {name!r}; try apiref.search({name.split('.')[-1]!r})")
    return _registry()[k]


def _entry_html(e: Entry, brief: bool) -> str:
    esc = _html.escape
    if brief:
        return f"<li><code>{esc(e.sig)}</code> &nbsp; {esc(e.what)}</li>"
    s = f"<p><b><code>{esc(e.sig)}</code></b><br>{esc(e.what)}</p>"
    if e.example:
        s += f"<pre>{esc(e.example)}</pre>"
    if e.note:
        s += f"<p><i>Watch out:</i> {esc(e.note)}</p>"
    if e.see:
        s += "<p>See also: " + ", ".join(f"<code>{esc(x)}</code>" for x in e.see) + "</p>"
    return s


def box(names, title: str | None = None, brief: bool = False, open: bool = False, missing: list | None = None) -> str:
    """HTML of one fold-out box. `brief` = one line per function (for functions met in an earlier notebook)."""
    found, lost = [], []
    for n in names:
        k = _resolve(n)
        (found if k else lost).append(k or n)
    if missing is not None:
        missing.extend(lost)
    reg = _registry()
    title = title or ("Functions from earlier notebooks, used again here" if brief else "Functions used here")
    label = " · ".join(f"<code>{_html.escape(k)}</code>" for k in found + lost)
    body = "".join(_entry_html(reg[k], brief) for k in found)
    if brief:
        body = f"<ul>{body}</ul>"
    body += "".join(f"<p><code>{_html.escape(n)}</code>: no entry yet</p>" for n in lost)
    hint = "" if brief else '<p><small>Any of these again later: <code>apiref.show("name")</code>.</small></p>'
    return (f'<details class="api-ref" data-api="{_html.escape(",".join(found + lost))}"{" open" if open else ""}>'
            f"<summary><b>{_html.escape(title)}</b> (click to open): {label}</summary>\n{body}{hint}</details>")


def show(*names, brief: bool = False):
    """Display the reference entries for these functions under the cell."""
    from IPython.display import HTML, display
    display(HTML(box(names, title="Reference", brief=brief, open=True)))


def search(text: str):
    """Entries whose name or description contains `text`."""
    t = text.lower()
    hits = [e for e in _registry().values() if t in e.key.lower() or t in e.what.lower() or t in e.sig.lower()]
    for e in hits:
        print(f"{e.key:<34} {e.what[:90]}")
    if not hits:
        print("nothing found")


def index():
    """All entries, grouped by library."""
    from IPython.display import HTML, display
    reg, out, used = _registry(), [], set()
    for prefix, heading in _GROUPS:
        ks = sorted(k for k in reg if k.startswith(prefix) and k not in used)
        used.update(ks)
        if ks:
            out.append(f"<p><b>{_html.escape(heading)}</b><br>" + " · ".join(f"<code>{_html.escape(k)}</code>" for k in ks) + "</p>")
    rest = sorted(k for k in reg if k not in used)
    if rest:
        out.append("<p><b>other</b><br>" + " · ".join(f"<code>{_html.escape(k)}</code>" for k in rest) + "</p>")
    display(HTML("".join(out) + f"<p><small>{len(reg)} entries. apiref.show(\"name\") opens one.</small></p>"))
