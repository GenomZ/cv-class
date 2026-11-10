"""Generate notebooks/cvcourse/labs/index.html from the <meta> tags of every lab.

    python tools/build_labs_index.py
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

from cvcourse import labs  # noqa: E402

DAYS = {
    "D0": "Before the course",
    "D1": "Day 1 · ConvNets: from pixels to transfer learning",
    "D2": "Day 2 · Object detection",
    "D3": "Day 3 · Segmentation and the frontier",
}

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interactive labs</title>
<link rel="stylesheet" href="lab.css">
<style>
  .day {{ margin: 22px 0 8px; font-size: 13px; font-weight: 700; color: var(--ink-2); text-transform: uppercase; letter-spacing: .05em; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px; }}
  a.card {{ display: block; text-decoration: none; color: inherit; border: 1px solid var(--border); border-radius: 10px; padding: 11px 13px; background: var(--surface); }}
  a.card:hover {{ border-color: var(--accent); background: var(--wash); }}
  a.card b {{ display: block; font-size: 14.5px; margin: 3px 0 3px; }}
  a.card span.s {{ color: var(--ink-2); font-size: 12.5px; }}
</style>
</head>
<body class="standalone">
<div class="lab">
  <div class="lab-head"><h1>Interactive labs</h1><span class="lab-tag">{n} labs · offline · no install</span></div>
  <p class="lab-lede">One mechanism per page. Predict what will happen, then drag, paint and slide to check. Every lab is also
    embedded in its notebook with <code>labs.show("name")</code>.</p>
{body}
</div>
<script src="lab.js"></script>
</body>
</html>
"""


def main() -> None:
    rows = labs.catalog(verbose=False)
    parts = []
    for key, title in DAYS.items():
        day = [r for r in rows if r.get("notebook", "").startswith(key)]
        if not day:
            continue
        parts.append(f'  <div class="day">{html.escape(title)}</div>\n  <div class="cards">')
        for r in day:
            parts.append(
                f'    <a class="card" href="{r["name"]}.html"><span class="lab-tag">{html.escape(r.get("notebook", ""))}</span>'
                f'<b>{html.escape(r["title"])}</b><span class="s">{html.escape(r.get("summary", ""))}</span></a>'
            )
        parts.append("  </div>")
    out = labs.LAB_DIR / "index.html"
    out.write_text(PAGE.format(n=len(rows), body="\n".join(parts)), encoding="utf-8", newline="\n")
    print(f"wrote {out.relative_to(ROOT)} ({len(rows)} labs)")


if __name__ == "__main__":
    main()
