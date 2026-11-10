"""Smoke-test the interactive labs in a headless browser.

    python tools/check_labs.py                     # every lab: load it, fail on script errors, save a screenshot
    python tools/check_labs.py iou                 # only labs whose name contains "iou"
    python tools/check_labs.py conv --state "kernel=sharpen&stride=2" --theme dark --width 700

Screenshots go to notebooks/cvcourse/labs/_shots/ (ignored by git) unless --out is given.
A lab "fails" when lab.js had to show its red error banner, or when the page is missing the
pieces every lab must have (see notebooks/cvcourse/labs/README.md). Needs Chrome or Edge.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABS = ROOT / "notebooks" / "cvcourse" / "labs"

CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser", "msedge",
]


def find_browser() -> str:
    for c in CANDIDATES:
        if os.path.isfile(c):
            return c
        w = shutil.which(c)
        if w:
            return w
    sys.exit("No Chrome / Edge found; open the labs by hand instead.")


def run(browser: str, args: list[str], timeout: int = 90) -> str:
    with tempfile.TemporaryDirectory() as profile:
        cmd = [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
               "--allow-file-access-from-files", f"--user-data-dir={profile}", "--virtual-time-budget=4000", *args]
        try:
            p = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ""
        return p.stdout.decode("utf-8", "replace")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", nargs="?", default="")
    ap.add_argument("--state", default="", help='initial state as a URL hash, e.g. "stride=2&pad=0"')
    ap.add_argument("--out", default=str(LABS / "_shots"))
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--suffix", default="", help="appended to the screenshot file name")
    a = ap.parse_args()

    browser = find_browser()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for page in sorted(LABS.glob("*.html")):
        if a.pattern not in page.stem:
            continue
        url = page.as_uri() + f"#theme={a.theme}" + (f"&{a.state}" if a.state else "")
        extra: list[str] = []
        dom = run(browser, [*extra, "--dump-dom", url])
        shot = out / f"{page.stem}{a.suffix}{'-dark' if a.theme == 'dark' else ''}.png"
        run(browser, [*extra, f"--window-size={a.width},{a.height}", f"--screenshot={shot}", url])
        problems = []
        if not dom:
            problems.append("page did not load")
        err = re.search(r'class="lab-error"[^>]*>(.*?)</div>', dom, re.S)
        if err:
            problems.append(err.group(1).strip())
        if page.stem != "index":
            for needle, what in [('name="lab-notebook"', "lab-notebook meta"), ('name="lab-summary"', "lab-summary meta"),
                                 ('class="try"', '"Try this" section'), ("<canvas", "a canvas")]:
                if dom and needle not in dom:
                    problems.append(f"missing {what}")
        if problems:
            failed += 1
            print(f"[FAIL] {page.stem}: " + "; ".join(problems))
        else:
            print(f"[PASS] {page.stem}  -> {shot}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
