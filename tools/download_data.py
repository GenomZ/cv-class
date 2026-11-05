"""Download every dataset the course uses into <repo>/data. Safe to re-run.

    python tools/download_data.py                 # everything
    python tools/download_data.py fashion pet     # only some: fashion | pet | pennfudan

What it fetches (about 850 MB in total):

* FashionMNIST            via torchvision           -> data/FashionMNIST/
* Oxford-IIIT Pet         images + annotations      -> data/oxford-iiit-pet/{images,annotations}/
  (https://thor.robots.ox.ac.uk/~vgg/data/pets/)
* Penn-Fudan pedestrians  PennFudanPed.zip          -> data/penn-fudan/PennFudanPed/

Each finished dataset gets a `.complete` marker; anything already present is skipped.
It also creates data/checkpoints/ where the notebooks save (and look for) trained weights.
"""
from __future__ import annotations

import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

PET_URLS = [
    "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz",
    "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz",
]
PENN_FUDAN_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"


def _download(url: str, dest: Path) -> Path:
    """Stream `url` to `dest` (via a .part file, so an interrupted download is never mistaken for a finished one)."""
    if dest.exists():
        print(f"  have {dest.name}")
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    t0 = time.time()
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print(f"\r    {done / 1e6:7.1f} MB in {time.time() - t0:.0f} s" + " " * 20)
    part.replace(dest)
    return dest


def _extract(archive: Path, into: Path) -> None:
    print(f"  extracting {archive.name}")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for m in z.namelist():                       # refuse absolute / parent paths
                if m.startswith(("/", "\\")) or ".." in Path(m).parts:
                    raise RuntimeError(f"unsafe path in {archive}: {m}")
            z.extractall(into)
    else:
        with tarfile.open(archive) as t:
            t.extractall(into, filter="data")


def fashion() -> None:
    print("FashionMNIST")
    marker = DATA / "FashionMNIST" / ".complete"
    if marker.exists():
        print("  already complete")
        return
    from torchvision import datasets

    for train in (True, False):
        datasets.FashionMNIST(str(DATA), train=train, download=True)
    marker.touch()
    print("  done")


def pet() -> None:
    print("Oxford-IIIT Pet")
    base = DATA / "oxford-iiit-pet"
    marker = base / ".complete"
    if marker.exists():
        print("  already complete")
        return
    base.mkdir(parents=True, exist_ok=True)
    for url, sub in zip(PET_URLS, ["images", "annotations"]):
        if (base / sub).is_dir() and any((base / sub).iterdir()) and (base / f".{sub}.extracted").exists():
            print(f"  have {sub}/")
            continue
        archive = _download(url, base / url.rsplit("/", 1)[1])
        _extract(archive, base)
        (base / f".{sub}.extracted").touch()
    n_img = len(list((base / "images").glob("*.jpg")))
    n_xml = len(list((base / "annotations" / "xmls").glob("*.xml")))
    print(f"  {n_img} images, {n_xml} head-box xmls")
    marker.touch()
    for name in ("images.tar.gz", "annotations.tar.gz"):
        (base / name).unlink(missing_ok=True)            # archives are no longer needed


def pennfudan() -> None:
    print("Penn-Fudan pedestrians")
    base = DATA / "penn-fudan"
    marker = base / ".complete"
    if marker.exists():
        print("  already complete")
        return
    base.mkdir(parents=True, exist_ok=True)
    if not (base / "PennFudanPed" / "PNGImages").is_dir():
        archive = _download(PENN_FUDAN_URL, base / "PennFudanPed.zip")
        _extract(archive, base)
        archive.unlink(missing_ok=True)
    shutil.rmtree(base / "__MACOSX", ignore_errors=True)
    print(f"  {len(list((base / 'PennFudanPed' / 'PNGImages').glob('*.png')))} images")
    marker.touch()


TASKS = {"fashion": fashion, "pet": pet, "pennfudan": pennfudan}

if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    (DATA / "checkpoints").mkdir(exist_ok=True)
    wanted = sys.argv[1:] or list(TASKS)
    unknown = [w for w in wanted if w not in TASKS]
    if unknown:
        sys.exit(f"unknown dataset(s) {unknown}; choose from {list(TASKS)}")
    for w in wanted:
        TASKS[w]()
    print(f"\nData folder: {DATA}\nCheckpoints: {DATA / 'checkpoints'} (the notebooks save trained weights here "
          f"and reuse them on re-runs)")
