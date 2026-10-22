"""Download KnowAir-V2 from Zenodo into examples/phase2/data/knowair_v2/.

Prefer system curl (better resume on Windows). Falls back to urllib.

Usage:
  python examples/phase2/_fetch_knowair_v2.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

RECORD_API = "https://zenodo.org/api/records/15614907"
RECORD_PAGE = "https://zenodo.org/records/15614907"
OUT_DIR = Path(__file__).resolve().parent / "data" / "knowair_v2"

# Human-facing browser/download-manager links
BROWSER_LINKS = {
    "dataset_bthsa.nc": f"{RECORD_PAGE}/files/dataset_bthsa.nc?download=1",
    "dataset_yrd.nc": f"{RECORD_PAGE}/files/dataset_yrd.nc?download=1",
    "stations_bthsa.csv": f"{RECORD_PAGE}/files/stations_bthsa.csv?download=1",
    "stations_yrd.csv": f"{RECORD_PAGE}/files/stations_yrd.csv?download=1",
}


def _list_files() -> list[dict]:
    with urllib.request.urlopen(RECORD_API, timeout=120) as resp:
        data = json.load(resp)
    return list(data.get("files") or [])


def _download_curl(url: str, dest: Path) -> None:
    cmd = [
        "curl",
        "-L",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "-C",
        "-",
        "--progress-bar",
        "-o",
        str(dest),
        url,
    ]
    print(" ", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def _download_urllib(url: str, dest: Path) -> None:
    print(f"  urllib -> {dest.name}", flush=True)
    urllib.request.urlretrieve(url, dest)


def main() -> int:
    print(f"Record page: {RECORD_PAGE}", flush=True)
    print("Browser / IDM direct links:", flush=True)
    for name, link in BROWSER_LINKS.items():
        print(f"  - {name}: {link}", flush=True)

    files = _list_files()
    if not files:
        print("No files on Zenodo record.", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    use_curl = shutil.which("curl") is not None
    print(f"\nSaving to {OUT_DIR} (curl={use_curl})", flush=True)

    for f in files:
        name = f["key"]
        size = int(f.get("size") or 0)
        url = f["links"]["self"]
        dest = OUT_DIR / name
        if dest.exists() and dest.stat().st_size == size:
            print(f"SKIP {name}", flush=True)
            continue
        print(f"GET {name} ({size / 1e6:.1f} MB)", flush=True)
        try:
            if use_curl:
                _download_curl(url, dest)
            else:
                _download_urllib(url, dest)
        except Exception as exc:
            print(f"FAILED {name}: {exc}", file=sys.stderr, flush=True)
            print(f"Please download manually:\n  {BROWSER_LINKS.get(name, url)}", flush=True)
            return 1
        got = dest.stat().st_size if dest.exists() else 0
        if size and got != size:
            print(f"WARN size mismatch {name}: got {got}, expect {size}", flush=True)
        else:
            print(f"OK {name}", flush=True)

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
