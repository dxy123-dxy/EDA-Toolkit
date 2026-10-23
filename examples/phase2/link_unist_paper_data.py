"""Link UniST official paper datasets into third_party/UniST/dataset/.

Default source (cross-drive): E:\\数据分析模型训练数据\\data_release
Links BikeNYC_short.json + TrafficHZ_short.json (symlink; copy fallback).

Usage (repo root):
  .venv-ml\\Scripts\\python.exe examples/phase2/link_unist_paper_data.py
  .venv-ml\\Scripts\\python.exe examples/phase2/link_unist_paper_data.py --also-list
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "third_party" / "UniST" / "dataset"
DEFAULT_SRC = Path(r"E:\数据分析模型训练数据\data_release")

# Demo D1 defaults (manageable size)
DEMO_FILES = ("BikeNYC_short.json", "TrafficHZ_short.json")


def _link_or_copy(src: Path, dst: Path, *, force: bool) -> str:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        if not force:
            kind = "symlink" if dst.is_symlink() else "file"
            return f"skip existing {kind}: {dst.name}"
        if dst.is_dir() and not dst.is_symlink():
            raise IsADirectoryError(f"refusing to replace directory: {dst}")
        dst.unlink()

    try:
        os.symlink(src, dst)
        return f"symlink → {src}"
    except OSError as exc:
        print(f"  symlink failed ({exc}); copying {src.name} …", flush=True)
        shutil.copy2(src, dst)
        return f"copied → {dst} ({dst.stat().st_size / 1e6:.1f} MB)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Official data_release folder")
    ap.add_argument(
        "--files",
        nargs="+",
        default=list(DEMO_FILES),
        help="JSON basenames to link",
    )
    ap.add_argument("--force", action="store_true", help="Replace existing links/files")
    ap.add_argument(
        "--also-list",
        action="store_true",
        help="Print inventory of all *_short.json under --src (no link)",
    )
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"ERROR: source not found: {args.src}", file=sys.stderr)
        print("Expected UniST paper JSONs (BikeNYC_short.json, …).", file=sys.stderr)
        return 1

    DEST.mkdir(parents=True, exist_ok=True)

    if args.also_list:
        print(f"Inventory under {args.src}:")
        for p in sorted(args.src.glob("*_short.json")):
            print(f"  {p.name:28s} {p.stat().st_size / 1e6:8.1f} MB")

    print(f"Linking into {DEST}")
    for name in args.files:
        src = args.src / name
        dst = DEST / name
        try:
            msg = _link_or_copy(src, dst, force=args.force)
            print(f"  {name}: {msg}")
        except FileNotFoundError:
            print(f"  ERROR missing: {src}", file=sys.stderr)
            return 1

    manifest = {
        "source": str(args.src.resolve()),
        "dest": str(DEST.resolve()),
        "files": args.files,
        "note": "Symlink preferred; copy used if symlink privilege unavailable (cross-drive).",
    }
    out = ROOT / "output" / "phase2" / "eval" / "paper_data_link.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        __import__("json").dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
