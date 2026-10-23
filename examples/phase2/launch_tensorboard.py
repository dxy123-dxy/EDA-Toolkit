"""Launch TensorBoard to replay completed UniST training curves (demo playback).

Does not retrain. Reads event files under third_party/UniST/src/logs/.

Usage (repo root):
  .venv-ml\\Scripts\\python.exe examples/phase2/launch_tensorboard.py
  .venv-ml\\Scripts\\python.exe examples/phase2/launch_tensorboard.py --preset long
  .venv-ml\\Scripts\\python.exe examples/phase2/launch_tensorboard.py --preset prompt --port 6007
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS_ROOT = ROOT / "third_party" / "UniST" / "src" / "logs"
TB_EXE = ROOT / ".venv-ml" / "Scripts" / "tensorboard.exe"

# Relative to LOGS_ROOT — demo narrative presets
PRESETS: dict[str, str | None] = {
    "all": None,  # entire logs root
    "long": "Pretrain_Dataset_TrafficKnowAirLong_Task_short_FewRatio_1.0",
    "prompt": "Prompt_Dataset_TrafficKnowAirYRDLong_Task_short_FewRatio_0.5",
    "scratch": "Pretrain_Dataset_TrafficKnowAirYRDLong_Task_short_FewRatio_0.5",
}

HINTS: dict[str, str] = {
    "long": (
        "Missing Long pretrain logs. Re-run:\n"
        "  .\\.venv-ml\\Scripts\\python.exe examples\\phase2\\run_unist_pretrain_long.py"
    ),
    "prompt": (
        "Missing YRD Prompt logs. Re-run:\n"
        "  .\\.venv-ml\\Scripts\\python.exe examples\\phase2\\run_unist_prompt_cross.py "
        "--from-long --epochs 5"
    ),
    "scratch": (
        "Missing YRD Scratch logs. Re-run with --also-scratch:\n"
        "  .\\.venv-ml\\Scripts\\python.exe examples\\phase2\\run_unist_prompt_cross.py "
        "--from-long --epochs 5 --also-scratch"
    ),
    "all": (
        "No TensorBoard event files under third_party/UniST/src/logs/.\n"
        "Train first, e.g.:\n"
        "  .\\.venv-ml\\Scripts\\python.exe examples\\phase2\\run_unist_pretrain_long.py\n"
        "  .\\.venv-ml\\Scripts\\python.exe examples\\phase2\\run_unist_prompt_cross.py "
        "--from-long --epochs 5 --also-scratch"
    ),
}


def _has_events(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return any(directory.rglob("events.out.tfevents*"))


def _resolve_tensorboard() -> list[str]:
    if TB_EXE.is_file():
        return [str(TB_EXE)]
    which = shutil.which("tensorboard")
    if which:
        return [which]
    # Fallback: python -m tensorboard
    py = ROOT / ".venv-ml" / "Scripts" / "python.exe"
    if py.is_file():
        return [str(py), "-m", "tensorboard.main"]
    return [sys.executable, "-m", "tensorboard.main"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="all",
        help="Which completed run to open (default: all runs under logs/)",
    )
    ap.add_argument(
        "--logdir",
        type=Path,
        default=None,
        help="Override log directory (absolute or relative to repo root)",
    )
    ap.add_argument("--port", type=int, default=6006)
    ap.add_argument("--host", default="localhost")
    args = ap.parse_args()

    if args.logdir is not None:
        logdir = args.logdir if args.logdir.is_absolute() else ROOT / args.logdir
        hint_key = "all"
    else:
        rel = PRESETS[args.preset]
        logdir = LOGS_ROOT if rel is None else LOGS_ROOT / rel
        hint_key = args.preset

    if not _has_events(logdir):
        print(f"ERROR: no events.out.tfevents* under:\n  {logdir}", file=sys.stderr)
        print(HINTS.get(hint_key, HINTS["all"]), file=sys.stderr)
        return 1

    runs = sorted({p.parent.name for p in logdir.rglob("events.out.tfevents*")})
    print(f"logdir: {logdir}")
    print(f"runs ({len(runs)}): {', '.join(runs)}")
    print("Look at: Training/Loss_step, Test_RMSE/*, Test_MAE/*, Evaluation/*")
    print("(Replay of finished runs — not live training.)")
    url = f"http://{args.host}:{args.port}"
    print(f"Opening TensorBoard → {url}")

    cmd = _resolve_tensorboard() + [
        "--logdir",
        str(logdir),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    print(">>", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
