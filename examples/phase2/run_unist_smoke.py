"""Smoke-run UniST Stage-1 pretrain on TrafficKnowAir (few epochs).

Prereq:
  1) build_knowair_grid.py
  2) export_unist_json.py
  3) patch_unist_windows.py

Usage (repo root):
  .venv-ml\\Scripts\\python.exe examples/phase2/run_unist_smoke.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNIST_SRC = ROOT / "third_party" / "UniST" / "src"
JSON = ROOT / "third_party" / "UniST" / "dataset" / "TrafficKnowAir_short.json"
PY = ROOT / ".venv-ml" / "Scripts" / "python.exe"


def main() -> int:
    if not JSON.exists():
        print(f"Missing {JSON}. Run export_unist_json.py first.", file=sys.stderr)
        return 1
    (UNIST_SRC / "experiments").mkdir(parents=True, exist_ok=True)
    (UNIST_SRC / "logs").mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PY),
        "main.py",
        "--device_id",
        "0",
        "--machine",
        "local",
        "--dataset",
        "TrafficKnowAir",
        "--task",
        "short",
        "--size",
        "middle",
        "--mask_strategy_random",
        "batch",
        "--lr",
        "3e-4",
        "--used_data",
        "single",
        "--prompt_ST",
        "0",
        "--his_len",
        "6",
        "--pred_len",
        "6",
        "--total_epoches",
        "2",
        "--lr_anneal_steps",
        "2",
        "--log_interval",
        "1",
        "--batch_size_1",
        "4",
        "--early_stop",
        "2",
        "--process_name",
        "phase2_smoke",
    ]
    print("cwd:", UNIST_SRC)
    print("cmd:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(UNIST_SRC))


if __name__ == "__main__":
    raise SystemExit(main())
