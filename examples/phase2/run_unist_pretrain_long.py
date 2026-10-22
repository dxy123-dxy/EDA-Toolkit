"""Longer UniST Stage-1 pretrain on KnowAir (larger window + more epochs).

Reads examples/phase2/configs/pretrain_long.json, builds tensors/JSON if needed,
then launches UniST main.py.

Usage (repo root):
  .venv-ml\\Scripts\\python.exe examples/phase2/run_unist_pretrain_long.py
  .venv-ml\\Scripts\\python.exe examples/phase2/run_unist_pretrain_long.py --skip-build
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG = Path(__file__).resolve().parent / "configs" / "pretrain_long.json"
PY = ROOT / ".venv-ml" / "Scripts" / "python.exe"
UNIST_SRC = ROOT / "third_party" / "UniST" / "src"
OUT_EVAL = ROOT / "output" / "phase2" / "eval"


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    print(">>", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=CFG)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    region = cfg["regions"]["pretrain"]
    h, w = cfg["grid_hw"]
    t0, t1 = cfg["time_start"], cfg["time_end"]
    stride = int(cfg["stride_hours"])
    epochs = int(args.epochs or cfg["epochs"])

    stem = f"knowair_{region}_{h}x{w}_{t0}_{t1}"
    npz = ROOT / "output" / "phase2" / "tensors" / f"{stem}.npz"
    meta = ROOT / "output" / "phase2" / "tensors" / f"{stem}.meta.json"
    ds_name = "TrafficKnowAirLong" if region == "bthsa" else f"TrafficKnowAir{region.upper()}Long"

    if not args.skip_build:
        code = _run(
            [
                str(PY),
                str(ROOT / "examples/phase2/build_knowair_grid.py"),
                "--region",
                region,
                "--grid",
                str(h),
                str(w),
                "--time-start",
                t0,
                "--time-end",
                t1,
                "--stride",
                str(stride),
            ],
            cwd=ROOT,
        )
        if code != 0:
            return code
        code = _run(
            [
                str(PY),
                str(ROOT / "examples/phase2/export_unist_json.py"),
                "--npz",
                str(npz),
                "--meta",
                str(meta),
                "--seq-len",
                str(cfg["seq_len"]),
                "--stride",
                str(cfg["window_stride"]),
                "--max-samples",
                str(cfg["max_samples"]),
                "--name",
                ds_name,
            ],
            cwd=ROOT,
        )
        if code != 0:
            return code
        # Also build transfer region JSON with same calendar window
        tr = cfg["regions"]["transfer"]
        tr_stem = f"knowair_{tr}_{h}x{w}_{t0}_{t1}"
        tr_npz = ROOT / "output" / "phase2" / "tensors" / f"{tr_stem}.npz"
        tr_meta = ROOT / "output" / "phase2" / "tensors" / f"{tr_stem}.meta.json"
        tr_name = "TrafficKnowAirYRDLong"
        code = _run(
            [
                str(PY),
                str(ROOT / "examples/phase2/build_knowair_grid.py"),
                "--region",
                tr,
                "--grid",
                str(h),
                str(w),
                "--time-start",
                t0,
                "--time-end",
                t1,
                "--stride",
                str(stride),
            ],
            cwd=ROOT,
        )
        if code != 0:
            return code
        code = _run(
            [
                str(PY),
                str(ROOT / "examples/phase2/export_unist_json.py"),
                "--npz",
                str(tr_npz),
                "--meta",
                str(tr_meta),
                "--seq-len",
                str(cfg["seq_len"]),
                "--stride",
                str(cfg["window_stride"]),
                "--max-samples",
                str(cfg["max_samples"]),
                "--name",
                tr_name,
            ],
            cwd=ROOT,
        )
        if code != 0:
            return code

    json_path = ROOT / "third_party" / "UniST" / "dataset" / f"{ds_name}_short.json"
    if not json_path.exists():
        print(f"Missing {json_path}", file=sys.stderr)
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
        ds_name,
        "--task",
        "short",
        "--size",
        str(cfg["size"]),
        "--mask_strategy_random",
        "batch",
        "--lr",
        str(cfg["lr"]),
        "--used_data",
        "single",
        "--prompt_ST",
        "0",
        "--his_len",
        str(cfg["his_len"]),
        "--pred_len",
        str(cfg["pred_len"]),
        "--few_ratio",
        str(cfg["few_ratio"]),
        "--total_epoches",
        str(epochs),
        "--lr_anneal_steps",
        str(epochs),
        "--log_interval",
        "5",
        "--batch_size_1",
        str(cfg["batch_size_1"]),
        "--batch_size_2",
        str(cfg["batch_size_1"]),
        "--early_stop",
        str(cfg["early_stop"]),
        "--process_name",
        "phase2_pretrain_long",
    ]
    code = _run(cmd, cwd=UNIST_SRC)
    if code != 0:
        return code

    # Snapshot result
    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    exp = UNIST_SRC / "experiments" / f"Pretrain_Dataset_{ds_name}_Task_short_FewRatio_{cfg['few_ratio']}"
    # few_ratio 1.0 may format as 1.0
    cands = list((UNIST_SRC / "experiments").glob(f"Pretrain_Dataset_{ds_name}_Task_short_FewRatio_*"))
    summary = {
        "config": cfg,
        "epochs_requested": epochs,
        "dataset": ds_name,
        "json": str(json_path),
        "npz": str(npz),
        "experiments": [str(c) for c in cands],
    }
    for c in cands:
        rt = c / "result.txt"
        if rt.exists():
            summary["result_txt"] = rt.read_text(encoding="utf-8", errors="ignore")
            summary["ckpt"] = str(c / "model_save" / "model_best.pkl")
    out = OUT_EVAL / "pretrain_long_bthsa.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
