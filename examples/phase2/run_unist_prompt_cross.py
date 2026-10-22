"""Cross-scene Prompt: BTHSA pretrained → YRD prompt-tuning.

Narrative (plan §2.2 / 附录 A):
  Region A (BTHSA) Stage-1 pretrain → Region B (YRD) Prompt / few-shot.

Usage (repo root):
  # Smoke weights (TrafficKnowAir → TrafficKnowAirYRD)
  .venv-ml\\Scripts\\python.exe examples/phase2/run_unist_prompt_cross.py --epochs 3 --also-scratch

  # Long pretrain weights (TrafficKnowAirLong → TrafficKnowAirYRDLong)
  .venv-ml\\Scripts\\python.exe examples/phase2/run_unist_prompt_cross.py --from-long --epochs 5 --also-scratch
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNIST_SRC = ROOT / "third_party" / "UniST" / "src"
PY = ROOT / ".venv-ml" / "Scripts" / "python.exe"
OUT = ROOT / "output" / "phase2" / "eval"

# Smoke defaults
SMOKE_CKPT_REL = Path(
    "experiments/Pretrain_Dataset_TrafficKnowAir_Task_short_FewRatio_0.5/model_save/model_best"
)
SMOKE_TARGET = "TrafficKnowAirYRD"
SMOKE_SOURCE = "TrafficKnowAir (BTHSA)"

# Long-pretrain defaults (aligned with configs/pretrain_long.json)
LONG_CKPT_REL = Path(
    "experiments/Pretrain_Dataset_TrafficKnowAirLong_Task_short_FewRatio_1.0/model_save/model_best"
)
LONG_TARGET = "TrafficKnowAirYRDLong"
LONG_SOURCE = "TrafficKnowAirLong (BTHSA 2022–2023)"


def _run(cmd: list[str]) -> int:
    print("cmd:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(UNIST_SRC))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=None, help="Prompt/scratch epochs (default: 3 smoke / 5 long)")
    p.add_argument("--few-ratio", type=float, default=0.5)
    p.add_argument("--also-scratch", action="store_true", help="Also train YRD from scratch for contrast")
    p.add_argument(
        "--from-long",
        action="store_true",
        help="Load TrafficKnowAirLong Stage-1 weights; target TrafficKnowAirYRDLong",
    )
    p.add_argument("--batch-size", type=int, default=None, help="Override batch_size_1/2 (default: 4 smoke / 8 long)")
    args = p.parse_args()

    if args.from_long:
        ckpt_rel = LONG_CKPT_REL
        target = LONG_TARGET
        source = LONG_SOURCE
        epochs = args.epochs if args.epochs is not None else 5
        batch = args.batch_size if args.batch_size is not None else 8
        out_name = "cross_scene_bthsa_long_to_yrd.json"
    else:
        ckpt_rel = SMOKE_CKPT_REL
        target = SMOKE_TARGET
        source = SMOKE_SOURCE
        epochs = args.epochs if args.epochs is not None else 3
        batch = args.batch_size if args.batch_size is not None else 4
        out_name = "cross_scene_bthsa_to_yrd.json"

    yrd_json = ROOT / "third_party" / "UniST" / "dataset" / f"{target}_short.json"
    ckpt_pkl = UNIST_SRC / f"{ckpt_rel}.pkl"

    if not yrd_json.exists():
        print(f"Missing {yrd_json}", file=sys.stderr)
        if args.from_long:
            print("Hint: run run_unist_pretrain_long.py (builds YRDLong JSON) or export_unist_json for YRDLong", file=sys.stderr)
        return 1
    if not ckpt_pkl.exists():
        print(f"Missing BTHSA checkpoint {ckpt_pkl}", file=sys.stderr)
        hint = "run_unist_pretrain_long.py" if args.from_long else "run_unist_smoke.py"
        print(f"Hint: run examples/phase2/{hint} first", file=sys.stderr)
        return 1

    (UNIST_SRC / "experiments").mkdir(parents=True, exist_ok=True)
    (UNIST_SRC / "logs").mkdir(parents=True, exist_ok=True)

    ckpt_arg = str(ckpt_rel).replace("\\", "/")

    prompt_cmd = [
        str(PY),
        "main.py",
        "--device_id", "0",
        "--machine", "local",
        "--dataset", target,
        "--task", "short",
        "--size", "middle",
        "--prompt_ST", "1",
        "--pred_len", "6",
        "--his_len", "6",
        "--num_memory_spatial", "256",
        "--num_memory_temporal", "256",
        "--prompt_content", "s_p_c",
        "--lr", "3e-4",
        "--used_data", "single",
        "--file_load_path", ckpt_arg,
        "--few_ratio", str(args.few_ratio),
        "--total_epoches", str(epochs),
        "--lr_anneal_steps", str(epochs),
        "--log_interval", "1",
        "--batch_size_1", str(batch),
        "--batch_size_2", str(batch),
        "--early_stop", "5",
        "--process_name", "phase2_cross_prompt_long" if args.from_long else "phase2_cross_prompt",
    ]
    code = _run(prompt_cmd)
    if code != 0:
        return code

    scratch_code = None
    if args.also_scratch:
        scratch_cmd = [
            str(PY),
            "main.py",
            "--device_id", "0",
            "--machine", "local",
            "--dataset", target,
            "--task", "short",
            "--size", "middle",
            "--mask_strategy_random", "batch",
            "--lr", "3e-4",
            "--used_data", "single",
            "--prompt_ST", "0",
            "--his_len", "6",
            "--pred_len", "6",
            "--few_ratio", str(args.few_ratio),
            "--total_epoches", str(epochs),
            "--lr_anneal_steps", str(epochs),
            "--log_interval", "1",
            "--batch_size_1", str(batch),
            "--early_stop", "5",
            "--process_name", "phase2_yrd_scratch_long" if args.from_long else "phase2_yrd_scratch",
        ]
        scratch_code = _run(scratch_cmd)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_pretrain": source,
        "target_prompt": f"{target} (YRD)",
        "from_long": bool(args.from_long),
        "few_ratio": args.few_ratio,
        "epochs": epochs,
        "batch_size": batch,
        "file_load_path": ckpt_arg,
        "prompt_result_dir": None,
        "scratch_result_dir": None,
    }
    for d in sorted((UNIST_SRC / "experiments").glob(f"Prompt_*{target}*"), key=lambda x: x.stat().st_mtime):
        summary["prompt_result_dir"] = str(d)
        rt = d / "result.txt"
        if rt.exists():
            summary["prompt_result_txt"] = rt.read_text(encoding="utf-8", errors="ignore")
    if args.also_scratch:
        for d in sorted((UNIST_SRC / "experiments").glob(f"Pretrain_*{target}*"), key=lambda x: x.stat().st_mtime):
            summary["scratch_result_dir"] = str(d)
            rt = d / "result.txt"
            if rt.exists():
                summary["scratch_result_txt"] = rt.read_text(encoding="utf-8", errors="ignore")

    out_json = OUT / out_name
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")
    if scratch_code not in (None, 0):
        return scratch_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
