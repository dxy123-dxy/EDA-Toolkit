"""Short UniST Stage-1 pretrain on official paper datasets (Demo D1).

Default: BikeNYC (few epochs). Optional: --also-hz for TrafficHZ.

Prereq:
  .venv-ml\\Scripts\\python.exe examples/phase2/link_unist_paper_data.py
  .venv-ml\\Scripts\\python.exe examples/phase2/patch_unist_windows.py

Usage (repo root):
  .venv-ml\\Scripts\\python.exe examples/phase2/run_unist_paper_demo.py
  .venv-ml\\Scripts\\python.exe examples/phase2/run_unist_paper_demo.py --also-hz --epochs 3
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNIST_SRC = ROOT / "third_party" / "UniST" / "src"
DATASET = ROOT / "third_party" / "UniST" / "dataset"
PY = ROOT / ".venv-ml" / "Scripts" / "python.exe"
OUT_EVAL = ROOT / "output" / "phase2" / "eval"


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    print(">>", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


def _parse_result_txt(text: str) -> dict:
    """Extract last test mask RMSE/MAE dict from UniST result.txt."""
    out: dict = {"raw": text, "test_masks": {}, "best_rmse_norm": None}
    # best rmse line (normalized)
    m = re.search(r"best rmse:\s*([0-9.eE+-]+)", text)
    if m:
        out["best_rmse_norm"] = float(m.group(1))

    # Prefer last "test rmse" block dict
    blocks = list(re.finditer(r"test rmse:\s*[0-9.eE+-]+\s*\n(\{.*?\})", text, re.S))
    if not blocks:
        # fall back to any dict with mask keys
        blocks = list(re.finditer(r"(\{'[^']+':\s*\{.*?\}\})", text, re.S))
    if not blocks:
        return out

    blob = blocks[-1].group(1)
    try:
        parsed = ast.literal_eval(blob)
    except (SyntaxError, ValueError):
        return out

    # parsed: {dataset: {strategy: {ratio: rmse} or {ratio: {rmse, mae}}}}
    for _ds, masks in parsed.items():
        for strategy, ratios in masks.items():
            for ratio, val in ratios.items():
                if isinstance(val, dict):
                    out["test_masks"][strategy] = {
                        "rmse": float(val.get("rmse", float("nan"))),
                        "mae": float(val.get("mae", float("nan"))),
                        "mask_ratio": float(ratio),
                    }
                else:
                    out["test_masks"][strategy] = {
                        "rmse": float(val),
                        "mae": float("nan"),
                        "mask_ratio": float(ratio),
                    }
    return out


def _exp_dir(dataset: str, few_ratio: float) -> Path:
    folder = f"Pretrain_Dataset_{dataset}_Task_short_FewRatio_{few_ratio}"
    return UNIST_SRC / "experiments" / folder


def train_one(
    dataset: str,
    *,
    epochs: int,
    batch_size: int,
    few_ratio: float,
    device_id: str,
) -> dict:
    json_path = DATASET / f"{dataset}_short.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing {json_path}. Run link_unist_paper_data.py first."
        )

    (UNIST_SRC / "experiments").mkdir(parents=True, exist_ok=True)
    (UNIST_SRC / "logs").mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PY),
        "main.py",
        "--device_id",
        device_id,
        "--machine",
        "local",
        "--dataset",
        dataset,
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
        "--few_ratio",
        str(few_ratio),
        "--total_epoches",
        str(epochs),
        "--lr_anneal_steps",
        str(max(epochs, 2)),
        "--log_interval",
        "5",
        "--batch_size_1",
        str(batch_size),
        "--batch_size_2",
        str(max(2, batch_size // 2)),
        "--batch_size_3",
        str(max(1, batch_size // 4)),
        "--early_stop",
        str(max(epochs, 2)),
        "--process_name",
        f"phase2_paper_{dataset}",
    ]
    code = _run(cmd, cwd=UNIST_SRC)
    exp = _exp_dir(dataset, few_ratio)
    result_txt = exp / "result.txt"
    ckpt = exp / "model_save" / "model_best.pkl"
    parsed = _parse_result_txt(result_txt.read_text(encoding="utf-8", errors="replace")) if result_txt.exists() else {}

    report = {
        "dataset": dataset,
        "epochs": epochs,
        "few_ratio": few_ratio,
        "exit_code": code,
        "json": str(json_path),
        "experiments": str(exp),
        "ckpt": str(ckpt) if ckpt.exists() else None,
        "result_txt": result_txt.read_text(encoding="utf-8", errors="replace") if result_txt.exists() else "",
        "metrics": parsed,
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=3, help="Stage-1 epochs (demo smoke)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device-id", default="0")
    ap.add_argument("--also-hz", action="store_true", help="Also train TrafficHZ")
    ap.add_argument("--skip-patch", action="store_true")
    ap.add_argument("--skip-link", action="store_true")
    args = ap.parse_args()

    if not args.skip_link:
        code = _run([str(PY), str(ROOT / "examples/phase2/link_unist_paper_data.py")], cwd=ROOT)
        if code != 0:
            return code
    if not args.skip_patch:
        code = _run([str(PY), str(ROOT / "examples/phase2/patch_unist_windows.py")], cwd=ROOT)
        if code != 0:
            return code

    datasets = ["BikeNYC"]
    if args.also_hz:
        datasets.append("TrafficHZ")

    reports = []
    for ds in datasets:
        print(f"\n=== Stage-1 paper demo: {ds} ({args.epochs} epoch) ===\n", flush=True)
        # few_ratio=1.0 → pure pretrain (mode=training), matching KnowAir Long
        rep = train_one(
            ds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            few_ratio=1.0,
            device_id=args.device_id,
        )
        reports.append(rep)
        if rep["exit_code"] != 0:
            print(f"WARNING: {ds} exited {rep['exit_code']}", file=sys.stderr)

    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": "unist_paper_stage1_demo",
        "source": r"E:\数据分析模型训练数据\data_release",
        "note": (
            "Short Stage-1 on official UniST JSONs for Demo D1. "
            "Not a full paper SOTA reproduction; few-shot/zero-shot left as next step."
        ),
        "runs": reports,
    }
    out = OUT_EVAL / "paper_demo_bikenyc.json"
    # If HZ included, write combined + per-ds
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")

    for rep in reports:
        ds = rep["dataset"].lower()
        one = OUT_EVAL / f"paper_demo_{ds}.json"
        one.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {one}")

        # Markdown table for quick demo reading
        masks = (rep.get("metrics") or {}).get("test_masks") or {}
        lines = [
            f"# UniST Stage-1 demo — {rep['dataset']}\n",
            f"- epochs: {rep['epochs']}  \n",
            f"- checkpoint: `{rep.get('ckpt')}`  \n\n",
            "| Mask | RMSE | MAE |\n|------|------|-----|\n",
        ]
        for name in ("random", "temporal", "tube", "block"):
            m = masks.get(name, {})
            lines.append(
                f"| {name} | {m.get('rmse', float('nan')):.4f} | {m.get('mae', float('nan')):.4f} |\n"
            )
        md = OUT_EVAL / f"paper_demo_{ds}.md"
        md.write_text("".join(lines), encoding="utf-8")
        print(f"wrote {md}")

    bad = [r for r in reports if r["exit_code"] != 0]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
