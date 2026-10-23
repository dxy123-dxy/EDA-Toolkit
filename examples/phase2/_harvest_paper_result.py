"""Harvest UniST result.txt into output/phase2/eval/paper_demo_{dataset}.json."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "phase2" / "eval"
UNIST_EXP = ROOT / "third_party" / "UniST" / "src" / "experiments"


def parse_result(text: str, dataset: str) -> dict:
    out: dict = {"raw": text, "test_masks": {}, "best_rmse_norm": None}
    m = re.search(r"best rmse:\s*([0-9.eE+-]+)", text)
    if m:
        out["best_rmse_norm"] = float(m.group(1))

    def _ingest(parsed: dict) -> None:
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

    # Prefer dict immediately after the last "test rmse:" line
    lines = text.splitlines()
    last_test_idx = max((i for i, ln in enumerate(lines) if "test rmse:" in ln), default=-1)
    candidates: list[str] = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("{") and dataset in s:
            candidates.append((i, s))
    if last_test_idx >= 0:
        after = [s for i, s in candidates if i > last_test_idx]
        if after:
            _ingest(ast.literal_eval(after[-1]))
            return out
    if candidates:
        _ingest(ast.literal_eval(candidates[-1][1]))
    return out


def harvest(dataset: str, few_ratio: float = 1.0, epochs: int = 3) -> Path:
    exp = UNIST_EXP / f"Pretrain_Dataset_{dataset}_Task_short_FewRatio_{few_ratio}"
    result_txt = exp / "result.txt"
    if not result_txt.exists():
        raise FileNotFoundError(result_txt)
    text = result_txt.read_text(encoding="utf-8", errors="replace")
    metrics = parse_result(text, dataset)
    ckpt = exp / "model_save" / "model_best.pkl"
    rep = {
        "dataset": dataset,
        "epochs": epochs,
        "few_ratio": few_ratio,
        "exit_code": 0,
        "json": str(ROOT / "third_party" / "UniST" / "dataset" / f"{dataset}_short.json"),
        "experiments": str(exp),
        "ckpt": str(ckpt) if ckpt.exists() else None,
        "result_txt": text,
        "metrics": metrics,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    key = dataset.lower()
    out = OUT / f"paper_demo_{key}.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    masks = metrics.get("test_masks") or {}
    lines = [
        f"# UniST Stage-1 demo - {dataset}\n\n",
        f"- epochs: {epochs}\n",
        f"- checkpoint: `{rep.get('ckpt')}`\n\n",
        "| Mask | RMSE | MAE |\n|------|------|-----|\n",
    ]
    for name in ("random", "temporal", "tube", "block"):
        m = masks.get(name, {})
        lines.append(
            f"| {name} | {m.get('rmse', float('nan')):.4f} | {m.get('mae', float('nan')):.4f} |\n"
        )
    (OUT / f"paper_demo_{key}.md").write_text("".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print("masks:", masks)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--few-ratio", type=float, default=1.0)
    args = ap.parse_args()
    harvest(args.dataset, few_ratio=args.few_ratio, epochs=args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
