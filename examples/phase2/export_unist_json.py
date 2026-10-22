"""Export KnowAir grid npz → UniST dataset JSON (short task).

UniST expects (see third_party/UniST dataset README + DataLoader):
  X_*[0]: (N, T, H, W)
  X_*[1]: (N, T, P, H, W) periodical features
  timestamps: for names matching 'Traffic*', a numeric tensor (N, T, 2)

We name the dataset TrafficKnowAir so the existing Traffic* timestamp branch applies.

Usage:
  .venv-ml\\Scripts\\python.exe examples/phase2/export_unist_json.py \\
    --npz output/phase2/tensors/knowair_bthsa_16x16_2023-01-01_2023-01-31.npz \\
    --meta  output/phase2/tensors/knowair_bthsa_16x16_2023-01-01_2023-01-31.meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

UNIST_DATASET = Path(__file__).resolve().parents[2] / "third_party" / "UniST" / "dataset"
SEQ_LEN = 12  # his_len 6 + pred_len 6
P_PERIOD = 2


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--stride", type=int, default=1, help="Sliding window stride on time axis")
    p.add_argument("--max-samples", type=int, default=120, help="Cap total windows before split")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--name", default="TrafficKnowAir")
    return p.parse_args()


def _make_windows(
    grid_thw: np.ndarray,
    times: pd.DatetimeIndex,
    seq_len: int,
    stride: int,
    max_samples: int,
) -> tuple[np.ndarray, list, np.ndarray]:
    """grid_thw: (T, H, W) → windows (N, seq_len, H, W)."""
    t_len = grid_thw.shape[0]
    xs, tss, periods = [], [], []
    for start in range(0, t_len - seq_len + 1, stride):
        end = start + seq_len
        win = grid_thw[start:end]
        ts = []
        for t in times[start:end]:
            # (weekday, half-hour index) compatible with Traffic* numeric path
            half = int(t.hour) * 2 + (1 if t.minute >= 30 else 0)
            ts.append([int(t.weekday()), half])
        # Period: previous seq_len blocks if available, else repeat win
        period_stack = []
        for p in range(1, P_PERIOD + 1):
            back = start - p * seq_len
            if back >= 0:
                period_stack.append(grid_thw[back : back + seq_len])
            else:
                period_stack.append(win)
        period = np.stack(period_stack, axis=1)  # (T, P, H, W)
        xs.append(win)
        tss.append(ts)
        periods.append(period)
        if len(xs) >= max_samples:
            break
    x = np.stack(xs, axis=0).astype(np.float32)
    period = np.stack(periods, axis=0).astype(np.float32)
    return x, tss, period


def main() -> int:
    args = _parse_args()
    blob = np.load(args.npz)
    data = blob["data"]  # (T, C, H, W)
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    t0 = pd.Timestamp(meta["time_start"])
    step_h = int(meta.get("stride", 1))
    times = pd.date_range(t0, periods=data.shape[0], freq=f"{step_h}h")

    grid = data[:, 0].astype(np.float32)  # raw PM2.5; UniST DataLoader applies its own MinMax

    x, timestamps, period = _make_windows(
        grid, times, args.seq_len, args.stride, args.max_samples
    )
    n = x.shape[0]
    n_train = max(1, int(n * args.train_ratio))
    n_val = max(1, int(n * args.val_ratio))
    n_test = max(1, n - n_train - n_val)
    # Adjust if rounding overflow
    while n_train + n_val + n_test > n:
        n_test -= 1
    i0, i1 = n_train, n_train + n_val

    def pack(sl: slice) -> list:
        return [x[sl].tolist(), period[sl].tolist()]

    payload = {
        "X_train": pack(slice(0, i0)),
        "X_val": pack(slice(i0, i1)),
        "X_test": pack(slice(i1, i1 + n_test)),
        "timestamps": {
            "train": timestamps[:i0],
            "val": timestamps[i0:i1],
            "test": timestamps[i1 : i1 + n_test],
        },
        "meta": {
            "source_npz": str(args.npz),
            "protocol": meta.get("protocol"),
            "region": meta.get("region"),
            "seq_len": args.seq_len,
            "n_train": i0,
            "n_val": i1 - i0,
            "n_test": n_test,
            "shape_example": list(x.shape),
        },
    }

    UNIST_DATASET.mkdir(parents=True, exist_ok=True)
    out = UNIST_DATASET / f"{args.name}_short.json"
    # JSON dump can be large; write compact
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps(payload["meta"], indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
