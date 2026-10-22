"""Rasterize KnowAir station panel → UniST-friendly T×C×H×W tensor.

Protocol (v1, freeze for Phase 2 demos):
  - Grid: equal lon/lat bins over station bbox (padding 0.05°)
  - Cell value: mean of stations falling in the cell (per timestep)
  - Empty cells: NaN in raw grid; filled with 0.0 in model tensor (mask saved)
  - Default variable: PM2.5 (C=1); optional multi-var via --vars
  - Time subsample via --time-start/--time-end and optional --stride hours

Usage (repo root):
  .venv-ml\\Scripts\\python.exe examples/phase2/build_knowair_grid.py --region bthsa --grid 16 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DATA = Path(__file__).resolve().parent / "data" / "knowair_v2"
OUT = Path(__file__).resolve().parent.parent.parent / "output" / "phase2" / "tensors"
PROTOCOL_VERSION = "knowair_grid_v1"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region", choices=("bthsa", "yrd"), default="bthsa")
    p.add_argument("--grid", nargs=2, type=int, default=[16, 16], metavar=("H", "W"))
    p.add_argument("--vars", nargs="+", default=["PM2.5"])
    p.add_argument("--time-start", default="2023-01-01")
    p.add_argument("--time-end", default="2023-03-31")
    p.add_argument("--stride", type=int, default=1, help="Keep every Nth timestep")
    p.add_argument("--pad-deg", type=float, default=0.05)
    return p.parse_args()


def build_grid(
    region: str,
    h: int,
    w: int,
    variables: list[str],
    time_start: str,
    time_end: str,
    stride: int,
    pad_deg: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    ds = xr.open_dataset(DATA / f"dataset_{region}.nc")
    st = pd.read_csv(DATA / f"stations_{region}.csv")
    st = st.set_index(st.station_id.astype(str))

    # Align station order to NetCDF
    station_ids = [str(s) for s in ds.station.values]
    lon = st.loc[station_ids, "lon"].to_numpy(dtype=np.float64)
    lat = st.loc[station_ids, "lat"].to_numpy(dtype=np.float64)

    lon0, lon1 = float(lon.min()) - pad_deg, float(lon.max()) + pad_deg
    lat0, lat1 = float(lat.min()) - pad_deg, float(lat.max()) + pad_deg
    lon_edges = np.linspace(lon0, lon1, w + 1)
    lat_edges = np.linspace(lat0, lat1, h + 1)
    # row 0 = north (lat max)
    col_idx = np.clip(np.digitize(lon, lon_edges) - 1, 0, w - 1)
    row_idx = np.clip(np.digitize(lat, lat_edges) - 1, 0, h - 1)
    row_idx = (h - 1) - row_idx

    sub = ds.sel(time=slice(time_start, time_end))
    if stride > 1:
        sub = sub.isel(time=slice(None, None, stride))
    times = pd.to_datetime(sub.time.values)

    t_len = sub.sizes["time"]
    c_len = len(variables)
    grid = np.full((t_len, c_len, h, w), np.nan, dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.int32)

    # static occupancy
    for r, c in zip(row_idx, col_idx):
        counts[r, c] += 1

    for ci, var in enumerate(variables):
        if var not in sub:
            raise KeyError(f"variable {var} not in dataset; have {list(sub.data_vars)}")
        arr = sub[var].values.astype(np.float32)  # (T, N)
        for ti in range(t_len):
            acc = np.zeros((h, w), dtype=np.float64)
            cnt = np.zeros((h, w), dtype=np.int32)
            vals = arr[ti]
            for si in range(vals.shape[0]):
                v = vals[si]
                if np.isnan(v):
                    continue
                r, c = int(row_idx[si]), int(col_idx[si])
                acc[r, c] += float(v)
                cnt[r, c] += 1
            mask = cnt > 0
            cell = np.full((h, w), np.nan, dtype=np.float32)
            cell[mask] = (acc[mask] / cnt[mask]).astype(np.float32)
            grid[ti, ci] = cell

    occupied = counts > 0
    filled = np.nan_to_num(grid, nan=0.0).astype(np.float32)
    meta = {
        "protocol": PROTOCOL_VERSION,
        "region": region,
        "shape_TCHW": list(filled.shape),
        "variables": variables,
        "time_start": str(times[0]),
        "time_end": str(times[-1]),
        "n_times": int(t_len),
        "stride": stride,
        "grid_hw": [h, w],
        "lon_edges": lon_edges.tolist(),
        "lat_edges": lat_edges.tolist(),
        "pad_deg": pad_deg,
        "n_stations": len(station_ids),
        "occupied_cells": int(occupied.sum()),
        "occupancy_frac": float(occupied.mean()),
        "empty_fill": 0.0,
        "aggregation": "mean of stations in lon/lat cell",
    }
    ds.close()
    return filled, occupied.astype(np.uint8), meta


def main() -> int:
    args = _parse_args()
    h, w = args.grid
    filled, occupied, meta = build_grid(
        region=args.region,
        h=h,
        w=w,
        variables=args.vars,
        time_start=args.time_start,
        time_end=args.time_end,
        stride=args.stride,
        pad_deg=args.pad_deg,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"knowair_{args.region}_{h}x{w}_{args.time_start}_{args.time_end}"
    npz_path = OUT / f"{stem}.npz"
    meta_path = OUT / f"{stem}.meta.json"
    np.savez_compressed(npz_path, data=filled, occupied=occupied)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"wrote {npz_path}")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
