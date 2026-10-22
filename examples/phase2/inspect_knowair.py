"""Summarize KnowAir-V2 NetCDF + station CSVs for Phase 2.

Usage (repo root):
  .venv-ml\\Scripts\\python.exe examples/phase2/inspect_knowair.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import xarray as xr

DATA = Path(__file__).resolve().parent / "data" / "knowair_v2"
REGIONS = ("bthsa", "yrd")


def _inspect_region(region: str) -> dict:
    nc = DATA / f"dataset_{region}.nc"
    stations = DATA / f"stations_{region}.csv"
    out: dict = {"region": region, "nc": str(nc), "stations_csv": str(stations)}
    if not nc.exists():
        out["error"] = "nc missing"
        return out
    if nc.stat().st_size < 50_000_000:
        out["warning"] = f"nc size looks incomplete ({nc.stat().st_size} bytes)"

    try:
        ds = xr.open_dataset(nc)
    except OSError as exc:
        out["nc_bytes"] = nc.stat().st_size
        out["error"] = f"cannot open nc (download incomplete?): {exc}"
        return out
    st = pd.read_csv(stations) if stations.exists() else None
    out.update(
        {
            "nc_bytes": nc.stat().st_size,
            "dims": dict(ds.sizes),
            "variables": list(ds.data_vars),
            "time_start": str(ds.time.values[0]),
            "time_end": str(ds.time.values[-1]),
            "n_stations_nc": int(ds.sizes.get("station", 0)),
        }
    )
    if st is not None:
        nc_ids = set(map(str, ds.station.values))
        csv_ids = set(st.station_id.astype(str))
        out["stations_csv_rows"] = len(st)
        out["station_id_overlap"] = len(nc_ids & csv_ids)
        out["lon_range"] = [float(st.lon.min()), float(st.lon.max())]
        out["lat_range"] = [float(st.lat.min()), float(st.lat.max())]
    if "PM2.5" in ds:
        pm = ds["PM2.5"]
        out["pm25_nan_frac"] = float(pm.isnull().mean().item())
        out["pm25_min"] = float(pm.min().item())
        out["pm25_max"] = float(pm.max().item())
    ds.close()
    return out


def main() -> int:
    rows = [_inspect_region(r) for r in REGIONS]
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    print(text)
    out_dir = Path(__file__).resolve().parent.parent.parent / "output" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "knowair_inspect.json"
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
