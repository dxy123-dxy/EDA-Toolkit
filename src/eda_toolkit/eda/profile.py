"""Data quality profiling for spatio-temporal datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from eda_toolkit.io.dataset import SpatioTemporalDataset


def build_profile(dataset: SpatioTemporalDataset) -> dict[str, Any]:
    """Build profile dict: dimensions, CRS, time range, missing stats, duplicates."""
    gdf = dataset.gdf
    profile: dict[str, Any] = {
        "version": "0.1",
        "summary": dataset.summary_dict(),
        "columns": {},
        "quality": {},
    }

    for col in gdf.columns:
        if col == "geometry":
            continue
        series = gdf[col]
        info: dict[str, Any] = {
            "dtype": str(series.dtype),
            "n_missing": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()) if len(gdf) else 0.0,
            "n_unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            info["stats"] = {k: float(v) for k, v in desc.items() if pd.notna(v)}
        profile["columns"][col] = info

    invalid_geom = 0
    if not gdf.empty and hasattr(gdf.geometry, "is_valid"):
        invalid_geom = int((~gdf.geometry.is_valid).sum())
    profile["quality"]["invalid_geometry_count"] = invalid_geom
    profile["quality"]["crs_defined"] = gdf.crs is not None

    dup_keys: list[str] = []
    if dataset.has_time and dataset.time_column:
        dup_keys.append(dataset.time_column)
    if dataset.region_id_column and dataset.region_id_column in gdf.columns:
        dup_keys.append(dataset.region_id_column)
    if dup_keys:
        n_dup = int(gdf.duplicated(subset=dup_keys).sum())
        profile["quality"]["duplicate_spatiotemporal_keys"] = {
            "keys": dup_keys,
            "count": n_dup,
        }

    return profile


def save_profile(profile: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "profile.json"
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
