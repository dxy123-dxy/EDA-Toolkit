"""Inverse Distance Weighting interpolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from eda_toolkit.io.dataset import SpatioTemporalDataset


def idw_interpolate(
    dataset: SpatioTemporalDataset,
    column: str,
    *,
    power: float = 2.0,
    grid_size: int = 40,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    IDW on a regular grid over the projected bounding box.

    z(x) = sum(w_i * z_i) / sum(w_i), w_i = 1 / d_i^power
    """
    gdf = dataset.to_projected()
    if column not in gdf.columns:
        raise ValueError(f"Column '{column}' not found")

    mask = gdf[column].notna() & gdf.geometry.notna()
    gdf = gdf.loc[mask]
    if len(gdf) < 2:
        raise ValueError("IDW requires at least 2 valid observations")

    xs = gdf.geometry.x.values.astype(float)
    ys = gdf.geometry.y.values.astype(float)
    zs = gdf[column].astype(float).values

    xmin, ymin, xmax, ymax = gdf.total_bounds
    xx = np.linspace(xmin, xmax, grid_size)
    yy = np.linspace(ymin, ymax, grid_size)
    grid_x, grid_y = np.meshgrid(xx, yy)
    qx = grid_x.ravel()
    qy = grid_y.ravel()

    # Broadcast distances: (n_query, n_obs)
    dx = qx[:, None] - xs[None, :]
    dy = qy[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    # Exact matches: take observed value
    exact = dist < 1e-12
    dist = np.where(exact, np.nan, dist)
    weights = 1.0 / np.power(dist, power)
    weights = np.where(np.isnan(weights), 0.0, weights)
    # Rows with an exact match: one-hot that observation
    if exact.any():
        for i in np.where(exact.any(axis=1))[0]:
            j = np.where(exact[i])[0][0]
            weights[i, :] = 0.0
            weights[i, j] = 1.0

    wsum = weights.sum(axis=1)
    wsum = np.where(wsum == 0, np.nan, wsum)
    zhat = (weights * zs[None, :]).sum(axis=1) / wsum
    zgrid = zhat.reshape(grid_size, grid_size)

    result: dict[str, Any] = {
        "method": "IDW",
        "column": column,
        "power": power,
        "grid_size": grid_size,
        "n_obs": int(len(zs)),
        "crs": str(gdf.crs) if gdf.crs else None,
        "bounds": [float(xmin), float(ymin), float(xmax), float(ymax)],
        "z_min": float(np.nanmin(zgrid)),
        "z_max": float(np.nanmax(zgrid)),
        "z_mean": float(np.nanmean(zgrid)),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows = [
            {"x": float(grid_x[i, j]), "y": float(grid_y[i, j]), "z": float(zgrid[i, j])}
            for i in range(grid_size)
            for j in range(grid_size)
        ]
        csv_path = out / "idw_grid.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        step = max(1, grid_size // 25)
        geoms, vals = [], []
        for i in range(0, grid_size, step):
            for j in range(0, grid_size, step):
                geoms.append(Point(grid_x[i, j], grid_y[i, j]))
                vals.append(float(zgrid[i, j]))
        grid_gdf = gpd.GeoDataFrame({"z": vals}, geometry=geoms, crs=gdf.crs)
        try:
            grid_gdf = grid_gdf.to_crs("EPSG:4326")
        except Exception:
            pass
        geo_path = out / "idw_grid.geojson"
        grid_gdf.to_file(geo_path, driver="GeoJSON")

        summary_path = out / "idw_summary.json"
        summary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["artifacts"] = {
            "idw_grid_csv": str(csv_path),
            "idw_grid_geojson": str(geo_path),
            "idw_summary_json": str(summary_path),
        }

    result["zgrid"] = zgrid
    result["grid_x"] = grid_x
    result["grid_y"] = grid_y
    return result
