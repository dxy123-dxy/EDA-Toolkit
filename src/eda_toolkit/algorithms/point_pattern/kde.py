"""Kernel density estimation for spatial point patterns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from eda_toolkit.io.dataset import SpatioTemporalDataset


def kde_heatmap(
    dataset: SpatioTemporalDataset,
    *,
    bandwidth: float | None = None,
    grid_size: int = 50,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Gaussian KDE on projected point coordinates.

    Returns grid extent, density values, and optional GeoJSON / CSV artifacts.
    """
    from scipy.stats import gaussian_kde

    gdf = dataset.to_projected()
    if gdf.empty:
        raise ValueError("Empty dataset")
    coords = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
    if len(coords) < 2:
        raise ValueError("KDE requires at least 2 points")

    xmin, ymin, xmax, ymax = gdf.total_bounds
    pad_x = (xmax - xmin) * 0.05 or 1.0
    pad_y = (ymax - ymin) * 0.05 or 1.0
    xmin, xmax = xmin - pad_x, xmax + pad_x
    ymin, ymax = ymin - pad_y, ymax + pad_y

    xx = np.linspace(xmin, xmax, grid_size)
    yy = np.linspace(ymin, ymax, grid_size)
    grid_x, grid_y = np.meshgrid(xx, yy)
    positions = np.vstack([grid_x.ravel(), grid_y.ravel()])

    kde = gaussian_kde(coords.T, bw_method=bandwidth)
    density = kde(positions).reshape(grid_size, grid_size)

    result: dict[str, Any] = {
        "method": "gaussian_kde",
        "n_points": int(len(coords)),
        "grid_size": grid_size,
        "bandwidth": float(kde.factor),
        "crs": str(gdf.crs) if gdf.crs else None,
        "bounds": [float(xmin), float(ymin), float(xmax), float(ymax)],
        "density_min": float(density.min()),
        "density_max": float(density.max()),
        "density_mean": float(density.mean()),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(grid_size):
            for j in range(grid_size):
                rows.append(
                    {
                        "x": float(grid_x[i, j]),
                        "y": float(grid_y[i, j]),
                        "density": float(density[i, j]),
                    }
                )
        csv_path = out / "kde_grid.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        # Subsample for GeoJSON to keep file size reasonable
        step = max(1, grid_size // 25)
        geoms = []
        dens_vals = []
        for i in range(0, grid_size, step):
            for j in range(0, grid_size, step):
                geoms.append(Point(grid_x[i, j], grid_y[i, j]))
                dens_vals.append(float(density[i, j]))
        grid_gdf = gpd.GeoDataFrame(
            {"density": dens_vals}, geometry=geoms, crs=gdf.crs
        )
        # Reproject to WGS84 for map clients if possible
        try:
            grid_gdf = grid_gdf.to_crs("EPSG:4326")
        except Exception:
            pass
        geo_path = out / "kde_grid.geojson"
        grid_gdf.to_file(geo_path, driver="GeoJSON")

        summary_path = out / "kde_summary.json"
        summary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["artifacts"] = {
            "kde_grid_csv": str(csv_path),
            "kde_grid_geojson": str(geo_path),
            "kde_summary_json": str(summary_path),
        }

    result["density"] = density
    result["grid_x"] = grid_x
    result["grid_y"] = grid_y
    return result
