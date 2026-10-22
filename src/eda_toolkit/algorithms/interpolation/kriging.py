"""Ordinary Kriging and empirical variogram (PyKrige)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from eda_toolkit.io.dataset import SpatioTemporalDataset


def empirical_variogram(
    dataset: SpatioTemporalDataset,
    column: str,
    *,
    n_lags: int = 12,
    max_dist: float | None = None,
) -> dict[str, Any]:
    """
    Pairwise empirical semivariogram (binned).

    Useful as EDA before choosing a theoretical model for kriging.
    """
    gdf = dataset.to_projected()
    if column not in gdf.columns:
        raise ValueError(f"Column '{column}' not found")
    mask = gdf[column].notna() & gdf.geometry.notna()
    gdf = gdf.loc[mask]
    xs = gdf.geometry.x.values.astype(float)
    ys = gdf.geometry.y.values.astype(float)
    zs = gdf[column].astype(float).values
    n = len(zs)
    if n < 3:
        raise ValueError("Variogram requires at least 3 points")

    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    gamma = 0.5 * (zs[:, None] - zs[None, :]) ** 2

    iu = np.triu_indices(n, k=1)
    d_flat = dist[iu]
    g_flat = gamma[iu]

    if max_dist is None:
        max_dist = float(np.percentile(d_flat, 50))
    bins = np.linspace(0, max_dist, n_lags + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    means = []
    counts = []
    for i in range(n_lags):
        sel = (d_flat >= bins[i]) & (d_flat < bins[i + 1])
        counts.append(int(sel.sum()))
        means.append(float(g_flat[sel].mean()) if sel.any() else float("nan"))

    return {
        "column": column,
        "n_pairs": int(len(d_flat)),
        "max_dist": float(max_dist),
        "lags": [float(c) for c in centers],
        "semivariance": means,
        "pair_counts": counts,
    }


def ordinary_kriging(
    dataset: SpatioTemporalDataset,
    column: str,
    *,
    variogram_model: str = "linear",
    grid_size: int = 40,
    nlags: int = 6,
    leave_one_out_cv: bool = True,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Ordinary Kriging via PyKrige.

    Requires optional dependency: pip install 'eda-toolkit[kriging]'
    """
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError as exc:
        raise ImportError(
            "Install kriging extras: pip install 'eda-toolkit[kriging]'"
        ) from exc

    gdf = dataset.to_projected()
    if column not in gdf.columns:
        raise ValueError(f"Column '{column}' not found")
    mask = gdf[column].notna() & gdf.geometry.notna()
    gdf = gdf.loc[mask]
    if len(gdf) < 3:
        raise ValueError("Ordinary Kriging requires at least 3 points")

    xs = gdf.geometry.x.values.astype(float)
    ys = gdf.geometry.y.values.astype(float)
    zs = gdf[column].astype(float).values

    xmin, ymin, xmax, ymax = gdf.total_bounds
    gridx = np.linspace(xmin, xmax, grid_size)
    gridy = np.linspace(ymin, ymax, grid_size)

    ok = OrdinaryKriging(
        xs,
        ys,
        zs,
        variogram_model=variogram_model,
        nlags=nlags,
        verbose=False,
        enable_plotting=False,
    )
    zgrid, ss = ok.execute("grid", gridx, gridy)
    zgrid = np.asarray(zgrid, dtype=float)
    ss = np.asarray(ss, dtype=float)

    result: dict[str, Any] = {
        "method": "OrdinaryKriging",
        "column": column,
        "variogram_model": variogram_model,
        "grid_size": grid_size,
        "n_obs": int(len(zs)),
        "crs": str(gdf.crs) if gdf.crs else None,
        "bounds": [float(xmin), float(ymin), float(xmax), float(ymax)],
        "z_min": float(np.nanmin(zgrid)),
        "z_max": float(np.nanmax(zgrid)),
        "z_mean": float(np.nanmean(zgrid)),
        "variance_mean": float(np.nanmean(ss)),
    }

    if leave_one_out_cv and len(zs) <= 80:
        preds = []
        for i in range(len(zs)):
            mask_i = np.ones(len(zs), dtype=bool)
            mask_i[i] = False
            ok_i = OrdinaryKriging(
                xs[mask_i],
                ys[mask_i],
                zs[mask_i],
                variogram_model=variogram_model,
                nlags=nlags,
                verbose=False,
                enable_plotting=False,
            )
            zi, _ = ok_i.execute("points", np.array([xs[i]]), np.array([ys[i]]))
            preds.append(float(np.asarray(zi).ravel()[0]))
        preds_arr = np.asarray(preds)
        rmse = float(np.sqrt(np.mean((preds_arr - zs) ** 2)))
        mae = float(np.mean(np.abs(preds_arr - zs)))
        result["cv"] = {"method": "leave_one_out", "rmse": rmse, "mae": mae}

    # Empirical variogram for diagnostics
    result["empirical_variogram"] = empirical_variogram(
        dataset, column, n_lags=max(nlags, 8)
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        grid_x, grid_y = np.meshgrid(gridx, gridy)
        rows = [
            {
                "x": float(grid_x[i, j]),
                "y": float(grid_y[i, j]),
                "z": float(zgrid[i, j]),
                "variance": float(ss[i, j]),
            }
            for i in range(grid_size)
            for j in range(grid_size)
        ]
        csv_path = out / "kriging_grid.csv"
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
        geo_path = out / "kriging_grid.geojson"
        grid_gdf.to_file(geo_path, driver="GeoJSON")

        summary = {k: v for k, v in result.items()}
        summary_path = out / "kriging_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["artifacts"] = {
            "kriging_grid_csv": str(csv_path),
            "kriging_grid_geojson": str(geo_path),
            "kriging_summary_json": str(summary_path),
        }

    result["zgrid"] = zgrid
    result["variance_grid"] = ss
    result["gridx"] = gridx
    result["gridy"] = gridy
    return result
