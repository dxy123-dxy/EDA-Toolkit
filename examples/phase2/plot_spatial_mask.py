"""Visualize spatial-mask eval: metric bars + station map + pred-vs-true scatter.

Usage (repo root):
  .venv-ml\\Scripts\\python.exe examples/phase2/plot_spatial_mask.py
  # then open output/phase2/eval/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pykrige.ok import OrdinaryKriging

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data" / "knowair_v2"
OUT = ROOT / "output" / "phase2" / "eval"


def _idw(obs_xy, obs_z, qry_xy, power=2.0):
    dx = qry_xy[:, None, 0] - obs_xy[None, :, 0]
    dy = qry_xy[:, None, 1] - obs_xy[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    exact = dist < 1e-9
    dist = np.where(exact, np.nan, dist)
    w = 1.0 / np.power(dist, power)
    w = np.where(np.isnan(w), 0.0, w)
    for i in np.where(exact.any(axis=1))[0]:
        j = int(np.where(exact[i])[0][0])
        w[i, :] = 0.0
        w[i, j] = 1.0
    wsum = np.where(w.sum(axis=1) == 0, np.nan, w.sum(axis=1))
    return (w * obs_z[None, :]).sum(axis=1) / wsum


def _kriging(obs_xy, obs_z, qry_xy):
    ok = OrdinaryKriging(
        obs_xy[:, 0], obs_xy[:, 1], obs_z,
        variogram_model="linear", verbose=False, enable_plotting=False,
    )
    z, _ = ok.execute("points", qry_xy[:, 0], qry_xy[:, 1])
    return np.asarray(z, dtype=float).ravel()


def plot_metrics(report: dict, out: Path, tag: str) -> Path:
    station = report["station_holdout"]
    grid = report["grid_cell_holdout"]
    labels, mae, rmse = [], [], []

    for name, block in [
        ("IDW\n(station)", station["idw"]),
        ("Kriging\n(station)", station["ordinary_kriging"]),
        ("IDW\n(grid cell)", grid["idw"]),
        ("Kriging\n(grid cell)", grid["ordinary_kriging"]),
    ]:
        labels.append(name)
        mae.append(block["mae"])
        rmse.append(block["rmse"])

    if "unist" in report and "masks" in report["unist"]:
        tube = report["unist"]["masks"]["tube"]["0.5"]
        labels.append("UniST tube\n(smoke)")
        mae.append(tube["mae"])
        rmse.append(tube["rmse"])

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width / 2, mae, width, label="MAE", color="#3b82f6")
    ax.bar(x + width / 2, rmse, width, label="RMSE", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PM2.5 (μg/m³)")
    ax.set_title("Spatial mask eval — metrics (BTHSA, seed=42, holdout=20%)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / f"01_metrics_bars_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_station_map(region: str, seed: int, holdout: float, out: Path) -> list[Path]:
    ds = xr.open_dataset(DATA / f"dataset_{region}.nc")
    st = pd.read_csv(DATA / f"stations_{region}.csv")
    st["station_id"] = st["station_id"].astype(str)
    st = st.set_index("station_id")
    ids = [str(s) for s in ds.station.values]
    lon = st.loc[ids, "lon"].to_numpy(float)
    lat = st.loc[ids, "lat"].to_numpy(float)
    xy = np.column_stack([lon, lat])

    # mid January 2023 snapshot
    t = np.datetime64("2023-01-15T12:00:00")
    z = ds["PM2.5"].sel(time=t, method="nearest").values.astype(float)
    t_used = pd.Timestamp(ds["PM2.5"].sel(time=t, method="nearest").time.values)

    rng = np.random.default_rng(seed)
    n = len(ids)
    n_hold = max(1, int(round(n * holdout)))
    hold = rng.choice(n, size=n_hold, replace=False)
    obs = np.ones(n, dtype=bool)
    obs[hold] = False

    y_true = z[hold]
    idw_pred = _idw(xy[obs], z[obs], xy[hold])
    kri_pred = _kriging(xy[obs], z[obs], xy[hold])
    ds.close()

    paths = []
    # --- map ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True, sharey=True)
    vmin = float(np.nanpercentile(z, 5))
    vmax = float(np.nanpercentile(z, 95))

    axes[0].scatter(lon[obs], lat[obs], c=z[obs], s=28, cmap="YlOrRd", vmin=vmin, vmax=vmax, edgecolors="k", linewidths=0.3)
    axes[0].scatter(lon[hold], lat[hold], facecolors="none", edgecolors="cyan", s=80, linewidths=1.5, label="hold-out")
    axes[0].set_title(f"Observed + hold-out\n{t_used}")
    axes[0].legend(loc="lower left", fontsize=8)

    sc1 = axes[1].scatter(lon[hold], lat[hold], c=idw_pred, s=55, cmap="YlOrRd", vmin=vmin, vmax=vmax, edgecolors="k", linewidths=0.4)
    axes[1].set_title("IDW at hold-out")

    sc2 = axes[2].scatter(lon[hold], lat[hold], c=kri_pred, s=55, cmap="YlOrRd", vmin=vmin, vmax=vmax, edgecolors="k", linewidths=0.4)
    axes[2].set_title("Ordinary Kriging at hold-out")

    for ax in axes:
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(sc2, ax=axes.ravel().tolist(), shrink=0.85, label="PM2.5 (μg/m³)")
    fig.suptitle(f"Station spatial hold-out ({region.upper()}, 20%, seed={seed})", y=1.02)
    fig.tight_layout()
    p1 = out / f"02_station_holdout_map_{region}.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p1)

    # --- scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, pred, name in [
        (axes[0], idw_pred, "IDW"),
        (axes[1], kri_pred, "Ordinary Kriging"),
    ]:
        ax.scatter(y_true, pred, alpha=0.75, edgecolors="k", linewidths=0.3)
        lo = float(np.nanmin([y_true.min(), pred.min()]))
        hi = float(np.nanmax([y_true.max(), pred.max()]))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
        mae = float(np.mean(np.abs(y_true - pred)))
        rmse = float(np.sqrt(np.mean((y_true - pred) ** 2)))
        ax.set_title(f"{name}\nMAE={mae:.2f}  RMSE={rmse:.2f}")
        ax.set_xlabel("True PM2.5")
        ax.set_ylabel("Predicted PM2.5")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(f"Hold-out pred vs true @ {t_used.date()}")
    fig.tight_layout()
    p2 = out / f"03_pred_vs_true_scatter_{region}.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)

    # --- error map ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    err_idw = idw_pred - y_true
    err_kri = kri_pred - y_true
    lim = float(np.nanmax(np.abs(np.concatenate([err_idw, err_kri]))))
    for ax, err, name in [
        (axes[0], err_idw, "IDW error (pred−true)"),
        (axes[1], err_kri, "Kriging error (pred−true)"),
    ]:
        sc = ax.scatter(lon[hold], lat[hold], c=err, s=60, cmap="coolwarm", vmin=-lim, vmax=lim, edgecolors="k", linewidths=0.4)
        ax.set_title(name)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, label="μg/m³")
    fig.tight_layout()
    p3 = out / f"04_holdout_error_map_{region}.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p3)
    return paths


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="bthsa")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
    )
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.metrics_json is None:
        args.metrics_json = OUT / f"spatial_mask_{args.region}_seed{args.seed}.json"

    report = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    paths = [plot_metrics(report, OUT, args.region)]
    paths.extend(plot_station_map(args.region, args.seed, args.holdout, OUT))

    index = OUT / f"README_plots_{args.region}.md"
    lines = ["# Spatial mask plots\n\n"]
    for path in paths:
        lines.append(f"- `{path.name}`\n")
        lines.append(f"  ![]({path.name})\n\n")
    index.write_text("".join(lines), encoding="utf-8")

    print("Wrote:")
    for path in paths:
        print(" ", path)
    print(" ", index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
