"""Regenerate Phase-2 Demo D1/D2 figures from latest eval JSON.

Produces under ``output/phase2/eval/``:
  - ``01_metrics_bars_{region}.png`` — same-mask IDW / Kriging / UniST
  - ``02_station_holdout_map_{region}.png``
  - ``03_pred_vs_true_scatter_{region}.png``
  - ``04_holdout_error_map_{region}.png``
  - ``05_cross_scene_prompt_vs_scratch.png`` (D2 narrative)

Then run ``pack_demos.py`` to copy into ``demo_d1`` / ``demo_d2``.

Usage (repo root):
  .venv-ml\\Scripts\\python.exe examples/phase2/plot_demo_figures.py
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
EVAL = ROOT / "output" / "phase2" / "eval"

REGION_TITLE = {"bthsa": "BTHSA", "yrd": "YRD"}
UNIST_LABEL = {
    "bthsa": "UniST\n(Long)",
    "yrd": "UniST\n(Prompt)",
}


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
        obs_xy[:, 0],
        obs_xy[:, 1],
        obs_z,
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
    )
    z, _ = ok.execute("points", qry_xy[:, 0], qry_xy[:, 1])
    return np.asarray(z, dtype=float).ravel()


def plot_same_mask_bars(region: str, seed: int, out: Path) -> Path:
    path = EVAL / f"same_mask_{region}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}; run eval_same_mask.py --region {region} first"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    methods = report["methods"]
    tag = REGION_TITLE[region]
    unist_lbl = UNIST_LABEL[region]

    labels = ["IDW", "Ordinary\nKriging", unist_lbl]
    keys = ["idw", "ordinary_kriging", "unist"]
    mae = [methods[k]["mae"] for k in keys]
    rmse = [methods[k]["rmse"] for k in keys]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - width / 2, mae, width, label="MAE", color="#3b82f6")
    ax.bar(x + width / 2, rmse, width, label="RMSE", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PM2.5 (μg/m³)")
    ax.set_title(
        f"Same-mask occupied-grid — {tag} "
        f"(seed={seed}, holdout≈{report.get('holdout_frac_cells', 0.2):.0%})"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for i, (m, r) in enumerate(zip(mae, rmse)):
        ax.text(i - width / 2, m, f"{m:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, r, f"{r:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    dest = out / f"01_metrics_bars_{region}.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


def plot_cross_scene_bars(out: Path) -> Path | None:
    path = EVAL / "cross_scene_bthsa_long_to_yrd.json"
    if not path.exists():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    labels = ["Prompt\n(BTHSA→YRD)", "Scratch\n(YRD, same budget)"]
    rmse = [
        float(m["prompt_test_temporal_rmse"]),
        float(m["scratch_test_temporal_rmse"]),
    ]
    mae_prompt = float(m.get("prompt_test_temporal_mae", float("nan")))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(x, rmse, width=0.55, color=["#16a34a", "#f97316"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test temporal RMSE (PM2.5 μg/m³)")
    ax.set_title("Cross-scene: Prompt vs Scratch (YRDLong, 5 epoch, few_ratio=0.5)")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, rmse):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    if np.isfinite(mae_prompt):
        ax.annotate(
            f"Prompt MAE ≈ {mae_prompt:.2f}",
            xy=(0, rmse[0]),
            xytext=(0.35, rmse[0] + 0.8),
            fontsize=8,
            color="#166534",
            arrowprops=dict(arrowstyle="->", color="#166534", lw=0.8),
        )
    fig.tight_layout()
    dest = out / "05_cross_scene_prompt_vs_scratch.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


def plot_station_figures(region: str, seed: int, holdout: float, out: Path) -> list[Path]:
    ds = xr.open_dataset(DATA / f"dataset_{region}.nc")
    st = pd.read_csv(DATA / f"stations_{region}.csv")
    st["station_id"] = st["station_id"].astype(str)
    st = st.set_index("station_id")
    ids = [str(s) for s in ds.station.values]
    lon = st.loc[ids, "lon"].to_numpy(float)
    lat = st.loc[ids, "lat"].to_numpy(float)
    xy = np.column_stack([lon, lat])

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

    tag = REGION_TITLE[region]
    paths: list[Path] = []

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True, sharey=True)
    vmin = float(np.nanpercentile(z, 5))
    vmax = float(np.nanpercentile(z, 95))

    axes[0].scatter(
        lon[obs], lat[obs], c=z[obs], s=28, cmap="YlOrRd",
        vmin=vmin, vmax=vmax, edgecolors="k", linewidths=0.3,
    )
    axes[0].scatter(
        lon[hold], lat[hold], facecolors="none", edgecolors="cyan",
        s=80, linewidths=1.5, label="hold-out",
    )
    axes[0].set_title(f"Observed + hold-out\n{t_used}")
    axes[0].legend(loc="lower left", fontsize=8)

    axes[1].scatter(
        lon[hold], lat[hold], c=idw_pred, s=55, cmap="YlOrRd",
        vmin=vmin, vmax=vmax, edgecolors="k", linewidths=0.4,
    )
    axes[1].set_title("IDW at hold-out")

    sc2 = axes[2].scatter(
        lon[hold], lat[hold], c=kri_pred, s=55, cmap="YlOrRd",
        vmin=vmin, vmax=vmax, edgecolors="k", linewidths=0.4,
    )
    axes[2].set_title("Ordinary Kriging at hold-out")

    for ax in axes:
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")

    fig.colorbar(sc2, ax=axes.ravel().tolist(), shrink=0.85, label="PM2.5 (μg/m³)")
    fig.suptitle(f"Station spatial hold-out ({tag}, {holdout:.0%}, seed={seed})", y=1.02)
    fig.tight_layout()
    p1 = out / f"02_station_holdout_map_{region}.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p1)

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
    fig.suptitle(f"Hold-out pred vs true @ {t_used.date()} ({tag})")
    fig.tight_layout()
    p2 = out / f"03_pred_vs_true_scatter_{region}.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    err_idw = idw_pred - y_true
    err_kri = kri_pred - y_true
    lim = float(np.nanmax(np.abs(np.concatenate([err_idw, err_kri]))))
    for ax, err, name in [
        (axes[0], err_idw, "IDW error (pred−true)"),
        (axes[1], err_kri, "Kriging error (pred−true)"),
    ]:
        sc = ax.scatter(
            lon[hold], lat[hold], c=err, s=60, cmap="coolwarm",
            vmin=-lim, vmax=lim, edgecolors="k", linewidths=0.4,
        )
        ax.set_title(name)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, label="μg/m³")
    fig.suptitle(f"Hold-out error map ({tag})", y=1.02)
    fig.tight_layout()
    p3 = out / f"04_holdout_error_map_{region}.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p3)
    return paths


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--regions", nargs="+", default=["bthsa", "yrd"])
    args = p.parse_args()

    EVAL.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for region in args.regions:
        written.append(plot_same_mask_bars(region, args.seed, EVAL))
        written.extend(plot_station_figures(region, args.seed, args.holdout, EVAL))

    cross = plot_cross_scene_bars(EVAL)
    if cross:
        written.append(cross)

    # Keep untagged aliases used by older pack paths
    for alias, src in [
        ("01_metrics_bars.png", "01_metrics_bars_bthsa.png"),
        ("02_station_holdout_map.png", "02_station_holdout_map_bthsa.png"),
        ("03_pred_vs_true_scatter.png", "03_pred_vs_true_scatter_bthsa.png"),
        ("04_holdout_error_map.png", "04_holdout_error_map_bthsa.png"),
    ]:
        s = EVAL / src
        if s.exists():
            (EVAL / alias).write_bytes(s.read_bytes())
            written.append(EVAL / alias)

    print("Wrote:")
    for path in written:
        print(" ", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
