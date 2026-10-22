"""Spatial hold-out: UniST (tube) vs IDW vs Ordinary Kriging.

Common protocol (seed-fixed):
  1) Station hold-out (primary D1 narrative)
     - Randomly hide 20% stations at selected timestamps
     - Fit IDW / Ordinary Kriging on remaining stations
     - Predict at hold-out station coordinates → MAE / RMSE
  2) Grid-cell hold-out (aligns with UniST T×H×W)
     - Hide 20% of occupied cells on the KnowAir grid tensor
     - IDW / Kriging at cell centers
  3) UniST tube mask on TrafficKnowAir test windows (pretrained ckpt)

Usage (repo root):
  .venv-ml\\Scripts\\python.exe examples/phase2/eval_spatial_mask.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data" / "knowair_v2"
OUT = ROOT / "output" / "phase2" / "eval"
NPZ_DEFAULT = ROOT / "output" / "phase2" / "tensors" / "knowair_bthsa_16x16_2023-01-01_2023-01-31.npz"
META_DEFAULT = ROOT / "output" / "phase2" / "tensors" / "knowair_bthsa_16x16_2023-01-01_2023-01-31.meta.json"
CKPT = (
    ROOT
    / "third_party"
    / "UniST"
    / "src"
    / "experiments"
    / "Pretrain_Dataset_TrafficKnowAir_Task_short_FewRatio_0.5"
    / "model_save"
    / "model_best.pkl"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region", default="bthsa", choices=("bthsa", "yrd"))
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-times", type=int, default=8, help="How many timestamps to average")
    p.add_argument("--time-start", default="2023-01-01")
    p.add_argument("--time-end", default="2023-01-31")
    p.add_argument("--npz", type=Path, default=NPZ_DEFAULT)
    p.add_argument("--meta", type=Path, default=META_DEFAULT)
    p.add_argument("--skip-unist", action="store_true")
    return p.parse_args()


def _idw_points(
    obs_xy: np.ndarray, obs_z: np.ndarray, qry_xy: np.ndarray, power: float = 2.0
) -> np.ndarray:
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
    wsum = w.sum(axis=1)
    wsum = np.where(wsum == 0, np.nan, wsum)
    return (w * obs_z[None, :]).sum(axis=1) / wsum


def _kriging_points(
    obs_xy: np.ndarray, obs_z: np.ndarray, qry_xy: np.ndarray
) -> np.ndarray:
    ok = OrdinaryKriging(
        obs_xy[:, 0],
        obs_xy[:, 1],
        obs_z,
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
    )
    zhat, _ = ok.execute("points", qry_xy[:, 0], qry_xy[:, 1])
    return np.asarray(zhat, dtype=float).ravel()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    yt, yp = y_true[m], y_pred[m]
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(root_mean_squared_error(yt, yp)),
        "n": int(m.sum()),
    }


def eval_station_holdout(
    region: str,
    holdout: float,
    seed: int,
    n_times: int,
    time_start: str,
    time_end: str,
) -> dict:
    ds = xr.open_dataset(DATA / f"dataset_{region}.nc")
    st = pd.read_csv(DATA / f"stations_{region}.csv")
    st = st.set_index(st.station_id.astype(str))
    station_ids = [str(s) for s in ds.station.values]
    lon = st.loc[station_ids, "lon"].to_numpy(dtype=float)
    lat = st.loc[station_ids, "lat"].to_numpy(dtype=float)
    xy = np.column_stack([lon, lat])

    sub = ds.sel(time=slice(time_start, time_end))
    times = pd.to_datetime(sub.time.values)
    # evenly pick n_times indices
    idxs = np.linspace(0, len(times) - 1, num=min(n_times, len(times)), dtype=int)
    rng = np.random.default_rng(seed)
    n = len(station_ids)
    n_hold = max(1, int(round(n * holdout)))

    idw_scores, kri_scores = [], []
    for ti in idxs:
        z = sub["PM2.5"].values[ti].astype(float)
        hold = rng.choice(n, size=n_hold, replace=False)
        obs_mask = np.ones(n, dtype=bool)
        obs_mask[hold] = False
        if obs_mask.sum() < 3:
            continue
        y_true = z[hold]
        idw_pred = _idw_points(xy[obs_mask], z[obs_mask], xy[hold])
        try:
            kri_pred = _kriging_points(xy[obs_mask], z[obs_mask], xy[hold])
        except Exception:
            kri_pred = np.full_like(y_true, np.nan)
        idw_scores.append(_metrics(y_true, idw_pred))
        kri_scores.append(_metrics(y_true, kri_pred))

    ds.close()

    def _avg(scores: list[dict]) -> dict:
        if not scores:
            return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
        return {
            "mae": float(np.nanmean([s["mae"] for s in scores])),
            "rmse": float(np.nanmean([s["rmse"] for s in scores])),
            "n_per_time": int(np.nanmean([s["n"] for s in scores])),
            "n_times": len(scores),
        }

    return {
        "protocol": "station_holdout",
        "region": region,
        "holdout_frac": holdout,
        "seed": seed,
        "time_start": time_start,
        "time_end": time_end,
        "idw": _avg(idw_scores),
        "ordinary_kriging": _avg(kri_scores),
    }


def eval_grid_cell_holdout(
    npz_path: Path, meta_path: Path, holdout: float, seed: int, n_times: int
) -> dict:
    blob = np.load(npz_path)
    data = blob["data"]  # (T,C,H,W)
    occupied = blob["occupied"].astype(bool)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lon_edges = np.asarray(meta["lon_edges"], dtype=float)
    lat_edges = np.asarray(meta["lat_edges"], dtype=float)
    h, w = occupied.shape
    # cell centers (lon, lat); row0 = north
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    rr, cc = np.where(occupied)
    cell_xy = np.column_stack([lon_c[cc], lat_c[rr]])

    rng = np.random.default_rng(seed)
    t_idxs = np.linspace(0, data.shape[0] - 1, num=min(n_times, data.shape[0]), dtype=int)
    n_occ = len(rr)
    n_hold = max(1, int(round(n_occ * holdout)))

    idw_scores, kri_scores = [], []
    for ti in t_idxs:
        z_all = data[ti, 0]
        z_occ = z_all[rr, cc].astype(float)
        hold = rng.choice(n_occ, size=n_hold, replace=False)
        obs = np.ones(n_occ, dtype=bool)
        obs[hold] = False
        if obs.sum() < 3:
            continue
        y_true = z_occ[hold]
        idw_pred = _idw_points(cell_xy[obs], z_occ[obs], cell_xy[hold])
        try:
            kri_pred = _kriging_points(cell_xy[obs], z_occ[obs], cell_xy[hold])
        except Exception:
            kri_pred = np.full_like(y_true, np.nan)
        idw_scores.append(_metrics(y_true, idw_pred))
        kri_scores.append(_metrics(y_true, kri_pred))

    def _avg(scores: list[dict]) -> dict:
        return {
            "mae": float(np.nanmean([s["mae"] for s in scores])),
            "rmse": float(np.nanmean([s["rmse"] for s in scores])),
            "n_per_time": int(np.nanmean([s["n"] for s in scores])),
            "n_times": len(scores),
        }

    return {
        "protocol": "grid_cell_holdout",
        "npz": str(npz_path),
        "holdout_frac": holdout,
        "seed": seed,
        "occupied_cells": int(n_occ),
        "idw": _avg(idw_scores),
        "ordinary_kriging": _avg(kri_scores),
    }


def eval_unist_tube(ckpt: Path) -> dict:
    """Load smoke checkpoint and report tube/block MAE/RMSE on test set."""
    if not ckpt.exists():
        return {"error": f"checkpoint missing: {ckpt}"}

    import os

    unist_src = ROOT / "third_party" / "UniST" / "src"
    sys.path.insert(0, str(unist_src))
    prev_cwd = Path.cwd()
    os.chdir(unist_src)
    import torch
    from DataLoader import data_load_main
    from model import UniST_model
    from train import TrainLoop
    from utils import add_dict_to_argparser
    import argparse as ap

    defaults = dict(
        task="short",
        dataset="TrafficKnowAir",
        mode="testing",
        file_load_path="",
        used_data="single",
        process_name="phase2_eval",
        prompt_ST=0,
        his_len=6,
        pred_len=6,
        few_ratio=1.0,
        stage=0,
        mask_ratio=0.5,
        patch_size=2,
        t_patch_size=2,
        size="middle",
        no_qkv_bias=0,
        pos_emb="SinCos",
        num_memory_spatial=512,
        num_memory_temporal=512,
        conv_num=3,
        prompt_content="s_p_c",
        random=True,
        mask_strategy="tube",
        mask_strategy_random="batch",
        lr=1e-3,
        min_lr=1e-5,
        early_stop=5,
        weight_decay=1e-6,
        batch_size=256,
        log_interval=20,
        total_epoches=1,
        device_id="0",
        machine="local",
        clip_grad=0.05,
        lr_anneal_steps=1,
        batch_size_1=4,
        batch_size_2=4,
        batch_size_3=4,
    )
    parser = ap.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args([])
    args.folder = "Eval_SpatialMask/"
    args.model_path = str(unist_src / "experiments" / args.folder)
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    (Path(args.model_path) / "model_save").mkdir(exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data, test_data, val_data, args.scaler = data_load_main(args)
    assert args.his_len + args.pred_len == args.seq_len

    model = UniST_model(args=args).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    class _DummyWriter:
        def add_scalar(self, *a, **k):
            return None

    loop = TrainLoop(
        args=args,
        writer=_DummyWriter(),
        model=model,
        data=data,
        test_data=test_data,
        val_data=val_data,
        device=device,
        early_stop=args.early_stop,
    )
    # Call Sample directly so we keep both RMSE and MAE (Evaluation drops MAE when random≠none)
    out = {
        "checkpoint": str(ckpt),
        "device": str(device),
        "dataset": "TrafficKnowAir",
        "masks": {},
    }
    try:
        for strategy in ("tube", "block", "random", "temporal"):
            rmse, mae, _loss = loop.Sample(
                test_data,
                0,
                mask_ratio=0.5,
                mask_strategy=strategy,
                seed=42,
                dataset="TrafficKnowAir",
                index=0,
                Type="test",
            )
            out["masks"][strategy] = {
                "0.5": {"rmse": float(rmse), "mae": float(mae)}
            }
            print(f"  UniST {strategy}: MAE={mae:.4f} RMSE={rmse:.4f}", flush=True)
    finally:
        os.chdir(prev_cwd)
    return out


def main() -> int:
    args = _parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # Region-specific default tensors
    if args.region == "yrd" and args.npz == NPZ_DEFAULT:
        args.npz = ROOT / "output/phase2/tensors/knowair_yrd_16x16_2023-01-01_2023-01-31.npz"
        args.meta = ROOT / "output/phase2/tensors/knowair_yrd_16x16_2023-01-01_2023-01-31.meta.json"

    station = eval_station_holdout(
        args.region, args.holdout, args.seed, args.n_times, args.time_start, args.time_end
    )
    grid = eval_grid_cell_holdout(args.npz, args.meta, args.holdout, args.seed, args.n_times)

    report = {
        "seed": args.seed,
        "holdout_frac": args.holdout,
        "station_holdout": station,
        "grid_cell_holdout": grid,
        "notes": [
            "Station hold-out: D1 narrative; IDW/Kriging on raw monitor PM2.5 (ug/m3).",
            "Grid-cell hold-out: same knowair_grid_v1 tensor as UniST; cell-center IDW/Kriging.",
            "UniST: tube/block/random/temporal reconstruction on test windows (ug/m3 after scaler inverse).",
            "Smoke checkpoint is only ~2 epochs on a tiny window — expect UniST >> classical errors until longer pretrain; empty grid cells filled with 0 also inflate UniST error.",
            "Protocols are complementary, not identical pointwise matches.",
        ],
    }

    if not args.skip_unist:
        print("Running UniST tube/block eval...", flush=True)
        report["unist"] = eval_unist_tube(CKPT)

    out_path = OUT / f"spatial_mask_{args.region}_seed{args.seed}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown summary table
    md = []
    md.append(f"# Spatial mask vs baselines ({args.region})\n")
    md.append(f"- seed={args.seed}, holdout={args.holdout}\n")
    md.append("\n## Station hold-out (PM2.5)\n")
    md.append("| Method | MAE | RMSE |\n|--------|-----|------|\n")
    md.append(
        f"| IDW | {station['idw']['mae']:.4f} | {station['idw']['rmse']:.4f} |\n"
    )
    md.append(
        f"| Ordinary Kriging | {station['ordinary_kriging']['mae']:.4f} | {station['ordinary_kriging']['rmse']:.4f} |\n"
    )
    md.append("\n## Grid-cell hold-out\n")
    md.append("| Method | MAE | RMSE |\n|--------|-----|------|\n")
    md.append(
        f"| IDW | {grid['idw']['mae']:.4f} | {grid['idw']['rmse']:.4f} |\n"
    )
    md.append(
        f"| Ordinary Kriging | {grid['ordinary_kriging']['mae']:.4f} | {grid['ordinary_kriging']['rmse']:.4f} |\n"
    )
    if "unist" in report and "masks" in report["unist"]:
        md.append("\n## UniST (grid mask reconstruction on test windows)\n")
        md.append("| Mask | MAE | RMSE |\n|------|-----|------|\n")
        for s, ratios in report["unist"]["masks"].items():
            for m, v in ratios.items():
                md.append(f"| {s}@{m} | {v['mae']:.4f} | {v['rmse']:.4f} |\n")
        md.append(
            "\n> UniST 为冒烟权重（约 2 epoch）+ 空格填 0，误差偏大属预期；"
            "与站点 hold-out 协议不完全相同，解读时以管线跑通为主。\n"
        )
    md_path = OUT / f"spatial_mask_{args.region}_seed{args.seed}.md"
    md_path.write_text("".join(md), encoding="utf-8")

    print("".join(md))
    print(f"wrote {out_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
