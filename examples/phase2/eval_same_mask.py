"""Same spatial mask: UniST vs IDW vs Ordinary Kriging.

Protocol (tube-style spatial hold-out, seed-fixed):
  1) Pick holdout_frac of occupied cells (seed).
  2) Map those cells → spatial patches (patch_size=2); those patches are masked
     for UniST tube evaluation (forced, not random).
  3) On every frame of each UniST test window:
       - IDW / Ordinary Kriging: predict holdout occupied cells from remaining
       - UniST: reconstruct with forced tube mask; score the same cells after unpatchify
  4) Report one table in original PM2.5 units (scaler inverse).

Usage (repo root):
  # In-domain: BTHSA Long Stage-1
  .venv-ml\\Scripts\\python.exe examples/phase2/eval_same_mask.py --region bthsa

  # Cross-scene: YRD Prompt (BTHSA Long → YRD)
  .venv-ml\\Scripts\\python.exe examples/phase2/eval_same_mask.py --region yrd
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
UNIST_SRC = ROOT / "third_party" / "UniST" / "src"
OUT = ROOT / "output" / "phase2" / "eval"
PATCH = 2

CKPTS = {
    "bthsa": {
        "dataset": "TrafficKnowAirLong",
        "ckpt": UNIST_SRC
        / "experiments"
        / "Pretrain_Dataset_TrafficKnowAirLong_Task_short_FewRatio_1.0"
        / "model_save"
        / "model_best.pkl",
        "npz": ROOT / "output/phase2/tensors/knowair_bthsa_16x16_2022-01-01_2023-12-31.npz",
        "meta": ROOT / "output/phase2/tensors/knowair_bthsa_16x16_2022-01-01_2023-12-31.meta.json",
        "prompt_ST": 0,
        "num_memory": 512,
    },
    "yrd": {
        "dataset": "TrafficKnowAirYRDLong",
        "ckpt": UNIST_SRC
        / "experiments"
        / "Prompt_Dataset_TrafficKnowAirYRDLong_Task_short_FewRatio_0.5"
        / "model_save"
        / "model_best.pkl",
        "npz": ROOT / "output/phase2/tensors/knowair_yrd_16x16_2022-01-01_2023-12-31.npz",
        "meta": ROOT / "output/phase2/tensors/knowair_yrd_16x16_2022-01-01_2023-12-31.meta.json",
        "prompt_ST": 1,
        "num_memory": 256,
    },
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region", choices=("bthsa", "yrd"), default="yrd")
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-windows", type=int, default=24, help="Cap test windows for speed")
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def _idw(obs_xy: np.ndarray, obs_z: np.ndarray, qry_xy: np.ndarray, power: float = 2.0) -> np.ndarray:
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


def _kriging(obs_xy: np.ndarray, obs_z: np.ndarray, qry_xy: np.ndarray) -> np.ndarray:
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
    if int(m.sum()) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    yt, yp = y_true[m], y_pred[m]
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(root_mean_squared_error(yt, yp)),
        "n": int(m.sum()),
    }


def _cell_centers(meta: dict, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    lon_edges = np.asarray(meta["lon_edges"], dtype=float)
    lat_edges = np.asarray(meta["lat_edges"], dtype=float)
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    return lon_c, lat_c


def _choose_holdout(
    occupied: np.ndarray, holdout: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return hold_rr, hold_cc, obs_rr, obs_cc (occupied only)."""
    rr, cc = np.where(occupied)
    n = len(rr)
    n_hold = max(1, int(round(n * holdout)))
    rng = np.random.default_rng(seed)
    hold_i = rng.choice(n, size=n_hold, replace=False)
    obs_i = np.array([i for i in range(n) if i not in set(hold_i.tolist())], dtype=int)
    return rr[hold_i], cc[hold_i], rr[obs_i], cc[obs_i]


def _cells_to_patches(hold_rr: np.ndarray, hold_cc: np.ndarray, h: int, w: int, p: int = PATCH) -> np.ndarray:
    ph, pw = h // p, w // p
    patches = sorted({int(r // p) * pw + int(c // p) for r, c in zip(hold_rr, hold_cc)})
    return np.asarray(patches, dtype=np.int64), ph, pw


def _install_fixed_tube_mask(hold_patches: np.ndarray, n_spatial: int) -> None:
    """Force tube_masking_evaluate to mask exactly hold_patches.

    UniST does ``from mask_strategy import *`` into ``model``, so patch both
    the source module and ``model.tube_masking_evaluate``.
    """
    import mask_strategy as ms
    import model as unist_model

    hold = np.asarray(hold_patches, dtype=np.int64)
    keep = np.asarray([i for i in range(n_spatial) if i not in set(hold.tolist())], dtype=np.int64)
    if len(keep) == 0:
        raise ValueError("holdout left zero patches to keep")

    def tube_masking_evaluate(x, mask_ratio, T):  # noqa: ARG001
        N, L, D = x.shape
        x = x.reshape(N, T, L // T, D)
        N, T_, Lsp, _C = x.shape
        if Lsp != n_spatial:
            raise ValueError(f"spatial tokens {Lsp} != expected {n_spatial}")
        device = x.device
        len_keep = len(keep)
        ids_shuffle = torch.empty(N, Lsp, dtype=torch.long, device=device)
        ids_shuffle[:, :len_keep] = torch.as_tensor(keep, device=device)
        ids_shuffle[:, len_keep:] = torch.as_tensor(hold, device=device)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=2, index=ids_keep.unsqueeze(1).unsqueeze(-1).repeat(1, T_, 1, D)
        )
        mask = torch.ones([N, T_, Lsp], device=device)
        mask[:, :, :len_keep] = 0
        mask = torch.gather(mask, dim=2, index=ids_restore.unsqueeze(1).repeat(1, T_, 1)).reshape(N, -1)
        ids_keep = ids_keep.unsqueeze(1).repeat(1, T_, 1).reshape(N, -1)
        x_masked = x_masked.reshape(N, -1, x_masked.shape[-1])
        return x_masked, mask, ids_restore, ids_keep

    ms.tube_masking_evaluate = tube_masking_evaluate
    unist_model.tube_masking_evaluate = tube_masking_evaluate


def _load_unist(cfg: dict, batch_size: int):
    sys.path.insert(0, str(UNIST_SRC))
    prev = Path.cwd()
    os.chdir(UNIST_SRC)
    from DataLoader import data_load_main
    from model import UniST_model
    from utils import add_dict_to_argparser
    import argparse as ap

    defaults = dict(
        task="short",
        dataset=cfg["dataset"],
        mode="testing",
        file_load_path="",
        used_data="single",
        process_name="phase2_same_mask",
        prompt_ST=cfg["prompt_ST"],
        his_len=6,
        pred_len=6,
        few_ratio=1.0,
        stage=0,
        mask_ratio=0.5,
        patch_size=PATCH,
        t_patch_size=2,
        size="middle",
        no_qkv_bias=0,
        pos_emb="SinCos",
        num_memory_spatial=cfg["num_memory"],
        num_memory_temporal=cfg["num_memory"],
        conv_num=3,
        prompt_content="s_p_c",
        random=True,
        mask_strategy="tube",
        mask_strategy_random="none",
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
        batch_size_1=batch_size,
        batch_size_2=batch_size,
        batch_size_3=batch_size,
    )
    parser = ap.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args([])
    args.folder = "Eval_SameMask/"
    args.model_path = str(UNIST_SRC / "experiments" / args.folder)
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    (Path(args.model_path) / "model_save").mkdir(exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data, test_data, val_data, args.scaler = data_load_main(args)
    model = UniST_model(args=args).to(device)
    state = torch.load(cfg["ckpt"], map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return args, model, test_data, device, prev


def main() -> int:
    args = _parse_args()
    cfg = CKPTS[args.region]
    if not cfg["ckpt"].exists():
        print(f"Missing checkpoint {cfg['ckpt']}", file=sys.stderr)
        return 1
    if not cfg["npz"].exists() or not cfg["meta"].exists():
        print(f"Missing tensor/meta for {args.region}", file=sys.stderr)
        return 1

    blob = np.load(cfg["npz"])
    occupied = blob["occupied"].astype(bool)
    meta = json.loads(cfg["meta"].read_text(encoding="utf-8"))
    h, w = occupied.shape
    lon_c, lat_c = _cell_centers(meta, h, w)

    hold_rr, hold_cc, obs_rr, obs_cc = _choose_holdout(occupied, args.holdout, args.seed)
    hold_patches, ph, pw = _cells_to_patches(hold_rr, hold_cc, h, w)
    n_spatial = ph * pw
    mask_ratio = float(len(hold_patches) / n_spatial)

    hold_xy = np.column_stack([lon_c[hold_cc], lat_c[hold_rr]])
    obs_xy = np.column_stack([lon_c[obs_cc], lat_c[obs_rr]])

    print(
        f"region={args.region} occupied={int(occupied.sum())} "
        f"hold_cells={len(hold_rr)} hold_patches={len(hold_patches)}/{n_spatial} "
        f"mask_ratio≈{mask_ratio:.3f}",
        flush=True,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    unist_args, model, test_loaders, device, prev_cwd = _load_unist(cfg, args.batch_size)
    scaler = unist_args.scaler[cfg["dataset"]]
    _install_fixed_tube_mask(hold_patches, n_spatial)

    idw_yt, idw_yp = [], []
    kri_yt, kri_yp = [], []
    dl_yt, dl_yp = [], []
    n_windows = 0

    try:
        loader = test_loaders[0]
        with torch.no_grad():
            for batch in loader:
                if n_windows >= args.max_windows:
                    break
                batch = [b.to(device) for b in batch]
                imgs = batch[0]  # N,1,T,H,W scaled
                n_in_batch = imgs.shape[0]
                take = min(n_in_batch, args.max_windows - n_windows)
                if take < n_in_batch:
                    batch = [b[:take] for b in batch]
                    imgs = batch[0]

                loss1, loss2, pred, target, mask = model(
                    batch,
                    mask_ratio=mask_ratio,
                    mask_strategy="tube",
                    seed=args.seed,
                    data=cfg["dataset"],
                    mode="forward",
                )
                # pred/target: (N, L, patch) → unpatchify needs patch_info set by patchify in forward_loss
                pred_grid = model.unpatchify(pred).detach().cpu().numpy()  # N,T,H,W scaled
                true_grid = model.unpatchify(target).detach().cpu().numpy()

                # Classical baselines on inverse-transformed true frames
                true_inv = scaler.inverse_transform(true_grid.reshape(-1, 1)).reshape(true_grid.shape)
                pred_inv = scaler.inverse_transform(pred_grid.reshape(-1, 1)).reshape(pred_grid.shape)

                n_b, t_len, _, _ = true_inv.shape
                for ni in range(n_b):
                    for ti in range(t_len):
                        frame = true_inv[ni, ti]
                        y_true = frame[hold_rr, hold_cc].astype(float)
                        z_obs = frame[obs_rr, obs_cc].astype(float)
                        if np.isfinite(z_obs).sum() < 3:
                            continue
                        # drop non-finite obs
                        m_obs = np.isfinite(z_obs)
                        idw_pred = _idw(obs_xy[m_obs], z_obs[m_obs], hold_xy)
                        try:
                            kri_pred = _kriging(obs_xy[m_obs], z_obs[m_obs], hold_xy)
                        except Exception:
                            kri_pred = np.full_like(y_true, np.nan)

                        idw_yt.append(y_true)
                        idw_yp.append(idw_pred)
                        kri_yt.append(y_true)
                        kri_yp.append(kri_pred)

                        dl_pred = pred_inv[ni, ti, hold_rr, hold_cc].astype(float)
                        dl_yt.append(y_true)
                        dl_yp.append(dl_pred)

                n_windows += take
                print(f"  windows done: {n_windows}/{args.max_windows}", flush=True)
    finally:
        os.chdir(prev_cwd)

    def _cat(chunks: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(chunks, axis=0) if chunks else np.array([])

    report = {
        "protocol": "same_spatial_tube_holdout",
        "region": args.region,
        "dataset": cfg["dataset"],
        "checkpoint": str(cfg["ckpt"]),
        "seed": args.seed,
        "holdout_frac_cells": args.holdout,
        "n_hold_cells": int(len(hold_rr)),
        "n_obs_cells": int(len(obs_rr)),
        "n_hold_patches": int(len(hold_patches)),
        "n_spatial_patches": int(n_spatial),
        "mask_ratio_patches": mask_ratio,
        "n_windows": n_windows,
        "methods": {
            "idw": _metrics(_cat(idw_yt), _cat(idw_yp)),
            "ordinary_kriging": _metrics(_cat(kri_yt), _cat(kri_yp)),
            "unist": _metrics(_cat(dl_yt), _cat(dl_yp)),
        },
        "notes": [
            "All three methods scored on the SAME occupied holdout cells.",
            "Holdout cells → UniST tube patches (patch_size=2); mask forced (not random).",
            "IDW/Kriging are purely spatial per-frame; UniST uses full-window tube reconstruction.",
            "Metrics in original PM2.5 units after MinMax inverse.",
            "Empty (non-occupied) cells excluded from scoring.",
        ],
    }

    out_json = OUT / f"same_mask_{args.region}_seed{args.seed}.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    m = report["methods"]
    md = []
    md.append(f"# Same-mask comparison ({args.region})\n\n")
    md.append(
        f"- protocol: spatial tube hold-out on occupied cells  \n"
        f"- seed={args.seed}, holdout_cells={args.holdout}, "
        f"hold_cells={len(hold_rr)}, hold_patches={len(hold_patches)}/{n_spatial}  \n"
        f"- windows={n_windows}, dataset=`{cfg['dataset']}`  \n\n"
    )
    md.append("| Method | MAE | RMSE | n |\n|--------|-----|------|---|\n")
    for name, label in (
        ("idw", "IDW"),
        ("ordinary_kriging", "Ordinary Kriging"),
        ("unist", "UniST"),
    ):
        s = m[name]
        md.append(f"| {label} | {s['mae']:.4f} | {s['rmse']:.4f} | {s['n']} |\n")
    md.append(
        "\n> Same occupied holdout cells for all methods. "
        "UniST = forced tube mask on patches covering those cells.\n"
    )
    out_md = OUT / f"same_mask_{args.region}_seed{args.seed}.md"
    out_md.write_text("".join(md), encoding="utf-8")

    print("".join(md))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
