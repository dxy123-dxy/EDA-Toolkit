"""UniST backend for ``dl_infer`` (loads third_party/UniST checkpoint)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "dl_infer (UniST) requires torch. Install the ML env: "
            'pip install -e ".[ml]" (see examples/phase2/README.md)'
        ) from exc
    return torch


def _cell_centers(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    lon_edges = np.asarray(meta["lon_edges"], dtype=float)
    lat_edges = np.asarray(meta["lat_edges"], dtype=float)
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    return lon_c, lat_c


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(m.sum())
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    yt, yp = y_true[m], y_pred[m]
    err = yp - yt
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "n": n,
    }


def run_unist_infer(params: dict[str, Any]) -> dict[str, Any]:
    """
    Run UniST tube/block mask reconstruction on the named dataset's test split.

    Required params: checkpoint, dataset.
    Optional: unist_src, prompt_ST, num_memory, mask_ratio, mask_strategy, seed,
    max_windows, batch_size, meta_path, tensor_path, column, device.
    """
    torch = _require_torch()

    checkpoint = Path(params["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    dataset = str(params["dataset"])
    unist_src = Path(params.get("unist_src") or _default_unist_src())
    if not unist_src.is_dir():
        raise FileNotFoundError(f"UniST src not found: {unist_src}")

    mask_ratio = float(params.get("mask_ratio", 0.5))
    mask_strategy = str(params.get("mask_strategy", "tube"))
    seed = int(params.get("seed", 42))
    max_windows = int(params.get("max_windows", 8))
    batch_size = int(params.get("batch_size", 4))
    patch_size = int(params.get("patch_size", 2))
    prompt_st = int(params.get("prompt_ST", 0))
    num_memory = int(params.get("num_memory", 512))
    column = str(params.get("column", "pm25"))

    meta: dict | None = None
    occupied: np.ndarray | None = None
    meta_path = params.get("meta_path")
    tensor_path = params.get("tensor_path")
    if meta_path:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    if tensor_path and Path(tensor_path).exists():
        blob = np.load(tensor_path)
        if "occupied" in blob.files:
            occupied = blob["occupied"].astype(bool)

    sys.path.insert(0, str(unist_src))
    prev_cwd = Path.cwd()
    os.chdir(unist_src)
    try:
        from DataLoader import data_load_main
        from model import UniST_model
        from utils import add_dict_to_argparser
        import argparse as ap

        defaults = dict(
            task="short",
            dataset=dataset,
            mode="testing",
            file_load_path="",
            used_data="single",
            process_name="eda_toolkit_dl_infer",
            prompt_ST=prompt_st,
            his_len=6,
            pred_len=6,
            few_ratio=1.0,
            stage=0,
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            t_patch_size=2,
            size="middle",
            no_qkv_bias=0,
            pos_emb="SinCos",
            num_memory_spatial=num_memory,
            num_memory_temporal=num_memory,
            conv_num=3,
            prompt_content="s_p_c",
            random=True,
            mask_strategy=mask_strategy,
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
        args.folder = "DlInfer/"
        args.model_path = str(unist_src / "experiments" / args.folder)
        Path(args.model_path).mkdir(parents=True, exist_ok=True)
        (Path(args.model_path) / "model_save").mkdir(exist_ok=True)

        if params.get("device"):
            device = torch.device(str(params["device"]))
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        _data, test_data, _val, args.scaler = data_load_main(args)
        model = UniST_model(args=args).to(device)
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()
        scaler = args.scaler[dataset]

        yt_all: list[np.ndarray] = []
        yp_all: list[np.ndarray] = []
        last_pred: np.ndarray | None = None
        last_true: np.ndarray | None = None
        n_windows = 0

        loader = test_data[0]
        with torch.no_grad():
            for batch in loader:
                if n_windows >= max_windows:
                    break
                batch = [b.to(device) for b in batch]
                n_in = batch[0].shape[0]
                take = min(n_in, max_windows - n_windows)
                if take < n_in:
                    batch = [b[:take] for b in batch]

                _l1, _l2, pred, target, mask = model(
                    batch,
                    mask_ratio=mask_ratio,
                    mask_strategy=mask_strategy,
                    seed=seed,
                    data=dataset,
                    mode="forward",
                )
                # mask: 1 = masked (scored); pred/target patch tokens
                pred_grid = model.unpatchify(pred).detach().cpu().numpy()
                true_grid = model.unpatchify(target).detach().cpu().numpy()
                mask_np = mask.detach().cpu().numpy()

                # Expand patch mask → cell mask (patch_size×patch_size)
                # mask shape after UniST: (N, T_tokens * spatial) flattened; use grids for scoring
                # Score all cells where occupied (if known), else all finite cells.
                true_inv = scaler.inverse_transform(true_grid.reshape(-1, 1)).reshape(true_grid.shape)
                pred_inv = scaler.inverse_transform(pred_grid.reshape(-1, 1)).reshape(pred_grid.shape)

                # Approximate masked cells via patch mask unpatchify of ones
                ones = torch.ones_like(pred)
                # Rebuild mask grid: UniST mask is over tokens; use |pred-target| proxy only on occupied
                n_b, t_len, h, w = true_inv.shape
                cell_mask = _token_mask_to_cells(mask_np, h, w, patch_size, t_len)

                for ni in range(n_b):
                    for ti in range(t_len):
                        mcell = cell_mask[ni, ti].astype(bool)
                        if occupied is not None:
                            mcell = mcell & occupied
                        yt = true_inv[ni, ti][mcell].astype(float)
                        yp = pred_inv[ni, ti][mcell].astype(float)
                        if yt.size:
                            yt_all.append(yt)
                            yp_all.append(yp)

                last_pred = pred_inv[-1, t_len // 2]  # mid-frame of last window
                last_true = true_inv[-1, t_len // 2]
                last_cell_mask = cell_mask[-1, t_len // 2]
                n_windows += take

        yt_cat = np.concatenate(yt_all) if yt_all else np.array([])
        yp_cat = np.concatenate(yp_all) if yp_all else np.array([])
        scores = _metrics(yt_cat, yp_cat)

        features: list[dict[str, Any]] = []
        if last_pred is not None and meta is not None:
            lon_c, lat_c = _cell_centers(meta)
            h, w = last_pred.shape
            for r in range(h):
                for c in range(w):
                    if occupied is not None and not occupied[r, c]:
                        continue
                    features.append(
                        {
                            "row": int(r),
                            "col": int(c),
                            "lon": float(lon_c[c]),
                            "lat": float(lat_c[r]),
                            column: float(last_pred[r, c]),
                            f"{column}_true": float(last_true[r, c]),
                            "masked": bool(last_cell_mask[r, c]),
                            "occupied": True if occupied is None else bool(occupied[r, c]),
                        }
                    )

        return {
            "backend": "unist",
            "dataset": dataset,
            "checkpoint": str(checkpoint),
            "device": str(device),
            "mask_ratio": mask_ratio,
            "mask_strategy": mask_strategy,
            "seed": seed,
            "n_windows": n_windows,
            "column": column,
            "metrics": scores,
            "features": features,
            "meta_path": str(meta_path) if meta_path else None,
            "tensor_path": str(tensor_path) if tensor_path else None,
        }
    finally:
        os.chdir(prev_cwd)


def _default_unist_src() -> Path:
    from eda_toolkit.ml.presets import repo_root

    return repo_root() / "third_party" / "UniST" / "src"


def _token_mask_to_cells(
    mask_np: np.ndarray,
    h: int,
    w: int,
    patch_size: int,
    t_len: int,
) -> np.ndarray:
    """
    Convert UniST flattened token mask (N, L) into (N, T, H, W) cell mask.

    UniST tube mask is over spatial patches, repeated across time patches.
    We treat any temporal token that masks a spatial patch as masking that cell
    for all frames in the window (conservative spatial hold-out view).
    """
    n, l = mask_np.shape
    ph, pw = h // patch_size, w // patch_size
    n_spatial = ph * pw
    # L = T_patch * spatial; t_patch_size usually 2 → n_t_tokens = t_len // t_patch
    if l % n_spatial != 0:
        # Fallback: mark all cells masked
        return np.ones((n, t_len, h, w), dtype=np.float32)

    n_t_tok = l // n_spatial
    m = mask_np.reshape(n, n_t_tok, n_spatial)
    # Spatial patch masked if any time-token marks it
    spat = (m.max(axis=1) > 0.5).astype(np.float32)  # (N, n_spatial)
    spat = spat.reshape(n, ph, pw)
    # Upsample patches → cells
    cell = np.repeat(np.repeat(spat, patch_size, axis=1), patch_size, axis=2)
    # Broadcast over time
    return np.broadcast_to(cell[:, None, :, :], (n, t_len, h, w)).copy()
