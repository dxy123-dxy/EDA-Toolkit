"""Plot UniST-first Demo D1/D2 figures (no IDW on main stage).

D1 (paper BikeNYC / TrafficHZ):
  - 01_unist_mask_bars_{dataset}.png
  - 02_unist_recon_heatmap_{dataset}.png  (true vs recon, if checkpoint)

D2 (KnowAir cross-scene):
  - 05_cross_scene_prompt_vs_scratch.png
  - 06_unist_cross_scene_summary.png

Usage (repo root):
  .venv-ml\\Scripts\\python.exe examples/phase2/plot_unist_demo_figures.py
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "output" / "phase2" / "eval"
UNIST_SRC = ROOT / "third_party" / "UniST" / "src"
PY_DATASET = ROOT / "third_party" / "UniST" / "dataset"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _masks_from_report(rep: dict) -> dict:
    metrics = rep.get("metrics") or {}
    masks = metrics.get("test_masks") or {}
    if masks:
        return masks
    # parse raw result_txt if needed
    raw = rep.get("result_txt") or metrics.get("raw") or ""
    if not raw:
        return {}
    blocks = list(re.finditer(r"test rmse:\s*[0-9.eE+-]+\s*\n(\{.*?\})", raw, re.S))
    if not blocks:
        return {}
    try:
        parsed = ast.literal_eval(blocks[-1].group(1))
    except (SyntaxError, ValueError):
        return {}
    out = {}
    for _ds, m in parsed.items():
        for strategy, ratios in m.items():
            for ratio, val in ratios.items():
                if isinstance(val, dict):
                    out[strategy] = {
                        "rmse": float(val.get("rmse", float("nan"))),
                        "mae": float(val.get("mae", float("nan"))),
                    }
                else:
                    out[strategy] = {"rmse": float(val), "mae": float("nan")}
    return out


def plot_mask_bars(dataset: str, masks: dict, out: Path) -> Path | None:
    order = ["random", "temporal", "tube", "block"]
    present = [k for k in order if k in masks]
    if not present:
        return None
    rmse = [masks[k]["rmse"] for k in present]
    mae = [masks[k].get("mae", float("nan")) for k in present]
    has_mae = all(np.isfinite(mae))

    x = np.arange(len(present))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    if has_mae:
        w = 0.36
        ax.bar(x - w / 2, mae, w, label="MAE", color="#3b82f6")
        ax.bar(x + w / 2, rmse, w, label="RMSE", color="#f59e0b")
        for i, (m, r) in enumerate(zip(mae, rmse)):
            ax.text(i - w / 2, m, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + w / 2, r, f"{r:.2f}", ha="center", va="bottom", fontsize=8)
        ax.legend()
    else:
        bars = ax.bar(x, rmse, width=0.55, color="#2563eb", label="RMSE")
        for bar, v in zip(bars, rmse):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(present)
    ax.set_ylabel("Error (original scale)")
    ax.set_title(f"UniST Stage-1 mask reconstruction — {dataset} (paper data)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    dest = out / f"01_unist_mask_bars_{dataset.lower()}.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


def plot_recon_heatmap(dataset: str, ckpt: Path | None, out: Path) -> Path | None:
    """True vs tube-recon mid-frame heatmap for one test window."""
    if ckpt is None or not Path(ckpt).exists():
        return None
    torch = __import__("torch")
    if not torch.cuda.is_available():
        # still try CPU for tiny BikeNYC window
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    sys.path.insert(0, str(UNIST_SRC))
    prev = Path.cwd()
    os.chdir(UNIST_SRC)
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
            process_name="plot",
            prompt_ST=0,
            his_len=6,
            pred_len=6,
            few_ratio=0.0,
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
            mask_strategy_random="none",
            lr=3e-4,
            min_lr=1e-5,
            early_stop=5,
            weight_decay=1e-6,
            batch_size=4,
            log_interval=20,
            total_epoches=1,
            device_id="0",
            machine="local",
            clip_grad=0.05,
            lr_anneal_steps=1,
            batch_size_1=4,
            batch_size_2=2,
            batch_size_3=1,
        )
        parser = ap.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        args = parser.parse_args([])
        data, test_data, val_data, scaler = data_load_main(args)
        args.scaler = scaler
        assert args.his_len + args.pred_len == args.seq_len

        model = UniST_model(args=args).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()

        # first test loader
        loader = test_data[0]
        batch = next(iter(loader))
        # batch: list of tensors or collated
        with torch.no_grad():
            # mirror train.model_forward path via model __call__
            loss, pred, target, mask = None, None, None, None
            # UniST_model forward signature varies; use train helper if needed
            from train import TrainLoop

            class _W:
                def add_scalar(self, *a, **k):
                    pass

            loop = TrainLoop(args, _W(), model, data, test_data, val_data, device, early_stop=1)
            # take one batch from test_data[0]
            for b in loader:
                loss, _, pred, target, mask = loop.model_forward(
                    b, model, 0.5, "tube", seed=42, data=dataset, mode="forward"
                )
                break

        pred = torch.clamp(pred, -1, 1)
        # pred/target are patch tokens (N, L, u*p*p); restore to (N, T, H, W)
        try:
            pred_grid = model.unpatchify(pred)
            tgt_grid = model.unpatchify(target)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip heatmap: unpatchify failed ({exc}); pred={tuple(pred.shape)}")
            return None
        pred_np = pred_grid.detach().cpu().numpy()
        tgt_np = tgt_grid.detach().cpu().numpy()
        if pred_np.ndim != 4:
            print(f"  skip heatmap: unexpected shape {pred_np.shape}")
            return None
        t_mid = pred_np.shape[1] // 2
        sc = scaler[dataset]

        def inv(g):
            flat = g.reshape(-1, 1)
            return sc.inverse_transform(flat).reshape(g.shape)

        true_g = inv(tgt_np[0, t_mid])
        pred_g = inv(pred_np[0, t_mid])

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        vmin = float(np.nanpercentile(true_g, 5))
        vmax = float(np.nanpercentile(true_g, 95))
        im0 = axes[0].imshow(true_g, cmap="YlOrRd", vmin=vmin, vmax=vmax, origin="lower")
        axes[0].set_title("Ground truth")
        im1 = axes[1].imshow(pred_g, cmap="YlOrRd", vmin=vmin, vmax=vmax, origin="lower")
        axes[1].set_title("UniST recon (tube)")
        err = pred_g - true_g
        lim = float(np.nanmax(np.abs(err))) + 1e-6
        im2 = axes[2].imshow(err, cmap="coolwarm", vmin=-lim, vmax=lim, origin="lower")
        axes[2].set_title("Error (pred−true)")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im1, ax=axes[:2].ravel().tolist(), shrink=0.85, fraction=0.046)
        fig.colorbar(im2, ax=axes[2], shrink=0.85, fraction=0.046)
        fig.suptitle(f"UniST reconstruction — {dataset}", y=1.02)
        fig.tight_layout()
        dest = out / f"02_unist_recon_heatmap_{dataset.lower()}.png"
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return dest
    except Exception as exc:  # noqa: BLE001
        print(f"  heatmap failed ({dataset}): {exc}")
        return None
    finally:
        os.chdir(prev)


def plot_cross_scene(out: Path) -> list[Path]:
    paths: list[Path] = []
    cross = _load_json(EVAL / "cross_scene_bthsa_long_to_yrd.json") or {}
    m = cross.get("metrics") or {}
    if not m:
        return paths

    labels = ["Prompt\n(BTHSA→YRD)", "Scratch\n(YRD, same budget)"]
    rmse = [
        float(m["prompt_test_temporal_rmse"]),
        float(m["scratch_test_temporal_rmse"]),
    ]
    mae_p = float(m.get("prompt_test_temporal_mae", float("nan")))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(x, rmse, width=0.55, color=["#16a34a", "#f97316"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test temporal RMSE (PM2.5 μg/m³)")
    ax.set_title("UniST cross-scene: Prompt vs Scratch (KnowAir YRDLong)")
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
    if np.isfinite(mae_p):
        ax.annotate(
            f"Prompt MAE ≈ {mae_p:.2f}",
            xy=(0, rmse[0]),
            xytext=(0.4, rmse[0] + 0.9),
            fontsize=8,
            color="#166534",
            arrowprops=dict(arrowstyle="->", color="#166534", lw=0.8),
        )
    fig.tight_layout()
    p1 = out / "05_cross_scene_prompt_vs_scratch.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)

    # Summary card-style figure for D2 hero (ASCII to avoid missing CJK glyphs)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axis("off")
    improve = (rmse[1] - rmse[0]) / rmse[1] * 100
    ax.text(0.02, 0.75, "Demo D2 · UniST cross-scene transfer", fontsize=14, fontweight="bold", transform=ax.transAxes)
    ax.text(
        0.02,
        0.42,
        f"Prompt RMSE  {rmse[0]:.2f}    vs    Scratch RMSE  {rmse[1]:.2f}",
        fontsize=13,
        transform=ax.transAxes,
        family="monospace",
    )
    ax.text(
        0.02,
        0.15,
        f"~{improve:.0f}% lower than same-budget scratch  ·  BTHSA Long -> YRD Prompt (5 epoch)",
        fontsize=11,
        color="#166534",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    p2 = out / "06_unist_cross_scene_summary.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)
    return paths


def plot_dataset_inventory(out: Path) -> Path | None:
    link = _load_json(EVAL / "paper_data_link.json")
    src = Path((link or {}).get("source") or r"E:\数据分析模型训练数据\data_release")
    if not src.is_dir():
        return None
    files = sorted(src.glob("*_short.json"))
    if not files:
        return None
    names = [p.stem.replace("_short", "") for p in files]
    sizes = [p.stat().st_size / 1e9 for p in files]
    highlight = {"BikeNYC", "TrafficHZ"}

    fig, ax = plt.subplots(figsize=(10, 4.2))
    colors = ["#2563eb" if n in highlight else "#94a3b8" for n in names]
    ax.barh(names, sizes, color=colors)
    ax.set_xlabel("Size (GB)")
    ax.set_title("UniST official paper datasets (blue = Demo D1)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    dest = out / "00_paper_dataset_inventory.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-heatmap", action="store_true")
    args = ap.parse_args()
    EVAL.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    inv = plot_dataset_inventory(EVAL)
    if inv:
        written.append(inv)

    for ds_key in ("bikenyc", "traffichz"):
        rep = _load_json(EVAL / f"paper_demo_{ds_key}.json")
        if not rep:
            # try combined
            comb = _load_json(EVAL / "paper_demo_bikenyc.json")
            if comb and "runs" in comb:
                rep = next((r for r in comb["runs"] if r.get("dataset", "").lower() == ds_key), None)
        if not rep:
            print(f"skip bars: no metrics for {ds_key}")
            continue
        dataset = rep.get("dataset") or ds_key
        masks = _masks_from_report(rep)
        p = plot_mask_bars(dataset, masks, EVAL)
        if p:
            written.append(p)
            print("wrote", p)
        if not args.skip_heatmap:
            ckpt = rep.get("ckpt")
            hp = plot_recon_heatmap(dataset, Path(ckpt) if ckpt else None, EVAL)
            if hp:
                written.append(hp)
                print("wrote", hp)

    written.extend(plot_cross_scene(EVAL))
    print("Wrote:")
    for p in written:
        print(" ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
