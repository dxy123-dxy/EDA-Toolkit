"""Assemble Phase-2 Demo folders for M3/M4 presentation.

D1 站点填图: station hold-out maps + classic baselines (+ same-mask table)
D2 面/网格补全: cross-scene Prompt + same-mask grid comparison

Usage (repo root):
  .venv-ml\\Scripts\\python.exe examples/phase2/pack_demos.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "output" / "phase2" / "eval"
DEMO1 = ROOT / "output" / "phase2" / "demo_d1"
DEMO2 = ROOT / "output" / "phase2" / "demo_d2"


def _copy(src: Path, dst_dir: Path) -> str | None:
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return src.name


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pack_d1() -> list[str]:
    """D1: station fill-in narrative (BTHSA)."""
    figs = DEMO1 / "figures"
    metrics = DEMO1 / "metrics"
    copied: list[str] = []

    for name in (
        "02_station_holdout_map_bthsa.png",
        "02_station_holdout_map.png",
        "03_pred_vs_true_scatter_bthsa.png",
        "04_holdout_error_map_bthsa.png",
        "01_metrics_bars_bthsa.png",
    ):
        got = _copy(EVAL / name, figs)
        if got:
            copied.append(f"figures/{got}")

    for name in (
        "spatial_mask_bthsa_seed42.json",
        "spatial_mask_bthsa_seed42.md",
        "same_mask_bthsa_seed42.json",
        "same_mask_bthsa_seed42.md",
    ):
        got = _copy(EVAL / name, metrics)
        if got:
            copied.append(f"metrics/{got}")

    station = _load(EVAL / "spatial_mask_bthsa_seed42.json")
    same = _load(EVAL / "same_mask_bthsa_seed42.json")

    idw_s = station["station_holdout"]["idw"] if station else {}
    kri_s = station["station_holdout"]["ordinary_kriging"] if station else {}
    same_m = same["methods"] if same else {}

    readme = f"""# Demo D1 — 站点填图（未观测站点补全）

## 叙事

在 KnowAir **BTHSA** 监测网上，随机遮罩约 20% 站点（seed=42），仅用剩余站点做空间插值，在遮罩站点上评估 MAE/RMSE。  
对照：IDW、普通克里金；网格同遮罩协议下另报 UniST（见同遮罩表）。

## 站点 hold-out（经典基线）

| Method | MAE | RMSE |
|--------|-----|------|
| IDW | {idw_s.get('mae', float('nan')):.2f} | {idw_s.get('rmse', float('nan')):.2f} |
| Ordinary Kriging | {kri_s.get('mae', float('nan')):.2f} | {kri_s.get('rmse', float('nan')):.2f} |

## 同遮罩 occupied-grid（BTHSA Long UniST）

| Method | MAE | RMSE |
|--------|-----|------|
| IDW | {same_m.get('idw', {}).get('mae', float('nan')):.2f} | {same_m.get('idw', {}).get('rmse', float('nan')):.2f} |
| Ordinary Kriging | {same_m.get('ordinary_kriging', {}).get('mae', float('nan')):.2f} | {same_m.get('ordinary_kriging', {}).get('rmse', float('nan')):.2f} |
| UniST | {same_m.get('unist', {}).get('mae', float('nan')):.2f} | {same_m.get('unist', {}).get('rmse', float('nan')):.2f} |

> 纯空间缺测时经典插值通常更强；深度学习主叙事见 **Demo D2 跨场景**。

## 目录

```
demo_d1/
  README.md          ← 本页
  figures/           ← 站点图、误差图、散点
  metrics/           ← JSON / Markdown 指标
```

## 复现

```powershell
.\\.venv-ml\\Scripts\\python.exe examples\\phase2\\eval_spatial_mask.py --region bthsa --skip-unist
.\\.venv-ml\\Scripts\\python.exe examples\\phase2\\plot_spatial_mask.py --region bthsa
.\\.venv-ml\\Scripts\\python.exe examples\\phase2\\eval_same_mask.py --region bthsa
.\\.venv-ml\\Scripts\\python.exe examples\\phase2\\pack_demos.py
```
"""
    (DEMO1 / "README.md").write_text(readme, encoding="utf-8")
    copied.append("README.md")
    return copied


def pack_d2() -> list[str]:
    """D2: grid / cross-scene areal completion (YRD)."""
    figs = DEMO2 / "figures"
    metrics = DEMO2 / "metrics"
    copied: list[str] = []

    for name in (
        "01_metrics_bars_yrd.png",
        "02_station_holdout_map_yrd.png",
        "04_holdout_error_map_yrd.png",
        "03_pred_vs_true_scatter_yrd.png",
    ):
        got = _copy(EVAL / name, figs)
        if got:
            copied.append(f"figures/{got}")

    for name in (
        "cross_scene_bthsa_long_to_yrd.json",
        "cross_scene_bthsa_to_yrd.json",
        "same_mask_yrd_seed42.json",
        "same_mask_yrd_seed42.md",
        "spatial_mask_yrd_seed42.json",
        "spatial_mask_yrd_seed42.md",
        "pretrain_long_bthsa.json",
    ):
        got = _copy(EVAL / name, metrics)
        if got:
            copied.append(f"metrics/{got}")

    cross = _load(EVAL / "cross_scene_bthsa_long_to_yrd.json") or {}
    same = _load(EVAL / "same_mask_yrd_seed42.json") or {}
    m = cross.get("metrics", {})
    same_m = same.get("methods", {})

    readme = f"""# Demo D2 — 面/网格指标补全 + 跨场景

## 叙事（跨场景最小定义）

1. 在区域 **A（BTHSA）** 完成 UniST Stage-1 加长预训练（40 epoch）。  
2. 在区域 **B（YRD）** 仅做 Prompt tuning（5 epoch，few_ratio=0.5）。  
3. 同预算下对比 YRD **从零训练（Scratch）**；并在同一占用格点遮罩下对照 IDW/克里金。

## 跨场景 Prompt vs Scratch（YRDLong，temporal）

| 设定 | Test RMSE | Test MAE |
|------|-----------|----------|
| **Prompt（BTHSA Long → YRD）** | **≈ {m.get('prompt_test_temporal_rmse', float('nan')):.2f}** | **≈ {m.get('prompt_test_temporal_mae', float('nan')):.2f}** |
| Scratch（同预算） | ≈ {m.get('scratch_test_temporal_rmse', float('nan')):.2f} | — |

→ 跨场景 Prompt **优于** 目标域从零训练（本期核心可演示结论）。

## 同遮罩 occupied-grid（YRD）

| Method | MAE | RMSE |
|--------|-----|------|
| IDW | {same_m.get('idw', {}).get('mae', float('nan')):.2f} | {same_m.get('idw', {}).get('rmse', float('nan')):.2f} |
| Ordinary Kriging | {same_m.get('ordinary_kriging', {}).get('mae', float('nan')):.2f} | {same_m.get('ordinary_kriging', {}).get('rmse', float('nan')):.2f} |
| UniST (Prompt) | {same_m.get('unist', {}).get('mae', float('nan')):.2f} | {same_m.get('unist', {}).get('rmse', float('nan')):.2f} |

> 纯空间补全上经典方法仍更强；与上表「跨场景相对 Scratch」叙事互补，不互相替代。

## 目录

```
demo_d2/
  README.md
  figures/
  metrics/
```

## 复现

```powershell
.\\.venv-ml\\Scripts\\python.exe examples\\phase2\\run_unist_prompt_cross.py --from-long --epochs 5 --also-scratch
.\\.venv-ml\\Scripts\\python.exe examples\\phase2\\eval_same_mask.py --region yrd
.\\.venv-ml\\Scripts\\python.exe examples\\phase2\\pack_demos.py
```
"""
    (DEMO2 / "README.md").write_text(readme, encoding="utf-8")
    copied.append("README.md")
    return copied


def main() -> int:
    DEMO1.mkdir(parents=True, exist_ok=True)
    DEMO2.mkdir(parents=True, exist_ok=True)
    c1 = pack_d1()
    c2 = pack_d2()

    brief_src = ROOT / "docs" / "阶段2-演示说明.md"
    brief_dst = ROOT / "output" / "phase2" / "DEMO_BRIEF.md"
    if brief_src.exists():
        shutil.copy2(brief_src, brief_dst)
        brief_note = f"一页总览（拷贝）：`output/phase2/DEMO_BRIEF.md` ← `{brief_src.relative_to(ROOT).as_posix()}`"
    else:
        brief_note = "一页总览：见 `docs/阶段2-演示说明.md`"

    index = f"""# Phase 2 Demos

| Demo | 路径 | 一句话 |
|------|------|--------|
| **总览** | `docs/阶段2-演示说明.md` · `output/phase2/DEMO_BRIEF.md` | Demo + 结论 + 跨场景协议 + 验收命令 |
| D1 站点填图 | `output/phase2/demo_d1/` | BTHSA 站点 hold-out：IDW / 克里金（+ 同遮罩 UniST） |
| D2 面/网格 + 跨场景 | `output/phase2/demo_d2/` | BTHSA→YRD Prompt 优于 Scratch；同遮罩对照经典基线 |

{brief_note}

重新打包：`python examples/phase2/pack_demos.py`

D1 文件数: {len(c1)} · D2 文件数: {len(c2)}
"""
    (ROOT / "output" / "phase2" / "demo_INDEX.md").write_text(index, encoding="utf-8")
    print(f"demo_d1: {len(c1)} items → {DEMO1}")
    print(f"demo_d2: {len(c2)} items → {DEMO2}")
    if brief_src.exists():
        print(f"wrote {brief_dst}")
    print(f"wrote {ROOT / 'output' / 'phase2' / 'demo_INDEX.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
