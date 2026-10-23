"""Assemble Phase-2 Demo folders — UniST-first narrative.

D1 官方论文数据：BikeNYC / TrafficHZ Stage-1 掩码重建
D2 KnowAir 跨城：Prompt vs Scratch（主胜利）；同遮罩 IDW 仅附录

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


def _mask_row(masks: dict, name: str) -> tuple[float, float]:
    m = masks.get(name) or {}
    return float(m.get("mae", float("nan"))), float(m.get("rmse", float("nan")))


def pack_d1() -> list[str]:
    """D1: UniST on official paper datasets."""
    figs = DEMO1 / "figures"
    metrics = DEMO1 / "metrics"
    # Clear stale IDW-centric artefacts from older packs
    if figs.exists():
        for p in figs.glob("*.png"):
            p.unlink()
    if metrics.exists():
        for p in metrics.glob("*"):
            if p.is_file():
                p.unlink()
    copied: list[str] = []

    for name in (
        "00_paper_dataset_inventory.png",
        "01_unist_mask_bars_bikenyc.png",
        "01_unist_mask_bars_traffichz.png",
        "02_unist_recon_heatmap_bikenyc.png",
        "02_unist_recon_heatmap_traffichz.png",
    ):
        got = _copy(EVAL / name, figs)
        if got:
            copied.append(f"figures/{got}")

    for name in (
        "paper_demo_bikenyc.json",
        "paper_demo_bikenyc.md",
        "paper_demo_traffichz.json",
        "paper_demo_traffichz.md",
        "paper_data_link.json",
    ):
        got = _copy(EVAL / name, metrics)
        if got:
            copied.append(f"metrics/{got}")

    bike = _load(EVAL / "paper_demo_bikenyc.json") or {}
    # combined summary may nest runs
    if "runs" in bike:
        bike = next((r for r in bike["runs"] if r.get("dataset") == "BikeNYC"), bike)
    hz = _load(EVAL / "paper_demo_traffichz.json") or {}

    bike_masks = (bike.get("metrics") or {}).get("test_masks") or {}
    hz_masks = (hz.get("metrics") or {}).get("test_masks") or {}

    def _tbl(masks: dict) -> str:
        # UniST result.txt often stores RMSE only under batch mask_strategy_random
        lines = ["| Mask | Test RMSE |", "|------|-----------|"]
        for k in ("random", "temporal", "tube", "block"):
            _mae, rmse = _mask_row(masks, k)
            lines.append(f"| {k} | {rmse:.2f} |")
        return "\n".join(lines)

    readme = f"""# Demo D1 — UniST 通用时空能力（官方论文数据）

## 叙事

接入 UniST 官方发布网格数据（`data_release`），在 **BikeNYC**（主）与 **TrafficHZ**（辅）上做 Stage-1 掩码预训练短训，展示：

1. 我们跑的就是论文里的 **UniST**（不是插值基线 Demo）  
2. 换场景（共享单车 → 城市交通）仍可训、可评  
3. tube / temporal / block / random 多种掩码重建指标可读

> IDW / 克里金不在本 Demo 主舞台；经典对照见 Demo D2 附录。

## BikeNYC（主）

{_tbl(bike_masks) if bike_masks else "_指标待训练产出：`run_unist_paper_demo.py`_"}

## TrafficHZ（辅）

{_tbl(hz_masks) if hz_masks else "_可选：`run_unist_paper_demo.py --also-hz`_"}

## 目录

```
demo_d1/
  README.md
  figures/   ← 官方数据清单、UniST mask 柱状图、重建热力图
  metrics/   ← paper_demo_*.json / md
```

## 复现

```powershell
.\\.venv-ml\\Scripts\\python.exe examples\\phase2\\link_unist_paper_data.py
.\\.venv-ml\\Scripts\\python.exe examples\\phase2\\run_unist_paper_demo.py --also-hz --epochs 3
.\\.venv-ml\\Scripts\\python.exe examples\\phase2\\plot_unist_demo_figures.py
.\\.venv-ml\\Scripts\\python.exe examples\\phase2\\pack_demos.py
```

数据源：`E:\\数据分析模型训练数据\\data_release`（见 `metrics/paper_data_link.json`）。
"""
    (DEMO1 / "README.md").write_text(readme, encoding="utf-8")
    copied.append("README.md")
    return copied


def pack_d2() -> list[str]:
    """D2: UniST cross-scene Prompt vs Scratch (KnowAir)."""
    figs = DEMO2 / "figures"
    metrics = DEMO2 / "metrics"
    if figs.exists():
        for p in figs.glob("*.png"):
            p.unlink()
    if metrics.exists():
        for p in metrics.glob("*"):
            if p.is_file():
                p.unlink()
    copied: list[str] = []

    # Hero first
    for name in (
        "05_cross_scene_prompt_vs_scratch.png",
        "06_unist_cross_scene_summary.png",
        # appendix / optional spatial context (not main narrative)
        "01_metrics_bars_yrd.png",
    ):
        got = _copy(EVAL / name, figs)
        if got:
            copied.append(f"figures/{got}")

    for name in (
        "cross_scene_bthsa_long_to_yrd.json",
        "cross_scene_bthsa_to_yrd.json",
        "pretrain_long_bthsa.json",
        "same_mask_yrd_seed42.json",
        "same_mask_yrd_seed42.md",
    ):
        got = _copy(EVAL / name, metrics)
        if got:
            copied.append(f"metrics/{got}")

    cross = _load(EVAL / "cross_scene_bthsa_long_to_yrd.json") or {}
    same = _load(EVAL / "same_mask_yrd_seed42.json") or {}
    m = cross.get("metrics", {})
    same_m = same.get("methods", {})
    p_rmse = float(m.get("prompt_test_temporal_rmse", float("nan")))
    s_rmse = float(m.get("scratch_test_temporal_rmse", float("nan")))
    p_mae = float(m.get("prompt_test_temporal_mae", float("nan")))

    readme = f"""# Demo D2 — UniST 跨城 Prompt（KnowAir 落地）

## 叙事（主胜利）

同一套 UniST：在区域 **A = BTHSA** 完成 Stage-1 加长预训练 → 在区域 **B = YRD** 仅做 Prompt tuning（5 epoch，few_ratio=0.5），对比同预算 **Scratch**。

| 设定 | Test temporal RMSE | Test temporal MAE |
|------|--------------------|-------------------|
| **Prompt（BTHSA Long → YRD）** | **≈ {p_rmse:.2f}** | **≈ {p_mae:.2f}** |
| Scratch（YRD 从零，同预算） | ≈ {s_rmse:.2f} | — |

→ **跨场景 Prompt 优于从零训练**——本期选型 UniST 的核心可演示结论。

## 附录 · 同遮罩 occupied-grid（另一张考卷）

纯空间填洞上经典插值仍更强；**不要**与上表混成「全面碾压」。

| Method | MAE | RMSE |
|--------|-----|------|
| IDW | {same_m.get('idw', {}).get('mae', float('nan')):.2f} | {same_m.get('idw', {}).get('rmse', float('nan')):.2f} |
| Ordinary Kriging | {same_m.get('ordinary_kriging', {}).get('mae', float('nan')):.2f} | {same_m.get('ordinary_kriging', {}).get('rmse', float('nan')):.2f} |
| UniST (Prompt) | {same_m.get('unist', {}).get('mae', float('nan')):.2f} | {same_m.get('unist', {}).get('rmse', float('nan')):.2f} |

## 目录

```
demo_d2/
  README.md
  figures/   ← Prompt vs Scratch 主图 + 摘要卡；（可选）同遮罩附录图
  metrics/
```

## 复现

```powershell
.\\.venv-ml\\Scripts\\python.exe examples\\phase2\\run_unist_prompt_cross.py --from-long --epochs 5 --also-scratch
.\\.venv-ml\\Scripts\\python.exe examples\\phase2\\plot_unist_demo_figures.py
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

    index = f"""# Phase 2 Demos（UniST 优先）

| Demo | 路径 | 一句话 |
|------|------|--------|
| **总览** | `docs/阶段2-演示说明.md` · `output/phase2/DEMO_BRIEF.md` | UniST 论文场景 + KnowAir 跨城 |
| **演示文稿** | `docs/阶段2-技术演示文稿.md` | UniST 主角叙事 + 插图 |
| **D1 论文能力** | `output/phase2/demo_d1/` | 官方 BikeNYC / TrafficHZ Stage-1 |
| **D2 跨城落地** | `output/phase2/demo_d2/` | Prompt 6.14 vs Scratch 9.92 |

{brief_note}

重新出图并打包：

```powershell
.\\.venv-ml\\Scripts\\python.exe examples\\phase2\\plot_unist_demo_figures.py
.\\.venv-ml\\Scripts\\python.exe examples\\phase2\\pack_demos.py
```

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
