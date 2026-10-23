# Phase 2 — 环境与复现说明（UniST）

对应计划：[阶段2-时空基础模型与预测计划.md](../../docs/阶段2-时空基础模型与预测计划.md)  
演示一页：[阶段2-演示说明.md](../../docs/阶段2-演示说明.md)（`pack_demos.py` 同步到 `output/phase2/DEMO_BRIEF.md`）

## 机器基线（本机已验证 · 2026-10-21）

| 项 | 值 |
|----|-----|
| GPU | NVIDIA GeForce RTX 3060 12GB |
| Driver | 560.94（报告 CUDA 12.6） |
| Python | 3.11.9 |
| 虚拟环境 | 仓库根目录 `.venv-ml/`（与阶段 1 的 `.venv` 分离） |
| PyTorch | `2.6.0+cu124` |
| torchvision | `0.21.0+cu124` |
| timm | `0.9.2`（对齐 UniST 官方 pin） |

自检命令（应全部通过）：

```powershell
.\.venv-ml\Scripts\python.exe examples\phase2\verify_env.py
```

期望：`cuda_available: True`，末尾 `OK: phase2 ML environment looks ready.`

阶段 1 EDA / 经典算法继续用 `.venv`；训练 UniST 与 KnowAir 请用 `.venv-ml`。

## 安装步骤（Windows / PowerShell）

在仓库根目录执行。国内建议非 torch 包走清华源；**torch / torchvision 必须走 PyTorch 官方 CUDA 索引**。

```powershell
# 0) 若上次安装中断，先清锁
Get-Process -Name uv -ErrorAction SilentlyContinue | Stop-Process -Force
if (Test-Path .venv-ml\.lock) { Remove-Item -Force .venv-ml\.lock }

# 1) 创建环境
uv venv .venv-ml --python 3.11

# 2) CUDA 版 PyTorch（约 2.4GB）
uv pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --python .venv-ml\Scripts\python.exe

# 3) 其余 ML 依赖（不要在此步装 torch）
uv pip install -r examples\phase2\requirements-ml.txt --python .venv-ml\Scripts\python.exe --default-index https://pypi.tuna.tsinghua.edu.cn/simple

# 4) 本仓库（含 IDW/克里金等基线）
#    注意：该步可能从 PyPI 拉来 CPU 版 torch，第 5 步必须重装 CUDA 轮
uv pip install -e ".[spatial,kriging,ml]" --python .venv-ml\Scripts\python.exe --default-index https://pypi.tuna.tsinghua.edu.cn/simple

# 5) 强制恢复 CUDA torch + 对齐 torchvision / timm
uv pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --python .venv-ml\Scripts\python.exe
uv pip install timm==0.9.2 --no-deps --python .venv-ml\Scripts\python.exe --default-index https://pypi.tuna.tsinghua.edu.cn/simple

# 6) 自检
.\.venv-ml\Scripts\python.exe examples\phase2\verify_env.py
```

### 激活环境（可选）

```powershell
.\.venv-ml\Scripts\Activate.ps1
```

## UniST 仓库（已 clone）

```text
third_party/UniST @ 6ee69db00d8f67e550f9bc3c06f004f9f7885703
```

复现 clone：

```powershell
mkdir third_party -ErrorAction SilentlyContinue
git clone https://github.com/tsinghua-fib-lab/UniST.git third_party\UniST
cd third_party\UniST
git rev-parse HEAD
```

官方曾 pin `torch==2.0.0+cu117`；本期用 `2.6.0+cu124`。目录在 `.gitignore` 的 `third_party/` 下，不进 git。

## 数据与栅格协议

```powershell
# 下载 / 续传 KnowAir-V2（YRD 约 357MB，可 curl -C - 续传）
.\.venv-ml\Scripts\python.exe examples\phase2\_fetch_knowair_v2.py

# 摸底
.\.venv-ml\Scripts\python.exe examples\phase2\inspect_knowair.py

# 站点 → T×C×H×W（协议见 configs/grid_protocol_v1.json）
.\.venv-ml\Scripts\python.exe examples\phase2\build_knowair_grid.py --region bthsa --grid 16 16 --time-start 2023-01-01 --time-end 2023-01-31 --stride 3
```

产物：`output/phase2/tensors/knowair_*.npz` + `*.meta.json`（已 gitignore `output/`）。

KnowAir 原数据是 `(time, station)` 面板；UniST 需要网格。栅格协议 v1：经纬度等分格、格内均值、空格填 0 并保存 `occupied` mask。

## UniST 冒烟（本机已跑通 · 2026-10-22）

```powershell
# 1) 栅格化（若尚未生成）
.\.venv-ml\Scripts\python.exe examples\phase2\build_knowair_grid.py --region bthsa --grid 16 16 --time-start 2023-01-01 --time-end 2023-01-31 --stride 3

# 2) 导出 UniST JSON + Windows 补丁
.\.venv-ml\Scripts\python.exe examples\phase2\export_unist_json.py --npz output\phase2\tensors\knowair_bthsa_16x16_2023-01-01_2023-01-31.npz --meta output\phase2\tensors\knowair_bthsa_16x16_2023-01-01_2023-01-31.meta.json --max-samples 80
.\.venv-ml\Scripts\python.exe examples\phase2\patch_unist_windows.py

# 3) Stage-1 预训练冒烟（2 epoch）
.\.venv-ml\Scripts\python.exe examples\phase2\run_unist_smoke.py
```

成功标志：日志出现 `model initialized!`，`training loss` 下降，权重落在 `third_party/UniST/src/experiments/.../model_save/`。

数据集名：`TrafficKnowAir`（走官方 Traffic* 时间戳分支）。

跨场景数据（YRD，已校验）：

```powershell
.\.venv-ml\Scripts\python.exe examples\phase2\build_knowair_grid.py --region yrd --grid 16 16 --time-start 2023-01-01 --time-end 2023-01-31 --stride 3
.\.venv-ml\Scripts\python.exe examples\phase2\export_unist_json.py --npz output\phase2\tensors\knowair_yrd_16x16_2023-01-01_2023-01-31.npz --meta output\phase2\tensors\knowair_yrd_16x16_2023-01-01_2023-01-31.meta.json --max-samples 80 --name TrafficKnowAirYRD
```

| 区域 | 站点 | UniST JSON |
|------|------|------------|
| BTHSA | 228 | `TrafficKnowAir_short.json` |
| YRD | 127 | `TrafficKnowAirYRD_short.json` |

## 空间遮罩 vs IDW / 克里金

```powershell
.\.venv-ml\Scripts\python.exe examples\phase2\eval_spatial_mask.py --region bthsa
# 只要经典基线：加 --skip-unist
```

产物：`output/phase2/eval/spatial_mask_bthsa_seed42.json` 与 `.md`。

可视化：

```powershell
.\.venv-ml\Scripts\python.exe examples\phase2\plot_spatial_mask.py --region bthsa
.\.venv-ml\Scripts\python.exe examples\phase2\plot_spatial_mask.py --region yrd
explorer.exe output\phase2\eval
```

### 同遮罩对比（推荐 · 公平协议）

同一批 occupied holdout cells 上比 UniST / IDW / 克里金：

```powershell
.\.venv-ml\Scripts\python.exe examples\phase2\eval_same_mask.py --region yrd
.\.venv-ml\Scripts\python.exe examples\phase2\eval_same_mask.py --region bthsa
```

| 区域 | 最优 | UniST RMSE |
|------|------|------------|
| YRD（Prompt） | IDW ≈ 5.65 | ≈ 20.8 |
| BTHSA（Long） | IDW ≈ 8.37 | ≈ 23.5 |

产物：`output/phase2/eval/same_mask_{yrd,bthsa}_seed42.json` / `.md`  
说明：纯空间补全上经典插值更强属预期；UniST 主叙事见下方跨场景 Prompt vs Scratch。

## 跨场景 Prompt（BTHSA → YRD）

```powershell
# 冒烟权重（短窗）
.\.venv-ml\Scripts\python.exe examples\phase2\run_unist_prompt_cross.py --epochs 3 --also-scratch

# 长训权重 TrafficKnowAirLong → TrafficKnowAirYRDLong（推荐）
.\.venv-ml\Scripts\python.exe examples\phase2\run_unist_prompt_cross.py --from-long --epochs 5 --also-scratch
```

| 产物 | 说明 |
|------|------|
| `output/phase2/eval/cross_scene_bthsa_to_yrd.json` | 冒烟跨场景 |
| `output/phase2/eval/cross_scene_bthsa_long_to_yrd.json` | 长训权重跨场景 |

进度总览：`output/phase2/PROGRESS.md`

说明：站点 hold-out 是 D1 主叙事；跨场景以「Prompt vs 从零」同预算对比为主叙事。长训 Prompt test temporal RMSE ≈ 6.1（优于同预算 Scratch ≈ 9.9）。

## 查看训练曲线（TensorBoard 回放）

已完成实验的标量在 `third_party/UniST/src/logs/`（不是现场重训）：

```powershell
# 全部 run（默认 http://localhost:6006）
.\.venv-ml\Scripts\python.exe examples\phase2\launch_tensorboard.py

# 只看长训 / Prompt / Scratch
.\.venv-ml\Scripts\python.exe examples\phase2\launch_tensorboard.py --preset long
.\.venv-ml\Scripts\python.exe examples\phase2\launch_tensorboard.py --preset prompt
.\.venv-ml\Scripts\python.exe examples\phase2\launch_tensorboard.py --preset scratch
```

浏览器里优先看 `Training/Loss_step`、`Test_RMSE/*`、`Test_MAE/*`。Preset：`long` = BTHSA 40 epoch；`prompt` / `scratch` = YRD 跨场景对照。

## Demo 打包（D1 / D2）

```powershell
.\.venv-ml\Scripts\python.exe examples\phase2\pack_demos.py
explorer.exe output\phase2\demo_d1
explorer.exe output\phase2\demo_d2
```

| 目录 | 说明 |
|------|------|
| `output/phase2/demo_d1/` | 站点填图：图 + 指标 + 一页 README |
| `output/phase2/demo_d2/` | 跨场景网格补全：Prompt vs Scratch + 同遮罩 |
| `output/phase2/demo_INDEX.md` | 总览 |

## ETL `dl_infer` 算子

契约见 [docs/etl_operators.md](../../docs/etl_operators.md)。训练仍用本目录脚本；推理可挂治理平台 DAG：

```powershell
# 列表中应出现 dl_infer
.\.venv-ml\Scripts\python.exe -m eda_toolkit.cli.main operators list

# BTHSA 长训权重推理（写出 predictions.geojson + metrics.json）
.\.venv-ml\Scripts\python.exe -m eda_toolkit.cli.main operators execute dl_infer `
  -i output\phase2\tensors\knowair_bthsa_16x16_2022-01-01_2023-12-31.meta.json `
  -o output\phase2\op_dl_infer_bthsa `
  -p examples\config\dl_infer_bthsa_long.json

# 跨场景 YRD Prompt 权重
.\.venv-ml\Scripts\python.exe -m eda_toolkit.cli.main operators execute dl_infer `
  -i output\phase2\tensors\knowair_yrd_16x16_2022-01-01_2023-12-31.meta.json `
  -o output\phase2\op_dl_infer_yrd `
  -p examples\config\dl_infer_yrd_prompt.json
```

| preset | 权重 |
|--------|------|
| `bthsa_long` | TrafficKnowAirLong Stage-1 |
| `yrd_prompt` | BTHSA Long → YRDLong Prompt |

## 常见问题

1. **`cuda_available: False` / 版本变成 `+cpu`**  
   几乎都是第 4 步 editable install 覆盖了 CUDA 轮。重跑第 5 步后自检。

2. **`timm` 报 `torchvision::nms does not exist`**  
   torchvision 与 torch 版本不匹配。按第 5 步同时装 `torch` + `torchvision` 的 `cu124` 轮，再 `timm==0.9.2 --no-deps`。

3. **`uv` 卡在 Waiting to acquire exclusive lock**  
   结束残留 `uv` 进程并删除 `.venv-ml\.lock`（见第 0 步）。

4. **与 `.venv` 搞混**  
   训练命令写死 `.venv-ml\Scripts\python.exe`。
