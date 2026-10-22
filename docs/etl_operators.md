# ETL 算子说明

## 算子契约

每个算子接收：

| 字段 | 说明 |
|------|------|
| `input_path` | GeoJSON / GeoParquet / CSV（含 lon/lat） |
| `output_dir` | 结果目录 |
| `params` | JSON 参数字典 |
| `time_column` | 可选，时间列名 |

输出写入 `operator_result.json`，包含 `artifacts` 路径列表。

## 已注册算子（v0.2 / 阶段 1.2）

| name | 说明 | 关键 params |
|------|------|----------------|
| `profile` | 生成 `profile.json` | — |
| `eda_run` | 完整 EDA，`params` / `params.eda_config` 同 CLI config | 见 `eda_default.json` |
| `areal_autocorr` | 全局 Moran / Geary + 局部 LISA | `column`（必填）, `weight_type`, `k`, `include_geary`, `include_lisa`, `p_threshold` |
| `point_pattern` | 点模式 KDE + Ripley K/L/G（可选 CSR） | `kde`, `ripley`, `functions`, `support`, `n_simulations`, `kde_grid_size`, `bandwidth` |
| `variogram` | 经验半变异函数 | `column`（必填）, `n_lags`, `max_dist` |
| `idw` | 反距离权重插值 | `column`（必填）, `power`, `grid_size` |
| `kriging` | 普通克里金（需 `[kriging]` 依赖） | `column`（必填）, `variogram_model`, `grid_size`, `nlags`, `leave_one_out_cv` |
| `dl_infer` | UniST 空间遮罩推理（需 `[ml]` / `.venv-ml` + 权重） | `preset` 或 `checkpoint`+`dataset`；可选 `meta_path`/`mask_ratio`/`max_windows` |

点数据上跑 `areal_autocorr` 时，默认权重为 `knn`；面数据默认 `queen`。

### `dl_infer`（阶段 2）

加载 UniST checkpoint，在测试窗上做 tube/block 遮罩重建，写出：

| 产物 | 说明 |
|------|------|
| `predictions.geojson` | 占用格点中心点：`pm25` 预测、`pm25_true`、`masked` |
| `metrics.json` | MAE / RMSE（遮罩格点，原 PM2.5 单位） |
| `diagnostics.json` / `operator_result.json` | 设备、权重路径、窗数等 |

**params 要点**

| 字段 | 说明 |
|------|------|
| `preset` | `bthsa_long` 或 `yrd_prompt`（自动填 checkpoint / dataset / meta） |
| `checkpoint` + `dataset` | 不用 preset 时必填 |
| `mask_ratio` / `mask_strategy` / `seed` | 默认 `0.5` / `tube` / `42` |
| `max_windows` / `batch_size` | 默认 `8` / `4`（冒烟可再降） |

`input_path` 可为 `*.meta.json` / `*.npz`（补全 meta/tensor）；也可用任意占位路径 + `preset`（preset 已带 meta）。训练仍在 `examples/phase2/` 脚本侧，不进算子。

## CLI 调用示例

```bash
# 概况 / EDA
steda operators execute profile -i examples/data/sample_points.geojson -o output/op_profile --time-column timestamp
steda operators execute eda_run -i examples/data/sample_points.geojson -o output/op_eda -p examples/config/eda_default.json --time-column timestamp

# 面数据 LISA
steda operators execute areal_autocorr -i examples/data/sample_polygons.geojson -o output/op_lisa -p examples/config/areal_gdp.json

# 点模式
steda operators execute point_pattern -i examples/data/sample_events.geojson -o output/op_pp -p examples/config/point_pattern.json --time-column timestamp

# 插值
steda operators execute variogram -i examples/data/sample_events.geojson -o output/op_var -p examples/config/idw_pm25.json --time-column timestamp
steda operators execute idw -i examples/data/sample_events.geojson -o output/op_idw -p examples/config/idw_pm25.json --time-column timestamp
steda operators execute kriging -i examples/data/sample_events.geojson -o output/op_krig -p examples/config/kriging_pm25.json --time-column timestamp

# 阶段 2：UniST 推理（请用 .venv-ml）
.\.venv-ml\Scripts\python.exe -m eda_toolkit.cli.main operators execute dl_infer ^
  -i output\phase2\tensors\knowair_bthsa_16x16_2022-01-01_2023-12-31.meta.json ^
  -o output\phase2\op_dl_infer_bthsa ^
  -p examples\config\dl_infer_bthsa_long.json
```

端到端两条流水线样例：

```bash
python examples/run_phase12_pipelines.py
```

- **Pipeline A**：区县面指标 → Profile → Moran/LISA  
- **Pipeline B**：点事件 → Profile → 点模式 → 变异函数 → IDW / 克里金  

## 治理平台对接

将上述命令封装为 DAG 节点；失败时根据退出码重试。`output_dir` 挂载到平台任务产物目录即可回写元数据。
