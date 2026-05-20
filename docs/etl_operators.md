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

## 已注册算子（v0.1）

| name | 说明 |
|------|------|
| `profile` | 生成 `profile.json` |
| `eda_run` | 完整 EDA，`params.eda_config` 同 CLI config |

## CLI 调用示例

```bash
steda operators execute profile -i examples/data/sample_points.geojson -o output/op_profile --time-column timestamp

steda operators execute eda_run -i examples/data/sample_points.geojson -o output/op_eda -p examples/config/eda_default.json --time-column timestamp
```

## 治理平台对接

将上述命令封装为 DAG 节点；失败时根据退出码重试。`output_dir` 挂载到平台任务产物目录即可回写元数据。
