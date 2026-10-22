# EDA-Toolkit

时空数据探索性分析（EDA）与经典空间统计算法工具包，对应「时空数据分析挖掘」模块阶段 1。

- 远程仓库：[dxy123-dxy/EDA-Toolkit](https://github.com/dxy123-dxy/EDA-Toolkit)
- 技术路线：见 [docs/技术路线-时空数据分析挖掘.md](docs/技术路线-时空数据分析挖掘.md)
- 使用流程与分析说明：见 [docs/使用流程与分析说明.md](docs/使用流程与分析说明.md)
- **阶段 1.2 交付说明**：见 [docs/阶段1.2-经典算法交付说明.md](docs/阶段1.2-经典算法交付说明.md)
- **阶段 2 计划（基础模型 + 未观测预测）**：见 [docs/阶段2-时空基础模型与预测计划.md](docs/阶段2-时空基础模型与预测计划.md)
- **阶段 2 演示说明（Demo + 结论）**：见 [docs/阶段2-演示说明.md](docs/阶段2-演示说明.md)
- ETL 算子：见 [docs/etl_operators.md](docs/etl_operators.md)

## 功能（当前 v0.2）

| 模块 | 状态 |
|------|------|
| 数据接入与 `SpatioTemporalDataset` | ✅ |
| Profile（质量检查 + 元数据 JSON） | ✅ |
| 单变量 / 多变量 EDA | ✅ 基础版 |
| 可视化（单变量/多变量丰富图表） | ✅ |
| ETL 算子契约与注册表 | ✅ |
| **Web 可视化界面** | ✅ Streamlit |
| 面数据 Moran / Geary / LISA | ✅ |
| 点模式 KDE + Ripley K/L/G | ✅ |
| IDW / 普通克里金 / 经验变异函数 | ✅ |

## 安装

```bash
cd 探索性数据分析
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev,ui,spatial,kriging]"
```

建议使用 **Python 3.10–3.12**（部分地理库对 3.14 可能尚未完全支持）。

## 快速开始

```bash
# 数据概况
steda profile --input examples/data/sample_points.geojson --output output/profile

# 完整 EDA（配置见 examples/config/eda_default.json）
steda run --input examples/data/sample_points.geojson --config examples/config/eda_default.json --output output/eda

# 列出已注册 ETL 算子
steda operators list

# 阶段 1.2：面 LISA / 点模式 / 插值
steda operators execute areal_autocorr -i examples/data/sample_polygons.geojson -o output/lisa -p examples/config/areal_gdp.json
steda operators execute point_pattern -i examples/data/sample_events.geojson -o output/pp -p examples/config/point_pattern.json -t timestamp
steda operators execute idw -i examples/data/sample_events.geojson -o output/idw -p examples/config/idw_pm25.json -t timestamp
steda operators execute kriging -i examples/data/sample_events.geojson -o output/krig -p examples/config/kriging_pm25.json -t timestamp

# 两条端到端流水线样例
python examples/run_phase12_pipelines.py

# 启动可视化 Web 界面（浏览器打开 http://localhost:8501）
steda ui
```

### Web 界面

安装 UI 依赖后运行 `steda ui`，在浏览器中：

1. 上传 GeoJSON / CSV，或加载内置样例  
2. 配置时间列、EDA 选项  
3. 在标签页查看地图预览、Profile、单/多变量结果与图表  

也可直接：`streamlit run src/eda_toolkit/ui/app.py`

## 项目结构

```text
src/eda_toolkit/
  io/           # 读写与 SpatioTemporalDataset
  eda/          # profile, univariate, multivariate, runner
  viz/          # 绘图
  algorithms/   # 点模式、面数据、插值（逐步实现）
  etl/          # 算子注册与 JSON Schema
  cli/          # Typer 命令行
```

## 开发

```bash
pytest tests/ -v
```

## 许可证

MIT
