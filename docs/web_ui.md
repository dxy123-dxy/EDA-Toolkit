# Web 可视化界面

基于 [Streamlit](https://streamlit.io/)，在浏览器中完成数据上传、EDA 分析与结果查看。

## 安装

```bash
pip install -e ".[ui]"
```

## 启动

```bash
steda ui
```

浏览器访问：**http://localhost:8501**

指定端口：

```bash
steda ui --port 8502
```

或直接：

```bash
streamlit run src/eda_toolkit/ui/app.py
```

## 界面说明

| 标签页 | 功能 |
|--------|------|
| 数据预览 | 指标卡片、地图点分布、属性表 |
| 概况 Profile | 缺失率、CRS、时间范围、字段统计 |
| 单变量 | 空间/时间/时空统计 |
| 多变量 | 相关矩阵、PCA |
| 完整 EDA | 一键跑全流程并展示图表 |
| ETL 算子 | 查看并试跑 `profile` / `eda_run` |

侧边栏可上传 **GeoJSON、CSV** 等，或一键加载内置样例数据。
