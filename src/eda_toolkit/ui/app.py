"""
EDA-Toolkit 可视化 Web 界面。

启动: steda ui  或  streamlit run src/eda_toolkit/ui/app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from eda_toolkit.eda.profile import build_profile
from eda_toolkit.eda.runner import run_eda
from eda_toolkit.eda.univariate import analyze_univariate
from eda_toolkit.eda.multivariate import analyze_multivariate
from eda_toolkit.etl.registry import get_operator, list_operators
from eda_toolkit.etl.operator import OperatorContext
from eda_toolkit.io.loaders import load_dataset, load_dataset_from_upload

ROOT = Path(__file__).resolve().parents[3]
SAMPLE_PATH = ROOT / "examples" / "data" / "sample_points.geojson"


def _init_session() -> None:
    defaults = {
        "dataset": None,
        "file_name": None,
        "profile": None,
        "eda_result": None,
        "last_error": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _load_uploaded(
    uploaded: Any,
    time_column: str | None,
    lon_col: str,
    lat_col: str,
) -> None:
    try:
        ds = load_dataset_from_upload(
            uploaded.name,
            uploaded.getvalue(),
            time_column=time_column or None,
            lon_column=lon_col,
            lat_column=lat_col,
        )
        st.session_state.dataset = ds
        st.session_state.file_name = uploaded.name
        st.session_state.profile = None
        st.session_state.eda_result = None
        st.session_state.last_error = None
    except Exception as exc:
        st.session_state.last_error = str(exc)
        st.error(f"加载失败: {exc}")


def _load_sample(time_column: str) -> None:
    try:
        ds = load_dataset(SAMPLE_PATH, time_column=time_column or "timestamp")
        st.session_state.dataset = ds
        st.session_state.file_name = "sample_points.geojson"
        st.session_state.profile = None
        st.session_state.eda_result = None
        st.session_state.last_error = None
    except Exception as exc:
        st.session_state.last_error = str(exc)
        st.error(f"加载样例失败: {exc}")


def _gdf_for_map(gdf: pd.DataFrame) -> pd.DataFrame | None:
    try:
        df = gdf.copy()
        if "latitude" in df.columns and "longitude" in df.columns:
            return df.dropna(subset=["latitude", "longitude"])
        df = df.copy()
        df["latitude"] = df.geometry.y
        df["longitude"] = df.geometry.x
        return df.dropna(subset=["latitude", "longitude"])
    except Exception:
        return None


def _render_sidebar() -> dict[str, Any]:
    st.sidebar.header("数据与配置")
    source = st.sidebar.radio(
        "数据来源",
        ["上传文件", "内置样例"],
        horizontal=True,
    )

    time_column = st.sidebar.text_input(
        "时间列名（可留空自动识别）",
        value="timestamp",
        help="如 timestamp、date、time 等",
    )
    lon_col = st.sidebar.text_input("CSV 经度列", value="longitude")
    lat_col = st.sidebar.text_input("CSV 纬度列", value="latitude")

    if source == "上传文件":
        uploaded = st.sidebar.file_uploader(
            "选择数据文件",
            type=["geojson", "json", "csv", "tsv", "shp", "gpkg", "parquet", "geoparquet"],
        )
        if uploaded is not None:
            if st.sidebar.button("加载数据", type="primary", use_container_width=True):
                _load_uploaded(uploaded, time_column.strip() or None, lon_col, lat_col)
    else:
        if st.sidebar.button("加载内置样例", type="primary", use_container_width=True):
            _load_sample(time_column.strip() or "timestamp")

    st.sidebar.divider()
    st.sidebar.subheader("EDA 选项")
    cfg = {
        "profile": st.sidebar.checkbox("数据概况 Profile", value=True),
        "univariate": st.sidebar.checkbox("单变量分析", value=True),
        "multivariate": st.sidebar.checkbox("多变量分析", value=True),
        "plots": st.sidebar.checkbox("生成图表", value=True),
        "pca_components": int(st.sidebar.slider("PCA 主成分数", 2, 10, 3)),
    }
    return cfg


def _tab_data_preview(ds: Any) -> None:
    st.subheader("数据预览")
    gdf = ds.gdf
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("行数", len(gdf))
    c2.metric("字段数", len(gdf.columns))
    c3.metric("几何类型", ds.geometry_type)
    c4.metric("含时间维", "是" if ds.has_time else "否")

    st.caption(f"CRS: {gdf.crs} | 时间列: {ds.time_column or '（未识别）'}")

    map_df = _gdf_for_map(gdf)
    if map_df is not None and len(map_df) > 0:
        st.map(map_df, latitude="latitude", longitude="longitude", size=20)

    with st.expander("属性表", expanded=True):
        preview = gdf.drop(columns="geometry", errors="ignore")
        st.dataframe(preview, use_container_width=True, height=320)


def _tab_profile(ds: Any) -> None:
    st.subheader("数据概况 Profile")
    if st.button("生成 / 刷新 Profile", type="primary"):
        with st.spinner("分析中..."):
            st.session_state.profile = build_profile(ds)

    profile = st.session_state.profile
    if profile is None:
        st.info("点击上方按钮生成数据概况。")
        return

    summary = profile.get("summary", {})
    cols = st.columns(3)
    cols[0].json({"CRS": summary.get("crs"), "bounds": summary.get("bounds")})
    cols[1].json({"时间范围": summary.get("time_range")})
    cols[2].json({"数值字段": summary.get("numeric_columns")})

    quality = profile.get("quality", {})
    if quality:
        st.warning(
            f"无效几何: {quality.get('invalid_geometry_count', 0)} | "
            f"CRS 已定义: {quality.get('crs_defined')}"
        )

    col_df = pd.DataFrame(
        [
            {
                "字段": name,
                "类型": info.get("dtype"),
                "缺失率": f"{info.get('missing_rate', 0) * 100:.1f}%",
                "唯一值数": info.get("n_unique"),
            }
            for name, info in profile.get("columns", {}).items()
        ]
    )
    st.dataframe(col_df, use_container_width=True, hide_index=True)

    with st.expander("完整 profile.json"):
        st.json(profile)


def _tab_univariate(ds: Any) -> None:
    st.subheader("单变量分析")
    numeric = ds.numeric_columns
    if not numeric:
        st.warning("无数值字段可分析。")
        return

    selected = st.multiselect("选择字段", numeric, default=numeric[: min(3, len(numeric))])
    if st.button("运行单变量分析", type="primary"):
        with st.spinner("计算中..."):
            result = analyze_univariate(ds, columns=selected or None)
        st.session_state.univariate = result

    result = st.session_state.get("univariate")
    if not result:
        st.info("选择字段后点击运行。")
        return

    for col, stats in result.get("columns", {}).items():
        with st.expander(f"字段: {col}", expanded=len(result.get("columns", {})) <= 2):
            if "spatial" in stats:
                st.markdown("**空间统计**")
                st.json(stats["spatial"])
            if "temporal" in stats:
                st.markdown("**时间统计**")
                st.json(stats["temporal"])
            if "spatiotemporal" in stats:
                st.markdown("**时空联合**")
                st.json(stats["spatiotemporal"])


def _tab_multivariate(ds: Any) -> None:
    st.subheader("多变量分析")
    numeric = ds.numeric_columns
    if len(numeric) < 2:
        st.warning("至少需要 2 个数值字段。")
        return

    pca_n = st.slider("PCA 主成分数", 2, min(10, len(numeric)), 3)
    if st.button("运行多变量分析", type="primary"):
        with st.spinner("计算中..."):
            result = analyze_multivariate(ds, pca_components=pca_n)
        st.session_state.multivariate = result

    result = st.session_state.get("multivariate")
    if not result:
        st.info("点击运行多变量分析。")
        return

    if "pearson" in result:
        st.markdown("**Pearson 相关矩阵**")
        st.dataframe(pd.DataFrame(result["pearson"]), use_container_width=True)
    if "pca" in result:
        st.markdown("**PCA 方差解释比**")
        st.bar_chart(
            pd.Series(
                result["pca"]["explained_variance_ratio"],
                index=[f"PC{i+1}" for i in range(len(result["pca"]["explained_variance_ratio"]))],
            )
        )
    with st.expander("完整结果 JSON"):
        st.json(result)


def _tab_plots_and_run(ds: Any, cfg: dict[str, Any]) -> None:
    st.subheader("一键完整 EDA")
    st.caption("按侧边栏选项运行 Profile + 单/多变量 + 图表，结果在下方展示。")

    if st.button("运行完整 EDA", type="primary", use_container_width=True):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with st.spinner("正在运行 EDA 流水线..."):
                result = run_eda(ds, config=cfg, output_dir=tmp)
            st.session_state.eda_result = result
            st.session_state.eda_tmp = tmp
            # persist figure bytes for display after temp dir is gone
            figures: dict[str, bytes] = {}
            for name, path in result.get("figures", {}).items():
                p = Path(path)
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"} and p.exists():
                    figures[name] = p.read_bytes()
            st.session_state.eda_figures = figures

    result = st.session_state.get("eda_result")
    if not result:
        st.info("点击「运行完整 EDA」开始分析。")
        return

    st.success("分析完成")

    figures = st.session_state.get("eda_figures", {})
    if figures:
        st.markdown("**图表**")
        cols = st.columns(min(len(figures), 2))
        for i, (name, data) in enumerate(figures.items()):
            cols[i % len(cols)].image(data, caption=name, use_container_width=True)

    tabs = st.tabs(["Profile", "单变量", "多变量"])
    with tabs[0]:
        if "profile" in result:
            st.json(result["profile"].get("summary", {}))
    with tabs[1]:
        if "univariate" in result.get("tables", {}):
            st.json(result["tables"]["univariate"])
    with tabs[2]:
        if "multivariate" in result.get("tables", {}):
            st.json(result["tables"]["multivariate"])


def _tab_operators() -> None:
    st.subheader("ETL 算子")
    st.caption("与数据治理流水线对接的算子列表（只读展示）。")

    ops = list_operators()
    st.dataframe(pd.DataFrame(ops), use_container_width=True, hide_index=True)

    ds = st.session_state.dataset
    if ds is None:
        st.info("请先在侧边栏加载数据。")
        return

    op_name = st.selectbox("选择算子", [o["name"] for o in ops])
    if st.button(f"执行算子: {op_name}", type="secondary"):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "input.geojson"
            ds.gdf.to_file(tmp_path, driver="GeoJSON")
            ctx = OperatorContext(
                input_path=tmp_path,
                output_dir=Path(tmp) / "out",
                params={"eda_config": {
                    "profile": True,
                    "univariate": True,
                    "multivariate": True,
                    "plots": False,
                }},
                time_column=ds.time_column,
            )
            try:
                res = get_operator(op_name)().run(ctx)
                st.success(res.message)
                st.json(res.diagnostics)
                if res.artifacts:
                    st.markdown("**产物**")
                    st.json(res.artifacts)
            except Exception as exc:
                st.error(str(exc))


def main() -> None:
    st.set_page_config(
        page_title="EDA-Toolkit",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session()

    st.title("EDA-Toolkit 时空数据探索分析")
    st.markdown(
        "上传 **GeoJSON / CSV** 等时空数据，进行概况检查、单变量/多变量 EDA 与可视化。"
    )

    cfg = _render_sidebar()
    ds = st.session_state.dataset

    if st.session_state.last_error and ds is None:
        st.error(st.session_state.last_error)

    if ds is None:
        st.info("👈 请在左侧 **上传文件** 或 **加载内置样例** 开始。")
        with st.expander("支持格式"):
            st.markdown(
                """
                - **矢量**: GeoJSON, Shapefile, GeoPackage, GeoParquet
                - **表格**: CSV / TSV（需含经度、纬度列）
                - **时间列**: 自动识别或手动填写列名
                """
            )
        return

    st.success(f"已加载: **{st.session_state.file_name}**（{len(ds.gdf)} 行）")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["数据预览", "概况 Profile", "单变量", "多变量", "完整 EDA", "ETL 算子"]
    )
    with tab1:
        _tab_data_preview(ds)
    with tab2:
        _tab_profile(ds)
    with tab3:
        _tab_univariate(ds)
    with tab4:
        _tab_multivariate(ds)
    with tab5:
        _tab_plots_and_run(ds, cfg)
    with tab6:
        _tab_operators()


if __name__ == "__main__":
    main()
