"""Visualization helpers (decoupled from computation)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda_toolkit.io.dataset import SpatioTemporalDataset

# Consistent style for reports
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _save_fig(fig: plt.Figure, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def figure_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Legacy / shared
# ---------------------------------------------------------------------------


def plot_timeseries(
    dataset: SpatioTemporalDataset,
    value_col: str,
    output_path: str | Path | None = None,
    *,
    title: str | None = None,
    show_band: bool = True,
) -> plt.Figure:
    if not dataset.has_time or not dataset.time_column:
        raise ValueError("Dataset has no time column for timeseries plot")

    gdf = dataset.gdf
    tcol = dataset.time_column
    grouped = gdf.groupby(tcol)[value_col]
    mean = grouped.mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(mean))
    ax.plot(x, mean.values, marker="o", linewidth=2, label="空间均值", color="#2563eb")
    if show_band and len(grouped) > 1:
        gmin = grouped.min()
        gmax = grouped.max()
        ax.fill_between(x, gmin.values, gmax.values, alpha=0.2, color="#2563eb", label="空间 min–max")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in mean.index], rotation=25, ha="right")
    ax.set_xlabel(tcol)
    ax.set_ylabel(value_col)
    ax.set_title(title or f"{value_col} 时间演化（空间聚合）")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig


def plot_choropleth(
    dataset: SpatioTemporalDataset,
    column: str,
    output_path: str | Path | None = None,
    *,
    scheme: str = "Quantiles",
    k: int = 5,
) -> plt.Figure:
    gdf = dataset.gdf
    if column not in gdf.columns:
        raise ValueError(f"Column '{column}' not in dataset")

    plot_gdf = gdf
    if dataset.has_time and dataset.time_column:
        latest = gdf[dataset.time_column].max()
        plot_gdf = gdf[gdf[dataset.time_column] == latest]

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_gdf.plot(
        column=column,
        ax=ax,
        legend=True,
        scheme=scheme,
        k=k,
        cmap="YlOrRd",
        edgecolor="gray",
        linewidth=0.3,
    )
    ax.set_title(f"{column} 空间分布")
    ax.set_axis_off()
    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Univariate charts
# ---------------------------------------------------------------------------


def plot_distribution(
    dataset: SpatioTemporalDataset,
    value_col: str,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Histogram + KDE for spatial marginal distribution."""
    series = dataset.gdf[value_col].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(series, bins=min(30, max(10, len(series) // 3)), density=True, alpha=0.65, color="#3b82f6", edgecolor="white")
    if len(series) > 5:
        try:
            series.plot.kde(ax=ax, color="#1d4ed8", linewidth=2)
        except Exception:
            pass
    ax.set_xlabel(value_col)
    ax.set_ylabel("密度")
    ax.set_title(f"{value_col} 分布（直方图 + 核密度）")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig


def plot_boxplot(
    dataset: SpatioTemporalDataset,
    value_col: str,
    output_path: str | Path | None = None,
    *,
    by_time: bool = True,
) -> plt.Figure:
    """Boxplot: overall or grouped by time step."""
    gdf = dataset.gdf
    fig, ax = plt.subplots(figsize=(8, 4))
    if by_time and dataset.has_time and dataset.time_column:
        tcol = dataset.time_column
        groups = [gdf.loc[gdf[tcol] == t, value_col].dropna().values for t in gdf[tcol].dropna().unique()]
        labels = [str(t) for t in gdf[tcol].dropna().unique()]
        ax.boxplot(groups, tick_labels=labels, patch_artist=True)
        ax.set_xlabel(tcol)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")
        ax.set_title(f"{value_col} 按时间分箱")
    else:
        ax.boxplot(gdf[value_col].dropna(), patch_artist=True)
        ax.set_title(f"{value_col} 箱线图")
    ax.set_ylabel(value_col)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig


def plot_spatial_values(
    dataset: SpatioTemporalDataset,
    value_col: str,
    output_path: str | Path | None = None,
) -> plt.Figure | None:
    """Point/scatter map colored by attribute; polygon uses choropleth."""
    gtype = dataset.geometry_type
    if gtype in ("Polygon", "MultiPolygon"):
        return plot_choropleth(dataset, value_col, output_path)

    gdf = dataset.gdf.dropna(subset=[value_col])
    if gdf.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 6))
    gdf.plot(
        column=value_col,
        ax=ax,
        legend=True,
        cmap="viridis",
        markersize=80,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_title(f"{value_col} 空间分布（点）")
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig


def plot_temporal_decomposition(
    dataset: SpatioTemporalDataset,
    value_col: str,
    output_path: str | Path | None = None,
) -> plt.Figure | None:
    """Line chart of spatial mean/std per time step."""
    if not dataset.has_time or not dataset.time_column:
        return None
    tcol = dataset.time_column
    agg = dataset.gdf.groupby(tcol)[value_col].agg(["mean", "std", "min", "max"])
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(agg))
    ax.plot(x, agg["mean"], "o-", label="均值", color="#2563eb")
    ax.fill_between(x, agg["min"], agg["max"], alpha=0.15, color="#2563eb", label="min–max")
    ax.fill_between(
        x,
        agg["mean"] - agg["std"].fillna(0),
        agg["mean"] + agg["std"].fillna(0),
        alpha=0.25,
        color="#93c5fd",
        label="±1 标准差",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in agg.index], rotation=25, ha="right")
    ax.set_xlabel(tcol)
    ax.set_ylabel(value_col)
    ax.set_title(f"{value_col} 时间聚合（均值 / 波动带）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Multivariate charts
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(
    corr: dict[str, dict[str, float]],
    output_path: str | Path | None = None,
    *,
    title: str = "相关矩阵",
    cmap: str = "RdBu_r",
) -> plt.Figure:
    df = pd.DataFrame(corr)
    fig, ax = plt.subplots(figsize=(max(5, len(df.columns) * 0.9), max(4, len(df.index) * 0.8)))
    im = ax.imshow(df.values.astype(float), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=35, ha="right")
    ax.set_yticklabels(df.index)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(j, i, f"{df.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="相关系数")
    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig


def plot_pairwise_scatter(
    dataset: SpatioTemporalDataset,
    columns: list[str],
    output_path: str | Path | None = None,
    *,
    max_cols: int = 4,
) -> plt.Figure | None:
    cols = columns[:max_cols]
    if len(cols) < 2:
        return None
    df = dataset.gdf[cols].dropna()
    if len(df) < 2:
        return None
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    if n == 1:
        axes = np.array([[axes]])
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                ax.hist(df[ci], bins=12, color="#60a5fa", alpha=0.8)
                ax.set_ylabel(ci if j == 0 else "")
            else:
                ax.scatter(df[cj], df[ci], alpha=0.6, s=18, c="#2563eb", edgecolors="none")
            if i == n - 1:
                ax.set_xlabel(cj, fontsize=8)
            if j == 0:
                ax.set_ylabel(ci, fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("两两散点矩阵", fontsize=12, y=1.02)
    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig


def plot_pca_scree(
    explained_ratio: list[float],
    output_path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(1, len(explained_ratio) + 1)
    ax.bar(x, explained_ratio, color="#6366f1", alpha=0.85, label="单个主成分")
    ax.plot(x, np.cumsum(explained_ratio), "o-", color="#dc2626", label="累计")
    ax.set_xlabel("主成分")
    ax.set_ylabel("方差解释比")
    ax.set_title("PCA 碎石图 / 累计贡献")
    ax.set_xticks(list(x))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig


def plot_pca_loadings(
    loadings: dict[str, list[float]],
    output_path: str | Path | None = None,
    *,
    n_components: int = 2,
) -> plt.Figure:
    variables = list(loadings.keys())
    n_comp = min(n_components, len(next(iter(loadings.values()))))
    fig, axes = plt.subplots(1, n_comp, figsize=(5 * n_comp, 4))
    if n_comp == 1:
        axes = [axes]
    for k, ax in enumerate(axes[:n_comp]):
        vals = [loadings[v][k] for v in variables]
        colors = ["#2563eb" if v >= 0 else "#dc2626" for v in vals]
        ax.barh(variables, vals, color=colors, alpha=0.85)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_title(f"PC{k + 1} 载荷")
        ax.grid(True, alpha=0.25, axis="x")
    fig.suptitle("PCA 变量载荷", fontsize=12)
    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Batch generation for EDA runner
# ---------------------------------------------------------------------------


def generate_univariate_figures(
    dataset: SpatioTemporalDataset,
    columns: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """Generate all univariate charts; return name -> path mapping."""
    cols = columns or dataset.numeric_columns
    figures: dict[str, str] = {}
    if not cols:
        return figures

    out = Path(output_dir) if output_dir else None
    sub = (out / "univariate") if out else None
    if sub:
        sub.mkdir(parents=True, exist_ok=True)

    for col in cols:
        prefix = f"uni_{col}"
        if sub:
            plot_distribution(dataset, col, sub / f"{prefix}_distribution.png")
            figures[f"{prefix}_distribution"] = str(sub / f"{prefix}_distribution.png")
            plot_boxplot(dataset, col, sub / f"{prefix}_boxplot.png")
            figures[f"{prefix}_boxplot"] = str(sub / f"{prefix}_boxplot.png")
            fig = plot_spatial_values(dataset, col, sub / f"{prefix}_spatial.png")
            if fig is not None:
                figures[f"{prefix}_spatial"] = str(sub / f"{prefix}_spatial.png")
            if dataset.has_time:
                plot_timeseries(dataset, col, sub / f"{prefix}_timeseries.png")
                figures[f"{prefix}_timeseries"] = str(sub / f"{prefix}_timeseries.png")
                fig = plot_temporal_decomposition(dataset, col, sub / f"{prefix}_temporal_band.png")
                if fig is not None:
                    figures[f"{prefix}_temporal_band"] = str(sub / f"{prefix}_temporal_band.png")

    return figures


def generate_multivariate_figures(
    dataset: SpatioTemporalDataset,
    multi_result: dict[str, Any],
    columns: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """Generate multivariate charts from analyze_multivariate output."""
    cols = columns or dataset.numeric_columns
    figures: dict[str, str] = {}
    if len(cols) < 2:
        return figures

    out = Path(output_dir) if output_dir else None
    sub = (out / "multivariate") if out else None
    if sub:
        sub.mkdir(parents=True, exist_ok=True)

    if "pearson" in multi_result and sub:
        plot_correlation_heatmap(
            multi_result["pearson"],
            sub / "corr_pearson.png",
            title="Pearson 相关矩阵",
        )
        figures["corr_pearson"] = str(sub / "corr_pearson.png")
    if "spearman" in multi_result and sub:
        plot_correlation_heatmap(
            multi_result["spearman"],
            sub / "corr_spearman.png",
            title="Spearman 相关矩阵",
        )
        figures["corr_spearman"] = str(sub / "corr_spearman.png")
    if sub:
        fig = plot_pairwise_scatter(dataset, cols, sub / "pairwise_scatter.png")
        if fig is not None:
            figures["pairwise_scatter"] = str(sub / "pairwise_scatter.png")

    pca = multi_result.get("pca")
    if pca and sub:
        plot_pca_scree(pca["explained_variance_ratio"], sub / "pca_scree.png")
        figures["pca_scree"] = str(sub / "pca_scree.png")
        if "loadings" in pca:
            plot_pca_loadings(pca["loadings"], sub / "pca_loadings.png")
            figures["pca_loadings"] = str(sub / "pca_loadings.png")

    return figures
