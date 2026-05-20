"""Visualization helpers (decoupled from computation)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from eda_toolkit.io.dataset import SpatioTemporalDataset


def plot_timeseries(
    dataset: SpatioTemporalDataset,
    value_col: str,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    if not dataset.has_time or not dataset.time_column:
        raise ValueError("Dataset has no time column for timeseries plot")

    gdf = dataset.gdf
    ts = gdf.groupby(dataset.time_column)[value_col].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ts.plot(ax=ax, marker="o")
    ax.set_xlabel(dataset.time_column)
    ax.set_ylabel(value_col)
    ax.set_title(title or f"Mean {value_col} over time")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_choropleth(
    dataset: SpatioTemporalDataset,
    column: str,
    output_path: str | Path,
    *,
    scheme: str = "Quantiles",
    k: int = 5,
) -> Path:
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
    ax.set_title(f"{column} (spatial)")
    ax.set_axis_off()
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
