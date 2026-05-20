"""Multivariate spatio-temporal EDA."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from eda_toolkit.io.dataset import SpatioTemporalDataset


def analyze_multivariate(
    dataset: SpatioTemporalDataset,
    columns: list[str] | None = None,
    *,
    pca_components: int | None = 3,
) -> dict[str, Any]:
    """Correlation matrices and optional PCA on numeric columns."""
    cols = columns or dataset.numeric_columns
    if len(cols) < 2:
        return {"note": "Need at least 2 numeric columns for multivariate analysis"}

    gdf = dataset.gdf
    df = gdf[cols].dropna()
    result: dict[str, Any] = {
        "pearson": df.corr(method="pearson").round(6).to_dict(),
        "spearman": df.corr(method="spearman").round(6).to_dict(),
    }

    if dataset.has_time and dataset.time_column:
        time_col = dataset.time_column
        by_time: dict[str, Any] = {}
        for t, group in gdf.groupby(time_col):
            sub = group[cols].dropna()
            if len(sub) >= 2:
                by_time[str(t)] = sub.corr(method="pearson").round(6).to_dict()
        result["pearson_by_time"] = by_time

    if pca_components and len(df) >= pca_components:
        result["pca"] = _run_pca(df, n_components=pca_components)

    return result


def _run_pca(df: pd.DataFrame, n_components: int) -> dict[str, Any]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_components = min(n_components, df.shape[1], len(df))
    X = StandardScaler().fit_transform(df.values)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    return {
        "explained_variance_ratio": [
            float(x) for x in pca.explained_variance_ratio_
        ],
        "components_by_pc": {
            f"PC{i + 1}": {
                df.columns[j]: float(pca.components_[i, j])
                for j in range(len(df.columns))
            }
            for i in range(n_components)
        },
        "loadings": {
            df.columns[j]: [float(pca.components_[i, j]) for i in range(n_components)]
            for j in range(len(df.columns))
        },
        "score_summary": {
            f"PC{i+1}": {"mean": float(scores[:, i].mean()), "std": float(scores[:, i].std())}
            for i in range(n_components)
        },
    }
