"""Univariate spatio-temporal EDA."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from eda_toolkit.io.dataset import SpatioTemporalDataset

_MIN_SAMPLES_ADF = 20


def analyze_univariate(
    dataset: SpatioTemporalDataset,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Temporal, spatial, and spatio-temporal summaries for numeric columns."""
    cols = columns or dataset.numeric_columns
    if not cols:
        return {"columns": {}, "note": "No numeric columns found"}

    gdf = dataset.gdf
    result: dict[str, Any] = {"columns": {}}

    for col in cols:
        if col not in gdf.columns:
            continue
        col_result: dict[str, Any] = {
            "spatial": _spatial_summary(gdf[col]),
        }
        if dataset.has_time and dataset.time_column:
            col_result["temporal"] = _temporal_summary(
                gdf, col, dataset.time_column
            )
            col_result["spatiotemporal"] = _spatiotemporal_summary(
                gdf, col, dataset.time_column
            )
        result["columns"][col] = col_result

    return result


def _spatial_summary(series: pd.Series) -> dict[str, Any]:
    s = series.dropna()
    if s.empty:
        return {}
    q = s.quantile([0.25, 0.5, 0.75, 0.95]).to_dict()
    return {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
        "quantiles": {str(k): float(v) for k, v in q.items()},
    }


def _temporal_summary(
    gdf: pd.DataFrame, value_col: str, time_col: str
) -> dict[str, Any]:
    ts = (
        gdf.groupby(time_col)[value_col]
        .agg(["mean", "std", "count"])
        .dropna(how="all")
    )
    if ts.empty:
        return {}

    out: dict[str, Any] = {
        "global_mean_over_time": float(ts["mean"].mean()),
        "global_std_over_time": float(ts["mean"].std()) if len(ts) > 1 else 0.0,
        "n_time_steps": int(len(ts)),
        "series_head": ts.head(5).reset_index().astype(str).to_dict(orient="records"),
    }

    values = ts["mean"].dropna()
    if len(values) >= _MIN_SAMPLES_ADF:
        try:
            from statsmodels.tsa.stattools import adfuller

            adf = adfuller(values, autolag="AIC")
            out["adf"] = {
                "statistic": float(adf[0]),
                "pvalue": float(adf[1]),
                "hint": "likely stationary" if adf[1] < 0.05 else "may be non-stationary",
            }
        except Exception as exc:
            out["adf_error"] = str(exc)

    if len(values) >= 8:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            period = min(12, max(2, len(values) // 2))
            if len(values) >= 2 * period:
                decomp = seasonal_decompose(
                    values, model="additive", period=period, extrapolate_trend="freq"
                )
                out["seasonal_strength"] = float(
                    1
                    - np.var(decomp.resid.dropna())
                    / (np.var(decomp.seasonal.dropna() + decomp.resid.dropna()) + 1e-12)
                )
        except Exception:
            pass

    return out


def _spatiotemporal_summary(
    gdf: pd.DataFrame, value_col: str, time_col: str
) -> dict[str, Any]:
    by_time = gdf.groupby(time_col)[value_col].agg(["mean", "var"]).dropna(how="all")
    if by_time.empty:
        return {}
    return {
        "mean_of_spatial_means": float(by_time["mean"].mean()),
        "variance_of_spatial_means": float(by_time["mean"].var())
        if len(by_time) > 1
        else 0.0,
        "mean_spatial_variance": float(by_time["var"].mean()),
    }
