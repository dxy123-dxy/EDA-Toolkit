"""Orchestrate EDA pipeline: profile + univariate + multivariate + optional plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eda_toolkit.eda.multivariate import analyze_multivariate
from eda_toolkit.eda.profile import build_profile, save_profile
from eda_toolkit.eda.univariate import analyze_univariate
from eda_toolkit.io.dataset import SpatioTemporalDataset
from eda_toolkit.viz.plots import plot_choropleth, plot_timeseries


def run_eda(
    dataset: SpatioTemporalDataset,
    config: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run EDA according to config.

    Returns dict with keys: profile, tables, figures (paths if output_dir set).
    """
    config = config or {}
    output_dir = Path(output_dir) if output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    columns = config.get("columns")
    result: dict[str, Any] = {"tables": {}, "figures": {}}

    if config.get("profile", True):
        profile = build_profile(dataset)
        result["profile"] = profile
        if output_dir:
            save_profile(profile, output_dir)
            result["figures"]["profile_json"] = str(output_dir / "profile.json")

    if config.get("univariate", True):
        uni = analyze_univariate(dataset, columns=columns)
        result["tables"]["univariate"] = uni
        if output_dir:
            _save_json(output_dir / "univariate.json", uni)

    if config.get("multivariate", True):
        multi = analyze_multivariate(
            dataset,
            columns=columns,
            pca_components=config.get("pca_components", 3),
        )
        result["tables"]["multivariate"] = multi
        if output_dir:
            _save_json(output_dir / "multivariate.json", multi)

    if config.get("plots", True) and output_dir:
        fig_cfg = config.get("plot_options", {})
        _run_plots(dataset, columns, output_dir, fig_cfg, result)

    if output_dir:
        manifest = {
            "profile": bool(config.get("profile", True)),
            "univariate": bool(config.get("univariate", True)),
            "multivariate": bool(config.get("multivariate", True)),
            "output_dir": str(output_dir),
        }
        _save_json(output_dir / "manifest.json", manifest)
        result["manifest"] = manifest

    return result


def _run_plots(
    dataset: SpatioTemporalDataset,
    columns: list[str] | None,
    output_dir: Path,
    fig_cfg: dict[str, Any],
    result: dict[str, Any],
) -> None:
    cols = columns or dataset.numeric_columns
    if not cols:
        return

    if dataset.has_time and dataset.time_column:
        path = output_dir / "timeseries.png"
        plot_timeseries(
            dataset,
            value_col=cols[0],
            output_path=path,
            title=fig_cfg.get("timeseries_title"),
        )
        result["figures"]["timeseries"] = str(path)

    value_col = cols[0]
    if dataset.geometry_type in ("Polygon", "MultiPolygon"):
        path = output_dir / "choropleth.png"
        plot_choropleth(dataset, column=value_col, output_path=path)
        result["figures"]["choropleth"] = str(path)


def _save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
