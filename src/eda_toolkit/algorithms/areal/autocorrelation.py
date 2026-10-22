"""Spatial autocorrelation for polygon / point-attribute datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from eda_toolkit.algorithms.areal.weights import build_weights
from eda_toolkit.io.dataset import SpatioTemporalDataset

_LISA_LABELS = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}


def moran_global(
    dataset: SpatioTemporalDataset,
    column: str,
    *,
    weight_type: str = "queen",
    k: int = 8,
) -> dict[str, Any]:
    """Global Moran's I (PySAL esda)."""
    try:
        from esda.moran import Moran
    except ImportError as exc:
        raise ImportError(
            "Install spatial extras: pip install 'eda-toolkit[spatial]'"
        ) from exc

    gdf = dataset.to_projected()
    _require_column(gdf, column)
    w = build_weights(gdf, weight_type, k=k)
    y = gdf[column].astype(float).values
    try:
        moran = Moran(y, w, alternative="two-sided")
    except TypeError:
        moran = Moran(y, w)
    return {
        "metric": "Moran_I",
        "I": float(moran.I),
        "EI": float(moran.EI),
        "p_sim": float(moran.p_sim),
        "z_sim": float(moran.z_sim),
        "weight_type": weight_type,
        "n": int(len(y)),
        "inference": "significant" if moran.p_sim < 0.05 else "not_significant",
    }


def geary_global(
    dataset: SpatioTemporalDataset,
    column: str,
    *,
    weight_type: str = "queen",
    k: int = 8,
) -> dict[str, Any]:
    """Global Geary's C (PySAL esda)."""
    try:
        from esda.geary import Geary
    except ImportError as exc:
        raise ImportError(
            "Install spatial extras: pip install 'eda-toolkit[spatial]'"
        ) from exc

    gdf = dataset.to_projected()
    _require_column(gdf, column)
    w = build_weights(gdf, weight_type, k=k)
    y = gdf[column].astype(float).values
    try:
        geary = Geary(y, w, alternative="two-sided")
    except TypeError:
        geary = Geary(y, w)
    return {
        "metric": "Geary_C",
        "C": float(geary.C),
        "p_sim": float(geary.p_sim),
        "z_sim": float(geary.z_sim),
        "weight_type": weight_type,
        "n": int(len(y)),
        "inference": "significant" if geary.p_sim < 0.05 else "not_significant",
    }


def lisa_local(
    dataset: SpatioTemporalDataset,
    column: str,
    *,
    weight_type: str = "queen",
    k: int = 8,
    p_threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Local Indicators of Spatial Association (Moran Local / LISA).

    Returns summary plus a per-feature table with Ii, p, quadrant label.
    """
    try:
        from esda.moran import Moran_Local
    except ImportError as exc:
        raise ImportError(
            "Install spatial extras: pip install 'eda-toolkit[spatial]'"
        ) from exc

    gdf = dataset.to_projected()
    _require_column(gdf, column)
    w = build_weights(gdf, weight_type, k=k)
    y = gdf[column].astype(float).values
    try:
        lisa = Moran_Local(y, w, alternative="two-sided")
    except TypeError:
        lisa = Moran_Local(y, w)

    labels = []
    for q, p in zip(lisa.q, lisa.p_sim):
        if p >= p_threshold:
            labels.append("NS")
        else:
            labels.append(_LISA_LABELS.get(int(q), "NS"))

    features = pd.DataFrame(
        {
            "index": gdf.index.astype(str),
            "value": y,
            "Ii": lisa.Is,
            "p_sim": lisa.p_sim,
            "quadrant": lisa.q,
            "cluster": labels,
        }
    )

    cluster_counts = features["cluster"].value_counts().to_dict()
    return {
        "metric": "Moran_Local",
        "weight_type": weight_type,
        "p_threshold": p_threshold,
        "n": int(len(y)),
        "cluster_counts": {str(k): int(v) for k, v in cluster_counts.items()},
        "features": features.to_dict(orient="records"),
    }


def run_areal_analysis(
    dataset: SpatioTemporalDataset,
    column: str,
    *,
    weight_type: str = "queen",
    k: int = 8,
    include_geary: bool = True,
    include_lisa: bool = True,
    p_threshold: float = 0.05,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run global Moran (+ optional Geary/LISA) and optionally persist results."""
    result: dict[str, Any] = {
        "column": column,
        "moran": moran_global(dataset, column, weight_type=weight_type, k=k),
    }
    if include_geary:
        result["geary"] = geary_global(
            dataset, column, weight_type=weight_type, k=k
        )
    if include_lisa:
        result["lisa"] = lisa_local(
            dataset,
            column,
            weight_type=weight_type,
            k=k,
            p_threshold=p_threshold,
        )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary = {k: v for k, v in result.items() if k != "lisa"}
        if "lisa" in result:
            summary["lisa"] = {
                kk: vv
                for kk, vv in result["lisa"].items()
                if kk != "features"
            }
            feat_path = out / "lisa_features.csv"
            pd.DataFrame(result["lisa"]["features"]).to_csv(feat_path, index=False)
            result["artifacts"] = {"lisa_features_csv": str(feat_path)}
            # Attach cluster labels to a GeoJSON copy for mapping
            gdf_out = dataset.gdf.copy()
            feat_df = pd.DataFrame(result["lisa"]["features"])
            gdf_out = gdf_out.reset_index(drop=True)
            gdf_out["lisa_Ii"] = feat_df["Ii"].values
            gdf_out["lisa_p"] = feat_df["p_sim"].values
            gdf_out["lisa_cluster"] = feat_df["cluster"].values
            geo_path = out / "lisa_map.geojson"
            gdf_out.to_file(geo_path, driver="GeoJSON")
            result["artifacts"]["lisa_map_geojson"] = str(geo_path)

        summary_path = out / "areal_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result.setdefault("artifacts", {})["areal_summary_json"] = str(summary_path)

    return result


def _require_column(gdf: pd.DataFrame, column: str) -> None:
    if column not in gdf.columns:
        raise ValueError(f"Column '{column}' not found")
