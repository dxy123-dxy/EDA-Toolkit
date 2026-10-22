"""Spatial weights helpers for areal / point attribute analysis."""

from __future__ import annotations

from typing import Any

import geopandas as gpd


def build_weights(
    gdf: gpd.GeoDataFrame,
    weight_type: str = "queen",
    *,
    k: int = 8,
) -> Any:
    """
    Build a PySAL weights object.

    weight_type: queen | rook | knn
    For point geometries, prefer knn (contiguity is undefined / unstable).
    """
    try:
        from libpysal.weights import KNN, Queen, Rook
    except ImportError as exc:
        raise ImportError(
            "Install spatial extras: pip install 'eda-toolkit[spatial]'"
        ) from exc

    wt = weight_type.lower()
    if wt == "queen":
        return Queen.from_dataframe(gdf, use_index=True)
    if wt == "rook":
        return Rook.from_dataframe(gdf, use_index=True)
    if wt == "knn":
        return KNN.from_dataframe(gdf, k=k, use_index=True)
    raise ValueError(f"Unknown weight_type: {weight_type}")
