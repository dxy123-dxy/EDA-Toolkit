"""Spatial autocorrelation for polygon datasets."""

from __future__ import annotations

from typing import Any

from eda_toolkit.io.dataset import SpatioTemporalDataset


def moran_global(
    dataset: SpatioTemporalDataset,
    column: str,
    *,
    weight_type: str = "queen",
) -> dict[str, Any]:
    """
    Global Moran's I using PySAL (requires optional spatial deps).

    weight_type: queen | rook | knn
    """
    try:
        from libpysal.weights import KNN, Queen, Rook
        from esda.moran import Moran
    except ImportError as exc:
        raise ImportError(
            "Install spatial extras: pip install 'eda-toolkit[spatial]'"
        ) from exc

    gdf = dataset.to_projected()
    if column not in gdf.columns:
        raise ValueError(f"Column '{column}' not found")

    if weight_type == "queen":
        w = Queen.from_dataframe(gdf)
    elif weight_type == "rook":
        w = Rook.from_dataframe(gdf)
    elif weight_type == "knn":
        w = KNN.from_dataframe(gdf, k=8)
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")

    y = gdf[column].values
    moran = Moran(y, w)
    return {
        "I": float(moran.I),
        "p_sim": float(moran.p_sim),
        "z_sim": float(moran.z_sim),
        "weight_type": weight_type,
        "inference": "significant" if moran.p_sim < 0.05 else "not_significant",
    }
