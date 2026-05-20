"""Unified spatio-temporal dataset abstraction."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd

_TIME_CANDIDATES = (
    "time",
    "timestamp",
    "datetime",
    "date",
    "dt",
    "t",
    "period",
    "year_month",
)


class SpatioTemporalDataset:
    """Wrapper around GeoDataFrame with time-column semantics."""

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        *,
        time_column: str | None = None,
        region_id_column: str | None = None,
    ) -> None:
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError("gdf must be a GeoDataFrame")
        self.gdf = gdf.copy()
        self.time_column = time_column or self._infer_time_column()
        self.region_id_column = region_id_column
        if self.time_column and self.time_column in self.gdf.columns:
            self.gdf[self.time_column] = pd.to_datetime(
                self.gdf[self.time_column], errors="coerce"
            )

    def _infer_time_column(self) -> str | None:
        lower_map = {c.lower(): c for c in self.gdf.columns}
        for candidate in _TIME_CANDIDATES:
            if candidate in lower_map:
                return lower_map[candidate]
        return None

    @property
    def has_time(self) -> bool:
        return self.time_column is not None and self.time_column in self.gdf.columns

    @property
    def numeric_columns(self) -> list[str]:
        exclude = {self.time_column, "geometry"}
        if self.region_id_column:
            exclude.add(self.region_id_column)
        return [
            c
            for c in self.gdf.select_dtypes(include="number").columns
            if c not in exclude
        ]

    @property
    def geometry_type(self) -> str:
        if self.gdf.empty:
            return "empty"
        modes = self.gdf.geometry.geom_type.mode()
        if len(modes):
            return str(modes.iloc[0])
        return str(self.gdf.geometry.geom_type.iloc[0])

    def to_projected(self, crs: str | None = None) -> gpd.GeoDataFrame:
        """Return a copy projected to a metric CRS (for distance-based analysis)."""
        gdf = self.gdf
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        if crs:
            return gdf.to_crs(crs)
        try:
            return gdf.to_crs(gdf.estimate_utm_crs())
        except Exception:
            return gdf.to_crs("EPSG:3857")

    def time_slice(self, value: Any) -> gpd.GeoDataFrame:
        if not self.has_time:
            raise ValueError("Dataset has no time column")
        return self.gdf[self.gdf[self.time_column] == value]

    def summary_dict(self) -> dict[str, Any]:
        bounds = self.gdf.total_bounds.tolist() if not self.gdf.empty else None
        time_range = None
        if self.has_time:
            ts = self.gdf[self.time_column].dropna()
            if len(ts):
                time_range = {
                    "min": str(ts.min()),
                    "max": str(ts.max()),
                    "n_unique": int(ts.nunique()),
                }
        return {
            "n_rows": len(self.gdf),
            "n_columns": len(self.gdf.columns),
            "crs": str(self.gdf.crs) if self.gdf.crs else None,
            "geometry_type": self.geometry_type,
            "bounds": bounds,
            "time_column": self.time_column,
            "has_time": self.has_time,
            "numeric_columns": self.numeric_columns,
            "time_range": time_range,
        }
