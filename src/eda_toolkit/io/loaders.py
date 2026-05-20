"""Load spatio-temporal data from common file formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from eda_toolkit.io.dataset import SpatioTemporalDataset

_VECTOR_EXTENSIONS = {".geojson", ".json", ".shp", ".gpkg", ".geoparquet", ".parquet"}
_TABULAR_EXTENSIONS = {".csv", ".tsv", ".txt"}


def load_dataset(
    path: str | Path,
    *,
    time_column: str | None = None,
    lon_column: str = "longitude",
    lat_column: str = "latitude",
    crs: str | None = "EPSG:4326",
    **read_kwargs: Any,
) -> SpatioTemporalDataset:
    """Load a dataset from path and wrap as :class:`SpatioTemporalDataset`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in _VECTOR_EXTENSIONS:
        gdf = gpd.read_file(path, **read_kwargs)
        return SpatioTemporalDataset(gdf, time_column=time_column)

    if suffix in _TABULAR_EXTENSIONS:
        sep = "\t" if suffix == ".tsv" else read_kwargs.pop("sep", ",")
        df = pd.read_csv(path, sep=sep, **read_kwargs)
        for col in (lon_column, lat_column):
            if col not in df.columns:
                raise ValueError(f"CSV must contain column '{col}'")
        geometry = gpd.points_from_xy(df[lon_column], df[lat_column])
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
        return SpatioTemporalDataset(gdf, time_column=time_column)

    raise ValueError(f"Unsupported file extension: {suffix}")
