"""Load spatio-temporal data from common file formats."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, BinaryIO

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


def load_dataset_from_upload(
    file_name: str,
    file_bytes: bytes | BinaryIO,
    *,
    time_column: str | None = None,
    lon_column: str = "longitude",
    lat_column: str = "latitude",
    crs: str | None = "EPSG:4326",
    **read_kwargs: Any,
) -> SpatioTemporalDataset:
    """Load dataset from an uploaded file (e.g. Streamlit file uploader)."""
    suffix = Path(file_name).suffix.lower()
    if suffix not in _VECTOR_EXTENSIONS | _TABULAR_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {suffix}")

    data = file_bytes.read() if hasattr(file_bytes, "read") else file_bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    return load_dataset(
        tmp_path,
        time_column=time_column,
        lon_column=lon_column,
        lat_column=lat_column,
        crs=crs,
        **read_kwargs,
    )
