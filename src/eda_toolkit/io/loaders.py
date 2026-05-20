"""Load spatio-temporal data from common file formats."""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

import geopandas as gpd
import pandas as pd

from eda_toolkit.io.dataset import SpatioTemporalDataset

_VECTOR_EXTENSIONS = {".geojson", ".json", ".shp", ".gpkg", ".geoparquet", ".parquet"}
_TABULAR_EXTENSIONS = {".csv", ".tsv", ".txt"}
_SHAPEFILE_SIDECAR = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".xml", ".qix", ".qmd"}
_ZIP_EXTENSIONS = {".zip"}


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
        if suffix == ".shp":
            _enable_shapefile_restore_shx()
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


def _enable_shapefile_restore_shx() -> None:
    """Allow GDAL to rebuild missing .shx when possible (fallback only)."""
    os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")


def _read_bytes(file_bytes: bytes | BinaryIO) -> bytes:
    return file_bytes.read() if hasattr(file_bytes, "read") else file_bytes


def _stage_uploads_to_directory(
    uploads: list[tuple[str, bytes]],
) -> tuple[Path, Path]:
    """
    Write uploaded files into one temp directory.

    Returns (temp_dir, main_vector_path).
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="eda_upload_"))
    shp_paths: list[Path] = []

    for name, data in uploads:
        safe_name = Path(name).name
        dest = tmp_dir / safe_name
        dest.write_bytes(data)
        ext = dest.suffix.lower()
        if ext == ".shp":
            shp_paths.append(dest)
        if ext in _ZIP_EXTENSIONS:
            _extract_zip(dest, tmp_dir)
            dest.unlink(missing_ok=True)
            shp_paths.extend(tmp_dir.rglob("*.shp"))

    if not shp_paths:
        # Single-file formats: pick first recognizable vector/tabular file
        for f in sorted(tmp_dir.iterdir()):
            if f.suffix.lower() in _VECTOR_EXTENSIONS | _TABULAR_EXTENSIONS:
                return tmp_dir, f
        raise ValueError(
            "未找到可读取的数据文件。Shapefile 请同时上传 .shp、.shx、.dbf（及 .prj），"
            "或上传包含完整 Shapefile 的 .zip。"
        )

    if len(shp_paths) > 1:
        names = ", ".join(p.name for p in shp_paths)
        raise ValueError(
            f"检测到多个 .shp 文件（{names}），请一次只加载一套 Shapefile，或分别打包为 zip。"
        )

    shp_path = shp_paths[0]
    _validate_shapefile_bundle(shp_path)
    return tmp_dir, shp_path


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            target = dest_dir / Path(member).name
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def _validate_shapefile_bundle(shp_path: Path) -> None:
    """Warn via exception if mandatory sidecar files are missing."""
    stem = shp_path.with_suffix
    missing = []
    for ext in (".shx", ".dbf"):
        if not stem(ext).exists():
            missing.append(ext)
    if missing:
        raise ValueError(
            f"Shapefile 不完整：缺少 {', '.join(missing)}。"
            f"请在 Web 端 **一次选择多个文件**（.shp、.shx、.dbf、.prj 等），"
            f"或把整套文件打成 zip 上传。仅上传 .shp 无法读取。"
            f"\n当前临时目录仅有: {[p.name for p in shp_path.parent.iterdir()]}"
        )


def load_dataset_from_uploads(
    uploads: list[tuple[str, bytes]],
    *,
    time_column: str | None = None,
    lon_column: str = "longitude",
    lat_column: str = "latitude",
    crs: str | None = "EPSG:4326",
    **read_kwargs: Any,
) -> SpatioTemporalDataset:
    """Load from one or more uploaded files (supports complete Shapefile sets and zip)."""
    if not uploads:
        raise ValueError("未提供任何文件")

    if len(uploads) == 1:
        name, data = uploads[0]
        suffix = Path(name).suffix.lower()
        if suffix in _ZIP_EXTENSIONS:
            pass  # fall through to directory staging
        elif suffix in _VECTOR_EXTENSIONS | _TABULAR_EXTENSIONS:
            return load_dataset_from_upload(
                name,
                data,
                time_column=time_column,
                lon_column=lon_column,
                lat_column=lat_column,
                crs=crs,
                **read_kwargs,
            )
        else:
            raise ValueError(f"不支持的文件类型: {suffix}")

    tmp_dir, main_path = _stage_uploads_to_directory(uploads)
    try:
        return load_dataset(
            main_path,
            time_column=time_column,
            lon_column=lon_column,
            lat_column=lat_column,
            crs=crs,
            **read_kwargs,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
    """Load dataset from a single uploaded file."""
    suffix = Path(file_name).suffix.lower()
    if suffix not in _VECTOR_EXTENSIONS | _TABULAR_EXTENSIONS | _ZIP_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {suffix}")

    data = _read_bytes(file_bytes)

    if suffix == ".shp":
        raise ValueError(
            "仅上传了 .shp 文件。Shapefile 需同时包含 .shx、.dbf 等文件："
            "请在文件选择框中 **按住 Ctrl 多选** 同目录下的全部相关文件，或上传 zip。"
        )

    if suffix in _ZIP_EXTENSIONS:
        return load_dataset_from_uploads(
            [(file_name, data)],
            time_column=time_column,
            lon_column=lon_column,
            lat_column=lat_column,
            crs=crs,
            **read_kwargs,
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        return load_dataset(
            tmp_path,
            time_column=time_column,
            lon_column=lon_column,
            lat_column=lat_column,
            crs=crs,
            **read_kwargs,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
