import zipfile
from pathlib import Path

import pytest

from eda_toolkit.io.loaders import load_dataset_from_upload, load_dataset_from_uploads

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "data" / "sample_points.geojson"


def test_single_shp_rejected_with_helpful_message():
    data = SAMPLE.read_bytes()  # wrong ext but tests path; use fake shp name
    with pytest.raises(ValueError, match="仅上传了 .shp"):
        load_dataset_from_upload("test.shp", data)


def test_shapefile_bundle_from_directory(tmp_path):
    gdf = __import__("geopandas").read_file(SAMPLE)
    shp_path = tmp_path / "pts.shp"
    gdf.to_file(shp_path, driver="ESRI Shapefile")
    uploads = [
        (p.name, p.read_bytes())
        for p in sorted(tmp_path.glob("pts.*"))
        if p.is_file()
    ]
    assert len(uploads) >= 3
    ds = load_dataset_from_uploads(uploads)
    assert len(ds.gdf) == len(gdf)


def test_shapefile_zip_upload(tmp_path):
    gdf = __import__("geopandas").read_file(SAMPLE)
    shp_path = tmp_path / "pts.shp"
    gdf.to_file(shp_path, driver="ESRI Shapefile")
    zpath = tmp_path / "data.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for f in tmp_path.glob("pts.*"):
            if f.is_file():
                zf.write(f, f.name)
    ds = load_dataset_from_upload("data.zip", zpath.read_bytes())
    assert len(ds.gdf) >= 1
