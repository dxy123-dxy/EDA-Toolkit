import geopandas as gpd
from shapely.geometry import Point

from eda_toolkit.io.dataset import SpatioTemporalDataset


def test_infer_time_column():
    gdf = gpd.GeoDataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "value": [1.0, 2.0],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        crs="EPSG:4326",
    )
    ds = SpatioTemporalDataset(gdf)
    assert ds.time_column == "date"
    assert ds.has_time


def test_numeric_columns_excludes_time():
    gdf = gpd.GeoDataFrame(
        {
            "timestamp": ["2024-01-01"],
            "pm25": [10.0],
            "geometry": [Point(116.4, 39.9)],
        },
        crs="EPSG:4326",
    )
    ds = SpatioTemporalDataset(gdf, time_column="timestamp")
    assert "pm25" in ds.numeric_columns
    assert "timestamp" not in ds.numeric_columns
