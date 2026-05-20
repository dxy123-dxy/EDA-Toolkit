import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from eda_toolkit.eda.profile import build_profile
from eda_toolkit.io.dataset import SpatioTemporalDataset


def test_profile_with_array_column():
    gdf = gpd.GeoDataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02"],
            "value": [1.0, 2.0],
            "coords": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        crs="EPSG:4326",
    )
    ds = SpatioTemporalDataset(gdf, time_column="timestamp")
    profile = build_profile(ds)
    assert "value" in profile["columns"]
    assert profile["columns"]["value"]["n_unique"] == 2
