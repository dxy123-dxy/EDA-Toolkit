"""Phase 1.2 classic algorithm tests."""

from pathlib import Path

import pytest

from eda_toolkit.algorithms.areal import (
    geary_global,
    lisa_local,
    moran_global,
    run_areal_analysis,
)
from eda_toolkit.algorithms.interpolation import (
    empirical_variogram,
    idw_interpolate,
    ordinary_kriging,
)
from eda_toolkit.algorithms.point_pattern import kde_heatmap, ripley_functions
from eda_toolkit.io.loaders import load_dataset

ROOT = Path(__file__).resolve().parents[1]
POLYGONS = ROOT / "examples" / "data" / "sample_polygons.geojson"
EVENTS = ROOT / "examples" / "data" / "sample_events.geojson"


@pytest.fixture
def poly_ds():
    return load_dataset(POLYGONS)


@pytest.fixture
def event_ds():
    return load_dataset(EVENTS, time_column="timestamp")


def test_moran_geary_lisa(poly_ds):
    m = moran_global(poly_ds, "gdp", weight_type="queen")
    assert "I" in m
    assert m["n"] == 16

    g = geary_global(poly_ds, "gdp", weight_type="queen")
    assert "C" in g

    lisa = lisa_local(poly_ds, "gdp", weight_type="queen")
    assert len(lisa["features"]) == 16
    assert "cluster_counts" in lisa


def test_run_areal_analysis_writes(poly_ds, tmp_path):
    out = run_areal_analysis(poly_ds, "gdp", output_dir=tmp_path)
    assert (tmp_path / "areal_summary.json").exists()
    assert (tmp_path / "lisa_features.csv").exists()
    assert (tmp_path / "lisa_map.geojson").exists()
    assert "artifacts" in out


def test_kde_and_ripley(event_ds, tmp_path):
    kde = kde_heatmap(event_ds, grid_size=20, output_dir=tmp_path / "kde")
    assert kde["density_max"] > kde["density_min"]
    assert (tmp_path / "kde" / "kde_grid.csv").exists()

    ripley = ripley_functions(
        event_ds,
        functions=("K", "G"),
        support=8,
        n_simulations=9,
        run_csr_test=True,
        output_dir=tmp_path / "ripley",
    )
    assert set(ripley["curves"]) == {"K", "G"}
    assert (tmp_path / "ripley" / "ripley_k.csv").exists()


def test_ripley_without_csr(event_ds):
    ripley = ripley_functions(
        event_ds, functions=("K",), support=8, run_csr_test=False
    )
    assert "statistic" in ripley["curves"]["K"]
    assert "pvalue" not in ripley["curves"]["K"]


def test_idw_and_variogram(event_ds, tmp_path):
    vario = empirical_variogram(event_ds, "pm25", n_lags=8)
    assert len(vario["lags"]) == 8

    idw = idw_interpolate(event_ds, "pm25", grid_size=15, output_dir=tmp_path)
    assert idw["z_max"] >= idw["z_min"]
    assert (tmp_path / "idw_grid.csv").exists()


def test_ordinary_kriging(event_ds, tmp_path):
    out = ordinary_kriging(
        event_ds,
        "pm25",
        grid_size=12,
        leave_one_out_cv=False,
        output_dir=tmp_path,
    )
    assert out["z_max"] >= out["z_min"]
    assert "empirical_variogram" in out
    assert (tmp_path / "kriging_grid.csv").exists()
