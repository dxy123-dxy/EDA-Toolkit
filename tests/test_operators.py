from pathlib import Path

from eda_toolkit.etl.operator import OperatorContext
from eda_toolkit.etl.registry import get_operator, list_operators

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "data" / "sample_points.geojson"
POLYGONS = ROOT / "examples" / "data" / "sample_polygons.geojson"
EVENTS = ROOT / "examples" / "data" / "sample_events.geojson"


def test_list_operators():
    ops = list_operators()
    names = {o["name"] for o in ops}
    assert "profile" in names
    assert "eda_run" in names
    assert "areal_autocorr" in names
    assert "point_pattern" in names
    assert "idw" in names
    assert "kriging" in names
    assert "variogram" in names
    assert "dl_infer" in names


def test_profile_operator(tmp_path):
    op = get_operator("profile")()
    ctx = OperatorContext(
        input_path=SAMPLE,
        output_dir=tmp_path,
        params={},
        time_column="timestamp",
    )
    result = op.run(ctx)
    assert result.success
    assert (tmp_path / "profile.json").exists()


def test_areal_operator(tmp_path):
    op = get_operator("areal_autocorr")()
    ctx = OperatorContext(
        input_path=POLYGONS,
        output_dir=tmp_path,
        params={"column": "gdp", "weight_type": "queen"},
    )
    result = op.run(ctx)
    assert result.success
    assert "moran" in result.diagnostics
    assert (tmp_path / "operator_result.json").exists()


def test_point_pattern_operator(tmp_path):
    op = get_operator("point_pattern")()
    ctx = OperatorContext(
        input_path=EVENTS,
        output_dir=tmp_path,
        params={
            "kde": True,
            "ripley": True,
            "n_simulations": 9,
            "support": 8,
            "kde_grid_size": 15,
            "functions": ["K"],
        },
        time_column="timestamp",
    )
    result = op.run(ctx)
    assert result.success
    assert "kde" in result.diagnostics
    assert "ripley" in result.diagnostics


def test_idw_and_kriging_operators(tmp_path):
    idw = get_operator("idw")()
    ctx = OperatorContext(
        input_path=EVENTS,
        output_dir=tmp_path / "idw",
        params={"column": "pm25", "grid_size": 12},
        time_column="timestamp",
    )
    r1 = idw.run(ctx)
    assert r1.success

    krig = get_operator("kriging")()
    ctx2 = OperatorContext(
        input_path=EVENTS,
        output_dir=tmp_path / "krig",
        params={
            "column": "pm25",
            "grid_size": 10,
            "leave_one_out_cv": True,
            "variogram_model": "linear",
        },
        time_column="timestamp",
    )
    r2 = krig.run(ctx2)
    assert r2.success
    assert r2.diagnostics.get("method") == "OrdinaryKriging"
