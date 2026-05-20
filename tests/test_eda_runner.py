from pathlib import Path

from eda_toolkit.io.loaders import load_dataset
from eda_toolkit.eda.runner import run_eda

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "data" / "sample_points.geojson"


def test_run_eda_minimal(tmp_path):
    ds = load_dataset(SAMPLE, time_column="timestamp")
    result = run_eda(
        ds,
        config={"profile": True, "univariate": True, "multivariate": True, "plots": False},
        output_dir=tmp_path,
    )
    assert "profile" in result
    assert "univariate" in result["tables"]
    assert (tmp_path / "profile.json").exists()


def test_run_eda_with_plots(tmp_path):
    ds = load_dataset(SAMPLE, time_column="timestamp")
    result = run_eda(ds, config={"plots": True}, output_dir=tmp_path)
    assert "timeseries" in result.get("figures", {})
