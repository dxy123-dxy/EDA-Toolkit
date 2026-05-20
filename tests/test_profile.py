from pathlib import Path

from eda_toolkit.io.loaders import load_dataset
from eda_toolkit.eda.profile import build_profile, save_profile

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "data" / "sample_points.geojson"


def test_build_profile():
    ds = load_dataset(SAMPLE, time_column="timestamp")
    profile = build_profile(ds)
    assert profile["summary"]["n_rows"] == 9
    assert profile["summary"]["has_time"] is True
    assert "pm25" in profile["columns"]


def test_save_profile(tmp_path):
    ds = load_dataset(SAMPLE, time_column="timestamp")
    profile = build_profile(ds)
    path = save_profile(profile, tmp_path)
    assert path.exists()
    assert "profile.json" in path.name
