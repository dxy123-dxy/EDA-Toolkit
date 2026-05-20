from pathlib import Path

from eda_toolkit.etl.registry import get_operator, list_operators
from eda_toolkit.etl.operator import OperatorContext

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "data" / "sample_points.geojson"


def test_list_operators():
    ops = list_operators()
    names = {o["name"] for o in ops}
    assert "profile" in names
    assert "eda_run" in names


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
