"""End-to-end sample: load points, profile, full EDA."""

from pathlib import Path

from eda_toolkit.io.loaders import load_dataset
from eda_toolkit.eda.runner import run_eda

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data" / "sample_points.geojson"
OUT = ROOT / "output" / "sample_eda"
CONFIG = ROOT / "examples" / "config" / "eda_default.json"


def main() -> None:
    import json

    ds = load_dataset(DATA, time_column="timestamp")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = run_eda(ds, config=cfg, output_dir=OUT)
    print("Done. Figures:", result.get("figures"))


if __name__ == "__main__":
    main()
