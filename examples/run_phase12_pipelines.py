"""End-to-end phase 1.2 pipelines: areal LISA + point pattern + interpolation."""

from __future__ import annotations

import json
from pathlib import Path

from eda_toolkit.etl.operator import OperatorContext
from eda_toolkit.etl.registry import get_operator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
CFG = ROOT / "examples" / "config"
OUT = ROOT / "output" / "phase12"


def _run(name: str, input_path: Path, params: dict, out_subdir: str, time_column: str | None = None) -> None:
    out = OUT / out_subdir
    ctx = OperatorContext(
        input_path=input_path,
        output_dir=out,
        params=params,
        time_column=time_column,
    )
    result = get_operator(name)().run(ctx)
    print(f"[{name}] {result.message}")
    for k, v in result.artifacts.items():
        print(f"  {k}: {v}")


def main() -> None:
    areal = json.loads((CFG / "areal_gdp.json").read_text(encoding="utf-8"))
    point = json.loads((CFG / "point_pattern.json").read_text(encoding="utf-8"))
    # Faster demo run
    point["n_simulations"] = 49
    idw = json.loads((CFG / "idw_pm25.json").read_text(encoding="utf-8"))
    krig = json.loads((CFG / "kriging_pm25.json").read_text(encoding="utf-8"))

    print("=== Pipeline A: district indicators (areal) ===")
    _run("profile", DATA / "sample_polygons.geojson", {}, "A_profile")
    _run("areal_autocorr", DATA / "sample_polygons.geojson", areal, "A_lisa")

    print("=== Pipeline B: point events (pattern + interpolation) ===")
    _run(
        "profile",
        DATA / "sample_events.geojson",
        {},
        "B_profile",
        time_column="timestamp",
    )
    _run(
        "point_pattern",
        DATA / "sample_events.geojson",
        point,
        "B_point_pattern",
        time_column="timestamp",
    )
    _run(
        "variogram",
        DATA / "sample_events.geojson",
        {"column": "pm25"},
        "B_variogram",
        time_column="timestamp",
    )
    _run(
        "idw",
        DATA / "sample_events.geojson",
        idw,
        "B_idw",
        time_column="timestamp",
    )
    _run(
        "kriging",
        DATA / "sample_events.geojson",
        krig,
        "B_kriging",
        time_column="timestamp",
    )
    print(f"Done. Outputs under {OUT}")


if __name__ == "__main__":
    main()
