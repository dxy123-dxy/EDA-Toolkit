"""Public DL inference API used by the ``dl_infer`` ETL operator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import Point

from eda_toolkit.ml.presets import resolve_preset


def _resolve_input_paths(input_path: Path | None, params: dict[str, Any]) -> dict[str, Any]:
    """Fill meta_path / tensor_path from input_path when not already set."""
    out = dict(params)
    if input_path is None:
        return out
    p = Path(input_path)
    suffix = "".join(p.suffixes).lower() if p.suffixes else p.suffix.lower()
    name = p.name.lower()

    # Explicit -i meta/npz overrides preset paths
    if name.endswith(".meta.json"):
        out["meta_path"] = str(p)
        twin = p.with_name(p.name.replace(".meta.json", ".npz"))
        if twin.exists():
            out["tensor_path"] = str(twin)
    elif p.suffix.lower() == ".npz":
        out["tensor_path"] = str(p)
        twin = p.with_name(p.stem + ".meta.json")
        if twin.exists():
            out["meta_path"] = str(twin)
    elif suffix.endswith(".geojson") or p.suffix.lower() in {".json", ".parquet", ".csv"}:
        # Point/areal input is accepted for ETL contract; UniST still needs meta/preset.
        pass
    return out


def merge_params(params: dict[str, Any], input_path: Path | None = None) -> dict[str, Any]:
    """Apply preset defaults, then user params, then input_path hints."""
    merged: dict[str, Any] = {}
    preset = params.get("preset")
    if preset:
        merged.update(resolve_preset(str(preset)))
        merged["preset"] = str(preset)
    for k, v in params.items():
        if k == "preset":
            continue
        merged[k] = v
    return _resolve_input_paths(input_path, merged)


def run_dl_infer(
    *,
    output_dir: str | Path,
    params: dict[str, Any],
    input_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load a Phase-2 checkpoint and write prediction GeoJSON + metrics.

    Parameters
    ----------
    output_dir :
        Artifact directory.
    params :
        Operator params. Either ``preset`` (``bthsa_long`` / ``yrd_prompt``) or
        explicit ``checkpoint`` + ``dataset`` (+ optional ``meta_path``).
    input_path :
        Optional ``*.meta.json`` / ``*.npz`` / GeoJSON. Meta/npz paths
        override when provided; GeoJSON alone is not enough for UniST (use preset).

    Returns
    -------
    dict with ``artifacts``, ``diagnostics`` (and internal ``features`` stripped).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged = merge_params(params, Path(input_path) if input_path else None)

    backend = str(merged.get("backend", "unist")).lower()
    if backend != "unist":
        raise ValueError(f"Unsupported dl_infer backend: {backend}")

    if "checkpoint" not in merged or "dataset" not in merged:
        raise ValueError(
            "dl_infer requires params.checkpoint + params.dataset "
            "(or params.preset=bthsa_long|yrd_prompt)"
        )

    from eda_toolkit.ml.unist_infer import run_unist_infer

    raw = run_unist_infer(merged)
    features = raw.pop("features", [])
    column = str(raw.get("column", "pm25"))

    artifacts: dict[str, str] = {}

    metrics_path = out / "metrics.json"
    metrics_payload = {
        "backend": raw["backend"],
        "dataset": raw["dataset"],
        "checkpoint": raw["checkpoint"],
        "mask_ratio": raw["mask_ratio"],
        "mask_strategy": raw["mask_strategy"],
        "seed": raw["seed"],
        "n_windows": raw["n_windows"],
        "metrics": raw["metrics"],
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    artifacts["metrics_json"] = str(metrics_path)

    if features:
        geoms = [Point(f["lon"], f["lat"]) for f in features]
        rows = [{k: v for k, v in f.items() if k not in ("lon", "lat")} for f in features]
        gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
        geo_path = out / "predictions.geojson"
        gdf.to_file(geo_path, driver="GeoJSON")
        artifacts["predictions_geojson"] = str(geo_path)

        csv_path = out / "predictions.csv"
        gdf.drop(columns="geometry").assign(lon=[f["lon"] for f in features], lat=[f["lat"] for f in features]).to_csv(
            csv_path, index=False
        )
        artifacts["predictions_csv"] = str(csv_path)

    diagnostics = {
        "backend": raw["backend"],
        "dataset": raw["dataset"],
        "checkpoint": raw["checkpoint"],
        "device": raw["device"],
        "preset": merged.get("preset"),
        "mask_ratio": raw["mask_ratio"],
        "mask_strategy": raw["mask_strategy"],
        "seed": raw["seed"],
        "n_windows": raw["n_windows"],
        "n_features": len(features),
        "column": column,
        "metrics": raw["metrics"],
        "meta_path": raw.get("meta_path"),
        "tensor_path": raw.get("tensor_path"),
    }
    diag_path = out / "diagnostics.json"
    diag_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    artifacts["diagnostics_json"] = str(diag_path)

    return {
        "success": True,
        "message": (
            f"dl_infer completed ({backend}): "
            f"MAE={raw['metrics'].get('mae')} RMSE={raw['metrics'].get('rmse')} "
            f"n={raw['metrics'].get('n')}"
        ),
        "artifacts": artifacts,
        "diagnostics": diagnostics,
    }
