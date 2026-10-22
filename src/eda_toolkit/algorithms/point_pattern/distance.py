"""Second-order point pattern statistics (K / L / G) with optional CSR tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eda_toolkit.io.dataset import SpatioTemporalDataset


def _xy_coords(dataset: SpatioTemporalDataset) -> np.ndarray:
    gdf = dataset.to_projected()
    if gdf.empty:
        raise ValueError("Empty dataset")
    return np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])


def ripley_functions(
    dataset: SpatioTemporalDataset,
    *,
    functions: tuple[str, ...] = ("K", "L", "G"),
    support: int = 20,
    n_simulations: int = 99,
    run_csr_test: bool = True,
    random_state: int = 42,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Estimate Ripley's K, L, and/or nearest-neighbour G.

    When run_csr_test=True, uses Monte Carlo CSR envelopes (pointpats *_test).
    Simulations default to 99 for ETL-friendly runtime.
    """
    try:
        from pointpats.distance_statistics import g, g_test, k, k_test, l, l_test
    except ImportError as exc:
        raise ImportError(
            "Install spatial extras: pip install 'eda-toolkit[spatial]'"
        ) from exc

    coords = _xy_coords(dataset)
    if len(coords) < 3:
        raise ValueError("Point pattern statistics require at least 3 points")

    np.random.seed(random_state)
    result: dict[str, Any] = {
        "n_points": int(len(coords)),
        "support_bins": support,
        "n_simulations": n_simulations if run_csr_test else 0,
        "csr_test": run_csr_test,
        "curves": {},
    }

    fn_map = {
        "K": (k, k_test),
        "L": (l, l_test),
        "G": (g, g_test),
    }

    for name in functions:
        key = name.upper()
        if key not in fn_map:
            raise ValueError(f"Unknown function: {name}. Use K, L, or G.")
        est_fn, test_fn = fn_map[key]
        if run_csr_test:
            # n_jobs=1 for reproducible / Windows-friendly runs
            tres = test_fn(
                coords,
                support=support,
                n_simulations=n_simulations,
                keep_simulations=False,
                n_jobs=1,
            )
            curve = {
                "support": [float(x) for x in tres.support],
                "statistic": [float(x) for x in tres.statistic],
                "pvalue": [float(x) for x in tres.pvalue],
            }
            # Simple CSR summary: fraction of distances with p < 0.05
            pvals = np.asarray(tres.pvalue)
            curve["csr_significant_fraction"] = float(np.mean(pvals < 0.05))
            curve["inference"] = (
                "deviates_from_CSR"
                if curve["csr_significant_fraction"] > 0.2
                else "consistent_with_CSR"
            )
        else:
            support_vals, statistic = est_fn(coords, support=support)
            curve = {
                "support": [float(x) for x in support_vals],
                "statistic": [float(x) for x in statistic],
            }
        result["curves"][key] = curve

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, str] = {}
        for key, curve in result["curves"].items():
            df = pd.DataFrame(
                {
                    "support": curve["support"],
                    "statistic": curve["statistic"],
                    **(
                        {"pvalue": curve["pvalue"]}
                        if "pvalue" in curve
                        else {}
                    ),
                }
            )
            path = out / f"ripley_{key.lower()}.csv"
            df.to_csv(path, index=False)
            artifacts[f"ripley_{key.lower()}_csv"] = str(path)

        summary = {
            k: v
            for k, v in result.items()
            if k != "curves"
        }
        summary["curves"] = {
            k: {kk: vv for kk, vv in c.items() if kk not in ("support", "statistic", "pvalue")}
            | {
                "n_support": len(c["support"]),
                "statistic_max": max(c["statistic"]) if c["statistic"] else None,
            }
            for k, c in result["curves"].items()
        }
        summary_path = out / "point_pattern_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        artifacts["point_pattern_summary_json"] = str(summary_path)
        result["artifacts"] = artifacts

    return result
