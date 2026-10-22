"""Named presets for UniST checkpoints used in Phase-2 demos."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def repo_root() -> Path:
    """Walk up from this file to the package root (contains ``pyproject.toml``)."""
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return here.parents[3]


def resolve_preset(name: str, root: Path | None = None) -> dict[str, Any]:
    """Return absolute-path params for a known Phase-2 preset."""
    root = root or repo_root()
    unist = root / "third_party" / "UniST" / "src"
    tensors = root / "output" / "phase2" / "tensors"
    presets: dict[str, dict[str, Any]] = {
        "bthsa_long": {
            "backend": "unist",
            "dataset": "TrafficKnowAirLong",
            "checkpoint": str(
                unist
                / "experiments"
                / "Pretrain_Dataset_TrafficKnowAirLong_Task_short_FewRatio_1.0"
                / "model_save"
                / "model_best.pkl"
            ),
            "meta_path": str(tensors / "knowair_bthsa_16x16_2022-01-01_2023-12-31.meta.json"),
            "tensor_path": str(tensors / "knowair_bthsa_16x16_2022-01-01_2023-12-31.npz"),
            "prompt_ST": 0,
            "num_memory": 512,
            "column": "pm25",
        },
        "yrd_prompt": {
            "backend": "unist",
            "dataset": "TrafficKnowAirYRDLong",
            "checkpoint": str(
                unist
                / "experiments"
                / "Prompt_Dataset_TrafficKnowAirYRDLong_Task_short_FewRatio_0.5"
                / "model_save"
                / "model_best.pkl"
            ),
            "meta_path": str(tensors / "knowair_yrd_16x16_2022-01-01_2023-12-31.meta.json"),
            "tensor_path": str(tensors / "knowair_yrd_16x16_2022-01-01_2023-12-31.npz"),
            "prompt_ST": 1,
            "num_memory": 256,
            "column": "pm25",
        },
    }
    key = name.strip().lower()
    if key not in presets:
        raise ValueError(f"Unknown preset '{name}'. Available: {sorted(presets)}")
    return dict(presets[key])
