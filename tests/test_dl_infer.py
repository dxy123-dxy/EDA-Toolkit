"""Tests for dl_infer registration, presets, and optional GPU smoke."""

from __future__ import annotations

from pathlib import Path

import pytest

from eda_toolkit.etl.operator import OperatorContext
from eda_toolkit.etl.registry import get_operator
from eda_toolkit.ml.presets import resolve_preset, repo_root

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "output/phase2/tensors/knowair_bthsa_16x16_2022-01-01_2023-12-31.meta.json"
CKPT = (
    ROOT
    / "third_party/UniST/src/experiments"
    / "Pretrain_Dataset_TrafficKnowAirLong_Task_short_FewRatio_1.0"
    / "model_save/model_best.pkl"
)


def test_dl_infer_registered():
    op = get_operator("dl_infer")
    assert op.name == "dl_infer"


def test_resolve_preset_bthsa():
    p = resolve_preset("bthsa_long", root=repo_root())
    assert p["dataset"] == "TrafficKnowAirLong"
    assert Path(p["checkpoint"]).name == "model_best.pkl"
    assert "meta_path" in p


def test_dl_infer_fails_without_checkpoint(tmp_path):
    op = get_operator("dl_infer")()
    ctx = OperatorContext(
        input_path=ROOT / "examples/data/sample_points.geojson",
        output_dir=tmp_path,
        params={
            "backend": "unist",
            "dataset": "TrafficKnowAirLong",
            "checkpoint": str(tmp_path / "missing.pkl"),
        },
    )
    result = op.run(ctx)
    assert result.success is False
    assert "failed" in result.message.lower() or "error" in result.diagnostics


@pytest.mark.skipif(not META.exists() or not CKPT.exists(), reason="Phase2 tensor/ckpt missing")
def test_dl_infer_smoke_bthsa(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for UniST smoke")

    op = get_operator("dl_infer")()
    ctx = OperatorContext(
        input_path=META,
        output_dir=tmp_path,
        params={
            "preset": "bthsa_long",
            "max_windows": 2,
            "batch_size": 2,
            "mask_ratio": 0.5,
            "seed": 42,
        },
    )
    result = op.run(ctx)
    assert result.success, result.message
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "operator_result.json").exists()
    assert "metrics" in result.diagnostics
