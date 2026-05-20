"""Operator registry for ETL / governance platform discovery."""

from __future__ import annotations

from typing import Type

from eda_toolkit.etl.operator import OperatorContext, OperatorResult, SpatioTemporalOperator
from eda_toolkit.eda.profile import build_profile, save_profile
from eda_toolkit.eda.runner import run_eda

_REGISTRY: dict[str, Type[SpatioTemporalOperator]] = {}


def register_operator(cls: Type[SpatioTemporalOperator]) -> Type[SpatioTemporalOperator]:
    _REGISTRY[cls.name] = cls
    return cls


def list_operators() -> list[dict[str, str]]:
    return [
        {"name": cls.name, "version": cls.version, "description": cls.description}
        for cls in _REGISTRY.values()
    ]


def get_operator(name: str) -> Type[SpatioTemporalOperator]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown operator: {name}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


@register_operator
class ProfileOperator(SpatioTemporalOperator):
    name = "profile"
    version = "0.1.0"
    description = "Build data profile (quality, CRS, time range, missing stats)"

    def run(self, ctx: OperatorContext) -> OperatorResult:
        ds = ctx.load_dataset()
        profile = build_profile(ds)
        path = save_profile(profile, ctx.output_dir)
        result = OperatorResult(
            success=True,
            diagnostics={"n_rows": len(ds.gdf)},
            artifacts={"profile_json": str(path)},
            message="Profile completed",
        )
        result.save_manifest(ctx.output_dir)
        return result


@register_operator
class EdaRunOperator(SpatioTemporalOperator):
    name = "eda_run"
    version = "0.1.0"
    description = "Full EDA: profile + univariate + multivariate + optional plots"

    def run(self, ctx: OperatorContext) -> OperatorResult:
        ds = ctx.load_dataset()
        config = ctx.params.get("eda_config", {})
        out = run_eda(ds, config=config, output_dir=ctx.output_dir)
        artifacts = {k: str(v) for k, v in out.get("figures", {}).items()}
        result = OperatorResult(
            success=True,
            diagnostics={"config": config},
            artifacts=artifacts,
            message="EDA run completed",
        )
        result.save_manifest(ctx.output_dir)
        return result
