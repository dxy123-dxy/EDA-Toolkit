"""Operator registry for ETL / governance platform discovery."""

from __future__ import annotations

from typing import Type

from eda_toolkit.eda.profile import build_profile, save_profile
from eda_toolkit.eda.runner import run_eda
from eda_toolkit.etl.operator import OperatorContext, OperatorResult, SpatioTemporalOperator

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


def _require_column(params: dict) -> str:
    column = params.get("column")
    if not column:
        raise ValueError("params.column is required")
    return str(column)


def _strip_arrays(payload: dict) -> dict:
    """Drop large numpy arrays from diagnostics / manifests."""
    skip = {"density", "grid_x", "grid_y", "zgrid", "variance_grid", "gridx", "gridy"}
    return {k: v for k, v in payload.items() if k not in skip and k != "artifacts"}


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
        config = ctx.params.get("eda_config", ctx.params)
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


@register_operator
class ArealAutocorrOperator(SpatioTemporalOperator):
    name = "areal_autocorr"
    version = "0.1.0"
    description = "Global Moran/Geary + local LISA for areal or point attributes"

    def run(self, ctx: OperatorContext) -> OperatorResult:
        from eda_toolkit.algorithms.areal import run_areal_analysis

        ds = ctx.load_dataset()
        column = _require_column(ctx.params)
        gtype = ds.geometry_type
        default_w = "knn" if gtype in ("Point", "MultiPoint") else "queen"
        weight_type = ctx.params.get("weight_type", default_w)
        out = run_areal_analysis(
            ds,
            column,
            weight_type=weight_type,
            k=int(ctx.params.get("k", 8)),
            include_geary=bool(ctx.params.get("include_geary", True)),
            include_lisa=bool(ctx.params.get("include_lisa", True)),
            p_threshold=float(ctx.params.get("p_threshold", 0.05)),
            output_dir=ctx.output_dir,
        )
        artifacts = {k: str(v) for k, v in out.get("artifacts", {}).items()}
        diagnostics = {
            "column": column,
            "weight_type": weight_type,
            "moran": out.get("moran"),
            "geary": out.get("geary"),
            "lisa_cluster_counts": (out.get("lisa") or {}).get("cluster_counts"),
        }
        result = OperatorResult(
            success=True,
            diagnostics=diagnostics,
            artifacts=artifacts,
            message="Areal autocorrelation completed",
        )
        result.save_manifest(ctx.output_dir)
        return result


@register_operator
class PointPatternOperator(SpatioTemporalOperator):
    name = "point_pattern"
    version = "0.1.0"
    description = "Point pattern KDE and/or Ripley K/L/G with CSR test"

    def run(self, ctx: OperatorContext) -> OperatorResult:
        from eda_toolkit.algorithms.point_pattern import kde_heatmap, ripley_functions

        ds = ctx.load_dataset()
        do_kde = bool(ctx.params.get("kde", True))
        do_ripley = bool(ctx.params.get("ripley", True))
        artifacts: dict[str, str] = {}
        diagnostics: dict = {}

        if do_kde:
            kde_dir = (
                ctx.output_dir / "kde"
                if ctx.params.get("separate_dirs", True)
                else ctx.output_dir
            )
            kde_out = kde_heatmap(
                ds,
                bandwidth=ctx.params.get("bandwidth"),
                grid_size=int(ctx.params.get("kde_grid_size", 40)),
                output_dir=kde_dir,
            )
            artifacts.update(kde_out.get("artifacts", {}))
            diagnostics["kde"] = _strip_arrays(kde_out)

        if do_ripley:
            funcs = ctx.params.get("functions", ["K", "L", "G"])
            if isinstance(funcs, str):
                funcs = [f.strip() for f in funcs.split(",")]
            ripley_out = ripley_functions(
                ds,
                functions=tuple(funcs),
                support=int(ctx.params.get("support", 15)),
                n_simulations=int(ctx.params.get("n_simulations", 99)),
                run_csr_test=bool(ctx.params.get("run_csr_test", True)),
                random_state=int(ctx.params.get("random_state", 42)),
                output_dir=ctx.output_dir / "ripley"
                if ctx.params.get("separate_dirs", True)
                else ctx.output_dir,
            )
            artifacts.update(ripley_out.get("artifacts", {}))
            diagnostics["ripley"] = {
                k: v for k, v in ripley_out.items() if k != "curves"
            }
            diagnostics["ripley_inference"] = {
                name: curve.get("inference")
                for name, curve in ripley_out.get("curves", {}).items()
            }

        result = OperatorResult(
            success=True,
            diagnostics=diagnostics,
            artifacts=artifacts,
            message="Point pattern analysis completed",
        )
        result.save_manifest(ctx.output_dir)
        return result


@register_operator
class IdwOperator(SpatioTemporalOperator):
    name = "idw"
    version = "0.1.0"
    description = "Inverse Distance Weighting spatial interpolation"

    def run(self, ctx: OperatorContext) -> OperatorResult:
        from eda_toolkit.algorithms.interpolation import idw_interpolate

        ds = ctx.load_dataset()
        column = _require_column(ctx.params)
        out = idw_interpolate(
            ds,
            column,
            power=float(ctx.params.get("power", 2.0)),
            grid_size=int(ctx.params.get("grid_size", 40)),
            output_dir=ctx.output_dir,
        )
        result = OperatorResult(
            success=True,
            diagnostics=_strip_arrays(out),
            artifacts={k: str(v) for k, v in out.get("artifacts", {}).items()},
            message="IDW interpolation completed",
        )
        result.save_manifest(ctx.output_dir)
        return result


@register_operator
class KrigingOperator(SpatioTemporalOperator):
    name = "kriging"
    version = "0.1.0"
    description = "Ordinary Kriging interpolation with optional LOO CV"

    def run(self, ctx: OperatorContext) -> OperatorResult:
        from eda_toolkit.algorithms.interpolation import ordinary_kriging

        ds = ctx.load_dataset()
        column = _require_column(ctx.params)
        out = ordinary_kriging(
            ds,
            column,
            variogram_model=str(ctx.params.get("variogram_model", "linear")),
            grid_size=int(ctx.params.get("grid_size", 30)),
            nlags=int(ctx.params.get("nlags", 6)),
            leave_one_out_cv=bool(ctx.params.get("leave_one_out_cv", True)),
            output_dir=ctx.output_dir,
        )
        result = OperatorResult(
            success=True,
            diagnostics=_strip_arrays(out),
            artifacts={k: str(v) for k, v in out.get("artifacts", {}).items()},
            message="Ordinary Kriging completed",
        )
        result.save_manifest(ctx.output_dir)
        return result


@register_operator
class VariogramOperator(SpatioTemporalOperator):
    name = "variogram"
    version = "0.1.0"
    description = "Empirical semivariogram for interpolation EDA"

    def run(self, ctx: OperatorContext) -> OperatorResult:
        import json
        from pathlib import Path

        from eda_toolkit.algorithms.interpolation import empirical_variogram

        ds = ctx.load_dataset()
        column = _require_column(ctx.params)
        out = empirical_variogram(
            ds,
            column,
            n_lags=int(ctx.params.get("n_lags", 12)),
            max_dist=ctx.params.get("max_dist"),
        )
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        path = Path(ctx.output_dir) / "variogram.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        result = OperatorResult(
            success=True,
            diagnostics=out,
            artifacts={"variogram_json": str(path)},
            message="Empirical variogram completed",
        )
        result.save_manifest(ctx.output_dir)
        return result


@register_operator
class DlInferOperator(SpatioTemporalOperator):
    name = "dl_infer"
    version = "0.1.0"
    description = (
        "Phase-2 UniST spatial-mask inference: predictions.geojson + metrics "
        "(params.preset or checkpoint+dataset; needs [ml] / .venv-ml)"
    )

    def run(self, ctx: OperatorContext) -> OperatorResult:
        from eda_toolkit.ml.infer import run_dl_infer

        try:
            out = run_dl_infer(
                output_dir=ctx.output_dir,
                params=ctx.params,
                input_path=ctx.input_path,
            )
        except Exception as exc:
            result = OperatorResult(
                success=False,
                diagnostics={"error": str(exc)},
                artifacts={},
                message=f"dl_infer failed: {exc}",
            )
            result.save_manifest(ctx.output_dir)
            return result

        result = OperatorResult(
            success=bool(out.get("success", True)),
            diagnostics=out.get("diagnostics", {}),
            artifacts={k: str(v) for k, v in out.get("artifacts", {}).items()},
            message=str(out.get("message", "dl_infer completed")),
        )
        result.save_manifest(ctx.output_dir)
        return result
