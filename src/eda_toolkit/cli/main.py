"""Command-line interface for EDA-Toolkit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from eda_toolkit.eda.profile import build_profile, save_profile
from eda_toolkit.eda.runner import run_eda
from eda_toolkit.etl.registry import get_operator, list_operators
from eda_toolkit.etl.operator import OperatorContext
from eda_toolkit.io.loaders import load_dataset

UI_APP = Path(__file__).resolve().parent.parent / "ui" / "app.py"

app = typer.Typer(
    name="steda",
    help="Spatio-temporal EDA toolkit (EDA-Toolkit)",
    no_args_is_help=True,
)
operators_app = typer.Typer(help="ETL operator commands")
app.add_typer(operators_app, name="operators")
console = Console()


@app.command("profile")
def cmd_profile(
    input: Path = typer.Option(..., "--input", "-i", help="Input data path"),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    time_column: Optional[str] = typer.Option(None, "--time-column", "-t"),
) -> None:
    """Generate profile.json for a dataset."""
    ds = load_dataset(input, time_column=time_column)
    profile = build_profile(ds)
    path = save_profile(profile, output)
    console.print(f"[green]Profile saved:[/green] {path}")


@app.command("run")
def cmd_run(
    input: Path = typer.Option(..., "--input", "-i"),
    output: Path = typer.Option(..., "--output", "-o"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="EDA config JSON file"
    ),
    time_column: Optional[str] = typer.Option(None, "--time-column", "-t"),
) -> None:
    """Run full EDA pipeline."""
    ds = load_dataset(input, time_column=time_column)
    cfg: dict = {}
    if config and config.exists():
        cfg = json.loads(config.read_text(encoding="utf-8"))
    result = run_eda(ds, config=cfg, output_dir=output)
    console.print(f"[green]EDA complete.[/green] Output: {output}")
    if result.get("figures"):
        for name, p in result["figures"].items():
            console.print(f"  {name}: {p}")


@operators_app.command("list")
def cmd_operators_list() -> None:
    """List registered ETL operators."""
    table = Table("Name", "Version", "Description")
    for op in list_operators():
        table.add_row(op["name"], op["version"], op["description"])
    console.print(table)


@operators_app.command("execute")
def cmd_operators_execute(
    name: str = typer.Argument(..., help="Operator name, e.g. profile, eda_run"),
    input: Path = typer.Option(..., "--input", "-i"),
    output: Path = typer.Option(..., "--output", "-o"),
    params: Optional[Path] = typer.Option(
        None, "--params", "-p", help="Operator params JSON"
    ),
    time_column: Optional[str] = typer.Option(None, "--time-column", "-t"),
) -> None:
    """Execute a registered ETL operator."""
    op_cls = get_operator(name)
    param_dict: dict = {}
    if params and params.exists():
        param_dict = json.loads(params.read_text(encoding="utf-8"))
    ctx = OperatorContext(
        input_path=input,
        output_dir=output,
        params=param_dict,
        time_column=time_column,
    )
    result = op_cls().run(ctx)
    if result.success:
        console.print(f"[green]{result.message}[/green]")
    else:
        console.print(f"[red]{result.message}[/red]")
        raise typer.Exit(1)
    for k, v in result.artifacts.items():
        console.print(f"  {k}: {v}")


@app.command("ui")
def cmd_ui(
    port: int = typer.Option(8501, "--port", "-p", help="Streamlit server port"),
    host: str = typer.Option("localhost", "--host", help="Bind address"),
) -> None:
    """Launch the visual web UI (Streamlit)."""
    import subprocess
    import sys

    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        console.print(
            "[red]缺少 UI 依赖。请执行:[/red] pip install -e \".[ui]\""
        )
        raise typer.Exit(1) from exc

    console.print(f"[green]启动 Web 界面:[/green] http://{host}:{port}")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(UI_APP),
        "--server.port",
        str(port),
        "--server.address",
        host,
    ]
    raise typer.Exit(subprocess.call(cmd))


if __name__ == "__main__":
    app()
