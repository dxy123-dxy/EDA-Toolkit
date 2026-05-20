"""ETL operator contract for data governance pipelines."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eda_toolkit.io.dataset import SpatioTemporalDataset
from eda_toolkit.io.loaders import load_dataset


@dataclass
class OperatorContext:
    """Standard inputs for a pipeline step."""

    input_path: Path
    output_dir: Path
    params: dict[str, Any] = field(default_factory=dict)
    time_column: str | None = None

    def load_dataset(self) -> SpatioTemporalDataset:
        return load_dataset(
            self.input_path,
            time_column=self.time_column or self.params.get("time_column"),
            lon_column=self.params.get("lon_column", "longitude"),
            lat_column=self.params.get("lat_column", "latitude"),
            crs=self.params.get("crs", "EPSG:4326"),
        )


@dataclass
class OperatorResult:
    """Standard outputs for audit and downstream steps."""

    success: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    message: str = ""

    def save_manifest(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "operator_result.json"
        payload = {
            "success": self.success,
            "message": self.message,
            "diagnostics": self.diagnostics,
            "artifacts": self.artifacts,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


class SpatioTemporalOperator(ABC):
    """Base class for stateless ETL steps."""

    name: str = "base"
    version: str = "0.1.0"
    description: str = ""

    @abstractmethod
    def run(self, ctx: OperatorContext) -> OperatorResult:
        ...

    def validate_params(self, params: dict[str, Any]) -> None:
        """Override to validate params against JSON Schema."""
