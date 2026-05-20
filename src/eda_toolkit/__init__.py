"""EDA-Toolkit: spatio-temporal exploratory analysis and spatial statistics."""

__version__ = "0.1.0"

from eda_toolkit.io.dataset import SpatioTemporalDataset
from eda_toolkit.eda.runner import run_eda

__all__ = ["SpatioTemporalDataset", "run_eda", "__version__"]
