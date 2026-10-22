"""Spatial interpolation — IDW, Ordinary Kriging, empirical variogram."""

from eda_toolkit.algorithms.interpolation.idw import idw_interpolate
from eda_toolkit.algorithms.interpolation.kriging import (
    empirical_variogram,
    ordinary_kriging,
)

__all__ = ["idw_interpolate", "empirical_variogram", "ordinary_kriging"]
