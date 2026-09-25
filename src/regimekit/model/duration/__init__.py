"""Sojourn-duration families."""

from regimekit.model.duration.base import DurationFamily
from regimekit.model.duration.geometric import GeometricDuration
from regimekit.model.duration.negbin import NegBinDuration

__all__ = ["DurationFamily", "GeometricDuration", "NegBinDuration"]
