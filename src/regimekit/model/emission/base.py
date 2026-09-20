"""`EmissionFamily` protocol stub (signatures only; bodies land in E3-T01/E3-T02).

Declares just enough structure for `ModelSpec` (E1-T05) to type-check under `mypy --strict`.
"""

from __future__ import annotations

from typing import Literal, Protocol

from regimekit.model.params import Params
from regimekit.utils.typing import Array

Support = Literal["real", "count", "binary", "categorical"]


class EmissionFamily(Protocol):
    """Structural type for emission families (GBM, Student-t, ...)."""

    support: Support
    scale_by_dt: bool

    def log_prob(self, y: Array, theta: Params, dt: Array) -> Array:
        """Per-step, per-regime emission log-density of `y` given constrained `theta`."""
        ...
