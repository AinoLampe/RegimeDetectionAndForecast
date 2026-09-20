"""`DurationFamily` protocol stub (signatures only; bodies land in E3-T03/E3-T04).

Declares just enough structure for `ModelSpec` (E1-T05) to type-check under `mypy --strict`.
"""

from __future__ import annotations

from typing import Protocol

from regimekit.model.params import Params
from regimekit.utils.typing import Array


class DurationFamily(Protocol):
    """Structural type for sojourn-duration families (geometric, negbin, explicit, ...)."""

    supports_explicit: bool

    def logpmf(self, d: Array, theta: Params, dt: float, d_max: int) -> Array:
        """Log-pmf of sojourn length `d` (steps), truncated at `d_max` and renormalized."""
        ...

    def init_from_mean_sojourn(self, mean_steps: Array, nominal_dt: float) -> Params:
        """Initial `theta` reproducing `mean_steps` as this family's mean sojourn."""
        ...

    def advance_prob_t(self, theta: Params, dt: float) -> Array:
        """Per-step continuation probability (the geometric-equivalent hazard at `dt`)."""
        ...
