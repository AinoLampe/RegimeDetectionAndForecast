"""GBM emission: the finance alias of the core Gaussian family (E10-T06, E3-T01).

`y_t | i ~ N((mu_i(t) - sigma_i(t)**2 / 2) * dt_t, sigma_i(t)**2 * dt_t)` with the drift link
`mu_i(t) = mu_i + gamma_i . z^e_t` and the log-volatility link
`log sigma_i(t) = log sigma_i + delta_i . z^e_t`.

It is `GaussianFamily` with `scale_by_dt=True, ito_correction=True` fixed, keeping the Part I vocabulary:
parameters `emission.mu`, `emission.sigma`, config keys `mu`, `sigma`, covariate keys `drift`, `log_vol`.
v1/v2 finance configurations and 0.1.x/0.2.x bundles therefore load unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from regimekit.model.emission.gaussian import GaussianFamily
from regimekit.plugins import register


@register("emission", "gbm")
@dataclass(frozen=True)
class GBMFamily(GaussianFamily):
    """Per-state geometric Brownian motion emission (Gaussian log-returns)."""

    name: str = "gbm"
    scale_by_dt: bool = True
    ito_correction: bool = True

    mean_param: ClassVar[str] = "mu"
    scale_param: ClassVar[str] = "sigma"
    mean_cov_key: ClassVar[str] = "drift"
    scale_cov_key: ClassVar[str] = "log_vol"
    default_scale: ClassVar[float] = 0.1
    fixed_options: ClassVar[bool] = True

    @property
    def drift_idx(self) -> tuple[int, ...]:
        """Columns of the emission covariates that feed the drift link."""
        return self.mean_idx

    @property
    def logvol_idx(self) -> tuple[int, ...]:
        """Columns of the emission covariates that feed the log-volatility link."""
        return self.scale_idx
