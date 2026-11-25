"""Finance observation transform `log_returns` (E10-T05), moved from `regimekit.data.transforms`."""

from __future__ import annotations

import numpy as np

from regimekit.plugins import REGISTRY, provenance_of


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Log-returns `y_t = log(P_t / P_{t-1})` per series.

    Args:
        prices: `(T+1, M)` prices on the master grid.

    Returns:
        `(T, M)` float64 log-returns. Non-positive prices give `±inf`/NaN; the caller masks them.
    """
    p = np.asarray(prices, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(p[1:]) - np.log(p[:-1])


class LogReturnsTransform:
    """`y_t = log(P_t / P_{t-1})`; the first row is dropped."""

    name = "log_returns"

    def __call__(self, values: np.ndarray, times: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
        """Log-returns of the price columns and the shortened times."""
        return log_returns(values), None if times is None else times[1:]


REGISTRY.register(
    "transform", LogReturnsTransform.name, LogReturnsTransform(), provenance=provenance_of(__name__)
)
