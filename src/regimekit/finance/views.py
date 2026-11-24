"""Finance-side views of the neutral data contracts (E10-T04a).

`ObservationSet` is domain-neutral (`series_names`, `values`, `source`); finance code that wants the old
vocabulary reads it through `price_view`.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

from regimekit.data.contracts import ObservationSet
from regimekit.utils.errors import DataError

Interval = Literal["1h", "1d", "1wk", "1mo"]


class PriceView(NamedTuple):
    """The price vocabulary of an `ObservationSet` produced by a provider."""

    symbols: tuple[str, ...]
    times: np.ndarray
    prices: np.ndarray
    interval: Interval
    adjusted: bool
    provider: str


def price_view(obs: ObservationSet) -> PriceView:
    """Read `obs` with the finance names (`interval` and `adjusted` come from `obs.meta`).

    Raises:
        DataError: `obs` has no timestamps or no `interval` / `adjusted` metadata (not a provider result).
    """
    if obs.times is None:
        raise DataError("price observations need timestamps", path="times")
    if "interval" not in obs.meta or "adjusted" not in obs.meta:
        raise DataError("observations carry no `interval` / `adjusted` metadata", path="meta")
    return PriceView(
        symbols=obs.series_names,
        times=obs.times,
        prices=obs.values,
        interval=obs.meta["interval"],
        adjusted=bool(obs.meta["adjusted"]),
        provider=obs.source,
    )
