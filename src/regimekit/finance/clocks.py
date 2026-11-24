"""The `trading` clock (E10-T03): trading days, `Δ = 1 / (252 · bars_per_day)` years, weekdays only."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from regimekit.data.clocks import NS_PER_DAY, LastTime, check_horizon, check_step, continue_times, median_step
from regimekit.plugins import REGISTRY, provenance_of

TRADING_DAYS_PER_YEAR = 252


class TradingOpts(BaseModel):
    """`data.clock_options` of the `trading` clock."""

    model_config = ConfigDict(extra="forbid")

    bars_per_day: int = Field(default=1, ge=1)


class TradingClock:
    """Constant `Δ = 1 / (252 · bars_per_day)` years; the future grid skips weekends (no holiday calendar)."""

    name = "trading"
    unit: Literal["year", "step"] = "year"
    options_model = TradingOpts
    days_per_year: float | None = float(TRADING_DAYS_PER_YEAR)

    def dt(self, times: np.ndarray | None, n: int, opts: BaseModel) -> np.ndarray:
        """`n` equal spacings."""
        bars = _bars(opts)
        return np.full(n, 1.0 / (TRADING_DAYS_PER_YEAR * bars), dtype=np.float64)

    def future_grid(
        self,
        last_time: LastTime,
        h: int,
        opts: BaseModel,
        times: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Next weekday bars (spacing: the median of `times`, default one day / `bars_per_day`)."""
        check_horizon(h)
        bars = _bars(opts)
        step_ns = median_step(times, np.timedelta64(NS_PER_DAY // bars, "ns"))
        check_step(step_ns)
        out = continue_times(last_time, h, step_ns, skip_weekends=True)
        return out, np.full(h, 1.0 / (TRADING_DAYS_PER_YEAR * bars))

    def steps_per_day(self, opts: BaseModel) -> int:
        """`bars_per_day`."""
        return _bars(opts)


def _bars(opts: BaseModel) -> int:
    return int(opts.bars_per_day) if isinstance(opts, TradingOpts) else 1


REGISTRY.register("clock", TradingClock.name, TradingClock(), provenance=provenance_of(__name__))
