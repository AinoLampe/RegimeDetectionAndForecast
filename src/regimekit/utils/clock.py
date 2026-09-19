"""Clock-unit conversions (overview §4.3, CLAUDE.md §3.2).

`utils/clock.py` is the only module allowed a literal `252`/`365.25` outside `data/calendar.py`
(which imports the constants from here); no other module may branch on the clock's *name* either
(both restrictions live only here and in `data/calendar.py`).

Created by E1-T02a, which is the first ticket needing a days->steps conversion (to size
`model.d_max`'s default from `init_mean_sojourn_days`), ahead of E2-T06/E3-T04 in build order.
Those tickets add `future_grid`/`is_irregular` here rather than recreating this module.
"""

from __future__ import annotations

from typing import Literal

Clock = Literal["trading", "calendar", "steps"]

TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365.25


def days_to_steps(days: float, clock: Clock, nominal_dt: float, bars_per_day: int) -> float:
    """Approximate the number of `nominal_dt`-sized (years) steps that `days` clock-days span.

    A config-time heuristic used before any real dataset exists (`nominal_dt` is not yet a fitted
    value): `nominal_dt` is treated as one calendar/trading day's worth of years, so `bars_per_day`
    intraday bars each span `nominal_dt / bars_per_day` years. Under `clock="steps"`, `days` is
    already a step count and is returned unchanged.
    """
    if clock == "steps":
        return days
    days_per_year = TRADING_DAYS_PER_YEAR if clock == "trading" else CALENDAR_DAYS_PER_YEAR
    return (days / days_per_year) * bars_per_day / nominal_dt
