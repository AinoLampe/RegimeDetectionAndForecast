"""E10-T03: the `trading` clock of the finance package (Appendix E numbers, holiday gap)."""

from __future__ import annotations

import numpy as np
import pytest

from regimekit.data.calendar import build_grid
from regimekit.data.contracts import ObservationSet
from regimekit.finance.clocks import TRADING_DAYS_PER_YEAR, TradingClock, TradingOpts

pytestmark = pytest.mark.finance


def test_e10_t03_trading_constant_and_unit() -> None:
    """The only literal 252 lives in the finance clock."""
    assert TRADING_DAYS_PER_YEAR == 252
    assert TradingClock.unit == "year"


def test_e10_t03_trading_clock_is_constant_across_a_holiday_gap() -> None:
    """E2-T06: a holiday gap in the timestamps does not change `Δ` under the trading clock."""
    times = np.array(["2024-12-23", "2024-12-24", "2024-12-27"], dtype="datetime64[ns]")
    obs = ObservationSet(("X",), times, np.array([[1.0], [2.0], [3.0]]), "t", "s")
    grid = build_grid(obs, "drop", clock="trading")
    assert grid.dt.tolist() == [1 / 252, 1 / 252]
    assert grid.irregular is False


def test_e10_t03_appendix_e_per_step_probabilities() -> None:
    """Appendix E: mean sojourns of 126 and 42 steps give exit rates 0.0079 / 0.0235."""
    dt = float(TradingClock().dt(None, 1, TradingOpts())[0])
    calm = 1 - np.exp(-dt * 252 / 126.0)
    turb = 1 - np.exp(-dt * 252 / 42.0)
    assert round(calm, 4) == 0.0079
    assert round(turb, 4) == 0.0235
