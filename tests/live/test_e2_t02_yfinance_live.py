"""E2-T02 AC3: live fetch of the last 30 days of `^GDAXI` daily bars (`-m live`, needs network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("yfinance")
pytest.importorskip("curl_cffi")

from regimekit.finance.providers import YFinanceProvider

pytestmark = pytest.mark.live


def test_e2_t02_live_gdaxi_last_30_days_daily() -> None:
    """The real provider returns a valid, sorted, float64 daily `^GDAXI` series."""
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    start = end - pd.Timedelta(days=30)
    provider = YFinanceProvider()
    obs = provider.fetch_prices(["^GDAXI"], str(start.date()), str(end.date()), "1d", True)
    assert len(obs.times) >= 10
    assert np.isfinite(obs.prices).all()
    assert (obs.prices > 0).all()
    assert obs.times.dtype == np.dtype("datetime64[ns]")
    assert np.all(np.diff(obs.times.astype("int64")) > 0)
    assert obs.prices.dtype == np.float64
    assert obs.snapshot_id == ""
