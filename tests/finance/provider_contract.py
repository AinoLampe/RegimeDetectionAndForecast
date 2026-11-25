"""Shared provider contract helper (E2-T01); not collected by pytest (no `test_` prefix).

Every adapter's test module calls `check_price_contract` / `check_series_contract` with an adapter
instance (backed by an offline fixture or injected downloader) and known symbols.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimekit.data.contracts import CovariateSeries, ObservationSet
from regimekit.finance.errors import ProviderError
from regimekit.finance.providers.base import DataProvider


def _check_times(times: np.ndarray) -> None:
    assert times.dtype == np.dtype("datetime64[ns]")
    assert np.all(np.diff(times.astype("int64")) > 0), "times must be strictly increasing (sorted, unique)"


def check_price_contract(
    provider: DataProvider,
    symbols: list[str],
    start: str,
    end: str,
    unknown: str = "NOPE",
    interval: str = "1d",
    adjusted_changes_prices: bool = True,
) -> None:
    """Assert the `fetch_prices` contract for any adapter.

    Set `adjusted_changes_prices=False` for providers with a single adjustment convention.
    """
    assert isinstance(provider.tos_note, str)
    assert provider.tos_note.strip()
    assert isinstance(provider.name, str)
    assert provider.name
    for adjusted in (False, True):
        obs = provider.fetch_prices(symbols, start, end, interval, adjusted)
        assert isinstance(obs, ObservationSet)
        _check_times(obs.times)
        assert obs.values.dtype == np.float64
        assert obs.values.shape == (len(obs.times), len(symbols))
        assert obs.series_names == tuple(symbols)
        assert obs.meta["interval"] == interval
        assert obs.meta["adjusted"] is adjusted
        assert obs.source == provider.name
        assert obs.snapshot_id == ""
        assert obs.times[0] >= np.datetime64(start, "ns")
        assert obs.times[-1] < np.datetime64(end, "ns") + np.timedelta64(1, "D")
    raw = provider.fetch_prices(symbols, start, end, interval, False)
    adj = provider.fetch_prices(symbols, start, end, interval, True)
    if adjusted_changes_prices:
        assert not np.array_equal(raw.values, adj.values, equal_nan=True), "adjusted flag must change prices"
    with pytest.raises(ProviderError):
        provider.fetch_prices([unknown], start, end, interval, False)


def check_window_inclusive(
    provider: DataProvider, symbol: str, first: str, last: str, interval: str = "1d"
) -> None:
    """`[first, last]` must be returned with both boundary timestamps present."""
    obs = provider.fetch_prices([symbol], first, last, interval, False)
    assert obs.times[0] == np.datetime64(first, "ns")
    assert obs.times[-1] == np.datetime64(last, "ns")


def check_series_contract(
    provider: DataProvider, series_id: str, start: str, end: str, unknown: str = "NOPE"
) -> None:
    """Assert the `fetch_series` contract for any macro adapter."""
    ser = provider.fetch_series(series_id, start, end)
    assert isinstance(ser, CovariateSeries)
    _check_times(ser.times)
    assert ser.values.dtype == np.float64
    assert ser.values.shape == ser.times.shape
    assert ser.source == provider.name
    assert ser.series_id == series_id
    assert ser.snapshot_id == ""
    assert ser.frequency
    with pytest.raises(ProviderError):
        provider.fetch_series(unknown, start, end)
