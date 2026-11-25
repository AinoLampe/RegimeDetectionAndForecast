"""`yfinance` price adapter (default provider, E2-T02).

The network call is isolated in an injectable `downloader`, so tests replay recorded frames and never touch
the network. The real downloader imports `yfinance` and `curl_cffi` lazily (extra `providers`).

Downloader contract: `downloader(symbols, start, end, interval, adjusted) -> DataFrame` indexed by bar time
(naive values are UTC) with one *Close* column per symbol found. `end` is exclusive, as in yfinance.
"""

from __future__ import annotations

import importlib
import importlib.util
import time
import warnings
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd

from regimekit.data.contracts import CovariateSeries, ObservationSet
from regimekit.finance.errors import ProviderError
from regimekit.finance.providers.base import (
    check_interval,
    clean_frame,
    clip_window,
    register,
    throttle,
    to_utc_ns,
    window_bounds,
    with_retries,
)
from regimekit.utils.logging import get_logger

Downloader = Callable[[list[str], str, str | None, str, bool], pd.DataFrame]

INTRADAY_CAP_DAYS: dict[str, int] = {"1h": 730}
TOS_NOTE = (
    "Yahoo! data via yfinance is for personal, non-commercial use under Yahoo's terms of service; "
    "do not redistribute it."
)

_log = get_logger("regimekit.finance.providers.yfinance")


def extract_close(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Reduce a raw `yfinance.download` frame to one Close column per symbol.

    Handles both the multi-index layout `(Price, Ticker)` and the flat single-ticker layout.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame(index=raw.index)
        close = cast(pd.DataFrame, raw["Close"])
        return close.loc[:, [s for s in symbols if s in close.columns]]
    if "Close" not in raw.columns or len(symbols) != 1:
        return pd.DataFrame(index=raw.index)
    return raw[["Close"]].rename(columns={"Close": symbols[0]})


def _require_extra() -> None:
    """Fail fast (outside the retry loop) when the `providers` extra is missing."""
    if importlib.util.find_spec("yfinance") is None or importlib.util.find_spec("curl_cffi") is None:
        raise ProviderError(
            "yfinance/curl_cffi are not installed; install the extra with `uv sync --extra providers`",
            path="data.provider",
        )


def _download_yfinance(
    symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool
) -> pd.DataFrame:  # pragma: no cover  (network; exercised by tests/live)
    yf = importlib.import_module("yfinance")
    curl_requests = importlib.import_module("curl_cffi.requests")
    try:
        session = curl_requests.Session(impersonate="chrome")
        raw = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=adjusted,
            progress=False,
            threads=False,
            session=session,
        )
    except Exception as exc:  # network/parse failures of a third-party client are retryable
        raise ProviderError(f"yfinance download failed: {exc}", path="data.symbols") from exc
    return extract_close(raw, symbols)


@register
class YFinanceProvider:
    """Adjusted (or raw) prices for indices, ETFs and FX from Yahoo! Finance.

    Args:
        downloader: Replaces the network call (see module docstring); default uses `yfinance`.
        min_interval_s: Minimum spacing between download calls.
        max_attempts: Attempts per download (`with_retries`).
        today: Injectable "now" (UTC date) for the intraday window cap.
        clock: Monotonic clock for throttling. sleep: Sleep function for throttling and retry backoff.
    """

    name = "yfinance"
    supports_intraday = True
    tos_note = TOS_NOTE

    def __init__(
        self,
        downloader: Downloader | None = None,
        min_interval_s: float = 0.5,
        max_attempts: int = 5,
        today: Callable[[], pd.Timestamp] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Configure the adapter; no network access happens here."""
        self._downloader: Downloader = downloader if downloader is not None else _download_yfinance
        self._max_attempts = max_attempts
        self._today = today if today is not None else (lambda: pd.Timestamp.now(tz="UTC").tz_localize(None))
        self._sleep = sleep
        self._call = throttle(min_interval_s, clock=clock, sleep=sleep)(self._download_once)

    def _download_once(
        self, symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool
    ) -> pd.DataFrame:
        return self._downloader(symbols, start, end, interval, adjusted)

    def _cap_start(self, start: str, interval: str) -> str:
        cap_days = INTRADAY_CAP_DAYS.get(interval)
        if cap_days is None:
            return start
        earliest = self._today().normalize() - pd.Timedelta(days=cap_days)
        lo = pd.Timestamp(to_utc_ns([start])[0])
        if lo >= earliest:
            return start
        msg = f"yfinance serves <= {cap_days} days of {interval!r} data; clipping start to {earliest.date()}"
        warnings.warn(msg, UserWarning, stacklevel=3)
        _log.warning(
            "intraday_window_clipped", provider=self.name, interval=interval, start=str(earliest.date())
        )
        return str(earliest.date())

    def fetch_prices(
        self, symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool
    ) -> ObservationSet:
        """Download `symbols`, union-align on time (NaN where missing) and clip to `[start, end]`."""
        check_interval(interval)
        if not symbols:
            raise ProviderError("no symbols requested", path="data.symbols")
        _log.info("fetch", provider=self.name, tos=self.tos_note, symbols=symbols, interval=interval)
        start = self._cap_start(start, interval)
        _, hi = window_bounds(start, end)
        if hi is not None and pd.Timestamp(hi) <= pd.Timestamp(to_utc_ns([start])[0]):
            raise ProviderError(
                f"window ends before the {INTRADAY_CAP_DAYS.get(interval)}-day {interval!r} limit",
                path="data.end",
            )
        end_exclusive = None if hi is None else str(pd.Timestamp(hi))
        if self._downloader is _download_yfinance:
            _require_extra()
        raw = with_retries(
            lambda: self._call(list(symbols), start, end_exclusive, interval, adjusted),
            max_attempts=self._max_attempts,
            sleep=self._sleep,
        )
        missing = [s for s in symbols if s not in raw.columns or raw[s].dropna().empty]
        if missing:
            raise ProviderError(f"yfinance returned no data for {missing}", path="data.symbols")
        frame = raw.loc[:, list(symbols)].astype("float64")
        frame.index = pd.DatetimeIndex(to_utc_ns(pd.DatetimeIndex(frame.index)))
        frame = clip_window(clean_frame(frame), start, end)
        if frame.empty:
            raise ProviderError(f"no rows for {symbols} in [{start}, {end}]", path="data.start")
        return ObservationSet(
            series_names=tuple(symbols),
            times=np.asarray(frame.index.values, dtype="datetime64[ns]"),
            values=frame.to_numpy(dtype="float64"),
            source=self.name,
            snapshot_id="",
            meta={"interval": interval, "adjusted": adjusted},
        )

    def fetch_series(self, series_id: str, start: str, end: str | None) -> CovariateSeries:
        """Not supported: yfinance is a price provider (use `fred`/`cboe` for macro series)."""
        raise ProviderError(
            f"provider 'yfinance' serves prices only, not series {series_id!r}", path="data.covariates"
        )
