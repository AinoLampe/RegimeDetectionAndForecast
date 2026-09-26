"""`cboe` volatility-index adapter (daily history CSV files, E2-T04).

No API key. Each `series_id` names a CSV file whose URL comes from the `urls` mapping (adapter kwarg;
defaults in `DEFAULT_URLS`). The network call is an injectable `downloader(url) -> str` returning the CSV
text. CBOE files have a `DATE` column (`MM/DD/YYYY`) and either `OPEN, HIGH, LOW, CLOSE` (value = `CLOSE`)
or a single value column (e.g. `VVIX_History.csv` has `DATE, VVIX`).
"""

from __future__ import annotations

import io
import time
import urllib.request
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from regimekit.data.contracts import CovariateSeries, ObservationSet
from regimekit.finance.providers.base import (
    clean_frame,
    clip_window,
    register,
    throttle,
    to_utc_ns,
    with_retries,
)
from regimekit.utils.errors import ProviderError
from regimekit.utils.logging import get_logger

Downloader = Callable[[str], str]

_BASE = "https://cdn.cboe.com/api/global/us_indices/daily_prices"
DEFAULT_URLS: dict[str, str] = {
    "VIX_History.csv": f"{_BASE}/VIX_History.csv",
    "VVIX_History.csv": f"{_BASE}/VVIX_History.csv",
}
TOS_NOTE = (
    "CBOE index history is published by Cboe Global Markets for informational use under cboe.com's terms; "
    "do not redistribute it."
)

_log = get_logger("regimekit.finance.providers.cboe")


@throttle(0.5)
def _download_cboe(url: str) -> str:  # pragma: no cover  (network)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return str(response.read().decode("utf-8"))
    except Exception as exc:
        raise ProviderError(f"cboe download failed: {exc}", path="data.covariates") from exc


def _parse_dates(raw: pd.Series) -> np.ndarray:
    """CBOE `MM/DD/YYYY` dates (ISO dates also accepted) -> naive-UTC `datetime64[ns]` at 00:00."""
    try:
        parsed = pd.to_datetime(raw, format="%m/%d/%Y")
    except (ValueError, TypeError):
        return to_utc_ns(raw.astype(str).to_numpy())
    return to_utc_ns(pd.DatetimeIndex(parsed))


def parse_cboe_csv(text: str, series_id: str) -> pd.DataFrame:
    """Parse a CBOE history CSV into a sorted frame with a `value` column on a `DatetimeIndex`."""
    try:
        raw = pd.read_csv(io.StringIO(text))
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ProviderError(f"cboe file {series_id!r} is not a CSV: {exc}", path="data.covariates") from None
    raw.columns = [str(c).strip().upper() for c in raw.columns]
    value_cols = [c for c in raw.columns if c != "DATE"]
    if "DATE" not in raw.columns or not value_cols:
        raise ProviderError(f"cboe file {series_id!r} has no DATE/value columns", path="data.covariates")
    col = "CLOSE" if "CLOSE" in value_cols else value_cols[0] if len(value_cols) == 1 else None
    if col is None:
        raise ProviderError(
            f"cboe file {series_id!r}: cannot pick a value column from {value_cols}", path="data.covariates"
        )
    try:
        values = pd.to_numeric(raw[col], errors="raise").to_numpy(dtype="float64")
    except (ValueError, TypeError) as exc:
        raise ProviderError(f"cboe file {series_id!r} has non-numeric {col}: {exc}") from None
    frame = pd.DataFrame({"value": values}, index=pd.DatetimeIndex(_parse_dates(raw["DATE"])))
    return clean_frame(frame)


@register
class CboeProvider:
    """Daily volatility-index closes (VIX, VVIX, ...) from CBOE history files; not a price provider.

    Args:
        urls: `series_id -> URL`; replaces `DEFAULT_URLS` entirely when given.
        downloader: Replaces the network call (see module docstring).
        max_attempts: Attempts per download. sleep: Retry backoff sleep.
    """

    name = "cboe"
    supports_intraday = False
    tos_note = TOS_NOTE

    def __init__(
        self,
        urls: Mapping[str, str] | None = None,
        downloader: Downloader | None = None,
        max_attempts: int = 5,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        """Configure the adapter; no network access happens here."""
        self.urls: dict[str, str] = dict(urls) if urls is not None else dict(DEFAULT_URLS)
        self._downloader: Downloader = downloader if downloader is not None else _download_cboe
        self._max_attempts = max_attempts
        self._sleep = sleep if sleep is not None else time.sleep

    def fetch_series(self, series_id: str, start: str, end: str | None) -> CovariateSeries:
        """Fetch the daily close of `series_id` (a key of `urls`) on `[start, end]`."""
        try:
            url = self.urls[series_id]
        except KeyError:
            raise ProviderError(
                f"unknown cboe series {series_id!r} (configured: {', '.join(sorted(self.urls)) or '<none>'})",
                path="data.covariates",
            ) from None
        _log.info("fetch", provider=self.name, tos=self.tos_note, series=series_id)
        text = with_retries(lambda: self._downloader(url), max_attempts=self._max_attempts, sleep=self._sleep)
        frame = clip_window(parse_cboe_csv(text, series_id), start, end)
        if frame.empty:
            raise ProviderError(f"no rows for {series_id!r} in [{start}, {end}]", path="data.start")
        return CovariateSeries(
            name=series_id,
            times=np.asarray(frame.index.values, dtype="datetime64[ns]"),
            values=frame["value"].to_numpy(dtype="float64"),
            provider=self.name,
            series_id=series_id,
            snapshot_id="",
            frequency="d",
        )

    def fetch_prices(
        self, symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool
    ) -> ObservationSet:
        """Not supported: CBOE history files are used as covariate series, not prices."""
        raise ProviderError(
            f"provider 'cboe' serves covariate series only, not prices for {symbols!r}", path="data.provider"
        )
