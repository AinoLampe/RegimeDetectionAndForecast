"""`stooq` daily-EOD adapter (fallback provider, EU coverage, E2-T03).

Uses stooq's CSV endpoint (no extra dependency). The API key comes from the environment only
(`STOOQ_API_KEY`) and is never logged or included in error messages. The network call is an injectable
`downloader`; the daily request quota is a persisted counter (`quota_dir/stooq_quota.json`).

Downloader contract: `downloader(stooq_symbol, start, end, api_key) -> DataFrame` indexed by date with a
`Close` column (stooq's CSV layout, `Date` may also be a column); an unknown symbol yields an empty frame.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from regimekit.data.contracts import CovariateSeries, ObservationSet
from regimekit.finance.errors import ProviderError
from regimekit.finance.providers.base import (
    PERIOD_FREQ,
    check_interval,
    clean_frame,
    clip_window,
    last_close_per_period,
    register,
    to_utc_ns,
    with_retries,
)
from regimekit.utils.logging import get_logger

Downloader = Callable[[str, str, str | None, str], pd.DataFrame]

API_KEY_ENV = "STOOQ_API_KEY"
STOOQ_SYMBOLS: dict[str, str] = {"^GDAXI": "^dax", "^GSPC": "^spx"}
"""Yahoo-style symbol -> stooq symbol; anything not listed is passed through unchanged."""
TOS_NOTE = (
    "Stooq data is for personal use under stooq.com's terms; an API key and a daily request quota apply. "
    "Do not redistribute it."
)

_log = get_logger("regimekit.finance.providers.stooq")


def to_stooq_symbol(symbol: str) -> str:
    """Map a Yahoo-style symbol to stooq's (`^GDAXI -> ^dax`); unknown symbols pass through."""
    return STOOQ_SYMBOLS.get(symbol, symbol)


def _download_stooq(
    symbol: str, start: str, end: str | None, api_key: str
) -> pd.DataFrame:  # pragma: no cover  (network; the URL carries the key, so never echo it)
    d1 = pd.Timestamp(start).strftime("%Y%m%d")
    d2 = (pd.Timestamp(end) if end else pd.Timestamp.now()).strftime("%Y%m%d")
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d&d1={d1}&d2={d2}&apikey={api_key}"
    try:
        frame = pd.read_csv(url, index_col="Date", parse_dates=True)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:  # message may contain the URL (and thus the key): report the type only
        raise ProviderError(f"stooq request failed ({type(exc).__name__})", path="data.symbols") from None
    return frame


@register
class StooqProvider:
    """Daily prices from stooq; `1wk`/`1mo` are resampled (last close). Intraday is not supported.

    Stooq closes are split-adjusted only, so `adjusted` is echoed but does not change the prices.

    Args:
        downloader: Replaces the network call (see module docstring).
        api_key: Overrides env `STOOQ_API_KEY`.
        max_calls_per_day: Client-side daily request budget (one call per symbol; retries are free).
        quota_dir: Directory of the persisted quota counter (default `~/.cache/regimekit`).
        today: Injectable current UTC date. env: Injectable environment mapping.
        max_attempts: Attempts per download. sleep: Retry backoff sleep.
    """

    name = "stooq"
    supports_intraday = False
    tos_note = TOS_NOTE

    def __init__(
        self,
        downloader: Downloader | None = None,
        api_key: str | None = None,
        max_calls_per_day: int = 100,
        quota_dir: Path | str | None = None,
        today: Callable[[], date] | None = None,
        env: Mapping[str, str] | None = None,
        max_attempts: int = 5,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        """Configure the adapter; no network access and no file access happen here."""
        self._downloader: Downloader = downloader if downloader is not None else _download_stooq
        self._api_key = api_key
        self._max_calls = max_calls_per_day
        self._quota_file = (
            Path(quota_dir) if quota_dir is not None else Path.home() / ".cache" / "regimekit"
        ) / "stooq_quota.json"
        self._today = today if today is not None else (lambda: pd.Timestamp.now(tz="UTC").date())
        self._env: Mapping[str, str] = env if env is not None else os.environ
        self._max_attempts = max_attempts
        self._sleep = sleep if sleep is not None else time.sleep

    def _key(self) -> str:
        key = self._api_key or self._env.get(API_KEY_ENV)
        if not key:
            raise ProviderError(
                f"stooq needs an API key: set the environment variable {API_KEY_ENV} "
                "(get a key at https://stooq.com, then e.g. `export STOOQ_API_KEY=...`)",
                path="data.provider",
            )
        return key

    def _calls_today(self) -> int:
        try:
            state = json.loads(self._quota_file.read_text(encoding="utf-8"))
            if state.get("date") == self._today().isoformat():
                return int(state["calls"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return 0

    def _charge(self, n_calls: int) -> None:
        used = self._calls_today()
        if used + n_calls > self._max_calls:
            raise ProviderError(
                f"stooq quota: {used} of {self._max_calls} daily calls used, {n_calls} more requested; "
                "retry tomorrow, raise `max_calls_per_day`, or use another provider",
                path="data.provider",
            )
        self._quota_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._quota_file.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps({"date": self._today().isoformat(), "calls": used + n_calls}), encoding="utf-8"
        )
        tmp.replace(self._quota_file)

    def _fetch_one(self, symbol: str, start: str, end: str | None, key: str, interval: str) -> pd.Series:
        stooq_symbol = to_stooq_symbol(symbol)
        raw = with_retries(
            lambda: self._downloader(stooq_symbol, start, end, key),
            max_attempts=self._max_attempts,
            sleep=self._sleep,
        )
        if "Date" in raw.columns:
            raw = raw.set_index("Date")
        if raw.empty or "Close" not in raw.columns or raw["Close"].dropna().empty:
            raise ProviderError(f"stooq returned no data for {symbol!r}", path="data.symbols")
        frame = raw[["Close"]].astype("float64")
        frame.index = pd.DatetimeIndex(to_utc_ns(pd.DatetimeIndex(frame.index)))
        close = clip_window(clean_frame(frame), start, end)["Close"].dropna()
        if close.empty:
            raise ProviderError(f"no rows for {symbol!r} in [{start}, {end}]", path="data.start")
        if interval in PERIOD_FREQ:
            close = last_close_per_period(close, interval)
        return close.rename(symbol)

    def fetch_prices(
        self, symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool
    ) -> ObservationSet:
        """Fetch daily closes per symbol (one quota call each), union-aligned on time."""
        check_interval(interval)
        if interval == "1h":
            raise ProviderError(
                "stooq provides daily data only; interval '1h' is not supported", path="data.interval"
            )
        if not symbols:
            raise ProviderError("no symbols requested", path="data.symbols")
        key = self._key()
        self._charge(len(symbols))
        _log.info("fetch", provider=self.name, tos=self.tos_note, symbols=symbols, interval=interval)
        joined = pd.concat(
            [self._fetch_one(s, start, end, key, interval) for s in symbols], axis=1, join="outer", sort=True
        )
        meta: dict[str, object] = {}
        if interval in PERIOD_FREQ:
            meta["resampled_from"] = "1d (last close of each period)"
        return ObservationSet(
            series_names=tuple(symbols),
            times=np.asarray(joined.index.values, dtype="datetime64[ns]"),
            values=joined.to_numpy(dtype="float64"),
            source=self.name,
            snapshot_id="",
            meta={"interval": interval, "adjusted": adjusted, **meta},
        )

    def fetch_series(self, series_id: str, start: str, end: str | None) -> CovariateSeries:
        """Not supported: stooq is used for prices only (use `fred`/`cboe` for macro series)."""
        raise ProviderError(
            f"provider 'stooq' serves prices only, not series {series_id!r}", path="data.covariates"
        )
