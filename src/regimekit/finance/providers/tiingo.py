"""`tiingo` US end-of-day adapter (optional provider, E2-T09).

Uses Tiingo's REST endpoint `https://api.tiingo.com/tiingo/daily/<ticker>/prices` through the standard
library (no extra dependency, so the adapter is always registered). The API key comes from the environment
only (`TIINGO_API_KEY`); it is sent in the `Authorization` header, never in a URL, and never logged.
Free-tier limits are enforced client-side by a quota persisted in `quota_dir/tiingo_quota.json`.

Downloader contract: `downloader(ticker, start, end, api_key) -> list[dict] | None`, the decoded JSON rows
(`date`, `close`, `adjClose`, ...), or `None` when Tiingo does not know the ticker.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

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
    throttle,
    to_utc_ns,
    with_retries,
)
from regimekit.utils.logging import get_logger

Downloader = Callable[[str, str, str | None, str], list[dict[str, Any]] | None]

API_KEY_ENV = "TIINGO_API_KEY"
HOUR_S = 3600.0
DAY_S = 86400.0
TOS_NOTE = (
    "Tiingo data is licensed for the account holder's own use under tiingo.com's terms; the free tier has "
    "request and symbol limits. Do not redistribute it."
)

_log = get_logger("regimekit.finance.providers.tiingo")


@throttle(0.5)
def _download_tiingo(
    ticker: str, start: str, end: str | None, api_key: str
) -> list[dict[str, Any]] | None:  # pragma: no cover  (network)
    query = {"startDate": pd.Timestamp(start).strftime("%Y-%m-%d"), "format": "json"}
    if end is not None:
        query["endDate"] = pd.Timestamp(end).strftime("%Y-%m-%d")
    url = (
        f"https://api.tiingo.com/tiingo/daily/{urllib.parse.quote(ticker.lower())}/prices?"
        f"{urllib.parse.urlencode(query)}"
    )
    request = urllib.request.Request(
        url, headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            rows = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise ProviderError(f"tiingo request failed: HTTP {exc.code}", path="data.symbols") from None
    except Exception as exc:
        raise ProviderError(f"tiingo request failed ({type(exc).__name__})", path="data.symbols") from None
    return list(rows)


class TiingoQuota:
    """Free-tier limits, persisted across processes: rolling hour/day request counts, symbols per month.

    Args:
        path: JSON state file.
        clock: Wall-clock time in epoch seconds (injectable; wall time because the state is persisted).
        max_per_hour: Requests in any rolling hour.
        max_per_day: Requests in any rolling 24 h.
        max_symbols_per_month: Distinct symbols per UTC calendar month.
    """

    def __init__(
        self,
        path: Path,
        clock: Callable[[], float] = time.time,
        max_per_hour: int = 50,
        max_per_day: int = 1000,
        max_symbols_per_month: int = 500,
    ) -> None:
        """No file access happens here."""
        self.path = path
        self._clock = clock
        self.max_per_hour = max_per_hour
        self.max_per_day = max_per_day
        self.max_symbols_per_month = max_symbols_per_month

    def _load(self) -> tuple[list[float], dict[str, list[str]]]:
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
            return [float(t) for t in state["requests"]], {k: list(v) for k, v in state["symbols"].items()}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return [], {}

    def charge(self, symbols: list[str]) -> None:
        """Charge one request per symbol, or raise `ProviderError` (charging nothing) if a limit is hit."""
        now = self._clock()
        requests, months = self._load()
        requests = [t for t in requests if now - t < DAY_S]
        n = len(symbols)
        in_hour = sum(1 for t in requests if now - t < HOUR_S)
        month = pd.Timestamp(now, unit="s", tz="UTC").strftime("%Y-%m")
        seen = set(months.get(month, [])) | set(symbols)
        checks = (
            (in_hour + n > self.max_per_hour, f"{in_hour} of {self.max_per_hour} requests used this hour"),
            (
                len(requests) + n > self.max_per_day,
                f"{len(requests)} of {self.max_per_day} requests used today",
            ),
            (
                len(seen) > self.max_symbols_per_month,
                f"{len(seen)} distinct symbols this month exceed {self.max_symbols_per_month}",
            ),
        )
        for exceeded, detail in checks:
            if exceeded:
                raise ProviderError(
                    f"tiingo free-tier limit: {detail} ({n} more requested); wait, raise the limit, "
                    "or use another provider",
                    path="data.provider",
                )
        state = {"requests": [*requests, *([now] * n)], "symbols": {month: sorted(seen)}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(self.path)


@register
class TiingoProvider:
    """Clean US end-of-day prices from Tiingo; `1wk`/`1mo` are resampled (last close). No intraday.

    Args:
        downloader: Replaces the network call (see module docstring).
        api_key: Programmatic override of env `TIINGO_API_KEY` (never fill it from config).
        quota_dir: Directory of the persisted quota (default `~/.cache/regimekit`).
        clock: Wall clock for the quota, epoch seconds.
        max_per_hour: Free-tier requests in any rolling hour.
        max_per_day: Free-tier requests in any rolling 24 h.
        max_symbols_per_month: Free-tier distinct symbols per UTC calendar month.
        env: Injectable environment mapping.
        max_attempts: Attempts per download.
        sleep: Retry backoff sleep.
    """

    name = "tiingo"
    supports_intraday = False
    tos_note = TOS_NOTE

    def __init__(
        self,
        downloader: Downloader | None = None,
        api_key: str | None = None,
        quota_dir: Path | str | None = None,
        clock: Callable[[], float] | None = None,
        max_per_hour: int = 50,
        max_per_day: int = 1000,
        max_symbols_per_month: int = 500,
        env: Mapping[str, str] | None = None,
        max_attempts: int = 5,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        """Configure the adapter; no network access and no file access happen here."""
        self._downloader: Downloader = downloader if downloader is not None else _download_tiingo
        self._api_key = api_key
        root = Path(quota_dir) if quota_dir is not None else Path.home() / ".cache" / "regimekit"
        self.quota = TiingoQuota(
            root / "tiingo_quota.json",
            clock if clock is not None else time.time,
            max_per_hour,
            max_per_day,
            max_symbols_per_month,
        )
        self._env: Mapping[str, str] = env if env is not None else os.environ
        self._max_attempts = max_attempts
        self._sleep = sleep if sleep is not None else time.sleep

    def _key(self) -> str:
        key = self._api_key or self._env.get(API_KEY_ENV)
        if not key:
            raise ProviderError(
                f"tiingo needs an API key: set the environment variable {API_KEY_ENV} (create a free "
                "account at https://www.tiingo.com, then e.g. `export TIINGO_API_KEY=...`)",
                path="data.provider",
            )
        return key

    def _fetch_one(
        self, symbol: str, start: str, end: str | None, key: str, interval: str, adjusted: bool
    ) -> pd.Series:
        rows = with_retries(
            lambda: self._downloader(symbol, start, end, key),
            max_attempts=self._max_attempts,
            sleep=self._sleep,
        )
        if rows is None:
            raise ProviderError(f"tiingo does not know ticker {symbol!r}", path="data.symbols")
        field = "adjClose" if adjusted else "close"
        raw = pd.DataFrame(rows)
        if raw.empty or not {"date", field} <= set(raw.columns):
            raise ProviderError(f"tiingo returned no {field!r} data for {symbol!r}", path="data.symbols")
        frame = pd.DataFrame(
            {"close": pd.to_numeric(raw[field], errors="coerce").to_numpy(dtype="float64")},
            index=pd.DatetimeIndex(to_utc_ns(raw["date"].astype(str).to_numpy())),
        )
        close = clip_window(clean_frame(frame), start, end)["close"].dropna()
        if close.empty:
            raise ProviderError(f"no rows for {symbol!r} in [{start}, {end}]", path="data.start")
        if interval in PERIOD_FREQ:
            close = last_close_per_period(close, interval)
        return close.rename(symbol)

    def fetch_prices(
        self, symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool
    ) -> ObservationSet:
        """Fetch daily `adjClose` (or `close`) per symbol, one quota request each, union-aligned on time."""
        check_interval(interval)
        if interval == "1h":
            raise ProviderError(
                "tiingo end-of-day data only; interval '1h' is not supported", path="data.interval"
            )
        if not symbols:
            raise ProviderError("no symbols requested", path="data.symbols")
        key = self._key()
        self.quota.charge(symbols)
        _log.info("fetch", provider=self.name, tos=self.tos_note, symbols=symbols, interval=interval)
        joined = pd.concat(
            [self._fetch_one(s, start, end, key, interval, adjusted) for s in symbols],
            axis=1,
            join="outer",
            sort=True,
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
        """Not supported: tiingo is used for prices only (use `fred`/`cboe` for covariate series)."""
        raise ProviderError(
            f"provider 'tiingo' serves prices only, not series {series_id!r}", path="data.covariates"
        )
