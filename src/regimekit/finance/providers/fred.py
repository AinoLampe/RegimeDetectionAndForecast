"""`fred` macro/covariate adapter (FRED, E2-T04).

The API key comes from the environment only (`FRED_API_KEY`) and is never logged or included in error
messages. Network access goes through an injectable `FredClient`; the default client wraps `fredapi`
(extra `providers`, imported lazily). Every HTTP request (retries included) passes a shared in-process
`RateLimiter` of 120 requests per rolling minute.

Client contract:
- `series_info(series_id, api_key) -> Mapping | None`: FRED series metadata (at least `frequency_short`),
  `None` when the series does not exist.
- `observations(series_id, start, end, api_key) -> DataFrame` with columns `date` and `value`; values may be
  FRED's raw strings, where `"."` marks a missing observation.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np
import pandas as pd

from regimekit.data.contracts import CovariateSeries, ObservationSet
from regimekit.finance.providers.base import (
    RateLimiter,
    clean_frame,
    clip_window,
    register,
    to_utc_ns,
    with_retries,
)
from regimekit.utils.errors import ProviderError
from regimekit.utils.logging import get_logger

API_KEY_ENV = "FRED_API_KEY"
MAX_REQUESTS_PER_MINUTE = 120
FREQUENCIES: dict[str, str] = {"D": "d", "W": "w", "M": "m", "Q": "q"}
"""FRED `frequency_short` -> `CovariateSeries.frequency`; other frequencies are not supported."""
TOS_NOTE = (
    "FRED data is provided by the Federal Reserve Bank of St. Louis under its terms of use; some series "
    "carry third-party copyrights. Do not redistribute it."
)

_log = get_logger("regimekit.finance.providers.fred")
_SHARED_LIMITER = RateLimiter(MAX_REQUESTS_PER_MINUTE, 60.0)


def _require_extra() -> None:
    """Fail fast (outside the retry loop) when the `providers` extra is missing."""
    if importlib.util.find_spec("fredapi") is None:
        raise ProviderError(
            "fredapi is not installed; install the extra with `uv sync --extra providers`",
            path="data.covariates",
        )


class FredClient(Protocol):
    """Network access to FRED (see the module docstring for the contract)."""

    def series_info(self, series_id: str, api_key: str) -> Mapping[str, object] | None:
        """Series metadata, or `None` if the series does not exist."""
        ...

    def observations(self, series_id: str, start: str, end: str | None, api_key: str) -> pd.DataFrame:
        """Observations as a `date`, `value` frame."""
        ...


class _FredApiClient:  # pragma: no cover  (network)
    """`fredapi`-backed client; errors report the exception type only (the URL carries the key)."""

    def series_info(self, series_id: str, api_key: str) -> Mapping[str, object] | None:
        fred: Any = importlib.import_module("fredapi").Fred(api_key=api_key)
        try:
            info = fred.get_series_info(series_id)
        except ValueError as exc:  # fredapi's ValueError carries FRED's own error text, never the URL/key
            if "does not exist" in str(exc):
                return None
            raise ProviderError(f"FRED request failed: {exc}", path="data.covariates") from None
        except Exception as exc:
            raise ProviderError(
                f"FRED request failed ({type(exc).__name__})", path="data.covariates"
            ) from None
        return dict(info)

    def observations(self, series_id: str, start: str, end: str | None, api_key: str) -> pd.DataFrame:
        fred: Any = importlib.import_module("fredapi").Fred(api_key=api_key)
        try:
            ser = fred.get_series(series_id, observation_start=start, observation_end=end)
        except Exception as exc:
            raise ProviderError(
                f"FRED request failed ({type(exc).__name__})", path="data.covariates"
            ) from None
        return pd.DataFrame({"date": pd.DatetimeIndex(ser.index).strftime("%Y-%m-%d"), "value": ser.values})


def parse_values(raw: pd.Series, series_id: str) -> np.ndarray:
    """FRED values to float64: `"."` (and empty/None) -> NaN; anything else unparseable -> `ProviderError`."""
    text = raw.astype("object").where(raw.notna(), None)
    missing = text.map(lambda v: v is None or (isinstance(v, str) and v.strip() in {".", ""}))
    try:
        values = pd.to_numeric(text.where(~missing, np.nan), errors="raise")
    except (ValueError, TypeError) as exc:
        raise ProviderError(f"FRED series {series_id!r} has non-numeric values: {exc}") from None
    return np.asarray(values, dtype="float64")


@register
class FredProvider:
    """Macro and volatility series (`VIXCLS`, `T10Y2Y`, ...) from FRED; not a price provider.

    Args:
        client: Replaces the network client (see module docstring).
        api_key: Overrides env `FRED_API_KEY`.
        limiter: Request limiter; default is one process-wide 120-requests-per-minute window.
        env: Injectable environment mapping.
        max_attempts: Attempts per request. sleep: Retry backoff sleep.
    """

    name = "fred"
    supports_intraday = False
    tos_note = TOS_NOTE

    def __init__(
        self,
        client: FredClient | None = None,
        api_key: str | None = None,
        limiter: RateLimiter | None = None,
        env: Mapping[str, str] | None = None,
        max_attempts: int = 5,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        """Configure the adapter; no network access happens here."""
        self._client: FredClient = client if client is not None else _FredApiClient()
        self._api_key = api_key
        self._limiter = limiter if limiter is not None else _SHARED_LIMITER
        self._env: Mapping[str, str] = env if env is not None else os.environ
        self._max_attempts = max_attempts
        self._sleep = sleep if sleep is not None else time.sleep

    def _key(self) -> str:
        key = self._api_key or self._env.get(API_KEY_ENV)
        if not key:
            raise ProviderError(
                f"fred needs an API key: set the environment variable {API_KEY_ENV} (request a free key at "
                "https://fredaccount.stlouisfed.org/apikeys, then e.g. `export FRED_API_KEY=...`)",
                path="data.covariates",
            )
        return key

    def _request[T](self, fn: Callable[[], T]) -> T:
        def limited() -> T:
            self._limiter.acquire()
            return fn()

        return with_retries(limited, max_attempts=self._max_attempts, sleep=self._sleep)

    def fetch_series(self, series_id: str, start: str, end: str | None) -> CovariateSeries:
        """Fetch one FRED series on `[start, end]`; `"."` becomes NaN; times are UTC dates at 00:00."""
        key = self._key()
        if isinstance(self._client, _FredApiClient):
            _require_extra()
        _log.info("fetch", provider=self.name, tos=self.tos_note, series=series_id)
        info = self._request(lambda: self._client.series_info(series_id, key))
        if info is None:
            raise ProviderError(f"FRED series {series_id!r} does not exist", path="data.covariates")
        freq_short = str(info.get("frequency_short", "")).strip().upper()
        if freq_short not in FREQUENCIES:
            raise ProviderError(
                f"FRED series {series_id!r} has frequency {freq_short!r}; supported: {sorted(FREQUENCIES)}",
                path="data.covariates",
            )
        raw = self._request(lambda: self._client.observations(series_id, start, end, key))
        if raw.empty or not {"date", "value"} <= set(raw.columns):
            raise ProviderError(f"FRED returned no observations for {series_id!r}", path="data.covariates")
        frame = pd.DataFrame(
            {"value": parse_values(raw["value"].reset_index(drop=True), series_id)},
            index=pd.DatetimeIndex(to_utc_ns(raw["date"].astype(str).to_numpy())),
        )
        frame = clip_window(clean_frame(frame), start, end)
        if frame.empty:
            raise ProviderError(f"no rows for {series_id!r} in [{start}, {end}]", path="data.start")
        return CovariateSeries(
            name=series_id,
            times=np.asarray(frame.index.values, dtype="datetime64[ns]"),
            values=frame["value"].to_numpy(dtype="float64"),
            source=self.name,
            series_id=series_id,
            snapshot_id="",
            frequency=FREQUENCIES[freq_short],
        )

    def fetch_prices(
        self, symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool
    ) -> ObservationSet:
        """Not supported: FRED serves covariate series, not prices."""
        raise ProviderError(
            f"provider 'fred' serves covariate series only, not prices for {symbols!r}", path="data.provider"
        )
