"""Provider protocol, registry and shared adapter helpers (FR-D01, ADR-0012).

Adapters return `snapshot_id=""`; the parquet cache (E2-T05) computes and fills the real snapshot id from
the raw bytes it stores.
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from regimekit.data.contracts import CovariateSeries, ObservationSet
from regimekit.utils.errors import ProviderError

VALID_INTERVALS: tuple[str, ...] = ("1h", "1d", "1wk", "1mo")


@runtime_checkable
class DataProvider(Protocol):
    """A market-data source. Price-only providers raise `ProviderError` from `fetch_series`."""

    name: str
    supports_intraday: bool
    tos_note: str

    def fetch_prices(
        self, symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool
    ) -> ObservationSet:
        """Fetch prices for `symbols` on `[start, end]` (both inclusive)."""
        ...

    def fetch_series(self, series_id: str, start: str, end: str | None) -> CovariateSeries:
        """Fetch one macro/covariate series on `[start, end]` (both inclusive)."""
        ...


_REGISTRY: dict[str, type] = {}


def register[P: type](provider_cls: P) -> P:
    """Class decorator registering a provider under its `name`; duplicates raise `ProviderError`."""
    name = getattr(provider_cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ProviderError(f"provider class {provider_cls.__name__} has no string `name`")
    if name in _REGISTRY:
        raise ProviderError(f"provider {name!r} is already registered", path=f"providers.{name}")
    _REGISTRY[name] = provider_cls
    return provider_cls


def get_provider(name: str, **kwargs: object) -> DataProvider:
    """Instantiate the provider registered as `name` with `kwargs`."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ProviderError(
            f"unknown provider {name!r} (registered: {known})", path="data.provider"
        ) from None
    provider: DataProvider = cls(**kwargs)
    return provider


def registered_providers() -> tuple[str, ...]:
    """Names of all registered providers, sorted."""
    return tuple(sorted(_REGISTRY))


def check_interval(interval: str) -> None:
    """Raise `ProviderError` unless `interval` is one of `1h, 1d, 1wk, 1mo`."""
    if interval not in VALID_INTERVALS:
        raise ProviderError(f"interval {interval!r} not in {VALID_INTERVALS}", path="data.interval")


def to_utc_ns(values: list[str] | np.ndarray | pd.Index | pd.Series) -> np.ndarray:
    """Convert timestamps to naive-UTC `datetime64[ns]` (naive values are taken to be UTC)."""
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(values, utc=True, format="ISO8601"))
    except (ValueError, TypeError) as exc:
        raise ProviderError(f"unparseable timestamps: {exc}") from exc
    return np.asarray(idx.tz_localize(None).astype("datetime64[ns]"))


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort by time and drop duplicate timestamps (keep last); expects a `DatetimeIndex` in ns."""
    frame = frame.sort_index(kind="stable")
    return frame[~frame.index.duplicated(keep="last")]


def window_bounds(start: str, end: str | None) -> tuple[np.datetime64, np.datetime64 | None]:
    """Inclusive `[start, end]` bounds; a date-only `end` covers that whole day.

    Returns:
        `(lo, hi)` where `lo` is inclusive and `hi` is *exclusive* (or `None` for open-ended).
    """
    lo = to_utc_ns([start])[0]
    if end is None:
        return lo, None
    hi_ts = pd.Timestamp(end)
    date_only = hi_ts == hi_ts.normalize() and len(end.strip()) <= 10
    hi = to_utc_ns([end])[0]
    hi = hi + np.timedelta64(1, "D") if date_only else hi + np.timedelta64(1, "ns")
    return lo, hi


def infer_frequency(times: np.ndarray) -> str:
    """Map the median sampling gap to `"d"`, `"w"`, `"m"` or `"q"`."""
    if len(times) < 2:
        return "d"
    gap_days = float(np.median(np.diff(times)) / np.timedelta64(1, "D"))
    if gap_days <= 4:
        return "d"
    if gap_days <= 10:
        return "w"
    return "m" if gap_days <= 45 else "q"


def with_retries[T](
    fn: Callable[[], T],
    max_attempts: int = 5,
    backoff: Callable[[int], float] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn` up to `max_attempts` times, retrying on `ProviderError`; re-raise the last one.

    Args:
        fn: Zero-argument callable performing the network request.
        max_attempts: Total number of calls (not extra retries).
        backoff: `attempt -> seconds` (0-based); default exponential `0.5 * 2**attempt`.
        sleep: Injectable sleep, so tests never wait.
    """
    if max_attempts < 1:
        raise ProviderError("max_attempts must be >= 1", path="max_attempts")
    delay = backoff if backoff is not None else (lambda attempt: 0.5 * 2.0**attempt)
    for attempt in range(max_attempts):
        try:
            return fn()
        except ProviderError:
            if attempt == max_attempts - 1:
                raise
            sleep(delay(attempt))
    raise AssertionError("unreachable")  # pragma: no cover


def throttle[T](
    min_interval_s: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator enforcing at least `min_interval_s` between successive calls of the wrapped function."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        lock = threading.Lock()
        last: list[float | None] = [None]

        @functools.wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> T:
            with lock:
                if last[0] is not None:
                    wait = min_interval_s - (clock() - last[0])
                    if wait > 0:
                        sleep(wait)
                last[0] = clock()
            return fn(*args, **kwargs)

        return wrapper

    return decorator
