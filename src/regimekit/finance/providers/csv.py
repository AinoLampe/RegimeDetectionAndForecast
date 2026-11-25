"""Local CSV provider: reference implementation of `DataProvider` (E2-T01)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regimekit.data.contracts import CovariateSeries, ObservationSet
from regimekit.finance.errors import ProviderError
from regimekit.finance.providers.base import (
    check_interval,
    clean_frame,
    clip_window,
    infer_frequency,
    register,
    to_utc_ns,
)


def _read(path: Path, required: tuple[str, ...], what: str) -> pd.DataFrame:
    if not path.is_file():
        raise ProviderError(f"no CSV file for {what!r} at {path}", path=what)
    try:
        raw = pd.read_csv(path, na_values=[".", ""], keep_default_na=True)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ProviderError(f"cannot parse {path}: {exc}", path=what) from exc
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ProviderError(f"{path.name} is missing columns {missing}", path=what)
    raw.index = pd.DatetimeIndex(to_utc_ns(raw["time"]))
    return clean_frame(raw.drop(columns="time"))


@register
class CsvProvider:
    """Reads `<root>/<symbol>.csv` and `<root>/<series_id>.csv` (columns: see below).

    Price files: `time, [open, high, low,] close[, adj_close, volume]`; series files: `time, value`.
    `time` values without a timezone are taken to be UTC. `"."` and empty cells become NaN. The
    requested `interval` is echoed, not resampled. `adj_close` is required only when `adjusted=True`.
    """

    name = "csv"
    supports_intraday = True
    tos_note = "Local files supplied by the user; their redistribution terms are the user's responsibility."

    def __init__(self, root: Path | str) -> None:
        """Store the directory containing the CSV files."""
        self.root = Path(root)

    def fetch_prices(
        self, symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool
    ) -> ObservationSet:
        """Read one CSV per symbol and outer-join them on time (missing cells are NaN)."""
        check_interval(interval)
        if not symbols:
            raise ProviderError("no symbols requested", path="data.symbols")
        column = "adj_close" if adjusted else "close"
        columns = []
        for sym in symbols:
            frame = _read(self.root / f"{sym}.csv", ("time", column), sym)
            columns.append(clip_window(frame, start, end)[column].astype("float64").rename(sym))
        joined = pd.concat(columns, axis=1, join="outer", sort=True)
        return ObservationSet(
            series_names=tuple(symbols),
            times=np.asarray(joined.index.values, dtype="datetime64[ns]"),
            values=joined.to_numpy(dtype="float64"),
            source=self.name,
            snapshot_id="",
            meta={"interval": interval, "adjusted": adjusted},
        )

    def fetch_series(self, series_id: str, start: str, end: str | None) -> CovariateSeries:
        """Read `<root>/<series_id>.csv` with columns `time, value`."""
        frame = clip_window(_read(self.root / f"{series_id}.csv", ("time", "value"), series_id), start, end)
        times = np.asarray(frame.index.values, dtype="datetime64[ns]")
        return CovariateSeries(
            name=series_id,
            times=times,
            values=frame["value"].to_numpy(dtype="float64"),
            source=self.name,
            series_id=series_id,
            snapshot_id="",
            frequency=infer_frequency(times),
        )
