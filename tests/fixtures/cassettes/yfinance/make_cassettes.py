"""Regenerate the synthetic `^GDAXI` cassettes (deterministic; no provider data is stored).

Run from the repository root: `uv run python tests/fixtures/cassettes/yfinance/make_cassettes.py`.
Each file is the *processed* downloader output: a DataFrame with one Close column named `^GDAXI`.
The daily index is tz-naive; the hourly index is tz-aware (Europe/Berlin), as yfinance returns it.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ADJUSTMENT_FACTOR = 0.97


def _close(n: int) -> np.ndarray:
    i = np.arange(n, dtype="float64")
    return 18000.0 + 10.0 * i + 50.0 * np.sin(i / 3.0)


def main() -> None:
    """Write the four parquet cassettes next to this script."""
    daily_idx = pd.bdate_range("2024-06-03", "2024-06-28")
    hourly = [
        pd.Timestamp(day, tz="Europe/Berlin") + pd.Timedelta(hours=h)
        for day in pd.bdate_range("2024-06-03", "2024-06-07")
        for h in range(9, 18)
    ]
    frames = {
        "1d": pd.DataFrame({"^GDAXI": _close(len(daily_idx))}, index=daily_idx),
        "1h": pd.DataFrame({"^GDAXI": _close(len(hourly))}, index=pd.DatetimeIndex(hourly)),
    }
    for interval, raw in frames.items():
        raw.to_parquet(HERE / f"GDAXI_{interval}_raw.parquet")
        (raw * ADJUSTMENT_FACTOR).to_parquet(HERE / f"GDAXI_{interval}_adjusted.parquet")


if __name__ == "__main__":
    main()
