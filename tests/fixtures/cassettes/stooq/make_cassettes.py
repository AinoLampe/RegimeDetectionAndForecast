"""Regenerate the synthetic stooq cassette `dax_1d.parquet` (deterministic; no provider data is stored).

Run from the repository root: `uv run python tests/fixtures/cassettes/stooq/make_cassettes.py`.
The frame mimics stooq's CSV layout: a `Date` index and `Open, High, Low, Close, Volume` columns.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def main() -> None:
    """Write `dax_1d.parquet` (business days 2024-01-02 .. 2024-06-28) next to this script."""
    idx = pd.bdate_range("2024-01-02", "2024-06-28", name="Date")
    i = np.arange(len(idx), dtype="float64")
    close = 16500.0 + 8.0 * i + 60.0 * np.sin(i / 4.0)
    frame = pd.DataFrame(
        {
            "Open": close - 5.0,
            "High": close + 20.0,
            "Low": close - 20.0,
            "Close": close,
            "Volume": 1000.0 + i,
        },
        index=idx,
    )
    frame.to_parquet(HERE / "dax_1d.parquet")


if __name__ == "__main__":
    main()
