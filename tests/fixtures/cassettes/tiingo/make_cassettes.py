"""Regenerate the synthetic Tiingo cassette `SPY.json` (deterministic; no provider data is stored).

Run from the repository root: `uv run python tests/fixtures/cassettes/tiingo/make_cassettes.py`.
Rows mimic Tiingo's daily-prices JSON (`date` as ISO-8601 UTC, raw OHLCV plus `adj*` fields). A synthetic
dividend on 2024-03-15 makes `adjClose` differ from `close` before that date.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def main() -> None:
    """Write `SPY.json` (business days 2024-01-02 .. 2024-06-28) next to this script."""
    days = pd.bdate_range("2024-01-02", "2024-06-28")
    i = np.arange(len(days), dtype="float64")
    close = 470.0 + 0.4 * i + 3.0 * np.sin(i / 5.0)
    factor = np.where(days < pd.Timestamp("2024-03-15"), 0.9965, 1.0)
    rows = [
        {
            "date": day.strftime("%Y-%m-%dT00:00:00.000Z"),
            "close": round(float(c), 4),
            "high": round(float(c) + 2.0, 4),
            "low": round(float(c) - 2.0, 4),
            "open": round(float(c) - 0.5, 4),
            "volume": int(60_000_000 + 1000 * k),
            "adjClose": round(float(c * f), 4),
            "adjHigh": round(float((c + 2.0) * f), 4),
            "adjLow": round(float((c - 2.0) * f), 4),
            "adjOpen": round(float((c - 0.5) * f), 4),
            "adjVolume": int(60_000_000 + 1000 * k),
            "divCash": 1.75 if day == pd.Timestamp("2024-03-15") else 0.0,
            "splitFactor": 1.0,
        }
        for k, (day, c, f) in enumerate(zip(days, close, factor, strict=True))
    ]
    (HERE / "SPY.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
