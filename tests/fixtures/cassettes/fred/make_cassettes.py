"""Regenerate the synthetic FRED cassettes (deterministic; no provider data is stored).

Run from the repository root: `uv run python tests/fixtures/cassettes/fred/make_cassettes.py`.
Each series gets `<id>.series.json` (shaped like `fred/series`) and `<id>.observations.json` (shaped like
`fred/series/observations`, values as strings, `"."` for holidays).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
START, END = "2024-01-02", "2024-02-29"
HOLIDAYS = {"2024-01-15", "2024-02-19"}
"""US market holidays in the window: FRED reports `"."` on these business days."""


def _write(series_id: str, title: str, units: str, level: float, amplitude: float) -> None:
    days = pd.bdate_range(START, END)
    i = np.arange(len(days), dtype="float64")
    obs = []
    for day, x in zip(days.strftime("%Y-%m-%d"), level + amplitude * np.sin(i / 3.0), strict=True):
        value = "." if day in HOLIDAYS else f"{x:.2f}"
        obs.append({"realtime_start": END, "realtime_end": END, "date": day, "value": value})
    info = {
        "seriess": [
            {
                "id": series_id,
                "title": f"{title} (synthetic fixture)",
                "frequency": "Daily",
                "frequency_short": "D",
                "units": units,
            }
        ]
    }
    (HERE / f"{series_id}.series.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    body = {"observation_start": START, "observation_end": END, "count": len(obs), "observations": obs}
    (HERE / f"{series_id}.observations.json").write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    """Write the `VIXCLS` and `T10Y2Y` cassettes next to this script."""
    _write("VIXCLS", "CBOE Volatility Index: VIX", "Index", 14.0, 2.0)
    _write("T10Y2Y", "10-Year Minus 2-Year Treasury Yield Spread", "Percent", -0.35, 0.10)


if __name__ == "__main__":
    main()
