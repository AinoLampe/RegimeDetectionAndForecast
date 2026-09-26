"""Regenerate the synthetic CBOE cassettes (deterministic; no provider data is stored).

Run from the repository root: `uv run python tests/fixtures/cassettes/cboe/make_cassettes.py`.
`VIX_History.csv` mimics CBOE's layout (`DATE,OPEN,HIGH,LOW,CLOSE`, `MM/DD/YYYY`);
`VVIX_History.csv` mimics the single-column layout (`DATE,VVIX`).
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def main() -> None:
    """Write both CSV files (business days 2024-01-02 .. 2024-02-29) next to this script."""
    days = pd.bdate_range("2024-01-02", "2024-02-29")
    i = np.arange(len(days), dtype="float64")
    close = 13.5 + 1.5 * np.sin(i / 3.0)
    dates = days.strftime("%m/%d/%Y")
    vix = pd.DataFrame(
        {"DATE": dates, "OPEN": close + 0.1, "HIGH": close + 0.6, "LOW": close - 0.5, "CLOSE": close}
    )
    vix.to_csv(HERE / "VIX_History.csv", index=False, float_format="%.6f", lineterminator="\n")
    vvix = pd.DataFrame({"DATE": dates, "VVIX": 85.0 + 5.0 * np.cos(i / 4.0)})
    vvix.to_csv(HERE / "VVIX_History.csv", index=False, float_format="%.2f", lineterminator="\n")


if __name__ == "__main__":
    main()
