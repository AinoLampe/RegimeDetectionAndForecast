"""E10-T04a: the finance helper that reads provider results with the old attribute names."""

from __future__ import annotations

import numpy as np
import pytest

from regimekit.data.contracts import ObservationSet
from regimekit.finance.views import price_view
from regimekit.utils.errors import DataError

pytestmark = pytest.mark.finance

DAYS = np.arange("2024-01-01", "2024-01-13", dtype="datetime64[D]").astype("datetime64[ns]")


def test_e10_t04a_price_view_reads_the_finance_names_from_meta() -> None:
    """The finance helper gives the old attribute names; plain data without provider metadata is rejected."""
    obs = ObservationSet(
        ("X",), DAYS, np.ones((12, 1)), "csv", "s", {"interval": "1d", "adjusted": True, "note": 1}
    )
    view = price_view(obs)
    assert (view.symbols, view.interval, view.adjusted, view.provider) == (("X",), "1d", True, "csv")
    assert view.prices is obs.values
    with pytest.raises(DataError):
        price_view(ObservationSet(("a",), DAYS, np.ones((12, 1)), "test", "s"))
    with pytest.raises(DataError):
        price_view(ObservationSet(("a",), None, np.ones((12, 1)), "test", "s"))
