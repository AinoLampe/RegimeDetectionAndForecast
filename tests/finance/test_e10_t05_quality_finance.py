"""E10-T05: the data-quality report is split into the core's generic checks and the finance checks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from regimekit.data.contracts import ObservationSet
from regimekit.data.quality import duplicate_times, generic_checks, index_gaps, mad_outliers
from regimekit.finance.quality import build_quality_report
from regimekit.finance.transforms import log_returns

pytestmark = pytest.mark.finance


def _obs(values: np.ndarray, times: np.ndarray) -> ObservationSet:
    return ObservationSet(
        ("X",), times, values.reshape(-1, 1), "csv", "s", {"interval": "1d", "adjusted": True}
    )


def test_e10_t05_generic_checks_find_duplicates_gaps_and_outliers() -> None:
    """Duplicated stamps, a gap `> 1.5 x` the median spacing and a `> 8 MAD` outlier."""
    days = np.array(
        ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-10", "2024-01-11"],
        dtype="datetime64[ns]",
    )
    values = np.array([1.0, 1.1, 1.0, 0.9, 1.0, 50.0, 1.05])
    obs = _obs(values, days)
    assert duplicate_times(days) == ["2024-01-02"]
    assert index_gaps(days) == ["2024-01-10"]
    out = mad_outliers(obs)
    assert [o["at"] for o in out] == ["2024-01-10"]
    report = generic_checks(obs, {"v": np.array([0, 0, 1, -1])})
    assert report.n_rows == 7
    assert report.staleness == {"v": {"0": 2, "1": 1, "no_value": 1}}


def test_e10_t05_finance_report_keeps_its_findings_and_adds_the_generic_section() -> None:
    """AC4: the finance keys are unchanged; `generic` is an extra key, in JSON and Markdown."""
    days = np.arange("2024-01-01", "2024-02-20", dtype="datetime64[D]").astype("datetime64[ns]")
    days = days[(days.astype("datetime64[D]").astype(np.int64) + 3) % 7 < 5]
    rng = np.random.default_rng(1)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, days.shape[0])))
    prices[10] = -1.0
    report = build_quality_report(_obs(prices, days))
    data = json.loads(report.to_json())
    assert {
        "symbols",
        "n_rows",
        "gaps",
        "duplicates",
        "non_positive_prices",
        "outliers",
        "adjustment",
    } <= set(data)
    assert len(data["non_positive_prices"]) == 1
    assert set(data["generic"]) == {"n_rows", "duplicates", "index_gaps", "outliers", "staleness"}
    assert "## Generic checks" in report.to_markdown()


def test_e10_t05_log_returns_are_the_moved_finance_transform() -> None:
    """`log_returns` lives in `regimekit.finance.transforms`, registered as the `log_returns` transform."""
    from regimekit.data.transforms import get_observation_transform

    prices = np.array([[100.0], [101.0], [99.0]])
    times = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]")
    y, t = get_observation_transform("log_returns")(prices, times)
    np.testing.assert_array_equal(y, log_returns(prices))
    assert t is not None
    assert t.tolist() == times[1:].tolist()
