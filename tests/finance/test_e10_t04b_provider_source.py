"""E10-T04b: the finance `provider` source kind, the moved cache and the fetch path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from regimekit.finance.config import FinanceDataSection
from regimekit.finance.errors import ProviderError
from regimekit.finance.providers.csv import CsvProvider
from regimekit.finance.providers.source import FetchContext
from regimekit.plugins import REGISTRY
from regimekit.testing.source_contract import FIXTURES, SourceCase, check_source
from regimekit.utils.errors import ConfigError

pytestmark = pytest.mark.finance


def _provider_case(tmp_path: Path) -> SourceCase:
    """Contract case: the CSV provider adapter on the `gauss` fixture, offline."""
    root = tmp_path / "csv"
    root.mkdir()
    frame = pd.read_csv(FIXTURES / "gauss.csv")
    for col in ("y1", "y2"):
        pd.DataFrame({"time": frame["t"], "close": frame[col], "adj_close": frame[col]}).to_csv(
            root / f"{col}.csv", index=False
        )
    pd.DataFrame({"time": frame["t"], "value": frame["x"]}).to_csv(root / "x.csv", index=False)

    def options(time: bool) -> dict[str, Any]:
        return {
            "provider": "csv",
            "symbols": ["y1", "y2"],
            "interval": "1d",
            "start": "2023-12-01",
            "adjusted": False,
        }

    def context(time: bool) -> FetchContext:
        data = FinanceDataSection.model_validate(
            {
                "source": {"kind": "provider", **options(True)},
                "cache": {"dir": str(tmp_path / "cache")},
                "covariates": [
                    {
                        "name": "x",
                        "source": {"kind": "provider", "provider": "csv", "series": "x"},
                        "lag": 1,
                        "use_in": ["emission"],
                    }
                ],
            }
        )
        return FetchContext(data, root)

    return SourceCase(
        options=options,
        obj=context,
        missing=lambda: ({**options(True), "symbols": ["y1", "nope"]}, context(True), "nope"),
        supports_timeless=False,
        expected_source="csv",
        error=ProviderError,
    )


def test_e10_t04b_provider_source_passes_the_contract_suite(tmp_path: Path) -> None:
    """AC1: the `provider` kind (CSV adapter, offline) passes `source_contract`; provenance is finance."""
    entry = next(e for e in REGISTRY.entries("source") if e.name == "provider")
    assert entry.provenance == "regimekit.finance"
    check_source("provider", _provider_case(tmp_path))


def test_e10_t04b_second_load_is_served_by_the_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC2: the second fetch hits the cache; the provider adapter is called once per request."""
    case = _provider_case(tmp_path)
    calls = {"prices": 0, "series": 0}
    real_prices, real_series = CsvProvider.fetch_prices, CsvProvider.fetch_series

    def prices(self: CsvProvider, *a: Any, **k: Any) -> Any:  # noqa: ANN401
        calls["prices"] += 1
        return real_prices(self, *a, **k)

    def series(self: CsvProvider, *a: Any, **k: Any) -> Any:  # noqa: ANN401
        calls["series"] += 1
        return real_series(self, *a, **k)

    monkeypatch.setattr(CsvProvider, "fetch_prices", prices)
    monkeypatch.setattr(CsvProvider, "fetch_series", series)
    from regimekit.data.sources.base import get_source, validate_options

    source = get_source("provider")
    opts = validate_options("provider", case.options(True), "data.source")
    assert opts is not None
    first, covs1 = source.load(opts, obj=case.obj(True))
    second, covs2 = source.load(opts, obj=case.obj(True))
    assert calls == {"prices": 1, "series": 1}
    assert first.snapshot_id == second.snapshot_id
    assert first == second
    assert covs1 == covs2


def test_e10_t04b_provider_source_needs_a_fetch_context(tmp_path: Path) -> None:
    """Without the context there is no cache to go through: `ConfigError` on `data.source.kind`."""
    from regimekit.data.sources.base import get_source, validate_options

    case = _provider_case(tmp_path)
    opts = validate_options("provider", case.options(True), "data.source")
    assert opts is not None
    with pytest.raises(ConfigError) as info:
        get_source("provider").load(opts)
    assert info.value.path == "data.source.kind"


def test_e10_t04b_covariate_provider_options_are_validated_with_their_own_model() -> None:
    """A covariate's `provider` source takes `provider, series`; a stray key fails with the key path."""
    with pytest.raises(ConfigError) as info:
        FinanceDataSection.model_validate(
            {
                "source": {
                    "kind": "provider",
                    "provider": "csv",
                    "symbols": ["a"],
                    "interval": "1d",
                    "start": "2020-01-01",
                },
                "covariates": [
                    {
                        "name": "x",
                        "source": {"kind": "provider", "provider": "csv", "serie": "x"},
                        "lag": 1,
                        "use_in": ["emission"],
                    }
                ],
            }
        )
    assert info.value.path is not None
    assert "serie" in info.value.path or "series" in info.value.path
