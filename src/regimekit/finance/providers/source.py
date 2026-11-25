"""The `provider` source kind (E10-T04b): market-data providers behind the core `Source` protocol.

`ProviderSource.options_model` is the Part I `PriceSource` (`provider, symbols, interval, start, end,
adjusted, field`); a covariate's `provider` source takes `CovariateProviderOpts` (`provider, series`).
`load` resolves the adapter by name from the finance provider registry (E2-T01), goes through the cache
(`regimekit.finance.cache`) and returns the prices plus the declared covariate series.

The cache, the snapshot policy and the config directory reach `load` through the `FetchContext` passed as
`obj`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from regimekit.data.contracts import CovariateSeries, ObservationSet
from regimekit.finance.cache import Cache, CachedProvider, SnapshotPolicy
from regimekit.finance.config import CovariateProviderOpts, PriceSource
from regimekit.finance.providers.base import get_provider
from regimekit.plugins import REGISTRY, provenance_of
from regimekit.utils.errors import ConfigError, DataError
from regimekit.utils.logging import get_logger

if TYPE_CHECKING:
    from regimekit.finance.config import FinanceDataSection


def make_provider(name: str, data: FinanceDataSection, config_dir: Path) -> object:
    """Instantiate a provider adapter for `fetch`.

    `csv` files are resolved relative to the config file's directory (the schema has no provider
    options); quota-tracking providers keep their quota files in the cache directory. API keys come
    from the environment only (CLAUDE.md section 9).
    """
    kwargs: dict[str, object] = {}
    if name == "csv":
        kwargs["root"] = config_dir
    elif name in ("stooq", "tiingo"):
        kwargs["quota_dir"] = Path(data.cache.dir).expanduser()
    return get_provider(name, **kwargs)


@dataclass
class FetchContext:
    """What a provider fetch needs besides the options: the `data:` section, where it lives, the cache."""

    data: FinanceDataSection
    config_dir: Path
    offline: bool = False
    cache: Cache = field(init=False)
    policy: SnapshotPolicy = field(init=False)
    primary_name: str = field(init=False, default="")
    _providers: dict[str, CachedProvider] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Open the cache and build the snapshot policy from `data.cache`."""
        cfg = self.data.cache
        self.cache = Cache(cfg.dir)
        self.policy = SnapshotPolicy(cfg.snapshot, cfg.max_age_days, self.offline)

    def provider(self, name: str) -> CachedProvider:
        """The cached adapter `name` (created and announced once)."""
        if name not in self._providers:
            inner = make_provider(name, self.data, self.config_dir)
            self._providers[name] = CachedProvider(inner, self.cache, self.policy)  # type: ignore[arg-type]
            get_logger("regimekit.fetch").info(
                "provider_terms_of_use", provider=name, note=self._providers[name].tos_note
            )
        return self._providers[name]


class ProviderSource:
    """Prices of a provider (`data.source`) and the provider covariates (`data.covariates[].source`)."""

    name = "provider"
    options_model: type[BaseModel] = PriceSource
    covariate_options_model: type[BaseModel] = CovariateProviderOpts

    def load(
        self,
        opts: BaseModel,
        *,
        obj: Any | None = None,  # noqa: ANN401
    ) -> tuple[ObservationSet, list[CovariateSeries]]:
        """Fetch the primary prices and every covariate series through the cache.

        Args:
            opts: the `PriceSource` options.
            obj: the `FetchContext`.

        Raises:
            ConfigError: no context was passed.
            DataError: the primary request returned fewer than two rows.
            ProviderError: a provider failed.
        """
        if not isinstance(obj, FetchContext):
            raise ConfigError(
                "the `provider` source needs a FetchContext (cache and config directory)",
                path="data.source.kind",
            )
        assert isinstance(opts, PriceSource)
        primary = obj.provider(opts.provider)
        obs = primary.fetch_prices(list(opts.symbols), opts.start, opts.end, opts.interval, opts.adjusted)
        covs = [
            obj.provider(cov.provider).fetch_series(cov.series, opts.start, opts.end)
            for cov in obj.data.covariates
        ]
        if obs.values.shape[0] < 2:
            raise DataError(f"the primary request returned {obs.values.shape[0]} rows", path="data.primary")
        obj.primary_name = primary.name
        return obs, covs


REGISTRY.register("source", ProviderSource.name, ProviderSource(), provenance=provenance_of(__name__))
