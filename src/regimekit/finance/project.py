"""`FinanceProject`: the finance application's `Project` (E10-T07b).

It adds `fetch` (prices and covariates from market-data providers through the parquet cache, the
data-quality report), the default strategy hook of a backtest and the finance bundle extension. The
core `Project` keeps `load_data`, `fit`, `decode`, `forecast`, `simulate`, `backtest` and `compare`.
`regimekit.Project` is an alias of `FinanceProject` (the 0.1.x / 0.2.x API).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from regimekit.data.contracts import ObservationSet, PreparedDataset
from regimekit.data.prepare import Preparer, PrepDiagnostics, prep_stats_to_json
from regimekit.evaluation.backtest import OriginHook
from regimekit.finance.cache import Cache, SnapshotPolicy
from regimekit.finance.config import FinanceConfig, PriceSource
from regimekit.finance.persistence import extension_payloads
from regimekit.finance.providers.source import FetchContext
from regimekit.finance.quality import DataQualityReport, build_quality_report
from regimekit.finance.report import strategy_section
from regimekit.finance.strategies import StrategyHook, make_strategy
from regimekit.persistence.results import write_prepared
from regimekit.plugins import REGISTRY
from regimekit.project import FetchedInputs, Project, _train_end
from regimekit.utils.errors import ConfigError, DataError, PersistenceError
from regimekit.utils.logging import get_logger

__all__ = ["FinanceProject"]


def _two_newest_generations(
    cache: Cache, provider: str, primary: PriceSource
) -> tuple[ObservationSet, ObservationSet] | None:
    """The two newest cached generations of the primary request `(newest, previous)`, if both exist."""
    key = cache.price_key(provider, primary.symbols, primary.interval, primary.adjusted)
    newest = cache.generations(key)[:2]
    if len(newest) < 2:
        return None
    loaded = [cache.load_prices(key, g.snapshot_id) for g in newest]
    if loaded[0] is None or loaded[1] is None:  # pragma: no cover  (listed generations exist on disk)
        return None
    return loaded[0], loaded[1]


def _dropped_rows(diagnostics: PrepDiagnostics | None, completed: bool) -> dict[str, int]:
    if diagnostics is None:
        return {}
    if not completed:  # the policies did not finish: only the missing-price count is known
        return {"missing_price": diagnostics.missing_rows}
    return {
        "missing_price": diagnostics.missing_rows,
        "warmup": diagnostics.warmup_rows,
        **{f"covariate:{name}": n for name, n in diagnostics.dropped.items()},
    }


def _check_pinned(config: FinanceConfig, offline: bool) -> None:
    data = config.data
    policy = SnapshotPolicy(data.cache.snapshot, data.cache.max_age_days, offline)
    if policy.pinned is not None and data.covariates:
        raise ConfigError(
            "a pinned snapshot id identifies one cache entry and cannot pin covariates; use "
            "`snapshot: latest` with `--offline` to replay a run with covariates",
            path="data.cache.snapshot",
        )


class FinanceProject(Project):
    """A `Project` with market-data providers: `fetch`, the quality report and the strategy hook.

    Attributes:
        last_quality: The data-quality report of the last `fetch`.
    """

    config_cls = FinanceConfig

    def __init__(
        self,
        config: FinanceConfig,
        *,
        config_dir: Path | str | None = None,
        out_dir: Path | str | None = None,
        offline: bool = False,
    ) -> None:
        """Wrap `config`; no network access and no file writes happen here."""
        super().__init__(config, config_dir=config_dir, out_dir=out_dir, offline=offline)
        self.config: FinanceConfig = config
        self.last_quality: DataQualityReport | None = None

    def _load_raw(self, obj: Any, offline: bool) -> FetchedInputs:  # noqa: ANN401
        """Prices and covariates through the cache (kind `provider`); other kinds use the core sources."""
        data = self.config.data
        if data.source.kind != "provider":
            return super()._load_raw(obj, offline)
        _check_pinned(self.config, offline)
        context = FetchContext(data, self.config_dir, offline)
        source = REGISTRY.get("source", data.source.kind)
        obs, covs = source.load(data.primary, obj=context)
        return FetchedInputs(obs, covs, context)

    def fetch_inputs(self, *, offline: bool | None = None) -> FetchedInputs:
        """Fetch the primary prices and covariates through the cache (no preparation, no writes).

        Raises:
            ConfigError: a pinned snapshot together with covariates.
            DataError: the primary request returned fewer than two rows.
            ProviderError: a provider failed.
        """
        return self.load_inputs(offline=offline)

    def load_data(
        self,
        obj: object = None,
        *,
        offline: bool | None = None,
        write: bool = False,
    ) -> PreparedDataset:
        """`fetch` for provider sources, the core `load_data` for the others."""
        if self.config.data.source.kind != "provider":
            return super().load_data(obj, offline=offline, write=write)
        return self.fetch(offline=offline, write=write)

    def fetch(self, *, offline: bool | None = None, write: bool = False) -> PreparedDataset:
        """Fetch through the cache and prepare; `write=True` writes the dataset, stats, snapshots, quality.

        The quality report (`last_quality`, and its files when writing) exists even when preparation
        fails (e.g. `missing: error`); the error is then re-raised.

        Raises:
            ConfigError: a pinned `data.cache.snapshot` combined with covariates.
            DataError: an empty primary request, or preparation failed.
            ProviderError: a provider failed.
            PersistenceError: the output directory cannot be written (exit 1, CLAUDE.md section 3.8).
        """
        offline = self.offline if offline is None else offline
        config, data_cfg, out_dir = self.config, self.config.data, self.out_dir
        _check_pinned(config, offline)
        log = get_logger("regimekit.fetch")
        outputs = [out_dir / n for n in ("prepared.parquet", "prep_stats.json", "snapshots.json")]
        if write:
            try:  # a failed run must not leave an earlier run's dataset next to its own report
                for stale in outputs:
                    stale.unlink(missing_ok=True)
            except OSError as exc:
                raise PersistenceError(f"cannot clear {out_dir}: {exc}", path=str(out_dir)) from exc
        fetched = self.fetch_inputs(offline=offline)
        obs, covs, p = fetched.obs, fetched.covs, data_cfg.primary
        generations = _two_newest_generations(fetched.context.cache, fetched.context.primary_name, p)

        preparer = Preparer(data_cfg)
        failure: DataError | None = None
        try:
            train_end = _train_end(obs, config.evaluation.backtest)
            dataset = preparer.fit(obs, covs, train_end).transform(obs, covs)
        except DataError as exc:
            failure = exc
        diagnostics = preparer.diagnostics
        report = build_quality_report(
            obs,
            generations=generations,
            staleness=None if diagnostics is None else diagnostics.staleness,
            dropped_rows=_dropped_rows(diagnostics, completed=failure is None),
        )
        self.last_quality = report
        if write:
            try:
                report.write(out_dir)
            except OSError as exc:
                raise PersistenceError(
                    f"cannot write the quality report to {out_dir}: {exc}", path=str(out_dir)
                ) from exc
        if failure is not None:
            raise failure

        if write:
            snapshots = {
                "primary": {
                    "provider": p.provider,
                    "symbols": list(p.symbols),
                    "interval": p.interval,
                    "adjusted": p.adjusted,
                    "snapshot_id": obs.snapshot_id,
                },
                "covariates": [
                    {"name": c.name, "provider": c.provider, "series": c.series, "snapshot_id": s.snapshot_id}
                    for c, s in zip(data_cfg.covariates, covs, strict=True)
                ],
            }
            try:
                write_prepared(dataset, outputs[0])
                outputs[1].write_text(prep_stats_to_json(dataset.prep_stats), encoding="utf-8")
                outputs[2].write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
            except OSError as exc:
                raise PersistenceError(
                    f"cannot write the prepared dataset to {out_dir}: {exc}", path=str(out_dir)
                ) from exc
        self.last_data = dataset
        log.info("fetch_done", out=str(out_dir) if write else None, rows=int(dataset.times.shape[0]))
        return dataset

    def _hooks(self, grid: PreparedDataset) -> list[OriginHook]:
        """The strategy of `evaluation.strategy` as a hook (none without a strategy)."""
        cfg_strategy = self.config.evaluation.strategy
        if cfg_strategy is None:
            return []
        strategy = make_strategy(cfg_strategy, grid.nominal_dt, int(grid.observations.shape[1]))
        return [StrategyHook(strategy, cfg_strategy.cost_bps)]

    def _report_sections(self) -> list[Any]:
        """The strategy section of the backtest report."""
        return [strategy_section]

    def _extensions(self) -> dict[str, Any]:
        """The provider metadata of the data source (`extensions/regimekit.json`)."""
        return extension_payloads(self.config)
