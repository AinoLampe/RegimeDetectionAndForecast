"""Finance configuration: `FinanceConfig` (schema version 2) on top of the neutral `ProjectConfig` (E10-T02b).

The core configuration (`regimekit.config.schema`) holds no finance vocabulary. This module subclasses its
sections where the finance application needs more: the provider source options (`PriceSource`), the cache,
the clock options (`bars_per_day`), `init_mean_sojourn_days`, the trading strategy and its output. Version-1
configurations (Part I, with `profile`, `data.primary`, ...) are migrated by `migrate_1_to_2`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

import pydantic
from pydantic import ConfigDict, Field, field_validator, model_validator

from regimekit.config.migrate import register_migration
from regimekit.config.schema import (
    CovariateCfg,
    DataSection,
    DurationCfg,
    EmissionCfg,
    EvaluationSection,
    ModelSection,
    OutputSection,
    ParamCfg,
    PrepCfg,
    ProjectConfig,
    _Base,
)
from regimekit.finance import clocks as clocks  # registers the `trading` clock
from regimekit.plugins import register
from regimekit.utils.clock import days_to_steps, get_clock
from regimekit.utils.errors import ConfigError

SCHEMA_ID = "regimekit"
SCHEMA_VERSION = 2


@register("source", "provider")
class PriceSource(_Base):
    """Options of the `provider` source kind (`data.source:`): the primary price series to fetch."""

    provider: str
    symbols: list[str]
    interval: Literal["1h", "1d", "1wk", "1mo"]
    start: str
    end: str | None = None
    adjusted: bool = True
    field: str = "close"


class SeriesSource(_Base):
    """Options of a covariate's `provider` source (`data.covariates[].source:`)."""

    provider: str
    series: str


class CacheCfg(_Base):
    """`data.cache:` — local parquet cache settings."""

    dir: Path = Path("~/.cache/regimekit")
    snapshot: str = "latest"
    max_age_days: int = 1


class FinanceCovariateCfg(CovariateCfg):
    """A covariate fetched from a data provider (`source: {kind: provider, provider, series}`)."""

    @property
    def provider(self) -> str:
        """Name of the provider the series comes from."""
        return str(self._series_source().provider)

    @property
    def series(self) -> str:
        """The provider's series id."""
        return str(self._series_source().series)

    def _series_source(self) -> SeriesSource:
        if self.source is None:
            raise ConfigError("a finance covariate needs a source", path="data.covariates[].source")
        return SeriesSource.model_validate(self.source.options())


class FinancePrepCfg(PrepCfg):
    """Finance preparation defaults: log-returns of the prices, rows with a missing price dropped."""

    observation_transform: str | None = "log_returns"
    missing_observations: Literal["mask", "drop", "error"] = "drop"

    @field_validator("observation_transform")
    @classmethod
    def _only_log_returns(cls, value: str | None) -> str | None:
        if value != "log_returns":
            raise ConfigError(
                f"unknown observation transform {value!r}; registered: log_returns",
                path="data.preparation.observation_transform",
            )
        return value

    @field_validator("missing_observations")
    @classmethod
    def _drop_or_error(cls, value: str) -> str:
        if value not in ("drop", "error"):
            raise ConfigError(
                f"missing_observations must be 'drop' or 'error', got {value!r}",
                path="data.preparation.missing_observations",
            )
        return value


def _convert(exc: pydantic.ValidationError, prefix: str) -> ConfigError:
    first = exc.errors()[0]
    loc = ".".join(str(part) for part in first["loc"])
    return ConfigError(first["msg"], path=f"{prefix}.{loc}" if loc else prefix)


class FinanceDataSection(DataSection):
    """`data:` of the finance application: provider source, trading clock, cache."""

    clock: str = "trading"
    cache: CacheCfg = Field(default_factory=CacheCfg)
    covariates: list[FinanceCovariateCfg] = Field(default_factory=list)  # type: ignore[assignment]
    preparation: FinancePrepCfg = Field(default_factory=FinancePrepCfg)

    @model_validator(mode="after")
    def _check_sources_and_clock_options(self) -> FinanceDataSection:
        try:
            PriceSource.model_validate(self.source.options())
        except pydantic.ValidationError as exc:
            raise _convert(exc, "data.source") from exc
        for i, cov in enumerate(self.covariates):
            if cov.source is None or cov.source.kind != "provider":
                raise ConfigError(
                    "a covariate needs a `provider` source", path=f"data.covariates[{i}].source"
                )
            try:
                SeriesSource.model_validate(cov.source.options())
            except pydantic.ValidationError as exc:
                raise _convert(exc, f"data.covariates[{i}].source") from exc
        return self

    @property
    def primary(self) -> PriceSource:
        """The primary price series (the options of the `provider` source)."""
        return PriceSource.model_validate(self.source.options())

    @property
    def bars_per_day(self) -> int:
        """Intraday bars per trading day (`data.clock_options.bars_per_day`, default 1)."""
        return int(self.clock_options.get("bars_per_day", 1))


class FinanceParamCfg(ParamCfg):
    """A parameter config that may state its initial mean sojourn in days."""

    init_mean_sojourn_days: list[float] | None = None


class FinanceEmissionCfg(EmissionCfg):
    """`model.emission:` with finance parameter configs."""

    params: dict[str, FinanceParamCfg]  # type: ignore[assignment]


class FinanceDurationCfg(DurationCfg):
    """`model.duration:` with finance parameter configs."""

    params: dict[str, FinanceParamCfg]  # type: ignore[assignment]


class FinanceModelSection(ModelSection):
    """`model:` with finance parameter configs."""

    emission: FinanceEmissionCfg
    duration: FinanceDurationCfg


class StrategyCfg(_Base):
    """`evaluation.strategy:` — a simple regime-conditioned trading rule."""

    type: Literal["regime_long_flat", "regime_target_vol"]
    flat_if: dict[str, Any] | None = None
    target_vol: float | None = None
    cost_bps: float = Field(default=0.0, ge=0.0)
    cap: float = 1.0  # regime_target_vol: largest absolute position
    allow_short: bool = False  # regime_target_vol: short when the predictive mean is negative


class FinanceEvaluationSection(EvaluationSection):
    """`evaluation:` plus the optional trading strategy."""

    strategy: StrategyCfg | None = None


class FinanceOutputSection(OutputSection):
    """`output:` plus the `strategy` file."""

    extra_outputs: ClassVar[tuple[str, ...]] = ("strategy",)


class FinanceConfig(ProjectConfig):
    """The whole resolved configuration of the finance application (schema version 2)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    SCHEMA_ID: ClassVar[str] = SCHEMA_ID
    SCHEMA_VERSION: ClassVar[int] = SCHEMA_VERSION

    schema_version: int = SCHEMA_VERSION
    data: FinanceDataSection
    model: FinanceModelSection
    evaluation: FinanceEvaluationSection
    output: FinanceOutputSection

    @model_validator(mode="after")
    def _check_days_rules(self) -> FinanceConfig:
        n = self.model.n_states
        for path, pc in _named_params(self):
            days = pc.init_mean_sojourn_days
            if days is not None and pc.init_mean_sojourn_steps is not None:
                raise ConfigError(
                    "at most one of init_mean_sojourn_days/init_mean_sojourn_steps",
                    path=f"{path}.init_mean_sojourn_days",
                )
            if days is not None and len(days) != n:
                raise ConfigError(
                    f"init_mean_sojourn_days must have length n_states={n}",
                    path=f"{path}.init_mean_sojourn_days",
                )
            if days is not None and self.data.clock == "steps":
                raise ConfigError(
                    "init_mean_sojourn_days is not allowed under clock='steps'; use init_mean_sojourn_steps",
                    path=f"{path}.init_mean_sojourn_days",
                )
        return self

    def expected_sojourn_steps(self) -> float | None:
        """As the core, also counting `init_mean_sojourn_days` (converted with the clock's nominal step)."""
        clock = self.data.clock
        nominal_dt = (
            1.0 if get_clock(clock).unit == "step" else 1.0 / float(get_clock(clock).days_per_year or 1.0)
        )
        values: list[float] = []
        for pc in self.model.duration.params.values():
            if pc.init_mean_sojourn_steps is not None:
                values.extend(pc.init_mean_sojourn_steps)
            elif pc.init_mean_sojourn_days is not None:
                values.extend(
                    days_to_steps(d, clock, nominal_dt, self.data.clock_options)
                    for d in pc.init_mean_sojourn_days
                )
            elif pc.init is not None and clock == "steps":
                values.extend(pc.init)
        return max(values) if values else None


def _named_params(config: FinanceConfig) -> list[tuple[str, FinanceParamCfg]]:
    return [(f"model.emission.params.{k}", pc) for k, pc in config.model.emission.params.items()] + [
        (f"model.duration.params.{k}", pc) for k, pc in config.model.duration.params.items()
    ]


@register_migration(SCHEMA_ID, 1)
def migrate_1_to_2(raw: dict[str, Any]) -> dict[str, Any]:
    """Part I (`schema_version: 1`) -> version 2: `profile` dropped, the neutral names and locations."""
    cfg = dict(raw)
    profile = cfg.pop("profile", "finance")
    if profile == "core":
        raise ConfigError("profile 'core' is not executable in this release; use 'finance'", path="profile")
    data = dict(cfg.get("data", {}))
    primary = data.pop("primary", None)
    if primary is not None:
        data["source"] = {"kind": "provider", **primary}
    covariates = []
    for cov in data.get("covariates", []):
        cov = dict(cov)
        provider, series = cov.pop("provider", None), cov.pop("series", None)
        if provider is not None or series is not None:
            cov["source"] = {"kind": "provider", "provider": provider, "series": series}
        covariates.append(cov)
    if "covariates" in data:
        data["covariates"] = covariates
    if "bars_per_day" in data:
        bars = data.pop("bars_per_day")
        if data.get("clock", "trading") == "trading" or bars != 1:  # `1` is the no-op default of other clocks
            data["clock_options"] = {**data.get("clock_options", {}), "bars_per_day": bars}
    prep = dict(data.get("preparation", {}))
    if "returns" in prep:
        prep["observation_transform"] = {"log": "log_returns"}.get(prep.pop("returns"), prep.get("returns"))
    if "missing_returns" in prep:
        prep["missing_observations"] = prep.pop("missing_returns")
    if "preparation" in data:
        data["preparation"] = prep
    cfg["data"] = data
    evaluation = dict(cfg.get("evaluation", {}))
    backtest = dict(evaluation.get("backtest", {}))
    for old, new in (("initial_train_years", "initial_train"), ("rolling_years", "rolling_window")):
        if old in backtest:
            backtest[new] = backtest.pop(old)
    if "backtest" in evaluation:
        evaluation["backtest"] = backtest
        cfg["evaluation"] = evaluation
    emission = cfg.get("model", {}).get("emission")
    if emission is not None and emission.get("family") == "student_t" and "scale_by_dt" not in emission:
        cfg["model"] = {**cfg["model"], "emission": {**emission, "scale_by_dt": True}}
    cfg["schema_version"] = 2
    return cfg
