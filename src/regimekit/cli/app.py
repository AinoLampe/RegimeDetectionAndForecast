"""Typer CLI entry point (further commands are added by their own tickets)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydantic
import typer

from regimekit.config.schema import DataSection, PriceSource, ProjectConfig
from regimekit.config.yaml_io import dump_resolved, export_json_schema, load_config
from regimekit.data.cache import Cache, CachedProvider, SnapshotPolicy
from regimekit.data.contracts import ObservationSet
from regimekit.data.prepare import Preparer, PrepDiagnostics, prep_stats_to_json
from regimekit.data.quality import build_quality_report
from regimekit.finance.providers import get_provider
from regimekit.persistence.results import write_prepared
from regimekit.utils.clock import CALENDAR_DAYS_PER_YEAR
from regimekit.utils.errors import EXIT_CODES, ConfigError, DataError, PersistenceError, RegimekitError
from regimekit.utils.logging import configure_logging, get_logger

app = typer.Typer(name="regimekit")


@dataclass
class GlobalOptions:
    """CLI-wide options set on the `regimekit` app itself, read by subcommands."""

    out: Path | None = None
    offline: bool = False
    device: str | None = None
    seed: int | None = None
    log_level: str | None = None


_global_options = GlobalOptions()


@app.callback()
def _main_callback(
    out: Path | None = typer.Option(None, "--out", help="Output directory (overrides project.output_dir)."),
    offline: bool = typer.Option(False, "--offline", help="Do not access the network."),
    device: str | None = typer.Option(None, "--device", help="cpu|gpu|auto (overrides estimation.device)"),
    seed: int | None = typer.Option(None, "--seed", help="Overrides project.seed."),
    log_level: str | None = typer.Option(None, "--log-level", help="Overrides project.log_level."),
) -> None:
    """regimekit: JAX-based Hidden Semi-Markov Model regime detection and forecasting."""
    global _global_options
    _global_options = GlobalOptions(out=out, offline=offline, device=device, seed=seed, log_level=log_level)


def _apply_overrides(config: ProjectConfig, options: GlobalOptions) -> None:
    """Apply `--seed`/`--device` (and any other global overrides) onto `config`, in place.

    `validate_assignment=True` (E1-T02a) re-validates each field on assignment, raising
    `pydantic.ValidationError` for e.g. `--device bogus` -- converted here to `ConfigError` so
    an invalid override surfaces the same way any other invalid config value would (exit 2, path
    in the message), rather than an unhandled pydantic traceback.
    """
    try:
        if options.seed is not None:
            config.project.seed = options.seed
        if options.device is not None:
            config.estimation.device = options.device  # type: ignore[assignment]
        if options.log_level is not None:
            config.project.log_level = options.log_level  # type: ignore[assignment]
    except pydantic.ValidationError as exc:
        first = exc.errors()[0]
        path = ".".join(str(part) for part in first["loc"])
        raise ConfigError(first["msg"], path=path or None) from exc


def _summary_lines(config: ProjectConfig) -> list[str]:
    emission_covariates = config.model.emission.covariates
    entry_points = sorted(
        {ep for cov in config.data.covariates for ep in cov.use_in}
        | ({"transition"} if config.model.transition.covariates else set())
        | ({"duration"} if config.model.duration.covariates else set())
        | ({"emission"} if emission_covariates.drift or emission_covariates.log_vol else set())
    )
    return [
        f"n_states: {config.model.n_states}",
        f"engine: {config.model.engine}",
        f"emission family: {config.model.emission.family}",
        f"duration family: {config.model.duration.family}",
        f"covariate entry points: {', '.join(entry_points) if entry_points else '(none)'}",
        f"warnings: {'; '.join(config.warnings) if config.warnings else '(none)'}",
    ]


@app.command("validate-config")
def validate_config_cmd(config_path: Path) -> None:
    """Validate `config_path`, print a summary, and write the resolved config alongside it."""
    try:
        config = load_config(config_path)
        _apply_overrides(config, _global_options)
        out_dir = _global_options.out or config.project.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_resolved(config, out_dir / "config.resolved.yaml")
    except RegimekitError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=EXIT_CODES.get(type(exc), 1)) from exc
    for line in _summary_lines(config):
        typer.echo(line)


@app.command("config-schema")
def config_schema_cmd(out_path: Path) -> None:
    """Write `ProjectConfig`'s JSON-Schema to `out_path`."""
    export_json_schema(out_path)


def _make_provider(name: str, data: DataSection, config_dir: Path) -> object:
    """Instantiate a provider adapter for `fetch`.

    `csv` files are resolved relative to the config file's directory (the schema has no provider
    options); quota-tracking providers keep their quota files in the cache directory. API keys come
    from the environment only (CLAUDE.md §9).
    """
    kwargs: dict[str, object] = {}
    if name == "csv":
        kwargs["root"] = config_dir
    elif name in ("stooq", "tiingo"):
        kwargs["quota_dir"] = Path(data.cache.dir).expanduser()
    return get_provider(name, **kwargs)


def _train_end(obs: ObservationSet, initial_train_years: float) -> np.datetime64:
    """End of the initial training window: first timestamp + `initial_train_years`, capped at the last."""
    seconds = round(initial_train_years * CALENDAR_DAYS_PER_YEAR * 86_400)
    end: np.datetime64 = obs.times[0] + np.timedelta64(seconds, "s")
    last: np.datetime64 = obs.times[-1]
    return min(end, last)


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
        return {"missing_price": diagnostics.missing_price_rows}
    return {
        "missing_price": diagnostics.missing_price_rows,
        "warmup": diagnostics.warmup_rows,
        **{f"covariate:{name}": n for name, n in diagnostics.dropped.items()},
    }


def run_fetch(config: ProjectConfig, config_dir: Path, out_dir: Path, offline: bool) -> None:
    """Fetch through the cache, prepare, and write the dataset, stats, snapshot ids and quality report.

    The quality report is written even when preparation fails (e.g. `missing: error`); the error is
    then re-raised.

    Raises:
        ConfigError: a pinned `data.cache.snapshot` combined with covariates (one id names one cache
            entry; pinning multi-source runs awaits an owner decision).
        DataError: an empty primary request, or preparation failed.
        ProviderError: a provider failed.
        PersistenceError: the output directory cannot be written (exit 1, CLAUDE.md §3.8).
    """
    data = config.data
    policy = SnapshotPolicy(data.cache.snapshot, data.cache.max_age_days, offline)
    if policy.pinned is not None and data.covariates:
        raise ConfigError(
            "a pinned snapshot id identifies one cache entry and cannot pin covariates; use "
            "`snapshot: latest` with `--offline` to replay a run with covariates",
            path="data.cache.snapshot",
        )
    log = get_logger("regimekit.fetch")
    outputs = [out_dir / n for n in ("prepared.parquet", "prep_stats.json", "snapshots.json")]
    try:  # a failed run must not leave an earlier run's dataset next to its own report
        for stale in outputs:
            stale.unlink(missing_ok=True)
    except OSError as exc:
        raise PersistenceError(f"cannot clear {out_dir}: {exc}", path=str(out_dir)) from exc
    cache = Cache(data.cache.dir)
    providers: dict[str, CachedProvider] = {}

    def cached(name: str) -> CachedProvider:
        if name not in providers:
            inner = _make_provider(name, data, config_dir)
            providers[name] = CachedProvider(inner, cache, policy)  # type: ignore[arg-type]
            log.info("provider_terms_of_use", provider=name, note=providers[name].tos_note)
        return providers[name]

    p = data.primary
    primary = cached(p.provider)
    obs = primary.fetch_prices(list(p.symbols), p.start, p.end, p.interval, p.adjusted)
    covs = [cached(c.provider).fetch_series(c.series, p.start, p.end) for c in data.covariates]
    if obs.times.shape[0] < 2:
        raise DataError(f"the primary request returned {obs.times.shape[0]} rows", path="data.primary")
    generations = _two_newest_generations(cache, primary.name, p)

    preparer = Preparer(data)
    failure: DataError | None = None
    try:
        train_end = _train_end(obs, config.evaluation.backtest.initial_train_years)
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
    try:
        report.write(out_dir)
    except OSError as exc:
        raise PersistenceError(
            f"cannot write the quality report to {out_dir}: {exc}", path=str(out_dir)
        ) from exc
    if failure is not None:
        raise failure

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
            for c, s in zip(data.covariates, covs, strict=True)
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
    log.info("fetch_done", out=str(out_dir), rows=int(dataset.times.shape[0]))


@app.command("fetch")
def fetch_cmd(config_path: Path) -> None:
    """Fetch data through the cache, prepare it, and write the dataset plus a data-quality report."""
    try:
        config = load_config(config_path)
        _apply_overrides(config, _global_options)
        configure_logging(config.project.log_level)
        out_dir = _global_options.out or config.project.output_dir
        run_fetch(config, config_path.resolve().parent, out_dir, _global_options.offline)
    except RegimekitError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=EXIT_CODES.get(type(exc), 1)) from exc


def main() -> None:  # pragma: no cover
    """Run the regimekit CLI, mapping `RegimekitError` subclasses to their exit codes."""
    try:
        app()
    except RegimekitError as exc:
        typer.echo(str(exc), err=True)
        sys.exit(EXIT_CODES.get(type(exc), 1))


if __name__ == "__main__":  # pragma: no cover
    main()
