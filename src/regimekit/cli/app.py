"""Typer CLI entry point (further commands are added by their own tickets)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pydantic
import typer

from regimekit.config.schema import ProjectConfig
from regimekit.config.yaml_io import dump_resolved, export_json_schema, load_config
from regimekit.utils.errors import EXIT_CODES, ConfigError, RegimekitError

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


def main() -> None:  # pragma: no cover
    """Run the regimekit CLI, mapping `RegimekitError` subclasses to their exit codes."""
    try:
        app()
    except RegimekitError as exc:
        typer.echo(str(exc), err=True)
        sys.exit(EXIT_CODES.get(type(exc), 1))


if __name__ == "__main__":  # pragma: no cover
    main()
