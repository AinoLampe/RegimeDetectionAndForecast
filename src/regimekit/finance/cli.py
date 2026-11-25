"""The finance CLI (E10-T07b): the core commands plus `fetch`, and `ProviderError` → exit code 5."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import typer

from regimekit.cli.app import CliApp, build_app, exit_on_error, register_exit_code, run
from regimekit.finance.config import FinanceConfig
from regimekit.finance.errors import ProviderError
from regimekit.finance.project import FinanceProject

register_exit_code(ProviderError, 5)


def add_fetch(app: typer.Typer) -> None:
    """Register `fetch`: providers through the cache, preparation, the dataset and the quality report."""
    ctx = cast(CliApp, app).context

    @app.command("fetch")
    def fetch_cmd(config_path: Path) -> None:
        """Fetch data through the cache, prepare it, and write the dataset plus a data-quality report."""
        with exit_on_error():
            project = cast(FinanceProject, ctx.project(config_path))
            project.fetch(write=True)


app = build_app(
    name="regimekit", config_cls=FinanceConfig, project_cls=FinanceProject, extra_commands=[add_fetch]
)


def main() -> None:
    """Entry point of the console script `regimekit`."""
    run(app)


if __name__ == "__main__":  # pragma: no cover
    main()
