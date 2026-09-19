"""Typer CLI entry point (real commands are added by later tickets)."""

import typer

app = typer.Typer(name="regimekit")


def main() -> None:  # pragma: no cover
    """Run the regimekit CLI."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
