"""Typer CLI entry point (commands are added by later tickets)."""

import typer

app = typer.Typer(name="regimekit")


def main() -> None:
    """Run the regimekit CLI."""
    app()


if __name__ == "__main__":
    main()
