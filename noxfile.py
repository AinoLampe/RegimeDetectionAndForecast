"""Nox sessions for regimekit's quality gates.

Sessions run inside the project's own uv-managed environment (`uv run ...`) rather than
nox-managed virtualenvs, so `uv sync --all-groups` is the single source of truth for
dependencies.
"""

import os

import nox

nox.options.default_venv_backend = "none"
nox.options.reuse_existing_virtualenvs = True


@nox.session
def lint(session: nox.Session) -> None:
    """Ruff lint + format check."""
    session.run("uv", "run", "ruff", "check", ".", external=True)
    session.run("uv", "run", "ruff", "format", "--check", ".", external=True)


@nox.session
def types(session: nox.Session) -> None:
    """Mypy strict on src/."""
    session.run("uv", "run", "mypy", external=True)


@nox.session
def tests(session: nox.Session) -> None:
    """Fast test suite with coverage (excludes slow/live/benchmark/gpu)."""
    env = dict(os.environ, REGIMEKIT_TYPECHECK="1", JAX_PLATFORMS="cpu")
    session.run(
        "uv",
        "run",
        "pytest",
        "-n",
        "auto",
        "--cov",
        "--cov-report=json",
        "--cov-report=term-missing",
        env=env,
        external=True,
    )
    session.run("uv", "run", "python", "scripts/check_package_coverage.py", env=env, external=True)


@nox.session
def recovery(session: nox.Session) -> None:
    """Parameter-recovery suite (slow)."""
    env = dict(os.environ, REGIMEKIT_TYPECHECK="1", JAX_PLATFORMS="cpu")
    session.run("uv", "run", "pytest", "-m", "slow", env=env, external=True)


@nox.session
def docs(session: nox.Session) -> None:
    """Strict mkdocs build."""
    session.run("uv", "run", "mkdocs", "build", "--strict", external=True)


@nox.session
def bench(session: nox.Session) -> None:
    """Benchmark suite."""
    env = dict(os.environ, REGIMEKIT_TYPECHECK="1", JAX_PLATFORMS="cpu")
    session.run("uv", "run", "pytest", "-m", "benchmark", env=env, external=True)
