"""E10-T07b: the finance app is the core app plus `fetch`; `regimekit.Project` is the finance facade."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import regimekit
from regimekit.cli import app as core_app_module
from regimekit.cli.app import exit_code_of
from regimekit.finance.cli import app as finance_app
from regimekit.finance.config import FinanceConfig
from regimekit.finance.errors import ProviderError
from regimekit.finance.project import FinanceProject
from regimekit.project import Project

pytestmark = pytest.mark.finance

ROOT = Path(__file__).resolve().parents[2]


def test_e10_t07b_finance_app_is_the_core_commands_plus_fetch() -> None:
    """AC1/AC2: the finance app has every core command and `fetch`; the core app has no `fetch`."""
    core = {c.name for c in core_app_module.app.registered_commands}
    finance = {c.name for c in finance_app.registered_commands}
    assert finance == core | {"fetch"}
    assert "fetch" not in core
    assert finance_app.context.config_cls is FinanceConfig
    assert finance_app.context.project_cls is FinanceProject


def test_e10_t07b_provider_error_exits_5_in_the_finance_app_only() -> None:
    """The finance module registers `ProviderError -> 5`; `DataError` keeps 3."""
    assert exit_code_of(ProviderError("x")) == 5
    assert issubclass(ProviderError, Exception)


def test_e10_t07b_regimekit_project_is_the_finance_facade() -> None:
    """AC4: `from regimekit import Project` is `FinanceProject`, a subclass of the core `Project`."""
    assert regimekit.Project is FinanceProject
    assert issubclass(FinanceProject, Project)
    assert FinanceProject.config_cls is FinanceConfig
    assert Project.config_cls is not FinanceConfig
    for method in ("fetch", "fetch_inputs"):
        assert hasattr(FinanceProject, method)
        assert not hasattr(Project, method)


def test_e10_t07b_console_scripts() -> None:
    """`regimekit` points at the finance app, `regimekit-core` at the core app."""
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    assert scripts["regimekit"] == "regimekit.finance.cli:main"
    assert scripts["regimekit-core"] == "regimekit.cli.app:main"
