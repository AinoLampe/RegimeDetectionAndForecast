"""E5-T05a: fast structural tests for `estimation.recovery` (default suite, not `-m slow`).

Tiny `t_len` so these run in the default suite and keep `estimation`'s coverage >= 85% (CLAUDE.md
§3.2) without paying for the nightly suite's `T=5000` fits; they check the harness's plumbing
(shapes, check bookkeeping, the empirical-SE fallback), not tolerance recovery.
"""

from __future__ import annotations

import os

import pytest

from regimekit.estimation.recovery import (
    SCENARIOS,
    ParamCheck,
    RecoveryReport,
    _default_t_len,
    run_recovery_suite,
    run_scenario,
)

_SMOKE_T = 120


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_e5_t05a_recovery_run_scenario_well_formed(name: str) -> None:
    """Every scenario simulates, fits and produces a non-empty, well-typed `RecoveryReport`."""
    report = run_scenario(name, seed=0, t_len=_SMOKE_T)
    assert isinstance(report, RecoveryReport)
    assert report.scenario == name
    assert report.t_len == _SMOKE_T
    assert len(report.checks) > 0
    for check in report.checks:
        assert isinstance(check, ParamCheck)
        assert check.line()  # must not raise, must be non-empty
    # sigma/mean-sojourn/jump checks never need an SE, so they are always evaluated.
    always_evaluated = [c for c in report.checks if "sigma" in c.name or "mean_sojourn" in c.name]
    assert always_evaluated
    assert all(c.evaluated for c in always_evaluated)


def test_e5_t05a_recovery_mu_fixed_scenario_has_no_mu_checks() -> None:
    """Scenario 6 (`fixed_drift`): `emission.mu` is fixed, so no `mu` tolerance check is emitted."""
    report = run_scenario("fixed_drift", seed=0, t_len=_SMOKE_T)
    assert not any(c.name.startswith("emission.mu") for c in report.checks)


def test_e5_t05a_recovery_empirical_se_fills_in_missing_hessian() -> None:
    """`explicit_gamma`'s Hessian is unavailable (no JVP for `igamma`'s shape derivative).

    `run_recovery_suite` must still evaluate its `mu` checks via the across-seed empirical spread.
    """
    suite = run_recovery_suite("explicit_gamma", seeds=(0, 1), t_len=_SMOKE_T)
    assert len(suite.reports) == 2
    mu_checks = [c for r in suite.reports for c in r.checks if c.name.startswith("emission.mu")]
    assert mu_checks
    assert all(c.evaluated for c in mu_checks)
    assert all(c.se is not None for c in mu_checks)
    # n_pass/passed/failures() must not raise regardless of outcome.
    assert suite.n_pass == sum(r.passed for r in suite.reports)
    suite.failures()


def test_e5_t05a_recovery_t_len_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC3: `REGIMEKIT_RECOVERY_T` controls the default `t_len` (nightly-suite sizing knob)."""
    monkeypatch.setenv("REGIMEKIT_RECOVERY_T", "777")
    assert _default_t_len() == 777
    monkeypatch.delenv("REGIMEKIT_RECOVERY_T", raising=False)
    assert _default_t_len() == 5000


def test_e5_t05a_recovery_t_len_env_var_used_when_not_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_scenario(t_len=None)` reads the env var (kept tiny here to stay in the default suite)."""
    monkeypatch.setenv("REGIMEKIT_RECOVERY_T", str(_SMOKE_T))
    report = run_scenario("gbm_negbin", seed=0, t_len=None)
    assert report.t_len == _SMOKE_T
    assert os.environ["REGIMEKIT_RECOVERY_T"] == str(_SMOKE_T)
