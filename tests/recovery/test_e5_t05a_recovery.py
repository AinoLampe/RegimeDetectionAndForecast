"""E5-T05a: the nightly parameter-recovery suite (`-m slow`).

Six scenarios, 5 seeds each, `T=5000` by default (`REGIMEKIT_RECOVERY_T` env var overrides, CLAUDE.md
§9 / ticket AC3). Each scenario must pass at least 4/5 seeds against concept Appendix D's tolerances
(`regimekit.estimation.recovery`, never loosened per CLAUDE.md §6). AC2: a failing assertion prints
every failing seed's truth/estimate/SE lines (`SuiteResult.failures()`).
"""

from __future__ import annotations

import pytest

from regimekit.estimation.recovery import SCENARIOS, run_recovery_suite

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_e5_t05a_recovery(name: str) -> None:
    """AC1: every scenario passes >= 4/5 seeds. AC2: failures are reported with truth/estimate/SE."""
    suite = run_recovery_suite(name)
    assert suite.passed, f"{name}: only {suite.n_pass}/5 seeds passed\n{suite.failures()}"
