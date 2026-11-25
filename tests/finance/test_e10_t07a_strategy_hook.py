"""E10-T07a: the finance strategy as an `OriginHook`, the strategy metrics on `extras`, the report section."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from regimekit.data.prepare import Preparer
from regimekit.evaluation.backtest import OriginHook, run_backtest
from regimekit.evaluation.metrics.registry import better_of, resolve
from regimekit.evaluation.results import BacktestResult
from regimekit.finance.config import StrategyCfg
from regimekit.finance.metrics import max_drawdown, sharpe
from regimekit.finance.report import strategy_section
from regimekit.finance.strategies import StrategyHook, make_strategy, strategy_frame
from regimekit.plugins import REGISTRY
from regimekit.utils.errors import ConfigError, DataError
from tests.helpers import backtest_fixtures as bf

pytestmark = pytest.mark.finance

SHORT = {"initial_train": 100, "refit_every": 30, "horizons": (1,), "restarts": 1, "n_sim": 50}


def test_e10_t07a_strategy_hook_is_an_origin_hook_named_strategy() -> None:
    """The hook satisfies the protocol and is keyed `strategy`."""
    hook = StrategyHook(
        make_strategy(StrategyCfg(type="regime_long_flat", flat_if={"regime": 0, "prob_above": 0.5}), 1.0)
    )
    assert isinstance(hook, OriginHook)
    assert hook.name == "strategy"


def test_e10_t07a_finalize_without_rows_is_a_data_error() -> None:
    """No decided position at all: `DataError` (as the runner before the hook)."""
    hook = StrategyHook(
        make_strategy(StrategyCfg(type="regime_long_flat", flat_if={"regime": 0, "prob_above": 0.5}), 1.0)
    )
    with pytest.raises(DataError):
        hook.finalize([])


def test_e10_t07a_the_hook_skips_origins_whose_next_observation_is_missing() -> None:
    """`on_origin` gives `{}` when the next row is masked, so that no position is recorded there."""
    hook = StrategyHook(
        make_strategy(StrategyCfg(type="regime_long_flat", flat_if={"regime": 0, "prob_above": 0.5}), 1.0)
    )
    mask = np.ones((6, 1), dtype=bool)
    mask[3, 0] = False
    data = SimpleNamespace(mask=jnp.asarray(mask))
    posterior = SimpleNamespace(
        log_alpha=jnp.log(jnp.asarray([[0.7, 0.3]])), index=SimpleNamespace(collapse=jnp.eye(2))
    )
    assert hook.on_origin(0, 2, posterior, None, data) == {}  # type: ignore[arg-type]
    row = hook.on_origin(0, 1, posterior, None, data)  # type: ignore[arg-type]
    assert float(row["position"][0]) == 0.0  # P(regime 0) = 0.7 > 0.5: flat


def test_e10_t07a_strategy_metrics_read_extras_and_declare_their_direction() -> None:
    """Metrics need `strategy`, read `ctx["extras"]["strategy"]` and rank by their own `better`."""
    trace = pd.DataFrame(
        {"return": [0.01, -0.02, 0.03], "position": [1.0, 1.0, 1.0], "gross": [0.01, -0.02, 0.03],
         "turnover": [1.0, 0.0, 0.0], "dt": [1.0, 1.0, 1.0]}
    )  # fmt: skip
    extras = {"strategy": trace}
    assert max_drawdown(extras=extras) == pytest.approx(1.0 - np.exp(-0.02), rel=1e-12)
    assert np.isfinite(sharpe(extras=extras))
    with pytest.raises(ConfigError):
        sharpe(extras={})
    assert [m.needs for m in resolve(["sharpe", "hit_rate"])] == [frozenset({"strategy"})] * 2
    assert (better_of("sharpe"), better_of("max_drawdown"), better_of("turnover")) == (
        "higher",
        "lower",
        None,
    )
    assert next(e for e in REGISTRY.entries("metric") if e.name == "sharpe").provenance == "regimekit.finance"


def test_e10_t07a_backtest_with_the_hook_fills_extras_and_the_report_section(tmp_path: Path) -> None:
    """`StrategyHook` in `run_backtest`: `extras["strategy"]`, per-fold strategy metrics, report section."""
    strategy_cfg = StrategyCfg(
        type="regime_long_flat", flat_if={"regime": 0, "prob_above": 0.6}, cost_bps=5.0
    )
    obs = bf.observations(bf.simulate_returns(199, seed=5))
    cfg = bf.make_config(tmp_path, metrics=("log_score", "sharpe", "max_drawdown"), **SHORT)  # type: ignore[arg-type]
    cfg.evaluation.strategy = strategy_cfg
    grid = Preparer(cfg.data).fit(obs, [], train_end=obs.times[-1]).transform(obs, [])
    hook = StrategyHook(make_strategy(strategy_cfg, grid.nominal_dt, 1), strategy_cfg.cost_bps)
    result = run_backtest(cfg, lambda: (obs, []), seed=5, hooks=[hook])
    frame = strategy_frame(result)
    assert frame is not None
    assert frame is result.extras["strategy"]
    assert {"fold", "time", "position", "return", "equity"} <= set(frame.columns)
    per_fold = result.metrics[result.metrics["metric"] == "sharpe"]
    assert (per_fold["horizon"] == 0).all()
    assert len(per_fold) == result.folds.shape[0]
    section = strategy_section({"cfg": cfg, "result": result, "meta": {}})
    assert section.startswith("## Strategy")
    assert "regime_long_flat" in section
    assert strategy_frame(BacktestResult(result.folds, result.metrics, result.summary)) is None
    empty = strategy_section(
        {"cfg": cfg, "result": BacktestResult(result.folds, result.metrics, result.summary), "meta": {}}
    )
    assert "No strategy was configured." in empty
