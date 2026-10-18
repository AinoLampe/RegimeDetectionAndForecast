"""Synthetic look-alike data for the offline example, tutorial and CI (no provider data is committed).

`write_offline_project` simulates three regimes of the Appendix A model (volatilities 12/18/35 %,
drifts -30/0/+20 %, mean sojourns 60/120/30 days) as daily index levels plus a VIX-like and a
spread-like covariate, writes them as csv files, and writes a copy of an example config that reads
them through the `csv` provider. Everything is deterministic in `seed`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

SIGMA = np.array([0.12, 0.18, 0.35])
MU = np.array([-0.30, 0.00, 0.20])
MEAN_SOJOURN = np.array([60.0, 120.0, 30.0])
DT = 1.0 / 252.0

# Cheap settings so an offline run takes minutes, not hours (the shipped example configs keep the
# Appendix A values for real data).
FAST_SETTINGS: dict[str, Any] = {
    "estimation": {"restarts": 2, "max_iter": 300, "std_errors": False},
    "forecast": {"horizon": 10, "n_sim": 500},
    "evaluation": {
        "backtest": {"initial_train_years": 1.5, "refit_every_steps": 126, "horizons": [1, 5]},
    },
}


def simulate_regime_path(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """A 3-regime path with geometric sojourns of the Appendix A means; jumps go to another regime."""
    regimes = np.empty(n_days, dtype=np.int64)
    state, t = int(rng.integers(3)), 0
    while t < n_days:
        length = int(rng.geometric(1.0 / MEAN_SOJOURN[state]))
        regimes[t : t + length] = state
        t += length
        state = int(rng.choice([s for s in range(3) if s != state]))
    return regimes


def write_synthetic_csv(directory: Path, *, years: float = 3.0, seed: int = 11) -> int:
    """Write `SYNTH_INDEX.csv`, `SYNTH_VIX.csv` and `SYNTH_SPREAD.csv` into `directory`; returns the rows."""
    rng = np.random.default_rng(seed)
    n_days = round(years * 252)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    regimes = simulate_regime_path(n_days, rng)
    sigma, mu = SIGMA[regimes], MU[regimes]
    log_ret = (mu - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * rng.standard_normal(n_days)
    close = 3000.0 * np.exp(np.cumsum(log_ret))
    text = dates.strftime("%Y-%m-%d")
    pd.DataFrame({"time": text, "close": close, "adj_close": close}).to_csv(
        directory / "SYNTH_INDEX.csv", index=False
    )
    level = np.empty(n_days)
    level[0] = 15.0
    for t in range(1, n_days):  # mean-reverting VIX-like level whose target rises with volatility
        target = 10.0 + 60.0 * sigma[t]
        level[t] = level[t - 1] + 0.1 * (target - level[t - 1]) + rng.normal(0.0, 0.6)
    pd.DataFrame({"time": text, "value": np.maximum(level, 5.0)}).to_csv(
        directory / "SYNTH_VIX.csv", index=False
    )
    spread = np.sin(np.arange(n_days) / 40.0) + rng.normal(0.0, 0.1, n_days)
    pd.DataFrame({"time": text, "value": spread}).to_csv(directory / "SYNTH_SPREAD.csv", index=False)
    return n_days


def _merge(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value


def write_offline_project(
    workdir: Path,
    example: Path,
    *,
    years: float = 3.0,
    seed: int = 11,
    fast: bool = True,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Write the synthetic csv files and an offline copy of `example` into `workdir`; returns its path.

    The copy reads the csv files through the `csv` provider, caches under `workdir/cache`, writes under
    `workdir/runs`, and (with `fast`) uses `FAST_SETTINGS`. `overrides` is a nested dict merged last.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    write_synthetic_csv(workdir, years=years, seed=seed)
    yaml = YAML(typ="safe")
    cfg = yaml.load(example.read_text(encoding="utf-8"))
    data = cfg["data"]
    data["primary"].update(provider="csv", symbols=["SYNTH_INDEX"], start="2021-01-01", end=None)
    data["cache"]["dir"] = str(workdir / "cache")
    data["covariates"][0].update(provider="csv", series="SYNTH_VIX")
    data["covariates"][1].update(provider="csv", series="SYNTH_SPREAD")
    cfg["project"]["output_dir"] = str(workdir / "runs")
    if fast:
        _merge(cfg, FAST_SETTINGS)
    _merge(cfg, overrides or {})
    path = workdir / "config.yaml"
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh)
    return path
