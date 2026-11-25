"""E8-T03 AC4: the six-command flow on real `^GDAXI` and `^GSPC` data (`-m live`, needs network + FRED key).

Run manually: `uv run pytest -m live tests/live/test_e8_t03_real_data.py`. The configs are the shipped
examples with a shorter history (2015 onwards), fewer restarts and a quick backtest so a run takes minutes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from ruamel.yaml import YAML
from typer.testing import CliRunner

pytest.importorskip("yfinance")

from regimekit.finance.cli import app

pytestmark = pytest.mark.live

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "configs"
_runner = CliRunner()


@pytest.mark.skipif(not os.environ.get("FRED_API_KEY"), reason="FRED_API_KEY is not set")
@pytest.mark.parametrize("example", ["dax_3state_gbm_nb.yaml", "spx_3state_gbm_nb.yaml"])
def test_e8_t03_real_data_flow(example: str, tmp_path: Path) -> None:
    """validate-config, fetch, fit, decode, forecast and backtest all exit 0 on the real index."""
    yaml = YAML(typ="safe")
    cfg = yaml.load((EXAMPLES / example).read_text(encoding="utf-8"))
    cfg["data"]["primary"]["start"] = "2015-01-02"
    cfg["data"]["cache"]["dir"] = str(tmp_path / "cache")
    cfg["project"]["output_dir"] = str(tmp_path / "runs")
    cfg["estimation"].update(restarts=2, max_iter=300, std_errors=False)
    cfg["forecast"].update(horizon=10, n_sim=500)
    cfg["evaluation"]["backtest"].update(initial_train_years=6, refit_every_steps=252, horizons=[1, 5])
    config = tmp_path / "config.yaml"
    with config.open("w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh)
    out = tmp_path / "runs"
    model = out / f"{cfg['project']['name']}.rkmodel"
    for step in (
        ("validate-config", str(config)),
        ("fetch", str(config)),
        ("fit", str(config)),
        ("decode", str(config), "--model", str(model)),
        ("forecast", str(config), "--model", str(model)),
        ("backtest", str(config)),
    ):
        result = _runner.invoke(app, ["--out", str(out), *step])
        assert result.exit_code == 0, f"{step[0]}: {result.output}"
    for name in ("posterior.parquet", "forecast.parquet", "backtest_metrics.parquet", "quality.json"):
        assert (out / name).is_file(), name
