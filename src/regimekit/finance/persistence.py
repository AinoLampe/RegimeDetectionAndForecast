"""Finance extension of the model bundle (E10-T07a): provider metadata in `extensions/regimekit.json`."""

from __future__ import annotations

from typing import Any

from regimekit.finance.config import FinanceConfig


def extension_payloads(cfg: FinanceConfig) -> dict[str, Any]:
    """`{"regimekit": {primary, covariates}}`: where the data came from (what `snapshots.json` held in v1)."""
    source = cfg.data.source
    return {
        "regimekit": {
            "primary": {
                "provider": source.option("provider"),
                "symbols": list(source.option("symbols", [])),
                "interval": source.option("interval"),
                "adjusted": source.option("adjusted"),
            },
            "covariates": [
                {
                    "name": c.name,
                    "provider": None if c.source is None else c.source.option("provider"),
                    "series": None if c.source is None else c.source.option("series"),
                }
                for c in cfg.data.covariates
            ],
        }
    }
