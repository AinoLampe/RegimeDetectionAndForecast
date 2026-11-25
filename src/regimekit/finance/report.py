"""Finance sections of the backtest report (E10-T07a): the strategy table."""

from __future__ import annotations

from typing import Any

import pandas as pd

from regimekit.evaluation.report import md_table
from regimekit.finance.metrics import hit_rate, max_drawdown, sharpe, turnover


def strategy_section(context: dict[str, Any]) -> str:
    """`## Strategy`: the overall statistics of the strategy table (or a note that none was configured)."""
    trace = context["result"].extras.get("strategy")
    if trace is None:
        return "## Strategy\n\nNo strategy was configured.\n"
    s = getattr(context["cfg"].evaluation, "strategy", None)  # the finance evaluation section adds it
    extras = {"strategy": trace}
    overall = {
        "total log-return": float(trace["return"].sum()),
        "final equity": float(trace["equity"].iloc[-1]),
        "sharpe": sharpe(extras=extras),
        "max_drawdown": max_drawdown(extras=extras),
        "turnover": turnover(extras=extras),
        "hit_rate": hit_rate(extras=extras),
    }
    desc = f"`{s.type}`, cost {s.cost_bps:g} bp" if s is not None else "configured strategy"
    table = pd.DataFrame({"statistic": list(overall), "value": list(overall.values())})
    return "\n".join(
        [
            "## Strategy",
            "",
            f"Strategy {desc}; {len(trace)} out-of-sample periods in {trace['fold'].nunique()} folds.",
            "",
            md_table(table),
            "",
        ]
    )
