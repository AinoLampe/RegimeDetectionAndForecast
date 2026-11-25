"""Finance profile: providers and finance-specific glue on top of the core."""

from regimekit.finance import clocks as clocks  # registers the `trading` clock
from regimekit.finance import config as config
from regimekit.finance import init as init  # registers the `rollvol_kmeans` alias
from regimekit.finance import metrics as metrics  # registers sharpe, max_drawdown, turnover, hit_rate
from regimekit.finance import transforms as transforms  # registers `log_returns`
from regimekit.finance.cache import Cache, SnapshotPolicy
from regimekit.finance.config import FinanceConfig, FinanceDataSection, PriceSource
from regimekit.finance.persistence import extension_payloads
from regimekit.finance.providers import get_provider
from regimekit.finance.providers import source as source  # registers the `provider` source kind
from regimekit.finance.providers.source import FetchContext, make_provider
from regimekit.finance.quality import DataQualityReport, build_quality_report
from regimekit.finance.report import strategy_section
from regimekit.finance.strategies import StrategyHook, make_strategy

__all__ = [
    "Cache",
    "DataQualityReport",
    "FetchContext",
    "FinanceConfig",
    "FinanceDataSection",
    "PriceSource",
    "SnapshotPolicy",
    "StrategyHook",
    "build_quality_report",
    "config",
    "extension_payloads",
    "get_provider",
    "make_provider",
    "make_strategy",
    "strategy_section",
]
