"""Finance profile: providers and finance-specific glue on top of the core."""

from regimekit.finance import clocks as clocks  # registers the `trading` clock
from regimekit.finance import config as config
from regimekit.finance import transforms as transforms  # registers `log_returns`
from regimekit.finance.cache import Cache, SnapshotPolicy
from regimekit.finance.config import FinanceConfig, FinanceDataSection, PriceSource
from regimekit.finance.providers import get_provider
from regimekit.finance.providers import source as source  # registers the `provider` source kind
from regimekit.finance.providers.source import FetchContext, make_provider
from regimekit.finance.quality import DataQualityReport, build_quality_report

__all__ = [
    "Cache",
    "DataQualityReport",
    "FetchContext",
    "FinanceConfig",
    "FinanceDataSection",
    "PriceSource",
    "SnapshotPolicy",
    "build_quality_report",
    "config",
    "get_provider",
    "make_provider",
]
