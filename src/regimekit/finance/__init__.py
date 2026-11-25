"""Finance profile: providers and finance-specific glue on top of the core."""

from regimekit.finance import clocks as clocks  # registers the `trading` clock
from regimekit.finance import config as config
from regimekit.finance.cache import Cache, SnapshotPolicy
from regimekit.finance.config import FinanceConfig, FinanceDataSection, PriceSource
from regimekit.finance.providers import get_provider
from regimekit.finance.providers import source as source  # registers the `provider` source kind
from regimekit.finance.providers.source import FetchContext, make_provider

__all__ = [
    "Cache",
    "FetchContext",
    "FinanceConfig",
    "FinanceDataSection",
    "PriceSource",
    "SnapshotPolicy",
    "config",
    "get_provider",
    "make_provider",
]
