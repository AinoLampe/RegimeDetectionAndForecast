"""Finance profile: providers and finance-specific glue on top of the core."""

from regimekit.finance import clocks as clocks  # registers the `trading` clock
from regimekit.finance import config as config  # registers the `provider` source kind
from regimekit.finance.config import FinanceConfig, FinanceDataSection, PriceSource
from regimekit.finance.providers import get_provider
from regimekit.finance.providers import source as source  # registers the `provider` source kind

__all__ = ["FinanceConfig", "FinanceDataSection", "PriceSource", "config", "get_provider"]
