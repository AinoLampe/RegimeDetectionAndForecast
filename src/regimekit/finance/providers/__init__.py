"""Market-data provider adapters (yfinance, stooq, fred, cboe, tiingo) and their registry.

Importing a provider class registers it under its `name`.
"""

from regimekit.finance.providers.base import DataProvider, get_provider, register, registered_providers
from regimekit.finance.providers.csv import CsvProvider
from regimekit.finance.providers.stooq import StooqProvider
from regimekit.finance.providers.yfinance import YFinanceProvider

__all__ = [
    "CsvProvider",
    "DataProvider",
    "StooqProvider",
    "YFinanceProvider",
    "get_provider",
    "register",
    "registered_providers",
]
