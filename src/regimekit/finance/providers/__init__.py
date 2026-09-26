"""Market-data provider adapters (yfinance, stooq, fred, cboe, tiingo) and their registry."""

from regimekit.finance.providers import csv as _csv  # noqa: F401  (registers "csv")
from regimekit.finance.providers.base import DataProvider, get_provider, register, registered_providers
from regimekit.finance.providers.csv import CsvProvider

__all__ = ["CsvProvider", "DataProvider", "get_provider", "register", "registered_providers"]
