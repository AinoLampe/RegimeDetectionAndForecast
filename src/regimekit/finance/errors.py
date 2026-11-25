"""Finance exceptions (E10-T07b): `ProviderError` moved out of the core error hierarchy."""

from __future__ import annotations

from regimekit.utils.errors import DataError


class ProviderError(DataError):
    """A market-data provider failed to fetch or parse data (exit code 5 in the finance CLI)."""
