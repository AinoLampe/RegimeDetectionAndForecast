"""Finance alias of an initialization method (E10-T07a): `rollvol_kmeans` is `rolling_kmeans`."""

from __future__ import annotations

from regimekit.estimation.init import rolling_kmeans
from regimekit.plugins import FINANCE_PROVENANCE, REGISTRY

REGISTRY.register("init", "rollvol_kmeans", rolling_kmeans, provenance=FINANCE_PROVENANCE)
