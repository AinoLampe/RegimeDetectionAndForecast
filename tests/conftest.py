"""Shared pytest configuration.

Sets `REGIMEKIT_TYPECHECK=1` so `@jaxtyped(typechecker=beartype)` is active during tests, imports
`regimekit` before any JAX array is created (x64 must be enabled first, see `utils.jax_setup`,
E0-T05), and registers a `deadline=None` Hypothesis profile (first-call JIT compilation exceeds the
default deadline).
"""

import os

os.environ.setdefault("REGIMEKIT_TYPECHECK", "1")

from hypothesis import HealthCheck, settings

import regimekit

settings.register_profile(
    "regimekit",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("regimekit")

_ = regimekit.__version__  # touch the import so linters don't flag it as unused
