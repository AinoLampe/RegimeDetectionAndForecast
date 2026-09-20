"""`StateIndex`: the expanded state space's index arrays (E1-T05; built from a spec in E4-T02a).

Index arrays are ordinary (non-static) `eqx.Module` fields, so they are JAX pytree **leaves** --
not `eqx.field(static=True)` metadata. `regime_of`/`phase_of`/`memory_of`/`collapse`/
`row_of_state` are plain integer/bool/float arrays (valid JAX dtypes), so treating them as leaves
avoids the "numpy arrays are unhashable" gotcha (CLAUDE.md §5) entirely, while still supporting
`jax.jit`'s shape-based caching (verified: no recompilation across distinct index-array values of
the same shape). Only `n_expanded`/`n_states`/`order` (plain ints) are `static=True`.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from regimekit.utils.typing import Array


def _as_int_array(x: object) -> Array:
    return jnp.asarray(x, dtype=jnp.int32)


def _as_float_array(x: object) -> Array:
    return jnp.asarray(x, dtype=jnp.float64)


class StateIndex(eqx.Module):
    """Static-content (but dynamically-registered) index arrays for the expanded state space."""

    regime_of: Array = eqx.field(converter=_as_int_array)  # (S,) int
    phase_of: Array = eqx.field(converter=_as_int_array)  # (S,) int, 1-based phase within regime
    memory_of: Array = eqx.field(converter=_as_int_array)  # (S, k-1) int; (S, 0) for order 1
    collapse: Array = eqx.field(converter=_as_float_array)  # C (S, N) 0/1, C[s, regime_of[s]] = 1
    row_of_state: Array = eqx.field(converter=_as_int_array)  # (S,) int: jump-matrix row per state
    n_expanded: int = eqx.field(static=True)
    n_states: int = eqx.field(static=True)
    order: int = eqx.field(static=True)
