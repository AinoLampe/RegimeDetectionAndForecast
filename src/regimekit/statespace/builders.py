"""`build_expanded(spec) -> StateSpace`: enumerate `(memory, regime, phase)` (E4-T02a).

Concept refs: §4.5, §4.7.1, Appendix B (`S` formula).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from regimekit.model.params import Params
from regimekit.model.transition import MemoryLayout, build_memory_layout
from regimekit.statespace.index import StateIndex, StateSpace
from regimekit.utils.errors import ConfigError
from regimekit.utils.typing import Array

if TYPE_CHECKING:
    from regimekit.model.spec import ModelSpec


def _shift(memory: tuple[int, ...], current: int, order: int) -> tuple[int, ...]:
    """The new memory tuple (length `order - 1`) once `current` has just been entered."""
    if order == 1:
        return ()
    return (*memory[1:], current)


def build_expanded(spec: ModelSpec) -> StateSpace:
    """Enumerate the expanded state space `(memory, regime, phase)` from a `ModelSpec`."""
    n_states = spec.n_states
    order = spec.order
    layout = build_memory_layout(n_states, order)
    r_rows = layout.tuples.shape[0]
    memories = layout.tuples[:, :-1]  # (R, order-1)

    regime_of: list[int] = []
    phase_of: list[int] = []
    memory_of: list[np.ndarray] = []
    row_of_state: list[int] = []
    row_start = np.empty(r_rows, dtype=np.int64)

    s = 0
    for row_idx in range(r_rows):
        current = int(layout.current_of[row_idx])
        row_start[row_idx] = s
        n_phases = spec.duration.n_phases(current)
        for phase in range(1, n_phases + 1):
            regime_of.append(current)
            phase_of.append(phase)
            memory_of.append(memories[row_idx])
            row_of_state.append(row_idx)
            s += 1

    n_expanded = s
    if order > 2:
        warnings.warn(
            f"transition.order={order} > 2 expands the state space to S={n_expanded} states",
            stacklevel=2,
        )

    regime_of_arr = np.asarray(regime_of, dtype=np.int64)
    phase_of_arr = np.asarray(phase_of, dtype=np.int64)
    memory_of_arr = np.stack(memory_of, axis=0) if order > 1 else np.zeros((n_expanded, 0), dtype=np.int64)
    row_of_state_arr = np.asarray(row_of_state, dtype=np.int64)
    collapse = np.eye(n_states, dtype=np.float64)[regime_of_arr]  # (S, N)

    index = StateIndex(
        regime_of=jnp.asarray(regime_of_arr),
        phase_of=jnp.asarray(phase_of_arr),
        memory_of=jnp.asarray(memory_of_arr),
        collapse=jnp.asarray(collapse),
        row_of_state=jnp.asarray(row_of_state_arr),
        n_expanded=n_expanded,
        n_states=n_states,
        order=order,
    )

    entry_state_of = _build_entry_state_of(layout, memories, row_start, n_states, order)
    n_rates, rate_index_of = _build_rate_index(
        spec.duration_conditioning, n_states, order, regime_of_arr, memory_of_arr
    )
    phase1_idx = np.flatnonzero(phase_of_arr == 1)

    return StateSpace(
        index=index,
        layout=layout,
        n_expanded=n_expanded,
        n_rates=n_rates,
        rate_index_of=jnp.asarray(rate_index_of),
        entry_state_of=jnp.asarray(entry_state_of),
        phase1_idx=jnp.asarray(phase1_idx),
    )


def _build_entry_state_of(
    layout: MemoryLayout, memories: np.ndarray, row_start: np.ndarray, n_states: int, order: int
) -> np.ndarray:
    """`(R, N)`: the `phase == 1` state entered by `target` from each row (`-1` if `target == current`)."""
    row_lookup = {tuple(row): idx for idx, row in enumerate(layout.tuples)}
    r_rows = layout.tuples.shape[0]
    entry_state_of = np.full((r_rows, n_states), -1, dtype=np.int64)
    for row_idx in range(r_rows):
        current = int(layout.current_of[row_idx])
        memory_r = tuple(int(m) for m in memories[row_idx])
        shifted = _shift(memory_r, current, order)
        for target in range(n_states):
            if target == current:
                continue
            entry_row = row_lookup[(*shifted, target)]
            entry_state_of[row_idx, target] = row_start[entry_row]
    return entry_state_of


def _build_rate_index(
    duration_conditioning: str, n_states: int, order: int, regime_of: np.ndarray, memory_of: np.ndarray
) -> tuple[int, np.ndarray]:
    """`(n_rates, rate_index_of (S,))` for `duration_conditioning in {"current", "pair"}`."""
    if duration_conditioning == "current":
        return n_states, regime_of.copy()
    if order < 2:
        raise ConfigError(
            "duration_conditioning='pair' requires transition.order >= 2",
            path="model.transition.duration_conditioning",
        )
    pairs = [(i, j) for i in range(n_states) for j in range(n_states) if i != j]
    pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}
    rate_index_of = np.asarray(
        [pair_to_idx[(int(memory_of[s, -1]), int(regime_of[s]))] for s in range(regime_of.shape[0])],
        dtype=np.int64,
    )
    return len(pairs), rate_index_of


def initial_log_rho(spec: ModelSpec, space: StateSpace, theta: Params) -> Array:
    """`(S,)` initial log-probabilities (`-inf` off the `phase == 1` support)."""
    s = space.n_expanded
    phase1_idx = space.phase1_idx
    n_phase1 = phase1_idx.shape[0]
    if spec.initial == "uniform":
        log_p = jnp.full((s,), -jnp.inf)
        return log_p.at[phase1_idx].set(-jnp.log(n_phase1))
    if spec.initial == "free":
        probs = theta["initial.logits"]  # (S_phase1,), already the constrained simplex values
        log_p = jnp.full((s,), -jnp.inf)
        return log_p.at[phase1_idx].set(jnp.log(probs))
    if spec.initial == "stationary":
        raise NotImplementedError("initial='stationary' is built by E4-T02b (needs the mean A_t)")
    raise ConfigError(f"unknown initial distribution {spec.initial!r}", path="model.initial_state")
