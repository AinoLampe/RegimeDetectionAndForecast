"""`TransitionModel` protocol stub and `MemoryLayout` (E1-T05; bodies land in E3-T05).

`MemoryLayout` is a real (not stub) dataclass per CLAUDE.md §3.1: `jump_probs(theta, z_pi,
memory: MemoryLayout)` needs concrete field access for `ModelSpec`/`ParamManager` to type-check
now. Its exact fields are a placeholder shape; E3-T05 owns finalizing them and may adjust this
file (logged as a follow-up in STATUS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from regimekit.model.params import Params
from regimekit.utils.typing import Array


@dataclass(frozen=True)
class MemoryLayout:
    """Maps each expanded state's recent-regime history to the jump-matrix row it uses.

    Placeholder shape (E1-T05); `order = 1` chains use `row_of_state = regime_of` trivially.
    """

    n_states: int
    order: int
    row_of_state: np.ndarray  # (S,) int: jump-matrix row index each expanded state uses


class TransitionModel(Protocol):
    """Structural type for the jump/generator transition model."""

    mode: Literal["jump", "generator"]

    def jump_probs(self, theta: Params, z_pi: Array, memory: MemoryLayout) -> Array:
        """Jump probability matrix/rows for the whole sequence, given covariates `z_pi`."""
        ...

    def jump_probs_t(self, theta: Params, z_pi_t: Array, memory: MemoryLayout) -> Array:
        """Per-step wrapper of `jump_probs`, for use inside a `jax.lax.scan` body."""
        ...
