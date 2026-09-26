"""`ExpandedHMMEngine`: log-space forward recursion over the expanded state space (E4-T03a).

Concept refs: §4.7.2 (forward), Appendix C, §6.3. Index convention (CLAUDE.md §3.7): code index `t`
(0-based) is math step `t+1`; `A_t` moves from code index `t-1` to `t`, covariate row `t` is the
lagged value usable at step `t`.

The recursion is one `jax.lax.scan` over `t = 1..T-1` whose body builds `log A_t` from
`space.transition_fn` -- `(T, S, S)` is never materialized. `spec` is a static jit argument,
`space` and `data` are pytrees.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from regimekit.statespace.builders import initial_log_rho
from regimekit.utils.errors import NumericalError
from regimekit.utils.typing import Array

if TYPE_CHECKING:
    from regimekit.data.contracts import PreparedDataset
    from regimekit.model.params import Params
    from regimekit.model.spec import ModelSpec
    from regimekit.statespace.index import StateSpace


def _safe_logsumexp(x: Array, axis: int) -> Array:
    """`logsumexp` that returns `-inf` with a zero (not NaN) gradient on an all-`-inf` slice.

    A phase-`k > 1` state has `alpha = -inf` until the chain has had time to reach it, so its
    incoming column can be entirely `-inf` (CLAUDE.md §5).
    """
    m = jnp.max(x, axis=axis, keepdims=True)
    m_safe = jnp.where(jnp.isfinite(m), m, 0.0)
    total = jnp.sum(jnp.exp(x - m_safe), axis=axis)
    total_safe = jnp.where(total > 0, total, 1.0)
    out = jnp.squeeze(m_safe, axis=axis) + jnp.log(total_safe)
    # NaN inputs must propagate (invariant 5), not collapse to -inf.
    return jnp.where(total > 0, out, jnp.where(jnp.isnan(total), jnp.nan, -jnp.inf))


def _block(x: Array | None, t: int) -> Array:
    """A covariate block, or `(T, 0)` zeros when absent (CLAUDE.md §3.7)."""
    return jnp.zeros((t, 0)) if x is None else x


@partial(jax.jit, static_argnames=("spec",))
def _forward_core(
    theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
) -> tuple[Array, Array]:
    y, dt, mask = data.observations, data.dt, data.mask
    t_len = y.shape[0]
    z_pi, z_d, z_e = (
        _block(data.cov_transition, t_len),
        _block(data.cov_duration, t_len),
        _block(data.cov_emission, t_len),
    )

    # Masked observations contribute zero log-density; sanitize `y` too so a NaN print cannot
    # leak NaN gradients through the untaken branch (double `where`).
    observed = jnp.all(mask, axis=-1)  # (T,)
    y_safe = jnp.where(mask, y, 0.0)
    log_emis = spec.emission.log_prob(y_safe, dt, theta, z_e)  # (T, N)
    log_emis = jnp.where(observed[:, None], log_emis, 0.0)
    ell = jnp.take(log_emis, space.index.regime_of, axis=1)  # (T, S)

    log_rho = initial_log_rho(spec, space, theta, data.nominal_dt)
    alpha_0 = log_rho + ell[0]

    def step(alpha_prev: Array, xs: tuple[Array, Array, Array, Array]) -> tuple[Array, Array]:
        ell_t, z_pi_t, z_d_t, dt_t = xs
        _, log_a = space.transition_fn(theta, z_pi_t, z_d_t, dt_t)
        alpha_t = ell_t + _safe_logsumexp(alpha_prev[:, None] + log_a, axis=0)
        return alpha_t, alpha_t

    _, rest = jax.lax.scan(step, alpha_0, (ell[1:], z_pi[1:], z_d[1:], dt[1:]))
    log_alpha = jnp.concatenate([alpha_0[None, :], rest], axis=0)
    return log_alpha, _safe_logsumexp(log_alpha[-1], axis=0)


def _check_finite(loglik: Array) -> None:
    """Raise `NumericalError` on a non-finite loglik; skipped under tracing (grad/jit callers)."""
    if isinstance(loglik, jax.core.Tracer):
        return
    if not bool(jnp.isfinite(loglik)):
        raise NumericalError("non-finite log-likelihood from the forward recursion", path="loglik")


class ExpandedHMMEngine:
    """HMM inference on the phase/memory-expanded state space (`spec.engine == "expanded"`)."""

    def forward(
        self, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
    ) -> tuple[Array, Array]:
        """`(log_alpha (T, S), loglik)`; `loglik` is a 0-d `Array`.

        Raises:
            NumericalError: if `loglik` is not finite (checked outside jit, only when called with
                concrete arrays).
        """
        log_alpha, loglik = _forward_core(theta, spec, space, data)
        _check_finite(loglik)
        return log_alpha, loglik

    def loglik(self, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset) -> Array:
        """`log p(y_1:T)`, differentiable w.r.t. `theta`."""
        return self.forward(theta, spec, space, data)[1]

    def approx(self) -> None:
        """No approximate-inference plug-in for the expanded engine (FR-X01 hook)."""
        return None
