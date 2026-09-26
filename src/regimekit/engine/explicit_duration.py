"""`ExplicitDurationEngine`: Guedon/Yu explicit-duration forward-backward with right-censoring (E4-T06a).

Concept refs: §4.7.3, Appendix C, ADR-0007. Segment-based recursion over boundaries `tau = 0..T`
(`tau` steps consumed; a segment starting at boundary `s` covers steps `s..s+d-1`), with cumulative
emission sums `C[tau, j] = sum_{u < tau} ell[u, j]` so a segment's emission is `C[s+d, j] - C[s, j]`:

    E[t, j]     = C[t, j] + logsumexp_d (G[t-d, j] + lp_{t-d}[j, d])        segment ends exactly at t
    inner[s, j] = log rho_j (s = 0)  |  logsumexp_i (E[s, i] + log pi_s[i, j])   segment starts at s
    G[s, j]     = inner[s, j] - C[s, j]

`lp_s[j, d]` is the log duration pmf and `log pi_s` the log jump matrix of a segment *starting* at `s`
(entry-time covariates, CLAUDE.md §3.7: covariate row `s` is usable at step `s`). Forward is a scan over
`s` that scatters each segment start into a rolling `(D_max, N)` buffer of pending segment ends, so a
step costs `O(N^2 + N D_max)`. The last segment may still be in progress at `T`: the log-likelihood uses
the survival `log P(d >= T - s)` for it. The augmented state is `(regime, elapsed)` with
`s = j D_max + (e - 1)`; `log_alpha[t, (j, e)] = log P(y_{1:t}, regime j, e steps elapsed at t)`,
`smoothed` the elapsed posterior `P(z_t = j, e | y)` and `log_beta = log(smoothed) + loglik - log_alpha`
(exact: the `(regime, elapsed)` chain is Markov, so `alpha * beta` is the joint).

The grid is assumed regular: `nominal_dt` replaces `dt` everywhere and an irregular grid warns once
per call. Full `(T, N, D_max)` tables are returned, so `T * N * D_max > 5e7` raises `NumericalError`
(chunked scan: E4-T06b).
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from regimekit.engine.base import Posterior
from regimekit.engine.expanded_hmm import _check_finite, _check_posterior, _regime_log_emis, _safe_logsumexp
from regimekit.utils.clock import is_irregular
from regimekit.utils.errors import ConfigError, NumericalError
from regimekit.utils.typing import Array

if TYPE_CHECKING:
    from regimekit.data.contracts import PreparedDataset
    from regimekit.model.params import Params
    from regimekit.model.spec import ModelSpec
    from regimekit.statespace.index import StateSpace

MAX_CELLS = 5e7
_NEG_INF = -jnp.inf


class _Inputs(NamedTuple):
    """Recursion inputs: `(N,)` initial, `(T+1, N)` cumulative emissions, jump and pmf tables."""

    log_rho: Array  # (N,)
    cum: Array  # (T+1, N)
    log_pi: Array  # (N, N) or (T, N, N), indexed by segment start
    log_pmf: Array  # (N, D) or (T, N, D), indexed by segment start


def _logaddexp(a: Array, b: Array) -> Array:
    """`log(exp(a) + exp(b))` that is `-inf` (with zero gradient) when both are `-inf`."""
    a, b = jnp.broadcast_arrays(a, b)
    return _safe_logsumexp(jnp.stack([a, b]), axis=0)


def _log_survival(log_pmf: Array) -> Array:
    """`log P(d >= e)` at `[..., j, e-1]`: reverse cumulative `logsumexp` over the duration axis."""
    return jax.lax.cumlogsumexp(log_pmf, axis=log_pmf.ndim - 1, reverse=True)


def _log_jump(spec: ModelSpec, space: StateSpace, theta: Params, data: PreparedDataset) -> Array:
    """`log pi`: `(N, N)` when constant, `(T, N, N)` with transition covariates (row = segment start)."""
    z_pi = data.cov_transition
    if z_pi is None or z_pi.shape[-1] == 0 or "transition.pi_rows" in theta:
        pi = spec.transition.jump_probs_t(theta, None, space.layout)
    else:
        pi = spec.transition.jump_probs(theta, z_pi, space.layout)
    with_zero_safe = jnp.where(pi > 0, pi, 1.0)
    return jnp.where(pi > 0, jnp.log(with_zero_safe), _NEG_INF)


def _log_pmf(spec: ModelSpec, theta: Params, data: PreparedDataset) -> Array:
    """`(N, D_max)`, or `(T, N, D_max)` with duration covariates (entry-time link at the segment start)."""
    family, d_max, nominal_dt = spec.duration, spec.d_max, data.nominal_dt
    z_d = data.cov_duration
    if z_d is None or z_d.shape[-1] == 0 or not getattr(family, "cov_idx", ()):
        return jnp.asarray(family.log_pmf(theta, d_max, nominal_dt, None))
    return jax.vmap(lambda z: family.log_pmf(theta, d_max, nominal_dt, z))(z_d)


def _log_rho(spec: ModelSpec, theta: Params) -> Array:
    if spec.initial == "uniform":
        return jnp.full((spec.n_states,), -jnp.log(spec.n_states))
    if spec.initial == "free":
        return jnp.log(theta["initial.logits"])
    raise ConfigError(
        "the explicit-duration engine supports initial in {'uniform', 'free'}", path="model.initial_state"
    )


def _prepare(theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset) -> _Inputs:
    """Validate the call (once per public method) and build the recursion inputs."""
    t_len, n, d = data.observations.shape[0], spec.n_states, spec.d_max
    if t_len * n * d > MAX_CELLS:
        raise NumericalError(
            f"T*N*D_max = {t_len * n * d:.3g} > {MAX_CELLS:.0e}: use chunked scan", path="model.d_max"
        )
    if space.index.n_expanded != n * d:
        raise ConfigError(
            "state space does not match the explicit engine; build it with build_explicit(spec)",
            path="model.engine",
        )
    if not isinstance(data.dt, jax.core.Tracer) and is_irregular(data.dt):
        warnings.warn(
            "explicit-duration engine assumes a regular grid: using nominal_dt for an irregular dt",
            UserWarning,
            stacklevel=3,
        )
    return _build_inputs(theta, spec, space, data)


def _build_inputs(theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset) -> _Inputs:
    dt = jnp.full_like(data.dt, data.nominal_dt)
    ell = _regime_log_emis(theta, spec, data, dt)  # (T, N)
    cum = jnp.concatenate([jnp.zeros((1, ell.shape[1])), jnp.cumsum(ell, axis=0)], axis=0)
    return _Inputs(
        _log_rho(spec, theta), cum, _log_jump(spec, space, theta, data), _log_pmf(spec, theta, data)
    )


def _at(x: Array, s: int) -> Array:
    """`x[s]` for a per-start table, `x` itself for a constant one (`ndim` is static)."""
    return x[s] if x.ndim == 3 else x


def _forward(inp: _Inputs) -> tuple[Array, Array]:
    """`(inner (T, N), G (T, N))`: log-prob of a segment starting at each boundary `0..T-1`."""
    cum, lp = inp.cum, inp.log_pmf
    t_len, (d_max, n) = cum.shape[0] - 1, (lp.shape[-1], lp.shape[-2])
    pad = jnp.full((1, n), _NEG_INF)

    def scatter(pend: Array, g: Array, lp_s: Array) -> Array:
        # contributions of a start at s to the ends at s+1..s+D; `pend[k]` is the end at boundary s+1+k
        return _logaddexp(jnp.concatenate([pend[1:], pad], axis=0), g[None, :] + lp_s.T)

    g0 = inp.log_rho  # cum[0] == 0
    pend1 = scatter(jnp.full((d_max, n), _NEG_INF), g0, _at(lp, 0))

    dummy = jnp.zeros((t_len - 1,))
    xs = (cum[1:t_len], inp.log_pi[1:] if inp.log_pi.ndim == 3 else dummy, lp[1:] if lp.ndim == 3 else dummy)

    def body(pend: Array, s_in: tuple[Array, Array, Array]) -> tuple[Array, tuple[Array, Array]]:
        c_s, pi_x, lp_x = s_in
        log_pi_s = pi_x if inp.log_pi.ndim == 3 else inp.log_pi
        end = c_s + pend[0]  # log E[s, :]
        inner = _safe_logsumexp(end[:, None] + log_pi_s, axis=0)
        g = inner - c_s
        return scatter(pend, g, lp_x if lp.ndim == 3 else lp), (inner, g)

    _, (inner_rest, g_rest) = jax.lax.scan(body, pend1, xs)
    inner = jnp.concatenate([inp.log_rho[None, :], inner_rest], axis=0)
    g = jnp.concatenate([g0[None, :], g_rest], axis=0)
    return inner, g


def _terminal(inp: _Inputs) -> Array:
    """`(T, N)`: `log S_s[j, T-s] + C[T, j] - C[s, j]`, a final segment from `s` (`-inf` if `T-s > D`)."""
    cum, lp = inp.cum, inp.log_pmf
    t_len, n, d_max = cum.shape[0] - 1, lp.shape[-2], lp.shape[-1]
    surv = _log_survival(lp)
    length = t_len - jnp.arange(t_len)  # observed final-segment length d = T - s
    idx = jnp.clip(length - 1, 0, d_max - 1)
    if surv.ndim == 2:
        picked = surv[:, idx].T  # (T, N)
    else:
        picked = surv[jnp.arange(t_len)[:, None], jnp.arange(n)[None, :], idx[:, None]]
    value = picked + cum[t_len][None, :] - cum[:t_len]
    return jnp.where((length <= d_max)[:, None], value, _NEG_INF)


def _loglik(inner: Array, term: Array) -> Array:
    return _safe_logsumexp((inner + term).reshape(-1), axis=0)


def _backward(inp: _Inputs, term: Array) -> tuple[Array, Array]:
    """`(B (T, N), cont0 (N,))`: `B[tau, i]` = future log-prob given a regime-`i` segment ends at `tau`."""
    cum, lp, log_pi = inp.cum, inp.log_pmf, inp.log_pi
    t_len, n, d_max = cum.shape[0] - 1, lp.shape[-2], lp.shape[-1]

    def body(hbuf: Array, s_in: tuple[Array, Array, Array, Array]) -> tuple[Array, tuple[Array, Array]]:
        c_s, term_s, pi_x, lp_x = s_in
        pi_s = pi_x if log_pi.ndim == 3 else log_pi
        lp_s = lp_x if lp.ndim == 3 else lp
        closed = _safe_logsumexp(lp_s.T + hbuf, axis=0) - c_s  # (N,), `hbuf[k]` = H[tau+1+k]
        cont = _logaddexp(closed, term_s)
        beta = _safe_logsumexp(pi_s + cont[None, :], axis=1)
        h = c_s + beta
        return jnp.concatenate([h[None, :], hbuf[:-1]], axis=0), (beta, cont)

    dummy = jnp.zeros((t_len,))
    xs = (cum[:t_len], term, log_pi if log_pi.ndim == 3 else dummy, lp if lp.ndim == 3 else dummy)
    _, (beta, cont) = jax.lax.scan(body, jnp.full((d_max, n), _NEG_INF), xs, reverse=True)
    return beta, cont[0]


class _Tables(NamedTuple):
    log_alpha: Array  # (T, S)
    log_beta: Array  # (T, S)
    smoothed: Array  # (T, S)
    loglik: Array  # 0-d


def _elapsed_index(t_len: int, d_max: int) -> tuple[Array, Array, Array]:
    """`(start_c (T, 1, D), valid (T, 1, D), e (1, D))`: start of the segment covering `t` at elapsed `e`."""
    e = jnp.arange(1, d_max + 1)[None, :]
    start = jnp.arange(t_len)[:, None] - e + 1
    return jnp.clip(start, 0, t_len - 1)[:, None, :], (start >= 0)[:, None, :], e


def _log_alpha(inp: _Inputs, g: Array) -> Array:
    """`(T, N, D)`: `log P(y_{1:t}, j, e) = G[s, j] + log S_s[j, e] + C[t+1, j]`, `s = t-e+1`."""
    cum, lp = inp.cum, inp.log_pmf
    t_len, n, d_max = cum.shape[0] - 1, lp.shape[-2], lp.shape[-1]
    start_c, valid, e = _elapsed_index(t_len, d_max)
    jn = jnp.arange(n)[None, :, None]
    surv = _log_survival(lp)
    surv_at = surv[start_c, jn, (e - 1)[None]] if surv.ndim == 3 else surv[None]
    log_alpha = g[start_c, jn] + surv_at + cum[1:][:, :, None]
    return jnp.where(valid, log_alpha, _NEG_INF)


def _tables(inp: _Inputs, inner: Array, g: Array, term: Array, beta: Array, loglik: Array) -> _Tables:
    """Filtered, backward and smoothed `(regime, elapsed)` tables, all `(T, N * D_max)` (regime-major)."""
    cum, lp = inp.cum, inp.log_pmf
    t_len, n, d_max = cum.shape[0] - 1, lp.shape[-2], lp.shape[-1]
    start_c, valid, e = _elapsed_index(t_len, d_max)
    jn = jnp.arange(n)[None, :, None]
    log_alpha = _log_alpha(inp, g)

    # smoothed: mass of all segments starting at s in regime j with duration >= e (+ the final segment)
    h_ext = jnp.concatenate([cum[:t_len] + beta, jnp.full((d_max, n), _NEG_INF)], axis=0)
    h_gather = h_ext[jnp.arange(t_len)[:, None] + jnp.arange(1, d_max + 1)[None, :]]  # (T, D, N)
    lp_full = lp if lp.ndim == 3 else lp[None]
    closed = inner[:, :, None] - cum[:t_len][:, :, None] + lp_full + jnp.swapaxes(h_gather, 1, 2)
    final = (inner + term)[:, :, None]
    weight = _logaddexp(jax.lax.cumlogsumexp(closed, axis=2, reverse=True), final)  # (T, N, D)
    log_elapsed = weight[start_c, jn, (e - 1)[None]]
    finite = valid & jnp.isfinite(log_elapsed)
    smoothed = jnp.where(finite, jnp.exp(jnp.where(finite, log_elapsed - loglik, 0.0)), 0.0)

    log_gamma = jnp.where(smoothed > 0, jnp.log(jnp.where(smoothed > 0, smoothed, 1.0)) + loglik, _NEG_INF)
    has_both = jnp.isfinite(log_gamma) & jnp.isfinite(log_alpha)
    log_beta = jnp.where(has_both, jnp.where(has_both, log_gamma - log_alpha, 0.0), _NEG_INF)
    flat = (t_len, n * d_max)
    return _Tables(log_alpha.reshape(flat), log_beta.reshape(flat), smoothed.reshape(flat), loglik)


@partial(jax.jit, static_argnames=("spec",))
def _loglik_core(theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset) -> Array:
    inp = _build_inputs(theta, spec, space, data)
    inner, _ = _forward(inp)
    return _loglik(inner, _terminal(inp))


@partial(jax.jit, static_argnames=("spec",))
def _forward_core(
    theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
) -> tuple[Array, Array]:
    inp = _build_inputs(theta, spec, space, data)
    inner, g = _forward(inp)
    log_alpha = _log_alpha(inp, g)
    return log_alpha.reshape(log_alpha.shape[0], -1), _loglik(inner, _terminal(inp))


@partial(jax.jit, static_argnames=("spec",))
def _forward_backward_core(
    theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
) -> tuple[Array, Array, Array, Array, Array]:
    inp = _build_inputs(theta, spec, space, data)
    inner, g = _forward(inp)
    term = _terminal(inp)
    loglik = _loglik(inner, term)
    beta, cont0 = _backward(inp, term)
    loglik_b = _safe_logsumexp(inp.log_rho + cont0, axis=0)
    tab = _tables(inp, inner, g, term, beta, loglik)
    return tab.log_alpha, tab.log_beta, tab.smoothed, loglik, loglik_b


class ExplicitDurationEngine:
    """HSMM inference with explicit sojourn pmfs (`engine == "explicit"`); state `(regime, elapsed)`."""

    def forward(
        self, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
    ) -> tuple[Array, Array]:
        """`(log_alpha (T, N * D_max), loglik)`; `loglik` is the right-censored 0-d log-likelihood."""
        _prepare(theta, spec, space, data)
        log_alpha, loglik = _forward_core(theta, spec, space, data)
        _check_finite(loglik)
        return log_alpha, loglik

    def loglik(self, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset) -> Array:
        """`log p(y_1:T)`, differentiable w.r.t. `theta`."""
        _prepare(theta, spec, space, data)
        loglik: Array = _loglik_core(theta, spec, space, data)
        _check_finite(loglik)
        return loglik

    def forward_backward(
        self, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
    ) -> Posterior:
        """Posterior over `(regime, elapsed)`; `Posterior.engine == "explicit"`."""
        _prepare(theta, spec, space, data)
        log_alpha, log_beta, smoothed, loglik, _ = _forward_backward_core(theta, spec, space, data)
        _check_posterior(log_alpha, log_beta, smoothed, loglik)
        return Posterior(
            log_alpha=log_alpha,
            log_beta=log_beta,
            smoothed=smoothed,
            loglik=loglik,
            index=space.index,
            engine="explicit",
        )

    def approx(self) -> None:
        """No approximate-inference plug-in (FR-X01 hook)."""
        return None
