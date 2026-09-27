"""Shared one-step-ahead predictive diagnostics: PIT and Ljung-Box (E5-T04; reused by E7-T01).

Concept refs: §4.10 Diagnostics. `predictive_regime_probs` computes `P(regime_t | y_{1:t-1})` from an
already-computed `Posterior` (its `log_alpha` is the *filtered* joint, `P(y_{1:t}, s_t)`): step 0 is the
prior `softmax(log_rho)`; step `t >= 1` reuses the same per-step `A_t` builder the engines use
(`space.transition_fn`), applied to `softmax(log_alpha[t-1])`, following the same covariate/`dt` timing
as the forward recursion (CLAUDE.md §3.7: step `t`'s transition uses covariate/`dt` row `t`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy.special
import numpy as np

from regimekit.statespace.builders import initial_log_rho

if TYPE_CHECKING:
    from regimekit.data.contracts import PreparedDataset
    from regimekit.engine.base import Posterior
    from regimekit.model.params.manager import Params
    from regimekit.model.spec import ModelSpec
    from regimekit.statespace.index import StateSpace
    from regimekit.utils.typing import Array


def _block(x: Array | None, t: int) -> Array:
    return jnp.zeros((t, 0)) if x is None else x


def predictive_regime_probs(
    posterior: Posterior, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
) -> Array:
    """`(T, N)`: `P(regime_t = i | y_{1:t-1})`, the one-step-ahead predictive regime distribution."""
    t_len = data.observations.shape[0]
    log_rho = initial_log_rho(spec, space, theta, data.nominal_dt)
    pred_0 = jax.nn.softmax(log_rho)
    if t_len == 1:
        state_probs = pred_0[None, :]
    else:
        z_pi = _block(data.cov_transition, t_len)[1:]
        z_d = _block(data.cov_duration, t_len)[1:]
        dt = data.dt[1:]
        log_alpha_prev = posterior.log_alpha[:-1]

        def step(carry: None, xs: tuple[Array, Array, Array, Array]) -> tuple[None, Array]:
            del carry
            la_prev, z_pi_t, z_d_t, dt_t = xs
            a_t, _log_a = space.transition_fn(theta, z_pi_t, z_d_t, dt_t)
            return None, jax.nn.softmax(la_prev) @ a_t

        _, rest = jax.lax.scan(step, None, (log_alpha_prev, z_pi, z_d, dt))
        state_probs = jnp.concatenate([pred_0[None, :], rest], axis=0)
    out: Array = state_probs @ space.index.collapse
    return out


def pit_values(
    predictive_regime_probs_: Array, spec: ModelSpec, theta: Params, data: PreparedDataset, series: int = 0
) -> Array:
    """`(T,)`: PIT `F_t(y_t) = sum_i P(regime_t = i | y_{1:t-1}) F_i(y_t)` for one series.

    `F_i` is the emission family's own `cdf` (per-regime predictive CDF). Single-series only: PIT is a
    univariate calibration check, so `series` selects which of the `M` observed series to evaluate.
    `y` is sanitized (`NaN` at masked rows -> `0.0`, like `engine/expanded_hmm.py`'s `_regime_log_emis`)
    before `cdf`, and the corresponding PIT entries are set to `NaN` (never a fabricated real number)
    so callers can filter them out rather than silently treating an unobserved step as calibrated.
    """
    t_len = data.observations.shape[0]
    z_e = _block(data.cov_emission, t_len)
    observed = jnp.all(data.mask[:, series : series + 1], axis=-1)
    y_safe = jnp.where(data.mask, data.observations, 0.0)
    cdf_t = spec.emission.cdf(y_safe, data.dt, theta, z_e)[:, :, series]  # (T, N)
    pit = jnp.clip(jnp.sum(predictive_regime_probs_ * cdf_t, axis=-1), 0.0, 1.0)
    out: Array = jnp.where(observed, pit, jnp.nan)
    return out


def pit_histogram(pit: Array | np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, float, float]:
    """`(hist (n_bins,), ks_statistic, ks_p_value)`: PIT histogram and a KS test against `Uniform(0, 1)`.

    The p-value uses the standard asymptotic Kolmogorov distribution (no `scipy` dependency): `p =
    2 sum_{k=1}^inf (-1)^(k-1) exp(-2 k^2 n D^2)`, truncated at `k = 100` (converges within machine
    precision well before that for any `D` seen in practice). `NaN` entries (masked/unobserved steps,
    see `pit_values`) are dropped first.
    """
    values = np.asarray(pit)
    values = values[np.isfinite(values)]
    n = values.shape[0]
    hist, _ = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    sorted_v = np.sort(values)
    ecdf_upper = np.arange(1, n + 1) / n
    ecdf_lower = np.arange(0, n) / n
    d_stat = float(np.max(np.maximum(ecdf_upper - sorted_v, sorted_v - ecdf_lower)))
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d_stat
    k = np.arange(1, 101)
    p = 2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * k**2 * lam**2))
    return hist, d_stat, float(np.clip(p, 0.0, 1.0))


def ljung_box(
    residuals: Array | np.ndarray, lags: tuple[int, ...] = (10, 20)
) -> dict[int, tuple[float, float]]:
    """`{lag: (Q_statistic, p_value)}`: Ljung-Box portmanteau test for residual autocorrelation.

    `Q = n(n+2) sum_{k=1}^{lag} rho_k^2 / (n-k)`, asymptotically `chi2(lag)` under the null of no
    autocorrelation; the p-value is the upper tail `1 - gammainc(lag / 2, Q / 2)` (regularized lower
    incomplete gamma, `jax.scipy.special.gammainc`, matching a chi-square CDF). `NaN` entries
    (masked/unobserved steps) are dropped first (a small gap in the lag structure, not a NaN leak).
    """
    r = np.asarray(residuals, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = r.shape[0]
    r = r - r.mean()
    denom = np.sum(r**2)
    out: dict[int, tuple[float, float]] = {}
    for lag in lags:
        if lag >= n:
            out[lag] = (float("nan"), float("nan"))
            continue
        acf = np.array([np.sum(r[k:] * r[:-k]) / denom for k in range(1, lag + 1)])
        q_stat = float(n * (n + 2) * np.sum(acf**2 / (n - np.arange(1, lag + 1))))
        p_value = float(1.0 - jax.scipy.special.gammainc(lag / 2.0, q_stat / 2.0))
        out[lag] = (q_stat, p_value)
    return out
