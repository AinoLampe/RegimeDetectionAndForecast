# Durations

!!! note
    Explicit families (E3-T04) still land as a stub page section below.

Each `DurationFamily` gives the sojourn-length pmf `p_i(d)` over `d` steps in regime `i` (geometric, shifted
negative-binomial, Poisson, Gamma, Weibull, log-normal, or nonparametric over `D_max`, per family):

$$
P(\text{sojourn} = d \mid z = i) = p_i(d; \theta_i), \qquad \text{mean\_sojourn}(\theta_i) = \sum_{d} d \, p_i(d)
$$

## Phase representation: `geometric` and `negbin` (E3-T03)

The expanded engine represents a regime's sojourn as a chain of `r_i` phases: the chain enters phase 1 on
regime entry, advances one phase per step with probability `a_i(t)`, and exits (the regime ends) leaving
phase `r_i`. On a constant grid, `d - r_i \sim \text{NegBin}(r_i, a_i)`:

$$
a_i(t) = 1 - \exp(-\lambda_i(t) \, \Delta_t), \qquad
\log \lambda_i(t) = \log \lambda_i + c_i^\top z^d_t
$$

$$
E[d] = \frac{r_i}{a_i}, \qquad \text{Var}[d] = \frac{r_i (1 - a_i)}{a_i^2}, \qquad \min(d) = r_i
$$

`GeometricDuration` is the `r_i = 1` special case (plain HMM sojourn); `advance_prob`/`log_pmf` agree with
`NegBinDuration(r=(1, ..., 1))` exactly, since the advance-probability formula does not depend on `r`.

`\lambda_i` (exit/advance rate, `duration.rate (N,)` positive) is in clock units: per year under the
`trading`/`calendar` clocks, per step under `steps` (`\Delta_t \equiv 1`) — no code branches on the clock
name (CLAUDE.md §3.2). `mean_sojourn(theta, nominal_dt)` reports `E[d] \cdot \text{nominal\_dt}` (renamed
from `mean_sojourn_years`, Part II §15.7/CLAUDE.md §3.1), and `init_from_mean_sojourn(mean_steps,
nominal_dt)` inverts it to a rate: `\lambda_i = r_i / (m_i \cdot \text{nominal\_dt})` (`r_i = 1` for
geometric). `init_mean_sojourn_days` needs a clock/`bars_per_day`-aware days-to-steps conversion that only
`model/build.py` has (CLAUDE.md §3.3); `DurationFamily.from_config` accepts `init_mean_sojourn_steps`
directly and rejects `init_mean_sojourn_days` with a `ConfigError` pointing at that boundary.

A duration covariate link (`duration.c (N, K_d)`, default `NormalPrior(0, coef_prior_sd)`) is evaluated
every step for `advance_prob` (used by the engine's per-step scan) and at a single, fixed entry-time value
`z_entry` for `log_pmf` (the explicit engine's constant-per-segment pmf, ADR-0006's "entry-time evaluation").
