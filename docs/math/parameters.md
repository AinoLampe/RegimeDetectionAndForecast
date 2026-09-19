# Parameters: bijectors and priors

Unconstrained parameters `u` map to constrained parameters `\theta` through a domain-specific
bijector `g`, with the objective (E5-T01) correcting for the change of variables via the
log-determinant-Jacobian:

$$
\theta = g(u), \qquad \log p(\theta) = \log p_{\text{prior}}(\theta) + \log \left| \det \frac{\partial \theta}{\partial u} \right|
$$

## Bijector table

| Domain | `forward(u)` | Notes |
|---|---|---|
| `real` | `u` | identity |
| `positive` | `softplus(u)` (default) or `exp(u)` (`variant="exp"`) | |
| `interval[lo, hi]` | `lo + (hi - lo) \cdot \text{sigmoid}(u)` | |
| `simplex_ref` | `\text{softmax}(\text{concat}([0], u))` | reference category first; `u` has `n-1` free entries, returns `n` |
| `nu_gt_2` | `2 + \text{softplus}(u)` | Student-t degrees of freedom |
| `pmf` | same construction as `simplex_ref` | nonparametric duration pmf over `1..D_max` |
| `increasing` | `\theta_0 = g(u_0)`, `\theta_i = \theta_{i-1} + \text{softplus}(u_i)` | ordering constraint; `g` is the inner bijector for the base domain (e.g. `positive` for `\sigma`) |

Log-det-Jacobians are analytic sums of scalar terms for every domain except `simplex_ref`/`pmf`,
where they are computed via `jax.jacfwd` + `jnp.linalg.slogdet` on the `n-1` free coordinates
(cheap: called once per objective evaluation). Every bijector's analytic (or autodiff-based)
`log_det_jac` is tested against a full autodiff `slogdet` reference to `\le 10^{-8}`
(`tests/property/test_e1_t03a_bijectors.py`).

## Priors

Evaluated on `\theta`, returning a summed log-density (`model/params/priors.py`):

- `NormalPrior(mean=0, sd)`: `-0.5((\theta - m)/sd)^2 - \log(sd) - 0.5 \log(2\pi)`
- `LogNormalPrior(mean_log, sd)`: the above on `\log \theta`, plus `-\log \theta` (the Jacobian of
  the log transform)
- `DirichletPrior(alpha)`: `\sum_i (\alpha - 1) \log p_i` per jump-matrix row, plus the Dirichlet
  normalizing constant `\log\Gamma(n\alpha) - n\log\Gamma(\alpha)`

Default coefficient prior: `NormalPrior(0, coef_prior_sd=1.0)`.
