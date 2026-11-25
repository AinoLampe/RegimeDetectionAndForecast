# Finance emissions

## `gbm`: geometric Brownian motion

`gbm` is the finance alias of the core [`gaussian`](../math/emissions.md) emission with
`scale_by_dt: true` and `ito_correction: true` fixed (E10-T06). It keeps the Part I vocabulary, so version-1 and
version-2 finance configurations and 0.1.x / 0.2.x model bundles load unchanged:

| | `gbm` | `gaussian` |
|---|---|---|
| parameters | `emission.mu`, `emission.sigma`, `emission.gamma`, `emission.delta` | `emission.mean`, `emission.scale`, `emission.gamma`, `emission.delta` |
| config keys | `mu`, `sigma` | `mean`, `scale` |
| covariate keys | `drift`, `log_vol` | `mean`, `log_scale` |

$$
y_t \mid z_t = i \sim \mathcal{N}\!\left(\left(\mu_i(t) - \tfrac12\sigma_i(t)^2\right)\Delta_t,\; \sigma_i(t)^2 \Delta_t\right),
\quad \mu_i(t) = \mu_i + \gamma_i^\top z^e_t, \quad \log\sigma_i(t) = \log\sigma_i + \delta_i^\top z^e_t.
$$

`\mu_i` and `\sigma_i` are annualized under the `trading` / `calendar` clocks and per step under `steps`.
The `gbm` family registers when a finance configuration is validated (or when `regimekit.finance.emission` is
imported); the core never imports it. `gbm` ignores `scale_by_dt` / `ito_correction` in the configuration: a
value that contradicts the family is a `ConfigError` at model build.

The experimental MS-GARCH emission (E9-T03, ADR-0022) lives next to it in `regimekit.finance.emission.msgarch`.
