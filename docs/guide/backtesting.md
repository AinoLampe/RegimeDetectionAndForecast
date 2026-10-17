# Backtesting

!!! note
    Stub page (E0-T04). Filled in by E7-T01…E7-T04 (metrics/walk-forward/strategies/CLI); completed in E8-T04.

See [Math → Evaluation](../math/evaluation.md) for the governing metric formulas.

## Metrics

`regimekit.evaluation.metrics` holds a name-resolved registry (`resolve(names)`; an unknown name raises
`ConfigError` at `evaluation.metrics`). A metric is a function of a context (`forecast`, `realized`,
`horizon`, `posterior`, `fit`, `labels`, `strategy`) with declared `needs`; a runner skips metrics whose
needs it cannot meet. Predictive metrics compare the realized cumulative return `Y_h = sum_{k<=h} y_{t+k}`
with the forecast samples of the cumulative return at step `h`.

| Name | Needs | Value |
|---|---|---|
| `log_score` | forecast | log of a Gaussian-KDE density (Silverman bandwidth) of the samples at `Y_h`; higher is better |
| `log_score_mixture` | forecast, fit | `horizon == 1` only: `log sum_i w_i f_i(y)` with the emission family's exact densities (`dt_next`, optional `z_emission_next` in the context). Weights: `mixture_weights` from the context when given (exact, `one_step_mixture_weights`), else the forecast's one-step regime probabilities (Monte-Carlo estimate) |
| `crps` | forecast | sample CRPS `E|X - Y| - 1/2 E|X - X'|`, computed in `O(n log n)` from the sorted samples; lower is better |
| `pit_coverage` | forecast | `pit` (share of samples `<= Y_h`) plus `coverage_<level>` hit indicators for each central interval `(q, 1 - q)` among the configured quantiles; average the indicators for the hit rate, pool `pit` for a KS test |
| `rmse`, `mae` | forecast | `abs(mean[h-1] - Y_h)` per origin; `rmse` declares `aggregate = "rms"` |
| `aic`, `bic`, `icl` | posterior | information criteria of the decoded fit |
| `regime_occupancy` | posterior | share of steps per regime on the Viterbi path |
| `regime_persistence` | posterior | mean Viterbi segment length per regime (regimes that never occur are omitted) |
| `balanced_accuracy` | posterior, labels | mean per-label recall of the Viterbi regimes against `labels`, maximised over relabellings (at most 6 distinct regimes/labels) |
| `sharpe`, `max_drawdown` | strategy | registered by the strategy ticket (E7-T03); until then calling them raises |

The forecast metrics read `forecast.samples`, so the forecast must keep its samples. With `M > 1` series,
scalar metrics return `<name>_<m>` entries.
