# Data contracts

`regimekit.data.contracts` defines the frozen dataclasses that are the only shared types between
the data layer and the model layer (FR-D07):

- `ObservationSet`, `CovariateSeries` — raw, per-provider data before alignment.
- `PrepStats` — preparation statistics (covariate mean/std, transforms, lags) fitted on the
  training window only.
- `BoundaryConstraints` — a Part II supervision hook; always `None` in 0.1.0.
- `PreparedDataset` — **the only object the model layer consumes**: a JAX pytree of aligned
  observations, `dt`, a mask, optional lagged/standardized covariate blocks per entry point
  (`transition`/`duration`/`emission`), and metadata (`clock`, `nominal_dt`, `prep_stats`,
  `snapshot_ids`, and the Part II hooks `support`, `n_categories`, `labels`, `boundaries`, `group`).

`PreparedDataset.validate()` raises a typed `DataError` (with the offending field's path) on any
contract violation — wrong shape/dtype, non-monotonic times, non-positive `dt`, non-finite
observations where observed, mismatched covariate metadata, or out-of-range boundary indices.
`PreparedDataset.window(start, end)` slices `[start, end)` along `T` without re-fitting
`prep_stats`, used by the backtester's walk-forward folds.

Only `regimekit.data.contracts` may be imported from `model`, `statespace`, `engine`,
`estimation`, or `inference` — enforced by an AST-based layer-rule test (ADR-0012).

## Calendar and alignment

`regimekit.data.calendar` turns raw provider data into the time axis of a `PreparedDataset`:

- **Master grid.** `build_grid(obs, missing_returns, clock=..., bars_per_day=...)` takes the primary
  instrument's timestamps `τ_0 < … < τ_T`. Under `missing_returns: drop`, every row where any series
  has a missing price is removed. Under `error`, the first such row raises a `DataError`. Observation
  `t` covers `(τ_{t-1}, τ_t]`, so a grid of `T + 1` times yields `T` observations.
- **`Δ_t` per clock.** `trading`: `1 / (252 · bars_per_day)` years, constant (holidays and weekends
  do not lengthen a step). `calendar`: elapsed days / 365.25, so a Thursday → Monday holiday gap
  gives `4/365.25`. `steps`: `1.0` everywhere (ADR-0013). `nominal_dt = median(Δ_t)`. The grid is
  flagged `irregular` when `max Δ_t / min Δ_t > 1.5`, which a daily calendar clock is because of
  weekends. The explicit-duration engine warns on irregular grids (ADR-0007).
- **Covariates.** `align_covariate` takes, for each grid time, the last non-NaN print dated at or
  before it (as `merge_asof(direction="backward")` would). It then shifts the result by `lag ≥ 1`
  grid steps, so observation `(τ_t, τ_{t+1}]` sees the value as of `τ_{t+1-lag}` and never a print
  dated after its start. After that, values are carried forward for at most `max_staleness` grid
  steps from the step where the print became visible; older values become NaN, and E2-T07's
  missing policy decides what happens to them. The staleness counter (`0` = fresh, `-1` = no print
  yet) is returned alongside the values for the data-quality report. A print dated before the
  first grid time is aged in median grid spacings, so a long-discontinued series is not treated as
  fresh at the start of the grid. A monthly macro series
  therefore becomes a step function; give it a `max_staleness` of about one month of grid steps and
  a `lag` that covers its publication delay.
