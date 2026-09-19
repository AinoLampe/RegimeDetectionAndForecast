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
