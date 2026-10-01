# Forecasting

!!! note
    Stub page (E0-T04). Forecasting/simulation are filled in by E6-T03…E6-T05; completed in E8-T04.

See [Math → Forecasting](../math/forecasting.md) for the governing Monte Carlo aggregation formula.

## Decode

`regimekit decode CONFIG --model DIR [--augmented]` turns a fitted model plus `CONFIG`'s prepared
dataset into a decoded posterior: filtered and smoothed regime probabilities, the Viterbi path, its
run-length-encoded segments, and the AIC/BIC/ICL information criteria.

`--model DIR` points at a directory written by `persistence.temp_model.write_temp_model` -- a
temporary `params.npz` + `spec.json` pair that stands in for the real model bundle until E8-T01 lands
(CLAUDE.md §4 invariant 6 forbids adding the `safetensors` dependency ahead of its own ADR-0011, which
E8-T01 owns). `prepared.parquet` is read from the output directory, or fetched first if missing and
`--offline` is not set.

Outputs, written to the output directory:

| File | Contents |
|---|---|
| `posterior.parquet` | `time`, `filtered_0..N-1`, `smoothed_0..N-1`, `viterbi` (`T` rows, `2N+2` columns) |
| `augmented.parquet` | `time` x `S` smoothed expanded-state posterior (only with `--augmented`) |
| `segments.json` | Run-length encoding of the Viterbi regime path: `{"start", "end", "regime"}` (`end` inclusive) |
| `criteria.json` | `loglik`, `aic`, `bic`, `icl` |

The same logic is available as a library call: `regimekit.inference.decode.decode(fit, data) ->
PosteriorResult`.

## Covariate forecasters

Multi-step forecasting needs future values for every covariate the model conditions on
(`forecast.covariate_forecaster`). Four strategies (`regimekit.inference.covariate_forecasters`):

| `type` | Behaviour |
|---|---|
| `freeze` | Every forecast step repeats the last known value. |
| `scenario` | A user-supplied path for named columns (`path`, loaded by the caller -- `inference/` has no file I/O); unmentioned columns freeze. |
| `bootstrap` | A stationary block bootstrap of historical first differences, cumulated forward from the last known value (`mean_block_length`). |
| `ar` | A VAR(1) fit by OLS on the training window, with Gaussian innovations (`diagonal: true` fits each column independently). |

Every forecaster works on the *unified* set of covariate columns -- the union across the
transition/duration/emission blocks, de-duplicated by name (`ColumnMap`) -- since the same covariate
can drive more than one part of the model.
