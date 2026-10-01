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
