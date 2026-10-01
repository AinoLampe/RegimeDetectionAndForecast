# Validation: the parameter-recovery suite

`regimekit.estimation.recovery` proves end-to-end estimation correctness: simulate `T` observations
from a known truth `theta*`, MAP-fit the simulated data, and check that the fit recovers `theta*`
within the tolerances of the concept's Appendix D. It is the suite the nightly CI job runs
(`-m slow`), and the engine behind the `regimekit selftest` CLI command (E5-T05b).

## Scenarios

Six scenarios, each `N=3` regimes on a daily (`clock="trading"`) grid, `T=5000` observations by
default:

1. **`gbm_negbin`** — GBM emissions, shifted negative-binomial phase durations (`r=(3,3,2)`), order 1,
   no covariates.
2. **`transition_covariates`** — scenario 1 plus two standardized AR(1) transition covariates
   (coefficients `±0.5`).
3. **`duration_covariates`** — scenario 1 plus one duration covariate (`c = (0.3, -0.3, 0.2)`).
4. **`order2`** — scenario 1 with second-order (`duration_conditioning: current`) memory.
5. **`explicit_gamma`** — explicit-duration Gamma sojourns (`engine: explicit`, `D_max=400`) instead
   of the phase representation.
6. **`fixed_drift`** — `emission.mu` fixed at its truth value (not estimated); only `sigma` is free.

Every scenario fixes `constraint: increasing` on `emission.sigma`, which acts as the identifiability
guard (no post-hoc state relabelling is needed).

## Tolerances

Quoted from the concept's Appendix D and never loosened:

| Parameter | Tolerance |
|---|---|
| `emission.sigma` | <= 10% relative error |
| `emission.mu` (when free) | <= 2 standard errors |
| Mean sojourn | <= 25% relative error |
| Jump probabilities | <= 0.15 absolute error |
| Covariate coefficients | sign-correct and <= 2 standard errors |

Standard errors come from the Hessian of the MAP objective (`estimation.report.std_errors`) when it
is positive-definite and JAX can differentiate through it twice. Two cases fall back to the empirical
spread across 5 seeds instead (`run_recovery_suite`'s own fallback, the alternative the concept's
Appendix D documents): a non-positive-definite Hessian, and the `explicit_gamma` scenario, where JAX
has no second-derivative rule for `jax.scipy.special.gammainc`'s shape-parameter gradient (the first
derivative used by the fit itself works; `jax.hessian` does not).

## Pass criterion

A scenario passes if **at least 4 of its 5 seeds** meet every applicable tolerance
(`SuiteResult.passed`). A failing assertion reports, for each failing seed, every failing check's
truth value, estimate, standard error and the rule it failed (`SuiteResult.failures()`).

## Running it

```bash
# The full nightly suite (slow: ~6 scenarios x 5 seeds x T=5000 MAP fits).
uv run pytest tests/recovery/ -m slow -q

# A smaller, faster run (also what REGIMEKIT_RECOVERY_T controls in CI).
REGIMEKIT_RECOVERY_T=1000 uv run pytest tests/recovery/ -m slow -q

# A single scenario, programmatically:
python -c "
from regimekit.estimation.recovery import run_recovery_suite
suite = run_recovery_suite('gbm_negbin')
print(suite.passed, suite.n_pass)
"
```

`tests/unit/test_e5_t05a_recovery_smoke.py` runs the same harness at a tiny `T` in the default
(non-`slow`) suite, to keep coverage up without paying for the full nightly cost.
