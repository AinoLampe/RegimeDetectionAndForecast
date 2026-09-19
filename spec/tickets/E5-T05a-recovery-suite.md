# E5-T05a — Parameter-recovery suite (nightly)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | NFR-04, §1.4 criterion 1 | M3 | §6.4.4, Appendix D |

## Goal
Prove end-to-end estimation correctness: simulate from known `θ*`, fit, and assert recovery within the concept's tolerances for six scenarios.

## Context (self-contained)
Scenarios (`T=5000`, daily `Δ`, `N=3` unless noted, `restarts=4`, 5 seeds each, pass if ≥ 4/5 seeds meet all tolerances):
1. GBM / NB `r=(3,3,2)`, order 1, no covariates.
2. + transition covariates (2 standardized AR(1) covariates, coefficients ±0.5).
3. + duration covariates (1 covariate, `c = (0.3, −0.3, 0.2)`).
4. Order 2 (`duration_conditioning: current`), no covariates.
5. Explicit gamma durations (`engine: explicit`, `D_max=400`).
6. Fixed-drift lattice (`mu` fixed at `(−0.30, 0.00, 0.20)`, σ free increasing).
Tolerances (Appendix D): σ ≤ 10 % rel; μ ≤ 2 SE (when free; SE from E5-T04 or from the 5-seed spread); mean sojourn ≤ 25 % rel; jump probabilities ≤ 0.15 abs; coefficients sign-correct and ≤ 2 SE.
Truth values: `σ* = (0.10, 0.18, 0.35)`, `μ* = (0.10, 0.00, −0.20)` (free in 1–5), mean sojourns `(60, 120, 30)` days, `π*` rows e.g. `(·, 0.7, 0.3), (0.5, ·, 0.5), (0.2, 0.8, ·)`. Use `constraint: increasing` on σ in all scenarios.
Harness: `tests/recovery/harness.py` with `run_scenario(name, seed) -> RecoveryReport` (table of truth/estimate/tolerance/pass) so E5-T05b can reuse it; reports written to `runs/recovery/<scenario>_<seed>.json` in CI artifacts.

## Creates
- `tests/recovery/{harness,scenarios,test_e5_t05a_recovery}.py` (`@pytest.mark.slow`)
- `docs/guide/validation.md` (what the suite checks)

## Acceptance criteria
1. All six scenarios pass ≥ 4/5 seeds in the nightly job.
2. Each failed tolerance is reported with truth/estimate/SE in the assertion message.
3. Total nightly wall time ≤ 60 min on the CI runner (sizes may be reduced via a `REGIMEKIT_RECOVERY_T` env var, default 5000).

## Out of scope
`selftest` CLI (E5-T05b).

## Connections
- Depends on: E5-T03, E6-T01, E4-T06b (scenario 5), E5-T04 (SE)
- Blocks: E5-T05b, E8-T04 (release criterion)
- Relates to: E0-T03 nightly job
