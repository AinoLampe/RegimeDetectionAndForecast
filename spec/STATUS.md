# Status

## Tickets
<!-- Status: todo | in-progress | done | blocked | needs-owner -->
| ID | Title | Size | Phase | Deps | Status | Commit | Notes |
|---|---|---|---|---|---|---|---|
| E0-T01 | Repository skeleton | S | 0 | — | done | d976269 | AC1/AC2/AC4 verified locally (Windows host, uv 0.11.17, Python 3.13.13); AC3 (uv lock) resolves `jax-cuda12-plugin` universally, install skipped (Linux-only wheels, expected — see Needs CI/network/GPU) |
| E0-T02 | Quality tooling | S | 0 | E0-T01 | done | ac0c2d6 | AC1-AC4 verified locally (`nox -s lint types tests` green, `pre-commit run --all-files` green); coverage gate + markers + hypothesis profile in place |
| E0-T03 | CI workflows | S | 0 | E0-T02 | done | 9bd0cc0 | AC2-AC4 verified locally; AC1 (<10min) unverifiable without a GH Actions run (no remote configured) — YAML validated structurally instead |
| E0-T04 | Docs skeleton + ADR-0001…0010, 0012, 0013 | M | 0 | E0-T01 | done | ad483b2 | AC1-AC3 verified locally; all 12 ADRs reviewed for substance, not just structure; independent review found no issues |
| E0-T05 | JAX runtime bootstrap, RNG, errors, logging | S | 0 | E0-T01 | done | d53f2cc | AC1 guard verified via monkeypatch (natural repro doesn't trigger in installed JAX, see Findings); AC2-AC5 verified with real tests |
| E1-T01 | Data contracts | M | 1 | E0-T05 | done | d0c55e0 | AC1-AC6 verified with real tests; PreparedDataset uses a hand-written pytree registration (not register_dataclass) for `times`, see Findings; independent review caught a PrepStats hash/eq contract bug, fixed |
| E1-T02a | Config schema (pydantic) + validation rules | M | 1 | E1-T01 | done | 54330e6 | AC1-AC5 verified with real tests, Appendix A + steps-clock fixtures, ~15 cross-field rules each with a ConfigError.path test; independent review caught a dead literal-252 constant and imprecise ParamCfg error paths, both fixed |
| E1-T02b | YAML I/O, resolved config, migration hook, `validate-config` | M | 1 | E1-T02a | done | <PENDING> | AC1-AC6 verified with real tests incl. real CLI invocations; independent review caught an invalid --device/--log-level override crashing with a raw pydantic traceback instead of ConfigError+exit2, fixed |
| E1-T03a | Bijectors and priors | M | 1 | E0-T05 | done | a8e70bd | AC1-AC4 verified with real tests; independent review caught a loosened round-trip tolerance (1e-6 vs required 1e-9), fixed |
| E1-T03b | `ParamSpec` / `ParamManager` | M | 1 | E1-T03a | done | a99f87d | AC1-AC5 verified with real tests inc. Appendix B n_free=20; independent review caught an unguarded tied-group shape mismatch and a missing increasing+prior/bounded test, both fixed |
| E1-T04 | Distributions (`dists`) | M | 1 | E0-T05 | done | c4bb1e1 | AC1-AC5 verified with real tests vs scipy; independent review caught a fragile skip-heavy AC3 test, fixed to rejection-sample instead |
| E1-T05 | Model-side contracts | S | 1 | E1-T03b | todo | | |
| E2-T01 | Provider protocol, registry, csv adapter, contract suite | M | 2 | E1-T01 | todo | | |
| E2-T02 | yfinance adapter | M | 2 | E2-T01 | todo | | |
| E2-T03 | stooq adapter | S | 2 | E2-T01 | todo | | |
| E2-T04 | fred + cboe adapters | M | 2 | E2-T01 | todo | | |
| E2-T05 | Parquet cache, snapshot IDs, offline replay | M | 2 | E2-T01 | todo | | |
| E2-T06 | Calendar, Δ, alignment, lagging | M | 2 | E1-T01 | todo | | |
| E2-T07 | `Preparer` fit/transform | M | 2 | E2-T06 | todo | | |
| E2-T08 | `fetch` command + data-quality report | S | 2 | E2-T05, E2-T07, E1-T02b | todo | | |
| E2-T09 | tiingo adapter (optional) | S | 2 | E2-T01 | todo | | |
| E3-T01 | `EmissionFamily` protocol + GBM | M | 2 | E1-T03b, E1-T04 | todo | | |
| E3-T02 | Student-t emission | S | 2 | E3-T01 | todo | | |
| E3-T03 | `DurationFamily` + geometric + negbin | M | 2 | E1-T03b, E1-T04 | todo | | |
| E3-T04 | Explicit duration families | M | 2 | E3-T03 | todo | | |
| E3-T05 | `TransitionModel` (jump, logit link, generator) | M | 2 | E1-T03b | todo | | |
| E3-T06 | Lattice + ordering + post-sort | S | 2 | E3-T01 | todo | | |
| E4-T01 | NumPy reference implementations | M | 2 | E1-T04 | todo | | |
| E4-T02a | `StateIndex` + `build_expanded` | M | 3 | E1-T05, E3-T03, E3-T05 | todo | | |
| E4-T02b | Per-step `A_t` builder | M | 3 | E4-T02a | todo | | |
| E4-T03a | Expanded engine: forward / loglik | M | 3 | E4-T01, E4-T02b, E3-T01 | todo | | |
| E4-T03b | Expanded engine: backward / smoothed / `Posterior` | M | 3 | E4-T03a | todo | | |
| E4-T04 | Augmented posterior utilities | S | 3 | E4-T03b | todo | | |
| E4-T05 | Viterbi + segments + AIC/BIC/ICL | S | 3 | E4-T03b | todo | | |
| E4-T06a | Explicit-duration engine: forward/backward | M | 4 | E4-T01, E3-T04 | todo | | |
| E4-T06b | Explicit engine: segmental Viterbi, chunking, equivalence | M | 4 | E4-T06a | todo | | |
| E4-T07 | Benchmarks, GPU markers, `associative_scan` flag | M | 3 | E4-T03b, E4-T06b | todo | | |
| E5-T01 | Objective (NLL + priors + log-det-J) | M | 4 | E4-T03b, E1-T03b, E4-T06a | todo | | soft dep on E4-T06a is hard per CLAUDE.md §6 |
| E5-T02 | Optimizers (L-BFGS, Adam), convergence | M | 4 | E5-T01 | todo | | |
| E5-T03 | Init strategies + multi-start (`estimation/map.py`) | M | 4 | E5-T02 | todo | | |
| E5-T04 | Fit report and diagnostics | M | 4 | E5-T03, E4-T05 | todo | | |
| E5-T05a | Parameter-recovery suite | M | 4 | E5-T03, E6-T01 | todo | | |
| E5-T05b | `selftest` command | S | 4 | E5-T05a | todo | | |
| E5-T06 | EM for GBM (v1.x) | M | 7 | E5-T03 | todo | | post-release |
| E6-T01 | Simulator | M | 4 | E4-T02b, E3-T01 | todo | | |
| E6-T02 | `decode` API + CLI | S | 5 | E4-T05, E1-T02b | todo | | |
| E6-T03 | Covariate forecasters | M | 5 | E1-T01 | todo | | |
| E6-T04a | Monte-Carlo path sampler | M | 5 | E6-T01, E4-T04, E6-T03 | todo | | |
| E6-T04b | Forecast aggregation + `ForecastResult` | M | 5 | E6-T04a | todo | | |
| E6-T05 | `forecast` / `simulate` CLI + writers | S | 5 | E6-T04b | todo | | |
| E7-T01 | Metric registry + metrics | M | 6 | E6-T04b | todo | | |
| E7-T02a | Fold scheduler | S | 6 | E2-T07 | todo | | |
| E7-T02b | Walk-forward runner | M | 6 | E5-T03, E6-T04b, E7-T01, E7-T02a | todo | | |
| E7-T03 | Strategies + costs + strategy metrics | M | 6 | E7-T02b | todo | | |
| E7-T04 | `backtest` / `compare` CLI + report | M | 6 | E7-T03 | todo | | |
| E8-T01 | `ModelStore` bundle | M | 7 | E5-T03 | todo | | |
| E8-T02 | `Project` facade + CLI completion | M | 7 | E8-T01, E7-T04 | todo | | |
| E8-T03 | End-to-end example + tutorial | M | 7 | E8-T02 | todo | | |
| E8-T04 | Documentation completion + release 0.1.0 | M | 7 | E8-T03 | todo | | stops before `git tag v0.1.0` |
| E9-T01 | `ProductEmission` for M > 1 | M | 7 | E3-T01, E2-T06 | todo | | post-release |
| E9-T02 | Factorial/coupled design spike (ADR-0008) | M | 7 | E4-T02b | todo | | post-release |
| E9-T03 | MS-GARCH emission spike | M | 7 | E3-T01, E4-T03b | todo | | post-release, ADR-0022 |
| E9-T04 | Endogenous covariates design spike | S | 7 | E9-T01 | todo | | post-release, ADR-0023, ends needs-owner |

## Handoff
Phase 0 and all of Phase 1 (E1-T01 through E1-T05) done this session. Phase 1 exit criterion met: config
schema validates Appendix A; `ParamManager` and `dists` fully tested. Phase 2 (E2-*/E3-*/E4-T01) is next.
`.venv` exists at repo root (uv-managed, Python 3.13.13); `uv sync --all-groups` works from a clean checkout.

## Blocked / needs owner

## Spec fixes

## Findings for the owner
- E1-T05: `Posterior`'s `index` field uses an identity-based `_StaticRef` wrapper for its custom pytree
  registration (same pattern as `ParamManager`, E1-T03b), unlike `PrepStats`'s content-based hash (E1-T01).
  Independent review confirmed this cannot cause *wrong* results (identity mismatch only ever forces an
  extra recompile, never a stale/incorrect cache hit), but it is a real, if modest, performance divergence
  from the content-hashing pattern already established for `PrepStats`: whichever ticket first calls the
  same jitted function repeatedly with distinct-but-structurally-identical `StateIndex` objects (most likely
  E7-T02b's walk-forward folds, or E5-T03's multi-start restarts) will pay for a fresh trace every time
  rather than reusing one. Follow-up for whoever wires those hot loops: consider a content-based hash for
  `StateIndex` (e.g. hashing `.tobytes()` of its array fields, mirroring `PrepStats`) if profiling shows this
  matters in practice.
- E1-T02b: independent review found `--device bogus` (or any invalid `--device`/`--log-level` override)
  crashed with an unhandled `pydantic.ValidationError` and Click's generic exit code 1, instead of the
  documented `ConfigError` + exit 2, because `validate_assignment=True` (E1-T02a) raises `ValidationError` on
  assignment, which `validate_config_cmd`'s `except RegimekitError` didn't catch. Fixed by converting
  `pydantic.ValidationError` to `ConfigError(path=...)` inside `_apply_overrides`, the same way
  `parse_project_config` already does (commit e6fc9e7).
- E1-T03b: `constraint="increasing"` requires a single uniform `mode` string, never a per-state tuple — the
  ticket text only literally says "cannot be combined with per-state `fixed`," so rejecting *all* per-state
  tuples (not just ones containing `fixed`) is an interpretation beyond the literal spec. Judgment call: a
  joint ordering transform across states doesn't have a clean per-state-independent-mode semantics (which
  coordinate would a `tied` or `fixed` state's `u`/`theta` even refer to inside one whole-array bijector
  application?), so the restriction was kept; flagged per independent review rather than left silent. Modes
  `free`/`bounded`/`prior` are all supported with `increasing` (tested), just not per-state mixtures.
  Also fixed: two specs sharing a `tie_group` with incompatible domain/cell shape were silently accepted
  (the second spec silently reused the first member's bijector); now raises `ConfigError` (commit 244dc73).
- E1-T02a: created `src/regimekit/utils/clock.py` (`TRADING_DAYS_PER_YEAR`, `CALENDAR_DAYS_PER_YEAR`,
  `days_to_steps`) ahead of its assumed owner (E2-T06/E3-T04), since E1-T02a is the first ticket needing a
  days->steps conversion (sizing `model.d_max`'s default from `init_mean_sojourn_days`) — per CLAUDE.md §3.2's
  own exact spec for the module's contents. E2-T06/E3-T04 should add `future_grid`/`is_irregular` to this same
  file rather than recreating it. `parse_project_config()` converts pydantic's `ValidationError` into
  `ConfigError(path=...)` (reporting only the first error found — multi-error reporting isn't specified by the
  ticket and wasn't implemented); custom cross-field rules raise `ConfigError` directly from
  `@model_validator` methods, which pydantic v2 does not intercept (only `ValueError`/`TypeError`/
  `AssertionError` are wrapped), so they propagate unwrapped with the caller's own dotted path.
- E1-T02a: independent review caught (1) a dead, unused `TRADING_DAYS_PER_YEAR = 252` in
  `config/defaults.py` that duplicated `utils/clock.py`'s constant — exactly the literal CLAUDE.md §3.2
  forbids outside `clock.py` — removed; (2) `ParamCfg`'s own `model_validator` raised `ConfigError` with a
  bare field name (e.g. `path="values"`) since it has no ancestry context, losing which spec/state failed once
  a config has multiple `ParamCfg`s — moved those checks to the top-level cross-field validator with full
  dotted paths (`model.emission.params.mu.values`); (3) added a missing test for `transition.order >= 1`
  (only the separate `order > 2` warning had a test). Commit 32644d8.
- E1-T03a: independent review found `tests/property/test_e1_t03a_bijectors.py`'s round-trip checks
  (`inverse(forward(u)) == u`) used `atol=rtol=1e-6`, three orders of magnitude looser than AC1's required
  `1e-9`, with no justification (CLAUDE.md §6: never loosen a tolerance to get a suite green). Verified 1e-9
  holds for every bijector under x64 and tightened it (commit 4af0c18); no other changes were needed.
- E1-T04: `pyproject.toml`'s pytest `addopts` gains `--import-mode=importlib`. Needed because the ticket's
  own "Creates" list names `test_e1_t04_dists.py` identically in both `tests/unit/` and `tests/property/`,
  which pytest's default `prepend` import mode cannot collect together (module-name collision) without
  `__init__.py` files in every test directory; `--import-mode=importlib` is the standard fix for src-layout
  projects and has no observed side effects on the rest of the suite.
- E1-T04: `disc_lognormal_logpmf`'s cdf helper uses `jax.scipy.special.ndtr` instead of the textbook
  `0.5*(1+erf(z/sqrt(2)))` — found while calibrating test parameter ranges: the erf form loses precision
  catastrophically in the left tail (`erf(z) -> -1`), which showed up as an fake ~1e-1 mismatch against
  scipy for realistic (not even extreme) parameter draws. `ndtr` is scipy/jax's standard numerically stable
  normal-cdf implementation; confirmed via independent review this is a genuine stability fix, not a
  correctness change.
- E1-T04: independent review found the AC3 test (`test_e1_t04_negbin_pmf_mean_matches_analytic_mean`)
  originally skipped 9/20 parametrized seeds (fixed `a` range didn't reliably land in the
  negligible-tail-mass regime the AC requires). Fixed by rejection-sampling `a` within each seed instead of
  skipping, so all 20 cases now exercise the assertion every run (commit a0047aa).
- E1-T01: `PreparedDataset.times` (a `datetime64[ns]` numpy array) cannot be a JAX pytree leaf (unsupported
  dtype under `jax.jit`) and cannot be plain pytree aux data either (a raw multi-element numpy array breaks
  JAX's own aux-equality check during jit tracing — reproduced directly: `ValueError: ambiguous truth value`).
  Resolved with a hand-written `jax.tree_util.register_pytree_node(PreparedDataset, flatten, unflatten)` pair
  instead of the ticket's literal `register_dataclass(..., meta_fields=[...])`; `times` is wrapped in a small
  `_HashableArray` (hash/eq via `.tobytes()`) only inside flatten/unflatten, so `dataset.times` stays a plain
  `np.ndarray` at the public API. All five contract dataclasses also override `__eq__`/`__hash__`, since the
  default dataclass-generated equality on array fields returns an array instead of a bool (also reproduced
  directly) — this is exactly CLAUDE.md §5's "NumPy arrays are unhashable" gotcha, applied more broadly than
  just to pytree registration. Independent review confirmed both deviations are correct and necessary, and
  caught a related bug this introduced: `PrepStats.__hash__` was identity-based while its `__eq__` (needed for
  the same reason) was value-based, violating the hash/eq contract for a class that sits in the pytree's aux
  data — this would have caused spurious `jax.jit` recompilation for structurally-identical-but-distinct
  `PrepStats` instances. Fixed with a content-based hash (commit eb1fb5f) and a regression test
  (`test_e1_t01_jit_of_function_taking_prepared_dataset_works`) asserting no extra compilation.
- E1-T01: the AST layer/profile-rule scanner (`test_e1_t01_layer_rule.py`) is import-based (matches `Import`/
  `ImportFrom` nodes) and would not catch a violation done purely via attribute access after a bare
  `import regimekit` (e.g. `regimekit.data.cache.foo()`); noted as a known, low-risk gap rather than fixed,
  since no code in the repo does this and the pattern is unusual.
- E0-T03: `.github/workflows/pr.yml`'s `docs` job runs `nox -s docs` (`mkdocs build --strict`), which needs
  `mkdocs.yml` + docs content from E0-T04 to exist; it will fail if run before E0-T04 lands. E0-T04 is planned
  later in this same session, so by session end this is resolved, but flagging since E0-T03 was implemented
  first per the phase-row order.
- E0-T03: `gpu.yml`'s "workflow_dispatch only; runs on a self-hosted runner if labelled, otherwise skipped"
  is implemented as plain `runs-on: [self-hosted, gpu]` with no additional `if:` gate. Per independent review,
  GitHub Actions doesn't actually report "skipped" for an unmatched self-hosted runner — the job just stays
  queued indefinitely. Low severity (no self-hosted runner exists for 0.1.0, CLAUDE.md §9: "nothing may require
  a GPU"); left as-is with a comment, flagged here rather than engineered further.
- E0-T05: AC1 ("importing regimekit after `jax.config.update('jax_enable_x64', False)` in a fresh subprocess
  raises `NumericalError`") does not reproduce as literally written: in the installed JAX version (0.11.2),
  `jax.config.update("jax_enable_x64", ...)` is fully dynamic/live, so `setup()`'s own `update(True)` call
  successfully re-enables x64 regardless of any prior `update(False)` in the same process — verified
  empirically (`tests/unit/test_e0_t05_jax_setup.py::test_e0_t05_setup_succeeds_after_prior_x64_disable`
  and `::test_e0_t05_import_in_fresh_subprocess_ends_with_x64_enabled`). The defensive guard itself
  (`utils/jax_setup.py`'s post-update dtype check, which is what actually raises `NumericalError` if x64
  ends up disabled) is verified directly by forcing the failure via monkeypatch
  (`::test_e0_t05_setup_raises_when_x64_check_fails`). Judgment call: implemented and tested the guard's
  real intent rather than stopping, since a viable test approach existed; flagged here per independent
  review rather than left undocumented.
- E0-T05: `EXIT_CODES` omits `PersistenceError`, matching the ticket's literal text and consistent with
  CLAUDE.md §3.8 (this is an open point for E8-T02 to resolve, not E0-T05's problem to fix).
- E0-T05: `.pre-commit-config.yaml`'s `mypy` hook was switched from the isolated `pre-commit/mirrors-mypy`
  repo (set up in E0-T02) to a local `uv run mypy` hook. Reason: the isolated hook environment only had
  `pydantic`/`typer` as `additional_dependencies` and lacked `jax`/`structlog`/`jaxtyping`/`beartype`, so it
  mis-evaluated `# type: ignore` comments needed for untyped `jax.config.update` calls as "unused" —
  inconsistent with the project's own `uv run mypy` (which correctly requires them). Running mypy against
  the project's own uv-managed environment removes the dependency-mismatch class of bug entirely, at the
  cost of the hook no longer being sandboxed from the dev environment (acceptable: same tradeoff
  `noxfile.py` already makes with `venv_backend="none"`). Scope creep into E0-T02's file, logged here per
  independent review.
- E0-T01: this session runs on native Windows, not Linux/WSL2 as CLAUDE.md §9 assumes. `uv sync --all-groups`,
  `uv build` and the import tests were verified here; anything that is genuinely Linux/WSL2-specific (e.g. an
  actual GPU-wheel install) could not be verified and is logged under "Needs CI / network / GPU" instead of
  claimed as done.
- E0-T01: `src/regimekit/cli/app.py` was created with a minimal working `typer.Typer()` + `main()` (not just an
  `__init__.py` stub) so the `regimekit = "regimekit.cli.app:main"` entry point in `pyproject.toml` resolves for
  AC1/AC4. This is one file beyond the ticket's literal "Creates" list (which only lists `__init__.py` per
  package); flagged here per independent review rather than left silent. No other module bodies beyond
  `__init__.py` docstrings were added.
- E0-T02: independent review flagged that a `version` CLI command was added to `cli/app.py` beyond the
  ticket's scope (it was added to give the coverage gate something to exercise). Reverted: `cli/app.py` is
  back to just `app`/`main`, `main()` is `# pragma: no cover`, and a small import-only test
  (`test_e0_t02_cli_app_module.py`) covers the two module-level statements instead. Real CLI commands remain
  for E6-T02/E6-T05/E8-T02.
- E0-T02: no `changelog.d/` fragment was added (same N/A reasoning as E0-T01 below — quality tooling has no
  user-visible behavior of its own).
- E0-T01: no `changelog.d/` fragment was added. Judgment call: an empty, non-functional package skeleton has no
  user-visible behavior yet, so CLAUDE.md §7's "changelog fragment for user-visible changes" is treated as N/A
  here; the first fragment will land with the first ticket that ships real behavior.

## Follow-ups

## Needs CI / network / GPU
- E0-T03 AC1: "PR pipeline completes in < 10 minutes on the skeleton" needs a real GitHub Actions run (no
  remote configured for this repo yet); workflow YAML validated structurally instead
  (`tests/unit/test_e0_t03_workflows.py`).
- E0-T03: `gpu.yml` needs a real self-hosted GPU runner to confirm dispatch behavior; N/A for 0.1.0.
- E0-T01 AC3: `uv sync --extra gpu` install (not just resolution) needs a Linux x86_64 host with matching CUDA
  wheels; `jax-cuda12-plugin` has no `win_amd64` wheel. `uv lock` already resolves the extra successfully
  (universal resolution, confirmed in `uv.lock`).
- E0-T01 AC1: "succeeds on Linux and WSL2" — verified functionally equivalent on Windows/uv 0.11.17 instead;
  needs a Linux/WSL2 run to fully close per the ticket's literal wording.

## Measurements
