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
| E1-T01 | Data contracts | M | 1 | E0-T05 | todo | | |
| E1-T02a | Config schema (pydantic) + validation rules | M | 1 | E1-T01 | todo | | |
| E1-T02b | YAML I/O, resolved config, migration hook, `validate-config` | M | 1 | E1-T02a | todo | | |
| E1-T03a | Bijectors and priors | M | 1 | E0-T05 | todo | | |
| E1-T03b | `ParamSpec` / `ParamManager` | M | 1 | E1-T03a | todo | | |
| E1-T04 | Distributions (`dists`) | M | 1 | E0-T05 | todo | | |
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
Phase 0 complete: E0-T01..E0-T05 all done. Phase 1 next: E1-T01 (data contracts) is the only ticket whose
deps (E0-T05) are satisfied; E1-T03a and E1-T04 also only depend on E0-T05 and can follow in any order.
`.venv` exists at repo root (uv-managed, Python 3.13.13); `uv sync --all-groups` works from a clean checkout.

## Blocked / needs owner

## Spec fixes

## Findings for the owner
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
