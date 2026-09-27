# Estimation

!!! note
    Stub page (E0-T04). Filled in by E5-T01…E5-T04 (objective, optimizers, init, diagnostics); completed in E8-T04.

MAP estimation (ADR-0005) maximizes the log-posterior in the unconstrained parameterization `u` (parameters
section):

$$
\hat{u} = \arg\max_u \; \log p(y_{1:T} \mid \theta = g(u)) + \log p_{\text{prior}}(\theta) + \log\left|\det \frac{\partial \theta}{\partial u}\right|
$$

## The objective (E5-T01)

`estimation/objective.py` builds the scalar the optimizers minimize, for any family and engine:

$$
J(u) = -\ell(\theta(u)) - \log p(\theta(u)) - \log\left|\det \frac{\partial \theta}{\partial u}\right| \quad(\text{MAP}),
\qquad J(u) = -\ell(\theta(u)) \quad(\text{MLE, } \texttt{use\_priors=False}).
$$

$\ell$ is `engine.loglik`, $\log p$ is `ParamManager.log_prior` plus the transition model's own extra term (the
Dirichlet prior on the jump matrix when covariate logits are configured), and the last term is the
change of variables from the constrained $\theta$ to the unconstrained $u$. `make_objective(spec, space, engine,
manager, data, use_priors)` returns an `Objective` with jitted `value(u)` and `value_and_grad(u)`; `spec`,
`engine` and `manager` are static, `space` and `data` are pytree arguments. Passing another dataset of the same
shapes (`value_and_grad(u, data=fold)`) reuses the compilation: the static bookkeeping in `PreparedDataset`
(`times`, `prep_stats`, `snapshot_ids`) never reaches an engine and is replaced by constants before the call.
`check_finite(J, g, u, names)` raises `NumericalError` naming the first non-finite gradient coordinate.

## Optimizers (E5-T02)

`estimation/optimizers.py`'s `minimize(objective, u0, cfg, log) -> RunRecord` drives one run from one starting
point. `cfg.optimizer` selects the algorithm:

- `lbfgs` (default): `optax.lbfgs` with a zoom line search (`optax.scale_by_zoom_linesearch`,
  `max_linesearch_steps=20`), driven by the `value_and_grad_from_state` idiom so the line search's own
  evaluations at the accepted point are reused rather than recomputed.
- `adam`: learning rate `1e-2` under `optax.cosine_decay_schedule` to `0` over `cfg.max_iter` steps. Its
  convergence tests are only evaluated every 10 iterations (its per-step objective is noisier than
  lbfgs's line-search-verified one); disabled entirely (`tol_f`/`tol_g` at or below zero) it simply runs
  the full decay schedule, which is how a "ground truth" reference run is obtained in practice — with
  early stopping enabled, a decaying-to-zero step size makes `|\Delta J| \le \texttt{tol\_f}` trivially
  true near the end of *any* fixed-length run regardless of how close `u` actually is to the optimum.

Convergence, checked once per iteration for `lbfgs` and every 10th for `adam`:

$$
|\Delta J| \le \texttt{tol\_f} \cdot \max(1, |J|) \quad\text{(`reason="tol_f"`)}, \qquad
\lVert \nabla J \rVert_\infty \le \texttt{tol\_g} \quad\text{(`reason="tol_g"`)},
$$

else the run stops at `cfg.max_iter` with `converged=False, reason="max_iter"`. A non-finite `J` or
gradient at any iteration aborts the run — never raises — returning the last known-finite `u` with
`reason="non_finite"` (or `"non_finite_start"` when `u0` itself is already non-finite). The whole
optimizer update (line search included) is fused into one `jax.jit`-ed step per iteration, since an
unfused, eagerly-dispatched line search dominates the wall clock otherwise. `minimize` has no randomness
and never touches an RNG key: identical inputs give bitwise identical output.
