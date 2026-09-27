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
