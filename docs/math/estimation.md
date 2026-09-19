# Estimation

!!! note
    Stub page (E0-T04). Filled in by E5-T01…E5-T04 (objective, optimizers, init, diagnostics); completed in E8-T04.

MAP estimation (ADR-0005) maximizes the log-posterior in the unconstrained parameterization `u` (parameters
section):

$$
\hat{u} = \arg\max_u \; \log p(y_{1:T} \mid \theta = g(u)) + \log p_{\text{prior}}(\theta) + \log\left|\det \frac{\partial \theta}{\partial u}\right|
$$
