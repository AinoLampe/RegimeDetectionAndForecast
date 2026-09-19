# Parameters: bijectors and priors

!!! note
    Stub page (E0-T04). The bijector table and full derivations are added by E1-T03a; completed in E8-T04.

Unconstrained parameters `u` map to constrained parameters `\theta` through a domain-specific bijector
`g`, with the objective (E5-T01) correcting for the change of variables via the log-determinant-Jacobian:

$$
\theta = g(u), \qquad \log p(\theta) = \log p_{\text{prior}}(\theta) + \log \left| \det \frac{\partial \theta}{\partial u} \right|
$$
