# Engines

!!! note
    Expanded engine (E4-T03a/b) documented below; explicit-duration engine (E4-T06a/b) and the final pass follow in E8-T04.

The forward recursion in log space, over the expanded state space built by `statespace` (ADR-0004):

$$
\log \alpha_t(j) = \log\!\left(\sum_i \alpha_{t-1}(i)\, A_t(i,j)\right) + \log b_j(y_t), \qquad
\log p(y_{1:T}) = \operatorname{logsumexp}_j \log \alpha_T(j)
$$

## Backward recursion and smoothing

With $\ell_t(s) = \log b_{r(s)}(y_t)$ (regime of state $s$; a fully masked step has $\ell_t \equiv 0$),
the backward pass runs a reverse `lax.scan`, rebuilding $A_{t+1}$ at each step (the $(T, S, S)$ array is never stored):

$$
\log \beta_T(s) = 0, \qquad
\log \beta_t(s) = \operatorname{logsumexp}_{s'}\!\big[\log A_{t+1}(s, s') + \ell_{t+1}(s') + \log \beta_{t+1}(s')\big]
$$

The smoothed posterior and its regime marginal (with the collapse matrix $C$, $C_{s,i} = 1$ iff state $s$ belongs to regime $i$):

$$
\gamma_t(s) = \operatorname{softmax}_s\big(\log \alpha_t(s) + \log \beta_t(s)\big), \qquad
\gamma^{\text{regime}}_t = \gamma_t\, C
$$

Invariants used by the tests: $\operatorname{logsumexp}_s(\log\alpha_t + \log\beta_t) = \log p(y_{1:T})$ for every $t$, and
the log-likelihood does not depend on the ordering of the expanded states. Structural zeros ($-\infty$) are masked
before `exp`, and a `logsumexp` over an all-$-\infty$ slice returns $-\infty$ with a zero gradient.

## Augmented posterior summaries

`engine/augmented.py` buckets the smoothed posterior by `(regime, phase)` (`phase_marginals`, `(T, N, r_max)`) and by
memory row (`memory_marginals`, `(T, R)`). For the negative-binomial phase chain, in phase $c$ of $r_i$ the expected
remaining sojourn is $(r_i - c + 1)/a_i(t)$ steps, and the phase-based lower bound on the elapsed time is
$(c - 1)/a_i(t)$; both are averaged over the phase marginals conditional on regime $i$ (0 where the regime has no mass).
Exact elapsed-duration posteriors come from the explicit-duration engine.
