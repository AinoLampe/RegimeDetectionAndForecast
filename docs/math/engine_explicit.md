# Explicit-duration engine

`engine/explicit_duration.py` (`ExplicitDurationEngine`, `spec.engine == "explicit"`) runs the
Guédon/Yu forward-backward over *segments* for an arbitrary sojourn pmf $p_j(d)$, $d = 1..D_{\max}$.
The grid is assumed regular (ADR-0007): `nominal_dt` replaces $\Delta_t$ everywhere and an irregular
grid warns once per call.

## Recursion

Boundary $\tau \in \{0, \dots, T\}$ means $\tau$ steps consumed; a segment starting at boundary $s$
covers steps $s, \dots, s+d-1$. With $C[\tau, j] = \sum_{u<\tau} \ell_u(j)$ (masked steps add 0), the
emission of that segment is $C[s+d, j] - C[s, j]$ and

$$
E[t, j] = C[t, j] + \operatorname{logsumexp}_d \big( G[t-d, j] + \log p_{t-d}(j, d) \big), \qquad
\text{inner}[s, j] = \begin{cases} \log\rho_j & s = 0 \
\operatorname{logsumexp}_i \big( E[s, i] + \log\pi_s(i, j) \big) & s > 0 \end{cases}
$$

with $G = \text{inner} - C$. $\log p_s$ and $\log\pi_s$ are those of a segment *starting* at $s$
(entry-time covariates use covariate row $s$). The forward pass is one `lax.scan` over $s$ that
scatters each start into a rolling `(D_max, N)` buffer of pending ends: $O(N^2 + N D_{\max})$ per step.

## Right-censoring

The last segment may still be running at $T$. With $S_s(j, d) = P(\text{duration} \ge d)$ from the
same pmf,

$$
\log p(y_{1:T}) = \operatorname{logsumexp}_{s, j} \big( \text{inner}[s, j] + \log S_s(j, T-s)
+ C[T, j] - C[s, j] \big), \qquad T - s \le D_{\max}.
$$

## Backward and smoothing

$B[\tau, i]$ is the log-probability of the future given that a regime-$i$ segment ends at $\tau$. With
$H = C + B$ it needs only the last $D_{\max}$ values of $H$ (a reverse scan), the "difference of
cumulative" trick, and the final segment enters through the survival term above. The elapsed posterior
is, for step $t$ at elapsed $e$ (segment start $s = t - e + 1$),

$$
P(z_t = j, e \mid y) = \exp\!\Big( \operatorname{logsumexp}_{d \ge e} w(s, j, d) \oplus w_T(s, j) - \ell \Big)
$$

where $w$ are the closed-segment weights, $w_T$ the final-segment weight and $\ell$ the log-likelihood.
Suffix sums over $d$ are a reverse `cumlogsumexp`: $O(T N D_{\max})$ in total.

## Augmented state

State $s = j D_{\max} + (e - 1)$ (`statespace.builders.build_explicit`). The `Posterior` fields are

* `log_alpha[t, (j, e)] = log P(y_{1:t}, regime j, e steps elapsed) = G[t-e+1, j] + log S + C[t+1, j]`
  (so `softmax(log_alpha)` is the filter and `logsumexp(log_alpha[-1])` the log-likelihood),
* `smoothed[t, (j, e)]` the elapsed posterior (`Posterior.elapsed_posterior()` is the `(T, N, D_max)` view),
* `log_beta = log(smoothed) + loglik - log_alpha`, exact because `(regime, elapsed)` is Markov.

## Limits

Full `(T, N, D_max)` tables are returned, so $T N D_{\max} > 5 \cdot 10^7$ raises `NumericalError`
("use chunked scan"; chunking is E4-T06b). Only `order == 1`, `duration_conditioning == "current"` and
`initial in {uniform, free}` (over the first segment's regime) are supported.
