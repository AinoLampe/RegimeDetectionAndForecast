# Engines

!!! note
    Stub page (E0-T04). Filled in by E4-T03a/b (expanded), E4-T06a/b (explicit duration); completed in E8-T04.

The forward recursion in log space, over the expanded state space built by `statespace` (ADR-0004):

$$
\log \alpha_t(j) = \log\!\left(\sum_i \alpha_{t-1}(i)\, A_t(i,j)\right) + \log b_j(y_t), \qquad
\log p(y_{1:T}) = \operatorname{logsumexp}_j \log \alpha_T(j)
$$
