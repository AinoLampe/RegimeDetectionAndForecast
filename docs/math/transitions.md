# Transitions

!!! note
    Stub page (E0-T04). Filled in by E3-T05 (`TransitionModel`); completed in E8-T04.

Between-regime jump probabilities, optionally covariate-dependent via a logit link, conditioned on a memory
layout for higher-order chains:

$$
P(z_{t} = j \mid z_{t-1} = i,\, z_{t} \neq i,\, x_t) = \operatorname{softmax}_j\big(a_{ij} + b_{ij}^\top x_t\big)
$$
