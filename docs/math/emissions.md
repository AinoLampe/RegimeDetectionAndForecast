# Emissions

!!! note
    Stub page (E0-T04). Filled in by E3-T01 (GBM), E3-T02 (Student-t); completed in E8-T04.

Each `EmissionFamily` gives the per-step, per-regime log-density of the observation `y_t` given the active
regime `i`, in continuous-time parameters converted to the step's `\Delta_t` (ADR-0001):

$$
\log p(y_t \mid z_t = i, \Delta_t) = \ell_i(y_t; \theta_i, \Delta_t)
$$
