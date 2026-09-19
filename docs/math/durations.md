# Durations

!!! note
    Stub page (E0-T04). Filled in by E3-T03 (geometric/negbin), E3-T04 (explicit families); completed in E8-T04.

Each `DurationFamily` gives the sojourn-length pmf `p_i(d)` over `d` steps in regime `i` (geometric, shifted
negative-binomial, Poisson, Gamma, Weibull, log-normal, or nonparametric over `D_max`, per family):

$$
P(\text{sojourn} = d \mid z = i) = p_i(d; \theta_i), \qquad \text{mean\_sojourn}(\theta_i) = \sum_{d} d \, p_i(d)
$$
