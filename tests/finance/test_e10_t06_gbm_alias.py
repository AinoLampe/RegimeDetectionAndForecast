"""E10-T06: `gbm` is the finance alias of `gaussian(scale_by_dt=True, ito_correction=True)`."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from regimekit.config.schema import EmissionCfg, ParamCfg
from regimekit.config.yaml_io import load_config
from regimekit.finance.config import FinanceConfig
from regimekit.finance.emission.gbm import GBMFamily
from regimekit.model.build import EMISSION_FAMILIES, BuildInputs, build_model_spec_from
from regimekit.model.emission.gaussian import GaussianFamily
from regimekit.plugins import REGISTRY
from regimekit.utils.errors import ConfigError

pytestmark = pytest.mark.finance

ROOT = Path(__file__).resolve().parents[2]

N, T = 3, 50
MU = np.array([[0.05, -0.02], [0.01, 0.03], [-0.04, 0.0]])
SIGMA = np.array([[0.1, 0.2], [0.15, 0.25], [0.3, 0.4]])


def test_e10_t06_gbm_is_registered_by_finance_and_is_a_gaussian() -> None:
    """Provenance finance; subclass of the core family with the fixed options."""
    entry = next(e for e in REGISTRY.entries("emission") if e.name == "gbm")
    assert entry.provenance == "regimekit.finance"
    assert EMISSION_FAMILIES["gbm"] is GBMFamily
    fam = GBMFamily()
    assert isinstance(fam, GaussianFamily)
    assert (fam.scale_by_dt, fam.ito_correction, fam.native_time) == (True, True, "continuous")


@pytest.mark.parametrize("dt_value", [1 / 252, 1 / 52, 1.0])
@pytest.mark.parametrize("m", [1, 2])
def test_e10_t06_gaussian_with_gbm_options_is_bitwise_the_gbm_density(dt_value: float, m: int) -> None:
    """AC1: bitwise equal log-densities on daily / weekly / steps grids (`T=50, N=3, M in {1, 2}`)."""
    rng = np.random.default_rng(m)
    y = jnp.asarray(rng.normal(0.0, 0.2, size=(T, m)))
    dt = jnp.full((T,), dt_value)
    gbm_theta = {"emission.mu": jnp.asarray(MU[:, :m]), "emission.sigma": jnp.asarray(SIGMA[:, :m])}
    gau_theta = {"emission.mean": gbm_theta["emission.mu"], "emission.scale": gbm_theta["emission.sigma"]}
    a = np.asarray(GBMFamily().log_prob(y, dt, gbm_theta, None))
    b = np.asarray(GaussianFamily(scale_by_dt=True, ito_correction=True).log_prob(y, dt, gau_theta, None))
    assert np.array_equal(a, b)


def test_e10_t06_gbm_keeps_the_part_i_parameter_and_config_names() -> None:
    """Spec names `emission.mu/sigma/gamma/delta`; config keys `mu`/`sigma`, covariates `drift`/`log_vol`."""
    cfg = EmissionCfg(
        family="gbm",
        params={"mu": ParamCfg(mode="free"), "sigma": ParamCfg(mode="free", constraint="increasing")},
        covariates={"drift": ["a"], "log_vol": ["a"]},  # type: ignore[arg-type]
    )
    fam, specs = GBMFamily.from_config(cfg, 3, 1, ("a",))
    assert [s.name for s in specs] == ["emission.mu", "emission.sigma", "emission.gamma", "emission.delta"]
    assert (fam.drift_idx, fam.logvol_idx) == ((0,), (0,))
    assert fam.series_param_names == ("emission.mu", "emission.sigma")
    assert next(s for s in specs if s.name == "emission.sigma").shape == (3,)
    assert next(s for s in specs if s.name == "emission.sigma").init is not None
    assert float(np.asarray(GBMFamily().param_specs(3, 1, ())[1].init)[0, 0]) == 0.1


def test_e10_t06_gbm_options_cannot_be_changed_by_the_config() -> None:
    """`scale_by_dt: false` on a gbm config contradicts the family and is reported when the model is built."""
    cfg = load_config(ROOT / "tests/fixtures/configs/v2/appendix_a_v2.yaml", FinanceConfig)
    cov_names = {
        ep: tuple(c.name for c in cfg.data.covariates if ep in c.use_in)
        for ep in ("transition", "duration", "emission")
    }
    inputs = BuildInputs(n_series=1, cov_names=cov_names, clock="trading", nominal_dt=1 / 252)
    spec = build_model_spec_from(cfg, inputs)
    assert isinstance(spec.emission, GBMFamily)
    bad = cfg.model_copy(
        update={
            "model": cfg.model.model_copy(
                update={"emission": cfg.model.emission.model_copy(update={"scale_by_dt": False})}
            )
        }
    )
    with pytest.raises(ConfigError) as info:
        build_model_spec_from(bad, inputs)
    assert info.value.path == "model.emission.scale_by_dt"
