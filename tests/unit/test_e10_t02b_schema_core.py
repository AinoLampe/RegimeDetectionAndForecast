"""E10-T02b: the neutral core schema versus the finance subclass (FR-G05, FR-G07, D24)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from ruamel.yaml import YAML

from regimekit.config.schema import ProjectConfig, parse_project_config
from regimekit.config.yaml_io import dump_resolved, export_json_schema, load_config
from regimekit.finance import FinanceConfig
from regimekit.model.params.manager import ParamManager
from regimekit.utils.errors import ConfigError
from tests.helpers.finance_config import parse_v1

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "configs"
_YAML = YAML(typ="safe")


def _v2() -> dict[str, Any]:
    """A fresh copy of the version-2 Appendix A configuration."""
    return copy.deepcopy(_YAML.load(FIXTURES / "v2" / "appendix_a_v2.yaml"))


def test_e10_t02b_appendix_a_v1_migrates_and_equals_the_v2_fixture() -> None:
    """AC1: Appendix A (v1, with `profile`) loads through FinanceConfig and equals `appendix_a_v2.yaml`."""
    v1 = load_config(FIXTURES / "appendix_a.yaml", FinanceConfig)
    v2 = load_config(FIXTURES / "v2" / "appendix_a_v2.yaml", FinanceConfig)
    assert isinstance(v1, FinanceConfig)
    assert v1.schema_version == 2
    assert v1.model_dump(mode="json") == v2.model_dump(mode="json")
    assert v1 == v2


def test_e10_t02b_migrated_config_builds_the_same_model_as_the_v2_fixture() -> None:
    """AC1: the migrated v1 config and the v2 fixture build identical ModelSpecs (param specs, layout)."""
    from regimekit.model.build import BuildInputs, build_model_spec_from

    def build(path: Path) -> Any:  # noqa: ANN401
        cfg = load_config(path, FinanceConfig)
        names = {
            ep: tuple(c.name for c in cfg.data.covariates if ep in c.use_in)
            for ep in ("transition", "duration", "emission")
        }
        return build_model_spec_from(cfg, BuildInputs(1, names, "trading", 1 / 252))

    one, two = build(FIXTURES / "appendix_a.yaml"), build(FIXTURES / "v2" / "appendix_a_v2.yaml")
    layout = [(s.name, tuple(s.shape)) for s in one.param_specs]
    assert layout == [(s.name, tuple(s.shape)) for s in two.param_specs]
    assert ParamManager(one.param_specs).n_free == ParamManager(two.param_specs).n_free > 0
    assert "emission.mu" in [n for n, _ in layout]


@pytest.mark.parametrize(
    ("key_path", "value"),
    [
        (("profile",), "finance"),
        (("data", "primary"), {"provider": "csv", "symbols": ["A"], "interval": "1d", "start": "2020-01-01"}),
        (("data", "bars_per_day"), 1),
        (("evaluation", "strategy"), {"type": "regime_long_flat"}),
    ],
)
def test_e10_t02b_core_rejects_finance_keys(key_path: tuple[str, ...], value: Any) -> None:  # noqa: ANN401
    """AC2: the core ProjectConfig rejects `profile`, `data.primary`, `bars_per_day` and `strategy`."""
    raw = _v2()
    raw["data"]["source"] = {"kind": "provider", **raw["data"]["source"]}
    node = raw
    for part in key_path[:-1]:
        node = node[part]
    node[key_path[-1]] = value
    with pytest.raises(ConfigError) as err:
        parse_project_config(raw)
    assert err.value.path is not None


def test_e10_t02b_finance_accepts_them_only_through_migration_or_v2_locations() -> None:
    """AC2: v1 keys migrate; the v2 locations are accepted; a v1 key inside a v2 file is rejected."""
    assert parse_v1(copy.deepcopy(_YAML.load(FIXTURES / "appendix_a.yaml"))).schema_version == 2
    assert parse_project_config(_v2(), FinanceConfig).data.primary.symbols == ["^GDAXI"]
    raw = _v2()
    raw["data"]["bars_per_day"] = 1
    with pytest.raises(ConfigError, match="bars_per_day"):
        parse_project_config(raw, FinanceConfig)
    raw = _v2()
    raw["profile"] = "finance"
    with pytest.raises(ConfigError) as err:
        parse_project_config(raw, FinanceConfig)
    assert err.value.path == "profile"


def test_e10_t02b_unknown_emission_family_lists_registered_names() -> None:
    """AC3: `emission.family: foo` -> ConfigError(path="model.emission.family") listing the names."""
    raw = _v2()
    raw["model"]["emission"]["family"] = "foo"
    with pytest.raises(ConfigError) as err:
        parse_project_config(raw, FinanceConfig)
    assert err.value.path == "model.emission.family"
    assert "gbm (regimekit.core)" in str(err.value)
    assert "student_t (regimekit.core)" in str(err.value)


@pytest.mark.parametrize(
    ("section", "key", "path"),
    [
        (("duration",), "family", "model.duration.family"),
        (("estimation",), "init", "estimation.init[0]"),
        (("forecast", "covariate_forecaster"), "type", "forecast.covariate_forecaster.type"),
        (("evaluation",), "metrics", "evaluation.metrics[0]"),
    ],
)
def test_e10_t02b_registry_validated_names_report_their_field_path(
    section: tuple[str, ...], key: str, path: str
) -> None:
    """Every registry-validated name field reports its own config path."""
    raw = _v2()
    node = raw["model"] if section == ("duration",) else raw
    for part in section if section != ("duration",) else ("duration",):
        node = node[part]
    node[key] = "nope" if key in ("family", "type") else ["nope"]
    with pytest.raises(ConfigError) as err:
        parse_project_config(raw, FinanceConfig)
    assert err.value.path == path


def test_e10_t02b_unknown_source_kind_and_clock_are_rejected() -> None:
    """Unknown source kind and clock are rejected."""
    raw = _v2()
    raw["data"]["source"]["kind"] = "nope"
    with pytest.raises(ConfigError) as err:
        parse_project_config(raw, FinanceConfig)
    assert err.value.path == "data.source.kind"
    raw = _v2()
    raw["data"]["clock"] = "lunar"
    with pytest.raises(ConfigError) as err:
        parse_project_config(raw, FinanceConfig)
    assert err.value.path == "data.clock"


def test_e10_t02b_backtest_pairs_need_exactly_one_member() -> None:
    """Backtest pairs need exactly one member."""
    raw = _v2()
    raw["evaluation"]["backtest"]["initial_train_steps"] = 100
    with pytest.raises(ConfigError, match="exactly one"):
        parse_project_config(raw, FinanceConfig)
    raw = _v2()
    del raw["evaluation"]["backtest"]["initial_train"]
    with pytest.raises(ConfigError, match="exactly one"):
        parse_project_config(raw, FinanceConfig)
    raw = _v2()
    raw["evaluation"]["backtest"].update(rolling_window=1.0, rolling_window_steps=10)
    with pytest.raises(ConfigError, match="at most one"):
        parse_project_config(raw, FinanceConfig)


def test_e10_t02b_resolved_dump_has_schema_version_2_and_no_profile(tmp_path: Path) -> None:
    """AC4: the resolved dump of a finance config contains `schema_version: 2` and no `profile`."""
    cfg = load_config(FIXTURES / "appendix_a.yaml", FinanceConfig)
    dump_resolved(cfg, tmp_path / "resolved.yaml")
    data = _YAML.load(tmp_path / "resolved.yaml")
    assert data["schema_version"] == 2
    assert "profile" not in data
    assert load_config(tmp_path / "resolved.yaml", FinanceConfig) == cfg


def test_e10_t02b_json_schema_export_contains_registered_names_as_enums(tmp_path: Path) -> None:
    """AC4: the export has the registered family names as an enum."""
    out = tmp_path / "schema.json"
    export_json_schema(out, FinanceConfig)
    defs = json.loads(out.read_text(encoding="utf-8"))["$defs"]
    assert defs["FinanceEmissionCfg"]["properties"]["family"]["enum"] == ["gbm", "student_t"]
    assert "nonparametric" in defs["FinanceDurationCfg"]["properties"]["family"]["enum"]
    assert defs["CovariateForecasterCfg"]["properties"]["type"]["enum"] == [
        "ar",
        "bootstrap",
        "freeze",
        "scenario",
    ]
    assert "rollvol_kmeans" in defs["EstimationSection"]["properties"]["init"]["items"]["enum"]


def test_e10_t02b_core_schema_ids() -> None:
    """Core schema ids."""
    assert (ProjectConfig.SCHEMA_ID, ProjectConfig.SCHEMA_VERSION) == ("core", 1)
    assert (FinanceConfig.SCHEMA_ID, FinanceConfig.SCHEMA_VERSION) == ("regimekit", 2)


def test_e10_t02b_core_config_uses_only_neutral_names() -> None:
    """A core-only config (steps clock) validates once a source kind is registered by the application."""
    raw = _v2()
    assert "profile" not in raw
    cfg = parse_project_config(raw, FinanceConfig)
    assert cfg.data.clock_option("bars_per_day") == 1
