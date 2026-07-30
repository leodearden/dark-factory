"""Deep merge-ahead α: the ``merge_deep.chain_cap`` knob + kill switch.

Task 3183; PRD ``plans/deep-merge-ahead-prd.md`` task α (Phase 1 foundation).
Covers decision #6 (cap staging 0 -> 6 -> 32, with ``chain_cap=0`` as the kill
switch) and decision #7 (the knob is green-tier hot-reloadable, so
enable/retune/kill never needs a process restart).

This task ships ONLY the knob — β (3184, chain builder) and γ (3185, dispatch
gate) are the consumers, so nothing here asserts dispatch behaviour.

Fixtures are kept MODULE-LOCAL (not conftest.py) — a conftest.py edit trips
verify.py's has_conftest and forces the merge-time verify to fall back to
running the full owning-package suite instead of a scoped subset (mirrors
test_config_psi_admission_reload.py:4-7's stated rationale). Harness helpers
are likewise replicated rather than imported (test_harness_config_reload.py:31-34).
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from orchestrator.config import (
    MergeDeepConfig,
    OrchestratorConfig,
    census_config_keys,
    load_config,
)


class TestMergeDeepConfigDefaults:
    """The shipped default is the kill switch: ``chain_cap=0``."""

    def test_pydantic_default_chain_cap_is_zero(self):
        """The kill-switch default is a pydantic-level fact, not just a YAML one."""
        field_info = MergeDeepConfig.model_fields['chain_cap']
        assert field_info.default == 0

    def test_reachable_as_orchestrator_config_attribute(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig()
        assert isinstance(cfg.merge_deep, MergeDeepConfig)
        assert cfg.merge_deep.chain_cap == 0

    def test_submodel_is_mutable_in_place(self):
        """Pins the plain-BaseModel requirement (no frozen / no
        validate_assignment) that ``_set_leaf`` (config.py:4876) depends on:
        a two-component reload path is written with a plain ``setattr`` on the
        submodel so held references observe the update (invariant I3).
        """
        m = MergeDeepConfig()
        m.chain_cap = 6
        assert m.chain_cap == 6


class TestMergeDeepKnobInProjectYaml:
    """The operator deploy path that ``scripts/merge-deep-set-cap.sh`` (already
    on main) writes: ``merge_deep:\\n  chain_cap: <cap>`` in the project YAML.
    """

    def test_project_yaml_override_loads_through(self, monkeypatch, tmp_path):
        """Project layer beats package defaults (test_config.py:982 precedent)."""
        project_cfg = tmp_path / 'orchestrator.yaml'
        project_cfg.write_text(yaml.dump({'merge_deep': {'chain_cap': 6}}))
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        cfg = load_config(project_cfg)
        assert cfg.merge_deep.chain_cap == 6

    def test_knob_is_a_known_census_key(self, monkeypatch, tmp_path):
        """Registering the submodel makes ``merge_deep.chain_cap`` a KNOWN key,
        so the deploy script's edit is neither silently dropped by pydantic
        ``extra='ignore'`` nor censused as a phantom key (which would file a
        born-at-L2 every time the canary is deployed).
        """
        project_cfg = tmp_path / 'orchestrator.yaml'
        project_cfg.write_text(yaml.dump({'merge_deep': {'chain_cap': 6}}))
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        assert census_config_keys(project_cfg).unknown == []

    def test_typo_under_the_knob_still_censuses(self, monkeypatch, tmp_path):
        """Negative control: the census must still descend INTO merge_deep, so a
        typo'd leaf is reported rather than swallowed by the new known parent.
        """
        project_cfg = tmp_path / 'orchestrator.yaml'
        project_cfg.write_text(yaml.dump({'merge_deep': {'chain_capp': 6}}))
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        paths = [uk.path for uk in census_config_keys(project_cfg).unknown]
        assert 'merge_deep.chain_capp' in paths, (
            f'expected the typo to be censused under the known merge_deep parent; got {paths!r}'
        )


class TestChainCapBoundValidation:
    """``chain_cap`` must be >= 0, with deliberately NO upper bound."""

    @pytest.mark.parametrize('bad_value', [-1, -6, -32])
    def test_negative_cap_rejected(self, bad_value):
        """A negative cap is meaningless in
        ``target_depth = min(len(queue), cap, halving_state)`` — it would silently
        win that min() and disable or underflow the gate, so it must fail loudly
        at construction (loud over silent degradation).
        """
        with pytest.raises(ValidationError):
            MergeDeepConfig(chain_cap=bad_value)

    def test_negative_cap_rejected_at_load(self, monkeypatch, tmp_path):
        """Covers the 'rejected at load' signal: OrchestratorConfig builds the
        merge_deep submodel from the nested dict shape YAML deserializes into, so
        a bad project YAML fails at load rather than reaching dispatch.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        with pytest.raises(ValidationError):
            OrchestratorConfig(merge_deep={'chain_cap': -1})  # type: ignore[arg-type]

    @pytest.mark.parametrize('good_value', [0, 1, 6, 32])
    def test_whole_staging_ladder_is_representable(self, good_value):
        """0 = kill switch, 1 = the halving floor (d=1, byte-identical to today's
        adjacent verify), 6 = the ζ canary depth, 32 = η2's "uncapped in
        practice". No upper bound is imposed — θ owns the ceiling question.
        """
        assert MergeDeepConfig(chain_cap=good_value).chain_cap == good_value
