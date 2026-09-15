"""C′: the verify_host_policy knob — default, validation, and check-config census
(task 5097; PRD plans/merge-lane-throughput-prd.md task C′).

Fixtures are kept MODULE-LOCAL (not conftest.py) — a conftest.py edit trips
verify.py's has_conftest and forces the merge-time verify to fall back to
running the full owning-package suite instead of a scoped subset (mirrors
test_config_verify_admission_reload.py's stated rationale).  conftest.py is
additionally a merge-lane ratchet cluster path with a frozen line count.

RATCHET CONSTRAINT: this module must NEVER import orchestrator.merge_queue or
anything under orchestrator.merge_lane.  scripts/merge_lane_metrics.py::
imports_lane_module would then admit this file to the ratchet's `tests` map,
staling the committed baseline that test_merge_lane_ratchet.py::
test_baseline_matches_a_fresh_measurement compares byte-for-byte.  Importing
orchestrator.config only keeps this file invisible to both ratchet gates.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orchestrator.config import (
    OrchestratorConfig,
    census_config_keys,
)


def _write_yaml(tmp_path: Path, data, name: str = 'orchestrator.yaml') -> Path:
    """Local copy of test_config_unknown_keys.py's helper — orchestrator/tests
    has no __init__.py, so this module stays self-contained.
    """
    p = tmp_path / name
    p.write_text(yaml.dump(data))
    return p


class TestVerifyHostPolicyDefault:
    """The default preserves today's dispatch order byte-for-byte: local is
    taken first when free, remotes take overflow.  Flipping the knob is an
    opt-in act by an operator, never a silent change under an upgrade.
    """

    def test_default_is_prefer_local(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig()
        assert cfg.verify_host_policy == 'prefer_local'

    def test_prefer_remote_is_accepted(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig(verify_host_policy='prefer_remote')
        assert cfg.verify_host_policy == 'prefer_remote'

    def test_prefer_local_is_accepted_explicitly(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig(verify_host_policy='prefer_local')
        assert cfg.verify_host_policy == 'prefer_local'


class TestVerifyHostPolicyValidation:
    """The Literal is the guard, not a free string: a typo'd or empty policy is
    rejected at construction rather than falling through to the prefer_local
    else-branch at dispatch time, where an operator would see no error and the
    wrong host.
    """

    @pytest.mark.parametrize(
        'bad_value',
        ['prefer_laptop', '', None, 'PREFER_REMOTE', 'remote', 0],
    )
    def test_bogus_policy_rejected(self, bad_value, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        with pytest.raises(ValidationError):
            OrchestratorConfig(verify_host_policy=bad_value)


class TestVerifyHostPolicyCheckConfigCensus:
    """`orchestrator check-config` must accept the new key.

    census_config_keys is the exact engine cli.py's check-config command
    drives, so calling it directly pins the accepts-the-new-key contract
    without a CLI subprocess.  A key dark-factory itself consumes must be
    genuinely KNOWN — the census correctly refuses a config_key_census.ignore
    excuse for it.
    """

    def test_key_is_known_to_the_census(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        p = _write_yaml(tmp_path, {'verify_host_policy': 'prefer_remote'})

        census = census_config_keys(p)

        assert census.parse_error is None
        assert [uk.path for uk in census.unknown] == []
        assert 'verify_host_policy' not in [ik.path for ik in census.ignored]

    def test_key_is_known_beside_its_lever_c_siblings(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        p = _write_yaml(
            tmp_path,
            {
                'verify_host_policy': 'prefer_local',
                'verify_host_unreachable_escalate_after_n': 3,
                'verify_host_reprobe_interval_s': 120.0,
            },
        )

        census = census_config_keys(p)

        assert census.parse_error is None
        assert [uk.path for uk in census.unknown] == []
