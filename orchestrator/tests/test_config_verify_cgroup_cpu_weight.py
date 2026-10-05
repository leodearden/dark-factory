"""Per-role cgroup CPUWeight knobs for verify scopes + green-tier reload
registration (task 5205).

Fixtures are kept MODULE-LOCAL (not conftest.py) — a conftest.py edit trips
verify.py's has_conftest and forces the merge-time verify to fall back to
running the full owning-package suite instead of a scoped subset (mirrors
test_config_verify_admission_reload.py's stated rationale).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from orchestrator.config import OrchestratorConfig, diff_config

WEIGHT_FIELDS = [
    'verify_cgroup_cpu_weight_merge',
    'verify_cgroup_cpu_weight_task',
    'verify_cgroup_cpu_weight_background',
]


@pytest.mark.usefixtures('code_default_config')
class TestVerifyCgroupCpuWeightDefaults:
    """Merge stays at systemd's default share (parity with each orchestrator
    unit); task and background are lowered, keeping the 3:1 merge:task intent.
    """

    def test_defaults_on_bare_config(self):
        cfg = OrchestratorConfig()
        assert cfg.verify_cgroup_cpu_weight_merge == 100
        assert cfg.verify_cgroup_cpu_weight_task == 33
        assert cfg.verify_cgroup_cpu_weight_background == 10


class TestVerifyCgroupCpuWeightBounds:
    """cgroup v2 cpu.weight accepts 1..10000; anything outside is rejected at
    construction rather than surfacing as a systemd-run failure at spawn.
    """

    @pytest.mark.parametrize('field', WEIGHT_FIELDS)
    @pytest.mark.parametrize('bad_value', [0, -1, 10001])
    def test_out_of_range_rejected(self, field, bad_value):
        with pytest.raises(ValidationError):
            OrchestratorConfig(**{field: bad_value})

    @pytest.mark.parametrize('field', WEIGHT_FIELDS)
    @pytest.mark.parametrize('good_value', [1, 10000])
    def test_range_endpoints_accepted(self, field, good_value):
        cfg = OrchestratorConfig(**{field: good_value})
        assert getattr(cfg, field) == good_value


class TestVerifyCgroupCpuWeightReloadDisposition:
    """Every weight knob is green-tier: hot-reloadable without a restart.

    That a reloaded weight reaches the next scope spawn is checked end to end
    in test_verify_scope_cpu_weight.py.
    """

    @pytest.mark.parametrize('field', WEIGHT_FIELDS)
    def test_weight_edit_lands_in_applied_candidates_not_restart_required(
        self, field, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        before: dict[str, Any] = {field: 33}
        after: dict[str, Any] = {field: 50}
        diff = diff_config(OrchestratorConfig(**before), OrchestratorConfig(**after))
        assert field in diff.applied_candidates
        assert field not in diff.restart_required
