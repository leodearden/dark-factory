"""δ (task 6670): the four verify_task_* knobs — defaults, validation, census and
green-tier reload (PRD plans/remote-task-leg-verify-prd.md §5.7, decisions 6
and 9).

Fixtures are kept MODULE-LOCAL (not conftest.py), for the reason
test_config_verify_host_policy.py states: a conftest.py edit widens the
merge-time verify to the owning package's full suite.

RATCHET CONSTRAINT: this module imports orchestrator.config only — never
orchestrator.merge_lane or orchestrator.git_ops — so it stays outside both
merge-lane ratchet gates (scripts/merge_lane_metrics.py::imports_lane_module).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    apply_reload,
    census_config_keys,
    diff_config,
    load_config,
)

TASK_VERIFY_KEYS = (
    'verify_task_host_policy',
    'verify_task_remote_modules',
    'verify_task_remote_cross_check_every_n',
    'verify_task_remote_red_confirms_locally',
)


def _write_yaml(tmp_path: Path, data, name: str = 'orchestrator.yaml') -> Path:
    p = tmp_path / name
    p.write_text(yaml.dump(data))
    return p


@pytest.mark.usefixtures('code_default_config')
class TestDefaults:
    """The shipped default keeps every task verify local, so an upgrade changes
    nothing until an operator flips the policy.
    """

    def test_defaults_keep_task_verifies_local(self):
        cfg = OrchestratorConfig()

        assert cfg.verify_task_host_policy == 'local'
        assert cfg.verify_task_remote_modules == []
        assert cfg.verify_task_remote_cross_check_every_n == 0
        assert cfg.verify_task_remote_red_confirms_locally is False

    def test_remote_modules_with_modules_is_accepted(self):
        cfg = OrchestratorConfig(
            verify_task_host_policy='remote_modules',
            verify_task_remote_modules=['orchestrator'],
        )

        assert cfg.verify_task_host_policy == 'remote_modules'
        assert cfg.verify_task_remote_modules == ['orchestrator']

    def test_remote_all_needs_no_module_list(self):
        cfg = OrchestratorConfig(verify_task_host_policy='remote_all')

        assert cfg.verify_task_host_policy == 'remote_all'
        assert cfg.verify_task_remote_modules == []


@pytest.mark.usefixtures('code_default_config')
class TestValidation:
    @pytest.mark.parametrize('bad_value', ['remote', '', None, 'REMOTE_ALL'])
    def test_bogus_policy_rejected(self, bad_value):
        with pytest.raises(ValidationError):
            OrchestratorConfig(verify_task_host_policy=bad_value)

    def test_negative_cross_check_cadence_rejected(self):
        with pytest.raises(ValidationError):
            OrchestratorConfig(verify_task_remote_cross_check_every_n=-1)


@pytest.mark.usefixtures('code_default_config')
class TestRemoteModulesNeedsModules:
    """Row 18 (INV-1): 'remote_modules' with no modules is a config error at
    load, naming both keys, rather than a policy that silently never matches.
    """

    def test_constructor_rejects_and_names_both_keys(self):
        with pytest.raises(ValidationError) as exc:
            OrchestratorConfig(
                verify_task_host_policy='remote_modules',
                verify_task_remote_modules=[],
            )

        assert 'verify_task_host_policy' in str(exc.value)
        assert 'verify_task_remote_modules' in str(exc.value)

    def test_load_config_rejects_and_names_both_keys(self, tmp_path, monkeypatch):
        p = _write_yaml(
            tmp_path,
            {'verify_task_host_policy': 'remote_modules', 'verify_task_remote_modules': []},
        )
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(p))

        with pytest.raises(ValidationError) as exc:
            load_config(p)

        assert 'verify_task_host_policy' in str(exc.value)
        assert 'verify_task_remote_modules' in str(exc.value)


class TestReloadDisposition:
    """All four are green-tier: the dispatcher reads them per dispatch, so a
    flip and its rollback are one reload_config each.
    """

    @pytest.mark.parametrize('key', TASK_VERIFY_KEYS)
    def test_field_is_reloadable(self, key):
        assert key in RELOADABLE_FIELDS

    @pytest.mark.usefixtures('code_default_config')
    def test_every_changed_key_is_an_applied_candidate(self):
        live = OrchestratorConfig()
        fresh = OrchestratorConfig(
            verify_task_host_policy='remote_modules',
            verify_task_remote_modules=['orchestrator'],
            verify_task_remote_cross_check_every_n=5,
            verify_task_remote_red_confirms_locally=True,
        )

        diff = diff_config(live, fresh)

        for key in TASK_VERIFY_KEYS:
            assert key in diff.applied_candidates
            assert key not in diff.restart_required

    @pytest.mark.usefixtures('code_default_config')
    def test_flip_and_rollback_apply_in_place(self):
        live = OrchestratorConfig()

        flip = apply_reload(
            live,
            OrchestratorConfig(
                verify_task_host_policy='remote_modules',
                verify_task_remote_modules=['orchestrator'],
            ),
        )

        assert flip['reloaded'] is True
        assert flip['applied']['verify_task_host_policy'] == {
            'old': 'local', 'new': 'remote_modules',
        }
        assert live.verify_task_host_policy == 'remote_modules'
        assert live.verify_task_remote_modules == ['orchestrator']

        back = apply_reload(live, OrchestratorConfig())

        assert back['reloaded'] is True
        assert back['applied']['verify_task_host_policy'] == {
            'old': 'remote_modules', 'new': 'local',
        }
        assert live.verify_task_host_policy == 'local'


class TestCheckConfigCensus:
    def test_keys_are_known_to_the_census(self, tmp_path):
        p = _write_yaml(
            tmp_path,
            {
                'verify_task_host_policy': 'remote_modules',
                'verify_task_remote_modules': ['orchestrator'],
                'verify_task_remote_cross_check_every_n': 5,
                'verify_task_remote_red_confirms_locally': True,
            },
        )

        census = census_config_keys(p)

        assert census.parse_error is None
        assert [uk.path for uk in census.unknown] == []
