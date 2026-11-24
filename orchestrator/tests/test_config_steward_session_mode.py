"""The green-tier steward_session_mode knob: its closed vocabulary, its code
default, and its hot-reload registration (task 5819).

Fixtures are kept MODULE-LOCAL (not conftest.py) for the reason stated in
test_config_pre_merge_reverify_reload.py: a conftest.py edit forces the
merge-time verify to run the full owning-package suite.
"""

from __future__ import annotations

import pytest
import yaml
from click.testing import CliRunner
from pydantic import ValidationError

from orchestrator.cli import main
from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    apply_reload,
    census_config_keys,
    diff_config,
)


@pytest.mark.usefixtures("code_default_config")
class TestConfigKnob:
    def test_code_default_is_fresh_per_l0(self):
        assert OrchestratorConfig().steward_session_mode == 'fresh_per_l0'

    def test_resume_is_a_member(self):
        config = OrchestratorConfig.model_validate({'steward_session_mode': 'resume'})
        assert config.steward_session_mode == 'resume'

    def test_a_non_member_is_rejected_naming_the_field(self):
        with pytest.raises(ValidationError) as excinfo:
            OrchestratorConfig.model_validate({'steward_session_mode': 'fresh'})
        assert 'steward_session_mode' in str(excinfo.value)


@pytest.mark.usefixtures("code_default_config")
class TestGreenTier:
    def test_field_is_reloadable(self):
        assert 'steward_session_mode' in RELOADABLE_FIELDS

    def test_edit_lands_in_applied_candidates_not_restart_required(self):
        diff = diff_config(
            OrchestratorConfig(), OrchestratorConfig(steward_session_mode='resume'),
        )
        assert 'steward_session_mode' in diff.applied_candidates
        assert 'steward_session_mode' not in diff.restart_required

    def test_apply_reload_flips_the_live_object_in_place(self):
        live = OrchestratorConfig()
        report = apply_reload(live, OrchestratorConfig(steward_session_mode='resume'))
        assert report['reloaded'] is True
        assert report['applied'] == {
            'steward_session_mode': {'old': 'fresh_per_l0', 'new': 'resume'},
        }
        assert report['restart_required'] == {}
        assert live.steward_session_mode == 'resume'


class TestCensusAndCli:
    def _write_yaml(self, tmp_path):
        path = tmp_path / 'orchestrator.yaml'
        path.write_text(yaml.safe_dump({'steward_session_mode': 'resume'}))
        return path

    def test_census_knows_the_key(self, tmp_path):
        path = self._write_yaml(tmp_path)
        unknown_paths = [key.path for key in census_config_keys(path).unknown]
        assert 'steward_session_mode' not in unknown_paths

    def test_check_config_passes(self, tmp_path):
        path = self._write_yaml(tmp_path)
        result = CliRunner().invoke(main, ['check-config', '--config', str(path)])
        assert result.exit_code == 0, result.output
