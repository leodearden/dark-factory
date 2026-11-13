"""Per-role verify-cache modes and the per-module never-cache opt-out (task 5652).

Fixtures are kept MODULE-LOCAL (not conftest.py) for the reason stated in
test_config_verify_admission_reload.py: a conftest.py edit trips verify.py's
has_conftest and forces the merge-time verify to run the full owning-package
suite instead of a scoped subset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orchestrator.config import (
    RELOADABLE_FIELDS,
    ModuleConfig,
    OrchestratorConfig,
    VerifyCacheConfig,
    apply_reload,
    load_config,
)
from orchestrator.verify_cache.modes import CacheMode


def _project(tmp_path: Path, top_level: dict) -> Path:
    config_path = tmp_path / 'dark-factory-orchestrator.yaml'
    config_path.write_text(yaml.safe_dump({'project_root': str(tmp_path), **top_level}))
    return config_path


@pytest.mark.usefixtures('code_default_config')
def test_both_roles_default_to_off():
    modes = OrchestratorConfig().verify_cache

    assert modes.task_mode is CacheMode.OFF
    assert modes.merge_mode is CacheMode.OFF


class TestModeFor:
    def test_task_on_leaves_merge_off(self):
        modes = VerifyCacheConfig(task_mode=CacheMode.ON)

        assert modes.mode_for('task') is CacheMode.ON
        assert modes.mode_for('merge') is CacheMode.OFF

    def test_merge_log_leaves_task_off(self):
        modes = VerifyCacheConfig(merge_mode=CacheMode.LOG)

        assert modes.mode_for('merge') is CacheMode.LOG
        assert modes.mode_for('task') is CacheMode.OFF

    def test_background_is_always_off(self):
        modes = VerifyCacheConfig(task_mode=CacheMode.ON, merge_mode=CacheMode.ON)

        assert modes.mode_for('background') is CacheMode.OFF

    def test_a_mode_outside_the_vocabulary_is_rejected(self):
        with pytest.raises(ValidationError):
            VerifyCacheConfig.model_validate({'task_mode': 'maybe'})


def test_each_role_mode_is_its_own_reload_leaf():
    assert {'verify_cache.task_mode', 'verify_cache.merge_mode'} <= RELOADABLE_FIELDS


@pytest.mark.usefixtures('code_default_config')
class TestReloadIndependence:
    def test_a_task_mode_edit_applies_alone(self):
        live = OrchestratorConfig()

        report = apply_reload(
            live, OrchestratorConfig(verify_cache=VerifyCacheConfig(task_mode=CacheMode.LOG)),
        )

        assert report['reloaded'] is True
        assert report['applied'] == {
            'verify_cache.task_mode': {'old': CacheMode.OFF, 'new': CacheMode.LOG},
        }
        assert not any(key.startswith('verify_cache') for key in report['restart_required'])
        assert live.verify_cache.task_mode is CacheMode.LOG
        assert live.verify_cache.merge_mode is CacheMode.OFF

    def test_a_merge_mode_edit_leaves_the_live_task_mode(self):
        live = OrchestratorConfig(verify_cache=VerifyCacheConfig(task_mode=CacheMode.LOG))

        report = apply_reload(
            live,
            OrchestratorConfig(
                verify_cache=VerifyCacheConfig(task_mode=CacheMode.LOG, merge_mode=CacheMode.ON),
            ),
        )

        assert set(report['applied']) == {'verify_cache.merge_mode'}
        assert live.verify_cache.merge_mode is CacheMode.ON
        assert live.verify_cache.task_mode is CacheMode.LOG


def test_the_project_yaml_sets_one_role(tmp_path):
    config = load_config(_project(tmp_path, {'verify_cache': {'task_mode': 'log'}}))

    assert config.verify_cache.task_mode is CacheMode.LOG
    assert config.verify_cache.merge_mode is CacheMode.OFF


class TestModuleNeverCache:
    def test_a_module_is_cacheable_by_default(self):
        assert ModuleConfig(prefix='x').verify_cache == 'default'

    def test_a_module_can_opt_out(self):
        assert ModuleConfig(prefix='x', verify_cache='never').verify_cache == 'never'

    def test_a_misspelt_opt_out_is_an_error_naming_the_module(self):
        as_loaded_from_yaml = yaml.safe_load('verify_cache: nevr')

        with pytest.raises(ValueError, match=r"'x'.*'nevr'.*default.*never"):
            ModuleConfig(prefix='x', **as_loaded_from_yaml)

    def test_a_module_yaml_declares_the_opt_out(self, tmp_path):
        guards = tmp_path / 'guards'
        guards.mkdir()
        (guards / 'orchestrator.yaml').write_text(
            yaml.safe_dump({'test_command': 'pytest guards', 'verify_cache': 'never'}),
        )

        config = load_config(_project(tmp_path, {}))

        assert config.module_configs_or_empty['guards'].verify_cache == 'never'

    def test_a_misspelt_module_yaml_value_loads_as_never_and_logs_the_module(self, tmp_path, caplog):
        guards = tmp_path / 'guards'
        guards.mkdir()
        (guards / 'orchestrator.yaml').write_text(
            yaml.safe_dump({'test_command': 'pytest guards', 'verify_cache': 'nevr'}),
        )

        with caplog.at_level(logging.ERROR, logger='orchestrator.config'):
            config = load_config(_project(tmp_path, {}))

        module = config.module_configs_or_empty['guards']
        assert (module.verify_cache, module.test_command) == ('never', 'pytest guards')
        assert any("'guards'" in record.getMessage() and "'nevr'" in record.getMessage() for record in caplog.records)
