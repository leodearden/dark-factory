"""The ``defect_tests:`` config group (plans/defect-tests-prd.md, Config and I5).

Fixtures stay module-local, following test_config_merge_deep.py: a conftest.py
edit would widen the merge-time verify to the whole owning package.
"""

from __future__ import annotations

import re

import pytest
import yaml
from pydantic import ValidationError

from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    apply_reload,
    load_config,
)
from orchestrator.defect_tests_config import DefectTestsConfig, DefectTestsMode

_M1_TEST_FILES = [
    'pkg/tests/x.py',
    'test/a.py',
    'src/test_foo.py',
    'foo_test.py',
    'cmd/a_test.go',
    'ui/x.test.ts',
    'ui/x.test.tsx',
    'ui/x.spec.ts',
    'ui/x.spec.tsx',
    'pkg/conftest.py',
]
_M1_NON_TEST_FILES = ['src/foo.py', 'docs/testing.md', 'src/contest.py', 'src/latest.py']


@pytest.mark.usefixtures('code_default_config')
class TestDefaults:
    def test_bare_config_carries_the_shipped_defaults(self):
        group = OrchestratorConfig().defect_tests
        assert group.mode is DefectTestsMode.SHADOW
        assert group.max_plan_repairs == 1
        assert group.max_proof_replans == 1
        assert group.proof_timeout_secs == 1800.0
        assert group.streak_threshold == 5

    @pytest.mark.parametrize('path', _M1_TEST_FILES)
    def test_default_pattern_matches_every_m1_rule(self, path):
        pattern = OrchestratorConfig().defect_tests.test_file_pattern
        assert re.search(pattern, path), f'{path!r} is an M1 test file'

    @pytest.mark.parametrize('path', _M1_NON_TEST_FILES)
    def test_default_pattern_rejects_non_test_files(self, path):
        pattern = OrchestratorConfig().defect_tests.test_file_pattern
        assert not re.search(pattern, path), f'{path!r} is not an M1 test file'


class TestLoad:
    def test_project_yaml_stanza_loads_through(self, tmp_path):
        project_cfg = tmp_path / 'orchestrator.yaml'
        project_cfg.write_text(
            yaml.dump({'defect_tests': {'mode': 'enforce', 'streak_threshold': 3}})
        )
        group = load_config(project_cfg).defect_tests
        assert group.mode is DefectTestsMode.ENFORCE
        assert group.streak_threshold == 3


class TestValidation:
    def test_unknown_mode_names_the_three_modes(self):
        with pytest.raises(ValidationError) as excinfo:
            DefectTestsConfig.model_validate({'mode': 'sometimes'})
        message = str(excinfo.value)
        for mode in ('off', 'shadow', 'enforce'):
            assert mode in message

    def test_uncompilable_pattern_names_the_field_and_the_re_error(self):
        with pytest.raises(ValidationError) as excinfo:
            DefectTestsConfig(test_file_pattern='(')
        message = str(excinfo.value)
        assert 'test_file_pattern' in message
        assert 'missing )' in message

    @pytest.mark.parametrize(
        'kwargs',
        [
            {'streak_threshold': 0},
            {'proof_timeout_secs': 0},
            {'max_plan_repairs': -1},
        ],
    )
    def test_out_of_bounds_knob_is_refused(self, kwargs):
        with pytest.raises(ValidationError):
            DefectTestsConfig(**kwargs)


class TestInjectsPlugin:
    @pytest.mark.parametrize('mode', list(DefectTestsMode))
    def test_only_off_stops_injection(self, mode):
        assert DefectTestsConfig(mode=mode).injects_plugin is (mode is not DefectTestsMode.OFF)


class TestReloadTier:
    @pytest.mark.parametrize('leaf', list(DefectTestsConfig.model_fields))
    def test_every_leaf_is_green_tier(self, leaf):
        assert f'defect_tests.{leaf}' in RELOADABLE_FIELDS

    @pytest.mark.usefixtures('code_default_config')
    def test_mode_flip_applies_live(self):
        live = OrchestratorConfig(defect_tests=DefectTestsConfig(mode=DefectTestsMode.SHADOW))
        fresh = OrchestratorConfig(defect_tests=DefectTestsConfig(mode=DefectTestsMode.ENFORCE))
        report = apply_reload(live, fresh)
        assert report['reloaded'] is True
        assert report['applied']['defect_tests.mode'] == {
            'old': DefectTestsMode.SHADOW,
            'new': DefectTestsMode.ENFORCE,
        }
        assert 'defect_tests.mode' not in report['restart_required']
        assert live.defect_tests.mode is DefectTestsMode.ENFORCE
