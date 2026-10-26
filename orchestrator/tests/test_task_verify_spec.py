"""Tests for the remote task-verify spec (``orchestrator.task_verify_spec``):
its shape and invariants, its strict wire codec, the workstation-side build
and the laptop-side apply.

PRD: plans/remote-task-leg-verify-prd.md §4.1, §5.1, §6 row 1.
"""

import dataclasses
import json
import operator
from pathlib import Path

import pytest

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.task_verify_spec import (
    CollapsedOverride,
    TaskVerifySpec,
    apply_task_verify_spec,
    build_task_verify_spec,
    collapsed_module_overrides,
    spec_from_dict,
    spec_to_dict,
    task_spec_from_json,
    task_spec_to_json,
)
from orchestrator.verify_spec_shared import VerifyCommand

_TIP = 'a' * 40


def _spec(**overrides) -> TaskVerifySpec:
    fields = {
        'verify_commands': (
            VerifyCommand(prefix='m', test_command='t-m', lint_command='l-m', type_check_command='y-m'),
            VerifyCommand(prefix='n', test_command='t-n'),
        ),
        'global_verify_command': None,
        'task_files': (),
        'force_workspace': False,
        'verify_env': {'A': '1', 'B': '2'},
        'scope_cargo': False,
        'verify_cold_preprovision_command': '',
        'attempt_id': 3,
        'task_id': '6667',
        'tip_sha': _TIP,
    }
    return TaskVerifySpec(**{**fields, **overrides})


def _host_config(tmp_path: Path, **overrides) -> OrchestratorConfig:
    fields = {
        'project_root': tmp_path,
        'verify_env': {'HOST_ONLY': 'h', 'A': 'host'},
        'verify_cold_preprovision_command': 'host-prep',
        'scope_cargo': True,
        'test_command': 'host-test',
        'lint_command': 'host-lint',
        'type_check_command': 'host-type',
    }
    return OrchestratorConfig(**{**fields, **overrides})


class TestShape:
    def test_wire_keys_are_exactly_the_contract(self):
        assert set(spec_to_dict(_spec())) == {
            'verify_commands',
            'global_verify_command',
            'task_files',
            'force_workspace',
            'verify_env',
            'scope_cargo',
            'verify_cold_preprovision_command',
            'attempt_id',
            'task_id',
            'tip_sha',
        }

    def test_is_frozen(self):
        spec = _spec()

        pytest.raises(dataclasses.FrozenInstanceError, setattr, spec, 'attempt_id', 4)

    def test_verify_env_is_immutable(self):
        spec = _spec()

        pytest.raises(TypeError, operator.setitem, spec.verify_env, 'A', 'changed')

    def test_verify_env_is_a_copy(self):
        env = {'A': '1'}

        spec = _spec(verify_env=env)
        env['A'] = 'changed'

        assert dict(spec.verify_env) == {'A': '1'}


class TestInvariants:
    def test_task_files_none_is_rejected(self):
        with pytest.raises(TypeError, match='task_files'):
            _spec(task_files=None)

    def test_task_files_list_is_rejected(self):
        with pytest.raises(TypeError, match='task_files'):
            _spec(task_files=['a'])

    @pytest.mark.parametrize('tip_sha', ['abc', 'A' * 40, 'g' * 40])
    def test_tip_sha_must_be_full_lowercase_hex(self, tip_sha: str):
        with pytest.raises(ValueError, match=tip_sha):
            _spec(tip_sha=tip_sha)

    def test_full_hex_tip_sha_is_accepted(self):
        assert _spec(tip_sha='0123456789abcdef' * 2 + '01234567').tip_sha.startswith('0123')


class TestCodec:
    def test_round_trip_two_modules_with_env(self):
        spec = _spec()

        assert task_spec_from_json(task_spec_to_json(spec)) == spec

    def test_round_trip_global_command_and_task_files(self):
        spec = _spec(
            verify_commands=(),
            global_verify_command=VerifyCommand(prefix='', test_command='pytest'),
            task_files=('x.py',),
        )

        assert task_spec_from_json(task_spec_to_json(spec)) == spec

    def test_encoding_is_canonical(self):
        encoded = task_spec_to_json(_spec())

        assert task_spec_to_json(task_spec_from_json(encoded)) == encoded

    def test_extra_top_level_key_is_rejected(self):
        document = {**spec_to_dict(_spec()), 'role': 'task'}

        with pytest.raises(TypeError, match='role'):
            spec_from_dict(document)

    def test_missing_key_is_rejected(self):
        document = spec_to_dict(_spec())
        del document['scope_cargo']

        with pytest.raises(TypeError, match='scope_cargo'):
            spec_from_dict(document)

    def test_extra_key_in_a_verify_command_is_rejected(self):
        document = spec_to_dict(_spec())
        document['verify_commands'][0]['timeout_secs'] = 5

        with pytest.raises(TypeError, match='timeout_secs'):
            spec_from_dict(document)

    def test_json_list_document_is_a_type_error(self):
        with pytest.raises(TypeError):
            task_spec_from_json(json.dumps([1, 2]))

    @pytest.mark.parametrize(('field', 'wire_value'), [
        ('force_workspace', 1),
        ('scope_cargo', 'false'),
        ('verify_env', {'A': 1}),
        ('verify_env', None),
        ('verify_cold_preprovision_command', None),
        ('attempt_id', '3'),
        ('task_id', 6667),
        ('tip_sha', None),
    ])
    def test_mistyped_field_is_rejected_naming_it(self, field: str, wire_value: object):
        document = {**spec_to_dict(_spec()), field: wire_value}

        with pytest.raises(TypeError, match=f'TaskVerifySpec.{field} must be'):
            spec_from_dict(document)


class TestBuild:
    def _modules(self) -> list[ModuleConfig]:
        return [
            ModuleConfig(prefix='m', test_command='t-m', lint_command='l-m', type_check_command='y-m'),
            ModuleConfig(prefix='n', test_command='t-n', lock_depth=2),
        ]

    def test_projects_config_and_modules(self, tmp_path: Path):
        config = _host_config(tmp_path, scope_cargo=False, verify_cold_preprovision_command='prep')

        spec = build_task_verify_spec(
            config, self._modules(), ['f.py'],
            force_workspace=False, attempt_id=3, task_id='6667', tip_sha='b' * 40,
        )

        assert spec.verify_commands == (
            VerifyCommand(prefix='m', test_command='t-m', lint_command='l-m', type_check_command='y-m'),
            VerifyCommand(prefix='n', test_command='t-n'),
        )
        assert spec.task_files == ('f.py',)
        assert dict(spec.verify_env) == config.effective_verify_env
        assert spec.scope_cargo is False
        assert spec.verify_cold_preprovision_command == 'prep'
        assert (spec.force_workspace, spec.attempt_id, spec.task_id, spec.tip_sha) == (
            False, 3, '6667', 'b' * 40,
        )

    def test_zero_modules_ship_the_global_command(self, tmp_path: Path):
        spec = build_task_verify_spec(
            _host_config(tmp_path), [], [],
            force_workspace=True, attempt_id=1, task_id='6667', tip_sha=_TIP,
        )

        assert spec.global_verify_command == VerifyCommand(
            prefix='', test_command='host-test', lint_command='host-lint',
            type_check_command='host-type',
        )

    def test_modules_ship_no_global_command(self, tmp_path: Path):
        spec = build_task_verify_spec(
            _host_config(tmp_path), self._modules(), [],
            force_workspace=False, attempt_id=1, task_id='6667', tip_sha=_TIP,
        )

        assert spec.global_verify_command is None

    def test_collapsed_module_overrides_names_each_one(self):
        modules = [
            ModuleConfig(prefix='m', verify_env={'X': '1'}),
            ModuleConfig(prefix='n', scope_cargo=False),
            ModuleConfig(prefix='o'),
        ]

        assert collapsed_module_overrides(modules) == (
            CollapsedOverride('m', 'verify_env'),
            CollapsedOverride('n', 'scope_cargo'),
        )

    def test_build_refuses_a_collapsed_override(self, tmp_path: Path):
        modules = [ModuleConfig(prefix='odd/module', verify_env={'X': '1'})]

        with pytest.raises(ValueError, match='odd/module') as excinfo:
            build_task_verify_spec(
                _host_config(tmp_path), modules, [],
                force_workspace=False, attempt_id=1, task_id='6667', tip_sha=_TIP,
            )

        assert 'verify_env' in str(excinfo.value)


class TestApply:
    def test_spec_env_wins_and_host_only_keys_survive(self, tmp_path: Path):
        config = _host_config(tmp_path)

        applied, _ = apply_task_verify_spec(config, _spec())

        assert applied.verify_env == {**config.effective_verify_env, 'A': '1', 'B': '2'}
        assert applied.verify_env['HOST_ONLY'] == 'h'

    def test_non_empty_spec_preprovision_replaces_the_hosts(self, tmp_path: Path):
        applied, _ = apply_task_verify_spec(
            _host_config(tmp_path), _spec(verify_cold_preprovision_command='spec-prep'),
        )

        assert applied.verify_cold_preprovision_command == 'spec-prep'

    def test_empty_spec_preprovision_keeps_the_hosts(self, tmp_path: Path):
        applied, _ = apply_task_verify_spec(
            _host_config(tmp_path), _spec(verify_cold_preprovision_command=''),
        )

        assert applied.verify_cold_preprovision_command == 'host-prep'

    def test_scope_cargo_comes_from_the_spec(self, tmp_path: Path):
        applied, _ = apply_task_verify_spec(_host_config(tmp_path, scope_cargo=True), _spec(scope_cargo=False))

        assert applied.scope_cargo is False

    def test_global_command_is_applied_when_set(self, tmp_path: Path):
        spec = _spec(
            verify_commands=(),
            global_verify_command=VerifyCommand(
                prefix='', test_command='spec-test', lint_command='spec-lint',
                type_check_command='spec-type',
            ),
        )

        applied, _ = apply_task_verify_spec(_host_config(tmp_path), spec)

        assert (applied.test_command, applied.lint_command, applied.type_check_command) == (
            'spec-test', 'spec-lint', 'spec-type',
        )

    def test_host_commands_kept_without_a_global_command(self, tmp_path: Path):
        applied, _ = apply_task_verify_spec(_host_config(tmp_path), _spec())

        assert (applied.test_command, applied.lint_command, applied.type_check_command) == (
            'host-test', 'host-lint', 'host-type',
        )

    def test_modules_are_rebuilt_host_shaped_and_installed(self, tmp_path: Path):
        config = _host_config(tmp_path)
        config._module_configs = {'host/only': ModuleConfig(prefix='host/only')}

        applied, modules = apply_task_verify_spec(config, _spec())

        assert [
            (mc.prefix, mc.test_command, mc.lint_command, mc.type_check_command)
            for mc in modules
        ] == [('m', 't-m', 'l-m', 'y-m'), ('n', 't-n', None, None)]
        assert all(mc.verify_env is None for mc in modules)
        assert all(mc.verify_cold_command_timeout_secs is None for mc in modules)
        assert applied.module_configs_or_empty == {mc.prefix: mc for mc in modules}

    def test_zero_modules_install_an_empty_registry(self, tmp_path: Path):
        config = _host_config(tmp_path)
        config._module_configs = {'host/only': ModuleConfig(prefix='host/only')}

        applied, modules = apply_task_verify_spec(config, _spec(verify_commands=()))

        assert modules == []
        assert applied.module_configs_or_empty == {}

    def test_caller_config_is_not_mutated(self, tmp_path: Path):
        config = _host_config(tmp_path)
        config._module_configs = {'host/only': ModuleConfig(prefix='host/only')}
        original_env = dict(config.verify_env)
        original_registry = config.module_configs_or_empty

        apply_task_verify_spec(config, _spec())

        assert config.verify_env == original_env
        assert config.module_configs_or_empty is original_registry
        assert original_registry == {'host/only': ModuleConfig(prefix='host/only')}
        assert config.verify_cold_preprovision_command == 'host-prep'
