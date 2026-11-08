"""Tests for the spec-agnostic layer every verify spec is built from and
applied with (``orchestrator.verify_spec_shared``).

Both the merge spec (``verify_runner.MergeVerifySpec``) and the task spec
(``task_verify_spec.TaskVerifySpec``) project ModuleConfigs into
VerifyCommands, rebuild them on the consuming host, and apply the spec onto
that host's config. These tests pin each rule at its one home.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify_spec_shared import (
    VerifyCommand,
    apply_spec_to_host_config,
    canonical_json,
    global_verify_command_for,
    module_config_from_command,
    require_exact_keys,
)


def _host_config(tmp_path: Path, **overrides) -> OrchestratorConfig:
    fields = {
        'project_root': tmp_path,
        'test_command': 'host-test',
        'lint_command': 'host-lint',
        'type_check_command': 'host-type',
        'verify_env': {'HOST_ONLY': 'h', 'SHARED': 'host'},
    }
    return OrchestratorConfig(**{**fields, **overrides})


class TestVerifyCommand:
    def test_is_frozen(self):
        vc = VerifyCommand(prefix='m', test_command='t')

        pytest.raises(dataclasses.FrozenInstanceError, setattr, vc, 'prefix', 'other')

    def test_dict_round_trip(self):
        vc = VerifyCommand(prefix='m', test_command='t', lint_command='l', type_check_command='y')

        assert VerifyCommand.from_dict(vc.to_dict()) == vc

    def test_from_module_config_projects_prefix_and_commands_only(self):
        mc = ModuleConfig(
            prefix='m', test_command='t', lint_command='l', type_check_command='y',
            verify_env={'X': '1'},
        )

        assert VerifyCommand.from_module_config(mc) == VerifyCommand(
            prefix='m', test_command='t', lint_command='l', type_check_command='y',
        )


class TestGlobalVerifyCommandFor:
    def test_zero_modules_with_global_commands_ships_them(self, tmp_path: Path):
        config = _host_config(tmp_path)

        assert global_verify_command_for(config, ()) == VerifyCommand(
            prefix='', test_command='host-test', lint_command='host-lint',
            type_check_command='host-type',
        )

    def test_non_empty_modules_ship_no_global_command(self, tmp_path: Path):
        config = _host_config(tmp_path)

        assert global_verify_command_for(config, (VerifyCommand(prefix='m'),)) is None

    def test_no_global_command_set_ships_none(self, tmp_path: Path):
        config = _host_config(tmp_path, test_command='', lint_command='', type_check_command='')

        assert global_verify_command_for(config, ()) is None


class TestModuleConfigFromCommand:
    def test_copies_prefix_and_commands(self):
        vc = VerifyCommand(prefix='m', test_command='t', lint_command='l', type_check_command='y')

        mc = module_config_from_command(vc)

        assert (mc.prefix, mc.test_command, mc.lint_command, mc.type_check_command) == (
            'm', 't', 'l', 'y',
        )
        assert mc.verify_env is None
        assert mc.verify_cold_command_timeout_secs is None

    def test_verify_env_is_a_fresh_copy(self):
        env = {'X': '1'}

        mc = module_config_from_command(VerifyCommand(prefix='m'), verify_env=env)
        env['X'] = 'changed'

        assert mc.verify_env == {'X': '1'}

    def test_cold_timeout_is_threaded_verbatim(self):
        mc = module_config_from_command(VerifyCommand(prefix='m'), cold_timeout_secs=42.5)

        assert mc.verify_cold_command_timeout_secs == 42.5


class TestApplySpecToHostConfig:
    def _apply(self, config: OrchestratorConfig, **overrides) -> OrchestratorConfig:
        arguments = {
            'verify_env': {},
            'global_verify_command': None,
            'module_configs': [],
            'field_overrides': {},
            'verify_cold_preprovision_command': '',
        }
        return apply_spec_to_host_config(config, **{**arguments, **overrides})

    def test_spec_env_wins_and_host_only_keys_survive(self, tmp_path: Path):
        config = _host_config(tmp_path)

        applied = self._apply(config, verify_env={'SHARED': 'spec', 'SPEC_ONLY': 's'})

        assert applied.verify_env == {
            **config.effective_verify_env, 'SHARED': 'spec', 'SPEC_ONLY': 's',
        }
        assert applied.verify_env['HOST_ONLY'] == 'h'

    def test_global_command_overwrites_host_commands(self, tmp_path: Path):
        config = _host_config(tmp_path)
        global_command = VerifyCommand(
            prefix='', test_command='spec-test', lint_command='spec-lint',
            type_check_command='spec-type',
        )

        applied = self._apply(config, global_verify_command=global_command)

        assert (applied.test_command, applied.lint_command, applied.type_check_command) == (
            'spec-test', 'spec-lint', 'spec-type',
        )

    def test_no_global_command_keeps_host_commands(self, tmp_path: Path):
        applied = self._apply(_host_config(tmp_path))

        assert (applied.test_command, applied.lint_command, applied.type_check_command) == (
            'host-test', 'host-lint', 'host-type',
        )

    def test_module_registry_is_exactly_the_spec_set(self, tmp_path: Path):
        config = _host_config(tmp_path)
        config._module_configs = {'host/only': ModuleConfig(prefix='host/only')}
        spec_modules = [ModuleConfig(prefix='a'), ModuleConfig(prefix='b')]

        applied = self._apply(config, module_configs=spec_modules)

        assert applied.module_configs_or_empty == {mc.prefix: mc for mc in spec_modules}

    def test_zero_spec_modules_install_an_empty_registry(self, tmp_path: Path):
        config = _host_config(tmp_path)
        config._module_configs = {'host/only': ModuleConfig(prefix='host/only')}

        applied = self._apply(config, module_configs=[])

        assert applied.module_configs_or_empty == {}

    def test_field_overrides_are_applied(self, tmp_path: Path):
        applied = self._apply(_host_config(tmp_path, scope_cargo=True), field_overrides={'scope_cargo': False})

        assert applied.scope_cargo is False

    def test_unknown_field_override_is_rejected_naming_it(self, tmp_path: Path):
        with pytest.raises(TypeError, match='scope_cargoo'):
            self._apply(_host_config(tmp_path), field_overrides={'scope_cargoo': False})

    @pytest.mark.parametrize(('field', 'value'), [
        ('scope_cargo', 'false'),
        ('merge_verify_workspace', 1),
        ('merge_verify_breadth', 'everything'),
        ('verify_cold_preprovision_command', None),
    ])
    def test_mistyped_field_override_is_rejected_naming_it(
        self, tmp_path: Path, field: str, value: object,
    ):
        with pytest.raises(TypeError, match=field):
            self._apply(_host_config(tmp_path), field_overrides={field: value})

    def test_non_empty_preprovision_replaces_the_hosts(self, tmp_path: Path):
        config = _host_config(tmp_path, verify_cold_preprovision_command='host-prep')

        applied = self._apply(config, verify_cold_preprovision_command='spec-prep')

        assert applied.verify_cold_preprovision_command == 'spec-prep'

    def test_empty_preprovision_keeps_the_hosts(self, tmp_path: Path):
        config = _host_config(tmp_path, verify_cold_preprovision_command='host-prep')

        applied = self._apply(config, verify_cold_preprovision_command='')

        assert applied.verify_cold_preprovision_command == 'host-prep'

    @pytest.mark.parametrize('value', [1, None])
    def test_mistyped_preprovision_is_rejected_naming_it(self, tmp_path: Path, value: object):
        with pytest.raises(TypeError, match=r'OrchestratorConfig\.verify_cold_preprovision_command'):
            self._apply(_host_config(tmp_path), verify_cold_preprovision_command=value)

    def test_caller_config_is_not_mutated(self, tmp_path: Path):
        config = _host_config(tmp_path, verify_cold_preprovision_command='host-prep')
        host_modules = {'host/only': ModuleConfig(prefix='host/only')}
        config._module_configs = host_modules
        original_env = dict(config.verify_env)
        original_registry = config.module_configs_or_empty

        self._apply(
            config,
            verify_env={'SHARED': 'spec'},
            module_configs=[ModuleConfig(prefix='a')],
            verify_cold_preprovision_command='spec-prep',
        )

        assert config.verify_env == original_env
        assert config.module_configs_or_empty is original_registry
        assert original_registry == {'host/only': ModuleConfig(prefix='host/only')}
        assert config.verify_cold_preprovision_command == 'host-prep'


class TestCanonicalJson:
    def test_sorts_keys_and_keeps_non_ascii_literal(self):
        encoded = canonical_json({'b': 1, 'a': 'é'})

        assert encoded == '{"a": "é", "b": 1}'
        assert json.loads(encoded) == {'a': 'é', 'b': 1}


class TestRequireExactKeys:
    def test_exact_match_passes(self):
        assert require_exact_keys({'a': 1, 'b': 2}, frozenset({'a', 'b'}), 'Thing') is None

    def test_mismatch_names_type_unknown_and_missing_keys(self):
        with pytest.raises(TypeError) as excinfo:
            require_exact_keys({'a': 1, 'extra': 2}, frozenset({'a', 'gone'}), 'Thing')

        message = str(excinfo.value)
        assert 'Thing' in message
        assert 'extra' in message
        assert 'gone' in message

    def test_non_dict_document_is_a_type_error(self):
        with pytest.raises(TypeError) as excinfo:
            require_exact_keys([1, 2], frozenset({'a'}), 'Thing')

        assert 'Thing' in str(excinfo.value)
