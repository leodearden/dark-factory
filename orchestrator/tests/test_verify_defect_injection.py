"""Verify's injection of the ``_df_defect_plugin`` pytest plugin (plans/defect-tests-prd.md decision 19).

Part 1 covers the ``verify_cmd`` mutator and the constants that name where the
plugin is imported from.
"""

from __future__ import annotations

import pytest

from orchestrator.verify_cmd import (
    DEFECT_PLUGIN_DIR,
    DEFECT_PLUGIN_MODULE,
    ToolKind,
    apply_pytest_numprocesses,
    parse_config_command,
    render,
    serial_pytest,
    with_defect_plugin,
    with_junitxml,
)

_PLUGIN_PAIR = ('-p', DEFECT_PLUGIN_MODULE)


def _carries_pair(flags: tuple[str, ...], pair: tuple[str, str]) -> bool:
    return any(flags[i : i + 2] == pair for i in range(len(flags) - 1))


class TestWithDefectPlugin:
    def test_structured_pytest_gains_the_pair(self):
        injected = with_defect_plugin(parse_config_command('pytest tests/x.py'))
        assert render(injected) == 'pytest -p _df_defect_plugin tests/x.py'
        assert injected.base_flags[-2:] == _PLUGIN_PAIR

    @pytest.mark.parametrize('order', ['plugin_first', 'junit_first'])
    def test_composes_with_junitxml_and_survives_a_reparse(self, order):
        cmd = parse_config_command('pytest tests/x.py')
        if order == 'plugin_first':
            cmd = with_junitxml(with_defect_plugin(cmd), '/tmp/r.xml')
        else:
            cmd = with_defect_plugin(with_junitxml(cmd, '/tmp/r.xml'))
        reparsed = with_junitxml(parse_config_command(render(cmd)), '/tmp/s.xml')
        assert _carries_pair(reparsed.base_flags, _PLUGIN_PAIR)
        assert DEFECT_PLUGIN_MODULE not in reparsed.targets
        assert reparsed.targets == ('tests/x.py',)

    def test_already_injected_reparsed_command_is_returned_itself(self):
        reparsed = parse_config_command(render(with_defect_plugin(parse_config_command('pytest tests/'))))
        assert with_defect_plugin(reparsed) is reparsed

    def test_mutating_twice_returns_the_injected_command_itself(self):
        once = with_defect_plugin(parse_config_command('pytest tests/'))
        assert with_defect_plugin(once) is once

    @pytest.mark.parametrize(
        'raw',
        [
            'cd shared && uv run pytest tests/ && cd ../orchestrator && uv run pytest tests/',
            'make test',
            'ruff check src/',
            'pyright src/',
        ],
    )
    def test_non_structured_pytest_is_returned_itself(self, raw):
        cmd = parse_config_command(raw)
        if raw.startswith('cd shared'):
            assert cmd.tool is ToolKind.PYTEST and cmd.raw is not None
        assert with_defect_plugin(cmd) is cmd

    def test_coexists_with_serial_recovery(self):
        injected = with_defect_plugin(serial_pytest(parse_config_command('pytest tests/')))
        assert _carries_pair(injected.base_flags, ('-p', 'no:xdist'))
        assert _carries_pair(injected.base_flags, _PLUGIN_PAIR)
        assert apply_pytest_numprocesses(injected, '4') is injected

    @pytest.mark.parametrize('order', ['plugin_first', 'cap_first'])
    def test_coexists_with_the_worker_cap(self, order):
        cmd = parse_config_command('pytest tests/')
        if order == 'plugin_first':
            cmd = apply_pytest_numprocesses(with_defect_plugin(cmd), '4')
        else:
            cmd = with_defect_plugin(apply_pytest_numprocesses(cmd, '4'))
        assert _carries_pair(cmd.base_flags, _PLUGIN_PAIR)
        assert _carries_pair(cmd.base_flags, ('-n', '4'))


class TestPluginLocation:
    def test_the_named_module_is_in_the_named_directory(self):
        assert DEFECT_PLUGIN_DIR.name == '_pytest_inject'
        assert (DEFECT_PLUGIN_DIR / f'{DEFECT_PLUGIN_MODULE}.py').is_file()
