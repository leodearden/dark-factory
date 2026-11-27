"""Verify's injection of the ``_df_defect_plugin`` pytest plugin (plans/defect-tests-prd.md decision 19).

Part 1 covers the ``verify_cmd`` mutator and the constants that name where the
plugin is imported from. Part 2 covers the PYTHONPATH merge in the target
subprocess env. Every Part 2 test sets or deletes PYTHONPATH itself, because a
verify run of this suite carries the plugin directory once this lands. Part 3
covers the sibling collection map in the junit report's lifecycle. Part 4 is the
user-observable signal (B18, B21, B22): a REAL ``run_verification`` driving a
real pytest over a non-workspace fixture project, after
test_merge_gate_rerun_end_to_end.py.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import shlex
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT
from shared.defect_plugin_location import DEFECT_PLUGIN_DIR

from orchestrator import verify as verify_module
from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.defect_tests_config import DefectTestsConfig, DefectTestsMode
from orchestrator.verify import (
    VerifyResult,
    _archive_junit_report,
    _clear_junit_report,
    _run_cmd,
    _target_subprocess_env,
    collection_map_path,
    run_verification,
)
from orchestrator.verify_cmd import (
    DEFECT_PLUGIN_MODULE,
    ToolKind,
    apply_pytest_numprocesses,
    carries_defect_plugin,
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


class TestCarriesDefectPlugin:
    @pytest.mark.parametrize('order', ['plugin_only', 'junit_after'])
    def test_an_injected_command_carries_it_after_a_reparse(self, order):
        cmd = with_defect_plugin(parse_config_command('pytest tests/'))
        if order == 'junit_after':
            cmd = with_junitxml(cmd, '/tmp/r.xml')
        assert carries_defect_plugin(parse_config_command(render(cmd)))

    def test_an_uninjected_command_does_not(self):
        assert not carries_defect_plugin(parse_config_command('pytest -p xdist tests/'))

    @pytest.mark.parametrize(
        'raw',
        [
            'pytest -p _df_defect_plugin tests/ && pytest -p _df_defect_plugin other/',
            'PYTHONPATH=src pytest -p _df_defect_plugin tests/',
        ],
    )
    def test_a_command_the_mutator_cannot_rewrite_never_does(self, raw):
        assert not carries_defect_plugin(parse_config_command(raw))


class TestPluginLocation:
    def test_the_named_module_is_in_the_named_directory(self):
        assert DEFECT_PLUGIN_DIR.name == '_pytest_inject'
        assert (DEFECT_PLUGIN_DIR / f'{DEFECT_PLUGIN_MODULE}.py').is_file()


_DIR = str(DEFECT_PLUGIN_DIR)


class TestPythonpathMerge:
    def test_off_passes_ambient_through(self, monkeypatch):
        monkeypatch.setenv('PYTHONPATH', '/a')
        assert _target_subprocess_env(None)['PYTHONPATH'] == '/a'

    def test_off_leaves_absent_absent(self, monkeypatch):
        monkeypatch.delenv('PYTHONPATH', raising=False)
        assert 'PYTHONPATH' not in _target_subprocess_env(None)

    def test_on_with_none_is_the_dir_alone(self, monkeypatch):
        monkeypatch.delenv('PYTHONPATH', raising=False)
        assert _target_subprocess_env(None, defect_plugin=True)['PYTHONPATH'] == _DIR

    def test_on_appends_after_ambient_entries(self, monkeypatch):
        monkeypatch.setenv('PYTHONPATH', os.pathsep.join(['/a', '/b']))
        env = _target_subprocess_env(None, defect_plugin=True)
        assert env['PYTHONPATH'] == os.pathsep.join(['/a', '/b', _DIR])

    def test_on_merges_after_the_caller_overlay(self, monkeypatch):
        monkeypatch.setenv('PYTHONPATH', '/a')
        env = _target_subprocess_env({'PYTHONPATH': '/x', 'FOO': '1'}, defect_plugin=True)
        assert env['PYTHONPATH'] == os.pathsep.join(['/x', _DIR])
        assert env['FOO'] == '1'

    @pytest.mark.parametrize('source', ['ambient', 'extra'])
    def test_on_does_not_duplicate_a_present_dir(self, monkeypatch, source):
        value = os.pathsep.join(['/a', _DIR])
        if source == 'ambient':
            monkeypatch.setenv('PYTHONPATH', value)
            env = _target_subprocess_env(None, defect_plugin=True)
        else:
            monkeypatch.delenv('PYTHONPATH', raising=False)
            env = _target_subprocess_env({'PYTHONPATH': value}, defect_plugin=True)
        assert env['PYTHONPATH'] == value

    def test_on_with_empty_ambient_is_the_dir_alone(self, monkeypatch):
        monkeypatch.setenv('PYTHONPATH', '')
        assert _target_subprocess_env(None, defect_plugin=True)['PYTHONPATH'] == _DIR

    def test_every_other_key_is_unchanged(self, monkeypatch, tmp_path):
        monkeypatch.setenv('PYTHONPATH', '/a')
        monkeypatch.setenv('ORCH_CONFIG_PATH', '/leaked')
        off = _target_subprocess_env({'FOO': '1'}, worktree=tmp_path)
        on = _target_subprocess_env({'FOO': '1'}, worktree=tmp_path, defect_plugin=True)
        off.pop('PYTHONPATH')
        on.pop('PYTHONPATH')
        assert on == off
        assert on['RUFF_CACHE_DIR'] == str(tmp_path / '.ruff_cache')
        assert on['PYTHONUNBUFFERED'] == '1'
        assert 'ORCH_CONFIG_PATH' not in on


class TestRunCmdForwardsTheFlag:
    @pytest.mark.asyncio
    async def test_on_reaches_the_child(self, monkeypatch, tmp_path: Path):
        monkeypatch.delenv('PYTHONPATH', raising=False)
        rc, output, _ = await _run_cmd('printf %s "$PYTHONPATH"', tmp_path, 60.0, defect_plugin=True)
        assert rc == 0
        assert output.endswith(_DIR)

    @pytest.mark.asyncio
    async def test_off_does_not(self, monkeypatch, tmp_path: Path):
        monkeypatch.delenv('PYTHONPATH', raising=False)
        rc, output, _ = await _run_cmd('printf %s "$PYTHONPATH"', tmp_path, 60.0)
        assert rc == 0
        assert not output.endswith(_DIR)


_MAP_BYTES = b'{"tests/test_x.py::test_d": {"criterion": null, "defects": ["human:20261119T000000Z:abcdef12"]}}'


def _live_report(directory: Path, *, with_map: bool) -> Path:
    junit = directory / 'r.xml'
    junit.write_text('<testsuites/>', encoding='utf-8')
    if with_map:
        collection_map_path(junit).write_bytes(_MAP_BYTES)
    return junit


class TestCollectionMapPath:
    def test_live_report(self):
        assert collection_map_path(Path('/w/r.xml')) == Path('/w/r.xml.tests.json')

    def test_archived_report(self):
        archived = Path('/a/t1/attempt-1.junit-20261119T000000_000000Z.xml.gz')
        assert collection_map_path(archived) == Path(
            '/a/t1/attempt-1.junit-20261119T000000_000000Z.xml.gz.tests.json'
        )


class TestClearJunitReport:
    def test_removes_the_report_and_its_map(self, tmp_path):
        junit = _live_report(tmp_path, with_map=True)
        _clear_junit_report(junit)
        assert not junit.exists()
        assert not collection_map_path(junit).exists()

    @pytest.mark.parametrize('present', ['report_only', 'map_only', 'neither'])
    def test_quiet_when_either_is_absent(self, tmp_path, caplog, present):
        junit = tmp_path / 'r.xml'
        if present == 'report_only':
            junit.write_text('<testsuites/>', encoding='utf-8')
        elif present == 'map_only':
            collection_map_path(junit).write_bytes(_MAP_BYTES)
        with caplog.at_level(logging.WARNING, logger='orchestrator.verify'):
            _clear_junit_report(junit)
        assert not junit.exists()
        assert not collection_map_path(junit).exists()
        assert caplog.records == []


class TestArchiveJunitReport:
    def test_archives_the_map_beside_the_archived_report(self, tmp_path):
        junit = _live_report(tmp_path, with_map=True)
        dest = _archive_junit_report(junit, tmp_path / 'archive', 't1', 1, module_prefix='pkg')
        assert dest is not None
        assert dest.name.startswith('attempt-1.pkg.junit-') and dest.name.endswith('.xml.gz')
        assert gzip.decompress(dest.read_bytes()) == b'<testsuites/>'
        assert collection_map_path(dest).read_bytes() == _MAP_BYTES

    def test_no_map_archives_only_the_report(self, tmp_path):
        junit = _live_report(tmp_path, with_map=False)
        dest = _archive_junit_report(junit, tmp_path / 'archive', 't1', 1, module_prefix='pkg')
        assert dest is not None and dest.is_file()
        assert list((tmp_path / 'archive' / 't1').glob('*.tests.json')) == []

    def test_missing_report_archives_no_stray_map(self, tmp_path):
        junit = tmp_path / 'r.xml'
        collection_map_path(junit).write_bytes(_MAP_BYTES)
        assert _archive_junit_report(junit, tmp_path / 'archive', 't1', 1, module_prefix='pkg') is None
        archived = tmp_path / 'archive' / 't1'
        assert not archived.exists() or list(archived.iterdir()) == []

    def test_segment_archive_carries_its_map(self, tmp_path):
        junit = _live_report(tmp_path, with_map=True)
        dest = _archive_junit_report(
            junit, tmp_path / 'archive', 't1', 1, module_prefix='pkg', segment_label='seg0',
        )
        assert dest is not None
        assert dest.name.startswith('attempt-1.pkg.seg0.junit-')
        assert collection_map_path(dest).read_bytes() == _MAP_BYTES

    def test_map_copy_failure_warns_and_keeps_the_report(self, tmp_path, caplog, monkeypatch):
        junit = _live_report(tmp_path, with_map=True)

        def _refuse(*_args, **_kwargs):
            raise PermissionError('denied')

        monkeypatch.setattr(shutil, 'copyfile', _refuse)
        with caplog.at_level(logging.WARNING, logger='orchestrator.verify'):
            dest = _archive_junit_report(junit, tmp_path / 'archive', 't1', 1, module_prefix='pkg')
        assert dest is not None and dest.is_file()
        assert not collection_map_path(dest).exists()
        assert any(str(collection_map_path(junit)) in record.getMessage() for record in caplog.records)


_FIXTURE_REF = 'human:20261119T000000Z:abcdef12'
_EXPECTED_FIXTURE_MAP = {
    'pkg/tests/test_x.py::test_defect': {'defects': [_FIXTURE_REF], 'criterion': None},
}


def _make_fixture_project(root: Path, *, registers_defect_marker: bool = False) -> Path:
    """A NON-workspace pytest project: no ``shared`` of its own, a strict ini, a marker of its own.

    Without the plugin, its ``defect`` mark is unknown and ``filterwarnings =
    error`` makes that a collection error, so a run that injects nothing passes
    only when the project registers the marker itself, as every workspace member does.
    Its sessions keep their basetemp inside the project, never in the host-wide pytest
    temp root, whose exit-time cleanup of other sessions' dead dirs is unbounded; see
    ``test_an_xdist_leg_leaves_the_hosts_default_pytest_temp_root_untouched``.
    """
    root.mkdir(parents=True)
    own_defect = '    defect(ref): registered by the project\n' if registers_defect_marker else ''
    (root / 'pytest.ini').write_text(
        '[pytest]\naddopts = --basetemp=.fixture-basetemp\n'
        'markers =\n    own: the project\'s own marker\n'
        f'{own_defect}filterwarnings = error\npythonpath = .\n',
        encoding='utf-8',
    )
    (root / 'testing.py').write_text("ORIGIN = 'project'\n", encoding='utf-8')
    tests = root / 'pkg' / 'tests'
    tests.mkdir(parents=True)
    (tests / 'test_x.py').write_text(
        'import pytest\n\nimport testing\n\n\n'
        '@pytest.mark.own\ndef test_own():\n    assert testing.ORIGIN == \'project\'\n\n\n'
        f'@pytest.mark.defect({_FIXTURE_REF!r})\ndef test_defect():\n    pass\n\n\n'
        'def test_plain():\n    pass\n',
        encoding='utf-8',
    )
    return root


def _make_config(worktree: Path, **overrides) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=worktree,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin', worktree_dir='.worktrees',
        ),
        verify_admission_enabled=False,
        verify_cold_preprovision_command='',
        verify_env={'PATH': f'{Path(sys.executable).parent}{os.pathsep}{os.environ["PATH"]}'},
        **overrides,
    )


def _verify(
    worktree: Path, config: OrchestratorConfig, test_command: str, **kwargs,
) -> tuple[VerifyResult, list[tuple[str, dict]]]:
    module = ModuleConfig(
        prefix='pkg', test_command=test_command, lint_command=None, type_check_command=None,
    )
    real_run_cmd = verify_module._run_cmd
    seen: list[tuple[str, dict]] = []

    async def spy(cmd, *args, **spy_kwargs):
        seen.append((cmd, spy_kwargs))
        return await real_run_cmd(cmd, *args, **spy_kwargs)

    with patch.object(verify_module, '_run_cmd', spy):
        result = asyncio.run(run_verification(worktree, config, module, **kwargs))
    return result, [(cmd, kw) for cmd, kw in seen if 'pytest' in cmd]


def _carries_plugin(cmd: str) -> bool:
    return _carries_pair(tuple(shlex.split(cmd)), _PLUGIN_PAIR)


@pytest.mark.xdist_group('defect_plugin_real_pytest')
@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
class TestVerifyLegInjectionEndToEnd:
    @pytest.mark.parametrize('test_command', [
        pytest.param('pytest', id='serial'),
        pytest.param('pytest -n 2', id='xdist'),
    ])
    def test_shadow_mode_injects_and_the_map_lands_where_verify_looks(
        self, tmp_path, monkeypatch, test_command,
    ):
        monkeypatch.delenv('PYTHONPATH', raising=False)
        worktree = _make_fixture_project(tmp_path / 'worktree')
        report = tmp_path / 'out' / 'r.xml'
        result, legs = _verify(
            worktree, _make_config(worktree), test_command, junit_report=report,
        )
        assert result.passed, result.test_output
        assert legs and all(_carries_plugin(cmd) for cmd, _ in legs), legs
        assert all(kw.get('defect_plugin') is True for _, kw in legs)
        assert json.loads(collection_map_path(report).read_text()) == _EXPECTED_FIXTURE_MAP

    def test_an_xdist_leg_resolves_its_pytest_basetemp_inside_the_fixture_project(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.delenv('PYTHONPATH', raising=False)
        contained = tmp_path / 'contained-temproot'
        contained.mkdir()
        monkeypatch.setenv('PYTEST_DEBUG_TEMPROOT', str(contained))
        worktree = _make_fixture_project(tmp_path / 'worktree')
        recorded = worktree / 'pkg' / 'tests' / 'session.json'
        (recorded.parent / 'test_session.py').write_text(
            'import json\nimport pathlib\n\nfrom xdist import get_xdist_worker_id\n\n\n'
            'def test_records_its_session(request, tmp_path_factory):\n'
            "    pathlib.Path(__file__).with_name('session.json').write_text(json.dumps({\n"
            "        'basetemp': str(tmp_path_factory.getbasetemp()),\n"
            "        'xdist_workerid': get_xdist_worker_id(request),\n"
            "    }), encoding='utf-8')\n",
            encoding='utf-8',
        )

        result, _ = _verify(
            worktree, _make_config(worktree), 'pytest -n 2', junit_report=tmp_path / 'out' / 'r.xml',
        )

        assert result.passed, result.test_output
        session = json.loads(recorded.read_text(encoding='utf-8'))
        assert session['xdist_workerid'].startswith('gw'), (
            'the recorder ran outside any xdist worker, so the leg had no xdist '
            'controller, which is what resolves a basetemp and arms the exit-time '
            'sweep: the assertion below would pass vacuously '
            f"(worker id {session['xdist_workerid']!r})"
        )
        controller_basetemp = Path(session['basetemp']).parent
        assert controller_basetemp.is_relative_to(worktree.resolve()), (
            'a nested xdist verify leg must resolve its basetemp inside the fixture '
            'project: a basetemp under PYTEST_DEBUG_TEMPROOT makes a numbered dir '
            "there and arms pytest's unbounded exit-time sweep of other sessions' "
            f'dead dirs\ncontroller basetemp: {controller_basetemp}\n'
            f'fixture project:     {worktree.resolve()}'
        )

    def test_task_cost_archive_carries_the_map(self, tmp_path, monkeypatch):
        monkeypatch.delenv('PYTHONPATH', raising=False)
        worktree = _make_fixture_project(tmp_path / 'worktree')
        (worktree / '.task').mkdir()
        archive = tmp_path / 'archive'
        result, _ = _verify(
            worktree, _make_config(worktree), 'pytest',
            role='task', attempt_id=1, task_id='t1', archive_root=archive,
        )
        assert result.passed, result.test_output
        archived = list((archive / 't1').glob('attempt-1.pkg.junit-*.xml.gz'))
        assert len(archived) == 1, list((archive / 't1').iterdir())
        assert json.loads(collection_map_path(archived[0]).read_text()) == _EXPECTED_FIXTURE_MAP

    def test_mode_off_injects_nothing(self, tmp_path, monkeypatch):
        monkeypatch.delenv('PYTHONPATH', raising=False)
        worktree = _make_fixture_project(tmp_path / 'worktree', registers_defect_marker=True)
        report = tmp_path / 'out' / 'r.xml'
        config = _make_config(worktree, defect_tests=DefectTestsConfig(mode=DefectTestsMode.OFF))
        result, legs = _verify(worktree, config, 'pytest', junit_report=report)
        assert result.passed, result.test_output
        assert legs and not any(DEFECT_PLUGIN_MODULE in cmd for cmd, _ in legs)
        assert all('defect_plugin' not in kw for _, kw in legs)
        assert not collection_map_path(report).exists()

    def test_raw_retained_chain_runs_byte_identical_and_says_so(self, tmp_path, monkeypatch, caplog):
        monkeypatch.delenv('PYTHONPATH', raising=False)
        worktree = _make_fixture_project(tmp_path / 'worktree', registers_defect_marker=True)
        chain = 'pytest pkg/tests/test_x.py && pytest pkg/tests/test_x.py'
        assert parse_config_command(chain).raw is not None
        with caplog.at_level(logging.INFO, logger='orchestrator.verify'):
            result, legs = _verify(worktree, _make_config(worktree), chain)
        assert result.passed, result.test_output
        assert [cmd for cmd, _ in legs] == [chain]
        assert all('defect_plugin' not in kw for _, kw in legs), legs
        assert any(
            chain in record.getMessage() and 'collection map' in record.getMessage()
            for record in caplog.records
        ), [record.getMessage() for record in caplog.records]
