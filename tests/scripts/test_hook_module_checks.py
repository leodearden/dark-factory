"""scripts/hook_module_checks.py lists the merge gate's checks for a staged file set.

Driven through its CLI only (staged paths on stdin, ``unit<TAB>command`` lines
on stdout, exit 0 or 2) over throwaway layouts from ``hook_checks_fixture``.
Contract: plans/merge-lane-policy-prd.md D8 / §6 'Hook contract'.
"""
from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml
from hook_checks_fixture import (
    DEFAULT_MODULES,
    ESCALATION,
    ORCHESTRATOR,
    REPO_ROOT,
    SCREEN_MANIFEST_REL,
    SCRIPTS,
    SHARED,
    ModuleSpec,
    make_layout,
    run_helper,
    stub_uv_env,
    write_file,
)
from orchestrator.verify_cmd import (
    apply_pytest_numprocesses,
    parse_config_command,
    render,
    scope_to,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


def _legs(module: ModuleSpec) -> dict[str, str]:
    legs = {'lint': module.lint_command, 'type': module.type_check_command}
    return {f'module:{module.prefix}:{leg}': cmd for leg, cmd in legs.items() if cmd is not None}


def _assert_units(units: list[tuple[str, str]], expected: dict[str, str]) -> None:
    assert len(units) == len(dict(units)), f'a unit is listed twice: {units!r}'
    assert dict(units) == expected


def test_a_staged_scripts_file_lists_that_modules_legs_verbatim(tmp_path: Path) -> None:
    run = run_helper(make_layout(tmp_path), ['scripts/foo.py'])

    assert run.rc == 0, run.stderr
    _assert_units(run.units, _legs(SCRIPTS))


def test_source_and_test_files_of_one_module_list_only_that_module(tmp_path: Path) -> None:
    run = run_helper(make_layout(tmp_path), ['orchestrator/src/x.py', 'orchestrator/tests/test_y.py'])

    assert run.rc == 0, run.stderr
    _assert_units(run.units, _legs(ORCHESTRATOR))


@pytest.mark.parametrize('dependency', [SHARED, ESCALATION], ids=lambda m: m.prefix)
def test_a_dependency_module_type_checks_every_module(tmp_path: Path, dependency: ModuleSpec) -> None:
    run = run_helper(make_layout(tmp_path), [f'{dependency.prefix}/src/x.py'])

    every_type_leg = {
        f'module:{m.prefix}:type': m.type_check_command
        for m in DEFAULT_MODULES
        if m.type_check_command is not None
    }
    assert run.rc == 0, run.stderr
    lint_leg = {unit: cmd for unit, cmd in _legs(dependency).items() if unit.endswith(':lint')}
    _assert_units(run.units, {**lint_leg, **every_type_leg})


def test_a_dependency_fan_out_skips_a_module_without_a_type_check(tmp_path: Path) -> None:
    untyped = ModuleSpec('sampler', 'uv run --directory sampler ruff check src/', None, None, True)
    run = run_helper(make_layout(tmp_path, (*DEFAULT_MODULES, untyped)), ['shared/src/x.py'])

    assert run.rc == 0, run.stderr
    assert 'module:sampler:type' not in dict(run.units)
    assert 'module:orchestrator:type' in dict(run.units)


@pytest.mark.parametrize(
    'staged',
    [['conftest.py'], ['scripts/run.sh'], []],
    ids=['py-under-no-module', 'non-py-non-screen', 'nothing-staged'],
)
def test_nothing_to_check_is_an_empty_listing(tmp_path: Path, staged: list[str]) -> None:
    run = run_helper(make_layout(tmp_path), staged)

    assert (run.rc, run.units) == (0, []), run.stderr


def test_a_missing_top_level_config_cannot_list(tmp_path: Path) -> None:
    run = run_helper(make_layout(tmp_path, top_config=False), ['scripts/foo.py'])

    assert run.rc == 2
    assert run.units == []
    assert run.stderr.startswith('hook_module_checks: ')
    assert 'dark-factory-orchestrator.yaml' in run.stderr


def test_an_unregistered_dependency_module_cannot_list(tmp_path: Path) -> None:
    layout = make_layout(tmp_path, (SCRIPTS, SHARED, ORCHESTRATOR))

    run = run_helper(layout, ['scripts/foo.py'])

    assert run.rc == 2
    assert run.units == []
    assert 'escalation' in run.stderr
    assert 'dependency module' in run.stderr


ORCH_NODE_IDS = ['tests/test_a.py::test_x', 'tests/test_b.py']
SCRIPTS_NODE_IDS = ['scripts/tests/test_c.py']
ORCH_CHECK: dict[str, object] = {'id': 'orch_check', 'module': 'orchestrator', 'node_ids': ORCH_NODE_IDS}
SCRIPTS_CHECK: dict[str, object] = {'id': 'scripts_check', 'module': 'scripts', 'node_ids': SCRIPTS_NODE_IDS}
NODE_FILES = ('orchestrator/tests/test_a.py', 'orchestrator/tests/test_b.py', 'scripts/tests/test_c.py')
CHAINED = ModuleSpec('chained', None, None, 'uv run --project shared pytest chained/tests/ -q && echo done', False)


def _manifest(*checks: Mapping[str, object], **top: object) -> str:
    return yaml.safe_dump({'checks': list(checks), **top}, sort_keys=False)


def _screen_layout(
    root: Path,
    manifest: str | None = None,
    modules: tuple[ModuleSpec, ...] = DEFAULT_MODULES,
) -> Path:
    body = _manifest(ORCH_CHECK, SCRIPTS_CHECK) if manifest is None else manifest
    layout = make_layout(root, modules, screen_manifest=body)
    for rel in (*NODE_FILES, 'chained/tests/test_z.py'):
        write_file(layout, rel, 'def test_x():\n    pass\n')
    return layout


def _expected_screen_command(module: ModuleSpec, node_ids: list[str]) -> str:
    assert module.test_command is not None
    parsed = parse_config_command(module.test_command)
    return render(apply_pytest_numprocesses(scope_to(parsed, node_ids), '0'))


@pytest.mark.parametrize(
    'staged', ['plans/x.md', 'docs/a.txt', 'review/r.txt', 'foo.json', 'scripts/baseline.json'],
)
def test_a_tree_read_file_runs_every_screen_check_in_manifest_order(tmp_path: Path, staged: str) -> None:
    run = run_helper(_screen_layout(tmp_path), [staged])

    assert run.rc == 0, run.stderr
    assert [unit for unit, _ in run.units] == ['screen:orch_check', 'screen:scripts_check']


@pytest.mark.parametrize('staged', ['shared/notes.md', 'orchestrator/x.yaml', 'scripts/foo.py'])
def test_package_data_and_python_do_not_trigger_the_screen(tmp_path: Path, staged: str) -> None:
    run = run_helper(_screen_layout(tmp_path), [staged])

    assert run.rc == 0, run.stderr
    assert not [unit for unit, _ in run.units if unit.startswith('screen:')]


def test_a_screen_command_is_its_modules_test_command_scoped_to_the_node_ids(tmp_path: Path) -> None:
    run = run_helper(_screen_layout(tmp_path), ['plans/x.md'])

    commands = dict(run.units)
    orch_command = commands['screen:orch_check']
    assert run.rc == 0, run.stderr
    assert orch_command == _expected_screen_command(ORCHESTRATOR, ORCH_NODE_IDS)
    assert orch_command.startswith('cd orchestrator && uv run pytest')
    assert all(node_id in orch_command for node_id in ORCH_NODE_IDS)
    assert '-n 0' in orch_command
    assert 'tests/' not in orch_command.split()
    assert commands['screen:scripts_check'] == _expected_screen_command(SCRIPTS, SCRIPTS_NODE_IDS)


@pytest.mark.parametrize(
    ('manifest', 'named'),
    [
        (_manifest({**ORCH_CHECK, 'extra': 1}), ['orch_check', 'extra']),
        (_manifest(ORCH_CHECK, other=1), ['other']),
        (_manifest({**ORCH_CHECK, 'module': 'nope'}), ['orch_check', 'nope']),
        (_manifest({**ORCH_CHECK, 'node_ids': []}), ['orch_check', 'node_ids']),
        (
            _manifest({**ORCH_CHECK, 'node_ids': ['tests/test_missing.py::test_x']}),
            ['orch_check', 'tests/test_missing.py'],
        ),
        (_manifest(ORCH_CHECK, ORCH_CHECK), ['orch_check', 'duplicate']),
        (_manifest({**ORCH_CHECK, 'id': 'bad id'}), ['bad id', 'whitespace']),
        (
            _manifest({'id': 'chained_check', 'module': 'chained', 'node_ids': ['chained/tests/test_z.py']}),
            ['chained_check', 'chained'],
        ),
    ],
    ids=[
        'unknown-entry-key', 'unknown-top-key', 'unregistered-module', 'empty-node-ids',
        'missing-node-file', 'duplicate-id', 'whitespace-id', 'unscopable-test-command',
    ],
)
def test_an_invalid_manifest_cannot_list_a_triggered_screen(
    tmp_path: Path, manifest: str, named: list[str],
) -> None:
    run = run_helper(_screen_layout(tmp_path, manifest, (*DEFAULT_MODULES, CHAINED)), ['plans/x.md'])

    assert (run.rc, run.units) == (2, [])
    assert run.stderr.startswith('hook_module_checks: ')
    assert all(name in run.stderr for name in named), run.stderr


def test_a_missing_manifest_cannot_list_a_triggered_screen(tmp_path: Path) -> None:
    layout = make_layout(tmp_path)

    run = run_helper(layout, ['plans/x.md'])

    assert (run.rc, run.units) == (2, [])
    assert SCREEN_MANIFEST_REL in run.stderr


def test_an_untriggered_screen_never_reads_the_manifest(tmp_path: Path) -> None:
    layout = make_layout(tmp_path, screen_manifest='checks: [not: a, valid: entry\n')

    run = run_helper(layout, ['scripts/foo.py'])

    assert run.rc == 0, run.stderr
    assert [unit for unit, _ in run.units] == ['module:scripts:lint', 'module:scripts:type']


def test_the_real_tree_screens_exactly_the_committed_manifest() -> None:
    committed = yaml.safe_load((REPO_ROOT / SCREEN_MANIFEST_REL).read_text(encoding='utf-8'))

    run = run_helper(REPO_ROOT, ['plans/_probe.md'])

    assert run.rc == 0, run.stderr
    assert [unit for unit, _ in run.units] == [f'screen:{check["id"]}' for check in committed['checks']]


COLLECT_TIMEOUT_SECS = 300


def test_every_real_screen_unit_collects_its_node_ids(tmp_path: Path) -> None:
    """A renamed screen test fails here, where the rename is made, not on main's next docs commit."""
    run = run_helper(REPO_ROOT, ['plans/_probe.md'])
    assert run.rc == 0, run.stderr
    env = {**stub_uv_env(tmp_path, tmp_path / 'uv.log'), 'PYTEST_ADDOPTS': '--collect-only'}

    procs = {
        unit: subprocess.Popen(
            ['bash', '-c', command],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for unit, command in run.units
    }
    collected: dict[str, tuple[int, str]] = {}
    try:
        for unit, proc in procs.items():
            out, _ = proc.communicate(timeout=COLLECT_TIMEOUT_SECS)
            collected[unit] = (proc.returncode, out)
    finally:
        for proc in procs.values():
            proc.kill()
            proc.wait()

    failed = {unit: out[-2000:] for unit, (rc, out) in collected.items() if rc != 0}
    assert failed == {}
