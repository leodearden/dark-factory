"""scripts/hook_module_checks.py lists the merge gate's checks for a staged file set.

Driven through its CLI only (staged paths on stdin, ``unit<TAB>command`` lines
on stdout, exit 0 or 2) over throwaway layouts from ``hook_checks_fixture``.
Contract: plans/merge-lane-policy-prd.md D8 / §6 'Hook contract'.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from hook_checks_fixture import (
    DEFAULT_MODULES,
    ESCALATION,
    ORCHESTRATOR,
    SCRIPTS,
    SHARED,
    ModuleSpec,
    make_layout,
    run_helper,
)


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
    _assert_units(
        run.units,
        {f'module:{dependency.prefix}:lint': dependency.lint_command, **every_type_leg},
    )


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
