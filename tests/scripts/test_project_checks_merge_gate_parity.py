"""A main-branch commit runs the merge gate's module checks and screen, through a REAL ``git commit``.

Boundary rows 10 and 11 of plans/merge-lane-policy-prd.md D8: a type error in a
staged module file, and a tree-read file the screen's manifest test rejects,
are both refused at commit time with the failing unit named. Each case commits
in a throwaway repo through the real hooks/pre-commit and hooks/project-checks;
the stub ``uv`` from ``hook_checks_fixture`` runs real pyright and real pytest,
so the refusals are genuine failures rather than stub verdicts.
"""
from __future__ import annotations

import dataclasses
import re
import subprocess
from pathlib import Path

import yaml
from hook_checks_fixture import (
    ESCALATION,
    ORCHESTRATOR,
    REPO_ROOT,
    SCRIPTS,
    SHARED,
    HookRepo,
    make_hook_repo,
)

BAD_TYPE = 'x: int = "not an int"\n'
TIMED_LINE = re.compile(r'merge-gate checks\b.* in [0-9]+s')
SCRIPTS_TYPE_COMMAND = 'uv run --project shared pyright scripts/'
MODULES = (
    dataclasses.replace(SCRIPTS, type_check_command=SCRIPTS_TYPE_COMMAND),
    SHARED,
    ESCALATION,
    ORCHESTRATOR,
)
SCREEN_CHECK_ID = 'capability_manifest_script_targets'
SCREEN_MANIFEST = yaml.safe_dump({
    'checks': [{
        'id': SCREEN_CHECK_ID,
        'module': 'shared',
        'node_ids': [
            'tests/test_capability_manifest.py::TestCheckedInScriptCheckTargets::'
            'test_target_exists_and_is_executable',
            'tests/test_capability_manifest.py::TestCheckedInScriptCheckTargets::'
            'test_target_is_committed_executable',
        ],
    }],
})
CAPABILITY_TEST_COPIES = {
    f'shared/tests/{name}': REPO_ROOT / 'shared' / 'tests' / name
    for name in ('test_capability_manifest.py', 'capability_manifest_corpus.py')
}


def _capability_manifest(script: str) -> str:
    return yaml.safe_dump({
        'prd': 'plans/x.md',
        'schema_version': 1,
        'tasks': [{
            'label': 'a',
            'capabilities': [{
                'name': 'tool-runs',
                'binding': 'capability->producer',
                'verdict': 'PASS',
                'delivered_check': {'kind': 'script', 'script': script, 'timeout_secs': 60},
            }],
        }],
    })


def _assert_lands(repo: HookRepo, before: str, result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stdout
    assert repo.head() != before


def _assert_refused(repo: HookRepo, before: str, result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode != 0, result.stdout
    assert repo.head() == before


def test_row_10_a_type_error_in_a_staged_module_file_is_refused(tmp_path: Path) -> None:
    repo = make_hook_repo(tmp_path, MODULES)
    before = repo.head()
    repo.stage('scripts/foo.py', BAD_TYPE)

    refused = repo.commit()

    out = refused.stdout
    _assert_refused(repo, before, refused)
    assert 'module:scripts:type' in out
    assert SCRIPTS_TYPE_COMMAND in out
    assert re.search(r'scripts/foo\.py:1:\d+ - error', out), out
    assert TIMED_LINE.search(out), out
    _assert_lands(repo, before, repo.commit('--no-verify'))


def test_row_10_control_a_well_typed_module_file_lands(tmp_path: Path) -> None:
    repo = make_hook_repo(tmp_path, MODULES)
    before = repo.head()
    repo.stage('scripts/foo.py', 'x: int = 1\n')

    landed = repo.commit()

    _assert_lands(repo, before, landed)
    assert 'ok module:scripts:type' in landed.stdout
    assert TIMED_LINE.search(landed.stdout), landed.stdout


def _screen_repo(tmp_path: Path) -> HookRepo:
    return make_hook_repo(tmp_path, MODULES, screen_manifest=SCREEN_MANIFEST, copies=CAPABILITY_TEST_COPIES)


def test_row_11_a_tree_read_file_the_screen_rejects_is_refused(tmp_path: Path) -> None:
    repo = _screen_repo(tmp_path)
    before = repo.head()
    repo.stage('plans/x.capability-manifest.yaml', _capability_manifest('scripts/missing_tool.py'))

    refused = repo.commit()

    out = refused.stdout
    _assert_refused(repo, before, refused)
    assert f'screen:{SCREEN_CHECK_ID}' in out
    assert 'test_target_exists_and_is_executable' in out
    assert 'scripts/missing_tool.py' in out
    assert TIMED_LINE.search(out), out


def test_row_11_control_a_tree_read_file_the_screen_accepts_lands(tmp_path: Path) -> None:
    repo = _screen_repo(tmp_path)
    before = repo.head()
    repo.stage('scripts/present_tool.py', '#!/usr/bin/env python3\n', executable=True)
    repo.stage('plans/x.capability-manifest.yaml', _capability_manifest('scripts/present_tool.py'))

    landed = repo.commit()

    _assert_lands(repo, before, landed)
    assert f'ok screen:{SCREEN_CHECK_ID}' in landed.stdout
    assert TIMED_LINE.search(landed.stdout), landed.stdout


def test_a_commit_with_nothing_to_check_runs_no_check(tmp_path: Path) -> None:
    repo = make_hook_repo(tmp_path, MODULES)
    before = repo.head()
    repo.stage('notes.txt', 'hello\n')

    landed = repo.commit()

    _assert_lands(repo, before, landed)
    assert not {'pyright', 'pytest'} & set(repo.uv_tools()), repo.uv_tools()
    assert TIMED_LINE.search(landed.stdout), landed.stdout


def test_a_branch_commit_runs_no_merge_gate_check(tmp_path: Path) -> None:
    repo = make_hook_repo(tmp_path, MODULES)
    repo.git('checkout', '--quiet', '-b', 'feature')
    before = repo.head()
    repo.stage('scripts/foo.py', BAD_TYPE)

    landed = repo.commit()

    _assert_lands(repo, before, landed)
    assert repo.uv_tools() == []


def test_a_helper_that_cannot_list_refuses_the_commit(tmp_path: Path) -> None:
    repo = make_hook_repo(tmp_path, MODULES, top_config=False)
    before = repo.head()
    repo.stage('scripts/foo.py', 'x: int = 1\n')

    refused = repo.commit()

    _assert_refused(repo, before, refused)
    assert 'could NOT RUN' in refused.stdout
    assert 'dark-factory-orchestrator.yaml' in refused.stdout
