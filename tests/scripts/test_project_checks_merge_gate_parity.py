"""A main-branch commit runs the merge gate's module checks and screen, through a REAL ``git commit``.

Boundary rows 10 and 11 of plans/merge-lane-policy-prd.md D8: a type error in a
staged module file, and a tree-read file the screen's manifest test rejects,
are both refused at commit time with the failing unit named. Each case commits
in a throwaway repo through the real hooks/pre-commit and hooks/project-checks;
the stub ``uv`` from ``hook_checks_fixture`` runs real pyright and real pytest,
so the refusals are genuine failures rather than stub verdicts. The hook runs
at most ``GATE_SLOTS`` units at once, and an interrupted commit stops them.
"""
from __future__ import annotations

import dataclasses
import os
import re
import shlex
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from hook_checks_fixture import (
    ESCALATION,
    ORCHESTRATOR,
    REPO_ROOT,
    SCRIPTS,
    SHARED,
    HookRepo,
    ModuleSpec,
    make_hook_repo,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

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


def _failure_line(unit: str) -> re.Pattern[str]:
    return re.compile(rf'^pre-commit: {re.escape(unit)} FAILED \(rc=[1-9][0-9]*\): (?P<command>.+)$', re.MULTILINE)


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
    failure = _failure_line('module:scripts:type').search(out)
    _assert_refused(repo, before, refused)
    assert failure is not None, out
    assert failure['command'] == SCRIPTS_TYPE_COMMAND
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
    assert _failure_line(f'screen:{SCREEN_CHECK_ID}').search(out), out
    assert re.search(
        r'^FAILED \S*test_capability_manifest\.py::TestCheckedInScriptCheckTargets::'
        r'test_target_exists_and_is_executable\b',
        out,
        re.MULTILINE,
    ), out
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


GATE_SLOTS = 4  # hooks/project-checks::GATE_SLOTS
SLOT_PROBE = """mkdir "$1/$3"
ls "$1" | wc -l >> "$2"
sleep 2
rmdir "$1/$3"
"""


def _type_only_modules(command_for: Callable[[str], str], *extra: str) -> tuple[ModuleSpec, ...]:
    dependencies = tuple(ModuleSpec(prefix, None, command_for(prefix), None, True) for prefix in ('shared', 'escalation'))
    return (*dependencies, *(ModuleSpec(prefix, None, command_for(prefix), None, False) for prefix in extra))


def test_at_most_gate_slots_units_run_at_once(tmp_path: Path) -> None:
    running = tmp_path / 'running'
    running.mkdir()
    occupancy = tmp_path / 'occupancy.log'
    probe = tmp_path / 'slot_probe.sh'
    probe.write_text(SLOT_PROBE, encoding='utf-8')
    modules = _type_only_modules(
        lambda prefix: shlex.join(['sh', str(probe), str(running), str(occupancy), prefix]),
        'm1', 'm2', 'm3', 'm4',
    )
    repo = make_hook_repo(tmp_path, modules)
    before = repo.head()
    repo.stage('shared/src/x.py', 'x = 1\n')

    landed = repo.commit()

    seen = [int(line) for line in occupancy.read_text(encoding='utf-8').split()]
    _assert_lands(repo, before, landed)
    assert len(seen) == len(modules), landed.stdout
    assert 1 < max(seen) <= GATE_SLOTS, seen


def _pid_alive(pid: int) -> bool:
    try:
        stat = Path(f'/proc/{pid}/stat').read_text(encoding='utf-8')
    except FileNotFoundError:
        return False
    return stat.rpartition(')')[2].split()[0] != 'Z'


def _eventually(condition: Callable[[], bool], timeout_secs: float) -> bool:
    deadline = time.monotonic() + timeout_secs
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.2)
    return True


def _hook_pid(git_pid: int) -> int:
    children = Path(f'/proc/{git_pid}/task/{git_pid}/children').read_text(encoding='utf-8').split()
    assert len(children) == 1, children
    return int(children[0])


def _ctrl_c(git_pid: int) -> None:
    os.killpg(git_pid, signal.SIGINT)


def _terminate_the_hook(git_pid: int) -> None:
    os.kill(_hook_pid(git_pid), signal.SIGTERM)


@pytest.mark.parametrize(
    ('unit_prelude', 'interrupt'),
    [("trap '' INT; ", _ctrl_c), ('', _terminate_the_hook)],
    ids=['ctrl-c-with-a-sigint-immune-unit', 'sigterm-to-the-hook-alone'],
)
def test_an_interrupted_commit_stops_its_units_and_removes_their_logs(
    tmp_path: Path, unit_prelude: str, interrupt: Callable[[int], None],
) -> None:
    pid_file = tmp_path / 'unit.pid'
    sleeper = f'sh -c {shlex.quote(f"{unit_prelude}echo $$ > {pid_file}; exec sleep 300")}; true'
    repo = make_hook_repo(tmp_path, _type_only_modules(lambda prefix: sleeper if prefix == 'shared' else 'true'))
    before = repo.head()
    repo.stage('shared/src/x.py', 'x = 1\n')
    hook_tmp = tmp_path / 'hook_tmp'
    hook_tmp.mkdir()
    out_path = tmp_path / 'commit.out'

    with out_path.open('w', encoding='utf-8') as out:
        commit = subprocess.Popen(
            ['git', 'commit', '-m', 't'],
            cwd=repo.root,
            env={**repo.env, 'TMPDIR': str(hook_tmp)},
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    unit_pid: int | None = None
    try:
        assert _eventually(lambda: pid_file.is_file() and pid_file.read_text().strip() != '', 120), (
            out_path.read_text(encoding='utf-8')
        )
        pid = unit_pid = int(pid_file.read_text())
        interrupt(commit.pid)
        commit.wait(timeout=60)
        stopped = _eventually(lambda: not _pid_alive(pid), 30)
        logs_removed = _eventually(lambda: not any(hook_tmp.iterdir()), 30)
    finally:
        if commit.poll() is None:
            os.killpg(commit.pid, signal.SIGKILL)
            commit.wait()
        if unit_pid is not None and _pid_alive(unit_pid):
            os.kill(unit_pid, signal.SIGKILL)

    assert repo.head() == before
    assert stopped, f'unit pid {unit_pid} outlived the interrupted commit'
    assert logs_removed, sorted(path.name for path in hook_tmp.iterdir())
