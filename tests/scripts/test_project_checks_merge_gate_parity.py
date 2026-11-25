"""A main-branch commit runs the merge gate's module checks and screen, through a REAL ``git commit``.

Boundary rows 10 and 11 of plans/merge-lane-policy-prd.md D8: a type error in a
staged module file, and a tree-read file the screen's manifest test rejects,
are both refused at commit time with the failing unit named. Each case commits
in a throwaway repo through the real hooks/pre-commit and hooks/project-checks;
the stub ``uv`` from ``hook_checks_fixture`` runs real pyright and real pytest,
so the refusals are genuine failures rather than stub verdicts. A staged
repo-root ``.py``, which no module owns, is refused the same way under the
hook's own ``root:type`` unit. The hook runs
at most ``GATE_SLOTS`` units at once, and an interrupted commit stops them.
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
import re
import shlex
import shutil
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
from shared.proc_group import pid_running, read_stat_fields, unsafe_pgid_reason
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
REPO_GUARDS = ModuleSpec('tests/repo_guards', None, None, 'uv run --project shared pytest tests/repo_guards/ -q', False)
SCREEN_MODULES = (*MODULES, REPO_GUARDS)
SCREEN_CHECK_ID = 'capability_manifest_script_targets'
SCREEN_MANIFEST = yaml.safe_dump({
    'checks': [{
        'id': SCREEN_CHECK_ID,
        'module': REPO_GUARDS.prefix,
        'node_ids': [
            'tests/repo_guards/test_checked_in_capability_manifests.py::TestCheckedInScriptCheckTargets::'
            'test_target_exists_and_is_executable',
            'tests/repo_guards/test_checked_in_capability_manifests.py::TestCheckedInScriptCheckTargets::'
            'test_target_is_committed_executable',
        ],
    }],
})
CAPABILITY_TEST_COPIES = {
    rel: REPO_ROOT / rel
    for rel in (
        'tests/repo_guards/test_checked_in_capability_manifests.py',
        'tests/repo_guards/conftest.py',
        'tests/repo_guards/session_prebuild.py',
        'tests/repo_guards/silent_fallthrough_scan.py',
        'shared/tests/cross_suite_helpers/capability_manifest_corpus.py',
    )
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


def test_a_type_error_in_a_staged_root_py_is_refused(tmp_path: Path) -> None:
    repo = make_hook_repo(tmp_path, MODULES)
    before = repo.head()
    repo.stage('conftest.py', BAD_TYPE)

    refused = repo.commit()

    out = refused.stdout
    failure = _failure_line('root:type').search(out)
    _assert_refused(repo, before, refused)
    assert failure is not None, out
    assert failure['command'] == 'uv run --project shared pyright conftest.py'
    assert re.search(r'conftest\.py:1:\d+ - error', out), out
    assert TIMED_LINE.search(out), out
    _assert_lands(repo, before, repo.commit('--no-verify'))


def test_a_well_typed_staged_root_py_lands(tmp_path: Path) -> None:
    repo = make_hook_repo(tmp_path, MODULES)
    before = repo.head()
    repo.stage('conftest.py', 'x: int = 1\n')

    landed = repo.commit()

    _assert_lands(repo, before, landed)
    assert 'ok root:type' in landed.stdout


def _screen_repo(tmp_path: Path) -> HookRepo:
    return make_hook_repo(tmp_path, SCREEN_MODULES, screen_manifest=SCREEN_MANIFEST, copies=CAPABILITY_TEST_COPIES)


def test_row_11_a_tree_read_file_the_screen_rejects_is_refused(tmp_path: Path) -> None:
    repo = _screen_repo(tmp_path)
    before = repo.head()
    repo.stage('plans/x.capability-manifest.yaml', _capability_manifest('scripts/missing_tool.py'))

    refused = repo.commit()

    out = refused.stdout
    _assert_refused(repo, before, refused)
    assert _failure_line(f'screen:{SCREEN_CHECK_ID}').search(out), out
    assert re.search(
        r'^FAILED \S*test_checked_in_capability_manifests\.py::TestCheckedInScriptCheckTargets::'
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


def _eventually(condition: Callable[[], bool], timeout_secs: float) -> bool:
    deadline = time.monotonic() + timeout_secs
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.2)
    return True


def _start_commit(
    repo: HookRepo, hook_tmp: Path, out_path: Path, *, bin_dir: Path | None = None,
) -> subprocess.Popen[bytes]:
    env = {**repo.env, 'TMPDIR': str(hook_tmp)}
    if bin_dir is not None:
        env['PATH'] = f'{bin_dir}{os.pathsep}{repo.env["PATH"]}'
    with out_path.open('w', encoding='utf-8') as out:
        return subprocess.Popen(
            ['git', 'commit', '-m', 't'],
            cwd=repo.root,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


@dataclasses.dataclass(frozen=True)
class _Unit:
    pid: int
    pgid: int


def _written(path: Path) -> bool:
    return path.is_file() and path.read_text().strip() != ''


def _await_written(path: Path, out_path: Path) -> str:
    """What a unit wrote to *path*, once it has written anything."""
    assert _eventually(lambda: _written(path), 120), out_path.read_text(encoding='utf-8')
    return path.read_text()


def _started_unit(pid_file: Path, out_path: Path) -> _Unit:
    """The unit that wrote *pid_file*, with its process group read while it is still alive."""
    pid = int(_await_written(pid_file, out_path))
    stat = read_stat_fields(Path(f'/proc/{pid}'), skip_malformed=False)
    assert stat is not None, f'unit pid {pid} exited before its process group could be read'
    return _Unit(pid, stat.pgrp)


def _kill_leftovers(commit: subprocess.Popen[bytes], units: list[_Unit]) -> None:
    """Each unit runs in its own session, which a SIGKILL to the commit's group never reaches."""
    if commit.poll() is None:
        os.killpg(commit.pid, signal.SIGKILL)
        commit.wait()
    for unit in units:
        if pid_running(unit.pid) and unsafe_pgid_reason(unit.pgid) is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(unit.pgid, signal.SIGKILL)


def _hook_pid(git_pid: int) -> int:
    children = Path(f'/proc/{git_pid}/task/{git_pid}/children').read_text(encoding='utf-8').split()
    assert len(children) == 1, children
    return int(children[0])


def _wchan(pid: int) -> str:
    """The kernel function *pid* sleeps in, or '' once it has exited."""
    try:
        return Path(f'/proc/{pid}/wchan').read_text(encoding='utf-8')
    except OSError:
        return ''


def _path_stub(bin_dir: Path, name: str, body: str) -> Path:
    """*bin_dir*, holding an executable *name* that shadows the real one once *bin_dir* leads PATH."""
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / name
    stub.write_text(body, encoding='utf-8')
    stub.chmod(0o755)
    return bin_dir


def _ctrl_c(git_pid: int) -> None:
    os.killpg(git_pid, signal.SIGINT)


def _terminate_the_hook(git_pid: int) -> None:
    os.kill(_hook_pid(git_pid), signal.SIGTERM)


def _hang_up(git_pid: int) -> None:
    os.killpg(git_pid, signal.SIGHUP)


HELD_MKTEMP = """#!/bin/sh
trap '' INT
made=$({real} "$@") || exit
echo "$made"
grep -q '^  running ' {out} || exit 0
exec >&-
: > {held}
until [ -e {release} ]; do sleep 0.05; done
exit 0
"""


@pytest.mark.parametrize(
    ('unit_prelude', 'unit_tail', 'interrupt'),
    [
        ("trap '' INT; ", '; true', _ctrl_c),
        ('', '; true', _terminate_the_hook),
        ("trap '' INT; ", ' & exec sleep 300', _ctrl_c),
        ('', '; true', _hang_up),
    ],
    ids=[
        'ctrl-c-with-a-sigint-immune-unit',
        'sigterm-to-the-hook-alone',
        'ctrl-c-that-kills-the-unit-shell',
        'sighup-to-the-commit',
    ],
)
def test_an_interrupted_commit_stops_its_units_and_removes_their_logs(
    tmp_path: Path, unit_prelude: str, unit_tail: str, interrupt: Callable[[int], None],
) -> None:
    pid_file = tmp_path / 'unit.pid'
    sleeper = f'sh -c {shlex.quote(f"{unit_prelude}echo $$ > {pid_file}; exec sleep 300")}{unit_tail}'
    repo = make_hook_repo(tmp_path, _type_only_modules(lambda prefix: sleeper if prefix == 'shared' else 'true'))
    before = repo.head()
    repo.stage('shared/src/x.py', 'x = 1\n')
    hook_tmp = tmp_path / 'hook_tmp'
    hook_tmp.mkdir()
    out_path = tmp_path / 'commit.out'

    commit = _start_commit(repo, hook_tmp, out_path)
    units: list[_Unit] = []
    try:
        units.append(_started_unit(pid_file, out_path))
        pid = units[-1].pid
        interrupt(commit.pid)
        commit.wait(timeout=60)
        stopped = _eventually(lambda: not pid_running(pid), 30)
        logs_removed = _eventually(lambda: not any(hook_tmp.iterdir()), 30)
    finally:
        _kill_leftovers(commit, units)

    assert repo.head() == before
    assert stopped, f'unit pid {pid} outlived the interrupted commit'
    assert logs_removed, sorted(path.name for path in hook_tmp.iterdir())


def test_an_interrupted_commit_stops_what_an_uncollected_unit_left_when_its_shell_exited(tmp_path: Path) -> None:
    blocker_pid_file = tmp_path / 'blocker.pid'
    orphan_pid_file = tmp_path / 'orphan.pid'
    last_unit_file = tmp_path / 'last_unit.pid'
    blocker = f'sh -c {shlex.quote(f"echo $$ > {blocker_pid_file}; exec sleep 300")}'
    orphan = shlex.quote(
        f"trap '' INT; while kill -0 \"$1\" 2>/dev/null; do sleep 0.1; done; echo $$ > {orphan_pid_file}; exec sleep 300"
    )
    commands = {
        'escalation': blocker,
        'm1': f'sh -c {orphan} unit-shell $$ &',
        'shared': f'echo $$ > {shlex.quote(str(last_unit_file))}',
    }
    repo = make_hook_repo(tmp_path, _type_only_modules(commands.__getitem__, 'm1'))
    before = repo.head()
    repo.stage('shared/src/x.py', 'x = 1\n')
    hook_tmp = tmp_path / 'hook_tmp'
    hook_tmp.mkdir()
    out_path = tmp_path / 'commit.out'

    commit = _start_commit(repo, hook_tmp, out_path)
    units: list[_Unit] = []
    try:
        units.append(_started_unit(blocker_pid_file, out_path))
        units.append(_started_unit(orphan_pid_file, out_path))
        pid = units[-1].pid
        _await_written(last_unit_file, out_path)
        _ctrl_c(commit.pid)
        commit.wait(timeout=60)
        stopped = _eventually(lambda: not pid_running(pid), 30)
    finally:
        _kill_leftovers(commit, units)

    output = out_path.read_text(encoding='utf-8')
    assert repo.head() == before
    assert re.findall(r'^  running (\S+):', output, re.MULTILINE) == [
        'module:escalation:type', 'module:m1:type', 'module:shared:type',
    ], 'the orphan must come from a unit launched after the blocker and before the last unit'
    assert stopped, f'pid {pid}, left running by a unit whose shell had exited, outlived the interrupted commit'


def test_a_ctrl_c_that_lands_while_the_hook_reaps_a_helper_mid_launch_stops_the_launched_unit(tmp_path: Path) -> None:
    """Guards the hook's rule against a command substitution once its traps are armed, for mktemp only.

    The stub mktemp holds only when the hook calls it after launching a unit, which is where the
    per-unit ``$(mktemp)`` used to run. A command substitution of any other helper, or of a builtin,
    in the launch loop or in gate_collect_next is not caught here.
    """
    blocker_pid_file = tmp_path / 'blocker.pid'
    last_unit_file = tmp_path / 'last_unit.pid'
    commands = {
        'escalation': f'sh -c {shlex.quote(f"echo $$ > {blocker_pid_file}; exec sleep 300")}',
        'shared': f'echo $$ > {shlex.quote(str(last_unit_file))}',
    }
    repo = make_hook_repo(tmp_path, _type_only_modules(commands.__getitem__))
    before = repo.head()
    repo.stage('shared/src/x.py', 'x = 1\n')
    hook_tmp = tmp_path / 'hook_tmp'
    hook_tmp.mkdir()
    out_path = tmp_path / 'commit.out'
    held = tmp_path / 'mktemp.held'
    release = tmp_path / 'mktemp.release'
    real_mktemp = shutil.which('mktemp')
    assert real_mktemp is not None
    held_bin = _path_stub(tmp_path / 'held_bin', 'mktemp', HELD_MKTEMP.format(
        real=shlex.quote(real_mktemp), out=shlex.quote(str(out_path)),
        held=shlex.quote(str(held)), release=shlex.quote(str(release)),
    ))

    commit = _start_commit(repo, hook_tmp, out_path, bin_dir=held_bin)
    units: list[_Unit] = []
    try:
        units.append(_started_unit(blocker_pid_file, out_path))
        hook = _hook_pid(commit.pid)
        assert _eventually(
            lambda: (held.exists() or _written(last_unit_file)) and _wchan(hook) == 'do_wait',
            120,
        ), out_path.read_text(encoding='utf-8')
        _ctrl_c(commit.pid)
        release.touch()
        commit.wait(timeout=60)
        stopped = _eventually(lambda: not pid_running(units[0].pid), 30)
        logs_removed = _eventually(lambda: not any(hook_tmp.iterdir()), 30)
    finally:
        release.touch()
        _kill_leftovers(commit, units)

    assert repo.head() == before
    assert stopped, (
        f'blocker pid {units[0].pid} outlived a Ctrl-C that landed while the hook was reaping a helper mid-launch'
    )
    assert logs_removed, sorted(path.name for path in hook_tmp.iterdir())
    assert not held.exists(), 'the hook ran mktemp after launching a unit'


def test_a_unit_log_directory_that_cannot_be_made_refuses_the_commit_before_any_unit_runs(tmp_path: Path) -> None:
    repo = make_hook_repo(tmp_path, _type_only_modules(lambda prefix: 'true'))
    before = repo.head()
    repo.stage('shared/src/x.py', 'x = 1\n')
    hook_tmp = tmp_path / 'hook_tmp'
    hook_tmp.mkdir()
    out_path = tmp_path / 'commit.out'
    failing_bin = _path_stub(tmp_path / 'failing_bin', 'mktemp', "#!/bin/sh\necho 'mktemp stub: refused' >&2; exit 1\n")

    commit = _start_commit(repo, hook_tmp, out_path, bin_dir=failing_bin)
    try:
        rc = commit.wait(timeout=60)
    finally:
        _kill_leftovers(commit, [])

    output = out_path.read_text(encoding='utf-8')
    assert rc != 0, output
    assert repo.head() == before
    assert re.findall(r'^  running ', output, re.MULTILINE) == [], output
    assert 'unit log directory' in output, output
