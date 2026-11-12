"""relocated_scopes against a fabricated /proc tree: which scopes a spawned session moved into."""

from __future__ import annotations

import os
from pathlib import Path

from orchestrator.relocated_scopes import relocated_scopes

_APP = '/user.slice/user-1000.slice/user@1000.service/app.slice'
OWN = f'{_APP}/orchestrator-x.service'
LAUNCH_UNIT = 'df-verify-t-abc.scope'
LAUNCH = f'{_APP}/{LAUNCH_UNIT}'
MOVED = '/user.slice/user-1000.slice/user@1000.service/govslice.slice/run-u42.scope'
LEADER = 500


def _proc(
    root: Path, pid: int, *, ppid: int, pgrp: int, session: int, cgroup: str | None
) -> None:
    entry = root / str(pid)
    entry.mkdir(parents=True)
    (entry / 'stat').write_text(
        f'{pid} (comm x) S {ppid} {pgrp} {session} 0 -1 4194304 ' + '0 ' * 20
    )
    if cgroup is not None:
        (entry / 'cgroup').write_text(f'0::{cgroup}\n')


def _self(root: Path, cgroup: str) -> None:
    entry = root / 'self'
    entry.mkdir(parents=True, exist_ok=True)
    (entry / 'cgroup').write_text(f'0::{cgroup}\n')


def _leader(root: Path, cgroup: str, *, ppid: int | None = None, session: int = LEADER) -> None:
    _proc(
        root,
        LEADER,
        ppid=os.getpid() if ppid is None else ppid,
        pgrp=LEADER,
        session=session,
        cgroup=cgroup,
    )


def _member(root: Path, pid: int, cgroup: str | None, *, session: int = LEADER) -> None:
    _proc(root, pid, ppid=LEADER, pgrp=pid, session=session, cgroup=cgroup)


def test_a_setpgid_member_moved_with_its_leader_names_the_new_scope(tmp_path):
    _self(tmp_path, OWN)
    _leader(tmp_path, MOVED)
    _member(tmp_path, 501, MOVED)

    assert relocated_scopes(LEADER, launched_in=LAUNCH_UNIT, proc_root=tmp_path) == frozenset(
        {'run-u42.scope'}
    )


def test_members_still_in_the_launched_scope_name_nothing(tmp_path):
    _self(tmp_path, OWN)
    _leader(tmp_path, LAUNCH)
    _member(tmp_path, 501, LAUNCH)

    assert relocated_scopes(LEADER, launched_in=LAUNCH_UNIT, proc_root=tmp_path) == frozenset()


def test_unscoped_members_in_the_callers_own_cgroup_name_nothing(tmp_path):
    _self(tmp_path, OWN)
    _leader(tmp_path, OWN)
    _member(tmp_path, 501, OWN)

    assert relocated_scopes(LEADER, launched_in=None, proc_root=tmp_path) == frozenset()


def test_the_callers_own_scope_is_never_named(tmp_path):
    runner_scope = f'{_APP}/df-verify-runner.scope'
    _self(tmp_path, runner_scope)
    _leader(tmp_path, runner_scope)
    _member(tmp_path, 501, runner_scope)

    assert relocated_scopes(LEADER, launched_in=None, proc_root=tmp_path) == frozenset()


def test_a_process_of_another_session_is_ignored(tmp_path):
    _self(tmp_path, OWN)
    _leader(tmp_path, LAUNCH)
    _member(tmp_path, 901, f'{_APP}/run-u900.scope', session=900)

    assert relocated_scopes(LEADER, launched_in=LAUNCH_UNIT, proc_root=tmp_path) == frozenset()


def test_members_in_two_scopes_name_both(tmp_path):
    other = '/user.slice/user-1000.slice/user@1000.service/govslice.slice/run-u43.scope'
    _self(tmp_path, OWN)
    _leader(tmp_path, LAUNCH)
    _member(tmp_path, 501, MOVED)
    _member(tmp_path, 502, other)

    assert relocated_scopes(LEADER, launched_in=LAUNCH_UNIT, proc_root=tmp_path) == frozenset(
        {'run-u42.scope', 'run-u43.scope'}
    )


def test_several_members_in_one_scope_name_it_once(tmp_path):
    _self(tmp_path, OWN)
    _leader(tmp_path, MOVED)
    for pid in (501, 502, 503):
        _member(tmp_path, pid, MOVED)

    assert relocated_scopes(LEADER, launched_in=LAUNCH_UNIT, proc_root=tmp_path) == frozenset(
        {'run-u42.scope'}
    )


def test_a_member_in_a_sub_cgroup_names_its_deepest_scope(tmp_path):
    _self(tmp_path, OWN)
    _leader(tmp_path, LAUNCH)
    _member(tmp_path, 501, f'{_APP}/outer.scope/govslice.slice/run-u7.scope/payload')

    assert relocated_scopes(LEADER, launched_in=LAUNCH_UNIT, proc_root=tmp_path) == frozenset(
        {'run-u7.scope'}
    )


def test_a_member_outside_any_scope_names_nothing(tmp_path):
    _self(tmp_path, OWN)
    _leader(tmp_path, LAUNCH)
    _member(tmp_path, 501, f'{_APP}/other.service')
    _member(tmp_path, 502, '/user.slice/user-1000.slice/user@1000.service/govslice.slice')

    assert relocated_scopes(LEADER, launched_in=LAUNCH_UNIT, proc_root=tmp_path) == frozenset()


def test_a_leader_that_is_not_our_child_names_nothing(tmp_path):
    _self(tmp_path, OWN)
    _leader(tmp_path, MOVED, ppid=os.getpid() + 1)
    _member(tmp_path, 501, MOVED)

    assert relocated_scopes(LEADER, launched_in=LAUNCH_UNIT, proc_root=tmp_path) == frozenset()


def test_an_absent_leader_names_nothing(tmp_path):
    _self(tmp_path, OWN)
    _member(tmp_path, 501, MOVED)

    assert relocated_scopes(LEADER, launched_in=LAUNCH_UNIT, proc_root=tmp_path) == frozenset()


def test_a_leader_not_leading_its_own_session_names_nothing(tmp_path):
    _self(tmp_path, OWN)
    _leader(tmp_path, MOVED, session=LEADER + 100)
    _member(tmp_path, 501, MOVED, session=LEADER + 100)

    assert relocated_scopes(LEADER, launched_in=LAUNCH_UNIT, proc_root=tmp_path) == frozenset()


def test_unreadable_and_malformed_entries_are_skipped(tmp_path):
    _self(tmp_path, OWN)
    _leader(tmp_path, LAUNCH)
    _member(tmp_path, 501, MOVED)
    _member(tmp_path, 502, None)
    _member(tmp_path, 503, None)
    (tmp_path / '503' / 'cgroup').write_text('1:name=systemd:/x\n')
    garbage = tmp_path / '504'
    garbage.mkdir()
    (garbage / 'stat').write_text('garbage')
    (tmp_path / 'sys').mkdir()

    assert relocated_scopes(LEADER, launched_in=LAUNCH_UNIT, proc_root=tmp_path) == frozenset(
        {'run-u42.scope'}
    )


def test_a_missing_proc_root_names_nothing(tmp_path):
    assert (
        relocated_scopes(LEADER, launched_in=LAUNCH_UNIT, proc_root=tmp_path / 'absent')
        == frozenset()
    )
