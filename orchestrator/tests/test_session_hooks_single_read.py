"""Pins that one hook event does one /proc ancestor walk and one record.json read, and acts on that single read, including the unreadable-snapshot fault lane (task 4662)."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _process_tree_fakes import claude_process_tree, counting_walks
from _session_hook_fixtures import (
    assert_hook_env_isolated,
    isolate_hook_env,
    launching_record,
    owner_ppid_env,
    write_bound_parent,
)

from orchestrator import session_hooks as sh
from orchestrator import session_registry as sr
from orchestrator.session_ownership_probes import EventProbes, ProcessTree


@pytest.fixture(autouse=True)
def _isolate_hook_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolate_hook_env(monkeypatch, tmp_path)


def test_hook_env_is_isolated_for_this_module(tmp_path: Path) -> None:
    assert_hook_env_isolated(tmp_path)


# ---------------------------------------------------------------------------
# ONE EVENT, ONE /proc WALK (task 4662)
#
# The fixture below is the exact shape where all three walks fire today: a
# still-LAUNCHING, still-unbound spawn record whose OWNER_PPID verdict is
# True -- the owner's own pre-SessionStart event, task 4193 L2 ruling item
# 4-iii. _env_slug_ownership takes its verdict, _withhold_from_launching
# takes it again, and _bind_claude_session_id stamps claude_owner_pid from a
# third walk. Each assertion pairs the count with an OUTCOME check, so the
# pin cannot be satisfied by simply skipping a probe.
# ---------------------------------------------------------------------------


def _owner_shape(
    slug: str, tmp_path: Path, *, pid: int
) -> tuple[dict[str, str], ProcessTree, list[int]]:
    """LAUNCHING+unbound record whose OWNER_PPID verdict resolves to True."""
    launching_record(slug, tmp_path, pid=pid)
    tree, walks = counting_walks(claude_process_tree(4193501, 4193500))
    return owner_ppid_env(slug, 4193500), tree, walks


def test_notification_walks_proc_exactly_once(tmp_path: Path) -> None:
    slug = 'session-cockpit-4662001'
    env, tree, walks = _owner_shape(slug, tmp_path, pid=4662001)

    sh.run_notification(
        {
            'session_id': 'uuid-owner',
            'cwd': '/home/leo/src/dark-factory',
            'message': 'may I proceed?',
        },
        env,
        root=tmp_path,
        process_tree=tree,
    )

    assert len(walks) == 1
    # OUTCOME PARITY (mirrors test_owner_stop_is_not_withheld_...).
    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.AWAITING_INPUT
    assert record.question is not None and record.question.text == 'may I proceed?'
    assert record.claude_session_id == 'uuid-owner'


def test_stop_walks_proc_exactly_once(tmp_path: Path) -> None:
    slug = 'session-cockpit-4662002'
    env, tree, walks = _owner_shape(slug, tmp_path, pid=4662002)

    sh.run_stop(
        {'session_id': 'uuid-owner', 'cwd': '/home/leo/src/dark-factory'},
        env,
        root=tmp_path,
        process_tree=tree,
    )

    assert len(walks) == 1
    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.IDLE
    assert record.claude_session_id == 'uuid-owner'


def test_session_start_walks_proc_exactly_once(tmp_path: Path) -> None:
    slug = 'session-cockpit-4662003'
    env, tree, walks = _owner_shape(slug, tmp_path, pid=4662003)

    sh.run_session_start(
        {
            'session_id': 'uuid-owner',
            'cwd': '/home/leo/src/dark-factory',
            'source': 'startup',
        },
        env,
        root=tmp_path,
        process_tree=tree,
    )

    assert len(walks) == 1
    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.RUNNING
    assert record.claude_session_id == 'uuid-owner'
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_nested_session_start_walks_proc_at_most_once(tmp_path: Path) -> None:
    """The fork mirror: memoizing must not weaken the nested-inheritor guard.

    Mirrors
    orchestrator/tests/test_session_hooks_launch_window.py::test_nested_session_start_cannot_capture_an_unbound_launching_record
    -- the spawn record must be left unbound and LAUNCHING -- while pinning
    that the fork path (ownership verdict, the nested-claude liveness pid,
    and the bind-time stamp) also settles on one walk.
    """
    slug = 'session-cockpit-4662004'
    launching_record(slug, tmp_path, pid=4662004)
    env = owner_ppid_env(slug, 4193500)
    # A nested claude's parent is its agent's Bash-tool shell, not the
    # payload bash spawn-claude.sh exported.
    tree, walks = counting_walks(claude_process_tree(4193777, 4193776))

    nested = {
        'session_id': 'uuid-nested',
        'cwd': '/home/leo/src/dark-factory',
        'source': 'startup',
    }
    sh.run_session_start(nested, env, root=tmp_path, process_tree=tree)

    assert len(walks) <= 1
    spawn_record = sr.read_record(slug, root=tmp_path)
    assert spawn_record.claude_session_id is None
    assert spawn_record.status is sr.Status.LAUNCHING
    forked = sr.read_record(
        sh.hook_session_slug(nested, env, root=tmp_path, process_tree=tree), root=tmp_path
    )
    assert forked.claude_session_id == 'uuid-nested'
    assert forked.parent_session_id == slug


# ---------------------------------------------------------------------------
# _RecordSnapshot / _read_record_snapshot -- the TRI-STATE read (task 4662)
#
# One read must serve two consumers that want OPPOSITE things from a failed
# one. The ownership probe must fail soft to "adopt" (_env_slug_ownership:
# "every failure mode resolves to adopt (True), never to fork"), so it treats
# corrupt and absent alike. The refresh must NOT: session_registry's
# refresh_record guarantees "a *corrupt* existing body is NOT treated as
# absent". A two-state record|None would let a hook synthesize a fresh record
# over a corrupt body -- silent loss of the record holding role/prompt/
# result_file. Hence ABSENT and UNREADABLE are distinct states.
# ---------------------------------------------------------------------------


def test_record_snapshot_reads_an_existing_record(tmp_path: Path) -> None:
    slug = 'session-cockpit-4662010'
    launching_record(slug, tmp_path, pid=4662010)

    snapshot = sh._read_record_snapshot(slug, tmp_path)

    assert snapshot.slug == slug
    assert snapshot.unreadable is False
    assert snapshot.record is not None
    assert snapshot.record.to_dict() == sr.read_record(slug, root=tmp_path).to_dict()


def test_record_snapshot_absent_is_not_unreadable_and_is_silent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """ABSENT is the state that MAY be synthesized over -- and it is routine.

    An ordinary fresh spawn must log nothing, matching _env_slug_ownership's
    existing dedicated FileNotFoundError arm ("so an ordinary fresh spawn
    logs nothing").
    """
    with caplog.at_level(logging.WARNING):
        snapshot = sh._read_record_snapshot('session-cockpit-4662011', tmp_path)

    assert snapshot.record is None
    assert snapshot.unreadable is False
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_record_snapshot_corrupt_body_is_unreadable_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    slug = 'session-cockpit-4662012'
    record_path = sr.record_path_for_slug(slug, root=tmp_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text('not-json{{{')
    with pytest.raises(sr.CorruptSessionRecord):
        sr.read_record(slug, root=tmp_path)

    with caplog.at_level(logging.WARNING):
        snapshot = sh._read_record_snapshot(slug, tmp_path)

    assert snapshot.record is None
    assert snapshot.unreadable is True
    # Degradation is never silent (repo's no-silent-fail-soft norm).
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_record_snapshot_arbitrary_oserror_is_unreadable_and_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError('disk on fire')

    monkeypatch.setattr(sr, 'read_record', _boom)

    with caplog.at_level(logging.WARNING):
        snapshot = sh._read_record_snapshot('session-cockpit-4662013', tmp_path)

    assert snapshot.record is None
    assert snapshot.unreadable is True
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_record_snapshot_is_frozen(tmp_path: Path) -> None:
    """Immutable: a snapshot is one observation, not a mutable scratchpad."""
    slug = 'session-cockpit-4662014'
    launching_record(slug, tmp_path, pid=4662014)
    snapshot = sh._read_record_snapshot(slug, tmp_path)

    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.record = None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _HookSlugResolution -- carry the probe's read forward (task 4662)
#
# The ownership probe already reads the candidate slug's record. On the ADOPT
# branch that candidate IS the slug the handler goes on to write, so its
# snapshot is exactly the record about to be refreshed -- handing it forward
# is what collapses the event onto one read. On the FORK branch the resolver
# returns a DIFFERENT slug, so offering the rejected spawner's snapshot would
# apply a nested claude's decision to the spawning session's record: the
# ownership inversion tasks 4193 and 2511 both exist to prevent. Hence the
# invariant asserted on EVERY shape below.
# ---------------------------------------------------------------------------


def test_resolution_adopts_and_carries_the_probes_snapshot(tmp_path: Path) -> None:
    slug = 'session-cockpit-4662020'
    write_bound_parent(slug, tmp_path, 4662020)
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    probes = EventProbes(env, claude_process_tree(None))
    resolution = sh._resolve_hook_slug(hook_input, env, tmp_path, probes=probes)

    assert resolution.slug == slug
    assert resolution.rejected_env_slug is None
    assert resolution.snapshot is not None
    assert resolution.snapshot.slug == resolution.slug
    assert resolution.snapshot.record is not None
    assert (
        resolution.snapshot.record.to_dict()
        == sr.read_record(slug, root=tmp_path).to_dict()
    )
    assert sh.hook_session_slug(hook_input, env, root=tmp_path) == resolution.slug


def test_resolution_forks_without_offering_the_rejected_snapshot(
    tmp_path: Path,
) -> None:
    """The snapshot belongs to the REJECTED slug, so it must not travel."""
    slug = 'session-cockpit-4662021'
    write_bound_parent(slug, tmp_path, 4662021)
    # Proven mismatch: the record is bound to 'uuid-parent', this is not it.
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    probes = EventProbes(env, claude_process_tree(None))
    resolution = sh._resolve_hook_slug(hook_input, env, tmp_path, probes=probes)

    assert resolution.slug != slug
    assert resolution.rejected_env_slug == slug
    assert resolution.snapshot is None
    assert sh.hook_session_slug(hook_input, env, root=tmp_path) == resolution.slug


def test_resolution_reads_nothing_for_a_hand_launched_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No env slug: nothing to probe, so nothing is read (must-not-be-called)."""

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError('nothing to probe for a hand-launched session')

    monkeypatch.setattr(sr, 'read_record', _boom)
    hook_input = {'session_id': 'sess-hand', 'cwd': '/home/leo/src/dark-factory'}

    probes = EventProbes({}, claude_process_tree(None))
    resolution = sh._resolve_hook_slug(hook_input, {}, tmp_path, probes=probes)

    assert resolution.snapshot is None
    assert resolution.rejected_env_slug is None
    assert resolution.may_bind is True
    assert sh.hook_session_slug(hook_input, {}, root=tmp_path) == resolution.slug


def test_resolution_blank_stdin_session_id_reads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-discriminator early return reads nothing, so carries nothing."""

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError('the blank-session_id early return reads no record')

    monkeypatch.setattr(sr, 'read_record', _boom)
    slug = 'session-cockpit-4662022'
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    hook_input = {'session_id': '  ', 'cwd': '/home/leo/src/dark-factory'}

    probes = EventProbes(env, claude_process_tree(None))
    resolution = sh._resolve_hook_slug(hook_input, env, tmp_path, probes=probes)

    assert resolution.slug == slug
    assert resolution.may_bind is True
    assert resolution.snapshot is None
    assert sh.hook_session_slug(hook_input, env, root=tmp_path) == resolution.slug


def test_resolution_snapshot_slug_invariant_holds_on_every_shape(
    tmp_path: Path,
) -> None:
    """INVARIANT: a snapshot is attached only when it IS the returned slug.

    Asserted across all four resolver shapes rather than left to the reader
    of the branch, because a mis-attached snapshot would fail silently and
    only under a nested-claude race.
    """
    adopted = 'session-cockpit-4662023'
    forked = 'session-cockpit-4662024'
    write_bound_parent(adopted, tmp_path, 4662023)
    write_bound_parent(forked, tmp_path, 4662024)
    cwd = '/home/leo/src/dark-factory'
    shapes = [
        ({'session_id': 'uuid-parent', 'cwd': cwd}, {'CLAUDE_SPAWN_SESSION_ID': adopted}),
        ({'session_id': 'uuid-nested', 'cwd': cwd}, {'CLAUDE_SPAWN_SESSION_ID': forked}),
        ({'session_id': 'sess-hand', 'cwd': cwd}, {}),
        ({'session_id': '  ', 'cwd': cwd}, {'CLAUDE_SPAWN_SESSION_ID': adopted}),
    ]
    for hook_input, env in shapes:
        tree = claude_process_tree(None)
        probes = EventProbes(env, tree)
        resolution = sh._resolve_hook_slug(hook_input, env, tmp_path, probes=probes)
        assert resolution.snapshot is None or resolution.snapshot.slug == resolution.slug
        # hook_session_slug's public signature and str return are unchanged.
        public = sh.hook_session_slug(hook_input, env, root=tmp_path, process_tree=tree)
        assert isinstance(public, str)
        assert public == resolution.slug


# ---------------------------------------------------------------------------
# ONE READ, ONE DECISION, ONE WRITE for Notification/Stop (task 4662)
#
# Today one adopted-spawn event reads record.json THREE times (the ownership
# probe, the pre-refresh snapshot, and refresh_record's own internal read)
# and writes up to TWICE. The counts are the efficiency half; the
# same-snapshot assertion below is the correctness half -- today the withhold
# decision is computed from read #2 while the body written derives from read
# #3, so two racing hook events decide against stale state and write against
# fresh state.
# ---------------------------------------------------------------------------


class _RegistryIO:
    """Count reads per slug and record every body handed to write_record.

    Each read stamps a per-call marker into the returned body's ``cwd`` so a
    written record can be traced back to the read it derives from -- the
    technique that turns "how many reads" into "which read did the write
    come from", which is the TOCTOU property, not merely an I/O count.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.reads: list[str] = []
        self.writes: list[sr.SessionRecord] = []
        real_read, real_write = sr.read_record, sr.write_record

        def _read(slug: str, *args: object, **kwargs: object) -> sr.SessionRecord:
            self.reads.append(slug)
            record = real_read(slug, *args, **kwargs)  # type: ignore[arg-type]
            record.cwd = f'read-{len(self.reads)}'
            return record

        def _write(record: sr.SessionRecord, *args: object, **kwargs: object) -> None:
            self.writes.append(record)
            real_write(record, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(sr, 'read_record', _read)
        monkeypatch.setattr(sr, 'write_record', _write)

    def reads_of(self, slug: str) -> int:
        return self.reads.count(slug)


def _adopted_bound_record(slug: str, tmp_path: Path) -> dict[str, object]:
    """A record already bound to this event's stdin session_id (adopt path)."""
    write_bound_parent(slug, tmp_path, 4662030)
    return {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}


@pytest.mark.parametrize('handler', ['notification', 'stop'])
def test_refresh_path_reads_the_written_record_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, handler: str
) -> None:
    slug = f'session-cockpit-466203{0 if handler == "notification" else 1}'
    hook_input = _adopted_bound_record(slug, tmp_path)
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    io = _RegistryIO(monkeypatch)

    if handler == 'notification':
        sh.run_notification({**hook_input, 'message': 'may I proceed?'}, env, root=tmp_path)
        expected = sr.Status.AWAITING_INPUT
    else:
        sh.run_stop(hook_input, env, root=tmp_path)
        expected = sr.Status.IDLE

    assert io.reads_of(slug) == 1
    # OUTCOME PARITY.
    record = sr.read_record(slug, root=tmp_path)
    assert record.status is expected
    if handler == 'notification':
        assert record.question is not None and record.question.text == 'may I proceed?'


def test_hand_launched_refresh_writes_the_body_its_decision_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook_input = {'session_id': 'sess-hand', 'cwd': '/home/leo/src/dark-factory'}
    sh.run_session_start(hook_input, {}, root=tmp_path)
    slug = sh.hook_session_slug(hook_input, {}, root=tmp_path)
    io = _RegistryIO(monkeypatch)

    sh.run_stop(hook_input, {}, root=tmp_path)

    # Read 1 is the session-id pointer's vouch; read 2 is the one the decision and the write share.
    assert io.reads_of(slug) == 2
    assert [written.cwd for written in io.writes] == ['read-2']
    assert sr.read_record(slug, root=tmp_path).status is sr.Status.IDLE


def test_refresh_writes_the_body_it_decided_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE TOCTOU INVARIANT, not merely a count.

    The withhold/bind decision and the body written must come from the SAME
    read. Today the decision is made on read #2 while refresh_record writes
    a body it re-read as #3, so a concurrent writer's change can land in
    between and be decided against stale state.
    """
    slug = 'session-cockpit-4662032'
    hook_input = _adopted_bound_record(slug, tmp_path)
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    io = _RegistryIO(monkeypatch)

    sh.run_stop(hook_input, env, root=tmp_path)

    assert len(io.writes) == 1
    assert io.writes[0].cwd == 'read-1'


def test_first_bind_event_writes_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two-write case: an adoptable record that this event first BINDS.

    Status, question and binding must land in ONE body, which is what the
    handler's docstring already claimed but the two-write shape did not
    deliver.
    """
    slug = 'session-cockpit-4662033'
    # RUNNING + unbound: adoptable (not in the launch window), so this event
    # both refreshes the status and stamps the first binding.
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug, status=sr.Status.RUNNING, launcher_pid=4662033
        ),
        root=tmp_path,
    )
    env = owner_ppid_env(slug, 4193500)
    io = _RegistryIO(monkeypatch)

    sh.run_notification(
        {
            'session_id': 'uuid-owner',
            'cwd': '/home/leo/src/dark-factory',
            'message': 'may I proceed?',
        },
        env,
        root=tmp_path,
        process_tree=claude_process_tree(4193501, 4193500),
    )

    assert io.reads_of(slug) == 1
    assert len(io.writes) == 1
    written = io.writes[0]
    assert written.status is sr.Status.AWAITING_INPUT
    assert written.question is not None and written.question.text == 'may I proceed?'
    assert written.claude_session_id == 'uuid-owner'


def test_fork_path_reads_each_of_its_two_slugs_at_most_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ACHIEVABILITY GUARD: a fork legitimately touches TWO records.

    So the bound is per-slug, never one read in total -- asserting the
    latter would be unsatisfiable without breaking the ownership split.
    """
    spawner = 'session-cockpit-4662034'
    write_bound_parent(spawner, tmp_path, 4662034)
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': spawner}
    forked = sh.hook_session_slug(hook_input, env, root=tmp_path)
    io = _RegistryIO(monkeypatch)

    sh.run_stop(hook_input, env, root=tmp_path)

    assert io.reads_of(spawner) <= 1
    assert io.reads_of(forked) <= 1
    # The spawner's record is untouched; the fork carries the IDLE.
    assert sr.read_record(spawner, root=tmp_path).status is sr.Status.RUNNING
    assert sr.read_record(forked, root=tmp_path).status is sr.Status.IDLE


def test_refresh_never_overwrites_a_corrupt_body(tmp_path: Path) -> None:
    """CHARACTERIZATION (green before and after) -- pinned because step 12
    puts it at risk. refresh_record's "a *corrupt* existing body is NOT
    treated as absent" contract means run_stop must leave the bytes alone.
    """
    slug = 'session-cockpit-4662035'
    record_path = sr.record_path_for_slug(slug, root=tmp_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text('not-json{{{')
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}

    with contextlib.suppress(sr.CorruptSessionRecord):
        sh.run_stop(hook_input, env, root=tmp_path)

    assert record_path.read_text() == 'not-json{{{'


# ---------------------------------------------------------------------------
# The same one-read pin for SessionStart, which shares the resolution path
# (task 4662). Today the adopted path reads twice: _env_slug_ownership's
# probe read, then run_session_start's own read_record.
# ---------------------------------------------------------------------------


def test_session_start_reads_the_written_record_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slug = 'session-cockpit-4662040'
    hook_input = _adopted_bound_record(slug, tmp_path)
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    io = _RegistryIO(monkeypatch)

    sh.run_session_start({**hook_input, 'source': 'startup'}, env, root=tmp_path)

    assert io.reads_of(slug) == 1
    assert sr.read_record(slug, root=tmp_path).status is sr.Status.RUNNING


def test_session_start_writes_the_body_it_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SAME-SNAPSHOT DERIVATION: the record the ownership/remint decision was
    made on is the record that gets written."""
    slug = 'session-cockpit-4662041'
    hook_input = _adopted_bound_record(slug, tmp_path)
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    io = _RegistryIO(monkeypatch)

    sh.run_session_start({**hook_input, 'source': 'startup'}, env, root=tmp_path)

    assert len(io.writes) == 1
    assert io.writes[0].cwd == 'read-1'


def test_session_start_fork_path_reads_each_slug_at_most_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawner = 'session-cockpit-4662042'
    write_bound_parent(spawner, tmp_path, 4662042)
    nested = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': spawner}
    forked = sh.hook_session_slug(nested, env, root=tmp_path)
    io = _RegistryIO(monkeypatch)

    sh.run_session_start({**nested, 'source': 'startup'}, env, root=tmp_path)

    assert io.reads_of(spawner) <= 1
    assert io.reads_of(forked) <= 1
    assert sr.read_record(spawner, root=tmp_path).status is sr.Status.RUNNING
    assert sr.read_record(forked, root=tmp_path).parent_session_id == spawner


def test_session_start_hand_launched_reads_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook_input = {'session_id': 'sess-hand', 'cwd': '/home/leo/src/dark-factory'}
    slug = sh.hook_session_slug(hook_input, {}, root=tmp_path)
    io = _RegistryIO(monkeypatch)

    sh.run_session_start(hook_input, {}, root=tmp_path)

    assert io.reads_of(slug) == 1
    assert sr.read_record(slug, root=tmp_path).status is sr.Status.RUNNING


def test_session_start_fault_lane_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CHARACTERIZATION of test_main_session_start_fail_soft_when_probe_read_
    raises' scenario, asserted here so the single-snapshot rewrite cannot
    quietly narrow it: a probe read that raises still completes the write,
    without CLAIMING the record (adopting is fail-soft, binding is permanent).
    """
    slug = 'session-cockpit-4662043'
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    real_read = sr.read_record
    calls: list[int] = []

    def _boom_once(*args: object, **kwargs: object) -> sr.SessionRecord:
        calls.append(1)
        if len(calls) == 1:
            raise OSError('disk on fire')
        return real_read(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sr, 'read_record', _boom_once)

    sh.run_session_start(
        {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'},
        env,
        root=tmp_path,
    )

    assert len(calls) >= 1
    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.RUNNING
    assert record.claude_session_id is None
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_session_start_first_sight_still_captures_richly(tmp_path: Path) -> None:
    """OUTCOME PARITY for the rich-capture path (no prior record)."""
    hook_input = {
        'session_id': 'sess-rich',
        'cwd': '/home/leo/src/dark-factory',
        'transcript_path': '/tmp/ignored.jsonl',
    }
    env = {
        'CLAUDE_SPAWN_ROLE': 'unblock',
        'CLAUDE_SPAWN_PROJECT': 'df',
        'CLAUDE_SPAWN_TASK_ID': '4662',
        'CLAUDE_SPAWN_ESCALATION_ID': 'esc-4662-1',
        'CLAUDE_SPAWN_LAUNCHER_PID': '4662044',
    }

    record = sh.run_session_start(hook_input, env, root=tmp_path)

    assert record.role == 'unblock'
    assert record.project == 'df'
    assert record.task_id == '4662'
    assert record.escalation_id == 'esc-4662-1'
    assert record.cwd == '/home/leo/src/dark-factory'
    assert record.launcher_pid == 4662044
    assert record.start_ts
    assert record.transcript_path == sr.transcript_path_for_cwd(
        '/home/leo/src/dark-factory'
    )


# ---------------------------------------------------------------------------
# The UNREADABLE-snapshot fault lane must still honour the task-4193
# launch-window withhold guard (task 4662, reviewer finding).
#
# MEASURED on task/4662 @ 38265c5462, same scenario both sides (LAUNCHING +
# unbound spawn record, nested `claude` Notification, read_record raising on
# call #1 then succeeding):
#   main        -> status=launching,      question=None            (withheld)
#   this branch -> status=awaiting-input, question='nested question'
# i.e. the nested session's status and prompt land on the SPAWNING session's
# record -- the ownership inversion tasks 4193/2511 exist to prevent.
#
# Before task 4662 the handler made TWO independent reads (the ownership
# probe's AND _prior_record_or_none's), so a TRANSIENT fault on the first
# left the second free to succeed and the guard still ran. One snapshot now
# serves both, and one fault disarms it -- the fault lane never carried a
# withhold evaluation of its own. The existing fault-lane tests
# (orchestrator/tests/test_session_hooks_ownership.py::test_run_stop_fail_soft_when_probe_read_raises,
# test_session_start_fault_lane_parity) both use an ABSENT record, so neither
# exercises the interaction; that gap is why this shipped.
# ---------------------------------------------------------------------------


def _boom_once_read(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Fault the FIRST ``read_record`` (the ownership probe's), then behave.

    The suite's established ``_boom_once`` idiom, hoisted so the fault-lane
    tests below share one definition of "transient fault on the probe read".
    """
    real_read = sr.read_record
    calls: list[int] = []

    def _boom_once(*args: object, **kwargs: object) -> sr.SessionRecord:
        calls.append(1)
        if len(calls) == 1:
            raise OSError('disk on fire')
        return real_read(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sr, 'read_record', _boom_once)
    return calls


def _spawn_launching_record(slug: str, root: Path) -> None:
    """The body that makes the stake real: role/project/prompt/result_file.

    This is the record spawn-claude.sh's ``finish()`` writes ``exited`` to,
    so a nested inheritor rewriting it is not a cosmetic status lie.
    """
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.LAUNCHING,
            role='session',
            project='cockpit',
            prompt='/spawn cockpit',
            result_file='/tmp/spawn-4662.json',
            launcher_pid=4662050,
            start_ts=datetime.now(UTC).isoformat(),
        ),
        root=root,
    )


@pytest.mark.parametrize('handler', ['notification', 'stop'])
def test_fault_lane_still_withholds_from_an_unbound_launching_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, handler: str
) -> None:
    """The direct mirror of test_refresh_path_does_not_bind_a_still_launching_
    record / ..._withholds_a_question_during_the_unbound_launch_window, with a
    transient fault injected into the ownership probe's read.

    A fault on ONE read must not cost this event its withhold DECISION. The
    record's provenance is still unknowable, so nothing this nested event
    carries -- not the status, not the question -- may land on it.
    """
    slug = f'session-cockpit-466205{0 if handler == "notification" else 1}'
    _spawn_launching_record(slug, tmp_path)
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    calls = _boom_once_read(monkeypatch)

    if handler == 'notification':
        sh.run_notification({**hook_input, 'message': 'nested question'}, env, root=tmp_path)
    else:
        sh.run_stop(hook_input, env, root=tmp_path)

    # The probe was consulted (and degraded) rather than skipped.
    assert len(calls) >= 1
    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.LAUNCHING
    assert record.question is None
    assert record.claude_session_id is None
    # Withholding is not forking: the event lands on this slug and is then
    # dropped -- it must not also mint a second, nested-owned row.
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_fault_lane_does_not_withhold_from_a_positively_identified_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PARITY IN THE OTHER DIRECTION, so the fix cannot be "always withhold
    when the snapshot faulted".

    The owner's own pre-SessionStart Notification (task 4193 L2 item 4-iii):
    CLAUDE_SPAWN_OWNER_PPID positively identifies this event, so its status
    and question belong on the record whether or not the read faulted.
    """
    slug = 'session-cockpit-4662052'
    _spawn_launching_record(slug, tmp_path)
    env = owner_ppid_env(slug, 4193500)
    calls = _boom_once_read(monkeypatch)

    sh.run_notification(
        {
            'session_id': 'uuid-owner',
            'cwd': '/home/leo/src/dark-factory',
            'message': 'may I proceed?',
        },
        env,
        root=tmp_path,
        process_tree=claude_process_tree(4193501, 4193500),
    )

    assert len(calls) >= 1
    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.AWAITING_INPUT
    assert record.question is not None and record.question.text == 'may I proceed?'
    # The spawn record's own payload survives the refresh untouched.
    assert record.prompt == '/spawn cockpit'
    assert record.result_file == '/tmp/spawn-4662.json'


def test_fault_lane_reads_the_written_slug_at_most_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE STRUCTURAL BOUND: the faulted probe read plus ONE retry.

    Restoring the withhold decision must not degrade into an unbounded or
    third re-read of the record. The outcome is asserted alongside the count
    so the bound cannot be met by skipping the retry.
    """
    slug = 'session-cockpit-4662053'
    _spawn_launching_record(slug, tmp_path)
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    reads: list[str] = []
    real_read = sr.read_record

    def _counting_boom_once(slug_arg: str, *args: object, **kwargs: object) -> sr.SessionRecord:
        reads.append(slug_arg)
        if len(reads) == 1:
            raise OSError('disk on fire')
        return real_read(slug_arg, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sr, 'read_record', _counting_boom_once)

    sh.run_stop(hook_input, env, root=tmp_path)

    assert reads.count(slug) <= 2
    # OUTCOME, so the bound cannot be satisfied by skipping the retry: the
    # retry is what supplies the observation the withhold decision needs.
    assert sr.read_record(slug, root=tmp_path).status is sr.Status.LAUNCHING


def test_fault_lane_propagates_a_persistently_corrupt_body(tmp_path: Path) -> None:
    """STRENGTHENS test_refresh_never_overwrites_a_corrupt_body.

    That test suppresses CorruptSessionRecord, so it would also pass if the
    handler silently no-op'd. A body that is corrupt on EVERY read must still
    PROPAGATE to main()'s blanket except -- refresh_record's "a *corrupt*
    existing body is NOT treated as absent" contract -- and leave the bytes
    exactly as it found them.
    """
    slug = 'session-cockpit-4662054'
    record_path = sr.record_path_for_slug(slug, root=tmp_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text('not-json{{{')
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}

    with pytest.raises(sr.CorruptSessionRecord):
        sh.run_stop(hook_input, env, root=tmp_path)

    assert record_path.read_text() == 'not-json{{{'
