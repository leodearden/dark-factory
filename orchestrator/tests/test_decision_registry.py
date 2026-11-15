"""Tests for orchestrator.decision_registry, the fleet-global DecisionRecord registry.

Covers the record schema and its round-trips, decision paths and
project-qualified ids, the store and its fail-soft setters, the per-id lock,
the close/reopen/audit transitions, queue and project-token normalisation,
the escalation-status read, reap and migrate. The refile merges are tested in
test_decision_merges.py, and the decision verbs that session_registry.py
serves are tested apart from this library.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import json
import logging
import os
import threading
import time
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _decision_fixtures import (
    EVIDENCE,
    claims_zero_matches,
    make_decision,
    names_the_destination_token,
    seed_mixed_spellings,
)
from shared.project_id import PROJECT_ID_ALIASES, fold_project_token

from orchestrator import decision_registry as dr


def _identity_of(record: dr.DecisionRecord) -> dict[str, str]:
    """close_decision_with_evidence's compare-and-swap expectations for *record*, as it was filed."""
    return {'expected_project': record.project, 'expected_escalations_dir': record.escalations_dir}


def _assert_utc_instant_within(stamp: str, before: datetime, after: datetime) -> None:
    instant = datetime.fromisoformat(stamp)
    assert instant.utcoffset() == timedelta(0)
    assert before <= instant <= after


def test_make_decision_defaults_to_the_unset_queue_sentinel() -> None:
    """The fixture must not hand escalations_dir a non-'' default.

    Guards the one exception documented on make_decision: escalations_dir is
    a control-flow guard input to ``_run_reap_decisions._status``, and a
    non-empty shared default silently neuters every reap-decisions test that
    does not override it (task 3528 -- it cost three live regression tests at
    once, each still passing with its premise inverted). Omitting the key is
    the structural fix; this is the one-line tripwire against re-adding it.
    """
    assert make_decision().escalations_dir == ''


def test_decision_state_enum_values() -> None:
    assert issubclass(dr.DecisionState, str)
    assert {m.value for m in dr.DecisionState} == {'open', 'answered', 'dropped'}
    assert dr.DecisionState.OPEN.value == 'open'
    assert dr.DecisionState.ANSWERED.value == 'answered'
    assert dr.DecisionState.DROPPED.value == 'dropped'


def test_decision_record_dict_round_trip_is_lossless() -> None:
    # escalations_dir is stated explicitly (the fixture deliberately omits it
    # and lets the '' dataclass default stand): a dropped key would round-trip
    # indistinguishably from '', so only a non-empty value keeps this test's
    # field-drop catch power.
    d = make_decision(escalations_dir='/p/data/reconciliation/escalations')
    assert dr.DecisionRecord.from_dict(d.to_dict()) == d


def test_decision_record_json_round_trip_is_lossless() -> None:
    d = make_decision(escalations_dir='/p/data/reconciliation/escalations')
    assert dr.DecisionRecord.from_json(d.to_json()) == d


def test_decision_record_round_trip_with_null_fields() -> None:
    """session_id/task_id/escalation_id/options may legitimately be None (a
    project-level decision with no session/task/escalation context yet);
    the round trip must preserve that, not coerce it or drop the key.
    """
    d = make_decision(
        session_id=None,
        task_id=None,
        escalation_id=None,
        options=None,
        escalations_dir='/p/data/reconciliation/escalations',
    )
    round_tripped = dr.DecisionRecord.from_dict(d.to_dict())
    assert round_tripped == d
    assert round_tripped.session_id is None
    assert round_tripped.task_id is None
    assert round_tripped.escalation_id is None
    assert round_tripped.options is None


def test_decision_record_defaults() -> None:
    d = dr.DecisionRecord(
        id='dec-1', project='df', text='approve?', filed_at='2026-07-07T00:00:00+00:00'
    )
    assert d.manual_boost == 0
    assert d.state == dr.DecisionState.OPEN
    assert d.severity == ''
    assert d.escalations_dir == ''


def test_decision_record_to_dict_includes_severity() -> None:
    d = make_decision(severity='blocking')
    assert d.to_dict()['severity'] == 'blocking'


def test_decision_record_parses_pre_severity_dict_additive() -> None:
    """A decision dict written before the severity field existed -- containing
    none of the severity key -- must still parse via from_dict without
    raising, defaulting severity to '' (migration-free additive contract,
    mirrors test_session_record_parses_rail_vintage_dict_migration_free).
    """
    pre_severity = {
        'id': 'dec-1',
        'project': 'df',
        'text': 'approve?',
        'filed_at': '2026-07-07T00:00:00+00:00',
        'session_id': 'unblock-df-2085-4242',
        'task_id': '2085',
        'escalation_id': 'esc-1',
        'options': ['yes', 'no'],
        'manual_boost': 2,
        'state': 'answered',
    }
    record = dr.DecisionRecord.from_dict(pre_severity)
    assert record.severity == ''


def test_decision_record_parses_pre_escalations_dir_dict_additive() -> None:
    """Every one of the ~300 decision records already on disk predates the
    escalations_dir field (task 3528). Such a dict must still parse, yielding
    the '' unset/legacy sentinel that makes the reaper fall back to today's
    project-only scoping -- the backward-compatibility half of the fix.
    """
    pre_queue = {
        'id': 'dec-1',
        'project': 'df',
        'text': 'approve?',
        'filed_at': '2026-07-07T00:00:00+00:00',
        'escalation_id': 'esc-1',
        'state': 'open',
        'severity': 'blocking',
    }
    record = dr.DecisionRecord.from_dict(pre_queue)
    assert record.escalations_dir == ''


def test_decision_record_parses_null_escalations_dir_as_empty() -> None:
    """An explicit JSON null (a hand-edited or third-party-written record)
    must also collapse to '', not None -- the annotation is `str`, and the
    reaper's queue guard is a single `if decision_dir and ...` test with no
    None-vs-''-vs-missing branching. Fail-soft, not a raise.
    """
    record = dr.DecisionRecord.from_dict(
        {
            'id': 'dec-1',
            'project': 'df',
            'text': 'approve?',
            'filed_at': '2026-07-07T00:00:00+00:00',
            'escalations_dir': None,
        }
    )
    assert record.escalations_dir == ''


def test_decision_record_to_dict_always_includes_escalations_dir() -> None:
    """The key is emitted unconditionally -- including for a queue-less
    record -- so a round-tripped record never loses its queue stamp and the
    cockpit's read-modify-write helpers carry it through untouched.
    """
    stamped = make_decision(escalations_dir='/p/data/reconciliation/escalations')
    assert stamped.to_dict()['escalations_dir'] == '/p/data/reconciliation/escalations'
    assert 'escalations_dir' in make_decision().to_dict()


def test_decision_path_for_id_under_decisions_dir(tmp_path: Path) -> None:
    assert dr.decision_path_for_id('dec-1', root=tmp_path) == tmp_path / 'decisions' / 'dec-1.json'


def test_decision_path_for_id_sanitizes_unsafe_id(tmp_path: Path) -> None:
    # An id containing '/' must not escape decisions_dir: it still resolves
    # to a single file directly inside decisions_dir (no nested directory,
    # no traversal) -- mirrors lease_path_for_name's path-escape guard.
    path = dr.decision_path_for_id('../../etc/passwd', root=tmp_path)
    assert path.parent == dr.decisions_dir(root=tmp_path)
    # Lock the actual sanitized stem, not just "has the right parent": every
    # '/' maps to '-' via _DECISION_ID_SANITIZE_RE while '.' is preserved.
    assert path.name == '..-..-etc-passwd.json'
    assert '/' not in path.name


def test_qualify_decision_id_prefixes_the_canonical_project() -> None:
    assert dr.qualify_decision_id('dark_factory', 'esc-42-1') == 'dark_factory-esc-42-1'


@pytest.mark.parametrize('project', ['df', 'Dark-Factory', '  DARK_FACTORY '])
def test_qualify_decision_id_folds_the_project(project: str) -> None:
    assert dr.qualify_decision_id(project, 'esc-42-1') == 'dark_factory-esc-42-1'


def test_qualify_decision_id_is_injective_across_projects() -> None:
    assert dr.qualify_decision_id('dark_factory', 'esc-42-1') != dr.qualify_decision_id(
        'reify', 'esc-42-1'
    )
    # Canonical tokens never contain '-', so the separator cannot be smuggled
    # in from the project side of the join.
    assert dr.qualify_decision_id('a', 'b-c') != dr.qualify_decision_id('a_b', 'c')


@pytest.mark.parametrize(
    ('project', 'local_id'),
    [('dark_factory', 'esc-42-1'), ('reify', 'recon-esc-7459-1'), ('df', 'watcher-lease-orphan-df')],
)
def test_qualify_decision_id_survives_the_path_sanitizer(
    tmp_path: Path, project: str, local_id: str
) -> None:
    qualified = dr.qualify_decision_id(project, local_id)
    assert dr.decision_path_for_id(qualified, root=tmp_path).stem == qualified


@pytest.mark.parametrize('project', [None, '', '  '])
def test_qualify_decision_id_leaves_an_unset_project_unqualified(project: object) -> None:
    assert dr.qualify_decision_id(project, 'esc-42-1') == 'esc-42-1'


def test_write_then_list_decision(tmp_path: Path) -> None:
    rec = make_decision()

    assert dr.write_decision(rec, root=tmp_path) is True

    listed = dr.list_decisions(root=tmp_path)
    assert len(listed) == 1
    assert listed[0] == rec


def test_update_decision_state_persists(tmp_path: Path) -> None:
    rec = make_decision(state=dr.DecisionState.OPEN)
    dr.write_decision(rec, root=tmp_path)

    updated = dr.update_decision_state(rec.id, dr.DecisionState.ANSWERED, root=tmp_path)

    assert updated is not None
    assert updated.state == dr.DecisionState.ANSWERED
    [reread] = [d for d in dr.list_decisions(root=tmp_path) if d.id == rec.id]
    assert reread.state == dr.DecisionState.ANSWERED


def test_set_manual_boost_persists(tmp_path: Path) -> None:
    rec = make_decision(manual_boost=0)
    dr.write_decision(rec, root=tmp_path)

    updated = dr.set_manual_boost(rec.id, 3, root=tmp_path)

    assert updated is not None
    assert updated.manual_boost == 3
    [reread] = [d for d in dr.list_decisions(root=tmp_path) if d.id == rec.id]
    assert reread.manual_boost == 3


def test_set_decision_escalations_dir_persists_normalized(tmp_path: Path) -> None:
    """The back-fill's writer (task 3640) normalizes ON THE WAY IN.

    A non-canonical spelling passed by a caller must not be stored raw: the
    reaper's axis-2 guard compares stored-value against reaper-value, and a
    dotted/trailing-slash spelling stored verbatim would compare unequal to
    the very queue it names -- the fail-open direction, so the record would
    silently never close again. Normalizing here means every writer of this
    field (write-decision, the back-fill) stores ONE spelling.
    """
    rec = make_decision(id='dec-setescdir')
    dr.write_decision(rec, root=tmp_path)
    queue = tmp_path / 'escalations'
    queue.mkdir()
    (tmp_path / 'sub').mkdir()
    dotted = str(tmp_path / 'sub' / '..' / 'escalations') + '/'
    assert dotted != str(queue)

    updated = dr.set_decision_escalations_dir(rec.id, dotted, root=tmp_path)

    assert updated is not None
    assert updated.escalations_dir == dr.normalize_escalations_dir(queue)
    [reread] = [d for d in dr.list_decisions(root=tmp_path) if d.id == rec.id]
    assert reread.escalations_dir == dr.normalize_escalations_dir(queue)


def test_set_decision_escalations_dir_stores_the_unknown_sentinel_verbatim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The composition that matters: sentinel in, sentinel on disk.

    This is the one path the back-fill uses for every record whose owning
    queue it could not determine. If the writer's normalize step turned the
    sentinel into a cwd-relative path (the bug fixed in step-2), the live
    fleet would end up with hundreds of records stamped
    '/home/leo/src/dark-factory/.worktrees/3640/<unknown>' -- a value that
    both lies about the record and varies with whoever ran the migration.
    Running under chdir pins that the stored value is cwd-INdependent.
    """
    monkeypatch.chdir(tmp_path)
    rec = make_decision(id='dec-setescdir-unknown')
    dr.write_decision(rec, root=tmp_path)

    updated = dr.set_decision_escalations_dir(rec.id, dr.UNKNOWN_QUEUE, root=tmp_path)

    assert updated is not None
    assert updated.escalations_dir == dr.UNKNOWN_QUEUE
    [reread] = [d for d in dr.list_decisions(root=tmp_path) if d.id == rec.id]
    assert reread.escalations_dir == dr.UNKNOWN_QUEUE


def test_set_decision_escalations_dir_preserves_every_other_field(tmp_path: Path) -> None:
    """A read-modify-write must modify exactly ONE field.

    write_decision rewrites the whole file, so a dropped/defaulted field in
    the round-trip is silent data loss -- and the back-fill runs this over the
    entire open cockpit population at once, where losing e.g. `options` or
    `manual_boost` would quietly degrade real human gates. make_decision
    gives every field a distinguishable value precisely so this can catch it.
    """
    rec = make_decision(id='dec-setescdir-preserve', state=dr.DecisionState.OPEN)
    dr.write_decision(rec, root=tmp_path)

    updated = dr.set_decision_escalations_dir(rec.id, dr.UNKNOWN_QUEUE, root=tmp_path)

    assert updated is not None
    [reread] = [d for d in dr.list_decisions(root=tmp_path) if d.id == rec.id]
    assert reread == dataclasses.replace(rec, escalations_dir=dr.UNKNOWN_QUEUE)
    # Spelled out as well as compared wholesale, so a failure names the field.
    assert reread.state == dr.DecisionState.OPEN
    assert reread.manual_boost == rec.manual_boost
    assert reread.severity == rec.severity
    assert reread.escalation_id == rec.escalation_id
    assert reread.session_id == rec.session_id
    assert reread.task_id == rec.task_id
    assert reread.options == rec.options
    assert reread.text == rec.text
    assert reread.filed_at == rec.filed_at
    assert reread.project == rec.project


def test_set_decision_escalations_dir_fail_soft_on_unknown_id(tmp_path: Path) -> None:
    """Same fail-soft contract as its two siblings: None, never a raise.

    The back-fill reads the whole decision list and then writes each id back;
    a record closed or removed by a live watcher in between is expected, not
    exceptional, and must not abort a migration mid-population.
    """
    assert dr.set_decision_escalations_dir('no-such-id', '/tmp/q', root=tmp_path) is None


def test_set_decision_escalations_dir_fail_soft_on_corrupt_body(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A corrupt on-disk body returns None (logged at ERROR), not a traceback."""
    corrupt_path = dr.decision_path_for_id('dec-setescdir-corrupt', root=tmp_path)
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text('{not valid json')

    with caplog.at_level(logging.ERROR):
        result = dr.set_decision_escalations_dir('dec-setescdir-corrupt', '/tmp/q', root=tmp_path)

    assert result is None
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_set_decision_escalations_dir_is_repeatable(tmp_path: Path) -> None:
    """Called twice on the same id it simply succeeds twice (last write wins).

    The back-fill is re-runnable by design, and the lock is a stable sidecar
    the first call creates -- so a second call must not trip over its own
    lock file. The locking contract itself is covered by TestDecisionIdLock;
    this only pins that repeated calls compose.
    """
    rec = make_decision(id='dec-setescdir-twice')
    dr.write_decision(rec, root=tmp_path)

    first = dr.set_decision_escalations_dir(rec.id, dr.UNKNOWN_QUEUE, root=tmp_path)
    second = dr.set_decision_escalations_dir(rec.id, tmp_path / 'q', root=tmp_path)

    assert first is not None
    assert second is not None
    [reread] = [d for d in dr.list_decisions(root=tmp_path) if d.id == rec.id]
    assert reread.escalations_dir == dr.normalize_escalations_dir(tmp_path / 'q')


def test_decisions_are_per_file_isolated(tmp_path: Path) -> None:
    """Distinct <id>.json paths are the whole isolation guarantee: writing or
    updating one decision must never touch another's file (no global index,
    no shared body -- mirrors each session record's own directory).
    """
    rec_a = make_decision(id='dec-a')
    rec_b = make_decision(id='dec-b')
    dr.write_decision(rec_a, root=tmp_path)
    dr.write_decision(rec_b, root=tmp_path)

    path_a = dr.decision_path_for_id('dec-a', root=tmp_path)
    path_b = dr.decision_path_for_id('dec-b', root=tmp_path)
    assert path_a.is_file()
    assert path_b.is_file()
    assert path_a != path_b
    assert {d.id for d in dr.list_decisions(root=tmp_path)} == {'dec-a', 'dec-b'}

    bytes_b_before = path_b.read_bytes()
    dr.set_manual_boost('dec-a', 9, root=tmp_path)
    assert path_b.read_bytes() == bytes_b_before


def test_list_decisions_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert dr.list_decisions(root=tmp_path) == []


def test_list_decisions_skips_corrupt_file(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rec = make_decision()
    dr.write_decision(rec, root=tmp_path)
    corrupt_path = dr.decisions_dir(root=tmp_path) / 'bogus.json'
    corrupt_path.write_text('{not valid json')

    with caplog.at_level(logging.ERROR):
        listed = dr.list_decisions(root=tmp_path)

    assert [d.id for d in listed] == [rec.id]
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_write_decision_fail_soft_on_unwritable_root(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Mirrors test_main_launching_fail_soft_when_fleet_root_under_a_file: a
    # root rooted under a pre-created regular file makes the write's
    # mkdir(parents=True) raise NotADirectoryError deterministically.
    blocker = tmp_path / 'blocker'
    blocker.write_text('not a directory')
    bad_root = blocker / 'fleet'

    with caplog.at_level(logging.ERROR):
        result = dr.write_decision(make_decision(), root=bad_root)

    assert result is False
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert not (blocker / 'fleet').exists()


def test_update_and_set_boost_fail_soft_when_absent(tmp_path: Path) -> None:
    assert dr.update_decision_state('no-such-id', dr.DecisionState.ANSWERED, root=tmp_path) is None
    assert dr.set_manual_boost('no-such-id', 5, root=tmp_path) is None


def test_all_decision_setters_fail_soft_when_lock_acquisition_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A lock-acquisition fault (mkdir/os.open/flock raising OSError inside
    decision_id_lock) must be absorbed by the same fail-soft try/except as a
    read/write fault, not propagate into C8/cockpit callers.

    Regression guard for the `with decision_id_lock(...)` placement: if it
    were ever moved outside (or above) the helpers' existing try/except, this
    would start raising instead of returning None.

    Covers ALL THREE writers, including task 3640's set_decision_escalations_dir
    -- the back-fill iterates the whole live decision population, so a single
    unabsorbed lock fault there would abort a migration mid-way and leave it
    half-stamped.
    """
    rec = make_decision(id='dec-lockfault', state=dr.DecisionState.OPEN, manual_boost=0)
    dr.write_decision(rec, root=tmp_path)

    @contextlib.contextmanager
    def raising_lock(decision_id: str, root: Path | str | None = None):
        raise OSError('simulated lock-acquisition fault')
        yield  # pragma: no cover -- unreachable, satisfies generator-function shape

    monkeypatch.setattr(dr, 'decision_id_lock', raising_lock)

    with caplog.at_level(logging.ERROR):
        state_result = dr.update_decision_state('dec-lockfault', dr.DecisionState.ANSWERED, root=tmp_path)
        boost_result = dr.set_manual_boost('dec-lockfault', 5, root=tmp_path)
        stamp_result = dr.set_decision_escalations_dir('dec-lockfault', '/tmp/q', root=tmp_path)

    assert state_result is None
    assert boost_result is None
    assert stamp_result is None
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


class TestDecisionIdLock:
    """Unit tests for the dr.decision_id_lock(decision_id, root=...) context manager.

    Mirrors TestEscalationIdLock (escalation/tests/test_queue.py:2553) -- same
    stable-sidecar-flock contract (task 1609), retargeted to decisions_dir /
    decision_path_for_id and the ``<id>.json.lock`` sidecar path.
    """

    def test_entering_context_creates_sidecar_file(self, tmp_path: Path) -> None:
        """(b) Entering the context creates the sidecar .lock file at the expected path."""
        lock_path = dr.decisions_dir(root=tmp_path) / 'dec-1.json.lock'

        assert not lock_path.exists(), 'Sidecar must not exist before context entry'
        with dr.decision_id_lock('dec-1', root=tmp_path):
            assert lock_path.exists(), 'Sidecar must exist while context is held'
        # File persists after release -- stable inode, never renamed/replaced.
        assert lock_path.exists(), 'Sidecar must persist after context exit (stable inode)'

    def test_held_lock_blocks_second_acquire_nonblocking(self, tmp_path: Path) -> None:
        """(c) While the context is held, a non-blocking flock on the same sidecar raises BlockingIOError."""
        lock_path = dr.decisions_dir(root=tmp_path) / 'dec-1.json.lock'

        with dr.decision_id_lock('dec-1', root=tmp_path):
            fd2 = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd2)

        # After context exit the lock is released -- non-blocking acquire succeeds.
        fd3 = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd3, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd3, fcntl.LOCK_UN)
        finally:
            os.close(fd3)

    def test_lock_sidecar_invisible_to_list_decisions(self, tmp_path: Path) -> None:
        """(d) The .lock sidecar never becomes a phantom decision via list_decisions' `*.json` glob."""
        rec = make_decision(id='dec-1')
        dr.write_decision(rec, root=tmp_path)

        with dr.decision_id_lock('dec-1', root=tmp_path):
            listed = dr.list_decisions(root=tmp_path)

        assert len(listed) == 1
        assert listed[0] == rec


class TestDecisionHelpersAdoptLock:
    """Spy tests (mirror TestSubmitResolveAdoptLock, test_queue.py:2913):
    update_decision_state, set_manual_boost, set_decision_escalations_dir and
    the write-decision verb must EACH acquire decision_id_lock for the
    correct decision id.

    One case per public caller, deliberately, even though the three setters
    now share _mutate_decision: the lock is a per-caller CONTRACT, and a
    setter that grows its own body (or a wrapper that mutates before
    delegating) would slip past a single shared-implementation test. The
    write-decision verb is precisely such a caller -- it does NOT go through
    _mutate_decision at all (it is an upsert, not a strict RMW), so nothing
    but this spy would notice its lock disappearing. TestDecisionIdLock
    covers the lock primitive itself; these cover its ADOPTION.
    """

    def test_update_decision_state_acquires_lock_for_decision_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rec = make_decision(id='dec-spy-1', state=dr.DecisionState.OPEN)
        dr.write_decision(rec, root=tmp_path)

        real_lock = dr.decision_id_lock
        acquired: list[str] = []

        @contextlib.contextmanager
        def recording_lock(decision_id: str, root: Path | str | None = None):
            acquired.append(decision_id)
            with real_lock(decision_id, root=root):
                yield

        monkeypatch.setattr(dr, 'decision_id_lock', recording_lock)

        dr.update_decision_state('dec-spy-1', dr.DecisionState.ANSWERED, root=tmp_path)

        assert 'dec-spy-1' in acquired, f'Expected lock acquisition for dec-spy-1; got {acquired}'

    def test_set_manual_boost_acquires_lock_for_decision_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rec = make_decision(id='dec-spy-2', manual_boost=0)
        dr.write_decision(rec, root=tmp_path)

        real_lock = dr.decision_id_lock
        acquired: list[str] = []

        @contextlib.contextmanager
        def recording_lock(decision_id: str, root: Path | str | None = None):
            acquired.append(decision_id)
            with real_lock(decision_id, root=root):
                yield

        monkeypatch.setattr(dr, 'decision_id_lock', recording_lock)

        dr.set_manual_boost('dec-spy-2', 5, root=tmp_path)

        assert 'dec-spy-2' in acquired, f'Expected lock acquisition for dec-spy-2; got {acquired}'

    def test_set_decision_escalations_dir_acquires_lock_for_decision_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The queue-stamp setter is the THIRD writer on a decision id.

        Its docstring makes the lock the central claim -- the task-3640
        back-fill runs against live records while the C8 watchers and the
        cockpit are up, so an unserialized read-modify-write here would drop
        a concurrent state transition. Without this spy, deleting the
        `with decision_id_lock(...)` span would leave every other test in the
        suite green.
        """
        rec = make_decision(id='dec-spy-3', escalations_dir='')
        dr.write_decision(rec, root=tmp_path)

        real_lock = dr.decision_id_lock
        acquired: list[str] = []

        @contextlib.contextmanager
        def recording_lock(decision_id: str, root: Path | str | None = None):
            acquired.append(decision_id)
            with real_lock(decision_id, root=root):
                yield

        monkeypatch.setattr(dr, 'decision_id_lock', recording_lock)

        dr.set_decision_escalations_dir('dec-spy-3', '/tmp/some-queue', root=tmp_path)

        assert 'dec-spy-3' in acquired, f'Expected lock acquisition for dec-spy-3; got {acquired}'

    def test_close_decision_with_evidence_acquires_lock_for_decision_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The FIFTH writer (task 5376): the sitting preparer's apply step, via close-decision."""
        rec = make_decision(id='dec-spy-6', state=dr.DecisionState.OPEN)
        dr.write_decision(rec, root=tmp_path)

        real_lock = dr.decision_id_lock
        acquired: list[str] = []

        @contextlib.contextmanager
        def recording_lock(decision_id: str, root: Path | str | None = None):
            acquired.append(decision_id)
            with real_lock(decision_id, root=root):
                yield

        monkeypatch.setattr(dr, 'decision_id_lock', recording_lock)

        dr.close_decision_with_evidence(
            'dec-spy-6', dr.DecisionState.ANSWERED, 'gate evidence', root=tmp_path, **_identity_of(rec)
        )

        assert 'dec-spy-6' in acquired, f'Expected lock acquisition for dec-spy-6; got {acquired}'


@pytest.mark.timeout(30)
def test_concurrent_state_and_boost_updates_do_not_lose_a_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """END-TO-END lost-update test (deterministic, in-process, two threads).

    Seeds a decision at state=OPEN/manual_boost=0, then races
    update_decision_state (-> ANSWERED) against set_manual_boost (-> 7) on
    the SAME id in two threads. A fixed delay is injected into the
    module-level write_decision seam -- AFTER each helper's read, BEFORE
    its write -- so both threads deterministically read the pre-mutation
    snapshot before either writes.

    Without decision_id_lock serializing the read+mutate+write span, the
    later os.replace() clobbers the earlier call's field mutation (each
    thread's local record only carries ONE of the two field changes, since
    neither thread observed the other's write) -- so the conjunction below
    fails no matter which thread's write lands last. With the lock, the two
    RMW spans serialize: whichever thread runs second re-reads the other's
    already-persisted change, so both fields survive.
    """
    rec = make_decision(id='dec-race', state=dr.DecisionState.OPEN, manual_boost=0)
    dr.write_decision(rec, root=tmp_path)

    real_write_decision = dr.write_decision

    def delayed_write_decision(record: dr.DecisionRecord, root: Path | str | None = None) -> bool:
        time.sleep(0.3)
        return real_write_decision(record, root=root)

    monkeypatch.setattr(dr, 'write_decision', delayed_write_decision)

    t1 = threading.Thread(
        target=dr.update_decision_state,
        args=('dec-race', dr.DecisionState.ANSWERED, tmp_path),
    )
    t2 = threading.Thread(target=dr.set_manual_boost, args=('dec-race', 7, tmp_path))

    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert not t1.is_alive(), 'update_decision_state thread did not finish in time'
    assert not t2.is_alive(), 'set_manual_boost thread did not finish in time'

    [reread] = [d for d in dr.list_decisions(root=tmp_path) if d.id == 'dec-race']
    assert reread.state == dr.DecisionState.ANSWERED
    assert reread.manual_boost == 7


def test_normalize_escalations_dir_makes_a_relative_path_absolute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A queue passed relative to the watcher's cwd must be stored/compared as
    an absolute path, so a reaper invoked from a different cwd still matches.
    """
    (tmp_path / 'esc').mkdir()
    monkeypatch.chdir(tmp_path)

    assert dr.normalize_escalations_dir('esc') == str((tmp_path / 'esc').resolve())


def test_normalize_escalations_dir_collapses_equivalent_spellings(tmp_path: Path) -> None:
    """The whole point of the helper: two spellings of the SAME queue dir must
    produce ONE identical string, so the reaper's queue guard compares equal
    for a decision stamped via a dotted/trailing-slash spelling.
    """
    esc = tmp_path / 'esc'
    esc.mkdir()
    (tmp_path / 'sub').mkdir()
    dotted = str(tmp_path / 'sub' / '..' / 'esc') + '/'

    assert dr.normalize_escalations_dir(dotted) == dr.normalize_escalations_dir(esc)


def test_normalize_escalations_dir_empty_and_blank_are_the_unset_sentinel() -> None:
    """'' means "unset/legacy" and is never a path; a whitespace-only value
    from a sloppy shell interpolation must collapse to the same sentinel
    rather than normalizing to the cwd.
    """
    assert dr.normalize_escalations_dir('') == ''
    assert dr.normalize_escalations_dir('   ') == ''
    assert dr.normalize_escalations_dir('\t\n') == ''


def test_normalize_escalations_dir_expands_user(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Watchers write queue paths as ~/... in SKILL.md snippets; those must
    expand, or a stamped record would never match a reaper's absolute path.
    """
    monkeypatch.setenv('HOME', str(tmp_path))

    assert dr.normalize_escalations_dir('~/data/escalations') == str(
        (tmp_path / 'data' / 'escalations').resolve()
    )


def test_normalize_escalations_dir_nonexistent_dir_still_normalizes(tmp_path: Path) -> None:
    """Path.resolve() is non-strict in Python 3.11+: a well-formed queue path
    that does not exist yet (a project checked out but never escalated, or a
    record migrated between machines) still normalizes rather than faulting.
    """
    missing = tmp_path / 'never' / 'created'

    assert dr.normalize_escalations_dir(missing) == str(missing)


def test_normalize_escalations_dir_fail_soft_on_unresolvable_value() -> None:
    """Fail-soft, matching the module's contract for helpers a C8 watch loop
    calls directly: a value the OS cannot resolve (embedded NUL) degrades to
    the raw string instead of raising into the caller. The raw string simply
    won't match any real queue, which is the fail-OPEN direction.
    """
    assert dr.normalize_escalations_dir('a\x00b') == 'a\x00b'


def test_unknown_queue_sentinel_is_distinguishable_from_the_unset_sentinel() -> None:
    """UNKNOWN_QUEUE is a THIRD queue state (task 3640), not a respelling of ''.

    '' means "nobody told us" (legacy/unset -- the reaper falls back to
    project-only scoping and MAY close the record). UNKNOWN_QUEUE means "we
    investigated and could not determine the owning queue" -- the reaper must
    refuse to close it. Collapsing the two would silently hand every
    back-filled undeterminable record back to the false-closure hazard task
    3528 exists to remove, so the values must never compare equal, and the
    sentinel must be TRUTHY (the reaper's axis-2 guard is gated on
    `if decision_dir and ...`).

    The angle brackets are load-bearing, not decoration: a resolved queue path
    always begins with '/', so a real queue and this sentinel can never
    collide no matter what a project is named.
    """
    assert dr.UNKNOWN_QUEUE
    assert dr.UNKNOWN_QUEUE != ''
    assert not dr.UNKNOWN_QUEUE.startswith('/')


def test_normalize_escalations_dir_preserves_the_unknown_sentinel_verbatim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """THE RED DRIVER (task 3640): the sentinel must round-trip unchanged.

    Without a special case, `'<unknown>'` is a bare word and falls through to
    `Path(raw).expanduser().resolve()`, which resolves it RELATIVE TO THE
    CALLING PROCESS'S CWD -- so the same record normalizes to
    `<cwd-A>/<unknown>` in the back-fill script and `<cwd-B>/<unknown>` in a
    watcher's reaper. A stamp whose value depends on who reads it is not a
    contract at all: the two would never compare equal, and worse, whichever
    accidental absolute path got STORED would be an outright lie about where
    the record's escalation lives.

    Asserting under two different cwds is the point -- a single call would
    pass against a buggy implementation that merely happened to be invoked
    from the right directory.
    """
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()

    monkeypatch.chdir(tmp_path / 'a')
    from_a = dr.normalize_escalations_dir(dr.UNKNOWN_QUEUE)
    monkeypatch.chdir(tmp_path / 'b')
    from_b = dr.normalize_escalations_dir(dr.UNKNOWN_QUEUE)

    assert from_a == dr.UNKNOWN_QUEUE
    assert from_b == dr.UNKNOWN_QUEUE


def test_project_token_alias_declines_registry_shape() -> None:
    """The DECLINE registry is the mirror image of PROJECT_ID_ALIASES, and
    it is what makes task 3813's decision durable rather than prose.

    A decision task's product has to survive the session that made it. Task
    3807 already tried a docstring ("KNOWN RESIDUAL GAP ... owned by its own
    filed decision task") and prose cannot fail a test -- someone could add
    ``solar_challenge -> my_solar_challenge`` tomorrow and nothing would
    object. The DISJOINTNESS assertion below is that objection: promoting a
    declined alias into the live table without first REMOVING the decline
    fails here by name, forcing the next editor to read the recorded
    evidence instead of rediscovering it from scratch.
    """
    declined = dr.PROJECT_TOKEN_ALIASES_DECLINED
    assert isinstance(declined, dict)

    # Exactly one recorded decline: the solar-challenge alias (task 3813).
    assert set(declined) == {'solar_challenge'}
    assert declined['solar_challenge'][0] == 'my_solar_challenge'

    # A decline can never be added without STATING WHY. Existence and
    # non-emptiness of the reason field only -- deliberately not a pin on
    # its wording.
    for alias, entry in declined.items():
        assert len(entry) == 2, alias
        assert isinstance(entry[1], str), alias
        assert entry[1].strip(), alias

    # THE GUARD: the two tables partition the alias space. Adding a declined
    # alias to the live table without removing the decline fails right here.
    assert set(declined) & set(PROJECT_ID_ALIASES) == set()

    # Same folded-to-folded hygiene invariant the live table carries (see
    # test_project_token_aliases_maps_folded_to_folded), so promoting an
    # entry is a one-line move between the two dicts.
    for alias, (canonical, _reason) in declined.items():
        assert fold_project_token(alias) == alias
        assert fold_project_token(canonical) == canonical

    # The decline is IN FORCE in the fold, not merely recorded beside it.
    assert fold_project_token('solar_challenge') != fold_project_token(
        'my_solar_challenge'
    )


def test_declined_project_token_hint_names_both_tokens() -> None:
    """The hint must name BOTH tokens: what the operator typed, and where the
    rows actually are.

    Declining the alias makes the naming mismatch PERMANENT, so the honest
    companion to the decline is that the mismatch announces itself. A message
    naming only one side leaves the operator exactly where the silent
    zero-row no-op did.
    """
    hint = dr.declined_project_token_hint('my_solar_challenge')

    assert hint is not None
    assert isinstance(hint, str)
    # Token PRESENCE, deliberately not sentence wording. The destination
    # token goes through names_the_destination_token because the bare
    # `'solar_challenge' in hint` spelling is VACUOUS -- it is implied by the
    # line above it, so a message that named only the typed token would pass.
    assert 'my_solar_challenge' in hint
    assert names_the_destination_token(hint)

    # The fold runs FIRST, so a spelling variant still hits. This is the case
    # that matters: an operator copying the config-declared value with
    # different case/separators must still be warned.
    assert dr.declined_project_token_hint('My-Solar-Challenge') is not None
    assert dr.declined_project_token_hint('  MY_SOLAR_CHALLENGE  ') is not None


def test_declined_project_token_hint_consequence_is_verb_aware() -> None:
    """One message cannot be true for both callers, so *action* picks one.

    ``reap-decisions`` MATCHES, so its consequence is a zero-row no-op.
    ``write-decision`` CREATES: it matches nothing by definition, and one
    line after the warning it files a row under exactly the token passed --
    so the reap wording would be false the moment it is acted on. Its real
    consequence is also the worse of the two (the row lands where no
    documented reap scopes, and can never auto-close), which the shared
    wording left unstated entirely.

    Asserts on the CLAIM each message makes, not its sentences: the
    zero-match claim must be present on the reap path and absent on the
    write path, and every variant must still name both tokens.
    """
    reap = dr.declined_project_token_hint('my_solar_challenge', action='reap')
    file_ = dr.declined_project_token_hint('my_solar_challenge', action='file')

    assert reap is not None and file_ is not None
    assert reap != file_
    for message in (reap, file_):
        assert 'my_solar_challenge' in message
        assert names_the_destination_token(message)

    assert claims_zero_matches(reap)
    assert not claims_zero_matches(file_)


@pytest.mark.parametrize('action', ['', 'bogus-verb'])
def test_declined_project_token_hint_unknown_action_stays_verb_neutral(action: str) -> None:
    """An omitted or unrecognised *action* is not an error, and is not guessed.

    Fail-soft in the direction that matters for a message a human acts on: a
    future caller that forgets the argument gets the verb-neutral core --
    less specific, but TRUE on any path -- rather than a confident
    description of the wrong verb. Pinning this is what stops the default
    from quietly being set to one of the two real verbs later.
    """
    hint = dr.declined_project_token_hint('my_solar_challenge', action=action)

    assert hint is not None
    assert 'my_solar_challenge' in hint
    assert names_the_destination_token(hint)
    # No verb-specific claim, in either direction.
    assert not claims_zero_matches(hint)
    assert hint != dr.declined_project_token_hint('my_solar_challenge', action='reap')
    assert hint != dr.declined_project_token_hint('my_solar_challenge', action='file')
    # The default really is the neutral variant, not one of the two verbs.
    assert dr.declined_project_token_hint('my_solar_challenge') == hint


@pytest.mark.parametrize(
    'token',
    [
        'dark_factory',
        # Aliases to dark_factory -- a healthy project must never be warned.
        'df',
        # The RECOMMENDED token, in both spellings. A warning here would be
        # pure noise a watcher accrues every Main Loop cycle.
        'solar_challenge',
        'solar-challenge',
        # The collapse-guard sibling: a distinct project root. A false
        # warning here would be actively misleading.
        'solar_challenge_platform',
        'reify',
        '',
        '   ',
    ],
)
def test_declined_project_token_hint_is_none_for_everything_else(token: str) -> None:
    """The hint fires on the declined VALUE, never on the declined KEY.

    The trap being closed is "operator reads their project's config, types
    the declared ``memory.project_id``, matches zero rows". So the warning
    must land on ``my_solar_challenge`` and stay SILENT on
    ``solar_challenge`` -- the token both SKILL.md files tell people to
    pass -- or the signal degrades into noise.
    """
    assert dr.declined_project_token_hint(token) is None


@pytest.mark.parametrize('token', [None, 42, 3.5, Path('/tmp/x'), object()])
def test_declined_project_token_hint_fail_soft_on_non_str(token: object) -> None:
    """Fail-soft, mirroring fold_project_token's contract: a helper a C8
    watch loop calls on its filing path never raises into the caller.
    """
    assert dr.declined_project_token_hint(token) is None


def test_read_escalation_status_reads_queue_root_file(tmp_path: Path) -> None:
    """A still-pending escalation lives directly under the queue root."""
    escalations_dir = tmp_path / 'escalations'
    escalations_dir.mkdir()
    (escalations_dir / 'esc-2303-2.json').write_text(
        json.dumps({'status': 'pending', 'id': 'esc-2303-2'})
    )

    assert dr.read_escalation_status(escalations_dir, 'esc-2303-2') == 'pending'


def test_read_escalation_status_falls_back_to_archive(tmp_path: Path) -> None:
    """A resolved/dismissed escalation has been moved into archive/YYYY-MM-DD/
    by escalation.queue._archive_resolved; read_escalation_status must find
    it there when the queue-root file is absent.
    """
    escalations_dir = tmp_path / 'escalations'
    archive_dir = escalations_dir / 'archive' / '2026-07-16'
    archive_dir.mkdir(parents=True)
    (archive_dir / 'esc-2303-2.json').write_text(json.dumps({'status': 'resolved'}))

    assert dr.read_escalation_status(escalations_dir, 'esc-2303-2') == 'resolved'


def test_read_escalation_status_multiple_archive_matches_picks_newest(tmp_path: Path) -> None:
    """Duplicate ids across dated archive subdirs are not expected in normal
    operation, but read_escalation_status's tie-break -- sorted matches,
    take the last -- is an explicit, load-bearing contract and must prefer
    the newer dated subdir.
    """
    escalations_dir = tmp_path / 'escalations'
    older_dir = escalations_dir / 'archive' / '2026-07-15'
    newer_dir = escalations_dir / 'archive' / '2026-07-16'
    older_dir.mkdir(parents=True)
    newer_dir.mkdir(parents=True)
    (older_dir / 'esc-2303-2.json').write_text(json.dumps({'status': 'dismissed'}))
    (newer_dir / 'esc-2303-2.json').write_text(json.dumps({'status': 'resolved'}))

    assert dr.read_escalation_status(escalations_dir, 'esc-2303-2') == 'resolved'


def test_read_escalation_status_unknown_id_returns_none(tmp_path: Path) -> None:
    """No file at the root and none in the archive -> None, not a raise."""
    escalations_dir = tmp_path / 'escalations'
    escalations_dir.mkdir()

    assert dr.read_escalation_status(escalations_dir, 'esc-does-not-exist') is None


def test_read_escalation_status_corrupt_body_returns_none(tmp_path: Path) -> None:
    """A corrupt/non-JSON escalation body must fail soft, returning None."""
    escalations_dir = tmp_path / 'escalations'
    escalations_dir.mkdir()
    (escalations_dir / 'esc-bad.json').write_text('{not valid json')

    assert dr.read_escalation_status(escalations_dir, 'esc-bad') is None


def test_reap_answered_decisions_matrix(tmp_path: Path) -> None:
    """One reap_answered_decisions call exercises the full close-on-resolve
    matrix: resolved->ANSWERED, dismissed->DROPPED, pending/unknown->left
    OPEN, an already-closed decision is skipped WITHOUT consulting the
    callback, and a decision with no escalation_id is skipped WITHOUT
    consulting the callback either.
    """
    dr.write_decision(
        make_decision(id='dec-resolved', escalation_id='esc-1', state=dr.DecisionState.OPEN),
        root=tmp_path,
    )
    dr.write_decision(
        make_decision(id='dec-dismissed', escalation_id='esc-2', state=dr.DecisionState.OPEN),
        root=tmp_path,
    )
    dr.write_decision(
        make_decision(id='dec-pending', escalation_id='esc-3', state=dr.DecisionState.OPEN),
        root=tmp_path,
    )
    dr.write_decision(
        make_decision(id='dec-unknown', escalation_id='esc-4', state=dr.DecisionState.OPEN),
        root=tmp_path,
    )
    dr.write_decision(
        make_decision(
            id='dec-already-answered', escalation_id='esc-5', state=dr.DecisionState.ANSWERED
        ),
        root=tmp_path,
    )
    dr.write_decision(
        make_decision(
            id='dec-already-dropped', escalation_id='esc-6', state=dr.DecisionState.DROPPED
        ),
        root=tmp_path,
    )
    dr.write_decision(
        make_decision(id='dec-no-escalation', escalation_id=None, state=dr.DecisionState.OPEN),
        root=tmp_path,
    )

    # esc-4 deliberately absent from this map -> callback returns None for it.
    # esc-5/esc-6 map to a close-worthy status too, but must never be
    # consulted -- their decisions are already closed.
    statuses = {
        'esc-1': 'resolved',
        'esc-2': 'dismissed',
        'esc-3': 'pending',
        'esc-5': 'resolved',
        'esc-6': 'dismissed',
    }
    consulted: list[str] = []

    def fake_status(decision: dr.DecisionRecord) -> str | None:
        consulted.append(decision.id)
        if decision.escalation_id is None:
            return None
        return statuses.get(decision.escalation_id)

    reaped = dr.reap_answered_decisions(root=tmp_path, escalation_status=fake_status)

    reaped_by_id = {r.id: (r.escalation_id, r.new_state) for r in reaped}
    assert reaped_by_id == {
        'dec-resolved': ('esc-1', 'answered'),
        'dec-dismissed': ('esc-2', 'dropped'),
    }
    # Only the still-OPEN, escalation-bearing decisions are ever passed to
    # the callback -- already-closed and no-escalation_id decisions are
    # skipped before it is invoked.
    assert set(consulted) == {'dec-resolved', 'dec-dismissed', 'dec-pending', 'dec-unknown'}

    listed = {d.id: d.state for d in dr.list_decisions(root=tmp_path)}
    assert listed['dec-resolved'] == dr.DecisionState.ANSWERED
    assert listed['dec-dismissed'] == dr.DecisionState.DROPPED
    assert listed['dec-pending'] == dr.DecisionState.OPEN
    assert listed['dec-unknown'] == dr.DecisionState.OPEN
    assert listed['dec-already-answered'] == dr.DecisionState.ANSWERED
    assert listed['dec-already-dropped'] == dr.DecisionState.DROPPED
    assert listed['dec-no-escalation'] == dr.DecisionState.OPEN


def test_project_token_census_counts_every_state_under_the_folded_token() -> None:
    decisions = [
        make_decision(id='a', project='df', state=dr.DecisionState.OPEN),
        make_decision(id='b', project='dark-factory', state=dr.DecisionState.ANSWERED),
        make_decision(id='c', project='dark_factory', state=dr.DecisionState.DROPPED),
        make_decision(id='d', project='reify', state=dr.DecisionState.OPEN),
    ]

    assert dr.project_token_census(decisions) == Counter({'dark_factory': 3, 'reify': 1})


def test_migrate_decision_project_tokens_reports_only_changed_records(tmp_path: Path) -> None:
    """Returns one MigratedDecision per CHANGED record and nothing else.

    An already-canonical row and an unrelated project's row are not
    "migrated" -- reporting them would make the operator's `| wc -l` lie
    about how much of the legacy population actually needed repair.
    """
    seed_mixed_spellings(tmp_path)

    migrated = dr.migrate_decision_project_tokens(root=tmp_path)

    assert {(m.id, m.old_project, m.new_project) for m in migrated} == {
        ('d-df', 'df', 'dark_factory'),
        ('d-hyphen', 'dark-factory', 'dark_factory'),
        ('d-mixed', 'Dark_Factory', 'dark_factory'),
    }


def test_migrate_decision_project_tokens_canonicalizes_on_disk(tmp_path: Path) -> None:
    """The point of the sweep: after it, every dark-factory row -- whatever
    it was filed under -- reads back as the ONE canonical token, and an
    unrelated project is untouched.
    """
    seed_mixed_spellings(tmp_path)

    dr.migrate_decision_project_tokens(root=tmp_path)

    on_disk = {d.id: d.project for d in dr.list_decisions(root=tmp_path)}
    assert on_disk == {
        'd-df': 'dark_factory',
        'd-hyphen': 'dark_factory',
        'd-mixed': 'dark_factory',
        'd-canon': 'dark_factory',
        'd-reify': 'reify',
    }


def test_migrate_decision_project_tokens_changes_only_the_project_field(tmp_path: Path) -> None:
    """PRESERVATION: exactly one key may differ, on every record.

    Asserted as a whole-dict diff rather than a field checklist so a future
    field added to DecisionRecord cannot silently escape the check. This is
    the invariant that makes the sweep safe to run against the live
    population: an ANSWERED/DROPPED row must never be reopened, and filed_at
    (the cockpit's age/ordering signal) must not be restamped.
    """
    before = seed_mixed_spellings(tmp_path)

    dr.migrate_decision_project_tokens(root=tmp_path)

    after = {d.id: d.to_dict() for d in dr.list_decisions(root=tmp_path)}
    assert set(after) == set(before)
    for decision_id, before_dict in before.items():
        differing = {
            key
            for key in set(before_dict) | set(after[decision_id])
            if before_dict.get(key) != after[decision_id].get(key)
        }
        assert differing <= {'project'}, decision_id
    # Spelled out for the states that must survive, so the intent survives a
    # future refactor of the diff above.
    states = {d.id: d.state for d in dr.list_decisions(root=tmp_path)}
    assert states['d-df'] == dr.DecisionState.OPEN
    assert states['d-hyphen'] == dr.DecisionState.ANSWERED
    assert states['d-mixed'] == dr.DecisionState.DROPPED


def test_migrate_decision_project_tokens_is_idempotent(tmp_path: Path) -> None:
    """A second sweep is a no-op: nothing reported, no file rewritten.

    Follows from fold_project_token's idempotence, and is what makes the
    verb safe to keep as a permanent repair tool rather than a one-shot
    script that must be deleted after use.
    """
    seed_mixed_spellings(tmp_path)
    dr.migrate_decision_project_tokens(root=tmp_path)
    settled = {
        path: path.read_text() for path in sorted((tmp_path / 'decisions').glob('*.json'))
    }

    assert dr.migrate_decision_project_tokens(root=tmp_path) == []
    assert {path: path.read_text() for path in sorted((tmp_path / 'decisions').glob('*.json'))} == (
        settled
    )


def test_migrated_decision_is_a_frozen_result_record() -> None:
    """Mirrors the sibling ReapedDecision contract: a frozen dataclass, so a
    caller cannot mutate the audit trail of what the sweep actually did.
    """
    migrated = dr.MigratedDecision(id='d-1', old_project='df', new_project='dark_factory')

    assert (migrated.id, migrated.old_project, migrated.new_project) == (
        'd-1',
        'df',
        'dark_factory',
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        migrated.new_project = 'other'  # type: ignore[misc]


def test_migrate_decision_project_tokens_dry_run_reports_without_writing(
    tmp_path: Path,
) -> None:
    """dry_run=True previews the exact same work but touches nothing.

    An operator must be able to see the full rewrite of the live population
    -- 391 records at filing -- before committing to it. Compared as RAW FILE
    TEXT before/after, so even a byte-identical re-serialization would fail:
    "side-effect-free" means the files are not written at all, not that they
    happen to round-trip.
    """
    seed_mixed_spellings(tmp_path)
    before_text = {
        path: path.read_text() for path in sorted((tmp_path / 'decisions').glob('*.json'))
    }

    previewed = dr.migrate_decision_project_tokens(root=tmp_path, dry_run=True)

    assert {(m.id, m.old_project, m.new_project) for m in previewed} == {
        ('d-df', 'df', 'dark_factory'),
        ('d-hyphen', 'dark-factory', 'dark_factory'),
        ('d-mixed', 'Dark_Factory', 'dark_factory'),
    }
    assert {
        path: path.read_text() for path in sorted((tmp_path / 'decisions').glob('*.json'))
    } == before_text


def test_migrate_decision_project_tokens_dry_run_does_not_consume_the_work(
    tmp_path: Path,
) -> None:
    """A preview must not be mistaken for the migration: the real run
    afterwards still reports and performs the identical set of rewrites.
    """
    seed_mixed_spellings(tmp_path)

    previewed = dr.migrate_decision_project_tokens(root=tmp_path, dry_run=True)
    performed = dr.migrate_decision_project_tokens(root=tmp_path)

    assert {(m.id, m.old_project, m.new_project) for m in performed} == {
        (m.id, m.old_project, m.new_project) for m in previewed
    }
    assert {d.id for d in dr.list_decisions(root=tmp_path) if d.project == 'dark_factory'} == {
        'd-df',
        'd-hyphen',
        'd-mixed',
        'd-canon',
    }


def test_migrate_decision_project_tokens_dry_run_creates_no_lock_sidecar(
    tmp_path: Path,
) -> None:
    """dry_run takes no lock, so it leaves no ``<id>.json.lock`` sidecar.

    decision_id_lock documents that merely TAKING a lock materializes a
    sidecar (its ORPHAN SIDECARS note). A preview that littered one per
    record across the whole fleet would not be the side-effect-free
    operation its name promises.
    """
    seed_mixed_spellings(tmp_path)

    dr.migrate_decision_project_tokens(root=tmp_path, dry_run=True)

    assert list((tmp_path / 'decisions').glob('*.lock')) == []


def test_migrate_decision_project_tokens_takes_the_per_id_lock(tmp_path: Path) -> None:
    """POSITIVE counterpart to the dry-run sidecar test: a real run DOES lock.

    decision_id_lock materializes ``<id>.json.lock`` on entry (its ORPHAN
    SIDECARS note), so the sidecar's presence is the observable proof the
    read-modify-write actually ran under the same lock every other mutator
    takes -- invariant (a). Without this, a regression that dropped the lock
    would leave every other migration test green.
    """
    dr.write_decision(make_decision(id='d-df', project='df'), root=tmp_path)

    dr.migrate_decision_project_tokens(root=tmp_path)

    assert (tmp_path / 'decisions' / 'd-df.json.lock').exists()


def test_migrate_decision_project_tokens_does_not_roll_back_a_racing_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INVARIANT (b): a state transition landing between the scan and the
    write survives the sweep.

    This is the entire safety argument for running the sweep against the live
    population, and it is what a regression reusing the ``list_decisions``
    snapshot record (instead of re-reading under the lock) would break --
    silently resurrecting an ANSWERED decision back to OPEN while every other
    test stayed green.

    The race is injected deterministically by wrapping decision_id_lock: the
    concurrent transition is applied to the file just after the lock is held
    and just before the migration's re-read, which is exactly the window the
    invariant claims to cover. It writes the file directly rather than via
    update_decision_state, which would re-enter the patched lock.
    """
    dr.write_decision(
        make_decision(id='d-race', project='df', state=dr.DecisionState.OPEN), root=tmp_path
    )
    real_lock = dr.decision_id_lock
    path = tmp_path / 'decisions' / 'd-race.json'

    @contextlib.contextmanager
    def _racing_lock(decision_id: str, root: Path | str | None = None) -> Iterator[None]:
        with real_lock(decision_id, root=root):
            # A human answers the decision in the scan->write window.
            raced = json.loads(path.read_text())
            raced['state'] = dr.DecisionState.ANSWERED.value
            path.write_text(json.dumps(raced))
            yield

    monkeypatch.setattr(dr, 'decision_id_lock', _racing_lock)

    migrated = dr.migrate_decision_project_tokens(root=tmp_path)

    settled = {d.id: d for d in dr.list_decisions(root=tmp_path)}
    assert settled['d-race'].project == 'dark_factory'
    assert settled['d-race'].state == dr.DecisionState.ANSWERED
    assert {m.id for m in migrated} == {'d-race'}


def test_migrate_decision_project_tokens_reports_the_token_it_actually_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INVARIANT (b), audit half: ``old_project`` names what was ON DISK at
    write time, not what the pre-lock scan happened to see.

    Here a concurrent writer re-spells the token inside the scan->write
    window. Deriving the new token from the stale snapshot would clobber that
    writer's value, and reporting the snapshot's ``old_project`` would put a
    token in the audit trail that was never the one replaced.
    """
    dr.write_decision(make_decision(id='d-respell', project='df'), root=tmp_path)
    real_lock = dr.decision_id_lock
    path = tmp_path / 'decisions' / 'd-respell.json'

    @contextlib.contextmanager
    def _respelling_lock(decision_id: str, root: Path | str | None = None) -> Iterator[None]:
        with real_lock(decision_id, root=root):
            raced = json.loads(path.read_text())
            raced['project'] = 'Dark-Factory'
            path.write_text(json.dumps(raced))
            yield

    monkeypatch.setattr(dr, 'decision_id_lock', _respelling_lock)

    migrated = dr.migrate_decision_project_tokens(root=tmp_path)

    assert [(m.id, m.old_project, m.new_project) for m in migrated] == [
        ('d-respell', 'Dark-Factory', 'dark_factory')
    ]


def test_migrate_decision_project_tokens_skips_a_record_canonicalized_in_the_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record another writer already canonicalized in the scan->write window
    is skipped, not re-reported.

    The pre-filter runs off the snapshot, so such a record still reaches the
    lock; the re-read is what turns it into a no-op. Reporting it would make
    the operator's audit trail claim a rewrite that this sweep never made.
    """
    dr.write_decision(make_decision(id='d-beaten', project='df'), root=tmp_path)
    real_lock = dr.decision_id_lock
    path = tmp_path / 'decisions' / 'd-beaten.json'

    @contextlib.contextmanager
    def _beating_lock(decision_id: str, root: Path | str | None = None) -> Iterator[None]:
        with real_lock(decision_id, root=root):
            raced = json.loads(path.read_text())
            raced['project'] = 'dark_factory'
            path.write_text(json.dumps(raced))
            yield

    monkeypatch.setattr(dr, 'decision_id_lock', _beating_lock)

    assert dr.migrate_decision_project_tokens(root=tmp_path) == []
    assert {d.id: d.project for d in dr.list_decisions(root=tmp_path)} == {
        'd-beaten': 'dark_factory'
    }


def test_migrate_decision_project_tokens_skips_a_corrupt_neighbour(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One unparseable file must not abort the sweep of the rest.

    list_decisions already skips it on the read side; this pins that the
    whole call still migrates every good record and leaves the bad file
    byte-for-byte alone rather than "repairing" something it cannot parse.
    """
    for decision_id in ('d-good-1', 'd-good-2'):
        dr.write_decision(make_decision(id=decision_id, project='df'), root=tmp_path)
    broken = tmp_path / 'decisions' / 'broken.json'
    broken.write_text('{not json at all')

    with caplog.at_level(logging.ERROR):
        migrated = dr.migrate_decision_project_tokens(root=tmp_path)

    assert {m.id for m in migrated} == {'d-good-1', 'd-good-2'}
    assert {d.id: d.project for d in dr.list_decisions(root=tmp_path)} == {
        'd-good-1': 'dark_factory',
        'd-good-2': 'dark_factory',
    }
    assert broken.read_text() == '{not json at all'


def test_migrate_decision_project_tokens_skips_an_unwritable_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A record whose write fails is absent from the result and does not stop
    the sweep -- mirrors reap_answered_decisions' "only record it when the
    update actually succeeded" contract. Reporting a rewrite that never
    landed would make the operator's audit trail lie.
    """
    for decision_id in ('d-ok', 'd-bad'):
        dr.write_decision(make_decision(id=decision_id, project='df'), root=tmp_path)
    real_write = dr.write_decision

    def _flaky_write(record: dr.DecisionRecord, root: Path | str | None = None) -> bool:
        if record.id == 'd-bad':
            return False
        return real_write(record, root=root)

    monkeypatch.setattr(dr, 'write_decision', _flaky_write)

    with caplog.at_level(logging.ERROR):
        migrated = dr.migrate_decision_project_tokens(root=tmp_path)

    assert {m.id for m in migrated} == {'d-ok'}
    assert {d.id: d.project for d in dr.list_decisions(root=tmp_path)} == {
        'd-ok': 'dark_factory',
        'd-bad': 'df',
    }


def test_migrate_decision_project_tokens_survives_a_raising_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A per-record fault mid-loop is logged and skipped, never propagated.

    Matches the module-wide fail-soft convention (write_decision,
    update_decision_state and list_decisions all self-guard): one record
    faulting must not deny the operator the repair of the other 390.
    """
    for decision_id in ('d-ok', 'd-boom'):
        dr.write_decision(make_decision(id=decision_id, project='df'), root=tmp_path)
    real_write = dr.write_decision

    def _exploding_write(record: dr.DecisionRecord, root: Path | str | None = None) -> bool:
        if record.id == 'd-boom':
            raise OSError('disk on fire')
        return real_write(record, root=root)

    monkeypatch.setattr(dr, 'write_decision', _exploding_write)

    with caplog.at_level(logging.ERROR):
        migrated = dr.migrate_decision_project_tokens(root=tmp_path)

    assert {m.id for m in migrated} == {'d-ok'}
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


class TestClosingEvidence:
    """The deciding evidence, quoted verbatim on the closed record, and the one mutator that writes it."""

    def _seed(self, root: Path, **overrides: object) -> Path:
        dr.write_decision(make_decision(id='dec-close', **{'state': dr.DecisionState.OPEN, **overrides}), root=root)
        return dr.decision_path_for_id('dec-close', root=root)

    def _close(self, root: Path, state: str = dr.DecisionState.ANSWERED, evidence: str = EVIDENCE):
        return dr.close_decision_with_evidence(
            'dec-close', state, evidence, root=root, **_identity_of(make_decision())
        )

    def test_defaults_to_empty(self) -> None:
        d = dr.DecisionRecord(id='dec-1', project='df', text='approve?', filed_at='2026-07-07T00:00:00+00:00')

        assert d.closing_evidence == ''

    def test_round_trips_losslessly(self) -> None:
        d = make_decision(closing_evidence=EVIDENCE, escalations_dir='/p/data/escalations')

        assert d.to_dict()['closing_evidence'] == EVIDENCE
        assert dr.DecisionRecord.from_dict(d.to_dict()) == d
        assert dr.DecisionRecord.from_json(d.to_json()) == d

    @pytest.mark.parametrize('present', [{}, {'closing_evidence': None}], ids=['absent', 'null'])
    def test_an_absent_or_null_key_parses_as_empty(self, present: dict) -> None:
        data = make_decision().to_dict()
        del data['closing_evidence']

        assert dr.DecisionRecord.from_dict({**data, **present}).closing_evidence == ''

    def test_a_pre_severity_era_minimal_record_parses_as_empty(self) -> None:
        minimal = {'id': 'dec-old', 'project': 'df', 'text': 'q', 'filed_at': '2026-05-01T00:00:00+00:00'}

        assert dr.DecisionRecord.from_dict(minimal).closing_evidence == ''

    @pytest.mark.parametrize('state', [dr.DecisionState.ANSWERED, dr.DecisionState.DROPPED])
    def test_open_to_terminal_sets_state_and_evidence_in_one_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
    ) -> None:
        self._seed(tmp_path)
        real_write = dr.write_decision
        written: list[dict[str, Any]] = []

        def recording_write(record: dr.DecisionRecord, root: Path | str | None = None) -> bool:
            written.append(record.to_dict())
            return real_write(record, root=root)

        monkeypatch.setattr(dr, 'write_decision', recording_write)

        before = datetime.now(UTC)
        updated = self._close(tmp_path, state)
        after = datetime.now(UTC)

        assert updated is not None
        assert (updated.state, updated.closing_evidence) == (state, EVIDENCE)
        [write] = written
        assert (write['state'], write['closing_evidence'], write['closed_at']) == (state, EVIDENCE, updated.closed_at)
        _assert_utc_instant_within(updated.closed_at, before, after)
        [reread] = dr.list_decisions(root=tmp_path)
        assert reread == updated

    def test_evidence_attaches_to_a_record_the_reaper_already_closed(self, tmp_path: Path) -> None:
        self._seed(tmp_path, state=dr.DecisionState.ANSWERED)

        updated = self._close(tmp_path)

        assert updated is not None
        assert (updated.state, updated.closing_evidence) == (dr.DecisionState.ANSWERED, EVIDENCE)

    @pytest.mark.parametrize(('seeded', 'state', 'evidence', 'reason'), [
        ({}, dr.DecisionState.OPEN, EVIDENCE, r"not 'open'"),
        ({}, dr.DecisionState.ANSWERED, '', 'evidence is empty'),
        ({}, dr.DecisionState.ANSWERED, '  \n\t', 'evidence is empty'),
        ({'state': dr.DecisionState.ANSWERED}, dr.DecisionState.DROPPED, EVIDENCE, 'between terminal states'),
        ({'state': dr.DecisionState.ANSWERED, 'closing_evidence': 'the reaper-era evidence'},
         dr.DecisionState.ANSWERED, EVIDENCE, 'overwrite'),
    ], ids=['to-open', 'empty', 'whitespace', 'terminal-to-terminal', 'overwrite'])
    def test_refusals_raise_naming_the_refusal_and_write_nothing(
        self, tmp_path: Path, seeded: dict, state: str, evidence: str, reason: str
    ) -> None:
        path = self._seed(tmp_path, **seeded)
        before = path.read_bytes()

        with pytest.raises(dr.DecisionCloseRefused, match=reason):
            self._close(tmp_path, state, evidence)

        assert path.read_bytes() == before
        assert dr.DecisionRecord.from_json(path.read_text()).closed_at == ''

    @pytest.mark.parametrize(('state', 'evidence'), [
        (dr.DecisionState.OPEN, EVIDENCE),
        (dr.DecisionState.ANSWERED, ' '),
    ])
    def test_argument_refusals_take_no_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str, evidence: str
    ) -> None:
        """Mirrors test_main_write_decision_refusal_takes_no_lock: a lock sidecar is never cleaned up."""
        self._seed(tmp_path)
        acquired: list[str] = []

        @contextlib.contextmanager
        def recording_lock(decision_id: str, root: Path | str | None = None):
            acquired.append(decision_id)
            yield

        monkeypatch.setattr(dr, 'decision_id_lock', recording_lock)

        with pytest.raises(dr.DecisionCloseRefused):
            self._close(tmp_path, state, evidence)

        assert acquired == []

    def test_fail_soft_when_absent(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.ERROR):
            assert self._close(tmp_path) is None

        assert any(r.levelno >= logging.ERROR for r in caplog.records)

    def test_fail_soft_on_a_corrupt_body(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        path = dr.decision_path_for_id('dec-close', root=tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text('{not valid json')

        with caplog.at_level(logging.ERROR):
            assert self._close(tmp_path) is None

        assert any(r.levelno >= logging.ERROR for r in caplog.records)

    def test_fail_soft_on_an_unwritable_root(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        blocker = tmp_path / 'blocker'
        blocker.write_text('not a directory')

        with caplog.at_level(logging.ERROR):
            assert dr.close_decision_with_evidence(
                'dec-close', dr.DecisionState.ANSWERED, EVIDENCE, root=blocker / 'fleet',
                **_identity_of(make_decision()),
            ) is None

        assert any(r.levelno >= logging.ERROR for r in caplog.records)

    @pytest.mark.timeout(30)
    def test_a_concurrent_boost_loses_neither_mutation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same technique as test_concurrent_state_and_boost_updates_do_not_lose_a_field."""
        self._seed(tmp_path, manual_boost=0)
        real_write = dr.write_decision

        def delayed_write(record: dr.DecisionRecord, root: Path | str | None = None) -> bool:
            time.sleep(0.3)
            return real_write(record, root=root)

        monkeypatch.setattr(dr, 'write_decision', delayed_write)
        threads = [
            threading.Thread(target=self._close, args=(tmp_path,)),
            threading.Thread(target=dr.set_manual_boost, args=('dec-close', 7, tmp_path)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not any(thread.is_alive() for thread in threads)
        [reread] = dr.list_decisions(root=tmp_path)
        assert (reread.state, reread.closing_evidence, reread.manual_boost) == (
            dr.DecisionState.ANSWERED, EVIDENCE, 7,
        )

    def test_same_queue_refile_keeps_the_evidence(self) -> None:
        closed_at = '2026-09-20T05:31:00+00:00'
        existing = make_decision(
            state=dr.DecisionState.ANSWERED, closing_evidence=EVIDENCE, closed_at=closed_at, escalations_dir='/q'
        )
        incoming = make_decision(state=dr.DecisionState.OPEN, escalations_dir='/q')

        for merged in (dr.merge_same_queue_refile(existing, incoming), dr.merge_decision_enrichment(existing, incoming)):
            assert (merged.closing_evidence, merged.closed_at) == (EVIDENCE, closed_at)


class TestClosedAt:
    """When the evidence-carrying close happened: custody, stamped by close_decision_with_evidence alone."""

    def test_defaults_to_empty(self) -> None:
        d = dr.DecisionRecord(id='dec-1', project='df', text='approve?', filed_at='2026-07-07T00:00:00+00:00')

        assert d.closed_at == ''

    def test_round_trips_losslessly(self) -> None:
        d = make_decision(closing_evidence=EVIDENCE, closed_at='2026-09-20T05:31:00+00:00')

        assert d.to_dict()['closed_at'] == '2026-09-20T05:31:00+00:00'
        assert dr.DecisionRecord.from_dict(d.to_dict()) == d
        assert dr.DecisionRecord.from_json(d.to_json()) == d

    @pytest.mark.parametrize('present', [{}, {'closed_at': None}], ids=['absent', 'null'])
    def test_an_absent_or_null_key_parses_as_empty(self, present: dict) -> None:
        data = make_decision().to_dict()
        del data['closed_at']

        assert dr.DecisionRecord.from_dict({**data, **present}).closed_at == ''

    def test_a_pre_severity_era_minimal_record_parses_as_empty(self) -> None:
        minimal = {'id': 'dec-old', 'project': 'df', 'text': 'q', 'filed_at': '2026-05-01T00:00:00+00:00'}

        assert dr.DecisionRecord.from_dict(minimal).closed_at == ''

    def test_attaching_evidence_after_the_reaper_closed_stamps_it(self, tmp_path: Path) -> None:
        seeded = make_decision(id='dec-close', state=dr.DecisionState.ANSWERED)
        dr.write_decision(seeded, root=tmp_path)

        before = datetime.now(UTC)
        updated = dr.close_decision_with_evidence(
            'dec-close', dr.DecisionState.ANSWERED, EVIDENCE, root=tmp_path, **_identity_of(seeded)
        )
        after = datetime.now(UTC)

        assert updated is not None
        _assert_utc_instant_within(updated.closed_at, before, after)

    def test_the_other_state_and_boost_writers_never_stamp_it(self, tmp_path: Path) -> None:
        dr.write_decision(make_decision(id='dec-close', state=dr.DecisionState.OPEN), root=tmp_path)

        closed = dr.update_decision_state('dec-close', dr.DecisionState.ANSWERED, root=tmp_path)
        boosted = dr.set_manual_boost('dec-close', 7, root=tmp_path)

        assert closed is not None and boosted is not None
        assert (closed.closed_at, boosted.closed_at) == ('', '')


class TestReopenDecision:
    """reopen_decision re-opens a closed record IN PLACE, under the same compare-and-swap as a close (task 4835)."""

    _A_PROJECT = 'know_live'

    _A_QUEUE = '/a/data/escalations'

    _B_PROJECT = 'reify'

    _B_QUEUE = '/b/data/escalations'

    _FILED_AT = '2026-07-07T00:00:00+00:00'

    def _seed(self, root: Path, **overrides: object) -> dr.DecisionRecord:
        fields = {
            'id': 'esc-42-1', 'project': self._A_PROJECT, 'escalations_dir': self._A_QUEUE,
            'state': dr.DecisionState.DROPPED, 'closing_evidence': EVIDENCE,
            'closed_at': '2026-09-20T09:00:00+00:00', 'manual_boost': 3, 'filed_at': self._FILED_AT,
            **overrides,
        }
        seeded = make_decision(**fields)
        dr.write_decision(seeded, root=root)
        return seeded

    def _path(self, root: Path) -> Path:
        return dr.decision_path_for_id('esc-42-1', root=root)

    @pytest.mark.parametrize('closed_state', [dr.DecisionState.DROPPED, dr.DecisionState.ANSWERED])
    def test_a_closed_record_reopens_with_its_closing_fields_cleared(
        self, tmp_path: Path, closed_state: str
    ) -> None:
        seeded = self._seed(tmp_path, state=closed_state)

        reopened = dr.reopen_decision('esc-42-1', root=tmp_path, **_identity_of(seeded))

        [reread] = dr.list_decisions(root=tmp_path)
        for record in (reopened, reread):
            assert record is not None
            assert (record.state, record.closing_evidence, record.closed_at) == (dr.DecisionState.OPEN, '', '')
            assert (record.filed_at, record.text, record.manual_boost, record.escalations_dir) == (
                self._FILED_AT, seeded.text, 3, self._A_QUEUE,
            )

    def test_an_open_record_stays_open(self, tmp_path: Path) -> None:
        seeded = self._seed(tmp_path, state=dr.DecisionState.OPEN, closing_evidence='', closed_at='')

        reopened = dr.reopen_decision('esc-42-1', root=tmp_path, **_identity_of(seeded))

        assert reopened is not None
        assert reopened.state == dr.DecisionState.OPEN

    @pytest.mark.parametrize(('project', 'queue'), [
        (_A_PROJECT, _B_QUEUE),
        (_B_PROJECT, _A_QUEUE),
        (_B_PROJECT, _B_QUEUE),
    ], ids=['queue-only', 'project-only', 'both'])
    def test_another_projects_record_at_the_same_id_is_refused_and_untouched(
        self, tmp_path: Path, project: str, queue: str
    ) -> None:
        self._seed(tmp_path)
        before = self._path(tmp_path).read_bytes()

        with pytest.raises(dr.DecisionReopenRefused) as refused:
            dr.reopen_decision(
                'esc-42-1', root=tmp_path, expected_project=project, expected_escalations_dir=queue
            )

        message = str(refused.value)
        for named in ('esc-42-1', self._A_PROJECT, self._A_QUEUE, project, queue):
            assert named in message
        assert self._path(tmp_path).read_bytes() == before

    @pytest.mark.parametrize('stamp', ['', dr.UNKNOWN_QUEUE], ids=['legacy', 'unknown'])
    def test_a_sentinel_stamp_reopens_under_the_same_sentinel(self, tmp_path: Path, stamp: str) -> None:
        self._seed(tmp_path, escalations_dir=stamp)

        reopened = dr.reopen_decision(
            'esc-42-1', root=tmp_path, expected_project=self._A_PROJECT, expected_escalations_dir=stamp
        )

        assert reopened is not None
        assert reopened.state == dr.DecisionState.OPEN

    @pytest.mark.parametrize(('stamp', 'queue'), [
        ('', '/q/data/escalations'),
        (dr.UNKNOWN_QUEUE, ''),
        (dr.UNKNOWN_QUEUE, '/q/data/escalations'),
    ], ids=['legacy-vs-real', 'unknown-vs-legacy', 'unknown-vs-real'])
    def test_a_sentinel_stamp_is_refused_under_any_other_expectation(
        self, tmp_path: Path, stamp: str, queue: str
    ) -> None:
        self._seed(tmp_path, escalations_dir=stamp)
        before = self._path(tmp_path).read_bytes()

        with pytest.raises(dr.DecisionReopenRefused, match='esc-42-1'):
            dr.reopen_decision(
                'esc-42-1', root=tmp_path, expected_project=self._A_PROJECT, expected_escalations_dir=queue
            )

        assert self._path(tmp_path).read_bytes() == before

    def test_a_missing_record_is_none(self, tmp_path: Path) -> None:
        assert dr.reopen_decision(
            'esc-42-1', root=tmp_path, expected_project=self._A_PROJECT, expected_escalations_dir=self._A_QUEUE
        ) is None

    def test_a_reopened_record_can_be_closed_with_fresh_evidence(self, tmp_path: Path) -> None:
        seeded = self._seed(tmp_path)
        dr.reopen_decision('esc-42-1', root=tmp_path, **_identity_of(seeded))

        closed = dr.close_decision_with_evidence(
            'esc-42-1', dr.DecisionState.ANSWERED, 'new evidence', root=tmp_path, **_identity_of(seeded)
        )

        assert closed is not None
        assert (closed.state, closed.closing_evidence) == (dr.DecisionState.ANSWERED, 'new evidence')
