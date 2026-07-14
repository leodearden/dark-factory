"""Tests for orchestrator.session_registry module.

Covers: schema/contract round-trip (SCHEMA_VERSION, Status enum, SessionRecord
to/from dict/json); slug sanitization + path/transcript encoding; single-writer
atomic write/read/update; TTL/pid stale-record reaper matrix; CLI subcommands
+ fail-soft; and the G5 two-way boundary test (write by a real spawn -> refresh
by a simulated hook -> reap), exercised jointly with
tests/scripts/test_spawn_claude.py's bash-level harness.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import re
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

from orchestrator import session_registry as sr

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_record(**overrides: object) -> sr.SessionRecord:
    """Build a fully-populated SessionRecord for round-trip/identity tests.

    Every field is given a concrete, distinguishable value so a round-trip
    test can catch a field being dropped/mis-typed; ``overrides`` lets a
    test tweak just the field(s) it cares about. Includes the C1 schema
    extensions (parent_session_id/spawn_mode/display/question) alongside the
    original rail fields.
    """
    # Declared as a bare `dict` (not `dict[str, object]`) so pyright treats it
    # as dict[Unknown, Unknown] at the **fields unpack below -- mirrors
    # test_merge_types_invariants.py's _real_kwargs/_decided_kwargs/_entry_kwargs,
    # the established idiom in this suite for a kwargs-builder helper whose
    # result is unpacked into a strictly-typed constructor.
    fields: dict = {
        'session_slug': 'unblock-df-2085-4242',
        'status': sr.Status.LAUNCHING,
        'title': 'unblock:df#2085 routing-mechanism',
        'role': 'unblock',
        'project': 'df',
        'task_id': '2085',
        'escalation_id': 'esc-1',
        'prompt': '/unblock 2085',
        'cwd': '/home/leo/src/dark-factory',
        'launcher_pid': 4242,
        'start_ts': '2026-07-07T00:00:00+00:00',
        'exit_code': None,
        'result_file': None,
        'transcript_path': '~/.claude/projects/-home-leo-src-dark-factory',
        'parent_session_id': 'root-df-1-1',
        'spawn_mode': 'sibling',
        'display': sr.Display(
            kind='wm', wm_title='unblock:df#2085 slug', wm_window_id='0x1a', tmux_target=None
        ),
        'question': sr.Question(text='approve rollout?', asked_at='2026-07-07T00:00:00+00:00'),
    }
    fields.update(overrides)
    return sr.SessionRecord(**fields)


# A pid virtually guaranteed dead on any host (matches the convention already
# established by orchestrator.harness._pid_alive's own test suite: 2**31 - 1,
# orchestrator/tests/test_reconcile_stranded.py:34).
_DEAD_PID = 2**31 - 1

_NOW = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)


def _set_mtime(path: Path, now: datetime, age: timedelta) -> None:
    """Backdate *path*'s mtime to ``now - age`` (reap_stale_records' heartbeat clock)."""
    ts = (now - age).timestamp()
    os.utime(path, (ts, ts))


# ---------------------------------------------------------------------------
# Step-1: schema / contract
# ---------------------------------------------------------------------------


def test_schema_version_is_int() -> None:
    assert isinstance(sr.SCHEMA_VERSION, int)


def test_status_enum_has_exact_six_wire_values() -> None:
    assert issubclass(sr.Status, str)
    values = {member.value for member in sr.Status}
    assert values == {
        'launching',
        'running',
        'awaiting-input',
        'idle',
        'exited',
        'failed-to-start',
    }
    assert len(list(sr.Status)) == 6
    # Spot-check the documented wire-value <-> member mapping.
    assert sr.Status('launching') is sr.Status.LAUNCHING
    assert sr.Status('running') is sr.Status.RUNNING
    assert sr.Status('awaiting-input') is sr.Status.AWAITING_INPUT
    assert sr.Status('idle') is sr.Status.IDLE
    assert sr.Status('exited') is sr.Status.EXITED
    assert sr.Status('failed-to-start') is sr.Status.FAILED_TO_START


def test_session_record_schema_version_defaults_to_schema_version() -> None:
    r = _make_record()
    assert r.schema_version == sr.SCHEMA_VERSION


def test_session_record_dict_round_trip_is_lossless() -> None:
    r = _make_record()
    assert sr.SessionRecord.from_dict(r.to_dict()) == r


def test_session_record_json_round_trip_is_lossless() -> None:
    r = _make_record()
    assert sr.SessionRecord.from_json(r.to_json()) == r


def test_session_record_json_round_trip_with_null_fields() -> None:
    """exit_code/result_file/task_id/escalation_id may legitimately be None
    (a freshly-launched record has no exit code yet); the round trip must
    preserve that, not coerce it to a string or drop the key.
    """
    r = _make_record(
        task_id=None,
        escalation_id=None,
        exit_code=None,
        result_file=None,
        transcript_path=None,
    )
    assert sr.SessionRecord.from_json(r.to_json()) == r


# ---------------------------------------------------------------------------
# Step-1: SessionRecord C1 schema extensions (B1) -- Fleet Cockpit
# parent_session_id/spawn_mode/display/question, SCHEMA_MINOR, SpawnMode,
# DisplayKind, Display, Question. Additive + migration-free: a rail-vintage
# dict (no C1 keys) must still parse, defaulting the new fields.
# ---------------------------------------------------------------------------


def test_session_record_round_trip_includes_new_c1_fields() -> None:
    r = _make_record()
    assert sr.SessionRecord.from_dict(r.to_dict()) == r
    assert sr.SessionRecord.from_json(r.to_json()) == r

    # A display whose optional wm_window_id/tmux_target are both None must
    # also round-trip losslessly (None-safe nested (dis)assembly).
    r_none_optionals = _make_record(
        display=sr.Display(
            kind='wm', wm_title='unblock:df#2085 slug', wm_window_id=None, tmux_target=None
        ),
    )
    assert sr.SessionRecord.from_dict(r_none_optionals.to_dict()) == r_none_optionals
    assert sr.SessionRecord.from_json(r_none_optionals.to_json()) == r_none_optionals

    # A record with no parent/display/question at all (the common
    # human-launched-root shape) must also round-trip losslessly.
    r_all_none = _make_record(parent_session_id=None, display=None, question=None)
    assert sr.SessionRecord.from_dict(r_all_none.to_dict()) == r_all_none
    assert sr.SessionRecord.from_json(r_all_none.to_json()) == r_all_none


def test_session_record_parses_rail_vintage_dict_migration_free() -> None:
    """A dict written by the pre-C1 (rail) module -- containing only the
    original schema keys, no parent_session_id/spawn_mode/display/question --
    must still parse via from_dict without raising, defaulting the new C1
    fields to None/'child'/None/None (migration-free additive contract).
    """
    rail_vintage = {
        'schema_version': 1,
        'session_slug': 'unblock-df-2085-4242',
        'title': 'unblock:df#2085 routing-mechanism',
        'role': 'unblock',
        'project': 'df',
        'task_id': '2085',
        'escalation_id': 'esc-1',
        'prompt': '/unblock 2085',
        'cwd': '/home/leo/src/dark-factory',
        'launcher_pid': 4242,
        'start_ts': '2026-07-07T00:00:00+00:00',
        'status': 'launching',
        'exit_code': None,
        'result_file': None,
        'transcript_path': '~/.claude/projects/-home-leo-src-dark-factory',
    }
    record = sr.SessionRecord.from_dict(rail_vintage)
    assert record.parent_session_id is None
    assert record.spawn_mode == sr.SpawnMode.CHILD
    assert record.display is None
    assert record.question is None


def test_schema_minor_is_int_bumped() -> None:
    assert isinstance(sr.SCHEMA_MINOR, int)
    assert sr.SCHEMA_MINOR >= 1
    # The PERSISTED major must stay migration-free: bumping it would make
    # rail-vintage and C1 records version-distinguishable on disk.
    assert sr.SCHEMA_VERSION == 1


def test_spawn_mode_enum_values() -> None:
    assert issubclass(sr.SpawnMode, str)
    assert {m.value for m in sr.SpawnMode} == {'child', 'sibling', 'detached'}
    assert sr.SpawnMode.CHILD.value == 'child'
    assert sr.SpawnMode.SIBLING.value == 'sibling'
    assert sr.SpawnMode.DETACHED.value == 'detached'


def test_display_kind_enum_values() -> None:
    assert issubclass(sr.DisplayKind, str)
    assert {m.value for m in sr.DisplayKind} == {'wm', 'tmux'}
    assert sr.DisplayKind.WM.value == 'wm'
    assert sr.DisplayKind.TMUX.value == 'tmux'


def test_display_round_trip() -> None:
    with_optionals = sr.Display(
        kind='wm', wm_title='unblock:df#2085 slug', wm_window_id='0x1a', tmux_target=None
    )
    assert sr.Display.from_dict(with_optionals.to_dict()) == with_optionals

    without_optionals = sr.Display(
        kind='tmux', wm_title='', wm_window_id=None, tmux_target='session:0.1'
    )
    assert sr.Display.from_dict(without_optionals.to_dict()) == without_optionals


def test_question_round_trip() -> None:
    q = sr.Question(text='approve rollout?', asked_at='2026-07-07T00:00:00+00:00')
    assert sr.Question.from_dict(q.to_dict()) == q


# ---------------------------------------------------------------------------
# Step-3: DecisionRecord type + paths (B2) -- Fleet Cockpit
# ---------------------------------------------------------------------------


def _make_decision(**overrides: object) -> sr.DecisionRecord:
    """Build a fully-populated DecisionRecord for round-trip/identity tests.

    Mirrors _make_record's kwargs-builder idiom: every field is given a
    concrete, distinguishable value so a round-trip test can catch a field
    being dropped/mis-typed; ``overrides`` lets a test tweak just the
    field(s) it cares about.
    """
    fields: dict = {
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
        'severity': 'critical',
    }
    fields.update(overrides)
    return sr.DecisionRecord(**fields)


def test_decision_state_enum_values() -> None:
    assert issubclass(sr.DecisionState, str)
    assert {m.value for m in sr.DecisionState} == {'open', 'answered', 'dropped'}
    assert sr.DecisionState.OPEN.value == 'open'
    assert sr.DecisionState.ANSWERED.value == 'answered'
    assert sr.DecisionState.DROPPED.value == 'dropped'


def test_decision_record_dict_round_trip_is_lossless() -> None:
    d = _make_decision()
    assert sr.DecisionRecord.from_dict(d.to_dict()) == d


def test_decision_record_json_round_trip_is_lossless() -> None:
    d = _make_decision()
    assert sr.DecisionRecord.from_json(d.to_json()) == d


def test_decision_record_round_trip_with_null_fields() -> None:
    """session_id/task_id/escalation_id/options may legitimately be None (a
    project-level decision with no session/task/escalation context yet);
    the round trip must preserve that, not coerce it or drop the key.
    """
    d = _make_decision(session_id=None, task_id=None, escalation_id=None, options=None)
    round_tripped = sr.DecisionRecord.from_dict(d.to_dict())
    assert round_tripped == d
    assert round_tripped.session_id is None
    assert round_tripped.task_id is None
    assert round_tripped.escalation_id is None
    assert round_tripped.options is None


def test_decision_record_defaults() -> None:
    d = sr.DecisionRecord(
        id='dec-1', project='df', text='approve?', filed_at='2026-07-07T00:00:00+00:00'
    )
    assert d.manual_boost == 0
    assert d.state == sr.DecisionState.OPEN
    assert d.severity == ''


def test_decision_record_to_dict_includes_severity() -> None:
    d = _make_decision(severity='blocking')
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
    record = sr.DecisionRecord.from_dict(pre_severity)
    assert record.severity == ''


def test_decision_path_for_id_under_decisions_dir(tmp_path: Path) -> None:
    assert sr.decision_path_for_id('dec-1', root=tmp_path) == tmp_path / 'decisions' / 'dec-1.json'


def test_decision_path_for_id_sanitizes_unsafe_id(tmp_path: Path) -> None:
    # An id containing '/' must not escape decisions_dir: it still resolves
    # to a single file directly inside decisions_dir (no nested directory,
    # no traversal) -- mirrors lease_path_for_name's path-escape guard.
    path = sr.decision_path_for_id('../../etc/passwd', root=tmp_path)
    assert path.parent == sr.decisions_dir(root=tmp_path)
    # Lock the actual sanitized stem, not just "has the right parent": every
    # '/' maps to '-' via _DECISION_ID_SANITIZE_RE while '.' is preserved.
    assert path.name == '..-..-etc-passwd.json'
    assert '/' not in path.name


# ---------------------------------------------------------------------------
# Step-3: identity, paths, transcript encoding
# ---------------------------------------------------------------------------


def test_build_session_slug_joins_role_project_task_pid() -> None:
    assert sr.build_session_slug('unblock', 'df', '2085', 4242) == 'unblock-df-2085-4242'


@pytest.mark.parametrize('task_id', [None, ''])
def test_build_session_slug_omits_task_segment_when_absent(task_id: str | None) -> None:
    assert sr.build_session_slug('unblock', 'df', task_id, 4242) == 'unblock-df-4242'


def test_build_session_slug_sanitizes_special_chars() -> None:
    slug = sr.build_session_slug('un block', 'df/prod', '20#85', 4242)
    assert slug == 'un-block-df-prod-20-85-4242'
    assert re.fullmatch(r'[A-Za-z0-9._-]+', slug)


# ---------------------------------------------------------------------------
# Task 2511 step-1: sanitize_slug (shared filesystem-safe-slug normalizer)
# ---------------------------------------------------------------------------


def test_sanitize_slug_leaves_already_safe_slug_unchanged() -> None:
    assert sr.sanitize_slug('session-cockpit-3215033') == 'session-cockpit-3215033'


def test_sanitize_slug_maps_unsafe_chars_to_dash() -> None:
    # '/' and '#' fall outside [A-Za-z0-9._-] (record_path_for_slug's
    # path-escape guard) and are mapped to '-'; '.' is itself in the allowed
    # set (mirrors _SLUG_SANITIZE_RE, the idiom this delegates to), so a
    # path separator can never escape sessions_dir via record_path_for_slug.
    slug = sr.sanitize_slug('weird/id#x.y')
    assert slug == 'weird-id-x.y'
    assert re.fullmatch(r'[A-Za-z0-9._-]+', slug)


def test_sanitize_slug_matches_build_session_slug_regression() -> None:
    # Pins build_session_slug's output as byte-identical to today once it
    # delegates to sanitize_slug (step-2's behavior-preserving refactor).
    assert sr.build_session_slug('unblock', 'df', '2085', 4242) == 'unblock-df-2085-4242'
    slug = sr.build_session_slug('un block', 'df/prod', '20#85', 4242)
    assert slug == 'un-block-df-prod-20-85-4242'
    assert slug == sr.sanitize_slug('un block-df/prod-20#85-4242')


def test_fleet_root_defaults_to_dot_claude_fleet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('CLAUDE_FLEET_ROOT', raising=False)
    monkeypatch.setenv('HOME', '/home/fakeuser')
    assert sr.fleet_root() == Path('/home/fakeuser/.claude/fleet')


def test_fleet_root_honors_env_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    assert sr.fleet_root() == tmp_path


def test_fleet_root_explicit_root_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', '/should/not/be/used')
    other = tmp_path / 'other'
    assert sr.fleet_root(root=other) == other


def test_record_path_for_slug(tmp_path: Path) -> None:
    path = sr.record_path_for_slug('unblock-df-2085-4242', root=tmp_path)
    assert path == tmp_path / 'sessions' / 'unblock-df-2085-4242' / 'record.json'


def test_transcript_path_for_cwd_encodes_slash() -> None:
    assert (
        sr.transcript_path_for_cwd('/home/leo/src/dark-factory')
        == '~/.claude/projects/-home-leo-src-dark-factory'
    )


def test_transcript_path_for_cwd_encodes_dot() -> None:
    # Regression fixture confirmed against a real ~/.claude/projects/ dir
    # (plans/session-attention-rail-prd.md §3): '.' -> '-' too, so a cwd
    # component starting with '.' yields a doubled '--' (one dash from the
    # preceding '/', one from the '.').
    assert (
        sr.transcript_path_for_cwd('/home/leo/.openclaw-workspace')
        == '~/.claude/projects/-home-leo--openclaw-workspace'
    )


# ---------------------------------------------------------------------------
# Step-5: single-writer atomic write / read / update
# ---------------------------------------------------------------------------


def test_write_record_creates_record_json_matching_the_record(tmp_path: Path) -> None:
    r = _make_record()
    sr.write_record(r, root=tmp_path)
    path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    assert path.is_file()
    assert sr.SessionRecord.from_json(path.read_text()) == r


def test_write_record_leaves_no_leftover_tmp_file(tmp_path: Path) -> None:
    r = _make_record()
    sr.write_record(r, root=tmp_path)
    slug_dir = sr.record_path_for_slug(r.session_slug, root=tmp_path).parent
    leftovers = [p for p in slug_dir.iterdir() if p.suffix == '.tmp']
    assert leftovers == []


def test_read_record_returns_equal_record(tmp_path: Path) -> None:
    r = _make_record()
    sr.write_record(r, root=tmp_path)
    assert sr.read_record(r.session_slug, root=tmp_path) == r


def test_update_status_mutates_status_and_exit_code_in_place(tmp_path: Path) -> None:
    r = _make_record(status=sr.Status.LAUNCHING, exit_code=None)
    sr.write_record(r, root=tmp_path)

    sr.update_status(r.session_slug, root=tmp_path, status=sr.Status.EXITED, exit_code=3)

    reread = sr.read_record(r.session_slug, root=tmp_path)
    assert reread.status == sr.Status.EXITED
    assert reread.exit_code == 3
    # Every other field survives the read-modify-write untouched.
    assert reread.session_slug == r.session_slug
    assert reread.title == r.title


def test_refresh_record_updates_existing_record_under_same_key(tmp_path: Path) -> None:
    r = _make_record(status=sr.Status.LAUNCHING)
    sr.write_record(r, root=tmp_path)

    sr.refresh_record(r.session_slug, root=tmp_path, status=sr.Status.RUNNING)

    path_before = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    reread = sr.read_record(r.session_slug, root=tmp_path)
    assert reread.status == sr.Status.RUNNING
    assert sr.record_path_for_slug(r.session_slug, root=tmp_path) == path_before


# ---------------------------------------------------------------------------
# Step-5: decision helpers (B2) -- Fleet Cockpit
# ---------------------------------------------------------------------------


def test_write_then_list_decision(tmp_path: Path) -> None:
    rec = _make_decision()

    assert sr.write_decision(rec, root=tmp_path) is True

    listed = sr.list_decisions(root=tmp_path)
    assert len(listed) == 1
    assert listed[0] == rec


def test_update_decision_state_persists(tmp_path: Path) -> None:
    rec = _make_decision(state=sr.DecisionState.OPEN)
    sr.write_decision(rec, root=tmp_path)

    updated = sr.update_decision_state(rec.id, sr.DecisionState.ANSWERED, root=tmp_path)

    assert updated is not None
    assert updated.state == sr.DecisionState.ANSWERED
    [reread] = [d for d in sr.list_decisions(root=tmp_path) if d.id == rec.id]
    assert reread.state == sr.DecisionState.ANSWERED


def test_set_manual_boost_persists(tmp_path: Path) -> None:
    rec = _make_decision(manual_boost=0)
    sr.write_decision(rec, root=tmp_path)

    updated = sr.set_manual_boost(rec.id, 3, root=tmp_path)

    assert updated is not None
    assert updated.manual_boost == 3
    [reread] = [d for d in sr.list_decisions(root=tmp_path) if d.id == rec.id]
    assert reread.manual_boost == 3


def test_decisions_are_per_file_isolated(tmp_path: Path) -> None:
    """Distinct <id>.json paths are the whole isolation guarantee: writing or
    updating one decision must never touch another's file (no global index,
    no shared body -- mirrors each session record's own directory).
    """
    rec_a = _make_decision(id='dec-a')
    rec_b = _make_decision(id='dec-b')
    sr.write_decision(rec_a, root=tmp_path)
    sr.write_decision(rec_b, root=tmp_path)

    path_a = sr.decision_path_for_id('dec-a', root=tmp_path)
    path_b = sr.decision_path_for_id('dec-b', root=tmp_path)
    assert path_a.is_file()
    assert path_b.is_file()
    assert path_a != path_b
    assert {d.id for d in sr.list_decisions(root=tmp_path)} == {'dec-a', 'dec-b'}

    bytes_b_before = path_b.read_bytes()
    sr.set_manual_boost('dec-a', 9, root=tmp_path)
    assert path_b.read_bytes() == bytes_b_before


def test_list_decisions_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert sr.list_decisions(root=tmp_path) == []


def test_list_decisions_skips_corrupt_file(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rec = _make_decision()
    sr.write_decision(rec, root=tmp_path)
    corrupt_path = sr.decisions_dir(root=tmp_path) / 'bogus.json'
    corrupt_path.write_text('{not valid json')

    with caplog.at_level(logging.ERROR):
        listed = sr.list_decisions(root=tmp_path)

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
        result = sr.write_decision(_make_decision(), root=bad_root)

    assert result is False
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert not (blocker / 'fleet').exists()


def test_update_and_set_boost_fail_soft_when_absent(tmp_path: Path) -> None:
    assert sr.update_decision_state('no-such-id', sr.DecisionState.ANSWERED, root=tmp_path) is None
    assert sr.set_manual_boost('no-such-id', 5, root=tmp_path) is None


def test_update_and_set_boost_fail_soft_when_lock_acquisition_raises(
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
    """
    rec = _make_decision(id='dec-lockfault', state=sr.DecisionState.OPEN, manual_boost=0)
    sr.write_decision(rec, root=tmp_path)

    @contextlib.contextmanager
    def raising_lock(decision_id: str, root: Path | str | None = None):
        raise OSError('simulated lock-acquisition fault')
        yield  # pragma: no cover -- unreachable, satisfies generator-function shape

    monkeypatch.setattr(sr, 'decision_id_lock', raising_lock)

    with caplog.at_level(logging.ERROR):
        state_result = sr.update_decision_state('dec-lockfault', sr.DecisionState.ANSWERED, root=tmp_path)
        boost_result = sr.set_manual_boost('dec-lockfault', 5, root=tmp_path)

    assert state_result is None
    assert boost_result is None
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


# ---------------------------------------------------------------------------
# Step-6: decision_id_lock per-decision-id sidecar flock (task 2427)
# ---------------------------------------------------------------------------


class TestDecisionIdLock:
    """Unit tests for the sr.decision_id_lock(decision_id, root=...) context manager.

    Mirrors TestEscalationIdLock (escalation/tests/test_queue.py:2553) -- same
    stable-sidecar-flock contract (task 1609), retargeted to decisions_dir /
    decision_path_for_id and the ``<id>.json.lock`` sidecar path.
    """

    def test_importable_from_session_registry(self) -> None:
        """(a) decision_id_lock is importable from orchestrator.session_registry."""
        from orchestrator.session_registry import decision_id_lock  # noqa: F401

    def test_entering_context_creates_sidecar_file(self, tmp_path: Path) -> None:
        """(b) Entering the context creates the sidecar .lock file at the expected path."""
        lock_path = sr.decisions_dir(root=tmp_path) / 'dec-1.json.lock'

        assert not lock_path.exists(), 'Sidecar must not exist before context entry'
        with sr.decision_id_lock('dec-1', root=tmp_path):
            assert lock_path.exists(), 'Sidecar must exist while context is held'
        # File persists after release -- stable inode, never renamed/replaced.
        assert lock_path.exists(), 'Sidecar must persist after context exit (stable inode)'

    def test_held_lock_blocks_second_acquire_nonblocking(self, tmp_path: Path) -> None:
        """(c) While the context is held, a non-blocking flock on the same sidecar raises BlockingIOError."""
        lock_path = sr.decisions_dir(root=tmp_path) / 'dec-1.json.lock'

        with sr.decision_id_lock('dec-1', root=tmp_path):
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
        rec = _make_decision(id='dec-1')
        sr.write_decision(rec, root=tmp_path)

        with sr.decision_id_lock('dec-1', root=tmp_path):
            listed = sr.list_decisions(root=tmp_path)

        assert len(listed) == 1
        assert listed[0] == rec


# ---------------------------------------------------------------------------
# Step-6b: update_decision_state / set_manual_boost serialize their RMW via
# decision_id_lock (task 2427)
# ---------------------------------------------------------------------------


class TestDecisionHelpersAdoptLock:
    """Spy tests (mirror TestSubmitResolveAdoptLock, test_queue.py:2913):
    update_decision_state and set_manual_boost must each acquire
    decision_id_lock for the correct decision id.
    """

    def test_update_decision_state_acquires_lock_for_decision_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rec = _make_decision(id='dec-spy-1', state=sr.DecisionState.OPEN)
        sr.write_decision(rec, root=tmp_path)

        real_lock = sr.decision_id_lock
        acquired: list[str] = []

        @contextlib.contextmanager
        def recording_lock(decision_id: str, root: Path | str | None = None):
            acquired.append(decision_id)
            with real_lock(decision_id, root=root):
                yield

        monkeypatch.setattr(sr, 'decision_id_lock', recording_lock)

        sr.update_decision_state('dec-spy-1', sr.DecisionState.ANSWERED, root=tmp_path)

        assert 'dec-spy-1' in acquired, f'Expected lock acquisition for dec-spy-1; got {acquired}'

    def test_set_manual_boost_acquires_lock_for_decision_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rec = _make_decision(id='dec-spy-2', manual_boost=0)
        sr.write_decision(rec, root=tmp_path)

        real_lock = sr.decision_id_lock
        acquired: list[str] = []

        @contextlib.contextmanager
        def recording_lock(decision_id: str, root: Path | str | None = None):
            acquired.append(decision_id)
            with real_lock(decision_id, root=root):
                yield

        monkeypatch.setattr(sr, 'decision_id_lock', recording_lock)

        sr.set_manual_boost('dec-spy-2', 5, root=tmp_path)

        assert 'dec-spy-2' in acquired, f'Expected lock acquisition for dec-spy-2; got {acquired}'


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
    rec = _make_decision(id='dec-race', state=sr.DecisionState.OPEN, manual_boost=0)
    sr.write_decision(rec, root=tmp_path)

    real_write_decision = sr.write_decision

    def delayed_write_decision(record: sr.DecisionRecord, root: Path | str | None = None) -> bool:
        time.sleep(0.3)
        return real_write_decision(record, root=root)

    monkeypatch.setattr(sr, 'write_decision', delayed_write_decision)

    t1 = threading.Thread(
        target=sr.update_decision_state,
        args=('dec-race', sr.DecisionState.ANSWERED, tmp_path),
    )
    t2 = threading.Thread(target=sr.set_manual_boost, args=('dec-race', 7, tmp_path))

    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert not t1.is_alive(), 'update_decision_state thread did not finish in time'
    assert not t2.is_alive(), 'set_manual_boost thread did not finish in time'

    [reread] = [d for d in sr.list_decisions(root=tmp_path) if d.id == 'dec-race']
    assert reread.state == sr.DecisionState.ANSWERED
    assert reread.manual_boost == 7


# ---------------------------------------------------------------------------
# Step-7: TTL / pid stale-record reaper matrix
# ---------------------------------------------------------------------------


def test_ttl_constants_are_module_level_timedeltas() -> None:
    assert timedelta(hours=24) == sr.TERMINAL_TTL
    assert timedelta(hours=1) == sr.NON_TERMINAL_HEARTBEAT_TTL


def test_reap_removes_terminal_record_past_ttl(tmp_path: Path) -> None:
    r = _make_record(status=sr.Status.EXITED, launcher_pid=os.getpid())
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    _set_mtime(record_path, _NOW, sr.TERMINAL_TTL + timedelta(hours=1))

    reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

    assert len(reaped) == 1
    assert reaped[0].session_slug == r.session_slug
    assert reaped[0].reason == 'terminal_ttl'
    assert reaped[0].path == record_path.parent
    assert not record_path.parent.exists()


def test_reap_keeps_terminal_record_within_ttl(tmp_path: Path) -> None:
    r = _make_record(status=sr.Status.EXITED, launcher_pid=os.getpid())
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    _set_mtime(record_path, _NOW, sr.TERMINAL_TTL - timedelta(hours=1))

    reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

    assert reaped == []
    assert record_path.is_file()


def test_reap_removes_non_terminal_dead_pid_past_heartbeat_ttl(tmp_path: Path) -> None:
    r = _make_record(status=sr.Status.LAUNCHING, launcher_pid=_DEAD_PID)
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    _set_mtime(record_path, _NOW, sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1))

    reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

    assert len(reaped) == 1
    assert reaped[0].session_slug == r.session_slug
    assert reaped[0].reason == 'stale_pid'
    assert not record_path.parent.exists()


def test_reap_keeps_non_terminal_live_pid_regardless_of_age(tmp_path: Path) -> None:
    r = _make_record(status=sr.Status.RUNNING, launcher_pid=os.getpid())
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    _set_mtime(record_path, _NOW, timedelta(days=30))  # past both TTLs

    reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

    assert reaped == []
    assert record_path.is_file()


def test_reap_keeps_non_terminal_dead_pid_within_heartbeat_ttl(tmp_path: Path) -> None:
    r = _make_record(status=sr.Status.LAUNCHING, launcher_pid=_DEAD_PID)
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    _set_mtime(record_path, _NOW, sr.NON_TERMINAL_HEARTBEAT_TTL - timedelta(minutes=1))

    reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

    assert reaped == []
    assert record_path.is_file()


def test_reap_removes_corrupt_body_past_heartbeat_ttl(tmp_path: Path) -> None:
    r = _make_record(status=sr.Status.LAUNCHING, launcher_pid=os.getpid())
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    record_path.write_text('{not valid json')
    _set_mtime(record_path, _NOW, sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1))

    reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

    assert len(reaped) == 1
    assert reaped[0].session_slug == r.session_slug
    assert reaped[0].reason == 'corrupt'
    assert not record_path.parent.exists()


def test_reap_removes_missing_body_past_heartbeat_ttl(tmp_path: Path) -> None:
    """A slug dir with no record.json at all is still reaped by path identity."""
    r = _make_record(status=sr.Status.LAUNCHING, launcher_pid=os.getpid())
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    record_path.unlink()
    _set_mtime(record_path.parent, _NOW, sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1))

    reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

    assert len(reaped) == 1
    assert reaped[0].session_slug == r.session_slug
    assert reaped[0].reason == 'corrupt'
    assert not record_path.parent.exists()


def test_reap_handles_mixed_population_in_one_sweep(tmp_path: Path) -> None:
    """The reaper's whole point is sweeping many records in one pass; pin that
    a kept/terminal_ttl/stale_pid/corrupt mix in a single sessions dir reaps
    exactly the expected subset in one call and leaves survivors untouched.
    """
    kept_running = _make_record(
        session_slug='kept-running', status=sr.Status.RUNNING, launcher_pid=os.getpid()
    )
    kept_recent_terminal = _make_record(
        session_slug='kept-recent-terminal', status=sr.Status.EXITED, launcher_pid=os.getpid()
    )
    reap_terminal = _make_record(
        session_slug='reap-terminal', status=sr.Status.EXITED, launcher_pid=os.getpid()
    )
    reap_stale = _make_record(
        session_slug='reap-stale-pid', status=sr.Status.LAUNCHING, launcher_pid=_DEAD_PID
    )

    for r in (kept_running, kept_recent_terminal, reap_terminal, reap_stale):
        sr.write_record(r, root=tmp_path)

    _set_mtime(
        sr.record_path_for_slug(kept_running.session_slug, root=tmp_path),
        _NOW,
        timedelta(days=30),  # long-lived but a live pid -> kept regardless of age
    )
    _set_mtime(
        sr.record_path_for_slug(kept_recent_terminal.session_slug, root=tmp_path),
        _NOW,
        sr.TERMINAL_TTL - timedelta(hours=1),
    )
    _set_mtime(
        sr.record_path_for_slug(reap_terminal.session_slug, root=tmp_path),
        _NOW,
        sr.TERMINAL_TTL + timedelta(hours=1),
    )
    _set_mtime(
        sr.record_path_for_slug(reap_stale.session_slug, root=tmp_path),
        _NOW,
        sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1),
    )

    # A corrupt-body record dir, aged past the heartbeat TTL.
    corrupt_dir = sr.sessions_dir(root=tmp_path) / 'reap-corrupt'
    corrupt_dir.mkdir(parents=True)
    corrupt_record = corrupt_dir / 'record.json'
    corrupt_record.write_text('{not valid json')
    _set_mtime(corrupt_record, _NOW, sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1))

    reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

    reaped_by_slug = {r.session_slug: r.reason for r in reaped}
    assert reaped_by_slug == {
        'reap-terminal': 'terminal_ttl',
        'reap-stale-pid': 'stale_pid',
        'reap-corrupt': 'corrupt',
    }

    remaining = {p.name for p in sr.sessions_dir(root=tmp_path).iterdir()}
    assert remaining == {'kept-running', 'kept-recent-terminal'}


def test_reap_continues_sweep_when_one_directory_fails_to_remove(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A single unreapable directory (permission error, a file held open, an
    ENOTEMPTY race with a concurrent writer) must not abort the sweep -- a
    later stale directory in the same sessions dir is still reaped.
    """
    poison = _make_record(session_slug='poison', status=sr.Status.EXITED, launcher_pid=os.getpid())
    victim = _make_record(session_slug='victim', status=sr.Status.EXITED, launcher_pid=os.getpid())
    for r in (poison, victim):
        sr.write_record(r, root=tmp_path)
        _set_mtime(
            sr.record_path_for_slug(r.session_slug, root=tmp_path),
            _NOW,
            sr.TERMINAL_TTL + timedelta(hours=1),
        )

    poison_dir = sr.record_path_for_slug(poison.session_slug, root=tmp_path).parent
    real_rmtree = sr.shutil.rmtree

    def _flaky_rmtree(path: str | os.PathLike[str], *args: Any, **kwargs: Any) -> None:
        if Path(path) == poison_dir:
            raise OSError('simulated ENOTEMPTY race')
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(sr.shutil, 'rmtree', _flaky_rmtree)

    with caplog.at_level(logging.ERROR):
        reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

    assert {r.session_slug for r in reaped} == {'victim'}
    assert poison_dir.is_dir()  # left behind, not silently claimed as reaped
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


# ---------------------------------------------------------------------------
# Task 2511 step-5: mark_orphaned_sessions_exited (liveness sweep, Part 2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'status',
    [sr.Status.LAUNCHING, sr.Status.RUNNING, sr.Status.IDLE, sr.Status.AWAITING_INPUT],
)
def test_mark_orphaned_marks_non_terminal_dead_pid_past_heartbeat_ttl(
    status: sr.Status, tmp_path: Path
) -> None:
    r = _make_record(session_slug='orphan-1', status=status, launcher_pid=_DEAD_PID, exit_code=None)
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    _set_mtime(record_path, _NOW, sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1))

    marked = sr.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW)

    assert {m.session_slug for m in marked} == {'orphan-1'}
    assert record_path.parent.is_dir()  # marked, NOT deleted (reap_stale_records' job)
    assert record_path.is_file()
    reloaded = sr.read_record('orphan-1', root=tmp_path)
    assert reloaded.status == sr.Status.EXITED
    assert reloaded.exit_code == sr.ORPHAN_EXIT_CODE


def test_mark_orphaned_keeps_non_terminal_live_pid_regardless_of_age(tmp_path: Path) -> None:
    r = _make_record(session_slug='kept-live', status=sr.Status.RUNNING, launcher_pid=os.getpid())
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    _set_mtime(record_path, _NOW, timedelta(days=30))  # past both TTLs

    marked = sr.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW)

    assert marked == []
    assert sr.read_record('kept-live', root=tmp_path).status == sr.Status.RUNNING


def test_mark_orphaned_keeps_non_terminal_dead_pid_within_heartbeat_ttl(tmp_path: Path) -> None:
    r = _make_record(session_slug='kept-recent', status=sr.Status.LAUNCHING, launcher_pid=_DEAD_PID)
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    _set_mtime(record_path, _NOW, sr.NON_TERMINAL_HEARTBEAT_TTL - timedelta(minutes=1))

    marked = sr.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW)

    assert marked == []
    assert sr.read_record('kept-recent', root=tmp_path).status == sr.Status.LAUNCHING


def test_mark_orphaned_leaves_already_terminal_record_untouched(tmp_path: Path) -> None:
    r = _make_record(
        session_slug='already-exited', status=sr.Status.EXITED, exit_code=0, launcher_pid=_DEAD_PID
    )
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    _set_mtime(record_path, _NOW, sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(days=2))

    marked = sr.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW)

    assert marked == []
    reloaded = sr.read_record('already-exited', root=tmp_path)
    assert reloaded.status == sr.Status.EXITED
    assert reloaded.exit_code == 0  # NOT re-stamped with ORPHAN_EXIT_CODE


def test_mark_orphaned_preserves_all_other_fields(tmp_path: Path) -> None:
    r = _make_record(
        session_slug='orphan-fields',
        status=sr.Status.RUNNING,
        launcher_pid=_DEAD_PID,
        role='unblock',
        project='df',
        task_id='2085',
        result_file='/tmp/whatever/result.md',
        start_ts='2026-07-07T00:00:00+00:00',
    )
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    _set_mtime(record_path, _NOW, sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1))

    sr.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW)

    reloaded = sr.read_record('orphan-fields', root=tmp_path)
    assert reloaded.status == sr.Status.EXITED
    assert reloaded.exit_code == sr.ORPHAN_EXIT_CODE
    assert reloaded.role == 'unblock'
    assert reloaded.project == 'df'
    assert reloaded.task_id == '2085'
    assert reloaded.result_file == '/tmp/whatever/result.md'
    assert reloaded.launcher_pid == _DEAD_PID
    assert reloaded.start_ts == '2026-07-07T00:00:00+00:00'


def test_mark_orphaned_handles_mixed_population_in_one_sweep(tmp_path: Path) -> None:
    marked_orphan = _make_record(
        session_slug='will-be-marked', status=sr.Status.RUNNING, launcher_pid=_DEAD_PID
    )
    kept_live = _make_record(
        session_slug='kept-live-mix', status=sr.Status.RUNNING, launcher_pid=os.getpid()
    )
    kept_recent = _make_record(
        session_slug='kept-recent-mix', status=sr.Status.LAUNCHING, launcher_pid=_DEAD_PID
    )
    kept_terminal = _make_record(
        session_slug='kept-terminal-mix',
        status=sr.Status.EXITED,
        exit_code=0,
        launcher_pid=_DEAD_PID,
    )
    for r in (marked_orphan, kept_live, kept_recent, kept_terminal):
        sr.write_record(r, root=tmp_path)

    _set_mtime(
        sr.record_path_for_slug('will-be-marked', root=tmp_path),
        _NOW,
        sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1),
    )
    _set_mtime(
        sr.record_path_for_slug('kept-live-mix', root=tmp_path),
        _NOW,
        timedelta(days=30),
    )
    _set_mtime(
        sr.record_path_for_slug('kept-recent-mix', root=tmp_path),
        _NOW,
        sr.NON_TERMINAL_HEARTBEAT_TTL - timedelta(minutes=1),
    )
    _set_mtime(
        sr.record_path_for_slug('kept-terminal-mix', root=tmp_path),
        _NOW,
        sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(days=2),
    )

    marked = sr.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW)

    assert {m.session_slug for m in marked} == {'will-be-marked'}
    assert sr.read_record('will-be-marked', root=tmp_path).status == sr.Status.EXITED
    assert sr.read_record('kept-live-mix', root=tmp_path).status == sr.Status.RUNNING
    assert sr.read_record('kept-recent-mix', root=tmp_path).status == sr.Status.LAUNCHING
    reloaded_terminal = sr.read_record('kept-terminal-mix', root=tmp_path)
    assert reloaded_terminal.status == sr.Status.EXITED
    assert reloaded_terminal.exit_code == 0


# ---------------------------------------------------------------------------
# Step-9: CLI + fail-soft
# ---------------------------------------------------------------------------


def _launching_env(
    tmp_path: Path,
    *,
    role: str = 'unblock',
    project: str = 'df',
    task_id: str = '2085',
    escalation_id: str = 'esc-9',
    title: str = 'unblock:df#2085 routing-mechanism',
    prompt: str = '/unblock 2085',
    cwd: str = '/home/leo/src/dark-factory',
    launcher_pid: str = '4242',
) -> dict[str, str]:
    """Build the CLAUDE_SPAWN_* env mapping spawn-claude.sh passes to `launching`."""
    return {
        'CLAUDE_FLEET_ROOT': str(tmp_path),
        'CLAUDE_SPAWN_ROLE': role,
        'CLAUDE_SPAWN_PROJECT': project,
        'CLAUDE_SPAWN_TASK_ID': task_id,
        'CLAUDE_SPAWN_ESCALATION_ID': escalation_id,
        'CLAUDE_SPAWN_TITLE': title,
        'CLAUDE_SPAWN_PROMPT': prompt,
        'CLAUDE_SPAWN_CWD': cwd,
        'CLAUDE_SPAWN_LAUNCHER_PID': launcher_pid,
    }


def _set_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def test_main_launching_writes_record_and_prints_only_record_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_env(monkeypatch, _launching_env(tmp_path))

    rc = sr.main(['launching'])

    assert rc == 0
    slug = sr.build_session_slug('unblock', 'df', '2085', 4242)
    expected_dir = sr.record_path_for_slug(slug, root=tmp_path).parent
    captured = capsys.readouterr()
    assert captured.out.strip() == str(expected_dir)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.LAUNCHING
    assert record.role == 'unblock'
    assert record.project == 'df'
    assert record.task_id == '2085'
    assert record.escalation_id == 'esc-9'
    assert record.title == 'unblock:df#2085 routing-mechanism'
    assert record.prompt == '/unblock 2085'
    assert record.cwd == '/home/leo/src/dark-factory'
    assert record.launcher_pid == 4242


def test_main_exit_sets_status_exited_and_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_env(monkeypatch, _launching_env(tmp_path))
    sr.main(['launching'])
    record_dir = capsys.readouterr().out.strip()

    rc = sr.main(['exit', '--record', record_dir, '--code', '3'])

    assert rc == 0
    slug = sr.build_session_slug('unblock', 'df', '2085', 4242)
    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.EXITED
    assert record.exit_code == 3


def test_main_launching_populates_result_file_and_exit_preserves_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Attention Rail T5: `launching` must allocate a deterministic result.md
    path inside the record dir, and the `exit` read-modify-write must preserve
    it through to the EXITED record (result_file is set once, at launch, and
    never touched again).
    """
    _set_env(monkeypatch, _launching_env(tmp_path))

    sr.main(['launching'])
    record_dir = capsys.readouterr().out.strip()

    slug = sr.build_session_slug('unblock', 'df', '2085', 4242)
    expected_result_file = str(sr.record_path_for_slug(slug, root=tmp_path).parent / 'result.md')

    launching_record = sr.read_record(slug, root=tmp_path)
    assert launching_record.result_file is not None
    assert launching_record.result_file == expected_result_file

    rc = sr.main(['exit', '--record', record_dir, '--code', '3'])
    assert rc == 0

    exited_record = sr.read_record(slug, root=tmp_path)
    assert exited_record.result_file == expected_result_file


def test_main_refresh_sets_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_env(monkeypatch, _launching_env(tmp_path))
    sr.main(['launching'])
    record_dir = capsys.readouterr().out.strip()

    rc = sr.main(['refresh', '--record', record_dir, '--status', 'running'])

    assert rc == 0
    slug = sr.build_session_slug('unblock', 'df', '2085', 4242)
    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.RUNNING


def test_main_set_display_stamps_and_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fleet Cockpit C6: the `set-display` verb stamps display post-hoc (the
    tmux target is only known after `tmux new-window` runs, well after
    `launching` allocated the record), and a later `exit` read-modify-write
    must preserve it (both re-read the whole record and re-write it).
    """
    _set_env(monkeypatch, _launching_env(tmp_path))
    sr.main(['launching'])
    record_dir = capsys.readouterr().out.strip()

    rc = sr.main(
        [
            'set-display',
            '--record',
            record_dir,
            '--kind',
            'tmux',
            '--tmux-target',
            'fleet-df:2',
            '--wm-title',
            'watcher:df#2085',
        ]
    )

    assert rc == 0
    slug = sr.build_session_slug('unblock', 'df', '2085', 4242)
    record = sr.read_record(slug, root=tmp_path)
    assert record.display is not None
    assert record.display.kind == 'tmux'
    assert record.display.tmux_target == 'fleet-df:2'
    assert record.display.wm_title == 'watcher:df#2085'
    assert record.status == sr.Status.LAUNCHING
    assert record.session_slug == slug

    rc = sr.main(['exit', '--record', record_dir, '--code', '0'])

    assert rc == 0
    exited_record = sr.read_record(slug, root=tmp_path)
    assert exited_record.status == sr.Status.EXITED
    assert exited_record.display is not None
    assert exited_record.display.kind == 'tmux'
    assert exited_record.display.tmux_target == 'fleet-df:2'


# --- identity fallback (parse_spawn_identity) -------------------------------


def test_parse_spawn_identity_env_first_overrides_conflicting_title() -> None:
    env = {
        'CLAUDE_SPAWN_ROLE': 'review',
        'CLAUDE_SPAWN_PROJECT': 'other',
        'CLAUDE_SPAWN_TASK_ID': '999',
        'CLAUDE_SPAWN_ESCALATION_ID': 'esc-5',
    }
    identity = sr.parse_spawn_identity(
        env=env,
        title='unblock:df#2085 routing-mechanism',
        prompt='',
        cwd='/x',
    )
    assert identity.role == 'review'
    assert identity.project == 'other'
    assert identity.task_id == '999'
    assert identity.escalation_id == 'esc-5'


def test_parse_spawn_identity_falls_back_to_task_scoped_title() -> None:
    identity = sr.parse_spawn_identity(
        env={},
        title='unblock:df#2085 routing-mechanism',
        prompt='',
        cwd='/x',
    )
    assert identity.role == 'unblock'
    assert identity.project == 'df'
    assert identity.task_id == '2085'


def test_parse_spawn_identity_falls_back_to_project_level_title_with_no_hash() -> None:
    identity = sr.parse_spawn_identity(
        env={},
        title='prd:df attention-rail',
        prompt='',
        cwd='/x',
    )
    assert identity.role == 'prd'
    assert identity.project == 'df'
    assert identity.task_id is None


def test_parse_spawn_identity_defaults_when_env_and_title_absent() -> None:
    identity = sr.parse_spawn_identity(
        env={},
        title='',
        prompt='',
        cwd='/home/leo/src/dark-factory',
    )
    assert identity.role == 'session'
    assert identity.project == 'dark-factory'
    assert identity.task_id is None
    assert identity.escalation_id is None


def test_parse_spawn_identity_defaults_project_unknown_when_cwd_has_no_basename() -> None:
    identity = sr.parse_spawn_identity(env={}, title='', prompt='', cwd='')
    assert identity.role == 'session'
    assert identity.project == 'unknown'


def test_parse_spawn_identity_defaults_project_strips_trailing_slash_from_cwd() -> None:
    # CLAUDE_SPAWN_CWD comes from a bash positional arg and is not guaranteed
    # normalized; a trailing '/' must not make basename() degrade to '' (and
    # thus the project silently fall through to 'unknown').
    identity = sr.parse_spawn_identity(
        env={},
        title='',
        prompt='',
        cwd='/home/leo/src/dark-factory/',
    )
    assert identity.role == 'session'
    assert identity.project == 'dark-factory'


# --- fail-soft ---------------------------------------------------------------


def test_main_launching_fail_soft_when_write_record_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _set_env(monkeypatch, _launching_env(tmp_path))

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError('disk on fire')

    monkeypatch.setattr(sr, 'write_record', _boom)

    with caplog.at_level(logging.ERROR):
        rc = sr.main(['launching'])

    assert rc == 0
    assert capsys.readouterr().out.strip() == ''
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_main_launching_fail_soft_when_fleet_root_under_a_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A CLAUDE_FLEET_ROOT rooted under a pre-created regular file makes the
    # write's mkdir(parents=True) raise NotADirectoryError deterministically.
    blocker = tmp_path / 'blocker'
    blocker.write_text('not a directory')
    env = _launching_env(tmp_path)
    env['CLAUDE_FLEET_ROOT'] = str(blocker / 'fleet')
    _set_env(monkeypatch, env)

    with caplog.at_level(logging.ERROR):
        rc = sr.main(['launching'])

    assert rc == 0
    assert capsys.readouterr().out.strip() == ''
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert not (blocker / 'fleet').exists()


def test_main_launching_fail_soft_when_launcher_pid_not_numeric(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # CLAUDE_SPAWN_LAUNCHER_PID is spawn-claude.sh's own "$$" and should
    # always be numeric, but a malformed/corrupted env value must fail soft
    # like any other registry fault -- int() raising ValueError inside
    # _run_launching must not escape main(), must not print a bogus record
    # dir, and must leave no record behind.
    env = _launching_env(tmp_path)
    env['CLAUDE_SPAWN_LAUNCHER_PID'] = 'not-a-pid'
    _set_env(monkeypatch, env)

    with caplog.at_level(logging.ERROR):
        rc = sr.main(['launching'])

    assert rc == 0
    assert capsys.readouterr().out.strip() == ''
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert not sr.sessions_dir(root=tmp_path).exists()


# ---------------------------------------------------------------------------
# Task 2511 step-7: driving the liveness sweep from `launching`/`reap`
# ---------------------------------------------------------------------------


def test_main_launching_drains_prior_orphan_via_sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A stale orphan O (non-terminal, dead pid, aged past the heartbeat TTL)
    # pre-exists in the same sessions dir BEFORE this spawn's own `launching`
    # write -- every spawn should opportunistically drain prior orphans.
    orphan = _make_record(session_slug='orphan-o', status=sr.Status.RUNNING, launcher_pid=_DEAD_PID)
    sr.write_record(orphan, root=tmp_path)
    _set_mtime(
        sr.record_path_for_slug('orphan-o', root=tmp_path),
        datetime.now(UTC),
        sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1),
    )

    _set_env(monkeypatch, _launching_env(tmp_path))

    rc = sr.main(['launching'])

    assert rc == 0
    slug = sr.build_session_slug('unblock', 'df', '2085', 4242)
    expected_dir = sr.record_path_for_slug(slug, root=tmp_path).parent
    # (i) the NEW launching record's dir is printed unchanged.
    assert capsys.readouterr().out.strip() == str(expected_dir)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.LAUNCHING

    # (ii) the pre-existing orphan is now marked exited.
    orphan_reloaded = sr.read_record('orphan-o', root=tmp_path)
    assert orphan_reloaded.status == sr.Status.EXITED
    assert orphan_reloaded.exit_code == sr.ORPHAN_EXIT_CODE


def test_main_launching_fail_soft_when_sweep_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_env(monkeypatch, _launching_env(tmp_path))

    calls: list[None] = []

    def _boom(*_args: object, **_kwargs: object) -> list[sr.SessionRecord]:
        calls.append(None)
        raise OSError('sweep on fire')

    monkeypatch.setattr(sr, 'mark_orphaned_sessions_exited', _boom)

    rc = sr.main(['launching'])

    assert rc == 0
    assert calls  # the sweep really was invoked (and really did raise)
    slug = sr.build_session_slug('unblock', 'df', '2085', 4242)
    expected_dir = sr.record_path_for_slug(slug, root=tmp_path).parent
    # The sweep fault is swallowed INSIDE _run_launching -- the printed dir
    # (what spawn-claude.sh captures into SESSION_RECORD_DIR) stays exactly
    # the new record's dir, never corrupted or suppressed by the fault.
    assert capsys.readouterr().out.strip() == str(expected_dir)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.LAUNCHING


def test_main_reap_marks_then_deletes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    # A fresh stale orphan: non-terminal, dead pid, past the heartbeat TTL --
    # this pass must MARK it exited (its mtime is bumped by the mark, so it
    # is not also independently terminal_ttl-stale in this SAME pass).
    fresh_orphan = _make_record(
        session_slug='fresh-orphan', status=sr.Status.RUNNING, launcher_pid=_DEAD_PID
    )
    sr.write_record(fresh_orphan, root=tmp_path)
    _set_mtime(
        sr.record_path_for_slug('fresh-orphan', root=tmp_path),
        datetime.now(UTC),
        sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1),
    )

    # A separately-aged terminal record, well past TERMINAL_TTL -- this pass
    # must DELETE it (reason='terminal_ttl'), unchanged from today.
    old_terminal = _make_record(
        session_slug='old-terminal', status=sr.Status.EXITED, exit_code=0, launcher_pid=os.getpid()
    )
    sr.write_record(old_terminal, root=tmp_path)
    _set_mtime(
        sr.record_path_for_slug('old-terminal', root=tmp_path),
        datetime.now(UTC),
        sr.TERMINAL_TTL + timedelta(hours=1),
    )

    rc = sr.main(['reap'])

    assert rc == 0
    fresh_dir = sr.record_path_for_slug('fresh-orphan', root=tmp_path).parent
    assert fresh_dir.is_dir()  # marked, NOT deleted, this pass
    fresh_reloaded = sr.read_record('fresh-orphan', root=tmp_path)
    assert fresh_reloaded.status == sr.Status.EXITED
    assert fresh_reloaded.exit_code == sr.ORPHAN_EXIT_CODE

    assert not sr.record_path_for_slug('old-terminal', root=tmp_path).parent.exists()


# ---------------------------------------------------------------------------
# Step-10: role leases (Attention Rail T7)
# ---------------------------------------------------------------------------

# --- paths / names / contract types -----------------------------------------


def test_leases_dir_is_fleet_root_slash_leases(tmp_path: Path) -> None:
    assert sr.leases_dir(root=tmp_path) == tmp_path / 'leases'


def test_lease_path_for_name_joins_leases_dir_and_lease_suffix(tmp_path: Path) -> None:
    path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    assert path == tmp_path / 'leases' / 'watcher-df.lease'


def test_lease_path_for_name_preserves_hash_for_task_scoped_names(tmp_path: Path) -> None:
    path = sr.lease_path_for_name('unblock-df#2085', root=tmp_path)
    assert path == tmp_path / 'leases' / 'unblock-df#2085.lease'


def test_lease_path_for_name_sanitizes_path_separators(tmp_path: Path) -> None:
    # A name containing '/' must not escape leases_dir: it still resolves to
    # a single file directly inside leases_dir (no nested directory, no
    # traversal), unlike _SLUG_SANITIZE_RE this sanitizer PRESERVES '#'.
    path = sr.lease_path_for_name('../../etc/passwd', root=tmp_path)
    assert path.parent == sr.leases_dir(root=tmp_path)
    assert '/' not in path.name


def test_build_lease_name_watcher() -> None:
    assert sr.build_lease_name('watcher', 'df') == 'watcher-df'


def test_build_lease_name_recon_watcher() -> None:
    assert sr.build_lease_name('recon-watcher', 'df') == 'recon-watcher-df'


def test_build_lease_name_unblock_is_task_scoped_with_hash() -> None:
    assert sr.build_lease_name('unblock', 'df', '2085') == 'unblock-df#2085'


def test_lease_policy_has_exactly_stand_down_and_warn_and_proceed() -> None:
    values = {member.value for member in sr.LeasePolicy}
    assert values == {'stand-down', 'warn-and-proceed'}
    assert sr.LeasePolicy.STAND_DOWN.value == 'stand-down'
    assert sr.LeasePolicy.WARN_AND_PROCEED.value == 'warn-and-proceed'


def test_lease_decision_has_exactly_acquired_stand_down_proceed() -> None:
    values = {member.value for member in sr.LeaseDecision}
    assert values == {'acquired', 'stand-down', 'proceed'}
    assert sr.LeaseDecision.ACQUIRED.value == 'acquired'
    assert sr.LeaseDecision.STAND_DOWN.value == 'stand-down'
    assert sr.LeaseDecision.PROCEED.value == 'proceed'


def test_lease_heartbeat_ttl_is_a_timedelta() -> None:
    assert isinstance(sr.LEASE_HEARTBEAT_TTL, timedelta)


def test_lease_holder_json_round_trip_is_lossless() -> None:
    holder = sr.LeaseHolder(
        session_slug='watcher-df-100', pid=4242, start_ts='2026-07-07T12:00:00+00:00'
    )
    assert sr.LeaseHolder.from_json(holder.to_json()) == holder


def test_lease_holder_dict_round_trip_is_lossless() -> None:
    holder = sr.LeaseHolder(
        session_slug='watcher-df-100', pid=4242, start_ts='2026-07-07T12:00:00+00:00'
    )
    assert sr.LeaseHolder.from_dict(holder.to_dict()) == holder


# --- claim_lease: free lease --------------------------------------------


def test_claim_lease_acquires_a_free_lease(tmp_path: Path) -> None:
    holder = sr.LeaseHolder(session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat())

    claim = sr.claim_lease('watcher-df', holder=holder, root=tmp_path, now=_NOW)

    assert claim.decision == sr.LeaseDecision.ACQUIRED
    assert claim.acquired is True
    assert claim.holder is not None
    assert claim.holder.session_slug == 'watcher-df-100'

    assert sr.leases_dir(root=tmp_path).is_dir()
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    assert lease_path.is_file()
    assert sr.LeaseHolder.from_json(lease_path.read_text()) == holder


# --- claim_lease: held by a live holder ----------------------------------


def test_claim_lease_held_by_live_holder_stand_down_policy(tmp_path: Path) -> None:
    original = sr.LeaseHolder(session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat())
    sr.claim_lease('watcher-df', holder=original, root=tmp_path, now=_NOW)
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    _set_mtime(lease_path, _NOW, timedelta(seconds=42))

    contender = sr.LeaseHolder(session_slug='watcher-df-200', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = sr.claim_lease(
        'watcher-df', holder=contender, policy=sr.LeasePolicy.STAND_DOWN, root=tmp_path, now=_NOW
    )

    assert claim.acquired is False
    assert claim.decision == sr.LeaseDecision.STAND_DOWN
    assert claim.holder is not None
    assert claim.holder.session_slug == 'watcher-df-100'
    assert claim.holder_alive is True
    assert claim.heartbeat_age_secs == 42
    assert claim.message == 'lease held by watcher-df-100 (alive, heartbeat 42s ago) — standing down'
    # No clobber: the on-disk body still names the ORIGINAL holder.
    assert sr.LeaseHolder.from_json(lease_path.read_text()) == original


def test_claim_lease_held_by_live_holder_warn_and_proceed_policy(tmp_path: Path) -> None:
    original = sr.LeaseHolder(session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat())
    sr.claim_lease('unblock-df#2085', holder=original, root=tmp_path, now=_NOW)
    lease_path = sr.lease_path_for_name('unblock-df#2085', root=tmp_path)
    _set_mtime(lease_path, _NOW, timedelta(seconds=42))

    contender = sr.LeaseHolder(session_slug='unblock-df-200', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = sr.claim_lease(
        'unblock-df#2085',
        holder=contender,
        policy=sr.LeasePolicy.WARN_AND_PROCEED,
        root=tmp_path,
        now=_NOW,
    )

    assert claim.acquired is False
    assert claim.decision == sr.LeaseDecision.PROCEED
    assert claim.holder is not None
    assert claim.holder.session_slug == 'watcher-df-100'
    assert 'lease held by watcher-df-100' in claim.message
    assert 'proceeding' in claim.message
    # No clobber: the on-disk body still names the ORIGINAL holder.
    assert sr.LeaseHolder.from_json(lease_path.read_text()) == original


# --- claim_lease: stale-lease reap-and-reclaim + the AND boundary --------


def test_claim_lease_reaps_and_reclaims_a_stale_lease(tmp_path: Path) -> None:
    stale_holder = sr.LeaseHolder(session_slug='watcher-df-dead', pid=_DEAD_PID, start_ts=_NOW.isoformat())
    sr.claim_lease('watcher-df', holder=stale_holder, root=tmp_path, now=_NOW)
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    _set_mtime(lease_path, _NOW, sr.LEASE_HEARTBEAT_TTL + timedelta(minutes=1))

    new_holder = sr.LeaseHolder(session_slug='watcher-df-new', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = sr.claim_lease('watcher-df', holder=new_holder, root=tmp_path, now=_NOW)

    assert claim.acquired is True
    assert claim.decision == sr.LeaseDecision.ACQUIRED
    assert claim.holder == new_holder
    assert sr.LeaseHolder.from_json(lease_path.read_text()) == new_holder


def test_claim_lease_does_not_reap_a_dead_holder_within_ttl(tmp_path: Path) -> None:
    # Proves the reap rule is (age > TTL) AND (pid dead), not either alone:
    # a dead-pid holder with a still-fresh heartbeat is NOT reaped.
    stale_holder = sr.LeaseHolder(session_slug='watcher-df-dead', pid=_DEAD_PID, start_ts=_NOW.isoformat())
    sr.claim_lease('watcher-df', holder=stale_holder, root=tmp_path, now=_NOW)
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    _set_mtime(lease_path, _NOW, sr.LEASE_HEARTBEAT_TTL - timedelta(minutes=1))

    new_holder = sr.LeaseHolder(session_slug='watcher-df-new', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = sr.claim_lease('watcher-df', holder=new_holder, root=tmp_path, now=_NOW)

    assert claim.acquired is False
    assert claim.holder is not None
    assert claim.holder.session_slug == 'watcher-df-dead'
    assert sr.LeaseHolder.from_json(lease_path.read_text()) == stale_holder


def test_claim_lease_survives_lease_vanishing_between_create_and_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulates a concurrent release/reap landing in the narrow window
    # between our failed O_EXCL create and _read_lease_holder_state's
    # path.stat() -- this must reclaim the (now-free) lease rather than
    # propagate an uncaught FileNotFoundError.
    original = sr.LeaseHolder(session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat())
    sr.claim_lease('watcher-df', holder=original, root=tmp_path, now=_NOW)
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)

    real_stat = Path.stat

    def _vanished_stat(self: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        if self == lease_path:
            raise FileNotFoundError('simulated concurrent release')
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'stat', _vanished_stat)

    new_holder = sr.LeaseHolder(session_slug='watcher-df-new', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = sr.claim_lease('watcher-df', holder=new_holder, root=tmp_path, now=_NOW)

    assert claim.acquired is True
    assert claim.decision == sr.LeaseDecision.ACQUIRED
    assert claim.holder == new_holder


def test_claim_lease_does_not_clobber_a_lease_reclaimed_between_stale_check_and_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulates a competitor's reap-and-reclaim landing in the gap between
    # claim_lease's first staleness read and its re-verify read immediately
    # before unlinking: the re-verify must see the competitor's fresh, live
    # holder and skip the unlink rather than clobbering it.
    stale_holder = sr.LeaseHolder(session_slug='watcher-df-dead', pid=_DEAD_PID, start_ts=_NOW.isoformat())
    sr.claim_lease('watcher-df', holder=stale_holder, root=tmp_path, now=_NOW)
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    _set_mtime(lease_path, _NOW, sr.LEASE_HEARTBEAT_TTL + timedelta(minutes=1))

    competitor_holder = sr.LeaseHolder(
        session_slug='watcher-df-competitor', pid=os.getpid(), start_ts=_NOW.isoformat()
    )
    # First read (top of claim_lease) observes the true, stale state; the
    # re-verify read (immediately before unlinking) reports that a
    # competitor has already reaped-and-reclaimed it in the gap.
    responses = iter(
        [
            (stale_holder, False, sr.LEASE_HEARTBEAT_TTL.total_seconds() + 60),
            (competitor_holder, True, 1.0),
        ]
    )
    monkeypatch.setattr(sr, '_read_lease_holder_state', lambda path, *, now: next(responses))

    new_holder = sr.LeaseHolder(session_slug='watcher-df-new', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = sr.claim_lease('watcher-df', holder=new_holder, root=tmp_path, now=_NOW)

    assert claim.acquired is False
    assert claim.holder == competitor_holder
    # unlink() must never have been called: the on-disk body (still the
    # pre-mock stale_holder written at setup) is untouched, proving the
    # competitor's (simulated) fresh lease was not clobbered.
    assert sr.LeaseHolder.from_json(lease_path.read_text()) == stale_holder


# --- claim_lease: corrupt/unreadable lease body ---------------------------


def test_claim_lease_reaps_and_reclaims_a_corrupt_and_aged_lease(tmp_path: Path) -> None:
    corrupt_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text('{not valid json')
    _set_mtime(corrupt_path, _NOW, sr.LEASE_HEARTBEAT_TTL + timedelta(minutes=1))

    new_holder = sr.LeaseHolder(session_slug='watcher-df-new', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = sr.claim_lease('watcher-df', holder=new_holder, root=tmp_path, now=_NOW)

    assert claim.acquired is True
    assert claim.decision == sr.LeaseDecision.ACQUIRED
    assert claim.holder == new_holder
    assert sr.LeaseHolder.from_json(corrupt_path.read_text()) == new_holder


def test_claim_lease_contention_on_corrupt_body_within_ttl_reports_unknown_holder(
    tmp_path: Path,
) -> None:
    corrupt_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text('{not valid json')
    _set_mtime(corrupt_path, _NOW, sr.LEASE_HEARTBEAT_TTL - timedelta(minutes=1))

    contender = sr.LeaseHolder(session_slug='watcher-df-200', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = sr.claim_lease(
        'watcher-df', holder=contender, policy=sr.LeasePolicy.STAND_DOWN, root=tmp_path, now=_NOW
    )

    assert claim.acquired is False
    assert claim.decision == sr.LeaseDecision.STAND_DOWN
    assert claim.holder is None
    assert claim.holder_alive is False
    assert '<unknown>' in claim.message
    assert 'standing down' in claim.message
    # Not reaped (within TTL): the corrupt body is left exactly as-is.
    assert corrupt_path.read_text() == '{not valid json'


# --- heartbeat_lease / release_lease --------------------------------------


def test_heartbeat_lease_advances_mtime(tmp_path: Path) -> None:
    holder = sr.LeaseHolder(session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat())
    sr.claim_lease('watcher-df', holder=holder, root=tmp_path, now=_NOW)
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    _set_mtime(lease_path, _NOW, timedelta(hours=1))
    old_mtime = lease_path.stat().st_mtime

    result = sr.heartbeat_lease('watcher-df', root=tmp_path)

    assert result is True
    assert lease_path.stat().st_mtime > old_mtime


def test_heartbeat_lease_on_absent_lease_returns_false_without_raising(tmp_path: Path) -> None:
    result = sr.heartbeat_lease('watcher-df', root=tmp_path)
    assert result is False


def test_release_lease_removes_the_lease_file(tmp_path: Path) -> None:
    holder = sr.LeaseHolder(session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat())
    sr.claim_lease('watcher-df', holder=holder, root=tmp_path, now=_NOW)
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)

    result = sr.release_lease('watcher-df', root=tmp_path)

    assert result is True
    assert not lease_path.exists()


def test_release_lease_is_idempotent_on_a_second_call(tmp_path: Path) -> None:
    holder = sr.LeaseHolder(session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat())
    sr.claim_lease('watcher-df', holder=holder, root=tmp_path, now=_NOW)

    first = sr.release_lease('watcher-df', root=tmp_path)
    second = sr.release_lease('watcher-df', root=tmp_path)

    assert first is True
    assert second is False


# ---------------------------------------------------------------------------
# reap_stale_leases matrix
# ---------------------------------------------------------------------------


def test_reap_stale_leases_matrix(tmp_path: Path) -> None:
    live_holder = sr.LeaseHolder(session_slug='kept-live', pid=os.getpid(), start_ts=_NOW.isoformat())
    sr.claim_lease('kept-live', holder=live_holder, root=tmp_path, now=_NOW)
    _set_mtime(sr.lease_path_for_name('kept-live', root=tmp_path), _NOW, timedelta(days=30))

    dead_within_ttl = sr.LeaseHolder(
        session_slug='kept-dead-fresh', pid=_DEAD_PID, start_ts=_NOW.isoformat()
    )
    sr.claim_lease('kept-dead-fresh', holder=dead_within_ttl, root=tmp_path, now=_NOW)
    _set_mtime(
        sr.lease_path_for_name('kept-dead-fresh', root=tmp_path),
        _NOW,
        sr.LEASE_HEARTBEAT_TTL - timedelta(minutes=1),
    )

    dead_past_ttl = sr.LeaseHolder(
        session_slug='reap-dead-stale', pid=_DEAD_PID, start_ts=_NOW.isoformat()
    )
    sr.claim_lease('reap-dead-stale', holder=dead_past_ttl, root=tmp_path, now=_NOW)
    _set_mtime(
        sr.lease_path_for_name('reap-dead-stale', root=tmp_path),
        _NOW,
        sr.LEASE_HEARTBEAT_TTL + timedelta(minutes=1),
    )

    corrupt_path = sr.lease_path_for_name('reap-corrupt', root=tmp_path)
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text('{not valid json')
    _set_mtime(corrupt_path, _NOW, sr.LEASE_HEARTBEAT_TTL + timedelta(minutes=1))

    reaped = sr.reap_stale_leases(root=tmp_path, now=_NOW)

    reaped_by_name = {r.lease_name: r.reason for r in reaped}
    assert reaped_by_name == {
        'reap-dead-stale': 'stale_pid',
        'reap-corrupt': 'corrupt',
    }
    remaining = {p.stem for p in sr.leases_dir(root=tmp_path).iterdir()}
    assert remaining == {'kept-live', 'kept-dead-fresh'}


def test_reap_stale_leases_continues_sweep_when_one_removal_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    poison_holder = sr.LeaseHolder(session_slug='poison', pid=_DEAD_PID, start_ts=_NOW.isoformat())
    sr.claim_lease('poison', holder=poison_holder, root=tmp_path, now=_NOW)
    victim_holder = sr.LeaseHolder(session_slug='victim', pid=_DEAD_PID, start_ts=_NOW.isoformat())
    sr.claim_lease('victim', holder=victim_holder, root=tmp_path, now=_NOW)
    for lease_name in ('poison', 'victim'):
        _set_mtime(
            sr.lease_path_for_name(lease_name, root=tmp_path),
            _NOW,
            sr.LEASE_HEARTBEAT_TTL + timedelta(minutes=1),
        )

    poison_path = sr.lease_path_for_name('poison', root=tmp_path)
    real_unlink = Path.unlink

    def _flaky_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == poison_path:
            raise OSError('simulated permission error')
        real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', _flaky_unlink)

    with caplog.at_level(logging.ERROR):
        reaped = sr.reap_stale_leases(root=tmp_path, now=_NOW)

    assert {r.lease_name for r in reaped} == {'victim'}
    assert poison_path.is_file()  # left behind, not silently claimed as reaped
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


# ---------------------------------------------------------------------------
# CLI lease-* verbs + fail-open fail-soft
# ---------------------------------------------------------------------------


def test_main_lease_claim_on_free_name_acquires_and_prints_decision_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main(
        ['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100', '--pid', str(os.getpid())]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == 'decision=acquired'
    assert sr.lease_path_for_name('watcher-df', root=tmp_path).is_file()


def test_main_lease_claim_when_held_by_live_holder_stands_down(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    sr.main(
        ['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100', '--pid', str(os.getpid())]
    )
    capsys.readouterr()  # discard the first claim's output

    rc = sr.main(
        [
            'lease-claim',
            '--name',
            'watcher-df',
            '--slug',
            'watcher-df-200',
            '--pid',
            str(os.getpid()),
            '--policy',
            'stand-down',
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == 'decision=stand-down'
    assert re.search(r'lease held by \S+ \(alive, heartbeat \d+s ago\) — standing down', lines[1])
    # the on-disk holder must still be the ORIGINAL claimant (no clobber).
    body = sr.lease_path_for_name('watcher-df', root=tmp_path).read_text()
    assert 'watcher-df-100' in body
    assert 'watcher-df-200' not in body


def test_main_lease_heartbeat_bumps_mtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    sr.main(['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100', '--pid', str(os.getpid())])
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    _set_mtime(lease_path, _NOW, timedelta(hours=1))
    backdated_mtime = lease_path.stat().st_mtime

    rc = sr.main(['lease-heartbeat', '--name', 'watcher-df'])

    assert rc == 0
    assert lease_path.stat().st_mtime > backdated_mtime


def test_main_lease_release_removes_the_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    sr.main(['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100', '--pid', str(os.getpid())])
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    assert lease_path.is_file()

    rc = sr.main(['lease-release', '--name', 'watcher-df'])

    assert rc == 0
    assert not lease_path.exists()


def test_main_lease_reap_removes_a_stale_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    sr.main(['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100', '--pid', str(_DEAD_PID)])
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    _set_mtime(lease_path, _NOW, sr.LEASE_HEARTBEAT_TTL + timedelta(minutes=1))

    rc = sr.main(['lease-reap'])

    assert rc == 0
    assert not lease_path.exists()


def test_main_lease_claim_is_fail_open_never_stand_down_on_a_fault(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    def _boom(*_args: object, **_kwargs: object) -> sr.LeaseClaim:
        raise OSError('lease substrate on fire')

    monkeypatch.setattr(sr, 'claim_lease', _boom)

    with caplog.at_level(logging.ERROR):
        rc = sr.main(
            ['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100', '--pid', str(os.getpid())]
        )

    assert rc == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == 'decision=proceed'
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


# ---------------------------------------------------------------------------
# CLI write-decision verb (Fleet Cockpit C8: park-to-registry for watchers)
# ---------------------------------------------------------------------------


def test_main_write_decision_files_open_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main(
        [
            'write-decision',
            '--id',
            'dec-park-1',
            '--project',
            'df',
            '--text',
            'Approve risky merge?',
            '--task-id',
            '2085',
            '--escalation-id',
            'esc-1',
            '--session-id',
            'watcher-df-99',
        ]
    )

    assert rc == 0
    listed = sr.list_decisions(root=tmp_path)
    assert len(listed) == 1
    rec = listed[0]
    assert rec.id == 'dec-park-1'
    assert rec.project == 'df'
    assert rec.text == 'Approve risky merge?'
    assert rec.task_id == '2085'
    assert rec.escalation_id == 'esc-1'
    assert rec.session_id == 'watcher-df-99'
    assert rec.state == sr.DecisionState.OPEN
    assert rec.filed_at != ''


def test_main_write_decision_stamps_severity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A watcher's park moment supplies the escalation's severity so the
    cockpit decision queue can weight this ask (Fleet Cockpit F7 fix 1).
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main(
        [
            'write-decision',
            '--id',
            'dec-sev',
            '--project',
            'df',
            '--text',
            'q',
            '--severity',
            'critical',
        ]
    )

    assert rc == 0
    listed = sr.list_decisions(root=tmp_path)
    assert len(listed) == 1
    assert listed[0].severity == 'critical'


def test_main_write_decision_severity_defaults_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Omitting --severity yields severity='' on the filed record (a watcher
    that doesn't know/supply a severity still files a well-formed decision).
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main(['write-decision', '--id', 'dec-nosev', '--project', 'df', '--text', 'q'])

    assert rc == 0
    listed = sr.list_decisions(root=tmp_path)
    assert len(listed) == 1
    assert listed[0].severity == ''


def test_main_write_decision_prints_filed_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The verb echoes the filed decision id on stdout so a watcher can
    cross-link it into its in-session note / afk-digest line.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main(['write-decision', '--id', 'dec-park-2', '--project', 'df', '--text', 'q'])

    assert rc == 0
    assert 'dec-park-2' in capsys.readouterr().out


def test_main_write_decision_fail_soft_when_fleet_root_under_a_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mirrors test_write_decision_fail_soft_on_unwritable_root, at the CLI
    boundary: write_decision's own fail-soft guard (not main()'s outer
    try/except) must absorb the fault, so _run_write_decision must not print
    a false confirmation when the underlying write never happened.
    """
    blocker = tmp_path / 'blocker'
    blocker.write_text('not a directory')
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(blocker / 'fleet'))

    with caplog.at_level(logging.ERROR):
        rc = sr.main(['write-decision', '--id', 'dec-park-3', '--project', 'df', '--text', 'q'])

    assert rc == 0
    assert 'dec-park-3' not in capsys.readouterr().out
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert not (blocker / 'fleet').exists()


def test_main_write_decision_refiling_same_id_overwrites_not_duplicates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SKILL.md promises watchers can safely re-file the same stable id
    across restarts: the second write must overwrite the first record in
    place rather than accumulating a duplicate.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc1 = sr.main(['write-decision', '--id', 'dec-park-4', '--project', 'df', '--text', 'first?'])
    rc2 = sr.main(['write-decision', '--id', 'dec-park-4', '--project', 'df', '--text', 'second?'])

    assert rc1 == 0
    assert rc2 == 0
    listed = sr.list_decisions(root=tmp_path)
    assert len(listed) == 1
    assert listed[0].id == 'dec-park-4'
    assert listed[0].text == 'second?'
