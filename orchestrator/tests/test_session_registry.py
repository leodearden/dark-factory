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
import dataclasses
import fcntl
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
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

    ONE DELIBERATE EXCEPTION: ``escalations_dir`` is absent from the map
    below, so the DecisionRecord dataclass default ('' -- unset/legacy)
    supplies it. It is the only field that is a GUARD INPUT to the reaper's
    axis-2 queue check (``_run_reap_decisions._status``), so a non-empty
    shared default here would make every _make_decision()-built decision
    carry a queue FOREIGN to the tmp_path queue a reap test invokes the
    reaper with -- short-circuiting _status before read_escalation_status
    runs and collapsing the test into a vacuous ``assert state == OPEN``. Do
    not add it here; a test needing a real queue passes ``escalations_dir=``
    at its own call site.
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


def test_make_decision_defaults_to_the_unset_queue_sentinel() -> None:
    """The fixture must not hand escalations_dir a non-'' default.

    Guards the one exception documented on _make_decision: escalations_dir is
    a control-flow guard input to ``_run_reap_decisions._status``, and a
    non-empty shared default silently neuters every reap-decisions test that
    does not override it (task 3528 -- it cost three live regression tests at
    once, each still passing with its premise inverted). Omitting the key is
    the structural fix; this is the one-line tripwire against re-adding it.
    """
    assert _make_decision().escalations_dir == ''


def test_decision_state_enum_values() -> None:
    assert issubclass(sr.DecisionState, str)
    assert {m.value for m in sr.DecisionState} == {'open', 'answered', 'dropped'}
    assert sr.DecisionState.OPEN.value == 'open'
    assert sr.DecisionState.ANSWERED.value == 'answered'
    assert sr.DecisionState.DROPPED.value == 'dropped'


def test_decision_record_dict_round_trip_is_lossless() -> None:
    # escalations_dir is stated explicitly (the fixture deliberately omits it
    # and lets the '' dataclass default stand): a dropped key would round-trip
    # indistinguishably from '', so only a non-empty value keeps this test's
    # field-drop catch power.
    d = _make_decision(escalations_dir='/p/data/reconciliation/escalations')
    assert sr.DecisionRecord.from_dict(d.to_dict()) == d


def test_decision_record_json_round_trip_is_lossless() -> None:
    d = _make_decision(escalations_dir='/p/data/reconciliation/escalations')
    assert sr.DecisionRecord.from_json(d.to_json()) == d


def test_decision_record_round_trip_with_null_fields() -> None:
    """session_id/task_id/escalation_id/options may legitimately be None (a
    project-level decision with no session/task/escalation context yet);
    the round trip must preserve that, not coerce it or drop the key.
    """
    d = _make_decision(
        session_id=None,
        task_id=None,
        escalation_id=None,
        options=None,
        escalations_dir='/p/data/reconciliation/escalations',
    )
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
    assert d.escalations_dir == ''


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
    record = sr.DecisionRecord.from_dict(pre_queue)
    assert record.escalations_dir == ''


def test_decision_record_parses_null_escalations_dir_as_empty() -> None:
    """An explicit JSON null (a hand-edited or third-party-written record)
    must also collapse to '', not None -- the annotation is `str`, and the
    reaper's queue guard is a single `if decision_dir and ...` test with no
    None-vs-''-vs-missing branching. Fail-soft, not a raise.
    """
    record = sr.DecisionRecord.from_dict(
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
    stamped = _make_decision(escalations_dir='/p/data/reconciliation/escalations')
    assert stamped.to_dict()['escalations_dir'] == '/p/data/reconciliation/escalations'
    assert 'escalations_dir' in _make_decision().to_dict()


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


def test_transcript_path_for_cwd_encodes_underscore() -> None:
    # The third character (task 3272). Derived from 738 real (encoded-dir,
    # decoded-cwd) pairs sampled off a live ~/.claude/projects tree, where
    # two thirds of the dirs carried an underscore — the former two-character
    # rule mismatched 492 of the 738.
    assert (
        sr.transcript_path_for_cwd('/media/leo/data_lv_1/leo/reify-build')
        == '~/.claude/projects/-media-leo-data-lv-1-leo-reify-build'
    )


def test_encode_cwd_returns_bare_dir_name() -> None:
    # encode_cwd is the bare encoded dir name — no '~/.claude/projects/'
    # prefix — so callers that need a lookup key (not a display path) have
    # one canonical source instead of string-slicing the prefix back off.
    assert sr.encode_cwd('/home/leo/src/dark-factory') == '-home-leo-src-dark-factory'


def test_encode_cwd_preserves_case() -> None:
    # A real on-disk dir name with its capitals intact — the encoder does NOT
    # lowercase, which rules out a case-folding step in the rule.
    assert sr.encode_cwd('/opt/Auto-Claude/resources/backend') == (
        '-opt-Auto-Claude-resources-backend'
    )


def test_encode_cwd_maps_all_three_characters() -> None:
    # '/' , '.' and '_' all collapse to '-'; a leading '_' on a path component
    # yields a doubled '--' just as a leading '.' does.
    assert sr.encode_cwd('/home/leo/src/warm-lanes/worktrees/_lane-39') == (
        '-home-leo-src-warm-lanes-worktrees--lane-39'
    )
    assert sr.encode_cwd('/home/leo/src/dark-factory/.eval-worktrees/df_task_12') == (
        '-home-leo-src-dark-factory--eval-worktrees-df-task-12'
    )


def test_transcript_path_for_cwd_is_prefix_plus_encoding() -> None:
    # Pins the '~/.claude/projects/' prefix AND the encoding together over a
    # cwd exercising all three characters, as a hard-coded literal.
    #
    # Asserting `== f'~/.claude/projects/{sr.encode_cwd(cwd)}'` instead would
    # be a character-for-character restatement of the one-line implementation
    # — it could not fail while the implementation keeps that shape, so it
    # would pin nothing. That is the same self-consistency trap task 3272
    # calls out for fixtures built with the encoder under test.
    assert (
        sr.transcript_path_for_cwd('/home/leo/src/dark-factory/.eval-worktrees/df_task_12')
        == '~/.claude/projects/-home-leo-src-dark-factory--eval-worktrees-df-task-12'
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


def test_set_decision_escalations_dir_persists_normalized(tmp_path: Path) -> None:
    """The back-fill's writer (task 3640) normalizes ON THE WAY IN.

    A non-canonical spelling passed by a caller must not be stored raw: the
    reaper's axis-2 guard compares stored-value against reaper-value, and a
    dotted/trailing-slash spelling stored verbatim would compare unequal to
    the very queue it names -- the fail-open direction, so the record would
    silently never close again. Normalizing here means every writer of this
    field (write-decision, the back-fill) stores ONE spelling.
    """
    rec = _make_decision(id='dec-setescdir')
    sr.write_decision(rec, root=tmp_path)
    queue = tmp_path / 'escalations'
    queue.mkdir()
    (tmp_path / 'sub').mkdir()
    dotted = str(tmp_path / 'sub' / '..' / 'escalations') + '/'
    assert dotted != str(queue)

    updated = sr.set_decision_escalations_dir(rec.id, dotted, root=tmp_path)

    assert updated is not None
    assert updated.escalations_dir == sr.normalize_escalations_dir(queue)
    [reread] = [d for d in sr.list_decisions(root=tmp_path) if d.id == rec.id]
    assert reread.escalations_dir == sr.normalize_escalations_dir(queue)


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
    rec = _make_decision(id='dec-setescdir-unknown')
    sr.write_decision(rec, root=tmp_path)

    updated = sr.set_decision_escalations_dir(rec.id, sr.UNKNOWN_QUEUE, root=tmp_path)

    assert updated is not None
    assert updated.escalations_dir == sr.UNKNOWN_QUEUE
    [reread] = [d for d in sr.list_decisions(root=tmp_path) if d.id == rec.id]
    assert reread.escalations_dir == sr.UNKNOWN_QUEUE


def test_set_decision_escalations_dir_preserves_every_other_field(tmp_path: Path) -> None:
    """A read-modify-write must modify exactly ONE field.

    write_decision rewrites the whole file, so a dropped/defaulted field in
    the round-trip is silent data loss -- and the back-fill runs this over the
    entire open cockpit population at once, where losing e.g. `options` or
    `manual_boost` would quietly degrade real human gates. _make_decision
    gives every field a distinguishable value precisely so this can catch it.
    """
    rec = _make_decision(id='dec-setescdir-preserve', state=sr.DecisionState.OPEN)
    sr.write_decision(rec, root=tmp_path)

    updated = sr.set_decision_escalations_dir(rec.id, sr.UNKNOWN_QUEUE, root=tmp_path)

    assert updated is not None
    [reread] = [d for d in sr.list_decisions(root=tmp_path) if d.id == rec.id]
    assert reread == dataclasses.replace(rec, escalations_dir=sr.UNKNOWN_QUEUE)
    # Spelled out as well as compared wholesale, so a failure names the field.
    assert reread.state == sr.DecisionState.OPEN
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
    assert sr.set_decision_escalations_dir('no-such-id', '/tmp/q', root=tmp_path) is None


def test_set_decision_escalations_dir_fail_soft_on_corrupt_body(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A corrupt on-disk body returns None (logged at ERROR), not a traceback."""
    corrupt_path = sr.decision_path_for_id('dec-setescdir-corrupt', root=tmp_path)
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text('{not valid json')

    with caplog.at_level(logging.ERROR):
        result = sr.set_decision_escalations_dir('dec-setescdir-corrupt', '/tmp/q', root=tmp_path)

    assert result is None
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_set_decision_escalations_dir_is_repeatable(tmp_path: Path) -> None:
    """Called twice on the same id it simply succeeds twice (last write wins).

    The back-fill is re-runnable by design, and the lock is a stable sidecar
    the first call creates -- so a second call must not trip over its own
    lock file. The locking contract itself is covered by TestDecisionIdLock;
    this only pins that repeated calls compose.
    """
    rec = _make_decision(id='dec-setescdir-twice')
    sr.write_decision(rec, root=tmp_path)

    first = sr.set_decision_escalations_dir(rec.id, sr.UNKNOWN_QUEUE, root=tmp_path)
    second = sr.set_decision_escalations_dir(rec.id, tmp_path / 'q', root=tmp_path)

    assert first is not None
    assert second is not None
    [reread] = [d for d in sr.list_decisions(root=tmp_path) if d.id == rec.id]
    assert reread.escalations_dir == sr.normalize_escalations_dir(tmp_path / 'q')


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
        stamp_result = sr.set_decision_escalations_dir('dec-lockfault', '/tmp/q', root=tmp_path)

    assert state_result is None
    assert boost_result is None
    assert stamp_result is None
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
    update_decision_state, set_manual_boost and set_decision_escalations_dir
    must EACH acquire decision_id_lock for the correct decision id.

    One case per public helper, deliberately, even though all three now share
    _mutate_decision: the lock is a per-helper CONTRACT, and a future setter
    that grows its own body (or a wrapper that mutates before delegating)
    would slip past a single shared-implementation test. TestDecisionIdLock
    covers the lock primitive itself; these cover its ADOPTION.
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
        rec = _make_decision(id='dec-spy-3', escalations_dir='')
        sr.write_decision(rec, root=tmp_path)

        real_lock = sr.decision_id_lock
        acquired: list[str] = []

        @contextlib.contextmanager
        def recording_lock(decision_id: str, root: Path | str | None = None):
            acquired.append(decision_id)
            with real_lock(decision_id, root=root):
                yield

        monkeypatch.setattr(sr, 'decision_id_lock', recording_lock)

        sr.set_decision_escalations_dir('dec-spy-3', '/tmp/some-queue', root=tmp_path)

        assert 'dec-spy-3' in acquired, f'Expected lock acquisition for dec-spy-3; got {acquired}'


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


def test_reap_respects_limit_stops_after_n_removals(tmp_path: Path) -> None:
    """A positive `limit` stops the sweep after that many dirs are actually
    removed -- bounding both the rmtree work and the per-call scan cost, so
    an opportunistic per-spawn prune driver stays cheap regardless of how
    large the on-disk backlog has grown.
    """
    records = [
        _make_record(session_slug=slug, status=sr.Status.EXITED, launcher_pid=os.getpid())
        for slug in ('term-a', 'term-b', 'term-c')
    ]
    for r in records:
        sr.write_record(r, root=tmp_path)
        _set_mtime(
            sr.record_path_for_slug(r.session_slug, root=tmp_path),
            _NOW,
            sr.TERMINAL_TTL + timedelta(hours=1),
        )

    reaped = sr.reap_stale_records(root=tmp_path, now=_NOW, limit=1)

    assert len(reaped) == 1
    remaining = {p.name for p in sr.sessions_dir(root=tmp_path).iterdir()}
    assert len(remaining) == 2


def test_reap_default_limit_none_is_unbounded(tmp_path: Path) -> None:
    """Explicit `limit=None` -- and the implicit default -- must reproduce
    today's unbounded full sweep, so every pre-existing reap test keeps
    passing unchanged.
    """
    records = [
        _make_record(session_slug=slug, status=sr.Status.EXITED, launcher_pid=os.getpid())
        for slug in ('term-d', 'term-e', 'term-f')
    ]
    for r in records:
        sr.write_record(r, root=tmp_path)
        _set_mtime(
            sr.record_path_for_slug(r.session_slug, root=tmp_path),
            _NOW,
            sr.TERMINAL_TTL + timedelta(hours=1),
        )

    reaped = sr.reap_stale_records(root=tmp_path, now=_NOW, limit=None)

    assert len(reaped) == 3
    assert list(sr.sessions_dir(root=tmp_path).iterdir()) == []


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
# Task 2934 step-1/2: _normalize_wm_window_id (decimal-vs-hex canonicalization)
# ---------------------------------------------------------------------------


def test_normalize_wm_window_id_decimal_and_padded_hex_collapse_to_same_canonical() -> None:
    assert sr._normalize_wm_window_id('0x0000001a') == '0x1a'
    assert sr._normalize_wm_window_id('26') == '0x1a'
    # The crux: a decimal-captured id and a zero-padded-hex wmctrl id for the
    # SAME window must compare equal after normalization.
    assert sr._normalize_wm_window_id('26') == sr._normalize_wm_window_id('0x0000001a')


def test_normalize_wm_window_id_already_canonical_round_trips() -> None:
    assert sr._normalize_wm_window_id('0x1a') == '0x1a'


@pytest.mark.parametrize(
    'raw',
    [None, '', '   ', '0xZZ', 'nope', '-1'],
    ids=['none', 'empty', 'whitespace', 'unparseable-hex', 'unparseable-decimal', 'negative'],
)
def test_normalize_wm_window_id_unparseable_or_negative_returns_none(raw: str | None) -> None:
    assert sr._normalize_wm_window_id(raw) is None


# ---------------------------------------------------------------------------
# Task 2934 step-3/4: _wmctrl_live_window_ids (single-probe live-window-id set)
# ---------------------------------------------------------------------------


def _fake_wmctrl_run(
    stdout_lines: list[str], returncode: int = 0
) -> Callable[[list[str]], subprocess.CompletedProcess[str]]:
    """Build a fake ``run`` callable returning a canned ``wmctrl -l`` result."""

    def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv, returncode=returncode, stdout='\n'.join(stdout_lines)
        )

    return _run


def test_wmctrl_live_window_ids_rc0_multiline_returns_normalized_set() -> None:
    run = _fake_wmctrl_run(
        [
            '0x0000001a  0 host  first window title',
            '0x2b 1 host second-window-title',
        ]
    )
    assert sr._wmctrl_live_window_ids(run) == {'0x1a', '0x2b'}


def test_wmctrl_live_window_ids_collapses_mixed_casing_and_padding() -> None:
    run = _fake_wmctrl_run(['0X0000002B  0 host  some window'])
    assert sr._wmctrl_live_window_ids(run) == {'0x2b'}


def test_wmctrl_live_window_ids_rc0_empty_stdout_returns_empty_set_not_none() -> None:
    run = _fake_wmctrl_run([])
    result = sr._wmctrl_live_window_ids(run)
    assert result == set()
    assert result is not None


def test_wmctrl_live_window_ids_rc127_missing_binary_returns_none() -> None:
    run = _fake_wmctrl_run(['0x1a  0 host  irrelevant'], returncode=127)
    assert sr._wmctrl_live_window_ids(run) is None


def test_wmctrl_live_window_ids_rc124_transient_timeout_returns_none() -> None:
    run = _fake_wmctrl_run(['0x1a  0 host  irrelevant'], returncode=124)
    assert sr._wmctrl_live_window_ids(run) is None


def test_wmctrl_live_window_ids_other_nonzero_rc_returns_none() -> None:
    run = _fake_wmctrl_run(['0x1a  0 host  irrelevant'], returncode=1)
    assert sr._wmctrl_live_window_ids(run) is None


def test_wmctrl_live_window_ids_run_raising_returns_none() -> None:
    def _boom(argv: list[str]) -> subprocess.CompletedProcess[str]:
        raise OSError('wmctrl exploded')

    assert sr._wmctrl_live_window_ids(_boom) is None


def test_wmctrl_live_window_ids_titleless_short_line_still_contributes_id() -> None:
    # No desktop/host/title columns -- id-only leniency: a reaper must fail
    # toward keeping live sessions, so column-0-only lines still count.
    run = _fake_wmctrl_run(['0x3c'])
    assert sr._wmctrl_live_window_ids(run) == {'0x3c'}


# ---------------------------------------------------------------------------
# Task 2934 step-5/6: mark_windowless_wm_sessions_exited (happy path)
# ---------------------------------------------------------------------------


def test_mark_windowless_marks_wm_record_whose_window_id_is_absent(tmp_path: Path) -> None:
    r = _make_record(
        session_slug='windowless-1',
        status=sr.Status.AWAITING_INPUT,
        display=sr.Display(kind='wm', wm_title='some title', wm_window_id='0x1a'),
    )
    sr.write_record(r, root=tmp_path)
    run = _fake_wmctrl_run(['0x0000ff99  0 host  some other window'])

    marked = sr.mark_windowless_wm_sessions_exited(root=tmp_path, run=run)

    assert {m.session_slug for m in marked} == {'windowless-1'}
    record_path = sr.record_path_for_slug('windowless-1', root=tmp_path)
    assert record_path.parent.is_dir()  # marked, NOT deleted
    assert record_path.is_file()
    reloaded = sr.read_record('windowless-1', root=tmp_path)
    assert reloaded.status == sr.Status.EXITED
    assert reloaded.exit_code == sr.ORPHAN_EXIT_CODE


def test_mark_windowless_keeps_decimal_captured_id_matching_padded_hex_live_window(
    tmp_path: Path,
) -> None:
    # DECIMAL-vs-hex KEEP: '26' (decimal) is the SAME window as the live,
    # zero-padded-hex '0x0000001a' wmctrl reports -- must NOT be reaped.
    r = _make_record(
        session_slug='decimal-kept',
        status=sr.Status.RUNNING,
        display=sr.Display(kind='wm', wm_title='t', wm_window_id='26'),
    )
    sr.write_record(r, root=tmp_path)
    run = _fake_wmctrl_run(['0x0000001a  0 host  live window'])

    marked = sr.mark_windowless_wm_sessions_exited(root=tmp_path, run=run)

    assert marked == []
    assert sr.read_record('decimal-kept', root=tmp_path).status == sr.Status.RUNNING


def test_mark_windowless_preserves_all_other_fields(tmp_path: Path) -> None:
    r = _make_record(
        session_slug='windowless-fields',
        status=sr.Status.IDLE,
        role='unblock',
        project='df',
        task_id='2085',
        launcher_pid=os.getpid(),
        start_ts='2026-07-07T00:00:00+00:00',
        display=sr.Display(kind='wm', wm_title='t', wm_window_id='0x1a'),
    )
    sr.write_record(r, root=tmp_path)
    run = _fake_wmctrl_run(['0x99  0 host  other window'])

    sr.mark_windowless_wm_sessions_exited(root=tmp_path, run=run)

    reloaded = sr.read_record('windowless-fields', root=tmp_path)
    assert reloaded.status == sr.Status.EXITED
    assert reloaded.exit_code == sr.ORPHAN_EXIT_CODE
    assert reloaded.role == 'unblock'
    assert reloaded.project == 'df'
    assert reloaded.task_id == '2085'
    assert reloaded.launcher_pid == os.getpid()
    assert reloaded.start_ts == '2026-07-07T00:00:00+00:00'


# ---------------------------------------------------------------------------
# Task 2934 step-7/8: mark_windowless_wm_sessions_exited fail-soft, scope,
# and pid/age-independence
# ---------------------------------------------------------------------------


def test_mark_windowless_fail_soft_when_wmctrl_unavailable(tmp_path: Path) -> None:
    r = _make_record(
        session_slug='no-wmctrl',
        status=sr.Status.AWAITING_INPUT,
        display=sr.Display(kind='wm', wm_title='t', wm_window_id='0x1a'),
    )
    sr.write_record(r, root=tmp_path)
    run = _fake_wmctrl_run([], returncode=127)  # missing binary -- no window evidence

    marked = sr.mark_windowless_wm_sessions_exited(root=tmp_path, run=run)

    assert marked == []
    # A headless run reaps NOTHING, even though this record's window would
    # (if we could prove it) be gone -- "no evidence" != "window gone".
    assert sr.read_record('no-wmctrl', root=tmp_path).status == sr.Status.AWAITING_INPUT


def test_mark_windowless_never_touches_tmux_display(tmp_path: Path) -> None:
    r = _make_record(
        session_slug='tmux-session',
        status=sr.Status.RUNNING,
        display=sr.Display(kind='tmux', wm_title='', wm_window_id=None, tmux_target='fleet-df:2'),
    )
    sr.write_record(r, root=tmp_path)
    run = _fake_wmctrl_run(['0x99  0 host  unrelated'])

    marked = sr.mark_windowless_wm_sessions_exited(root=tmp_path, run=run)

    assert marked == []
    assert sr.read_record('tmux-session', root=tmp_path).status == sr.Status.RUNNING


def test_mark_windowless_never_touches_displayless_record(tmp_path: Path) -> None:
    r = _make_record(session_slug='headless', status=sr.Status.RUNNING, display=None)
    sr.write_record(r, root=tmp_path)
    run = _fake_wmctrl_run(['0x99  0 host  unrelated'])

    marked = sr.mark_windowless_wm_sessions_exited(root=tmp_path, run=run)

    assert marked == []
    assert sr.read_record('headless', root=tmp_path).status == sr.Status.RUNNING


def test_mark_windowless_leaves_already_terminal_wm_record_untouched(tmp_path: Path) -> None:
    r = _make_record(
        session_slug='already-exited-wm',
        status=sr.Status.EXITED,
        exit_code=0,
        display=sr.Display(kind='wm', wm_title='t', wm_window_id='0x1a'),
    )
    sr.write_record(r, root=tmp_path)
    run = _fake_wmctrl_run(['0x99  0 host  unrelated'])  # window gone, but already terminal

    marked = sr.mark_windowless_wm_sessions_exited(root=tmp_path, run=run)

    assert marked == []
    reloaded = sr.read_record('already-exited-wm', root=tmp_path)
    assert reloaded.status == sr.Status.EXITED
    assert reloaded.exit_code == 0  # NOT re-stamped with ORPHAN_EXIT_CODE


@pytest.mark.parametrize(
    'wm_window_id', [None, '0xZZ', 'nope'], ids=['none', 'unparseable-hex', 'unparseable-decimal']
)
def test_mark_windowless_skips_unparseable_or_missing_window_id(
    wm_window_id: str | None, tmp_path: Path
) -> None:
    r = _make_record(
        session_slug='no-id',
        status=sr.Status.RUNNING,
        display=sr.Display(kind='wm', wm_title='t', wm_window_id=wm_window_id),
    )
    sr.write_record(r, root=tmp_path)
    run = _fake_wmctrl_run(['0x99  0 host  unrelated'])

    marked = sr.mark_windowless_wm_sessions_exited(root=tmp_path, run=run)

    assert marked == []
    assert sr.read_record('no-id', root=tmp_path).status == sr.Status.RUNNING


def test_mark_windowless_reaps_regardless_of_live_pid_and_fresh_age(tmp_path: Path) -> None:
    """THE CRUX: window-gone is definitive death evidence, independent of
    launcher_pid liveness or record age -- proving this sweep reaps exactly
    the zombies mark_orphaned_sessions_exited's pid/TTL guards currently keep.
    """
    r = _make_record(
        session_slug='live-pid-fresh-but-windowless',
        status=sr.Status.AWAITING_INPUT,
        launcher_pid=os.getpid(),  # ALIVE
        display=sr.Display(kind='wm', wm_title='t', wm_window_id='0x1a'),
    )
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug('live-pid-fresh-but-windowless', root=tmp_path)
    _set_mtime(record_path, _NOW, timedelta(minutes=1))  # far under NON_TERMINAL_HEARTBEAT_TTL
    run = _fake_wmctrl_run(['0x99  0 host  unrelated'])

    marked = sr.mark_windowless_wm_sessions_exited(root=tmp_path, run=run)

    assert {m.session_slug for m in marked} == {'live-pid-fresh-but-windowless'}
    reloaded = sr.read_record('live-pid-fresh-but-windowless', root=tmp_path)
    assert reloaded.status == sr.Status.EXITED
    assert reloaded.exit_code == sr.ORPHAN_EXIT_CODE


def test_mark_windowless_handles_mixed_population_in_one_sweep(tmp_path: Path) -> None:
    windowless_wm = _make_record(
        session_slug='mix-windowless',
        status=sr.Status.RUNNING,
        display=sr.Display(kind='wm', wm_title='t', wm_window_id='0x1a'),
    )
    present_window_wm = _make_record(
        session_slug='mix-present',
        status=sr.Status.RUNNING,
        display=sr.Display(kind='wm', wm_title='t2', wm_window_id='0x99'),
    )
    tmux_session = _make_record(
        session_slug='mix-tmux',
        status=sr.Status.RUNNING,
        display=sr.Display(kind='tmux', wm_title='', wm_window_id=None, tmux_target='fleet-df:2'),
    )
    headless = _make_record(session_slug='mix-headless', status=sr.Status.RUNNING, display=None)
    terminal_wm = _make_record(
        session_slug='mix-terminal',
        status=sr.Status.EXITED,
        exit_code=0,
        display=sr.Display(kind='wm', wm_title='t3', wm_window_id='0x1a'),
    )
    for r in (windowless_wm, present_window_wm, tmux_session, headless, terminal_wm):
        sr.write_record(r, root=tmp_path)

    run = _fake_wmctrl_run(['0x99  0 host  still-open'])

    marked = sr.mark_windowless_wm_sessions_exited(root=tmp_path, run=run)

    assert {m.session_slug for m in marked} == {'mix-windowless'}
    assert sr.read_record('mix-windowless', root=tmp_path).status == sr.Status.EXITED
    assert sr.read_record('mix-present', root=tmp_path).status == sr.Status.RUNNING
    assert sr.read_record('mix-tmux', root=tmp_path).status == sr.Status.RUNNING
    assert sr.read_record('mix-headless', root=tmp_path).status == sr.Status.RUNNING
    reloaded_terminal = sr.read_record('mix-terminal', root=tmp_path)
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


def test_main_launching_bound_prunes_old_terminal_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every spawn's `launching` path must also opportunistically bound-prune
    terminal/stale record dirs (reap_stale_records(limit=REAP_BATCH_LIMIT)),
    not just mark orphans exited -- otherwise the disk backlog is marked
    terminal on every spawn but never actually reclaimed.
    """
    assert isinstance(sr.REAP_BATCH_LIMIT, int)
    assert sr.REAP_BATCH_LIMIT > 0

    old_terminal = _make_record(
        session_slug='old-terminal', status=sr.Status.EXITED, exit_code=0, launcher_pid=os.getpid()
    )
    sr.write_record(old_terminal, root=tmp_path)
    _set_mtime(
        sr.record_path_for_slug('old-terminal', root=tmp_path),
        datetime.now(UTC),
        sr.TERMINAL_TTL + timedelta(hours=1),
    )

    _set_env(monkeypatch, _launching_env(tmp_path))

    rc = sr.main(['launching'])

    assert rc == 0
    slug = sr.build_session_slug('unblock', 'df', '2085', 4242)
    expected_dir = sr.record_path_for_slug(slug, root=tmp_path).parent
    assert capsys.readouterr().out.strip() == str(expected_dir)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.LAUNCHING

    assert not (sr.sessions_dir(root=tmp_path) / 'old-terminal').exists()


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


def test_main_launching_fail_soft_when_prune_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mirrors test_main_launching_fail_soft_when_sweep_raises for the bounded
    prune: a fault in reap_stale_records must never raise out of
    _run_launching (it would corrupt the printed record dir spawn-claude.sh
    captures into SESSION_RECORD_DIR), exactly like a fault in the mark sweep.
    """
    _set_env(monkeypatch, _launching_env(tmp_path))

    calls: list[None] = []

    def _boom(*_args: object, **_kwargs: object) -> list[sr.ReapedSessionRecord]:
        calls.append(None)
        raise OSError('prune on fire')

    monkeypatch.setattr(sr, 'reap_stale_records', _boom)

    rc = sr.main(['launching'])

    assert rc == 0
    assert calls  # the prune really was invoked (and really did raise)
    slug = sr.build_session_slug('unblock', 'df', '2085', 4242)
    expected_dir = sr.record_path_for_slug(slug, root=tmp_path).parent
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
# Task 2934 step-9/10: driving the WM-window sweep from `launching`/`reap`
# ---------------------------------------------------------------------------


def test_main_reap_marks_windowless_wm_session_before_deleting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    windowless = _make_record(
        session_slug='reap-windowless',
        status=sr.Status.AWAITING_INPUT,
        launcher_pid=os.getpid(),  # ALIVE -- the pid/TTL sweep alone would never catch this
        display=sr.Display(kind='wm', wm_title='t', wm_window_id='0x1a'),
    )
    sr.write_record(windowless, root=tmp_path)

    def _fake_wmctrl_list(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv, returncode=0, stdout='0x99  0 host  still-open window\n'
        )

    monkeypatch.setattr(sr, '_wmctrl_list', _fake_wmctrl_list)

    rc = sr.main(['reap'])

    assert rc == 0
    # Marked, not deleted, in this SAME pass -- the window sweep ran in the
    # mark phase before reap_stale_records (mark-then-delete order).
    reloaded_dir = sr.record_path_for_slug('reap-windowless', root=tmp_path).parent
    assert reloaded_dir.is_dir()
    reloaded = sr.read_record('reap-windowless', root=tmp_path)
    assert reloaded.status == sr.Status.EXITED
    assert reloaded.exit_code == sr.ORPHAN_EXIT_CODE


def test_main_launching_fail_soft_when_window_sweep_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mirrors test_main_launching_fail_soft_when_sweep_raises for the new WM
    window sweep: a fault must never raise out of _run_launching (it would
    corrupt the printed record dir spawn-claude.sh captures into
    SESSION_RECORD_DIR), exactly like a fault in the orphan mark sweep.
    """
    _set_env(monkeypatch, _launching_env(tmp_path))

    calls: list[None] = []

    def _boom(*_args: object, **_kwargs: object) -> list[sr.SessionRecord]:
        calls.append(None)
        raise OSError('window sweep on fire')

    monkeypatch.setattr(sr, 'mark_windowless_wm_sessions_exited', _boom)

    rc = sr.main(['launching'])

    assert rc == 0
    assert calls  # the sweep really was invoked (and really did raise)
    slug = sr.build_session_slug('unblock', 'df', '2085', 4242)
    expected_dir = sr.record_path_for_slug(slug, root=tmp_path).parent
    assert capsys.readouterr().out.strip() == str(expected_dir)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.LAUNCHING


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


def test_lease_mutation_has_exactly_applied_forced_absent_refused_faulted() -> None:
    values = {member.value for member in sr.LeaseMutation}
    assert values == {'applied', 'forced', 'absent', 'refused', 'faulted'}
    assert sr.LeaseMutation.APPLIED.value == 'applied'
    assert sr.LeaseMutation.FORCED.value == 'forced'
    assert sr.LeaseMutation.ABSENT.value == 'absent'
    assert sr.LeaseMutation.REFUSED.value == 'refused'
    assert sr.LeaseMutation.FAULTED.value == 'faulted'


def test_lease_mutation_is_a_str_enum_like_its_siblings() -> None:
    # Mirrors LeasePolicy/LeaseDecision: the CLI prints `.value` directly and
    # a StrEnum member compares equal to its own string form.
    assert issubclass(sr.LeaseMutation, str)
    assert sr.LeaseMutation.REFUSED == 'refused'


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


# --- _render_contention_message (task 3994 defect 3) ----------------------
#
# The old form was `lease held by <slug> ({alive|dead}, heartbeat <n>s ago)`,
# which reads as SELF-CONTRADICTORY on its most important branch: "dead,
# heartbeat 42s ago" invites the reader to conclude the holder is gone and
# the lease is reclaimable, when a fresh heartbeat means precisely the
# opposite. That exact string misled the 2026-08-08 rotation into
# force-releasing a live holder's lease. Liveness and freshness are two
# INDEPENDENT axes and each must now name itself.

_LIVE_HOLDER = sr.LeaseHolder(
    session_slug='watcher-df-100', pid=4242, start_ts='2026-07-07T12:00:00+00:00'
)


def test_contention_message_live_holder_names_the_slug_and_the_pid() -> None:
    msg = sr._render_contention_message(
        _LIVE_HOLDER, holder_alive=True, age_secs=42.0, policy=sr.LeasePolicy.STAND_DOWN
    )

    assert 'watcher-df-100' in msg
    assert '4242' in msg
    assert 'pid 4242 alive' in msg
    assert msg.endswith('— standing down')


def test_contention_message_dead_pid_but_fresh_heartbeat_says_not_reclaimable() -> None:
    ttl = timedelta(seconds=7200)
    msg = sr._render_contention_message(
        _LIVE_HOLDER,
        holder_alive=False,
        age_secs=1200.0,
        policy=sr.LeasePolicy.STAND_DOWN,
        ttl=ttl,
    )

    # The withdrawn string, pinned as an ANTI-assertion.
    assert '(dead, heartbeat' not in msg
    lowered = msg.lower()
    assert 'not running' in lowered  # the pid axis, named
    assert 'fresh' in lowered  # the heartbeat axis, named
    assert 'not reclaimable' in lowered  # the decision, explained
    assert '6000' in msg  # the remaining TTL, in seconds


def test_contention_message_remaining_ttl_tracks_the_injected_ttl() -> None:
    msg = sr._render_contention_message(
        _LIVE_HOLDER,
        holder_alive=False,
        age_secs=100.0,
        policy=sr.LeasePolicy.STAND_DOWN,
        ttl=timedelta(seconds=900),
    )
    assert '800' in msg


def test_contention_message_dead_pid_past_ttl_is_phrased_as_reclaim_eligible() -> None:
    msg = sr._render_contention_message(
        _LIVE_HOLDER,
        holder_alive=False,
        age_secs=9000.0,
        policy=sr.LeasePolicy.STAND_DOWN,
        ttl=timedelta(seconds=7200),
    )

    assert '(dead, heartbeat' not in msg
    lowered = msg.lower()
    assert 'not running' in lowered
    # Past the TTL the lease WAS reclaim-eligible; we are here only because
    # the reclaim race was lost to someone else, and the message must say so
    # rather than implying the holder is still healthy.
    assert 'reclaim' in lowered
    assert 'not reclaimable' not in lowered


def test_contention_message_corrupt_body_does_not_invent_a_holder() -> None:
    msg = sr._render_contention_message(
        None, holder_alive=False, age_secs=42.0, policy=sr.LeasePolicy.STAND_DOWN
    )

    assert 'unreadable' in msg.lower()
    assert '42s ago' in msg  # freshness is still reported
    assert msg.endswith('— standing down')


@pytest.mark.parametrize(
    ('policy', 'suffix'),
    [
        (sr.LeasePolicy.STAND_DOWN, '— standing down'),
        (sr.LeasePolicy.WARN_AND_PROCEED, '— proceeding anyway'),
    ],
)
@pytest.mark.parametrize(
    ('holder', 'alive', 'age'),
    [
        (_LIVE_HOLDER, True, 42.0),
        (_LIVE_HOLDER, False, 1200.0),
        (_LIVE_HOLDER, False, 9000.0),
        (None, False, 42.0),
    ],
    ids=['live', 'dead-fresh', 'dead-expired', 'corrupt'],
)
def test_contention_message_is_always_one_greppable_line(
    policy: sr.LeasePolicy,
    suffix: str,
    holder: sr.LeaseHolder | None,
    alive: bool,
    age: float,
) -> None:
    msg = sr._render_contention_message(holder, holder_alive=alive, age_secs=age, policy=policy)

    assert '\n' not in msg
    assert msg.endswith(suffix)


def test_claim_lease_dead_holder_within_ttl_reports_the_non_contradictory_message(
    tmp_path: Path,
) -> None:
    # End-to-end: the branch the 2026-08-08 rotation actually hit.
    dead = sr.LeaseHolder(session_slug='watcher-df-dead', pid=_DEAD_PID, start_ts=_NOW.isoformat())
    sr.claim_lease('watcher-df', holder=dead, root=tmp_path, now=_NOW)
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    _set_mtime(lease_path, _NOW, sr.LEASE_HEARTBEAT_TTL - timedelta(minutes=10))

    contender = sr.LeaseHolder(session_slug='watcher-df-200', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = sr.claim_lease('watcher-df', holder=contender, root=tmp_path, now=_NOW)

    assert claim.decision == sr.LeaseDecision.STAND_DOWN
    assert '(dead, heartbeat' not in claim.message
    assert 'not reclaimable' in claim.message.lower()
    assert str(_DEAD_PID) in claim.message


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
    assert claim.message == (
        f'lease held by watcher-df-100 (pid {os.getpid()} alive, heartbeat 42s ago) '
        '— standing down'
    )
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


# --- LeaseClaim.holder_session_state (task 3994 defect 4) -----------------
#
# The escape hatch's evidence. A lease whose holder pid is dead AND whose
# session record is absent/exited is very probably ORPHANED — and a
# stand-down that silently honours it is exactly the "23 pending L2 items,
# no consumer for hours" failure. This field is DIAGNOSTIC ONLY: the lease
# file stays authoritative, because auto-stealing on a dead-looking holder
# is the duplicate-spawn incident path.


def _contend(tmp_path: Path, *, holder_slug: str, holder_pid: int) -> sr.LeaseClaim:
    """Seed a lease for (holder_slug, holder_pid) and return a contender's claim."""
    held = sr.LeaseHolder(session_slug=holder_slug, pid=holder_pid, start_ts=_NOW.isoformat())
    sr.claim_lease('watcher-df', holder=held, root=tmp_path, now=_NOW)
    _set_mtime(sr.lease_path_for_name('watcher-df', root=tmp_path), _NOW, timedelta(seconds=42))
    contender = sr.LeaseHolder(session_slug='watcher-df-200', pid=os.getpid(), start_ts=_NOW.isoformat())
    return sr.claim_lease('watcher-df', holder=contender, root=tmp_path, now=_NOW)


def test_holder_session_state_is_absent_when_the_holder_has_no_record(tmp_path: Path) -> None:
    claim = _contend(tmp_path, holder_slug='watcher-df-100', holder_pid=_DEAD_PID)
    assert claim.holder_session_state == 'absent'


@pytest.mark.parametrize('status', [sr.Status.EXITED, sr.Status.FAILED_TO_START])
def test_holder_session_state_is_exited_for_a_terminal_record(
    tmp_path: Path, status: sr.Status
) -> None:
    sr.write_record(_make_record(session_slug='watcher-df-100', status=status), root=tmp_path)

    claim = _contend(tmp_path, holder_slug='watcher-df-100', holder_pid=_DEAD_PID)

    assert status in sr.TERMINAL_STATUSES
    assert claim.holder_session_state == 'exited'


def test_holder_session_state_is_live_for_a_non_terminal_record(tmp_path: Path) -> None:
    sr.write_record(
        _make_record(session_slug='watcher-df-100', status=sr.Status.RUNNING), root=tmp_path
    )

    claim = _contend(tmp_path, holder_slug='watcher-df-100', holder_pid=_DEAD_PID)

    assert claim.holder_session_state == 'live'


def test_holder_session_state_is_unknown_for_an_unparseable_record(tmp_path: Path) -> None:
    record_path = sr.record_path_for_slug('watcher-df-100', root=tmp_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text('{not valid json')

    claim = _contend(tmp_path, holder_slug='watcher-df-100', holder_pid=_DEAD_PID)

    assert claim.holder_session_state == 'unknown'


def test_holder_session_state_is_unknown_for_a_corrupt_lease_body(tmp_path: Path) -> None:
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_text('{not valid json')
    _set_mtime(lease_path, _NOW, timedelta(seconds=42))

    contender = sr.LeaseHolder(session_slug='watcher-df-200', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = sr.claim_lease('watcher-df', holder=contender, root=tmp_path, now=_NOW)

    assert claim.holder is None
    assert claim.holder_session_state == 'unknown'


def test_holder_session_state_resolution_is_fail_soft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> sr.SessionRecord:
        raise OSError('session registry substrate on fire')

    monkeypatch.setattr(sr, 'read_record', _boom)

    # A session-registry read fault must never break a lease claim.
    claim = _contend(tmp_path, holder_slug='watcher-df-100', holder_pid=_DEAD_PID)

    assert claim.decision == sr.LeaseDecision.STAND_DOWN
    assert claim.holder_session_state == 'unknown'


def test_holder_session_state_of_a_freshly_acquired_lease_is_live(tmp_path: Path) -> None:
    holder = sr.LeaseHolder(session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat())

    claim = sr.claim_lease('watcher-df', holder=holder, root=tmp_path, now=_NOW)

    # The acquirer is itself, and it is by construction running.
    assert claim.acquired is True
    assert claim.holder_session_state == 'live'


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


def test_claim_lease_reap_and_reclaim_emits_structured_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # THREAD 1 hardened detection (task 2796): a reap-and-reclaim displaces
    # whoever currently held the lease. That holder is SUPPOSED to be dead,
    # but if the TTL is ever mis-tuned again this same path would silently
    # steal a live-but-quiet watcher's lease. Make every reap-and-reclaim
    # loud/greppable so any residual displacement of a supposedly-live holder
    # is surfaced rather than silent.
    stale_holder = sr.LeaseHolder(session_slug='watcher-df-dead', pid=_DEAD_PID, start_ts=_NOW.isoformat())
    sr.claim_lease('watcher-df', holder=stale_holder, root=tmp_path, now=_NOW)
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    # Backdate past the (new) TTL so the lease IS reap-eligible.
    age = sr.LEASE_HEARTBEAT_TTL + timedelta(minutes=1)
    _set_mtime(lease_path, _NOW, age)

    new_holder = sr.LeaseHolder(session_slug='watcher-df-new', pid=os.getpid(), start_ts=_NOW.isoformat())
    with caplog.at_level(logging.WARNING, logger=sr.logger.name):
        claim = sr.claim_lease('watcher-df', holder=new_holder, root=tmp_path, now=_NOW)

    # Sanity: the reclaim actually happened (the WARNING is on that path).
    assert claim.acquired is True

    warnings = [
        r for r in caplog.records if r.levelno == logging.WARNING and r.name == sr.logger.name
    ]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    # Names the displaced holder (slug + pid), the observed heartbeat age, and
    # the new holder's slug.
    assert 'watcher-df-dead' in message
    assert str(_DEAD_PID) in message
    assert f'{age.total_seconds():.0f}' in message
    assert 'watcher-df-new' in message


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


def test_claim_lease_keeps_a_quiet_watchers_lease_across_a_full_slice(tmp_path: Path) -> None:
    # The required "duplicate-through" regression (task 2796 THREAD 1): a live
    # interactive watcher holds its lease with `--pid $$` (the ephemeral
    # Bash-tool shell, dead moments after the claim), so _pid_alive is
    # ~always False for a watcher lease and staleness collapses to a pure
    # heartbeat-TTL check. That watcher heartbeats only once per Main Loop
    # cycle, bounded by ONE canonical watcher-rearm.sh `--timeout` slice
    # (3600s). If LEASE_HEARTBEAT_TTL is below that slice, a live-but-quiet
    # holder's lease goes stale->reap-eligible during any quiet slice and a
    # duplicate reaps+reclaims it -- the exact code path that let a
    # live-lease-holder's duplicate spawn through. With TTL raised above the
    # slice, the duplicate must instead STAND DOWN.
    original = sr.LeaseHolder(session_slug='watcher-df-dead', pid=_DEAD_PID, start_ts=_NOW.isoformat())
    sr.claim_lease('watcher-df', holder=original, root=tmp_path, now=_NOW)
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    # Backdate the heartbeat by one full canonical wait slice (3600s).
    _set_mtime(lease_path, _NOW, timedelta(seconds=3600))

    duplicate = sr.LeaseHolder(session_slug='watcher-df-dup', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = sr.claim_lease(
        'watcher-df', holder=duplicate, policy=sr.LeasePolicy.STAND_DOWN, root=tmp_path, now=_NOW
    )

    # RED today: TTL=300s < 3600s slice age, so the live-but-quiet holder's
    # lease is reaped-and-reclaimed and the duplicate ACQUIRES (claim.acquired
    # is True) -- the duplicate-through bug. GREEN after TTL is raised: the
    # duplicate stands down.
    assert claim.acquired is False
    assert claim.decision == sr.LeaseDecision.STAND_DOWN
    assert claim.holder is not None
    assert claim.holder.session_slug == 'watcher-df-dead'
    # No clobber / no reap: the on-disk body still names the ORIGINAL holder.
    assert sr.LeaseHolder.from_json(lease_path.read_text()) == original

    # Invariant guard: the whole regression hinges on TTL exceeding one full
    # canonical slice. If a future edit lowers LEASE_HEARTBEAT_TTL back below
    # 3600s, this assertion fails loudly rather than silently re-opening the
    # gap.
    assert timedelta(seconds=3600) < sr.LEASE_HEARTBEAT_TTL


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

    result = sr.heartbeat_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    assert result is sr.LeaseMutation.APPLIED
    assert lease_path.stat().st_mtime > old_mtime


def test_heartbeat_lease_on_absent_lease_returns_false_without_raising(tmp_path: Path) -> None:
    result = sr.heartbeat_lease('watcher-df', slug='watcher-df-100', root=tmp_path)
    assert result is sr.LeaseMutation.ABSENT


def test_release_lease_removes_the_lease_file(tmp_path: Path) -> None:
    holder = sr.LeaseHolder(session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat())
    sr.claim_lease('watcher-df', holder=holder, root=tmp_path, now=_NOW)
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)

    result = sr.release_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    assert result is sr.LeaseMutation.APPLIED
    assert not lease_path.exists()


def test_release_lease_is_idempotent_on_a_second_call(tmp_path: Path) -> None:
    holder = sr.LeaseHolder(session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat())
    sr.claim_lease('watcher-df', holder=holder, root=tmp_path, now=_NOW)

    first = sr.release_lease('watcher-df', slug='watcher-df-100', root=tmp_path)
    second = sr.release_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    assert first is sr.LeaseMutation.APPLIED
    assert second is sr.LeaseMutation.ABSENT


# --- heartbeat_lease: ownership (task 3994 defect 1) -----------------------


def test_heartbeat_lease_by_a_stranger_is_refused_and_the_mtime_is_untouched(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lease_path = _seed_lease(tmp_path)
    _set_mtime(lease_path, _NOW, timedelta(hours=1))
    backdated_mtime = lease_path.stat().st_mtime

    with caplog.at_level(logging.ERROR):
        result = sr.heartbeat_lease('watcher-df', slug='watcher-df-STRANGER', root=tmp_path)

    assert result is sr.LeaseMutation.REFUSED
    # THE assertion: an unrelated caller can no longer keep someone else's
    # lease structurally unreapable by refreshing its heartbeat clock.
    assert lease_path.stat().st_mtime == backdated_mtime
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors
    blob = ' '.join(r.getMessage() for r in errors)
    assert 'watcher-df-100' in blob
    assert 'watcher-df-STRANGER' in blob


def test_heartbeat_lease_on_an_absent_lease_is_absent_and_quiet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR):
        result = sr.heartbeat_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    assert result is sr.LeaseMutation.ABSENT
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


def test_heartbeat_lease_on_a_corrupt_body_is_refused_with_mtime_untouched(
    tmp_path: Path,
) -> None:
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_text('{not valid json')
    _set_mtime(lease_path, _NOW, timedelta(hours=1))
    backdated_mtime = lease_path.stat().st_mtime

    result = sr.heartbeat_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    assert result is sr.LeaseMutation.REFUSED
    assert lease_path.stat().st_mtime == backdated_mtime


def test_heartbeat_lease_with_force_overrides_a_stranger_refusal_loudly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lease_path = _seed_lease(tmp_path)
    _set_mtime(lease_path, _NOW, timedelta(hours=1))
    backdated_mtime = lease_path.stat().st_mtime

    with caplog.at_level(logging.WARNING):
        result = sr.heartbeat_lease(
            'watcher-df', slug='watcher-df-STRANGER', force=True, root=tmp_path
        )

    assert result is sr.LeaseMutation.FORCED
    assert lease_path.stat().st_mtime > backdated_mtime
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings
    blob = ' '.join(r.getMessage() for r in warnings)
    assert 'watcher-df-100' in blob
    assert 'watcher-df-STRANGER' in blob


def test_heartbeat_lease_faults_soft_when_utime_itself_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _seed_lease(tmp_path)

    def _boom_utime(*_args: object, **_kwargs: object) -> None:
        raise OSError('simulated permission error')

    monkeypatch.setattr(sr.os, 'utime', _boom_utime)

    with caplog.at_level(logging.ERROR):
        result = sr.heartbeat_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    # Fail-soft: a lease-substrate fault never raises into a watcher's loop.
    assert result is sr.LeaseMutation.FAULTED
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_a_strangers_refused_heartbeat_leaves_a_dead_holders_lease_reapable(
    tmp_path: Path,
) -> None:
    # The structural-unreapability defect, pinned end to end: before task 3994
    # any caller could bump a dead holder's lease mtime forever, so the
    # (dead pid AND aged heartbeat) staleness AND-guard could never fire and
    # the lease outlived its holder indefinitely.
    lease_path = _seed_lease(tmp_path, slug='watcher-df-dead', pid=_DEAD_PID)
    _set_mtime(lease_path, _NOW, sr.LEASE_HEARTBEAT_TTL + timedelta(minutes=1))

    assert sr.heartbeat_lease('watcher-df', slug='watcher-df-STRANGER', root=tmp_path) is (
        sr.LeaseMutation.REFUSED
    )

    reaped = sr.reap_stale_leases(root=tmp_path, now=_NOW)

    assert {r.lease_name: r.reason for r in reaped} == {'watcher-df': 'stale_pid'}
    assert not lease_path.exists()


# --- release_lease: ownership (task 3994 defect 1) -------------------------


def _seed_lease(
    tmp_path: Path, name: str = 'watcher-df', *, slug: str = 'watcher-df-100', pid: int | None = None
) -> Path:
    """Claim *name* for *slug* and return its on-disk lease path."""
    holder = sr.LeaseHolder(
        session_slug=slug, pid=os.getpid() if pid is None else pid, start_ts=_NOW.isoformat()
    )
    sr.claim_lease(name, holder=holder, root=tmp_path, now=_NOW)
    return sr.lease_path_for_name(name, root=tmp_path)


def test_release_lease_by_a_stranger_is_refused_and_the_lease_survives(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lease_path = _seed_lease(tmp_path)
    original_body = lease_path.read_text()

    with caplog.at_level(logging.ERROR):
        result = sr.release_lease('watcher-df', slug='watcher-df-STRANGER', root=tmp_path)

    assert result is sr.LeaseMutation.REFUSED
    assert lease_path.is_file()
    # Byte-identical: a refused release must not touch the holder's body.
    assert lease_path.read_text() == original_body
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, 'a refused release must be loud (INV-2 structured-facts-at-failure)'
    blob = ' '.join(r.getMessage() for r in errors)
    # Both parties named: the real holder AND the would-be releaser.
    assert 'watcher-df-100' in blob
    assert 'watcher-df-STRANGER' in blob


def test_release_lease_on_an_absent_lease_is_absent_and_quiet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR):
        result = sr.release_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    assert result is sr.LeaseMutation.ABSENT
    # An idempotent release is not a failure -- it must not cry ERROR.
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


def test_release_lease_on_a_corrupt_body_is_refused_and_the_file_survives(
    tmp_path: Path,
) -> None:
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_text('{not valid json')

    result = sr.release_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    # Fail TOWARD held, mirroring _read_lease_holder_state's documented rule.
    assert result is sr.LeaseMutation.REFUSED
    assert lease_path.read_text() == '{not valid json'


def test_release_lease_with_force_overrides_a_stranger_refusal_loudly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lease_path = _seed_lease(tmp_path)

    with caplog.at_level(logging.WARNING):
        result = sr.release_lease(
            'watcher-df', slug='watcher-df-STRANGER', force=True, root=tmp_path
        )

    assert result is sr.LeaseMutation.FORCED
    assert not lease_path.exists()
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, 'an operator force must be attributable'
    blob = ' '.join(r.getMessage() for r in warnings)
    assert 'watcher-df-100' in blob  # the displaced holder
    assert 'watcher-df-STRANGER' in blob  # who forced it


def test_release_lease_with_force_on_a_corrupt_body_removes_it(tmp_path: Path) -> None:
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_text('{not valid json')

    result = sr.release_lease('watcher-df', slug='watcher-df-100', force=True, root=tmp_path)

    assert result is sr.LeaseMutation.FORCED
    assert not lease_path.exists()


def test_release_lease_faults_soft_when_the_unlink_itself_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    lease_path = _seed_lease(tmp_path)
    real_unlink = Path.unlink

    def _flaky_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == lease_path:
            raise OSError('simulated permission error')
        real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', _flaky_unlink)

    with caplog.at_level(logging.ERROR):
        result = sr.release_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    # Fail-soft: a lease-substrate fault never raises into a watcher's loop.
    assert result is sr.LeaseMutation.FAULTED
    assert lease_path.is_file()
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


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
# lease_status / lease-show (task 3994 defect 3)
#
# `cat <lease>` CANNOT show freshness: the body carries an immutable
# `start_ts` and the heartbeat lives only in the file mtime, so the archaeology
# a human actually needs is `cat` + `stat -c %y` + a TTL comparison done by
# hand. That is what misled the 2026-08-08 rotation. lease_status computes all
# of it through the same _read_lease_holder_state claim_lease decides on --
# one reader, one clock -- and lease-show prints it.
# ---------------------------------------------------------------------------


def _seed_lease(tmp_path: Path, *, slug: str = 'watcher-df-100', pid: int | None = None) -> Path:
    """Claim `watcher-df` for *slug*/*pid* under tmp_path; return its lease path."""
    holder = sr.LeaseHolder(
        session_slug=slug, pid=os.getpid() if pid is None else pid, start_ts=_NOW.isoformat()
    )
    sr.claim_lease('watcher-df', holder=holder, root=tmp_path, now=_NOW)
    return sr.lease_path_for_name('watcher-df', root=tmp_path)


def test_lease_status_reports_the_holder_and_the_heartbeat_clock(tmp_path: Path) -> None:
    path = _seed_lease(tmp_path)
    _set_mtime(path, _NOW, timedelta(minutes=30))

    status = sr.lease_status('watcher-df', root=tmp_path, now=_NOW)

    assert status.name == 'watcher-df'
    assert status.state == 'held'
    assert status.holder_slug == 'watcher-df-100'
    assert status.holder_pid == os.getpid()
    assert status.holder_pid_alive is True
    # Freshness is derived from the MTIME, the axis `cat` cannot show.
    assert int(status.heartbeat_age_secs) == 1800
    assert datetime.fromisoformat(status.heartbeat_ts).replace(microsecond=0) == _NOW - timedelta(
        minutes=30
    )
    assert status.reclaimable is False
    assert status.holder_session == 'absent'  # no session record was written


def test_lease_status_heartbeat_age_tracks_the_mtime(tmp_path: Path) -> None:
    path = _seed_lease(tmp_path)

    for age in (timedelta(seconds=5), timedelta(minutes=17), timedelta(hours=3)):
        _set_mtime(path, _NOW, age)
        status = sr.lease_status('watcher-df', root=tmp_path, now=_NOW)
        assert int(status.heartbeat_age_secs) == int(age.total_seconds())


@pytest.mark.parametrize(
    ('age', 'expected'),
    [
        (sr.LEASE_HEARTBEAT_TTL - timedelta(seconds=1), False),
        (sr.LEASE_HEARTBEAT_TTL, False),
        (sr.LEASE_HEARTBEAT_TTL + timedelta(seconds=1), True),
    ],
)
def test_lease_status_reclaimable_flips_exactly_at_the_ttl_for_a_dead_pid(
    tmp_path: Path, age: timedelta, expected: bool
) -> None:
    path = _seed_lease(tmp_path, pid=_DEAD_PID)
    _set_mtime(path, _NOW, age)

    status = sr.lease_status('watcher-df', root=tmp_path, now=_NOW)

    assert status.holder_pid_alive is False
    # Exactly claim_lease/reap_stale_leases' own predicate: staleness needs
    # BOTH a dead pid AND a heartbeat STRICTLY past the TTL. The display and
    # the decision cannot disagree, because they are the same computation.
    assert status.reclaimable is expected


def test_lease_status_never_reports_a_live_holder_reclaimable_at_any_age(tmp_path: Path) -> None:
    path = _seed_lease(tmp_path)  # our own, provably-live, pid
    _set_mtime(path, _NOW, timedelta(days=30))

    status = sr.lease_status('watcher-df', root=tmp_path, now=_NOW)

    assert status.holder_pid_alive is True
    assert int(status.heartbeat_age_secs) == 30 * 24 * 3600
    # A live holder is never reclaimable however quiet it has been -- the
    # inverse reading is the duplicate-spawn incident.
    assert status.reclaimable is False


@pytest.mark.parametrize(
    ('record_status', 'expected'),
    [(sr.Status.RUNNING, 'live'), (sr.Status.EXITED, 'exited')],
)
def test_lease_status_reports_the_holders_own_session_state(
    tmp_path: Path, record_status: sr.Status, expected: str
) -> None:
    _seed_lease(tmp_path, pid=_DEAD_PID)
    sr.write_record(
        _make_record(session_slug='watcher-df-100', status=record_status), root=tmp_path
    )

    status = sr.lease_status('watcher-df', root=tmp_path, now=_NOW)

    assert status.holder_session == expected


def test_lease_status_on_an_absent_lease_reports_state_absent(tmp_path: Path) -> None:
    status = sr.lease_status('watcher-df', root=tmp_path, now=_NOW)

    assert status.name == 'watcher-df'
    assert status.state == 'absent'
    assert status.holder_slug is None
    assert status.holder_pid is None
    assert status.holder_pid_alive is False
    assert status.reclaimable is False


def test_lease_status_on_a_corrupt_body_is_unreadable_but_still_dates_the_heartbeat(
    tmp_path: Path,
) -> None:
    path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{not valid json')
    _set_mtime(path, _NOW, timedelta(minutes=42))

    status = sr.lease_status('watcher-df', root=tmp_path, now=_NOW)

    # Fail-soft: an unreadable body is REPORTED, never raised, and never
    # rendered as a fake holder -- but its freshness is still real, which is
    # exactly what decides whether it is reapable.
    assert status.state == 'unreadable'
    assert status.holder_slug is None
    assert status.holder_pid is None
    assert int(status.heartbeat_age_secs) == 2520
    assert status.holder_session == 'unknown'


def test_lease_status_is_read_only(tmp_path: Path) -> None:
    path = _seed_lease(tmp_path)
    _set_mtime(path, _NOW, timedelta(hours=1))
    mtime_before = path.stat().st_mtime
    body_before = path.read_text()

    sr.lease_status('watcher-df', root=tmp_path, now=_NOW)

    # Inspecting a lease must never be a heartbeat: an operator running
    # lease-show cannot be allowed to keep a dead holder's lease unreapable.
    assert path.stat().st_mtime == mtime_before
    assert path.read_text() == body_before


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
    assert re.search(
        r'lease held by \S+ \(pid \d+ alive, heartbeat \d+s ago\) — standing down', lines[1]
    )
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

    rc = sr.main(['lease-heartbeat', '--name', 'watcher-df', '--slug', 'watcher-df-100'])

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

    rc = sr.main(['lease-release', '--name', 'watcher-df', '--slug', 'watcher-df-100'])

    assert rc == 0
    assert not lease_path.exists()


# --- CLI holder_liveness orphan signal (task 3994 defect 4) ---------------


def _contend_via_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], *, holder_pid: int
) -> list[str]:
    """Seed a lease held by watcher-df-100/holder_pid, contend it, return stdout lines."""
    sr.main(['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100', '--pid', str(holder_pid)])
    capsys.readouterr()
    rc = sr.main(
        ['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-200', '--pid', str(os.getpid())]
    )
    assert rc == 0
    return capsys.readouterr().out.splitlines()


def test_main_lease_claim_orphan_signal_is_additive_after_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    lines = _contend_via_cli(tmp_path, capsys, holder_pid=os.getpid())

    # `decision=` STAYS line 1 and the message line 2, so a skill parser that
    # reads only the first two lines is provably unaffected by the addition.
    assert lines[0] == 'decision=stand-down'
    assert lines[1].startswith('lease held by watcher-df-100')
    assert lines[2] == 'holder_liveness=held'


def test_main_lease_claim_reports_orphaned_for_a_dead_pid_with_no_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    lines = _contend_via_cli(tmp_path, capsys, holder_pid=_DEAD_PID)

    assert lines[0] == 'decision=stand-down'
    assert lines[-1] == 'holder_liveness=orphaned'


def test_main_lease_claim_reports_orphaned_for_a_dead_pid_with_an_exited_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    sr.write_record(
        _make_record(session_slug='watcher-df-100', status=sr.Status.EXITED), root=tmp_path
    )

    lines = _contend_via_cli(tmp_path, capsys, holder_pid=_DEAD_PID)

    assert lines[-1] == 'holder_liveness=orphaned'


def test_main_lease_claim_reports_held_for_a_dead_pid_with_a_live_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    sr.write_record(
        _make_record(session_slug='watcher-df-100', status=sr.Status.RUNNING), root=tmp_path
    )

    lines = _contend_via_cli(tmp_path, capsys, holder_pid=_DEAD_PID)

    # The pid may be a stale write; with a live record, do NOT cry orphan.
    assert lines[-1] == 'holder_liveness=held'


def test_main_lease_claim_reports_held_for_an_unreadable_lease_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_text('{not valid json')

    rc = sr.main(
        ['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-200', '--pid', str(os.getpid())]
    )

    assert rc == 0
    # 'unknown' must never be promoted to an orphan finding: fail toward held.
    assert capsys.readouterr().out.splitlines()[-1] == 'holder_liveness=held'


# --- resolve_session_pid (task 3994 defect 2) ------------------------------


def test_session_pid_env_names_claude_pid() -> None:
    assert sr.SESSION_PID_ENV == 'CLAUDE_PID'


def test_resolve_session_pid_prefers_claude_pid(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        pid = sr.resolve_session_pid({'CLAUDE_PID': '1348600'})

    assert pid == 1348600
    # The happy path is silent: a resolvable session pid is not a degradation.
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.parametrize(
    'env',
    [
        {},
        {'CLAUDE_PID': ''},
        {'CLAUDE_PID': '   '},
        {'CLAUDE_PID': 'not-a-pid'},
        {'CLAUDE_PID': '0'},
        {'CLAUDE_PID': '-1'},
    ],
    ids=['unset', 'empty', 'blank', 'non-numeric', 'zero', 'negative'],
)
def test_resolve_session_pid_falls_back_loudly(
    env: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        pid = sr.resolve_session_pid(env)

    # Fallback, never a raise -- and never a positive-looking lie.
    assert isinstance(pid, int)
    assert pid > 0
    # LOUD, not silent: the repo's loud-over-silent-degradation norm (INV-2).
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings
    blob = ' '.join(r.getMessage() for r in warnings).lower()
    assert 'degraded' in blob


def test_resolve_session_pid_defaults_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('CLAUDE_PID', '424242')
    assert sr.resolve_session_pid() == 424242


def test_resolve_session_pid_falls_back_to_getppid_when_getsid_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom_getsid(*_args: object) -> int:
        raise OSError('no getsid on this platform')

    monkeypatch.setattr(sr.os, 'getsid', _boom_getsid)

    assert sr.resolve_session_pid({}) == os.getppid()


def test_main_lease_claim_without_pid_writes_the_resolved_session_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    # Deliberately NOT this process's own pid: that is what the old
    # `--pid` default was, so an env-blind implementation would still pass.
    monkeypatch.setenv('CLAUDE_PID', str(_DEAD_PID))

    rc = sr.main(['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100'])

    assert rc == 0
    # The correct pid is STRUCTURAL: it no longer depends on every skill doc
    # getting one shell token right (the `--pid $$` defect).
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    assert sr.LeaseHolder.from_json(lease_path.read_text()).pid == _DEAD_PID


def test_main_lease_claim_explicit_pid_still_overrides_the_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(os.getpid()))

    rc = sr.main(
        ['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100', '--pid', str(_DEAD_PID)]
    )

    assert rc == 0
    # The operator-override path operators already used by hand.
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    assert sr.LeaseHolder.from_json(lease_path.read_text()).pid == _DEAD_PID


# --- CLI ownership on the mutating verbs (task 3994 defect 1) -------------


def _claim_via_cli(slug: str = 'watcher-df-100', pid: int | None = None) -> None:
    sr.main(
        [
            'lease-claim',
            '--name',
            'watcher-df',
            '--slug',
            slug,
            '--pid',
            str(os.getpid() if pid is None else pid),
        ]
    )


@pytest.mark.parametrize('verb', ['lease-heartbeat', 'lease-release'])
def test_main_lease_mutating_verbs_require_slug(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, verb: str
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    _claim_via_cli()

    # A pre-3994 invocation must now FAIL LOUDLY rather than silently
    # evicting/refreshing a lease it may not hold. argparse exits 2.
    with pytest.raises(SystemExit) as excinfo:
        sr.main([verb, '--name', 'watcher-df'])

    assert excinfo.value.code == 2


@pytest.mark.parametrize('verb', ['lease-heartbeat', 'lease-release'])
def test_main_lease_mutating_verbs_print_result_applied_for_the_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    verb: str,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    _claim_via_cli()
    capsys.readouterr()

    rc = sr.main([verb, '--name', 'watcher-df', '--slug', 'watcher-df-100'])

    assert rc == 0
    # Mirrors lease-claim's `decision=` convention: one parse rule for all
    # four lease verbs.
    assert capsys.readouterr().out.splitlines()[0] == 'result=applied'


@pytest.mark.parametrize('verb', ['lease-heartbeat', 'lease-release'])
def test_main_lease_mutating_verbs_refuse_a_stranger_without_changing_rc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    verb: str,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    _claim_via_cli()
    capsys.readouterr()
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    original_body = lease_path.read_text()

    rc = sr.main([verb, '--name', 'watcher-df', '--slug', 'watcher-df-STRANGER'])

    # Fail-soft: a refusal is an OUTCOME, not an exit-code change.
    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == 'result=refused'
    # The human line must name the REAL holder, so an agent reading stdout
    # learns who owns it rather than only that it was told no.
    assert 'watcher-df-100' in lines[1]
    assert lease_path.read_text() == original_body


@pytest.mark.parametrize('verb', ['lease-heartbeat', 'lease-release'])
def test_main_lease_mutating_verbs_force_overrides_a_stranger_refusal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    verb: str,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    _claim_via_cli()
    capsys.readouterr()

    rc = sr.main([verb, '--name', 'watcher-df', '--slug', 'watcher-df-STRANGER', '--force'])

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == 'result=forced'
    assert 'watcher-df-100' in lines[1]


@pytest.mark.parametrize('verb', ['lease-heartbeat', 'lease-release'])
def test_main_lease_mutating_verbs_report_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    verb: str,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main([verb, '--name', 'watcher-df', '--slug', 'watcher-df-100'])

    assert rc == 0
    assert capsys.readouterr().out.splitlines()[0] == 'result=absent'


# --- CLI lease-show (task 3994 defect 3) ----------------------------------


def _parse_kv(out: str) -> dict[str, str]:
    """Parse lease-show's `key=value` lines -- the whole point of the verb."""
    return dict(line.split('=', 1) for line in out.splitlines() if '=' in line)


def test_main_lease_show_prints_machine_readable_fields_for_a_held_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    _claim_via_cli()
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    _set_mtime(lease_path, datetime.now(UTC), timedelta(hours=1))
    capsys.readouterr()

    rc = sr.main(['lease-show', '--name', 'watcher-df'])

    assert rc == 0
    fields = _parse_kv(capsys.readouterr().out)
    assert fields['name'] == 'watcher-df'
    assert fields['state'] == 'held'
    assert fields['holder_slug'] == 'watcher-df-100'
    assert fields['holder_pid'] == str(os.getpid())
    assert fields['holder_pid_alive'] == 'true'
    assert fields['reclaimable'] == 'false'
    assert fields['holder_session'] == 'absent'
    # The freshness clock `cat <lease>` cannot show, replacing `stat -c %y`.
    assert abs(int(fields['heartbeat_age_secs']) - 3600) <= 5
    assert datetime.fromisoformat(fields['heartbeat_ts']).tzinfo is not None


def test_main_lease_show_reports_a_dead_and_expired_holder_reclaimable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    _claim_via_cli(pid=_DEAD_PID)
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    _set_mtime(lease_path, datetime.now(UTC), sr.LEASE_HEARTBEAT_TTL + timedelta(minutes=1))
    capsys.readouterr()

    rc = sr.main(['lease-show', '--name', 'watcher-df'])

    assert rc == 0
    fields = _parse_kv(capsys.readouterr().out)
    assert fields['holder_pid_alive'] == 'false'
    assert fields['reclaimable'] == 'true'
    assert fields['holder_session'] == 'absent'


def test_main_lease_show_on_an_absent_lease_reports_state_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main(['lease-show', '--name', 'watcher-df'])

    assert rc == 0
    assert 'state=absent' in capsys.readouterr().out.splitlines()


def test_main_lease_show_on_a_corrupt_body_is_fail_soft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_text('{not valid json')
    _set_mtime(lease_path, datetime.now(UTC), timedelta(hours=1))

    rc = sr.main(['lease-show', '--name', 'watcher-df'])

    assert rc == 0
    fields = _parse_kv(capsys.readouterr().out)
    # Marks the body unreadable rather than asserting a fake holder -- and
    # still reports a REAL heartbeat age, the axis that decides reapability.
    assert fields['state'] == 'unreadable'
    assert abs(int(fields['heartbeat_age_secs']) - 3600) <= 5
    assert lease_path.is_file()  # read-only: inspection never reaps


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


def test_main_write_decision_stamps_escalations_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A watcher stamps the queue its --escalation-id belongs to, so the
    fleet-global decision can later be joined back to the right per-queue
    escalation-id namespace (task 3528). The verb must store the NORMALIZED
    form, not the raw argv string: a watcher invoking with a relative/dotted
    spelling must still match a reaper passing the absolute one.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    recon = tmp_path / 'recon'
    recon.mkdir()
    (tmp_path / 'sub').mkdir()
    dotted = str(tmp_path / 'sub' / '..' / 'recon') + '/'

    rc = sr.main(
        [
            'write-decision',
            '--id',
            'dec-q',
            '--project',
            'df',
            '--text',
            'q',
            '--escalation-id',
            'esc-1',
            '--escalations-dir',
            dotted,
        ]
    )

    assert rc == 0
    listed = sr.list_decisions(root=tmp_path)
    assert len(listed) == 1
    assert listed[0].escalations_dir == sr.normalize_escalations_dir(recon)
    assert Path(listed[0].escalations_dir).is_absolute()


def test_main_write_decision_escalations_dir_defaults_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Omitting --escalations-dir yields '' on the filed record: the flag is
    optional, and a queue-less record keeps today's project-only-scoped
    reaper behaviour (mirrors --severity's default at
    test_main_write_decision_severity_defaults_empty).
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main(['write-decision', '--id', 'dec-noq', '--project', 'df', '--text', 'q'])

    assert rc == 0
    listed = sr.list_decisions(root=tmp_path)
    assert len(listed) == 1
    assert listed[0].escalations_dir == ''


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


# ---------------------------------------------------------------------------
# Step-6: decision reaper (C8 close-on-resolve)
# ---------------------------------------------------------------------------


def test_normalize_escalations_dir_makes_a_relative_path_absolute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A queue passed relative to the watcher's cwd must be stored/compared as
    an absolute path, so a reaper invoked from a different cwd still matches.
    """
    (tmp_path / 'esc').mkdir()
    monkeypatch.chdir(tmp_path)

    assert sr.normalize_escalations_dir('esc') == str((tmp_path / 'esc').resolve())


def test_normalize_escalations_dir_collapses_equivalent_spellings(tmp_path: Path) -> None:
    """The whole point of the helper: two spellings of the SAME queue dir must
    produce ONE identical string, so the reaper's queue guard compares equal
    for a decision stamped via a dotted/trailing-slash spelling.
    """
    esc = tmp_path / 'esc'
    esc.mkdir()
    (tmp_path / 'sub').mkdir()
    dotted = str(tmp_path / 'sub' / '..' / 'esc') + '/'

    assert sr.normalize_escalations_dir(dotted) == sr.normalize_escalations_dir(esc)


def test_normalize_escalations_dir_empty_and_blank_are_the_unset_sentinel() -> None:
    """'' means "unset/legacy" and is never a path; a whitespace-only value
    from a sloppy shell interpolation must collapse to the same sentinel
    rather than normalizing to the cwd.
    """
    assert sr.normalize_escalations_dir('') == ''
    assert sr.normalize_escalations_dir('   ') == ''
    assert sr.normalize_escalations_dir('\t\n') == ''


def test_normalize_escalations_dir_expands_user(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Watchers write queue paths as ~/... in SKILL.md snippets; those must
    expand, or a stamped record would never match a reaper's absolute path.
    """
    monkeypatch.setenv('HOME', str(tmp_path))

    assert sr.normalize_escalations_dir('~/data/escalations') == str(
        (tmp_path / 'data' / 'escalations').resolve()
    )


def test_normalize_escalations_dir_nonexistent_dir_still_normalizes(tmp_path: Path) -> None:
    """Path.resolve() is non-strict in Python 3.11+: a well-formed queue path
    that does not exist yet (a project checked out but never escalated, or a
    record migrated between machines) still normalizes rather than faulting.
    """
    missing = tmp_path / 'never' / 'created'

    assert sr.normalize_escalations_dir(missing) == str(missing)


def test_normalize_escalations_dir_fail_soft_on_unresolvable_value() -> None:
    """Fail-soft, matching the module's contract for helpers a C8 watch loop
    calls directly: a value the OS cannot resolve (embedded NUL) degrades to
    the raw string instead of raising into the caller. The raw string simply
    won't match any real queue, which is the fail-OPEN direction.
    """
    assert sr.normalize_escalations_dir('a\x00b') == 'a\x00b'


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
    assert sr.UNKNOWN_QUEUE
    assert sr.UNKNOWN_QUEUE != ''
    assert not sr.UNKNOWN_QUEUE.startswith('/')


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
    from_a = sr.normalize_escalations_dir(sr.UNKNOWN_QUEUE)
    monkeypatch.chdir(tmp_path / 'b')
    from_b = sr.normalize_escalations_dir(sr.UNKNOWN_QUEUE)

    assert from_a == sr.UNKNOWN_QUEUE
    assert from_b == sr.UNKNOWN_QUEUE


def test_read_escalation_status_reads_queue_root_file(tmp_path: Path) -> None:
    """A still-pending escalation lives directly under the queue root."""
    escalations_dir = tmp_path / 'escalations'
    escalations_dir.mkdir()
    (escalations_dir / 'esc-2303-2.json').write_text(
        json.dumps({'status': 'pending', 'id': 'esc-2303-2'})
    )

    assert sr.read_escalation_status(escalations_dir, 'esc-2303-2') == 'pending'


def test_read_escalation_status_falls_back_to_archive(tmp_path: Path) -> None:
    """A resolved/dismissed escalation has been moved into archive/YYYY-MM-DD/
    by escalation.queue._archive_resolved; read_escalation_status must find
    it there when the queue-root file is absent.
    """
    escalations_dir = tmp_path / 'escalations'
    archive_dir = escalations_dir / 'archive' / '2026-07-16'
    archive_dir.mkdir(parents=True)
    (archive_dir / 'esc-2303-2.json').write_text(json.dumps({'status': 'resolved'}))

    assert sr.read_escalation_status(escalations_dir, 'esc-2303-2') == 'resolved'


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

    assert sr.read_escalation_status(escalations_dir, 'esc-2303-2') == 'resolved'


def test_read_escalation_status_unknown_id_returns_none(tmp_path: Path) -> None:
    """No file at the root and none in the archive -> None, not a raise."""
    escalations_dir = tmp_path / 'escalations'
    escalations_dir.mkdir()

    assert sr.read_escalation_status(escalations_dir, 'esc-does-not-exist') is None


def test_read_escalation_status_corrupt_body_returns_none(tmp_path: Path) -> None:
    """A corrupt/non-JSON escalation body must fail soft, returning None."""
    escalations_dir = tmp_path / 'escalations'
    escalations_dir.mkdir()
    (escalations_dir / 'esc-bad.json').write_text('{not valid json')

    assert sr.read_escalation_status(escalations_dir, 'esc-bad') is None


def test_reap_answered_decisions_matrix(tmp_path: Path) -> None:
    """One reap_answered_decisions call exercises the full close-on-resolve
    matrix: resolved->ANSWERED, dismissed->DROPPED, pending/unknown->left
    OPEN, an already-closed decision is skipped WITHOUT consulting the
    callback, and a decision with no escalation_id is skipped WITHOUT
    consulting the callback either.
    """
    sr.write_decision(
        _make_decision(id='dec-resolved', escalation_id='esc-1', state=sr.DecisionState.OPEN),
        root=tmp_path,
    )
    sr.write_decision(
        _make_decision(id='dec-dismissed', escalation_id='esc-2', state=sr.DecisionState.OPEN),
        root=tmp_path,
    )
    sr.write_decision(
        _make_decision(id='dec-pending', escalation_id='esc-3', state=sr.DecisionState.OPEN),
        root=tmp_path,
    )
    sr.write_decision(
        _make_decision(id='dec-unknown', escalation_id='esc-4', state=sr.DecisionState.OPEN),
        root=tmp_path,
    )
    sr.write_decision(
        _make_decision(
            id='dec-already-answered', escalation_id='esc-5', state=sr.DecisionState.ANSWERED
        ),
        root=tmp_path,
    )
    sr.write_decision(
        _make_decision(
            id='dec-already-dropped', escalation_id='esc-6', state=sr.DecisionState.DROPPED
        ),
        root=tmp_path,
    )
    sr.write_decision(
        _make_decision(id='dec-no-escalation', escalation_id=None, state=sr.DecisionState.OPEN),
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

    def fake_status(decision: sr.DecisionRecord) -> str | None:
        consulted.append(decision.id)
        if decision.escalation_id is None:
            return None
        return statuses.get(decision.escalation_id)

    reaped = sr.reap_answered_decisions(root=tmp_path, escalation_status=fake_status)

    reaped_by_id = {r.id: (r.escalation_id, r.new_state) for r in reaped}
    assert reaped_by_id == {
        'dec-resolved': ('esc-1', 'answered'),
        'dec-dismissed': ('esc-2', 'dropped'),
    }
    # Only the still-OPEN, escalation-bearing decisions are ever passed to
    # the callback -- already-closed and no-escalation_id decisions are
    # skipped before it is invoked.
    assert set(consulted) == {'dec-resolved', 'dec-dismissed', 'dec-pending', 'dec-unknown'}

    listed = {d.id: d.state for d in sr.list_decisions(root=tmp_path)}
    assert listed['dec-resolved'] == sr.DecisionState.ANSWERED
    assert listed['dec-dismissed'] == sr.DecisionState.DROPPED
    assert listed['dec-pending'] == sr.DecisionState.OPEN
    assert listed['dec-unknown'] == sr.DecisionState.OPEN
    assert listed['dec-already-answered'] == sr.DecisionState.ANSWERED
    assert listed['dec-already-dropped'] == sr.DecisionState.DROPPED
    assert listed['dec-no-escalation'] == sr.DecisionState.OPEN


# ---------------------------------------------------------------------------
# CLI reap-decisions verb (Fleet Cockpit C8: close-on-resolve driver)
# ---------------------------------------------------------------------------


def test_main_reap_decisions_closes_answered_from_archived_escalation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An escalation archived (resolved) closes its OPEN decision to
    ANSWERED, and the closed id is echoed on stdout.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    escalations_dir = tmp_path / 'esc'
    archive_dir = escalations_dir / 'archive' / '2026-07-16'
    archive_dir.mkdir(parents=True)
    (archive_dir / 'esc-resolved.json').write_text(json.dumps({'status': 'resolved'}))
    sr.write_decision(
        _make_decision(
            id='dec-cli-resolved',
            project='df',
            escalation_id='esc-resolved',
            state=sr.DecisionState.OPEN,
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', str(escalations_dir)])

    assert rc == 0
    listed = {d.id: d.state for d in sr.list_decisions(root=tmp_path)}
    assert listed['dec-cli-resolved'] == sr.DecisionState.ANSWERED
    assert 'dec-cli-resolved' in capsys.readouterr().out


def test_main_reap_decisions_leaves_pending_escalation_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A still-pending (queue-root) escalation leaves its decision OPEN."""
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    escalations_dir = tmp_path / 'esc'
    escalations_dir.mkdir(parents=True)
    (escalations_dir / 'esc-pending.json').write_text(json.dumps({'status': 'pending'}))
    sr.write_decision(
        _make_decision(
            id='dec-cli-pending',
            project='df',
            escalation_id='esc-pending',
            state=sr.DecisionState.OPEN,
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', str(escalations_dir)])

    assert rc == 0
    listed = {d.id: d.state for d in sr.list_decisions(root=tmp_path)}
    assert listed['dec-cli-pending'] == sr.DecisionState.OPEN


def test_main_reap_decisions_scopes_to_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A decision from a DIFFERENT project is left OPEN even though its
    escalation has resolved -- decisions are fleet-global but escalations
    are per-project, so reap-decisions is a per-project driver.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    escalations_dir = tmp_path / 'esc'
    archive_dir = escalations_dir / 'archive' / '2026-07-16'
    archive_dir.mkdir(parents=True)
    (archive_dir / 'esc-other-project.json').write_text(json.dumps({'status': 'resolved'}))
    sr.write_decision(
        _make_decision(
            id='dec-other-project',
            project='not-df',
            escalation_id='esc-other-project',
            state=sr.DecisionState.OPEN,
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', str(escalations_dir)])

    assert rc == 0
    listed = {d.id: d.state for d in sr.list_decisions(root=tmp_path)}
    assert listed['dec-other-project'] == sr.DecisionState.OPEN


def _two_queues(tmp_path: Path) -> tuple[Path, Path]:
    """Build the observed two-queue collision on disk (task 3528).

    dark_factory runs TWO escalation queues over ONE ``esc-<taskid>-<n>`` id
    namespace: the orchestrator's ``data/escalations`` and the reconciliation
    watcher's ``data/reconciliation/escalations``. Here that is `orch` and
    `recon`, both holding an *unrelated* escalation that happens to share the
    id ``esc-3036-1`` -- RESOLVED (archived) in `orch`, still PENDING (queue
    root) in `recon`, exactly as observed when a blocking recon gate sat
    invisible in the cockpit for ~7 days.

    Reuses the layout pinned by test_read_escalation_status_reads_queue_root_file
    / ..._falls_back_to_archive: queue-root file = pending, dated
    ``archive/YYYY-MM-DD/`` file = resolved.
    """
    orch = tmp_path / 'orch'
    recon = tmp_path / 'recon'
    orch_archive = orch / 'archive' / '2026-07-26'
    orch_archive.mkdir(parents=True)
    recon.mkdir(parents=True)
    (orch_archive / 'esc-3036-1.json').write_text(json.dumps({'status': 'resolved'}))
    (recon / 'esc-3036-1.json').write_text(json.dumps({'status': 'pending'}))
    return orch, recon


def test_main_reap_decisions_does_not_close_across_queues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """THE REGRESSION (task 3528), cases (a) + (d) in ONE reaper run.

    (a) A decision stamped with the `recon` queue must NOT be closed by a
    reaper scanning the `orch` queue, even though `orch` holds a RESOLVED
    escalation with the same id -- they are unrelated escalations that merely
    collide in the shared id namespace. Before this fix the join was scoped
    on project alone, so this decision closed to ANSWERED and vanished from
    the cockpit queue while its own escalation was still PENDING.

    (d) In the SAME run, a queue-less (pre-change, escalations_dir='')
    decision on the SAME escalation id still closes exactly as before -- so
    one invocation proves the new guard is both blocking (a) and backward-
    compatible (d), and that the only thing distinguishing them is the queue
    stamp.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, recon = _two_queues(tmp_path)
    sr.write_decision(
        _make_decision(
            id='dec-recon-gate',
            project='df',
            escalation_id='esc-3036-1',
            state=sr.DecisionState.OPEN,
            escalations_dir=sr.normalize_escalations_dir(recon),
        ),
        root=tmp_path,
    )
    sr.write_decision(
        _make_decision(
            id='dec-legacy-queueless',
            project='df',
            escalation_id='esc-3036-1',
            state=sr.DecisionState.OPEN,
            escalations_dir='',
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', str(orch)])

    assert rc == 0
    listed = {d.id: d.state for d in sr.list_decisions(root=tmp_path)}
    assert listed['dec-recon-gate'] == sr.DecisionState.OPEN
    assert listed['dec-legacy-queueless'] == sr.DecisionState.ANSWERED


def test_main_reap_decisions_does_not_close_across_queues_mirrored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Case (b): the guard is symmetric. A decision stamped with the `orch`
    queue is equally protected from the recon watcher's reaper, which scans
    `recon` and finds a RESOLVED escalation of the same id there. Neither
    watcher is privileged; either one reaping the other's decisions is the
    same bug.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, recon = _two_queues(tmp_path)
    recon_archive = recon / 'archive' / '2026-07-26'
    recon_archive.mkdir(parents=True)
    (recon_archive / 'esc-mirror.json').write_text(json.dumps({'status': 'resolved'}))
    sr.write_decision(
        _make_decision(
            id='dec-orch-gate',
            project='df',
            escalation_id='esc-mirror',
            state=sr.DecisionState.OPEN,
            escalations_dir=sr.normalize_escalations_dir(orch),
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', str(recon)])

    assert rc == 0
    listed = {d.id: d.state for d in sr.list_decisions(root=tmp_path)}
    assert listed['dec-orch-gate'] == sr.DecisionState.OPEN


def test_main_reap_decisions_same_queue_still_closes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Case (c): the guard must not over-block. A decision stamped with the
    SAME queue the reaper is scanning still closes on its escalation's
    terminal status -- otherwise queue-scoping would quietly turn the reaper
    into a permanent no-op and every decision would need manual closure.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, _recon = _two_queues(tmp_path)
    sr.write_decision(
        _make_decision(
            id='dec-orch-same-queue',
            project='df',
            escalation_id='esc-3036-1',
            state=sr.DecisionState.OPEN,
            escalations_dir=sr.normalize_escalations_dir(orch),
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', str(orch)])

    assert rc == 0
    listed = {d.id: d.state for d in sr.list_decisions(root=tmp_path)}
    assert listed['dec-orch-same-queue'] == sr.DecisionState.ANSWERED


def test_main_reap_decisions_queue_match_is_spelling_insensitive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Case (e): the comparison normalizes BOTH sides, rather than doing a raw
    string compare of whatever each side happened to store.

    The decision here carries a dotted, trailing-slash spelling of the `orch`
    queue -- what a hand-repaired record, a future writer, or a record
    migrated between checkouts can hold, since it bypassed write-decision's
    write-time normalization. A raw compare would treat it as a foreign
    queue and fail OPEN forever; worse, the same laxness in reverse is how a
    false NON-match would reintroduce silent divergence between writer and
    reaper.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, _recon = _two_queues(tmp_path)
    (tmp_path / 'x').mkdir()
    dotted = str(orch.parent / 'x' / '..' / orch.name) + '/'
    assert dotted != str(orch)
    sr.write_decision(
        _make_decision(
            id='dec-dotted-queue',
            project='df',
            escalation_id='esc-3036-1',
            state=sr.DecisionState.OPEN,
            escalations_dir=dotted,
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', str(orch)])

    assert rc == 0
    listed = {d.id: d.state for d in sr.list_decisions(root=tmp_path)}
    assert listed['dec-dotted-queue'] == sr.DecisionState.ANSWERED


def test_main_reap_decisions_normalizes_the_reapers_own_escalations_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Case (e) MIRRORED: the REAPER's side of the compare is normalized too.

    ..._queue_match_is_spelling_insensitive only exercises the DECISION side
    -- it stores a dotted spelling and passes the already-canonical
    ``str(orch)`` to the CLI, so the reaper-side normalize is a no-op there.
    Here it is the other way round: the record carries the canonical form and
    the CLI is handed a relative, dotted spelling of the same queue.

    This is the real invocation shape, not a contrivance. Both SKILL.md files
    now promise "stored normalized, so any spelling of the same directory
    works", and the recon watcher's documented command is
    ``--escalations-dir $DARK_FACTORY_ROOT/data/reconciliation/escalations``
    where ``$DARK_FACTORY_ROOT`` may legitimately be relative or symlinked.
    Drop the reaper-side normalize and EVERY stamped decision becomes a
    permanent no-close -- fail-open, so invisible: nothing errors, decisions
    just quietly stop closing. Running under monkeypatch.chdir also pins that
    the reaper side resolves against the cwd, matching
    normalize_escalations_dir's documented expanduser/resolve contract.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, _recon = _two_queues(tmp_path)
    (tmp_path / 'x').mkdir()
    monkeypatch.chdir(tmp_path)
    relative = f'x/../{orch.name}/'
    assert not Path(relative).is_absolute()
    sr.write_decision(
        _make_decision(
            id='dec-canonical-queue',
            project='df',
            escalation_id='esc-3036-1',
            state=sr.DecisionState.OPEN,
            escalations_dir=sr.normalize_escalations_dir(orch),
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', relative])

    assert rc == 0
    listed = {d.id: d.state for d in sr.list_decisions(root=tmp_path)}
    assert listed['dec-canonical-queue'] == sr.DecisionState.ANSWERED


def test_main_reap_decisions_mode2_collapsed_decision_is_reapable_only_by_its_stamped_queue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pins the acknowledged MODE-2 tradeoff (task 3528, raised in review).

    A MODE-2 same-subject duplicate (esc-5914-1) collapses to ONE record by
    design -- but ``escalations_dir`` is single-valued and write_decision
    rewrites the whole file, so the survivor carries only the LAST filer's
    queue (pinned by ..._same_id_from_two_queues_stays_one_decision). The
    axis-2 guard then makes the OTHER queue's reaper skip it outright.

    So if the escalation that actually reaches a terminal status is the one
    in the NON-stamped queue -- entirely possible for two independently-filed
    escalations covering the same gate -- the decision now stays OPEN and
    needs human closure, where before this change either reaper would have
    closed it. That is a real behaviour change for the MODE-2 population,
    accepted because it is the fail-OPEN direction: an over-held decision is
    a visible, human-triageable cockpit row, while a falsely closed one is
    invisible (the ~7-day loss this task exists to prevent). Pinned here so
    the tradeoff is explicit rather than latent; documented on
    _run_reap_decisions and in the recon watcher's MODE-2 bullet.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, recon = _two_queues(tmp_path)
    # The same gate is filed as a separate escalation in each queue; only the
    # orchestrator's copy has resolved.
    (orch / 'archive' / '2026-07-26' / 'esc-5914-1.json').write_text(
        json.dumps({'status': 'resolved'})
    )
    (recon / 'esc-5914-1.json').write_text(json.dumps({'status': 'pending'}))
    for queue in (orch, recon):  # two watchers, one collapsed record; recon files last
        assert (
            sr.main(
                [
                    'write-decision',
                    '--id',
                    'esc-5914-1',
                    '--project',
                    'df',
                    '--text',
                    'Adopt the reify plan?',
                    '--escalation-id',
                    'esc-5914-1',
                    '--escalations-dir',
                    str(queue),
                ]
            )
            == 0
        )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', str(orch)])

    assert rc == 0
    listed = {d.id: d.state for d in sr.list_decisions(root=tmp_path)}
    assert listed['esc-5914-1'] == sr.DecisionState.OPEN


def test_main_write_decision_same_id_from_two_queues_stays_one_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Case (f): MODE-2 same-subject collapse (task 3528 ADDENDUM, req. (b)).

    Two collision modes must be kept apart. MODE 1 (esc-3036-1, the cases
    above) is two UNRELATED escalations sharing an id -- cross-queue closing
    there is a straight bug. MODE 2 (observed: esc-5914-1) is both queues
    surfacing the SAME underlying human gate; those must collapse to ONE
    cockpit decision, because a human asked the same question twice is a
    regression of its own.

    This design satisfies MODE 2 BY CONSTRUCTION: the queue is recorded as a
    FIELD on the record and the decision id is left untouched, so a second
    watcher filing the same question lands on the same id. This case passes
    both before and after the fix by design -- it is the guard that a future
    refactor to per-queue decision ids ('recon:esc-5914-1' vs
    'orch:esc-5914-1') would double-file the same question and must not be
    adopted. Complements test_main_write_decision_refiling_same_id_overwrites_not_duplicates,
    which pins the same-queue restart case.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, recon = _two_queues(tmp_path)

    rc1 = sr.main(
        [
            'write-decision',
            '--id',
            'esc-5914-1',
            '--project',
            'df',
            '--text',
            'Adopt the reify plan?',
            '--escalation-id',
            'esc-5914-1',
            '--escalations-dir',
            str(orch),
        ]
    )
    rc2 = sr.main(
        [
            'write-decision',
            '--id',
            'esc-5914-1',
            '--project',
            'df',
            '--text',
            'Adopt the reify plan?',
            '--escalation-id',
            'esc-5914-1',
            '--escalations-dir',
            str(recon),
        ]
    )

    assert rc1 == 0
    assert rc2 == 0
    listed = sr.list_decisions(root=tmp_path)
    assert [d.id for d in listed] == ['esc-5914-1']
    # The discriminator is a FIELD holding a normalized queue path, never a
    # namespace prefix baked into the id.
    assert listed[0].escalations_dir == sr.normalize_escalations_dir(recon)
    assert listed[0].id == 'esc-5914-1'


def test_main_reap_decisions_fail_soft_on_bad_escalations_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A nonexistent --escalations-dir must not raise or close anything:
    read_escalation_status returns None for every lookup, and
    reap_answered_decisions treats None as "leave OPEN".
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    bad_escalations_dir = tmp_path / 'does-not-exist'
    sr.write_decision(
        _make_decision(
            id='dec-cli-badescdir',
            project='df',
            escalation_id='esc-whatever',
            state=sr.DecisionState.OPEN,
        ),
        root=tmp_path,
    )

    rc = sr.main(
        ['reap-decisions', '--project', 'df', '--escalations-dir', str(bad_escalations_dir)]
    )

    assert rc == 0
    listed = {d.id: d.state for d in sr.list_decisions(root=tmp_path)}
    assert listed['dec-cli-badescdir'] == sr.DecisionState.OPEN


def test_main_reap_decisions_refuses_unknown_queue_but_still_closes_legacy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REGRESSION GUARD, not a RED driver (task 3640) -- and deliberately so.

    Probed against the merged 3528 code before this test was written: a
    decision stamped with ANY truthy non-matching value already survives the
    axis-2 `if decision_dir and decision_dir != reaper_dir` compare, so an
    UNKNOWN_QUEUE stamp is ALREADY safe today. A test asserting "the reaper
    closes unknown-stamped records" would therefore be doomed-green and could
    never drive an implementation. What this pins instead is that the safety
    SURVIVES: it is currently an accident of string inequality, and a future
    simplification of that compare (or of the explicit by-name guard step-2
    adds) would silently make every back-filled undeterminable record closable
    again, restoring the exact ~7-day invisible-close failure 3528 removed.

    Both arms run in ONE reaper invocation against a queue holding a RESOLVED
    escalation with the shared id, so the ONLY thing distinguishing them is
    the queue stamp:
      - the UNKNOWN_QUEUE-stamped record stays OPEN (refuse, never default to
        close -- it stays a visible cockpit row for human closure);
      - the legacy `escalations_dir=''` record on the SAME id still closes to
        ANSWERED. That second assertion is the load-bearing one: task 3640
        must NOT redefine '' under the human. '' keeps its 3528 meaning
        (fall back to project-only scoping); the back-fill DRAINS that
        population instead of changing what it means.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, _recon = _two_queues(tmp_path)
    sr.write_decision(
        _make_decision(
            id='dec-unknown-queue',
            project='df',
            escalation_id='esc-3036-1',
            state=sr.DecisionState.OPEN,
            escalations_dir=sr.UNKNOWN_QUEUE,
        ),
        root=tmp_path,
    )
    sr.write_decision(
        _make_decision(
            id='dec-legacy-unset',
            project='df',
            escalation_id='esc-3036-1',
            state=sr.DecisionState.OPEN,
            escalations_dir='',
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', str(orch)])

    assert rc == 0
    listed = {d.id: d.state for d in sr.list_decisions(root=tmp_path)}
    assert listed['dec-unknown-queue'] == sr.DecisionState.OPEN
    assert listed['dec-legacy-unset'] == sr.DecisionState.ANSWERED


def test_main_reap_decisions_refuses_unknown_queue_even_when_reaper_passes_the_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The degenerate case the string-inequality compare alone does NOT cover.

    If a reaper is invoked with the sentinel as its own ``--escalations-dir``
    (an operator copy-pasting a stamped value out of a record, or a wrapper
    threading the field straight through), then `decision_dir == reaper_dir`
    and the axis-2 guard does not fire at all. Today the record still survives
    only because read_escalation_status finds no queue at that bogus path and
    returns None -- safety by lucky accident, one directory named
    ``<unknown>`` away from failing. Step-2's explicit by-name guard is what
    makes the refusal intentional, and this pins it.

    Note the sentinel must reach the record VERBATIM for this to be a real
    test, which is exactly what step-2's normalize_escalations_dir case
    guarantees.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, _recon = _two_queues(tmp_path)
    # A real directory literally named '<unknown>', holding a RESOLVED
    # escalation for the shared id -- so a reaper that resolved the sentinel
    # as a relative path WOULD find a close-worthy status there.
    monkeypatch.chdir(tmp_path)
    bogus = tmp_path / sr.UNKNOWN_QUEUE
    bogus.mkdir()
    (bogus / 'esc-3036-1.json').write_text(json.dumps({'status': 'resolved'}))
    assert (orch / 'archive' / '2026-07-26' / 'esc-3036-1.json').is_file()
    sr.write_decision(
        _make_decision(
            id='dec-unknown-selfmatch',
            project='df',
            escalation_id='esc-3036-1',
            state=sr.DecisionState.OPEN,
            escalations_dir=sr.UNKNOWN_QUEUE,
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', sr.UNKNOWN_QUEUE])

    assert rc == 0
    listed = {d.id: d.state for d in sr.list_decisions(root=tmp_path)}
    assert listed['dec-unknown-selfmatch'] == sr.DecisionState.OPEN


class TestAtomicWriteSemantics:
    """``_atomic_write_text``'s on-disk semantics, pinned WITHOUT a delegation seam.

    Task 3223 consolidated the repo's tmp+rename writers into
    ``shared.safe_io.atomic_write_text``, but this module is the deliberate
    exception: it is stdlib-only so ``skills/spawn/spawn-claude.sh`` can run it
    with no venv (see ``TestStdlibOnlySelfContainment`` below and
    ``_ALLOWED_RENAMERS`` in ``shared/tests/test_safe_io.py``).

    So these assert the OBSERVABLE result on disk rather than that a particular
    helper was called. That is the more durable pin anyway: it holds whether
    the body is inlined or delegated, and it does not go stale the next time
    the implementation choice is revisited.
    """

    def test_writes_0600_and_leaves_no_temp_residue(self, tmp_path):
        """The record is created 0600 (mkstemp's mode), with no tmp left behind."""
        target = tmp_path / 'record.json'

        sr._atomic_write_text(target, '{"a": 1}')

        assert target.read_text() == '{"a": 1}'
        assert target.stat().st_mode & 0o777 == 0o600, (
            'session records hold session state and must stay owner-only'
        )
        assert sorted(p.name for p in tmp_path.iterdir()) == ['record.json']

    def test_creates_missing_parent_dirs(self, tmp_path):
        """The parent chain is created rather than surfacing FileNotFoundError."""
        target = tmp_path / 'nested' / 'deeper' / 'record.json'

        sr._atomic_write_text(target, '{"a": 1}')

        assert target.read_text() == '{"a": 1}'

    def test_overwrites_atomically_leaving_no_residue(self, tmp_path):
        """A rewrite replaces the contents wholesale and leaves no tmp file."""
        target = tmp_path / 'record.json'

        sr._atomic_write_text(target, '{"a": 1}')
        sr._atomic_write_text(target, '{"b": 2}')

        assert target.read_text() == '{"b": 2}'
        assert sorted(p.name for p in tmp_path.iterdir()) == ['record.json']

    def test_failed_write_leaves_original_intact_and_no_residue(
        self, tmp_path, monkeypatch
    ):
        """On a mid-write failure the original survives and the tmp is cleaned up."""
        target = tmp_path / 'record.json'
        sr._atomic_write_text(target, '{"original": true}')

        def _boom(*_args, **_kwargs):
            raise OSError('disk full')

        monkeypatch.setattr(sr.os, 'replace', _boom)

        with pytest.raises(OSError, match='disk full'):
            sr._atomic_write_text(target, '{"replacement": true}')

        assert target.read_text() == '{"original": true}'
        assert sorted(p.name for p in tmp_path.iterdir()) == ['record.json']

    def test_write_record_creates_missing_nested_dir_and_round_trips(self, tmp_path):
        """End-to-end: write_record into a non-existent nested root still works."""
        root = tmp_path / 'does' / 'not' / 'exist'
        assert not root.exists()

        record = _make_record(session_slug='sess-3223', task_id='3223')
        sr.write_record(record, root=root)

        loaded = sr.read_record('sess-3223', root=root)
        assert loaded is not None
        assert loaded.session_slug == 'sess-3223'
        assert loaded.task_id == '3223'


#: Repo root of this worktree, resolved from THIS FILE so the subprocess
#: contract test below is CWD-independent (the suite is run from
#: ``<worktree>/orchestrator``, but a scratch-CWD run must behave identically).
_SR_REPO_ROOT = Path(__file__).resolve().parents[2]


def _non_venv_interpreter() -> str | None:
    """Return an interpreter that has no workspace packages installed, or None.

    ``sys.executable`` is the venv interpreter, which HAS ``shared`` installed
    and therefore cannot discriminate — the contract under test is precisely
    "imports without a venv".  Prefer the system interpreter, then the base
    interpreter this venv was built from.
    """
    candidates = ['/usr/bin/python3', str(Path(sys.base_prefix) / 'bin' / 'python3')]
    for candidate in candidates:
        if Path(candidate).is_file() and Path(candidate).resolve() != Path(
            sys.executable
        ).resolve():
            return candidate
    return None


class TestStdlibOnlySelfContainment:
    """``session_registry`` must import with NO venv, install or workspace deps.

    This is an architectural contract, not a style preference.  The module
    docstring (lines 5-12) states it is "deliberately stdlib-only and
    self-contained (no intra-orchestrator imports)" so it can be "invoked
    directly by ``skills/spawn/spawn-claude.sh`` via an absolute path (no
    venv/PYTHONPATH/install required)".  That hook runs under an interpreter
    with no workspace packages available, so ANY ``from shared import ...`` or
    third-party import at module scope breaks the spawn path — and does so far
    from here, as a hook subprocess that silently never writes ``record.json``.

    Task 3223's consolidation added ``from shared import safe_io`` and broke
    exactly this (escalation esc-3223-2); the only symptom in the suite was two
    downstream ``test_session_hooks.py`` failures, which is why the contract is
    now pinned directly.  A future migration that finds this test in its way
    should read the module docstring before deleting it.
    """

    def test_imports_under_a_bare_interpreter_with_no_workspace_packages(self, tmp_path):
        """`import orchestrator.session_registry` succeeds with only orchestrator/src."""
        interpreter = _non_venv_interpreter()
        if interpreter is None:
            pytest.skip('no non-venv interpreter available on this host')

        env = {
            'PYTHONPATH': str(_SR_REPO_ROOT / 'orchestrator' / 'src'),
            'PATH': '/usr/bin:/bin',
            'HOME': os.environ.get('HOME', '/tmp'),
            'PYTHONDONTWRITEBYTECODE': '1',
        }

        # ``cwd=tmp_path`` on both runs below is LOAD-BEARING, not tidiness.
        # Python puts the process's own cwd on sys.path, so a run from the REPO
        # ROOT resolves ``<repo>/shared`` — the wrapper directory, NOT the real
        # ``shared/src/shared`` package — as an empty NAMESPACE package.  The
        # probe's ``shared`` import then succeeds, this test skips, and the
        # contract goes unpinned.  Measured: inert from the repo root, live from
        # ``orchestrator/``.  The canonical test_command runs
        # ``cd orchestrator && pytest tests/``, so it only bit an ad-hoc
        # ``pytest orchestrator/tests/...`` from the root — which is exactly what
        # someone checking this one file types.  A neutral cwd holds either way.
        #
        # Guard against a vacuous pass: on a host where ``shared`` really is
        # installed into this interpreter, the assertion below would succeed even
        # with the contract broken.  Probe the SAME import shape the contract
        # forbids — a bare ``import shared`` is satisfied by a namespace dir that
        # ``from shared import safe_io`` still fails on, so probing the bare form
        # can only ever over-skip.
        probe = subprocess.run(
            [interpreter, '-c', 'from shared import safe_io'],
            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60,
        )
        if probe.returncode == 0:
            pytest.skip(
                'this interpreter can already import `shared`; the test cannot '
                'discriminate a stdlib-only module from one that imports it'
            )

        result = subprocess.run(
            [interpreter, '-c', 'import orchestrator.session_registry'],
            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60,
        )

        assert result.returncode == 0, (
            'session_registry must import with no venv/install — it is executed '
            'by absolute path from skills/spawn/spawn-claude.sh. Remove the '
            'non-stdlib module-scope import.\n'
            f'stderr:\n{result.stderr}'
        )
