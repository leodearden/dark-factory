"""Tests for orchestrator.session_registry module.

Covers: schema/contract round-trip (SCHEMA_VERSION, Status enum, SessionRecord
to/from dict/json); slug sanitization + path/transcript encoding; single-writer
atomic write/read/update; TTL/pid stale-record reaper matrix; CLI subcommands
+ fail-soft; and the G5 two-way boundary test (write by a real spawn -> refresh
by a simulated hook -> reap), exercised jointly with
tests/scripts/test_spawn_claude.py's bash-level harness.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import fcntl
import importlib.util
import json
import logging
import os
import re
import socket
import subprocess
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import pytest
from _decision_fixtures import EVIDENCE, make_decision
from _process_identity_variants import IdentityVariant, own_identity
from _reap_report_wait import await_reap_report
from _utf8_codec import names_the_utf8_codec
from shared import proc_liveness
from shared.process_identity import ProcessIdentity
from shared.project_id import PROJECT_ID_ALIASES, fold_project_token

from orchestrator import decision_registry as dr
from orchestrator import fleet_files
from orchestrator import session_registry as sr

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_record(**overrides: object) -> sr.SessionRecord:
    """Build a fully-populated SessionRecord for round-trip/identity tests.

    Every field is given a concrete, distinguishable value so a round-trip
    test can catch a field being dropped/mis-typed; ``overrides`` lets a
    test tweak just the field(s) it cares about. Includes the C1 schema
    extensions (parent_session_id/spawn_mode/display/question) and the
    task-4193 hook-owner binding (claude_session_id, claude_owner_pid)
    alongside the original rail fields.
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
        'claude_session_id': 'uuid-claude-abc123',
        'claude_launch_session_id': 'uuid-claude-launch-0',
        'claude_owner_pid': 424242,
        'self_exit': True,
        't3_end': sr.T3End.SETTLE,
        'exit_request': sr.ExitRequest(
            state=sr.ExitRequestState.ATTEMPTED,
            reason=None,
            pid=424242,
            at='2026-07-07T00:00:00+00:00',
            marker_session_id='uuid-claude-abc123',
        ),
        'exit_requested_at': '2026-07-07T00:00:00+00:00',
    }
    fields.update(overrides)
    return sr.SessionRecord(**fields)


# A pid virtually guaranteed dead on any host (matches the convention in
# shared/tests/test_proc_liveness.py: 2**31 - 1).
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
    # Task 4193's additive hook-owner binding: absent on disk -> None, no
    # migration required.
    assert record.claude_session_id is None
    assert record.launcher_identity is None
    assert record.claude_owner_identity is None


def test_schema_minor_is_int_bumped() -> None:
    assert isinstance(sr.SCHEMA_MINOR, int)
    assert sr.SCHEMA_MINOR >= 1
    # The PERSISTED major must stay migration-free: bumping it would make
    # rail-vintage and C1 records version-distinguishable on disk.
    assert sr.SCHEMA_VERSION == 1


# ---------------------------------------------------------------------------
# Task 4193: SessionRecord.claude_session_id (hook-owner binding)
# ---------------------------------------------------------------------------


def test_session_record_round_trip_includes_claude_session_id() -> None:
    r = _make_record()
    assert sr.SessionRecord.from_dict(r.to_dict()) == r
    assert sr.SessionRecord.from_json(r.to_json()) == r
    # The wire key is pinned by name: session_hooks reads it back off disk.
    assert r.to_dict()['claude_session_id'] == 'uuid-claude-abc123'

    # The unbound shape a spawn-claude.sh `launching` write produces (no hook
    # has claimed the slug yet) must round-trip just as losslessly.
    r_unbound = _make_record(claude_session_id=None)
    assert sr.SessionRecord.from_dict(r_unbound.to_dict()) == r_unbound
    assert sr.SessionRecord.from_json(r_unbound.to_json()) == r_unbound
    assert r_unbound.to_dict()['claude_session_id'] is None


def test_session_record_defaults_claude_session_id_to_none() -> None:
    record = sr.SessionRecord(session_slug='s', status=sr.Status.LAUNCHING)
    assert record.claude_session_id is None


def test_session_record_round_trip_includes_claude_owner_pid() -> None:
    # esc-4193-11 suggestion 3: to_dict/from_dict must actually carry
    # claude_owner_pid -- without a non-None value in _make_record's fields,
    # the round-trip pin above is vacuous (both sides default to None even
    # if to_dict silently dropped the key).
    r = _make_record()
    assert r.claude_owner_pid == 424242
    assert r.to_dict()['claude_owner_pid'] == 424242
    assert sr.SessionRecord.from_dict(r.to_dict()) == r
    assert sr.SessionRecord.from_json(r.to_json()) == r


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        (True, None),  # bool is an int subclass; must be rejected before the int check
        ('x', None),
        (3.0, None),
        (0, None),
        (-1, None),
        (None, None),
        (4242, 4242),
    ],
)
def test_coerce_owner_pid(raw: object, expected: int | None) -> None:
    assert sr._coerce_owner_pid(raw) == expected


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        (True, None),
        (123, None),
        (3.0, None),
        (None, None),
        ('', None),
        ('   ', None),
        ('uuid-claude-abc123', 'uuid-claude-abc123'),
        ('  uuid-claude-abc123  ', 'uuid-claude-abc123'),
    ],
)
def test_coerce_session_id(raw: object, expected: str | None) -> None:
    assert sr._coerce_session_id(raw) == expected


def test_session_record_from_dict_coerces_non_str_claude_session_id() -> None:
    # esc-4193-11 suggestion 2: a hand-edited or future-writer record body
    # carrying a non-string claude_session_id must not survive into the
    # SessionRecord -- session_hooks does `(record.claude_session_id or
    # '').strip()` outside a try/except, so a raw int here would raise
    # AttributeError and lose the whole hook event.
    data = _make_record().to_dict()
    data['claude_session_id'] = 123
    record = sr.SessionRecord.from_dict(data)
    assert record.claude_session_id is None


def test_session_record_from_dict_coerces_non_int_claude_owner_pid() -> None:
    # Sibling to the claude_session_id regression test above (task 4660
    # review follow-up): pins the from_dict call site for claude_owner_pid
    # too, so replacing `_coerce_owner_pid(data.get(...))` with a bare
    # `data.get(...)` cannot go unnoticed. True is an int subclass and would
    # otherwise survive as a real-looking pid, which could silently compare
    # equal to a real pid in session_hooks._env_slug_ownership
    # (session_hooks.py:329).
    data = _make_record().to_dict()
    data['claude_owner_pid'] = True
    record = sr.SessionRecord.from_dict(data)
    assert record.claude_owner_pid is None


# ---------------------------------------------------------------------------
# Task 8125: SessionRecord.launcher_identity and claude_owner_identity
# ---------------------------------------------------------------------------

_IDENTITY_FIELDS = ('launcher_identity', 'claude_owner_identity')
_RECORD_PID = 1


def _identity_wire(**overrides: object) -> dict[str, object]:
    return {'pid': _RECORD_PID, 'start_ticks': 11, 'boot_id': 'boot-a', 'host': 'host-a', **overrides}


def _record_dict_with_pids(pid: int) -> dict[str, Any]:
    return _make_record(launcher_pid=pid, claude_owner_pid=pid).to_dict()


def test_session_record_round_trip_includes_both_identities() -> None:
    r = _make_record(
        launcher_pid=4242,
        launcher_identity=ProcessIdentity(4242, 11, 'boot-a', 'host-a'),
        claude_owner_pid=424242,
        claude_owner_identity=ProcessIdentity(424242, 12, 'boot-a', 'host-a'),
    )

    assert sr.SessionRecord.from_dict(r.to_dict()) == r
    assert sr.SessionRecord.from_json(r.to_json()) == r
    # The wire keys are pinned by name: watchers read them back off disk.
    assert r.to_dict()['launcher_identity'] == {
        'pid': 4242,
        'start_ticks': 11,
        'boot_id': 'boot-a',
        'host': 'host-a',
    }
    assert r.to_dict()['claude_owner_identity'] == {
        'pid': 424242,
        'start_ticks': 12,
        'boot_id': 'boot-a',
        'host': 'host-a',
    }


def test_session_record_defaults_both_identities_to_none() -> None:
    record = sr.SessionRecord(session_slug='s', status=sr.Status.LAUNCHING)

    assert record.launcher_identity is None
    assert record.claude_owner_identity is None


@pytest.mark.parametrize('identity_field', _IDENTITY_FIELDS)
def test_a_well_formed_identity_naming_its_pid_reads_back(identity_field: str) -> None:
    data = _record_dict_with_pids(_RECORD_PID)
    data[identity_field] = _identity_wire()

    record = sr.SessionRecord.from_dict(data)

    assert getattr(record, identity_field) == ProcessIdentity(_RECORD_PID, 11, 'boot-a', 'host-a')


@pytest.mark.parametrize('identity_field', _IDENTITY_FIELDS)
@pytest.mark.parametrize(
    'junk',
    [
        pytest.param('host-a', id='str'),
        pytest.param(_RECORD_PID, id='int'),
        pytest.param(list(_identity_wire().values()), id='list'),
        pytest.param(
            {k: v for k, v in _identity_wire().items() if k != 'host'}, id='missing-host'
        ),
        pytest.param(_identity_wire(pid=True), id='bool-pid'),
        pytest.param(_identity_wire(start_ticks='11'), id='str-start-ticks'),
        pytest.param(_identity_wire(start_ticks=-1), id='negative-start-ticks'),
        pytest.param(_identity_wire(boot_id=''), id='empty-boot-id'),
        pytest.param(_identity_wire(host=''), id='empty-host'),
        pytest.param(_identity_wire(pid=_RECORD_PID + 1), id='names-another-pid'),
    ],
)
def test_a_junk_identity_reads_none(identity_field: str, junk: object) -> None:
    data = _record_dict_with_pids(_RECORD_PID)
    data[identity_field] = junk

    assert getattr(sr.SessionRecord.from_dict(data), identity_field) is None


@pytest.mark.parametrize(
    ('identity_field', 'pid_field'),
    [('launcher_identity', 'launcher_pid'), ('claude_owner_identity', 'claude_owner_pid')],
)
def test_an_identity_on_a_record_without_its_pid_reads_none(
    identity_field: str, pid_field: str
) -> None:
    data = _record_dict_with_pids(_RECORD_PID)
    del data[pid_field]
    data[identity_field] = _identity_wire()

    assert getattr(sr.SessionRecord.from_dict(data), identity_field) is None


def test_spawn_mode_enum_values() -> None:
    assert issubclass(sr.SpawnMode, str)
    assert {m.value for m in sr.SpawnMode} == {'child', 'sibling', 'detached'}
    assert sr.SpawnMode.CHILD.value == 'child'
    assert sr.SpawnMode.SIBLING.value == 'sibling'
    assert sr.SpawnMode.DETACHED.value == 'detached'


def test_display_kind_enum_values() -> None:
    assert issubclass(sr.DisplayKind, str)
    assert {m.value for m in sr.DisplayKind} == {'wm', 'tmux', 't3'}
    assert sr.DisplayKind.WM.value == 'wm'
    assert sr.DisplayKind.TMUX.value == 'tmux'
    assert sr.DisplayKind.T3.value == 't3'


def test_display_round_trip() -> None:
    with_optionals = sr.Display(
        kind='wm', wm_title='unblock:df#2085 slug', wm_window_id='0x1a', tmux_target=None
    )
    assert sr.Display.from_dict(with_optionals.to_dict()) == with_optionals

    without_optionals = sr.Display(
        kind='tmux', wm_title='', wm_window_id=None, tmux_target='session:0.1'
    )
    assert sr.Display.from_dict(without_optionals.to_dict()) == without_optionals

    t3 = sr.Display(kind=sr.DisplayKind.T3.value, wm_title='unblock:df#7489', t3_thread_id='th-1')
    assert sr.Display.from_dict(t3.to_dict()) == t3


def test_display_from_a_body_without_t3_thread_id_reads_none() -> None:
    assert sr.Display.from_dict({'kind': 'wm'}).t3_thread_id is None


def test_question_round_trip() -> None:
    q = sr.Question(text='approve rollout?', asked_at='2026-07-07T00:00:00+00:00')
    assert sr.Question.from_dict(q.to_dict()) == q


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


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [('.', '-'), ('..', '--'), ('...', '---')],
)
def test_sanitize_slug_collapses_all_dots_slug(raw: str, expected: str) -> None:
    # Task 4112: the all-dots branch is the SOLE guard against a path escape
    # out of sessions_dir. '.' is inside _SLUG_SANITIZE_RE's allowed class, so
    # the character substitution alone passes '..' straight through, and
    # record_path_for_slug is a bare join that sanitizes nothing of its own --
    # so an unguarded '..' would be handed to the filesystem as a traversal
    # segment. Every '.' maps to '-', so length is preserved.
    slug = sr.sanitize_slug(raw)
    assert slug == expected
    assert re.fullmatch(r'[A-Za-z0-9._-]+', slug)


@pytest.mark.parametrize('raw', ['..foo', 'foo..', 'a.b', '.hidden-x'])
def test_sanitize_slug_leaves_partially_dotted_slug_untouched(raw: str) -> None:
    # The guard's LOWER edge: a value that merely CONTAINS dots alongside other
    # characters is a literal, non-traversing directory name (sanitize_slug's
    # docstring says so) and must pass through byte-identical. Pinning only the
    # collapse direction would leave a future "hardening" free to over-broaden
    # into stripping every dot, silently mangling ordinary slugs like 'a.b'.
    assert sr.sanitize_slug(raw) == raw


def test_record_path_for_slug(tmp_path: Path) -> None:
    path = sr.record_path_for_slug('unblock-df-2085-4242', root=tmp_path)
    assert path == tmp_path / 'sessions' / 'unblock-df-2085-4242' / 'record.json'


def test_record_path_for_slug_all_dots_slug_stays_under_sessions_dir(tmp_path: Path) -> None:
    # Task 4112: the containment property sanitize_slug's all-dots branch buys.
    # resolve() + is_relative_to, NOT a string prefix check -- str(root/'sessions'
    # /'..'/'record.json') startswith str(root/'sessions') is TRUE, so a prefix
    # check would pass for the very path that escapes. resolve() is non-strict
    # (these paths do not exist) and collapses '..' with the exact filesystem
    # semantics under defense.
    sessions = sr.sessions_dir(root=tmp_path).resolve()
    # '..' is the ONLY genuine escape vector in this loop; '.' and '...' are
    # non-traversing CONTROLS, not extra attack coverage. Unsanitized,
    # 'sessions/./record.json' still resolves under sessions_dir and '...' is a
    # literal POSIX dirname resolve() leaves in place -- so only the '..'
    # iteration discriminates against a guard-deleted build.
    for raw in ('.', '..', '...'):
        slug = sr.sanitize_slug(raw)
        path = sr.record_path_for_slug(slug, root=tmp_path)
        assert path.resolve().is_relative_to(sessions)

    # NON-VACUITY COUNTERFACTUAL: the UNsanitized token genuinely escapes, i.e.
    # record_path_for_slug performs NO sanitization of its own (it is a bare
    # join) -- which is exactly why sanitize_slug has to be the chokepoint.
    # Without this, the assertions above could pass for the wrong reason if
    # record_path_for_slug's shape ever changed, and nobody would notice the
    # test had stopped proving anything.
    assert not sr.record_path_for_slug('..', root=tmp_path).resolve().is_relative_to(sessions)


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


def test_read_record_decodes_with_an_explicit_utf8_encoding(tmp_path, monkeypatch):
    """The READ half of task 3387's round-trip names utf-8 too.

    Pinning only ``fleet_files.atomic_write_text`` left the round-trip ASYMMETRIC: on a
    genuinely non-UTF-8 host a record holding non-ASCII bytes was written
    correctly and then failed to DECODE on the way back in, and because
    ``UnicodeDecodeError`` subclasses ``ValueError`` it lands in
    ``read_record``'s ``except`` clause and resurfaces as
    ``CorruptSessionRecord`` — the exact unreadable-record outcome the write
    fix exists to prevent. Every ``read_text`` in the module is now explicit;
    this pins the one on the primary read path.

    A BOUNDARY SPY rather than a write-then-read-back round-trip, for the same
    measured reason the write pins in test_fleet_files.py are shaped the way they are:
    ``Path.read_text()`` with no encoding resolves its codec through the
    C-level locale (``io.text_encoding()`` returns the ``"locale"`` sentinel),
    unreachable from an in-process monkeypatch, and under this suite's UTF-8
    ambient locale a round-trip passes either way. Spying the argument fails on
    EVERY host instead: a locale-dependent read passes no ``encoding`` kwarg at
    all. (``test_fleet_files.py::TestAtomicWriteSemantics``'s strict child covers the write half
    end-to-end; a second child for the read half would buy nothing this does
    not already localise.)
    """
    r = _make_record()
    sr.write_record(r, root=tmp_path)

    seen: list[dict] = []
    real_read_text = Path.read_text

    def _spy(self, *args, **kwargs):
        seen.append(dict(kwargs))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', _spy)

    # The spy delegates, so the read really completed — this pin cannot pass
    # by disabling the thing it is measuring.
    assert sr.read_record(r.session_slug, root=tmp_path) == r

    assert len(seen) == 1, f'expected exactly one read_text call, recorded {seen}'
    encoding = seen[0].get('encoding')
    assert encoding is not None, (
        'the record was read with no encoding= argument, so it decodes in '
        'whatever the ambient locale happens to be — the task 3387 bug, '
        'mirrored onto the read half'
    )
    assert names_the_utf8_codec(encoding), (
        f'JSON on disk is utf-8 (RFC 8259) and must be decoded as utf-8 '
        f'regardless of ambient locale (task 3387); got encoding={encoding!r}'
    )


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
# apply_refresh -- the PURE (no-I/O) half of refresh_record (task 4662)
#
# session_hooks performs three reads of one record.json per hook event; the
# third is refresh_record's own internal read. Splitting the upsert body out
# as a pure function lets a caller that ALREADY holds the record produce the
# refreshed body without a second read, so one event can settle on ONE
# snapshot and ONE write. These tests pin that the split half is genuinely
# I/O-free and that refresh_record, re-expressed on top of it, is unchanged.
# ---------------------------------------------------------------------------


def _no_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ANY registry filesystem access explode.

    apply_refresh is the seam session_hooks needs precisely because it
    touches no disk; asserting "we passed no root" would not catch an
    implementation that reached for the default root, so the read/write
    entry points are booby-trapped instead.
    """

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError('apply_refresh must perform no filesystem access')

    monkeypatch.setattr(sr, 'read_record', _boom)
    monkeypatch.setattr(sr, 'write_record', _boom)


def test_apply_refresh_updates_prior_in_place_and_does_no_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = _make_record(status=sr.Status.LAUNCHING)
    before = prior.to_dict()
    _no_io(monkeypatch)

    result = sr.apply_refresh('unblock-df-2085-4242', prior, status=sr.Status.IDLE)

    # In-place read-modify, exactly like refresh_record/update_status today.
    assert result is prior
    assert result.status is sr.Status.IDLE
    # Every other field survives byte-identical.
    after = result.to_dict()
    del before['status'], after['status']
    assert after == before


def test_apply_refresh_with_no_status_is_a_pure_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """status=None must leave the prior status alone.

    This is the case the launch-window withhold path relies on: it wants the
    mtime heartbeat bumped without promoting a still-LAUNCHING record.
    """
    prior = _make_record(status=sr.Status.LAUNCHING)
    _no_io(monkeypatch)

    result = sr.apply_refresh('unblock-df-2085-4242', prior, status=None)

    assert result is prior
    assert result.status is sr.Status.LAUNCHING


def test_apply_refresh_synthesizes_a_well_formed_record_when_prior_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_io(monkeypatch)

    result = sr.apply_refresh('hand-launched-slug', None, status=sr.Status.RUNNING)

    assert result.session_slug == 'hand-launched-slug'
    assert result.status is sr.Status.RUNNING
    assert result.schema_version == sr.SCHEMA_VERSION
    # A parseable ISO-8601 start_ts, not merely a non-empty string.
    assert result.start_ts
    assert datetime.fromisoformat(result.start_ts)


def test_apply_refresh_synthesis_defaults_to_launching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LAUNCHING is refresh_record's documented default for a new record."""
    _no_io(monkeypatch)

    result = sr.apply_refresh('hand-launched-slug', None, status=None)

    assert result.status is sr.Status.LAUNCHING


@pytest.mark.parametrize('status', [sr.Status.RUNNING, None])
def test_refresh_record_equals_read_apply_write(
    tmp_path: Path, status: sr.Status | None
) -> None:
    """refresh_record(...) == write_record(apply_refresh(read_record(...))).

    The equivalence is the whole point of the split: session_hooks composes
    the second spelling from a snapshot it already holds, so it must produce
    a byte-identical body to the one refresh_record writes today.
    """
    seed = _make_record(status=sr.Status.LAUNCHING)
    sr.write_record(seed, root=tmp_path)
    via_refresh = sr.refresh_record(seed.session_slug, root=tmp_path, status=status)
    refreshed_bytes = sr.record_path_for_slug(
        seed.session_slug, root=tmp_path
    ).read_text(encoding='utf-8')

    # Re-seed and take the composed route over the same starting body.
    sr.write_record(_make_record(status=sr.Status.LAUNCHING), root=tmp_path)
    composed = sr.apply_refresh(
        seed.session_slug,
        sr.read_record(seed.session_slug, root=tmp_path),
        status=status,
    )
    sr.write_record(composed, root=tmp_path)
    composed_bytes = sr.record_path_for_slug(
        seed.session_slug, root=tmp_path
    ).read_text(encoding='utf-8')

    assert composed_bytes == refreshed_bytes
    assert composed.to_dict() == via_refresh.to_dict()


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
# Task 8125: the pid sweeps judge the launcher by its identity
# ---------------------------------------------------------------------------

def _launcher_slug(variant: IdentityVariant) -> str:
    return f'launcher-{variant}'


_UNQUALIFIED = 'launcher-unqualified'
_ELSEWHERE = _launcher_slug(IdentityVariant.OTHER_HOST)
_JUDGED_DEAD = {_launcher_slug(v) for v in IdentityVariant if v.judged_dead}
_JUDGED_KEPT = {_launcher_slug(v) for v in IdentityVariant if not v.judged_dead} | {_UNQUALIFIED}


def _launcher_identities() -> dict[str, ProcessIdentity | None]:
    """One record slug per identity, all naming this live test process: only the identity can make one dead."""
    return {_launcher_slug(v): own_identity(v) for v in IdentityVariant} | {_UNQUALIFIED: None}


def _seed_stale_launchers(root: Path) -> None:
    for slug, identity in _launcher_identities().items():
        record = _make_record(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=os.getpid(),
            launcher_identity=identity,
        )
        sr.write_record(record, root=root)
        _set_mtime(
            sr.record_path_for_slug(slug, root=root),
            _NOW,
            sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1),
        )


def _logged_other_host_skip(caplog: pytest.LogCaptureFixture, sweep: str, subject: str) -> bool:
    return any(
        sweep in r.getMessage() and subject in r.getMessage() and 'reason=other_host' in r.getMessage()
        for r in caplog.records
        if r.name == sr.__name__ and r.levelno == logging.INFO
    )


class TestPidSweepsJudgeTheLauncherByIdentity:
    def test_mark_orphaned_marks_exactly_the_launchers_judged_dead(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _seed_stale_launchers(tmp_path)

        with caplog.at_level(logging.INFO, logger=sr.__name__):
            marked = sr.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW)

        assert {m.session_slug for m in marked} == _JUDGED_DEAD
        for slug in _JUDGED_KEPT:
            assert sr.read_record(slug, root=tmp_path).status == sr.Status.RUNNING
        assert _logged_other_host_skip(caplog, 'mark_orphaned_sessions_exited', _ELSEWHERE)

    def test_reap_stale_records_reaps_exactly_the_launchers_judged_dead(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _seed_stale_launchers(tmp_path)

        with caplog.at_level(logging.INFO, logger=sr.__name__):
            reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

        assert {r.session_slug: r.reason for r in reaped} == dict.fromkeys(
            _JUDGED_DEAD, 'stale_pid'
        )
        for slug in _JUDGED_KEPT:
            assert sr.record_path_for_slug(slug, root=tmp_path).is_file()
        assert _logged_other_host_skip(caplog, 'reap_stale_records', _ELSEWHERE)


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


def _call_indexed_wmctrl_run(
    *results: tuple[int, str],
    on_call: Mapping[int, Callable[[], None]] | None = None,
) -> Callable[[list[str]], subprocess.CompletedProcess[str]]:
    """A fake ``run`` answering call N with ``results[N]``, repeating the last thereafter.

    ``on_call[N]`` runs inside call N, before it answers -- the hook a test
    uses to land a record write just after one of the sweep's probes.
    """
    calls = 0

    def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        if on_call is not None and calls in on_call:
            on_call[calls]()
        returncode, stdout = results[min(calls, len(results) - 1)]
        calls += 1
        return subprocess.CompletedProcess(argv, returncode=returncode, stdout=stdout)

    return _run


def test_mark_windowless_keeps_a_window_that_appears_after_the_first_probe(
    tmp_path: Path,
) -> None:
    def _set_display_lands() -> None:
        sr.write_record(
            _make_record(
                session_slug='late-window',
                status=sr.Status.RUNNING,
                display=sr.Display(kind='wm', wm_title='t', wm_window_id='0x1a'),
            ),
            root=tmp_path,
        )

    run = _call_indexed_wmctrl_run(
        (0, '0x99  0 host  other\n'),
        (0, '0x1a  0 host  late\n0x99  0 host  other\n'),
        on_call={0: _set_display_lands},
    )

    marked = sr.mark_windowless_wm_sessions_exited(root=tmp_path, run=run)

    assert 'late-window' not in {m.session_slug for m in marked}
    assert sr.read_record('late-window', root=tmp_path).status == sr.Status.RUNNING


def test_mark_windowless_marks_nothing_when_the_confirming_probe_fails(tmp_path: Path) -> None:
    sr.write_record(
        _make_record(
            session_slug='unconfirmed',
            status=sr.Status.RUNNING,
            display=sr.Display(kind='wm', wm_title='t', wm_window_id='0x1a'),
        ),
        root=tmp_path,
    )
    run = _call_indexed_wmctrl_run((0, '0x99  0 host  other\n'), (127, ''))

    marked = sr.mark_windowless_wm_sessions_exited(root=tmp_path, run=run)

    assert marked == []
    reloaded = sr.read_record('unconfirmed', root=tmp_path)
    assert reloaded.status == sr.Status.RUNNING
    assert reloaded.exit_code is None


def test_mark_windowless_leaves_a_record_rebound_to_a_live_window_after_the_pass(
    tmp_path: Path,
) -> None:
    def _write_wm_record(window_id: str) -> None:
        sr.write_record(
            _make_record(
                session_slug='rebound',
                status=sr.Status.RUNNING,
                display=sr.Display(kind='wm', wm_title='t', wm_window_id=window_id),
            ),
            root=tmp_path,
        )

    _write_wm_record('0x1a')
    run = _call_indexed_wmctrl_run(
        (0, '0x99  0 host  other\n'),
        (0, '0x2b  0 host  resumed\n0x99  0 host  other\n'),
        on_call={1: lambda: _write_wm_record('0x2b')},
    )

    marked = sr.mark_windowless_wm_sessions_exited(root=tmp_path, run=run)

    assert marked == []
    reloaded = sr.read_record('rebound', root=tmp_path)
    assert reloaded.status == sr.Status.RUNNING
    assert reloaded.display is not None
    assert reloaded.display.wm_window_id == '0x2b'


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


def test_main_launching_stamps_the_launcher_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_env(monkeypatch, _launching_env(tmp_path, launcher_pid=str(os.getpid())))

    rc = sr.main(['launching'])

    assert rc == 0
    slug = sr.build_session_slug('unblock', 'df', '2085', os.getpid())
    record_file = sr.record_path_for_slug(slug, root=tmp_path)
    assert capsys.readouterr().out.strip() == str(record_file.parent)
    assert sr.read_record(slug, root=tmp_path).launcher_identity == own_identity()
    on_disk = json.loads(record_file.read_text())
    assert on_disk['launcher_identity']['host'] == socket.gethostname()


def test_main_launching_with_a_dead_launcher_still_writes_an_unqualified_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_env(monkeypatch, _launching_env(tmp_path, launcher_pid=str(_DEAD_PID)))

    rc = sr.main(['launching'])

    assert rc == 0
    slug = sr.build_session_slug('unblock', 'df', '2085', _DEAD_PID)
    expected_dir = sr.record_path_for_slug(slug, root=tmp_path).parent
    assert capsys.readouterr().out.strip() == str(expected_dir)
    record = sr.read_record(slug, root=tmp_path)
    assert record.launcher_pid == _DEAD_PID
    assert record.launcher_identity is None


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


def test_main_logs_a_verb_the_parser_accepts_but_no_runner_handles(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A verb registered on the parser but dispatched nowhere is an ERROR, not a silent rc-0 no-op."""

    def parser_with_an_unrouted_verb() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_subparsers(dest='verb', required=True).add_parser('unrouted-verb')
        return parser

    monkeypatch.setattr(sr, '_build_parser', parser_with_an_unrouted_verb)

    with caplog.at_level(logging.ERROR):
        rc = sr.main(['unrouted-verb'])

    assert rc == 0
    [record] = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert 'unrouted-verb' in record.getMessage()
    assert 'no runner handles it' in record.getMessage()


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


def test_launching_returns_while_a_reap_holds_the_lock_and_its_detached_reap_drains_after(
    tmp_path: Path,
) -> None:
    """An inline sweep would block on the held lock, and a child that inherited
    stdout or stderr would keep ``run`` from returning: either way it times out.
    """
    sr.write_record(
        _make_record(
            session_slug='orphan-o', status=sr.Status.RUNNING, launcher_pid=_DEAD_PID, display=None
        ),
        root=tmp_path,
    )
    _set_mtime(
        sr.record_path_for_slug('orphan-o', root=tmp_path),
        datetime.now(UTC),
        sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1),
    )
    _write_aged_exited('old-terminal', tmp_path)
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLAUDE_SPAWN_')}
    env.update(_launching_env(tmp_path))
    slug = sr.build_session_slug('unblock', 'df', '2085', 4242)

    with sr.reap_lock(tmp_path):
        result = subprocess.run(
            [sys.executable, str(Path(sr.__file__).resolve()), 'launching'],
            env=env,
            capture_output=True,
            timeout=120,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(sr.record_path_for_slug(slug, root=tmp_path).parent)
        assert sr.read_record(slug, root=tmp_path).status == sr.Status.LAUNCHING
        assert sr.read_record('orphan-o', root=tmp_path).status == sr.Status.RUNNING
        assert (sr.sessions_dir(root=tmp_path) / 'old-terminal').is_dir()

    await_reap_report(tmp_path)
    orphan = sr.read_record('orphan-o', root=tmp_path)
    assert orphan.status == sr.Status.EXITED
    assert orphan.exit_code == sr.ORPHAN_EXIT_CODE
    assert not (sr.sessions_dir(root=tmp_path) / 'old-terminal').exists()


def _pids_with_cwd_under(directory: Path) -> list[int]:
    pids: list[int] = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cwd = Path(os.readlink(proc / 'cwd'))
        except OSError:
            continue
        if cwd == directory or directory in cwd.parents:
            pids.append(int(proc.name))
    return pids


def test_launchings_detached_reap_leaves_the_callers_cwd_and_drains_its_relative_fleet_root(
    tmp_path: Path,
) -> None:
    """A reap child whose cwd stayed in the caller's worktree would hold that
    worktree in use for as long as it queued on the held lock.
    """
    worktree = tmp_path / 'worktree'
    fleet = worktree / 'fleet'
    _write_aged_exited('old-terminal', fleet)
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLAUDE_SPAWN_')}
    env.update(_launching_env(fleet))
    env['CLAUDE_FLEET_ROOT'] = 'fleet'

    with sr.reap_lock(fleet):
        result = subprocess.run(
            [sys.executable, str(Path(sr.__file__).resolve()), 'launching'],
            cwd=worktree,
            env=env,
            capture_output=True,
            timeout=120,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert _pids_with_cwd_under(worktree) == []

    await_reap_report(fleet)
    assert not (sr.sessions_dir(root=fleet) / 'old-terminal').exists()


def test_main_launching_fail_soft_when_the_detached_reap_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_env(monkeypatch, _launching_env(tmp_path))
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'no-such-python'))

    rc = sr.main(['launching'])

    assert rc == 0
    slug = sr.build_session_slug('unblock', 'df', '2085', 4242)
    expected_dir = sr.record_path_for_slug(slug, root=tmp_path).parent
    assert capsys.readouterr().out == f'{expected_dir}\n'
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


def _write_aged_exited(slug: str, root: Path) -> None:
    sr.write_record(
        _make_record(session_slug=slug, status=sr.Status.EXITED, exit_code=0), root=root
    )
    _set_mtime(
        sr.record_path_for_slug(slug, root=root),
        datetime.now(UTC),
        sr.TERMINAL_TTL + timedelta(hours=1),
    )


def test_main_reap_drains_every_eligible_record_in_one_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('PATH', str(tmp_path / 'no-bin'))
    aged_slugs = [f'aged-exited-{i:03d}' for i in range(150)]
    for slug in aged_slugs:
        _write_aged_exited(slug, tmp_path)
    sr.write_record(
        _make_record(session_slug='drain-orphan', status=sr.Status.RUNNING, launcher_pid=_DEAD_PID),
        root=tmp_path,
    )
    _set_mtime(
        sr.record_path_for_slug('drain-orphan', root=tmp_path),
        datetime.now(UTC),
        sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1),
    )

    rc = sr.main(['reap'])

    assert rc == 0
    for slug in aged_slugs:
        assert not (sr.sessions_dir(root=tmp_path) / slug).exists()
    orphan = sr.read_record('drain-orphan', root=tmp_path)
    assert orphan.status == sr.Status.EXITED
    assert orphan.exit_code == sr.ORPHAN_EXIT_CODE

    report_path = sr.reap_report_path(tmp_path)
    assert report_path.is_file()
    report = json.loads(report_path.read_text())
    requested_at = datetime.fromisoformat(report['requested_at'])
    started_at = datetime.fromisoformat(report['started_at'])
    assert requested_at <= started_at <= datetime.fromisoformat(report['finished_at'])
    assert report['remaining_record_dirs'] == 1
    phases = report['phases']
    assert [phase['name'] for phase in phases] == [
        'index_unpointed_t3_records',
        'mark_orphaned_sessions_exited',
        'mark_windowless_wm_sessions_exited',
        'reap_stale_records',
        'reap_stale_session_pointers',
    ]
    for phase in phases:
        assert isinstance(phase['seconds'], (int, float))
        assert phase['seconds'] >= 0
        assert phase['error'] is None
    by_name = {phase['name']: phase for phase in phases}
    assert by_name['mark_orphaned_sessions_exited']['count'] == 1
    assert by_name['reap_stale_records']['count'] == 150


def test_main_reap_runs_later_phases_when_one_phase_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('PATH', str(tmp_path / 'no-bin'))

    def _boom(*_args: object, **_kwargs: object) -> list[sr.SessionRecord]:
        raise OSError('mark phase on fire')

    monkeypatch.setattr(sr, 'mark_orphaned_sessions_exited', _boom)
    _write_aged_exited('aged-despite-fault', tmp_path)

    rc = sr.main(['reap'])

    assert rc == 0
    assert not (sr.sessions_dir(root=tmp_path) / 'aged-despite-fault').exists()
    phases = json.loads(sr.reap_report_path(tmp_path).read_text())['phases']
    [failed] = [phase for phase in phases if phase['name'] == 'mark_orphaned_sessions_exited']
    assert 'OSError' in failed['error']
    assert failed['count'] is None


def _write_report_started(root: Path, started_at: datetime) -> str:
    report = json.dumps({'started_at': started_at.isoformat()})
    sr.reap_report_path(root).write_text(report)
    return report


def test_main_reap_skips_only_when_a_reap_started_after_it_was_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('PATH', str(tmp_path / 'no-bin'))
    _write_aged_exited('aged', tmp_path)
    later = _write_report_started(tmp_path, datetime.now(UTC) + timedelta(hours=1))

    assert sr.main(['reap']) == 0

    assert (sr.sessions_dir(root=tmp_path) / 'aged').is_dir()
    assert sr.reap_report_path(tmp_path).read_text() == later

    _write_report_started(tmp_path, datetime.now(UTC) - timedelta(hours=1))

    assert sr.main(['reap']) == 0

    assert not (sr.sessions_dir(root=tmp_path) / 'aged').exists()


def test_reap_lock_is_a_stable_sidecar_under_the_fleet_root(tmp_path: Path) -> None:
    lock_path = sr.reap_lock_path(tmp_path)
    with sr.reap_lock(tmp_path):
        assert lock_path.is_file()
        assert lock_path.parent == tmp_path
    assert lock_path.is_file()


def test_reap_lock_excludes_another_holder_until_released(tmp_path: Path) -> None:
    sr.reap_lock_path(tmp_path).touch()
    fd = os.open(sr.reap_lock_path(tmp_path), os.O_RDWR)
    try:
        with sr.reap_lock(tmp_path), pytest.raises(BlockingIOError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


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
    # Bash-tool shell, dead moments after the claim), so pid_alive is
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


def _seed_lease(tmp_path: Path, *, slug: str = 'watcher-df-100', pid: int | None = None) -> Path:
    """Claim `watcher-df` for *slug*/*pid* under tmp_path; return its lease path.

    Shared by the three task-3994 sections below (heartbeat ownership, release
    ownership, lease_status) -- defined once, here, at its first use.
    """
    holder = sr.LeaseHolder(
        session_slug=slug, pid=os.getpid() if pid is None else pid, start_ts=_NOW.isoformat()
    )
    sr.claim_lease('watcher-df', holder=holder, root=tmp_path, now=_NOW)
    return sr.lease_path_for_name('watcher-df', root=tmp_path)


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


def test_heartbeat_lease_on_a_reaped_lease_is_absent_and_the_owner_can_re_claim(
    tmp_path: Path,
) -> None:
    """ABSENT on a HEARTBEAT is not the harmless idempotence it is on a release.

    Code backing for the skill contract: the holder's own per-cycle heartbeat
    reporting ABSENT means its lease is GONE (reaped here; an operator
    `--force` release does the same), so the session is now running un-leased
    and any duplicate can take the name freely. The prescribed response is to
    re-claim, which must actually work — a watcher that kept beating into the
    void would sit at `result=absent` forever, which is the duplicate-watcher
    condition the lease exists to prevent.
    """
    lease_path = _seed_lease(tmp_path, pid=_DEAD_PID)
    _set_mtime(lease_path, _NOW, sr.LEASE_HEARTBEAT_TTL + timedelta(minutes=1))
    assert [r.reason for r in sr.reap_stale_leases(root=tmp_path, now=_NOW)] == ['stale_pid']

    result = sr.heartbeat_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    assert result is sr.LeaseMutation.ABSENT
    # ...and re-claiming is the documented recovery, so it must succeed.
    reclaim = sr.claim_lease(
        'watcher-df',
        holder=sr.LeaseHolder(
            session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat()
        ),
        root=tmp_path,
        now=_NOW,
    )
    assert reclaim.acquired
    assert sr.heartbeat_lease('watcher-df', slug='watcher-df-100', root=tmp_path) is (
        sr.LeaseMutation.APPLIED
    )


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
    # heartbeat_ts is None only for an absent lease; a held one always has one.
    assert status.heartbeat_ts is not None
    assert datetime.fromisoformat(status.heartbeat_ts).replace(microsecond=0) == _NOW - timedelta(
        minutes=30
    )
    assert status.reclaimable is False


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
    lines = capsys.readouterr().out.splitlines()
    # The WHOLE acquired-path contract, not just line 1: a successful claim has
    # no contending holder, so it must not print `holder_liveness=held` (which
    # reads as "someone else has it"). `none` is the third token in the
    # vocabulary, and it is pinned here because nothing else exercises it.
    assert lines[0] == 'decision=acquired'
    assert lines[2] == 'holder_liveness=none'
    # Task 4248 appended a `slug=` line after it (pinned in that task's own
    # section), and task 4237 a `holder_record=` line after that; anchored by
    # INDEX here so this keeps asserting the position of `holder_liveness`,
    # which is what this test is about, rather than silently re-aiming at
    # whatever line happens to be last.
    assert len(lines) == 5
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
#
# SINGLE-SIGNAL by construction: `holder_liveness=orphaned` means exactly
# "the pid recorded in the lease body is not running". These tests run under
# PRODUCTION-REALISTIC conditions -- no session record is written anywhere,
# because a lease's `session_slug` is a claimant-chosen ownership token
# (`watcher-<project>-<pid>`) and NOT a session-registry record key, so no
# production path can ever put a record there (see
# TestLeaseSlugIsNotASessionRecordKey below). The originally-designed
# session-record corroboration was withdrawn for exactly that reason; the
# emitted values are byte-identical either way.


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


def test_main_lease_claim_reports_orphaned_for_a_dead_holder_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    # No session record is written -- that is the production shape, and the
    # dead pid is the whole of the evidence.
    lines = _contend_via_cli(tmp_path, capsys, holder_pid=_DEAD_PID)

    assert lines[0] == 'decision=stand-down'
    # Anchored by INDEX, not [-1]: task 4248 appended a `slug=` line after this
    # one, and [-1] would have silently re-aimed at it instead of failing.
    assert lines[2] == 'holder_liveness=orphaned'


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
    # Index, not [-1] -- task 4248's `slug=` line now follows it.
    assert capsys.readouterr().out.splitlines()[2] == 'holder_liveness=held'


class TestLeaseSlugIsNotASessionRecordKey:
    """`read_record(<a lease slug>)` raises — pinned so the withdrawn corroboration cannot return.

    Task 3994 originally corroborated the orphan signal with the holder's own
    session_registry record, via ``read_record(holder.session_slug)``. That
    lookup is STRUCTURALLY IMPOSSIBLE, and the tests that appeared to exercise
    its branches only did so by hand-writing a record under a lease-shaped
    slug — a state no production path can produce. They were deleted; this
    class replaces them with the reason.

    MEASURED EVIDENCE (2026-08-15, live fleet root): the real
    ``~/.claude/fleet/leases/watcher-df.lease`` holds
    ``session_slug=watcher-df-1894895``, while 0 of the 46,624
    ``~/.claude/fleet/sessions/`` record dirs begin with ``watcher-``. Real
    record slugs look like ``architect-dark_factory-3133-25737ee4-...``.

    The namespaces differ in all three segments:
    - the LEASE slug is ``<lease name>-<session pid>``, built by
      ``default_lease_slug`` (task 4248 moved it out of shell in the SKILLs and
      into the CLI, for the reason 3994 moved the pid there);
    - the RECORD slug comes from ``build_session_slug(role, project, task,
      <uniqueness token>)``, where the token is a session UUID for a
      hand-launched session (``session_hooks.hook_session_slug``) and the
      project token is the registry's project id (``dark_factory``), not the
      lease's short name (``df``).

    No test here compares a literal this file wrote against
    ``build_session_slug``'s output, on purpose: the disjointness RATIONALE
    belongs to (and is recorded in) ``LeaseHolder.session_slug``'s docstring,
    and such a test cannot fail as a result of any production change. The
    first test has real coupling — it would break if ``read_record`` ever
    started resolving a lease slug, which is exactly the reintroduction worth
    catching. The second pins the pid pointer (task 4237) as the join that
    does reach the record, side by side with the lookup that cannot.
    """

    # A hand-launched session's uniqueness token: hook_session_slug passes the
    # hook's `session_id` (a UUID string) into build_session_slug's
    # launcher_pid slot. Kept literal so the test is deterministic.
    SESSION_UUID = '25737ee4-3b1e-4a7f-9f0e-6a1c2d3e4f50'

    def _lease_slug(self, project: str = 'df', pid: int = 1894895) -> str:
        """The lease slug exactly as production builds it (task 4248).

        Routed through ``default_lease_slug`` rather than hand-building the
        f-string: the disjointness this class pins is only interesting if the
        left-hand side is the REAL lease slug, so the helper must break when
        production's builder changes shape.
        """
        slug = sr.default_lease_slug(f'watcher-{project}', {sr.SESSION_PID_ENV: str(pid)})
        assert slug is not None  # pid > 0, so the builder always derives here
        return slug

    def _record_slug(self, project: str = 'dark_factory') -> str:
        """The record slug exactly as session_hooks.hook_session_slug builds it."""
        return sr.build_session_slug(
            'session',
            project,
            None,
            self.SESSION_UUID,  # type: ignore[arg-type]
        )

    def test_a_real_record_provably_does_not_live_under_its_lease_slug(
        self, tmp_path: Path
    ) -> None:
        # A well-formed record IS written for this notional session, under the
        # slug the registry actually keys on...
        record_slug = self._record_slug()
        sr.write_record(_make_record(session_slug=record_slug), root=tmp_path)
        assert sr.read_record(record_slug, root=tmp_path).session_slug == record_slug

        # ...and read_record(<lease slug>) STILL raises FileNotFoundError. That
        # is why holder_session_state was a constant 'absent' in production.
        with pytest.raises(FileNotFoundError):
            sr.read_record(self._lease_slug(), root=tmp_path)

    def test_the_pid_pointer_reaches_the_record_the_lease_slug_cannot(
        self, tmp_path: Path
    ) -> None:
        pid = 1894895
        record_slug = self._record_slug()
        sr.write_record(
            _make_record(session_slug=record_slug, claude_owner_pid=pid), root=tmp_path
        )
        sr.write_session_pointer(pid, record_slug, root=tmp_path)

        with pytest.raises(FileNotFoundError):
            sr.read_record(self._lease_slug(pid=pid), root=tmp_path)
        linked = sr.resolve_session_slug_for_pid(pid, root=tmp_path)
        assert linked is not None
        assert sr.read_record(linked, root=tmp_path).session_slug == record_slug


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
def test_resolve_session_pid_falls_back_loudly_to_a_provably_dead_pid(
    env: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        pid = sr.resolve_session_pid(env)

    # Fallback, never a raise -- and never a pid that merely LOOKS resolved.
    # THE assertion: the degraded pid must be one `pid_alive` reports dead,
    # so the lease's staleness AND-guard collapses to its heartbeat half (the
    # documented degradation) rather than to a never-reapable lease. The
    # earlier `os.getsid(0)` fallback failed this: the POSIX session leader is
    # the interactive shell, which outlives a crashed `claude`.
    assert pid == 0
    assert sr.pid_alive(pid) is False
    # LOUD, not silent: the repo's loud-over-silent-degradation norm (INV-2).
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings
    blob = ' '.join(r.getMessage() for r in warnings).lower()
    assert 'degraded' in blob


def test_resolve_session_pid_defaults_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('CLAUDE_PID', '424242')
    assert sr.resolve_session_pid() == 424242


# --- default_lease_slug (task 4248) ----------------------------------------
#
# 3994 moved the lease PID into code because it depended on every skill doc
# getting one shell token right. The SLUG stayed in shell -- and 3994 then made
# it load-bearing, since a mismatch on `lease-heartbeat`/`lease-release` is
# REFUSED. These tests pin the builder that moves it into code too.


def test_default_lease_slug_reproduces_the_measured_production_slug() -> None:
    """The defaulted slug is BYTE-IDENTICAL to what the WATCHER skills built.

    ``watcher-df-1894895`` is the slug measured in the live fleet root on
    2026-08-15 and recorded in ``TestLeaseSlugAndRecordSlugAreDisjoint``. This
    equality is the back-compat oracle FOR THE TWO WATCHER LEASES ONLY --
    ``watcher-<project>`` and ``recon-watcher-<project>``, whose pre-4248 skill
    prescription was already ``<name>-${CLAUDE_PID:-$PPID}``. For those, an
    IN-FLIGHT session that stops passing ``--slug`` keeps its lease with no
    restart, because the CLI re-derives exactly the token it claimed with.

    It does NOT generalise to ``/unblock`` -- see
    ``test_the_unblock_lease_slug_shape_changed_at_4248``, which records that
    asymmetry rather than leaving this docstring to imply it away.
    """
    assert sr.default_lease_slug('watcher-df', {sr.SESSION_PID_ENV: '1894895'}) == 'watcher-df-1894895'
    assert (
        sr.default_lease_slug('recon-watcher-df', {sr.SESSION_PID_ENV: '1894895'})
        == 'recon-watcher-df-1894895'
    )


def test_the_unblock_lease_slug_shape_changed_at_4248() -> None:
    """The `/unblock` lease slug is NOT back-compatible, and that is recorded here.

    The two watcher skills built ``<lease name>-<pid>``, so deriving from the
    name reproduces their token exactly. ``/unblock`` did not: its lease NAME
    is ``unblock-<project>#<TASK_ID>`` (``build_lease_name``'s ``#``) while its
    skill built the slug with a ``-`` separator and no ``#`` at all. Deriving
    from the name therefore yields a DIFFERENT token.

    ROLLOUT CONSEQUENCE, stated so nobody rediscovers it from a lingering
    lease: an `/unblock` session that CLAIMED under the pre-4248 prescription
    and then releases slug-less gets ``result=refused`` -- non-destructive, but
    its lease lingers and falsely reports a holder to the next `/unblock` on
    that task until the 2h ``LEASE_HEARTBEAT_TTL`` ages it out. Such a session
    must either keep passing its ORIGINAL ``--slug`` on release or accept the
    TTL wait; ``skills/unblock/SKILL.md`` says so at the release step. Sessions
    that claim after 4248 are unaffected -- both ends derive the same token.
    """
    name = sr.build_lease_name('unblock', 'df', '2085')
    derived = sr.default_lease_slug(name, {sr.SESSION_PID_ENV: '1894895'})

    assert derived == 'unblock-df#2085-1894895'
    # The exact string the pre-4248 SKILL.md prescribed, for contrast.
    assert derived != 'unblock-df-2085-1894895'


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
def test_default_lease_slug_refuses_to_derive_when_the_session_pid_is_unresolvable(
    env: dict[str, str],
) -> None:
    """An underivable slug is None -- and specifically NOT ``f'{name}-0'``.

    Same parametrize set as ``resolve_session_pid``'s degradation sweep, on
    purpose: these are exactly the envs where that function degrades to pid 0.

    THE assertion is the second one. ``resolve_session_pid`` may degrade to 0
    because a pid is a LIVENESS probe and 0 is provably dead, which keeps a
    crashed holder's lease reapable. A slug is an IDENTITY, and ``{name}-0``
    is not unique: two concurrently-degraded sessions contending the same lease
    name would compute an IDENTICAL slug, so each would pass the other's
    ``_is_lease_owner`` check and could heartbeat or release the other's lease
    -- 3994 defect 1 ("any caller may evict/refresh any lease") reopened on the
    degraded path. Refusing to synthesize a token it cannot make unique is the
    only answer that keeps the ownership guard honest; the CLI then exits 2 and
    names ``--slug`` as the escape hatch.
    """
    assert sr.default_lease_slug('watcher-df', env) is None
    assert sr.default_lease_slug('watcher-df', env) != 'watcher-df-0'


def test_default_lease_slug_defaults_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('CLAUDE_PID', '424242')
    assert sr.default_lease_slug('watcher-df') == 'watcher-df-424242'


def test_default_lease_slug_passes_a_task_scoped_name_through_verbatim() -> None:
    """``build_lease_name``'s ``#`` separator survives into the slug untouched.

    A lease slug is never a filesystem path: ``lease_path_for_name`` sanitizes
    only the NAME, and ``sanitize_slug`` applies only to session-RECORD slugs.
    A lease slug lives inside the lease's JSON body and is compared for
    equality alone, so the task-scoped ``unblock-df#2085`` needs no escaping.
    """
    name = sr.build_lease_name('unblock', 'df', '2085')
    assert name == 'unblock-df#2085'
    assert sr.default_lease_slug(name, {sr.SESSION_PID_ENV: '55'}) == 'unblock-df#2085-55'


def test_a_degraded_pid_lease_still_ages_out_and_is_reaped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: an unresolvable CLAUDE_PID must not make a lease immortal.

    The whole point of returning 0 rather than a durable-but-unrelated pid:
    both staleness rules need a DEAD pid AND an aged heartbeat, so a fallback
    pid that stays alive (the interactive shell) would make a crashed holder's
    lease permanently unreapable — recoverable only by a human `--force`.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.delenv('CLAUDE_PID', raising=False)

    sr.main(['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-degraded'])
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    assert json.loads(lease_path.read_text())['pid'] == 0

    # Fresh: NOT reaped — the heartbeat half of the guard still protects it.
    _set_mtime(lease_path, _NOW, timedelta(seconds=60))
    assert sr.reap_stale_leases(root=tmp_path, now=_NOW) == []
    assert lease_path.is_file()

    # Aged past the TTL: reaped, exactly as pre-3994 heartbeat-only behaviour.
    _set_mtime(lease_path, _NOW, sr.LEASE_HEARTBEAT_TTL + timedelta(minutes=1))
    reaped = sr.reap_stale_leases(root=tmp_path, now=_NOW)

    assert [r.reason for r in reaped] == ['stale_pid']
    assert not lease_path.exists()


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


def test_main_lease_claim_without_slug_writes_the_derived_lease_slug(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--slug` is now optional on lease-claim and defaults like `--pid` does.

    Same shape as the `--pid` pair above, on purpose: the two CLI-owned tokens
    should read as ONE pattern rather than two conventions.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    # Deliberately NOT this process's own pid, so an env-blind implementation
    # (e.g. one reaching for os.getpid()) cannot pass by coincidence.
    monkeypatch.setenv('CLAUDE_PID', str(_DEAD_PID))

    rc = sr.main(['lease-claim', '--name', 'watcher-df'])

    assert rc == 0
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    assert (
        sr.LeaseHolder.from_json(lease_path.read_text()).session_slug
        == f'watcher-df-{_DEAD_PID}'
    )


def test_main_lease_claim_derives_the_body_pid_and_the_slug_pid_from_one_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A no-flag claim's IDENTITY pid and its LIVENESS pid come from ONE resolution.

    The slug embeds a pid and the lease body records one. If each were resolved
    independently they would agree only because both happen to call the same
    function today: any later change making `resolve_session_pid`
    non-deterministic (a cache, a fallback probe, a re-read of a mutated env)
    would silently desynchronise the identity token from the pid
    `holder_liveness` probes — a lease whose slug names a different session than
    its own liveness check. So `main()` resolves once and feeds both, and this
    pins the resulting equality rather than the implementation.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(_DEAD_PID))

    rc = sr.main(['lease-claim', '--name', 'watcher-df'])

    assert rc == 0
    holder = sr.LeaseHolder.from_json(sr.lease_path_for_name('watcher-df', root=tmp_path).read_text())
    assert holder.pid == int(holder.session_slug.rsplit('-', 1)[1])
    assert holder.pid == _DEAD_PID


def test_an_explicit_pid_does_not_satisfy_an_underivable_slug(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--pid` is the body's LIVENESS pid, not this session's IDENTITY.

    The two CLI-owned tokens read as one pattern in the skills ("the CLI owns
    both"), so an operator on the degraded path may reach for the documented
    `--pid` and expect it to unblock a slug-less claim. It deliberately does
    not, and this pins that: `lease-heartbeat`/`lease-release` have no `--pid`
    at all, so a slug derived from one would be derivable by the CLAIM alone —
    an identity the session's own later release could not reproduce, which is
    the exact drift task 4248 removes. `--slug` is the escape hatch precisely
    because all three verbs honour it. The refusal must SAY so, or the operator
    just retries with a bigger `--pid`.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.delenv('CLAUDE_PID', raising=False)

    with pytest.raises(SystemExit) as excinfo:
        sr.main(['lease-claim', '--name', 'watcher-df', '--pid', str(_DEAD_PID)])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert '--pid' in err
    assert '--slug' in err
    assert not sr.lease_path_for_name('watcher-df', root=tmp_path).exists()


def test_main_lease_claim_explicit_slug_still_overrides_the_derived_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(os.getpid()))

    rc = sr.main(['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-OPERATOR'])

    assert rc == 0
    # An explicit --slug remains a deliberate operator override, never shadowed.
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    assert sr.LeaseHolder.from_json(lease_path.read_text()).session_slug == 'watcher-df-OPERATOR'


# --- lease-claim's `slug=` diagnostic line (task 4248) ---------------------
#
# ADDITIVE and LAST, under 3994's discipline: `decision=` stays line 1 and the
# human message line 2, so a not-yet-reloaded skill parser reading only the
# first two lines is unaffected. What the line is FOR, now that the skills no
# longer pass `--slug`: it makes the CLI-derived identity LEGIBLE, so an agent
# can cross-check it against `lease-show`'s holder_slug when a heartbeat comes
# back `result=refused`. It is a diagnostic, NOT a value to thread into the
# next call -- a shell variable cannot survive to the next Claude Code Bash
# tool call, which is exactly why the derivation had to move into the CLI.


def test_main_lease_claim_prints_the_derived_slug_last_on_the_acquired_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(_DEAD_PID))

    rc = sr.main(['lease-claim', '--name', 'watcher-df'])

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == 'decision=acquired'
    assert lines[2] == 'holder_liveness=none'
    assert lines[3] == f'slug=watcher-df-{_DEAD_PID}'
    # Task 4237 appended `holder_record=` after it (pinned in
    # test_session_record_linkage.py); `slug=` keeps its index.
    assert len(lines) == 5


def test_main_lease_claim_slug_line_names_this_caller_not_the_holder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """On a stand-down the `slug=` line is about US; `holder_liveness=` is about THEM.

    Confusing the two would make the line actively misleading in the one
    situation it exists for — reconciling a refusal against `lease-show`.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    _claim_via_cli(slug='watcher-df-OTHER', pid=_DEAD_PID)
    capsys.readouterr()
    monkeypatch.setenv('CLAUDE_PID', str(os.getpid()))

    rc = sr.main(['lease-claim', '--name', 'watcher-df'])

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == 'decision=stand-down'
    # DETERMINISTIC, not "either value": the holder above was seeded with
    # `_DEAD_PID`, which `test_main_lease_claim_reports_orphaned_for_a_dead_holder_pid`
    # already pins as `orphaned`. Accepting `held` too would buy nothing and
    # would read as if the outcome were nondeterministic.
    assert lines[2] == 'holder_liveness=orphaned'
    assert lines[3] == f'slug=watcher-df-{os.getpid()}'
    assert 'watcher-df-OTHER' not in lines[3]


def test_main_lease_claim_prints_the_slug_line_even_when_it_fails_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fail-open path keeps its 3994 silence about the HOLDER, but not about US.

    `holder_liveness=` stays ABSENT on a substrate fault, deliberately: we know
    nothing about a holder and must not assert one either way. The slug is a
    different fact — it is this caller's own derived identity, which the fault
    did not make unknown — so it is still emitted, and is in fact most useful
    here, where the claim's outcome is least certain.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(_DEAD_PID))

    def _boom(*_args: object, **_kwargs: object) -> sr.LeaseClaim:
        raise OSError('lease substrate on fire')

    monkeypatch.setattr(sr, 'claim_lease', _boom)

    with caplog.at_level(logging.ERROR):
        rc = sr.main(['lease-claim', '--name', 'watcher-df'])

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == 'decision=proceed'
    assert lines[2] == f'slug=watcher-df-{_DEAD_PID}'
    assert not any(line.startswith('holder_liveness=') for line in lines)


def test_main_lease_claim_echoes_an_explicit_slug_verbatim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(_DEAD_PID))

    rc = sr.main(['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-OPERATOR'])

    assert rc == 0
    # The line reports the slug ACTUALLY claimed with, not the one that would
    # have been derived — otherwise it would misreport the override path.
    assert capsys.readouterr().out.splitlines()[3] == 'slug=watcher-df-OPERATOR'


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


@pytest.mark.parametrize('verb', ['lease-claim', 'lease-heartbeat', 'lease-release'])
def test_main_lease_verbs_refuse_loudly_when_the_slug_cannot_be_derived(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    verb: str,
) -> None:
    """An underivable slug FAILS LOUDLY — it never drifts to a colliding token.

    This is the half of task 4248 that is not "move it into the CLI": having
    moved it, the CLI must refuse rather than synthesize. `{name}-0` would be
    IDENTICAL for two concurrently-degraded sessions, so each would pass the
    other's `_is_lease_owner` check — 3994 defect 1 reopened. Exit 2 is not a
    regression: pre-4248 a slug-less lease verb ALSO exited 2, so the
    underivable case keeps its exit code and only the derivable case changed.

    The stderr assertions matter as much as the code: an operator who is
    refused must also be told the escape hatch, not merely told no.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(os.getpid()))
    _claim_via_cli()
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    original_body = lease_path.read_text()
    capsys.readouterr()

    monkeypatch.delenv('CLAUDE_PID', raising=False)
    with pytest.raises(SystemExit) as excinfo:
        sr.main([verb, '--name', 'watcher-df'])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert 'CLAUDE_PID' in err
    assert '--slug' in err
    # Nothing was mutated on the way out.
    assert lease_path.read_text() == original_body


def test_the_read_only_lease_verbs_stay_usable_when_the_slug_cannot_be_derived(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """DIAGNOSIS must survive the degraded path the exit-2 refusal creates.

    `lease-show` and `lease-reap` are exactly what a refused or degraded
    operator is sent to next — `_run_lease_mutation`'s `result=refused` message
    ends "inspect with `lease-show --name <name>`", and both watcher SKILLs say
    the same — so the guard must cover only the three SLUG-BEARING verbs.
    Nothing else pins that:
    the guard is scoped by a hardcoded verb tuple, and widening it (or hoisting
    it above the verb check) would leave every new test green while removing
    the operator's only way to see WHY they were refused.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(os.getpid()))
    _claim_via_cli()
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    capsys.readouterr()

    monkeypatch.delenv('CLAUDE_PID', raising=False)

    # Read-only inspection: still prints its normal key=value report.
    assert sr.main(['lease-show', '--name', 'watcher-df']) == 0
    shown = dict(line.split('=', 1) for line in capsys.readouterr().out.splitlines())
    assert shown['name'] == 'watcher-df'
    assert shown['holder_slug'] == 'watcher-df-100'

    # The sweep: also unguarded, and a live holder is still not reaped.
    assert sr.main(['lease-reap']) == 0
    assert lease_path.is_file()


def test_an_explicit_slug_is_still_the_escape_hatch_under_an_unresolvable_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The refusal above is escapable, and does not break the degraded-pid path.

    `test_a_degraded_pid_lease_still_ages_out_and_is_reaped` depends on being
    able to CLAIM a lease with an unresolvable `CLAUDE_PID` (recording pid 0, so
    a crashed holder still ages out). Refusing to DERIVE a slug must not refuse
    to ACCEPT one — the operator supplies a stable token and the pid degrades
    exactly as 3994 designed.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.delenv('CLAUDE_PID', raising=False)

    rc = sr.main(['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-degraded'])

    assert rc == 0
    holder = sr.LeaseHolder.from_json(sr.lease_path_for_name('watcher-df', root=tmp_path).read_text())
    assert holder.session_slug == 'watcher-df-degraded'
    assert holder.pid == 0


# `test_main_lease_mutating_verbs_require_slug` lived here until task 4248.
# It asserted that OMITTING `--slug` exits 2 — true only while the token had
# to be spelled in shell. It is superseded, not dropped: its surviving half
# (an UNDERIVABLE slug exits 2 rather than defaulting to a colliding token) is
# now pinned across all THREE lease verbs by
# `test_main_lease_verbs_refuse_loudly_when_the_slug_cannot_be_derived`, and
# the ownership guarantee it was really protecting is pinned directly by
# `..._still_refuse_a_stranger_with_a_derived_slug` below. It was also quietly
# env-dependent: it never unset `CLAUDE_PID`, so post-4248 it would pass or
# fail on ambient environment rather than on behaviour.


@pytest.mark.parametrize('verb', ['lease-heartbeat', 'lease-release'])
def test_main_lease_mutating_verbs_derive_the_owners_slug_when_it_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    verb: str,
) -> None:
    """THE POINT of task 4248: the CLI re-derives the same slug across invocations.

    Claim and mutate are separate Claude Code Bash tool calls in production —
    separate `/bin/bash -c` processes — so nothing can be carried between them.
    Both derive from `$CLAUDE_PID`, so the owner matches itself without ever
    spelling the token in shell.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(_DEAD_PID))
    sr.main(['lease-claim', '--name', 'watcher-df'])
    capsys.readouterr()

    rc = sr.main([verb, '--name', 'watcher-df'])

    assert rc == 0
    assert capsys.readouterr().out.splitlines()[0] == 'result=applied'


@pytest.mark.parametrize('verb', ['lease-heartbeat', 'lease-release'])
def test_main_lease_mutating_verbs_still_refuse_a_stranger_with_a_derived_slug(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    verb: str,
) -> None:
    """THE GUARD TEST: defaulting `--slug` is a CALLER-DERIVED default, not a fallback.

    3994's `required=True` comment reads "a silent no-slug fallback would
    preserve the 'any caller may evict/refresh any lease' defect indefinitely".
    This pins that the defect stays fixed. The defect 3994 closed was that both
    verbs mutated UNCONDITIONALLY — no ownership check ran at all. The check
    still runs; it now compares against a slug derived from THIS caller's own
    `$CLAUDE_PID`, so a DIFFERENT session derives a DIFFERENT slug and is
    refused, with the lease body untouched. What 4248 retires is only the
    requirement that the token be spelled in shell — which is what let it drift.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(_DEAD_PID))
    sr.main(['lease-claim', '--name', 'watcher-df'])
    capsys.readouterr()
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    original_body = lease_path.read_text()

    # A different session: same lease name, its own (different) session pid.
    monkeypatch.setenv('CLAUDE_PID', str(_DEAD_PID - 1))
    rc = sr.main([verb, '--name', 'watcher-df'])

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == 'result=refused'
    # The human line names the REAL holder — the first session's derived slug.
    assert f'watcher-df-{_DEAD_PID}' in lines[1]
    assert lease_path.read_text() == original_body


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


def _assert_age_between_clock_reads(
    reported_secs: str, *, backdated_by: timedelta, before: datetime, after: datetime
) -> None:
    """lease-show's age is its own clock read, taken somewhere in [before, after], minus an mtime of before - backdated_by.

    The 1 s of upper slack covers filesystem mtime granularity.
    """
    reported = timedelta(seconds=int(reported_secs))
    assert backdated_by <= reported <= backdated_by + (after - before) + timedelta(seconds=1)


def test_main_lease_show_prints_machine_readable_fields_for_a_held_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    _claim_via_cli()
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    before = datetime.now(UTC)
    _set_mtime(lease_path, before, timedelta(hours=1))
    capsys.readouterr()

    rc = sr.main(['lease-show', '--name', 'watcher-df'])
    after = datetime.now(UTC)

    assert rc == 0
    fields = _parse_kv(capsys.readouterr().out)
    assert fields['name'] == 'watcher-df'
    assert fields['state'] == 'held'
    assert fields['holder_slug'] == 'watcher-df-100'
    assert fields['holder_pid'] == str(os.getpid())
    assert fields['holder_pid_alive'] == 'true'
    assert fields['reclaimable'] == 'false'
    # The freshness clock `cat <lease>` cannot show, replacing `stat -c %y`.
    _assert_age_between_clock_reads(
        fields['heartbeat_age_secs'], backdated_by=timedelta(hours=1), before=before, after=after
    )
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
    before = datetime.now(UTC)
    _set_mtime(lease_path, before, timedelta(hours=1))

    rc = sr.main(['lease-show', '--name', 'watcher-df'])
    after = datetime.now(UTC)

    assert rc == 0
    fields = _parse_kv(capsys.readouterr().out)
    # Marks the body unreadable rather than asserting a fake holder -- and
    # still reports a REAL heartbeat age, the axis that decides reapability.
    assert fields['state'] == 'unreadable'
    _assert_age_between_clock_reads(
        fields['heartbeat_age_secs'], backdated_by=timedelta(hours=1), before=before, after=after
    )
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
# Project-token canonicalization (task 3807)
# ---------------------------------------------------------------------------

def test_solar_challenge_alias_was_decided_and_declined() -> None:
    """The naming mismatch is a DECIDED, standing state -- not a pending gap.

    Folding merges solar-challenge's two filed spellings into one bucket --
    an improvement, since a reap scoped to either previously missed the
    other (re-measured 2026-09-07 over 748 records: 3 OPEN under
    ``solar-challenge``, 2 under ``solar_challenge``, ZERO under
    ``my_solar_challenge``; unchanged from 2026-08-07). But that bucket is
    ``solar_challenge``, while
    ``/home/leo/src/solar-challenge/dark-factory-orchestrator.yaml`` declares
    ``my_solar_challenge``, and no alias bridges them. So a reaper passing
    the config-declared token matches ZERO of those 5 rows.

    Task 3813 was the decision task for that bridge, and it DECLINED it --
    chiefly because the fold left no split to heal (an alias would rename a
    populated bucket onto an empty one) and because the identity question is
    an open human gate in that project, marked "Do NOT auto-act". See
    ``dr.PROJECT_TOKEN_ALIASES_DECLINED`` for the full evidence.

    So this is no longer "pending"; the assertions below pin a settled
    outcome. They keep their fail-loudly-on-promotion property -- adding the
    alias breaks the last two -- which
    test_project_token_alias_declines_registry_shape now also enforces from
    the other side, via the disjointness of the two tables. The skills'
    ``--project`` guidance carries the same, now-permanent caveat for the
    humans and watchers that read it, and both CLI verbs warn if you pass
    the config-declared token.
    """
    assert fold_project_token('solar-challenge') == 'solar_challenge'
    assert fold_project_token('solar-challenge') == fold_project_token(
        'solar_challenge'
    )
    assert fold_project_token('my_solar_challenge') != fold_project_token(
        'solar-challenge'
    )
    assert 'solar_challenge' not in PROJECT_ID_ALIASES


def test_project_token_aliases_maps_folded_to_folded() -> None:
    """Every PROJECT_ID_ALIASES key AND value must already be canonical
    under the fold, or a lookup would miss (key) or emit a non-canonical
    token (value) -- the alias map is applied AFTER folding, never before.
    """
    assert PROJECT_ID_ALIASES['df'] == 'dark_factory'
    for alias, canonical in PROJECT_ID_ALIASES.items():
        assert fold_project_token(alias) == canonical
        assert fold_project_token(canonical) == canonical


def test_write_record_creates_missing_nested_dir_and_round_trips(tmp_path):
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


# ---------------------------------------------------------------------------
# Bare-shell entrypoint registry + mutation matrix (task 4027)
# ---------------------------------------------------------------------------
#
# Production runs orchestrator modules from plain shell scripts with NO venv
# and no install, under MORE THAN ONE environment.  Each such invocation is
# registered below as a ROW, and `_MUST_BE_REJECTED` mutation-tests those rows
# so the guard is observed FAILING rather than merely observed green.
#
# Register a new bare-shell entrypoint as a ROW; do not write another ad-hoc
# test.  The rows are NOT interchangeable — see
# `test_tier1_is_strictly_stronger_than_tier2` below.

#: The directory that plays the role of ``orchestrator/src`` — i.e. the PARENT
#: of the ``orchestrator`` package.  A mutation run substitutes a sandbox for
#: it (see ``_inject_import_sandbox``), which is why every path below is
#: expressed relative to this rather than hardcoded.
_SR_PKG_ROOT = _SR_REPO_ROOT / 'orchestrator' / 'src'

#: The one repo-relative ``PYTHONPATH`` entry the hook shells export.  Any row
#: entry that RESOLVES to the same directory (``_row_env`` compares resolved
#: paths, not strings) is rewritten to the sandbox root during a mutation run,
#: so the row measures the MUTATED copy rather than the real tree.
_SR_PKG_ROOT_ENTRY = 'orchestrator/src'


class _EntrypointShape(StrEnum):
    """How a bare-shell caller invokes an entrypoint. The shapes differ.

    ``SCRIPT`` puts only the script's OWN directory on ``sys.path``; ``IMPORT``
    resolves off ``PYTHONPATH``.  A module can satisfy one and break the other,
    so a row records which one its callers actually use.
    """

    SCRIPT = 'script'
    """``python /abs/path/to/module.py <argv...>``"""

    IMPORT = 'import'
    """``python -c 'import <dotted.name>'``"""


@dataclasses.dataclass(frozen=True)
class _BareShellEntrypoint:
    """One production invocation of an orchestrator module without a venv.

    Attributes:
        name: human label, used verbatim as the pytest parametrization id.
        path: absolute path of the entrypoint, under ``_SR_PKG_ROOT``.
        pythonpath: repo-relative ``PYTHONPATH`` entries.  ``()`` means the key
            is ABSENT from the env — NOT set to ``''``.  CPython reads an empty
            entry as "the current directory", which would silently put the
            neutral cwd on ``sys.path`` and change what the row measures.
        shape: ``SCRIPT`` or ``IMPORT`` (see ``_EntrypointShape``).
        argv: argv tail for a ``SCRIPT`` row; unused by ``IMPORT`` rows.
            ``('--help',)`` exercises the module import PLUS ``_build_parser()``
            construction — the two things a bad module-scope import breaks —
            and exits 0 without touching ``~/.claude/fleet``.
        probe: an import that MUST FAIL under THIS row's own env.  If it
            succeeds, the row cannot discriminate a compliant module from a
            violating one, and the mutation test skips loudly rather than
            passing vacuously.  Per-row rather than global because the rows
            forbid different sets (tier-1 forbids strictly more than tier-2).
            It is a general health check for the row, NOT a stand-in for the
            payload: ``test_forbidden_import_is_rejected`` additionally requires
            the payload itself to be unreachable under this env, because probe
            and payload can name different modules.
        callers: the shell scripts that make this invocation, so a failure
            names the production path that breaks.
    """

    name: str
    path: Path
    pythonpath: tuple[str, ...]
    shape: _EntrypointShape
    argv: tuple[str, ...]
    probe: str
    callers: str


_BARE_SHELL_ENTRYPOINTS: tuple[_BareShellEntrypoint, ...] = (
    _BareShellEntrypoint(
        name='session_registry.py (tier-2: import with PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'session_registry.py',
        pythonpath=(_SR_PKG_ROOT_ENTRY,),
        shape=_EntrypointShape.IMPORT,
        argv=(),
        probe='from shared import safe_io',
        callers=(
            'downstream Python consumers that import orchestrator.session_registry '
            'with orchestrator/src on PYTHONPATH but no install (T4 verify, T5 '
            'result, T6 hooks, T7 leases). Nothing on that path reaches `shared`; '
            "the module's own bootstrap appends this checkout's shared/src, so only "
            'its stdlib-only shared leaves are importable. This row carries the '
            'assertion formerly made by test_imports_under_a_bare_interpreter_with_'
            'no_workspace_packages (task 3223 step-31).'
        ),
    ),
    _BareShellEntrypoint(
        name='session_registry.py (tier-1: bare shell, NO PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'session_registry.py',
        # THE POINT OF THIS ROW: running a script by absolute path puts only
        # the script's OWN directory on sys.path. Nothing else is reachable
        # from the env — not `shared`, not `orchestrator.<sibling>` — so the
        # only `shared` the module sees is the checkout's shared/src its own
        # bootstrap appends. That is a STRICTLY stronger contract than the
        # tier-2 row above, which the PYTHONPATH entry makes able to reach
        # every intra-orchestrator sibling.
        pythonpath=(),
        shape=_EntrypointShape.SCRIPT,
        argv=('--help',),
        probe='from orchestrator import stop_instruction',
        callers=(
            'skills/spawn/spawn-claude.sh lines 145 (launching), 428 (exit), 596 '
            'and 693 (refresh), 871 (set-display) — `python3 "$SESSION_REGISTRY_PY"` '
            'with no venv, no install and no PYTHONPATH — and the SKILL.md verb '
            'lines that run it the same way. FOUR OF THE FIVE spawn lines end in '
            '`|| true`, so a break here is SILENT: the subprocess simply never '
            'writes record.json and the session goes missing from the fleet.'
        ),
    ),
    _BareShellEntrypoint(
        name='fleet_files.py (tier-1: bare shell, NO PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'fleet_files.py',
        # An import-only module: running it by path executes its module scope
        # and exits 0, which is exactly what a bad module-scope import breaks.
        pythonpath=(),
        shape=_EntrypointShape.SCRIPT,
        argv=(),
        probe='from orchestrator import stop_instruction',
        callers=(
            'session_registry.py (and decision_registry.py) import this module at '
            'module scope as a bare sibling when skills/spawn/spawn-claude.sh, the '
            'watcher SKILL.md verb lines and scripts/sitting/payloads.py run '
            'session_registry.py by absolute path with no venv, no install and no '
            'PYTHONPATH. A non-stdlib import here breaks every one of those verbs.'
        ),
    ),
    _BareShellEntrypoint(
        name='decision_registry.py (tier-1: bare shell, NO PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'decision_registry.py',
        pythonpath=(),
        shape=_EntrypointShape.SCRIPT,
        argv=(),
        probe='from orchestrator import stop_instruction',
        callers=(
            'decision_cli.py imports this module at module scope as a bare sibling, '
            'and session_registry.py imports decision_cli the same way, when '
            'skills/spawn/spawn-claude.sh, the watcher SKILL.md verb lines '
            '(write-decision, reap-decisions, close-decision, reopen-decision, '
            'migrate-decision-projects) and scripts/sitting/payloads.py run '
            'session_registry.py by absolute path with no venv, no install and no '
            'PYTHONPATH. A non-stdlib import here breaks every decision verb.'
        ),
    ),
    _BareShellEntrypoint(
        name='decision_cli.py (tier-1: bare shell, NO PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'decision_cli.py',
        pythonpath=(),
        shape=_EntrypointShape.SCRIPT,
        argv=(),
        probe='from orchestrator import stop_instruction',
        callers=(
            'session_registry.py imports this module at module scope as a bare '
            'sibling and hands it the five decision verbs (write-decision, '
            'reap-decisions, close-decision, reopen-decision, '
            'migrate-decision-projects) when skills/spawn/spawn-claude.sh, the '
            'watcher SKILL.md verb lines and scripts/sitting/payloads.py run '
            'session_registry.py by absolute path with no venv, no install and no '
            'PYTHONPATH. A non-stdlib import here breaks every decision verb.'
        ),
    ),
    _BareShellEntrypoint(
        name='session_hooks.py (tier-2: bare shell + PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'session_hooks.py',
        # A THIRD distinct shape: script-by-absolute-path WITH PYTHONPATH. Not
        # reducible to either row above — the tier-1 row has no PYTHONPATH, and
        # the tier-2 IMPORT row does not put the script's own directory on
        # sys.path. This one is exactly what session-start.sh:25 + :29 does.
        pythonpath=(_SR_PKG_ROOT_ENTRY,),
        shape=_EntrypointShape.SCRIPT,
        argv=('--help',),
        probe='from shared import safe_io',
        callers=(
            'skills/spawn/hooks/{session-start,stop,notification,install-hooks}.sh '
            '— each exports PYTHONPATH="$REPO_ROOT/orchestrator/src" (line 25) and '
            'then runs session_hooks.py by absolute path (line 29). These fire on '
            'EVERY session start / stop / notification in every project, including '
            'hand-launched ones.'
        ),
    ),
    _BareShellEntrypoint(
        name='session_ownership_probes.py (tier-2: import with PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'session_ownership_probes.py',
        pythonpath=(_SR_PKG_ROOT_ENTRY,),
        shape=_EntrypointShape.IMPORT,
        argv=(),
        probe='from shared import safe_io',
        callers=(
            'session_hooks.py imports this module at module scope, and '
            'skills/spawn/hooks/{session-start,stop,notification}.sh run '
            'session_hooks.py by absolute path with PYTHONPATH=orchestrator/src and '
            'no venv. A non-stdlib import here makes session_hooks.py unimportable, '
            'which silently disables every hook.'
        ),
    ),
    _BareShellEntrypoint(
        name='spawn_t3_lane.py (tier-2: bare shell + PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'spawn_t3_lane.py',
        pythonpath=(_SR_PKG_ROOT_ENTRY,),
        shape=_EntrypointShape.SCRIPT,
        argv=('--help',),
        probe='from shared import safe_io',
        callers=(
            'skills/spawn/spawn-claude.sh runs it by absolute path for the t3 lane '
            '(detect, launch, await-started, wait, self-record, pair, status) with '
            'PYTHONPATH="$REPO_ROOT/orchestrator/src" and no venv. A non-stdlib import '
            'here turns every t3-lane spawn into a launcher failure.'
        ),
    ),
    _BareShellEntrypoint(
        name='t3_mcp_client.py (tier-2: import with PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 't3_mcp_client.py',
        pythonpath=(_SR_PKG_ROOT_ENTRY,),
        shape=_EntrypointShape.IMPORT,
        argv=(),
        probe='from shared import safe_io',
        callers=(
            'spawn_t3_lane.py imports this module at module scope, and '
            'skills/spawn/spawn-claude.sh runs spawn_t3_lane.py by absolute path '
            'with PYTHONPATH="$REPO_ROOT/orchestrator/src" and no venv. A non-stdlib '
            'import here makes the lane unimportable, so every t3-lane spawn becomes '
            'a launcher failure.'
        ),
    ),
    _BareShellEntrypoint(
        name='session_self_exit.py (tier-2: import with PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'session_self_exit.py',
        pythonpath=(_SR_PKG_ROOT_ENTRY,),
        shape=_EntrypointShape.IMPORT,
        argv=(),
        probe='from shared import safe_io',
        callers=(
            'session_hooks.py imports this module at module scope, and '
            'skills/spawn/hooks/stop.sh (with its session-start/notification '
            'siblings) runs session_hooks.py by absolute path with '
            'PYTHONPATH=orchestrator/src and no venv. A non-stdlib import here makes '
            'session_hooks.py unimportable, which silently disables every hook.'
        ),
    ),
)

#: Mutation matrix: ``(entrypoint filename, import injected at module scope)``.
#: Each pair MUST be rejected by at least one registry row covering that
#: entrypoint.  This is the load-bearing signal — a guard that has never been
#: observed to fail is not evidence that it guards anything.
_MUST_BE_REJECTED: tuple[tuple[str, str], ...] = (
    # session_registry's own bootstrap puts this checkout's shared/src on
    # sys.path, and its payloads are injected AFTER that (see
    # ``_INJECTION_ANCHORS``), so `shared` itself is reachable here by design.
    # What must still be refused is a shared module with a third-party
    # dependency: config_models fails on its pydantic import.
    ('session_registry.py', 'from shared import config_models'),
    ('session_registry.py', 'from orchestrator import config'),
    ('session_registry.py', 'from orchestrator import stop_instruction'),
    # The dependency-footprint half of the contract: a plain third-party import
    # at module scope. Without it the matrix only ever exercises workspace
    # packages, and nothing observes the rows rejecting `pip install`-ware.
    #
    # `pydantic` rather than `yaml` DELIBERATELY, and it is measured, not
    # assumed: on this host `import yaml` resolves out of the distro's
    # /usr/lib/python3/dist-packages and SUCCEEDS under every row's env (it is
    # not user-site, so PYTHONNOUSERSITE does not remove it), which would make
    # the payload non-discriminating and skip the row instead of pinning it.
    # `pydantic` is absent from the bare interpreter — that absence is the same
    # one that makes `from orchestrator import config` fail below.
    ('session_registry.py', 'import pydantic'),
    # session_hooks.py's contract is the WEAKER tier-2 one: it does
    # `from orchestrator import session_registry` at module scope (line 50) and
    # its callers export PYTHONPATH for exactly that, so `orchestrator.*` is
    # ALLOWED. `shared` and third-party are not: its env puts neither shared/src
    # nor a site-packages on the path, and the payload lands at the
    # `from __future__` anchor, before the session_registry import whose
    # bootstrap would otherwise make `shared` reachable as a side effect.
    ('session_hooks.py', 'from shared import safe_io'),
    ('session_hooks.py', 'import pydantic'),
    ('session_ownership_probes.py', 'from shared import safe_io'),
    ('session_ownership_probes.py', 'import pydantic'),
    ('session_self_exit.py', 'from shared import safe_io'),
    ('session_self_exit.py', 'import pydantic'),
    ('spawn_t3_lane.py', 'from shared import safe_io'),
    ('spawn_t3_lane.py', 'import pydantic'),
    # Lands at the `from __future__` anchor, BEFORE t3_mcp_client's own
    # session_registry import, whose bootstrap is what makes `shared` reachable.
    ('t3_mcp_client.py', 'from shared import safe_io'),
    ('t3_mcp_client.py', 'import pydantic'),
    ('fleet_files.py', 'import pydantic'),
    ('fleet_files.py', 'from orchestrator import stop_instruction'),
    ('decision_registry.py', 'import pydantic'),
    ('decision_registry.py', 'from orchestrator import stop_instruction'),
    ('decision_cli.py', 'import pydantic'),
    ('decision_cli.py', 'from orchestrator import stop_instruction'),
)


def _registry_row(filename: str, shape: _EntrypointShape) -> _BareShellEntrypoint:
    """The unique registry row for ``filename`` invoked as ``shape``.

    Anything needing "the tier-2 env" must read it off the ROW via ``_row_env``
    rather than hand-rolling a dict, so there is exactly ONE definition of each
    row's environment.  Two copies drift: a scrub added to ``_row_env`` later
    (``PYTHONNOUSERSITE`` was one) would reach the row but not the helper that
    claims to characterise it.
    """
    matches = [
        row
        for row in _BARE_SHELL_ENTRYPOINTS
        if row.path.name == filename and row.shape is shape
    ]
    assert len(matches) == 1, (
        f'expected exactly one _BARE_SHELL_ENTRYPOINTS row for {filename} invoked '
        f'as {shape}, found {[row.name for row in matches]}'
    )
    return matches[0]


def _row_env(row: _BareShellEntrypoint, *, pkg_root: Path) -> dict[str, str]:
    """Build ``row``'s scrubbed env, with ``pkg_root`` standing in for src.

    ``pkg_root`` is the real ``orchestrator/src`` for a positive run and a
    mutation sandbox for a rejection run.
    """
    env = {
        'PATH': '/usr/bin:/bin',
        'HOME': os.environ.get('HOME', '/tmp'),
        # Every row, always: a sandbox mirrors the package as SYMLINKS, so
        # bytecode written through one would land in the real source tree.
        'PYTHONDONTWRITEBYTECODE': '1',
        # STRICTER than production, deliberately.  The shells do not set this,
        # but ``~/.local/lib/pythonX.Y/site-packages`` is on ``sys.path`` by
        # default (``site.ENABLE_USER_SITE`` is True on the system interpreter),
        # so one developer's ``pip install --user`` of anything named ``shared``
        # would make these rows accept a module that breaks on every other box —
        # and would silently defeat the vacuity probes below, which measure
        # importability under exactly this env.  Erring toward the stricter
        # environment can only produce a loud false RED, never a false green.
        'PYTHONNOUSERSITE': '1',
    }
    # An empty tuple leaves the key genuinely ABSENT — see the dataclass
    # docstring; PYTHONPATH='' is not the same environment.
    if not row.pythonpath:
        return env

    real_pkg_root = _SR_PKG_ROOT.resolve()
    is_mutation_run = pkg_root.resolve() != real_pkg_root
    entries: list[str] = []
    for entry in row.pythonpath:
        resolved = (_SR_REPO_ROOT / entry).resolve()
        if resolved == real_pkg_root:
            # THE substitution that makes a mutation run measure the sandbox.
            entries.append(str(pkg_root))
            continue
        # Resolve-and-compare rather than string-equality against
        # ``_SR_PKG_ROOT_ENTRY``: a row spelled 'orchestrator/src/',
        # './orchestrator/src', or carrying a second entry that also reaches the
        # package, would otherwise miss the substitution and point a MUTATION run
        # at the real, UNMUTATED tree.  That row would then always exit 0 and
        # quietly stop pinning anything — masked, because
        # ``test_forbidden_import_is_rejected`` only needs ANY row to reject.
        assert not (is_mutation_run and resolved.is_relative_to(real_pkg_root)), (
            f'{row.name}: PYTHONPATH entry {entry!r} resolves inside the real '
            f'{real_pkg_root} but was not substituted for the mutation sandbox '
            f'{pkg_root}, so this run would import the UNMUTATED module and the '
            f'mutation test would measure nothing. Spell it exactly '
            f'{_SR_PKG_ROOT_ENTRY!r} so it is substituted.'
        )
        entries.append(str(resolved))
    env['PYTHONPATH'] = os.pathsep.join(entries)
    return env


def _run_row(
    row: _BareShellEntrypoint, *, pkg_root: Path, tmp_path: Path
) -> subprocess.CompletedProcess[str]:
    """Run ``row``'s production invocation shape against ``pkg_root``."""
    interpreter = _non_venv_interpreter()
    assert interpreter is not None, 'callers skip when no non-venv interpreter exists'

    target = pkg_root / row.path.relative_to(_SR_PKG_ROOT)
    if row.shape is _EntrypointShape.SCRIPT:
        argv = [interpreter, str(target), *row.argv]
    else:
        dotted = '.'.join(target.relative_to(pkg_root).with_suffix('').parts)
        argv = [interpreter, '-c', f'import {dotted}']

    # ``cwd=tmp_path`` here and in ``_probe_discriminates`` is LOAD-BEARING,
    # not tidiness (task 3223 step-31's measured finding, carried forward).
    # Python puts the process's own cwd on sys.path, so a run from the REPO
    # ROOT resolves ``<repo>/shared`` — the wrapper directory, NOT the real
    # ``shared/src/shared`` package — as an empty NAMESPACE package.  The probe's
    # ``shared`` import then succeeds, the test skips, and the contract goes
    # unpinned.  Measured: inert from the repo root, live from ``orchestrator/``.
    # The canonical test_command runs ``cd orchestrator && pytest tests/``, so it
    # only bit an ad-hoc ``pytest orchestrator/tests/...`` from the root — which
    # is exactly what someone checking this one file types.  A neutral cwd holds
    # either way.  (conftest.py's autouse ``_df_no_mock_derived_stray_dirs`` is a
    # second reason never to let a subprocess inherit the suite's cwd.)
    return subprocess.run(
        argv,
        env=_row_env(row, pkg_root=pkg_root),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )


class _BareShellProbes:
    """Session-scoped memo for the subprocess measurements the rows need.

    Two kinds, each a pure function of (row env, subject) and therefore
    identical for every test that asks:

    - ``rejects(row, statement)`` — does ``statement`` FAIL as a bare import
      under ``row``'s own documented env?  A row can only be counted as
      discriminating when this is True; if the interpreter reaches the package
      anyway, the row would accept a violating module too and its green is
      unearned.
    - ``control_run(row, entrypoint)`` — does ``row`` run CLEAN against a
      sandbox with NOTHING injected?

    Memoised because the mutation matrix asks the same questions repeatedly:
    unmemoised, every (matrix entry x covering row) pair spawns its own
    interpreter, so the two session_registry rows are re-probed once per payload
    with identical inputs.  That count grows multiplicatively as rows and
    payloads are added — exactly what a registry design invites — for results
    that cannot differ.
    """

    def __init__(self, cwd: Path) -> None:
        #: Neutral cwd for every subprocess — LOAD-BEARING, see ``_run_row``.
        self._cwd = cwd
        self._rejects: dict[tuple[str, str], bool] = {}
        self._control_sandboxes: dict[str, Path] = {}
        self._control_runs: dict[tuple[str, str], subprocess.CompletedProcess[str]] = {}

    def rejects(self, row: _BareShellEntrypoint, statement: str) -> bool:
        """True when ``statement`` FAILS as a bare import under ``row``'s env."""
        key = (row.name, statement)
        if key not in self._rejects:
            interpreter = _non_venv_interpreter()
            assert interpreter is not None, (
                'callers skip when no non-venv interpreter exists'
            )
            probe = subprocess.run(
                [interpreter, '-c', statement],
                env=_row_env(row, pkg_root=_SR_PKG_ROOT),
                cwd=self._cwd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self._rejects[key] = probe.returncode != 0
        return self._rejects[key]

    def control_run(
        self, row: _BareShellEntrypoint, entrypoint: str
    ) -> subprocess.CompletedProcess[str]:
        """Run ``row`` against a sandbox of ``entrypoint`` with NOTHING injected.

        The control for every rejection.  It proves the symlink farm and the
        source rewrite still produce a RUNNABLE module, so a non-zero exit on a
        mutated copy is attributable to the payload.  Without it, a silently
        degraded sandbox — a missing symlink, an anchor line that lands inside a
        docstring, a syntax-breaking rewrite — makes every covering row exit
        non-zero and every mutation test PASS while measuring nothing, which is
        the same silent-pass mode this registry exists to close.
        """
        if entrypoint not in self._control_sandboxes:
            # Empty payload: same code path, same symlink farm, same rewrite —
            # only the injected line differs from a mutation run.
            self._control_sandboxes[entrypoint] = _inject_import_sandbox(
                self._cwd, entrypoint, ''
            )
        key = (row.name, entrypoint)
        if key not in self._control_runs:
            self._control_runs[key] = _run_row(
                row, pkg_root=self._control_sandboxes[entrypoint], tmp_path=self._cwd
            )
        return self._control_runs[key]


@pytest.fixture(scope='session')
def bare_shell_probes(tmp_path_factory: pytest.TempPathFactory) -> _BareShellProbes:
    """Session-scoped so each (row, statement) pair is measured exactly once."""
    return _BareShellProbes(tmp_path_factory.mktemp('bare-shell-probes'))


#: Intra-orchestrator modules documented stdlib-only for LAYERING reasons
#: (cycle avoidance), NOT for bare-shell executability. Each is therefore
#: importable under the tier-2 env and NOT under tier-1 — which is exactly the
#: difference `test_tier1_is_strictly_stronger_than_tier2` demonstrates.
#:
#: `orchestrator.config` is deliberately NOT in this list. Measured: it fails
#: the TIER-2 env too, because orchestrator/config.py imports pydantic, which
#: the bare interpreter lacks — a dependency-footprint failure, not the
#: layering one being demonstrated. Using it as the probe would make the test
#: tautological and mislead the next reader. It stays in `_MUST_BE_REJECTED`
#: (it must fail tier-1, and does), it just cannot serve as the tier-2-accepts
#: probe.
_BARE_IMPORTABLE_SIBLINGS = ('stop_instruction', 'fm_retry', 'routing')


def _first_bare_importable_sibling(probes: _BareShellProbes) -> str | None:
    """Return the first sibling importable under the tier-2 env, or None.

    Probed at runtime rather than hardcoded: if the chosen module later grows a
    pydantic/yaml dependency it stops being bare-importable, and a hardcoded
    reference would turn its caller into a confusing false failure instead of a
    skip. Probing the list degrades to None only when ALL candidates stop
    qualifying.

    The env comes off the tier-2 ROW itself (see ``_registry_row``), never a
    local copy — this probe exists to characterise what that row accepts, so
    measuring a different environment than the row would silently weaken
    ``test_tier1_is_strictly_stronger_than_tier2``'s premise.
    """
    tier2 = _registry_row('session_registry.py', _EntrypointShape.IMPORT)
    for candidate in _BARE_IMPORTABLE_SIBLINGS:
        if not probes.rejects(tier2, f'from orchestrator import {candidate}'):
            return candidate
    return None


#: The line a payload is injected after, per entrypoint; any other entrypoint
#: uses ``from __future__ import annotations``.  session_registry's,
#: decision_registry's and decision_cli's payloads go after their last shared
#: import, so each is judged with the module's own shared/src bootstrap already
#: applied: that is the environment a later module-scope import in that file
#: really runs in.
_INJECTION_ANCHORS: dict[str, str] = {
    'session_registry.py': 'from shared.session_liveness import ',
    'decision_registry.py': 'from shared.project_id import ',
    'decision_cli.py': 'from shared.project_id import ',
}
_DEFAULT_INJECTION_ANCHOR = 'from __future__ import annotations'


def _inject_after_anchor(source: str, entrypoint_filename: str, payload: str) -> str:
    anchor = _INJECTION_ANCHORS.get(entrypoint_filename, _DEFAULT_INJECTION_ANCHOR)
    lines = source.splitlines(keepends=True)
    anchored = [index for index, line in enumerate(lines) if line.startswith(anchor)]
    assert anchored, (
        f'{entrypoint_filename} has no module-scope line starting `{anchor}` to '
        'anchor the injection on; pick another anchor rather than injecting blindly'
    )
    insert_at = anchored[0] + 1
    return ''.join([*lines[:insert_at], f'{payload}\n', *lines[insert_at:]])


def _inject_import_sandbox(tmp_path: Path, entrypoint_filename: str, payload: str) -> Path:
    """Mirror the checkout layout, with ``payload`` injected into one file.

    Returns the directory that stands in for ``orchestrator/src``.

    The sandbox is laid out like a checkout: the package sits at
    ``<sandbox>/orchestrator/src/orchestrator`` and ``<sandbox>/shared/src``
    links the real ``shared/src``, so a mutated session_registry.py resolves its
    ``__file__``-relative shared bootstrap exactly as the real file does.

    Every entry of the real package is mirrored as a SYMLINK (``__pycache__``
    excepted) and only the mutated entrypoint is a real file.  Faithfulness and
    cost: in production a ``SCRIPT`` row's own directory holds all ~90 sibling
    modules, so a lone-file temp copy would be over-strict and could manufacture
    false rejections; a full ``cp -r`` would copy 11M per mutation.  The symlink
    farm is both faithful and one inode.
    """
    slug = re.sub(r'\W+', '-', f'{entrypoint_filename}-{payload}').strip('-')
    sandbox = tmp_path / f'sandbox-{slug}'
    pkg_root = sandbox / 'orchestrator' / 'src'
    pkg = pkg_root / 'orchestrator'
    pkg.mkdir(parents=True)
    (sandbox / 'shared').mkdir()
    (sandbox / 'shared' / 'src').symlink_to(_SR_REPO_ROOT / 'shared' / 'src')

    real_pkg = _SR_PKG_ROOT / 'orchestrator'
    for entry in real_pkg.iterdir():
        if entry.name == '__pycache__':
            continue
        (pkg / entry.name).symlink_to(entry)

    source = (real_pkg / entrypoint_filename).read_text()
    (pkg / entrypoint_filename).unlink()
    (pkg / entrypoint_filename).write_text(
        _inject_after_anchor(source, entrypoint_filename, payload)
    )
    return pkg_root


class TestStdlibOnlySelfContainment:
    """The bare-shell entrypoints must run with NO venv, install or third-party deps.

    This is an architectural contract, not a style preference.  Two production
    shell paths execute orchestrator modules directly, and they do it under
    DIFFERENT environments — so this class pins a REGISTRY
    (``_BARE_SHELL_ENTRYPOINTS``) of invocations, not one module.  What such a
    module may import is the stdlib plus this checkout's stdlib-only ``shared``
    leaves, and the only route to those is session_registry's own bootstrap,
    which appends the checkout's ``shared/src`` (derived from ``__file__``) to
    ``sys.path``; no caller exports anything for it.

    - **tier-1 — bare shell, NO PYTHONPATH.**  ``skills/spawn/spawn-claude.sh``
      and the SKILL.md verb lines run ``session_registry.py`` by absolute path
      with nothing exported, so only the script's own directory is on
      ``sys.path``: no third-party, no ``orchestrator.<sibling>``, and no
      ``shared`` beyond what the bootstrap appends.
    - **tier-2 — bare shell (or plain import) WITH PYTHONPATH=orchestrator/src.**
      ``skills/spawn/hooks/*.sh`` run ``session_hooks.py``, and Python consumers
      import ``orchestrator.session_registry``.  Intra-orchestrator siblings are
      reachable here; third-party is still not, and ``shared`` only through the
      same bootstrap.

    Tier-1 is STRICTLY STRONGER, so the rows are not interchangeable and neither
    may be deleted as a duplicate of the other — pinned by
    ``test_tier1_is_strictly_stronger_than_tier2``.

    The registry is MUTATION-TESTED (``_MUST_BE_REJECTED``): each forbidden
    import is injected into a sandbox and some covering row must be observed
    REJECTING it, with a ``ModuleNotFoundError`` naming the payload's own
    dependency and against a control run of the SAME sandbox with nothing
    injected — so a degraded sandbox cannot masquerade as a working guard.  A
    guard that has never been observed to fail is not evidence of anything —
    task 3223's consolidation added ``from shared import safe_io`` before any
    bootstrap existed and broke exactly this contract (escalation esc-3223-2)
    while the only symptom in the suite was two downstream
    ``test_session_hooks.py`` failures.  A row counts
    as able to discriminate only when BOTH its own vacuity ``probe`` AND the
    payload under test are unreachable under its documented env; otherwise it
    SKIPS loudly rather than reporting a green it did not earn.

    Adding a bare-shell entrypoint?  Register a ROW plus a matrix entry.  Do not
    write another ad-hoc test — a second one drifts from what the shells run.

    DO NOT WIDEN THIS INTO A REPO-WIDE RULE.  "stdlib-only" names at least three
    different contracts in this repo, and only the first is what these rows pin:

    (a) bare-shell executability — ``session_registry``, ``fleet_files``,
        ``decision_registry``, ``decision_cli``, ``session_hooks``,
        ``session_ownership_probes``, ``session_self_exit``;
    (b) layering / cycle avoidance — ``routing``, ``stop_instruction``,
        ``fm_retry``, ``agents/triage``, ``fused_memory/config``/``schema``,
        where importing an intra-package sibling is perfectly fine;
    (c) dependency footprint — ``memory_eval_limits``, ``evals/report``
        ("no scipy/numpy"), which already import pydantic and
        ``shared.memory_eval_metrics`` by design.

    A blanket "no non-stdlib import in any file whose docstring says
    stdlib-only" check false-positives on (c) and misreads (b).
    """

    def test_every_matrix_entrypoint_has_a_registry_row(self):
        """Every entrypoint named in the mutation matrix is actually registered.

        Without this, a matrix entry naming an unregistered entrypoint makes
        ``test_forbidden_import_is_rejected`` iterate an EMPTY row set — it
        would neither fail nor skip while the contract went unpinned, which is
        precisely the silent-pass failure mode this registry exists to close.
        """
        registered = {row.path.name for row in _BARE_SHELL_ENTRYPOINTS}
        named = {entrypoint for entrypoint, _payload in _MUST_BE_REJECTED}

        assert named <= registered, (
            'these entrypoints are mutation-tested but have no '
            f'_BARE_SHELL_ENTRYPOINTS row: {sorted(named - registered)}. Register '
            'the production invocation (env + shape + vacuity probe + callers) as '
            'a row rather than leaving the mutation unpinned.'
        )

    @pytest.mark.parametrize('row', _BARE_SHELL_ENTRYPOINTS, ids=lambda row: row.name)
    def test_entrypoint_runs_clean_under_documented_env(self, row, tmp_path):
        """Every registered bare-shell invocation succeeds against the REAL tree."""
        if _non_venv_interpreter() is None:
            pytest.skip('no non-venv interpreter available on this host')

        result = _run_row(row, pkg_root=_SR_PKG_ROOT, tmp_path=tmp_path)

        assert result.returncode == 0, (
            f'{row.name} must run with no venv/install. Its callers:\n'
            f'  {row.callers}\n'
            'Remove the offending module-scope import (or, if the dependency is '
            'genuinely required, the caller can no longer run it bare).\n'
            f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
        )

    @pytest.mark.parametrize('flags', [(), ('-I', '-B')], ids=['default', 'isolated'])
    def test_bare_shell_write_decision_folds_project_through_shared(self, tmp_path, flags):
        """The production spawn-claude/SKILL invocation reaches the shared fold.

        ``write-decision`` run by absolute path with no PYTHONPATH stores its
        project folded by ``shared.project_id.fold_project_token``, so the fold
        the cockpit reads with is reachable from a bare shell. The isolated case
        (``-I``, which implies ``-P``) leaves the script's own directory off
        ``sys.path``, so the sibling imports must not depend on it being there.
        """
        interpreter = _non_venv_interpreter()
        if interpreter is None:
            pytest.skip('no non-venv interpreter available on this host')
        fleet = tmp_path / 'fleet'

        result = subprocess.run(
            [
                interpreter,
                *flags,
                str(_SR_PKG_ROOT / 'orchestrator' / 'session_registry.py'),
                'write-decision',
                '--id', 'esc-1-1',
                '--project', 'Dark-Factory',
                '--text', 'Approve the bare-shell filing?',
                '--escalations-dir', str(tmp_path / 'escalations'),
            ],
            env={
                'PATH': '/usr/bin:/bin',
                'HOME': str(tmp_path),
                'PYTHONNOUSERSITE': '1',
                'PYTHONDONTWRITEBYTECODE': '1',
                'CLAUDE_FLEET_ROOT': str(fleet),
            },
            # Neutral cwd, LOAD-BEARING for the reasons ``_run_row`` gives.
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f'stderr:\n{result.stderr}'
        assert [decision.project for decision in dr.list_decisions(root=fleet)] == [
            fold_project_token('Dark-Factory')
        ], f'stderr:\n{result.stderr}'

    _BARE_SHELL_QUEUE = '/a/data/escalations'

    @pytest.mark.parametrize(
        ('argv', 'seeded_state', 'stdout', 'final_state'),
        [
            pytest.param(
                ['close-decision', '--id', 'esc-1-1', '--state', 'answered', '--evidence', EVIDENCE,
                 '--project', 'df', '--escalations-dir', _BARE_SHELL_QUEUE],
                dr.DecisionState.OPEN, 'esc-1-1', dr.DecisionState.ANSWERED,
                id='close-decision',
            ),
            pytest.param(
                ['reopen-decision', '--id', 'esc-1-1', '--project', 'df',
                 '--escalations-dir', _BARE_SHELL_QUEUE],
                dr.DecisionState.DROPPED, 'esc-1-1', dr.DecisionState.OPEN,
                id='reopen-decision',
            ),
            pytest.param(
                ['reap-decisions', '--project', 'df', '--escalations-dir', _BARE_SHELL_QUEUE],
                None, '', None,
                id='reap-decisions',
            ),
            pytest.param(
                ['migrate-decision-projects', '--dry-run'],
                None, '', None,
                id='migrate-decision-projects',
            ),
        ],
    )
    def test_bare_shell_runs_the_other_decision_verbs(
        self,
        tmp_path: Path,
        argv: list[str],
        seeded_state: dr.DecisionState | None,
        stdout: str,
        final_state: dr.DecisionState | None,
    ) -> None:
        """Every SKILL-documented decision verb runs at ``session_registry.py <verb>`` from a bare shell.

        The in-process ``sr.main`` tests import the package, so they cannot see
        a break in the bare sibling imports that only a run by absolute path
        takes. write-decision has its own case above.
        """
        interpreter = _non_venv_interpreter()
        if interpreter is None:
            pytest.skip('no non-venv interpreter available on this host')
        fleet = tmp_path / 'fleet'
        if seeded_state is not None:
            seeded = make_decision(
                id='esc-1-1', escalations_dir=self._BARE_SHELL_QUEUE, state=seeded_state,
                closing_evidence=EVIDENCE if seeded_state != dr.DecisionState.OPEN else '',
            )
            dr.write_decision(seeded, root=fleet)

        result = subprocess.run(
            [interpreter, str(_SR_PKG_ROOT / 'orchestrator' / 'session_registry.py'), *argv],
            env={
                'PATH': '/usr/bin:/bin',
                'HOME': str(tmp_path),
                'PYTHONNOUSERSITE': '1',
                'PYTHONDONTWRITEBYTECODE': '1',
                'CLAUDE_FLEET_ROOT': str(fleet),
            },
            # Neutral cwd, LOAD-BEARING for the reasons ``_run_row`` gives.
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f'stderr:\n{result.stderr}'
        assert result.stdout.strip() == stdout, f'stderr:\n{result.stderr}'
        if final_state is not None:
            assert [decision.state for decision in dr.list_decisions(root=fleet)] == [final_state]

    def test_the_decision_modules_import_only_downward(self):
        """fleet_files <- decision_registry <- decision_cli, and none of them imports session_registry.

        Under tier-1 session_registry is ``__main__``, so a module below it
        that imported it back would silently load a second copy rather than
        fail. Read off every import statement, function-local ones included,
        in each of the relative, ``orchestrator.``-qualified and bare spellings.
        """
        package = _SR_PKG_ROOT / 'orchestrator'

        def is_sibling(name: str) -> bool:
            return (package / f'{name}.py').is_file() or (package / name / '__init__.py').is_file()

        def sibling_imports(module: str) -> set[str]:
            found: set[str] = set()
            for node in ast.walk(ast.parse((package / f'{module}.py').read_text(encoding='utf-8'))):
                if isinstance(node, ast.ImportFrom):
                    head = (node.module or '').split('.')
                    if node.level == 1:
                        found.update([head[0]] if node.module else [alias.name for alias in node.names])
                    elif head[0] == 'orchestrator':
                        found.update([head[1]] if len(head) > 1 else [alias.name for alias in node.names])
                    elif is_sibling(head[0]):
                        found.add(head[0])
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        head = alias.name.split('.')
                        if head[0] == 'orchestrator' and len(head) > 1:
                            found.add(head[1])
                        elif is_sibling(head[0]):
                            found.add(head[0])
            return found

        assert {module: sibling_imports(module) for module in ('fleet_files', 'decision_registry', 'decision_cli')} == {
            'fleet_files': set(),
            'decision_registry': {'fleet_files'},
            'decision_cli': {'decision_registry'},
        }

    def test_loads_by_path_under_a_foreign_dotless_module_name(self, monkeypatch):
        """The third load mode: by path, under a module name outside the package.

        tests/scripts/review_all_program_fixture.py::load_session_registry loads
        this checkout's session_registry.py with ``spec_from_file_location``
        under a dotless name, without putting its directory on ``sys.path``, so
        that a suite takes the registry from this checkout rather than from
        whichever ``orchestrator`` the venv resolves. There ``__package__`` is
        ``''``, so no relative sibling import resolves. The siblings must come
        from this file's own directory too, and that directory must not be left
        on the importer's ``sys.path``.
        """
        name = 'sr_foreign_dotless_load_probe'
        path = _SR_PKG_ROOT / 'orchestrator' / 'session_registry.py'
        assert str(path.parent) not in sys.path
        siblings = ('decision_cli', 'decision_registry', 'fleet_files')
        assert not any(sibling in sys.modules for sibling in siblings)
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        path_before = list(sys.path)

        try:
            spec.loader.exec_module(module)

            assert module.__package__ == ''
            assert sys.path == path_before
            assert {sibling: Path(sys.modules[sibling].__file__ or '') for sibling in siblings} == {
                sibling: path.parent / f'{sibling}.py' for sibling in siblings
            }
            assert module.decision_cli is sys.modules['decision_cli']
            assert module.fleet_root is sys.modules['fleet_files'].fleet_root
        finally:
            for sibling in siblings:
                sys.modules.pop(sibling, None)

    def test_tier1_is_strictly_stronger_than_tier2(self, tmp_path, bare_shell_probes):
        """The two session_registry rows encode DIFFERENT contracts, not one twice.

        They read like duplicates ("both just check that session_registry
        imports"), so a tidy-up could delete the tier-1 row as redundant and
        silently restore the exact hole task 4027 closed. They are not: tier-2
        ACCEPTS an intra-orchestrator sibling import that tier-1 REJECTS, and
        tier-1 is what spawn-claude.sh actually runs.
        """
        if _non_venv_interpreter() is None:
            pytest.skip('no non-venv interpreter available on this host')

        sibling = _first_bare_importable_sibling(bare_shell_probes)
        if sibling is None:
            pytest.skip(
                'no intra-orchestrator sibling is currently bare-importable under '
                'the tier-2 env, so the two tiers cannot be told apart by this '
                'probe (they still differ — see the row definitions)'
            )

        rows = {
            row.shape: row
            for row in _BARE_SHELL_ENTRYPOINTS
            if row.path.name == 'session_registry.py'
        }
        assert set(rows) == {_EntrypointShape.SCRIPT, _EntrypointShape.IMPORT}, (
            'session_registry.py must keep BOTH a SCRIPT row (tier-1, no '
            'PYTHONPATH — what spawn-claude.sh runs) and an IMPORT row (tier-2, '
            f'what Python consumers do). Present: {sorted(rows)}'
        )

        sandbox = _inject_import_sandbox(
            tmp_path, 'session_registry.py', f'from orchestrator import {sibling}'
        )
        tier2 = _run_row(rows[_EntrypointShape.IMPORT], pkg_root=sandbox, tmp_path=tmp_path)
        tier1 = _run_row(rows[_EntrypointShape.SCRIPT], pkg_root=sandbox, tmp_path=tmp_path)

        both_encode_different_contracts = (
            'The two session_registry rows are NOT duplicates and neither may be '
            'deleted as a copy of the other: tier-2 (import with PYTHONPATH) '
            'tolerates intra-orchestrator siblings, tier-1 (bare shell, no '
            'PYTHONPATH) does not, and tier-1 is the one spawn-claude.sh runs.\n'
            f'payload: from orchestrator import {sibling}\n'
        )
        assert tier2.returncode == 0, (
            f'{both_encode_different_contracts}'
            f'tier-2 was expected to ACCEPT it but exited {tier2.returncode}. If '
            f'`{sibling}` has grown a non-stdlib dependency it is no longer a valid '
            f'probe — extend _BARE_IMPORTABLE_SIBLINGS.\nstderr:\n{tier2.stderr}'
        )
        assert tier1.returncode != 0, (
            f'{both_encode_different_contracts}'
            'tier-1 was expected to REJECT it but exited 0, which means the tier-1 '
            'row is no longer measuring the no-PYTHONPATH environment.\n'
            f'stdout:\n{tier1.stdout}'
        )

    @pytest.mark.parametrize(('entrypoint', 'payload'), _MUST_BE_REJECTED)
    def test_forbidden_import_is_rejected(
        self, entrypoint, payload, tmp_path, bare_shell_probes
    ):
        """Inject a forbidden import and demand that some covering row REJECTS it.

        This is the signal that the rows above discriminate at all. A guard that
        has never been observed to fail is not evidence of anything.
        """
        if _non_venv_interpreter() is None:
            pytest.skip('no non-venv interpreter available on this host')

        covering = [row for row in _BARE_SHELL_ENTRYPOINTS if row.path.name == entrypoint]
        assert covering, (
            f'no _BARE_SHELL_ENTRYPOINTS row covers {entrypoint}, so the loop below '
            'would iterate zero rows and this test would PASS while the contract '
            'went unpinned — the exact silent-pass failure mode the registry exists '
            'to close. Register a row for it.'
        )

        # Discrimination is judged against THIS payload, not the row's fixed
        # `probe` alone.  The two can be about different modules, and then a row
        # is counted as discriminating on a probe unrelated to what it is being
        # asked to reject.  Concretely: the tier-1 row's probe is `from
        # orchestrator import stop_instruction` (no PYTHONPATH, so it fails on
        # essentially any host), while one payload it must reject is `from shared
        # import safe_io`.  On a host where `shared` IS reachable by the bare
        # interpreter, tier-2 is correctly filtered out on its own probe, tier-1
        # survives the filter on an unrelated one, and the mutated tier-1 run then
        # exits 0 — a hard RED reporting a fact about the HOST rather than about
        # the code.  The honest outcome there is the skip below.  Both conditions
        # are required: the row probe guards the row in general, the payload
        # guards this parametrization in particular.
        discriminating = [
            row
            for row in covering
            if bare_shell_probes.rejects(row, row.probe)
            and bare_shell_probes.rejects(row, payload)
        ]
        if not discriminating:
            pytest.skip(
                f'every registry row covering {entrypoint} can already import '
                f'`{payload}` (or its own vacuity probe) under its documented env '
                '(workspace or third-party packages reachable by the bare '
                'interpreter?), so none of them can discriminate a compliant module '
                'from a violating one'
            )

        # CONTROL BEFORE CONCLUSION: the identical sandbox with NOTHING injected
        # must run clean, or a non-zero exit on the mutated copy says nothing
        # about the payload.
        for row in discriminating:
            control = bare_shell_probes.control_run(row, entrypoint)
            assert control.returncode == 0, (
                f'the UNMUTATED sandbox of {entrypoint} already fails under '
                f'{row.name} (rc={control.returncode}), so any rejection below would '
                'be attributable to the sandbox rather than to the injected import. '
                '_inject_import_sandbox has degraded — a missing symlink, or an '
                'anchor line that no longer sits at module scope.\n'
                f'stderr:\n{control.stderr}'
            )

        sandbox = _inject_import_sandbox(tmp_path, entrypoint, payload)
        outcomes = [
            (row, _run_row(row, pkg_root=sandbox, tmp_path=tmp_path))
            for row in discriminating
        ]

        detail = '\n'.join(
            f'- {row.name}: rc={result.returncode}\n'
            f'  stderr: {result.stderr.strip()[:400]}'
            for row, result in outcomes
        )
        # ``ModuleNotFoundError``, not merely ``rc != 0``: the rejection must be
        # caused by the module the payload names — directly, or transitively (`from
        # orchestrator import config` fails on config.py's own pydantic import under
        # tier-2).  A SyntaxError or a crash also exits non-zero, so accepting any
        # non-zero exit would let a broken injection masquerade as a working guard.
        rejected = [
            row.name
            for row, result in outcomes
            if result.returncode != 0 and 'ModuleNotFoundError' in result.stderr
        ]
        assert rejected, (
            f'{entrypoint} with `{payload}` injected at module scope was not rejected '
            'for the right reason by any registry row covering it: each row either '
            'ACCEPTED it — nothing pins that part of the contract, so add the row '
            'whose environment rejects it — or failed WITHOUT a ModuleNotFoundError, '
            'meaning the injection broke the module instead of exercising the import '
            'guard.\n'
            f'{detail}'
        )


# ---------------------------------------------------------------------------
# Task 7489: the T3 lane's registry primitives
# ---------------------------------------------------------------------------


def test_main_set_display_stamps_a_t3_thread_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_env(monkeypatch, _launching_env(tmp_path))
    sr.main(['launching'])
    record_dir = capsys.readouterr().out.strip()

    rc = sr.main(
        ['set-display', '--record', record_dir, '--kind', 't3', '--t3-thread-id', 'th-9',
         '--wm-title', 'unblock:df#2085']
    )

    assert rc == 0
    display = sr.read_record(Path(record_dir).name, root=tmp_path).display
    assert display == sr.Display(kind='t3', wm_title='unblock:df#2085', t3_thread_id='th-9')


_SID = '0b6a3c1e-1111-4222-8333-444455556666'


def _bound_record(tmp_path: Path, slug: str, session_id: str | None) -> sr.SessionRecord:
    record = _make_record(session_slug=slug, claude_session_id=session_id)
    sr.write_record(record, root=tmp_path)
    return record


class TestClaudeSessionPointers:
    def test_pointer_dir_is_a_sibling_of_sessions(self, tmp_path: Path) -> None:
        assert sr.claude_session_pointers_dir(tmp_path) == tmp_path / 'sessions-by-claude-id'
        assert sr.claude_session_pointer_path(_SID, tmp_path).parent.parent == tmp_path

    def test_a_written_pointer_resolves_while_the_record_vouches(self, tmp_path: Path) -> None:
        _bound_record(tmp_path, 'session-df-7489-1', _SID)

        assert sr.write_claude_session_pointer(_SID, 'session-df-7489-1', root=tmp_path)
        assert sr.claude_session_pointer_path(_SID, tmp_path).read_text() == 'session-df-7489-1'
        assert sr.resolve_session_slug_for_claude_session_id(_SID, root=tmp_path) == (
            'session-df-7489-1'
        )

    def test_a_record_bound_to_another_session_does_not_vouch(self, tmp_path: Path) -> None:
        _bound_record(tmp_path, 'session-df-7489-1', 'some-other-session')
        sr.write_claude_session_pointer(_SID, 'session-df-7489-1', root=tmp_path)

        assert sr.resolve_session_slug_for_claude_session_id(_SID, root=tmp_path) is None

    def test_a_missing_record_or_pointer_resolves_none(self, tmp_path: Path) -> None:
        assert sr.resolve_session_slug_for_claude_session_id(_SID, root=tmp_path) is None
        sr.write_claude_session_pointer(_SID, 'gone-record', root=tmp_path)
        assert sr.resolve_session_slug_for_claude_session_id(_SID, root=tmp_path) is None

    @pytest.mark.parametrize('session_id', ['', '../escape', 'a/b', 'x' * 129])
    def test_an_unsafe_session_id_is_never_written_or_read(
        self, tmp_path: Path, session_id: str
    ) -> None:
        assert not sr.write_claude_session_pointer(session_id, 'session-df-1', root=tmp_path)
        assert sr.resolve_session_slug_for_claude_session_id(session_id, root=tmp_path) is None
        assert not sr.claude_session_pointers_dir(tmp_path).exists()

    def test_a_non_record_key_slug_is_not_written(self, tmp_path: Path) -> None:
        assert not sr.write_claude_session_pointer(_SID, '..', root=tmp_path)
        assert not sr.write_claude_session_pointer(_SID, 'a/b', root=tmp_path)

    def test_reap_removes_pointers_no_record_vouches_for(self, tmp_path: Path) -> None:
        _bound_record(tmp_path, 'vouched', _SID)
        _bound_record(tmp_path, 'rebound', 'newer-session')
        sr.write_claude_session_pointer(_SID, 'vouched', root=tmp_path)
        sr.write_claude_session_pointer('old-session', 'rebound', root=tmp_path)
        sr.write_claude_session_pointer('orphan-session', 'gone-record', root=tmp_path)
        pointers = sr.claude_session_pointers_dir(tmp_path)
        (pointers / 'bad name').write_text('vouched')
        fresh_temp = pointers / f'inflight{fleet_files.ATOMIC_WRITE_TEMP_SUFFIX}'
        fresh_temp.write_text('vouched')

        removed = sr.reap_stale_session_pointers(root=tmp_path)

        assert {path.name for path in removed} == {'old-session', 'orphan-session', 'bad name'}
        assert sorted(path.name for path in pointers.iterdir()) == sorted([_SID, fresh_temp.name])


def _t3_record(
    tmp_path: Path,
    slug: str,
    thread_id: str | None,
    *,
    cwd: str,
    status: sr.Status = sr.Status.RUNNING,
    kind: str = sr.DisplayKind.T3.value,
    environment_id: str | None = None,
    pointer: bool = False,
) -> None:
    sr.write_record(
        _make_record(
            session_slug=slug,
            status=status,
            cwd=cwd,
            exit_code=None,
            display=sr.Display(
                kind=kind, t3_thread_id=thread_id, t3_environment_id=environment_id
            ),
        ),
        root=tmp_path,
    )
    if pointer and thread_id is not None:
        assert sr.write_t3_thread_pointer(thread_id, slug, root=tmp_path)


_THREAD = '7d1f0c2a-5b3e-4c8d-9a6f-1e2d3c4b5a69'
_OTHER_THREAD = 'mcp:launch:' + '0f9e8d7c6b5a49388271605f4e3d2c1b' * 2


class TestT3ThreadPointers:
    def test_pointer_dir_is_a_sibling_of_sessions(self, tmp_path: Path) -> None:
        assert sr.t3_thread_pointers_dir(tmp_path) == tmp_path / 'sessions-by-t3-thread'
        assert sr.t3_thread_pointer_path(_THREAD, tmp_path).parent.parent == tmp_path

    def test_a_written_pointer_resolves_while_a_live_t3_record_names_the_thread(
        self, tmp_path: Path
    ) -> None:
        _t3_record(tmp_path, 'session-df-8124-1', _THREAD, cwd=str(tmp_path))

        assert sr.write_t3_thread_pointer(_THREAD, 'session-df-8124-1', root=tmp_path)
        assert sr.t3_thread_pointer_path(_THREAD, tmp_path).read_text() == 'session-df-8124-1'
        assert sr.resolve_session_slug_for_t3_thread(_THREAD, root=tmp_path) == (
            'session-df-8124-1'
        )

    @pytest.mark.parametrize(
        ('thread_id', 'status', 'kind'),
        [
            pytest.param(_THREAD, sr.Status.RUNNING, 'wm', id='not-t3'),
            pytest.param(_OTHER_THREAD, sr.Status.RUNNING, 't3', id='another-thread'),
            pytest.param(_THREAD, sr.Status.EXITED, 't3', id='terminal'),
        ],
    )
    def test_a_record_that_does_not_vouch_resolves_none(
        self, tmp_path: Path, thread_id: str, status: sr.Status, kind: str
    ) -> None:
        _t3_record(tmp_path, 'held', thread_id, cwd=str(tmp_path), status=status, kind=kind)
        sr.write_t3_thread_pointer(_THREAD, 'held', root=tmp_path)

        assert sr.resolve_session_slug_for_t3_thread(_THREAD, root=tmp_path) is None

    def test_a_missing_record_or_pointer_resolves_none(self, tmp_path: Path) -> None:
        assert sr.resolve_session_slug_for_t3_thread(_THREAD, root=tmp_path) is None
        sr.write_t3_thread_pointer(_THREAD, 'gone-record', root=tmp_path)
        assert sr.resolve_session_slug_for_t3_thread(_THREAD, root=tmp_path) is None

    @pytest.mark.parametrize('thread_id', ['', '../escape', 'a/b', 'x' * 129])
    def test_an_unsafe_thread_id_is_never_written_or_read(
        self, tmp_path: Path, thread_id: str
    ) -> None:
        assert not sr.write_t3_thread_pointer(thread_id, 'session-df-1', root=tmp_path)
        assert sr.resolve_session_slug_for_t3_thread(thread_id, root=tmp_path) is None
        assert not sr.t3_thread_pointers_dir(tmp_path).exists()

    def test_a_non_record_key_slug_is_not_written(self, tmp_path: Path) -> None:
        assert not sr.write_t3_thread_pointer(_THREAD, '..', root=tmp_path)
        assert not sr.write_t3_thread_pointer(_THREAD, 'a/b', root=tmp_path)

    def test_reap_removes_pointers_no_live_t3_record_vouches_for(self, tmp_path: Path) -> None:
        ended, gone, moved = (
            '1a2b3c4d-0000-4000-8000-000000000001',
            '1a2b3c4d-0000-4000-8000-000000000002',
            '1a2b3c4d-0000-4000-8000-000000000003',
        )
        _t3_record(tmp_path, 'live', _THREAD, cwd=str(tmp_path))
        _t3_record(tmp_path, 'ended', ended, cwd=str(tmp_path), status=sr.Status.EXITED)
        _t3_record(tmp_path, 'moved', _OTHER_THREAD, cwd=str(tmp_path))
        sr.write_t3_thread_pointer(_THREAD, 'live', root=tmp_path)
        sr.write_t3_thread_pointer(ended, 'ended', root=tmp_path)
        sr.write_t3_thread_pointer(gone, 'gone-record', root=tmp_path)
        sr.write_t3_thread_pointer(moved, 'moved', root=tmp_path)
        pointers = sr.t3_thread_pointers_dir(tmp_path)
        (pointers / 'bad name').write_text('live')
        fresh_temp = pointers / f'inflight{fleet_files.ATOMIC_WRITE_TEMP_SUFFIX}'
        fresh_temp.write_text('live')

        removed = sr.reap_stale_session_pointers(root=tmp_path)

        assert {path.name for path in removed} == {ended, gone, moved, 'bad name'}
        assert sorted(path.name for path in pointers.iterdir()) == sorted(
            [_THREAD, fresh_temp.name]
        )

    def test_the_exit_verb_drops_the_pointer_naming_the_exited_record(
        self, tmp_path: Path
    ) -> None:
        _t3_record(tmp_path, 'held', _THREAD, cwd=str(tmp_path))
        sr.write_t3_thread_pointer(_THREAD, 'held', root=tmp_path)

        assert sr.main(['exit', '--record', str(sr.sessions_dir(tmp_path) / 'held'), '--code', '0']) == 0

        assert sr.read_record('held', root=tmp_path).status == sr.Status.EXITED
        assert not sr.t3_thread_pointer_path(_THREAD, tmp_path).exists()

    def test_the_exit_verb_keeps_a_pointer_another_live_record_holds(
        self, tmp_path: Path
    ) -> None:
        _t3_record(tmp_path, 'holder', _THREAD, cwd=str(tmp_path))
        _t3_record(tmp_path, 'bystander', _THREAD, cwd=str(tmp_path))
        sr.write_t3_thread_pointer(_THREAD, 'holder', root=tmp_path)

        sr.main(['exit', '--record', str(sr.sessions_dir(tmp_path) / 'bystander'), '--code', '0'])

        assert sr.read_record('bystander', root=tmp_path).status == sr.Status.EXITED
        assert sr.resolve_session_slug_for_t3_thread(_THREAD, root=tmp_path) == 'holder'

    def test_a_failed_to_start_refresh_drops_the_pointer_naming_the_record(
        self, tmp_path: Path
    ) -> None:
        _t3_record(tmp_path, 'held', _OTHER_THREAD, cwd=str(tmp_path), pointer=True)
        record_dir = str(sr.sessions_dir(tmp_path) / 'held')

        assert sr.main(['refresh', '--record', record_dir, '--status', 'failed-to-start']) == 0

        assert sr.read_record('held', root=tmp_path).status == sr.Status.FAILED_TO_START
        assert not sr.t3_thread_pointer_path(_OTHER_THREAD, tmp_path).exists()

    def test_folding_a_record_drops_the_pointer_naming_it(self, tmp_path: Path) -> None:
        _t3_record(tmp_path, 'hand', _THREAD, cwd=str(tmp_path))
        sr.write_t3_thread_pointer(_THREAD, 'hand', root=tmp_path)

        assert sr.mark_session_folded('hand', root=tmp_path) is not None

        assert not sr.t3_thread_pointer_path(_THREAD, tmp_path).exists()


_ENVIRONMENT = '3f9a1c2e-5b7d-4e8f-9a0b-1c2d3e4f5a6b'
_GONE_THREAD = '2c4e6a8b-0d1f-4a3c-9e5b-7d9f1b3d5e70'


class _ThreadEnded:
    """A ``thread_ended`` predicate answering *answer* for every thread, recording each thread asked."""

    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.calls: list[str] = []

    def __call__(self, thread_id: str) -> bool:
        self.calls.append(thread_id)
        return self.answer


class TestMarkT3SessionsExited:
    def test_marks_only_indexed_records_of_this_environment_whose_thread_is_gone(
        self, tmp_path: Path
    ) -> None:
        foreign_thread, unindexed_thread = (
            '4e6a8c0d-2f1b-4c5e-8a7d-9b1d3f5a7c92',
            '6a8c0e2f-4b3d-4e7a-8c9f-1d3f5b7d9e14',
        )
        cwd = str(tmp_path)
        _t3_record(tmp_path, 'listed', _THREAD, cwd=cwd, environment_id=_ENVIRONMENT, pointer=True)
        _t3_record(tmp_path, 'gone', _GONE_THREAD, cwd=cwd, environment_id=_ENVIRONMENT, pointer=True)
        _t3_record(
            tmp_path, 'foreign', foreign_thread, cwd=cwd, environment_id='env-elsewhere', pointer=True
        )
        _t3_record(tmp_path, 'unindexed', unindexed_thread, cwd=cwd, environment_id=_ENVIRONMENT)

        marked = sr.mark_t3_sessions_exited(
            _ENVIRONMENT, frozenset({_THREAD}), root=tmp_path, thread_ended=_ThreadEnded()
        )

        assert marked == ['gone']
        gone = sr.read_record('gone', root=tmp_path)
        assert (gone.status, gone.exit_code) == (sr.Status.EXITED, sr.ORPHAN_EXIT_CODE)
        for slug in ('listed', 'foreign', 'unindexed'):
            assert sr.read_record(slug, root=tmp_path).status == sr.Status.RUNNING
        assert sorted(path.name for path in sr.t3_thread_pointers_dir(tmp_path).iterdir()) == (
            sorted([_THREAD, foreign_thread])
        )

    def test_a_record_with_no_environment_is_this_environments(self, tmp_path: Path) -> None:
        _t3_record(tmp_path, 'legacy', _GONE_THREAD, cwd=str(tmp_path), pointer=True)

        assert sr.mark_t3_sessions_exited(
            _ENVIRONMENT, frozenset(), root=tmp_path, thread_ended=_ThreadEnded()
        ) == ['legacy']

    def test_a_worktree_bound_record_is_marked_whatever_its_cwd(self, tmp_path: Path) -> None:
        worktree = tmp_path / 'proj' / '.worktrees' / '8124'
        _t3_record(
            tmp_path,
            'worktree-bound',
            _GONE_THREAD,
            cwd=str(worktree),
            environment_id=_ENVIRONMENT,
            pointer=True,
        )

        assert sr.mark_t3_sessions_exited(
            _ENVIRONMENT, frozenset(), root=tmp_path, thread_ended=_ThreadEnded()
        ) == ['worktree-bound']

    def test_a_record_written_after_the_listing_is_skipped_unasked(self, tmp_path: Path) -> None:
        _t3_record(tmp_path, 'just-launched', _GONE_THREAD, cwd=str(tmp_path), pointer=True)
        listed_at = datetime.now(UTC) - timedelta(minutes=1)
        thread_ended = _ThreadEnded()

        assert (
            sr.mark_t3_sessions_exited(
                _ENVIRONMENT,
                frozenset(),
                root=tmp_path,
                listed_at=listed_at,
                thread_ended=thread_ended,
            )
            == []
        )
        assert thread_ended.calls == []

    def test_an_unlisted_thread_not_confirmed_ended_keeps_its_record(self, tmp_path: Path) -> None:
        _t3_record(
            tmp_path, 'side-chat', _GONE_THREAD, cwd=str(tmp_path), environment_id=_ENVIRONMENT, pointer=True
        )

        marked = sr.mark_t3_sessions_exited(
            _ENVIRONMENT, frozenset(), root=tmp_path, thread_ended=_ThreadEnded(answer=False)
        )

        assert marked == []
        assert sr.read_record('side-chat', root=tmp_path).status == sr.Status.RUNNING
        assert sr.resolve_session_slug_for_t3_thread(_GONE_THREAD, root=tmp_path) == 'side-chat'

    def test_only_this_environments_unlisted_indexed_threads_are_confirmed(
        self, tmp_path: Path
    ) -> None:
        foreign_thread, unindexed_thread = (
            '4e6a8c0d-2f1b-4c5e-8a7d-9b1d3f5a7c92',
            '6a8c0e2f-4b3d-4e7a-8c9f-1d3f5b7d9e14',
        )
        cwd = str(tmp_path)
        _t3_record(tmp_path, 'listed', _THREAD, cwd=cwd, environment_id=_ENVIRONMENT, pointer=True)
        _t3_record(tmp_path, 'gone', _GONE_THREAD, cwd=cwd, environment_id=_ENVIRONMENT, pointer=True)
        _t3_record(
            tmp_path, 'foreign', foreign_thread, cwd=cwd, environment_id='env-elsewhere', pointer=True
        )
        _t3_record(tmp_path, 'unindexed', unindexed_thread, cwd=cwd, environment_id=_ENVIRONMENT)
        thread_ended = _ThreadEnded()

        sr.mark_t3_sessions_exited(
            _ENVIRONMENT, frozenset({_THREAD}), root=tmp_path, thread_ended=thread_ended
        )

        assert thread_ended.calls == [_GONE_THREAD]

    def test_one_info_line_reports_the_index_entries_scanned(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        for slug, thread in (
            ('listed', _THREAD),
            ('also-listed', _OTHER_THREAD),
            ('gone', _GONE_THREAD),
        ):
            _t3_record(tmp_path, slug, thread, cwd=str(tmp_path), pointer=True)

        with caplog.at_level(logging.INFO, logger='orchestrator.session_registry'):
            sr.mark_t3_sessions_exited(
                _ENVIRONMENT,
                frozenset({_THREAD, _OTHER_THREAD}),
                root=tmp_path,
                thread_ended=_ThreadEnded(),
            )

        [line] = [
            record.getMessage().replace(str(tmp_path), '<root>')
            for record in caplog.records
            if record.name == 'orchestrator.session_registry'
            and record.levelno == logging.INFO
            and 'mark_t3_sessions_exited' in record.getMessage()
        ]
        assert re.search(r'\b3\b', line), line

    @pytest.mark.parametrize(
        ('routine', 'ended', 'level'),
        [
            pytest.param(False, False, logging.INFO, id='launch-sweep-marking-nothing'),
            pytest.param(True, True, logging.INFO, id='routine-sweep-marking-one'),
            pytest.param(True, False, logging.DEBUG, id='routine-sweep-marking-nothing'),
        ],
    )
    def test_only_a_routine_sweep_that_marks_nothing_logs_its_line_at_debug(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        routine: bool,
        ended: bool,
        level: int,
    ) -> None:
        _t3_record(tmp_path, 'gone', _GONE_THREAD, cwd=str(tmp_path), pointer=True)

        with caplog.at_level(logging.DEBUG, logger='orchestrator.session_registry'):
            sr.mark_t3_sessions_exited(
                _ENVIRONMENT,
                frozenset(),
                root=tmp_path,
                thread_ended=_ThreadEnded(answer=ended),
                routine=routine,
            )

        assert [
            record.levelno
            for record in caplog.records
            if record.name == 'orchestrator.session_registry'
            and record.getMessage().startswith('mark_t3_sessions_exited: marked')
        ] == [level]


class TestT3RecordsAreExemptFromPidSweeps:
    def _stale_t3_record(self, tmp_path: Path, status: sr.Status) -> Path:
        _t3_record(tmp_path, 't3-quiet', 'th-1', cwd=str(tmp_path), status=status)
        record = sr.read_record('t3-quiet', root=tmp_path)
        record.launcher_pid = _DEAD_PID
        sr.write_record(record, root=tmp_path)
        path = sr.record_path_for_slug('t3-quiet', root=tmp_path)
        _set_mtime(path, _NOW, sr.TERMINAL_TTL + timedelta(hours=1))
        return path

    def test_mark_orphaned_skips_a_quiet_t3_record_with_a_dead_launcher(
        self, tmp_path: Path
    ) -> None:
        self._stale_t3_record(tmp_path, sr.Status.AWAITING_INPUT)

        assert sr.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW) == []
        assert sr.read_record('t3-quiet', root=tmp_path).status == sr.Status.AWAITING_INPUT

    def test_reap_keeps_a_non_terminal_t3_record_with_a_dead_launcher(self, tmp_path: Path) -> None:
        path = self._stale_t3_record(tmp_path, sr.Status.RUNNING)

        assert sr.reap_stale_records(root=tmp_path, now=_NOW) == []
        assert path.is_file()
        assert not sr.t3_thread_pointers_dir(tmp_path).exists()

    def test_the_reap_indexes_a_t3_record_whose_thread_no_pointer_holds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stale_t3_record(tmp_path, sr.Status.RUNNING)
        monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
        monkeypatch.setenv('PATH', str(tmp_path / 'no-bin'))

        assert sr.main(['reap']) == 0

        assert sr.resolve_session_slug_for_t3_thread('th-1', root=tmp_path) == 't3-quiet'
        assert sr.read_record('t3-quiet', root=tmp_path).status == sr.Status.RUNNING

    def test_indexing_never_moves_a_held_pointer_or_indexes_a_terminal_record(
        self, tmp_path: Path
    ) -> None:
        self._stale_t3_record(tmp_path, sr.Status.RUNNING)
        _t3_record(tmp_path, 'holder', 'th-1', cwd=str(tmp_path), pointer=True)
        _t3_record(tmp_path, 'ended', 'th-2', cwd=str(tmp_path), status=sr.Status.EXITED)

        assert sr.index_unpointed_t3_records(root=tmp_path) == []

        assert sr.resolve_session_slug_for_t3_thread('th-1', root=tmp_path) == 'holder'
        assert not sr.t3_thread_pointer_path('th-2', tmp_path).exists()

    def test_a_t3_record_displaced_from_its_thread_pointer_is_pid_swept(
        self, tmp_path: Path
    ) -> None:
        self._stale_t3_record(tmp_path, sr.Status.RUNNING)
        _t3_record(tmp_path, 'holder', 'th-1', cwd=str(tmp_path), pointer=True)

        reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

        assert [(r.session_slug, r.reason) for r in reaped] == [('t3-quiet', 'stale_pid')]
        assert sr.resolve_session_slug_for_t3_thread('th-1', root=tmp_path) == 'holder'

    def test_mark_orphaned_marks_a_displaced_t3_record_with_a_dead_launcher(
        self, tmp_path: Path
    ) -> None:
        self._stale_t3_record(tmp_path, sr.Status.AWAITING_INPUT)
        _t3_record(tmp_path, 'holder', 'th-1', cwd=str(tmp_path), pointer=True)

        marked = sr.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW)

        assert [r.session_slug for r in marked] == ['t3-quiet']

    def test_a_failed_index_write_leaves_the_t3_record_exempt(self, tmp_path: Path) -> None:
        path = self._stale_t3_record(tmp_path, sr.Status.RUNNING)
        sr.t3_thread_pointers_dir(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        sr.t3_thread_pointers_dir(tmp_path).write_text('not a dir')

        assert sr.index_unpointed_t3_records(root=tmp_path) == []
        assert sr.reap_stale_records(root=tmp_path, now=_NOW) == []
        assert sr.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW) == []
        assert path.is_file()
        assert sr.read_record('t3-quiet', root=tmp_path).status == sr.Status.RUNNING

    def test_reap_still_applies_terminal_ttl_to_a_t3_record(self, tmp_path: Path) -> None:
        self._stale_t3_record(tmp_path, sr.Status.EXITED)

        reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

        assert [(r.session_slug, r.reason) for r in reaped] == [('t3-quiet', 'terminal_ttl')]


def test_mark_session_folded_stamps_the_fold_code_once(tmp_path: Path) -> None:
    sr.write_record(_make_record(session_slug='hand', status=sr.Status.RUNNING), root=tmp_path)

    folded = sr.mark_session_folded('hand', root=tmp_path)

    assert folded is not None
    assert (folded.status, folded.exit_code) == (sr.Status.EXITED, sr.FOLDED_EXIT_CODE)
    assert sr.mark_session_folded('hand', root=tmp_path) is None


def test_is_t3_record_reads_the_display_kind() -> None:
    t3 = _make_record(display=sr.Display(kind=sr.DisplayKind.T3.value, t3_thread_id='th-1'))

    assert sr.is_t3_record(t3)
    assert not sr.is_t3_record(_make_record())
    assert not sr.is_t3_record(_make_record(display=None))


def test_pid_alive_is_the_public_liveness_probe() -> None:
    assert sr.pid_alive(os.getpid())
    assert not sr.pid_alive(_DEAD_PID)
    assert not sr.pid_alive(0)
    assert not sr.pid_alive(True)
    assert not sr.pid_alive(None)
    assert sr.pid_alive is proc_liveness.pid_alive
