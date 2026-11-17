"""Tests for orchestrator.session_records: the session-record schema and its store under the fleet root.

Covers the schema/contract round-trips (SCHEMA_VERSION, the enums, SessionRecord
to/from dict/json and its field coercions), slug sanitization and the
path/transcript encoding, and the single-writer atomic write/read/update/refresh.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from _session_record_fixtures import make_record
from _utf8_codec import names_the_utf8_codec
from shared.process_identity import ProcessIdentity

from orchestrator import session_records

# ---------------------------------------------------------------------------
# Step-1: schema / contract
# ---------------------------------------------------------------------------


def test_schema_version_is_int() -> None:
    assert isinstance(session_records.SCHEMA_VERSION, int)


def test_status_enum_has_exact_six_wire_values() -> None:
    assert issubclass(session_records.Status, str)
    values = {member.value for member in session_records.Status}
    assert values == {
        'launching',
        'running',
        'awaiting-input',
        'idle',
        'exited',
        'failed-to-start',
    }
    assert len(list(session_records.Status)) == 6
    # Spot-check the documented wire-value <-> member mapping.
    assert session_records.Status('launching') is session_records.Status.LAUNCHING
    assert session_records.Status('running') is session_records.Status.RUNNING
    assert session_records.Status('awaiting-input') is session_records.Status.AWAITING_INPUT
    assert session_records.Status('idle') is session_records.Status.IDLE
    assert session_records.Status('exited') is session_records.Status.EXITED
    assert session_records.Status('failed-to-start') is session_records.Status.FAILED_TO_START


def test_session_record_schema_version_defaults_to_schema_version() -> None:
    r = make_record()
    assert r.schema_version == session_records.SCHEMA_VERSION


def test_session_record_dict_round_trip_is_lossless() -> None:
    r = make_record()
    assert session_records.SessionRecord.from_dict(r.to_dict()) == r


def test_session_record_json_round_trip_is_lossless() -> None:
    r = make_record()
    assert session_records.SessionRecord.from_json(r.to_json()) == r


def test_session_record_json_round_trip_with_null_fields() -> None:
    """exit_code/result_file/task_id/escalation_id may legitimately be None
    (a freshly-launched record has no exit code yet); the round trip must
    preserve that, not coerce it to a string or drop the key.
    """
    r = make_record(
        task_id=None,
        escalation_id=None,
        exit_code=None,
        result_file=None,
        transcript_path=None,
    )
    assert session_records.SessionRecord.from_json(r.to_json()) == r


# ---------------------------------------------------------------------------
# Step-1: SessionRecord C1 schema extensions (B1) -- Fleet Cockpit
# parent_session_id/spawn_mode/display/question, SCHEMA_MINOR, SpawnMode,
# DisplayKind, Display, Question. Additive + migration-free: a rail-vintage
# dict (no C1 keys) must still parse, defaulting the new fields.
# ---------------------------------------------------------------------------


def test_session_record_round_trip_includes_new_c1_fields() -> None:
    r = make_record()
    assert session_records.SessionRecord.from_dict(r.to_dict()) == r
    assert session_records.SessionRecord.from_json(r.to_json()) == r

    # A display whose optional wm_window_id/tmux_target are both None must
    # also round-trip losslessly (None-safe nested (dis)assembly).
    r_none_optionals = make_record(
        display=session_records.Display(
            kind='wm', wm_title='unblock:df#2085 slug', wm_window_id=None, tmux_target=None
        ),
    )
    assert session_records.SessionRecord.from_dict(r_none_optionals.to_dict()) == r_none_optionals
    assert session_records.SessionRecord.from_json(r_none_optionals.to_json()) == r_none_optionals

    # A record with no parent/display/question at all (the common
    # human-launched-root shape) must also round-trip losslessly.
    r_all_none = make_record(parent_session_id=None, display=None, question=None)
    assert session_records.SessionRecord.from_dict(r_all_none.to_dict()) == r_all_none
    assert session_records.SessionRecord.from_json(r_all_none.to_json()) == r_all_none


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
    record = session_records.SessionRecord.from_dict(rail_vintage)
    assert record.parent_session_id is None
    assert record.spawn_mode == session_records.SpawnMode.CHILD
    assert record.display is None
    assert record.question is None
    # Task 4193's additive hook-owner binding: absent on disk -> None, no
    # migration required.
    assert record.claude_session_id is None
    assert record.launcher_identity is None
    assert record.claude_owner_identity is None


def test_schema_minor_is_int_bumped() -> None:
    assert isinstance(session_records.SCHEMA_MINOR, int)
    assert session_records.SCHEMA_MINOR >= 1
    # The PERSISTED major must stay migration-free: bumping it would make
    # rail-vintage and C1 records version-distinguishable on disk.
    assert session_records.SCHEMA_VERSION == 1


# ---------------------------------------------------------------------------
# Task 4193: SessionRecord.claude_session_id (hook-owner binding)
# ---------------------------------------------------------------------------


def test_session_record_round_trip_includes_claude_session_id() -> None:
    r = make_record()
    assert session_records.SessionRecord.from_dict(r.to_dict()) == r
    assert session_records.SessionRecord.from_json(r.to_json()) == r
    # The wire key is pinned by name: session_hooks reads it back off disk.
    assert r.to_dict()['claude_session_id'] == 'uuid-claude-abc123'

    # The unbound shape a spawn-claude.sh `launching` write produces (no hook
    # has claimed the slug yet) must round-trip just as losslessly.
    r_unbound = make_record(claude_session_id=None)
    assert session_records.SessionRecord.from_dict(r_unbound.to_dict()) == r_unbound
    assert session_records.SessionRecord.from_json(r_unbound.to_json()) == r_unbound
    assert r_unbound.to_dict()['claude_session_id'] is None


def test_session_record_defaults_claude_session_id_to_none() -> None:
    record = session_records.SessionRecord(session_slug='s', status=session_records.Status.LAUNCHING)
    assert record.claude_session_id is None


def test_session_record_round_trip_includes_claude_owner_pid() -> None:
    # esc-4193-11 suggestion 3: to_dict/from_dict must actually carry
    # claude_owner_pid -- without a non-None value in _make_record's fields,
    # the round-trip pin above is vacuous (both sides default to None even
    # if to_dict silently dropped the key).
    r = make_record()
    assert r.claude_owner_pid == 424242
    assert r.to_dict()['claude_owner_pid'] == 424242
    assert session_records.SessionRecord.from_dict(r.to_dict()) == r
    assert session_records.SessionRecord.from_json(r.to_json()) == r


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
    assert session_records._coerce_owner_pid(raw) == expected


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
    assert session_records._coerce_session_id(raw) == expected


def test_session_record_from_dict_coerces_non_str_claude_session_id() -> None:
    # esc-4193-11 suggestion 2: a hand-edited or future-writer record body
    # carrying a non-string claude_session_id must not survive into the
    # SessionRecord -- session_hooks does `(record.claude_session_id or
    # '').strip()` outside a try/except, so a raw int here would raise
    # AttributeError and lose the whole hook event.
    data = make_record().to_dict()
    data['claude_session_id'] = 123
    record = session_records.SessionRecord.from_dict(data)
    assert record.claude_session_id is None


def test_session_record_from_dict_coerces_non_int_claude_owner_pid() -> None:
    # Sibling to the claude_session_id regression test above (task 4660
    # review follow-up): pins the from_dict call site for claude_owner_pid
    # too, so replacing `_coerce_owner_pid(data.get(...))` with a bare
    # `data.get(...)` cannot go unnoticed. True is an int subclass and would
    # otherwise survive as a real-looking pid, which could silently compare
    # equal to a real pid in session_hooks._env_slug_ownership
    # (session_hooks.py:329).
    data = make_record().to_dict()
    data['claude_owner_pid'] = True
    record = session_records.SessionRecord.from_dict(data)
    assert record.claude_owner_pid is None


# ---------------------------------------------------------------------------
# Task 8125: SessionRecord.launcher_identity and claude_owner_identity
# ---------------------------------------------------------------------------

_IDENTITY_FIELDS = ('launcher_identity', 'claude_owner_identity')
_RECORD_PID = 1


def _identity_wire(**overrides: object) -> dict[str, object]:
    return {'pid': _RECORD_PID, 'start_ticks': 11, 'boot_id': 'boot-a', 'host': 'host-a', **overrides}


def _record_dict_with_pids(pid: int) -> dict[str, Any]:
    return make_record(launcher_pid=pid, claude_owner_pid=pid).to_dict()


def test_session_record_round_trip_includes_both_identities() -> None:
    r = make_record(
        launcher_pid=4242,
        launcher_identity=ProcessIdentity(4242, 11, 'boot-a', 'host-a'),
        claude_owner_pid=424242,
        claude_owner_identity=ProcessIdentity(424242, 12, 'boot-a', 'host-a'),
    )

    assert session_records.SessionRecord.from_dict(r.to_dict()) == r
    assert session_records.SessionRecord.from_json(r.to_json()) == r
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
    record = session_records.SessionRecord(session_slug='s', status=session_records.Status.LAUNCHING)

    assert record.launcher_identity is None
    assert record.claude_owner_identity is None


@pytest.mark.parametrize('identity_field', _IDENTITY_FIELDS)
def test_a_well_formed_identity_naming_its_pid_reads_back(identity_field: str) -> None:
    data = _record_dict_with_pids(_RECORD_PID)
    data[identity_field] = _identity_wire()

    record = session_records.SessionRecord.from_dict(data)

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

    assert getattr(session_records.SessionRecord.from_dict(data), identity_field) is None


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

    assert getattr(session_records.SessionRecord.from_dict(data), identity_field) is None


def test_spawn_mode_enum_values() -> None:
    assert issubclass(session_records.SpawnMode, str)
    assert {m.value for m in session_records.SpawnMode} == {'child', 'sibling', 'detached'}
    assert session_records.SpawnMode.CHILD.value == 'child'
    assert session_records.SpawnMode.SIBLING.value == 'sibling'
    assert session_records.SpawnMode.DETACHED.value == 'detached'


def test_display_kind_enum_values() -> None:
    assert issubclass(session_records.DisplayKind, str)
    assert {m.value for m in session_records.DisplayKind} == {'wm', 'tmux', 't3'}
    assert session_records.DisplayKind.WM.value == 'wm'
    assert session_records.DisplayKind.TMUX.value == 'tmux'
    assert session_records.DisplayKind.T3.value == 't3'


def test_display_round_trip() -> None:
    with_optionals = session_records.Display(
        kind='wm', wm_title='unblock:df#2085 slug', wm_window_id='0x1a', tmux_target=None
    )
    assert session_records.Display.from_dict(with_optionals.to_dict()) == with_optionals

    without_optionals = session_records.Display(
        kind='tmux', wm_title='', wm_window_id=None, tmux_target='session:0.1'
    )
    assert session_records.Display.from_dict(without_optionals.to_dict()) == without_optionals

    t3 = session_records.Display(kind=session_records.DisplayKind.T3.value, wm_title='unblock:df#7489', t3_thread_id='th-1')
    assert session_records.Display.from_dict(t3.to_dict()) == t3


def test_display_from_a_body_without_t3_thread_id_reads_none() -> None:
    assert session_records.Display.from_dict({'kind': 'wm'}).t3_thread_id is None


def test_question_round_trip() -> None:
    q = session_records.Question(text='approve rollout?', asked_at='2026-07-07T00:00:00+00:00')
    assert session_records.Question.from_dict(q.to_dict()) == q


# ---------------------------------------------------------------------------
# Step-3: identity, paths, transcript encoding
# ---------------------------------------------------------------------------


def test_build_session_slug_joins_role_project_task_pid() -> None:
    assert session_records.build_session_slug('unblock', 'df', '2085', 4242) == 'unblock-df-2085-4242'


@pytest.mark.parametrize('task_id', [None, ''])
def test_build_session_slug_omits_task_segment_when_absent(task_id: str | None) -> None:
    assert session_records.build_session_slug('unblock', 'df', task_id, 4242) == 'unblock-df-4242'


def test_build_session_slug_sanitizes_special_chars() -> None:
    slug = session_records.build_session_slug('un block', 'df/prod', '20#85', 4242)
    assert slug == 'un-block-df-prod-20-85-4242'
    assert re.fullmatch(r'[A-Za-z0-9._-]+', slug)


# ---------------------------------------------------------------------------
# Task 2511 step-1: sanitize_slug (shared filesystem-safe-slug normalizer)
# ---------------------------------------------------------------------------


def test_sanitize_slug_leaves_already_safe_slug_unchanged() -> None:
    assert session_records.sanitize_slug('session-cockpit-3215033') == 'session-cockpit-3215033'


def test_sanitize_slug_maps_unsafe_chars_to_dash() -> None:
    # '/' and '#' fall outside [A-Za-z0-9._-] (record_path_for_slug's
    # path-escape guard) and are mapped to '-'; '.' is itself in the allowed
    # set (mirrors _SLUG_SANITIZE_RE, the idiom this delegates to), so a
    # path separator can never escape sessions_dir via record_path_for_slug.
    slug = session_records.sanitize_slug('weird/id#x.y')
    assert slug == 'weird-id-x.y'
    assert re.fullmatch(r'[A-Za-z0-9._-]+', slug)


def test_sanitize_slug_matches_build_session_slug_regression() -> None:
    # Pins build_session_slug's output as byte-identical to today once it
    # delegates to sanitize_slug (step-2's behavior-preserving refactor).
    assert session_records.build_session_slug('unblock', 'df', '2085', 4242) == 'unblock-df-2085-4242'
    slug = session_records.build_session_slug('un block', 'df/prod', '20#85', 4242)
    assert slug == 'un-block-df-prod-20-85-4242'
    assert slug == session_records.sanitize_slug('un block-df/prod-20#85-4242')


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
    slug = session_records.sanitize_slug(raw)
    assert slug == expected
    assert re.fullmatch(r'[A-Za-z0-9._-]+', slug)


@pytest.mark.parametrize('raw', ['..foo', 'foo..', 'a.b', '.hidden-x'])
def test_sanitize_slug_leaves_partially_dotted_slug_untouched(raw: str) -> None:
    # The guard's LOWER edge: a value that merely CONTAINS dots alongside other
    # characters is a literal, non-traversing directory name (sanitize_slug's
    # docstring says so) and must pass through byte-identical. Pinning only the
    # collapse direction would leave a future "hardening" free to over-broaden
    # into stripping every dot, silently mangling ordinary slugs like 'a.b'.
    assert session_records.sanitize_slug(raw) == raw


def test_record_path_for_slug(tmp_path: Path) -> None:
    path = session_records.record_path_for_slug('unblock-df-2085-4242', root=tmp_path)
    assert path == tmp_path / 'sessions' / 'unblock-df-2085-4242' / 'record.json'


def test_record_path_for_slug_all_dots_slug_stays_under_sessions_dir(tmp_path: Path) -> None:
    # Task 4112: the containment property sanitize_slug's all-dots branch buys.
    # resolve() + is_relative_to, NOT a string prefix check -- str(root/'sessions'
    # /'..'/'record.json') startswith str(root/'sessions') is TRUE, so a prefix
    # check would pass for the very path that escapes. resolve() is non-strict
    # (these paths do not exist) and collapses '..' with the exact filesystem
    # semantics under defense.
    sessions = session_records.sessions_dir(root=tmp_path).resolve()
    # '..' is the ONLY genuine escape vector in this loop; '.' and '...' are
    # non-traversing CONTROLS, not extra attack coverage. Unsanitized,
    # 'sessions/./record.json' still resolves under sessions_dir and '...' is a
    # literal POSIX dirname resolve() leaves in place -- so only the '..'
    # iteration discriminates against a guard-deleted build.
    for raw in ('.', '..', '...'):
        slug = session_records.sanitize_slug(raw)
        path = session_records.record_path_for_slug(slug, root=tmp_path)
        assert path.resolve().is_relative_to(sessions)

    # NON-VACUITY COUNTERFACTUAL: the UNsanitized token genuinely escapes, i.e.
    # record_path_for_slug performs NO sanitization of its own (it is a bare
    # join) -- which is exactly why sanitize_slug has to be the chokepoint.
    # Without this, the assertions above could pass for the wrong reason if
    # record_path_for_slug's shape ever changed, and nobody would notice the
    # test had stopped proving anything.
    assert not session_records.record_path_for_slug('..', root=tmp_path).resolve().is_relative_to(sessions)


def test_transcript_path_for_cwd_encodes_slash() -> None:
    assert (
        session_records.transcript_path_for_cwd('/home/leo/src/dark-factory')
        == '~/.claude/projects/-home-leo-src-dark-factory'
    )


def test_transcript_path_for_cwd_encodes_dot() -> None:
    # Regression fixture confirmed against a real ~/.claude/projects/ dir
    # (plans/session-attention-rail-prd.md §3): '.' -> '-' too, so a cwd
    # component starting with '.' yields a doubled '--' (one dash from the
    # preceding '/', one from the '.').
    assert (
        session_records.transcript_path_for_cwd('/home/leo/.openclaw-workspace')
        == '~/.claude/projects/-home-leo--openclaw-workspace'
    )


def test_transcript_path_for_cwd_encodes_underscore() -> None:
    # The third character (task 3272). Derived from 738 real (encoded-dir,
    # decoded-cwd) pairs sampled off a live ~/.claude/projects tree, where
    # two thirds of the dirs carried an underscore — the former two-character
    # rule mismatched 492 of the 738.
    assert (
        session_records.transcript_path_for_cwd('/media/leo/data_lv_1/leo/reify-build')
        == '~/.claude/projects/-media-leo-data-lv-1-leo-reify-build'
    )


def test_encode_cwd_returns_bare_dir_name() -> None:
    # encode_cwd is the bare encoded dir name — no '~/.claude/projects/'
    # prefix — so callers that need a lookup key (not a display path) have
    # one canonical source instead of string-slicing the prefix back off.
    assert session_records.encode_cwd('/home/leo/src/dark-factory') == '-home-leo-src-dark-factory'


def test_encode_cwd_preserves_case() -> None:
    # A real on-disk dir name with its capitals intact — the encoder does NOT
    # lowercase, which rules out a case-folding step in the rule.
    assert session_records.encode_cwd('/opt/Auto-Claude/resources/backend') == (
        '-opt-Auto-Claude-resources-backend'
    )


def test_encode_cwd_maps_all_three_characters() -> None:
    # '/' , '.' and '_' all collapse to '-'; a leading '_' on a path component
    # yields a doubled '--' just as a leading '.' does.
    assert session_records.encode_cwd('/home/leo/src/warm-lanes/worktrees/_lane-39') == (
        '-home-leo-src-warm-lanes-worktrees--lane-39'
    )
    assert session_records.encode_cwd('/home/leo/src/dark-factory/.eval-worktrees/df_task_12') == (
        '-home-leo-src-dark-factory--eval-worktrees-df-task-12'
    )


def test_transcript_path_for_cwd_is_prefix_plus_encoding() -> None:
    # Pins the '~/.claude/projects/' prefix AND the encoding together over a
    # cwd exercising all three characters, as a hard-coded literal.
    #
    # Asserting `== f'~/.claude/projects/{session_records.encode_cwd(cwd)}'` instead would
    # be a character-for-character restatement of the one-line implementation
    # — it could not fail while the implementation keeps that shape, so it
    # would pin nothing. That is the same self-consistency trap task 3272
    # calls out for fixtures built with the encoder under test.
    assert (
        session_records.transcript_path_for_cwd('/home/leo/src/dark-factory/.eval-worktrees/df_task_12')
        == '~/.claude/projects/-home-leo-src-dark-factory--eval-worktrees-df-task-12'
    )


# ---------------------------------------------------------------------------
# Step-5: single-writer atomic write / read / update
# ---------------------------------------------------------------------------


def test_write_record_creates_record_json_matching_the_record(tmp_path: Path) -> None:
    r = make_record()
    session_records.write_record(r, root=tmp_path)
    path = session_records.record_path_for_slug(r.session_slug, root=tmp_path)
    assert path.is_file()
    assert session_records.SessionRecord.from_json(path.read_text()) == r


def test_write_record_leaves_no_leftover_tmp_file(tmp_path: Path) -> None:
    r = make_record()
    session_records.write_record(r, root=tmp_path)
    slug_dir = session_records.record_path_for_slug(r.session_slug, root=tmp_path).parent
    leftovers = [p for p in slug_dir.iterdir() if p.suffix == '.tmp']
    assert leftovers == []


def test_read_record_returns_equal_record(tmp_path: Path) -> None:
    r = make_record()
    session_records.write_record(r, root=tmp_path)
    assert session_records.read_record(r.session_slug, root=tmp_path) == r


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
    r = make_record()
    session_records.write_record(r, root=tmp_path)

    seen: list[dict] = []
    real_read_text = Path.read_text

    def _spy(self, *args, **kwargs):
        seen.append(dict(kwargs))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', _spy)

    # The spy delegates, so the read really completed — this pin cannot pass
    # by disabling the thing it is measuring.
    assert session_records.read_record(r.session_slug, root=tmp_path) == r

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
    r = make_record(status=session_records.Status.LAUNCHING, exit_code=None)
    session_records.write_record(r, root=tmp_path)

    session_records.update_status(r.session_slug, root=tmp_path, status=session_records.Status.EXITED, exit_code=3)

    reread = session_records.read_record(r.session_slug, root=tmp_path)
    assert reread.status == session_records.Status.EXITED
    assert reread.exit_code == 3
    # Every other field survives the read-modify-write untouched.
    assert reread.session_slug == r.session_slug
    assert reread.title == r.title


def test_refresh_record_updates_existing_record_under_same_key(tmp_path: Path) -> None:
    r = make_record(status=session_records.Status.LAUNCHING)
    session_records.write_record(r, root=tmp_path)

    session_records.refresh_record(r.session_slug, root=tmp_path, status=session_records.Status.RUNNING)

    path_before = session_records.record_path_for_slug(r.session_slug, root=tmp_path)
    reread = session_records.read_record(r.session_slug, root=tmp_path)
    assert reread.status == session_records.Status.RUNNING
    assert session_records.record_path_for_slug(r.session_slug, root=tmp_path) == path_before


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

    monkeypatch.setattr(session_records, 'read_record', _boom)
    monkeypatch.setattr(session_records, 'write_record', _boom)


def test_apply_refresh_updates_prior_in_place_and_does_no_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = make_record(status=session_records.Status.LAUNCHING)
    before = prior.to_dict()
    _no_io(monkeypatch)

    result = session_records.apply_refresh('unblock-df-2085-4242', prior, status=session_records.Status.IDLE)

    # In-place read-modify, exactly like refresh_record/update_status today.
    assert result is prior
    assert result.status is session_records.Status.IDLE
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
    prior = make_record(status=session_records.Status.LAUNCHING)
    _no_io(monkeypatch)

    result = session_records.apply_refresh('unblock-df-2085-4242', prior, status=None)

    assert result is prior
    assert result.status is session_records.Status.LAUNCHING


def test_apply_refresh_synthesizes_a_well_formed_record_when_prior_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_io(monkeypatch)

    result = session_records.apply_refresh('hand-launched-slug', None, status=session_records.Status.RUNNING)

    assert result.session_slug == 'hand-launched-slug'
    assert result.status is session_records.Status.RUNNING
    assert result.schema_version == session_records.SCHEMA_VERSION
    # A parseable ISO-8601 start_ts, not merely a non-empty string.
    assert result.start_ts
    assert datetime.fromisoformat(result.start_ts)


def test_apply_refresh_synthesis_defaults_to_launching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LAUNCHING is refresh_record's documented default for a new record."""
    _no_io(monkeypatch)

    result = session_records.apply_refresh('hand-launched-slug', None, status=None)

    assert result.status is session_records.Status.LAUNCHING


@pytest.mark.parametrize('status', [session_records.Status.RUNNING, None])
def test_refresh_record_equals_read_apply_write(
    tmp_path: Path, status: session_records.Status | None
) -> None:
    """refresh_record(...) == write_record(apply_refresh(read_record(...))).

    The equivalence is the whole point of the split: session_hooks composes
    the second spelling from a snapshot it already holds, so it must produce
    a byte-identical body to the one refresh_record writes today.
    """
    seed = make_record(status=session_records.Status.LAUNCHING)
    session_records.write_record(seed, root=tmp_path)
    via_refresh = session_records.refresh_record(seed.session_slug, root=tmp_path, status=status)
    refreshed_bytes = session_records.record_path_for_slug(
        seed.session_slug, root=tmp_path
    ).read_text(encoding='utf-8')

    # Re-seed and take the composed route over the same starting body.
    session_records.write_record(make_record(status=session_records.Status.LAUNCHING), root=tmp_path)
    composed = session_records.apply_refresh(
        seed.session_slug,
        session_records.read_record(seed.session_slug, root=tmp_path),
        status=status,
    )
    session_records.write_record(composed, root=tmp_path)
    composed_bytes = session_records.record_path_for_slug(
        seed.session_slug, root=tmp_path
    ).read_text(encoding='utf-8')

    assert composed_bytes == refreshed_bytes
    assert composed.to_dict() == via_refresh.to_dict()


# ---------------------------------------------------------------------------
# Nested-root write and the T3 record accessor
# ---------------------------------------------------------------------------


def test_write_record_creates_missing_nested_dir_and_round_trips(tmp_path):
    """End-to-end: write_record into a non-existent nested root still works."""
    root = tmp_path / 'does' / 'not' / 'exist'
    assert not root.exists()

    record = make_record(session_slug='sess-3223', task_id='3223')
    session_records.write_record(record, root=root)

    loaded = session_records.read_record('sess-3223', root=root)
    assert loaded is not None
    assert loaded.session_slug == 'sess-3223'
    assert loaded.task_id == '3223'


def test_is_t3_record_reads_the_display_kind() -> None:
    t3 = make_record(display=session_records.Display(kind=session_records.DisplayKind.T3.value, t3_thread_id='th-1'))

    assert session_records.is_t3_record(t3)
    assert not session_records.is_t3_record(make_record())
    assert not session_records.is_t3_record(make_record(display=None))
