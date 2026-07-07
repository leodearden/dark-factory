"""Tests for orchestrator.session_registry module.

Covers: schema/contract round-trip (SCHEMA_VERSION, Status enum, SessionRecord
to/from dict/json); slug sanitization + path/transcript encoding; single-writer
atomic write/read/update; TTL/pid stale-record reaper matrix; CLI subcommands
+ fail-soft; and the G5 two-way boundary test (write by a real spawn -> refresh
by a simulated hook -> reap), exercised jointly with
tests/scripts/test_spawn_claude.py's bash-level harness.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

from orchestrator import session_registry as sr

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_record(**overrides: object) -> sr.SessionRecord:
    """Build a fully-populated SessionRecord for round-trip/identity tests.

    Every field is given a concrete, distinguishable value so a round-trip
    test can catch a field being dropped/mis-typed; ``overrides`` lets a
    test tweak just the field(s) it cares about.
    """
    fields: dict[str, object] = {
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
    }
    fields.update(overrides)
    return sr.SessionRecord(**fields)


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


def test_session_record_carries_all_required_fields() -> None:
    r = _make_record()
    for field_name in (
        'schema_version',
        'session_slug',
        'title',
        'role',
        'project',
        'task_id',
        'escalation_id',
        'prompt',
        'cwd',
        'launcher_pid',
        'start_ts',
        'status',
        'exit_code',
        'result_file',
        'transcript_path',
    ):
        assert hasattr(r, field_name), f'SessionRecord missing field {field_name!r}'


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
        task_id=None, escalation_id=None, exit_code=None,
        result_file=None, transcript_path=None,
    )
    assert sr.SessionRecord.from_json(r.to_json()) == r


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


def test_fleet_root_defaults_to_dot_claude_fleet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('CLAUDE_FLEET_ROOT', raising=False)
    monkeypatch.setenv('HOME', '/home/fakeuser')
    assert sr.fleet_root() == Path('/home/fakeuser/.claude/fleet')


def test_fleet_root_honors_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    assert sr.fleet_root() == tmp_path


def test_fleet_root_explicit_root_overrides_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
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
