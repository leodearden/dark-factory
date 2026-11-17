"""Tests for orchestrator.session_identity: THIS session's identity, resolved from its env."""

from __future__ import annotations

import logging

import pytest

from orchestrator import session_identity


def test_session_pid_env_names_claude_pid() -> None:
    assert session_identity.SESSION_PID_ENV == 'CLAUDE_PID'


def test_resolve_session_pid_prefers_claude_pid(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        pid = session_identity.resolve_session_pid({'CLAUDE_PID': '1348600'})

    assert pid == 1348600
    # The happy path is silent: a resolvable session pid is not a degradation.
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_resolve_session_pid_defaults_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('CLAUDE_PID', '424242')
    assert session_identity.resolve_session_pid() == 424242


@pytest.mark.parametrize(
    ('token', 'accepted'),
    [
        ('0b6a3c1e-1111-4222-8333-444455556666', True),
        ('A-z0' * 32, True),
        ('', False),
        ('.', False),
        ('..', False),
        ('a/b', False),
        ('a.b', False),
        ('a b', False),
        ('a' * 129, False),
    ],
    ids=['uuid', '128-chars', 'empty', 'dot', 'dot-dot', 'slash', 'inner-dot', 'space', '129-chars'],
)
def test_is_session_id_accepts_only_a_bounded_alnum_dash_token(token: str, accepted: bool) -> None:
    assert session_identity.is_session_id(token) is accepted
