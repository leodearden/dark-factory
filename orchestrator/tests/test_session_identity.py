"""Tests for orchestrator.session_identity: THIS session's identity, resolved from its env."""

from __future__ import annotations

import logging

import pytest
from shared import proc_liveness

from orchestrator import session_identity


def test_session_pid_env_names_claude_pid() -> None:
    assert session_identity.SESSION_PID_ENV == 'CLAUDE_PID'


def test_resolve_session_pid_prefers_claude_pid(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        pid = session_identity.resolve_session_pid({'CLAUDE_PID': '1348600'})

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
        pid = session_identity.resolve_session_pid(env)

    # Fallback, never a raise -- and never a pid that merely LOOKS resolved.
    # THE assertion: the degraded pid must be one `pid_alive` reports dead,
    # so the lease's staleness AND-guard collapses to its heartbeat half (the
    # documented degradation) rather than to a never-reapable lease. The
    # earlier `os.getsid(0)` fallback failed this: the POSIX session leader is
    # the interactive shell, which outlives a crashed `claude`.
    assert pid == 0
    assert proc_liveness.pid_alive(pid) is False
    # LOUD, not silent: the repo's loud-over-silent-degradation norm (INV-2).
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings
    blob = ' '.join(r.getMessage() for r in warnings).lower()
    assert 'degraded' in blob


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
