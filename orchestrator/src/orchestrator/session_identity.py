"""orchestrator.session_identity — THIS session's identity, resolved from its env.

The claude pid, the session id, whether that pid is a shared ``claude agents``
host, and the pid a lease records for liveness.

It is stdlib-only and imports no orchestrator sibling and no ``shared``. It
never imports ``session_registry``: in a bare-shell (tier-1) run
session_registry is ``__main__``, so an upward import would load a second copy
of it. The bare-shell contract is pinned by
``orchestrator/tests/test_session_registry.py::TestStdlibOnlySelfContainment``.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


SESSION_PID_ENV = 'CLAUDE_PID'
"""Env var naming the long-lived ``claude`` process's pid (see resolve_session_pid)."""

SESSION_ID_ENV = 'CLAUDE_CODE_SESSION_ID'
"""Env var naming THIS Claude Code session's id; the CLI sets it beside ``CLAUDE_PID``."""

_SESSION_ID_RE = re.compile(r'[A-Za-z0-9-]{1,128}')

_SHARED_SESSION_HOST_SUBCOMMAND = b'agents'
"""``argv[1]`` of a ``claude agents`` process, which runs the sessions it hosts in-process."""


def is_session_id(token: str) -> bool:
    """Whether *token* has the shape of a ``CLAUDE_CODE_SESSION_ID``: 1-128 of ``[A-Za-z0-9-]``."""
    return _SESSION_ID_RE.fullmatch(token) is not None


def _parse_session_pid(env: Mapping[str, str]) -> int | None:
    """Parse *env*'s ``CLAUDE_PID`` as a positive int, else None. Never raises, never logs."""
    try:
        pid = int(env.get(SESSION_PID_ENV, '').strip())
    except (AttributeError, TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _parse_session_id(env: Mapping[str, str]) -> str | None:
    """Parse *env*'s ``CLAUDE_CODE_SESSION_ID`` as a session-id token, else None. Never raises, never logs."""
    try:
        session_id = env.get(SESSION_ID_ENV, '').strip()
    except (AttributeError, TypeError):
        return None
    return session_id if is_session_id(session_id) else None


def is_shared_session_host(pid: int) -> bool:
    """Whether *pid* is a ``claude agents`` process hosting several sessions.

    The one home of the rule that a pid hosting several sessions cannot vouch
    for any one of them: it outlives each. An unreadable
    ``/proc/<pid>/cmdline`` (no /proc, a dead pid) reads as not a host.
    """
    if pid <= 0:
        return False
    try:
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except (OSError, ValueError):
        return False
    return len(argv) >= 2 and argv[1] == _SHARED_SESSION_HOST_SUBCOMMAND


@dataclass(frozen=True)
class SessionIdentity:
    """THIS session's lease identity, resolved once from the env (resolve_session_identity).

    claude_pid: ``$CLAUDE_PID`` parsed, or None when unusable.
    session_id: ``$CLAUDE_CODE_SESSION_ID`` parsed, or None when unusable.
    shared_host: whether *claude_pid* is a shared ``claude agents`` host
        (is_shared_session_host), so that pid does not die with this session.
    """

    claude_pid: int | None
    session_id: str | None
    shared_host: bool

    @property
    def session_scoped_pid(self) -> int | None:
        """A pid that dies with THIS session, or None."""
        return None if self.shared_host else self.claude_pid

    def lease_slug(self, name: str) -> str | None:
        """``<name>-<claude pid>``; under a shared host ``<name>-<claude pid>-<session id>``.

        None when that token is underivable -- see ``session_leases.py::default_lease_slug``.
        """
        if self.claude_pid is None:
            return None
        if not self.shared_host:
            return f'{name}-{self.claude_pid}'
        if self.session_id is None:
            return None
        return f'{name}-{self.claude_pid}-{self.session_id}'


def resolve_session_identity(env: Mapping[str, str] | None = None) -> SessionIdentity:
    """Resolve THIS session's SessionIdentity from *env* (default ``os.environ``). Never logs."""
    if env is None:
        env = os.environ
    claude_pid = _parse_session_pid(env)
    return SessionIdentity(
        claude_pid=claude_pid,
        session_id=_parse_session_id(env),
        shared_host=claude_pid is not None and is_shared_session_host(claude_pid),
    )


def lease_liveness_pid(identity: SessionIdentity) -> int:
    """The pid a lease body records for *identity*: its session-scoped pid, else 0, LOUDLY.

    The single home of the lease liveness degradation WARNING; why the
    fallback is 0 is resolve_session_pid's docstring.
    """
    if identity.session_scoped_pid is not None:
        return identity.session_scoped_pid
    if identity.shared_host:
        logger.warning(
            'lease_liveness_pid: $%s=%s names a shared `claude agents` host, whose pid '
            'outlives every session it hosts, so it cannot vouch for this one. The lease '
            'pid/liveness guard is DEGRADED to heartbeat-only staleness: recording pid 0, '
            'which is never alive, so this lease ages out and is reaped once its heartbeat '
            'stops.',
            SESSION_PID_ENV,
            identity.claude_pid,
        )
    else:
        logger.warning(
            'lease_liveness_pid: $%s is unset or unusable; the lease pid/liveness '
            'guard is DEGRADED to heartbeat-only staleness. Recording pid 0, which is '
            'never alive, so a crashed holder still ages out and is reaped normally. '
            'Note: neither `$$` nor `$PPID` substitutes for it -- `$$` in a Claude Code '
            'Bash tool call is the transient /bin/bash -c wrapper, and `$PPID` is not '
            'stable across tool calls either. Set $CLAUDE_PID; if you cannot, the lease '
            'slug is underivable too, so pass an explicit --slug <stable-token>.',
            SESSION_PID_ENV,
        )
    return 0


def resolve_session_pid(env: Mapping[str, str] | None = None) -> int:
    """Resolve THIS Claude Code session's long-lived pid -- the lease's liveness anchor.

    WHY THIS EXISTS (task 3994, defect 2). ``$$`` inside a Claude Code Bash
    tool call is NOT the session: each tool call is a fresh ``/bin/bash -c``
    wrapper whose pid is dead the instant the call returns. Measured:
    ``$$``=2487079 (transient wrapper) vs ``$PPID``==``$CLAUDE_PID``=2345789,
    with ``ps -o comm=`` confirming the latter is the long-lived ``claude``
    process. Both watcher SKILLs nonetheless prescribed ``--pid $$`` for a
    year, so ``pid_alive`` was ~always False for a watcher lease and the
    (dead pid AND aged heartbeat) staleness AND-guard collapsed to a pure
    heartbeat check.

    Resolving it HERE rather than only documenting it in each SKILL.md is
    deliberate: documentation alone re-creates the failure mode it fixes,
    because the pid was wrong for that whole year precisely because it
    depended on every skill doc getting one shell token right.

    Chain, in order: ``CLAUDE_PID`` (parsed, must be > 0, and not a shared
    ``claude agents`` host -- see SessionIdentity) -> ``0``.

    THE FALLBACK IS 0 DELIBERATELY, and specifically NOT ``os.getsid(0)`` ->
    ``os.getppid()`` (which this returned before review). Those are the
    durable-pid idiom ``session_ownership_probes.py::EventProbes.hand_launched_liveness_pid``
    uses for T6's hand-launched RECORDS, and they are wrong HERE, because a
    lease pid is consumed by a REAPER rather than merely displayed: the POSIX
    session leader is ordinarily the interactive shell, which lives for the
    terminal's whole lifetime -- so it stays alive after the ``claude``
    process it stood in for has crashed.
    ``pid_alive`` would then answer True forever, and because BOTH staleness
    rules (``session_leases.py::claim_lease``'s ``is_stale`` and
    ``session_leases.py::reap_stale_leases``' ``stale_pid``)
    require a DEAD pid AND an aged heartbeat, a crashed holder's lease would
    become PERMANENTLY unreapable -- recoverable only by a human
    ``lease-release --force``. That is a strictly worse failure than the
    ``--pid $$`` defect this function exists to fix, and the exact opposite of
    the heartbeat-only degradation documented here and in
    ``session_leases.py::LEASE_HEARTBEAT_TTL``.

    0 makes that documented degradation literally true: ``pid_alive`` returns
    False for every pid <= 0, so an unresolvable session pid collapses the
    AND-guard to its heartbeat half -- precisely the pre-3994 behaviour, which
    LEASE_HEARTBEAT_TTL's 7200s is still sized to survive -- instead of to a
    never-reapable lease. The cost is bounded and non-destructive: a contender
    reads ``holder_liveness=orphaned`` for a degraded holder that may well be
    alive, but a FRESH heartbeat still forces that contender to stand down,
    and the prescribed response to ``orphaned`` is to file a DecisionRecord
    and exit -- never to steal.

    Degradation is LOUD, never silent: lease_liveness_pid emits the WARNING.
    Never raises -- a pid we cannot resolve must not break a lease claim.
    """
    return lease_liveness_pid(resolve_session_identity(env))
