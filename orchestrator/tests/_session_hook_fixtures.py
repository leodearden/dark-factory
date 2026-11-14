"""Setup shared by the ``orchestrator/tests/test_session_hooks*.py`` modules: a hermetic hook environment, a hook stdin payload, seeded session-registry records."""

from __future__ import annotations

import io
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from orchestrator import session_registry as sr


def isolate_hook_env(monkeypatch: pytest.MonkeyPatch, fleet_root: Path) -> None:
    """Scrub every ``CLAUDE_SPAWN_*`` var and ``T3CODE_THREAD_ID``, and pin ``CLAUDE_FLEET_ROOT`` to *fleet_root*.

    ``sh.main`` reads ``os.environ``, so a ``CLAUDE_SPAWN_*`` identity leaked
    from a fleet-spawned launching context would re-key or re-attribute the
    records a test asserts on (tasks 2643, 2940), and a ``T3CODE_THREAD_ID``
    leaked from a T3-hosted runner would key them on that thread's pointer
    and stamp them t3 (task 8124). ``hook_session_slug`` reads
    the registry, so the default root must not be the developer's real
    ``~/.claude/fleet`` (task 4193).

    Wrap it in a MODULE-LOCAL autouse fixture, never ``conftest.py``:
    ``test_session_registry.py`` exercises the unset fleet-root default.
    Creates no directories.
    """
    for name in [k for k in os.environ if k.startswith('CLAUDE_SPAWN_')]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv('T3CODE_THREAD_ID', raising=False)
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(fleet_root))


def assert_hook_env_isolated(fleet_root: Path) -> None:
    """Fail unless ``isolate_hook_env(..., fleet_root)`` is in effect for the running test.

    Every module that wraps ``isolate_hook_env`` calls this from a test of its
    own, so a module that loses its wrapper fails rather than silently reading
    the real fleet root.
    """
    assert [k for k in os.environ if k.startswith('CLAUDE_SPAWN_')] == []
    assert 'T3CODE_THREAD_ID' not in os.environ
    assert os.environ.get('CLAUDE_FLEET_ROOT') == str(fleet_root)


def stdin_json(monkeypatch: pytest.MonkeyPatch, payload: Mapping[str, object]) -> None:
    monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps(payload)))


def owner_ppid_env(slug: str, ppid: int) -> dict[str, str]:
    return {'CLAUDE_SPAWN_SESSION_ID': slug, 'CLAUDE_SPAWN_OWNER_PPID': str(ppid)}


def write_bound_parent(slug: str, root: Path, pid: int) -> None:
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=pid,
            claude_session_id='uuid-parent',
        ),
        root=root,
    )


def launching_record(
    slug: str,
    root: Path,
    *,
    pid: int = 4193001,
    start_ts: str | None = None,
) -> None:
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.LAUNCHING,
            launcher_pid=pid,
            role='cockpit',
            start_ts=start_ts if start_ts is not None else datetime.now(UTC).isoformat(),
        ),
        root=root,
    )
