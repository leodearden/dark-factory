"""Session hooks whose owning claude is a shared ``claude agents`` host (task 6945).

A host runs every session it hosts in-process, so each hosted session's
owning claude pid is the host's. That pid must never key a ``sessions-by-pid``
pointer, and owner-pid equality must never prove a re-mint is the same
session. The owning claude here is a REAL process with the host's argv, so
``session_identity.is_shared_session_host`` reads its real ``/proc`` cmdline.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from _process_tree_fakes import claude_process_tree
from _session_hook_fixtures import assert_hook_env_isolated, isolate_hook_env
from _shared_host_fixtures import REQUIRES_PROC, running_shared_host

from orchestrator import session_hooks as sh
from orchestrator import session_records

pytestmark = REQUIRES_PROC

_CWD = '/home/leo/src/dark-factory'


@pytest.fixture(autouse=True)
def _isolate_hook_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolate_hook_env(monkeypatch, tmp_path)


def test_hook_env_is_isolated_for_this_module(tmp_path: Path) -> None:
    assert_hook_env_isolated(tmp_path)


@pytest.fixture
def shared_host_pid(tmp_path_factory: pytest.TempPathFactory) -> Iterator[int]:
    with running_shared_host(tmp_path_factory.mktemp('agents-host')) as pid:
        yield pid


def _event(session_id: str, source: str | None = None) -> dict[str, str]:
    event = {'session_id': session_id, 'cwd': _CWD}
    if source is not None:
        event['source'] = source
    return event


def _session_start(
    tmp_path: Path, host: int, session_id: str, source: str | None = None
) -> session_records.SessionRecord:
    return sh.run_session_start(
        _event(session_id, source), {}, root=tmp_path, process_tree=claude_process_tree(host)
    )


def _slug_for(session_id: str, tmp_path: Path) -> str:
    slug = session_records.resolve_session_slug_for_claude_session_id(session_id, root=tmp_path)
    assert slug is not None, session_id
    return slug


def _bound_session_id(slug: str, tmp_path: Path) -> str | None:
    return session_records.read_record(slug, root=tmp_path).claude_session_id


def test_hosted_sessions_write_no_pid_pointer(tmp_path: Path, shared_host_pid: int) -> None:
    host = shared_host_pid

    for session_id in ('sess-a', 'sess-b'):
        _session_start(tmp_path, host, session_id)
        sh.run_stop(
            _event(session_id), {}, root=tmp_path, process_tree=claude_process_tree(host)
        )

    assert not session_records.session_pointer_path_for_pid(host, root=tmp_path).exists()
    slugs = {session_id: _slug_for(session_id, tmp_path) for session_id in ('sess-a', 'sess-b')}
    assert slugs['sess-a'] != slugs['sess-b']
    for session_id, slug in slugs.items():
        record = session_records.read_record(slug, root=tmp_path)
        assert record.claude_session_id == session_id
        assert record.claude_owner_pid == host


@pytest.mark.parametrize('source', ['clear', 'compact'])
def test_a_hosted_remint_does_not_capture_the_sibling_session(
    tmp_path: Path, shared_host_pid: int, source: str
) -> None:
    host = shared_host_pid
    _session_start(tmp_path, host, 'sess-a')
    _session_start(tmp_path, host, 'sess-b')
    slug_b = _slug_for('sess-b', tmp_path)

    _session_start(tmp_path, host, 'sess-a2', source)

    assert _bound_session_id(slug_b, tmp_path) == 'sess-b'
    slug_a2 = _slug_for('sess-a2', tmp_path)
    assert slug_a2 != slug_b
    assert _bound_session_id(slug_a2, tmp_path) == 'sess-a2'


def test_a_pointer_written_before_this_change_is_not_followed(
    tmp_path: Path, shared_host_pid: int
) -> None:
    host = shared_host_pid
    slug_b = 'session-dark-factory-hosted-b'
    session_records.write_record(
        session_records.SessionRecord(
            session_slug=slug_b,
            status=session_records.Status.RUNNING,
            claude_session_id='sess-b',
            claude_owner_pid=host,
        ),
        root=tmp_path,
    )
    session_records.session_pointers_dir(tmp_path).mkdir(parents=True)
    session_records.session_pointer_path_for_pid(host, root=tmp_path).write_text(
        slug_b, encoding='utf-8'
    )

    _session_start(tmp_path, host, 'sess-a2', 'clear')

    assert _bound_session_id(slug_b, tmp_path) == 'sess-b'
