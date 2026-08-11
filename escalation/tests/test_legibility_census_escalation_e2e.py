"""END-TO-END: the legibility census's fail-loud escalation must actually land.

The headline acceptance test for task 3644. Every pre-existing test of this
path monkeypatched the MCP boundary itself (`census._post_mcp_tool_call`) or
faked `httpx.post`, so all of them asserted only that `escalate_fn` was CALLED.
Nothing ever spoke the real streamable-HTTP protocol to a real escalation
server -- which is exactly why a TRANSPORT-level rejection stayed invisible
while every legibility escalation ever filed was silently dropped:

    HTTP/1.1 400 Bad Request
    mcp-session-id: 93599e03ba3b4baeb5bd0d2b6b399ddd
    {"jsonrpc":"2.0","id":"server-error",
     "error":{"code":-32600,"message":"Bad Request: Missing session ID"}}

(captured verbatim from a live probe against the running dark_factory
escalation server on :8103, 2026-08-05, re-confirmed 2026-08-10). The
best-effort closure in `census._build_default_escalate_fn` logs that and
swallows it, leaving the queue EMPTY -- green on paper, dead in practice.

So these tests drive the REAL protocol against a REAL escalation MCP server
(the `serve_escalation_mcp` fixture) and assert the escalation EXISTS
afterwards and is RETRIEVABLE BY ID, which is the user-observable signal the
task names: "force a census failure and observe a real escalation appear in
the queue, retrievable by id".

WHY THIS FILE LIVES IN `escalation/tests/` rather than `scripts/tests/`:
verified by probe, not assumed. `scripts/tests/` runs under
`uv run --project shared`, and `shared` does not depend on fastmcp -- an
in-process real-server test there would fail to import or, worse, be
`importorskip`-ed into a silent skip, reproducing the very green-on-paper
failure this task exists to close. Adding fastmcp+escalation to `shared`'s
deps would invert the package dependency direction. Under
`cd escalation && uv run pytest tests/` both sides import, and this suite
already hosts a cross-package client/server lockstep test of exactly this
shape (`test_capability_guard_http.py`).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# --- scoped legibility-scripts path setup (review finding #6, task 3644) ---
#
# Deliberately HERE and not in this directory's conftest.py: the legibility
# scripts dir contributes very generic flat top-level module names (`config`,
# `inventory`, `sampling`, `digest`, `coder`, `codebook`), and putting those on
# the path for all ~1080 tests in this suite would shadow any same-named module
# for every one of them, failing as a confusing wrong-module import rather than
# an ImportError. This module is the only consumer, so it does its own setup.
#
# APPEND, never insert(0, ...): repo and stdlib names must keep winning even
# once this module has been imported (sys.path edits are process-global and
# outlive collection of this file).
#
# `scripts/` provides the `legibility` PACKAGE (`from legibility import
# census_trigger`, which census.py itself does); `scripts/legibility/` provides
# the flat module names (`import census`). The `shared` stub installed by
# conftest.py is compatible: its `__path__` points at the real
# `shared/src/shared`, so census.py's `shared.cap_markers` import resolves
# against real source.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / 'scripts'), str(_REPO_ROOT / 'scripts' / 'legibility')):
    if _p not in sys.path:
        sys.path.append(_p)

import census  # noqa: E402  — needs the sys.path setup above
import check_transcript_persistence  # noqa: E402
import nightly  # noqa: E402
import pytest  # noqa: E402

_RAISED_MESSAGE = 'codebook merge produced an invalid codebook'
_PROJECT_ID = 'e2e_census_project'


def _write_legibility_yaml(project_root, *, escalation_port):
    """Write a minimal valid legibility.yaml at the default location.

    Same shape (and the same deliberate plain-text-lines approach, not a
    `yaml.safe_dump` round trip) as `_write_legibility_yaml` in
    `scripts/tests/test_legibility_census.py` -- kept independent of the
    module under test's own YAML writer. The only difference that matters
    here is `escalation_port`, pointed at the live test server's ephemeral
    port instead of the fleet's :8103.
    """
    config_path = project_root / 'docs' / 'legibility' / 'legibility.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        '\n'.join([
            f'project_id: {_PROJECT_ID}',
            f'project_root: {project_root}',
            f'escalation_port: {escalation_port}',
            'cwd_prefixes:',
            f'  - {project_root}',
        ]) + '\n',
        encoding='utf-8',
    )
    return config_path


@pytest.fixture
def live_census_project(tmp_path, serve_escalation_mcp, monkeypatch):
    """A tmp project wired to a live escalation MCP server, census set to fail.

    Yields `(project_root, queue)`. `run_census` is monkeypatched to raise, so
    `census.main` takes its fail-loud catch-all -- the exact production path
    that files the escalation.
    """
    _base_url, port, queue = serve_escalation_mcp(tmp_path / 'queue')
    project_root = tmp_path / 'project'
    project_root.mkdir()
    _write_legibility_yaml(project_root, escalation_port=port)

    def _raising_run_census(**kwargs):
        raise RuntimeError(_RAISED_MESSAGE)

    monkeypatch.setattr(census, 'run_census', _raising_run_census)
    yield project_root, queue


def test_census_hard_failure_lands_a_retrievable_escalation(
    live_census_project, caplog,
):
    """The acceptance assertion: a forced census failure leaves a REAL
    escalation in a REAL queue, retrievable by id -- over the real transport.

    Before task 3644 this failed with ZERO escalations in the queue: the bare
    `tools/call` POST 400s at the transport layer and the best-effort closure
    swallows it.
    """
    project_root, queue = live_census_project

    with caplog.at_level(logging.WARNING, logger='legibility.census'):
        exit_code = census.main(['--project-root', str(project_root), '--force'])

    # (1) The authoritative exit signal is never masked by the escalation POST.
    assert exit_code == 1

    # (2) The escalation EXISTS -- not "escalate_fn was called".
    pending = queue.get_pending()
    assert len(pending) == 1, (
        f'expected exactly one escalation in the queue, got {len(pending)}: '
        f'{[e.id for e in pending]}'
    )

    # (3) ...and is RETRIEVABLE BY ID, which is the user-observable signal the
    # task names. Read the id off the queue and resolve it back through the
    # queue's own lookup rather than trusting the in-memory record.
    fetched = queue.get(pending[0].id)
    assert fetched is not None, f'escalation {pending[0].id} is not retrievable by id'

    assert fetched.task_id == f'legibility-census-{_PROJECT_ID}'
    assert fetched.agent_role == 'legibility-census'
    assert fetched.category == 'infra_issue'
    assert fetched.severity == 'info'
    assert fetched.summary and _PROJECT_ID in fetched.summary
    assert _RAISED_MESSAGE in fetched.detail

    # (4) No swallow. `_build_default_escalate_fn`'s best-effort except logs
    # "escalation post failed" and returns {} -- pinning its ABSENCE is what
    # stops a future silent regression to that path from passing (1)-(3) is
    # not enough on its own, since a swallowed POST would fail (2) loudly, but
    # this makes the diagnosis immediate rather than a mystery empty queue).
    swallowed = [
        r.getMessage() for r in caplog.records
        if 'escalation post failed' in r.getMessage()
    ]
    assert not swallowed, f'the escalation POST was swallowed best-effort: {swallowed}'


def test_default_escalate_fn_returns_the_live_servers_response(
    tmp_path, serve_escalation_mcp,
):
    """The poster's RETURN VALUE is the real server's response, not a
    swallowed `{}` -- i.e. the census learns the id of what it filed.

    A closure that swallows everything and returns `{}` satisfies "no
    exception escaped" while filing nothing; asserting on the returned id
    (and resolving that id back out of the queue) is what makes the
    difference observable.
    """
    _base_url, port, queue = serve_escalation_mcp(tmp_path / 'queue')
    project_root = tmp_path / 'project'
    project_root.mkdir()
    _write_legibility_yaml(project_root, escalation_port=port)

    cfg = census.config.load_config(
        project_root / 'docs' / 'legibility' / 'legibility.yaml'
    )
    escalate_fn = census._build_default_escalate_fn(cfg)

    response = escalate_fn(
        category='infra_issue',
        severity='info',
        summary=f'legibility census run failed ({_PROJECT_ID}): {_RAISED_MESSAGE}',
        detail='traceback would go here',
    )

    assert response, 'escalate_fn returned a falsy response -- the POST was swallowed'
    escalation_id = response.get('id')
    assert escalation_id, f'no id in the escalation response: {response!r}'

    fetched = queue.get(escalation_id)
    assert fetched is not None, (
        f'escalate_fn reported id {escalation_id!r} but the queue cannot '
        f'resolve it -- the census learned an id for an escalation that does '
        f'not exist'
    )
    assert fetched.task_id == f'legibility-census-{_PROJECT_ID}'
    assert fetched.category == 'infra_issue'


# ---------------------------------------------------------------------------
# The two SIBLING posters, end to end (review finding #4, task 3644)
# ---------------------------------------------------------------------------
#
# `nightly._default_poster` and `check_transcript_persistence._default_poster`
# carried the identical defect and were rewritten by the same change, but were
# covered only by `install_fake_httpx` unit tests whose fakes were widened to
# keep passing -- exactly the test shape this file's docstring names as the
# reason the bug survived ("nothing ever spoke the real streamable-HTTP
# protocol to a real escalation server"). Without these, the claim that the
# trickle and transcript-loss alarms now land rests on a hand-run live probe
# recorded in a commit message rather than on a repeatable test.


def _live_cfg(tmp_path, port):
    """A LegibilityConfig pointed at the live test server's port."""
    project_root = tmp_path / 'project'
    project_root.mkdir()
    _write_legibility_yaml(project_root, escalation_port=port)
    return census.config.load_config(
        project_root / 'docs' / 'legibility' / 'legibility.yaml'
    )


def test_nightly_trickle_escalation_lands_against_a_live_server(
    tmp_path, serve_escalation_mcp,
):
    """The trickle's fail-loud escalation (extractor crash, coder storm, commit
    failure) must actually reach the queue over the real transport."""
    _base_url, port, queue = serve_escalation_mcp(tmp_path / 'queue')
    cfg = _live_cfg(tmp_path, port)

    landed = nightly.post_escalation(
        cfg,
        f'legibility trickle: extractor crashed ({_PROJECT_ID})',
        'traceback would go here',
    )

    assert landed is True, 'post_escalation reported failure against a live server'

    pending = queue.get_pending()
    assert len(pending) == 1, (
        f'expected exactly one escalation, got {len(pending)}: '
        f'{[e.id for e in pending]}'
    )
    fetched = queue.get(pending[0].id)
    assert fetched is not None, f'escalation {pending[0].id} is not retrievable by id'
    assert fetched.task_id == f'legibility-trickle-{_PROJECT_ID}'
    assert fetched.agent_role == 'legibility-trickle'
    assert fetched.category == 'infra_issue'
    assert _PROJECT_ID in fetched.summary


def test_transcript_loss_alarm_lands_against_a_live_server(
    tmp_path, serve_escalation_mcp,
):
    """The transcript-loss alarm must actually reach the queue too.

    `post_findings` -- NOT `post_escalation`; this module's escalation
    entrypoint takes a `Sequence[MissingTranscript]`, so a minimal non-empty
    findings list is what reaches the POST.
    """
    _base_url, port, queue = serve_escalation_mcp(tmp_path / 'queue')
    cfg = _live_cfg(tmp_path, port)

    finding = check_transcript_persistence.MissingTranscript(
        session_slug='sess-lost-e2e',
        cwd=str(tmp_path / 'project'),
        prompt_prefix='Diagnose the stuck reconciliation.',
        start_ts='2026-08-10T00:00:00+00:00',
        exit_code=0,
        expected_dir=tmp_path / 'projects' / 'enc',
    )

    landed = check_transcript_persistence.post_findings(cfg, [finding])

    assert landed is True, 'post_findings reported failure against a live server'

    pending = queue.get_pending()
    assert len(pending) == 1, (
        f'expected exactly one escalation, got {len(pending)}: '
        f'{[e.id for e in pending]}'
    )
    fetched = queue.get(pending[0].id)
    assert fetched is not None, f'escalation {pending[0].id} is not retrievable by id'
    assert fetched.task_id == f'legibility-transcript-check-{_PROJECT_ID}'
    assert fetched.agent_role == 'legibility-transcript-check'
    assert fetched.category == 'infra_issue'
    assert 'sess-lost-e2e' in fetched.detail


def test_the_transport_releases_its_session_on_the_real_server(
    tmp_path, serve_escalation_mcp, monkeypatch,
):
    """The session opened by the handshake is really GONE server-side.

    The unit tests can only assert that a DELETE was SENT; and the DELETE is
    best-effort by design, so a version that sent a request the server
    rejected would pass them and still leak. `StreamableHTTPSessionManager`
    keeps every live session in `_server_instances` with its own anyio task
    until an explicit DELETE, and the escalation server is a long-lived
    process serving the whole fleet -- so "the server no longer knows this
    session" is the assertion that actually matters (review finding #2, task
    3644).

    Proven black-box: re-POST with the SAME session id the transport used and
    require the server to reject it as unknown.
    """
    import httpx
    from legibility import census_trigger

    base_url, port, _queue = serve_escalation_mcp(tmp_path / 'queue')
    cfg = _live_cfg(tmp_path, port)

    captured = {}
    real_terminate = census_trigger._terminate_mcp_session

    def _spy(url, session_headers, *, timeout):
        captured['headers'] = dict(session_headers)
        return real_terminate(url, session_headers, timeout=timeout)

    monkeypatch.setattr(census_trigger, '_terminate_mcp_session', _spy)

    assert nightly.post_escalation(cfg, f'summary ({_PROJECT_ID})', 'detail') is True

    session_headers = captured.get('headers') or {}
    assert session_headers.get('mcp-session-id'), (
        'the transport handshook but never reached session termination'
    )

    replayed = httpx.post(
        f'{base_url}/mcp',
        json={
            'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
            'params': {'name': 'escalate_info', 'arguments': {}},
        },
        headers=session_headers,
        timeout=10.0,
    )
    assert replayed.status_code == 404, (
        f'the server still accepts session '
        f'{session_headers["mcp-session-id"]!r} (HTTP {replayed.status_code}) '
        f'-- the DELETE did not actually release it, so every legibility '
        f'escalation still leaks one server-side session: {replayed.text!r}'
    )
