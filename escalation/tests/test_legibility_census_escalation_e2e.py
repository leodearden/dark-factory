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
`cd escalation && uv run pytest tests/` both sides import (the `scripts/`
path inserts live in this directory's conftest.py), and this suite already
hosts a cross-package client/server lockstep test of exactly this shape
(`test_capability_guard_http.py`).
"""

from __future__ import annotations

import logging

import census
import pytest

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
