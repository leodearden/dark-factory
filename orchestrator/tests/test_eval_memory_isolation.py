"""Isolation + recording-sink tests for eval memory writes (D8).

PRD eval-framework-revival §ε, Contract C1.mem, Boundary test B5.

In eval mode every orchestrator-side memory write
(``TaskWorkflow._write_completion_to_memory`` / ``_write_decisions_to_memory``
/ ``_write_suggestions_to_memory``) POSTs raw httpx to ``{self.mcp.url}/mcp/``,
where ``self.mcp = _EvalMcpStub(orch_config.fused_memory.url)``. Because
``fused_memory.url`` defaults to the real production endpoint
(``http://localhost:8002``), an un-isolated green eval run would write into
PRODUCTION dark_factory memory.

These tests exercise the MECHANISM ε delivers — a real green eval run (real
architect/implementer/reviewer agents) belongs to the integration gate ι, which
depends on ε:
  - ``RecordingMemorySink`` is a real loopback HTTP endpoint that records every
    intended write and returns a benign JSON-RPC success envelope.
  - the EVAL_PROFILE default points ``fused_memory.url`` off-production.
  - the ``memory_endpoint`` override routes the REAL
    ``_write_completion_to_memory`` write to the sink and never to
    ``http://localhost:8002``.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.profile import EVAL_PROFILE
from orchestrator.evals.runner import (
    RecordingMemorySink,
    _EvalMcpStub,
    build_eval_orch_config,
    run_eval,
)
from orchestrator.workflow import TaskWorkflow

# The non-routable null sentinel the eval profile pins fused_memory.url to:
# 127.0.0.1:1 gives an immediate ECONNREFUSED (fast-fail, no slow DNS/timeout)
# and can never be the production dark_factory memory store.
_NULL_SENTINEL = 'http://127.0.0.1:1'
# The production fused-memory endpoint an un-isolated eval run would write into.
_PRODUCTION = 'http://localhost:8002'


def _add_memory_request(
    content: str,
    category: str,
    project_id: str,
    agent_id: str = 'orchestrator-task-t',
) -> dict:
    """The exact JSON-RPC body ``_write_completion_to_memory`` POSTs."""
    return {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'tools/call',
        'params': {
            'name': 'add_memory',
            'arguments': {
                'content': content,
                'category': category,
                'project_id': project_id,
                'agent_id': agent_id,
            },
        },
    }


def test_recording_sink_records_posted_write():
    """RecordingMemorySink captures a POSTed tools/call and replies with a success envelope."""
    with RecordingMemorySink() as sink:
        # Real ephemeral loopback endpoint (not a static literal).
        assert sink.url.startswith('http://127.0.0.1:')

        body = _add_memory_request(
            content='Completed: something',
            category='observations_and_summaries',
            project_id='dark_factory',
        )
        resp = httpx.post(f'{sink.url}/mcp/', json=body, timeout=10)

        # Well-formed JSON-RPC success envelope, no error.
        assert resp.status_code == 200
        payload = resp.json()
        assert payload['jsonrpc'] == '2.0'
        assert 'id' in payload
        assert 'result' in payload
        assert 'error' not in payload

        # The intended write is captured as (tool_name, arguments).
        assert len(sink.writes) == 1
        tool_name, arguments = sink.writes[0]
        assert tool_name == 'add_memory'
        assert arguments['content'] == 'Completed: something'
        assert arguments['category'] == 'observations_and_summaries'
        assert arguments['project_id'] == 'dark_factory'


def test_eval_profile_pins_fused_memory_url_to_null_sentinel():
    """EVAL_PROFILE isolates the fused-memory endpoint to a non-routable sentinel (D8)."""
    assert EVAL_PROFILE['fused_memory.url'] == _NULL_SENTINEL


@pytest.mark.usefixtures('code_default_config')
def test_build_eval_orch_config_default_isolates_memory_off_production(tmp_path):
    """With no memory_endpoint, the built eval config points fused_memory.url off production.

    Default isolation flows purely through the profile — no recording wiring
    and no CLI change required. Even a zero-wiring eval run can never reach the
    real dark_factory memory store.
    """
    base = OrchestratorConfig(project_root=tmp_path)
    # Premise: a fresh code-default base carries the real production endpoint,
    # so the sentinel below is a genuine off-production divergence.
    assert base.fused_memory.url == _PRODUCTION

    cfg = EvalConfig(name='t', backend='claude', model='sonnet', effort='high')
    task = {'id': 't', 'project_root': str(tmp_path)}

    result = build_eval_orch_config(cfg, task, base)

    assert result.fused_memory.url == _NULL_SENTINEL
    assert result.fused_memory.url != _PRODUCTION
    assert result.fused_memory.url != base.fused_memory.url


@pytest.mark.asyncio
async def test_memory_endpoint_override_routes_real_workflow_write_to_sink(tmp_path):
    """A memory_endpoint override routes the REAL _write_completion_to_memory write to the sink.

    Exercises the genuine workflow write path (not a re-implementation): the
    override lands on the single ``orch_config.fused_memory.url`` leaf the writes
    funnel through, and the real ``_write_completion_to_memory`` POST is captured
    by the recording sink — never at production ``http://localhost:8002``.
    """
    base = OrchestratorConfig(project_root=tmp_path)
    cfg = EvalConfig(name='t', backend='claude', model='sonnet', effort='high')
    task = {'id': 't', 'project_root': str(tmp_path)}

    with RecordingMemorySink() as sink:
        orch_config = build_eval_orch_config(cfg, task, base, memory_endpoint=sink.url)

        # The override lands on the single leaf the writes funnel through
        # (fused_memory.url → _EvalMcpStub.url → self.mcp.url).
        assert orch_config.fused_memory.url == sink.url
        assert _EvalMcpStub(orch_config.fused_memory.url).url == sink.url
        assert orch_config.fused_memory.url != _PRODUCTION
        # A sample profile field still holds — the override layers over the
        # profile rather than replacing it wholesale.
        assert orch_config.unblock_auto.enabled is False

        # Drive the REAL workflow memory-write method against the sink, wiring
        # only the minimal attrs _write_completion_to_memory reads.
        wf = object.__new__(TaskWorkflow)
        wf.mcp = _EvalMcpStub(orch_config.fused_memory.url)
        wf.config = orch_config
        wf.task = {'title': 'My task', 'description': 'does a thing'}
        wf.plan = {
            'analysis': 'because reasons',
            'design_decisions': [],
            'steps': [{'status': 'done'}, {'status': 'pending'}],
        }
        wf.modules = ['orchestrator/src/orchestrator/evals']
        wf.task_id = 't'

        await wf._write_completion_to_memory()

        # The real write was captured by the sink — proving isolation covers the
        # genuine write path, and the production write-count delta stays 0.
        assert len(sink.writes) == 1
        tool_name, arguments = sink.writes[0]
        assert tool_name == 'add_memory'
        assert arguments['category'] == 'observations_and_summaries'
        assert arguments['project_id'] == orch_config.fused_memory.project_id


@pytest.mark.asyncio
async def test_run_eval_forwards_memory_endpoint_to_build_eval_orch_config(tmp_path, monkeypatch):
    """run_eval threads its memory_endpoint straight into build_eval_orch_config.

    A behavioral guard (not a signature lock): it captures the value
    ``build_eval_orch_config`` actually receives at the runner's call site, then
    short-circuits before any workflow/worktree work. Deleting the
    ``memory_endpoint=memory_endpoint`` forward at that call site would make this
    fail (captured would be the default ``None``). No other test covers run_eval's
    forwarding — ``test_memory_endpoint_override_routes_real_workflow_write_to_sink``
    exercises ``build_eval_orch_config`` directly.
    """

    class _Sentinel(Exception):
        pass

    captured: dict = {}

    def _rec(config, task, base_config=None, *, memory_endpoint=None):
        captured['memory_endpoint'] = memory_endpoint
        raise _Sentinel

    monkeypatch.setattr('orchestrator.evals.runner.build_eval_orch_config', _rec)
    monkeypatch.setattr(
        'orchestrator.evals.runner.load_task',
        lambda _path: {'id': 't', 'project_root': str(tmp_path)},
    )

    cfg = EvalConfig(name='t', backend='claude', model='sonnet', effort='high')

    # worktree_path short-circuits create_eval_worktree, so no git op runs; the
    # recorder raises _Sentinel to stop before any workflow/worktree work.
    with pytest.raises(_Sentinel):
        await run_eval(
            tmp_path / 'task.json',
            cfg,
            memory_endpoint='http://127.0.0.1:54321',
            worktree_path=tmp_path,
        )

    assert captured['memory_endpoint'] == 'http://127.0.0.1:54321'


def _minimal_eval_workflow(orch_config) -> TaskWorkflow:
    """A bare TaskWorkflow wired only with the attrs the ``_write_*_to_memory`` methods read.

    Built via ``object.__new__`` (no ``__init__``) so no real orchestrator wiring
    is needed — just the ``mcp`` / ``config`` / ``task`` / ``plan`` / ``modules`` /
    ``task_id`` the three memory-write methods touch. The plan carries a
    design decision (with both ``decision`` and ``rationale``) so
    ``_write_decisions_to_memory`` emits exactly one write.
    """
    wf = object.__new__(TaskWorkflow)
    wf.mcp = _EvalMcpStub(orch_config.fused_memory.url)
    wf.config = orch_config
    wf.task = {'title': 'My task', 'description': 'does a thing'}
    wf.plan = {
        'analysis': 'because reasons',
        'design_decisions': [{'decision': 'chose A', 'rationale': 'A beats B'}],
        'steps': [{'status': 'done'}, {'status': 'pending'}],
    }
    wf.modules = ['orchestrator/src/orchestrator/evals']
    wf.task_id = 't'
    return wf


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('write_path', 'expected_category'),
    [
        (
            lambda wf: wf._write_completion_to_memory(),
            'observations_and_summaries',
        ),
        (
            lambda wf: wf._write_decisions_to_memory(),
            'decisions_and_rationale',
        ),
        (
            lambda wf: wf._write_suggestions_to_memory(
                SimpleNamespace(suggestions=[{'category': 'perf', 'description': 'do X'}])
            ),
            'preferences_and_norms',
        ),
    ],
    ids=['completion', 'decisions', 'suggestions'],
)
async def test_all_memory_write_paths_route_through_mcp_url_to_sink(
    tmp_path, write_path, expected_category
):
    """ALL THREE orchestrator-side memory writes funnel through ``self.mcp.url`` to the sink.

    The isolation guarantee rests on every ``_write_*_to_memory`` method POSTing to
    ``f'{self.mcp.url}/mcp/'`` — the single leaf the ``memory_endpoint`` override lands
    on. ``test_memory_endpoint_override_routes_real_workflow_write_to_sink`` proves this
    for the completion path only; here each real write path
    (``_write_completion_to_memory`` / ``_write_decisions_to_memory`` /
    ``_write_suggestions_to_memory``) is driven against the recording sink. If any
    were refactored to compute its URL differently, its write would escape the sink
    and this test would fail (0 captured writes) — catching the regression the
    shared-leaf mechanism otherwise hides.
    """
    base = OrchestratorConfig(project_root=tmp_path)
    cfg = EvalConfig(name='t', backend='claude', model='sonnet', effort='high')
    task = {'id': 't', 'project_root': str(tmp_path)}

    with RecordingMemorySink() as sink:
        orch_config = build_eval_orch_config(cfg, task, base, memory_endpoint=sink.url)
        wf = _minimal_eval_workflow(orch_config)

        await write_path(wf)

        # The real write landed in the sink — never at production.
        assert orch_config.fused_memory.url != _PRODUCTION
        assert len(sink.writes) == 1
        tool_name, arguments = sink.writes[0]
        assert tool_name == 'add_memory'
        assert arguments['category'] == expected_category
        assert arguments['project_id'] == orch_config.fused_memory.project_id
