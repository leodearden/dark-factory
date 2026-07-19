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

import inspect

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


def test_run_eval_accepts_memory_endpoint_kwarg():
    """run_eval exposes an optional memory_endpoint it threads to build_eval_orch_config."""
    params = inspect.signature(run_eval).parameters
    assert 'memory_endpoint' in params
    # Default None → the profile null-sentinel isolation stands with no wiring.
    assert params['memory_endpoint'].default is None
