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

import httpx

from orchestrator.evals.runner import RecordingMemorySink


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
