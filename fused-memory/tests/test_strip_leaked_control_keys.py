"""Tests for scripts/strip_leaked_control_keys.py (task 2735).

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution — mirrors the pattern in
test_cleanup_count_snapshots.py / test_migrate_cross_graph_leak.py.

The suite covers the script's pure transform, ``correct_task_metadata``, and
drives the per-project sweep, ``clean_one_project``, with an injected fake
client so a server's refusal of get_task or update_task is pinned as an error,
never as a clean.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from _fm_helpers import load_script_module
from shared.testing_mcp_stub import StubAnswer, StubMcpServer

from fused_memory.backends.task_backend_errors import done_provenance_via_update_task_error

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'strip_leaked_control_keys.py'


_mod = load_script_module(SCRIPT_PATH, mod_name='strip_leaked_control_keys')
correct_task_metadata = _mod.correct_task_metadata


# A realistic task-2682-shaped metadata dict: legit keys — including the
# non-blessed-but-legitimate `_causation_id` / `gate_escalated_at`
# annotations — plus the errant `append` key that leaked in from the
# Stage-2 recon LLM's call-flag/metadata-field conflation.
_TASK_2682_SHAPED_METADATA = {
    'source': 'reconciliation',
    'prd_path': 'plans/some-prd.md',
    'prd_task_label': 'Some Task',
    'memory_hints': {'entities': ['A'], 'queries': ['q1']},
    '_causation_id': '8d3af293-0000-0000-0000-000000000000',
    'gate_escalated_at': '2026-07-17T00:00:00+00:00',
    'always_escalates': True,
    'task_kind': 'deterministic',
    'append': True,
}


def test_correct_task_metadata_drops_append_preserves_every_other_key():
    """The corrected dict is the input minus `append`; every legit key —
    including non-blessed annotations — is preserved with identical values."""
    cleaned, dropped = correct_task_metadata(_TASK_2682_SHAPED_METADATA)
    expected = {k: v for k, v in _TASK_2682_SHAPED_METADATA.items() if k != 'append'}
    assert cleaned == expected
    assert dropped == ['append']


def test_correct_task_metadata_idempotent_on_already_corrected_dict():
    """Re-running against an already-clean dict is a no-op."""
    cleaned, _ = correct_task_metadata(_TASK_2682_SHAPED_METADATA)
    twice_cleaned, dropped_again = correct_task_metadata(cleaned)
    assert twice_cleaned == cleaned
    assert dropped_again == []


class _ScriptedClient:
    """Answers each tool from *replies*, raising a reply that is an exception."""

    def __init__(self, replies: dict[str, object]) -> None:
        self._replies = replies
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> object:
        self.calls.append((name, arguments))
        reply = self._replies[name]
        if isinstance(reply, Exception):
            raise reply
        return reply


_DIRTY_DONE_METADATA = {
    **_TASK_2682_SHAPED_METADATA,
    'done_provenance': {'kind': 'merged', 'commit': 'abc1234'},
}
_DIRTY_DONE_TASK = {'id': '2682', 'status': 'done', 'metadata': _DIRTY_DONE_METADATA}
_PROJECT_ROOT = '/tmp/df-6128-root'


async def _clean_2682(update_task_reply: object) -> tuple[tuple[int, int, int], _ScriptedClient]:
    client = _ScriptedClient({'get_task': _DIRTY_DONE_TASK, 'update_task': update_task_reply})
    counts = await _mod.clean_one_project(client, _PROJECT_ROOT, ['2682'], dry_run=False)
    return counts, client


@pytest.mark.asyncio
async def test_refused_update_task_is_an_error_not_a_clean(capsys):
    counts, _ = await _clean_2682(done_provenance_via_update_task_error('2682'))

    assert counts == (1, 0, 1)
    captured = capsys.readouterr()
    assert 'dropped' not in captured.out
    assert 'done_provenance_via_update_task' in captured.err


@pytest.mark.asyncio
async def test_accepted_update_task_counts_a_clean(capsys):
    accepted = {
        'id': '2682', 'message': 'Task 2682 updated', 'updated': True, 'updated_task': {'id': '2682'},
    }
    counts, client = await _clean_2682(accepted)

    assert counts == (1, 1, 0)
    assert "dropped ['append']" in capsys.readouterr().out
    [update_arguments] = [arguments for name, arguments in client.calls if name == 'update_task']
    assert update_arguments['metadata_mode'] == 'replace'
    expected = {k: v for k, v in _DIRTY_DONE_METADATA.items() if k != 'append'}
    assert json.loads(update_arguments['metadata']) == expected


@pytest.mark.asyncio
async def test_update_task_that_raises_is_an_error_not_a_clean(capsys):
    counts, _ = await _clean_2682(RuntimeError('update_task failed: boom'))

    assert counts == (1, 0, 1)
    captured = capsys.readouterr()
    assert 'dropped' not in captured.out
    assert 'boom' in captured.err


_TASK_NOT_FOUND_REPLY = {
    'error': 'TASKMASTER_TOOL_ERROR: No tasks found for ID(s): 9999',
    'error_type': 'TaskNotFoundError',
}
_NON_NUMERIC_ID_REPLY = {
    'error': "INVALID_TASK_ID: non-numeric task id: 'infra'",
    'error_type': 'TaskmasterError',
    'metadata': {},
}


@pytest.mark.asyncio
@pytest.mark.parametrize('get_task_reply', [_TASK_NOT_FOUND_REPLY, _NON_NUMERIC_ID_REPLY])
async def test_refused_get_task_is_an_error_not_a_noop(capsys, get_task_reply):
    client = _ScriptedClient({'get_task': get_task_reply})

    counts = await _mod.clean_one_project(client, _PROJECT_ROOT, ['9999'], dry_run=False)

    assert counts == (0, 0, 1)
    assert [name for name, _ in client.calls] == ['get_task']
    assert get_task_reply['error_type'] in capsys.readouterr().err


def _args_for(stub: StubMcpServer) -> argparse.Namespace:
    return argparse.Namespace(
        server_url=stub.url, project_roots=[_PROJECT_ROOT], task_ids='9999', dry_run=False,
    )


@pytest.mark.asyncio
async def test_main_async_exits_nonzero_on_a_refused_reply():
    with StubMcpServer(stateful=False, payload=_TASK_NOT_FOUND_REPLY) as stub:
        exit_code = await _mod.main_async(_args_for(stub))

    assert exit_code == 1


@pytest.mark.asyncio
async def test_main_async_exits_nonzero_on_an_is_error_tool_result(capsys):
    with StubMcpServer(stateful=False, answer=StubAnswer.IS_ERROR) as stub:
        exit_code = await _mod.main_async(_args_for(stub))

    assert exit_code == 1
    captured = capsys.readouterr()
    assert 'no-op' not in captured.out
    assert 'scripted stub failure' in captured.err
