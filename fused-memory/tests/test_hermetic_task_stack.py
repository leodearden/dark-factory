"""The hermetic task stack helper (``_hermetic_task_stack.py``) builds no curator, so it reaches no corpus service."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from _fm_sockets import refused_port
from _hermetic_task_stack import HermeticTaskStack, hermetic_task_stack

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def unreachable_corpus_services(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point the tracked config's Qdrant and OpenAI URLs at a loopback port that refuses connects.

    A regression that rebuilds a curator then fails on its assertion without
    touching the live Qdrant or OpenAI.
    """
    with refused_port() as port:
        monkeypatch.setenv('QDRANT_URL', f'http://127.0.0.1:{port}')
        monkeypatch.setenv('OPENAI_API_URL', f'http://127.0.0.1:{port}/v1')
        yield


async def file_planned(stack: HermeticTaskStack, root: str) -> str:
    submitted = await stack.call('submit_task', {
        'project_root': root,
        'title': 'Probe the hermetic stack',
        'description': 'A planned task the hermeticity tests commit or retitle.',
        'planning_mode': True,
    })
    assert submitted.get('status') == 'deferred', submitted
    return str(submitted['task_id'])


async def test_a_pending_commit_builds_no_curator(tmp_path: Path) -> None:
    root = str(tmp_path)
    async with hermetic_task_stack(tmp_path) as stack:
        task_id = await file_planned(stack, root)

        committed = await stack.call('commit_planning', {'project_root': root, 'task_ids': task_id})
        assert 'error' not in committed, committed
        task = await stack.call('get_task', {'id': task_id, 'project_root': root})
        assert task['status'] == 'pending'

        state = await stack.call('get_curator_state', {})

    assert state['registries'] == []


async def test_a_title_update_builds_no_curator(tmp_path: Path) -> None:
    root = str(tmp_path)
    async with hermetic_task_stack(tmp_path) as stack:
        task_id = await file_planned(stack, root)

        updated = await stack.call('update_task', {
            'id': task_id, 'project_root': root, 'title': 'Probe the hermetic stack, retitled',
        })
        assert 'error' not in updated, updated

        state = await stack.call('get_curator_state', {})

    assert state['registries'] == []
