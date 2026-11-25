"""The submit stack helper (``_submit_stack.py``): a hermetic curator corpus and a load-tolerant ticket wait."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from _curator_helpers import hold_first_prepare
from _fm_sockets import refused_port
from _submit_stack import Stack
from _submit_stack import curator_creates as curator_creates
from _submit_stack import passthrough_main_checkout as passthrough_main_checkout
from _submit_stack import stack as stack

from fused_memory.middleware.curator_types import PreparedCandidate


@pytest.fixture(autouse=True)
def unreachable_corpus_services(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point the tracked config's Qdrant and OpenAI URLs at a loopback port that refuses connects.

    Autouse, so the redirect is in place before ``stack`` builds its ``FusedMemoryConfig()``.
    """
    with refused_port() as port:
        monkeypatch.setenv('QDRANT_URL', f'http://127.0.0.1:{port}')
        monkeypatch.setenv('OPENAI_API_URL', f'http://127.0.0.1:{port}/v1')
        yield


def _followup(title: str) -> dict[str, Any]:
    return {
        'title': title,
        'description': f'{title}, described.',
        'priority': 'low',
        'metadata': {
            'source': 'agent-followup', 'spawn_context': 'agent_followup',
            'spawned_from': '101', 'files': ['src/app/a.py'],
        },
    }


@pytest.mark.asyncio
async def test_the_stack_curates_over_an_in_process_corpus(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    await stack.file_normal(**_followup('Tidy the a module'))

    [bundle] = curator_creates
    assert bundle.corpus_error is None
    assert bundle.withheld is not None and bundle.withheld.unavailable_streams == ()


@pytest.mark.asyncio
async def test_resolve_waits_again_while_the_ticket_is_still_pending(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    release = hold_first_prepare(stack.curator)
    try:
        submitted = await stack.call('submit_task', **_followup('Tidy the a module'))
        assert 'ticket' in submitted, submitted
        resolving = asyncio.create_task(stack.resolve(submitted['ticket'], wait_seconds=0))
        expired = await stack.call('resolve_ticket', ticket=submitted['ticket'], timeout_seconds=0)
    finally:
        release.set()
    answer = await resolving

    assert (expired['status'], expired['reason']) == ('pending', 'wait_timeout')
    assert answer['status'] == 'created'
    assert 'task_id' in answer


@pytest.mark.asyncio
async def test_resolve_returns_an_open_answer_after_shutdown_without_asking_again(stack: Stack) -> None:
    release = hold_first_prepare(stack.curator)
    try:
        submitted = await stack.call('submit_task', **_followup('Tidy the a module'))
        assert 'ticket' in submitted, submitted
        await stack.interceptor.close()
        answer = await stack.resolve(submitted['ticket'])
    finally:
        release.set()

    assert (answer['status'], answer['reason']) == ('pending', 'store_unavailable')
