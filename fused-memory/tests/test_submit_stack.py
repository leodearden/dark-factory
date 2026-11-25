"""The submit stack helper (``_submit_stack.py``): a hermetic curator corpus and a load-tolerant ticket wait."""

from __future__ import annotations

import socket
from typing import Any

import pytest
from _submit_stack import Stack
from _submit_stack import curator_creates as curator_creates
from _submit_stack import passthrough_main_checkout as passthrough_main_checkout
from _submit_stack import stack as stack

from fused_memory.middleware.curator_types import PreparedCandidate


@pytest.fixture(autouse=True)
def unreachable_corpus_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the tracked config's Qdrant and OpenAI URLs at a loopback port nothing listens on.

    Autouse, so the redirect is in place before ``stack`` builds its ``FusedMemoryConfig()``.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv('QDRANT_URL', f'http://127.0.0.1:{port}')
    monkeypatch.setenv('OPENAI_API_URL', f'http://127.0.0.1:{port}/v1')


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
