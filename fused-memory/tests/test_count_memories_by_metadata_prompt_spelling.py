"""Recon prompts spell count_memories_by_metadata with keyword args (task 7715).

The tool requires the keyword ``filters``; a positional spelling in a prompt
leads agents to pass the dict under another key, the call fails validation and
the stage2_suppress gate fails open. Pins the call spelling an LLM would copy.
"""

from __future__ import annotations

import pytest
from _count_call_spelling import non_keyword_count_calls
from _recon_prompt_write_scope import recon_system_prompts

_PROMPTS = recon_system_prompts()
_STAGE2_PROMPTS = sorted(name for name in _PROMPTS if 'stage2' in name.lower())

_KEYWORD_SUPPRESS_GATE = (
    "count_memories_by_metadata(project_id=..., "
    "filters={'task_id': str(task_id), 'stage2_suppress': True})"
)


@pytest.mark.parametrize('name', sorted(_PROMPTS))
def test_every_count_call_uses_keyword_filters(name: str) -> None:
    offending = non_keyword_count_calls(_PROMPTS[name])
    assert offending == [], (
        f'{name}: count_memories_by_metadata call(s) not spelled '
        f'`project_id=..., filters=...`: {offending}'
    )


@pytest.mark.parametrize('name', _STAGE2_PROMPTS)
def test_stage2_carries_the_keyword_suppress_gate(name: str) -> None:
    assert _KEYWORD_SUPPRESS_GATE in _PROMPTS[name]
