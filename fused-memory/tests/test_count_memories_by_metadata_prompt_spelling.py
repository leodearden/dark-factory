"""Recon prompts spell count_memories_by_metadata with keyword args (task 7715).

The tool requires the keyword ``filters``; a positional spelling in a prompt
leads agents to pass the dict under another key, the call fails validation and
the stage2_suppress gate fails open. Pins the call spelling an LLM would copy.
"""

from __future__ import annotations

import re

import pytest

from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT

_STAGE_PROMPTS = {
    'STAGE1_SYSTEM_PROMPT': STAGE1_SYSTEM_PROMPT,
    'STAGE2_SYSTEM_PROMPT': STAGE2_SYSTEM_PROMPT,
    "build_stage2_system_prompt('dark_factory')": build_stage2_system_prompt('dark_factory'),
    'STAGE3_SYSTEM_PROMPT': STAGE3_SYSTEM_PROMPT,
}

_CALL_OPENING = re.compile(r'count_memories_by_metadata\(')
_KEYWORD_CALL = re.compile(r'count_memories_by_metadata\(project_id=[^,()]*,\s*filters=')


@pytest.mark.parametrize('name', list(_STAGE_PROMPTS))
def test_every_count_call_uses_keyword_filters(name: str) -> None:
    prompt = _STAGE_PROMPTS[name]
    openings = len(_CALL_OPENING.findall(prompt))
    keyword_calls = len(_KEYWORD_CALL.findall(prompt))
    assert keyword_calls == openings, (
        f'{name}: {openings - keyword_calls} count_memories_by_metadata call(s) '
        'not spelled `project_id=..., filters=...`'
    )


def test_stage2_spells_the_gate_in_each_rule() -> None:
    assert len(_KEYWORD_CALL.findall(STAGE2_SYSTEM_PROMPT)) == 3
