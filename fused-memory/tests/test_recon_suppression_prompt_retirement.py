"""No recon stage prompt advertises the retired stage-facing suppression remedy (task 4863).

stage1_flag_suppression records are operator-managed recon_ledger rows; a
stage cannot create one (add_memory/add_system_record refuse the kind), and a
Mem0 record of that kind has no gate effect. So no stage prompt may name the
operator-only producer function, the optional Mem0 lookup of records the gate
never reads, or the Mem0 producer content template.

Following ``tests/test_recon_marker_kind_prompt_guidance.py``'s house rule,
these pin IDENTIFIERS an LLM would act on, never sentences. The kind value is
imported from the contract module rather than retyped.
"""

from __future__ import annotations

import re

import pytest

from fused_memory.reconciliation.flag_record_contract import STAGE1_FLAG_SUPPRESSION_KIND
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

_RETIRED_REMEDY_TOKENS = (
    'write_suppression_record',
    f'search(query="{STAGE1_FLAG_SUPPRESSION_KIND}',
    'STAGE 1 FLAG SUPPRESSION task_id=',
)


def _flattened(prompt: str) -> str:
    return re.sub(r'\s+', ' ', prompt)


@pytest.mark.parametrize('token', _RETIRED_REMEDY_TOKENS)
@pytest.mark.parametrize('prompt_name', sorted(_STAGE_PROMPTS))
def test_no_stage_prompt_carries_the_retired_remedy(prompt_name: str, token: str):
    assert token not in _flattened(_STAGE_PROMPTS[prompt_name])


@pytest.mark.parametrize('identifier', [STAGE1_FLAG_SUPPRESSION_KIND, 'recon_ledger'])
def test_stage1_still_names_the_code_gate(identifier: str):
    assert identifier in _flattened(STAGE1_SYSTEM_PROMPT)
