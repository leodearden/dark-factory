"""Contract tests for the declared ``kind`` on the recon stages' marker writes.

The Stage 1 ``flag_for_stage2`` marker and the Stage 2 ``stage2_suppress``
guard are written by LLM agents through the ``add_memory`` MCP tool, with the
metadata their stage prompt dictates, so the PROMPT is their writer. A
declared ``kind`` is what keeps such a write out of write triage
(``server/write_triage.py::declares_attach_keys``), so it is never filed as a
child of another memory.

Following ``tests/test_finding_provenance_prompt_guidance.py``'s house rule,
these pin the IDENTIFIERS an LLM writes to and no sentence around them. The
kind values are imported from the prompt package rather than retyped, so a
rename at the source cannot leave this test agreeing with itself.
"""

from __future__ import annotations

import re

import pytest

from fused_memory.memory_metadata import KIND_REGISTRY
from fused_memory.reconciliation.prompts import (
    FLAG_FOR_STAGE2_MARKER_KIND,
    STAGE2_SUPPRESS_GUARD_KIND,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)

_GUARD_WRITE_OPENING = "metadata={'stage2_suppress': True"
_COUNT_GATE_FILTER_OPENING = "{'task_id': str(task_id), 'stage2_suppress': True"
_COUNT_GATE_FILTER = _COUNT_GATE_FILTER_OPENING + '})'

_STAGE2_PROMPTS = {
    'STAGE2_SYSTEM_PROMPT': STAGE2_SYSTEM_PROMPT,
    "build_stage2_system_prompt('dark_factory')": build_stage2_system_prompt('dark_factory'),
}


def _flattened(prompt: str) -> str:
    return re.sub(r'\s+', ' ', prompt)


@pytest.mark.parametrize('kind', [FLAG_FOR_STAGE2_MARKER_KIND, STAGE2_SUPPRESS_GUARD_KIND])
def test_each_marker_kind_is_a_registered_kind(kind: str):
    assert kind in KIND_REGISTRY


def test_stage1_tells_the_flag_marker_writer_its_kind():
    assert f"`metadata.kind='{FLAG_FOR_STAGE2_MARKER_KIND}'`" in STAGE1_SYSTEM_PROMPT


@pytest.mark.parametrize('prompt_name', sorted(_STAGE2_PROMPTS))
def test_every_stage2_guard_write_declares_its_kind(prompt_name: str):
    prompt = _flattened(_STAGE2_PROMPTS[prompt_name])
    guard_writes = prompt.count(_GUARD_WRITE_OPENING)
    declared_kinds = prompt.count(f"'kind': '{STAGE2_SUPPRESS_GUARD_KIND}'")

    assert guard_writes >= 2, prompt_name
    assert declared_kinds == guard_writes, (
        f'{prompt_name}: {guard_writes} guard writes but {declared_kinds} declared kinds'
    )


@pytest.mark.parametrize('prompt_name', sorted(_STAGE2_PROMPTS))
def test_the_stage2_count_gate_filter_stays_kind_free(prompt_name: str):
    prompt = _flattened(_STAGE2_PROMPTS[prompt_name])

    assert _COUNT_GATE_FILTER in prompt, prompt_name
    assert prompt.count(_COUNT_GATE_FILTER_OPENING) == prompt.count(_COUNT_GATE_FILTER), (
        f'{prompt_name}: a count-gate filter carries a key beyond task_id and stage2_suppress'
    )
