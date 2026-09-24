"""The review checkpoint's reflection guidance must reach the review agent's prompt.

``orchestrator.review_checkpoint.REFLECTION_INSTRUCTIONS`` is the "Reflect on
your findings" step of the deep-review prompt. It is public so that
``fused-memory/tests/test_referent_declaration_examples.py`` can import it and
certify its ``entities=`` example against the live entities_gate. That shape
check needs fused_memory; this file needs nothing from it, which is why the
splice is pinned here in the orchestrator suite.

Two structural properties are pinned, and no prose:

* the template's ``.format()`` fields are exactly ``project_id`` and
  ``review_id``;
* the assembled prompt contains it, formatted with live values. Without this,
  the certified constant could become prose no agent reads, the failure that
  ``test_memory_instructions_are_spliced_into_at_least_one_role`` guards
  against for role prompts.
"""

from __future__ import annotations

import string
from unittest.mock import MagicMock

from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.review_checkpoint import REFLECTION_INSTRUCTIONS, ReviewCheckpoint
from orchestrator.verify import VerifyResult


def test_reflection_template_takes_exactly_project_id_and_review_id():
    assert isinstance(REFLECTION_INSTRUCTIONS, str) and REFLECTION_INSTRUCTIONS.strip()

    fields = {
        field for _, field, _, _ in string.Formatter().parse(REFLECTION_INSTRUCTIONS) if field
    }

    assert fields == {'project_id', 'review_id'}


def test_assembled_review_prompt_carries_the_formatted_template(tmp_path):
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.project_root = tmp_path
    config.fused_memory.project_id = 'splice-proj'
    checkpoint = ReviewCheckpoint(config, mcp=MagicMock(), usage_gate=None)

    prompt = checkpoint._build_prompt(
        mode='focused',
        phase1=VerifyResult(
            passed=True, summary='OK', test_output='', lint_output='', type_output=''
        ),
        modules=['orchestrator'],
        briefing_content='',
        review_id='REV-TEST',
    )

    expected = REFLECTION_INSTRUCTIONS.format(
        project_id=config.fused_memory.project_id, review_id='REV-TEST'
    )
    assert expected in prompt, (
        'REFLECTION_INSTRUCTIONS no longer reaches the assembled review prompt, so '
        'the reflection guidance certified in the fused-memory suite is read by no agent.'
    )
