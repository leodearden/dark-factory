"""The review checkpoint's reflection guidance must reach the review agent's prompt.

``orchestrator.review_checkpoint.render_reflection_instructions`` renders the
"Reflect on your findings" step of the deep-review prompt. It is public so that
``fused-memory/tests/test_referent_declaration_examples.py`` can import it and
certify its ``entities=`` example against the live entities_gate. That shape
check needs fused_memory; this file needs nothing from it, which is why the
splice is pinned here in the orchestrator suite.

One structural property is pinned, and no prose: the assembled prompt contains
the rendered guidance, rendered with live values. Without this, the certified
text could become prose no agent reads, the failure that
``test_memory_instructions_are_spliced_into_at_least_one_role`` guards against
for role prompts.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.review_checkpoint import ReviewCheckpoint, render_reflection_instructions
from orchestrator.verify import VerifyResult


def test_assembled_review_prompt_carries_the_rendered_reflection_instructions(tmp_path):
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

    expected = render_reflection_instructions(
        project_id=config.fused_memory.project_id, review_id='REV-TEST'
    )
    assert expected in prompt, (
        'render_reflection_instructions() no longer reaches the assembled review prompt, '
        'so the reflection guidance certified in the fused-memory suite is read by no agent.'
    )
