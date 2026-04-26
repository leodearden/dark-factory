"""Prompt-pin tests for set_task_status response shapes in STAGE2_SYSTEM_PROMPT."""

from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT


def test_stage2_prompt_documents_no_op_response_shape():
    assert 'no_op' in STAGE2_SYSTEM_PROMPT, (
        "STAGE2_SYSTEM_PROMPT must instruct agents to treat `no_op: True` "
        "set_task_status responses as successful no-ops, not discrepancies."
    )
