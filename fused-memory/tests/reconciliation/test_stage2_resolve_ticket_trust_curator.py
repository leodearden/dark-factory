"""Prompt-pin test for resolve_ticket trust-curator rule in STAGE2_SYSTEM_PROMPT."""

from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT


def test_stage2_prompt_trusts_curator_on_resolve_ticket_success():
    assert 'increment `tasks_created` directly' in STAGE2_SYSTEM_PROMPT, (
        "STAGE2_SYSTEM_PROMPT must instruct agents to trust `resolve_ticket` "
        "`status='created'/'combined'` + `task_id` as authoritative success and "
        "increment `tasks_created` directly without a follow-up `get_task` round-trip "
        "(decision: drop the unconditional verification added in task-1082 step-6 — "
        "see plan analysis Part B)."
    )
