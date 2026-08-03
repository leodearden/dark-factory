"""Wiring tests for the Task-Creation Accounting prompt contract (task 3046).

Stage 2 self-reports ``tasks_created`` as a bare LLM counter with no code-side
ground truth: ``submit_task``/``resolve_ticket`` are not journaled (see
``fused_memory.middleware.task_interceptor.TaskInterceptor._journal_around``,
which wraps only ``set_task_status``/``update_task``/``remove_tasks``/
``add_dependency``/``remove_dependency``), so ``derive_stage_stats`` cannot
recompute it and ``stats_verifier`` leaves it untouched (it is not in
``_COMPUTED_STAT_KEYS``). Run 507bc25b filed task 3045 via the Proactive Task
Sample / cross-project-routing surface yet Stage 2 self-reported
``tasks_created: 0`` — the increment mandate (``## Verifying Task
Operations``, ``prompts/stage2.py``) never declared the counter
path-agnostic, and (unlike the flag counters) had no action-record ground
truth analogous to ``flag_deleted_records``.

This module pins the WIRING of the fix's shared renderer,
``render_task_creation_accounting_section()`` (``recon_self_model.py``) —
exactly once per assembled Stage 2 prompt, on both
``build_stage2_system_prompt`` branches, never colliding with the
``'## Available Tools'`` injection sentinel or the ``mcp__recon-report__``
run_id drift guard (``test_recon_report_guidance_drift.py``) — and the
machine-readable contract tokens the Python side actually reads
(``tasks_created``, ``task_created_records``, the
``"action": "task_created"`` record shape). Per the
``test_recon_gate_closure_guidance.py`` house rule, prose wording beyond
those tokens is intentionally NOT pinned here.
"""

from __future__ import annotations

import pytest

from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)
from fused_memory.reconciliation.recon_self_model import (
    render_task_creation_accounting_section,
)


class TestTaskCreationAccountingWiring:
    """render_task_creation_accounting_section() is wired into Stage 2 exactly once."""

    def test_embedded_verbatim_exactly_once_in_stage2_prompt(self):
        section = render_task_creation_accounting_section()

        # prompt.count(section) == 1 subsumes non-emptiness: str.count('')
        # over a non-empty STAGE2_SYSTEM_PROMPT is len(prompt) + 1, never 1.
        assert STAGE2_SYSTEM_PROMPT.count(section) == 1, (
            'STAGE2_SYSTEM_PROMPT must interpolate '
            'render_task_creation_accounting_section() verbatim, exactly once.'
        )

    @pytest.mark.parametrize('project_id', ['dark_factory', 'autopilot_video'])
    def test_survives_build_stage2_system_prompt(self, project_id: str):
        """Both runtime builder branches keep the section, exactly once.

        `autopilot_video` splices its contamination guardrail at the
        `'## Available Tools'` sentinel; that injection must not drop or
        duplicate this section.
        """
        section = render_task_creation_accounting_section()
        built = build_stage2_system_prompt(project_id)

        assert built.count(section) == 1, (
            f'build_stage2_system_prompt({project_id!r}) must carry '
            'render_task_creation_accounting_section() verbatim, exactly once.'
        )

    def test_does_not_contain_available_tools_sentinel(self):
        """Must not carry the injection sentinel build_stage2_system_prompt
        requires to appear exactly once in STAGE2_SYSTEM_PROMPT."""
        section = render_task_creation_accounting_section()

        assert '## Available Tools' not in section, (
            'the rendered section must not contain the "## Available Tools" '
            'sentinel — build_stage2_system_prompt raises RuntimeError unless '
            'that sentinel appears exactly once in STAGE2_SYSTEM_PROMPT.'
        )

    def test_does_not_contain_recon_report_call_examples(self):
        """No mcp__recon-report__ example — keeps this section trivially
        compatible with test_recon_report_guidance_drift.py's run_id= guard."""
        section = render_task_creation_accounting_section()

        assert 'mcp__recon-report__' not in section

    def test_assembled_prompt_carries_machine_readable_contract_tokens(self):
        """The Python side reads tasks_created / task_created_records / the
        "action": "task_created" record shape — pin those tokens, not prose."""
        built = build_stage2_system_prompt('dark_factory')

        assert 'tasks_created' in built
        assert 'task_created_records' in built
        assert '"action": "task_created"' in built

    def test_task_created_records_declared_in_per_cycle_counter_schema(self):
        """task_created_records must be enumerated in the '## Per-Cycle
        Counter Schema' bullet list, not merely mentioned elsewhere."""
        built = build_stage2_system_prompt('dark_factory')
        marker = '**Per-Cycle Counter Schema**'

        start = built.find(marker)
        assert start != -1, f'{marker!r} not found in the assembled Stage 2 prompt.'
        end = built.find('\n##', start + 1)
        if end == -1:
            end = len(built)
        schema_block = built[start:end]

        assert 'task_created_records' in schema_block, (
            'task_created_records must be declared as a bullet in the '
            '"## Per-Cycle Counter Schema" block.'
        )
