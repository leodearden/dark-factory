"""Pins on the Live-Workflow reading rules as DELIVERED to the recon stage LLMs.

The rules are rendered by the module that owns the section's vocabulary
(``reconciliation/live_workflow_section.py::render_live_workflow_authority_rules``),
so these tests pin tokens taken from that module's constants and the
identifiers an agent must act on, never the prose wording.
"""

from __future__ import annotations

import pytest
from _live_workflow_region import live_workflow_region

from fused_memory.reconciliation.live_workflow_section import (
    CLAIMANT_FIELD,
    LIVE_WORKFLOW_SECTION_HEADER,
    PROJECT_LOCK_HELD,
    LiveSignal,
    LiveWorkflowRow,
    LiveWorkflowSnapshot,
    clipped_section_header,
    render_live_workflow_authority_rules,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
from fused_memory.reconciliation.task_filter import (
    ACTIVE_TASK_TREE_HEADER,
    FilteredTaskTree,
    format_filtered_task_tree,
)
from fused_memory.services.landed_on_main import LandingVerdict
from fused_memory.services.live_workflow_detector import ClaimantLabel


class TestLandedRuleInStagePrompts:
    def test_rules_are_delivered_once_to_each_stage_that_receives_the_section(self) -> None:
        rules = render_live_workflow_authority_rules()

        assert STAGE1_SYSTEM_PROMPT.count(rules) == 1
        assert STAGE2_SYSTEM_PROMPT.count(rules) == 1
        assert rules not in STAGE3_SYSTEM_PROMPT, (
            'Stage 3 never receives the Live-Workflow Signals section, so it must not get rules over it'
        )


class TestTieBreakerRule:
    @pytest.mark.parametrize('identifier', ['get_task', 'claimant_run_id', 'heartbeat_at'])
    def test_names_the_authoritative_check(self, identifier: str) -> None:
        assert identifier in render_live_workflow_authority_rules(), (
            'the tie-breaker must name the tool and both record fields that settle liveness'
        )

    @pytest.mark.parametrize('label', list(ClaimantLabel))
    def test_quotes_every_claimant_token_the_section_renders(self, label: ClaimantLabel) -> None:
        assert f'{CLAIMANT_FIELD}{label}' in render_live_workflow_authority_rules()

    def test_quotes_the_project_line_the_section_renders(self) -> None:
        assert PROJECT_LOCK_HELD in render_live_workflow_authority_rules()


def _snapshot(*, probed: int, total_active: int) -> LiveWorkflowSnapshot:
    row = LiveWorkflowRow(
        task_id='1',
        branch='task/1',
        is_live=True,
        signals=(LiveSignal.WORKTREE,),
        claimant=ClaimantLabel.NONE,
        landing=LandingVerdict.unknown(),
    )
    return LiveWorkflowSnapshot(
        rows=(row,), project_orchestrator_live=False, probed=probed, total_active=total_active,
    )


class TestProbedScopeRule:
    def test_a_clipped_render_opens_with_the_clipped_header(self) -> None:
        clipped_first_line = _snapshot(probed=2, total_active=5).render().splitlines()[0]
        whole_first_line = _snapshot(probed=5, total_active=5).render().splitlines()[0]

        assert clipped_first_line == clipped_section_header(2, 5)
        assert clipped_first_line.startswith(LIVE_WORKFLOW_SECTION_HEADER)
        assert whole_first_line == LIVE_WORKFLOW_SECTION_HEADER

    def test_quotes_the_clipped_header_the_section_renders(self) -> None:
        assert clipped_section_header('N', 'M') in render_live_workflow_authority_rules()

    def test_names_the_tree_that_bounds_the_probe(self) -> None:
        assert f'`{ACTIVE_TASK_TREE_HEADER}`' in render_live_workflow_authority_rules()

    def test_the_named_tree_is_the_header_the_tree_renders(self) -> None:
        tree = FilteredTaskTree(
            active_tasks=[{'id': 1, 'title': 't', 'status': 'pending'}],
            done_tasks=[],
            cancelled_tasks=[],
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=1,
            max_task_id=1,
        )

        assert format_filtered_task_tree(tree).startswith(ACTIVE_TASK_TREE_HEADER + '\n')


def _stage_local_live_workflow_text(prompt: str) -> str:
    """The stage's own `## Live-Workflow Authority` prose, without the shared rules block."""
    region = live_workflow_region(prompt)
    rules = render_live_workflow_authority_rules()
    assert region.count(rules) == 1, 'the shared rules must sit inside the region they are stripped from'
    return region.replace(rules, '')


@pytest.mark.parametrize(
    'prompt', [STAGE1_SYSTEM_PROMPT, STAGE2_SYSTEM_PROMPT], ids=['stage1', 'stage2'],
)
class TestStageAbsenceRulesAreScopedToTheTree:
    def test_both_absence_rules_name_the_tree(self, prompt: str) -> None:
        stage_local = _stage_local_live_workflow_text(prompt)

        assert stage_local.count(f'`{ACTIVE_TASK_TREE_HEADER}`') >= 2, (
            "the 'NO live signal ... no live row' rule and the 'If `### Live-Workflow "
            "Signals` is absent from the payload' rule must each be scoped to tasks the "
            f'`{ACTIVE_TASK_TREE_HEADER}` lists; either one left unqualified lets the '
            "model read 'never probed' as 'not live'"
        )
