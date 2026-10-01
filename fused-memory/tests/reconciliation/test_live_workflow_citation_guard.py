"""Behaviour of ``reconciliation/live_workflow_citation_guard.py``.

The guard catches a Stage 1 finding that asserts a live-workflow signal its own
payload's ``### Live-Workflow Signals`` section does not carry (reify run
6aa50844 cited task/5891 as live while the section rendered nothing).
Extraction is deliberately conservative: every negative case below must yield
no citation, because a false annotation is itself a false signal.
"""

from __future__ import annotations

import pytest

from fused_memory.reconciliation.live_workflow_citation_guard import (
    LiveWorkflowCitation,
    extract_live_workflow_citations,
)
from fused_memory.reconciliation.live_workflow_section import LiveSignal

INCIDENT = (
    'Live-Workflow Signals show task/5891 is live (worktree, orchestrator); '
    'deferring disposition'
)


class TestExtractLiveWorkflowCitations:
    def test_the_incident_finding_cites_worktree_and_orchestrator_for_5891(self):
        citations = extract_live_workflow_citations({'description': INCIDENT})

        assert citations == (
            LiveWorkflowCitation(
                '5891', frozenset({LiveSignal.WORKTREE, LiveSignal.ORCHESTRATOR}),
            ),
        )

    def test_a_citation_in_suggested_action_is_caught(self):
        citations = extract_live_workflow_citations(
            {'description': 'Task looks stranded', 'suggested_action': INCIDENT},
        )

        assert citations == (
            LiveWorkflowCitation(
                '5891', frozenset({LiveSignal.WORKTREE, LiveSignal.ORCHESTRATOR}),
            ),
        )

    def test_a_citation_in_the_structured_json_content_key_is_caught(self):
        citations = extract_live_workflow_citations({'content': INCIDENT})

        assert [c.task_id for c in citations] == ['5891']

    def test_the_same_citation_in_two_fields_is_one_citation(self):
        citations = extract_live_workflow_citations(
            {'description': INCIDENT, 'suggested_action': INCIDENT},
        )

        assert len(citations) == 1

    def test_recent_commit_spelled_with_a_space_maps_to_recent_commit(self):
        citations = extract_live_workflow_citations(
            {'description': 'task/6122 is live: it has a recent commit'},
        )

        assert citations == (
            LiveWorkflowCitation('6122', frozenset({LiveSignal.RECENT_COMMIT})),
        )

    def test_the_rendered_hyphenated_spelling_maps_to_recent_commit(self):
        citations = extract_live_workflow_citations(
            {'description': 'task/6122 is live (recent-commit)'},
        )

        assert citations == (
            LiveWorkflowCitation('6122', frozenset({LiveSignal.RECENT_COMMIT})),
        )

    def test_matching_is_case_insensitive(self):
        citations = extract_live_workflow_citations(
            {'description': 'TASK/5891 is LIVE (WORKTREE)'},
        )

        assert citations == (LiveWorkflowCitation('5891', frozenset({LiveSignal.WORKTREE})),)

    def test_citations_of_different_tasks_in_different_sentences_are_kept_apart(self):
        citations = extract_live_workflow_citations(
            {'description': 'task/5891 is live (worktree). task/6122 is live (orchestrator).'},
        )

        assert citations == (
            LiveWorkflowCitation('5891', frozenset({LiveSignal.WORKTREE})),
            LiveWorkflowCitation('6122', frozenset({LiveSignal.ORCHESTRATOR})),
        )

    @pytest.mark.parametrize(
        'description',
        [
            pytest.param('task/5891 needs attention', id='bare-task-ref'),
            pytest.param('the worktree is live and busy', id='signal-without-task-ref'),
            pytest.param(
                'cleaned up the worktree for task/5891 after merge', id='no-live-language',
            ),
            pytest.param(
                'task/5891 has no worktree and no orchestrator signal, so it is not live',
                id='negated-in-the-same-window',
            ),
            pytest.param('task/5891 is live without a worktree', id='without-cue'),
            pytest.param('task/5891 is live; worktree absent', id='signal-in-another-window'),
            pytest.param(
                'task/5891 is live\nworktree registered', id='newline-splits-windows',
            ),
            pytest.param(
                'task/5891 is live. The worktree is registered', id='period-splits-windows',
            ),
            pytest.param('task/5891 is live with a worktrees list', id='plural-is-not-a-token'),
            pytest.param('task/5891 has liveness via worktree', id='liveness-is-not-live'),
            pytest.param(
                'task/5891 and task/6122 are live (worktree)', id='ambiguous-two-tasks',
            ),
        ],
    )
    def test_conservative_negatives_yield_no_citation(self, description):
        assert extract_live_workflow_citations({'description': description}) == ()

    @pytest.mark.parametrize(
        'flag',
        [
            pytest.param(None, id='none'),
            pytest.param(INCIDENT, id='a-bare-string'),
            pytest.param([INCIDENT], id='a-list'),
            pytest.param({}, id='empty-dict'),
            pytest.param(
                {'description': None, 'suggested_action': None, 'content': None},
                id='none-fields',
            ),
            pytest.param(
                {'description': 5891, 'suggested_action': ['x'], 'content': {'text': INCIDENT}},
                id='non-str-fields',
            ),
            pytest.param({'flag_type': INCIDENT, 'task_id': INCIDENT}, id='unscanned-fields'),
        ],
    )
    def test_malformed_or_unscanned_input_yields_no_citation(self, flag):
        assert extract_live_workflow_citations(flag) == ()
