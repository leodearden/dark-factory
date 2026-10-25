"""Behaviour of ``reconciliation/live_workflow_citation_guard.py``.

The guard catches a Stage 1 finding that asserts a live-workflow signal its own
payload's ``### Live-Workflow Signals`` section does not carry (reify run
6aa50844 cited task/5891 as live while the section rendered nothing).
Extraction is deliberately conservative: every negative case below must yield
no citation, because a false annotation is itself a false signal.
"""

from __future__ import annotations

import logging

import pytest

from fused_memory.reconciliation.live_workflow_citation_guard import (
    LiveWorkflowCitation,
    check_live_workflow_citations,
    extract_live_workflow_citations,
)
from fused_memory.reconciliation.live_workflow_section import (
    LiveSignal,
    LiveWorkflowRow,
    LiveWorkflowSnapshot,
)
from fused_memory.services.landed_on_main import LandingEvidence, LandingVerdict
from fused_memory.services.live_workflow_detector import ClaimantLabel

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
            pytest.param(
                'Live-Workflow Signals list task/123 with landed=true, its worktree '
                'lingers after merge',
                id='the-section-name-is-not-live-language',
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


ANNOTATION = 'live_workflow_citation_contradictions'


def _row(
    task_id: str,
    *signals: LiveSignal,
    is_live: bool = True,
    landing: LandingVerdict | None = None,
) -> LiveWorkflowRow:
    return LiveWorkflowRow(
        task_id=task_id,
        branch=f'task/{task_id}',
        is_live=is_live,
        signals=signals,
        claimant=ClaimantLabel.NONE,
        landing=landing or LandingVerdict.not_landed(),
    )


def _snapshot(*rows: LiveWorkflowRow, lock_held: bool | None = False) -> LiveWorkflowSnapshot:
    return LiveWorkflowSnapshot(rows=rows, project_orchestrator_live=lock_held)


def _citing(task_id: str, *signals: str) -> dict:
    return {
        'task_id': task_id,
        'flag_type': 'task_stranded',
        'description': f'Live-Workflow Signals show task/{task_id} is live ({", ".join(signals)})',
    }


class _ExplodingFlag(dict):
    def get(self, key, default=None):
        raise RuntimeError('a flag whose accessor fails')


class TestCheckLiveWorkflowCitations:
    def test_the_incident_a_citation_against_an_empty_section_is_annotated(self):
        flag = {'task_id': '5891', 'description': INCIDENT}

        flags, contradictions = check_live_workflow_citations([flag], _snapshot())

        assert contradictions == 1
        assert flags[0] is not flag
        assert flags[0][ANNOTATION] == [
            {'task_id': '5891', 'cited_signals': ['orchestrator', 'worktree'], 'rendered_signals': None},
        ]

    def test_a_citation_matching_its_live_row_is_consistent(self):
        flag = _citing('6122', 'worktree', 'recent-commit')
        snapshot = _snapshot(_row('6122', LiveSignal.WORKTREE, LiveSignal.RECENT_COMMIT))

        flags, contradictions = check_live_workflow_citations([flag], snapshot)

        assert contradictions == 0
        assert ANNOTATION not in flags[0]

    def test_a_citation_beyond_its_rows_signals_is_annotated_with_what_was_rendered(self):
        flag = _citing('6122', 'worktree', 'recent-commit')
        snapshot = _snapshot(_row('6122', LiveSignal.WORKTREE))

        flags, contradictions = check_live_workflow_citations([flag], snapshot)

        assert contradictions == 1
        assert flags[0][ANNOTATION] == [
            {'task_id': '6122', 'cited_signals': ['recent-commit', 'worktree'], 'rendered_signals': ['worktree']},
        ]

    @pytest.mark.parametrize(
        ('lock_held', 'expected'),
        [
            pytest.param(True, 0, id='lock-held'),
            pytest.param(False, 1, id='lock-not-held'),
            pytest.param(None, 1, id='lock-unknown'),
        ],
    )
    def test_an_orchestrator_citation_for_a_live_row_needs_the_project_lock_held(
        self, lock_held, expected,
    ):
        snapshot = _snapshot(_row('6122', LiveSignal.WORKTREE), lock_held=lock_held)

        _, contradictions = check_live_workflow_citations(
            [_citing('6122', 'worktree', 'orchestrator')], snapshot,
        )

        assert contradictions == expected

    def test_any_liveness_citation_against_a_landed_only_row_is_annotated(self):
        landed = LandingVerdict.landed_by(LandingEvidence.MERGE_MARKER, 'a' * 40)
        snapshot = _snapshot(_row('7001', is_live=False, landing=landed), lock_held=True)

        flags, contradictions = check_live_workflow_citations(
            [_citing('7001', 'orchestrator')], snapshot,
        )

        assert contradictions == 1
        assert flags[0][ANNOTATION][0]['rendered_signals'] == []

    def test_no_snapshot_makes_the_guard_inert(self):
        flag = {'description': INCIDENT}

        flags, contradictions = check_live_workflow_citations([flag], None)

        assert contradictions == 0
        assert flags == [flag]
        assert flags[0] is flag

    def test_a_flag_with_no_citation_passes_through_as_the_same_object(self):
        flag = {'description': 'task/5891 needs attention'}

        flags, _ = check_live_workflow_citations([flag], _snapshot())

        assert flags[0] is flag

    def test_nothing_is_dropped_reordered_or_rewritten(self):
        inputs = [
            {'task_id': '1', 'description': 'unrelated finding'},
            {'task_id': '5891', 'description': INCIDENT, 'suggested_action': INCIDENT},
            {'task_id': '6122', 'description': 'task/6122 is live (worktree)'},
        ]
        originals = [dict(flag) for flag in inputs]
        snapshot = _snapshot(_row('6122', LiveSignal.WORKTREE))

        flags, contradictions = check_live_workflow_citations(inputs, snapshot)

        assert contradictions == 1
        assert [flag['task_id'] for flag in flags] == ['1', '5891', '6122']
        for flag, original in zip(flags, originals, strict=True):
            assert flag['description'] == original['description']
        assert inputs == originals

    def test_each_contradiction_is_logged_once_at_warning(self, caplog):
        flags = [_citing('5891', 'worktree'), _citing('6122', 'worktree', 'recent-commit')]
        snapshot = _snapshot(_row('6122', LiveSignal.WORKTREE))

        with caplog.at_level(logging.WARNING):
            _, contradictions = check_live_workflow_citations(flags, snapshot)

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert contradictions == 2
        assert len(warnings) == 2
        assert '5891' in warnings[0] and 'worktree' in warnings[0] and 'None' in warnings[0]
        assert '6122' in warnings[1] and 'recent-commit' in warnings[1]

    def test_a_flag_that_fails_extraction_passes_through_and_is_logged(self, caplog):
        flag = _ExplodingFlag(description=INCIDENT)

        with caplog.at_level(logging.WARNING):
            flags, contradictions = check_live_workflow_citations([flag], _snapshot())

        assert contradictions == 0
        assert flags[0] is flag
        assert any(r.levelno >= logging.WARNING for r in caplog.records)
