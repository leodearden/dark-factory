"""A re-dispatch after a branch-swapping quarantine reviews its new branch afresh (task 3663).

The 3404/2432 shape, end to end through a real ``TaskWorkflow.run()``: a
quarantine renamed ``task/<id>`` aside after an amendment round, and every
review record survived in the task root. Dispatch setup must retire that
lineage, so the new branch gets its own amendment round, the task-2523
adjudicator is not run against suggestions settled on the parked branch, and
the architect is briefed with them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _info_l0_router_fixtures import TASK_ID
from _info_l0_router_fixtures import assignment as _assignment
from _workflow_helpers import PLAN, AgentStub, FakeBriefing, _make_review
from shared.cli_invoke import AgentResult
from test_workflow_escalation_rulings import _WorkflowRun

from orchestrator.agents.review_briefing import ReviewBriefing
from orchestrator.artifacts import TaskArtifacts
from orchestrator.escalation_rulings import Ruling
from orchestrator.quarantine_provenance import stamp_quarantined_from
from orchestrator.review_lineage import RetiredReview
from orchestrator.review_suggestions.prior_round import load_prior_round_suggestions
from orchestrator.workflow import WorkflowOutcome

_PARKING_NAME = f'{TASK_ID}-20261112T000000Z'
_ADJUDICATOR = 'resettled_adjudicator'
_SETTLED = {
    'severity': 'suggestion',
    'kind': 'process',
    'location': 'lib.py:3',
    'category': 'naming',
    'description': 'helper name hides its side effect',
    'suggested_fix': 'rename helper',
}
_RAISED_ON_THE_NEW_BRANCH = {
    'severity': 'suggestion',
    'kind': 'test_quality',
    'location': 'lib.py:1',
    'category': 'coverage',
    'description': 'farewell has no empty-name case',
    'suggested_fix': 'add a test for farewell("")',
}


class _LineageRecordingBriefing(FakeBriefing):
    """Canned prompts that also record what the architect carried and what each amender pass was handed."""

    def __init__(self) -> None:
        self.carried: list[list[str]] = []
        self.amended: list[list[str]] = []

    async def build_architect_prompt(
        self, task: dict, worktree=None, context: str | None = None,
        *, include_prior_proposals: bool = False,
        committed_work: list[dict] | None = None,
        rulings: Sequence[Ruling] = (),
        retired_reviews: Sequence[RetiredReview] = (),
    ) -> str:
        self.carried.append([s['description'] for review in retired_reviews for s in review.suggestions])
        return await super().build_architect_prompt(task, worktree, context)

    async def build_amender_prompt(
        self, plan: dict, suggestions: list, locked_modules: list,
        context: str | None = None, task_id: str | None = None,
        *, rulings: Sequence[Ruling] = (),
        scope_drift=None,
    ) -> str:
        self.amended.append([s['description'] for s in suggestions])
        return await super().build_amender_prompt(plan, suggestions, locked_modules, context, task_id)


class _OneSuggestionReviewerStub(AgentStub):
    """Each reviewer raises one in-scope suggestion on its first review; the task-2523 adjudicator is told apart.

    The adjudicator's system prompt says it is "not a code reviewer", which
    the base stub would read as a reviewer.
    """

    def __init__(self) -> None:
        super().__init__()
        self._reviewed: set[str] = set()

    def _detect_role(self, system_prompt: str) -> str:
        if 'CLASSIFICATION JUDGE' in system_prompt:
            return _ADJUDICATOR
        return super()._detect_role(system_prompt)

    def _reviewer(self, role: str, output_schema: dict | None) -> AgentResult:
        if role in self._reviewed:
            return super()._reviewer(role, output_schema)
        self._reviewed.add(role)
        review = _make_review(role, 'ISSUES_FOUND', [dict(_RAISED_ON_THE_NEW_BRANCH)])
        return AgentResult(success=True, output=json.dumps(review), structured_output=review, cost_usd=0.10)


class _RedispatchRun(_WorkflowRun[_LineageRecordingBriefing]):
    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.stub = _OneSuggestionReviewerStub()
        super().__init__(
            tmp_path, monkeypatch, self.stub, _LineageRecordingBriefing(),
            replace(_assignment(), modules=list(PLAN['files'])),
            briefing=ReviewBriefing.DIFF_ONLY, router_enabled=True,
        )
        worktree = self.workflow.git_ops.worktree_base / TASK_ID
        self.artifacts = TaskArtifacts.for_task(worktree, TASK_ID)

    def seed_a_parked_lineage(self, *, stamped: bool) -> None:
        """One amendment round answered on the parked branch, and, when *stamped*, the quarantine that parked it."""
        self.artifacts.init(TASK_ID, 'X', 'Y')
        self.artifacts.write_plan({'task_id': TASK_ID, 'title': 'X', 'steps': []})
        self.artifacts.set_review_counters(amendment_rounds_total=1)
        archive = self.artifacts.review_archive_dir('amend-1')
        archive.mkdir(parents=True)
        (archive / 'analyst.json').write_text(json.dumps({'verdict': 'ISSUES_FOUND', 'issues': [_SETTLED]}))
        if stamped:
            assert stamp_quarantined_from(self.artifacts, {
                'branch': f'refs/heads/task/{_PARKING_NAME}',
                'tip_sha': 'b' * 40,
                'unique_commit_count': 3,
                'quarantined_at': datetime.now(UTC).isoformat(),
                'reason': 'orphan-reaper',
                'parking': f'/q/{_PARKING_NAME}',
                'stored_title': None,
            })

    def current_round_descriptions(self) -> list[str]:
        return [s['description'] for s in load_prior_round_suggestions(self.artifacts.review_archives_root)]


@pytest.mark.asyncio
class TestRedispatchAfterQuarantine:
    async def test_the_new_branch_gets_its_own_amendment_round(self, tmp_path: Path, monkeypatch):
        run = _RedispatchRun(tmp_path, monkeypatch)
        run.seed_a_parked_lineage(stamped=True)

        assert await run.run() == WorkflowOutcome.DONE

        assert run.briefing.amended
        assert all(handed == [_RAISED_ON_THE_NEW_BRANCH['description']] for handed in run.briefing.amended)
        assert _ADJUDICATOR not in run.stub.calls

    async def test_the_architect_is_briefed_with_the_parked_branch_suggestions(self, tmp_path: Path, monkeypatch):
        run = _RedispatchRun(tmp_path, monkeypatch)
        run.seed_a_parked_lineage(stamped=True)

        assert await run.run() == WorkflowOutcome.DONE

        assert run.briefing.carried == [[_SETTLED['description']]]

    async def test_the_new_branch_archives_its_own_amendment_round(self, tmp_path: Path, monkeypatch):
        run = _RedispatchRun(tmp_path, monkeypatch)
        run.seed_a_parked_lineage(stamped=True)

        assert await run.run() == WorkflowOutcome.DONE

        assert set(run.current_round_descriptions()) == {_RAISED_ON_THE_NEW_BRANCH['description']}

    async def test_without_a_stamp_the_parked_lineage_still_governs_the_review(self, tmp_path: Path, monkeypatch):
        run = _RedispatchRun(tmp_path, monkeypatch)
        run.seed_a_parked_lineage(stamped=False)

        assert await run.run() == WorkflowOutcome.DONE

        assert run.briefing.amended == []
        assert _ADJUDICATOR in run.stub.calls
        assert run.briefing.carried == [[]]
        assert run.current_round_descriptions() == [_SETTLED['description']]
