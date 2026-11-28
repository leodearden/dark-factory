"""An AgentStub whose first review round blocks, then replans past the block.

The comprehensive reviewer reports one blocking issue in round one. The
architect's replan adds step-3, the implementer lands it, and every later
review round passes. The blocking issue is a constructor argument, so a
scenario can choose what kind of finding forces the replan.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from _workflow_helpers import AgentStub, _make_review, agent_side_artifacts

from orchestrator.agents.invoke import AgentResult
from orchestrator.git_ops import _run

DEFAULT_BLOCKING_ISSUE: Mapping[str, str] = {
    'severity': 'blocking',
    'kind': 'test_quality',
    'location': 'lib.py:5',
    'category': 'missing_edge_case',
    'description': 'No test for empty name',
    'suggested_fix': 'Add test_farewell_empty',
}


class ReplanningReviewStub(AgentStub):
    """The comprehensive reviewer blocks the first round; the replan adds step-3, which the implementer lands."""

    def __init__(self, *, blocking_issue: Mapping[str, str] = DEFAULT_BLOCKING_ISSUE) -> None:
        super().__init__()
        self.replans = 0
        self._blocking_issue = dict(blocking_issue)

    def _reviewer(self, role: str, output_schema: dict | None) -> AgentResult:
        if self.replans == 0 and role == 'reviewer_comprehensive':
            review = _make_review(role, 'ISSUES_FOUND', [dict(self._blocking_issue)])
            return AgentResult(
                success=True, output=json.dumps(review),
                structured_output=review, cost_usd=0.10,
            )
        review = _make_review(role)
        return AgentResult(
            success=True, output=json.dumps(review), structured_output=review, cost_usd=0.10,
        )

    async def _architect(self, cwd: Path) -> AgentResult:
        plan_path = agent_side_artifacts(cwd).artifact_path('plan.json')
        if not plan_path.exists():
            return await super()._architect(cwd)
        plan = json.loads(plan_path.read_text())
        plan['steps'].append({
            'id': 'step-3', 'type': 'test', 'description': 'Add test for empty name',
            'status': 'pending', 'commit': None,
        })
        plan_path.write_text(json.dumps(plan, indent=2) + '\n')
        self.replans += 1
        return AgentResult(success=True, output='Plan updated', cost_usd=0.40)

    async def _implementer(self, cwd: Path) -> AgentResult:
        artifacts = agent_side_artifacts(cwd)
        if not any(step['id'] == 'step-3' for step in artifacts.get_pending_steps()):
            return await super()._implementer(cwd)
        test_file = cwd / 'test_lib.py'
        test_file.write_text(
            test_file.read_text()
            + '\ndef test_farewell_empty():\n    assert farewell("") == "Goodbye, "\n'
        )
        await _run(['git', 'add', '-A'], cwd=cwd)
        await _run(['git', 'commit', '-m', 'Complete step-3'], cwd=cwd)
        _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=cwd)
        artifacts.update_step_status('step-3', 'done', commit=sha)
        return AgentResult(success=True, output='Completed step-3', cost_usd=1.00)
