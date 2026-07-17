"""RED/GREEN tests for the completion judge routed through the verdict-tools
MCP artifact, with a transition-window fallback to ``result.structured_output``.

Task 2486 (PRD ``plans/mcp-verdict-servers-prd.md`` task ζ): unlike the
merger (task 2483 / PRD task γ) and reviewer (task 2484 / PRD task δ), which
cut over to an artifact-only read, the judge is completion-gating and this
orchestrator self-hosts the PRD's own tasks — a botched cutover that
silently returned "not complete" would wedge completion. So
``TaskWorkflow._run_completion_judge`` PREFERS
``TaskArtifacts.read_verdict('judge')`` but FALLS BACK to
``result.structured_output`` when the artifact is absent or malformed. Task
η removes the fallback after real-run verification.

Driven directly through the ``_workflow_helpers._make`` factory (real
on-disk ``TaskArtifacts``), mirroring ``test_merger_disposition_verdict.py``
and ``test_reviewer_verdict_routing.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _workflow_helpers import _make
from shared.cli_invoke import AgentResult

from orchestrator.mcp.verdict_tools import _envelope


def _invoke_with_judge_verdict(
    f, *, artifact: dict | None = None, structured_output: dict | None = None,
    success: bool = True, output: str = '',
) -> Callable:
    """Build an ``_invoke`` side_effect that optionally writes a judge verdict
    artifact and/or returns a legacy ``structured_output`` payload.

    ``artifact=None`` writes no verdict artifact at all — simulating a judge
    that never called ``submit_completion_verdict`` (or a pre-ζ judge relying
    solely on the ``--json-schema`` contract).
    """

    def _side_effect(*args, **kwargs):
        if artifact is not None:
            f.artifacts.write_verdict('judge', _envelope('judge', 'sid', artifact))
        return AgentResult(success=success, output=output, structured_output=structured_output)

    return _side_effect


@pytest.mark.asyncio
class TestRunCompletionJudgeVerdictRouting:
    """``_run_completion_judge`` prefers the verdict artifact, falling back to
    ``result.structured_output`` during the transition window.
    """

    def _setup(self, tmp_path: Path):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf.briefing.build_completion_judge_prompt = AsyncMock(return_value='prompt')
        f.wf.git_ops.get_diff_from_base = AsyncMock(
            return_value='diff --git a/x b/x\n+line',
        )
        return f

    async def test_artifact_verdict_wins_over_structured_output(self, tmp_path: Path):
        """(a) a written artifact is preferred over a disagreeing structured_output.

        This is the "arrived via the tool" precedence signal: the artifact
        says complete=True while the (stale/disagreeing) structured_output
        says complete=False — the artifact must win.
        """
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_with_judge_verdict(
                f,
                artifact={
                    'complete': True, 'reasoning': 'r',
                    'uncovered_plan_steps': [], 'substantive_work': True,
                },
                structured_output={
                    'complete': False, 'reasoning': 'stale legacy path',
                    'uncovered_plan_steps': ['1'], 'substantive_work': False,
                },
            ),
        )

        verdict = await f.wf._run_completion_judge([])

        assert verdict == {
            'complete': True, 'reasoning': 'r',
            'uncovered_plan_steps': [], 'substantive_work': True,
        }

    async def test_structured_output_fallback_when_no_artifact(self, tmp_path: Path):
        """(b) no artifact written => falls back to structured_output (transition window)."""
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_with_judge_verdict(
                f,
                artifact=None,
                structured_output={
                    'complete': True, 'reasoning': 'legacy path',
                    'uncovered_plan_steps': [], 'substantive_work': True,
                },
            ),
        )

        verdict = await f.wf._run_completion_judge([])

        assert verdict == {
            'complete': True, 'reasoning': 'legacy path',
            'uncovered_plan_steps': [], 'substantive_work': True,
        }

    async def test_absent_both_returns_none(self, tmp_path: Path):
        """(c) no artifact and no structured_output => None (fail-safe, keep iterating)."""
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_with_judge_verdict(
                f, artifact=None, structured_output=None,
            ),
        )

        verdict = await f.wf._run_completion_judge([])

        assert verdict is None
