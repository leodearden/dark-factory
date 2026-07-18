"""Tests for the completion judge routed through the verdict-tools MCP
artifact — the artifact-only contract (post-task-η).

Task 2487 (PRD ``plans/mcp-verdict-servers-prd.md`` task η) closes the ζ
transition window: like the merger (task 2483 / PRD task γ) and reviewer
(task 2484 / PRD task δ), the judge now reads
``TaskArtifacts.read_verdict('judge')`` ONLY. The transition-window fallback
to ``result.structured_output`` is GONE, and
``TaskWorkflow._run_completion_judge`` no longer passes ``output_schema`` to
``_invoke``. An absent or malformed verdict artifact yields ``None``
(I-FAIL-SAFE: keep iterating, never a false completion) even when a legacy
``result.structured_output`` payload is present.

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

from orchestrator.agents.roles import JUDGE
from orchestrator.mcp.verdict_tools import _envelope


class TestJudgeGrantSurface:
    """Structural contract for the judge's verdict-tools grant (β/task 2482).

    A membership/contract check (not a docstring/prose pin) pinning the
    β-delivered grant the migrated prompt (step-6) now depends on — a
    future edit to JUDGE.allowed_tools can't silently drop the tool while
    the prompt still instructs the judge to call it. Mirrors
    test_reviewer_verdict_routing.py::TestReviewerGrantSurface. Already
    green (grant present via task 2482); included as a regression guard.
    """

    def test_has_verdict_tools_grant(self):
        assert 'mcp__verdict-tools__*' in JUDGE.allowed_tools

    def test_declares_verdict_tools_family(self):
        assert 'verdict_tools' in JUDGE.mcp_families


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
    """``_run_completion_judge`` reads the verdict artifact ONLY (post-η
    artifact-only contract); an absent/malformed artifact ⇒ ``None`` even when
    a legacy ``result.structured_output`` payload is present.
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

    async def test_no_artifact_returns_none(self, tmp_path: Path):
        """(b) no artifact written ⇒ None, even with a complete legacy
        structured_output present (post-η: the fallback is gone).
        """
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

        assert verdict is None

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

    async def test_stale_verdict_is_cleared_before_spawn(self, tmp_path: Path):
        """(d) a stale prior verdict must never masquerade as this run's (I-FRESH)."""
        f = self._setup(tmp_path)
        # Pre-seed a stale complete=True verdict, as if left over from a
        # prior _run_completion_judge invocation on this same worktree.
        f.artifacts.write_verdict(
            'judge',
            _envelope('judge', 'stale-sid', {
                'complete': True, 'reasoning': 'stale',
                'uncovered_plan_steps': [], 'substantive_work': True,
            }),
        )
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_with_judge_verdict(
                f, artifact=None, structured_output=None,
            ),
        )

        verdict = await f.wf._run_completion_judge([])

        # If the stale verdict had survived uncleared, its complete=True
        # would leak through instead.
        assert verdict is None

    async def test_malformed_artifact_returns_none(
        self, tmp_path: Path,
    ):
        """(e) an envelope present but missing a dict 'verdict' key ⇒ None,
        even with a complete legacy structured_output present (post-η: a
        malformed tool write is untrusted and no longer degrades to the
        structured_output fallback).
        """
        f = self._setup(tmp_path)

        def _invoke_writes_malformed(*args, **kwargs):
            f.artifacts.write_verdict('judge', {'role': 'judge', 'schema_version': 1})
            return AgentResult(
                success=True, output='fine',
                structured_output={
                    'complete': True, 'reasoning': 'legacy path',
                    'uncovered_plan_steps': [], 'substantive_work': True,
                },
            )

        f.wf._invoke = AsyncMock(side_effect=_invoke_writes_malformed)  # type: ignore[method-assign]

        verdict = await f.wf._run_completion_judge([])

        assert verdict is None

    async def test_run_completion_judge_does_not_pass_output_schema(
        self, tmp_path: Path,
    ):
        """(f) the judge no longer requests ``--json-schema`` structured output.

        Post-η the completion verdict arrives via the submit_completion_verdict
        tool + on-disk artifact, so ``_run_completion_judge`` must NOT pass
        ``output_schema`` to ``_invoke`` at all. Mirrors
        ``test_steward.py::test_pre_triage_invocation_uses_verdict_tools_mcp_config``
        (the ε/triage precedent).
        """
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_with_judge_verdict(
                f,
                artifact={
                    'complete': True, 'reasoning': 'r',
                    'uncovered_plan_steps': [], 'substantive_work': True,
                },
            ),
        )

        await f.wf._run_completion_judge([])

        assert 'output_schema' not in f.wf._invoke.call_args.kwargs
