"""RED/GREEN tests for merger disposition routed through the verdict tool.

Task 2483 (PRD ``plans/mcp-verdict-servers-prd.md`` task γ): replaces the
fragile prose grep ``'BLOCKED' in merger_result.output.upper()`` in
``TaskWorkflow._resolve_and_resubmit`` with a structured verdict-tool read
(``TaskArtifacts.read_verdict('merger')``), mirroring the steward
pre-triage clear→spawn→read→fail-safe pattern (steward.py:721-790) and the
defensive-extraction philosophy of ``extract_triage_verdict``
(agents/triage.py:303) — an absent or malformed verdict is treated as
untrusted and maps to the same blocked-equivalent outcome as an explicit
``blocked=True``.

All six cases below drive ``_resolve_and_resubmit`` directly through the
``_workflow_helpers._make`` factory (real on-disk ``TaskArtifacts``, so
``write_verdict``/``read_verdict``/``clear_verdict`` hit actual files).
Cases (a)-(e) fail against the pre-γ substring-grep implementation — it
never reads ``verdicts/merger.json`` at all, so a verdict's ``blocked``
value cannot influence routing, and stray "BLOCKED" prose in the agent's
free-text output is (wrongly) load-bearing. Case (f) preserves the
existing "invocation failed ⇒ blocked" guard, which already passes
pre-γ (included as a regression guard, not a new behavior).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _workflow_helpers import _make

from orchestrator.mcp.verdict_tools import _envelope
from orchestrator.workflow import WorkflowOutcome
from shared.cli_invoke import AgentResult


def _invoke_with_verdict(
    f, *, blocked: bool | None, reason: str = '', output: str = '', success: bool = True,
) -> Callable:
    """Build an ``_invoke`` side_effect that optionally writes a merger verdict.

    ``blocked=None`` writes no verdict at all — simulating a merger agent
    that never called ``submit_merge_disposition``.
    """

    def _side_effect(*args, **kwargs):
        if blocked is not None:
            f.artifacts.write_verdict(
                'merger',
                _envelope('merger', 'sid', {'blocked': blocked, 'reason': reason}),
            )
        return AgentResult(success=success, output=output)

    return _side_effect


@pytest.mark.asyncio
class TestResolveAndResubmitVerdictRouting:
    """``_resolve_and_resubmit`` reads the merger's verdict instead of grepping prose."""

    def _setup(self, tmp_path: Path):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf.git_ops.rebase_onto_main = AsyncMock()
        f.wf.briefing.build_merger_prompt = AsyncMock(return_value='prompt')
        f.wf._submit_to_merge_queue = AsyncMock(  # type: ignore[method-assign]
            return_value=WorkflowOutcome.DONE,
        )
        f.wf._mark_blocked = AsyncMock(  # type: ignore[method-assign]
            return_value=WorkflowOutcome.BLOCKED,
        )
        f.wf._write_merge_failure_review = MagicMock()  # type: ignore[method-assign]
        return f

    async def test_blocked_false_verdict_proceeds_despite_blocked_prose(
        self, tmp_path: Path,
    ):
        """(a) verdict blocked=False wins over stray 'BLOCKED' prose in output.

        This is the user-observable win: a merger that says "BLOCKED" in
        prose but submits ``blocked=False`` now PROCEEDS instead of being
        false-blocked by the grep.
        """
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_with_verdict(
                f, blocked=False, output='Resolved cleanly. Not BLOCKED at all.',
            ),
        )

        outcome = await f.wf._resolve_and_resubmit('B', 'conflict_X', merge_phase=True)

        f.wf._submit_to_merge_queue.assert_awaited_once()
        f.wf._mark_blocked.assert_not_awaited()
        assert outcome == WorkflowOutcome.DONE

    async def test_blocked_true_verdict_blocks_with_reason(self, tmp_path: Path):
        """(b) verdict blocked=True routes to _mark_blocked with its reason."""
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_with_verdict(
                f, blocked=True, reason='ARCH mismatch', output='Looks fine.',
            ),
        )

        outcome = await f.wf._resolve_and_resubmit('B', 'conflict_X', merge_phase=True)

        f.wf._mark_blocked.assert_awaited_once()
        args, _kwargs = f.wf._mark_blocked.await_args
        assert 'ARCH' in args[0]
        f.wf._submit_to_merge_queue.assert_not_awaited()
        assert outcome == WorkflowOutcome.BLOCKED

    async def test_absent_verdict_is_failsafe_blocked(self, tmp_path: Path):
        """(c) no verdict written at all => fail-safe blocked-equivalent."""
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_with_verdict(
                f, blocked=None, output='no disposition emitted',
            ),
        )

        outcome = await f.wf._resolve_and_resubmit('B', 'conflict_X', merge_phase=True)

        f.wf._mark_blocked.assert_awaited_once()
        f.wf._submit_to_merge_queue.assert_not_awaited()
        assert outcome == WorkflowOutcome.BLOCKED

    async def test_stale_verdict_is_cleared_before_spawn(self, tmp_path: Path):
        """(d) a stale prior verdict must never masquerade as this run's (I-FRESH)."""
        f = self._setup(tmp_path)
        # Pre-seed a stale blocked=False verdict, as if left over from a
        # prior _resolve_and_resubmit invocation on this same worktree.
        f.artifacts.write_verdict(
            'merger',
            _envelope('merger', 'stale-sid', {'blocked': False, 'reason': ''}),
        )
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_with_verdict(f, blocked=None, output='nothing new'),
        )

        outcome = await f.wf._resolve_and_resubmit('B', 'conflict_X', merge_phase=True)

        # If the stale verdict had survived uncleared, its blocked=False
        # would let this proceed to _submit_to_merge_queue instead.
        f.wf._mark_blocked.assert_awaited_once()
        f.wf._submit_to_merge_queue.assert_not_awaited()
        assert outcome == WorkflowOutcome.BLOCKED

    async def test_malformed_envelope_is_failsafe_blocked(self, tmp_path: Path):
        """(e) an envelope present but missing the 'verdict' key is untrusted."""
        f = self._setup(tmp_path)

        def _invoke_writes_malformed(*args, **kwargs):
            f.artifacts.write_verdict('merger', {'role': 'merger', 'schema_version': 1})
            return AgentResult(success=True, output='fine')

        f.wf._invoke = AsyncMock(side_effect=_invoke_writes_malformed)  # type: ignore[method-assign]

        outcome = await f.wf._resolve_and_resubmit('B', 'conflict_X', merge_phase=True)

        f.wf._mark_blocked.assert_awaited_once()
        f.wf._submit_to_merge_queue.assert_not_awaited()
        assert outcome == WorkflowOutcome.BLOCKED

    async def test_invocation_failure_still_blocks(self, tmp_path: Path):
        """(f) _invoke success=False blocks regardless of verdict (preservation guard).

        May already pass pre-γ — the existing ``not merger_result.success``
        guard is preserved as the first blocked condition.
        """
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_with_verdict(
                f, blocked=False, output='ok', success=False,
            ),
        )

        outcome = await f.wf._resolve_and_resubmit('B', 'conflict_X', merge_phase=True)

        f.wf._mark_blocked.assert_awaited_once()
        f.wf._submit_to_merge_queue.assert_not_awaited()
        assert outcome == WorkflowOutcome.BLOCKED
