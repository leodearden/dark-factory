"""RED/GREEN tests for reviewer verdicts routed through the verdict-tools
MCP artifact instead of the structured-output/``json.loads`` cascade.

Task 2484 (PRD ``plans/mcp-verdict-servers-prd.md`` task δ): removes the
reviewer panel's ``output_schema``/``json.loads`` fallback in
``TaskWorkflow._run_reviewer`` in favor of reading
``TaskArtifacts.read_verdict(role.name)`` — the same
clear→invoke→read→defensive-extract→fail-safe shape task 2483 (PRD task γ)
established for the merger (see ``test_merger_disposition_verdict.py``) — and
removes the reviewer's now-inert ``mcp__jcodemunch__*`` grant (fold-in κ).

``TestRunReviewerVerdictRouting`` cases (a)-(b) (PASS / ISSUES_FOUND payloads
returned verbatim) fail against the pre-δ structured_output/json.loads
cascade — it never reads ``verdicts/reviewer_comprehensive.json``, so a
written verdict cannot be returned and the unparseable prose output falls
through to the synthesized ERROR fallback instead. Cases (c)-(e) (absent /
stale / malformed verdict => ERROR) already hold pre-δ, by coincidence —
the old cascade also defaults to ERROR on unparseable/absent output — and
are included as fail-safe/I-FRESH regression guards, not new RED behavior
(mirrors the merger test's case (f)). Cases (f1)-(f3) pin the *narrowed*
I-FAIL-SAFE contract for an unsuccessful invocation. fe37ca04a8 (task 2484
amendment pass) made ``not result.success`` a short-circuit disjunct ahead
of the payload check, so ANY unsuccessful run yielded ERROR even with a
schema-valid verdict on disk; because ``cli_invoke`` downgrades ``success``
on a ~98%-false-positive ``ended_awaiting_background`` flag (task 3639),
that discarded 13+ valid verdicts across 8 tasks in 20 days. The gate now
salvages a well-formed verdict from a failed-but-not-timed-out run (f1),
mirroring the architect's ``_finalized_at`` plan salvage, while still
synthesizing ERROR when nothing valid is on disk (f2) or the run timed out
(f3).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _workflow_helpers import _make
from shared.cli_invoke import AgentResult

from orchestrator.agents.roles import REVIEWER_COMPREHENSIVE
from orchestrator.mcp.verdict_tools import _envelope


class TestReviewerGrantSurface:
    """Structural contract for the reviewer's tool grants (fold-in κ)."""

    def test_no_jcodemunch_grant(self):
        """The reviewer's inert jcodemunch grant has been removed."""
        assert 'mcp__jcodemunch__*' not in REVIEWER_COMPREHENSIVE.allowed_tools

    def test_has_verdict_tools_grant(self):
        """The verdict-tools grant (added by β/task 2482) is present."""
        assert 'mcp__verdict-tools__*' in REVIEWER_COMPREHENSIVE.allowed_tools

    def test_declares_verdict_tools_family(self):
        """The role declares the verdict_tools MCP family (added by β)."""
        assert 'verdict_tools' in REVIEWER_COMPREHENSIVE.mcp_families


def _invoke_writes_review_verdict(
    f, *, verdict: str | None, issues: list | None = None, summary: str = '',
    output: str = '', success: bool = True, timed_out: bool = False,
) -> Callable:
    """Build an ``_invoke`` side_effect that optionally writes a reviewer
    verdict to ``verdicts/reviewer_comprehensive.json``.

    ``verdict=None`` writes no verdict at all — simulating a reviewer agent
    that never called ``submit_review_verdict``.
    """

    def _side_effect(*args, **kwargs):
        if verdict is not None:
            f.artifacts.write_verdict(
                'reviewer_comprehensive',
                _envelope('reviewer_comprehensive', 'sid', {
                    'reviewer': 'reviewer_comprehensive',
                    'verdict': verdict,
                    'issues': issues or [],
                    'summary': summary,
                }),
            )
        return AgentResult(success=success, output=output, timed_out=timed_out)

    return _side_effect


@pytest.mark.asyncio
class TestRunReviewerVerdictRouting:
    """``_run_reviewer`` reads the reviewer's verdict artifact instead of
    the structured_output/``json.loads`` cascade.

    Driven directly through the ``_workflow_helpers._make`` factory (real
    on-disk ``TaskArtifacts``; ``wf.artifacts`` IS ``f.artifacts`` at the
    legacy ``.task`` root, so the test's writes and ``_run_reviewer``'s
    reads hit the same root) — mirrors
    ``test_merger_disposition_verdict.py``.
    """

    def _setup(self, tmp_path: Path):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf.briefing.build_reviewer_prompt = AsyncMock(return_value='prompt')
        return f

    async def test_pass_verdict_ignores_stray_prose(self, tmp_path: Path):
        """(a) a PASS payload is returned verbatim; stray prose in the
        agent's free-text output is ignored.
        """
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_review_verdict(
                f, verdict='PASS', summary='Looks good.',
                output='I think this might have BLOCKING issues actually...',
            ),
        )

        result = await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

        assert result == {
            'reviewer': 'reviewer_comprehensive',
            'verdict': 'PASS',
            'issues': [],
            'summary': 'Looks good.',
        }

    async def test_issues_found_verdict_returned_verbatim(self, tmp_path: Path):
        """(b) an ISSUES_FOUND payload with a non-empty issues list is
        returned verbatim.
        """
        f = self._setup(tmp_path)
        issues = [{
            'severity': 'blocking',
            'location': 'src/foo.py:1',
            'category': 'bug',
            'description': 'desc',
            'suggested_fix': 'fix',
        }]
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_review_verdict(
                f, verdict='ISSUES_FOUND', issues=issues, summary='Found one issue.',
            ),
        )

        result = await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

        assert result == {
            'reviewer': 'reviewer_comprehensive',
            'verdict': 'ISSUES_FOUND',
            'issues': issues,
            'summary': 'Found one issue.',
        }

    async def test_absent_verdict_is_failsafe_error(self, tmp_path: Path):
        """(c) no verdict written at all => fail-safe ERROR — the existing
        ``_review`` retry/aggregate path keys off this signal.
        """
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_review_verdict(
                f, verdict=None, output='no verdict emitted',
            ),
        )

        result = await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

        assert result['verdict'] == 'ERROR'
        assert result['reviewer'] == 'reviewer_comprehensive'

    async def test_stale_verdict_is_cleared_before_spawn(self, tmp_path: Path):
        """(d) a stale prior verdict must never masquerade as this run's (I-FRESH)."""
        f = self._setup(tmp_path)
        # Pre-seed a stale PASS verdict, as if left over from a prior
        # _run_reviewer invocation on this same worktree.
        f.artifacts.write_verdict(
            'reviewer_comprehensive',
            _envelope('reviewer_comprehensive', 'stale-sid', {
                'reviewer': 'reviewer_comprehensive',
                'verdict': 'PASS',
                'issues': [],
                'summary': 'stale',
            }),
        )
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_review_verdict(f, verdict=None, output='nothing new'),
        )

        result = await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

        # If the stale verdict had survived uncleared, its PASS would leak
        # through instead.
        assert result['verdict'] == 'ERROR'

    async def test_missing_verdict_payload_is_failsafe_error(self, tmp_path: Path):
        """(e1) an envelope present but missing the 'verdict' payload key
        entirely is untrusted (defensive extraction).
        """
        f = self._setup(tmp_path)

        def _invoke_writes_malformed(*args, **kwargs):
            f.artifacts.write_verdict(
                'reviewer_comprehensive',
                {'role': 'reviewer_comprehensive', 'schema_version': 1},
            )
            return AgentResult(success=True, output='fine')

        f.wf._invoke = AsyncMock(side_effect=_invoke_writes_malformed)  # type: ignore[method-assign]

        result = await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

        assert result['verdict'] == 'ERROR'

    async def test_invalid_inner_verdict_value_is_failsafe_error(self, tmp_path: Path):
        """(e2) an envelope whose inner verdict is not PASS/ISSUES_FOUND is
        untrusted (defensive extraction).
        """
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_review_verdict(
                f, verdict='MAYBE', summary='unsure', output='fine',
            ),
        )

        result = await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

        assert result['verdict'] == 'ERROR'

    @pytest.mark.parametrize('verdict', ['PASS', 'ISSUES_FOUND'])
    async def test_invocation_failure_salvages_written_verdict(
        self, tmp_path: Path, verdict: str,
    ):
        """(f1) ``_invoke`` success=False but NOT timed out, with a
        well-formed verdict on disk => the verdict is salvaged, not
        discarded.

        AMENDS the pre-narrowing pin (``test_invocation_failure_is_failsafe_
        error``, fe37ca04a8), which asserted ERROR here. ``result.success``
        is not a trustworthy proxy for "the reviewer did not decide": a run
        that reaches ``end_turn`` and writes a schema-valid verdict is
        downgraded to success=False by ``cli_invoke``'s
        ``ended_awaiting_background`` flag, ~98% of the time spuriously
        (task 3639). Mirrors the architect's ``_finalized_at`` plan salvage
        in ``_plan``.
        """
        f = self._setup(tmp_path)
        issues = (
            [{
                'severity': 'blocking',
                'location': 'src/foo.py:1',
                'category': 'bug',
                'description': 'desc',
                'suggested_fix': 'fix',
            }]
            if verdict == 'ISSUES_FOUND' else []
        )
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_review_verdict(
                f, verdict=verdict, issues=issues, summary='Decided.',
                output='ok', success=False,
            ),
        )

        result = await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

        assert result == {
            'reviewer': 'reviewer_comprehensive',
            'verdict': verdict,
            'issues': issues,
            'summary': 'Decided.',
        }

    async def test_invocation_failure_without_verdict_is_failsafe_error(
        self, tmp_path: Path,
    ):
        """(f2) success=False with NO verdict on disk stays fail-safe ERROR
        — the narrowing salvages a written verdict, it does not invent one.
        """
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_review_verdict(
                f, verdict=None, output='crashed before deciding', success=False,
            ),
        )

        result = await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

        assert result['verdict'] == 'ERROR'
        assert result['reviewer'] == 'reviewer_comprehensive'

    async def test_invocation_failure_with_invalid_verdict_is_failsafe_error(
        self, tmp_path: Path,
    ):
        """(f2b) success=False with an out-of-set inner verdict stays
        fail-safe ERROR — salvage requires a *well-formed* payload.
        """
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_review_verdict(
                f, verdict='MAYBE', summary='unsure', output='ok', success=False,
            ),
        )

        result = await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

        assert result['verdict'] == 'ERROR'

    async def test_timed_out_invocation_is_failsafe_error(self, tmp_path: Path):
        """(f3) a TIMED-OUT invocation is fail-safe ERROR even with a
        well-formed verdict on disk.

        This is the residue of the fe37ca04a8 invariant that survives the
        narrowing: a wall-clock kill aborts the run mid-flight, so a verdict
        it left behind may reflect a partial pass over the diff — exactly
        the case the original fail-safe exists for.
        """
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_review_verdict(
                f, verdict='PASS', summary='Looks good.', output='ok',
                success=False, timed_out=True,
            ),
        )

        result = await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

        assert result['verdict'] == 'ERROR'
        assert result['reviewer'] == 'reviewer_comprehensive'
