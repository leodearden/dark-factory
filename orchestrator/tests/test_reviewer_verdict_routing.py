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

import asyncio
import json
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


def _invoke_writes_raw_verdict(
    f, payload: dict, *, success: bool = True, output: str = 'ok',
    timed_out: bool = False,
) -> Callable:
    """Build an ``_invoke`` side_effect that writes *payload* to
    ``verdicts/reviewer_comprehensive.json`` VERBATIM, bypassing
    ``_submit_review_verdict``.

    ``mcp/verdict_tools.py:_submit_review_verdict`` rejects a payload whose
    ``reviewer`` disagrees with the artifact's role at WRITE time ("the
    artifact filename is authoritative for this role"), so a cross-role
    payload can only be staged on disk by going around that server — which
    is exactly the corrupt/hand-edited-artifact case ``_run_reviewer``'s
    own identity check must defend against.
    """

    def _side_effect(*args, **kwargs):
        f.artifacts.write_verdict(
            'reviewer_comprehensive',
            _envelope('reviewer_comprehensive', 'sid', payload),
        )
        return AgentResult(success=success, output=output, timed_out=timed_out)

    return _side_effect


@pytest.mark.asyncio
class TestReviewerIdentityGate:
    """A salvaged payload must actually belong to the role being run.

    Defense-in-depth (task 3639): ``verdict_tools.py:86-95`` already rejects
    a mismatched ``reviewer`` at write time, so this gate only fires for a
    corrupt, hand-edited or cross-role artifact reaching disk another way.
    It matters because ``_run_reviewer``'s return is mirrored verbatim to
    ``reviews/<role>.json`` (``_review`` -> ``artifacts.write_review``) and
    that mirror is the SOLE input to ``aggregate_reviews()``, which makes
    the blocking-issue decision — so a cross-role payload returned verbatim
    would be filed, and counted, under the wrong reviewer's name.
    """

    def _setup(self, tmp_path: Path):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf.briefing.build_reviewer_prompt = AsyncMock(return_value='prompt')
        return f

    @pytest.mark.parametrize('success', [True, False])
    async def test_verdict_reviewer_identity_mismatch_is_failsafe_error(
        self, tmp_path: Path, success: bool,
    ):
        """A well-formed payload naming a DIFFERENT reviewer is untrusted.

        Parametrized over ``success`` so the identity gate is proven
        independent of the invocation-result signal — it is a property of
        the payload, not of how the run ended.
        """
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_raw_verdict(
                f,
                {
                    'reviewer': 'reviewer_security',
                    'verdict': 'PASS',
                    'issues': [],
                    'summary': 'Different reviewer entirely.',
                },
                success=success,
                output='cross-role artifact',
            ),
        )

        result = await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

        assert result['reviewer'] == 'reviewer_comprehensive'
        assert result['verdict'] == 'ERROR'
        assert result['issues'] == []
        assert result['summary'].startswith('Reviewer emitted no/invalid verdict:')

    async def test_absent_reviewer_field_is_failsafe_error(self, tmp_path: Path):
        """A payload with no ``reviewer`` key at all cannot be attributed."""
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_raw_verdict(
                f, {'verdict': 'PASS', 'issues': [], 'summary': 'anonymous'},
            ),
        )

        result = await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

        assert result['verdict'] == 'ERROR'
        assert result['reviewer'] == 'reviewer_comprehensive'

    @pytest.mark.parametrize('success', [True, False])
    async def test_matching_reviewer_identity_is_salvaged(
        self, tmp_path: Path, success: bool,
    ):
        """Positive control: the SAME staging path with a matching
        ``reviewer`` still returns the payload verbatim.

        Guards against the identity check being written inverted — this
        differs from the mismatch case in exactly one field.
        """
        f = self._setup(tmp_path)
        payload = {
            'reviewer': 'reviewer_comprehensive',
            'verdict': 'PASS',
            'issues': [],
            'summary': 'Mine, and it says PASS.',
        }
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_raw_verdict(f, payload, success=success),
        )

        result = await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

        assert result == payload


def _invoke_writes_then_raises(
    f, exc: BaseException, *, verdict: str | None = 'PASS',
    issues: list | None = None, summary: str = 'Decided.',
) -> Callable:
    """Build an ``_invoke`` side_effect that (optionally) writes a valid
    reviewer verdict and THEN raises *exc*.

    Models the real failure shape this closes: the reviewer agent runs to
    ``end_turn`` and calls ``submit_review_verdict`` — the verdict is on
    disk — and the invocation then dies in the post-agent work that shares
    the same await (transcript archival, telemetry, transport teardown).
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
        raise exc

    return _side_effect


def _suggestion_issues(n: int) -> list[dict]:
    """*n* schema-shaped suggestion-severity issues."""
    return [{
        'severity': 'suggestion',
        'location': f'src/mod_{i}.py:{i + 1}',
        'category': 'quality',
        'description': f'suggestion {i}',
        'suggested_fix': f'fix {i}',
    } for i in range(n)]


@pytest.mark.asyncio
class TestExceptionPathSalvage:
    """An exception escaping ``_invoke`` must not discard a well-formed
    on-disk verdict, and must not burn a retry.

    ``b4fe7171d3`` narrowed the post-invocation gate so a failed-but-written
    verdict is salvaged — but an exception escaping the ``_invoke`` await
    bypasses that gate ENTIRELY (the await sits one line after
    ``clear_verdict()``, and the ``read_verdict`` salvage is never reached).
    ``_review`` then treats the exception as a first-class outcome: its
    ``gather(..., return_exceptions=True)`` captures it, the retry loop
    re-enters ``_run_reviewer`` whose ``clear_verdict()`` destroys the
    recoverable verdict permanently, and on exhaustion it synthesizes
    ``{'verdict': 'ERROR', 'issues': []}`` without ever consulting disk.

    That is the esc-5777-7 shape: a good ``verdicts/`` artifact and an ERROR
    ``reviews/`` mirror written seconds later.
    """

    def _setup(self, tmp_path: Path):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf.briefing.build_reviewer_prompt = AsyncMock(return_value='prompt')
        return f

    @pytest.mark.parametrize('verdict', ['PASS', 'ISSUES_FOUND'])
    async def test_exception_after_verdict_written_is_salvaged(
        self, tmp_path: Path, verdict: str,
    ):
        """A well-formed verdict on disk survives an exception from ``_invoke``."""
        f = self._setup(tmp_path)
        issues = _suggestion_issues(1) if verdict == 'ISSUES_FOUND' else []
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_then_raises(
                f, RuntimeError('transport closed'),
                verdict=verdict, issues=issues,
            ),
        )

        result = await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

        assert result == {
            'reviewer': 'reviewer_comprehensive',
            'verdict': verdict,
            'issues': issues,
            'summary': 'Decided.',
        }

    async def test_exception_without_verdict_still_propagates(self, tmp_path: Path):
        """Nothing on disk => the exception propagates unchanged.

        Task 3321's no-emit case: ``_review``'s existing retry + synthesized
        ERROR fail-safe must be untouched. Salvage recovers a verdict that
        exists; it never invents one.
        """
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_then_raises(
                f, RuntimeError('died before deciding'), verdict=None,
            ),
        )

        with pytest.raises(RuntimeError, match='died before deciding'):
            await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

    async def test_timeout_exception_is_not_salvaged(self, tmp_path: Path):
        """``TimeoutError`` propagates even with a valid verdict on disk.

        Exception-path analogue of the already-pinned ``result.timed_out``
        exclusion (``test_timed_out_invocation_is_failsafe_error``): a
        wall-clock kill can land mid-write, so its artifact may reflect a
        partial pass over the diff.
        """
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_then_raises(f, TimeoutError('wall clock')),
        )

        with pytest.raises(TimeoutError):
            await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

    async def test_cancellation_is_never_swallowed(self, tmp_path: Path):
        """``asyncio.CancelledError`` propagates even with a verdict on disk.

        Cooperative cancellation (a clean SIGTERM shutdown) must never be
        converted into a salvage — mirrors ``_invoke``'s own
        preserve-and-re-raise contract.
        """
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_then_raises(f, asyncio.CancelledError()),
        )

        with pytest.raises(asyncio.CancelledError):
            await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

    async def test_exception_with_invalid_verdict_propagates(self, tmp_path: Path):
        """An out-of-set inner verdict is not salvageable => propagate."""
        f = self._setup(tmp_path)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_then_raises(
                f, RuntimeError('transport closed'), verdict='MAYBE',
            ),
        )

        with pytest.raises(RuntimeError, match='transport closed'):
            await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

    async def test_exception_with_cross_role_verdict_propagates(self, tmp_path: Path):
        """A cross-role payload is not salvageable => propagate.

        Proves the exception path routes through the SAME
        ``_salvageable_verdict_payload`` seam as the normal gate, rather
        than re-implementing a laxer check.
        """
        f = self._setup(tmp_path)

        def _side_effect(*args, **kwargs):
            f.artifacts.write_verdict(
                'reviewer_comprehensive',
                _envelope('reviewer_comprehensive', 'sid', {
                    'reviewer': 'reviewer_security',
                    'verdict': 'PASS',
                    'issues': [],
                    'summary': 'Not mine.',
                }),
            )
            raise RuntimeError('transport closed')

        f.wf._invoke = AsyncMock(side_effect=_side_effect)  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match='transport closed'):
            await f.wf._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')


@pytest.mark.asyncio
class TestReviewMirrorCarriesSalvagedVerdict:
    """``_review``'s ``reviews/<role>.json`` mirror — the esc-5777-7 surface.

    The escalation was observed AT the mirror (a good ``verdicts/`` artifact
    and an ``{'verdict': 'ERROR', 'issues': []}`` ``reviews/`` file written
    18s later), and no existing test drives ``_review`` at all. The mirror is
    what ``aggregate_reviews()`` reads, so a discarded verdict there is what
    actually loses the reviewer's issues.

    (``ALL_REVIEWERS`` is a single comprehensive reviewer today, so these
    drive the whole panel through one role.)
    """

    def _setup(self, tmp_path: Path, *, retries: int = 2):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf.briefing.build_reviewer_prompt = AsyncMock(return_value='prompt')
        f.wf.git_ops.get_diff_from_base = AsyncMock(return_value='diff')
        f.wf.git_ops.get_diff_from_main = AsyncMock(return_value='diff')
        f.wf.config.reviewer_stagger_secs = 0
        f.wf.config.max_reviewer_retries = retries
        return f

    def _mirror(self, f) -> dict:
        return json.loads(
            (f.artifacts.root / 'reviews' / 'reviewer_comprehensive.json').read_text(),
        )

    async def test_mirror_carries_verdict_salvaged_from_exception(
        self, tmp_path: Path,
    ):
        """An exception after the verdict was written must not blank the mirror."""
        f = self._setup(tmp_path)
        issues = _suggestion_issues(6)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_then_raises(
                f, RuntimeError('transport closed'),
                verdict='ISSUES_FOUND', issues=issues, summary='Six things.',
            ),
        )

        aggregation = await f.wf._review()

        mirror = self._mirror(f)
        assert mirror['verdict'] == 'ISSUES_FOUND'
        assert mirror['issues'] == issues
        assert mirror['reviewer'] == 'reviewer_comprehensive'
        # No retry burned => no clear_verdict() destroyed the recoverable
        # verdict. In the pre-fix shape this is 1 + max_reviewer_retries.
        assert f.wf._invoke.call_count == 1
        assert aggregation.reviewer_errors == []
        assert len(aggregation.suggestions) == 6

    async def test_mirror_carries_verdict_salvaged_from_failed_invocation(
        self, tmp_path: Path,
    ):
        """Regression pin for ``b4fe7171d3``: the non-exception salvage
        (success=False, verdict written, no raise) already reaches the mirror.
        """
        f = self._setup(tmp_path)
        issues = _suggestion_issues(6)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_review_verdict(
                f, verdict='ISSUES_FOUND', issues=issues, summary='Six things.',
                output='ok', success=False,
            ),
        )

        aggregation = await f.wf._review()

        mirror = self._mirror(f)
        assert mirror['verdict'] == 'ISSUES_FOUND'
        assert mirror['issues'] == issues
        assert f.wf._invoke.call_count == 1
        assert aggregation.reviewer_errors == []

    async def test_mirror_is_error_when_nothing_salvageable(self, tmp_path: Path):
        """Fail-safe intact: an exception with no verdict on disk still
        retries and still lands the synthesized ERROR mirror.
        """
        f = self._setup(tmp_path, retries=2)
        f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_invoke_writes_then_raises(
                f, RuntimeError('died before deciding'), verdict=None,
            ),
        )

        aggregation = await f.wf._review()

        mirror = self._mirror(f)
        assert mirror['verdict'] == 'ERROR'
        assert mirror['issues'] == []
        assert mirror['summary'].startswith('Reviewer exception:')
        assert f.wf._invoke.call_count == 3  # initial + 2 retries
        assert aggregation.reviewer_errors == ['reviewer_comprehensive']
