"""μ contract-agnostic driver scoring in evals/compare.py (task 2478, P4/B8).

The headline invariant (P4 / B8): the μ driver's per-result quality is computed
off PERSISTED review artifacts — the legacy ``reviews/<name>.json`` payload OR
the MCP verdict-tool ``verdicts/reviewer*.json`` envelope — NEVER a transcript /
session / log. Both persisted shapes reduce to ONE payload, so
``score_result_from_artifacts`` returns the SAME quality float regardless of
which contract era wrote the review; scoring tracks whichever reviewer/judge
output contract is live.

Every worktree here is built by hand via TaskArtifacts and holds ONLY the review
artifact (no transcript/session/log exists in it at all), so the score provably
comes from the persisted review — hermetic, no LLM.
"""

from __future__ import annotations

from pathlib import Path

from orchestrator.artifacts import TaskArtifacts
from orchestrator.evals.runner import EvalResult
from orchestrator.mcp import verdict_tools as _verdict_tools

_REVIEWER = 'reviewer_comprehensive'


def _payload(verdict: str = 'ISSUES_FOUND') -> dict:
    """One reviewer payload: 1 blocking + 2 suggestions. Both eras reduce to it."""
    return {
        'reviewer': _REVIEWER,
        'verdict': verdict,
        'issues': [
            {'severity': 'blocking', 'description': 'b1', 'location': 'm/x.py:1'},
            {'severity': 'suggestion', 'description': 's1', 'location': 'm/x.py:2'},
            {'severity': 'suggestion', 'description': 's2', 'location': 'm/x.py:3'},
        ],
        'summary': 'one blocking, two suggestions',
    }


def _clean_payload() -> dict:
    return {'reviewer': _REVIEWER, 'verdict': 'PASS', 'issues': [], 'summary': 'clean'}


def _legacy_worktree(tmp_path: Path, name: str, payload: dict) -> Path:
    """A worktree persisting ONLY the legacy ``reviews/<name>.json`` payload."""
    wt = tmp_path / name
    art = TaskArtifacts(wt)
    art.init('df_task', 'T', 'd')
    art.write_review(_REVIEWER, payload)
    return wt


def _envelope_worktree(tmp_path: Path, name: str, payload: dict) -> Path:
    """A worktree persisting ONLY the MCP verdict-tool envelope verdicts/reviewer*.json."""
    wt = tmp_path / name
    art = TaskArtifacts(wt)
    art.init('df_task', 'T', 'd')
    art.write_verdict(_REVIEWER, _verdict_tools._envelope(_REVIEWER, 'sid', payload))
    return wt


def _result(worktree: Path, *, plan_steps: int = 5) -> EvalResult:
    return EvalResult('df_task', 'cfg', 'done', {'plan_steps': plan_steps}, str(worktree))


class TestScoreResultFromArtifacts:
    def test_legacy_and_envelope_shapes_score_identically(self, tmp_path: Path):
        # The headline P4/B8 assertion: the SAME review content persisted in the
        # two different contract shapes yields the IDENTICAL quality float.
        from orchestrator.evals import compare

        legacy_wt = _legacy_worktree(tmp_path, 'wt_legacy', _payload())
        envelope_wt = _envelope_worktree(tmp_path, 'wt_env', _payload())

        legacy_score = compare.score_result_from_artifacts(_result(legacy_wt))
        envelope_score = compare.score_result_from_artifacts(_result(envelope_wt))

        assert isinstance(legacy_score, float)
        assert legacy_score == envelope_score

    def test_score_comes_from_persisted_artifact_not_a_transcript(self, tmp_path: Path):
        # The worktree holds ONLY the review artifact — no transcript/session/log
        # exists — so a non-constant, content-sensitive score PROVES the value came
        # from the persisted review. A clean PASS scores strictly higher, and the
        # value is EXACTLY scoring.quality_from_review_artifact on the same payload
        # (single-sourced to compute_composite — the driver re-derives nothing).
        from orchestrator.evals import compare, scoring

        issues_wt = _legacy_worktree(tmp_path, 'wt_issues', _payload())
        clean_wt = _legacy_worktree(tmp_path, 'wt_clean', _clean_payload())

        issues_score = compare.score_result_from_artifacts(_result(issues_wt))
        clean_score = compare.score_result_from_artifacts(_result(clean_wt))

        assert clean_score > issues_score
        assert issues_score == scoring.quality_from_review_artifact(_payload(), plan_steps=5)

    def test_error_verdict_zeroes_issues_matching_aggregate_reviews(self, tmp_path: Path):
        # An ERROR top-level verdict is skipped exactly as artifacts.aggregate_reviews
        # skips it (0 blocking / 0 suggestions): even carrying issues entries it
        # scores identically to a clean, zero-issue review.
        from orchestrator.evals import compare

        error_wt = _legacy_worktree(tmp_path, 'wt_error', _payload('ERROR'))
        clean_wt = _legacy_worktree(tmp_path, 'wt_clean2', _clean_payload())

        error_score = compare.score_result_from_artifacts(_result(error_wt))
        clean_score = compare.score_result_from_artifacts(_result(clean_wt))
        assert error_score == clean_score

    def test_envelope_error_verdict_also_zeroes_issues(self, tmp_path: Path):
        # Contract-agnostic even for the ERROR filter: an ERROR verdict persisted
        # in the MCP envelope shape zeroes issues just like the legacy shape.
        from orchestrator.evals import compare

        env_error_wt = _envelope_worktree(tmp_path, 'wt_env_error', _payload('ERROR'))
        legacy_error_wt = _legacy_worktree(tmp_path, 'wt_leg_error', _payload('ERROR'))

        assert (
            compare.score_result_from_artifacts(_result(env_error_wt))
            == compare.score_result_from_artifacts(_result(legacy_error_wt))
        )

    def test_plan_steps_falls_back_to_result_metrics_and_kwarg_overrides(self, tmp_path: Path):
        # plan_steps is sourced from result.metrics when the kwarg is omitted; an
        # explicit kwarg overrides it.
        from orchestrator.evals import compare, scoring

        wt = _legacy_worktree(tmp_path, 'wt_ps', _payload())
        result = _result(wt, plan_steps=8)

        from_metrics = compare.score_result_from_artifacts(result)
        assert from_metrics == scoring.quality_from_review_artifact(_payload(), plan_steps=8)

        override = compare.score_result_from_artifacts(result, plan_steps=3)
        assert override == scoring.quality_from_review_artifact(_payload(), plan_steps=3)
