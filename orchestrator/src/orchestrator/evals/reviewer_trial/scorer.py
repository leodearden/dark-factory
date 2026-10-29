"""Scorer for the reviewer panel trial.

Matches reviewer findings to ground-truth annotations using LLM-assisted
matching (haiku), then computes recall, precision, F1, and blocking recall.
"""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from shared.cli_invoke import no_mcp_servers_config

from orchestrator.agents.invoke import invoke_agent
from orchestrator.evals.measurement import (
    MeasurementStage,
    Scored,
    Unmeasured,
    json_answer,
    measured_call,
)
from orchestrator.evals.reviewer_trial.corpus import CorpusDiff, GroundTruthIssue
from orchestrator.evals.reviewer_trial.runner import PanelRunResult
from orchestrator.review_payload import ReviewPayload, normalize_review

if TYPE_CHECKING:
    from shared.cli_invoke import AgentResult
    from shared.usage_gate import UsageGate

logger = logging.getLogger(__name__)


@dataclass
class IssueMatch:
    """A match between a reviewer-found issue and a ground-truth issue."""

    reviewer_issue: dict
    ground_truth_id: str
    match_confidence: float
    match_reasoning: str


@dataclass(frozen=True)
class MatchedIssues:
    """What the matcher measured for one panel: its matches and its false positives."""

    matches: list[IssueMatch]
    false_positives: list[dict]


@dataclass
class ScoringResult:
    """Scoring outcome for one (variant, diff) pair.

    Cost fields (kept separate for audit granularity):

    - ``cost_usd``       — reviewer-panel cost only (sum of all reviewer
                           invocations for this diff via ``PanelRunResult``).
    - ``match_cost_usd`` — haiku matcher cost for this diff (the LLM call
                           inside ``match_issues()``).  0.0 when no LLM call
                           was made (empty inputs) or when the field is not
                           yet populated (backward-compatible default).

    Use the ``total_cost_usd`` property when you need the combined figure,
    so callers cannot accidentally read only ``cost_usd`` and undercount.

    - ``unmeasured``     — set when the matcher did not measure. The metric
                           fields then carry no measurement, matches /
                           false_positives / unmatched_gt are empty, and every
                           reducer excludes the record and counts it
                           (``campaign.summarize_arm``,
                           ``report.build_trial_report``).
    """

    variant_name: str
    diff_id: str
    matches: list[IssueMatch] = field(default_factory=list)
    unmatched_gt: list[str] = field(default_factory=list)
    false_positives: list[dict] = field(default_factory=list)
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    blocking_recall: float = 0.0
    cost_usd: float = 0.0       # panel cost only — see class docstring
    match_cost_usd: float = 0.0  # haiku matcher cost — see class docstring
    wall_clock_ms: int = 0
    unmeasured: Unmeasured | None = None

    @property
    def total_cost_usd(self) -> float:
        """Combined panel + matcher cost for this (variant, diff) pair."""
        return self.cost_usd + self.match_cost_usd


# Schema for the LLM matcher output
_MATCH_SCHEMA = {
    'type': 'object',
    'properties': {
        'matches': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'reviewer_issue_index': {'type': 'integer'},
                    'ground_truth_id': {'type': 'string'},
                    'confidence': {'type': 'number', 'minimum': 0.0, 'maximum': 1.0},
                    'reasoning': {'type': 'string'},
                },
                'required': ['reviewer_issue_index', 'ground_truth_id', 'confidence', 'reasoning'],
            },
        },
    },
    'required': ['matches'],
}


def _normalize_location(loc: str) -> str:
    """Normalize a file location for deduplication.

    Strips line numbers and normalizes path separators.
    "src/foo.py:42" -> "src/foo.py"
    "src/foo.py:42-50" -> "src/foo.py"
    """
    # Remove line number suffixes
    normalized = re.sub(r':\d+(-\d+)?$', '', loc.strip())
    return normalized.replace('\\', '/')


def _deduplicate_issues(reviews: Mapping[str, ReviewPayload]) -> list[dict]:
    """Deduplicate issues across reviewers within a panel.

    Groups by (normalized_location, category). When multiple reviewers
    flag the same file with the same category, keep the one with the
    longest description (most informative).
    """
    seen: dict[tuple[str, str], dict] = {}

    for reviewer_name, review in reviews.items():
        for issue in review.issues:
            loc = _normalize_location(issue.get('location', ''))
            cat = issue.get('category', '').lower().strip()
            key = (loc, cat)

            enriched = {**issue, 'reviewer': reviewer_name}
            if key not in seen:
                seen[key] = enriched
            else:
                # Keep the more detailed description
                existing_len = len(seen[key].get('description', ''))
                new_len = len(issue.get('description', ''))
                if new_len > existing_len:
                    seen[key] = enriched

    return list(seen.values())


async def match_issues(
    reviewer_issues: list[dict],
    ground_truth: list[GroundTruthIssue],
    diff_text: str,
    confidence_threshold: float = 0.5,
    *,
    usage_gate: UsageGate | None = None,
    label: str = 'issue matcher',
) -> Scored[MatchedIssues] | Unmeasured:
    """Use haiku to match reviewer findings to ground truth.

    With no findings or no ground truth there is nothing to match: that is a
    measurement made without an LLM call, at ``cost_usd=0.0``. Otherwise the
    call goes through ``measured_call`` (through *usage_gate* when given), so
    a refused call or an answer without a ``matches`` list is ``Unmeasured``
    with its real spend, never an empty match.
    """
    if not reviewer_issues or not ground_truth:
        nothing_matched = MatchedIssues(matches=[], false_positives=list(reviewer_issues))
        return Scored(nothing_matched, cost_usd=0.0)

    # Format issues for the matcher prompt
    ri_lines = []
    for i, issue in enumerate(reviewer_issues):
        ri_lines.append(
            f'  [{i}] location="{issue.get("location", "?")}" '
            f'category="{issue.get("category", "?")}" '
            f'severity="{issue.get("severity", "?")}" '
            f'description="{issue.get("description", "?")}"'
        )

    gt_lines = []
    for gt in ground_truth:
        gt_lines.append(
            f'  [{gt.id}] location="{gt.location}" '
            f'category="{gt.category}" '
            f'severity="{gt.severity}" '
            f'description="{gt.description}"'
        )

    # Truncate diff for context
    diff_context = diff_text[:10_000] if len(diff_text) > 10_000 else diff_text

    prompt = f"""\
Match reviewer findings to ground truth issues in this code diff.

## Diff (for context)

```diff
{diff_context}
```

## Reviewer Findings

{chr(10).join(ri_lines)}

## Ground Truth Issues

{chr(10).join(gt_lines)}

## Instructions

For each reviewer finding, determine if it matches a ground truth issue.
A match means the reviewer identified the same fundamental problem, even if
described differently or at a slightly different location.

Only report matches with confidence >= {confidence_threshold}.
Each ground truth issue should match at most one reviewer finding (pick the best).
Each reviewer finding should match at most one ground truth issue.

Output your matches as JSON.
"""

    system_prompt = (
        'You are a precise issue matcher. Match reviewer findings to ground truth '
        'issues based on whether they identify the same underlying problem. '
        'Output ONLY valid JSON.'
    )

    return await measured_call(
        invoke_agent,
        functools.partial(
            _parse_matches,
            reviewer_issues=reviewer_issues,
            ground_truth=ground_truth,
            confidence_threshold=confidence_threshold,
        ),
        stage=MeasurementStage.MATCHER,
        backend='claude',
        usage_gate=usage_gate,
        label=label,
        prompt=prompt,
        system_prompt=system_prompt,
        cwd=Path('/home/leo/src/dark-factory'),
        model='haiku',
        max_turns=3,
        max_budget_usd=0.50,
        output_schema=_MATCH_SCHEMA,
        effort='low',
        # The matcher needs no tools: deny the built-ins and scope MCP to an
        # empty server set so the .mcp.json at cwd is never ambient-merged
        # (allowed_tools=[] does not restrict under bypassPermissions).
        disallowed_tools=['*'],
        mcp_config=no_mcp_servers_config(),
        strict_mcp_config=True,
    )


def _parse_matches(
    result: AgentResult,
    *,
    reviewer_issues: list[dict],
    ground_truth: list[GroundTruthIssue],
    confidence_threshold: float,
) -> MatchedIssues:
    """The matcher's answer as matches and false positives.

    Raises ``KeyError`` / ``ValueError`` when the answer carries no ``matches``
    list of objects. An entry below the threshold, out of range, naming an
    unknown ground-truth id or breaking one-to-one is a match the matcher did
    not make, so it is skipped and its finding stays a false positive.
    """
    entries = json_answer(result)['matches']
    if not isinstance(entries, list):
        raise ValueError(f'matches is not a list but {type(entries).__name__}')
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f'a matches entry is not an object but {type(entry).__name__}')

    matches: list[IssueMatch] = []
    matched_indices: set[int] = set()
    gt_ids = {gt.id for gt in ground_truth}
    matched_gt_ids: set[str] = set()

    for m in entries:
        idx = m.get('reviewer_issue_index', -1)
        gt_id = m.get('ground_truth_id', '')
        confidence = m.get('confidence', 0.0)

        if (
            confidence < confidence_threshold
            or idx < 0
            or idx >= len(reviewer_issues)
            or gt_id not in gt_ids
            or idx in matched_indices
            or gt_id in matched_gt_ids
        ):
            continue

        matches.append(IssueMatch(
            reviewer_issue=reviewer_issues[idx],
            ground_truth_id=gt_id,
            match_confidence=confidence,
            match_reasoning=m.get('reasoning', ''),
        ))
        matched_indices.add(idx)
        matched_gt_ids.add(gt_id)

    # Unmatched reviewer issues = false positives
    unmatched = [
        issue for i, issue in enumerate(reviewer_issues)
        if i not in matched_indices
    ]
    return MatchedIssues(matches=matches, false_positives=unmatched)


async def score_panel_run(
    run: PanelRunResult,
    corpus_diff: CorpusDiff,
    *,
    usage_gate: UsageGate | None = None,
) -> ScoringResult:
    """Score one (variant, diff) pair.

    1. Collect every reviewer's issues, each review normalised by
       ``orchestrator/review_payload.py::normalize_review`` (an ERROR review
       carries none; an off-contract one raises ``ReviewPayloadError``)
    2. Deduplicate by (normalized_location, category)
    3. Match against ground truth via match_issues() (through *usage_gate*
       when given); an unmeasured match makes the whole result unmeasured,
       with no metric computed
    4. Compute recall, precision, F1, blocking_recall
    """
    reviews = {name: normalize_review(rev) for name, rev in run.reviews.items()}

    # Deduplicate across panel
    deduped = _deduplicate_issues(reviews)

    # Match against ground truth
    outcome = await match_issues(
        reviewer_issues=deduped,
        ground_truth=corpus_diff.ground_truth,
        diff_text=corpus_diff.diff_text,
        usage_gate=usage_gate,
        label=f'matcher {run.variant_name}__{run.diff_id}',
    )
    if isinstance(outcome, Unmeasured):
        return ScoringResult(
            variant_name=run.variant_name,
            diff_id=run.diff_id,
            cost_usd=run.total_cost_usd,
            match_cost_usd=round(outcome.cost_usd, 4),
            wall_clock_ms=run.wall_clock_ms,
            unmeasured=outcome,
        )
    matches = outcome.value.matches

    # Compute metrics
    gt_total = len(corpus_diff.ground_truth)
    matched_gt_ids = {m.ground_truth_id for m in matches}
    unmatched_gt = [gt.id for gt in corpus_diff.ground_truth if gt.id not in matched_gt_ids]

    recall = len(matches) / gt_total if gt_total > 0 else 0.0

    total_findings = len(deduped)
    precision = len(matches) / total_findings if total_findings > 0 else 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Blocking recall: what fraction of blocking ground-truth issues were found?
    blocking_gt = corpus_diff.blocking_issues()
    blocking_found = sum(1 for m in matches if m.ground_truth_id in {
        gt.id for gt in blocking_gt
    })
    blocking_recall = blocking_found / len(blocking_gt) if blocking_gt else 0.0

    return ScoringResult(
        variant_name=run.variant_name,
        diff_id=run.diff_id,
        matches=matches,
        unmatched_gt=unmatched_gt,
        false_positives=outcome.value.false_positives,
        recall=round(recall, 4),
        precision=round(precision, 4),
        f1=round(f1, 4),
        blocking_recall=round(blocking_recall, 4),
        cost_usd=run.total_cost_usd,
        match_cost_usd=round(outcome.cost_usd, 4),
        wall_clock_ms=run.wall_clock_ms,
    )
