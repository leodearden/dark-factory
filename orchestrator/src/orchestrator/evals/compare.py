"""LLM-powered qualitative head-to-head comparison of two model groups."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from orchestrator.agents.invoke import invoke_agent

from .judge import _strip_metadata
from .runner import EvalResult
from .snapshots import get_diff

logger = logging.getLogger(__name__)

OUTCOME_PRIORITY = {'done': 3, 'blocked': 2, 'timeout': 1}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TaskAssessment:
    """Per-task LLM assessment of both models."""

    task_id: str
    task_name: str
    winner: str                      # 'A' | 'B' | 'tie'
    confidence: float                # 0.0-1.0
    summary: str
    strengths_a: list[str] = field(default_factory=list)
    weaknesses_a: list[str] = field(default_factory=list)
    strengths_b: list[str] = field(default_factory=list)
    weaknesses_b: list[str] = field(default_factory=list)
    circumstances: str = ''
    metrics_a: dict[str, Any] = field(default_factory=dict)
    metrics_b: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonReport:
    """Full head-to-head comparison report."""

    group_a_name: str
    group_b_name: str
    group_a_configs: list[str]
    group_b_configs: list[str]
    assessments: list[TaskAssessment]
    # LLM synthesis
    overall_winner: str = ''
    executive_summary: str = ''
    circumstantial_analysis: str = ''
    cost_quality_tradeoff: str = ''
    workflow_recommendations: str = ''
    # Quantitative backdrop
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    no_contest: int = 0
    avg_score_a: float = 0.0
    avg_score_b: float = 0.0


# ---------------------------------------------------------------------------
# Combine-runs aliasing
# ---------------------------------------------------------------------------

def apply_combine_runs(
    results: list[EvalResult],
    combine_groups: list[list[str]],
) -> list[EvalResult]:
    """Alias config names to the first name in each combine group.

    Returns a new list with shallow-copied EvalResult objects whose
    config_name has been remapped.  Original list is not mutated.
    """
    alias_map: dict[str, str] = {}
    for group in combine_groups:
        canonical = group[0]
        for name in group:
            alias_map[name] = canonical

    out: list[EvalResult] = []
    for r in results:
        canonical = alias_map.get(r.config_name, r.config_name)
        if canonical != r.config_name:
            r = replace(r, config_name=canonical)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Result selection
# ---------------------------------------------------------------------------

def pick_best_result(results: list[EvalResult]) -> EvalResult | None:
    """Pick the most informative result for qualitative comparison.

    Priority: results with actual code changes first (gives the assessor
    something to compare), then composite_score, then outcome priority,
    then lines_changed (more work = more informative diff).
    """
    if not results:
        return None

    def key(r: EvalResult) -> tuple[int, float, int, int]:
        has_changes = 1 if r.metrics.get('lines_changed', 0) > 0 else 0
        return (
            has_changes,
            r.metrics.get('composite_score', 0.0),
            OUTCOME_PRIORITY.get(r.outcome, 0),
            r.metrics.get('lines_changed', 0),
        )

    return max(results, key=key)


# ---------------------------------------------------------------------------
# Review artifact reading
# ---------------------------------------------------------------------------

def _read_review_artifacts(worktree_path: str) -> dict[str, Any]:
    """Read review JSON files from a worktree's .task/reviews/ directory.

    Returns ``{reviewer_name: {verdict, issues, summary}}``.
    Returns empty dict if the worktree or reviews dir doesn't exist.
    """
    reviews_dir = Path(worktree_path) / '.task' / 'reviews'
    if not reviews_dir.is_dir():
        return {}

    reviews: dict[str, Any] = {}
    for path in sorted(reviews_dir.glob('*.json')):
        try:
            data = json.loads(path.read_text())
            name = data.get('reviewer', path.stem)
            reviews[name] = data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning('Failed to read review %s: %s', path, e)
    return reviews


def _format_review_artifacts(reviews: dict[str, Any]) -> str:
    """Format review artifacts as text for the assessor prompt."""
    if not reviews:
        return '(no review data available)'

    parts: list[str] = []
    for reviewer_name, data in reviews.items():
        verdict = data.get('verdict', 'unknown')
        summary = data.get('summary', '')
        issues = data.get('issues', [])

        lines = [f'### {reviewer_name} — {verdict}']
        if summary:
            lines.append(summary)

        for issue in issues:
            sev = issue.get('severity', '?')
            cat = issue.get('category', '')
            loc = issue.get('location', '')
            desc = issue.get('description', '')
            fix = issue.get('suggested_fix', '')
            lines.append(f'  [{sev}] {cat} at {loc}')
            lines.append(f'    {desc}')
            if fix:
                lines.append(f'    Fix: {fix}')

        parts.append('\n'.join(lines))

    return '\n\n'.join(parts)


# ---------------------------------------------------------------------------
# Per-task LLM assessment
# ---------------------------------------------------------------------------

ASSESSOR_SCHEMA = {
    'type': 'object',
    'properties': {
        'winner': {'type': 'string', 'enum': ['A', 'B', 'tie']},
        'confidence': {'type': 'number', 'minimum': 0.0, 'maximum': 1.0},
        'summary': {'type': 'string'},
        'strengths_a': {'type': 'array', 'items': {'type': 'string'}},
        'weaknesses_a': {'type': 'array', 'items': {'type': 'string'}},
        'strengths_b': {'type': 'array', 'items': {'type': 'string'}},
        'weaknesses_b': {'type': 'array', 'items': {'type': 'string'}},
        'circumstances': {'type': 'string'},
    },
    'required': [
        'winner', 'confidence', 'summary',
        'strengths_a', 'weaknesses_a', 'strengths_b', 'weaknesses_b',
        'circumstances',
    ],
}


def _build_assessor_prompt(
    task: dict,
    result_a: EvalResult,
    result_b: EvalResult,
    diff_a: str,
    diff_b: str,
    reviews_a: dict[str, Any],
    reviews_b: dict[str, Any],
    group_a_name: str,
    group_b_name: str,
) -> str:
    """Build the per-task assessor prompt with full context."""
    task_name = task.get('name', task.get('id', 'unknown'))
    task_desc = task.get('task_definition', {}).get('description', '')

    ma = result_a.metrics
    mb = result_b.metrics

    diff_a_display = _strip_metadata(diff_a) if diff_a else '(no diff available)'
    diff_b_display = _strip_metadata(diff_b) if diff_b else '(no diff available)'

    # Truncate very large diffs to avoid blowing context
    max_diff_chars = 30_000
    if len(diff_a_display) > max_diff_chars:
        diff_a_display = diff_a_display[:max_diff_chars] + '\n... (truncated)'
    if len(diff_b_display) > max_diff_chars:
        diff_b_display = diff_b_display[:max_diff_chars] + '\n... (truncated)'

    reviews_a_text = _format_review_artifacts(reviews_a)
    reviews_b_text = _format_review_artifacts(reviews_b)

    return f"""You are an expert code quality analyst comparing two model implementations \
of the same software engineering task. Produce a detailed qualitative assessment.

## Task
**{task_name}**
{task_desc}

## Model A: {group_a_name}
Outcome: {result_a.outcome} | Score: {ma.get('composite_score', 0):.2f} | \
Cost: ${ma.get('cost_usd', 0):.2f}
Iterations: {ma.get('iterations', 0)} | Debug cycles: {ma.get('debug_cycles', 0)} | \
Lines: {ma.get('lines_changed', 0)} | Files: {ma.get('files_changed', 0)}
Tests: {ma.get('tests_pass')} | Lint: {ma.get('lint_clean')} | \
Types: {ma.get('typecheck_clean')}

### Diff
```diff
{diff_a_display}
```

### Review Findings
{reviews_a_text}

## Model B: {group_b_name}
Outcome: {result_b.outcome} | Score: {mb.get('composite_score', 0):.2f} | \
Cost: ${mb.get('cost_usd', 0):.2f}
Iterations: {mb.get('iterations', 0)} | Debug cycles: {mb.get('debug_cycles', 0)} | \
Lines: {mb.get('lines_changed', 0)} | Files: {mb.get('files_changed', 0)}
Tests: {mb.get('tests_pass')} | Lint: {mb.get('lint_clean')} | \
Types: {mb.get('typecheck_clean')}

### Diff
```diff
{diff_b_display}
```

### Review Findings
{reviews_b_text}

## Assessment Instructions

Analyze both implementations and produce your assessment as JSON.

Focus on:
- **Correctness**: Does the code actually solve the task completely and correctly?
- **Design quality**: Idiomatic patterns, appropriate abstractions, fits codebase style
- **Test quality**: Meaningful assertions, edge cases, good isolation
- **Robustness**: Error handling, edge cases, defensive coding
- **Efficiency**: Minimal unnecessary changes, clean implementation path
- **Circumstances**: What specific task characteristics or patterns favor each model?

Be concrete and specific. Reference actual code patterns, function names, and design \
choices from the diffs. Don't just restate the metrics — explain *why* the code quality \
differs and what the metrics reveal about each model's approach."""


SYNTHESIZER_SCHEMA = {
    'type': 'object',
    'properties': {
        'overall_winner': {'type': 'string', 'enum': ['A', 'B', 'tie']},
        'executive_summary': {'type': 'string'},
        'circumstantial_analysis': {'type': 'string'},
        'cost_quality_tradeoff': {'type': 'string'},
        'workflow_recommendations': {'type': 'string'},
    },
    'required': [
        'overall_winner', 'executive_summary', 'circumstantial_analysis',
        'cost_quality_tradeoff', 'workflow_recommendations',
    ],
}


def _build_synthesizer_prompt(
    assessments: list[TaskAssessment],
    group_a_name: str,
    group_b_name: str,
    tasks: dict[str, dict],
    avg_score_a: float,
    avg_score_b: float,
) -> str:
    """Build the cross-task synthesis prompt."""
    assessments_json = json.dumps(
        [a.to_dict() for a in assessments], indent=2,
    )

    task_descriptions: list[str] = []
    for tid, task in sorted(tasks.items()):
        name = task.get('name', tid)
        desc = task.get('task_definition', {}).get('description', '')
        task_descriptions.append(f'- **{tid}** ({name}): {desc[:200]}')

    wins_a = sum(1 for a in assessments if a.winner == 'A')
    wins_b = sum(1 for a in assessments if a.winner == 'B')
    ties = sum(1 for a in assessments if a.winner == 'tie')

    return f"""You are synthesizing a cross-task qualitative comparison of two code \
generation models to guide model selection and workflow optimization decisions.

## Models
- **A**: {group_a_name}
- **B**: {group_b_name}

## Aggregate Metrics
Wins: A={wins_a}, B={wins_b}, Ties={ties}
Avg composite score: A={avg_score_a:.2f}, B={avg_score_b:.2f}

## Task Descriptions
{chr(10).join(task_descriptions)}

## Per-Task Assessments
{assessments_json}

## Instructions

Synthesize the per-task assessments into an overall qualitative comparison. \
Produce your analysis as JSON.

Focus on actionable insights:
- **Executive summary**: One paragraph capturing the overall comparison.
- **Circumstantial analysis**: Which model to use for which types of tasks and why. \
Be specific about task characteristics (complexity, domain, patterns involved).
- **Cost/quality tradeoff**: Is the cheaper model "good enough" for some tasks? \
Where is the cost/quality frontier?
- **Workflow recommendations**: Suggested changes to the eval pipeline, reviewer \
config, iteration limits, or deployment strategy based on what the comparison reveals.

Be concrete and reference specific tasks and patterns from the assessments. \
Avoid generic advice — ground everything in the evidence."""


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

async def _assess_task(
    task: dict,
    result_a: EvalResult,
    result_b: EvalResult,
    group_a_name: str,
    group_b_name: str,
) -> TaskAssessment:
    """Run the per-task LLM assessor on two results."""
    task_id = task.get('id', 'unknown')
    task_name = task.get('name', task_id)

    # Get diffs
    diff_a = ''
    diff_b = ''
    wt_a = Path(result_a.worktree_path)
    wt_b = Path(result_b.worktree_path)

    if wt_a.is_dir():
        try:
            diff_a = await get_diff(wt_a)
        except Exception as e:
            logger.warning('Failed to get diff for %s (%s): %s', task_id, group_a_name, e)
    else:
        logger.warning('Worktree missing for %s (%s): %s', task_id, group_a_name, wt_a)

    if wt_b.is_dir():
        try:
            diff_b = await get_diff(wt_b)
        except Exception as e:
            logger.warning('Failed to get diff for %s (%s): %s', task_id, group_b_name, e)
    else:
        logger.warning('Worktree missing for %s (%s): %s', task_id, group_b_name, wt_b)

    # Read review artifacts
    reviews_a = _read_review_artifacts(result_a.worktree_path)
    reviews_b = _read_review_artifacts(result_b.worktree_path)

    has_diffs = bool(diff_a or diff_b)
    has_reviews = bool(reviews_a or reviews_b)

    if not has_diffs and not has_reviews:
        logger.warning(
            'No diffs or reviews available for %s — metrics-only assessment',
            task_id,
        )

    prompt = _build_assessor_prompt(
        task, result_a, result_b,
        diff_a, diff_b, reviews_a, reviews_b,
        group_a_name, group_b_name,
    )

    logger.info('Assessing %s: %s vs %s', task_id, group_a_name, group_b_name)

    result = await invoke_agent(
        prompt=prompt,
        system_prompt=(
            'You are an expert code reviewer performing qualitative model comparison. '
            'Be precise, fair, and specific. Always ground your assessment in concrete '
            'code patterns from the diffs.'
        ),
        cwd=Path('/tmp'),
        model='opus',
        effort='high',
        backend='claude',
        max_budget_usd=5.0,
        max_turns=3,
        output_schema=ASSESSOR_SCHEMA,
    )

    try:
        verdict = result.structured_output or json.loads(result.output)
    except (json.JSONDecodeError, TypeError):
        logger.warning('Assessor produced unparseable output for %s: %s',
                       task_id, result.output[:200])
        verdict = {
            'winner': 'tie',
            'confidence': 0.0,
            'summary': 'Assessment failed: unparseable LLM output',
            'strengths_a': [],
            'weaknesses_a': [],
            'strengths_b': [],
            'weaknesses_b': [],
            'circumstances': '',
        }

    return TaskAssessment(
        task_id=task_id,
        task_name=task_name,
        winner=verdict.get('winner', 'tie'),
        confidence=verdict.get('confidence', 0.0),
        summary=verdict.get('summary', ''),
        strengths_a=verdict.get('strengths_a', []),
        weaknesses_a=verdict.get('weaknesses_a', []),
        strengths_b=verdict.get('strengths_b', []),
        weaknesses_b=verdict.get('weaknesses_b', []),
        circumstances=verdict.get('circumstances', ''),
        metrics_a=result_a.metrics,
        metrics_b=result_b.metrics,
    )


async def compare_models(
    results: list[EvalResult],
    group_a_configs: list[str],
    group_b_configs: list[str],
    tasks: dict[str, dict],
    group_a_name: str | None = None,
    group_b_name: str | None = None,
) -> ComparisonReport:
    """Run LLM-powered head-to-head comparison across all tasks.

    *tasks* is a dict of ``{task_id: task_dict}`` loaded from task JSON files.
    """
    group_a_name = group_a_name or group_a_configs[0]
    group_b_name = group_b_name or group_b_configs[0]
    group_a_set = set(group_a_configs)
    group_b_set = set(group_b_configs)

    # Group results by task
    by_task: dict[str, dict[str, list[EvalResult]]] = defaultdict(
        lambda: {'a': [], 'b': []},
    )
    for r in results:
        if r.config_name in group_a_set:
            by_task[r.task_id]['a'].append(r)
        elif r.config_name in group_b_set:
            by_task[r.task_id]['b'].append(r)

    assessments: list[TaskAssessment] = []
    no_contest = 0
    scores_a: list[float] = []
    scores_b: list[float] = []

    for task_id in sorted(by_task):
        best_a = pick_best_result(by_task[task_id]['a'])
        best_b = pick_best_result(by_task[task_id]['b'])

        if best_a:
            scores_a.append(best_a.metrics.get('composite_score', 0.0))
        if best_b:
            scores_b.append(best_b.metrics.get('composite_score', 0.0))

        if best_a is None or best_b is None:
            no_contest += 1
            missing = group_b_name if best_b is None else group_a_name
            logger.info(
                'Skipping %s: %s has no results (no contest)', task_id, missing,
            )
            continue

        task = tasks.get(task_id, {'id': task_id, 'name': task_id})
        assessment = await _assess_task(
            task, best_a, best_b, group_a_name, group_b_name,
        )
        assessments.append(assessment)

    avg_score_a = sum(scores_a) / len(scores_a) if scores_a else 0.0
    avg_score_b = sum(scores_b) / len(scores_b) if scores_b else 0.0
    wins_a = sum(1 for a in assessments if a.winner == 'A')
    wins_b = sum(1 for a in assessments if a.winner == 'B')
    ties = sum(1 for a in assessments if a.winner == 'tie')

    # Run synthesizer if we have assessments
    synthesis = {
        'overall_winner': 'tie',
        'executive_summary': '',
        'circumstantial_analysis': '',
        'cost_quality_tradeoff': '',
        'workflow_recommendations': '',
    }

    if assessments:
        synth_prompt = _build_synthesizer_prompt(
            assessments, group_a_name, group_b_name, tasks,
            avg_score_a, avg_score_b,
        )

        logger.info('Running cross-task synthesis...')
        synth_result = await invoke_agent(
            prompt=synth_prompt,
            system_prompt=(
                'You are a senior engineering manager synthesizing model evaluation '
                'results into actionable recommendations. Be concrete and specific.'
            ),
            cwd=Path('/tmp'),
            model='opus',
            effort='high',
            backend='claude',
            max_budget_usd=5.0,
            max_turns=3,
            output_schema=SYNTHESIZER_SCHEMA,
        )

        try:
            synthesis = synth_result.structured_output or json.loads(synth_result.output)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                'Synthesizer produced unparseable output: %s',
                synth_result.output[:200],
            )

    return ComparisonReport(
        group_a_name=group_a_name,
        group_b_name=group_b_name,
        group_a_configs=group_a_configs,
        group_b_configs=group_b_configs,
        assessments=assessments,
        overall_winner=synthesis.get('overall_winner', 'tie'),
        executive_summary=synthesis.get('executive_summary', ''),
        circumstantial_analysis=synthesis.get('circumstantial_analysis', ''),
        cost_quality_tradeoff=synthesis.get('cost_quality_tradeoff', ''),
        workflow_recommendations=synthesis.get('workflow_recommendations', ''),
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
        no_contest=no_contest,
        avg_score_a=round(avg_score_a, 2),
        avg_score_b=round(avg_score_b, 2),
    )


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def format_comparison_markdown(report: ComparisonReport) -> str:
    """Render the comparison report as markdown for console output."""
    a = report.group_a_name
    b = report.group_b_name
    lines: list[str] = []

    lines.append(f'# Model Comparison: {a} vs {b}')
    lines.append('')
    lines.append('Configs pooled:')
    lines.append(f'  {a}: {", ".join(report.group_a_configs)}')
    lines.append(f'  {b}: {", ".join(report.group_b_configs)}')
    lines.append('')

    # Executive summary
    if report.executive_summary:
        lines.append('## Executive Summary')
        lines.append('')
        lines.append(report.executive_summary)
        lines.append('')

    # Quantitative overview
    lines.append('## Quantitative Overview')
    lines.append('')
    contested = report.wins_a + report.wins_b + report.ties
    lines.append(f'  {"":22s} {a:>15s} {b:>15s}')
    lines.append(f'  {"Wins":22s} {report.wins_a:>15d} {report.wins_b:>15d}')
    lines.append(f'  {"Ties":22s} {report.ties:>15d}')
    if report.no_contest:
        lines.append(f'  {"No contest":22s} {report.no_contest:>15d}')
    lines.append(f'  {"Avg composite score":22s} {report.avg_score_a:>15.2f} {report.avg_score_b:>15.2f}')
    lines.append(f'  {"Contested tasks":22s} {contested:>15d}')
    lines.append('')

    # Per-task assessments
    if report.assessments:
        lines.append('## Per-Task Assessments')
        lines.append('')

        for assessment in report.assessments:
            winner_label = {
                'A': a, 'B': b, 'tie': 'TIE',
            }.get(assessment.winner, '?')

            lines.append(
                f'### {assessment.task_id}: {assessment.task_name} '
                f'— Winner: {winner_label} (confidence: {assessment.confidence:.1f})'
            )
            lines.append('')

            # Metrics summary
            ma = assessment.metrics_a
            mb = assessment.metrics_b
            lines.append(
                f'  {a}: {ma.get("composite_score", 0):.2f} score, '
                f'${ma.get("cost_usd", 0):.2f}, '
                f'{ma.get("iterations", 0)} iter, '
                f'{ma.get("debug_cycles", 0)} debug, '
                f'{ma.get("lines_changed", 0)} lines'
            )
            lines.append(
                f'  {b}: {mb.get("composite_score", 0):.2f} score, '
                f'${mb.get("cost_usd", 0):.2f}, '
                f'{mb.get("iterations", 0)} iter, '
                f'{mb.get("debug_cycles", 0)} debug, '
                f'{mb.get("lines_changed", 0)} lines'
            )
            lines.append('')

            # LLM assessment
            lines.append(assessment.summary)
            lines.append('')

            if assessment.strengths_a:
                lines.append(f'  {a} strengths:')
                for s in assessment.strengths_a:
                    lines.append(f'    - {s}')
            if assessment.weaknesses_a:
                lines.append(f'  {a} weaknesses:')
                for w in assessment.weaknesses_a:
                    lines.append(f'    - {w}')
            if assessment.strengths_b:
                lines.append(f'  {b} strengths:')
                for s in assessment.strengths_b:
                    lines.append(f'    - {s}')
            if assessment.weaknesses_b:
                lines.append(f'  {b} weaknesses:')
                for w in assessment.weaknesses_b:
                    lines.append(f'    - {w}')

            if assessment.circumstances:
                lines.append('')
                lines.append(f'  Circumstances: {assessment.circumstances}')

            lines.append('')

    # Synthesis sections
    if report.circumstantial_analysis:
        lines.append('## Circumstantial Analysis')
        lines.append('')
        lines.append(report.circumstantial_analysis)
        lines.append('')

    if report.cost_quality_tradeoff:
        lines.append('## Cost / Quality Tradeoff')
        lines.append('')
        lines.append(report.cost_quality_tradeoff)
        lines.append('')

    if report.workflow_recommendations:
        lines.append('## Workflow Recommendations')
        lines.append('')
        lines.append(report.workflow_recommendations)
        lines.append('')

    return '\n'.join(lines)
