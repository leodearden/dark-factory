"""Report generation for the reviewer panel trial.

Produces a structured report comparing variant performance across
the evaluation corpus, with per-language breakdowns and cost analysis.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from orchestrator.evals.reviewer_trial.corpus import CorpusManifest
from orchestrator.evals.reviewer_trial.scorer import ScoringResult
from orchestrator.evals.reviewer_trial.variants import VariantConfig


@dataclass
class VariantSummary:
    """Aggregated metrics for one variant across all corpus diffs."""

    name: str
    description: str
    num_reviewers: int
    mean_recall: float = 0.0
    mean_precision: float = 0.0
    mean_f1: float = 0.0
    mean_blocking_recall: float = 0.0
    total_cost_usd: float = 0.0
    mean_wall_clock_ms: float = 0.0
    by_language: dict[str, dict[str, float]] = field(default_factory=dict)
    by_source: dict[str, dict[str, float]] = field(default_factory=dict)
    f1_per_dollar: float = 0.0
    blocking_recall_per_dollar: float = 0.0


@dataclass
class HeadToHead:
    """Pairwise comparison between two variants."""

    variant_a: str
    variant_b: str
    a_wins: list[str] = field(default_factory=list)   # diff_ids where A > B
    b_wins: list[str] = field(default_factory=list)   # diff_ids where B > A
    ties: list[str] = field(default_factory=list)      # diff_ids where A == B


@dataclass
class TrialReport:
    """Complete trial report."""

    summaries: list[VariantSummary]
    head_to_heads: list[HeadToHead]
    missed_by_all: list[str]     # ground truth IDs missed by every variant
    per_diff_scores: dict[str, dict[str, float]]  # diff_id -> {variant -> f1}
    total_cost_usd: float = 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group_scores_by_variant(
    scores: list[ScoringResult],
) -> dict[str, list[ScoringResult]]:
    grouped: dict[str, list[ScoringResult]] = defaultdict(list)
    for s in scores:
        grouped[s.variant_name].append(s)
    return dict(grouped)


def _build_variant_summary(
    variant: VariantConfig,
    scores: list[ScoringResult],
    corpus: CorpusManifest,
) -> VariantSummary:
    recalls = [s.recall for s in scores]
    precisions = [s.precision for s in scores]
    f1s = [s.f1 for s in scores]
    blocking_recalls = [s.blocking_recall for s in scores]
    costs = [s.cost_usd for s in scores]
    wall_clocks = [s.wall_clock_ms for s in scores]

    total_cost = sum(costs)

    # By language
    by_language: dict[str, dict[str, float]] = {}
    for lang in ('python', 'rust', 'typescript'):
        lang_diffs = {d.diff_id for d in corpus.filter_by_language(lang)}
        lang_scores = [s for s in scores if s.diff_id in lang_diffs]
        if lang_scores:
            by_language[lang] = {
                'recall': round(_mean([s.recall for s in lang_scores]), 4),
                'precision': round(_mean([s.precision for s in lang_scores]), 4),
                'f1': round(_mean([s.f1 for s in lang_scores]), 4),
                'blocking_recall': round(_mean([s.blocking_recall for s in lang_scores]), 4),
            }

    # By source
    by_source: dict[str, dict[str, float]] = {}
    for source in ('synthetic', 'real_world'):
        src_diffs = {d.diff_id for d in corpus.filter_by_source(source)}
        src_scores = [s for s in scores if s.diff_id in src_diffs]
        if src_scores:
            by_source[source] = {
                'recall': round(_mean([s.recall for s in src_scores]), 4),
                'precision': round(_mean([s.precision for s in src_scores]), 4),
                'f1': round(_mean([s.f1 for s in src_scores]), 4),
                'blocking_recall': round(_mean([s.blocking_recall for s in src_scores]), 4),
            }

    mean_f1 = _mean(f1s)
    mean_br = _mean(blocking_recalls)

    return VariantSummary(
        name=variant.name,
        description=variant.description,
        num_reviewers=len(variant.reviewers),
        mean_recall=round(_mean(recalls), 4),
        mean_precision=round(_mean(precisions), 4),
        mean_f1=round(mean_f1, 4),
        mean_blocking_recall=round(mean_br, 4),
        total_cost_usd=round(total_cost, 2),
        mean_wall_clock_ms=round(_mean(wall_clocks), 0),
        by_language=by_language,
        by_source=by_source,
        f1_per_dollar=round(mean_f1 / total_cost, 4) if total_cost > 0 else 0.0,
        blocking_recall_per_dollar=round(mean_br / total_cost, 4) if total_cost > 0 else 0.0,
    )


def _build_head_to_heads(
    scores: list[ScoringResult],
    variants: list[VariantConfig],
) -> list[HeadToHead]:
    """Compare every pair of variants on each diff."""
    by_variant = _group_scores_by_variant(scores)
    results: list[HeadToHead] = []

    variant_names = [v.name for v in variants]
    for i, a_name in enumerate(variant_names):
        for b_name in variant_names[i + 1:]:
            a_scores = {s.diff_id: s.f1 for s in by_variant.get(a_name, [])}
            b_scores = {s.diff_id: s.f1 for s in by_variant.get(b_name, [])}
            all_diffs = set(a_scores) | set(b_scores)

            h2h = HeadToHead(variant_a=a_name, variant_b=b_name)
            for diff_id in sorted(all_diffs):
                a_f1 = a_scores.get(diff_id, 0.0)
                b_f1 = b_scores.get(diff_id, 0.0)
                if abs(a_f1 - b_f1) < 0.01:
                    h2h.ties.append(diff_id)
                elif a_f1 > b_f1:
                    h2h.a_wins.append(diff_id)
                else:
                    h2h.b_wins.append(diff_id)
            results.append(h2h)

    return results


def build_trial_report(
    scores: list[ScoringResult],
    variants: list[VariantConfig],
    corpus: CorpusManifest,
) -> TrialReport:
    """Build the full trial report from scoring results."""
    by_variant = _group_scores_by_variant(scores)

    summaries = [
        _build_variant_summary(v, by_variant.get(v.name, []), corpus)
        for v in variants
    ]
    # Sort by mean F1 descending
    summaries.sort(key=lambda s: s.mean_f1, reverse=True)

    head_to_heads = _build_head_to_heads(scores, variants)

    # Per-diff heatmap
    per_diff: dict[str, dict[str, float]] = defaultdict(dict)
    for s in scores:
        per_diff[s.diff_id][s.variant_name] = s.f1

    # Ground-truth issues missed by ALL variants
    all_gt_ids: set[str] = set()
    found_by_any: set[str] = set()
    for diff in corpus.diffs:
        for gt in diff.ground_truth:
            all_gt_ids.add(gt.id)
    for s in scores:
        for m in s.matches:
            found_by_any.add(m.ground_truth_id)
    missed_by_all = sorted(all_gt_ids - found_by_any)

    total_cost = sum(s.cost_usd for s in scores)

    return TrialReport(
        summaries=summaries,
        head_to_heads=head_to_heads,
        missed_by_all=missed_by_all,
        per_diff_scores=dict(per_diff),
        total_cost_usd=round(total_cost, 2),
    )


def format_markdown(report: TrialReport) -> str:
    """Format trial report as markdown."""
    lines: list[str] = []

    lines.append('# Reviewer Panel Trial Report\n')
    lines.append(f'Total trial cost: **${report.total_cost_usd:.2f}**\n')

    # 1. Variant leaderboard
    lines.append('## Variant Leaderboard\n')
    lines.append('| Rank | Variant | Reviewers | Mean F1 | Blocking Recall | Cost | F1/$ | BR/$ |')
    lines.append('|------|---------|-----------|---------|-----------------|------|------|------|')
    for i, s in enumerate(report.summaries, 1):
        lines.append(
            f'| {i} | {s.name} | {s.num_reviewers} | {s.mean_f1:.3f} | '
            f'{s.mean_blocking_recall:.3f} | ${s.total_cost_usd:.2f} | '
            f'{s.f1_per_dollar:.3f} | {s.blocking_recall_per_dollar:.3f} |'
        )
    lines.append('')

    # 2. Per-variant breakdown by language
    lines.append('## Per-Language Breakdown\n')
    for s in report.summaries:
        lines.append(f'### {s.name}\n')
        if s.by_language:
            lines.append('| Language | F1 | Recall | Precision | Blocking Recall |')
            lines.append('|----------|----|--------|-----------|-----------------|')
            for lang, metrics in sorted(s.by_language.items()):
                lines.append(
                    f'| {lang} | {metrics["f1"]:.3f} | {metrics["recall"]:.3f} | '
                    f'{metrics["precision"]:.3f} | {metrics["blocking_recall"]:.3f} |'
                )
            lines.append('')
        if s.by_source:
            lines.append('| Source | F1 | Recall | Precision | Blocking Recall |')
            lines.append('|--------|----|--------|-----------|-----------------|')
            for source, metrics in sorted(s.by_source.items()):
                lines.append(
                    f'| {source} | {metrics["f1"]:.3f} | {metrics["recall"]:.3f} | '
                    f'{metrics["precision"]:.3f} | {metrics["blocking_recall"]:.3f} |'
                )
            lines.append('')

    # 3. Per-diff heatmap
    lines.append('## Per-Diff F1 Scores\n')
    variant_names = [s.name for s in report.summaries]
    header = '| Diff | ' + ' | '.join(variant_names) + ' |'
    sep = '|------|' + '|'.join(['------'] * len(variant_names)) + '|'
    lines.append(header)
    lines.append(sep)
    for diff_id in sorted(report.per_diff_scores):
        scores = report.per_diff_scores[diff_id]
        cells = [f'{scores.get(v, 0.0):.3f}' for v in variant_names]
        lines.append(f'| {diff_id} | ' + ' | '.join(cells) + ' |')
    lines.append('')

    # 4. Head-to-head
    lines.append('## Head-to-Head Comparisons\n')
    for h2h in report.head_to_heads:
        lines.append(f'### {h2h.variant_a} vs {h2h.variant_b}\n')
        lines.append(f'- **{h2h.variant_a} wins**: {len(h2h.a_wins)} diffs ({", ".join(h2h.a_wins) or "none"})')
        lines.append(f'- **{h2h.variant_b} wins**: {len(h2h.b_wins)} diffs ({", ".join(h2h.b_wins) or "none"})')
        lines.append(f'- **Ties**: {len(h2h.ties)} diffs')
        lines.append('')

    # 5. Failure analysis
    if report.missed_by_all:
        lines.append('## Issues Missed by ALL Variants\n')
        for gt_id in report.missed_by_all:
            lines.append(f'- `{gt_id}`')
        lines.append('')

    return '\n'.join(lines)


def save_report(report: TrialReport, path: Path) -> Path:
    """Save report as both JSON and markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = path.with_suffix('.json')
    json_path.write_text(json.dumps(asdict(report), indent=2, default=str))

    # Markdown
    md_path = path.with_suffix('.md')
    md_path.write_text(format_markdown(report))

    return md_path
