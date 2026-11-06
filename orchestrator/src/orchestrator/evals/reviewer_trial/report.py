"""Report generation for the reviewer panel trial.

Produces a structured report comparing variant performance across
the evaluation corpus, with per-language breakdowns and cost analysis.
Every figure is over measured scores; a score whose matcher did not measure
is counted beside them, and its spend is still in the trial total.
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
    mean_recall: float | None = None
    mean_precision: float | None = None
    mean_f1: float | None = None
    mean_blocking_recall: float | None = None
    total_cost_usd: float = 0.0
    mean_wall_clock_ms: float | None = None
    by_language: dict[str, dict[str, float]] = field(default_factory=dict)
    by_source: dict[str, dict[str, float]] = field(default_factory=dict)
    f1_per_dollar: float | None = None
    blocking_recall_per_dollar: float | None = None
    unmeasured: int = 0
    unmeasured_cost_usd: float = 0.0


@dataclass
class HeadToHead:
    """Pairwise comparison between two variants."""

    variant_a: str
    variant_b: str
    a_wins: list[str] = field(default_factory=list)   # diff_ids where A > B
    b_wins: list[str] = field(default_factory=list)   # diff_ids where B > A
    ties: list[str] = field(default_factory=list)      # diff_ids where A == B
    unmeasured: list[str] = field(default_factory=list)  # diff_ids either side did not measure


@dataclass
class TrialReport:
    """Complete trial report."""

    summaries: list[VariantSummary]
    head_to_heads: list[HeadToHead]
    missed_by_all: list[str]     # ground truth IDs missed by every variant
    per_diff_scores: dict[str, dict[str, float | None]]  # diff_id -> {variant -> f1, None if unmeasured}
    total_cost_usd: float = 0.0


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rounded(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _per_dollar(mean_value: float | None, cost: float) -> float | None:
    if mean_value is None:
        return None
    return round(mean_value / cost, 4) if cost > 0 else 0.0


def _partition(
    scores: list[ScoringResult],
) -> tuple[list[ScoringResult], list[ScoringResult]]:
    """``(measured, unmeasured)`` — the one home of that rule in this module."""
    measured = [s for s in scores if s.unmeasured is None]
    unmeasured = [s for s in scores if s.unmeasured is not None]
    return measured, unmeasured


def _group_metrics(scores: list[ScoringResult]) -> dict[str, float]:
    return {
        'recall': round(sum(s.recall for s in scores) / len(scores), 4),
        'precision': round(sum(s.precision for s in scores) / len(scores), 4),
        'f1': round(sum(s.f1 for s in scores) / len(scores), 4),
        'blocking_recall': round(sum(s.blocking_recall for s in scores) / len(scores), 4),
    }


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
    measured, unmeasured = _partition(scores)
    total_cost = sum(s.total_cost_usd for s in measured)

    by_language: dict[str, dict[str, float]] = {}
    for lang in ('python', 'rust', 'typescript'):
        lang_diffs = {d.diff_id for d in corpus.filter_by_language(lang)}
        lang_scores = [s for s in measured if s.diff_id in lang_diffs]
        if lang_scores:
            by_language[lang] = _group_metrics(lang_scores)

    by_source: dict[str, dict[str, float]] = {}
    for source in ('synthetic', 'real_world'):
        src_diffs = {d.diff_id for d in corpus.filter_by_source(source)}
        src_scores = [s for s in measured if s.diff_id in src_diffs]
        if src_scores:
            by_source[source] = _group_metrics(src_scores)

    mean_f1 = _mean([s.f1 for s in measured])
    mean_br = _mean([s.blocking_recall for s in measured])

    return VariantSummary(
        name=variant.name,
        description=variant.description,
        num_reviewers=len(variant.reviewers),
        mean_recall=_rounded(_mean([s.recall for s in measured]), 4),
        mean_precision=_rounded(_mean([s.precision for s in measured]), 4),
        mean_f1=_rounded(mean_f1, 4),
        mean_blocking_recall=_rounded(mean_br, 4),
        total_cost_usd=round(total_cost, 2),
        mean_wall_clock_ms=_rounded(_mean([s.wall_clock_ms for s in measured]), 0),
        by_language=by_language,
        by_source=by_source,
        f1_per_dollar=_per_dollar(mean_f1, total_cost),
        blocking_recall_per_dollar=_per_dollar(mean_br, total_cost),
        unmeasured=len(unmeasured),
        unmeasured_cost_usd=round(sum(s.total_cost_usd for s in unmeasured), 2),
    )


def _f1_by_diff(scores: list[ScoringResult]) -> tuple[dict[str, float], set[str]]:
    """Measured F1 per diff, and the diffs that have only unmeasured scores."""
    measured, unmeasured = _partition(scores)
    f1s = {s.diff_id: s.f1 for s in measured}
    return f1s, {s.diff_id for s in unmeasured} - set(f1s)


def _build_head_to_heads(
    scores: list[ScoringResult],
    variants: list[VariantConfig],
) -> list[HeadToHead]:
    """Compare every pair of variants on each diff both of them measured."""
    by_variant = _group_scores_by_variant(scores)
    results: list[HeadToHead] = []

    variant_names = [v.name for v in variants]
    for i, a_name in enumerate(variant_names):
        for b_name in variant_names[i + 1:]:
            a_scores, a_unmeasured = _f1_by_diff(by_variant.get(a_name, []))
            b_scores, b_unmeasured = _f1_by_diff(by_variant.get(b_name, []))
            unmeasured = a_unmeasured | b_unmeasured
            all_diffs = set(a_scores) | set(b_scores) | unmeasured

            h2h = HeadToHead(variant_a=a_name, variant_b=b_name)
            for diff_id in sorted(all_diffs):
                if diff_id in unmeasured:
                    h2h.unmeasured.append(diff_id)
                    continue
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
    # Sort by mean F1 descending; a variant with nothing measured sorts last
    summaries.sort(key=lambda s: (s.mean_f1 is not None, s.mean_f1 or 0.0), reverse=True)

    head_to_heads = _build_head_to_heads(scores, variants)

    # Per-diff heatmap; a cell with no measured score is None
    measured, unmeasured = _partition(scores)
    per_diff: dict[str, dict[str, float | None]] = defaultdict(dict)
    for s in measured:
        per_diff[s.diff_id][s.variant_name] = s.f1
    for s in unmeasured:
        per_diff[s.diff_id].setdefault(s.variant_name, None)

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

    total_cost = sum(s.total_cost_usd for s in scores)

    return TrialReport(
        summaries=summaries,
        head_to_heads=head_to_heads,
        missed_by_all=missed_by_all,
        per_diff_scores=dict(per_diff),
        total_cost_usd=round(total_cost, 2),
    )


def _fmt(value: float | None, spec: str) -> str:
    return 'n/a' if value is None else format(value, spec)


def format_markdown(report: TrialReport) -> str:
    """Format trial report as markdown."""
    lines: list[str] = []

    lines.append('# Reviewer Panel Trial Report\n')
    lines.append(f'Total trial cost: **${report.total_cost_usd:.2f}**\n')

    # 1. Variant leaderboard
    lines.append('## Variant Leaderboard\n')
    lines.append('| Rank | Variant | Reviewers | Mean F1 | Blocking Recall | Cost | F1/$ | BR/$ | Unmeasured |')
    lines.append('|------|---------|-----------|---------|-----------------|------|------|------|------------|')
    for i, s in enumerate(report.summaries, 1):
        lines.append(
            f'| {i} | {s.name} | {s.num_reviewers} | {_fmt(s.mean_f1, ".3f")} | '
            f'{_fmt(s.mean_blocking_recall, ".3f")} | ${s.total_cost_usd:.2f} | '
            f'{_fmt(s.f1_per_dollar, ".3f")} | {_fmt(s.blocking_recall_per_dollar, ".3f")} | '
            f'{s.unmeasured} |'
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
        cells = [_fmt(scores.get(v), '.3f') for v in variant_names]
        lines.append(f'| {diff_id} | ' + ' | '.join(cells) + ' |')
    lines.append('')

    # 4. Head-to-head
    lines.append('## Head-to-Head Comparisons\n')
    for h2h in report.head_to_heads:
        lines.append(f'### {h2h.variant_a} vs {h2h.variant_b}\n')
        lines.append(f'- **{h2h.variant_a} wins**: {len(h2h.a_wins)} diffs ({", ".join(h2h.a_wins) or "none"})')
        lines.append(f'- **{h2h.variant_b} wins**: {len(h2h.b_wins)} diffs ({", ".join(h2h.b_wins) or "none"})')
        lines.append(f'- **Ties**: {len(h2h.ties)} diffs')
        if h2h.unmeasured:
            lines.append(f'- **Unmeasured**: {len(h2h.unmeasured)} diffs')
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
