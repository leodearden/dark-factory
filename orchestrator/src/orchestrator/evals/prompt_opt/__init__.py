"""Bespoke SkillOpt-discipline prompt-optimization loop (T6) + T8 canary.

See plans/tier1-prompt-optimization-prd.md T6/T8. This package generalizes
the reviewer_trial rollout+scoring engine into a corpus-agnostic optimization
loop over a pluggable ``(corpus, scorer, executor_model, heuristics_block)``,
and (T8) a post-deploy canary that compares pipeline-level ``runs.db``
metrics against a rolling baseline to catch the MAS net-negative failure mode.
"""

from __future__ import annotations

from orchestrator.evals.prompt_opt.canary import (
    CanaryThresholds,
    CanaryVerdict,
    MetricComparison,
    WindowMetrics,
    compare_windows,
    compute_window_metrics,
    load_window_rows,
    run_canary,
)
from orchestrator.evals.prompt_opt.curator_corpus import (
    AuditReport,
    CuratorCorpusManifest,
    CuratorReplayItem,
    audit_curator_corpus,
    build_curator_corpus,
    propose_curator_label_frontier,
    read_curator_decisions,
    select_spot_check_subset,
)
from orchestrator.evals.prompt_opt.curator_scorer import CuratorActionScorer
from orchestrator.evals.prompt_opt.engine import LoopResult, run_optimization_loop
from orchestrator.evals.prompt_opt.optimizer import propose_heuristics_edit
from orchestrator.evals.prompt_opt.scorer import ProposeFn, RolloutFn, ScoredItem, Scorer
from orchestrator.evals.prompt_opt.splits import CorpusSplit, split_corpus
from orchestrator.evals.prompt_opt.variance import (
    AcceptanceRecord,
    evaluate_acceptance,
    measure_repeatability_band,
    paired_delta,
    textual_lr,
)

__all__ = [
    'AcceptanceRecord',
    'AuditReport',
    'CanaryThresholds',
    'CanaryVerdict',
    'CorpusSplit',
    'CuratorActionScorer',
    'CuratorCorpusManifest',
    'CuratorReplayItem',
    'LoopResult',
    'MetricComparison',
    'ProposeFn',
    'RolloutFn',
    'ScoredItem',
    'Scorer',
    'WindowMetrics',
    'audit_curator_corpus',
    'build_curator_corpus',
    'compare_windows',
    'compute_window_metrics',
    'evaluate_acceptance',
    'load_window_rows',
    'measure_repeatability_band',
    'paired_delta',
    'propose_curator_label_frontier',
    'propose_heuristics_edit',
    'read_curator_decisions',
    'run_canary',
    'run_optimization_loop',
    'select_spot_check_subset',
    'split_corpus',
    'textual_lr',
]
