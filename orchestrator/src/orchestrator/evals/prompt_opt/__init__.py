"""Bespoke SkillOpt-discipline prompt-optimization loop (T6).

See plans/tier1-prompt-optimization-prd.md T6. This package generalizes the
reviewer_trial rollout+scoring engine into a corpus-agnostic optimization
loop over a pluggable ``(corpus, scorer, executor_model, heuristics_block)``.
"""

from __future__ import annotations

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
    'CorpusSplit',
    'LoopResult',
    'ProposeFn',
    'RolloutFn',
    'ScoredItem',
    'Scorer',
    'evaluate_acceptance',
    'measure_repeatability_band',
    'paired_delta',
    'propose_heuristics_edit',
    'run_optimization_loop',
    'split_corpus',
    'textual_lr',
]
