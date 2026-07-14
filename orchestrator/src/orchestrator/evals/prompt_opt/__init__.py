"""Bespoke SkillOpt-discipline prompt-optimization loop (T6).

See plans/tier1-prompt-optimization-prd.md T6. This package generalizes the
reviewer_trial rollout+scoring engine into a corpus-agnostic optimization
loop over a pluggable ``(corpus, scorer, executor_model, heuristics_block)``.

Public re-exports are populated in engine.py's implementation step; until
then this is just an (empty) package marker so ``orchestrator.evals.prompt_opt``
and its submodules are importable.
"""

from __future__ import annotations
