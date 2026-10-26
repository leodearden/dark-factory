"""Bespoke SkillOpt-discipline prompt-optimization loop (T6) + T8 canary.

See plans/tier1-prompt-optimization-prd.md T6/T8. This package generalizes
the reviewer_trial rollout+scoring engine into a corpus-agnostic optimization
loop over a pluggable ``(corpus, scorer, executor_model, heuristics_block)``,
and (T8) a post-deploy canary that compares pipeline-level ``runs.db``
metrics against a rolling baseline to catch the MAS net-negative failure mode.
"""
