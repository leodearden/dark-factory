#!/usr/bin/env python3
"""Offline haiku-vs-sonnet replay-agreement trial for the ``module_tagger`` role.

Task 2540 (Routing κ). A self-contained, decide-and-act pilot: replay historical
DONE tasks (which carry ground-truth ``metadata.files`` = the actual merge-diff
files) through BOTH haiku and a fresh sonnet run at the production call site,
score each model's file predictions vs ground truth (precision/recall/F1) and
haiku-vs-sonnet agreement (Jaccard), frontier-adjudicate (opus) the
disagreements, and compute a pass/marginal/fail verdict via documented
thresholds. On a clear PASS the trial flips ``module_tagger`` to haiku via a
hot-reload of ``dark-factory-orchestrator.yaml``; otherwise it escalates with the
measured numbers. Both branches always commit the markdown report.

The replay reuses byte-identical production inputs by importing the shared
``orchestrator.module_tagger_prompt`` builder used by ``harness._tag_task_modules``
(single source of truth — no drifting hand-copy).

Deterministic core (set_scores, build_replay_input, faithful_prompt_for,
parse_predictions, adjudication, decide, render_report, run_trial) is unit-tested
in tests/scripts/test_trial_module_tagger_haiku.py with a FAKE invoke_fn over
fixtures. The live model run + config flip are impl-only (live models +
conditional config).
"""
from __future__ import annotations

from orchestrator.module_charter import sanitize_files_for_persist


def set_scores(predicted: list, gold: list) -> dict[str, float]:
    """Score a predicted file set against a gold file set.

    Both sides are first passed through ``sanitize_files_for_persist`` (the
    production WRITE-path normalizer), so scoring happens on the same
    file-level sets production actually locks on — directory-shaped entries are
    stripped symmetrically and never count.

    Returns ``{precision, recall, f1, jaccard}`` over the sanitized sets:

    - ``precision`` = |P∩G| / |P|  (fraction of predictions that were correct)
    - ``recall``    = |P∩G| / |G|  (fraction of gold files that were predicted)
    - ``f1``        = harmonic mean of precision and recall
    - ``jaccard``   = |P∩G| / |P∪G|  (symmetric overlap)

    Edge cases are total (no ZeroDivision): when a denominator would be zero,
    the metric is 1.0 iff BOTH sets are empty (vacuously perfect) else 0.0.
    """
    p = set(sanitize_files_for_persist(list(predicted)))
    g = set(sanitize_files_for_persist(list(gold)))
    inter = len(p & g)
    union = len(p | g)

    precision = inter / len(p) if p else (1.0 if not g else 0.0)
    recall = inter / len(g) if g else (1.0 if not p else 0.0)
    jaccard = inter / union if union else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1, 'jaccard': jaccard}
