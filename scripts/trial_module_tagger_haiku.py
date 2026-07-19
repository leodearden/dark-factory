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

import json
from typing import Any

from orchestrator.harness import _extract_tagger_entries
from orchestrator.module_charter import sanitize_files_for_persist
from orchestrator.module_tagger_prompt import build_tagger_prompt


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


def build_replay_input(task: dict, top_level_dirs: list) -> dict:
    """Build the per-task summary the tagger replays over.

    Mirrors ``harness._tag_task_modules`` exactly: ``id`` is stringified, and
    ``description`` falls back to ``details`` (then to ``''``) so the tagger
    still has context when the description is empty. *top_level_dirs* is accepted
    for call-signature uniformity with ``faithful_prompt_for`` (the dir listing
    is embedded by the shared prompt builder, not by this per-task summary).
    """
    return {
        'id': str(task.get('id', '')),
        'title': task.get('title', ''),
        'description': task.get('description') or task.get('details') or '',
    }


def faithful_prompt_for(task: dict, top_level_dirs: list) -> str:
    """Build the module-tagger prompt for a SINGLE task, faithfully.

    Delegates to the shared production builder
    (``module_tagger_prompt.build_tagger_prompt``) over a single-element summary
    list, so the replay's prompt is byte-identical to what production would send
    "at the call site" — no drifting hand-copy.
    """
    summary = build_replay_input(task, top_level_dirs)
    return build_tagger_prompt(top_level_dirs, [summary])


def parse_predictions(agent_result: Any) -> list[str]:
    """Extract the predicted file list from a single-task tagger replay result.

    Reads ``AgentResult.structured_output`` first (falling back to
    ``json.loads(result.output)``), then reuses ``harness._extract_tagger_entries``
    to peel known StructuredOutput wrapper keys ('predictions'/legacy 'tasks')
    before flattening the entries' ``files`` into one list. A missing/unparseable
    payload yields ``[]`` (fail-safe, matching the harness bad-output early
    return). The replay invokes one task at a time, so the flattened list is that
    task's prediction.
    """
    payload = getattr(agent_result, 'structured_output', None)
    if not payload:
        output = getattr(agent_result, 'output', None)
        try:
            payload = json.loads(output) if output else None
        except (json.JSONDecodeError, TypeError):
            payload = None
    if not payload:
        return []

    files: list[str] = []
    for entry in _extract_tagger_entries(payload):
        if isinstance(entry, dict):
            files.extend(entry.get('files') or [])
    return files
