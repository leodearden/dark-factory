"""Recurring-human-gate dedupe guard (task 3588).

Bounds the population of recon-filed human-gate carrier tasks by refusing,
at the ``submit_task`` boundary, to mint a SECOND open gate for a subject
that already has one.

## The measured evidence

Reconciliation filed one carrier per cycle for the same subject: task 5879
accumulated carriers 5902 → 5916 → 5929 (and subject 5858 got 5881, 5917).
5929's own metadata carried ``prior_escalation_tasks=[5916, 5902]`` — the
filing agent demonstrably KNEW about both predecessors and filed anyway.
That is why a prompt-only rule is insufficient and this needs a code
chokepoint: the LLM's awareness of the duplicates was not the binding
constraint.

## Leaf contract

This is a LEAF ``middleware`` module: it imports nothing from
``reconciliation/`` or ``server/`` (the same contract
``live_task_write_guard.py`` states and realizes for itself). The task
corpus is supplied by an INJECTED async callable, so the module never
reaches for :class:`~fused_memory.middleware.task_interceptor.TaskInterceptor`
— the import cycle that already forced ``operational_routing_guard.py``
into a lazy in-function import — and every unit test can fake the corpus
with a one-line ``AsyncMock``.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    'GATE_SUBJECT_ALIASES',
    'GATE_SUBJECT_KEY',
    'extract_gate_subject',
    'is_gate_submission',
]

# The canonical metadata key a recon-filed human gate uses to name the
# subject it is gating. Rendered into the recon prompt authority by
# reconciliation/recon_self_model.py::render_source_completion_section.
GATE_SUBJECT_KEY = 'gate_subject'

# Resolution order. The two trailing names are READ-SIDE aliases honoured for
# already-filed carriers (5902/5916/5929 key their subject via
# ``stranded_task_id``); no already-filed task's metadata is ever rewritten,
# and the canonical key always wins when both are present. They are also
# honoured on the INCOMING submission so a Stage-2 run that has not yet
# internalized the new prompt is deduped anyway.
GATE_SUBJECT_ALIASES: tuple[str, ...] = (
    GATE_SUBJECT_KEY,
    'stranded_task_id',
    'related_task_id',
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_metadata(metadata: Any) -> dict:
    """Return *metadata* as a dict (best-effort; unknown shapes → empty dict).

    Behaviourally identical to ``execution_class_guard._parse_metadata`` —
    None → {}; empty string → {}; dict → returned as-is (callers that need
    to mutate must copy first); JSON string → parsed dict, or {} if the
    parse fails or yields a non-dict value; anything else → {}. Kept in
    lockstep deliberately: the two guards run back-to-back on the same
    ``submit_task`` payload and must never disagree about a malformed blob.
    """
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        if not metadata:
            return {}
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


# ---------------------------------------------------------------------------
# Public submission readers
# ---------------------------------------------------------------------------


def extract_gate_subject(metadata: str | dict[str, Any] | None) -> str | None:
    """Return the dedupe subject declared by *metadata*, or ``None``.

    Walks :data:`GATE_SUBJECT_ALIASES` in order and returns the first
    non-empty scalar, stripped and coerced to ``str``. Only ``str`` and
    ``int`` are accepted — ``bool`` is excluded EXPLICITLY because it is an
    ``int`` subclass, so ``gate_subject=True`` would otherwise become the
    subject ``'True'`` and dedupe two unrelated gates against each other.
    Dicts, lists and ``None`` are likewise not subjects; a key holding one
    is skipped so a later alias can still answer.
    """
    meta = _parse_metadata(metadata)
    for key in GATE_SUBJECT_ALIASES:
        value = meta.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            continue
        subject = str(value).strip()
        if subject:
            return subject
    return None


def is_gate_submission(metadata: str | dict[str, Any] | None) -> bool:
    """True when *metadata* declares a human-gate carrier submission.

    ``execution_class == 'operational'`` AND ``operational_mode`` is
    ``'gate'`` — where ABSENT counts as ``'gate'``. That default is not a
    convenience: ``TaskMetadata.operational_mode`` is declared
    ``Literal['gate', 'llm'] = 'gate'``, and ``inject_operational_routing``
    coerces "operational + gate/absent" into a deterministic pure gate, so
    an ``operational`` submission that merely OMITS the key still becomes a
    born-at-L2 critical ``milestone_gate`` on dispatch — exactly the
    population this guard exists to bound.

    The comparison is VALUE-sensitive, mirroring
    ``TaskInterceptor._is_gate_metadata``: it tests ``== 'gate'`` rather
    than truthiness, so ``operational_mode=1`` or ``'false'`` never
    satisfies the predicate (``bool('false')`` is True and would silently
    accept the opposite of the caller's intent).
    """
    meta = _parse_metadata(metadata)
    if meta.get('execution_class') != 'operational':
        return False
    return meta.get('operational_mode', 'gate') == 'gate'
