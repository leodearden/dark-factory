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
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

from shared.task_statuses import TERMINAL

__all__ = [
    'GATE_SUBJECT_ALIASES',
    'GATE_SUBJECT_KEY',
    'extract_gate_subject',
    'find_open_gate',
    'is_gate_submission',
    'recurring_gate_error',
    'recurring_gate_guard_error',
]

logger = logging.getLogger(__name__)

# Injected task-corpus reader. Mirrors live_task_write_guard's GetTaskFn /
# DoWriteFn / FileFindingFn aliases: the concrete callable is bound at the
# tools.py call site to task_interceptor.get_tasks, so this module stays a
# leaf and every unit test can supply a one-line AsyncMock.
GetTasksFn = Callable[[], Awaitable[Any]]

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


# ---------------------------------------------------------------------------
# Corpus predicate
# ---------------------------------------------------------------------------


def _iter_task_rows(result: Any) -> Iterable[Any]:
    """Normalize a raw ``get_tasks`` return into an iterable of task rows.

    ``{'tasks': [...]}`` → the list; a bare list → itself; anything else →
    ``()``. No flattening is performed and
    ``task_curator.flatten_task_tree`` is deliberately NOT imported: all
    task rows are top-level post-DF-D (see ``sqlite_task_backend``'s
    ``_row_to_task``, "All tasks are top-level after DF-D"), so a leaf guard
    would be coupling itself to the curator module for nothing.
    """
    if isinstance(result, Mapping):
        tasks = result.get('tasks')
        return tasks if isinstance(tasks, list) else ()
    if isinstance(result, list):
        return result
    return ()


def find_open_gate(tasks: Any, subject: str) -> dict[str, Any] | None:
    """Return the first NON-TERMINAL gate carrier for *subject*, else ``None``.

    Pure and total: *tasks* is any iterable of task-row mappings (or a raw
    ``get_tasks`` return, or ``None``), and no branch raises on a malformed
    row — a corpus arriving from the backend is treated as untrusted.

    The terminal set is imported from :data:`shared.task_statuses.TERMINAL`
    rather than re-spelled (the same single-sourcing ``live_task_write_guard``
    does), so a carrier that has been ``done``/``cancelled`` correctly stops
    blocking a genuinely fresh gate for the same subject.

    Terminal filtering is re-checked here even though the tools.py call site
    already narrows the query with ``statuses=``: this is a PURE function
    contracted to be correct on ANY caller-supplied corpus, and a future
    caller passing an unfiltered one must not silently get a wrong answer.
    """
    for row in _iter_task_rows(tasks):
        if not isinstance(row, Mapping):
            continue
        if str(row.get('status', '')) in TERMINAL:
            continue
        metadata = row.get('metadata')
        if extract_gate_subject(metadata) != subject:
            continue
        if not is_gate_submission(metadata):
            continue
        task_id = str(row.get('id', '') or '')
        if not task_id:
            continue
        return {
            'id': task_id,
            'title': str(row.get('title', '') or ''),
            'status': str(row.get('status', '') or ''),
        }
    return None


# ---------------------------------------------------------------------------
# Rejection payload
# ---------------------------------------------------------------------------


def recurring_gate_error(
    existing: Mapping[str, Any],
    subject: str,
) -> dict[str, Any]:
    """Return a structured RecurringGateViolation error dict.

    Shape mirrors ``lock_charter_guard.lock_charter_error`` (structured
    ``error`` + ``error_type`` + a domain-specific machine-readable field +
    ``hint``) so MCP callers handle this rejection uniformly with the
    existing ones. ``existing_gate_task_id`` and ``gate_subject`` are what
    ``directory_paths`` is there — the machine-readable half, so no consumer
    has to regex the prose to learn which carrier to amend.

    Total: a carrier mapping missing its ``title`` still yields a
    well-formed error naming the id.
    """
    task_id = str(existing.get('id', '') or '')
    title = str(existing.get('title', '') or '')
    title_clause = f' ({title!r})' if title else ''
    return {
        'error': (
            f'a non-terminal human gate for gate_subject={subject!r} already '
            f'exists: task {task_id}{title_clause}. Filing a second carrier '
            f'for the same subject is what produced the 5902 -> 5916 -> 5929 '
            f'chain this guard exists to bound.'
        ),
        'error_type': 'RecurringGateViolation',
        'existing_gate_task_id': task_id,
        'gate_subject': subject,
        'hint': (
            f'Do not file a new gate. Amend the existing one: call '
            f'update_task on task {task_id} to refresh its evidence and bump '
            f'metadata.recurrence_count. '
            f'CAUTION: update_task\'s append=True governs only details and '
            f'prompt — it does NOT append description, which always '
            f'overwrites. To amend the description, read the current text '
            f'first, write the full merged text, and verify the echoed '
            f'updated_task reflects what you intended. '
            f'If the subject genuinely needs a fresh gate, close the existing '
            f'carrier first (done or cancelled) — terminal carriers do not '
            f'block.'
        ),
    }


# ---------------------------------------------------------------------------
# Async orchestrator (the module's only I/O)
# ---------------------------------------------------------------------------


async def recurring_gate_guard_error(
    metadata: str | dict[str, Any] | None,
    agent_id: str | None,
    project_root: str,
    *,
    fetch_tasks: GetTasksFn,
) -> dict[str, Any] | None:
    """Reject a recon-stage human gate that duplicates an already-open one.

    Returns a :func:`recurring_gate_error` dict when the submission is a
    recon-stage gate carrying a ``gate_subject`` that already has a
    NON-TERMINAL carrier, and ``None`` otherwise — including for every
    non-recon caller, which is never enforced.

    Gates run cheapest-first, so the extra task read is issued ONLY on the
    narrow recon gate-filing path: a non-recon caller costs one string
    ``startswith`` and nothing else. ``fetch_tasks`` is awaited at most once.

    FAILS OPEN. If the lookup raises, this logs a WARNING and returns
    ``None``, letting the submission proceed — the same policy
    ``TaskInterceptor._check_escalation_idempotency`` adopts when its own
    ``get_tasks`` fails ("callers fall through to CREATE rather than
    treating a failure as a clean no-hit"). The failure modes are
    asymmetric: failing closed on a transient backend blip would block
    EVERY recon human gate including genuinely novel ones a human needs to
    see, whereas failing open costs at most one duplicate gate — precisely
    the bounded, already-tolerated cost this guard reduces.

    Args:
        metadata: The submission's declared metadata (dict, JSON string or
            None), read BEFORE ``inject_operational_routing`` normalizes it.
        agent_id: The resolved caller identity. Enforcement fires only when
            this is a string starting with ``'recon-stage-'``.
        project_root: Absolute path to the project root. Unused — the check
            is metadata-only — but kept to mirror ``execution_class_error``'s
            signature so the two guards read as a matched pair, and so a
            future project-scoped rule has a home without a signature change.
        fetch_tasks: Injected zero-arg async callable returning a raw
            ``get_tasks`` result. The caller is responsible for narrowing it
            to non-terminal statuses; :func:`find_open_gate` re-checks anyway.
    """
    if not (isinstance(agent_id, str) and agent_id.startswith('recon-stage-')):
        return None

    meta = _parse_metadata(metadata)
    if not is_gate_submission(meta):
        return None

    subject = extract_gate_subject(meta)
    if subject is None:
        # No dedupe key => nothing to enforce, and nothing to read.
        return None

    try:
        result = await fetch_tasks()
    except Exception:
        logger.warning(
            'recurring_gate_guard: task lookup failed for gate_subject=%s — '
            'failing open (submission proceeds)',
            subject,
            exc_info=True,
        )
        return None

    existing = find_open_gate(_iter_task_rows(result), subject)
    if existing is None:
        return None

    logger.warning(
        'recurring_gate_guard: rejecting duplicate human gate for '
        'gate_subject=%s — task %s is already open (status=%s)',
        subject,
        existing.get('id'),
        existing.get('status'),
    )
    return recurring_gate_error(existing, subject)
