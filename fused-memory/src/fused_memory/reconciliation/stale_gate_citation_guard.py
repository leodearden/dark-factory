"""Stale gate-citation guard (task 4919).

The invariant, enforced where a recon-stage ``details`` write meets the live
task row:

    In text that asserts THIS task's pending external gates, every cited
    task id must be a live gate: an element of the task's ``dependencies``
    array, or the task-id half of a ``metadata.external_deps`` entry.

:func:`stale_gate_citation_error` is the predicate, consumed by
``middleware/task_interceptor.py::TaskInterceptor.update_task``.
:func:`render_gate_citation_section` states the same rule to Stage 2, built
from the same marker vocabulary and :data:`ERROR_TYPE`.

Fail-open: input that is not conclusive passes — a non-recon caller, no
``details``, an uninterpretable dependency array, or no marker with an
adjacent id list. The incident, the corpus measurement behind the marker
vocabulary, and why no existing guard covers this are in the task 4919 record.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from shared.task_metadata import ExternalDep

from fused_memory.reconciliation.task_filter import TERMINAL_OUTCOME_RE

__all__ = [
    'ERROR_TYPE',
    'GATE_CITATION_MARKER',
    'GATE_CITATION_MARKER_PHRASES',
    'GATE_CITATION_RE',
    'find_gate_citation_ids',
    'render_gate_citation_section',
    'stale_gate_citation_error',
]

ERROR_TYPE = 'ReconStaleGateCitationRejected'

# Each phrase the Stage 2 prompt names as policed, paired with the regex arm
# that polices it, so neither can change without the other. Phrasings left out,
# each for a measured false positive on correct relay prose:
#   - bare 'gate': the correction relay "… no longer exists as a separate gate".
#   - 'blocked on': "3659: pending — blocked on 3212" cites a transitive gate.
#   - 'upstream gates': "via their own upstream gates 3212 and 4006", transitive.
#   - 'remediation levers': "remediation levers remain 3212 (in-progress) ->
#     3659", transitive.
_GATE_CITATION_VOCABULARY: tuple[tuple[str, str], ...] = (
    ('external deps', r'external\s+dep(?:s|endenc(?:y|ies))?'),
    ('pending external gates', r'pending\s+external\s+gates?'),
    ('external gates', r'external\s+gates?'),
    ('gating dependencies', r'gating\s+dependenc(?:y|ies)'),
)

GATE_CITATION_MARKER_PHRASES: tuple[str, ...] = tuple(
    phrase for phrase, _ in _GATE_CITATION_VOCABULARY
)

GATE_CITATION_MARKER = '|'.join(arm for _, arm in _GATE_CITATION_VOCABULARY)

# A whole 2-5 digit number that is not the year of a date like '2026-09-11'.
_TASK_ID = r'\d{2,5}\b(?!-\d)'

# A marker, then only the id list IMMEDIATELY following it: capture stops at
# the first non-list token, so ids inside a trailing parenthetical or after
# 'only' are not read as citations.
GATE_CITATION_RE: re.Pattern[str] = re.compile(
    rf'(?:{GATE_CITATION_MARKER})\s*(?:are|as|:|=)?\s*'
    rf'({_TASK_ID}(?:\s*(?:[/,+&]|\band\b)\s*{_TASK_ID})*)',
    re.IGNORECASE,
)

# The words allowed between an id list and the terminal outcome that makes it
# a retrospective, as in 'external deps 3658/3659 have all been merged'.
_RETROSPECTIVE_LEAD_RE = re.compile(
    r'(?:\s+(?:have|has|had|are|were|is|was|been|all|both|now|since|already))*\s+',
    re.IGNORECASE,
)


def _is_retrospective(text: str, list_end: int) -> bool:
    """Whether the id list ending at `list_end` is directly reported as landed."""
    lead = _RETROSPECTIVE_LEAD_RE.match(text, list_end)
    return lead is not None and TERMINAL_OUTCOME_RE.match(text, lead.end()) is not None


def find_gate_citation_ids(text: str) -> set[int]:
    """Return the task ids `text` cites as this task's pending external gates.

    Pure — no I/O. A marker with no adjacent id list contributes nothing, nor
    does a list reported as already landed: a retrospective is not a claim
    that those gates still block.
    """
    if not isinstance(text, str) or not text:
        return set()
    cited: set[int] = set()
    for m in GATE_CITATION_RE.finditer(text):
        if not _is_retrospective(text, m.end()):
            cited.update(int(t) for t in re.findall(r'\d+', m.group(1)))
    return cited


def _normalise_dependency_ids(value: Any) -> set[int] | None:
    """Coerce a live ``dependencies`` payload to a set of task ids.

    Returns ``None`` when the payload cannot be interpreted at all, which the
    caller reads as "fail open". An EMPTY list is interpretable and returns an
    empty set — a task with no dependencies genuinely has no pending external
    gates, so citing one is precisely the violation.

    Both int and str shapes arrive here in practice, so accepting both is
    load-bearing rather than defensive: ``sqlite_task_backend._row_to_task``
    types ``dependencies`` as ``list[int]`` on READ, while
    ``TaskBackend.update_task(..., dependencies: list[str] | None)`` takes
    ``list[str]`` on WRITE, and the interceptor prefers the write kwarg when the
    same call rewrites the array. Non-coercible entries are skipped rather than
    raising. Cross-project gates never appear here: they live in
    ``metadata.external_deps`` and are read by ``_external_dep_task_ids``.
    """
    if not isinstance(value, (list, tuple)):
        return None
    ids: set[int] = set()
    for entry in value:
        try:
            ids.add(int(entry))
        except (TypeError, ValueError):
            continue
    if value and not ids:
        return None
    return ids


def _external_dep_task_ids(metadata_payloads: Iterable[Any]) -> set[int]:
    """Task-id halves of every ``metadata.external_deps`` entry in the payloads.

    Cross-project gates (``"project_id:task_id"``, docs/task-authoring.md §3.2)
    are stored in ``metadata.external_deps``, not in ``dependencies``, yet are
    just as live. Payloads may be a dict or a JSON string; anything unreadable
    contributes nothing. Ids are unioned across payloads, so a collision with a
    local id only widens the live set — the fail-open direction.
    """
    ids: set[int] = set()
    for payload in metadata_payloads:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                continue
        if not isinstance(payload, dict):
            continue
        entries = payload.get('external_deps')
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, str):
                continue
            try:
                ids.add(int(ExternalDep.parse(entry).task_id))
            except ValueError:
                continue
    return ids


def stale_gate_citation_error(
    details: Any,
    agent_id: str | None,
    *,
    live_dependencies: Any,
    metadata_payloads: Iterable[Any] = (),
) -> dict[str, Any] | None:
    """Reject a recon-stage ``details`` write that cites a pending external gate
    absent from the task's live ``dependencies`` array.

    Returns a structured error dict (``{'error', 'error_type', 'hint'}``, the
    same flat shape as ``premise_lint_guard.premise_lint_error``) on a
    violation, else ``None``. See the module docstring for the fail-open
    direction.

    Args:
        details: The ``details`` text being written. Anything that is not a
            non-empty string is a no-op.
        agent_id: The resolved caller identity. Enforcement fires only for a
            string starting with ``'recon-stage-'`` — the scoping lives inside
            the function (as in ``premise_lint_error``) so the predicate is
            safe to unit-test and safe to call from any boundary.
        live_dependencies: The dependency array the write LEAVES BEHIND — the
            incoming kwarg when the same call rewrites it, else the live row's.
        metadata_payloads: The live row's ``metadata`` and any incoming
            ``metadata`` write. Their ``external_deps`` task ids count as live
            gates alongside ``live_dependencies``.
    """
    if not (isinstance(agent_id, str) and agent_id.startswith('recon-stage-')):
        return None
    if not isinstance(details, str) or not details:
        return None
    live = _normalise_dependency_ids(live_dependencies)
    if live is None:
        return None
    live |= _external_dep_task_ids(metadata_payloads)

    stale = sorted(find_gate_citation_ids(details) - live)
    if not stale:
        return None

    stale_text = ', '.join(str(i) for i in stale)
    live_text = ', '.join(str(i) for i in sorted(live)) or '(none)'
    return {
        'error': (
            f'Task details cite {stale_text} as a pending external gate, but '
            f"those ids are absent from the task's live `dependencies` array "
            f'({live_text}). A gate list copied forward from earlier relay '
            f'prose keeps a superseded dependency presenting as a live '
            f'blocker — the task-3708 incident this guard closes.'
        ),
        'error_type': ERROR_TYPE,
        'hint': (
            f'Re-derive the gate list from the live `dependencies` array read '
            f'this cycle ({live_text}), drop {stale_text}, and retry — the '
            f'array above is current, so no second read is needed. If an id is '
            f'being mentioned historically rather than as a live gate, phrase '
            f'it outside a gate assertion (e.g. "3660 was coalesced into '
            f'4856") or as a completed outcome (e.g. "external deps '
            f'3658/3659 have landed").'
        ),
    }


def render_gate_citation_section() -> str:
    """Render the gate-citation mandate for the Stage 2 system prompt, following
    the ``render_*_section()`` style used throughout ``prompts/stage2.py``.

    :data:`ERROR_TYPE` and :data:`GATE_CITATION_MARKER_PHRASES` are interpolated
    rather than restated, so the rule the prompt states and the rule
    :func:`stale_gate_citation_error` enforces are one thing.
    """
    phrases = ', '.join(f'"{p}"' for p in GATE_CITATION_MARKER_PHRASES)
    return (
        '## Pending External Gates Must Be Re-Derived, Never Copied Forward\n'
        "When you append an evidence relay to a task's `details`, RE-DERIVE any "
        'list of pending external gates from that task\'s live `dependencies` '
        'array as read THIS cycle. Never carry a gate list forward from earlier '
        'relay prose already in the same field — that field is append-only, so '
        'the text you are reading above your own append may predate several '
        'dependency changes.\n\n'
        f'POLICED PHRASINGS: {phrases}. An id list immediately following any of '
        'these reads as an assertion that those ids are gating the task NOW, and '
        'every id in it must be an element of the live `dependencies` array.\n\n'
        f'IF IT IS NOT, the write is rejected at the boundary with '
        f'`{ERROR_TYPE}` — the write does not land. The rejection carries both '
        'the stale ids and the full live `dependencies` array, so correct the '
        'sentence in place and retry in the same turn; you do NOT need another '
        'read to find out what the live gates are.\n\n'
        'MENTIONING A SUPERSEDED ID IS STILL FINE, outside a gate assertion. '
        'State it historically ("3660 was coalesced into 4856") or as a '
        'completed outcome ("external deps 3658/3659 have landed") rather than '
        'as a live gate.\n\n'
        'WHY: task 3708\'s relay named 3660 as its remaining blocker for three '
        'consecutive cycles after 3660 had been coalesced into 4856, because '
        'each cycle copied the gate list from the previous relay instead of from '
        'the dependency array. Re-reading the task is not enough on its own — '
        'the relay that introduced the error was written by an agent that had '
        'read the live task in that same cycle (task 4919).'
    )
