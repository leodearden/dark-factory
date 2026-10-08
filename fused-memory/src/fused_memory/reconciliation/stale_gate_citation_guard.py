"""Stale gate-citation guard (task 4919).

## The incident

Task 3708's ``details`` field carries an append-only evidence relay written
each reconciliation cycle by Stage 2. Across three consecutive cycles
(2026-08-26, 2026-08-27, 2026-08-29) every relay asserted "the only real
remediation lever remains external deps 3658/3659/3660 landing" — but 3660
had been coalesced into 4856 on 2026-08-28 and was no longer one of the
task's dependencies. The gate list was copy-forwarded from the previous
relay's prose instead of re-derived from the task's live ``dependencies``
array, so a superseded id kept presenting as a live blocker. Origin: Stage 1
finding ``5d5b31f3-2e16-48a4-8493-fded89fdb3af``.

## PREMISE CORRECTION: there is no evidence-relay generator

The task that commissioned this module described fixing "the task-3708-family
evidence-relay generator". No such generator exists. A repo-wide
``--include=*.py`` search for the relay prose signatures ("pending external
gate", "remediation lever", "appended by recon", "evidence relay") across
``fused-memory/``, ``orchestrator/``, ``shared/`` and ``escalation/`` returns
zero matches, and no prompt defines a relay format. The relay is freehand LLM
output from recon Stage 2 — ``agent_id='recon-stage-task_knowledge_sync'``,
the only stage holding ``update_task`` (Stages 1 and 3 have it in
``cli_stage_runner.py``'s ``DISALLOW_TASK_WRITES``). The only code in the path
is generic plumbing: ``sqlite_task_backend.py::TaskBackend.update_task``
blind-concatenates ``existing + '\\n\\n' + details`` when ``append=True``.

Prose written by an LLM cannot be made deterministic, but the INVARIANT it
must satisfy can be made non-optional at the write boundary, where the live
``dependencies`` array and the incoming ``details`` string are both already in
scope:

    In text that asserts THIS task's pending external gates, every cited
    task id must be an element of the task's live ``dependencies`` array.

:func:`stale_gate_citation_error` is that predicate, consumed by
``middleware/task_interceptor.py::TaskInterceptor.update_task``.
:func:`render_gate_citation_section` states the same rule to Stage 2 in its
system prompt, sharing :data:`ERROR_TYPE` and
:data:`GATE_CITATION_MARKER_PHRASES` with the predicate so the stated rule and
the enforced rule cannot drift. Pairing a code-side gate with source-side
discipline is the house pattern ``prompts/stage1.py`` already states for two
sibling defects: the gate alone cannot undo the wasted turn or the misleading
in-run narrative.

## FAIL DIRECTION: fail-open

Under-fire rather than block a legitimate relay. The scanner returns nothing
whenever the inputs are not conclusive, and :func:`stale_gate_citation_error`
returns ``None`` for a non-recon ``agent_id``, absent/empty ``details``,
``dependencies`` that is missing or wholly uninterpretable, no marker+list
match, or a citation whose every id is live.

## Measurement (2026-09-11, reproducible)

Validated against ``.taskmaster/tasks/tasks.db`` (tag=master). Note that
``dependencies`` is NOT a task column — it is the separate relational table
``dependencies(tag, task_id, depends_on)``, joined per task for this scan.

  - 1591 tasks carry non-empty ``details``.
  - 16 of them contain a marker phrase, over 33 marker occurrences.
  - Only 9 of those 33 markers have an adjacent id list, so 24 markers
    deliberately under-fire. All 9 matches are on task 3708 — still the only
    task in the corpus producing any gate-citation match at all, which is why
    this guard needs no allowlist and why an allowlist would be a
    single-element list that stopped protecting the moment the relay
    convention spread.
  - 3 of the 9 fire: exactly the three stale ``3660`` citations named above.
    6 pass.
  - Precision 100% (0 false positives over 1591 tasks); recall on the known
    incident 3/3.

OUT-OF-SAMPLE. The first scan (2026-09-08) found 7 matches; this one finds 9.
The two that appeared in between — ``GATING DEPENDENCY 4987 OBSERVED`` and
``remediation levers remain 3659 (→3212)`` — are relay prose written AFTER
this algorithm was designed, and both correctly pass. That is the evidence the
marker vocabulary generalises rather than being fitted to the incident.

## Non-duplication (INV-5)

No existing module asks this module's question — "does a gate assertion cite
an id absent from the live ``dependencies`` array" — and none can see the
citations, because the corpus spells them as BARE id lists
(``3658/3659/3660``) with no ``task``/``#`` anchor:

  - ``task_filter.find_conflicting_task_status_ids`` asks whether ONE id is
    framed as both terminal and non-terminal. It is clause-scoped and anchored
    on ``TASK_REF_RE``'s ``task N``/``#N`` forms, which match none of the
    citations here.
  - ``citation_verifier`` handles structured UUID citations only.
  - ``middleware/recon_code_fix_premise_guard.py`` is a curator pre-check that
    DROPS filed task candidates whose premise the live source refutes — a
    different subject (candidates, not ``details`` prose) and a different
    verdict (drop, not reject).
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

# The rejection this guard returns. Shared with render_gate_citation_section()
# below so the prompt names exactly the error the write boundary produces —
# renaming this constant cannot silently orphan the prompt.
ERROR_TYPE = 'ReconStaleGateCitationRejected'

# Marker vocabulary: the phrasings that assert THIS task's pending external
# gates. Deliberately narrow and corpus-derived — each exclusion below was
# measured to false-positive on real, correct relay prose:
#
#   - bare 'gate' fires on the CORRECTION relay itself ("3660 must be dropped
#     permanently, it no longer exists as a separate gate").
#   - 'blocked on' fires on "3659: pending — blocked on 3212 (pending, itself
#     blocked on 3207)", where 3212/3207 are correctly-cited TRANSITIVE gates
#     — gates of this task's gates, legitimately absent from its own
#     `dependencies`.
#   - 'upstream gates' fires on "(via their own upstream gates 3212 and 4006
#     respectively)" for the same reason.
#
# Widening the vocabulary trades this guard's only real asset — a
# zero-false-positive record over the live corpus — for recall the incident
# does not need.
#
# The vocabulary is ONE table with two views: each human-readable spelling (what
# render_gate_citation_section() shows a Stage 2 author) paired with the regex
# arm that matches it (what the write boundary enforces). Both
# GATE_CITATION_MARKER_PHRASES and GATE_CITATION_MARKER are derived from it, so
# an arm cannot be policed without the prompt naming it, nor named without being
# policed — the prompt-vs-rule drift this task exists to fix.
#
# The arms carry morphology the plain spellings cannot (singular/plural, and the
# 'remain/remains/are/is' connector), which is why the table pairs them rather
# than escaping the phrases mechanically. Arm ORDER is significant: 'pending
# external gates' precedes 'external gates' so the longer spelling wins where
# both could match.
_GATE_CITATION_VOCABULARY: tuple[tuple[str, str], ...] = (
    ('external deps', r'external\s+dep(?:s|endenc(?:y|ies))?'),
    ('pending external gates', r'pending\s+external\s+gates?'),
    ('external gates', r'external\s+gates?'),
    ('remediation levers remain', r'remediation\s+levers?\s+(?:remains?|are|is)'),
    ('gating dependencies', r'gating\s+dependenc(?:y|ies)'),
)

GATE_CITATION_MARKER_PHRASES: tuple[str, ...] = tuple(
    phrase for phrase, _ in _GATE_CITATION_VOCABULARY
)

GATE_CITATION_MARKER = '|'.join(arm for _, arm in _GATE_CITATION_VOCABULARY)

# Marker-anchored CONTIGUOUS id-list capture: match a marker, then take only
# the id list IMMEDIATELY following it, stopping at the first non-list token.
#
# A clause-scoped rule — the obvious design, and the one
# task_filter.find_conflicting_task_status_ids uses — would block the correct
# relay "remediation levers remain: 3659 (blocked on 3212→3207), 4856 (blocked
# on 3659+4006)", because those transitive gates are not in `dependencies`.
# Stopping at the first non-list token is also what handles the correction
# relay "…as 3658/3659/4856 only — 3660 must be dropped permanently" for free:
# the capture ends at ' only', so the historical 3660 is never read as a
# citation and no separate negation heuristic is needed.
#
# Exactly ONE capture group, so `findall` over group(1) stays a list of strings.
# Both flexibility points are corpus-forced, not speculative: the optional
# connector admits both "remediation levers remain: 3659" and the colon-less
# "remediation levers remain 3659", and IGNORECASE admits the ALL-CAPS
# "GATING DEPENDENCY 4987".
GATE_CITATION_RE: re.Pattern[str] = re.compile(
    rf'(?:{GATE_CITATION_MARKER})\s*(?:are|as|:|=)?\s*'
    rf'(\d{{2,5}}(?:\s*(?:[/,+&]|\band\b)\s*\d{{2,5}})*)',
    re.IGNORECASE,
)

# How far past the captured id list to look for a terminal-outcome cue. The
# width is measured, not chosen: it must be wide enough to span
# " have landed and this task is unblocked" (38 chars) yet narrow enough that
# the live stale spelling's tail — " landing so this task (γ) can be dispat" —
# still carries no cue. `\blanded\b` does not match "landing", so widening this
# would not break the true positives, but a wider window absorbs incidental
# later prose ("… once everything is done") into the decision.
_TERMINAL_TAIL_WINDOW = 40


def find_gate_citation_ids(text: str) -> set[int]:
    """Return the task ids cited as this task's pending external gates in `text`.

    Pure — no I/O. Returns an empty set for any input that does not
    conclusively contain a gate assertion with an adjacent id list; see the
    module docstring for the fail-open rationale and the corpus measurement.
    """
    if not isinstance(text, str) or not text:
        return set()
    cited: set[int] = set()
    for m in GATE_CITATION_RE.finditer(text):
        # A retrospective ("external deps 3658/3659 have landed") is not an
        # assertion that those gates are still pending, so it is a legitimate
        # relay even once `dependencies` has been emptied. Blocking it is the
        # false-positive class this escape removes — the fail-open direction.
        # TERMINAL_OUTCOME_RE is imported rather than re-spelled so this shares
        # one "framed as a terminal outcome" vocabulary with task_filter's
        # detectors.
        tail = text[m.end():m.end() + _TERMINAL_TAIL_WINDOW]
        if TERMINAL_OUTCOME_RE.search(tail):
            continue
        cited.update(int(t) for t in re.findall(r'\d{2,5}', m.group(1)))
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
    violation, else ``None``. See the module docstring for the incident, the
    corpus measurement, and the fail-open rationale.

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
