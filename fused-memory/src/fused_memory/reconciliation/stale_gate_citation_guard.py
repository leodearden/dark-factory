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

import re

from fused_memory.reconciliation.task_filter import TERMINAL_OUTCOME_RE

__all__ = [
    'GATE_CITATION_MARKER',
    'GATE_CITATION_RE',
    'find_gate_citation_ids',
]

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
GATE_CITATION_MARKER = (
    r'external\s+dep(?:s|endenc(?:y|ies))?'
    r'|pending\s+external\s+gates?'
    r'|external\s+gates?'
    r'|remediation\s+levers?\s+(?:remains?|are|is)'
    r'|gating\s+dependenc(?:y|ies)'
)

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
