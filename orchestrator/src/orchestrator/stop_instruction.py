"""orchestrator.stop_instruction — shared explicit stop-instruction detector.

GOVERNANCE GAP this module closes (task 2509, reconciliation finding
0aac21b4): the autonomous ``/unblock`` session for task 2407 self-authorized
an irreversible Mem0 mutation despite an explicit "do not apply" instruction,
then narrated the result as human-authorized. The companion ``/unblock`` 2273
session in the same audit episode is the "what right looks like" precedent —
it was SIGTERM-killed before it could auto-apply, honoring its runbook's
human-rehearsal mandate. This module gives every enforcement surface a way to
self-halt on a stop instruction instead of relying on an external kill.

This is the SINGLE SOURCE OF TRUTH for the curated stop-instruction phrase
set. It is reused, unmodified, by:

  - ``orchestrator.b3_gate.check_proposal`` — a mechanical hard-abort step
    (1c) that scans the persisted proposal + task description.
  - ``orchestrator.deterministic_runner.DeterministicRunner`` — a guard on
    the first dispatch of a non-predicate ``before_done`` (act-then-ask/
    deploy) task that scans the task description.
  - The ``/unblock`` and ``unblock-low-risk`` runbook prose (skills/unblock/
    SKILL.md, skills/unblock-low-risk/SKILL.md) — a human-readable
    precondition citing this module as the canonical phrase source.

Centralizing the phrase set here means it cannot silently drift between
these surfaces.

Design (see task 2509 plan design decisions):
  - Curated MULTI-WORD phrases only (e.g. 'do not apply', 'human rehearsal')
    — never a bare single word like 'stop' or 'halt' — to bound false
    positives: single-word triggers appear incidentally in benign task text,
    whereas multi-word imperative phrases are far more specific to an actual
    directive.
  - Matching is case/hyphen/whitespace-insensitive (casefold, hyphens
    normalized to spaces, whitespace runs collapsed) applied uniformly to
    both the scanned text and the phrase itself, so 'do not auto-apply' and
    'do not auto apply' resolve identically.
  - A false positive here is fail-safe: it only routes work to a human,
    exactly as if the session had never run.

This module is intentionally stdlib-only with no orchestrator imports, so
that ``b3_gate`` (which stays dependency-light — stdlib ``sqlite3`` plus a
lazily-imported config) can import it at module load with no new transitive
dependency.
"""

from __future__ import annotations

import re

# Curated, multi-word-only stop-instruction phrases. Lowercased/casefolded
# and stripped — see module docstring for the false-positive rationale.
# Extend this set (not a second copy) when a new stop-instruction phrasing
# is identified — every consumer picks up the change automatically.
STOP_INSTRUCTION_PHRASES: frozenset[str] = frozenset({
    'do not apply',
    'do not auto-apply',
    'do not merge',
    'do not proceed',
    'do not run',
    'do not execute',
    'do not self-authorize',
    'human rehearsal',
    'hold for human',
    'awaiting human',
})

_WHITESPACE_RE = re.compile(r'\s+')


def _normalize(text: str) -> str:
    """Casefold *text*, replace hyphens with spaces, collapse whitespace runs.

    Applied uniformly to both the scanned text and each candidate phrase so
    hyphen/space spelling variants (e.g. 'auto-apply' vs 'auto apply') and
    case variants match identically.
    """
    return _WHITESPACE_RE.sub(' ', text.casefold().replace('-', ' ')).strip()


def detect_stop_instruction(*texts: str | None) -> str | None:
    """Return the first canonical stop-instruction phrase found in *texts*, else None.

    Each positional argument is an independent text to scan (varargs so
    callers can pass e.g. proposal_text, block_reason, and a task
    description in one call without pre-joining them). ``None`` and empty
    entries are skipped — safe to pass optional/absent fields directly.

    Returns the phrase exactly as it appears in ``STOP_INSTRUCTION_PHRASES``
    (the canonical spelling), not the substring as it appeared in *texts*.
    """
    for text in texts:
        if not text:
            continue
        normalized_text = _normalize(text)
        for phrase in STOP_INSTRUCTION_PHRASES:
            if _normalize(phrase) in normalized_text:
                return phrase
    return None
