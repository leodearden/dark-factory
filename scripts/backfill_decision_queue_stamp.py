#!/usr/bin/env python3
"""Back-fill ``DecisionRecord.escalations_dir`` on legacy open cockpit decisions.

THE PROBLEM (task 3528 -> task 3640)
------------------------------------
DecisionRecords are FLEET-GLOBAL — they all live in one
``~/.claude/fleet/decisions/`` directory, shared by every project. Escalation
ids are NOT global: ``esc-<taskid>-<n>`` is unique only WITHIN one queue, and
the same id routinely exists in several queues as entirely unrelated
escalations. dark_factory alone runs TWO queues over that one id namespace:

  * ``<dark-factory>/data/escalations``                  (the orchestrator's)
  * ``<dark-factory>/data/reconciliation/escalations``   (the recon watcher's)

and the second is FLEET-WIDE — it holds reconciliation escalations for reify,
autopilot_video, solar_challenge, know_live and pump_web_ui too. So those
projects' ids collide with their OWN per-project orchestrator queues as well;
the ambiguity is not a dark_factory-only quirk.

Task 3528 fixed the forward direction: the ``reap-decisions`` reaper now scopes
its (decisions <-> escalations) join on TWO axes, project AND queue, and the
watchers stamp ``escalations_dir`` on every newly-filed decision. It did NOT
back-fill. A record still carrying ``escalations_dir == ''`` — the unset/legacy
sentinel — falls back to project-only scoping, which is exactly the state that
let an unrelated RESOLVED escalation in one queue silently close a still-PENDING
blocking human gate filed from the other. That decision then sat invisible in
the cockpit for ~7 days. This script drains that residual population, so the
fallback set is emptied rather than merely shrinking as new records are filed.

WHAT IT WRITES
--------------
For each OPEN record with a falsy ``escalations_dir`` it stamps either the one
queue that demonstrably holds its escalation id, or
``session_registry.UNKNOWN_QUEUE`` ("investigated, could not determine" — the
reaper refuses to close those, leaving them as visible cockpit rows for human
closure). It never rewrites an existing stamp and never touches a non-OPEN
record: the watchers' stamps are FIRST-HAND evidence from the filer, this
script's are inference, and inference must not overwrite evidence.

Writes go through ``session_registry.set_decision_escalations_dir`` rather than
rewriting decision JSON here, so they are atomic and serialized against the
live C8 watchers and the cockpit on the same ids via ``decision_id_lock``.

Queue topology is supplied ENTIRELY by explicit flags — there is no canonical
project_id -> project_root registry in this repo to discover it from, and the
live records' project spellings are inconsistent anyway (``df`` /
``dark_factory`` / ``dark-factory``; ``autopilot-video`` /
``autopilot_video``). Explicit flags make the script a pure function of its
arguments, testable on synthetic dirs, and — since the exact command line is
recorded in the audit report — make the migration reproducible after the fact.

Dry run is the DEFAULT; ``--apply`` is required to write anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

# --- self-locating import bootstrap -----------------------------------------
# scripts/ is not on sys.path when run standalone; bind `orchestrator` to the
# SAME checkout as this script via a __file__-relative path, never a hardcoded
# absolute. An editable install puts the MAIN checkout's orchestrator/src on
# sys.path for a bare `python3`, so without this a copy running from a worktree
# would silently use the main checkout's session_registry (tasks 2881/2882).
# Idempotent — a no-op under pytest, where conftest already inserted scripts/.
_ORCH_SRC = Path(__file__).resolve().parent.parent / 'orchestrator' / 'src'
if str(_ORCH_SRC) not in sys.path:
    sys.path.insert(0, str(_ORCH_SRC))

from orchestrator import session_registry as sr  # noqa: E402


# The measured provenance discriminator (see resolve_queue_for_decision).
_WATCHER_SESSION_PREFIX = 'watcher-'


def queues_holding(escalation_id: str, queues: list[Path]) -> list[Path]:
    """Every queue in *queues* that actually contains *escalation_id*.

    Reports, never adjudicates: returns ALL holders, in the caller's given
    order, so the caller can see that an id is ambiguous rather than being
    handed a single silently-chosen answer.

    Uses the SAME two-tier lookup as ``session_registry.read_escalation_status``
    — the queue-root file ``<queue>/<id>.json`` first (a still-pending
    escalation), then a recursive search under ``<queue>/archive/`` (a
    resolved/dismissed one, moved into a dated subdir by
    ``escalation.queue._archive_resolved``). Matching the reaper's own notion
    of "this queue contains that id" is load-bearing: if the two diverged, this
    script could stamp a queue the reaper would never match against, and the
    record would be pinned OPEN forever. The archive tier also carries most of
    the value here — legacy records overwhelmingly point at escalations that
    have long since resolved.

    Duplicate spellings of one queue collapse to a single holder (compared via
    ``normalize_escalations_dir``). The real invocation passes the fleet-wide
    recon queue as both a ``--queue`` and the ``--recon-queue``; counting that
    twice would misclassify a uniquely-attributable record as ambiguous.

    FAIL-SOFT PER QUEUE: a missing dir, a path that is a regular file, or an
    unreadable tree is skipped rather than raised on. One bad ``--queue``
    argument must not decide the fate of the other five.
    """
    holders: list[Path] = []
    seen: set[str] = set()
    for queue in queues:
        canonical = sr.normalize_escalations_dir(queue)
        if canonical in seen:
            continue
        try:
            if not queue.is_dir():
                continue
            if (queue / f'{escalation_id}.json').is_file():
                found = True
            else:
                found = any((queue / 'archive').rglob(f'{escalation_id}.json'))
        except OSError:
            continue
        if found:
            seen.add(canonical)
            holders.append(queue)
    return holders


def resolve_queue_for_decision(
    record: sr.DecisionRecord,
    *,
    queues: list[Path],
    recon_queue: Path | None,
    orch_queue_for_project: dict[str, Path],
) -> str:
    """The value to stamp on *record*: a normalized queue path, or UNKNOWN_QUEUE.

    An explicit ordered ladder. Direct evidence first; a measured tiebreak only
    to CHOOSE AMONG queues that demonstrably hold the id; UNKNOWN for
    everything else.

    THE TIEBREAK'S EVIDENCE BASIS, so a future reader can re-derive it rather
    than take it on faith: measured across the records ALREADY stamped by
    3528's watchers, ``session_id`` discriminated the two queue families
    perfectly at both measurements — 2026-08-05, 34 stamped records: 25/25
    recon-queue records had ``session_id is None`` and 9/9 orchestrator-queue
    records had a ``watcher-<slug>-<pid>`` session_id; re-derived 2026-08-06 on
    45 stamped records: 25/25 recon null, 20/20 orchestrator ``watcher-*``.
    That is a MEASURED REGULARITY on a related population, which is real
    evidence but strictly weaker than the id actually being present in a queue.
    So it is only ever used to pick between queues that already hold the id,
    never to invent one.

    THE ASYMMETRY OF HARM sets the fallback direction, and is the whole reason
    an explicit UNKNOWN disposition exists. A WRONG stamp silently re-creates
    the cross-queue false-closure hazard this task exists to remove: the record
    gets closed against an unrelated escalation and vanishes from the cockpit
    while the human gate behind it is still open. UNKNOWN merely withdraws the
    row from AUTOMATIC reaping and leaves it visible for human closure. Given
    that, refusing to answer is strictly better than answering plausibly.
    """
    # Rule 1: no escalation id -> nothing to attribute. These are the
    # manual/sentinel records; reap_answered_decisions already skips a falsy
    # escalation_id before consulting the queue, so they were never
    # false-closable. They are stamped anyway so "no OPEN record lacks a queue
    # stamp" becomes a checkable invariant instead of an aspiration.
    escalation_id = record.escalation_id
    if not escalation_id:
        return sr.UNKNOWN_QUEUE

    holders = queues_holding(escalation_id, queues)

    # Rule 3: the id resolves in no supplied queue -- archive retention pruned
    # it, or the owning project's queue was not passed. Refuse, don't guess.
    if not holders:
        return sr.UNKNOWN_QUEUE

    # Rule 2: exactly one holder -> direct evidence, no tiebreak runs. This
    # branch is deliberately ABOVE the heuristics: a measured regularity must
    # never override the id demonstrably being in one specific queue.
    if len(holders) == 1:
        return sr.normalize_escalations_dir(holders[0])

    # Ambiguous from here down (2+ queues hold the id). Infer, then CORROBORATE.
    inferred: Path | None = None
    session_id = record.session_id
    if session_id is None:
        # Rule 4: null session_id -> the fleet-wide reconciliation queue.
        inferred = recon_queue
    elif session_id.startswith(_WATCHER_SESSION_PREFIX):
        # Rule 5: watcher-<slug>-<pid> -> that record's OWN project's
        # orchestrator queue. Keyed on record.project so a df watcher can never
        # claim a reify record. Rule 8 (project absent from the mapping) falls
        # out of this .get returning None.
        inferred = orch_queue_for_project.get(record.project)
    # Rule 7: any other session_id shape (e.g. an /unblock session's
    # 'unblock-df-2085-4242') leaves `inferred` None -- the discriminator was
    # measured only over None and watcher-*, and extending it to "anything
    # not-None means orchestrator" would be inventing evidence.

    if inferred is not None:
        # THE CORROBORATION. Without this the inference could name a queue that
        # demonstrably does NOT hold this escalation -- a stamp that looks
        # authoritative while pointing at the wrong queue, which is precisely
        # the false-closure hazard being removed.
        canonical = sr.normalize_escalations_dir(inferred)
        if canonical in {sr.normalize_escalations_dir(h) for h in holders}:
            return canonical

    # Rule 6: ambiguous and uncorroborated -> UNKNOWN. Human triage, not a coin flip.
    return sr.UNKNOWN_QUEUE
