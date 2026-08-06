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

import argparse
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Exit codes, named rather than spelled inline so the epilog, main()'s
# docstring and the returns can never drift into disagreeing about what a
# number means. Each denotes EXACTLY ONE outcome -- 2 and 3 exist separately
# from 0 because 0 would otherwise cover both "examined everything, nothing to
# do" and "examined nothing at all", and an operator reading only the exit code
# would take a total no-op for a clean run.
EXIT_OK = 0            # ran clean, or --verify found no unstamped OPEN records
EXIT_WRITE_FAILED = 1  # at least one --apply write FAILED
EXIT_UNSTAMPED = 2     # --verify found OPEN records still lacking a queue stamp
EXIT_NO_QUEUE = 3      # no usable --queue argument; nothing could be resolved

# Disposition reasons. Named constants so the per-record lines, the summary
# block and the tests all read the SAME strings -- the report is the durable
# audit artifact for a migration over live state, and a reason label that
# drifts from what the code did makes it worthless.
REASON_UNIQUE = 'unique-hit'
REASON_TIEBREAK_RECON = 'tiebreak-recon'
REASON_TIEBREAK_ORCH = 'tiebreak-orch'
REASON_NO_ESCALATION_ID = 'no-escalation-id'
REASON_NO_HOLDERS = 'no-holders'
REASON_AMBIGUOUS = 'ambiguous-uncorroborated'


def classify_reason(
    record: sr.DecisionRecord,
    resolved: str,
    holders: list[Path],
) -> str:
    """Label WHY *resolved* was chosen, for the per-record line and the summary.

    Derived from the same inputs the resolver saw rather than returned
    alongside it, so the ladder stays a single expression of the policy and
    this stays a pure description of it.

    The distinction that matters is unique-hit vs. tiebreak-*: both produce a
    real queue path, but one rests on the id being in exactly one queue and the
    other on a measured session_id regularity choosing among several. Collapsing
    them would hide, from the human reviewing this migration, exactly which
    stamps rest on inference.
    """
    if not record.escalation_id:
        return REASON_NO_ESCALATION_ID
    if not holders:
        return REASON_NO_HOLDERS
    if resolved == sr.UNKNOWN_QUEUE:
        return REASON_AMBIGUOUS
    if len(holders) == 1:
        return REASON_UNIQUE
    return REASON_TIEBREAK_RECON if record.session_id is None else REASON_TIEBREAK_ORCH


def _parse_orch_queue_pairs(pairs: list[str]) -> dict[str, Path]:
    """Parse repeated ``PROJECT=PATH`` into a mapping. Malformed -> SystemExit.

    LOUD, never silent. Dropping a malformed pair would quietly disable the
    rule-5 tiebreak for that project and downgrade every one of its ambiguous
    records to UNKNOWN -- a strictly worse migration wearing a clean exit code.
    """
    mapping: dict[str, Path] = {}
    for pair in pairs:
        project, sep, path = pair.partition('=')
        if not sep or not project.strip() or not path.strip():
            raise SystemExit(
                f'--orch-queue expects PROJECT=PATH, got {pair!r}. '
                'Refusing to continue: silently dropping it would disable the '
                "orchestrator tiebreak for that project and downgrade its "
                'ambiguous records to <unknown>.'
            )
        mapping[project.strip()] = Path(path.strip())
    return mapping


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='backfill_decision_queue_stamp',
        description=(
            'Back-fill DecisionRecord.escalations_dir on OPEN cockpit decisions '
            'that predate task 3528 queue-stamping. DRY RUN BY DEFAULT: without '
            '--apply nothing is written. Only OPEN records with no existing '
            'stamp are ever touched.'
        ),
        epilog=(
            'exit codes: 0 = ran clean (or --verify found no unstamped OPEN '
            'records); 1 = at least one write FAILED; 2 = --verify found OPEN '
            'records still lacking a queue stamp; 3 = no usable --queue '
            'argument, so nothing could be resolved. Never read 2 or 3 as a '
            'clean run, and note that only 1 ever means a write was attempted '
            'and rejected.'
        ),
    )
    parser.add_argument(
        '--queue', dest='queues', action='append', default=[], metavar='PATH',
        help='An escalation queue to search for escalation ids. May be repeated.',
    )
    parser.add_argument(
        '--recon-queue', default=None, metavar='PATH',
        help=(
            'The fleet-wide reconciliation queue, used as the corroborated '
            'tiebreak target for records with a null session_id.'
        ),
    )
    parser.add_argument(
        '--orch-queue', dest='orch_queues', action='append', default=[], metavar='PROJECT=PATH',
        help=(
            "A project id -> its orchestrator queue, used as the corroborated "
            'tiebreak target for records filed by a watcher-<slug>-<pid> '
            'session. May be repeated; the live records spell project ids '
            'inconsistently, so map every spelling that appears.'
        ),
    )
    parser.add_argument(
        '--fleet-root', default=None, metavar='PATH',
        help='Override $CLAUDE_FLEET_ROOT (default: ~/.claude/fleet).',
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Actually write. Without this the run is inert.',
    )
    parser.add_argument(
        '--verify', action='store_true',
        help=(
            'Report-only: exit 2 while any OPEN record lacks a queue stamp, '
            '0 once none do. Writes nothing even with --apply.'
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Exit codes: 0 = ran clean (or ``--verify`` found no unstamped OPEN
    records); 1 = at least one write FAILED; 2 = ``--verify`` found unstamped
    OPEN records; 3 = no usable ``--queue`` argument.

    ``--verify`` short-circuits before any resolution: it answers exactly one
    question -- "does any OPEN record still lack a queue stamp?" -- which is
    task 3640's WORK-item-4 invariant, and it must be able to answer that
    without being handed the queue topology.
    """
    args = _build_parser().parse_args(argv)
    root = Path(args.fleet_root) if args.fleet_root else None

    if args.verify:
        residue = [
            d for d in sr.list_decisions(root=root)
            if d.state == sr.DecisionState.OPEN and not d.escalations_dir
        ]
        for decision in residue:
            print(f'UNSTAMPED {decision.id} project={decision.project} '
                  f'escalation_id={decision.escalation_id}')
        print(f'---- summary ----\nunstamped open records: {len(residue)}')
        return EXIT_UNSTAMPED if residue else EXIT_OK

    queues = [Path(q) for q in args.queues]
    if not queues:
        print(
            'no --queue given: nothing could be resolved. Refusing to report a '
            'clean run, which would be indistinguishable from "everything is '
            'already stamped".',
            file=sys.stderr,
        )
        return EXIT_NO_QUEUE

    recon_queue = Path(args.recon_queue) if args.recon_queue else None
    orch_queue_for_project = _parse_orch_queue_pairs(args.orch_queues)

    mode = 'APPLY' if args.apply else 'DRY RUN (pass --apply to write)'
    print(f'mode: {mode}')

    # Scope: OPEN and unstamped, nothing else. A non-OPEN record is not
    # false-closable (the reaper skips it before consulting the queue at all),
    # and an already-stamped record carries first-hand evidence from its filer
    # that this script's inference must not overwrite.
    candidates = [
        d for d in sr.list_decisions(root=root)
        if d.state == sr.DecisionState.OPEN and not d.escalations_dir
    ]
    print(f'candidates: {len(candidates)}')

    by_reason: dict[str, int] = {}
    written = 0
    failed = 0
    for record in candidates:
        holders = (
            queues_holding(record.escalation_id, queues) if record.escalation_id else []
        )
        resolved = resolve_queue_for_decision(
            record,
            queues=queues,
            recon_queue=recon_queue,
            orch_queue_for_project=orch_queue_for_project,
        )
        reason = classify_reason(record, resolved, holders)
        by_reason[reason] = by_reason.get(reason, 0) + 1
        holder_names = ','.join(str(h) for h in holders) or '-'
        line = (
            f'{record.id} project={record.project} '
            f'escalation_id={record.escalation_id} session_id={record.session_id} '
            f'holders=[{holder_names}] -> {resolved} ({reason})'
        )
        if args.apply:
            updated = sr.set_decision_escalations_dir(record.id, resolved, root=root)
            if updated is None:
                failed += 1
                print(f'{line} WRITE-FAILED')
                continue
            written += 1
        print(line)

    print('---- summary ----')
    print(f'candidates: {len(candidates)}')
    for reason in (
        REASON_UNIQUE,
        REASON_TIEBREAK_RECON,
        REASON_TIEBREAK_ORCH,
        REASON_NO_ESCALATION_ID,
        REASON_NO_HOLDERS,
        REASON_AMBIGUOUS,
    ):
        print(f'  {reason}: {by_reason.get(reason, 0)}')
    print(f'written: {written}   write-failed: {failed}')
    if not args.apply:
        print('nothing was written (dry run)')
    return EXIT_WRITE_FAILED if failed else EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
