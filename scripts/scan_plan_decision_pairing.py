#!/usr/bin/env python3
"""Detect semantically CROSS-PAIRED ``design_decisions`` entries in task plans.

READ-ONLY / DETECTION-ONLY: this module and its CLI never mutate a plan. Every
plan file is opened for reading and nothing here writes, renames, chmods or
unlinks anything; the walker it delegates to
(:func:`shared.decision_pairing.scan_design_decisions`) never assigns into the
document it is handed. ``scripts/tests/test_scan_plan_decision_pairing.py``
ASSERTS that rather than trusting it, snapshotting every input's bytes and
``st_mtime_ns`` before each scan — including on the unparseable and unreadable
error paths — and requiring both unchanged afterwards. That matters more here
than in the sibling sweeps: this scanner's real inputs are live plan documents
that a RUNNING task may be reading, so a stray write would be mutation of
another agent's runtime state, not merely a dirty run.

## The damage class

A ``design_decisions`` entry whose ``decision`` and ``rationale`` are each
perfectly well-formed prose, but whose *association* is wrong — the rationale
recorded under one decision-line actually argues for a different one. Both
texts are intact. Nothing is malformed, no sentinel is present, no parser is
upset, and so :func:`shared.toolcall_markup.detect` is structurally blind to
it. This is a DIFFERENT damage class from the envelope leakage that
``sweep_toolcall_markup.py`` and ``scan_task_toolcall_leaks.py`` chase, and the
two are disjoint at the detector.

Every accept/reject verdict is DELEGATED to :mod:`shared.decision_pairing`,
which is the single owner of the two marker literal sets and of the predicate
(INV-5). This script never re-spells a marker; it discovers files, reports, and
renders. The envelope verdict on each record likewise comes from
``toolcall_markup.detect``, unmodified — so ONE report distinguishes the two
damage classes and flags a plan carrying both (the task 3382 shape) instead of
making an operator run two sweeps and join them by hand.

## Why there is no ``--apply`` and never will be

A mis-paired document carries both texts but NO record of which belonged where.
The association itself is what was lost, and it is not recoverable from the
document — unlike envelope damage, which leaves residue around otherwise intact
argument text, so ``shared.toolcall_markup.repair`` can return a slice of its
own input. Here there is nothing to slice and nothing to reconstruct: a
"repair" would have to guess which rationale belongs to which decision, and a
guess written back into a plan is strictly worse than visible damage, because
it launders an unknown into a confident-looking record. Remediation is an
author appending a correction entry, or the supersede mechanism task 3865 will
add — never a bulk rewrite by this or any other script. Task 3692 reached the
same conclusion from the other direction and deliberately declined to assert
repair of its own damage specimen; that restraint is inherited here.

## Every count this script reports is a STRICT LOWER BOUND

The predicate can only ever find a mis-pairing that a human or an agent NOTICED
and then documented in a later correction entry. A mis-pairing nobody noticed
leaves no trace at all — both texts are well-formed, so there is nothing to key
on. Read any number below as a floor, never as a prevalence estimate, and never
report it as "N mis-pairings exist". See :mod:`shared.decision_pairing`'s own
docstring for the same caveat at the predicate.

## What it walks

``<root>/*/plan.json``, where *root* defaults to the repo's
``.worktrees/.task-meta`` — the durable per-lane plan location after the W11
relocation moved plan state out of the worktrees themselves. ``--root`` accepts
any tree of the same shape, notably ``.worktrees-orphaned/*/`` style trees.

A task id is taken from the LANE DIRECTORY, not from the document's own
``task_id`` field. Measured 2026-08-16 over 1299 live plans the two disagree on
four (tasks 2421, 2460, 2579 and 2921 carry ``df_task_<n>``); the directory is
the spelling the rest of the system uses and the one an operator navigates by,
and decisively it is the only one available for a plan whose bytes cannot be
parsed at all — so a skip and a hit key alike.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

# shared/src bootstrap. Same idiom and same precedence argument as
# scripts/sweep_toolcall_markup.py:75-77 and
# scripts/scan_task_toolcall_leaks.py:113-115: resolve it from __file__ and
# insert at sys.path[0], so a run inside a task worktree resolves `shared` to
# THIS checkout's copy of the marker enumeration rather than to whatever
# editable install happens to be on the path. The shared editable install is an
# ordinary .pth entry, so sys.path ORDER decides the winner and a hardcoded or
# install-provided path would silently scan with the main checkout's literals.
_SHARED_SRC = Path(__file__).resolve().parent.parent / 'shared' / 'src'
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))

from shared.decision_pairing import scan_design_decisions  # noqa: E402
from shared.toolcall_markup import detect  # noqa: E402

__all__ = [
    'DEFAULT_ROOT',
    'PLAN_FILENAME',
    'PairingRecord',
    'SKIP_MISSING',
    'SKIP_UNPARSEABLE',
    'SKIP_UNREADABLE',
    'SkippedFile',
    'TreeScan',
    'format_json',
    'format_report',
    'main',
    'scan_plan_file',
    'scan_tree',
]

#: The per-lane plan filename this scanner walks.
PLAN_FILENAME = 'plan.json'

#: The checkout this script lives in — the main checkout, or a task worktree.
_CHECKOUT_ROOT = Path(__file__).resolve().parent.parent

#: The directory holding the task worktrees, and beside them the lane tree.
_WORKTREES_DIR = '.worktrees'


def _default_root(checkout: Path) -> Path:
    """Where task plans live, as seen from *checkout*.

    Resolved from ``__file__`` rather than hardcoded, for the same reason the
    sys.path bootstrap above is: a scanner run should sweep the lane tree of
    the checkout it was invoked from.

    The worktree case is NOT the same path as the main-checkout case, and
    getting that wrong is silent. A task worktree lives at
    ``<repo>/.worktrees/<lane>/``, so the naive ``<checkout>/.worktrees/
    .task-meta`` resolves to ``<repo>/.worktrees/<lane>/.worktrees/.task-meta``
    — a directory that does not exist. Measured 2026-08-16 from this very
    worktree: the naive spelling yields a non-existent root, and the sweep then
    reports zero hits over zero files. That is exactly the failure
    :attr:`TreeScan.scanned` exists to make visible, but a default that walks
    into it by construction is a defect, not a diagnostic. When the checkout
    sits directly under a ``.worktrees`` directory, the lane tree is that
    directory's own ``.task-meta`` sibling-of-the-worktrees.
    """
    if checkout.parent.name == _WORKTREES_DIR:
        return checkout.parent / '.task-meta'
    return checkout / _WORKTREES_DIR / '.task-meta'


#: Where task plans live after the W11 lane-lifecycle relocation moved plan
#: state out of the worktrees themselves. See :func:`_default_root`.
DEFAULT_ROOT = _default_root(_CHECKOUT_ROOT)

#: Why a discovered ``plan.json`` was not scanned. Reported, never raised — one
#: bad file must not abort a sweep, and a silently dropped one would understate
#: a count this scanner already only ever states as a lower bound.
SKIP_MISSING = 'missing'        # the name exists but resolves to nothing
SKIP_UNREADABLE = 'unreadable'  # an OSError opening or reading it
SKIP_UNPARSEABLE = 'unparseable' # present and readable, but not JSON


class PairingRecord(NamedTuple):
    """One self-documented mis-pairing, with everything a triager needs.

    ``header`` and ``marker`` are :mod:`shared.decision_pairing`'s own declared
    literals rather than the matched text, so a reader is never sent to
    log-scrape which literal fired (INV-2). ``envelope_leak`` is the SECOND
    damage class's verdict on the same entry, carried here so one report covers
    both and can flag a plan suffering each at once.
    """

    task_id: str
    path: str
    index: int
    header: str
    marker: str
    field: str
    envelope_leak: bool


class SkippedFile(NamedTuple):
    """One discovered ``plan.json`` that could not be scanned, and why."""

    task_id: str
    path: str
    reason: str
    detail: str


class TreeScan(NamedTuple):
    """The result of one sweep: what matched, what was skipped, what was read.

    ``scanned`` is not redundant with ``len(records)``. Without it an empty
    ``records`` list means either "the corpus is clean" or "the root was wrong
    and nothing was ever opened" — the two outcomes an operator most needs to
    tell apart, and the reason the precedent scanner reserves a distinct exit
    code for "nothing was scanned at all".
    """

    records: list[PairingRecord]
    skipped: list[SkippedFile]
    scanned: int


def _sort_key(task_id: str) -> tuple[int, int, str]:
    """Numeric-aware ordering, so lane 10 sorts AFTER lane 2.

    An operator diffs one run's report against the next; a plain lexicographic
    order would scatter a numeric corpus and manufacture churn that reads as
    new damage. Non-numeric lane names (``_iact-spawn-transcript-persistence``
    is the one live example) sort last, deterministically, by name.
    """
    if task_id.isdigit():
        return (0, int(task_id), '')
    return (1, 0, task_id)


def _entry_has_envelope_residue(entry: object) -> bool:
    """True if ``toolcall_markup.detect`` fires on either field of *entry*.

    Total for adversarial shapes for the same reason the predicate is: this
    walks plan documents nobody validated, and ``detect`` already returns
    ``None`` for any non-``str``.
    """
    if not isinstance(entry, dict):
        return False
    return any(
        detect(entry.get(field)) is not None
        for field in ('decision', 'rationale')
    )


def scan_plan_file(path: Path | str) -> list[PairingRecord]:
    """Every self-documented mis-pairing in the plan at *path*, in index order.

    Reads and parses the file, then delegates every accept/reject verdict to
    :func:`shared.decision_pairing.scan_design_decisions`. Adversarial document
    shapes — a non-object top level, a missing or non-list ``design_decisions``,
    non-dict items, non-``str`` fields — yield no hit and no exception, because
    that walker is total.

    Propagates ``OSError`` and ``json.JSONDecodeError`` rather than swallowing
    them; :func:`scan_tree` is what turns those into reported skips. The split
    mirrors ``scan_task_toolcall_leaks.scan_db``, which likewise raises while
    its CLI driver warns and continues, and it keeps this function honest for a
    caller pointing it at one known file.
    """
    plan_path = Path(path)
    plan = json.loads(plan_path.read_text())
    entries = plan.get('design_decisions') if isinstance(plan, dict) else None
    task_id = plan_path.parent.name
    text = str(plan_path)
    records: list[PairingRecord] = []
    for hit in scan_design_decisions(plan):
        entry = entries[hit.index] if isinstance(entries, list) else None
        records.append(
            PairingRecord(
                task_id=task_id,
                path=text,
                index=hit.index if hit.index is not None else -1,
                header=hit.header,
                marker=hit.marker,
                field=hit.field,
                envelope_leak=_entry_has_envelope_residue(entry),
            )
        )
    return records


def scan_tree(root: Path | str) -> TreeScan:
    """Sweep every ``<root>/*/plan.json``, tolerating each unreadable input.

    An absent *root* yields an empty scan with ``scanned == 0`` rather than
    raising: a fresh checkout legitimately has no lane tree yet, and the zero
    is exactly the signal that distinguishes it from a clean sweep.

    Every per-file failure is REPORTED as a :class:`SkippedFile` and the sweep
    continues, so one truncated plan cannot hide the rest of the corpus.
    ``scanned`` counts only files actually read AND parsed, so it never
    overstates coverage.

    Results are sorted by :func:`_sort_key` then by entry index. The ordering is
    load-bearing rather than cosmetic — see that helper.
    """
    root_path = Path(root)
    records: list[PairingRecord] = []
    skipped: list[SkippedFile] = []
    scanned = 0

    if not root_path.is_dir():
        return TreeScan(records=[], skipped=[], scanned=0)

    for lane in root_path.iterdir():
        if not lane.is_dir():
            continue
        candidate = lane / PLAN_FILENAME
        # exists() follows symlinks; a lane whose plan.json is a DANGLING
        # symlink (the live shape — every lane's plan.json points into
        # .task-meta) must be reported, while a lane that simply never had a
        # plan is not an error at all and is passed over silently.
        if not candidate.exists():
            if candidate.is_symlink():
                skipped.append(
                    SkippedFile(
                        task_id=lane.name,
                        path=str(candidate),
                        reason=SKIP_MISSING,
                        detail='dangling symlink: '
                               f'{candidate.readlink()}',
                    )
                )
            continue
        try:
            file_records = scan_plan_file(candidate)
        except json.JSONDecodeError as exc:
            skipped.append(
                SkippedFile(
                    task_id=lane.name,
                    path=str(candidate),
                    reason=SKIP_UNPARSEABLE,
                    detail=str(exc),
                )
            )
            continue
        except OSError as exc:
            skipped.append(
                SkippedFile(
                    task_id=lane.name,
                    path=str(candidate),
                    reason=SKIP_UNREADABLE,
                    detail=str(exc),
                )
            )
            continue
        scanned += 1
        records.extend(file_records)

    records.sort(key=lambda r: (_sort_key(r.task_id), r.index))
    skipped.sort(key=lambda s: _sort_key(s.task_id))
    return TreeScan(records=records, skipped=skipped, scanned=scanned)


def format_report(scan: TreeScan) -> str:
    """Render *scan* as a grouped, deterministic, human-readable report.

    Always ends with a summary naming BOTH what matched and what was read, so a
    zero-hit run can never be mistaken for a run that opened nothing. An empty
    scan yields that summary rather than a blank string.
    """
    lines: list[str] = []
    by_task: dict[str, list[PairingRecord]] = {}
    for record in scan.records:
        by_task.setdefault(record.task_id, []).append(record)

    for task_id, task_records in by_task.items():
        lines.append(f'task {task_id}: {task_records[0].path}')
        for record in task_records:
            envelope = ' +envelope-residue' if record.envelope_leak else ''
            lines.append(
                f'  design_decisions[{record.index}] '
                f'header={record.header!r} marker={record.marker!r} '
                f'field={record.field}{envelope}'
            )

    if scan.skipped:
        lines.append('skipped:')
        for skip in scan.skipped:
            lines.append(
                f'  task {skip.task_id}: {skip.reason} — {skip.path} ({skip.detail})'
            )

    plans = len(by_task)
    lines.append(
        f'{len(scan.records)} mis-paired entries across {plans} plans '
        f'(scanned {scan.scanned} plan files, skipped {len(scan.skipped)}). '
        'This is a STRICT LOWER BOUND: only mis-pairings someone noticed and '
        'documented leave a trace.'
    )
    return '\n'.join(lines)


def format_json(scan: TreeScan) -> str:
    """Render *scan* as a deterministic JSON object.

    Carries ``skipped`` and ``scanned`` beside ``records`` so a consumer
    reading the machine format has the same coverage information the human
    report shows, rather than an unqualified hit list.
    """
    payload = {
        'records': [record._asdict() for record in scan.records],
        'skipped': [skip._asdict() for skip in scan.skipped],
        'scanned': scan.scanned,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'READ-ONLY sweep for semantically CROSS-PAIRED design_decisions '
            'entries in task plans -- an entry whose decision and rationale are '
            'each well-formed prose but whose association is wrong. '
            'Detection/reporting only; it never mutates a plan.'
        ),
        epilog=(
            'There is deliberately no --apply or --repair, and there never will '
            'be: a mis-paired document carries both texts but no record of '
            'which belonged where, so there is nothing to reconstruct and a '
            '"repair" could only guess. Remediation is an author appending a '
            'correction entry. Every count reported is a STRICT LOWER BOUND -- '
            'only a mis-pairing someone noticed and documented leaves a trace. '
            'exit codes: 0 = the sweep ran; 1 = --fail-on-hit was passed and at '
            'least one mis-paired entry was found. NOTE that 0 does not by '
            'itself mean the corpus is clean: a --root that does not exist also '
            'yields no hits, so read the "scanned N plan files" summary, which '
            'is printed on every run for exactly this reason.'
        ),
    )
    parser.add_argument(
        '--root', default=str(DEFAULT_ROOT),
        help='Tree of <root>/*/plan.json to sweep (default: %(default)s). '
             'Accepts any tree of that shape, e.g. a .worktrees-orphaned lane '
             'tree.',
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Emit a JSON object (records, skipped, scanned) instead of the '
             'human-readable report.',
    )
    parser.add_argument(
        '--fail-on-hit', action='store_true',
        help='Exit 1 when at least one mis-paired entry is found, for use as a '
             'CI or timer gate. Skipped files do NOT trigger it -- see the '
             'scanned count.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — a thin shell over the pure functions above.

    Exit codes: 0 = the sweep ran; 1 = ``--fail-on-hit`` was passed and at
    least one mis-paired entry was found.

    Exit 0 is NOT by itself evidence of a clean corpus. ``--fail-on-hit`` keys
    on hits, and a ``--root`` that does not exist produces none, so a
    mistyped root exits 0 exactly as a clean sweep does. The summary therefore
    states the scanned count on EVERY run, including the no-hits and JSON
    paths, so the distinction is visible to anyone reading output rather than
    only branching on ``$?``. That is the same failure mode the precedent
    scanner reserves its exit 3 for; here it is surfaced in the report rather
    than in a code, because a partial sweep (some lanes skipped) is normal for
    this corpus and would make an all-or-nothing code misleading.
    """
    args = _build_parser().parse_args(argv)
    scan = scan_tree(args.root)
    print(format_json(scan) if args.json else format_report(scan))
    if args.fail_on_hit and scan.records:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
