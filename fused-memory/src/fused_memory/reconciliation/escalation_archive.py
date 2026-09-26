"""One scan of the escalation queue + archive for recently-resolved fingerprints.

TWO CONTRACTS A READER CANNOT RECOVER FROM THE CODE BELOW.

(1) THIS IS SYNCHRONOUS AND UNBOUNDED, so every coroutine caller must reach it
through ``await asyncio.to_thread(...)``.  It globs the queue root and then
*rglobs* a dated archive subtree that ``escalation/queue.py`` describes as "an
archive shared across 7+ projects" — there is no ceiling on how many files that
is, and each one is an ``open`` + ``read`` + ``json.loads``.  Called inline from
a coroutine it wedges the event loop for the whole walk, which is the defect
task 5550 exists to remove: the reconciliation harness and the ``/alive`` route
the orchestrator watchdog probes share one loop, so a long scan here reads to
the watchdog as a dead process.  ``ReconciliationHarness._run_remediation_pass``
is its one coroutine caller, and takes that hop.

(2) THIS MODULE LIVES UNDER ``fused-memory/src/`` DELIBERATELY, not in the
``escalation`` package beside ``iter_all_escalation_paths`` where it would
otherwise belong.  ``shared/tests/test_loop_blocking_gate.py`` sweeps only
``_SCOPE_PREFIX = 'fused-memory/src/'``, and it finds a blocking call by
following sync helpers across modules *that are in its sources dict*.  A helper
under ``escalation/src/`` is invisible to it, so moving this file there would
take the ``_escalate -> _finding_recently_resolved -> read_text`` reach out of
the scanner's view and silently void the ten INV-8 ledger rows that record that
defect as still-open and owned by task 5270 — coverage loss disguised as a fix.
``shared/tests/test_loop_blocking_gate.py::TestScanHelperStaysInGateScope``
enforces this rather than trusting the comment.
"""
from __future__ import annotations

from collections.abc import Collection
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Mirrors the optional-import guard in reconciliation/harness.py: the escalation
# package is an optional dependency, and its absence degrades to "no suppressions"
# rather than an ImportError at module load.  Defense in depth — both callers
# already gate on HAS_ESCALATION before reaching this module.
try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import iter_all_escalation_paths  # type: ignore[import-untyped]
    HAS_ESCALATION = True
except ImportError:
    HAS_ESCALATION = False

_TERMINAL_STATUSES = ('resolved', 'dismissed')


def scan_recently_resolved_fingerprints(
    queue_dir: Path,
    *,
    now: datetime,
    window: timedelta,
    categories: Collection[str],
) -> frozenset[str]:
    """Fingerprints of escalations in *categories* resolved within *window* of *now*.

    A record contributes its ``dedupe_fingerprint`` iff it is terminal
    (``resolved`` or ``dismissed`` — a dismissal is a disposition too, and
    suppresses re-firing exactly as a resolution does), its category is in
    scope, and it carries both a fingerprint and a ``resolved_at`` that parses
    and falls inside the window.  A naive ``resolved_at`` is read as UTC.

    FAIL-OPEN PER RECORD AND IN AGGREGATE.  An unreadable or unparseable file is
    skipped rather than voiding the scan, and a missing *queue_dir* yields
    nothing (``iter_all_escalation_paths``' own contract).  Everything this set
    is used for is SUPPRESSION, so losing an entry costs at most one extra
    escalation while inventing one would silence a needed alarm — the asymmetry
    that makes skipping the right response to a bad record.
    """
    if not HAS_ESCALATION:
        return frozenset()

    fingerprints: set[str] = set()
    for path in iter_all_escalation_paths(Path(queue_dir)):
        try:
            esc = Escalation.from_json(path.read_text())
        except Exception:
            continue
        if not (
            esc.status in _TERMINAL_STATUSES
            and esc.category in categories
            and esc.dedupe_fingerprint
            and esc.resolved_at
        ):
            continue
        try:
            resolved = datetime.fromisoformat(esc.resolved_at)
        except (ValueError, TypeError):
            continue
        if resolved.tzinfo is None:
            resolved = resolved.replace(tzinfo=UTC)
        if now - resolved <= window:
            fingerprints.add(esc.dedupe_fingerprint)
    return frozenset(fingerprints)
