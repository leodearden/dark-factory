"""RED tests for the extracted escalation-archive scan helper.

``fused_memory.reconciliation.escalation_archive.scan_recently_resolved_fingerprints``
is the single implementation of a scan that task 5550 found duplicated twice in
``reconciliation/harness.py`` — once inline in ``_run_remediation_pass`` (the
per-pass ``resolved_fps`` build) and once in ``_finding_recently_resolved``'s
fallback arm.  Both copies walked ``iter_all_escalation_paths`` over the queue
root AND the dated archive subtree, parsed every hit, and kept the
``dedupe_fingerprint`` of in-window resolved/dismissed records.

These tests pin the filter semantics the two copies shared, so the extraction is
provably behaviour-preserving rather than merely plausible.  The corpus is a
REAL ``EscalationQueue`` on ``tmp_path`` holding REAL ``Escalation`` records —
never a fake — because the contract under test is "what this helper makes of
what the queue actually writes to disk".
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fused_memory.reconciliation.escalation_archive import (
    scan_recently_resolved_fingerprints,
)

escalation_archive_mod = pytest.importorskip('escalation.archive')
from escalation.models import Escalation  # noqa: E402
from escalation.queue import EscalationQueue  # noqa: E402

ARCHIVE_SUBDIR = escalation_archive_mod.ARCHIVE_SUBDIR

NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)
WINDOW = timedelta(seconds=86400)  # _RESOLVED_RECURRENCE_WINDOW_SECONDS
CATEGORIES = ('recon_integrity_issue', 'recon_failure')


def _esc(
    esc_id: str,
    *,
    category: str = 'recon_integrity_issue',
    status: str = 'resolved',
    fingerprint: str | None = None,
    resolved_at: str | None = None,
) -> Escalation:
    """One escalation record with only the fields this scan reads varied."""
    return Escalation(
        id=esc_id,
        task_id='recon-test',
        agent_role='reconciliation-harness',
        severity='info',
        category=category,
        summary=f'summary for {esc_id}',
        status=status,
        dedupe_fingerprint=fingerprint,
        resolved_at=resolved_at,
    )


def _write(directory, esc: Escalation) -> None:
    """Write *esc* as the ``esc-*.json`` file ``iter_all_escalation_paths`` globs."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{esc.id}.json').write_text(esc.to_json())


def _iso(delta: timedelta) -> str:
    return (NOW - delta).isoformat()


def _scan(queue_dir, *, categories=CATEGORIES) -> frozenset[str]:
    return scan_recently_resolved_fingerprints(
        queue_dir, now=NOW, window=WINDOW, categories=categories,
    )


@pytest.fixture
def queue_dir(tmp_path):
    """A real EscalationQueue's directory, ready for hand-written records."""
    return EscalationQueue(tmp_path / 'esc').queue_dir


def test_in_window_resolved_record_contributes_its_fingerprint(queue_dir):
    """(a) resolved 60s ago, category in scope → fingerprint is returned."""
    _write(queue_dir, _esc(
        'esc-recent-1', fingerprint='fp-recent', resolved_at=_iso(timedelta(seconds=60)),
    ))

    assert _scan(queue_dir) == frozenset({'fp-recent'})


def test_dismissed_is_treated_exactly_like_resolved(queue_dir):
    """(b) 'dismissed' is a terminal disposition too — it must suppress alike."""
    _write(queue_dir, _esc(
        'esc-dismissed-1', status='dismissed', fingerprint='fp-dismissed',
        resolved_at=_iso(timedelta(seconds=60)),
    ))

    assert _scan(queue_dir) == frozenset({'fp-dismissed'})


def test_record_resolved_outside_the_window_is_excluded(queue_dir):
    """(c) resolved 8 days ago is far outside a 24h window → re-firing is allowed."""
    _write(queue_dir, _esc(
        'esc-stale-1', fingerprint='fp-stale', resolved_at=_iso(timedelta(days=8)),
    ))

    assert _scan(queue_dir) == frozenset()


def test_record_outside_the_scoped_categories_is_excluded(queue_dir):
    """(d) the category gate mirrors submit_or_dedupe's — a foreign category never folds."""
    _write(queue_dir, _esc(
        'esc-other-cat', category='scope_violation', fingerprint='fp-other',
        resolved_at=_iso(timedelta(seconds=60)),
    ))

    assert _scan(queue_dir) == frozenset()


@pytest.mark.parametrize(
    ('esc_id', 'kwargs'),
    [
        ('esc-pending-1', {'status': 'pending', 'fingerprint': 'fp-pending',
                           'resolved_at': _iso(timedelta(seconds=60))}),
        ('esc-no-resolved-at', {'fingerprint': 'fp-no-ts', 'resolved_at': None}),
        ('esc-no-fingerprint', {'fingerprint': None,
                                'resolved_at': _iso(timedelta(seconds=60))}),
    ],
    ids=['still_pending', 'resolved_at_is_none', 'fingerprint_is_none'],
)
def test_records_missing_a_required_field_are_excluded(queue_dir, esc_id, kwargs):
    """(e) a record must be terminal AND timestamped AND fingerprinted to count."""
    _write(queue_dir, _esc(esc_id, **kwargs))

    assert _scan(queue_dir) == frozenset()


def test_record_living_only_in_the_dated_archive_is_found(queue_dir):
    """(f) the archive tier is scanned, not just the queue root.

    This is the reach that makes the scan UNBOUNDED — the archive is dated and
    shared across projects — so it is also the reach that must never silently
    stop being walked.
    """
    archived = queue_dir / ARCHIVE_SUBDIR / '2026-09-19'
    _write(archived, _esc(
        'esc-archived-1', fingerprint='fp-archived', resolved_at=_iso(timedelta(hours=2)),
    ))

    assert _scan(queue_dir) == frozenset({'fp-archived'})


def test_naive_resolved_at_is_coerced_to_utc_not_rejected(queue_dir):
    """(g) a tzinfo-less timestamp reads as UTC, matching both copies it replaces."""
    naive = (NOW - timedelta(seconds=60)).replace(tzinfo=None).isoformat()
    _write(queue_dir, _esc('esc-naive-1', fingerprint='fp-naive', resolved_at=naive))

    assert _scan(queue_dir) == frozenset({'fp-naive'})


def test_malformed_record_is_skipped_and_its_neighbours_still_return(queue_dir):
    """(h) one unreadable file must not void the whole scan (fail-open per record)."""
    _write(queue_dir, _esc(
        'esc-good-1', fingerprint='fp-good-1', resolved_at=_iso(timedelta(seconds=60)),
    ))
    (queue_dir / 'esc-corrupt.json').write_text('{not valid json at all')
    _write(queue_dir, _esc(
        'esc-good-2', fingerprint='fp-good-2', resolved_at=_iso(timedelta(hours=3)),
    ))

    assert _scan(queue_dir) == frozenset({'fp-good-1', 'fp-good-2'})


def test_missing_queue_dir_returns_empty_rather_than_raising(tmp_path):
    """(i) an absent directory is an empty result — the fail-open direction."""
    assert _scan(tmp_path / 'does-not-exist') == frozenset()


@pytest.mark.parametrize(
    ('age', 'expected'),
    [
        (WINDOW, frozenset({'fp-edge'})),
        (WINDOW + timedelta(microseconds=1), frozenset()),
    ],
    ids=['exactly_one_window_ago_is_included', 'just_past_the_window_is_excluded'],
)
def test_the_window_edge_is_inclusive(queue_dir, age, expected):
    """(j) resolved exactly *window* ago still suppresses; a microsecond later does not."""
    _write(queue_dir, _esc('esc-edge-1', fingerprint='fp-edge', resolved_at=_iso(age)))

    assert _scan(queue_dir) == expected
