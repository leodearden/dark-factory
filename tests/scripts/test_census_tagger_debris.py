"""Tests for scripts/census_tagger_debris.py — the READ-ONLY tagger-debris census.

PRD task epsilon of plans/module-tagger-retirement-prd.md (decision 3). The
census enumerates every task record carrying ``metadata.files_tagged_at`` — the
stamp the retired module tagger left behind — across all six project corpora,
and classifies each on three axes so DF 3113 P4a and DF 3427 can consume a
machine-readable candidate set instead of a prose claim.

NO TEST HERE ASSERTS A COUNT OR TASK ID DERIVED FROM THE LIVE DATABASES.
This norm is inherited verbatim from both sibling suites —
scripts/tests/test_audit_wiped_metadata_files.py and
tests/scripts/test_repair_wiped_metadata_files.py:11-21, the latter citing a
candidate count that moved 40 -> 43 -> 45 across a single task's planning
sessions with one id changing signature class in between. The six corpora are
mutated continuously by six running orchestrators, so a test pinning "the live
DB yields N stamped records" would be a guessed threshold that goes red the
moment any task merges. Every assertion below runs against synthetic tuples or
synthetic tmp_path databases whose contents the test controls exactly.

The one place the required POSITIVE CONTROLS (reify 6068/5602/5632,
dark_factory 3113) are asserted is against the COMMITTED ARTIFACT — a static
repo file, not a live database — which is stable under corpus drift and is
precisely the task's user-observable signal.

Mirrors the sibling split: pure functions get direct pytest coverage;
``main()`` gets subprocess coverage.
"""
from __future__ import annotations

import pytest

from census_tagger_debris import (
    NEVER_RECONCILED,
    NO_PRIOR_SCOPE,
    POST_WIPE_OVERWRITE,
    RECONCILED,
    STATUS_NON_TERMINAL,
    STATUS_TERMINAL,
    ScopeEvent,
    classify_record,
)

# ---------------------------------------------------------------------------
# The classification vocabulary and the pure three-axis classifier.
#
# Every input below is a hand-built tuple. Timestamps are ISO-8601 strings with
# a timezone, the shape measured in BOTH live columns the census compares:
# events.timestamp and metadata.files_tagged_at. The comparison is a plain
# string compare, which is total and correct for same-offset ISO-8601 — the
# stamp and the events are written by the same process family.
# ---------------------------------------------------------------------------

_STAMP = "2026-08-08T01:04:58+00:00"
_BEFORE = "2026-08-01T00:00:00+00:00"
_AFTER = "2026-08-15T00:00:00+00:00"


def _event(timestamp: str, event_type: str = "set_to_plan", event_id: int = 1) -> ScopeEvent:
    return ScopeEvent(
        timestamp=timestamp,
        event_type=event_type,
        event_id=event_id,
        fidelity="lock_level",
        file_count=2,
    )


def test_classification_vocabulary_constants_have_exact_string_values():
    """(a) The six labels are the artifact's public vocabulary.

    DF 3113 P4a and DF 3427 will read these strings out of the committed JSON,
    so a rename is a breaking change to a consumer that cannot see this repo's
    constants. Pinning the literals here makes that breakage a failing test
    rather than a silently-unjoinable artifact.
    """
    assert STATUS_TERMINAL == "terminal"
    assert STATUS_NON_TERMINAL == "non_terminal"
    assert RECONCILED == "plan_reconciled"
    assert NEVER_RECONCILED == "never_reconciled"
    assert POST_WIPE_OVERWRITE == "post_wipe_overwrite"
    assert NO_PRIOR_SCOPE == "no_prior_scope"


@pytest.mark.parametrize("status", ["done", "cancelled"])
def test_terminal_statuses_classify_terminal(status):
    """(b) The terminal axis is the repair's own allowlist, not a re-spelling."""
    result = classify_record(_STAMP, status, [])
    assert result.status_class == STATUS_TERMINAL


@pytest.mark.parametrize(
    "status", ["pending", "in-progress", "blocked", "deferred", "merge-deferred"]
)
def test_every_other_status_classifies_non_terminal(status):
    """(b) An ALLOWLIST, so a status the system grows later falls on the
    non_terminal side — reported as a live victim rather than silently
    excluded from the population the census exists to find."""
    result = classify_record(_STAMP, status, [])
    assert result.status_class == STATUS_NON_TERMINAL


def test_scope_event_after_the_stamp_is_plan_reconciled():
    """(c) A scope event postdating the stamp means the tagger's guess was
    superseded by a real derivation — the record is no longer a live victim."""
    result = classify_record(_STAMP, "pending", [_event(_AFTER)])
    assert result.reconciliation == RECONCILED
    assert result.wipe_signature == NO_PRIOR_SCOPE


def test_scope_event_before_the_stamp_is_post_wipe_overwrite():
    """(c) A scope event predating the stamp means an authoritative scope
    EXISTED and the tagger stamped over it — the damaging case."""
    result = classify_record(_STAMP, "pending", [_event(_BEFORE)])
    assert result.wipe_signature == POST_WIPE_OVERWRITE
    assert result.reconciliation == NEVER_RECONCILED


def test_events_on_both_sides_of_the_stamp_yield_both_classifications():
    """(c) The two axes are INDEPENDENT: a record can have been stamped over a
    prior scope AND later reconciled. Collapsing them to one label would lose
    exactly the distinction the repair needs."""
    result = classify_record(
        _STAMP, "pending", [_event(_BEFORE, event_id=1), _event(_AFTER, event_id=2)]
    )
    assert result.reconciliation == RECONCILED
    assert result.wipe_signature == POST_WIPE_OVERWRITE


def test_no_scope_events_at_all_is_never_reconciled_and_no_prior_scope():
    """(c) The live-victim cell: the tagger's guess is still the only scope
    this record has ever had."""
    result = classify_record(_STAMP, "pending", [])
    assert result.reconciliation == NEVER_RECONCILED
    assert result.wipe_signature == NO_PRIOR_SCOPE


def test_event_exactly_at_the_stamp_decides_neither_axis():
    """(d) THE BOUNDARY, pinned explicitly rather than left to inference.

    Comparison is strict (``>`` / ``<``), so an event bearing the same instant
    as the stamp is evidence of neither reconciliation nor overwrite. The two
    writes are not ordered with respect to each other at equal timestamps, and
    inventing an order would be a guess presented as a measurement.
    """
    result = classify_record(_STAMP, "pending", [_event(_STAMP)])
    assert result.reconciliation == NEVER_RECONCILED
    assert result.wipe_signature == NO_PRIOR_SCOPE


def test_reconciliation_evidence_names_the_deciding_event():
    """(e) INV-2: no classification is a prose-only claim.

    The EARLIEST post-stamp event is the deciding one — the first thing that
    superseded the tagger's guess.
    """
    events = [
        _event("2026-08-20T00:00:00+00:00", event_type="phase_skipped", event_id=9),
        _event("2026-08-10T00:00:00+00:00", event_type="set_to_plan", event_id=4),
    ]
    result = classify_record(_STAMP, "pending", events)

    assert result.reconciliation == RECONCILED
    assert result.reconciled_by.event_type == "set_to_plan"
    assert result.reconciled_by.event_id == 4
    assert result.reconciled_by.timestamp == "2026-08-10T00:00:00+00:00"


def test_overwrite_evidence_names_the_latest_prior_scope_event():
    """(e) The LATEST pre-stamp event is the deciding one — the most recent
    authoritative scope that the tagger's stamp wrote over."""
    events = [
        _event("2026-07-01T00:00:00+00:00", event_type="set_to_plan", event_id=2),
        _event("2026-08-07T00:00:00+00:00", event_type="phase_skipped", event_id=7),
    ]
    result = classify_record(_STAMP, "done", events)

    assert result.wipe_signature == POST_WIPE_OVERWRITE
    assert result.preceded_by.event_type == "phase_skipped"
    assert result.preceded_by.event_id == 7
    assert result.preceded_by.timestamp == "2026-08-07T00:00:00+00:00"


def test_absent_evidence_is_explicitly_null_not_a_missing_key():
    """(e) An unclassified axis still carries its evidence keys, all None. A
    MISSING key in the artifact would be indistinguishable from a serializer
    bug; a present null says "looked, found nothing"."""
    result = classify_record(_STAMP, "pending", [])

    assert result.reconciled_by._asdict() == {
        "event_type": None,
        "event_id": None,
        "timestamp": None,
    }
    assert result.preceded_by._asdict() == {
        "event_type": None,
        "event_id": None,
        "timestamp": None,
    }
