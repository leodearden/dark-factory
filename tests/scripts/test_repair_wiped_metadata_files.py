"""Tests for scripts/repair_wiped_metadata_files.py — the WRITE counterpart to
the read-only audit of the DONE-path ``metadata.files`` wipe.

Task 3329 backfills ``metadata.files`` on the records that
``TaskWorkflow._reconcile_metadata_files_for_done``
(orchestrator/src/orchestrator/workflow.py:2007) already emptied. Task 3113 owns
the forward fix and is still ``pending``, so the wiper is LIVE and the damaged
population keeps growing — which is precisely why the repair re-runs the audit
in-process on every invocation instead of consuming a pasted candidate list.

NO TEST HERE ASSERTS A COUNT OR TASK ID DERIVED FROM THE LIVE DATABASES.
This norm is inherited verbatim from scripts/tests/test_audit_wiped_metadata_files.py:
tasks.db and runs.db are mutated continuously by the running orchestrator, so a
test pinning "the live DB yields N candidates" would be a guessed threshold that
goes red the moment another task merges. It is not a hypothetical here — three
read-only runs of the audit across this task's planning sessions measured the
candidate count moving 40 -> 43 -> 45, with one id (3086) MOVING between
signature classes in between. Every assertion below runs against synthetic
``WipeCandidate`` tuples or synthetic temp databases whose contents the test
controls exactly, and no test points at /home/leo/src/dark-factory.

Mirrors the audit's split: pure functions get direct pytest coverage; ``main()``
gets subprocess coverage.
"""
from __future__ import annotations

from audit_wiped_metadata_files import (
    CLEAN_MERGE_SHA,
    CONFIRMED_NULL_SHA_DONE_PATH,
    CONTRADICTED_REAL_MERGE_SHA,
    FIDELITY_FILE_LEVEL,
    FIDELITY_LOCK_LEVEL,
    NO_MERGE_EVENT,
    NO_SUCCESSFUL_MERGE_SHA,
    WipeCandidate,
)

from repair_wiped_metadata_files import select_repairable_candidates

# ---------------------------------------------------------------------------
# Synthetic candidate builder.
#
# WipeCandidate carries a LEADING ``tag`` field, so it must be constructed by
# KEYWORD — a positional build would silently shift every field by one and make
# the fidelity/signature assertions below meaningless.
# ---------------------------------------------------------------------------


def _candidate(
    task_id=1,
    *,
    tag="master",
    status="done",
    plan_files=("orchestrator/src/orchestrator/workflow.py",),
    plan_files_source="meta_root_plan",
    plan_files_fidelity=FIDELITY_FILE_LEVEL,
    wipe_signature=CONFIRMED_NULL_SHA_DONE_PATH,
) -> WipeCandidate:
    return WipeCandidate(
        tag=tag,
        task_id=task_id,
        status=status,
        plan_files=tuple(plan_files),
        plan_files_source=plan_files_source,
        plan_files_fidelity=plan_files_fidelity,
        wipe_signature=wipe_signature,
    )


# ---------------------------------------------------------------------------
# select_repairable_candidates — constraints 1 (file-level only) and 2 (never
# repair a contradicted candidate), expressed as an EXCLUSION rule.
# ---------------------------------------------------------------------------


def test_select_drops_a_contradicted_candidate():
    """(a) constraint 2: a null-sha row AND a real merge sha coexist, so the
    null-sha DONE path is not what emptied this task. Never repair it."""
    kept = _candidate(1)
    dropped = _candidate(2, wipe_signature=CONTRADICTED_REAL_MERGE_SHA)

    assert select_repairable_candidates([kept, dropped]) == [kept]


def test_select_drops_a_lock_level_candidate_even_with_a_confirmed_signature():
    """(b) constraint 1: lock-level plan_files are MODULE paths, not plan.files
    entries. A confirmed wipe signature does not license writing them into
    metadata.files — fidelity is checked independently of signature."""
    dropped = _candidate(
        3,
        plan_files=("orchestrator/src/orchestrator",),
        plan_files_fidelity=FIDELITY_LOCK_LEVEL,
        wipe_signature=CONFIRMED_NULL_SHA_DONE_PATH,
    )

    assert select_repairable_candidates([dropped]) == []


def test_select_keeps_every_non_contradicted_file_level_signature():
    """(c) all four enumerated non-contradicted signatures are repairable —
    including CLEAN_MERGE_SHA, which the audit's own constant docstring states
    is NOT an exoneration: the ``if err is not None: files = []`` branch wipes
    WITH a real merge sha present."""
    candidates = [
        _candidate(10, wipe_signature=CONFIRMED_NULL_SHA_DONE_PATH),
        _candidate(11, wipe_signature=NO_MERGE_EVENT),
        _candidate(12, wipe_signature=NO_SUCCESSFUL_MERGE_SHA),
        _candidate(13, wipe_signature=CLEAN_MERGE_SHA),
    ]

    assert select_repairable_candidates(candidates) == candidates


def test_select_keeps_a_signature_class_the_audit_grows_later():
    """(d) THE POINT OF THE EXCLUSION FRAMING. An inclusion list of the three
    signatures the task description enumerates would have silently dropped
    CLEAN_MERGE_SHA, which did not exist in that snapshot but has members
    today. A future class must reach the report, not vanish from the feed."""
    future = _candidate(14, wipe_signature="some_future_signature")

    assert select_repairable_candidates([future]) == [future]


def test_select_preserves_input_order_and_does_not_mutate_the_input():
    """Ordering is the operator's audit trail; the caller's list is not ours."""
    candidates = [
        _candidate(30),
        _candidate(20, wipe_signature=CONTRADICTED_REAL_MERGE_SHA),
        _candidate(10),
    ]
    original = list(candidates)

    selected = select_repairable_candidates(candidates)

    assert [c.task_id for c in selected] == [30, 10]
    assert candidates == original


def test_select_on_an_empty_feed_returns_empty():
    assert select_repairable_candidates([]) == []
