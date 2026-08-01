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

import asyncio
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

# Both modules live in <repo>/scripts, which tests/scripts/conftest.py puts on
# sys.path at collection time. scripts/ is a flat script directory, not a
# package src root, so it is deliberately absent from [tool.pyright] extraPaths
# in the root pyproject.toml — hence the targeted ignore, matching the same
# convention at tests/scripts/test_reviewer_redundancy_diagnostic.py:6.
from audit_wiped_metadata_files import (  # pyright: ignore[reportMissingImports]
    CLEAN_MERGE_SHA,
    AuditCoverage,
    CONFIRMED_NULL_SHA_DONE_PATH,
    CONTRADICTED_REAL_MERGE_SHA,
    FIDELITY_FILE_LEVEL,
    FIDELITY_LOCK_LEVEL,
    NO_MERGE_EVENT,
    NO_SUCCESSFUL_MERGE_SHA,
    WipeCandidate,
)

from repair_wiped_metadata_files import (  # pyright: ignore[reportMissingImports]
    CLIENT_NAME,
    EXIT_NOTHING_SCANNED,
    EXIT_NO_ROOT,
    EXIT_OK,
    EXIT_SERVER_UNREACHABLE,
    PROVENANCE_KEY,
    REPAIR,
    REPAIR_TASK_ID,
    SKIP_FILES_PRESENT,
    SKIP_MISSING,
    SKIP_NOT_TERMINAL,
    ALL_DISPOSITIONS,
    FAILED,
    SKIP_CONTRADICTED,
    SKIP_LOCK_CHARTER,
    SKIP_LOCK_LEVEL_FIDELITY,
    RepairOutcome,
    RepairResult,
    build_repair_payload,
    classify_live_task,
    format_summary,
    plan_files_rejection_reason,
    repair_one,
    repair_project,
    select_repairable_candidates,
    writable_plan_files,
)

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


# ---------------------------------------------------------------------------
# plan_files_rejection_reason — the lock-charter PRE-CHECK.
#
# The interceptor's _reject_directory_locks_in_update_metadata
# (fused-memory/src/fused_memory/middleware/task_interceptor.py:5284) rejects a
# metadata.files write carrying a DIRECTORY entry with an opaque
# lock_charter_error. Pre-checking turns that into a named, attributable skip.
# ---------------------------------------------------------------------------


def test_plan_files_rejection_reason_accepts_an_all_file_level_list():
    """(a) every entry is file-level -> no reason. The systemd unit paths are
    deliberate: `timer` and `service` ARE in shared.locking.CODE_EXTENSIONS, so
    a naive "no dot-py suffix means directory" check would wrongly reject real
    repairable paths measured in today's population."""
    candidate = _candidate(
        1,
        plan_files=(
            "orchestrator/src/orchestrator/workflow.py",
            "scripts/reclaim-orphaned-worktrees.timer",
            "scripts/reclaim-orphaned-worktrees.service",
            "scripts/restart-all-orchestrators.sh",
            "dark-factory-orchestrator.yaml",
            "pyproject.toml",
        ),
    )

    assert plan_files_rejection_reason(candidate) is None


def test_plan_files_rejection_reason_names_every_directory_entry():
    """(b) a directory-looking entry -> a non-empty reason NAMING it, so the
    operator can act on the summary without re-deriving which path tripped."""
    candidate = _candidate(
        2,
        plan_files=(
            "orchestrator/src/orchestrator/workflow.py",
            "orchestrator/src/orchestrator",
            "shared/src/shared/",
        ),
    )

    reason = plan_files_rejection_reason(candidate)

    assert reason
    assert "orchestrator/src/orchestrator" in reason
    assert "shared/src/shared/" in reason
    # The clean entry is not slandered as an offender.
    assert "workflow.py" not in reason


def test_plan_files_rejection_reason_rejects_an_empty_plan_files():
    """(c) nothing to restore. A write setting files to [] would re-perform the
    very wipe this script exists to undo, so it must never be issued."""
    reason = plan_files_rejection_reason(_candidate(3, plan_files=()))

    assert reason
    assert "empty" in reason.lower()


def test_plan_files_rejection_reason_rejects_a_list_of_only_blanks():
    """A whitespace-only entry carries no scope either. directory_locks skips
    blanks, so a bare directory check would call this list clean and issue a
    write whose files list the server then coerces to nothing."""
    reason = plan_files_rejection_reason(_candidate(4, plan_files=("", "   ")))

    assert reason
    assert "empty" in reason.lower()


def test_the_gate_and_the_payload_read_one_sanitised_list(tmp_path):
    """VALIDATE-THEN-WRITE DIVERGENCE — the MIXED case, which is the one that
    actually reaches a write.

    The all-blank list above is rejected outright, so the divergence never
    shows there. ``("a.py", "   ")`` is the dangerous shape: the gate passes it
    on the filtered ``["a.py"]``, and a payload built from the RAW tuple would
    then ship ``["a.py", "   "]``. Nothing downstream catches that —
    ``shared.locking.directory_locks`` skips blanks, so the interceptor accepts
    the junk entry and it lands in metadata.files verbatim. The audit's
    ``_coerce_file_list`` KEEPS whitespace-only entries by design (see
    test_classify_live_task_agrees_with_the_audit_on_a_whitespace_only_entry),
    so this is a real reachable input, not a hypothetical.

    Asserted at all three layers that must agree: the helper, the payload, and
    the bytes actually handed to update_task.
    """
    candidate = _candidate(40, plan_files=("a.py", "   ", "", "b.py"))

    assert plan_files_rejection_reason(candidate) is None
    assert writable_plan_files(candidate) == ("a.py", "b.py")
    assert build_repair_payload(candidate, now_iso=_NOW)["files"] == ["a.py", "b.py"]

    client = _FakeClient()
    outcome = asyncio.run(repair_one(client, _ROOT, candidate, now_iso=_NOW))

    _, arguments = client.calls[0]
    assert json.loads(arguments["metadata"])["files"] == ["a.py", "b.py"]
    # And the report cannot claim a wider scope than the write carried.
    assert outcome.files == ("a.py", "b.py")


# ---------------------------------------------------------------------------
# build_repair_payload — a MINIMAL ADDITIVE PATCH, never a lifecycle write.
#
# now_iso is INJECTED rather than stamped inside, so these tests pin an exact
# value instead of matching a regex against a clock.
# ---------------------------------------------------------------------------

_NOW = "2026-08-01T00:00:00+00:00"


def test_build_repair_payload_has_exactly_two_top_level_keys():
    payload = build_repair_payload(_candidate(5), now_iso=_NOW)

    assert set(payload) == {"files", PROVENANCE_KEY}


def test_build_repair_payload_files_is_a_plain_list_in_plan_order():
    """(a) plan_files is a TUPLE on the candidate; the payload must carry a
    JSON-serialisable list, in the plan's own order."""
    files = (
        "orchestrator/src/orchestrator/workflow.py",
        "shared/src/shared/locking.py",
        "scripts/audit_wiped_metadata_files.py",
    )
    payload = build_repair_payload(_candidate(6, plan_files=files), now_iso=_NOW)

    assert payload["files"] == list(files)
    assert isinstance(payload["files"], list)


def test_build_repair_payload_provenance_carries_this_tasks_id_and_the_audit_fields():
    """(b) the record says WHO wrote it, from WHICH recovered source, under
    WHICH wipe signature and fidelity, and WHEN — everything a later reader
    needs to judge the backfill without re-deriving the audit."""
    candidate = _candidate(
        7,
        plan_files_source="phase_skipped_event",
        wipe_signature=NO_MERGE_EVENT,
        plan_files_fidelity=FIDELITY_FILE_LEVEL,
    )

    provenance = build_repair_payload(candidate, now_iso=_NOW)[PROVENANCE_KEY]

    assert provenance["task"] == REPAIR_TASK_ID == "3329"
    assert provenance["src"] == "phase_skipped_event"
    assert provenance["sig"] == NO_MERGE_EVENT
    assert provenance["fidelity"] == FIDELITY_FILE_LEVEL
    assert provenance["at"] == _NOW


def test_build_repair_payload_never_carries_done_provenance():
    """(c) DoneProvenanceWriteAuthorityError floor: update_task rejects any
    metadata write carrying done_provenance (set_task_status is its only
    sanctioned writer). Checked at the top level AND nested."""
    payload = build_repair_payload(_candidate(8), now_iso=_NOW)

    assert "done_provenance" not in payload
    for value in payload.values():
        if isinstance(value, dict):
            assert "done_provenance" not in value


def test_build_repair_payload_is_never_a_lifecycle_or_tagger_write():
    """(d) no status (StatusWriteAuthorityError floor), and no modules /
    files_tagged_at — this patch restores scope, it does not re-run the tagger
    or move the task through its lifecycle."""
    payload = build_repair_payload(_candidate(9), now_iso=_NOW)

    for forbidden in ("status", "modules", "files_tagged_at"):
        assert forbidden not in payload


def test_build_repair_payload_round_trips_through_json():
    """(e) the payload is handed to json.dumps by the write path, so anything
    non-serialisable here becomes a mid-batch TypeError instead of a write."""
    payload = build_repair_payload(_candidate(11), now_iso=_NOW)

    assert json.loads(json.dumps(payload)) == payload


def test_build_repair_payload_provenance_key_uses_the_x_namespace():
    """docs/task-authoring.md Tier-C: a one-off annotation key must live in the
    x_ forward-compat namespace, which shared/src/shared/task_metadata.py:933
    exempts from the unknown_key census warning. A bare
    `files_backfill_provenance` is not in _BLESSED_METADATA_KEYS and would emit
    one schema_warning line per repaired task — 35+ lines of exactly the drift
    noise that census exists to surface."""
    assert PROVENANCE_KEY.startswith("x_")
    assert PROVENANCE_KEY == "x_files_backfill_provenance"


# ---------------------------------------------------------------------------
# classify_live_task — the immediately-before-write re-read gate.
#
# NOT CEREMONY. Task 3113's correction addendum documents _stamp_optimistic_path
# (orchestrator/src/orchestrator/workflow.py:4550 — the method MOVED from 4413
# since this plan was first written, so locate it by NAME) writing a task's
# STALE dispatch-time metadata snapshot back as a whole blob. A repair applied
# underneath a live workflow is therefore not merely racy: it is guaranteed to
# be undone.
# ---------------------------------------------------------------------------


def test_classify_live_task_repairs_a_terminal_task_with_empty_files():
    """(a) both terminal statuses, with files empty and with files absent."""
    candidate = _candidate(20)

    assert classify_live_task({"status": "done", "metadata": {"files": []}}, candidate) == REPAIR
    assert classify_live_task({"status": "cancelled", "metadata": {}}, candidate) == REPAIR


def test_classify_live_task_skips_every_non_terminal_status():
    """(b) a live workflow holds a dispatch-time metadata snapshot it writes
    back wholesale, so repairing underneath one would be undone."""
    candidate = _candidate(21)

    for status in ("pending", "in-progress", "blocked", "deferred", "review"):
        live = {"status": status, "metadata": {"files": []}}
        assert classify_live_task(live, candidate) == SKIP_NOT_TERMINAL, status


def test_classify_live_task_terminal_set_is_an_allowlist_so_new_statuses_fail_closed():
    """A status the system grows later must be SKIPPED, not written to — the
    check is `status in {done, cancelled}`, never `status not in {...}`."""
    live = {"status": "some_future_status", "metadata": {"files": []}}

    assert classify_live_task(live, _candidate(22)) == SKIP_NOT_TERMINAL


def test_classify_live_task_skips_a_task_whose_files_are_already_present():
    """(c) someone else repaired it, or this is a re-run. This arm is what
    makes the script idempotent and makes the post-3113 second pass a cheap
    re-run rather than a re-derivation."""
    live = {"status": "done", "metadata": {"files": ["a.py"]}}

    assert classify_live_task(live, _candidate(23)) == SKIP_FILES_PRESENT


def test_classify_live_task_skips_a_missing_or_empty_live_read():
    """(d) get_task returned nothing usable — never write blind."""
    for live in (None, {}, "not-a-dict", {"status": ""}):
        assert classify_live_task(live, _candidate(24)) == SKIP_MISSING, live


def test_classify_live_task_still_repairs_a_corrupt_files_value():
    """(e) the emptiness predicate is the audit's _coerce_file_list, NOT bare
    truthiness. A non-list `files`, or a list of only blanks/non-strings, is
    corrupt data carrying no scope — the audit already nominated the task on
    exactly that reading, so a divergent predicate here would make the two
    disagree about the same record."""
    candidate = _candidate(25)

    for corrupt in ("a.py", 7, {"a": 1}, [""], [None], [None, ""]):
        live = {"status": "done", "metadata": {"files": corrupt}}
        assert classify_live_task(live, candidate) == REPAIR, corrupt


def test_classify_live_task_agrees_with_the_audit_on_a_whitespace_only_entry():
    """THE BOUNDARY OF 'byte-identical', asserted rather than left implicit.

    _coerce_file_list drops empty strings and non-strings but KEEPS a
    whitespace-only entry (`"  "` is a truthy str). A task whose files is
    `["  "]` therefore has non-empty metadata_files by the audit's reading and
    is never nominated as a candidate at all. The repair must read it the same
    way — treating it as 'no scope' here would make the repair claim damage the
    audit does not see, which is the divergence reusing _coerce_file_list
    exists to prevent."""
    live = {"status": "done", "metadata": {"files": ["  "]}}

    assert classify_live_task(live, _candidate(27)) == SKIP_FILES_PRESENT


# ---------------------------------------------------------------------------
# repair_one — the write itself, against a FAKE client. No network, no MCP
# server, no live database.
# ---------------------------------------------------------------------------

_ROOT = "/tmp/proj"


class _FakeClient:
    """Records every call_tool invocation; optionally raises or returns a shape."""

    def __init__(self, *, raises=None, returns=None):
        self.calls: list[tuple[str, dict]] = []
        self._raises = raises
        self._returns = returns if returns is not None else {"success": True}

    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        if self._raises is not None:
            raise self._raises
        return self._returns


class _DuplicateCandidateKeyError(RuntimeError):
    """Shape-alike for the real fused_memory DuplicateCandidateKeyError.

    Defined in fused-memory/src/fused_memory/backends/task_backend_errors.py:37
    and raised from the update_task path at sqlite_task_backend.py:2543:
    candidate_key is recomputed from (title, files) on EVERY metadata-touching
    update, so backfilling files onto a batch of tasks can genuinely collide
    with another non-cancelled row. Simulated rather than imported so this test
    stays a pure unit test of repair_one's error handling."""


def test_repair_one_issues_exactly_one_merge_mode_update_task():
    """(a) one update_task carrying id/project_root/metadata/metadata_mode."""
    client = _FakeClient()
    candidate = _candidate(2464, plan_files=("a.py", "b.py"))

    result = asyncio.run(repair_one(client, _ROOT, candidate, now_iso=_NOW))

    assert result.disposition == REPAIR
    assert len(client.calls) == 1
    name, arguments = client.calls[0]
    assert name == "update_task"
    assert arguments["id"] == "2464"
    assert isinstance(arguments["id"], str)
    assert arguments["project_root"] == _ROOT
    assert arguments["metadata_mode"] == "merge"
    assert json.loads(arguments["metadata"]) == build_repair_payload(
        candidate, now_iso=_NOW
    )


def test_repair_one_never_uses_replace_or_append_mode():
    """(b) replace is the primitive that CAUSED this wipe class: _execute_combine
    (task_interceptor.py:1850) writes a 3-key blob in replace mode and
    _merge_metadata (sqlite_task_backend.py:3301) returns `incoming` verbatim for
    that mode, deleting every untouched key. A repair must not use the wiper's
    own primitive."""
    client = _FakeClient()

    asyncio.run(repair_one(client, _ROOT, _candidate(1), now_iso=_NOW))

    _, arguments = client.calls[0]
    assert arguments["metadata_mode"] not in ("replace", "append")
    assert "append" not in arguments


def test_repair_one_passes_no_status_argument():
    """(c) StatusWriteAuthorityError floor (sqlite_task_backend.py:2575) — a
    metadata write carrying status is rejected outright."""
    client = _FakeClient()

    asyncio.run(repair_one(client, _ROOT, _candidate(1), now_iso=_NOW))

    _, arguments = client.calls[0]
    assert "status" not in arguments
    assert "done_provenance" not in json.loads(arguments["metadata"])


def test_repair_one_captures_a_raised_error_instead_of_propagating():
    """(d) a candidate_key collision is a genuine per-task risk across a
    35-task batch. It must be captured, attributed and reported — never allowed
    to abort the batch and strand the remaining candidates."""
    client = _FakeClient(
        raises=_DuplicateCandidateKeyError("candidate_key collides with task 9999")
    )

    result = asyncio.run(repair_one(client, _ROOT, _candidate(2464), now_iso=_NOW))

    assert result.disposition == FAILED
    assert result.task_id == 2464
    assert "9999" in (result.detail or "")


def test_repair_one_treats_an_error_shaped_result_as_failed():
    """(e) the server can answer with an error PAYLOAD rather than raising. A
    truthy-response check would count that as a success and report a repair
    that never happened."""
    for payload in (
        {"error": "lock_charter_error: directory lock rejected"},
        {"success": False, "error_type": "DuplicateCandidateKeyError"},
    ):
        client = _FakeClient(returns=payload)

        result = asyncio.run(repair_one(client, _ROOT, _candidate(3), now_iso=_NOW))

        assert result.disposition == FAILED, payload
        assert result.detail, payload


def test_repair_one_accepts_a_plain_success_shape():
    """The complement of the above: an ordinary success must not be misread as
    a failure just because it carries no explicit success flag."""
    for payload in ({}, {"success": True}, {"id": "2464", "status": "done"}):
        client = _FakeClient(returns=payload)

        result = asyncio.run(repair_one(client, _ROOT, _candidate(4), now_iso=_NOW))

        assert result.disposition == REPAIR, payload


def test_repair_one_carries_the_candidates_tag_verbatim():
    """(f) THE TAG IS HALF THE PRIMARY KEY, AND OMITTING IT IS A SILENT RETARGET.

    ``WipeCandidate.tag`` is the FIRST field of the audit's NamedTuple
    (scripts/audit_wiped_metadata_files.py:501) because the audit nominates
    candidates under ``(tag, id)``: ``load_task_records``'s docstring (:460-465)
    states the schema "permits the same numeric id under two tags and collapsing
    them would silently merge two distinct tasks", and the coverage comment
    (:609-612) states one plan record "matches two DISTINCT tasks" in that case.

    ``update_task`` accepts an optional ``tag``, and when it is absent the
    backend does not error — it substitutes ``DEFAULT_TAG = 'master'``
    (fused-memory/src/fused_memory/backends/sqlite_task_backend.py:127, applied
    on the update_task path at :2612). So dropping the tag writes the recovered
    scope onto a DIFFERENT task that merely shares the id, which is exactly the
    failure the four-gate model exists to prevent.

    Asserted with a NON-master tag so a hardcoded or defaulted ``'master'``
    cannot pass, and on the exact value — never merely ``"tag" in arguments``.
    """
    client = _FakeClient()
    candidate = _candidate(77, tag="feature-x")

    result = asyncio.run(repair_one(client, _ROOT, candidate, now_iso=_NOW))

    assert result.disposition == REPAIR
    assert len(client.calls) == 1
    name, arguments = client.calls[0]
    assert name == "update_task"
    assert arguments["tag"] == "feature-x"


# ---------------------------------------------------------------------------
# format_summary — THE HONESTY ARTIFACT.
#
# The candidate list is an OBSERVABLE SUBSET. Most tasks have no recoverable
# plan scope at all and are UNKNOWN — neither clean nor damaged. A summary that
# let a reader mistake this run for "fixed the blast radius" would be the
# no-silent-fail-soft violation named in docs/legibility/design-invariants.md.
# ---------------------------------------------------------------------------

_COVERAGE = AuditCoverage(
    project_root=_ROOT,
    total_tasks=3363,
    tasks_with_file_level_signal=951,
    tasks_with_lock_level_signal_only=135,
    tasks_without_plan_signal=2277,
    plan_records_without_task=6,
)


def _outcome(task_id, disposition, *, detail=None, files=("a.py",)):
    return RepairOutcome(
        task_id=task_id,
        tag="master",
        disposition=disposition,
        files=tuple(files),
        detail=detail,
    )


def _result(outcomes=(), *, applied=True):
    return RepairResult(
        project_root=_ROOT,
        applied=applied,
        outcomes=list(outcomes),
        coverage=_COVERAGE,
    )


def _disposition_counts(summary: str) -> dict[str, int]:
    """Parse the rendered '-- dispositions --' block back into {name: count}.

    Asserting on the PARSED COUNT is the whole point. `disposition in summary`
    is satisfied by the itemised sections and by the disposition strings alone,
    so it would still pass if format_summary stopped printing zero buckets
    entirely — which is the exact regression the caller below exists to catch.
    """
    lines = summary.splitlines()
    counts: dict[str, int] = {}
    for line in lines[lines.index("  -- dispositions --") + 1:]:
        parts = line.split()
        if len(parts) != 2 or not parts[1].isdigit():
            break
        counts[parts[0]] = int(parts[1])
    return counts


def test_format_summary_prints_every_disposition_including_the_zero_ones():
    """(a) a bucket that is merely ABSENT reads as 'did not happen' when it may
    mean 'was never evaluated'. Every disposition gets a line WITH ITS COUNT,
    and the zeros are rendered as zeros rather than omitted."""
    summary = format_summary(_result([_outcome(1, REPAIR)]))

    assert _disposition_counts(summary) == {
        disposition: (1 if disposition == REPAIR else 0)
        for disposition in ALL_DISPOSITIONS
    }


def test_format_summary_lists_each_failure_individually_with_its_error():
    """(b) an aggregate '2 failed' is not actionable. Each id and error text
    must appear so the operator can act without re-deriving them."""
    summary = format_summary(
        _result(
            [
                _outcome(2464, FAILED, detail="DuplicateCandidateKeyError: collides with 9999"),
                _outcome(2465, FAILED, detail="server returned an error: lock_charter_error"),
            ]
        )
    )

    assert "2464" in summary and "9999" in summary
    assert "2465" in summary and "lock_charter_error" in summary


def test_format_summary_lists_each_lock_charter_skip_individually():
    """(b) same for the pre-check skips — the offending path must be named."""
    summary = format_summary(
        _result(
            [
                _outcome(
                    77,
                    SKIP_LOCK_CHARTER,
                    detail="directory-classified entries: orchestrator/src/orchestrator",
                )
            ]
        )
    )

    assert "77" in summary
    assert "orchestrator/src/orchestrator" in summary


def test_format_summary_echoes_the_audit_coverage_block():
    """(c) the coverage counts travel with the result, verbatim."""
    summary = format_summary(_result([_outcome(1, REPAIR)]))

    for count in ("3363", "951", "135", "2277"):
        assert count in summary, count


def test_format_summary_states_the_observable_subset_caveat_even_with_zero_repairs():
    """(d) THE POINT. A run that repaired nothing must still say that 2277
    tasks are UNKNOWN — neither clean nor damaged — and must not read as
    'fixed the blast radius'."""
    summary = format_summary(_result([]))

    lowered = summary.lower()
    assert "observable subset" in lowered
    assert "unknown" in lowered
    assert "neither clean nor damaged" in lowered
    assert "2277" in summary


def test_format_summary_labels_a_dry_run_and_says_no_write_was_attempted():
    """(e) a dry-run summary that looked like an apply summary would let an
    operator believe a repair landed when nothing was written."""
    summary = format_summary(_result([_outcome(1, REPAIR)], applied=False))

    lowered = summary.lower()
    assert "dry run" in lowered or "dry-run" in lowered
    assert "no write" in lowered or "nothing was written" in lowered


def test_format_summary_apply_run_is_not_labelled_a_dry_run():
    """The complement: an apply run must not carry the dry-run disclaimer."""
    summary = format_summary(_result([_outcome(1, REPAIR)], applied=True))

    assert "no write was attempted" not in summary.lower()


def test_format_summary_reports_the_exclusions_as_decisions():
    """A contradicted or lock-level candidate was DECIDED against, not absent.
    Both must appear with their ids so the report and the audit agree on the
    same population."""
    summary = format_summary(
        _result(
            [
                _outcome(3086, SKIP_CONTRADICTED, detail="null-sha row and a REAL merge sha"),
                _outcome(3087, SKIP_LOCK_LEVEL_FIDELITY, detail="module paths, not plan.files"),
            ]
        )
    )

    assert "3086" in summary
    assert "3087" in summary


# ---------------------------------------------------------------------------
# main() / the CLI — by subprocess, mirroring the audit test's pattern.
#
# Every fixture below is a SYNTHETIC temp project root. Not one test points at
# /home/leo/src/dark-factory: those databases mutate continuously, so an
# assertion derived from them would be a guessed threshold.
# ---------------------------------------------------------------------------

_TASKS_SCHEMA = """
CREATE TABLE tasks (
    tag           TEXT NOT NULL DEFAULT 'master',
    id            INTEGER NOT NULL,
    title         TEXT NOT NULL,
    description   TEXT,
    details       TEXT,
    test_strategy TEXT,
    status        TEXT NOT NULL,
    priority      TEXT,
    metadata      TEXT,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (tag, id)
);
"""

_EVENTS_SCHEMA = """
CREATE TABLE events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    run_id      TEXT    NOT NULL,
    task_id     TEXT,
    event_type  TEXT    NOT NULL,
    phase       TEXT,
    role        TEXT,
    data        TEXT    DEFAULT '{}',
    cost_usd    REAL,
    duration_ms INTEGER
);
"""


def _make_tasks_db(dir_path: Path, rows: list[dict]) -> Path:
    """Copied from scripts/tests/test_audit_wiped_metadata_files.py — the two
    test directories cannot share imports (same note as
    orchestrator/tests/test_deterministic_runner.py:31-32). Schemas mirror the
    live ones so the audit exercises real column shapes."""
    db_path = dir_path / "tasks.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_TASKS_SCHEMA)
        for row in rows:
            metadata = row.get("metadata")
            if metadata is not None and not isinstance(metadata, str):
                metadata = json.dumps(metadata)
            conn.execute(
                "INSERT INTO tasks (tag, id, title, description, details, "
                "test_strategy, status, priority, metadata, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row.get("tag", "master"),
                    row["id"],
                    row.get("title", f"task {row['id']}"),
                    None,
                    None,
                    None,
                    row.get("status", "done"),
                    "medium",
                    metadata,
                    "2026-08-01T00:00:00+00:00",
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _make_runs_db(dir_path: Path, events: list[dict]) -> Path:
    db_path = dir_path / "runs.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_EVENTS_SCHEMA)
        for i, event in enumerate(events):
            data = event.get("data")
            if data is not None and not isinstance(data, str):
                data = json.dumps(data)
            conn.execute(
                "INSERT INTO events (timestamp, run_id, task_id, event_type, "
                "phase, role, data, cost_usd, duration_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"2026-08-01T00:00:{i:02d}+00:00",
                    "run-1",
                    None if event.get("task_id") is None else str(event["task_id"]),
                    event["event_type"],
                    None,
                    None,
                    data,
                    None,
                    None,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _make_project(tmp_path, tasks=(), events=(), plans=(), name="proj") -> Path:
    """Build a whole synthetic project root with the three inputs audit_project
    reads. *plans* is a list of (worktree_name, plan_dict)."""
    root = tmp_path / name
    tasks_dir = root / ".taskmaster" / "tasks"
    tasks_dir.mkdir(parents=True)
    _make_tasks_db(tasks_dir, list(tasks))

    runs_dir = root / "data" / "orchestrator"
    runs_dir.mkdir(parents=True)
    _make_runs_db(runs_dir, list(events))

    worktrees = root / ".worktrees"
    worktrees.mkdir(parents=True)
    for wt_name, plan in plans:
        path = worktrees / ".task-meta" / wt_name / "plan.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan))
    return root


def _wiped_project(tmp_path, name="proj", task_id=2464):
    """A project with exactly one repairable candidate: a done task whose
    metadata.files is empty but whose plan declared a file-level scope."""
    return _make_project(
        tmp_path,
        tasks=[{"id": task_id, "status": "done", "metadata": {"files": []}}],
        plans=[(str(task_id), {"task_id": task_id, "files": ["a.py", "b.py"]})],
        name=name,
    )


# ---------------------------------------------------------------------------
# repair_project END-TO-END on the LIVE RE-READ path — gate 3 must interrogate
# the candidate's OWN task, not the master-tag row that happens to share its id.
# ---------------------------------------------------------------------------


def test_repair_project_re_reads_the_candidates_own_tag_not_the_default(tmp_path):
    """The companion to the repair_one assertion above, on the OTHER MCP call.

    ``get_task`` takes the same optional ``tag``
    (fused-memory/src/fused_memory/server/tools.py:4182-4186) and defaults it
    the same silent way. Dropped there, gate 3 (``classify_live_task``) judges
    the WRONG task's live status — and idempotency dies with it, because
    SKIP_FILES_PRESENT would be evaluated against the master-tag row, so a
    re-run would re-write.

    The whole project is synthetic and the task row carries ``tag="feature-x"``
    (``_make_tasks_db`` already honours ``row.get("tag", "master")``), so the
    tag travels the real path: tasks.db column -> audit -> WipeCandidate -> wire.
    """
    root = _make_project(
        tmp_path,
        tasks=[
            {
                "id": 77,
                "tag": "feature-x",
                "status": "done",
                "metadata": {"files": []},
            }
        ],
        plans=[("77", {"task_id": 77, "files": ["a.py", "b.py"]})],
    )
    # One payload serves both calls: get_task reads it as a terminal task with
    # empty files (so gate 3 says REPAIR and the run reaches the write), and
    # update_task reads it as a plain non-error success.
    client = _FakeClient(returns={"id": "77", "status": "done", "metadata": {"files": []}})

    result = asyncio.run(repair_project(client, str(root), apply=True, now_iso=_NOW))

    assert [o.disposition for o in result.outcomes] == [REPAIR]
    assert [o.tag for o in result.outcomes] == ["feature-x"]
    calls = dict(client.calls)
    assert set(calls) == {"get_task", "update_task"}
    assert calls["get_task"]["tag"] == "feature-x"


def test_repair_project_declines_to_write_when_the_live_task_is_not_terminal(tmp_path):
    """GATE 3'S ENFORCEMENT, not just its classifier.

    ``classify_live_task`` has thorough unit coverage above, but a unit test of
    a pure classifier proves nothing about whether repair_project ACTS on its
    verdict: a refactor that dropped the ``continue`` after the disposition
    check would leave every one of those tests green while writing to a live
    task. This is the single most safety-critical branch in the script — the
    whole "repairing underneath a live workflow is guaranteed to be undone"
    argument rests on it — so the wiring is asserted end-to-end, on the absence
    of the update_task call.
    """
    root = _wiped_project(tmp_path)
    client = _FakeClient(
        returns={"id": "2464", "status": "in-progress", "metadata": {"files": []}}
    )

    result = asyncio.run(repair_project(client, str(root), apply=True, now_iso=_NOW))

    assert [o.disposition for o in result.outcomes] == [SKIP_NOT_TERMINAL]
    called = [name for name, _ in client.calls]
    assert called == ["get_task"], f"gate 3 did not stop the write: {client.calls}"


def test_repair_project_declines_to_write_when_the_live_task_already_has_files(tmp_path):
    """The idempotency arm of gate 3, enforced rather than merely classified.
    A second pass (the one expected once 3113 lands) must re-read cheaply and
    touch only what is still empty — so the write must not be issued here."""
    root = _wiped_project(tmp_path)
    client = _FakeClient(
        returns={"id": "2464", "status": "done", "metadata": {"files": ["a.py"]}}
    )

    result = asyncio.run(repair_project(client, str(root), apply=True, now_iso=_NOW))

    assert [o.disposition for o in result.outcomes] == [SKIP_FILES_PRESENT]
    called = [name for name, _ in client.calls]
    assert called == ["get_task"], f"an already-repaired task was rewritten: {client.calls}"


def test_repair_project_short_circuits_a_lock_charter_reject_before_any_mcp_call(tmp_path):
    """GATE 2 IS A PRE-CHECK, and its whole value is being one.

    The interceptor would reject a directory-carrying files list with an opaque
    lock_charter_error on task N of a batch. Catching it locally is only worth
    anything if it happens BEFORE the wire — so this asserts not merely that no
    write went out, but that the client was never called AT ALL.
    """
    root = _make_project(
        tmp_path,
        tasks=[{"id": 2464, "status": "done", "metadata": {"files": []}}],
        plans=[("2464", {"task_id": 2464, "files": ["orchestrator/src/orchestrator"]})],
    )
    client = _FakeClient(
        returns={"id": "2464", "status": "done", "metadata": {"files": []}}
    )

    result = asyncio.run(repair_project(client, str(root), apply=True, now_iso=_NOW))

    assert [o.disposition for o in result.outcomes] == [SKIP_LOCK_CHARTER]
    assert client.calls == [], f"the pre-check dialled the server: {client.calls}"
    assert "orchestrator/src/orchestrator" in (result.outcomes[0].detail or "")


def test_repair_project_reports_the_servers_own_reason_for_a_missing_task(tmp_path):
    """AN ERROR REPLY IS NOT AN EXCEPTION, and its text must not be discarded.

    fused-memory's tool layer never raises across the wire: ``@mcp_tool_errors``
    (fused-memory/src/fused_memory/server/tool_errors.py:44-59) turns every
    exception into an ordinary reply dict. A get_task for a task that no longer
    exists therefore arrives as a normal reply carrying
    ``error``/``error_type``, sails past the ``except`` arm, and — having no
    ``status`` key — would land in SKIP_MISSING with the generic detail 'live
    re-read immediately before the write'.

    ``RepairOutcome.detail``'s docstring promises the operator-actionable
    reason for EVERY non-REPAIR disposition, so the server's own explanation is
    read off the reply and carried through.
    """
    root = _wiped_project(tmp_path)
    client = _FakeClient(
        returns={
            "error": "No tasks found for ID(s): 2464",
            "error_type": "TaskNotFoundError",
        }
    )

    result = asyncio.run(repair_project(client, str(root), apply=True, now_iso=_NOW))

    assert [o.disposition for o in result.outcomes] == [SKIP_MISSING]
    detail = result.outcomes[0].detail or ""
    assert "TaskNotFoundError" in detail
    assert "No tasks found for ID(s): 2464" in detail
    assert "update_task" not in [name for name, _ in client.calls]


def test_repair_project_with_a_live_client_and_apply_false_issues_no_write(tmp_path):
    """THE DRY-RUN GUARANTEE, asserted where it actually lives.

    The subprocess dry-run test below proves today's CLI never dials. This is
    its companion one layer down: dry-run-ness must be a property of
    ``repair_project`` itself, not of the caller remembering to withhold the
    client. With a LIVE client and ``apply=False`` the candidate must still be
    reported as a would-be REPAIR (the selection is the dry run's whole
    output), the live re-read may happen, but ``update_task`` must never be
    called. Without the guard at the write site this records a real write and
    ``format_summary`` then prints ``NO WRITE WAS ATTEMPTED`` over the top of
    it.
    """
    root = _wiped_project(tmp_path)
    client = _FakeClient(returns={"id": "2464", "status": "done", "metadata": {"files": []}})

    result = asyncio.run(repair_project(client, str(root), apply=False, now_iso=_NOW))

    assert [o.disposition for o in result.outcomes] == [REPAIR]
    assert result.applied is False
    called = [name for name, _ in client.calls]
    assert "update_task" not in called, f"a dry run issued a write: {client.calls}"


_SCRIPT = str(Path(__file__).parent.parent.parent / "scripts" / "repair_wiped_metadata_files.py")

# A port nothing listens on. Pointing --server-url here PROVES the dry run
# never dials: if it did, the process would fail to connect rather than exit 0.
_CLOSED_PORT_URL = "http://127.0.0.1:9"


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, _SCRIPT, *args], capture_output=True, text=True
    )


def test_main_dry_run_is_the_default_and_never_dials_the_server(tmp_path):
    """(a) THE SAFETY PROPERTY. No --apply, an unreachable server URL, and a
    project that HAS a repairable candidate: the process must still exit 0
    having made zero MCP calls. An accidental bare invocation must be inert."""
    root = _wiped_project(tmp_path)

    result = _run_cli("--project-root", str(root), "--server-url", _CLOSED_PORT_URL)

    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    assert "2464" in result.stdout


def test_main_dry_run_summary_states_no_write_was_attempted(tmp_path):
    root = _wiped_project(tmp_path)

    result = _run_cli("--project-root", str(root))

    assert "NO WRITE WAS ATTEMPTED" in result.stdout


def test_main_json_emits_an_object_with_both_outcomes_and_coverage(tmp_path):
    """(c) parseable, and coverage travels with the results."""
    root = _wiped_project(tmp_path)

    result = _run_cli("--project-root", str(root), "--json")

    payload = json.loads(result.stdout)
    project = payload["projects"][0]
    assert project["project_root"] == str(root)
    assert project["applied"] is False
    assert project["coverage"]["total_tasks"] == 1
    assert [o["task_id"] for o in project["outcomes"]] == [2464]
    assert project["counts"][REPAIR] == 1
    # Every disposition is present in the machine-readable counts too.
    assert set(project["counts"]) == set(ALL_DISPOSITIONS)


def test_main_json_carries_the_observable_subset_caveat(tmp_path):
    """The honesty caveat is not a human-only decoration — a JSON consumer
    must receive it too, or the machine path silently loses the disclaimer."""
    root = _wiped_project(tmp_path)

    payload = json.loads(_run_cli("--project-root", str(root), "--json").stdout)

    caveat = payload["projects"][0]["observable_subset_caveat"].lower()
    assert "observable subset" in caveat
    assert "neither clean nor damaged" in caveat


def test_main_exit_0_on_a_clean_project_and_still_prints_coverage(tmp_path):
    """(d) exit 0 = ran, nothing failed — including when there is nothing to do.
    The coverage block prints anyway."""
    root = _make_project(tmp_path, tasks=[{"id": 1, "metadata": {"files": ["a.py"]}}])

    result = _run_cli("--project-root", str(root))

    assert result.returncode == 0
    assert "COVERAGE" in result.stdout
    assert "OBSERVABLE SUBSET" in result.stdout


def test_main_exit_2_when_no_project_root_resolves(tmp_path):
    """(d) 2 = no project root resolved to a readable tasks.db. Reusing the
    audit's discover_project_roots, which already drops roots with none."""
    result = _run_cli("--project-root", str(tmp_path / "no-such-project"))

    assert result.returncode == EXIT_NO_ROOT
    assert "no project root" in result.stderr.lower()


def test_main_apply_is_the_only_way_to_write(tmp_path):
    """(b) --apply on an unreachable server must FAIL rather than quietly
    succeed: the dry-run path's inertness comes from not dialling at all, so
    an apply run that also never dialled would be indistinguishable from a
    successful repair.

    A DOWN SERVER IS A NAMED OUTCOME, NOT A TRACEBACK. `returncode != 0` is not
    good enough for a script whose exit codes are an advertised contract: an
    unhandled httpx.ConnectError also exits non-zero, and it exits with 1 —
    which this CLI's own epilog defines as "at least one candidate FAILED to
    write", sending an operator hunting for a rejected write that was never
    attempted. So the code AND the message are pinned.
    """
    root = _wiped_project(tmp_path)

    result = _run_cli(
        "--project-root", str(root), "--apply", "--server-url", _CLOSED_PORT_URL
    )

    assert result.returncode == EXIT_SERVER_UNREACHABLE, result.stderr
    assert "could not reach the fused-memory MCP server" in result.stderr
    assert _CLOSED_PORT_URL in result.stderr
    assert "NOTHING was written" in result.stderr
    assert "Traceback" not in result.stderr


def _unreadable_project(tmp_path, name="corrupt"):
    """A project whose tasks.db EXISTS (so discovery keeps it) but is not a
    sqlite database, so audit_project raises sqlite3.DatabaseError on it."""
    root = _wiped_project(tmp_path, name=name, task_id=99)
    (root / ".taskmaster" / "tasks" / "tasks.db").write_bytes(b"not a database at all")
    return root


def test_main_one_unreadable_root_does_not_abort_the_other_roots(tmp_path):
    """A single sqlite3.Error must not eat the whole run.

    Without a per-root guard the exception escapes main_async, the summary is
    NEVER PRINTED, and on --apply the record of the writes already applied to
    the earlier roots is lost — precisely the reporting-honesty failure this
    module's docstring claims to prevent. The sibling audit CLI already handles
    this case (audit_wiped_metadata_files.py:899-933); this asserts the repair
    inherits the resilience along with the exit-code philosophy.
    """
    bad = _unreadable_project(tmp_path)
    good = _wiped_project(tmp_path, name="good", task_id=11)

    result = _run_cli(
        "--project-root", str(bad), "--project-root", str(good), "--json"
    )

    assert result.returncode == EXIT_OK, result.stderr
    payload = json.loads(result.stdout)
    assert [p["project_root"] for p in payload["projects"]] == [str(good)]
    assert [o["task_id"] for o in payload["projects"][0]["outcomes"]] == [11]
    # The skipped root is warned about, never silently dropped.
    assert "skipping unreadable project" in result.stderr
    assert str(bad) in result.stderr
    assert "incomplete" in result.stderr


def test_main_exit_3_when_every_resolved_root_is_unreadable(tmp_path):
    """NOTHING SCANNED IS NOT A CLEAN RUN — and it is not a failed write either.

    Exit 1 is the operator's signal that a write was attempted and rejected.
    Mapping an unreadable database onto it would make that signal ambiguous, so
    'roots resolved but every one failed' gets its own code, mirroring the
    audit's 3.
    """
    bad = _unreadable_project(tmp_path)

    result = _run_cli("--project-root", str(bad))

    assert result.returncode == EXIT_NOTHING_SCANNED, result.stdout
    assert "NOTHING was examined" in result.stderr
    assert "not a clean result" in result.stderr


_MIGRATE_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "migrate_metadata_modules_to_files.py"


def _protocol_versions(path: Path) -> set[str]:
    return set(
        re.findall(r"""protocolVersion["']\s*:\s*["']([^"']+)["']""", path.read_text())
    )


def test_repair_client_handshake_has_not_drifted_from_its_parent():
    """A DRIFT GUARD FOR A KNOWING CLONE.

    RepairFusedMemoryClient._initialize restates its parent's whole handshake —
    both JSON-RPC posts, the protocolVersion, the capabilities block — solely to
    change clientInfo.name, because the parent bakes that name into the middle
    of the procedure. scripts/migrate_metadata_modules_to_files.py is outside
    task 3329's lock scope, so the parent cannot be given a
    ``getattr(self, '_client_name', ...)`` seam here; the ``_client_name``
    attribute is set on the subclass ready for that change, and the override
    can be deleted the day it lands.

    Until then this is what stops the clone drifting silently: bump the
    parent's protocolVersion and the repair client would keep handshaking with
    the stale one, with nothing in the repo noticing. Asserted on the source
    text rather than by importing the client, so the check costs no httpx
    import and no server.
    """
    parent = _protocol_versions(_MIGRATE_SCRIPT)
    child = _protocol_versions(Path(_SCRIPT))

    assert len(parent) == 1, f"ambiguous parent protocolVersion: {parent}"
    assert len(child) == 1, f"ambiguous repair protocolVersion: {child}"
    assert parent == child, (
        "the repair client's handshake has drifted from FusedMemoryClient's: "
        f"parent={parent}, repair={child}"
    )
    # The one thing the override is FOR: an attributable clientInfo.name, so
    # these repair writes are not filed under the migration's agent_id.
    assert CLIENT_NAME not in _MIGRATE_SCRIPT.read_text()
    assert "'migrate-metadata'" in _MIGRATE_SCRIPT.read_text()


def test_main_project_root_is_repeatable(tmp_path):
    a = _wiped_project(tmp_path, name="a", task_id=11)
    b = _wiped_project(tmp_path, name="b", task_id=22)

    result = _run_cli(
        "--project-root", str(a), "--project-root", str(b), "--json"
    )

    payload = json.loads(result.stdout)
    assert [p["project_root"] for p in payload["projects"]] == [str(a), str(b)]


def test_main_does_not_repair_a_contradicted_candidate(tmp_path):
    """End-to-end proof of constraint 2 through the CLI: a task with both a
    null-sha row and a real merge sha lands in skip_contradicted, never in the
    repair list."""
    root = _make_project(
        tmp_path,
        tasks=[{"id": 3086, "status": "done", "metadata": {"files": []}}],
        plans=[("3086", {"task_id": 3086, "files": ["a.py"]})],
        events=[
            {
                "event_type": "merge_finalized",
                "task_id": 3086,
                "data": {"state": "blocked", "merge_sha": None},
            },
            {
                "event_type": "merge_finalized",
                "task_id": 3086,
                "data": {"state": "done", "merge_sha": "abc123"},
            },
        ],
    )

    payload = json.loads(_run_cli("--project-root", str(root), "--json").stdout)

    counts = payload["projects"][0]["counts"]
    assert counts[SKIP_CONTRADICTED] == 1
    assert counts[REPAIR] == 0


def test_main_help_documents_the_exit_codes():
    """The audit CLI's convention: exit codes live in the argparse epilog, not
    only in a docstring the operator will not see.

    EVERY advertised code, including 3 (nothing scanned) and 4 (server
    unreachable). Those two exist so that 1 keeps its single documented meaning
    — 'a write was attempted and failed' — so an epilog that omitted them would
    leave an operator mapping them back onto 1 by guesswork. Distinctness is
    asserted here too: two outcomes sharing a code is the same ambiguity.
    """
    result = _run_cli("--help")

    assert result.returncode == 0
    assert "exit codes" in result.stdout.lower()

    codes = (EXIT_OK, 1, EXIT_NO_ROOT, EXIT_NOTHING_SCANNED, EXIT_SERVER_UNREACHABLE)
    assert len(set(codes)) == len(codes), codes
    for code in codes:
        assert str(code) in result.stdout, code


def test_classify_live_task_treats_a_non_dict_metadata_as_no_files():
    """metadata itself can be NULL or malformed in the store; that is 'no
    scope recorded', which is repairable, not a crash."""
    for meta in (None, "null", 3, []):
        live = {"status": "done", "metadata": meta}
        assert classify_live_task(live, _candidate(26)) == REPAIR, meta
