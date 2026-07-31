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

from repair_wiped_metadata_files import (
    PROVENANCE_KEY,
    REPAIR,
    REPAIR_TASK_ID,
    SKIP_FILES_PRESENT,
    SKIP_MISSING,
    SKIP_NOT_TERMINAL,
    FAILED,
    build_repair_payload,
    classify_live_task,
    plan_files_rejection_reason,
    repair_one,
    select_repairable_candidates,
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


def test_classify_live_task_treats_a_non_dict_metadata_as_no_files():
    """metadata itself can be NULL or malformed in the store; that is 'no
    scope recorded', which is repairable, not a crash."""
    for meta in (None, "null", 3, []):
        live = {"status": "done", "metadata": meta}
        assert classify_live_task(live, _candidate(26)) == REPAIR, meta
