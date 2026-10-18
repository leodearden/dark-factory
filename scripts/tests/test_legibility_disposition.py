"""Tests for scripts/legibility/disposition.py: how a verified census finding
becomes work and how its task links back (plans/census-incremental-prd.md L4;
docs/quality-findings-contract.md §6, §7, §8)."""
from __future__ import annotations

import copy
import json
import re
from collections.abc import Sequence
from typing import Any

import pytest
from legibility import codebook, disposition, task_store, verdict
from legibility_task_store_fake import FakeTaskStore

_ROOT = "/abs/observed"
_RUN = "census-dark_factory-20261017"
_KEY_A = "fk-00000000000a"
_KEY_B = "fk-00000000000b"
_KEY_C = "fk-00000000000c"

Step = disposition.Step
Resolution = disposition.Resolution
Standing = disposition.Standing


def _filing(
    key: str | None = _KEY_A,
    title: str = "agents misread the merge queue",
    *,
    entry_id: str | None = "entry-cand-20261017-1",
    statement: str = "the queue state is not legible from the report",
    project_root: str = _ROOT,
) -> disposition.Filing:
    metadata: dict[str, Any] = {"source": disposition.CENSUS_SOURCE, "x_finding_run": _RUN}
    if key is not None:
        metadata["x_finding_key"] = key
    payload = {
        "project_root": project_root,
        "title": disposition.task_title(title),
        "description": statement,
        "task_kind": "normal",
        "priority": "medium",
        "metadata": metadata,
    }
    return disposition.Filing(entry_id=entry_id, finding_key=key, title=title, statement=statement, payload=payload)


class _Submit:
    """A fake curator-path ``submit_task``: answers ``{'ticket': 'tkt_<n>'}``
    unless told otherwise."""

    def __init__(self, reply: Any = None, *, raises: Exception | None = None) -> None:
        self._reply = reply
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **payload: Any) -> Any:
        self.calls.append(payload)
        if self._raises is not None:
            raise self._raises
        if self._reply is not None:
            return self._reply
        return {"ticket": f"tkt_{len(self.calls)}"}


def _hit(task_id: int, status: str) -> task_store.TaskHit:
    return task_store.TaskHit(task_id=task_id, title=f"task {task_id}", status=status)


def _task(task_id: int, status: str, **metadata: Any) -> task_store.TaskRecord:
    return task_store.TaskRecord(task_id=task_id, title=f"task {task_id}", status=status, metadata=metadata)


def _key_hits(*hits: task_store.TaskHit, key: str = _KEY_A) -> dict:
    return {(disposition.FINDING_KEY_METADATA, key): list(hits)}


def _accepting(task_id: int) -> task_store.TaskRecord:
    return _task(task_id, "cancelled", **{disposition.ACCEPTANCE_REASON_METADATA: "kept on purpose"})


def _only(disposal: disposition.Disposal) -> disposition.Decision:
    (decision,) = disposal.decisions
    return decision


def test_a_live_task_carrying_the_key_decides_at_step_one_and_files_nothing() -> None:
    store = FakeTaskStore(metadata_hits=_key_hits(_hit(501, "pending")))
    submit = _Submit()

    disposal = disposition.dispose([_filing()], store=store, submit=submit)

    decision = _only(disposal)
    assert submit.calls == []
    assert (decision.step, decision.resolution, decision.standing) == (Step.KEY, Resolution.EXISTING, Standing.FILED)
    assert decision.task_id == 501
    assert decision.disposition() == "filed:501"
    assert disposal.unfiled == ()
    assert disposal.unreachable is None
    assert store.arguments_of("find_tasks_by_metadata") == [
        {"project_root": _ROOT, "key": disposition.FINDING_KEY_METADATA, "value": _KEY_A},
    ]


def test_a_cancelled_task_carrying_an_acceptance_reason_is_an_acceptance() -> None:
    store = FakeTaskStore(metadata_hits=_key_hits(_hit(12, "cancelled")), tasks={12: _accepting(12)})
    submit = _Submit()

    decision = _only(disposition.dispose([_filing()], store=store, submit=submit))

    assert submit.calls == []
    assert (decision.step, decision.standing, decision.task_id) == (Step.KEY, Standing.ACCEPTED, 12)
    assert decision.disposition() == "accepted:12"
    assert "search_tasks" not in store.tools_called()


def test_a_cancelled_task_without_an_acceptance_reason_counts_as_absent() -> None:
    store = FakeTaskStore(metadata_hits=_key_hits(_hit(12, "cancelled")), tasks={12: _task(12, "cancelled")})
    submit = _Submit()

    decision = _only(disposition.dispose([_filing()], store=store, submit=submit))

    assert "search_tasks" in store.tools_called()
    assert len(submit.calls) == 1
    assert (decision.step, decision.ticket) == (Step.CURATOR, "tkt_1")


def test_a_done_task_carrying_the_key_is_linked_and_not_refiled() -> None:
    store = FakeTaskStore(metadata_hits=_key_hits(_hit(40, "done")))
    submit = _Submit()

    decision = _only(disposition.dispose([_filing()], store=store, submit=submit))

    assert submit.calls == []
    assert (decision.step, decision.standing, decision.task_id) == (Step.KEY, Standing.DONE, 40)
    assert decision.disposition() == "open (task 40 done; §8 step 1c re-verify not yet run)"
    link = decision.back_link(_RUN)
    assert link is not None
    assert (link.task_id, link.step, link.resolution) == (40, Step.KEY, Resolution.EXISTING)


@pytest.mark.parametrize(
    ("hits", "expected"),
    [
        ([_hit(10, "done"), _hit(12, "cancelled"), _hit(5, "pending"), _hit(7, "blocked")], (Standing.FILED, 7)),
        ([_hit(10, "done"), _hit(12, "cancelled"), _hit(3, "cancelled")], (Standing.ACCEPTED, 12)),
        ([_hit(3, "done"), _hit(9, "done"), _hit(4, "cancelled")], (Standing.DONE, 9)),
    ],
    ids=["live-wins", "accepted-over-done", "newest-done"],
)
def test_step_one_precedence_is_live_then_accepted_then_done_newest_first(hits, expected) -> None:
    store = FakeTaskStore(
        metadata_hits=_key_hits(*hits),
        tasks={12: _accepting(12), 3: _task(3, "cancelled"), 4: _task(4, "cancelled")},
    )

    decision = _only(disposition.dispose([_filing()], store=store, submit=_Submit()))

    assert (decision.standing, decision.task_id) == expected


def test_an_unreachable_step_one_lookup_files_nothing_at_all() -> None:
    unreachable = task_store.StoreUnreachable("find_tasks_by_metadata: ConnectionError: down")
    store = FakeTaskStore(metadata_hits={(disposition.FINDING_KEY_METADATA, _KEY_B): unreachable})
    submit = _Submit()
    filings = [_filing(_KEY_A, "first"), _filing(_KEY_B, "second")]

    disposal = disposition.dispose(filings, store=store, submit=submit)

    assert submit.calls == []
    assert disposal.decisions == ()
    assert disposal.unfiled == tuple(filings)
    assert disposal.unreachable is not None
    assert "ConnectionError: down" in disposal.unreachable


def test_an_unreachable_acceptance_read_back_is_an_unreachable_step_one() -> None:
    store = FakeTaskStore(metadata_hits=_key_hits(_hit(12, "cancelled")))
    submit = _Submit()

    disposal = disposition.dispose([_filing()], store=store, submit=submit)

    assert submit.calls == []
    assert disposal.unreachable is not None
    assert "get_task" in disposal.unreachable


def test_step_two_finds_a_live_task_by_meaning() -> None:
    filing = _filing()
    store = FakeTaskStore(
        search_hits={filing.title: [task_store.SearchHit(task_id=88, title="close", status="pending", score=0.8)]},
        tasks={88: _task(88, "pending")},
    )
    submit = _Submit()

    decision = _only(disposition.dispose([filing], store=store, submit=submit))

    assert submit.calls == []
    assert (decision.step, decision.resolution, decision.task_id) == (Step.SEMANTIC, Resolution.EXISTING, 88)
    (search,) = store.arguments_of("search_tasks")
    assert search["score_threshold"] == disposition.SEMANTIC_SCORE_THRESHOLD == 0.6
    assert filing.title in search["query"]
    assert store.arguments_of("get_task") == [{"project_root": _ROOT, "task_id": 88}]


def test_step_two_reads_back_the_top_hit_and_files_when_it_is_done() -> None:
    filing = _filing()
    store = FakeTaskStore(
        search_hits={filing.title: [
            task_store.SearchHit(task_id=88, title="close", status="pending", score=0.7),
            task_store.SearchHit(task_id=89, title="closer", status="pending", score=0.9),
        ]},
        tasks={88: _task(88, "pending"), 89: _task(89, "done")},
    )
    submit = _Submit()

    decision = _only(disposition.dispose([filing], store=store, submit=submit))

    assert store.arguments_of("get_task") == [{"project_root": _ROOT, "task_id": 89}]
    assert len(submit.calls) == 1
    assert decision.step is Step.CURATOR


def test_an_unreachable_step_two_goes_to_the_curator_and_says_so() -> None:
    store = FakeTaskStore(unreachable=["search_tasks"])
    submit = _Submit()

    decision = _only(disposition.dispose([_filing()], store=store, submit=submit))

    assert len(submit.calls) == 1
    assert decision.step is Step.CURATOR
    assert decision.note is not None
    assert "search_tasks" in decision.note


def test_step_three_files_through_the_curator_and_holds_the_pending_ticket() -> None:
    submit = _Submit({"ticket": "tkt_9"})
    filing = _filing()

    decision = _only(disposition.dispose([filing], store=FakeTaskStore(), submit=submit))

    assert submit.calls == [dict(filing.payload)]
    assert (decision.step, decision.ticket, decision.resolution, decision.task_id) == (
        Step.CURATOR, "tkt_9", None, None,
    )
    assert "tkt_9" in decision.disposition()
    link = decision.back_link(_RUN)
    assert link is not None
    assert (link.ticket, link.task_id, link.resolution, link.step) == ("tkt_9", None, None, Step.CURATOR)


@pytest.mark.parametrize(
    "submit",
    [_Submit(raises=RuntimeError("curator down")), _Submit({"error": "no", "error_type": "ValidationError"})],
    ids=["raises", "no-ticket"],
)
def test_a_submit_that_yields_no_ticket_is_a_decision_without_a_link(submit) -> None:
    decision = _only(disposition.dispose([_filing()], store=FakeTaskStore(), submit=submit))

    assert decision.step is Step.CURATOR
    assert decision.ticket is None
    assert decision.note
    assert decision.back_link(_RUN) is None


def test_dry_run_stops_before_the_curator() -> None:
    live = _filing(_KEY_A, "live")
    fresh = _filing(_KEY_B, "fresh")
    store = FakeTaskStore(metadata_hits=_key_hits(_hit(501, "pending"), key=_KEY_A))

    disposal = disposition.dispose([live, fresh], store=store, submit=None)

    assert [decision.filing for decision in disposal.decisions] == [live]
    assert disposal.unfiled == (fresh,)
    assert disposal.unreachable is None


def test_an_unkeyed_filing_skips_the_key_lookup() -> None:
    store = FakeTaskStore()

    decision = _only(disposition.dispose([_filing(None)], store=store, submit=_Submit()))

    assert "find_tasks_by_metadata" not in store.tools_called()
    assert decision.step is Step.CURATOR


def test_a_filing_whose_key_was_routed_earlier_this_run_is_linked_to_that_work_and_never_filed() -> None:
    store = FakeTaskStore()
    submit = _Submit()
    first = _filing(_KEY_A, "first title", entry_id="entry-a")
    second = _filing(_KEY_A, "second title", entry_id="entry-b")
    other = _filing(_KEY_B, "another finding", entry_id="entry-c")

    disposal = disposition.dispose([first, second, other], store=store, submit=submit)

    assert [payload["title"] for payload in submit.calls] == [first.payload["title"], other.payload["title"]]
    assert store.arguments_of("search_tasks") == [
        {"project_root": _ROOT, "query": filing.query, "score_threshold": disposition.SEMANTIC_SCORE_THRESHOLD}
        for filing in (first, other)
    ]
    by_title = {decision.filing.title: decision for decision in disposal.decisions}
    linked = by_title["second title"]
    assert (linked.step, linked.ticket, linked.routed) == (Step.KEY, by_title["first title"].ticket, True)
    assert linked.note is not None
    assert "first title" in linked.note
    assert disposal.unfiled == ()


def test_a_filing_whose_key_found_no_routed_work_earlier_this_run_is_decided_on_its_own() -> None:
    submit = _Submit(raises=RuntimeError("curator down"))
    filings = [_filing(_KEY_A, "first title"), _filing(_KEY_A, "second title")]

    disposal = disposition.dispose(filings, store=FakeTaskStore(), submit=submit)

    assert len(submit.calls) == 2
    assert [decision.step for decision in disposal.decisions] == [Step.CURATOR, Step.CURATOR]


def test_same_key_filings_on_one_entry_leave_one_row_and_one_resolve_call() -> None:
    filings = [_filing(_KEY_A, "first title", entry_id="entry-a"), _filing(_KEY_A, "second title", entry_id="entry-a")]
    disposal = disposition.dispose(filings, store=FakeTaskStore(), submit=_Submit())
    links = [link for decision in disposal.decisions if (link := decision.back_link(_RUN)) is not None]

    cb = disposition.with_back_links(_codebook(_entry("entry-a")), links)

    assert len(links) == 2
    assert [row["ticket"] for row in _rows(cb, "entry-a")] == ["tkt_1"]


# The submit_task seam's contract (fused_memory/server/tools.py::submit_task,
# fused_memory/middleware/task_interceptor.py::TaskInterceptor): success is
# {"ticket": "tkt_<id>"}, a TICKET id; there is no "id" key on this path.

def test_ticket_id_from_submit_result_reads_the_curator_ticket_key() -> None:
    assert disposition.ticket_id_from_submit_result({"ticket": "tkt_42"}) == "tkt_42"


def test_ticket_id_from_submit_result_rejects_the_error_shape() -> None:
    assert disposition.ticket_id_from_submit_result({"error": "boom", "error_type": "ValueError"}) is None


def test_ticket_id_from_submit_result_rejects_non_dict_results() -> None:
    assert disposition.ticket_id_from_submit_result(None) is None
    assert disposition.ticket_id_from_submit_result("tkt_1") is None
    assert disposition.ticket_id_from_submit_result(["tkt_1"]) is None


def test_ticket_id_from_submit_result_rejects_empty_dict() -> None:
    assert disposition.ticket_id_from_submit_result({}) is None


def test_ticket_id_from_submit_result_rejects_unusable_ticket_values() -> None:
    assert disposition.ticket_id_from_submit_result({"ticket": ""}) is None
    assert disposition.ticket_id_from_submit_result({"ticket": None}) is None
    assert disposition.ticket_id_from_submit_result({"ticket": {"id": 1}}) is None
    assert disposition.ticket_id_from_submit_result({"ticket": 42}) is None


# ---------------------------------------------------------------------------
# C1 back-links and in-run ticket resolution (D7)
# ---------------------------------------------------------------------------

_OLD_RUN = "census-dark_factory-20261001"


def _entry(entry_id: str, title: str = "a confusion", **fields: Any) -> dict:
    return {
        "id": entry_id,
        "title": title,
        "severity": "medium",
        "status": "open",
        "origin_phase": "unknown",
        "manifested_phase": "unknown",
        "sightings": [],
        **fields,
    }


def _codebook(*entries: dict, candidates: list[dict] | None = None) -> dict:
    return {"version": 2, "entries": list(entries), "candidates": list(candidates or [])}


def _link(entry_id: str = "entry-a", **overrides: Any) -> disposition.BackLink:
    fields: dict[str, Any] = {
        "entry_id": entry_id,
        "ticket": "tkt_1",
        "project_root": _ROOT,
        "run_id": _RUN,
        "finding_key": _KEY_A,
        "task_id": None,
        "resolution": None,
        "step": Step.CURATOR,
    }
    fields.update(overrides)
    return disposition.BackLink(**fields)


def _row(ticket: str | None, *, run_id: str = _RUN, **overrides: Any) -> dict:
    return _link(ticket=ticket, run_id=run_id, **overrides).to_row()


def _rows(cb: dict, entry_id: str) -> list[dict]:
    (entry,) = [entry for entry in cb["entries"] if entry["id"] == entry_id]
    return entry.get("filed_tickets", [])


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _outcome(status: task_store.TicketStatus, task_id: int | None = None, reason: str | None = None):
    return task_store.TicketOutcome(status, task_id, reason)


def test_with_back_links_appends_each_row_to_its_entry_without_mutating_the_input() -> None:
    cb = _codebook(_entry("entry-a"), _entry("entry-b"))
    links = [_link("entry-a"), _link("entry-b", ticket=None, task_id=501, resolution=Resolution.EXISTING, step=Step.KEY)]

    result = disposition.with_back_links(cb, links)

    assert _rows(result, "entry-a") == [links[0].to_row()]
    assert _rows(result, "entry-b") == [links[1].to_row()]
    assert "filed_tickets" not in cb["entries"][0]
    assert codebook.validate(result) == []
    assert disposition.with_back_links(result, links) == result


def test_with_back_links_refuses_a_link_to_an_absent_entry() -> None:
    with pytest.raises(ValueError, match="entry-missing"):
        disposition.with_back_links(_codebook(_entry("entry-a")), [_link("entry-missing")])


def test_a_back_link_is_held_to_the_c1_row_invariant() -> None:
    with pytest.raises(ValueError, match="entry-a"):
        _link("entry-a", finding_key="fk-ZZZ")


def test_resolve_tickets_resolves_every_unresolved_row_by_its_outcome() -> None:
    cb = _codebook(
        _entry("entry-a", filed_tickets=[_row("tkt_c")]),
        _entry("entry-b", filed_tickets=[_row("tkt_m")]),
        _entry("entry-c", filed_tickets=[_row("tkt_r")]),
        _entry("entry-d", filed_tickets=[_row("tkt_t")]),
        _entry("entry-e", filed_tickets=[_row("tkt_x")]),
        _entry("entry-f", filed_tickets=[_row("tkt_done", task_id=3, resolution=Resolution.CREATED)]),
    )
    store = FakeTaskStore(resolve_outcomes={
        "tkt_c": _outcome(task_store.TicketStatus.CREATED, 7),
        "tkt_m": _outcome(task_store.TicketStatus.COMBINED, 9),
        "tkt_r": _outcome(task_store.TicketStatus.REFUSED, reason="premise refused"),
        "tkt_t": _outcome(task_store.TicketStatus.FAILED, reason="timeout"),
        "tkt_x": _outcome(task_store.TicketStatus.FAILED, reason="expired"),
    })

    result, resolutions = disposition.resolve_tickets(cb, store=store, run_id=_RUN, clock=_Clock())

    assert [(row["resolution"], row["task_id"]) for entry in result["entries"] for row in entry["filed_tickets"]] == [
        ("created", 7), ("combined", 9), ("refused", None), (None, None), ("failed", None), ("created", 3),
    ]
    assert "tkt_done" not in [arguments["ticket"] for arguments in store.arguments_of("resolve_ticket")]
    assert resolutions.pending == ("tkt_t",)
    assert {resolved.ticket for resolved in resolutions.resolved} == {"tkt_c", "tkt_m", "tkt_r", "tkt_x"}
    assert resolutions.budget_exhausted is False
    assert cb["entries"][0]["filed_tickets"][0]["resolution"] is None

    again = FakeTaskStore(resolve_outcomes={"tkt_t": _outcome(task_store.TicketStatus.FAILED, reason="timeout")})
    disposition.resolve_tickets(result, store=again, run_id=_RUN, clock=_Clock())
    assert [arguments["ticket"] for arguments in again.arguments_of("resolve_ticket")] == ["tkt_t"]


def test_resolve_tickets_stops_when_its_budget_is_spent() -> None:
    assert disposition.RESOLVE_BUDGET_SECS == 300
    assert disposition.RESOLVE_CALL_CAP_SECS == 115
    clock = _Clock()

    def _spend() -> None:
        clock.now += 120

    tickets = ["tkt_1", "tkt_2", "tkt_3", "tkt_4"]
    cb = _codebook(*(_entry(f"entry-{n}", filed_tickets=[_row(ticket)]) for n, ticket in enumerate(tickets)))
    store = FakeTaskStore(on_resolve=_spend)

    result, resolutions = disposition.resolve_tickets(cb, store=store, run_id=_RUN, clock=clock)

    calls = store.arguments_of("resolve_ticket")
    assert [arguments["ticket"] for arguments in calls] == tickets[:3]
    assert [arguments["timeout_secs"] for arguments in calls] == [115, 115, 60]
    assert resolutions.budget_exhausted is True
    assert "tkt_4" in resolutions.pending
    assert _rows(result, "entry-3")[0]["resolution"] is None


def test_resolve_tickets_resolves_carried_over_rows_first() -> None:
    cb = _codebook(
        _entry("entry-a", filed_tickets=[_row("tkt_new")]),
        _entry("entry-b", filed_tickets=[_row("tkt_old", run_id=_OLD_RUN)]),
    )
    store = FakeTaskStore()

    _, resolutions = disposition.resolve_tickets(cb, store=store, run_id=_RUN, clock=_Clock())

    assert [arguments["ticket"] for arguments in store.arguments_of("resolve_ticket")] == ["tkt_old", "tkt_new"]
    assert [(resolved.ticket, resolved.carried) for resolved in resolutions.resolved] == [
        ("tkt_old", True), ("tkt_new", False),
    ]


def test_an_unreachable_ticket_stays_unresolved_and_the_next_one_is_still_resolved() -> None:
    cb = _codebook(
        _entry("entry-a", filed_tickets=[_row("tkt_1")]),
        _entry("entry-b", filed_tickets=[_row("tkt_2")]),
    )
    store = FakeTaskStore(resolve_outcomes={"tkt_1": task_store.StoreUnreachable("resolve_ticket: down")})

    result, resolutions = disposition.resolve_tickets(cb, store=store, run_id=_RUN, clock=_Clock())

    assert _rows(result, "entry-a")[0]["resolution"] is None
    assert _rows(result, "entry-b")[0]["resolution"] == "created"
    assert any("down" in error for error in resolutions.errors)


@pytest.mark.parametrize(
    ("outcome", "expected_rows", "expected_pending"),
    [
        (_outcome(task_store.TicketStatus.CREATED, 7), [("created", 7), ("created", 7)], ()),
        (_outcome(task_store.TicketStatus.FAILED, reason="timeout"), [(None, None), (None, None)], ("tkt_1",)),
    ],
    ids=["decided", "undecided"],
)
def test_a_ticket_two_rows_hold_is_asked_about_once(outcome, expected_rows, expected_pending) -> None:
    cb = _codebook(
        _entry("entry-a", filed_tickets=[_row("tkt_1")]),
        _entry("entry-b", filed_tickets=[_row("tkt_1", step=Step.KEY)]),
    )
    store = FakeTaskStore(resolve_outcomes={"tkt_1": outcome})

    result, resolutions = disposition.resolve_tickets(cb, store=store, run_id=_RUN, clock=_Clock())

    assert len(store.arguments_of("resolve_ticket")) == 1
    assert [
        (row["resolution"], row["task_id"]) for entry_id in ("entry-a", "entry-b") for row in _rows(result, entry_id)
    ] == expected_rows
    assert resolutions.pending == expected_pending


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (_outcome(task_store.TicketStatus.CREATED, 77), (Resolution.CREATED, 77, "filed:77")),
        (_outcome(task_store.TicketStatus.COMBINED, 78), (Resolution.COMBINED, 78, "filed:78")),
        (_outcome(task_store.TicketStatus.REFUSED, reason="premise refused"), (Resolution.REFUSED, None, None)),
    ],
    ids=["created", "combined", "refused"],
)
def test_a_disposal_carries_the_resolution_its_tickets_reached(outcome, expected) -> None:
    filing = _filing(entry_id="entry-a")
    disposal = disposition.dispose([filing], store=FakeTaskStore(), submit=_Submit({"ticket": "tkt_1"}))
    link = _only(disposal).back_link(_RUN)
    assert link is not None
    cb = disposition.with_back_links(_codebook(_entry("entry-a")), [link])
    store = FakeTaskStore(resolve_outcomes={"tkt_1": outcome})
    _, resolutions = disposition.resolve_tickets(cb, store=store, run_id=_RUN, clock=_Clock())

    decision = _only(disposal.resolved(resolutions))

    resolution, task_id, shown = expected
    assert (decision.resolution, decision.task_id) == (resolution, task_id)
    if shown is not None:
        assert decision.disposition() == shown
    else:
        assert decision.note is not None
        assert "premise refused" in decision.note


# ---------------------------------------------------------------------------
# The one-shot idempotent backfill of historical census tasks
# ---------------------------------------------------------------------------

_ESCAPED_TITLE = "agents quote </function_calls> in reports"
_IDENTITY = {
    "finding_key": _KEY_A,
    "finding_area": "orchestrator",
    "finding_anchor": "orchestrator/x.py::f",
    "finding_tags": ["h13", "kind:confusion"],
}


def _historical_codebook() -> dict:
    return _codebook(
        _entry("entry-plain", "plain title"),
        _entry("entry-escaped", _ESCAPED_TITLE),
        _entry("entry-retitled", "the corrected title"),
        _entry("entry-keyed", "keyed title", **_IDENTITY),
        _entry("entry-ticketed", "ticketed title", filed_tickets=[_row("tkt_open", run_id=_OLD_RUN)]),
        _entry(
            "entry-linked",
            "already linked",
            filed_tickets=[_row(None, task_id=5000, resolution=Resolution.CREATED, finding_key=None)],
        ),
        candidates=[{
            "id": "cand-20260801-1",
            "title": "the filed title",
            "first_seen": "2026-08-01",
            "disposition": "promoted",
            "sightings": [],
            "promoted_to": "entry-retitled",
        }],
    )


_UNMATCHED = task_store.TaskHit(106, "[legibility census] nothing carries this title", "done")


def _historical_store(**overrides: Any) -> FakeTaskStore:
    hits = [
        task_store.TaskHit(101, disposition.task_title("plain title"), "done"),
        task_store.TaskHit(102, disposition.task_title(_ESCAPED_TITLE), "pending"),
        task_store.TaskHit(103, disposition.task_title("the filed title"), "cancelled"),
        task_store.TaskHit(104, disposition.task_title("keyed title"), "done"),
        task_store.TaskHit(105, disposition.task_title("ticketed title"), "pending"),
        task_store.TaskHit(5000, disposition.task_title("already linked"), "done"),
        _UNMATCHED,
    ]
    return FakeTaskStore(metadata_hits={("source", disposition.CENSUS_SOURCE): hits}, **overrides)


def _backfilled_row(task_id: int, finding_key: str | None = None) -> dict:
    return {
        "ticket": None,
        "project_root": _ROOT,
        "run_id": _RUN,
        "finding_key": finding_key,
        "task_id": task_id,
        "resolution": "created",
        "step": "curator",
    }


def test_backfill_links_historical_census_tasks_to_entries_by_exact_title() -> None:
    assert disposition.task_title(_ESCAPED_TITLE) != f"[legibility census] {_ESCAPED_TITLE}"
    store = _historical_store()
    cb = _historical_codebook()

    result, backfill = disposition.backfill(cb, store=store, project_root=_ROOT, run_id=_RUN)

    assert store.arguments_of("find_tasks_by_metadata") == [
        {"project_root": _ROOT, "key": "source", "value": disposition.CENSUS_SOURCE},
    ]
    assert _rows(result, "entry-plain") == [_backfilled_row(101)]
    assert _rows(result, "entry-escaped") == [_backfilled_row(102)]
    assert _rows(result, "entry-retitled") == [_backfilled_row(103)]
    assert _rows(result, "entry-keyed") == [_backfilled_row(104, _KEY_A)]
    assert _rows(result, "entry-ticketed") == _rows(cb, "entry-ticketed")
    assert _rows(result, "entry-linked") == _rows(cb, "entry-linked")
    assert set(backfill.linked) == {
        (101, "entry-plain"), (102, "entry-escaped"), (103, "entry-retitled"), (104, "entry-keyed"),
    }
    assert backfill.unmatched == (_UNMATCHED,)
    assert backfill.error is None
    assert codebook.validate(result) == []


def test_backfill_runs_once_and_records_the_run_that_did() -> None:
    once, _ = disposition.backfill(_historical_codebook(), store=_historical_store(), project_root=_ROOT, run_id=_RUN)
    later = _historical_store()

    twice, backfill = disposition.backfill(once, store=later, project_root=_ROOT, run_id="census-dark_factory-20261018")

    assert once[codebook.HISTORICAL_BACKFILL] == {"run_id": _RUN, "unmatched_task_ids": [_UNMATCHED.task_id]}
    assert twice == once
    assert later.calls == []
    assert backfill == disposition.Backfill(done_in_run=_RUN)


def test_an_unreachable_backfill_changes_nothing_and_never_raises() -> None:
    cb = _historical_codebook()

    result, backfill = disposition.backfill(
        cb, store=_historical_store(unreachable=["find_tasks_by_metadata"]), project_root=_ROOT, run_id=_RUN,
    )

    assert result == cb
    assert codebook.HISTORICAL_BACKFILL not in result
    assert backfill.linked == ()
    assert backfill.error is not None
    assert "find_tasks_by_metadata" in backfill.error


# ---------------------------------------------------------------------------
# Rendering the Filed Tasks body and ## Structural from the JSON-safe record
# ---------------------------------------------------------------------------

_HARNESS_ROOT = "/abs/harness"


def _record(
    disposal: disposition.Disposal | None = None,
    resolutions: disposition.Resolutions | None = None,
    backfill: disposition.Backfill | None = None,
    structural: tuple[disposition.StructuralFinding, ...] = (),
) -> dict:
    report = disposition.FilingReport(
        disposal=disposal or disposition.Disposal((), ()),
        resolutions=resolutions or disposition.Resolutions(),
        backfill=backfill or disposition.Backfill(),
        structural=structural,
    )
    return json.loads(json.dumps(report.to_record()))


def _cells(row: str) -> list[str]:
    return [cell.replace("\\|", "|") for cell in re.split(r" \| ", row[2:-2])]


def _table(lines: list[str]) -> list[dict[str, str]]:
    header, _separator, *body = [line for line in lines if line.startswith("| ")]
    names = _cells(header)
    return [dict(zip(names, _cells(row), strict=True)) for row in body]


def _filed(record: dict) -> list[str]:
    return disposition.filed_tasks_lines(record, observed_root=_ROOT)


def _disposed_and_resolved(
    filings: list[disposition.Filing], store: FakeTaskStore, *, submit: _Submit | None = None,
) -> tuple[disposition.Disposal, disposition.Resolutions]:
    disposal = disposition.dispose(filings, store=store, submit=submit or _Submit())
    entries = [_entry(filing.entry_id) for filing in filings if filing.entry_id is not None]
    links = [link for decision in disposal.decisions if (link := decision.back_link(_RUN)) is not None]
    cb = disposition.with_back_links(_codebook(*entries), links)
    _, resolutions = disposition.resolve_tickets(cb, store=store, run_id=_RUN, clock=_Clock())
    return disposal, resolutions


def test_filed_tasks_names_each_finding_its_step_task_ticket_and_disposition() -> None:
    store = FakeTaskStore(
        metadata_hits={**_key_hits(_hit(42, "pending")), **_key_hits(_hit(5120, "cancelled"), key=_KEY_B)},
        tasks={5120: _accepting(5120)},
        resolve_outcomes={"tkt_1": _outcome(task_store.TicketStatus.CREATED, 77)},
    )
    filings = [
        _filing(_KEY_A, "live one", entry_id="entry-a"),
        _filing(_KEY_B, "accepted one", entry_id="entry-b"),
        _filing(_KEY_C, "fresh | one", entry_id="entry-c"),
    ]

    rows = {row["finding_key"]: row for row in _table(_filed(_record(*_disposed_and_resolved(filings, store))))}

    assert {key: (row["step"], row["resolution"], row["task_id"], row["ticket"], row["disposition"])
            for key, row in rows.items()} == {
        _KEY_A: ("key", "existing", "42", "", "filed:42"),
        _KEY_B: ("key", "existing", "5120", "", "accepted:5120"),
        _KEY_C: ("curator", "created", "77", "tkt_1", "filed:77"),
    }
    assert rows[_KEY_C]["title"] == "fresh | one"


def test_a_pending_ticket_is_named_and_said_to_resolve_next_run() -> None:
    store = FakeTaskStore(resolve_outcomes={"tkt_1": _outcome(task_store.TicketStatus.FAILED, reason="timeout")})

    (row,) = _table(_filed(_record(*_disposed_and_resolved([_filing()], store))))

    assert row["ticket"] == "tkt_1"
    assert row["resolution"] == ""
    assert "next run" in row["note"]


def test_a_refused_ticket_says_so_with_the_reason() -> None:
    store = FakeTaskStore(resolve_outcomes={"tkt_1": _outcome(task_store.TicketStatus.REFUSED, reason="premise")})

    (row,) = _table(_filed(_record(*_disposed_and_resolved([_filing()], store))))

    assert row["resolution"] == "refused"
    assert "premise" in row["note"]


def test_a_finding_filed_into_another_project_names_that_project() -> None:
    filings = [_filing(_KEY_A, "observed", entry_id="entry-a"), _filing(_KEY_B, "harness", entry_id="entry-b",
                                                                      project_root=_HARNESS_ROOT)]

    rows = {row["finding_key"]: row for row in _table(_filed(_record(*_disposed_and_resolved(filings, FakeTaskStore()))))}

    assert rows[_KEY_A]["project_root"] == ""
    assert rows[_KEY_B]["project_root"] == _HARNESS_ROOT


def test_an_unreachable_step_one_says_nothing_was_filed_and_lists_every_finding() -> None:
    filings = (_filing(_KEY_A, "first"), _filing(_KEY_B, "second"))
    disposal = disposition.Disposal((), filings, "find_tasks_by_metadata: ConnectionError: down")

    lines = _filed(_record(disposal))

    (statement_at,) = [index for index, line in enumerate(lines) if "step 1" in line]
    assert "NOTHING was filed" in lines[statement_at]
    assert "ConnectionError: down" in lines[statement_at]
    listed = "\n".join(lines[statement_at + 1:])
    for filing in filings:
        assert filing.title in listed
        assert str(filing.finding_key) in listed


def test_carried_resolutions_budget_and_backfill_each_render_only_when_present() -> None:
    carried = disposition.ResolvedTicket(
        "entry-b", "tkt_old", _ROOT, _outcome(task_store.TicketStatus.CREATED, 88), carried=True,
    )
    resolutions = disposition.Resolutions(resolved=(carried,), pending=("tkt_late",), budget_exhausted=True)
    backfill = disposition.Backfill(linked=((101, "entry-plain"),), unmatched=(_UNMATCHED,))

    text = "\n".join(_filed(_record(resolutions=resolutions, backfill=backfill)))

    assert "tkt_old" in text and "88" in text
    assert "budget" in text and "tkt_late" in text
    assert "101" in text and "entry-plain" in text
    assert str(_UNMATCHED.task_id) in text


def test_an_empty_record_renders_none_filed() -> None:
    assert _filed(_record()) == ["_none filed._"]


def test_structural_always_renders_its_heading_with_a_placeholder_when_empty() -> None:
    lines = disposition.structural_lines(_record())

    assert lines[:3] == ["", "## Structural", ""]
    assert len(lines) == 4


def test_structural_lists_each_finding_recorded_and_not_filed() -> None:
    cb = _codebook(_entry("entry-keyed", "a structural shape", **_IDENTITY), _entry("entry-other"))
    structural = disposition.structural_findings(cb, ["entry-keyed"])

    lines = disposition.structural_lines(_record(structural=structural))

    assert lines[:3] == ["", "## Structural", ""]
    assert any("NOT filed" in line and "§6" in line for line in lines)
    assert _table(lines) == [{
        "finding_key": _KEY_A,
        "entry_id": "entry-keyed",
        "title": "a structural shape",
        "anchor": _IDENTITY["finding_anchor"],
    }]


def _verdict(route: verdict.Route) -> verdict.Verdict:
    return verdict.Verdict(
        verified=True,
        reason="seen",
        anchor="orchestrator/x.py::f",
        tags=("h13", verdict.KIND_TAG),
        severity="medium",
        severity_reason="r",
        route=route,
        remediation=None,
    )


def test_only_a_mechanical_finding_files() -> None:
    assert disposition.files(_verdict(verdict.Route.MECHANICAL)) is True
    assert disposition.files(_verdict(verdict.Route.STRUCTURAL)) is False


# ---------------------------------------------------------------------------
# Outcome feedback: each back-linked task's standing, read from one
# get_statuses snapshot (plans/census-incremental-prd.md L5; contract §7, §8 1a-1c)
# ---------------------------------------------------------------------------


def _task_row(task_id: int, **overrides: Any) -> dict:
    return _row(None, task_id=task_id, resolution=Resolution.CREATED, **overrides)


def _read(cb: dict, statuses: dict | None, store: FakeTaskStore | None = None, *, error: str | None = None):
    return disposition.read_outcomes(
        cb, statuses=statuses, statuses_error=error, observed_root=_ROOT, store=store or FakeTaskStore(),
    )


def _entry_outcome(outcomes: disposition.Outcomes, entry_id: str) -> disposition.EntryOutcome:
    (found,) = [entry for entry in outcomes.entries if entry.entry_id == entry_id]
    return found


def _only_task(outcomes: disposition.Outcomes, entry_id: str = "entry-a") -> disposition.TaskOutcome:
    (task,) = _entry_outcome(outcomes, entry_id).tasks
    return task


@pytest.mark.parametrize(
    ("status", "standing"),
    [
        ("pending", Standing.FILED),
        ("in-progress", Standing.FILED),
        ("blocked", Standing.FILED),
        ("deferred", Standing.FILED),
        ("merge-deferred", Standing.FILED),
        ("done", Standing.DONE),
    ],
)
def test_an_observed_task_takes_its_standing_from_the_snapshot_alone(status, standing) -> None:
    store = FakeTaskStore()
    cb = _codebook(_entry("entry-a", filed_tickets=[_task_row(5120)]))

    task = _only_task(_read(cb, {"5120": status}, store))

    assert (task.ref, task.status, task.standing) == (disposition.TaskRef(_ROOT, 5120), status, standing)
    assert store.calls == []


@pytest.mark.parametrize(
    ("record", "standing"),
    [(_accepting(5120), "accepted"), (_task(5120, "cancelled"), "abandoned")],
    ids=["accepting", "abandoned"],
)
def test_a_cancelled_task_is_read_back_for_its_acceptance_reason(record, standing) -> None:
    store = FakeTaskStore(tasks={5120: record})
    cb = _codebook(_entry("entry-a", filed_tickets=[_task_row(5120)]))

    task = _only_task(_read(cb, {"5120": "cancelled"}, store))

    assert (task.status, task.standing) == ("cancelled", Standing(standing))
    assert store.arguments_of("get_task") == [{"project_root": _ROOT, "task_id": 5120}]


def test_another_projects_task_is_read_with_get_task_and_never_from_the_snapshot() -> None:
    store = FakeTaskStore(tasks={7: _task(7, "done")})
    cb = _codebook(_entry("entry-a", filed_tickets=[_task_row(7, project_root=_HARNESS_ROOT)]))

    task = _only_task(_read(cb, {"7": "pending"}, store))

    assert (task.ref, task.status, task.standing) == (disposition.TaskRef(_HARNESS_ROOT, 7), "done", Standing.DONE)
    assert store.arguments_of("get_task") == [{"project_root": _HARNESS_ROOT, "task_id": 7}]


def test_another_projects_cancelled_task_reads_its_acceptance_from_the_same_record() -> None:
    store = FakeTaskStore(tasks={7: _accepting(7)})
    cb = _codebook(_entry("entry-a", filed_tickets=[_task_row(7, project_root=_HARNESS_ROOT)]))

    task = _only_task(_read(cb, {}, store))

    assert task.standing is Standing.ACCEPTED
    assert len(store.arguments_of("get_task")) == 1


@pytest.mark.parametrize(
    ("statuses", "store", "error", "named"),
    [
        ({}, FakeTaskStore(), None, "5120"),
        (None, FakeTaskStore(), "get_statuses returned a tool-error envelope (ValidationError): 'boom'", "boom"),
        ({"5120": "cancelled"}, FakeTaskStore(unreachable=["get_task"]), None, "get_task"),
        ({"5120": "review"}, FakeTaskStore(), None, "review"),
    ],
    ids=["absent-from-snapshot", "no-snapshot", "unreachable-read-back", "unrecognised-status"],
)
def test_an_unobserved_task_has_no_standing_and_a_note_naming_why(statuses, store, error, named, caplog) -> None:
    cb = _codebook(_entry("entry-a", filed_tickets=[_task_row(5120)]))

    with caplog.at_level("WARNING", logger="legibility.disposition"):
        outcomes = _read(cb, statuses, store, error=error)

    task = _only_task(outcomes)
    assert task.standing is None
    assert task.note is not None
    assert named in task.note
    assert _entry_outcome(outcomes, "entry-a").governing is None
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert any("5120" in message for message in warnings)
    assert not any("done-count" in message for message in warnings)


def test_an_unobserved_read_never_raises_out_of_read_outcomes() -> None:
    cb = _codebook(_entry("entry-a", filed_tickets=[_task_row(5120), _task_row(7, project_root=_HARNESS_ROOT)]))

    outcomes = _read(cb, {"5120": "cancelled"}, FakeTaskStore(unreachable=["get_task"]))

    assert [task.standing for task in _entry_outcome(outcomes, "entry-a").tasks] == [None, None]


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ({"10": "done", "12": "cancelled", "5": "pending", "7": "blocked", "3": "cancelled"}, 7),
        ({"10": "done", "12": "cancelled", "3": "cancelled"}, 12),
        ({"10": "done", "9": "done", "3": "cancelled"}, 10),
        ({"3": "cancelled", "4": "cancelled"}, 4),
    ],
    ids=["live-wins", "accepted-over-done", "newest-done", "newest-abandoned"],
)
def test_the_governing_task_follows_the_step_one_precedence(statuses, expected) -> None:
    store = FakeTaskStore(tasks={12: _accepting(12), 3: _task(3, "cancelled"), 4: _task(4, "cancelled")})
    cb = _codebook(_entry("entry-a", filed_tickets=[_task_row(int(task_id)) for task_id in statuses]))

    governing = _entry_outcome(_read(cb, statuses, store), "entry-a").governing

    assert governing is not None
    assert governing.ref == disposition.TaskRef(_ROOT, expected)


def test_any_unobserved_task_leaves_the_entry_without_a_governing_task() -> None:
    cb = _codebook(_entry("entry-a", filed_tickets=[_task_row(5120), _task_row(5518)]))

    outcome = _entry_outcome(_read(cb, {"5120": "done"}), "entry-a")

    assert [task.standing for task in outcome.tasks] == [Standing.DONE, None]
    assert outcome.governing is None


def test_a_task_two_rows_of_one_entry_hold_is_read_once() -> None:
    store = FakeTaskStore(tasks={5120: _task(5120, "cancelled")})
    cb = _codebook(_entry("entry-a", filed_tickets=[
        _task_row(5120), _row("tkt_1", task_id=5120, resolution=Resolution.CREATED),
    ]))

    outcome = _entry_outcome(_read(cb, {"5120": "cancelled"}, store), "entry-a")

    assert [task.ref for task in outcome.tasks] == [disposition.TaskRef(_ROOT, 5120)]
    assert len(store.arguments_of("get_task")) == 1


def test_a_ticket_without_a_task_is_listed_and_triggers_no_read() -> None:
    store = FakeTaskStore()
    cb = _codebook(_entry("entry-a", filed_tickets=[_row("tkt_9")]))

    outcomes = _read(cb, {}, store)

    task = _only_task(outcomes)
    assert (task.ref, task.ticket, task.standing) == (None, "tkt_9", None)
    assert _entry_outcome(outcomes, "entry-a").governing is None
    assert outcomes.due == ()
    assert store.calls == []


def test_a_ticket_only_row_beside_an_observed_task_does_not_block_its_governance() -> None:
    cb = _codebook(_entry("entry-a", filed_tickets=[_row("tkt_9"), _task_row(5120)]))

    governing = _entry_outcome(_read(cb, {"5120": "done"}), "entry-a").governing

    assert governing is not None
    assert governing.ref == disposition.TaskRef(_ROOT, 5120)


def test_the_reverify_queue_holds_each_unfixed_entry_governed_by_a_done_task_in_codebook_order() -> None:
    cb = _codebook(
        _entry("entry-z", filed_tickets=[_task_row(30)]),
        _entry("entry-fixed", status="fixed", fixed_at_sha="abcdef1", filed_tickets=[_task_row(31)]),
        _entry("entry-retired", status="retired", filed_tickets=[_task_row(32)]),
        _entry("entry-live", filed_tickets=[_task_row(33), _task_row(34)]),
        _entry("entry-partial", status="partially", filed_tickets=[_task_row(35)]),
        _entry("entry-plain"),
    )
    statuses = {"30": "done", "31": "done", "32": "done", "33": "done", "34": "pending", "35": "done"}

    outcomes = _read(cb, statuses)

    assert outcomes.due == (
        disposition.Reverify("entry-z", disposition.TaskRef(_ROOT, 30)),
        disposition.Reverify("entry-partial", disposition.TaskRef(_ROOT, 35)),
    )
    assert [entry.entry_id for entry in outcomes.entries] == [
        "entry-z", "entry-fixed", "entry-retired", "entry-live", "entry-partial",
    ]
    assert _entry_outcome(outcomes, "entry-live").entry_status == "open"


def test_a_codebook_without_back_links_has_no_outcomes_at_all() -> None:
    outcomes = _read(_codebook(_entry("entry-a")), {"5120": "done"})

    assert outcomes.entries == ()
    assert outcomes.any_back_links is False
    assert outcomes.due == ()


def test_back_links_whose_read_changes_nothing_still_have_outcomes() -> None:
    outcomes = _read(_codebook(_entry("entry-a", filed_tickets=[_task_row(5120)])), {"5120": "pending"})

    assert outcomes.any_back_links is True
    assert outcomes.due == ()


def test_the_entry_outcome_carries_the_entry_key_and_the_snapshot_error() -> None:
    cb = _codebook(_entry("entry-a", filed_tickets=[_task_row(5120)], **_IDENTITY))

    outcomes = _read(cb, None, error="no snapshot")

    assert _entry_outcome(outcomes, "entry-a").finding_key == _KEY_A
    assert outcomes.statuses_error == "no snapshot"


def test_an_abandoned_task_carrying_the_key_still_counts_as_absent_for_step_one() -> None:
    store = FakeTaskStore(
        metadata_hits=_key_hits(_hit(12, "cancelled"), _hit(9, "done")), tasks={12: _task(12, "cancelled")},
    )

    decision = _only(disposition.dispose([_filing()], store=store, submit=_Submit()))

    assert (decision.standing, decision.task_id) == (Standing.DONE, 9)


# ---------------------------------------------------------------------------
# Typing one re-verify result (D8, INV-3): only a refuting Verdict is "gone"
# ---------------------------------------------------------------------------


def _parsed(*, verified: bool, route: str = "mechanical") -> verdict.Verdict:
    return verdict.parse_verdict(
        {
            "verified": verified,
            "reason": "checked at as_of_sha",
            "anchor": "orchestrator/x.py::f",
            "tags": ["h13", verdict.KIND_TAG],
            "severity": "medium",
            "severity_reason": "r",
            "route": route,
        },
        title="a confusion",
        project_root=_ROOT,
    )


def _due(entry_id: str = "entry-a", task_id: int = 5120) -> disposition.Reverify:
    return disposition.Reverify(entry_id, disposition.TaskRef(_ROOT, task_id))


def _reverify_cluster(entry_id: str = "entry-a", found: verdict.Verdict | None = None) -> dict:
    cluster: dict[str, Any] = {"title": "a confusion", disposition.REVERIFY_ENTRY: entry_id}
    if found is not None:
        cluster["verdict"] = found
    return cluster


def _verify_result(*, verified: Sequence[dict] = (), rejected: Sequence[dict] = ()) -> dict:
    return {"verified": list(verified), "rejected": list(rejected), "fixed": [], "headroom_probes": 0}


def _typed(result: dict, *offered: disposition.Reverify, deferred=(), error=None) -> dict:
    return dict(disposition.reverifications(result, offered=offered, deferred=deferred, error=error))


def test_a_verified_reverify_is_present_with_its_verdict() -> None:
    found = _parsed(verified=True)

    (typed,) = _typed(_verify_result(verified=[_reverify_cluster(found=found)]), _due()).values()

    assert (typed.entry_id, typed.task, typed.result, typed.found) == (
        "entry-a", disposition.TaskRef(_ROOT, 5120), disposition.ReverifyResult.PRESENT, found,
    )


def test_a_refuting_verdict_is_gone() -> None:
    found = _parsed(verified=False)

    typed = _typed(_verify_result(rejected=[_reverify_cluster(found=found)]), _due())["entry-a"]

    assert (typed.result, typed.found) == (disposition.ReverifyResult.GONE, found)


def test_a_rejection_without_a_verdict_is_inconclusive_never_gone() -> None:
    typed = _typed(_verify_result(rejected=[_reverify_cluster()]), _due())["entry-a"]

    assert typed.result is disposition.ReverifyResult.INCONCLUSIVE
    assert typed.found is None
    assert typed.reason is not None
    assert "fail" in typed.reason


def test_an_offered_entry_the_result_never_mentions_is_inconclusive() -> None:
    typed = _typed(_verify_result(verified=[_reverify_cluster("entry-other", _parsed(verified=True))]), _due())

    assert typed["entry-a"].result is disposition.ReverifyResult.INCONCLUSIVE
    assert "entry-other" not in typed


def test_a_reverify_the_cap_left_unoffered_is_deferred_to_the_next_run() -> None:
    typed = _typed(_verify_result(), deferred=[_due("entry-b", 7)])

    assert typed["entry-b"].result is disposition.ReverifyResult.DEFERRED
    assert typed["entry-b"].task == disposition.TaskRef(_ROOT, 7)
    assert typed["entry-b"].reason is not None
    assert "cap" in typed["entry-b"].reason


def test_an_errored_reverify_call_leaves_every_offered_entry_inconclusive_with_the_error() -> None:
    error = "headroom exhausted: verify reports no capacity"
    result = _verify_result(rejected=[_reverify_cluster(found=_parsed(verified=False))])

    typed = _typed(result, _due("entry-a"), _due("entry-b", 7), deferred=[_due("entry-c", 8)], error=error)

    assert [(each.result, error in (each.reason or "")) for each in (typed["entry-a"], typed["entry-b"])] == [
        (disposition.ReverifyResult.INCONCLUSIVE, True), (disposition.ReverifyResult.INCONCLUSIVE, True),
    ]
    assert typed["entry-c"].result is disposition.ReverifyResult.DEFERRED


@pytest.mark.parametrize(
    "result",
    [{}, {"verified": None, "rejected": "x"}, {"verified": [{"title": "no marker"}, 3], "rejected": [None]}, None],
    ids=["empty", "wrong-types", "no-marker", "not-a-mapping"],
)
def test_a_malformed_result_never_raises_and_types_nothing_as_gone(result) -> None:
    typed = _typed(result, _due())

    assert typed["entry-a"].result is disposition.ReverifyResult.INCONCLUSIVE


def test_a_verified_cluster_whose_verdict_refutes_is_not_gone() -> None:
    typed = _typed(_verify_result(verified=[_reverify_cluster(found=_parsed(verified=False))]), _due())

    assert typed["entry-a"].result is disposition.ReverifyResult.INCONCLUSIVE


@pytest.mark.parametrize(
    ("result", "found"),
    [("gone", None), ("gone", "verified"), ("present", None), ("present", "refuted")],
)
def test_a_reverification_holds_gone_and_present_to_their_verdicts(result, found) -> None:
    verdicts = {None: None, "verified": _parsed(verified=True), "refuted": _parsed(verified=False)}

    with pytest.raises(ValueError, match="entry-a") as excinfo:
        disposition.Reverification(
            "entry-a", disposition.TaskRef(_ROOT, 5120), disposition.ReverifyResult(result), verdicts[found], None,
        )

    assert "INV-3" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The projection (INV-9: status and accepted_by re-derived from the task store
# every run; INV-3: fixed only from a re-verify that found the finding gone)
# ---------------------------------------------------------------------------

_SHA = "1234567abcdef"
ReverifyResult = disposition.ReverifyResult


def _reverified(result: str, entry_id: str = "entry-a", task_id: int = 5120) -> dict:
    found = {"gone": _parsed(verified=False), "present": _parsed(verified=True)}.get(result)
    reason = None if found is not None else "not established"
    return {entry_id: disposition.Reverification(
        entry_id, disposition.TaskRef(_ROOT, task_id), ReverifyResult(result), found, reason,
    )}


def _projected(entry: dict, statuses: dict | None, *, store: FakeTaskStore | None = None, reverified=None):
    cb = _codebook(entry)
    before = copy.deepcopy(cb)
    outcomes = _read(cb, statuses, store)

    projected_cb, changes = disposition.project(cb, outcomes, reverified or {}, as_of_sha=_SHA)

    assert cb == before
    assert codebook.validate(projected_cb) == []
    (projected,) = projected_cb["entries"]
    return projected, tuple(changes)


def _state(entry: dict) -> tuple:
    return entry["status"], entry.get("accepted_by"), entry.get("fixed_at_sha")


def _change(from_status: str, to_status: str, accepted_by: int | None = None, fixed_at_sha: str | None = None):
    return disposition.StatusChange("entry-a", from_status, to_status, accepted_by, fixed_at_sha)


def test_an_accepting_cancel_makes_the_entry_accepted_by_that_task() -> None:
    entry = _entry("entry-a", filed_tickets=[_task_row(5120)])

    projected, changes = _projected(entry, {"5120": "cancelled"}, store=FakeTaskStore(tasks={5120: _accepting(5120)}))

    assert _state(projected) == ("accepted", 5120, None)
    assert changes == (_change("open", "accepted", accepted_by=5120),)


def test_a_stale_acceptance_whose_task_was_abandoned_returns_to_open() -> None:
    entry = _entry("entry-a", status="accepted", accepted_by=77, filed_tickets=[_task_row(77)])

    projected, changes = _projected(entry, {"77": "cancelled"}, store=FakeTaskStore(tasks={77: _task(77, "cancelled")}))

    assert _state(projected) == ("open", None, None)
    assert "accepted_by" not in projected
    assert changes == (_change("accepted", "open"),)


def test_an_acceptance_moves_to_the_task_that_now_governs() -> None:
    entry = _entry("entry-a", status="accepted", accepted_by=77, filed_tickets=[_task_row(77), _task_row(80)])
    store = FakeTaskStore(tasks={77: _accepting(77), 80: _accepting(80)})

    projected, changes = _projected(entry, {"77": "cancelled", "80": "cancelled"}, store=store)

    assert _state(projected) == ("accepted", 80, None)
    assert changes == (_change("accepted", "accepted", accepted_by=80),)


@pytest.mark.parametrize(
    "fields",
    [{"status": "accepted", "accepted_by": 77}, {"status": "fixed", "fixed_at_sha": "abcdef1"}],
    ids=["accepted", "fixed"],
)
def test_live_work_reopens_an_accepted_or_fixed_entry(fields) -> None:
    entry = _entry("entry-a", filed_tickets=[_task_row(77)], **fields)

    projected, changes = _projected(entry, {"77": "pending"})

    assert _state(projected) == ("open", None, None)
    assert changes == (_change(fields["status"], "open"),)


@pytest.mark.parametrize("status", ["open", "partially"])
def test_live_work_leaves_an_open_or_partial_entry_as_it_is(status) -> None:
    entry = _entry("entry-a", status=status, filed_tickets=[_task_row(77)])

    projected, changes = _projected(entry, {"77": "in-progress"})

    assert _state(projected) == (status, None, None)
    assert changes == ()


@pytest.mark.parametrize(
    "fields", [{}, {"status": "accepted", "accepted_by": 5120}], ids=["from-open", "from-accepted"],
)
def test_a_done_task_whose_reverify_found_the_finding_gone_fixes_the_entry_at_as_of_sha(fields) -> None:
    entry = _entry("entry-a", filed_tickets=[_task_row(5120)], **fields)

    projected, changes = _projected(entry, {"5120": "done"}, reverified=_reverified("gone"))

    assert _state(projected) == ("fixed", None, _SHA)
    assert changes == (_change(fields.get("status", "open"), "fixed", fixed_at_sha=_SHA),)


@pytest.mark.parametrize("result", ["present", "inconclusive", "deferred"])
def test_a_done_task_never_fixes_without_a_gone_reverify(result) -> None:
    accepted = _entry("entry-a", status="accepted", accepted_by=5120, filed_tickets=[_task_row(5120)])

    projected, changes = _projected(accepted, {"5120": "done"}, reverified=_reverified(result))

    assert _state(projected) == ("open", None, None)
    assert changes == (_change("accepted", "open"),)


@pytest.mark.parametrize("status", ["open", "partially", "mined-unverified"])
@pytest.mark.parametrize("result", ["present", "inconclusive", "deferred", None])
def test_a_done_task_without_a_gone_reverify_keeps_any_other_status(status, result) -> None:
    entry = _entry("entry-a", status=status, filed_tickets=[_task_row(5120)])
    reverified = None if result is None else _reverified(result)

    projected, changes = _projected(entry, {"5120": "done"}, reverified=reverified)

    assert _state(projected) == (status, None, None)
    assert changes == ()


def test_a_gone_reverify_of_a_task_that_no_longer_governs_fixes_nothing() -> None:
    entry = _entry("entry-a", filed_tickets=[_task_row(5120), _task_row(5200)])

    projected, changes = _projected(entry, {"5120": "done", "5200": "pending"}, reverified=_reverified("gone"))

    assert _state(projected) == ("open", None, None)
    assert changes == ()


@pytest.mark.parametrize(
    ("statuses", "tasks"),
    [({"77": "done"}, {}), ({"77": "cancelled"}, {77: _task(77, "cancelled")})],
    ids=["done", "abandoned"],
)
def test_a_fixed_entry_not_reopened_by_live_work_stays_fixed(statuses, tasks) -> None:
    entry = _entry("entry-a", status="fixed", fixed_at_sha="abcdef1", filed_tickets=[_task_row(77)])

    projected, changes = _projected(entry, statuses, store=FakeTaskStore(tasks=tasks))

    assert _state(projected) == ("fixed", None, "abcdef1")
    assert changes == ()


@pytest.mark.parametrize(
    ("fields", "statuses"),
    [
        ({"status": "retired"}, {"5120": "cancelled"}),
        ({"status": "accepted", "accepted_by": 5120}, {}),
        ({"status": "accepted", "accepted_by": 5120}, None),
    ],
    ids=["retired", "unobserved-task", "no-snapshot"],
)
def test_a_retired_or_ungoverned_entry_is_unchanged_whatever_the_outcomes_say(fields, statuses) -> None:
    entry = _entry("entry-a", filed_tickets=[_task_row(5120)], **fields)
    store = FakeTaskStore(tasks={5120: _accepting(5120)})

    projected, changes = _projected(entry, statuses, store=store, reverified=_reverified("gone"))

    assert projected == entry
    assert changes == ()


def test_an_entry_without_back_links_is_unchanged() -> None:
    entry = _entry("entry-a", status="accepted", accepted_by=12)

    projected, changes = _projected(entry, {"12": "pending"})

    assert projected == entry
    assert changes == ()


def test_refilings_re_file_a_present_mechanical_finding_and_a_recurring_abandoned_one() -> None:
    cb = _codebook(
        _entry("entry-present", filed_tickets=[_task_row(5120)]),
        _entry("entry-structural", filed_tickets=[_task_row(5121)]),
        _entry("entry-gone", filed_tickets=[_task_row(5122)]),
        _entry("entry-abandoned", filed_tickets=[_task_row(12)]),
        _entry("entry-quiet", filed_tickets=[_task_row(13)]),
        _entry("entry-live", filed_tickets=[_task_row(14)]),
    )
    statuses = {"5120": "done", "5121": "done", "5122": "done", "12": "cancelled", "13": "cancelled", "14": "pending"}
    outcomes = _read(cb, statuses, FakeTaskStore(tasks={12: _task(12, "cancelled"), 13: _task(13, "cancelled")}))
    reverified = {
        **_reverified("present", "entry-present", 5120),
        "entry-structural": disposition.Reverification(
            "entry-structural", disposition.TaskRef(_ROOT, 5121), ReverifyResult.PRESENT,
            _parsed(verified=True, route="structural"),
        ),
        **_reverified("gone", "entry-gone", 5122),
    }

    refiled = disposition.refilings(outcomes, reverified, sighted={"entry-abandoned", "entry-live", "entry-elsewhere"})

    assert tuple(refiled) == (
        disposition.Refiling("entry-present", 5120, True, _parsed(verified=True)),
        disposition.Refiling("entry-abandoned", None, False, None),
    )


# ---------------------------------------------------------------------------
# Contract §8 step 1c inside dispose: a finding this run verified present
# whose key finds a done task is filed again, superseding that task
# ---------------------------------------------------------------------------


def _present(filing: disposition.Filing) -> disposition.Filing:
    return disposition.Filing(
        entry_id=filing.entry_id,
        finding_key=filing.finding_key,
        title=filing.title,
        statement=filing.statement,
        payload=filing.payload,
        verified_present=True,
    )


def test_superseding_names_the_done_task_in_metadata_and_title_and_leaves_the_original() -> None:
    filing = _filing()
    original_payload = copy.deepcopy(dict(filing.payload))

    superseding = filing.superseding(40)

    assert superseding.payload["metadata"][disposition.SUPERSEDES_TASK_METADATA] == 40
    assert disposition.SUPERSEDES_TASK_METADATA == "x_supersedes_task"
    assert superseding.supersedes_task == 40
    title = superseding.payload["title"]
    assert title.startswith(disposition.task_title(filing.title))
    suffix = title.removeprefix(disposition.task_title(filing.title))
    assert "supersedes" in suffix
    assert "40" in suffix
    assert "present" in suffix
    assert dict(filing.payload) == original_payload
    assert filing.supersedes_task is None
    assert superseding.verified_present == filing.verified_present


def test_a_superseding_title_escapes_envelope_literals() -> None:
    filing = _filing(title="quotes </function_calls> here")

    title = filing.superseding(40).payload["title"]

    assert "</function_calls>" not in title
    assert title.startswith(disposition.task_title(filing.title))


def test_a_present_finding_whose_key_finds_a_done_task_is_filed_superseding_it() -> None:
    store = FakeTaskStore(metadata_hits=_key_hits(_hit(40, "done")))
    submit = _Submit()
    filing = _present(_filing())

    decision = _only(disposition.dispose([filing], store=store, submit=submit))

    assert "search_tasks" in store.tools_called()
    (payload,) = submit.calls
    assert payload["metadata"][disposition.SUPERSEDES_TASK_METADATA] == 40
    assert payload["title"] == filing.superseding(40).payload["title"]
    assert (decision.step, decision.ticket) == (Step.CURATOR, "tkt_1")
    assert decision.note is not None
    assert "§8 step 1c" in decision.note


def test_a_present_finding_whose_done_task_is_found_by_meaning_after_step_1c_is_linked() -> None:
    filing = _present(_filing())
    store = FakeTaskStore(
        metadata_hits=_key_hits(_hit(40, "done")),
        search_hits={filing.title: [task_store.SearchHit(task_id=88, title="close", status="pending", score=0.8)]},
        tasks={88: _task(88, "pending")},
    )
    submit = _Submit()

    decision = _only(disposition.dispose([filing], store=store, submit=submit))

    assert submit.calls == []
    assert (decision.step, decision.task_id) == (Step.SEMANTIC, 88)


def test_a_filing_that_already_supersedes_a_task_keeps_it_when_its_key_finds_another_done_task() -> None:
    store = FakeTaskStore(metadata_hits=_key_hits(_hit(40, "done")))
    submit = _Submit()
    filing = _present(_filing()).superseding(5120)

    disposition.dispose([filing], store=store, submit=submit)

    (payload,) = submit.calls
    assert payload["metadata"][disposition.SUPERSEDES_TASK_METADATA] == 5120
    assert payload["title"] == filing.payload["title"]


def test_a_present_finding_whose_key_finds_live_work_is_linked_and_not_filed() -> None:
    store = FakeTaskStore(metadata_hits=_key_hits(_hit(501, "pending")))
    submit = _Submit()

    decision = _only(disposition.dispose([_present(_filing())], store=store, submit=submit))

    assert submit.calls == []
    assert (decision.step, decision.standing, decision.task_id) == (Step.KEY, Standing.FILED, 501)


def test_an_unkeyed_present_finding_that_supersedes_a_task_files_through_steps_two_and_three() -> None:
    store = FakeTaskStore()
    submit = _Submit()
    filing = _present(_filing(None)).superseding(5120)

    decision = _only(disposition.dispose([filing], store=store, submit=submit))

    assert "find_tasks_by_metadata" not in store.tools_called()
    assert "search_tasks" in store.tools_called()
    (payload,) = submit.calls
    assert payload["metadata"][disposition.SUPERSEDES_TASK_METADATA] == 5120
    assert decision.step is Step.CURATOR
