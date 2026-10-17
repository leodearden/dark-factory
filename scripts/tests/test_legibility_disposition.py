"""Tests for scripts/legibility/disposition.py: how a verified census finding
becomes work and how its task links back (plans/census-incremental-prd.md L4;
docs/quality-findings-contract.md §6, §7, §8)."""
from __future__ import annotations

from typing import Any

import pytest
from legibility import codebook, disposition, task_store
from legibility_task_store_fake import FakeTaskStore

_ROOT = "/abs/observed"
_RUN = "census-dark_factory-20261017"
_KEY_A = "fk-00000000000a"
_KEY_B = "fk-00000000000b"

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
